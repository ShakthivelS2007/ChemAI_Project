import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MessagePassing
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. ARCHITECTURE DEFINITIONS
# ==============================================================================
class RadialBasisExpansion(nn.Module):
    def __init__(self, num_rbf=16, r_max=6.0):
        super(RadialBasisExpansion, self).__init__()
        self.register_buffer('centers', torch.linspace(0.0, r_max, num_rbf))
        self.gamma = 1.0 / (self.centers[1] - self.centers[0]).item() ** 2

    def forward(self, dists):
        return torch.exp(-self.gamma * (dists.unsqueeze(-1) - self.centers) ** 2)

class EquivariantMobilityConv(MessagePassing):
    def __init__(self, channels=128, num_rbf=16):
        super(EquivariantMobilityConv, self).__init__(aggr='add')
        self.radial_mlp = nn.Sequential(nn.Linear(num_rbf, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.vector_mlp = nn.Sequential(nn.Linear(channels, 1, bias=False))
        self.node_mlp = nn.Linear(channels, channels)
        self.rbf = RadialBasisExpansion(num_rbf=num_rbf, r_max=6.0)

    def forward(self, x, v, edge_index, pos):
        row, col = edge_index
        coord_diff = pos[col] - pos[row]
        dists = torch.norm(coord_diff, dim=-1)
        unit_vectors = coord_diff / (dists.unsqueeze(-1) + 1e-6)
        edge_weights = self.radial_mlp(self.rbf(dists))
        vec_j, w_v = v[row], edge_weights.unsqueeze(-1)
        dir_weights = self.vector_mlp(edge_weights).unsqueeze(-1)
        edge_vec_msg = (vec_j * w_v) + (unit_vectors.unsqueeze(1) * dir_weights)
        out_x, out_v = self.propagate(edge_index, x=x, edge_weights=edge_weights, edge_vec_msg=edge_vec_msg, size=(x.size(0), x.size(0)))
        return F.silu(x + self.node_mlp(out_x)), v + 0.5 * out_v

    def message(self, x_j, edge_weights, edge_vec_msg): return x_j * edge_weights, edge_vec_msg
    def aggregate(self, inputs, index, dim_size=None):
        msg_x, msg_v = inputs
        agg_x = torch.zeros(dim_size, msg_x.size(-1), device=msg_x.device).index_add_(0, index, msg_x)
        agg_v = torch.zeros(dim_size, msg_v.size(1), 3, device=msg_v.device).index_add_(0, index, msg_v)
        return agg_x, agg_v

class EquivariantBatteryTransformer(nn.Module):
    def __init__(self, node_in=3, hidden=128):
        super(EquivariantBatteryTransformer, self).__init__()
        self.node_embed = nn.Linear(node_in, hidden)
        self.convs = nn.ModuleList([EquivariantMobilityConv(channels=hidden) for _ in range(4)])
        self.fc_fused = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden), nn.SiLU())
        # NOTE: activation_energy dropped (no real data source this pass).
        # phase_stability restored (query bug fixed -- see dataset_builder.py).
        # stability_gate: auxiliary hurdle classifier for stability_window.
        self.heads = nn.ModuleDict({
            "band_gap": nn.Linear(hidden, 1),
            "dielectric": nn.Linear(hidden, 1),
            "ionic_conductivity": nn.Linear(hidden, 1),
            "stability_window": nn.Linear(hidden, 2),
            "stability_gate": nn.Linear(hidden, 1),
            "phase_stability": nn.Linear(hidden, 1),
            "elastic_moduli": nn.Linear(hidden, 2)
        })

    def forward(self, x, edge_index, pos, batch):
        h_x = F.silu(self.node_embed(x))
        v = torch.zeros(h_x.size(0), h_x.size(1), 3, device=x.device)
        for conv in self.convs: h_x, v = conv(h_x, v, edge_index, pos)
        latent = self.fc_fused(global_mean_pool(h_x, batch))
        return {k: head(latent) for k, head in self.heads.items()}

# ==============================================================================
# 2. LOSS FUNCTIONS (masked multi-task, heterogeneous label availability)
# ==============================================================================
def log_cosh_loss(pred, target):
    return torch.mean(torch.log(torch.cosh(pred - target + 1e-12)))

# Ionic conductivity spans many orders of magnitude (real S/cm values from
# OBELiX) so it needs the same log-space treatment as dielectric/elastic_moduli.
LOG_SPACE_KEYS = ["dielectric", "elastic_moduli", "ionic_conductivity"]
REGRESSION_KEYS = ["band_gap", "dielectric", "ionic_conductivity",
                   "stability_window", "phase_stability", "elastic_moduli"]

def compute_global_log_stats(dataset, keys=LOG_SPACE_KEYS):
    """
    Computes log-space mean/std ONCE from the training split, using ONLY
    samples where that key is actually valid (mask_dict[key] == True).
    Mixing in masked-out placeholder zeros here would corrupt the stats.
    """
    stats = {}
    for key in keys:
        valid_ys = [d.y_dict[key] for d in dataset if bool(d.mask_dict[key][0])]
        if not valid_ys:
            continue
        all_y = torch.cat(valid_ys, dim=0)
        log_y = torch.log(all_y + 1e-6)
        stats[key] = {'mu': log_y.mean().item(), 'sigma': (log_y.std() + 1e-6).item()}
    return stats

class MultiTaskLoss(nn.Module):
    def __init__(self, norm_stats, num_tasks=len(REGRESSION_KEYS)):
        super(MultiTaskLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.ones(num_tasks) * 1.5)
        self.weight_map = {
            "band_gap": 1.0, "dielectric": 5.0, "elastic_moduli": 5.0,
            "ionic_conductivity": 2.0, "stability_window": 1.0, "phase_stability": 1.0,
        }
        self.norm_stats = norm_stats
        self.gate_loss_fn = nn.BCEWithLogitsLoss()
        self.gate_weight = 1.0

    def forward(self, preds, targets, masks):
        total_loss = 0.0
        huber = nn.HuberLoss(delta=1.0)

        for idx, key in enumerate(REGRESSION_KEYS):
            pred = preds[key]
            target = targets[key]
            valid = masks[key].bool().view(-1)

            if key == "stability_window":
                if valid.any():
                    sub_pred, sub_target = pred[valid], target[valid]
                    is_nonzero = (sub_target.abs().sum(dim=1) > 1e-8)
                    gate_target = is_nonzero.float().unsqueeze(-1)
                    gate_loss = self.gate_loss_fn(preds["stability_gate"][valid], gate_target)
                    total_loss = total_loss + self.gate_weight * gate_loss
                    if is_nonzero.any():
                        loss = huber(sub_pred[is_nonzero], sub_target[is_nonzero]) * self.weight_map.get(key, 1.0)
                    else:
                        loss = pred.sum() * 0.0
                else:
                    loss = pred.sum() * 0.0
            elif key in LOG_SPACE_KEYS:
                if valid.any() and key in self.norm_stats:
                    mu, sigma = self.norm_stats[key]['mu'], self.norm_stats[key]['sigma']
                    target_log = torch.log(target[valid] + 1e-6)
                    target_norm = (target_log - mu) / sigma
                    loss = log_cosh_loss(pred[valid], target_norm) * self.weight_map.get(key, 5.0)
                else:
                    loss = pred.sum() * 0.0
            else:
                if valid.any():
                    loss = huber(pred[valid], target[valid]) * self.weight_map.get(key, 1.0)
                else:
                    loss = pred.sum() * 0.0

            total_loss = total_loss + torch.exp(-torch.clamp(self.log_vars[idx], -2.0, 5.0)) * loss + self.log_vars[idx]

        return total_loss

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================
def stratified_split_by_source(full_dataset, test_size=0.15, random_state=42):
    """
    Splits MP-origin and OBELiX-origin graphs separately (same ratio) then
    recombines. A plain random split risks putting a skewed fraction of the
    much smaller OBELiX subset into train or val purely by chance.
    """
    mp_data = [d for d in full_dataset if d.source == "mp"]
    obelix_data = [d for d in full_dataset if d.source == "obelix"]

    mp_train, mp_val = train_test_split(mp_data, test_size=test_size, random_state=random_state)
    if len(obelix_data) > 1:
        ob_train, ob_val = train_test_split(obelix_data, test_size=test_size, random_state=random_state)
    else:
        ob_train, ob_val = obelix_data, []

    return mp_train + ob_train, mp_val + ob_val

def train_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_dataset = torch.load('processed_data/equivariant_battery_dataset.pt', weights_only=False)
    train_data, val_data = stratified_split_by_source(full_dataset)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    norm_stats = compute_global_log_stats(train_data)
    print("Global log-space norm stats (from train split):", norm_stats)

    model = EquivariantBatteryTransformer(node_in=3, hidden=128).to(device)
    criterion = MultiTaskLoss(norm_stats=norm_stats).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=50)

    for epoch in range(1, 51):
        model.train()
        train_loss = 0.0
        for b in train_loader:
            b = b.to(device)
            optimizer.zero_grad()
            preds = model(b.x, b.edge_index, b.pos, b.batch)
            targets = {k: b.y_dict[k].view(b.num_graphs, -1) for k in REGRESSION_KEYS}
            masks = {k: b.mask_dict[k].view(b.num_graphs, -1) for k in REGRESSION_KEYS}
            loss = criterion(preds, targets, masks)

            loss.backward()
            for p in model.parameters():
                if p.grad is not None and p.grad.dim() > 1:
                    p.grad.add_(-p.grad.mean(dim=1, keepdim=True))

            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in val_loader:
                b = b.to(device)
                preds = model(b.x, b.edge_index, b.pos, b.batch)
                targets = {k: b.y_dict[k].view(b.num_graphs, -1) for k in REGRESSION_KEYS}
                masks = {k: b.mask_dict[k].view(b.num_graphs, -1) for k in REGRESSION_KEYS}
                val_loss += criterion(preds, targets, masks).item()

        print(f"Epoch {epoch:02d}/50 | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    os.makedirs('models', exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'norm_stats': norm_stats},
               'models/equivariant_battery_transformer.pth')

if __name__ == "__main__":
    train_pipeline()