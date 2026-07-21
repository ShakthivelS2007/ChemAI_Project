import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MessagePassing

class RadialBasisExpansion(nn.Module):
    def __init__(self, num_rbf=16, r_max=6.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0.0, r_max, num_rbf))
        self.gamma = 1.0 / (self.centers[1] - self.centers[0]).item() ** 2
    def forward(self, dists):
        return torch.exp(-self.gamma * (dists.unsqueeze(-1) - self.centers) ** 2)

class EquivariantMobilityConv(MessagePassing):
    def __init__(self, channels=128, num_rbf=16):
        super().__init__(aggr='add')
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
        super().__init__()
        self.node_embed = nn.Linear(node_in, hidden)
        self.convs = nn.ModuleList([EquivariantMobilityConv(channels=hidden) for _ in range(4)])
        self.fc_fused = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden), nn.SiLU())
        self.heads = nn.ModuleDict({
            "band_gap": nn.Linear(hidden, 1), "dielectric": nn.Linear(hidden, 1),
            "ionic_conductivity": nn.Linear(hidden, 1),
            "stability_window": nn.Linear(hidden, 2), "stability_gate": nn.Linear(hidden, 1),
            "phase_stability": nn.Linear(hidden, 1), "elastic_moduli": nn.Linear(hidden, 2)
        })
    def forward(self, x, edge_index, pos, batch):
        h_x = F.silu(self.node_embed(x))
        v = torch.zeros(h_x.size(0), h_x.size(1), 3, device=x.device)
        for conv in self.convs: h_x, v = conv(h_x, v, edge_index, pos)
        latent = self.fc_fused(global_mean_pool(h_x, batch))
        return {k: head(latent) for k, head in self.heads.items()}

LOG_SPACE_KEYS = ["dielectric", "elastic_moduli", "ionic_conductivity"]
REGRESSION_KEYS = ["band_gap", "dielectric", "ionic_conductivity",
                   "stability_window", "phase_stability", "elastic_moduli"]

def stratified_split_by_source(full_dataset, test_size=0.15, random_state=42):
    from sklearn.model_selection import train_test_split
    mp_data = [d for d in full_dataset if d.source == "mp"]
    obelix_data = [d for d in full_dataset if d.source == "obelix"]
    mp_train, mp_val = train_test_split(mp_data, test_size=test_size, random_state=random_state)
    if len(obelix_data) > 1:
        ob_train, ob_val = train_test_split(obelix_data, test_size=test_size, random_state=random_state)
    else:
        ob_train, ob_val = obelix_data, []
    return mp_train + ob_train, mp_val + ob_val

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_dataset = torch.load('processed_data/equivariant_battery_dataset.pt', weights_only=False)
    _, val_data = stratified_split_by_source(full_dataset)

    checkpoint = torch.load('models/equivariant_battery_transformer.pth', map_location=device)
    stats = checkpoint['norm_stats']
    model = EquivariantBatteryTransformer(node_in=3, hidden=128).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    abs_errors = {k: [] for k in REGRESSION_KEYS}
    log_abs_errors = {k: [] for k in REGRESSION_KEYS}  # orders-of-magnitude error, for wide-dynamic-range properties
    n_valid = {k: 0 for k in REGRESSION_KEYS}
    gate_correct, gate_total = 0, 0

    with torch.no_grad():
        for b in val_loader:
            b = b.to(device)
            preds = model(b.x, b.edge_index, b.pos, b.batch)
            for key in REGRESSION_KEYS:
                t = b.y_dict[key].view(b.num_graphs, -1)
                mask = b.mask_dict[key].view(b.num_graphs, -1).bool().view(-1)
                if not mask.any():
                    continue
                p_raw = preds[key][mask]
                t_sub = t[mask]

                if key == "stability_window":
                    gate_pred = (torch.sigmoid(preds["stability_gate"][mask]) > 0.5).float().view(-1)
                    is_nonzero = (t_sub.abs().sum(dim=1) > 1e-8).float()
                    gate_correct += (gate_pred == is_nonzero).sum().item()
                    gate_total += is_nonzero.numel()
                    p_val = p_raw * (gate_pred.unsqueeze(-1))
                elif key in LOG_SPACE_KEYS and key in stats:
                    mu, sigma = stats[key]['mu'], stats[key]['sigma']
                    p_val = torch.exp(p_raw * sigma + mu)
                    # FIX: raw-space MAE is misleading for properties spanning
                    # many orders of magnitude (ionic_conductivity especially --
                    # a tiny absolute error can still mean many orders of
                    # magnitude off). Report log-space error too.
                    log_abs_errors[key].append((p_raw * sigma + mu - torch.log(t_sub + 1e-6)).abs().mean(dim=1))
                else:
                    p_val = p_raw

                abs_errors[key].append((p_val - t_sub).abs().mean(dim=1))
                n_valid[key] += mask.sum().item()

    print("\n" + "=" * 75)
    print(f"{'Property':<22} | {'Val MAE':>10} | {'Log-space MAE':>14} | {'# valid':>8}")
    print("=" * 75)
    for key in REGRESSION_KEYS:
        if abs_errors[key]:
            mae = torch.cat(abs_errors[key]).mean().item()
            log_mae = torch.cat(log_abs_errors[key]).mean().item() if log_abs_errors[key] else float('nan')
            log_str = f"{log_mae:>14.4f}" if log_abs_errors[key] else f"{'--':>14}"
            print(f"{key.replace('_',' ').title():<22} | {mae:>10.4f} | {log_str} | {n_valid[key]:>8d}")
        else:
            print(f"{key.replace('_',' ').title():<22} | {'N/A':>10} | {'--':>14} | {0:>8d}")
    if gate_total:
        print(f"\nStability gate accuracy (nonzero-window classifier): {gate_correct/gate_total:.2%} ({gate_total} samples)")
    print("=" * 75)
    print("\nLog-space MAE for ionic_conductivity is in natural-log units --")
    print("roughly 'average orders of magnitude off' (divide by ln(10)=2.303")
    print("to convert to decades of magnitude). This matters much more than")
    print("raw MAE for a property spanning ~10 orders of magnitude.")

if __name__ == "__main__":
    main()