import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MessagePassing
from sklearn.model_selection import train_test_split

# ==============================================================================
# ARCHITECTURE (must match train.py / predict.py)
# ==============================================================================
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
            "ionic_conductivity": nn.Linear(hidden, 1), "activation_energy": nn.Linear(hidden, 1),
            "stability_window": nn.Linear(hidden, 2), "phase_stability": nn.Linear(hidden, 1),
            "elastic_moduli": nn.Linear(hidden, 2)
        })
    def forward(self, x, edge_index, pos, batch):
        h_x = F.silu(self.node_embed(x))
        v = torch.zeros(h_x.size(0), h_x.size(1), 3, device=x.device)
        for conv in self.convs: h_x, v = conv(h_x, v, edge_index, pos)
        latent = self.fc_fused(global_mean_pool(h_x, batch))
        return {k: head(latent) for k, head in self.heads.items()}

LOG_SPACE_KEYS = ["dielectric", "elastic_moduli"]

# ==============================================================================
# DIAGNOSTICS
# ==============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_dataset = torch.load('processed_data/equivariant_battery_dataset.pt', weights_only=False)
    train_data, val_data = train_test_split(full_dataset, test_size=0.15, random_state=42)

    checkpoint = torch.load('models/equivariant_battery_transformer.pth', map_location=device)
    stats = checkpoint['norm_stats']
    model = EquivariantBatteryTransformer(node_in=3, hidden=128).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    keys = ["band_gap", "dielectric", "ionic_conductivity", "activation_energy",
            "stability_window", "phase_stability", "elastic_moduli"]
    abs_errors = {k: [] for k in keys}
    all_targets = {k: [] for k in keys}

    with torch.no_grad():
        for b in val_loader:
            b = b.to(device)
            preds = model(b.x, b.edge_index, b.pos, b.batch)
            for key in keys:
                t = b.y_dict[key].view(b.num_graphs, -1)
                p_raw = preds[key]
                if key in LOG_SPACE_KEYS:
                    mu, sigma = stats[key]['mu'], stats[key]['sigma']
                    p = torch.exp(p_raw * sigma + mu)
                else:
                    p = p_raw
                abs_errors[key].append((p - t).abs().mean(dim=1))
                all_targets[key].append(t)

    print("\n" + "=" * 70)
    print(f"{'Property':<22} | {'Val MAE':>10} | {'Target Mean':>12} | {'Target Std':>10} | {'% == 0':>7}")
    print("=" * 70)
    for key in keys:
        errs = torch.cat(abs_errors[key])
        targets = torch.cat(all_targets[key])
        mae = errs.mean().item()
        t_mean = targets.mean().item()
        t_std = targets.std().item()
        pct_zero = (targets.abs() < 1e-8).float().mean().item() * 100
        print(f"{key.replace('_',' ').title():<22} | {mae:>10.4f} | {t_mean:>12.4f} | {t_std:>10.4f} | {pct_zero:>6.1f}%")
    print("=" * 70)
    print("\nIf '% == 0' is high for stability_window / phase_stability, those")
    print("targets are likely degenerate/placeholder values, not real continuous")
    print("labels -- worth checking your data prep for those two columns.")

if __name__ == "__main__":
    main()