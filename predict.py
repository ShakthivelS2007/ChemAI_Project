import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MessagePassing

# ==============================================================================
# 1. ARCHITECTURE DEFINITIONS (Must match train.py)
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

    def forward(self, x, v, edge_index, pos):
        row, col = edge_index
        coord_diff = pos[col] - pos[row]
        dists = torch.norm(coord_diff, dim=-1)
        unit_vectors = coord_diff / (dists.unsqueeze(-1) + 1e-6)
        rbf_engine = RadialBasisExpansion(num_rbf=16, r_max=6.0).to(x.device)
        edge_weights = self.radial_mlp(rbf_engine(dists))
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

# ==============================================================================
# 2. INFERENCE ENGINE
# ==============================================================================
def run_interactive_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EquivariantBatteryTransformer(node_in=3, hidden=128).to(device)
    model.load_state_dict(torch.load('models/equivariant_battery_transformer.pth', map_location=device))
    model.eval()
    
    # Load the full dataset to get global statistics for denormalization
    dataset = torch.load('processed_data/equivariant_battery_dataset.pt', weights_only=False)
    
    user_input = input(">> Enter indices to analyze (e.g., 0, 1, 2): ").strip()
    selected_indices = [int(idx.strip()) for idx in user_input.split(",")] if user_input else [0]
    graphs = [dataset[i] for i in selected_indices if 0 <= i < len(dataset)]
    loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
    
    # Pre-calculate global stats for dielectric and elastic_moduli
    stats = {}
    for key in ["dielectric", "elastic_moduli"]:
        # Extract all targets from the dataset for this property
        all_y = torch.cat([d.y_dict[key] for d in dataset], dim=0)
        log_y = torch.log(all_y + 1e-6)
        stats[key] = {'mu': log_y.mean(), 'sigma': log_y.std()}

    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            preds = model(b.x, b.edge_index, b.pos, b.batch)
            
            print("\n" + "="*80)
            print("                     EQUIVARIANT MULTI-HEAD PREDICTION PROFILING")
            print("="*80)
            
            for i in range(b.num_graphs):
                formula = getattr(b, 'formula', ['Unknown'] * b.num_graphs)[i]
                print(f"\n[COMPOUND: {formula} | MATRIX INDEX #{selected_indices[i]}]")
                
                for key, p_raw in preds.items():
                    if key in ["dielectric", "elastic_moduli"]:
                        mu, sigma = stats[key]['mu'], stats[key]['sigma']
                        # Denormalize: p_raw is the normalized Z-score
                        p_denorm = (p_raw[i] * sigma) + mu
                        p_val = torch.exp(p_denorm)
                    else:
                        p_val = p_raw[i]
                    
                    t_val = b.y_dict[key][i]
                    
                    # Ensure we handle tensors correctly for printing
                    p_final = p_val.mean().item()
                    t_final = t_val.mean().item()
                    
                    print(f"{key.replace('_', ' ').title():<25} | Pred: {p_final:<12.4f} | Actual: {t_final:<12.4f}")

if __name__ == "__main__":
    run_interactive_inference()
