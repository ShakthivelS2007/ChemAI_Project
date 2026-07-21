import os
import torch
from pymatgen.core import Structure

# Your exact training feature map: {Element: [Atomic Number, Electronegativity, Atomic Radius in pm]}
FEATURE_MAP = {
    'O': [8, 3.44, 60],    'Li': [3, 0.98, 145],  'Si': [14, 1.90, 110],
    'Al': [13, 1.61, 125], 'Fe': [26, 1.83, 140], 'P':  [15, 2.19, 100],
    'Zr': [40, 1.33, 160], 'Rb': [37, 0.82, 248], 'Ba': [56, 0.89, 215],
    'Ti': [22, 1.54, 140], 'K':  [19, 0.82, 227], 'Na': [11, 0.93, 186],
    'H':  [1, 2.20, 37],   'Br': [35, 2.96, 114], 'Co': [27, 1.88, 125],
    'Ge': [32, 2.01, 122], 'Cl': [17, 3.16, 99],  'F':  [9, 3.98, 50],
    'Au': [79, 2.54, 144], 'Mg': [12, 1.31, 160], 'Ca': [20, 1.00, 197],
    'Sr': [38, 0.95, 215], 'Hf': [72, 1.30, 155], 'Cu': [29, 1.90, 128],
    'Ag': [47, 1.93, 144], 'Ga': [31, 1.81, 135], 'As': [33, 2.18, 115],
    'C': [6, 2.55, 70],    'N': [7, 3.04, 65],    'S': [16, 2.58, 100],
    'Se': [34, 2.55, 115], 'Te': [52, 2.1, 140],  'I': [53, 2.66, 140],
    'B': [5, 2.04, 85],    'Be': [4, 1.57, 105],  'Sc': [21, 1.36, 160], 
    'V': [23, 1.63, 135],  'Cr': [24, 1.66, 140], 'Mn': [25, 1.55, 140], 
    'Ni': [28, 1.91, 135], 'Zn': [30, 1.65, 135], 'Y': [39, 1.22, 180],
    'Nb': [41, 1.6, 145],  'Mo': [42, 2.16, 145], 'Ru': [44, 2.2, 130],
    'Rh': [45, 2.28, 135], 'Pd': [46, 2.2, 140],  'Cd': [48, 1.69, 155], 
    'Sn': [50, 1.96, 145], 'Sb': [51, 2.05, 145], 'Cs': [55, 0.79, 260], 
    'La': [57, 1.1, 195],  'Ta': [73, 1.5, 145],  'W': [74, 2.36, 135],
    'Re': [75, 1.9, 135],  'Os': [76, 2.2, 130],  'Ir': [77, 2.2, 135],
    'Pt': [78, 2.28, 135], 'Hg': [80, 2.0, 150],  'Tl': [81, 1.62, 170], 
    'Pb': [82, 2.33, 180], 'Bi': [83, 2.02, 160], 'Ce': [58, 1.12, 185], 
    'Pr': [59, 1.13, 185], 'Nd': [60, 1.14, 185], 'Sm': [62, 1.17, 185], 
    'Gd': [64, 1.2, 180],  'Tb': [65, 1.1, 175],  'Dy': [66, 1.22, 175], 
    'Ho': [67, 1.23, 175], 'Er': [68, 1.24, 175], 'Tm': [69, 1.25, 175], 
    'Yb': [70, 1.1, 170],  'Lu': [71, 1.27, 175], 'Th': [90, 1.3, 180],
    'Pa': [91, 1.5, 180],  'U': [92, 1.38, 175],  'Ac': [89, 1.1, 195],
    'In': [49, 1.78, 155]
}

def parse_cif_to_graph(cif_path, bond_threshold=3.0):
    """
    Parses a .cif file and converts it into a graph structure matching your exact FEATURE_MAP.
    """
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"Could not find the CIF file at: {cif_path}")
        
    # 1. Load the crystal structure using pymatgen
    structure = Structure.from_file(cif_path)
    print(f"Successfully loaded formula: {structure.composition.reduced_formula}")
    print(f"Total atoms (nodes) in unit cell: {len(structure)}")

    node_features = []
    
    # 2. Extract Node Features directly matching FEATURE_MAP mapping arrays
    for site in structure:
        element_symbol = site.specie.symbol
        
        # Get traits from the mapped array. If element isn't in your dict, 
        # it defaults to a dummy array [0, 0.0, 0] to keep the shape safe.
        traits = FEATURE_MAP.get(element_symbol, [0, 0.0, 0])
        
        # Explicitly make sure everything is a float representation for PyTorch
        node_features.append([float(traits[0]), float(traits[1]), float(traits[2])])
        
    # Convert node features to a PyTorch tensor of shape [num_atoms, 3]
    x = torch.tensor(node_features, dtype=torch.float)

    # 3. Calculate Edges based on crystal spatial distance matrix
    edge_list = []
    
    for i in range(len(structure)):
        for j in range(len(structure)):
            if i == j:
                continue # Skip self-loops
                
            distance = structure.get_distance(i, j)
            
            if distance <= bond_threshold:
                edge_list.append([i, j])
                
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return {
        "formula": structure.composition.reduced_formula,
        "x": x,                  # Shape: [num_atoms, 3]
        "edge_index": edge_index # Shape: [2, num_edges]
    }

# --- Quick Local Verification Run ---
if __name__ == "__main__":
    try:
        # Swap this with a path to one of your locally saved .cif test targets
        graph_data = parse_cif_to_graph("sample_material.cif")
        print("\n--- Graph Conversion Summary ---")
        print(f"Formula: {graph_data['formula']}")
        print(f"Node Tensor Shape (Atoms, 3 Features): {graph_data['x'].shape}")
        print(f"Edge Index Tensor Shape: {graph_data['edge_index'].shape}")
        print("\nGenerated Feature Vectors [Atomic No., Electronegativity, Radius in pm]:\n", graph_data['x'][:5]) # print first 5 nodes
    except Exception as e:
        print(f"\nConfiguration Note: {e}")
