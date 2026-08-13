import os
import torch
import numpy as np
from mp_api.client import MPRester
from tqdm import tqdm
from torch_geometric.data import Data

# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================
# FIX: never hardcode API keys in source. Set this in your shell:
#   export MP_API_KEY="your-key-here"
API_KEY = os.environ.get("MP_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the MP_API_KEY environment variable before running this script.")

OUTPUT_PATH = 'processed_data/equivariant_battery_dataset.pt'
MAX_MATERIALS = 5000
EDGE_CUTOFF = 4.5  # Angstrom

FEATURE_MAP = {
    'O':  [8,  3.44, 60],  'Li': [3,  0.98, 145], 'Si': [14, 1.90, 110],
    'Al': [13, 1.61, 125], 'Fe': [26, 1.83, 140], 'P':  [15, 2.19, 100],
    'Zr': [40, 1.33, 175], 'Rb': [37, 0.82, 235], 'Ba': [56, 0.89, 215],
    'Ti': [22, 1.54, 160], 'K':  [19, 0.82, 220], 'Na': [11, 0.93, 180],
    'H':  [1,  2.20, 25],  'Br': [35, 2.96, 115], 'Co': [27, 1.88, 135],
    'Ge': [32, 2.01, 120], 'Cl': [17, 3.16, 100], 'F':  [9,  3.98, 50],
    'Au': [79, 2.54, 135], 'Mg': [12, 1.31, 145], 'Ca': [20, 1.00, 195],
    'Sr': [38, 0.95, 215], 'Hf': [72, 1.30, 175], 'Cu': [29, 1.90, 135],
    'Ag': [47, 1.93, 145], 'Ga': [31, 1.81, 130], 'As': [33, 2.18, 115],
    'C':  [6,  2.55, 70],  'N':  [7,  3.04, 65],  'S':  [16, 2.58, 100],
    'Se': [34, 2.55, 115], 'Te': [52, 2.10, 135], 'I':  [53, 2.66, 140],
    'B':  [5,  2.04, 85],  'Be': [4,  1.57, 105], 'Sc': [21, 1.36, 160],
    'V':  [23, 1.63, 135], 'Cr': [24, 1.66, 140], 'Mn': [25, 1.55, 140],
    'Ni': [28, 1.91, 135], 'Zn': [30, 1.65, 135], 'Y':  [39, 1.22, 180],
    'Nb': [41, 1.60, 145], 'Mo': [42, 2.16, 145], 'Ru': [44, 2.20, 130],
    'Rh': [45, 2.28, 135], 'Pd': [46, 2.20, 140], 'Cd': [48, 1.69, 140],
    'Sn': [50, 1.96, 140], 'Sb': [51, 2.05, 140], 'Cs': [55, 0.79, 260],
    'La': [57, 1.10, 195], 'Ta': [73, 1.50, 145], 'W':  [74, 2.36, 145],
    'Re': [75, 1.90, 135], 'Os': [76, 2.20, 130], 'Ir': [77, 2.20, 135],
    'Pt': [78, 2.28, 135], 'Hg': [80, 2.00, 135], 'Tl': [81, 1.62, 145],
    'Pb': [82, 2.33, 180], 'Bi': [83, 2.02, 160], 'Ce': [58, 1.12, 185],
    'Pr': [59, 1.13, 185], 'Nd': [60, 1.14, 185], 'Sm': [62, 1.17, 185],
    'Gd': [64, 1.20, 180], 'Tb': [65, 1.20, 175], 'Dy': [66, 1.22, 175],
    'Ho': [67, 1.23, 175], 'Er': [68, 1.24, 175], 'Tm': [69, 1.25, 175],
    'Yb': [70, 1.10, 175], 'Lu': [71, 1.27, 175], 'Th': [90, 1.30, 180],
    'Pa': [91, 1.50, 180], 'U':  [92, 1.38, 175], 'Ac': [89, 1.10, 195],
    'In': [49, 1.78, 155]
}

# All tasks the merged dataset can carry a value for. Every graph gets a
# y_dict entry AND a mask_dict entry for each of these -- mask says whether
# that particular value is trustworthy for that particular graph. Consumers
# (the loss function) must never use a value where mask is False.
ALL_TASKS = ["band_gap", "dielectric", "ionic_conductivity",
             "stability_window", "phase_stability", "elastic_moduli"]
# NOTE: activation_energy has been removed entirely -- the old
# estimate_transport_physics() function fabricated it (and ionic_conductivity)
# from a hand-written formula with injected noise, not real data. No
# comparable real dataset at this scale was found this pass. Revisit with
# He et al. 2020's BVSE-derived activation energies if/when you want it back.

def build_graph_from_structure(structure, elements):
    """Shared graph-construction logic for both MP and OBELiX structures."""
    if any(el not in FEATURE_MAP for el in elements):
        return None
    node_features = [FEATURE_MAP[el] for el in elements]
    x = torch.tensor(node_features, dtype=torch.float)
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)

    edge_index = []
    for i, site_i in enumerate(structure):
        for j, site_j in enumerate(structure):
            if i != j:
                dist = site_i.distance(site_j)
                if dist <= EDGE_CUTOFF:
                    edge_index.append([i, j])
    if not edge_index:
        return None
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return x, pos, edge_index_tensor

def make_empty_targets():
    """Placeholder y_dict/mask_dict with everything marked invalid. Callers
    fill in the tasks they actually have real data for and flip those masks
    to True."""
    y = {
        "band_gap": torch.tensor([0.0]), "dielectric": torch.tensor([0.0]),
        "ionic_conductivity": torch.tensor([0.0]),
        "stability_window": torch.tensor([0.0, 0.0]),
        "phase_stability": torch.tensor([0.0]),
        "elastic_moduli": torch.tensor([0.0, 0.0]),
    }
    mask = {k: torch.tensor([False]) for k in ALL_TASKS}
    return y, mask

# ==============================================================================
# SOURCE 1: Materials Project (band_gap, dielectric, elastic_moduli,
# phase_stability, stability_window)
# ==============================================================================
def build_mp_dataset():
    pyg_dataset = []
    with MPRester(API_KEY) as mpr:
        print("\n=== MP STEP 1: Querying Materials Project Database Summary ===")
        # FIX: is_stable=True guarantees energy_above_hull ~= 0 for every
        # result, which is exactly why phase_stability had zero variance.
        # Use a range instead so metastable phases (still physically
        # reasonable, just not on the exact convex hull) are included too.
        docs = mpr.materials.summary.search(
            has_props=["dielectric", "elasticity"],
            energy_above_hull=(0, 0.5),
            fields=["material_id", "structure", "band_gap", "energy_above_hull", "formula_pretty"]
        )[:MAX_MATERIALS]

        mat_ids = [doc.material_id for doc in docs]

        print("=== MP STEP 2: Pulling Secondary Property Sub-Endpoints ===")
        diel_docs = mpr.materials.dielectric.search(material_ids=mat_ids, fields=["material_id", "e_total"])
        diel_map = {str(d.material_id): float(d.e_total) for d in diel_docs if hasattr(d, 'e_total')}

        elastic_docs = mpr.materials.elasticity.search(material_ids=mat_ids, fields=["material_id", "bulk_modulus", "shear_modulus"])
        elastic_map = {}
        for e in elastic_docs:
            if hasattr(e, 'bulk_modulus') and e.bulk_modulus and hasattr(e, 'shear_modulus') and e.shear_modulus:
                k_voigt = float(getattr(e.bulk_modulus, "voigt", 0.0))
                g_voigt = float(getattr(e.shear_modulus, "voigt", 0.0))
                elastic_map[str(e.material_id)] = (k_voigt, g_voigt)

        print("-> Fetching real electrochemical stability records...")
        # FIX #3: there's no flat min_voltage/max_voltage field on this doc at
        # all -- confirmed from the error message listing every available
        # field. The real per-step voltage data lives inside `electrode_object`
        # (a pymatgen InsertionElectrode), specifically `.voltage_pairs`, each
        # of which has a `.voltage` attribute. min/max across those steps is
        # our real electrochemical stability window.
        from pymatgen.core import Element
        electro_map = {}
        mat_id_set = set(mat_ids)
        try:
            electrode_docs = mpr.materials.insertion_electrodes.search(
                working_ion=Element("Li"),
                fields=["material_ids", "electrode_object"]
            )
            for e in electrode_docs:
                eo = getattr(e, "electrode_object", None)
                if eo is None or not getattr(eo, "voltage_pairs", None):
                    continue
                voltages = [vp.voltage for vp in eo.voltage_pairs]
                v_low, v_high = min(voltages), max(voltages)
                for m_id in getattr(e, "material_ids", []) or []:
                    if str(m_id) in mat_id_set:
                        electro_map[str(m_id)] = (v_low, v_high)
        except Exception as ex:
            print(f"insertion_electrodes query failed ({ex}). stability_window will be masked out entirely.")
        print(f"-> Got real electrochemistry data for {len(electro_map)} materials.")

        print("=== MP STEP 3: Mapping Features into Geometric Data Formats ===")
        for doc in tqdm(docs):
            m_id = str(doc.material_id)
            if m_id not in diel_map or m_id not in elastic_map:
                continue

            struct = doc.structure
            elements = [str(spec.symbol) for spec in struct.species]
            graph = build_graph_from_structure(struct, elements)
            if graph is None:
                continue
            x, pos, edge_index_tensor = graph

            y, mask = make_empty_targets()
            y["band_gap"] = torch.tensor([float(doc.band_gap)]); mask["band_gap"][0] = True
            y["dielectric"] = torch.tensor([diel_map[m_id]]); mask["dielectric"][0] = True
            k_mod, g_mod = elastic_map[m_id]
            y["elastic_moduli"] = torch.tensor([k_mod, g_mod]); mask["elastic_moduli"][0] = True
            y["phase_stability"] = torch.tensor([float(doc.energy_above_hull)]); mask["phase_stability"][0] = True

            # FIX: only mark stability_window valid when the material actually
            # HAS electrochemistry data. Materials without it are simply
            # excluded from that head's loss -- not silently given a fake 0.0.
            if m_id in electro_map:
                v_low, v_high = electro_map[m_id]
                y["stability_window"] = torch.tensor([v_low, v_high])
                mask["stability_window"][0] = True
            # ionic_conductivity stays masked False -- MP doesn't have real data for it.

            data_obj = Data(x=x, edge_index=edge_index_tensor, pos=pos, y_dict=y, mask_dict=mask)
            data_obj.formula = doc.formula_pretty
            data_obj.source = "mp"
            pyg_dataset.append(data_obj)

    print(f">>> MP source: {len(pyg_dataset)} valid graphs.")
    mp_ids_used = {str(doc.material_id) for doc in docs}
    return pyg_dataset, mp_ids_used

# ==============================================================================
# SOURCE 1b: Materials Project, band_gap ONLY, no dielectric/elasticity
# co-requirement. band_gap doesn't need either of those to exist, and
# requiring them cut the usable pool down to ~1900 materials for no reason
# -- band_gap alone is available for a much larger fraction of MP. This is
# the single biggest lever for band_gap's ~0.83 eV MAE: more real training
# examples, not more architecture tuning.
# ==============================================================================
BANDGAP_ONLY_MAX_MATERIALS = 15000  # tune down if this makes a Colab run too slow

def build_mp_bandgap_only_dataset(exclude_ids):
    pyg_dataset = []
    with MPRester(API_KEY) as mpr:
        print("\n=== MP (band_gap only) STEP 1: Broad query, no dielectric/elasticity requirement ===")
        docs = mpr.materials.summary.search(
            energy_above_hull=(0, 0.5),
            fields=["material_id", "structure", "band_gap", "formula_pretty"]
        )[:BANDGAP_ONLY_MAX_MATERIALS]

        print("=== MP (band_gap only) STEP 2: Mapping Features into Geometric Data Formats ===")
        for doc in tqdm(docs):
            m_id = str(doc.material_id)
            if m_id in exclude_ids:
                continue  # already covered (with more properties) by the main MP source
            if doc.band_gap is None:
                continue

            struct = doc.structure
            elements = [str(spec.symbol) for spec in struct.species]
            graph = build_graph_from_structure(struct, elements)
            if graph is None:
                continue
            x, pos, edge_index_tensor = graph

            y, mask = make_empty_targets()
            y["band_gap"] = torch.tensor([float(doc.band_gap)])
            mask["band_gap"][0] = True
            # everything else stays masked False -- this source only has band_gap.

            data_obj = Data(x=x, edge_index=edge_index_tensor, pos=pos, y_dict=y, mask_dict=mask)
            data_obj.formula = doc.formula_pretty
            data_obj.source = "mp"  # same modality as the main MP source, grouped together for splitting
            pyg_dataset.append(data_obj)

    print(f">>> MP (band_gap only) source: {len(pyg_dataset)} additional valid graphs.")
    return pyg_dataset

# ==============================================================================
# SOURCE 2: OBELiX (real, experimentally measured ionic conductivity)
# pip install obelix-data
# ==============================================================================
def build_obelix_dataset():
    from obelix import OBELiX
    ob = OBELiX()
    pyg_dataset = []

    print("\n=== OBELiX: Building ionic conductivity graphs from real CIF structures ===")
    for entry in tqdm(ob.round_partial().with_cifs()):
        ic = entry.get("Ionic conductivity (S cm-1)")
        if ic is None or ic <= 0:
            continue
        structure = entry["structure"]
        elements = [str(spec.symbol) for spec in structure.species]
        graph = build_graph_from_structure(structure, elements)
        if graph is None:
            continue
        x, pos, edge_index_tensor = graph

        y, mask = make_empty_targets()
        y["ionic_conductivity"] = torch.tensor([float(ic)])
        mask["ionic_conductivity"][0] = True
        # everything else stays masked False -- OBELiX doesn't have band_gap,
        # dielectric, elastic_moduli, phase_stability, or stability_window.

        data_obj = Data(x=x, edge_index=edge_index_tensor, pos=pos, y_dict=y, mask_dict=mask)
        data_obj.formula = entry.get("Reduced Composition", "Unknown")
        data_obj.source = "obelix"
        pyg_dataset.append(data_obj)

    print(f">>> OBELiX source: {len(pyg_dataset)} valid graphs.")
    return pyg_dataset

# ==============================================================================
# MERGE + SAVE
# ==============================================================================
def build_equivariant_dataset():
    os.makedirs('processed_data', exist_ok=True)
    mp_data, mp_ids_used = build_mp_dataset()
    mp_bandgap_data = build_mp_bandgap_only_dataset(exclude_ids=mp_ids_used)
    try:
        obelix_data = build_obelix_dataset()
    except ImportError:
        print("obelix-data not installed (`pip install obelix-data`) -- skipping real ionic conductivity data.")
        obelix_data = []

    full_dataset = mp_data + mp_bandgap_data + obelix_data
    print(f"\n>>> Compiling Complete. Saved {len(full_dataset)} valid matrices "
          f"({len(mp_data)} MP full + {len(mp_bandgap_data)} MP band_gap-only + {len(obelix_data)} OBELiX) "
          f"to: {OUTPUT_PATH}")
    torch.save(full_dataset, OUTPUT_PATH)

if __name__ == "__main__":
    build_equivariant_dataset()