import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QPushButton, QLabel, QGroupBox, 
                             QListWidget, QFileDialog, QComboBox, QLineEdit, QFormLayout)
import pyqtgraph.opengl as gl
from pymatgen.core import Structure

# train.py lives one directory up (ChemAI_Project/), outside this "3D
# software" folder, so it's not importable by default. Add that parent
# directory to sys.path before importing -- this is resolved relative to
# this file's own location, so it works no matter what directory you
# launch main.py from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model architecture must be imported from train.py so the class definition
# can never drift out of sync with the one the checkpoint was actually
# trained with.
from train import EquivariantBatteryTransformer

# -------------------------------------------------------------------
# 1. FEATURE MAP -- must exactly match dataset_builder.py's table,
# since that is what the model was trained on.
# -------------------------------------------------------------------
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

class CADViewWidget(gl.GLViewWidget):
    """Custom 3D viewport that enables free panning and a locked corner orientation axis."""
    def __init__(self, parent=None):
        super().__init__(parent)
        from PyQt6.QtGui import QVector3D
        self.opts['center'] = QVector3D(0.0, 0.0, 0.0)
        
    def paintGL(self, *args, **kwds):
        super().paintGL(*args, **kwds)
        import OpenGL.GL as ogl
        
        ogl.glMatrixMode(ogl.GL_PROJECTION)
        ogl.glPushMatrix()
        ogl.glLoadIdentity()
        
        width = self.width()
        height = self.height()
        ogl.glViewport(10, 10, 80, 80)
        
        scale = 0.0414
        ogl.glFrustum(-scale, scale, -scale, scale, 0.1, 10.0)
        
        ogl.glMatrixMode(ogl.GL_MODELVIEW)
        ogl.glPushMatrix()
        ogl.glLoadIdentity()
        
        ogl.glTranslatef(0.0, 0.0, -2.5)
        
        elev = self.opts['elevation']
        azim = self.opts['azimuth']
        
        ogl.glRotatef(elev - 90, 1.0, 0.0, 0.0)
        ogl.glRotatef(-azim - 90, 0.0, 0.0, 1.0)
        
        ogl.glClear(ogl.GL_DEPTH_BUFFER_BIT)
        
        ogl.glLineWidth(3)
        ogl.glBegin(ogl.GL_LINES)
        
        ogl.glColor4f(1.0, 0.2, 0.2, 1.0)
        ogl.glVertex3f(0.0, 0.0, 0.0); ogl.glVertex3f(0.8, 0.0, 0.0)
        
        ogl.glColor4f(0.2, 1.0, 0.2, 1.0)
        ogl.glVertex3f(0.0, 0.0, 0.0); ogl.glVertex3f(0.0, 0.8, 0.0)
        
        ogl.glColor4f(0.2, 0.2, 1.0, 1.0)
        ogl.glVertex3f(0.0, 0.0, 0.0); ogl.glVertex3f(0.0, 0.0, 0.8)
        
        ogl.glEnd()
        
        ogl.glPopMatrix()
        ogl.glMatrixMode(ogl.GL_PROJECTION)
        ogl.glPopMatrix()
        ogl.glMatrixMode(ogl.GL_MODELVIEW)
        ogl.glViewport(0, 0, width, height)

# -------------------------------------------------------------------
# 3. CORE DESKTOP UI ENGINE
# -------------------------------------------------------------------
class ChemAIDesktopApp(QMainWindow):
    def update_preview_atom(self):
        if self.preview_mesh_item in self.canvas.items:
            self.canvas.removeItem(self.preview_mesh_item)
            
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())
            
            preview_mesh = gl.MeshData.sphere(rows=10, cols=20, radius=0.22)
            self.preview_mesh_item = gl.GLMeshItem(
                meshdata=preview_mesh, 
                smooth=True, 
                color=(1.0, 0.85, 0.0, 0.45),
                shader='shaded'
            )
            self.preview_mesh_item.translate(x, y, z)
            self.canvas.addItem(self.preview_mesh_item)
        except ValueError:
            pass

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChemAI - 3D Material Design Suite v4.7")
        self.setGeometry(100, 100, 1200, 750)
        
        self.device = torch.device("cpu")
        self.model = None
        self.norm_stats = None
        self.load_ml_model()
        
        self.current_formula = "None Loaded"
        self.atom_elements = []
        self.atom_positions = np.empty((0, 3), dtype=np.float32)
        
        self.init_ui()

    def load_ml_model(self):
        # models/ lives under ChemAI_Project/ (the parent of this "3D
        # software" folder), same as train.py -- anchor to that, not to
        # whatever directory the script happens to be launched from.
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "equivariant_battery_transformer.pth")
        try:
            self.model = EquivariantBatteryTransformer(node_in=3, hidden=128)
            if os.path.exists(MODEL_PATH):
                checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state'])
                self.norm_stats = checkpoint['norm_stats']
                self.model.eval()
                print("Model loaded successfully into standalone memory.")
                print("Loaded norm_stats:", self.norm_stats)
            else:
                print(f"Weights file not found at local target path: {MODEL_PATH}")
                self.model = None
        except Exception as e:
            print(f"Model weight warning: Offline extraction mode active. ({e})")
            self.model = None

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        self.canvas = CADViewWidget()
        self.canvas.setCameraPosition(distance=12)
        grid = gl.GLGridItem()
        self.canvas.addItem(grid)
        layout.addWidget(self.canvas, stretch=3)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        layout.addWidget(right_panel, stretch=1)
        
        import_btn = QPushButton("Import .cif File")
        import_btn.setStyleSheet("background-color: #2ca02c; color: white; font-weight: bold; padding: 6px;")
        import_btn.clicked.connect(self.handle_cif_import)
        right_layout.addWidget(import_btn)
        
        inspector_box = QGroupBox("Lattice Grid Inspector")
        inspector_layout = QVBoxLayout(inspector_box)
        self.atom_list = QListWidget()
        inspector_layout.addWidget(self.atom_list)
        right_layout.addWidget(inspector_box)
        
        placement_box = QGroupBox("Place New Custom Atom")
        placement_layout = QFormLayout(placement_box)
        
        self.element_input = QComboBox()
        self.element_input.addItems(sorted(list(FEATURE_MAP.keys())))
        
        self.x_input = QLineEdit("0.0")
        self.y_input = QLineEdit("0.0")
        self.z_input = QLineEdit("0.0")
        
        self.x_input.textChanged.connect(self.update_preview_atom)
        self.y_input.textChanged.connect(self.update_preview_atom)
        self.z_input.textChanged.connect(self.update_preview_atom)
        self.preview_mesh_item = None
        
        placement_layout.addRow("Element Selection:", self.element_input)
        placement_layout.addRow("Position X (A):", self.x_input)
        placement_layout.addRow("Position Y (A):", self.y_input)
        placement_layout.addRow("Position Z (A):", self.z_input)
        
        add_atom_btn = QPushButton("Inject Atom into Viewport")
        add_atom_btn.setStyleSheet("background-color: #ff7f0e; color: white; font-weight: bold;")
        add_atom_btn.clicked.connect(self.place_custom_atom)
        placement_layout.addRow(add_atom_btn)
        
        right_layout.addWidget(placement_box)
        
        modify_btn = QPushButton("Shift Selected Atom (+0.2 A on X)")
        modify_btn.clicked.connect(self.tweak_crystal_structure)
        right_layout.addWidget(modify_btn)
        
        predict_btn = QPushButton("Run Equivariant Inference")
        predict_btn.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 8px;")
        predict_btn.clicked.connect(self.run_live_inference)
        right_layout.addWidget(predict_btn)
        
        output_box = QGroupBox("Real-time Evaluation Metrics")
        output_layout = QVBoxLayout(output_box)
        self.result_formula = QLabel("Formula: None Loaded")
        self.result_bg = QLabel("Predicted Band Gap: --")
        self.result_diel = QLabel("Predicted Dielectric: --")
        self.result_ionic = QLabel("Predicted Ionic Conductivity: --")
        self.result_phase = QLabel("Predicted Phase Stability (E above hull): --")
        self.result_window = QLabel("Predicted Stability Window: --")
        self.result_elastic = QLabel("Predicted Elastic Moduli (K, G): --")
        self.result_status = QLabel("Screening Status: Waiting")
        
        output_layout.addWidget(self.result_formula)
        output_layout.addWidget(self.result_bg)
        output_layout.addWidget(self.result_diel)
        output_layout.addWidget(self.result_ionic)
        output_layout.addWidget(self.result_phase)
        output_layout.addWidget(self.result_window)
        output_layout.addWidget(self.result_elastic)
        output_layout.addWidget(self.result_status)
        right_layout.addWidget(output_box)
        
        right_layout.addStretch()

    def handle_cif_import(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Structure Target", "", "CIF Files (*.cif)")
        if file_path:
            try:
                structure = Structure.from_file(file_path)
                self.current_formula = structure.composition.reduced_formula
                self.atom_elements = [site.specie.symbol for site in structure]
                self.atom_positions = np.array([site.coords for site in structure], dtype=np.float32)
                
                self.refresh_atom_list_ui()
                self.update_3d_canvas()
                self.result_formula.setText(f"Formula: {self.current_formula}")
            except Exception as e:
                self.result_formula.setText(f"File Load Error: {str(e)}")

    def place_custom_atom(self):
        try:
            element = self.element_input.currentText()
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())
            new_pos = np.array([[x, y, z]], dtype=np.float32)
            
            self.atom_elements.append(element)
            if len(self.atom_positions) == 0:
                self.atom_positions = new_pos
            else:
                self.atom_positions = np.vstack([self.atom_positions, new_pos])
                
            self.refresh_atom_list_ui()
            self.update_3d_canvas()
            
            self.x_input.setText("0.0")
            self.y_input.setText("0.0")
            self.z_input.setText("0.0")
            if self.preview_mesh_item in self.canvas.items:
                self.canvas.removeItem(self.preview_mesh_item)
                
        except ValueError:
            print("Numerical parsing execution error inside coordinate inputs.")

    def refresh_atom_list_ui(self):
        self.atom_list.clear()
        for idx, element in enumerate(self.atom_elements):
            pos = self.atom_positions[idx]
            self.atom_list.addItem(f"[{idx}] {element} | Coordinates: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    def update_3d_canvas(self):
        for item in list(self.canvas.items):
            if isinstance(item, (gl.GLMeshItem, gl.GLLinePlotItem)):
                if item == self.preview_mesh_item:
                    continue
                self.canvas.removeItem(item)
                
        if len(self.atom_positions) == 0:
            return

        for idx, pos in enumerate(self.atom_positions):
            element = self.atom_elements[idx]
            traits = FEATURE_MAP.get(element, [1, 2.2, 25])
            raw_radius = float(traits[2])
            radius_scale = 0.035 * np.sqrt(raw_radius)
            
            color_map = {
                'Mg': (0.15, 0.55, 0.85, 1.0), 
                'H': (0.9, 0.9, 0.9, 1.0)
            }
            atom_color = color_map.get(element, (0.5, 0.5, 0.5, 1.0))
            
            sphere_mesh = gl.MeshData.sphere(rows=12, cols=24, radius=radius_scale)
            mesh_item = gl.GLMeshItem(meshdata=sphere_mesh, smooth=True, color=atom_color, shader='shaded')
            mesh_item.translate(*pos)
            self.canvas.addItem(mesh_item)

        min_coords = np.min(self.atom_positions, axis=0) - 0.4
        max_coords = np.max(self.atom_positions, axis=0) + 0.4
        
        corners = [
            [min_coords[0], min_coords[1], min_coords[2]], [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]], [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]], [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]], [min_coords[0], max_coords[1], max_coords[2]]
        ]
        box_edges = [0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7]
        for start, end in zip(box_edges[::2], box_edges[1::2]):
            pts = np.array([corners[start], corners[end]])
            box_line = gl.GLLinePlotItem(pos=pts, color=(0.5, 0.5, 0.5, 0.4), width=1)
            self.canvas.addItem(box_line)

        bond_threshold = 3.0
        num_atoms = len(self.atom_positions)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = np.linalg.norm(self.atom_positions[i] - self.atom_positions[j])
                if dist <= bond_threshold:
                    points = np.array([self.atom_positions[i], self.atom_positions[j]])
                    line_item = gl.GLLinePlotItem(pos=points, color=(0.65, 0.65, 0.65, 0.9), width=2, antialias=True)
                    self.canvas.addItem(line_item)

    def tweak_crystal_structure(self):
        if len(self.atom_positions) == 0: return
        selected = self.atom_list.currentRow()
        if selected == -1: selected = 0
        
        self.atom_positions[selected][0] += 0.2
        self.refresh_atom_list_ui()
        self.update_3d_canvas()

    def run_live_inference(self):
        if len(self.atom_positions) == 0:
            self.result_status.setText("Screening Status: No structure loaded")
            return
        if self.model is None:
            self.result_status.setText("Screening Status: Model not loaded")
            return

        num_atoms = len(self.atom_elements)

        # 1. Node features: RAW values, no normalization -- matches
        #    dataset_builder.py exactly (the model was trained on raw
        #    [atomic_number, electronegativity, covalent_radius]).
        node_features = []
        for element in self.atom_elements:
            traits = FEATURE_MAP.get(element, [1, 2.2, 25])
            node_features.append([float(traits[0]), float(traits[1]), float(traits[2])])
        x = torch.tensor(node_features, dtype=torch.float)

        # 2. Positions tensor -- required directly by the equivariant model
        #    (distances/directions are computed internally from `pos`).
        pos = torch.tensor(self.atom_positions, dtype=torch.float)

        # 3. Edges: cutoff of 4.5 A, matching dataset_builder.py's EDGE_CUTOFF
        #    (NOT the old 3.0 A bond_threshold used for the visual bonds).
        EDGE_CUTOFF = 4.5
        edge_list = []
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue
                dist = np.linalg.norm(self.atom_positions[i] - self.atom_positions[j])
                if dist <= EDGE_CUTOFF:
                    edge_list.append([i, j])

        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        batch = torch.zeros(num_atoms, dtype=torch.long)

        try:
            with torch.no_grad():
                preds = self.model(x, edge_index, pos, batch)

                # -- band_gap: raw value, no normalization applied at train time --
                predicted_bg = float(preds["band_gap"].item())

                # -- log-space heads: value = exp(pred * sigma + mu) - 1e-6 --
                def decode_log(key, idx=None):
                    stats = self.norm_stats.get(key)
                    if stats is None:
                        return None
                    mu, sigma = stats['mu'], stats['sigma']
                    raw = preds[key]
                    val = raw if idx is None else raw[:, idx:idx+1]
                    return float(torch.exp(val * sigma + mu).item() - 1e-6)

                predicted_diel = decode_log("dielectric")
                predicted_ionic = decode_log("ionic_conductivity")
                predicted_k = decode_log("elastic_moduli", idx=0)
                predicted_g = decode_log("elastic_moduli", idx=1)

                # -- phase_stability: raw (energy above hull, eV), no normalization --
                predicted_phase = float(preds["phase_stability"].item())

                # -- stability_window: gated. Only trust it if the auxiliary
                #    classifier says a real (nonzero) window exists. --
                gate_prob = torch.sigmoid(preds["stability_gate"]).item()
                has_window = gate_prob > 0.5
                if has_window:
                    window = preds["stability_window"][0]
                    v_low, v_high = float(window[0].item()), float(window[1].item())
                else:
                    v_low, v_high = None, None

        except Exception as e:
            print(f"Model Forward Pass Error: {str(e)}")
            self.result_status.setText(f"Error: {str(e)}")
            return

        self.result_bg.setText(f"Predicted Band Gap: {predicted_bg:.4f} eV")
        self.result_diel.setText(
            f"Predicted Dielectric: {predicted_diel:.4f}" if predicted_diel is not None else "Predicted Dielectric: N/A"
        )
        self.result_ionic.setText(
            f"Predicted Ionic Conductivity: {predicted_ionic:.6e} S/cm" if predicted_ionic is not None else "Predicted Ionic Conductivity: N/A"
        )
        self.result_phase.setText(f"Predicted Phase Stability (E above hull): {predicted_phase:.4f} eV")
        if has_window:
            self.result_window.setText(f"Predicted Stability Window: {v_low:.3f} V - {v_high:.3f} V (gate: {gate_prob:.2f})")
        else:
            self.result_window.setText(f"Predicted Stability Window: None predicted (gate: {gate_prob:.2f})")
        if predicted_k is not None and predicted_g is not None:
            self.result_elastic.setText(f"Predicted Elastic Moduli: K={predicted_k:.2f} GPa, G={predicted_g:.2f} GPa")
        else:
            self.result_elastic.setText("Predicted Elastic Moduli: N/A")

        is_candidate = bool(
            predicted_bg > 3.0
            and predicted_diel is not None and predicted_diel > 10.0
        )
        self.result_status.setText(f"Screening Status: {'PASSED CANDIDATE' if is_candidate else 'REJECTED'}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChemAIDesktopApp()
    window.show()
    sys.exit(app.exec())
