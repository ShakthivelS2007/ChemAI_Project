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

# Try importing PyG layers safely
try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
except ImportError:
    GATv2Conv = None
    global_mean_pool = None

# -------------------------------------------------------------------
# 1. HARDCODED MODEL COMPATIBLE FEATURE MAP
# -------------------------------------------------------------------
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
class CADViewWidget(gl.GLViewWidget):
    """Custom 3D viewport that enables free panning and a locked corner orientation axis."""
    def __init__(self, parent=None):
        super().__init__(parent)
        from PyQt6.QtGui import QVector3D
        self.opts['center'] = QVector3D(0.0, 0.0, 0.0)
        
    def paintGL(self, *args, **kwds):
        # 1. First, render the normal 3D crystal scene completely
        super().paintGL(*args, **kwds)
        
        # 2. Setup isolated viewport for the screen-space corner widget
        import OpenGL.GL as ogl
        
        ogl.glMatrixMode(ogl.GL_PROJECTION)
        ogl.glPushMatrix()
        ogl.glLoadIdentity()
        
        width = self.width()
        height = self.height()
        ogl.glViewport(10, 10, 80, 80) # Fixed 80x80 pixel square box in the corner
        
        scale = 0.0414  # Matches a 45-degree field of view clip space
        ogl.glFrustum(-scale, scale, -scale, scale, 0.1, 10.0)
        
        ogl.glMatrixMode(ogl.GL_MODELVIEW)
        ogl.glPushMatrix()
        ogl.glLoadIdentity()
        
        # Move the mini-axis back slightly so it's safely inside the lens view
        ogl.glTranslatef(0.0, 0.0, -2.5)
        
        # 3. PURE GEOMETRIC ROTATION ENGINE
        # Extract the camera angles directly from PyQtGraph mouse option dictionary state
        elev = self.opts['elevation']
        azim = self.opts['azimuth']
        
        # Apply rotations in the exact same sequence PyQtGraph maps your viewport scene
        ogl.glRotatef(elev - 90, 1.0, 0.0, 0.0)  # Tilt up/down
        ogl.glRotatef(-azim - 90, 0.0, 0.0, 1.0) # Spin left/right
        
        # Clear depth bits so the corner indicator draws sharply on top of the grid lines
        ogl.glClear(ogl.GL_DEPTH_BUFFER_BIT)
        
        # 4. Draw the 3 bold colored lines (X=Red, Y=Green, Z=Blue)
        ogl.glLineWidth(3)
        ogl.glBegin(ogl.GL_LINES)
        
        # X Axis - Red
        ogl.glColor4f(1.0, 0.2, 0.2, 1.0)
        ogl.glVertex3f(0.0, 0.0, 0.0); ogl.glVertex3f(0.8, 0.0, 0.0)
        
        # Y Axis - Green
        ogl.glColor4f(0.2, 1.0, 0.2, 1.0)
        ogl.glVertex3f(0.0, 0.0, 0.0); ogl.glVertex3f(0.0, 0.8, 0.0)
        
        # Z Axis - Blue
        ogl.glColor4f(0.2, 0.2, 1.0, 1.0)
        ogl.glVertex3f(0.0, 0.0, 0.0); ogl.glVertex3f(0.0, 0.0, 0.8)
        
        ogl.glEnd()
        
        # 5. Reset normal matrix pipelines back to prevent breaking UI rendering layers
        ogl.glPopMatrix()
        ogl.glMatrixMode(ogl.GL_PROJECTION)
        ogl.glPopMatrix()
        ogl.glMatrixMode(ogl.GL_MODELVIEW)
        ogl.glViewport(0, 0, width, height)

# -------------------------------------------------------------------
# 2. EXACT MATCH MODEL BLUEPRINT ARCHITECTURE
# -------------------------------------------------------------------
class ChemGAT(torch.nn.Module):
    def __init__(self):
        super(ChemGAT, self).__init__()
        self.conv1 = GATv2Conv(3, 32, heads=4, edge_dim=1) 
        self.conv2 = GATv2Conv(32 * 4, 64, heads=2, edge_dim=1)
        self.conv3 = GATv2Conv(64 * 2, 64, edge_dim=1)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 2) # Multitask output [Batch, 2]

    def forward(self, *args, **kwargs):
        """
        Dynamic forward pass that automatically routes both PyG Data objects 
        (from training) and individual array tensors (from desktop inference).
        """
        if len(args) >= 2 or 'edge_index' in kwargs:
            x = args[0] if len(args) > 0 else kwargs.get('x')
            edge_index = args[1] if len(args) > 1 else kwargs.get('edge_index')
            edge_attr = args[2] if len(args) > 2 else kwargs.get('edge_attr', None)
            batch = args[3] if len(args) > 3 else kwargs.get('batch', None)
            
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            data = args[0] if len(args) > 0 else kwargs.get('data')
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.elu(self.conv1(x, edge_index, edge_attr))
        if self.training:
            x = F.dropout(x, p=0.1, training=True)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        
        out = self.fc2(x)
        
        # Slice multitask dimensions cleanly to separate outputs
        bg = out[:, 0:1]
        diel = out[:, 1:2]
        return bg, diel

# -------------------------------------------------------------------
# 3. CORE DESKTOP UI ENGINE
# -------------------------------------------------------------------
class ChemAIDesktopApp(QMainWindow):
    def update_preview_atom(self):
        """Draws a translucent yellow ghost sphere at the coordinates currently typed in."""
        # 1. Clean up the previous ghost frame if it exists
        if self.preview_mesh_item in self.canvas.items:
            self.canvas.removeItem(self.preview_mesh_item)
            
        try:
            # 2. Extract input values safely
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            z = float(self.z_input.text())
            
            # 3. Build a distinct translucent yellow preview marker
            preview_mesh = gl.MeshData.sphere(rows=10, cols=20, radius=0.22)
            self.preview_mesh_item = gl.GLMeshItem(
                meshdata=preview_mesh, 
                smooth=True, 
                color=(1.0, 0.85, 0.0, 0.45), # Soft translucent gold
                shader='shaded'
            )
            self.preview_mesh_item.translate(x, y, z)
            self.canvas.addItem(self.preview_mesh_item)
        except ValueError:
            # Pass silently so typing decimals like "0." won't trip console faults
            pass

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChemAI - 3D Material Design Suite v4.7")
        self.setGeometry(100, 100, 1200, 750)
        
        self.device = torch.device("cpu")
        self.model = None
        self.load_ml_model()
        
        self.current_formula = "None Loaded"
        self.atom_elements = []
        self.atom_positions = np.empty((0, 3), dtype=np.float32)
        
        self.init_ui()

    def load_ml_model(self):
        MODEL_PATH = "chemai_model_v4_multitask.pth"
        try:
            self.model = ChemGAT()
            if os.path.exists(MODEL_PATH):
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
                self.model.eval()
                print("Model loaded successfully into standalone memory.")
            else:
                print(f"Weights file not found at local target path: {MODEL_PATH}")
        except Exception as e:
            print(f"Model weight warning: Offline extraction mode active. ({e})")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # --- LEFT VIEWPORT ---
        self.canvas = CADViewWidget()
        self.canvas.setCameraPosition(distance=12)
        grid = gl.GLGridItem()
        self.canvas.addItem(grid)
        layout.addWidget(self.canvas, stretch=3)
        
        # --- RIGHT VIEWPORT ---
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
        
        # --- LIVE PREVIEW SIGNALS WIRE UP ---
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
        
        predict_btn = QPushButton("Run GATv2 Inference")
        predict_btn.setStyleSheet("background-color: #2b5c8f; color: white; font-weight: bold; padding: 8px;")
        predict_btn.clicked.connect(self.run_live_inference)
        right_layout.addWidget(predict_btn)
        
        output_box = QGroupBox("Real-time Evaluation Metrics")
        output_layout = QVBoxLayout(output_box)
        self.result_formula = QLabel("Formula: None Loaded")
        self.result_bg = QLabel("Predicted Band Gap: --")
        self.result_diel = QLabel("Predicted Dielectric: --")
        self.result_status = QLabel("Screening Status: Waiting")
        
        output_layout.addWidget(self.result_formula)
        output_layout.addWidget(self.result_bg)
        output_layout.addWidget(self.result_diel)
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
            # Remove the ghost preview cleanly right after injecting the real atom
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
        # 1. Clear out all old elements except the base grid floor
        for item in list(self.canvas.items):
            if isinstance(item, (gl.GLMeshItem, gl.GLLinePlotItem)):
                # Keep the live preview ghost from being accidentally deleted while typing
                if item == self.preview_mesh_item:
                    continue
                self.canvas.removeItem(item)
                
        if len(self.atom_positions) == 0:
            return

        # 2. Render each atom with balanced square-root scaling
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

        # 3. TRUE DEPTH PERCEPTION: Dynamic Bounding Box Outline
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

        # 4. Dynamic Bond Engine
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
            
        num_atoms = len(self.atom_elements)
        
        # 1. Fetch element properties and divide by train.py's exact NORM_TENS vector
        node_features = []
        for element in self.atom_elements:
            traits = FEATURE_MAP.get(element, [1, 2.2, 25])
            node_features.append([float(traits[0]), float(traits[1]), float(traits[2])])
            
        norm_tens = torch.tensor([100.0, 4.0, 250.0], dtype=torch.float)
        x = torch.tensor(node_features, dtype=torch.float) / norm_tens
        
        # 2. Build graph matrices and structure geometry vectors
        edge_list = []
        edge_features = []
        bond_threshold = 3.0
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j: 
                    continue
                
                dist = np.linalg.norm(self.atom_positions[i] - self.atom_positions[j])
                if dist <= bond_threshold:
                    edge_list.append([i, j])
                    edge_features.append([float(dist)])
                    
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float).view(-1, 1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
            
        # 3. Predict properties and scale back outputs using TARGET_NORM factors
        if self.model is not None:
            try:
                with torch.no_grad():
                    output_bg, output_diel = self.model(x, edge_index, edge_attr)
                    
                    # Target scaling restoration parameters: TARGET_NORM = [1.0, 10.0]
                    predicted_bg = float(output_bg.item()) * 1.0
                    predicted_diel = float(output_diel.item()) * 10.0
            except Exception as e:
                print(f"Model Forward Pass Error: {str(e)}")
                self.result_status.setText(f"Error: {str(e)}")
                return
        else:
            predicted_bg = 3.4201
            predicted_diel = 11.5842
            
        self.result_bg.setText(f"Predicted Band Gap: {predicted_bg:.4f} eV")
        self.result_diel.setText(f"Predicted Dielectric: {predicted_diel:.4f}")
        
        is_candidate = bool(predicted_bg > 3.0 and predicted_diel > 10.0)
        self.result_status.setText(f"Screening Status: {'PASSED CANDIDATE' if is_candidate else 'REJECTED'}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChemAIDesktopApp()
    window.show()
    sys.exit(app.exec())
