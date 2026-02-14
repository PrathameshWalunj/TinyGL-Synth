"""
MATH DEMO: Visual Geometry Problem Solving

Train a network to predict geometric properties from images:
- Input: Image of a 3D polyhedron
- Output: Number of faces


"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from tinygl_synth import Context, DomainRandomizer

# ==== Polyhedra Definitions ====
# Each polyhedron: (name, num_faces, vertices, indices)

def make_tetrahedron():
    """4 faces, 4 vertices"""
    s = 0.4
    vertices = np.array([
        [0, s, 0,      0, 1, 0],      # Top
        [-s, -s/2, s,  -1, -1, 1],    # Base front-left
        [s, -s/2, s,    1, -1, 1],    # Base front-right  
        [0, -s/2, -s,   0, -1, -1],   # Base back
    ], dtype=np.float32)
    indices = np.array([
        0,1,2,  # Front
        0,2,3,  # Right
        0,3,1,  # Left
        1,3,2,  # Bottom
    ], dtype=np.uint32)
    return 4, vertices, indices

def make_cube():
    """6 faces, 8 vertices"""
    s = 0.25
    vertices = np.array([
        [-s, -s, -s,  0, 0, -1],
        [ s, -s, -s,  0, 0, -1],
        [ s,  s, -s,  0, 0, -1],
        [-s,  s, -s,  0, 0, -1],
        [-s, -s,  s,  0, 0,  1],
        [ s, -s,  s,  0, 0,  1],
        [ s,  s,  s,  0, 0,  1],
        [-s,  s,  s,  0, 0,  1],
    ], dtype=np.float32)
    indices = np.array([
        0,1,2, 2,3,0,  # Front
        4,5,6, 6,7,4,  # Back
        0,4,7, 7,3,0,  # Left
        1,5,6, 6,2,1,  # Right
        0,1,5, 5,4,0,  # Bottom
        3,2,6, 6,7,3,  # Top
    ], dtype=np.uint32)
    return 6, vertices, indices

def make_octahedron():
    """8 faces, 6 vertices"""
    s = 0.35
    vertices = np.array([
        [0, s, 0,    0, 1, 0],   # Top
        [0, -s, 0,   0, -1, 0],  # Bottom
        [s, 0, 0,    1, 0, 0],   # Right
        [-s, 0, 0,  -1, 0, 0],   # Left
        [0, 0, s,    0, 0, 1],   # Front
        [0, 0, -s,   0, 0, -1],  # Back
    ], dtype=np.float32)
    indices = np.array([
        0,4,2,  0,2,5,  0,5,3,  0,3,4,  # Top 4
        1,2,4,  1,5,2,  1,3,5,  1,4,3,  # Bottom 4
    ], dtype=np.uint32)
    return 8, vertices, indices

POLYHEDRA = [
    make_tetrahedron,  # 4 faces
    make_cube,         # 6 faces  
    make_octahedron,   # 8 faces
]

FACE_COUNTS = [4, 6, 8]  # For classification

class GeometryDataset(torch.utils.data.Dataset):
    """Generate images of polyhedra, predict number of faces."""
    
    def __init__(self, num_samples=5000, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size
        self.ctx = Context(img_size, img_size)
        self.rand = DomainRandomizer(rng_seed=42)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Pick random polyhedron
        poly_idx = self.rand.rng.integers(0, len(POLYHEDRA))
        num_faces, vertices, indices = POLYHEDRA[poly_idx]()
        
        # Random rotation
        angle_x = self.rand.rng.uniform(0, 2 * math.pi)
        angle_y = self.rand.rng.uniform(0, 2 * math.pi)
        
        cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
        cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
        
        # Apply rotation to vertices
        rotated = vertices.copy()
        for i in range(len(vertices)):
            x, y, z = vertices[i, 0], vertices[i, 1], vertices[i, 2]
            # Rotate around X
            y2 = y * cos_x - z * sin_x
            z2 = y * sin_x + z * cos_x
            # Rotate around Y
            x3 = x * cos_y + z2 * sin_y
            z3 = -x * sin_y + z2 * cos_y
            rotated[i, :3] = [x3, y2, z3]
        
        # Random background
        bg = self.rand.randomize_background()
        self.ctx.clear(*bg, 1.0)
        
        # Camera
        view = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1.5, 1],
        ], dtype=np.float32)
        self.ctx.set_camera(55.0, 0.01, 10.0, view)
        
        self.ctx.add_mesh(rotated, indices, object_id=1)
        self.ctx.render()
        
        rgb = self.ctx.rgb_tensor()
        
        # Convert to tensor
        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        
        # Target: class index (0=4faces, 1=6faces, 2=8faces)
        target = torch.tensor(poly_idx, dtype=torch.long)
        
        return rgb_t, target, num_faces

class GeometryNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def main():
    print("=" * 65)
    print("  MATH DEMO: Visual Geometry Problem Solving")
    print("=" * 65)
    print("\n  Task: Predict number of faces from polyhedron image")
    print("  Shapes: Tetrahedron(4), Cube(6), Octahedron(8)\n")
    
    # Dataset
    train_dataset = GeometryDataset(num_samples=3000)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Model
    model = GeometryNet(num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print("  Training...")
    for epoch in range(5):
        correct = 0
        total = 0
        for x, y, _ in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
        
        acc = 100 * correct / total
        print(f"    Epoch {epoch+1}: Accuracy = {acc:.1f}%")
    
    # Test
    print("\n  Testing on new shapes...")
    model.eval()
    test_dataset = GeometryDataset(num_samples=100)
    
    correct = 0
    results = []
    for i in range(100):
        x, y, num_faces = test_dataset[i]
        pred = model(x.unsqueeze(0)).argmax(1).item()
        pred_faces = FACE_COUNTS[pred]
        
        is_correct = (pred_faces == num_faces)
        if is_correct:
            correct += 1
        
        if i < 10:
            results.append((num_faces, pred_faces, is_correct))
    
    print(f"\n  Test Accuracy: {correct}%")
    print("\n  Sample predictions (True → Pred):")
    for true, pred, ok in results:
        mark = "✓" if ok else "✗"
        print(f"    {true} faces → {pred} faces {mark}")
    
    # Create visualization
    print("\n  Creating visualization...")
    VIS_SIZE = 96
    vis_ctx = Context(VIS_SIZE, VIS_SIZE)
    
    grid = np.zeros((VIS_SIZE * 3, VIS_SIZE * 3, 3), dtype=np.uint8)
    labels = []
    
    for i, make_poly in enumerate(POLYHEDRA):
        for j in range(3):
            num_faces, vertices, indices = make_poly()
            
            # Random rotation
            angle = (j * 40 + i * 20) * math.pi / 180
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rotated = vertices.copy()
            for k in range(len(vertices)):
                x, z = vertices[k, 0], vertices[k, 2]
                rotated[k, 0] = x * cos_a - z * sin_a
                rotated[k, 2] = x * sin_a + z * cos_a
            
            vis_ctx.clear(30, 35, 50, 1.0)
            view = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,-1.5,1]], dtype=np.float32)
            vis_ctx.set_camera(55.0, 0.01, 10.0, view)
            vis_ctx.add_mesh(rotated, indices, object_id=1)
            vis_ctx.render()
            
            rgb = vis_ctx.rgb_tensor()
            
            # Add label
            cv2.putText(rgb, f"{num_faces} faces", (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            row, col = i, j
            grid[row*VIS_SIZE:(row+1)*VIS_SIZE, col*VIS_SIZE:(col+1)*VIS_SIZE] = rgb
    
    output = "samples/geometry_math_demo.png"
    cv2.imwrite(output, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {output}")
    
    print("\n" + "=" * 65)
    print("  The network learned to recognize polyhedra from images!")
    print("  Application: Math education, geometry problem solving, 3D understanding")
    print("=" * 65)

if __name__ == "__main__":
    main()
