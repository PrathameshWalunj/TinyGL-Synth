"""
 Train + Visualize Grasp Predictions

"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from tinygl_synth import Context, DomainRandomizer

class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=5000, img_size=128):
        self.num_samples = num_samples
        self.img_size = img_size
        self.ctx = Context(img_size, img_size)
        self.rand = DomainRandomizer()
        self.indices = np.array([
            0,1,2, 2,3,0, 4,5,6, 6,7,4,
            0,4,7, 7,3,0, 1,5,6, 6,2,1,
            0,1,5, 5,4,0, 3,2,6, 6,7,3
        ], dtype=np.uint32)
    
    def _make_cube(self, pos, size=0.12):
        s = size / 2
        return np.array([
            [pos[0]-s, pos[1]-s, pos[2]-s,  0, 0,-1],
            [pos[0]+s, pos[1]-s, pos[2]-s,  0, 0,-1],
            [pos[0]+s, pos[1]+s, pos[2]-s,  0, 0,-1],
            [pos[0]-s, pos[1]+s, pos[2]-s,  0, 0,-1],
            [pos[0]-s, pos[1]-s, pos[2]+s,  0, 0, 1],
            [pos[0]+s, pos[1]-s, pos[2]+s,  0, 0, 1],
            [pos[0]+s, pos[1]+s, pos[2]+s,  0, 0, 1],
            [pos[0]-s, pos[1]+s, pos[2]+s,  0, 0, 1],
        ], dtype=np.float32)
    
    def __len__(self):
        return self.num_samples
    
    def pos_to_pixel(self, pos):
        """Convert 3D position to pixel coordinates (approximate projection)."""
        # Simple perspective projection approximation
        x_pix = int((pos[0] / 1.5 + 0.5) * self.img_size)
        y_pix = int((-pos[1] / 1.5 + 0.5) * self.img_size)  # Y flipped
        return np.clip(x_pix, 0, self.img_size-1), np.clip(y_pix, 0, self.img_size-1)
    
    def __getitem__(self, idx):
        # Random object position
        obj_pos = self.rand.rng.uniform([-0.4, -0.4, 0.5], [0.4, 0.4, 1.0])
        
        vertices = self._make_cube(obj_pos)
        
        view = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -2.0, 1],
        ], dtype=np.float32)
        
        bg = self.rand.randomize_background()
        self.ctx.clear(*bg, 1.0)
        self.ctx.set_camera(60.0, 0.01, 5.0, view)
        self.ctx.add_mesh(vertices, self.indices, object_id=1)
        self.ctx.render()
        
        rgb = self.ctx.rgb_tensor()
        
        rgb_t = torch.from_numpy(rgb).float().permute(2,0,1) / 255.0
        y = torch.tensor(obj_pos[:2], dtype=torch.float32)
        
        return rgb_t, y, rgb.copy()  # Also return raw RGB for visualization

class GraspNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
    
    def forward(self, x):
        return self.fc(self.conv(x))

def pos_to_pixel(pos, img_size):
    x_pix = int((pos[0] / 1.5 + 0.5) * img_size)
    y_pix = int((-pos[1] / 1.5 + 0.5) * img_size)
    return np.clip(x_pix, 0, img_size-1), np.clip(y_pix, 0, img_size-1)

def main():
    IMG_SIZE = 128
    
    print("=" * 60)
    print("SHOWCASE: Train + Visualize Grasp Predictions")
    print("=" * 60)
    
    # Training
    print("\n[1/3] Training on 3000 synthetic samples...")
    start = time.perf_counter()
    
    dataset = GraspDataset(num_samples=3000, img_size=IMG_SIZE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = GraspNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(5):
        total_loss = 0
        for rgb_t, y, _ in loader:
            optimizer.zero_grad()
            pred = model(rgb_t)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}/5: Loss = {total_loss/len(loader):.6f}")
    
    train_time = time.perf_counter() - start
    print(f"    Training completed in {train_time:.1f}s")
    
    # Generate test visualization
    print("\n[2/3] Generating visualization grid...")
    model.eval()
    
    GRID_SIZE = 5
    TILE_SIZE = IMG_SIZE
    vis_grid = np.zeros((TILE_SIZE * GRID_SIZE, TILE_SIZE * GRID_SIZE, 3), dtype=np.uint8)
    
    test_dataset = GraspDataset(num_samples=GRID_SIZE * GRID_SIZE, img_size=IMG_SIZE)
    
    errors = []
    for i in range(GRID_SIZE * GRID_SIZE):
        rgb_t, y_true, rgb_raw = test_dataset[i]
        
        with torch.no_grad():
            y_pred = model(rgb_t.unsqueeze(0)).squeeze().numpy()
        
        y_true_np = y_true.numpy()
        error = np.abs(y_pred - y_true_np).mean()
        errors.append(error)
        
        # Draw on image
        img = rgb_raw.copy()
        
        # True position (green circle)
        tx, ty = pos_to_pixel(y_true_np, IMG_SIZE)
        cv2.circle(img, (tx, ty), 8, (0, 255, 0), 2)
        
        # Predicted position (red cross)
        px, py = pos_to_pixel(y_pred, IMG_SIZE)
        cv2.drawMarker(img, (px, py), (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
        
        # Error text
        cv2.putText(img, f"E:{error:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        row, col = i // GRID_SIZE, i % GRID_SIZE
        y0, y1 = row * TILE_SIZE, (row + 1) * TILE_SIZE
        x0, x1 = col * TILE_SIZE, (col + 1) * TILE_SIZE
        vis_grid[y0:y1, x0:x1] = img
    
    avg_error = np.mean(errors)
    
    # Add legend
    legend = np.zeros((60, vis_grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, "GREEN = Ground Truth", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(legend, "RED = Network Prediction (learned from synthetic data only)", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 80), 1)
    
    # Stats bar
    stats = np.zeros((50, vis_grid.shape[1], 3), dtype=np.uint8)
    stats_text = f"tinygl-synth | 3000 samples | {train_time:.1f}s training | Avg Error: {avg_error:.3f} | CPU only | Zero real data"
    cv2.putText(stats, stats_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
    
    final = np.vstack([legend, vis_grid, stats])
    
    output_path = "samples/showcase_predictions.png"
    cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    
    print(f"\n[3/3] Results saved!")
    print(f"    Output: {output_path}")
    print(f"    Average Error: {avg_error:.4f}")
    print(f"    Training Time: {train_time:.1f}s")
    print("\n" + "=" * 60)
    print("The network learned to localize objects using ONLY synthetic data!")
    print("=" * 60)

if __name__ == "__main__":
    main()
