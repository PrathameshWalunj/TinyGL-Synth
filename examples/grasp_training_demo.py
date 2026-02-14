"""
Train a grasp point predictor using tinygl-synth.
Ground truth grasp points are "free" - we know them from scene generation!
"""
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from tinygl_synth import Context, DomainRandomizer
class GraspDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=5000):
        self.num_samples = num_samples
        self.ctx = Context(64, 64)  # Small for fast training
        self.rand = DomainRandomizer()
        
        # Simple cylinder-like object
        self.indices = np.array([0,1,2, 2,3,0, 4,5,6, 6,7,4], dtype=np.uint32)
    
    def _make_cylinder_vertices(self, pos, height=0.2, radius=0.05):
        """Make a simple 'graspable' cylinder at position."""
        h, r = height/2, radius
        return np.array([
            [pos[0]-r, pos[1]-h, pos[2],  -1, 0, 0],
            [pos[0]-r, pos[1]+h, pos[2],  -1, 0, 0],
            [pos[0]+r, pos[1]+h, pos[2],   1, 0, 0],
            [pos[0]+r, pos[1]-h, pos[2],   1, 0, 0],
            [pos[0], pos[1]-h, pos[2]-r,   0, 0,-1],
            [pos[0], pos[1]+h, pos[2]-r,   0, 0,-1],
            [pos[0], pos[1]+h, pos[2]+r,   0, 0, 1],
            [pos[0], pos[1]-h, pos[2]+r,   0, 0, 1],
        ], dtype=np.float32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random object position (this IS our ground truth grasp point!)
        obj_pos = self.rand.rng.uniform([-0.3, -0.3, 0.5], [0.3, 0.3, 1.0])
        
        # Make object
        vertices = self._make_cylinder_vertices(obj_pos)
        
        # Randomize camera slightly
        cam_jitter = self.rand.rng.uniform(-0.1, 0.1, 2)
        view = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [cam_jitter[0], cam_jitter[1], -2.0, 1],
        ], dtype=np.float32)
        
        # Render
        bg = self.rand.randomize_background()
        self.ctx.clear(*bg, 1.0)
        self.ctx.set_camera(60.0, 0.01, 5.0, view)
        self.ctx.add_mesh(vertices, self.indices, object_id=1)
        self.ctx.render()
        
        # Get RGB + Depth (multi-modal input)
        rgb = self.ctx.rgb_tensor()
        depth = self.ctx.depth_tensor()
        
        # Stack as 4-channel input
        rgb_t = torch.from_numpy(rgb).float().permute(2,0,1) / 255.0
        depth_t = torch.from_numpy(depth).float().unsqueeze(0)
        x = torch.cat([rgb_t, depth_t], dim=0)  # (4, 64, 64)
        
        # Ground truth: XY grasp point (we KNOW it from generation!)
        y = torch.tensor(obj_pos[:2], dtype=torch.float32)
        
        return x, y
class GraspNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),  # 4 channels: RGBD
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # XY grasp point
        )
    
    def forward(self, x):
        return self.fc(self.conv(x))
def main():
    print("=" * 50)
    print("Grasp Point Detection - Synthetic Training Demo")
    print("=" * 50)
    
    # Create dataset
    dataset = GraspDataset(num_samples=2000)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train
    model = GraspNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(3):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(loader):.6f}")
    
    # Test
    print("\nTest predictions:")
    model.eval()
    test_data = GraspDataset(num_samples=5)
    for i in range(5):
        x, y = test_data[i]
        pred = model(x.unsqueeze(0)).detach().squeeze()
        err = torch.abs(pred - y).mean().item()
        print(f"  Sample {i}: True={y.numpy()}, Pred={pred.numpy()}, Err={err:.4f}")
    
    print("\n Training complete!")
if __name__ == "__main__":
    main()