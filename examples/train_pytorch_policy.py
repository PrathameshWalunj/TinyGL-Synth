"""
Train a simple CNN policy to predict cube position from RGB images.
Uses tinygl-synth for unlimited synthetic training data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from tinygl_synth import Context, DomainRandomizer

class SyntheticVisionDataset(Dataset):
    """PyTorch dataset using tinygl-synth for on-the-fly data generation."""
    
    def __init__(self, num_samples=10000, width=128, height=128, seed=42):
        self.num_samples = num_samples
        self.ctx = Context(width, height)
        self.randomizer = DomainRandomizer(seed)
        
        self.base_vertices = np.array([
            [-0.1, -0.1, -0.1,  0,  0, -1],
            [ 0.1, -0.1, -0.1,  0,  0, -1],
            [ 0.1,  0.1, -0.1,  0,  0, -1],
            [-0.1,  0.1, -0.1,  0,  0, -1],
            [-0.1, -0.1,  0.1,  0,  0,  1],
            [ 0.1, -0.1,  0.1,  0,  0,  1],
            [ 0.1,  0.1,  0.1,  0,  0,  1],
            [-0.1,  0.1,  0.1,  0,  0,  1],
        ], dtype=np.float32)
        
        self.indices = np.array([
            0,1,2, 2,3,0, 4,5,6, 6,7,4,
            0,4,7, 7,3,0, 1,5,6, 6,2,1,
            0,1,5, 5,4,0, 3,2,6, 6,7,3,
        ], dtype=np.uint32)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Randomize cube position
        cube_pos = self.randomizer.rng.uniform([-0.5, -0.5, 0.0], [0.5, 0.5, 0.5])
        bg_color = self.randomizer.randomize_background()
        
        # Translate vertices
        vertices = self.base_vertices.copy()
        vertices[:, :3] += cube_pos
        
        # Randomize camera
        cam_z = self.randomizer.rng.uniform(-2.5, -1.5)
        view_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, cam_z, 1],
        ], dtype=np.float32)
        
        # Render
        self.ctx.clear(*bg_color, 1.0)
        self.ctx.set_camera(60.0, 0.01, 10.0, view_matrix)
        self.ctx.add_mesh(vertices, self.indices, object_id=1)
        self.ctx.render()
        
        rgb = self.ctx.rgb_tensor()
        
        # Normalize to [0, 1] and convert to CHW format
        rgb = torch.from_numpy(rgb).float() / 255.0
        rgb = rgb.permute(2, 0, 1)  # HWC -> CHW
        
        # Target: XY position only (ignore Z for simplicity)
        target = torch.tensor(cube_pos[:2], dtype=torch.float32)
        
        return rgb, target

class SimpleCNN(nn.Module):
    """Simple CNN to predict cube XY position from RGB image."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Predict XY
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
    
    
        return x
    



def debug_save_batch(dataloader, output_dir="debug_frames"):
    Path(output_dir).mkdir(exist_ok=True)
    rgb, target = next(iter(dataloader))
    
    # Convert back to uint8 image for saving
    # (B, C, H, W) -> (B, H, W, C)
    imgs = rgb.permute(0, 2, 3, 1).numpy() * 255.0
    imgs = imgs.astype(np.uint8)
    
    for i in range(min(5, len(imgs))):
        # CV2 uses BGR, PyTorch uses RGB
        img_bgr = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
        
        # Draw the ground truth position as text
        label = f"X:{target[i][0]:.2f} Y:{target[i][1]:.2f}"
        cv2.putText(img_bgr, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.imwrite(f"{output_dir}/sample_{i}.png", img_bgr)
    
    print(f"Saved 5 debug frames to {output_dir}/")

def train():
    print("=" * 60)
    print("Training Vision Policy with tinygl-synth")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create dataset and dataloader
    train_dataset = SyntheticVisionDataset(num_samples=5000, width=128, height=128)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    debug_save_batch(train_loader)
    
    # Model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (rgb, target) in enumerate(train_loader):
            rgb = rgb.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(rgb)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.6f}\n")
    
    # Test on a few examples
    print("=" * 60)
    print("Testing trained model")
    print("=" * 60)
    
    model.eval()
    test_dataset = SyntheticVisionDataset(num_samples=10, width=128, height=128, seed=999)
    
    with torch.no_grad():
        for i in range(5):
            rgb, target = test_dataset[i]
            rgb = rgb.unsqueeze(0).to(device)
            prediction = model(rgb).cpu().squeeze()
            
            error = torch.abs(prediction - target).mean().item()
            print(f"Sample {i}: Target={target.numpy()}, Pred={prediction.numpy()}, Error={error:.4f}")
    
    print("\n" + "=" * 60)
    print(" Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    train()