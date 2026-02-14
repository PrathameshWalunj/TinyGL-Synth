"""
 Object Pose Estimation

"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import sys
import time
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from tinygl_synth import Context, DomainRandomizer


class PoseDataset(torch.utils.data.Dataset):
    """
    Generate images of an arrow/pointer at known rotations.
    Task: Predict rotation angle from image.
    
    """
    
    def __init__(self, num_samples=10000, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size
        self.ctx = Context(img_size, img_size)
        self.rand = DomainRandomizer()
    
    def _make_arrow(self, angle_rad):
        """Create an arrow mesh pointing in direction of angle."""
        # Arrow shape: triangle pointing in direction
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Arrow vertices (base triangle + head)
        size = 0.25
        
        # Rotate base points
        def rotate(x, y):
            return x * cos_a - y * sin_a, x * sin_a + y * cos_a
        
        # Arrow body
        p0 = rotate(-size, -size * 0.3)  # Back left
        p1 = rotate(-size, size * 0.3)   # Back right
        p2 = rotate(size * 0.5, 0)        # Tip
        
        vertices = np.array([
            [p0[0], p0[1], 0.7,  0, 0, 1, 0.9, 0.3, 0.2],  # Red arrow
            [p1[0], p1[1], 0.7,  0, 0, 1, 0.9, 0.3, 0.2],
            [p2[0], p2[1], 0.7,  0, 0, 1, 0.9, 0.3, 0.2],
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2], dtype=np.uint32)
        return vertices, indices
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random rotation angle [0, 2*pi)
        angle = self.rand.rng.uniform(0, 2 * math.pi)
        
        # Create arrow at this angle
        vertices, indices = self._make_arrow(angle)
        
        # Randomize background
        bg = self.rand.randomize_background()
        self.ctx.clear(*bg, 1.0)
        
        # Fixed camera
        view = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -1.5, 1],
        ], dtype=np.float32)
        self.ctx.set_camera(60.0, 0.01, 5.0, view)
        
        self.ctx.add_mesh_colored(vertices, indices, object_id=1)
        self.ctx.render()
        
        rgb = self.ctx.rgb_tensor()
        
       
      
        target = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        
        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        y = torch.from_numpy(target)
        
        return rgb_t, y, angle  # Also return raw angle for visualization


class PoseNet(nn.Module):
    """Simple CNN to predict pose (cos, sin) from image."""
    
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
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # cos, sin
        )
    
    def forward(self, x):
        return self.fc(self.conv(x))


def angle_error(pred_cos_sin, true_angle):
    """Compute angular error in degrees."""
    pred_angle = math.atan2(pred_cos_sin[1], pred_cos_sin[0])
    error = abs(pred_angle - true_angle)
    # Handle wraparound
    if error > math.pi:
        error = 2 * math.pi - error
    return math.degrees(error)


def main():
    print("=" * 70)
    print("  POSE ESTIMATION ")
    print("=" * 70)
    print("\n  Task: Predict arrow rotation angle from image")

    
    IMG_SIZE = 64
    
    # ==================== Training ====================
    print("[1/3] Training on synthetic data...")
    start = time.perf_counter()
    
    dataset = PoseDataset(num_samples=8000, img_size=IMG_SIZE)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = PoseNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(8):
        total_loss = 0
        for rgb, target, _ in loader:
            optimizer.zero_grad()
            pred = model(rgb)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}/8: Loss = {total_loss/len(loader):.6f}")
    
    train_time = time.perf_counter() - start
    print(f"    Training completed in {train_time:.1f}s")
    
    # ==================== Testing ====================
    print("\n[2/3] Testing pose prediction accuracy...")
    model.eval()
    
    test_dataset = PoseDataset(num_samples=100, img_size=IMG_SIZE)
    errors = []
    
    for i in range(100):
        rgb, target, true_angle = test_dataset[i]
        with torch.no_grad():
            pred = model(rgb.unsqueeze(0)).squeeze().numpy()
        error = angle_error(pred, true_angle)
        errors.append(error)
    
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"    Mean angle error: {mean_error:.2f}°")
    print(f"    Max angle error:  {max_error:.2f}°")
    print(f"    Accuracy (<10°):  {100 * np.mean(np.array(errors) < 10):.0f}%")
    print(f"    Accuracy (<5°):   {100 * np.mean(np.array(errors) < 5):.0f}%")
    
    # ==================== Visualization ====================
    print("\n[3/3] Creating visualization...")
    
    VIS_SIZE = 64  # Same as training
    vis_dataset = PoseDataset(num_samples=25, img_size=VIS_SIZE)
    
    GRID = 5
    grid_img = np.zeros((VIS_SIZE * GRID, VIS_SIZE * GRID, 3), dtype=np.uint8)
    
    for i in range(25):
        rgb, target, true_angle = vis_dataset[i]
        
        with torch.no_grad():
            pred = model(rgb.unsqueeze(0)).squeeze().numpy()
        
        pred_angle = math.atan2(pred[1], pred[0])
        error = angle_error(pred, true_angle)
        
        # Get raw image
        img = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Draw prediction (blue line from center)
        cx, cy = VIS_SIZE // 2, VIS_SIZE // 2
        line_len = 40
        
        # True angle (green)
        ex_true = int(cx + line_len * math.cos(true_angle))
        ey_true = int(cy - line_len * math.sin(true_angle))  # Y flipped
        cv2.line(img, (cx, cy), (ex_true, ey_true), (0, 255, 0), 2)
        
        # Predicted angle (blue)
        ex_pred = int(cx + line_len * math.cos(pred_angle))
        ey_pred = int(cy - line_len * math.sin(pred_angle))
        cv2.line(img, (cx, cy), (ex_pred, ey_pred), (0, 100, 255), 2)
        
        # Error text
        cv2.putText(img, f"{error:.1f}deg", (5, 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        row, col = i // GRID, i % GRID
        y0, y1 = row * VIS_SIZE, (row + 1) * VIS_SIZE
        x0, x1 = col * VIS_SIZE, (col + 1) * VIS_SIZE
        grid_img[y0:y1, x0:x1] = img
    
    # Add legend
    legend = np.zeros((50, grid_img.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, "GREEN = True Angle | BLUE = Predicted | Numbers = Error in degrees", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    stats = np.zeros((40, grid_img.shape[1], 3), dtype=np.uint8)
    stats_text = f"tinygl-synth | 8000 samples | {train_time:.1f}s | Mean Error: {mean_error:.2f}deg | {100*np.mean(np.array(errors)<5):.0f}% within 5deg"
    cv2.putText(stats, stats_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1)
    
    final = np.vstack([legend, grid_img, stats])
    
    output_path = "samples/pose_estimation_demo.png"
    cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    
    print(f"\n    Output: {output_path}")
    
    # ==================== Results ====================
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"    Training samples:  8000 (synthetic)")
    print(f"    Training time:     {train_time:.1f}s")
    print(f"    Mean error:        {mean_error:.2f}°")
    print(f"    <5° accuracy:      {100*np.mean(np.array(errors)<5):.0f}%")
    print("\n" + "=" * 70)
    print("  The network learned pose estimation from synthetic data.")
    print("=" * 70)


if __name__ == "__main__":
    main()
