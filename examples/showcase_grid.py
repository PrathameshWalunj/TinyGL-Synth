"""
SHOWCASE: Generate 100 unique training samples in a grid.
Demonstrates: Speed + Domain Randomization + Multi-Modal Output
"""
import numpy as np
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from tinygl_synth import Context, DomainRandomizer
import cv2

def make_random_cube(rand, base_size=0.15):
    """Generate a cube with random position and size."""
    pos = rand.rng.uniform([-0.4, -0.4, 0.3], [0.4, 0.4, 0.8])
    size = base_size * rand.rng.uniform(0.5, 1.5)
    s = size / 2
    
    vertices = np.array([
        [pos[0]-s, pos[1]-s, pos[2]-s,  0, 0,-1],
        [pos[0]+s, pos[1]-s, pos[2]-s,  0, 0,-1],
        [pos[0]+s, pos[1]+s, pos[2]-s,  0, 0,-1],
        [pos[0]-s, pos[1]+s, pos[2]-s,  0, 0,-1],
        [pos[0]-s, pos[1]-s, pos[2]+s,  0, 0, 1],
        [pos[0]+s, pos[1]-s, pos[2]+s,  0, 0, 1],
        [pos[0]+s, pos[1]+s, pos[2]+s,  0, 0, 1],
        [pos[0]-s, pos[1]+s, pos[2]+s,  0, 0, 1],
    ], dtype=np.float32)
    
    indices = np.array([
        0,1,2, 2,3,0, 4,5,6, 6,7,4,
        0,4,7, 7,3,0, 1,5,6, 6,2,1,
        0,1,5, 5,4,0, 3,2,6, 6,7,3
    ], dtype=np.uint32)
    
    return vertices, indices, pos

def depth_to_heatmap(depth, seg, near=0.5, far=2.5):
    """Convert depth to colorful heatmap. Background = black."""
    d = depth.copy()
    
    # Normalize valid depth
    d = np.clip(d, near, far)
    d_norm = (d - near) / (far - near)
    d_viz = (1.0 - d_norm) * 255  # Close = bright, far = dark
    d_viz = d_viz.astype(np.uint8)
    
    # Apply colormap
    result = cv2.applyColorMap(d_viz, cv2.COLORMAP_INFERNO)
    
    # Make background black (where seg == 0)
    result[seg == 0] = [0, 0, 0]
    
    return result

def seg_to_color(seg):
    """Convert segmentation to color."""
    out = np.zeros((*seg.shape, 3), dtype=np.uint8)
    out[seg == 0] = [25, 25, 30]  # Background dark
    out[seg > 0] = [0, 255, 100]  # Object green
    return out

def main():
    print("=" * 60)
    print("tinygl-synth SHOWCASE: 100 Unique Samples")
    print("=" * 60)
    
    # Settings
    GRID_SIZE = 10  # 10x10 = 100 samples
    TILE_SIZE = 64
    ctx = Context(TILE_SIZE, TILE_SIZE)
    rand = DomainRandomizer(rng_seed=42)
    
    # Pre-allocate output grids
    rgb_grid = np.zeros((TILE_SIZE * GRID_SIZE, TILE_SIZE * GRID_SIZE, 3), dtype=np.uint8)
    depth_grid = np.zeros((TILE_SIZE * GRID_SIZE, TILE_SIZE * GRID_SIZE, 3), dtype=np.uint8)
    seg_grid = np.zeros((TILE_SIZE * GRID_SIZE, TILE_SIZE * GRID_SIZE, 3), dtype=np.uint8)
    
    print(f"Generating {GRID_SIZE * GRID_SIZE} unique samples...")
    
    start_time = time.perf_counter()
    
    sample_count = 0
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            # Random scene
            bg = rand.randomize_background()
            ctx.clear(*bg, 1.0)
            
            # Random camera
            cam_offset = rand.rng.uniform(-0.2, 0.2, 2)
            view = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [cam_offset[0], cam_offset[1], -2.0, 1],
            ], dtype=np.float32)
            ctx.set_camera(60.0, 0.01, 5.0, view)
            
            # Random object
            vertices, indices, _ = make_random_cube(rand)
            ctx.add_mesh(vertices, indices, object_id=1)
            ctx.render()
            
            # Get outputs
            rgb = ctx.rgb_tensor()
            depth = ctx.depth_tensor()
            seg = ctx.segmentation_tensor()
            
            # Convert for visualization
            depth_viz = depth_to_heatmap(depth, seg)
            seg_viz = seg_to_color(seg)
            
            # Place in grid
            y0, y1 = row * TILE_SIZE, (row + 1) * TILE_SIZE
            x0, x1 = col * TILE_SIZE, (col + 1) * TILE_SIZE
            
            rgb_grid[y0:y1, x0:x1] = rgb
            depth_grid[y0:y1, x0:x1] = depth_viz
            seg_grid[y0:y1, x0:x1] = seg_viz
            
            sample_count += 1
    
    elapsed = time.perf_counter() - start_time
    samples_per_sec = sample_count / elapsed
    
    print(f"\n{'=' * 40}")
    print(f"  Generated: {sample_count} samples")
    print(f"  Time: {elapsed:.3f} seconds")
    print(f"  Speed: {samples_per_sec:.1f} samples/sec")
    print(f"{'=' * 40}\n")
    
    # Combine into one mega image: RGB | Depth | Segmentation
    combined = np.hstack([rgb_grid, depth_grid, seg_grid])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_img = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(label_img, "RGB (Domain Randomized)", (20, 28), font, 0.7, (255,255,255), 2)
    cv2.putText(label_img, "DEPTH (Automatic)", (TILE_SIZE * GRID_SIZE + 20, 28), font, 0.7, (255,255,255), 2)
    cv2.putText(label_img, "SEGMENTATION (Free Ground Truth)", (TILE_SIZE * GRID_SIZE * 2 + 20, 28), font, 0.7, (255,255,255), 2)
    
    # Stats bar
    stats_img = np.zeros((50, combined.shape[1], 3), dtype=np.uint8)
    stats_text = f"tinygl-synth: {sample_count} unique samples | {elapsed:.2f}s | {samples_per_sec:.0f} samples/sec | Zero dependencies | CPU only"
    cv2.putText(stats_img, stats_text, (20, 35), font, 0.65, (0, 255, 200), 2)
    
    final = np.vstack([label_img, combined, stats_img])
    
    # Save
    output_path = "samples/showcase_100_samples.png"
    cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"Saved to: {output_path}")
    print("\nThis image shows 100 unique training samples with:")
    print("  - Randomized backgrounds")
    print("  - Randomized object positions/sizes")
    print("  - Randomized camera positions")
    print("  - Automatic depth maps")
    print("  - Automatic segmentation masks")

if __name__ == "__main__":
    main()
