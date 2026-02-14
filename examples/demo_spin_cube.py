"""
Demo: Spinning Cube with Multi-Modal Output

Shows a cube rotating in the center of the screen with:
- RGB rendering
- Depth heatmap  
- Segmentation mask
"""

import numpy as np
import imageio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from tinygl_synth import Context

def main():
    W, H = 192, 192
    ctx = Context(W, H)

    # Define a cube - will rotate it instead of orbiting camera
    def make_cube(angle):
        """Create rotated cube vertices."""
        s = 0.35  # Cube half-size
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Base cube vertices
        base = np.array([
            [-s, -s, -s],
            [ s, -s, -s],
            [ s,  s, -s],
            [-s,  s, -s],
            [-s, -s,  s],
            [ s, -s,  s],
            [ s,  s,  s],
            [-s,  s,  s],
        ])
        
        # Rotate around Y axis
        rotated = base.copy()
        rotated[:, 0] = base[:, 0] * cos_a - base[:, 2] * sin_a
        rotated[:, 2] = base[:, 0] * sin_a + base[:, 2] * cos_a
        
        # Add normals (pointing outward for each face)
        vertices = np.zeros((8, 6), dtype=np.float32)
        vertices[:, :3] = rotated
        # Simple normals
        vertices[:4, 3:] = [0, 0, -1]  # Front face
        vertices[4:, 3:] = [0, 0, 1]   # Back face
        
        return vertices

    indices = np.array([
        0,1,2, 2,3,0,  # Front
        4,5,6, 6,7,4,  # Back
        0,4,7, 7,3,0,  # Left
        1,5,6, 6,2,1,  # Right
        0,1,5, 5,4,0,  # Bottom
        3,2,6, 6,7,3   # Top
    ], dtype=np.uint32)

    # Fixed camera looking at origin
    view = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, -1.5, 1],  # Camera 1.5 units back
    ], dtype=np.float32)

    frames = []

    for i in range(90):
        angle = i * np.pi * 2 / 90  # Full rotation
        
        vertices = make_cube(angle)
        
        ctx.clear(25, 30, 45, 1.0)  # Dark blue background
        ctx.set_camera(55.0, 0.01, 10.0, view)
        ctx.add_mesh(vertices, indices, object_id=1)
        ctx.render()

        rgb = ctx.rgb_tensor()
        depth = ctx.depth_tensor()
        seg = ctx.segmentation_tensor()

        # Depth visualization
        d = depth.copy()
        valid = d < 100
        if valid.any():
            d_min, d_max = d[valid].min(), d[valid].max()
            d_norm = np.clip((d - d_min) / (d_max - d_min + 1e-6), 0, 1)
        else:
            d_norm = np.zeros_like(d)
        depth_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        depth_rgb[:,:,0] = (50 + 180 * (1 - d_norm)).astype(np.uint8)
        depth_rgb[:,:,1] = (80 + 150 * (1 - d_norm)).astype(np.uint8)
        depth_rgb[:,:,2] = (150 + 100 * (1 - d_norm)).astype(np.uint8)

        # Segmentation visualization
        seg_rgb = np.full((H, W, 3), 40, dtype=np.uint8)  # Dark gray bg
        seg_rgb[seg > 0] = [255, 80, 80]  # Red for object

        # Combine horizontally
        panel = np.hstack([rgb, depth_rgb, seg_rgb])
        frames.append(panel)

    output = "samples/spin_cube.gif"
    imageio.mimsave(output, frames, fps=30, loop=0)
    print(f"Saved {output} - {len(frames)} frames")

if __name__ == "__main__":
    main()