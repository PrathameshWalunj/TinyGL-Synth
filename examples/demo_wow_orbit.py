"""
 Colorful Orbiting Cubes


- Multiple colored objects
- Smooth camera orbit
- RGB + Depth + Segmentation in one view

"""

import numpy as np
import imageio
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from tinygl_synth import Context

W, H = 192, 192
ctx = Context(W, H)

def make_cube_colored(pos, color, size=0.12):
    """Create a colored cube at position."""
    s = size / 2
    r, g, b = color
    return np.array([
        [pos[0]-s, pos[1]-s, pos[2]-s,  0,0,-1, r,g,b],
        [pos[0]+s, pos[1]-s, pos[2]-s,  0,0,-1, r,g,b],
        [pos[0]+s, pos[1]+s, pos[2]-s,  0,0,-1, r,g,b],
        [pos[0]-s, pos[1]+s, pos[2]-s,  0,0,-1, r,g,b],
        [pos[0]-s, pos[1]-s, pos[2]+s,  0,0, 1, r,g,b],
        [pos[0]+s, pos[1]-s, pos[2]+s,  0,0, 1, r,g,b],
        [pos[0]+s, pos[1]+s, pos[2]+s,  0,0, 1, r,g,b],
        [pos[0]-s, pos[1]+s, pos[2]+s,  0,0, 1, r,g,b],
    ], dtype=np.float32)

cube_indices = np.array([
    0,1,2, 2,3,0, 4,5,6, 6,7,4,
    0,4,7, 7,3,0, 1,5,6, 6,2,1,
    0,1,5, 5,4,0, 3,2,6, 6,7,3
], dtype=np.uint32)

# Vibrant colors
COLORS = [
    (0.9, 0.2, 0.2),  # Red
    (0.2, 0.9, 0.2),  # Green
    (0.2, 0.4, 0.9),  # Blue
    (0.9, 0.9, 0.2),  # Yellow
    (0.9, 0.2, 0.9),  # Magenta
    (0.2, 0.9, 0.9),  # Cyan
]

frames = []
NUM_FRAMES = 90

print("Generating colorful orbit demo...")

for frame in range(NUM_FRAMES):
    t = frame / NUM_FRAMES * 2 * math.pi
    
    # Orbiting camera
    cam_dist = 2.0
    cx = math.sin(t) * cam_dist
    cz = math.cos(t) * cam_dist
    cy = 0.3
    
    # Look-at matrix (simplified)
    fwd = np.array([-cx, -cy, -cz])
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0, 1, 0])
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    
    view = np.array([
        [right[0], up[0], -fwd[0], 0],
        [right[1], up[1], -fwd[1], 0],
        [right[2], up[2], -fwd[2], 0],
        [-np.dot(right, [cx,cy,cz]), -np.dot(up, [cx,cy,cz]), np.dot(fwd, [cx,cy,cz]), 1],
    ], dtype=np.float32)
    
    # Dark blue background
    ctx.clear(20, 25, 40, 1.0)
    ctx.set_camera(50.0, 0.01, 10.0, view)
    
    # Add orbiting cubes
    for i, color in enumerate(COLORS):
        angle = t + i * (2 * math.pi / len(COLORS))
        radius = 0.5
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        y = math.sin(angle * 2 + t) * 0.15  # Bobbing
        
        verts = make_cube_colored([x, y, z], color, size=0.15)
        ctx.add_mesh_colored(verts, cube_indices, object_id=i+1)
    
    # Center cube (white)
    center_verts = make_cube_colored([0, 0, 0], (0.95, 0.95, 0.95), size=0.2)
    ctx.add_mesh_colored(center_verts, cube_indices, object_id=10)
    
    ctx.render()
    
    rgb = ctx.rgb_tensor()
    depth = ctx.depth_tensor()
    seg = ctx.segmentation_tensor()
    
    # Depth visualization (cool blue gradient)
    d_min, d_max = depth[depth < 1e6].min(), depth[depth < 1e6].max()
    depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
    depth_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth_rgb[:,:,0] = (50 + 150 * (1 - depth_norm)).astype(np.uint8)  # R
    depth_rgb[:,:,1] = (100 + 100 * (1 - depth_norm)).astype(np.uint8)  # G
    depth_rgb[:,:,2] = (180 + 75 * (1 - depth_norm)).astype(np.uint8)  # B
    
    # Segmentation (show each object in its color)
    seg_rgb = np.zeros_like(rgb)
    for i, color in enumerate(COLORS):
        mask = seg == (i + 1)
        seg_rgb[mask] = [int(c * 255) for c in color]
    seg_rgb[seg == 10] = [240, 240, 240]  # Center cube white
    
    # Combine
    combo = np.concatenate([rgb, depth_rgb, seg_rgb], axis=1)
    frames.append(combo)

# Save
output_path = "samples/wow_orbit_demo.gif"
imageio.mimsave(output_path, frames, fps=30, loop=0)
print(f"Saved: {output_path}")
print(f"Generated {NUM_FRAMES} frames showing RGB | Depth | Segmentation")
