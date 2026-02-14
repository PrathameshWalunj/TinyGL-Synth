"""
ULTIMATE FLEX DEMO: Speed Showcase

Renders 180 frames showing off speed. No GPU needed!
"""

import numpy as np
import imageio
import sys
import time
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from tinygl_synth import Context

W, H = 256, 256
ctx = Context(W, H)

def make_cube(pos, color, size=0.1):
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

indices = np.array([
    0,1,2, 2,3,0, 4,5,6, 6,7,4,
    0,4,7, 7,3,0, 1,5,6, 6,2,1,
    0,1,5, 5,4,0, 3,2,6, 6,7,3
], dtype=np.uint32)

COLORS = [
    (0.95, 0.3, 0.3), (0.3, 0.95, 0.3), (0.3, 0.5, 0.95),
    (0.95, 0.95, 0.3), (0.95, 0.3, 0.95), (0.3, 0.95, 0.95),
    (0.95, 0.6, 0.3), (0.6, 0.3, 0.95),
]

NUM_FRAMES = 120
print(f"\n{'='*60}")
print(f"  TINYGL-SYNTH SPEED FLEX")
print(f"{'='*60}")
print(f"\n  Rendering {NUM_FRAMES} unique frames at {W}x{H}...")

frames = []
start_time = time.perf_counter()

for frame in range(NUM_FRAMES):
    t = frame / NUM_FRAMES * 4 * math.pi
    
    # Orbiting camera
    cam_dist = 1.8
    cx = math.sin(t * 0.5) * cam_dist
    cz = math.cos(t * 0.5) * cam_dist
    cy = 0.2 + math.sin(t * 0.3) * 0.3
    
    fwd = np.array([-cx, -cy, -cz])
    fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
    up = np.array([0, 1, 0])
    right = np.cross(fwd, up)
    right = right / (np.linalg.norm(right) + 1e-6)
    up = np.cross(right, fwd)
    
    view = np.array([
        [right[0], up[0], -fwd[0], 0],
        [right[1], up[1], -fwd[1], 0],
        [right[2], up[2], -fwd[2], 0],
        [-np.dot(right, [cx,cy,cz]), -np.dot(up, [cx,cy,cz]), np.dot(fwd, [cx,cy,cz]), 1],
    ], dtype=np.float32)
    
    bg_r = int(20 + 15 * math.sin(t * 0.2))
    bg_g = int(25 + 10 * math.sin(t * 0.3))
    bg_b = int(45 + 20 * math.sin(t * 0.25))
    ctx.clear(bg_r, bg_g, bg_b, 1.0)
    ctx.set_camera(55.0, 0.01, 10.0, view)
    
    for i, color in enumerate(COLORS):
        angle = t + i * (2 * math.pi / len(COLORS))
        radius = 0.4 + 0.1 * math.sin(t * 2 + i)
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        y = math.sin(angle * 2 + t) * 0.12
        
        verts = make_cube([x, y, z], color, size=0.12)
        ctx.add_mesh_colored(verts, indices, object_id=i+1)
    
    pulse = 0.15 + 0.03 * math.sin(t * 3)
    center_verts = make_cube([0, 0, 0], (0.98, 0.98, 0.98), size=pulse)
    ctx.add_mesh_colored(center_verts, indices, object_id=20)
    
    ctx.render()
    
    rgb = ctx.rgb_tensor()
    depth = ctx.depth_tensor()
    seg = ctx.segmentation_tensor()
    
    valid = depth < 1e6
    if valid.any():
        d_min, d_max = depth[valid].min(), depth[valid].max()
        depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
    else:
        depth_norm = np.zeros_like(depth)
    depth_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth_rgb[:,:,0] = (40 + 180 * (1 - depth_norm)).astype(np.uint8)
    depth_rgb[:,:,1] = (80 + 140 * (1 - depth_norm)).astype(np.uint8)
    depth_rgb[:,:,2] = (160 + 95 * (1 - depth_norm)).astype(np.uint8)
    
    seg_rgb = np.zeros_like(rgb)
    for i, color in enumerate(COLORS):
        mask = seg == (i + 1)
        seg_rgb[mask] = [int(c * 255) for c in color]
    seg_rgb[seg == 20] = [250, 250, 250]
    
    combo = np.concatenate([rgb, depth_rgb, seg_rgb], axis=1)
    frames.append(combo)

elapsed = time.perf_counter() - start_time
fps = NUM_FRAMES / elapsed

print(f"\n  Rendered {NUM_FRAMES} frames in {elapsed:.2f} seconds")
print(f" That's {fps:.0f} FPS on CPU!")

imageio.mimsave("samples/speed_test.gif", frames, fps=30, loop=0)
print(f"  Saved: samples/speed_test.gif")
print(f"\n{'='*60}")
print(f"  Done.")
print(f"{'='*60}\n")
