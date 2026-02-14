"""
 See Questions + Answers on Images

This creates a grid where we can visually verify if the answers are correct:
- Each image shows objects
- Text overlay shows the QA pairs
- We can count objects and check!
"""

import numpy as np
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from tinygl_synth import Context, DomainRandomizer

W, H = 200, 200
ctx = Context(W, H)
rand = DomainRandomizer(rng_seed=123)

indices = np.array([
    0,1,2, 2,3,0, 4,5,6, 6,7,4,
    0,4,7, 7,3,0, 1,5,6, 6,2,1,
    0,1,5, 5,4,0, 3,2,6, 6,7,3
], dtype=np.uint32)

COLOR_NAMES = ["red", "green", "blue", "yellow"]
COLORS = [
    (0.95, 0.25, 0.25),  
    (0.25, 0.95, 0.25),  
    (0.25, 0.45, 0.95),  
    (0.95, 0.95, 0.25),  
]

def make_cube(pos, color, size=0.12):
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

def generate_and_verify():
    num_objects = rand.rng.integers(3, 8)
    color_indices = rand.rng.integers(0, len(COLORS), size=num_objects)
    
    positions = []
    for _ in range(num_objects):
        pos = rand.rng.uniform([-0.4, -0.4, 0.5], [0.4, 0.4, 1.0])
        positions.append(pos)
    
    bg = rand.randomize_background()
    ctx.clear(*bg, 1.0)
    
    view = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, -2.0, 1],
    ], dtype=np.float32)
    ctx.set_camera(55.0, 0.01, 10.0, view)
    
    for i, (ci, pos) in enumerate(zip(color_indices, positions)):
        verts = make_cube(pos, COLORS[ci])
        ctx.add_mesh_colored(verts, indices, object_id=i+1)
    
    ctx.render()
    rgb = ctx.rgb_tensor()
    
    # Count colors
    color_counts = {name: 0 for name in COLOR_NAMES}
    for ci in color_indices:
        color_counts[COLOR_NAMES[ci]] += 1
    
    # Generate QA
    qa = []
    qa.append(f"Total: {num_objects}")
    for name in COLOR_NAMES:
        if color_counts[name] > 0:
            qa.append(f"{name}: {color_counts[name]}")
    
    return rgb, qa, num_objects, color_counts

# Generate 9 samples for 3x3 grid
print("="*50)
print("  VERIFICATION DEMO")
print("="*50)
print("\n  Generating 9 samples we can manually verify...")

GRID = 3
grid_img = np.zeros((H * GRID, W * GRID, 3), dtype=np.uint8)

for i in range(GRID * GRID):
    row, col = i // GRID, i % GRID
    rgb, qa, num_obj, counts = generate_and_verify()
    
    img = rgb.copy()
    
    # Add QA overlay
    y = 18
    for q in qa:
        cv2.putText(img, q, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        y += 16
    
    grid_img[row*H:(row+1)*H, col*W:(col+1)*W] = img
    
    
    print(f"\n  Sample {i+1}: {counts}")
    print(f"    → Total objects: {num_obj}")
    for name, count in counts.items():
        if count > 0:
            print(f"    → {name}: {count}")

output = "samples/verify_qa_grid.png"
cv2.imwrite(output, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))


