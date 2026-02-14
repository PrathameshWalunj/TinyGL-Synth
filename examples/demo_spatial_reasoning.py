"""
SPATIAL REASONING 
- 3-10 objects per scene
- Duplicate colors allowed (multiple red cubes)


"""

import numpy as np
import cv2
import imageio
import sys
import time
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from tinygl_synth import Context, DomainRandomizer

W, H = 160, 160
ctx = Context(W, H)
rand = DomainRandomizer(rng_seed=42)

indices = np.array([
    0,1,2, 2,3,0, 4,5,6, 6,7,4,
    0,4,7, 7,3,0, 1,5,6, 6,2,1,
    0,1,5, 5,4,0, 3,2,6, 6,7,3
], dtype=np.uint32)

COLOR_NAMES = ["red", "green", "blue", "yellow", "purple", "orange"]
COLORS = [
    (0.95, 0.25, 0.25),  # red
    (0.25, 0.95, 0.25),  # green
    (0.25, 0.45, 0.95),  # blue
    (0.95, 0.95, 0.25),  # yellow
    (0.75, 0.25, 0.95),  # purple
    (0.95, 0.55, 0.15),  # orange
]

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

def generate_complex_scene():
    """Generate scene with 3-10 objects, duplicates allowed."""
    num_objects = rand.rng.integers(3, 11)  # 3 to 10 objects
    
    # Pick colors WITH replacement (duplicates!)
    color_indices = rand.rng.integers(0, len(COLORS), size=num_objects)
    
    # Random positions (spread out to avoid overlap)
    positions = []
    for _ in range(num_objects):
        pos = rand.rng.uniform([-0.5, -0.5, 0.4], [0.5, 0.5, 1.2])
        positions.append(pos)
    
    # Render
    bg = rand.randomize_background()
    ctx.clear(*bg, 1.0)
    
    view = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, -2.2, 1],
    ], dtype=np.float32)
    ctx.set_camera(55.0, 0.01, 10.0, view)
    
    for i, (ci, pos) in enumerate(zip(color_indices, positions)):
        size = rand.rng.uniform(0.08, 0.14)  # Random sizes too!
        verts = make_cube(pos, COLORS[ci], size=size)
        ctx.add_mesh_colored(verts, indices, object_id=i+1)
    
    ctx.render()
    rgb = ctx.rgb_tensor()
    
    # Count each color
    color_counts = {}
    for ci in color_indices:
        name = COLOR_NAMES[ci]
        color_counts[name] = color_counts.get(name, 0) + 1
    
    # Generate QA pairs
    qa_pairs = []
    
    # Q: How many total?
    qa_pairs.append({"q": "How many objects total?", "a": str(num_objects)})
    
    # Q: How many of each color?
    for color_name in COLOR_NAMES:
        count = color_counts.get(color_name, 0)
        qa_pairs.append({"q": f"How many {color_name}?", "a": str(count)})
    
    # Q: Which color has the most?
    if color_counts:
        most_common = max(color_counts, key=color_counts.get)
        qa_pairs.append({"q": "Which color appears most?", "a": most_common})
    
    # Q: Are there more X than Y?
    if len(color_counts) >= 2:
        colors = list(color_counts.keys())
        c1, c2 = colors[0], colors[1]
        more = "yes" if color_counts[c1] > color_counts.get(c2, 0) else "no"
        qa_pairs.append({"q": f"More {c1} than {c2}?", "a": more})
    
    # Q: Which is highest?
    highest_idx = np.argmax([p[1] for p in positions])
    qa_pairs.append({"q": "Which color is highest?", "a": COLOR_NAMES[color_indices[highest_idx]]})
    
    return rgb, qa_pairs, num_objects, color_counts

# ============== DEMO =================
print("\n" + "="*65)
print("  SPATIAL REASONING: Complex Scenes with Duplicates")
print("="*65)

# Generate many samples fast
print("\n  [Phase 1] Generating 5000 complex samples...")
start = time.perf_counter()

all_samples = []
total_qa = 0
for i in range(5000):
    rgb, qa_pairs, num_obj, color_counts = generate_complex_scene()
    all_samples.append((rgb, qa_pairs, num_obj, color_counts))
    total_qa += len(qa_pairs)

elapsed = time.perf_counter() - start
print(f" Generated 5000 samples in {elapsed:.2f}s ({5000/elapsed:.0f}/sec)")
print(f" Total QA pairs: {total_qa:,}")

# Show stats
print("\n  [Phase 2] Dataset Statistics:")
obj_counts = [s[2] for s in all_samples]
print(f"  Objects per scene: {min(obj_counts)} to {max(obj_counts)}")
print(f"  Average objects: {np.mean(obj_counts):.1f}")

# Create visualization GIF
print("\n  [Phase 3] Creating visualization GIF...")
frames = []

for i in range(60):
    rgb, qa_pairs, num_obj, color_counts = all_samples[i * 50]  # Sample every 50th
    
    # Create annotated frame
    frame = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_NEAREST)
    
    # Add info overlay
    y = 20
    cv2.putText(frame, f"{num_obj} objects", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    y += 18
    
    # Show color counts
    for color, count in sorted(color_counts.items()):
        if count > 0:
            cv2.putText(frame, f"{color}: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            y += 14
    
    frames.append(frame)

imageio.mimsave("samples/spatial_reasoning_demo.gif", frames, fps=4, loop=0)
print("Saved: samples/spatial_reasoning_demo.gif")

# Create sample grid
print("\n  [Phase 4] Creating sample grid...")
GRID = 6
grid_img = np.zeros((H * GRID, W * GRID, 3), dtype=np.uint8)
for i in range(GRID * GRID):
    row, col = i // GRID, i % GRID
    rgb, _, num_obj, _ = all_samples[i]
    img = rgb.copy()
    cv2.putText(img, str(num_obj), (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    grid_img[row*H:(row+1)*H, col*W:(col+1)*W] = img

cv2.imwrite("samples/spatial_reasoning_grid.png", cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
print("   Saved: samples/spatial_reasoning_grid.png")

# Example QA pairs
print("\n  [Phase 5] Example QA pairs from complex scenes:")
for i in [0, 1000, 3000]:
    _, qa, num_obj, counts = all_samples[i]
    print(f"\n  Scene {i}: {num_obj} objects, counts={dict(counts)}")
    for q in qa[:3]:
        print(f"    Q: {q['q']:<30} A: {q['a']}")

print("\n" + "="*65)
print("  5000 scenes × ~10 QA pairs = 50,000 training examples")
print("  Generated in {:.1f} seconds. Cost with human annotators: $$$$$".format(elapsed))
print("="*65 + "\n")
