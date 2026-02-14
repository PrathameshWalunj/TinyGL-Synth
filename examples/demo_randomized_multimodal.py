import imageio
import numpy as np
import sys
from pathlib import Path
import time
import platform
import multiprocessing

start_time = time.time()
frame_times = []


sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from tinygl_synth import Context, DomainRandomizer

W, H = 128, 128

ctx = Context(W, H)
rnd = DomainRandomizer(42)

# Cube with normals (Nx6)
# Larger cube for better visibility
cube_vertices = np.array([
    [-0.2, -0.2, -0.2,  0,  0, -1],
    [ 0.2, -0.2, -0.2,  0,  0, -1],
    [ 0.2,  0.2, -0.2,  0,  0, -1],
    [-0.2,  0.2, -0.2,  0,  0, -1],

    [-0.2, -0.2,  0.2,  0,  0,  1],
    [ 0.2, -0.2,  0.2,  0,  0,  1],
    [ 0.2,  0.2,  0.2,  0,  0,  1],
    [-0.2,  0.2,  0.2,  0,  0,  1],
], dtype=np.float32)

cube_indices = np.array([
    0,1,2, 2,3,0,
    4,5,6, 6,7,4,
    0,4,7, 7,3,0,
    1,5,6, 6,2,1,
    0,1,5, 5,4,0,
    3,2,6, 6,7,3,
], dtype=np.uint32)

frames = []

for i in range(120):
    # Random cube position (closer to camera)
    pos = rnd.rng.uniform([-0.3,-0.3,0.5], [0.3,0.3,0.8])

    verts = cube_vertices.copy()
    verts[:, :3] += pos

    # Closer camera for better visibility
    cam_z = rnd.rng.uniform(-1.5, -1.2)
    view = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,cam_z,1],
    ], dtype=np.float32)

    # Random background
    bg = rnd.randomize_background()

    ctx.clear(*bg, 1.0)
    ctx.set_camera(60, 0.01, 10, view)
    ctx.add_mesh(verts, cube_indices, object_id=1)
    frame_start = time.time()
    ctx.render()
    frame_end = time.time()
    frame_times.append(frame_end - frame_start)
    rgb = ctx.rgb_tensor()
    depth = ctx.depth_tensor()
    seg = ctx.segmentation_tensor()

    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype(np.uint8)
    depth_rgb = np.stack([depth_norm]*3, axis=2)

    seg_rgb = np.zeros_like(rgb)
    seg_rgb[seg == 1] = [255, 64, 64]

    combo = np.concatenate([rgb, depth_rgb, seg_rgb], axis=1)
    frames.append(combo)
    end_time = time.time()
total_time = end_time - start_time

avg_frame_time = np.mean(frame_times)
fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
triangles = len(cube_indices) // 3  # per cube
pixels = W * H

print("\n===== tinygl-synth =====")
print(f"CPU: {platform.processor() or platform.machine()}")
print(f"Physical Cores: {multiprocessing.cpu_count()}")
print(f"Resolution: {W}x{H}  ({pixels:,} pixels)")
print(f"Triangles per frame: {triangles}")
print(f"Frames rendered: {len(frames)}")
print(f"Average frame time: {avg_frame_time*1000:.3f} ms")
print(f"Throughput (FPS): {fps:.2f}")
print(f"Total script time: {total_time:.2f} sec")
print("Render device: CPU \n")
print("Saved git_test22.gif")
imageio.mimsave("synthetic_multimodal.gif", frames, fps=30)

