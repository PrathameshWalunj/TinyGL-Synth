# Getting Started with tinygl-synth

## Installation

### Build from source

```bash
git clone https://github.com/PrathameshWalunj/tinygl-synth.git
cd tinygl-synth
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

### Python package (local)

```bash
cd python
pip install -e .
```

> [!NOTE]
> Python searches for the native library in `build/`, `build/Release/`, and `build/Debug/`.
> On Windows, CMake also copies `tinygl_synth.dll` next to example/test executables after build.
> If you run binaries from a custom location, add `build/Release` (or `build/Debug`) to `PATH`.

## Your First Render

```python
from tinygl_synth import Context
import numpy as np

# 1. Create renderer (128x128 pixels)
ctx = Context(128, 128)

# 2. Define a triangle (position + normal per vertex)
vertices = np.array([
    [-1, -1, 0,  0, 0, 1],  # x, y, z, nx, ny, nz
    [ 1, -1, 0,  0, 0, 1],
    [ 0,  1, 0,  0, 0, 1],
], dtype=np.float32)

indices = np.array([0, 1, 2], dtype=np.uint32)

# 3. Set up camera
view_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, -3, 1],  # Camera 3 units back
], dtype=np.float32)

ctx.clear(50, 50, 50, 1.0)
ctx.set_camera(60.0, 0.01, 10.0, view_matrix)
ctx.add_mesh(vertices, indices, object_id=1)
ctx.render()

# 4. Get outputs - all in one pass!
rgb = ctx.rgb_tensor()            # (128, 128, 3) uint8
depth = ctx.depth_tensor()        # (128, 128) float32
seg = ctx.segmentation_tensor()   # (128, 128) uint32
normals = ctx.normal_tensor()     # (128, 128, 3) float32
```

## Multi-Modal Output

tinygl-synth renders all buffers in a single pass:

| Buffer | Type | Description |
|--------|------|-------------|
| RGB | uint8 (H,W,3) | Color image |
| Depth | float32 (H,W) | Distance per pixel |
| Segmentation | uint32 (H,W) | Object instance ID |
| Normals | float32 (H,W,3) | Surface orientation |

## Domain Randomization

```python
from tinygl_synth import Context, DomainRandomizer

ctx = Context(128, 128)
randomizer = DomainRandomizer(seed=42)

for epoch in range(100):
    # Randomize background
    bg_color = randomizer.randomize_background()
    ctx.clear(*bg_color, 1.0)
    
    # Randomize object position
    pos = randomizer.rng.uniform(-0.5, 0.5, 3)
    # ... modify vertices with pos offset
    
    ctx.render()
    rgb = ctx.rgb_tensor()  # Infinite varied training data!
```

## PyTorch Integration

Zero-copy conversion to PyTorch tensors:

```python
import torch

ctx.render()
rgb = torch.from_numpy(ctx.rgb_tensor()).float() / 255.0
rgb = rgb.permute(2, 0, 1)  # HWC -> CHW for PyTorch
```

## Examples

| Example | Description |
|---------|-------------|
| `examples/demo_spin_cube.py` | Animated cube with RGB/depth/segmentation |
| `examples/train_pytorch_policy.py` | Full CNN training loop |
| `examples/grasp_training_demo.py` | Synthetic grasping data/training workflow demo |
| `examples/basic_cube.c` | Pure C rendering example |

## Troubleshooting

**Q: Rendered image is black?**
A: Check camera position - make sure objects are within near/far planes.

**Q: DLL not found error on Windows?**
A: Ensure `tinygl_synth.dll` is in `build/Release/` or `build/Debug/`.

**Q: Slow rendering?**
A: Build in Release mode: `cmake --build . --config Release`

**Q: Import error in Python?**
A: Run from the project root, or add to PYTHONPATH:
```bash
export PYTHONPATH=/path/to/tinygl-synth/python:$PYTHONPATH
```

## Performance

Benchmark on Intel CPU (Windows 11):

| Metric | Value |
|--------|-------|
| Triangle throughput | ~2M triangles/sec |
| Mesh upload | ~3ms for 50K triangles |
| Render time | ~25ms for 50K triangles |

Run your own benchmark:
```bash
./build/examples/Release/benchmark.exe
```
