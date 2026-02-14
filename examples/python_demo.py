import numpy as np
import sys
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from tinygl_synth import Context

def main():
    print("=== Smooth Normal Test: Mustard Bottle ===")

    
    obj_path = "examples/ycb_assets/006_mustard_bottle/google_16k/textured.obj"
    print(f"Loading {obj_path}...")
    
    mesh = trimesh.load(obj_path, force='mesh')
    
    # Extract Data
    v_pos = mesh.vertices.astype(np.float32)
    v_norm = mesh.vertex_normals.astype(np.float32) # Smooth normals from Trimesh
    indices = mesh.faces.flatten().astype(np.uint32)

    # 3. Pack Data: [x, y, z, nx, ny, nz]
    # We skip RGB for this test to ensure we are visualizing Normals
    vertices = np.hstack([v_pos, v_norm]).astype(np.float32)
    
    print(f"Mesh Stats: {len(v_pos)} verts, {len(indices)//3} tris")

    # 4. Create Context
    ctx = Context(640, 480)
    ctx.clear(50, 50, 50, 1.0)

    # 5. Camera Setup
    view_matrix = np.eye(4, dtype=np.float32)
    view_matrix[3, 2] = -0.4  # Move camera very close to see curvature

    ctx.set_camera(60.0, 0.01, 5.0, view_matrix)

    # 6. Add Mesh (Normals are inside 'vertices' now)
    ctx.add_mesh(vertices, indices, object_id=1)

    # 7. Render
    ctx.render()

    # 8. Get Outputs
    rgb = ctx.rgb_tensor()
    normal = ctx.normal_tensor() # This should now contain smooth gradients

    print("RGB Shape:", rgb.shape)
    print("Normal Shape:", normal.shape)
    
    # Check center pixel normal
    cy, cx = 240, 320
    print(f"Sample Normal at center ({cx},{cy}): {normal[cy, cx]}")

    # 9. Visualize Normals
    # Normals are [-1, 1], map to [0, 1] for display
    n_viz = (normal + 1.0) / 2.0
    
    # Mask out background (where normal is 0,0,0 usually, or just use segmentation)
   
    mask = np.linalg.norm(normal, axis=2) > 0.1
    n_viz[~mask] = 0

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("RGB (Vertex Color)")
    
    plt.subplot(1, 2, 2)
    plt.imshow(n_viz)
    plt.title("Smooth Normals (RGB encoded)")
    
    plt.show()

if __name__ == "__main__":
    main()