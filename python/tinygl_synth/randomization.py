import numpy as np

class DomainRandomizer:
    """
    Fast domain randomization for synthetic data.
    Randomizes: camera pose, lighting, colors, object poses.
    """
    
    def __init__(self, rng_seed=None):
        self.rng = np.random.default_rng(rng_seed)
    
    def randomize_camera_pose(self, base_pos, jitter=0.1):
        """Jitter camera position."""
        offset = self.rng.uniform(-jitter, jitter, size=3)
        return base_pos + offset
    
    def randomize_object_color(self, vertices):
        """Randomize RGB channels in vertex colors (for colored meshes)."""
        vertices_copy = vertices.copy()
        if vertices.shape[1] >= 9:  # Has color channels
            color_scale = self.rng.uniform(0.5, 1.5, size=3)
            vertices_copy[:, 6:9] *= color_scale
            vertices_copy[:, 6:9] = np.clip(vertices_copy[:, 6:9], 0, 1)
        return vertices_copy
    
    def randomize_background(self):
        """Random background color."""
        return tuple(self.rng.integers(0, 100, size=3))