import gymnasium as gym
import numpy as np
from .context import Context

class TinyGLRobotEnv(gym.Env):
    """
    Base class for robot manipulation environments using tinygl-synth.
    Pairs with MuJoCo/PyBullet for physics, uses tinygl-synth for vision.
    """
    
    metadata = {"render_modes": ["rgb_array", "depth", "segmentation"]}
    
    def __init__(self, width=128, height=128, render_mode="rgb_array"):
        super().__init__()
        
        self.width = width
        self.height = height
        self.render_mode = render_mode
        
        # tinygl-synth renderer
        self.renderer = Context(width, height)
        
        # Observation space: RGB + depth
        self.observation_space = gym.spaces.Dict({
            "rgb": gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8),
            "depth": gym.spaces.Box(0, 10, (height, width), dtype=np.float32),
            "segmentation": gym.spaces.Box(0, 255, (height, width), dtype=np.uint32),
        })
        
        # Placeholder action space (override in subclass)
        self.action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)
        
        # Camera setup
        self.camera_fov = 60.0
        self.camera_near = 0.01
        self.camera_far = 10.0
        
    def _get_camera_matrix(self):
        """Override this to sync with your physics engine camera."""
        # Default: camera at (0, 0, 2) looking at origin
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, -2, 1],
        ], dtype=np.float32)
    
    def _sync_scene(self):
        """Override this to sync meshes from physics to renderer."""
        raise NotImplementedError("Subclass must implement _sync_scene()")
    
    def _get_obs(self):
        self.renderer.clear(50, 50, 50, 1.0)
        self.renderer.set_camera(
            self.camera_fov,
            self.camera_near,
            self.camera_far,
            self._get_camera_matrix()
        )
        
        self._sync_scene()
        self.renderer.render()
        
        return {
            "rgb": self.renderer.rgb_tensor(),
            "depth": self.renderer.depth_tensor(),
            "segmentation": self.renderer.segmentation_tensor(),
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Physics reset happens in subclass
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        # Physics step happens in subclass
        obs = self._get_obs()
        reward = 0.0  # Compute in subclass
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info