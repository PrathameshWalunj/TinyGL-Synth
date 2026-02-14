import numpy as np
import ctypes
from ._native import _lib

class Context:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._ctx = _lib.tgls_create_context(width, height)
        if not self._ctx:
            raise RuntimeError("Failed to create tinygl-synth context")
    
    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            _lib.tgls_destroy_context(self._ctx)
    
    def clear(self, r=0, g=0, b=0, depth=1.0):
        _lib.tgls_clear(self._ctx, r, g, b, depth)
    
    def set_camera(self, fov, near, far, view_matrix=None):
        if view_matrix is not None:
            view_array = np.asarray(view_matrix, dtype=np.float32).flatten()
            view_ptr = view_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            view_ptr = None
        _lib.tgls_set_camera(self._ctx, fov, near, far, view_ptr)
    
    def add_mesh(self, vertices, indices, object_id=1):
        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.uint32)
        
        if vertices.ndim != 2 or vertices.shape[1] != 6:
            raise ValueError("Vertices must be Nx6 array (x,y,z,nx,ny,nz)")
        
        num_vertices = vertices.shape[0]
        num_indices = indices.size
        
        v_ptr = vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        i_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        
        _lib.tgls_add_mesh(self._ctx, v_ptr, num_vertices, i_ptr, num_indices, object_id)
    

    def add_mesh_colored(self, vertices, indices, object_id=1):
        vertices = np.asarray(vertices, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.uint32)
    
        if vertices.ndim != 2 or vertices.shape[1] != 9:
            raise ValueError("Vertices must be Nx9 array (x,y,z,nx,ny,nz,r,g,b)")
    
        num_vertices = vertices.shape[0]
        num_indices = indices.size
    
        v_ptr = vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        i_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    
        _lib.tgls_add_mesh_colored(self._ctx, v_ptr, num_vertices, i_ptr, num_indices, object_id)
    
    def render(self):
        _lib.tgls_render(self._ctx)
    
    def rgb_tensor(self):
        ptr = _lib.tgls_get_rgb_buffer(self._ctx)
        buffer = np.ctypeslib.as_array(ptr, shape=(self.height * self.width * 4,))
        rgba = np.frombuffer(buffer, dtype=np.uint32).reshape(self.height, self.width)
        
        r = (rgba >> 16) & 0xFF
        g = (rgba >> 8) & 0xFF
        b = rgba & 0xFF
        
        return np.stack([r, g, b], axis=-1).astype(np.uint8)
    
    def depth_tensor(self):
        ptr = _lib.tgls_get_depth_buffer(self._ctx)
        return np.ctypeslib.as_array(ptr, shape=(self.height, self.width))
    
    def segmentation_tensor(self):
        ptr = _lib.tgls_get_segmentation_buffer(self._ctx)
        return np.ctypeslib.as_array(ptr, shape=(self.height, self.width))
    
    def normal_tensor(self):
        ptr = _lib.tgls_get_normal_buffer(self._ctx)
        return np.ctypeslib.as_array(ptr, shape=(self.height, self.width, 3))