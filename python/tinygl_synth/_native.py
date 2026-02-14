import ctypes
import os
import sys
from pathlib import Path

def find_library():
    if sys.platform == "win32":
        lib_name = "tinygl_synth.dll"
    elif sys.platform == "darwin":
        lib_name = "libtinygl_synth.dylib"
    else:
        lib_name = "libtinygl_synth.so"
    
    search_paths = [
        Path(__file__).parent.parent.parent / "build",
        Path(__file__).parent.parent.parent / "build" / "Release",
        Path(__file__).parent.parent.parent / "build" / "Debug",
        Path.cwd(),
    ]
    
    for path in search_paths:
        lib_path = path / lib_name
        if lib_path.exists():
            return str(lib_path)
    
    raise FileNotFoundError(f"Could not find {lib_name}")

_lib = ctypes.CDLL(find_library())

_lib.tgls_create_context.argtypes = [ctypes.c_int, ctypes.c_int]
_lib.tgls_create_context.restype = ctypes.c_void_p

_lib.tgls_destroy_context.argtypes = [ctypes.c_void_p]
_lib.tgls_destroy_context.restype = None

_lib.tgls_clear.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_float]
_lib.tgls_clear.restype = None

_lib.tgls_set_camera.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.POINTER(ctypes.c_float)]
_lib.tgls_set_camera.restype = None

_lib.tgls_add_mesh.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_int,
    ctypes.c_uint32
]
_lib.tgls_add_mesh.restype = None

_lib.tgls_add_mesh_colored.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.c_int,
    ctypes.c_uint32
]
_lib.tgls_add_mesh_colored.restype = None

_lib.tgls_render.argtypes = [ctypes.c_void_p]
_lib.tgls_render.restype = None

_lib.tgls_get_rgb_buffer.argtypes = [ctypes.c_void_p]
_lib.tgls_get_rgb_buffer.restype = ctypes.POINTER(ctypes.c_uint8)

_lib.tgls_get_depth_buffer.argtypes = [ctypes.c_void_p]
_lib.tgls_get_depth_buffer.restype = ctypes.POINTER(ctypes.c_float)

_lib.tgls_get_segmentation_buffer.argtypes = [ctypes.c_void_p]
_lib.tgls_get_segmentation_buffer.restype = ctypes.POINTER(ctypes.c_uint32)

_lib.tgls_get_normal_buffer.argtypes = [ctypes.c_void_p]
_lib.tgls_get_normal_buffer.restype = ctypes.POINTER(ctypes.c_float)

_lib.tgls_get_dimensions.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_lib.tgls_get_dimensions.restype = None