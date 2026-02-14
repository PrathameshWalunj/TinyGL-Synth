#ifndef TINYGL_SYNTH_CONTEXT_INTERNAL_H
#define TINYGL_SYNTH_CONTEXT_INTERNAL_H

#include "tinygl_synth/types.h"
#include "../../src/raster/zbuffer.h"
#include "../../src/math/matrix.h"
#include <stdint.h>

typedef struct {
    float x, y, z;
    float nx, ny, nz;
    float r, g, b, a;
} TGLSynthVertex;

typedef struct {
    TGLSynthVertex v[3];
    uint32_t object_id;
} TGLSynthTriangle;

struct TGLSynthContext {
    int width;
    int height;
    
    ZBuffer* zbuffer;
    
    uint32_t* id_buffer;
    float* normal_buffer;
    float* flow_buffer;
    
    bool enable_depth;
    bool enable_segmentation;
    bool enable_normals;
    bool enable_optical_flow;
    
    Mat4 projection_matrix;
    Mat4 view_matrix;
    
    TGLSynthTriangle* triangle_buffer;
    int triangle_count;
    int triangle_capacity;
    
    TGLSynthError last_error;
};

#endif