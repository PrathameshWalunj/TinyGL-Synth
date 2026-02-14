#include "tinygl_synth/synth.h"
#include "../core/synth_context.h"
#include "../core/synth_render.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

void tgls_set_camera(TGLSynthContext* ctx,
                     float fov_deg,
                     float near_plane,
                     float far_plane,
                     const float* view_matrix)
{
    if (!ctx) return;

    float aspect = (float)ctx->width / (float)ctx->height;
    float f = 1.0f / tanf(fov_deg * 0.5f * 3.1415926535f / 180.0f);

    Mat4* P = &ctx->projection_matrix;

    mat4_identity(P);

    // COLUMN-MAJOR PROJECTION MATRIX (is OpenGL style)
    P->data[0]  = f / aspect;       // column 0, row 0
    P->data[5]  = f;                // column 1, row 1
    P->data[10] = (far_plane + near_plane) / (near_plane - far_plane);
    P->data[14] = (2.0f * far_plane * near_plane) / (near_plane - far_plane);

    P->data[11] = -1.0f;  // row 2, col 3  → md[11]
    P->data[15] = 0.0f;   // row 3, col 3

    // View matrix
    if (view_matrix)
        memcpy(&ctx->view_matrix, view_matrix, sizeof(Mat4));
    else
        mat4_identity(&ctx->view_matrix);
}

void tgls_add_mesh(TGLSynthContext* ctx,
                   const float* vertices,
                   int num_vertices,
                   const uint32_t* indices,
                   int num_indices,
                   uint32_t object_id) {
    if (!ctx || !vertices || !indices) return;
    if (num_vertices <= 0 || num_indices <= 0) return;
    if (num_indices % 3 != 0) return;
    
    int num_triangles = num_indices / 3;
    
    if (ctx->triangle_count + num_triangles > ctx->triangle_capacity) {
        int new_capacity = ctx->triangle_capacity * 2;
        while (new_capacity < ctx->triangle_count + num_triangles) {
            new_capacity *= 2;
        }
        
        TGLSynthTriangle* new_buffer = (TGLSynthTriangle*)realloc(
            ctx->triangle_buffer,
            new_capacity * sizeof(TGLSynthTriangle)
        );
        
        if (!new_buffer) {
            ctx->last_error = TGLS_ERROR_OUT_OF_MEMORY;
            return;
        }
        
        ctx->triangle_buffer = new_buffer;
        ctx->triangle_capacity = new_capacity;
    }
    
    for (int i = 0; i < num_triangles; i++) {
        TGLSynthTriangle* tri = &ctx->triangle_buffer[ctx->triangle_count++];
        tri->object_id = object_id;
        
        for (int j = 0; j < 3; j++) {
            uint32_t idx = indices[i * 3 + j];
            if (idx >= (uint32_t)num_vertices) {
                ctx->last_error = TGLS_ERROR_INVALID_PARAMETER;
                ctx->triangle_count--;
                return;
            }
            
            const float* v = &vertices[idx * 6];
            tri->v[j].x = v[0];
            tri->v[j].y = v[1];
            tri->v[j].z = v[2];
            tri->v[j].nx = v[3];
            tri->v[j].ny = v[4];
            tri->v[j].nz = v[5];
            
            tri->v[j].r = 0.8f;
            tri->v[j].g = 0.8f;
            tri->v[j].b = 0.8f;
            tri->v[j].a = 1.0f;
        }
    }
}

void tgls_add_mesh_colored(TGLSynthContext* ctx,
                          const float* vertices,
                          int num_vertices,
                          const uint32_t* indices,
                          int num_indices,
                          uint32_t object_id) {
    if (!ctx || !vertices || !indices) return;
    if (num_vertices <= 0 || num_indices <= 0) return;
    if (num_indices % 3 != 0) return;
    
    int num_triangles = num_indices / 3;
    
    if (ctx->triangle_count + num_triangles > ctx->triangle_capacity) {
        int new_capacity = ctx->triangle_capacity * 2;
        while (new_capacity < ctx->triangle_count + num_triangles) {
            new_capacity *= 2;
        }
        
        TGLSynthTriangle* new_buffer = (TGLSynthTriangle*)realloc(
            ctx->triangle_buffer,
            new_capacity * sizeof(TGLSynthTriangle)
        );
        
        if (!new_buffer) {
            ctx->last_error = TGLS_ERROR_OUT_OF_MEMORY;
            return;
        }
        
        ctx->triangle_buffer = new_buffer;
        ctx->triangle_capacity = new_capacity;
    }
    
    for (int i = 0; i < num_triangles; i++) {
        TGLSynthTriangle* tri = &ctx->triangle_buffer[ctx->triangle_count++];
        tri->object_id = object_id;
        
        for (int j = 0; j < 3; j++) {
            uint32_t idx = indices[i * 3 + j];
            if (idx >= (uint32_t)num_vertices) {
                ctx->last_error = TGLS_ERROR_INVALID_PARAMETER;
                ctx->triangle_count--;
                return;
            }
            
            const float* v = &vertices[idx * 9];
            tri->v[j].x = v[0];
            tri->v[j].y = v[1];
            tri->v[j].z = v[2];
            tri->v[j].nx = v[3];
            tri->v[j].ny = v[4];
            tri->v[j].nz = v[5];
            tri->v[j].r = v[6];
            tri->v[j].g = v[7];
            tri->v[j].b = v[8];
            tri->v[j].a = 1.0f;
        }
    }
}

void tgls_render(TGLSynthContext* ctx) {
    if (!ctx) return;
    tgls_flush_triangles(ctx);
}