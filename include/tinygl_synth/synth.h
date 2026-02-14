#ifndef TINYGL_SYNTH_H
#define TINYGL_SYNTH_H

#include "types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

TGLSynthContext* tgls_create_context(int width, int height);
TGLSynthContext* tgls_create_context_ex(const TGLSynthContextCreateInfo* info);
void tgls_destroy_context(TGLSynthContext* ctx);

void tgls_clear(TGLSynthContext* ctx, uint8_t r, uint8_t g, uint8_t b, float depth);

void tgls_set_camera(TGLSynthContext* ctx, 
                     float fov, 
                     float near_plane, 
                     float far_plane,
                     const float* view_matrix);

void tgls_add_mesh(TGLSynthContext* ctx,
                   const float* vertices,
                   int num_vertices,
                   const uint32_t* indices,
                   int num_indices,
                   uint32_t object_id);

void tgls_add_mesh_colored(TGLSynthContext* ctx,
                           const float* vertices,
                           int num_vertices,
                           const uint32_t* indices,
                           int num_indices,
                           uint32_t object_id);

void tgls_render(TGLSynthContext* ctx);

uint8_t* tgls_get_rgb_buffer(TGLSynthContext* ctx);
float* tgls_get_depth_buffer(TGLSynthContext* ctx);
uint32_t* tgls_get_segmentation_buffer(TGLSynthContext* ctx);
float* tgls_get_normal_buffer(TGLSynthContext* ctx);

void tgls_get_dimensions(TGLSynthContext* ctx, int* width, int* height);

TGLSynthError tgls_get_last_error(TGLSynthContext* ctx);
const char* tgls_error_string(TGLSynthError error);

#ifdef __cplusplus
}
#endif

#endif