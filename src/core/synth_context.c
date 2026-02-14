#include "synth_context.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

TGLSynthContext* tgls_create_context_ex(const TGLSynthContextCreateInfo* info) {
    if (!info || info->width <= 0 || info->height <= 0) {
        return NULL;
    }
    
    TGLSynthContext* ctx = (TGLSynthContext*)malloc(sizeof(TGLSynthContext));
    if (!ctx) {
        return NULL;
    }
    
    memset(ctx, 0, sizeof(TGLSynthContext));
    
    ctx->width = info->width;
    ctx->height = info->height;
    ctx->enable_depth = info->enable_depth;
    ctx->enable_segmentation = info->enable_segmentation;
    ctx->enable_normals = info->enable_normals;
    ctx->enable_optical_flow = info->enable_optical_flow;
    
    ctx->zbuffer = zb_create(info->width, info->height, NULL);
    if (!ctx->zbuffer) {
        free(ctx);
        return NULL;
    }
    
    int buffer_size = info->width * info->height;
    
    if (info->enable_segmentation) {
        ctx->id_buffer = (uint32_t*)malloc(buffer_size * sizeof(uint32_t));
        if (!ctx->id_buffer) {
            zb_destroy(ctx->zbuffer);
            free(ctx);
            return NULL;
        }
        memset(ctx->id_buffer, 0, buffer_size * sizeof(uint32_t));
    }
    
    if (info->enable_normals) {
        ctx->normal_buffer = (float*)malloc(buffer_size * 3 * sizeof(float));
        if (!ctx->normal_buffer) {
            free(ctx->id_buffer);
            zb_destroy(ctx->zbuffer);
            free(ctx);
            return NULL;
        }
        memset(ctx->normal_buffer, 0, buffer_size * 3 * sizeof(float));
    }
    
    if (info->enable_optical_flow) {
        ctx->flow_buffer = (float*)malloc(buffer_size * 2 * sizeof(float));
        if (!ctx->flow_buffer) {
            free(ctx->normal_buffer);
            free(ctx->id_buffer);
            zb_destroy(ctx->zbuffer);
            free(ctx);
            return NULL;
        }
        memset(ctx->flow_buffer, 0, buffer_size * 2 * sizeof(float));
    }
    
    ctx->triangle_capacity = 1024;
    ctx->triangle_buffer = (TGLSynthTriangle*)malloc(ctx->triangle_capacity * sizeof(TGLSynthTriangle));
    if (!ctx->triangle_buffer) {
        free(ctx->flow_buffer);
        free(ctx->normal_buffer);
        free(ctx->id_buffer);
        zb_destroy(ctx->zbuffer);
        free(ctx);
        return NULL;
    }
    ctx->triangle_count = 0;
    
    mat4_identity(&ctx->projection_matrix);
    mat4_identity(&ctx->view_matrix);
    
    ctx->last_error = TGLS_SUCCESS;
    
    return ctx;
}

TGLSynthContext* tgls_create_context(int width, int height) {
    TGLSynthContextCreateInfo info = {
        .width = width,
        .height = height,
        .enable_depth = true,
        .enable_segmentation = true,
        .enable_normals = true,
        .enable_optical_flow = false
    };
    return tgls_create_context_ex(&info);
}

void tgls_destroy_context(TGLSynthContext* ctx) {
    if (!ctx) return;
    
    free(ctx->triangle_buffer);
    free(ctx->flow_buffer);
    free(ctx->normal_buffer);
    free(ctx->id_buffer);
    zb_destroy(ctx->zbuffer);
    free(ctx);
}

void tgls_clear(TGLSynthContext* ctx, uint8_t r, uint8_t g, uint8_t b, float depth) {
    if (!ctx) return;
    
    zb_clear(ctx->zbuffer, true, true, r, g, b, 255, depth);
    
    int buffer_size = ctx->width * ctx->height;
    
    if (ctx->id_buffer) {
        memset(ctx->id_buffer, 0, buffer_size * sizeof(uint32_t));
    }
    
    if (ctx->normal_buffer) {
        memset(ctx->normal_buffer, 0, buffer_size * 3 * sizeof(float));
    }
    
    if (ctx->flow_buffer) {
        memset(ctx->flow_buffer, 0, buffer_size * 2 * sizeof(float));
    }
    
    ctx->triangle_count = 0;
}

void tgls_get_dimensions(TGLSynthContext* ctx, int* width, int* height) {
    if (!ctx) return;
    if (width) *width = ctx->width;
    if (height) *height = ctx->height;
}

uint8_t* tgls_get_rgb_buffer(TGLSynthContext* ctx) {
    if (!ctx || !ctx->zbuffer) return NULL;
    return (uint8_t*)ctx->zbuffer->pixels;
}

float* tgls_get_depth_buffer(TGLSynthContext* ctx) {
    if (!ctx || !ctx->zbuffer) return NULL;
    return ctx->zbuffer->zbuffer;
}

uint32_t* tgls_get_segmentation_buffer(TGLSynthContext* ctx) {
    if (!ctx) return NULL;
    return ctx->id_buffer;
}

float* tgls_get_normal_buffer(TGLSynthContext* ctx) {
    if (!ctx) return NULL;
    return ctx->normal_buffer;
}

TGLSynthError tgls_get_last_error(TGLSynthContext* ctx) {
    if (!ctx) return TGLS_ERROR_INVALID_CONTEXT;
    return ctx->last_error;
}

const char* tgls_error_string(TGLSynthError error) {
    switch (error) {
        case TGLS_SUCCESS: return "Success";
        case TGLS_ERROR_INVALID_CONTEXT: return "Invalid context";
        case TGLS_ERROR_INVALID_PARAMETER: return "Invalid parameter";
        case TGLS_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case TGLS_ERROR_INVALID_OPERATION: return "Invalid operation";
        default: return "Unknown error";
    }
}