#include "tinygl_synth/synth.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("Testing multi-buffer creation...\n");
    
    TGLSynthContext* ctx = tgls_create_context(256, 256);
    if (!ctx) {
        printf("FAIL: Could not create context\n");
        return 1;
    }
    
    int width, height;
    tgls_get_dimensions(ctx, &width, &height);
    if (width != 256 || height != 256) {
        printf("FAIL: Incorrect dimensions\n");
        return 1;
    }
    
    uint8_t* rgb = tgls_get_rgb_buffer(ctx);
    if (!rgb) {
        printf("FAIL: RGB buffer is NULL\n");
        return 1;
    }
    
    float* depth = tgls_get_depth_buffer(ctx);
    if (!depth) {
        printf("FAIL: Depth buffer is NULL\n");
        return 1;
    }
    
    uint32_t* seg = tgls_get_segmentation_buffer(ctx);
    if (!seg) {
        printf("FAIL: Segmentation buffer is NULL\n");
        return 1;
    }
    
    float* normals = tgls_get_normal_buffer(ctx);
    if (!normals) {
        printf("FAIL: Normal buffer is NULL\n");
        return 1;
    }
    
    tgls_clear(ctx, 100, 150, 200, 1.0f);
    
    uint32_t first_pixel = ((uint32_t*)rgb)[0];
    uint8_t r = (first_pixel >> 16) & 0xFF;
    uint8_t g = (first_pixel >> 8) & 0xFF;
    uint8_t b = first_pixel & 0xFF;
    
    if (r != 100 || g != 150 || b != 200) {
        printf("FAIL: Clear color incorrect (got %d,%d,%d)\n", r, g, b);
        return 1;
    }
    
    if (depth[0] != 1.0f) {
        printf("FAIL: Clear depth incorrect\n");
        return 1;
    }
    
    tgls_destroy_context(ctx);
    
    printf("PASS: All buffer tests passed\n");
    return 0;
}