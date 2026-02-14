#include "tinygl_synth/synth.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    printf("Testing deterministic rendering...\n");
    
    TGLSynthContext* ctx1 = tgls_create_context(128, 128);
    TGLSynthContext* ctx2 = tgls_create_context(128, 128);
    
    if (!ctx1 || !ctx2) {
        printf("FAIL: Could not create contexts\n");
        return 1;
    }
    
    float vertices[] = {
        -1.0f, -1.0f, 0.0f,  0.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 0.0f,  0.0f, 0.0f, 1.0f,
         0.0f,  1.0f, 0.0f,  0.0f, 0.0f, 1.0f,
    };
    
    uint32_t indices[] = {0, 1, 2};
    
    float view[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, -3.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    
    tgls_clear(ctx1, 0, 0, 0, 1.0f);
    tgls_set_camera(ctx1, 60.0f, 0.1f, 100.0f, view);
    tgls_add_mesh(ctx1, vertices, 3, indices, 3, 1);
    tgls_render(ctx1);
    
    tgls_clear(ctx2, 0, 0, 0, 1.0f);
    tgls_set_camera(ctx2, 60.0f, 0.1f, 100.0f, view);
    tgls_add_mesh(ctx2, vertices, 3, indices, 3, 1);
    tgls_render(ctx2);
    
    uint8_t* rgb1 = tgls_get_rgb_buffer(ctx1);
    uint8_t* rgb2 = tgls_get_rgb_buffer(ctx2);
    
    int buffer_size = 128 * 128 * 4;
    if (memcmp(rgb1, rgb2, buffer_size) != 0) {
        printf("FAIL: RGB buffers do not match\n");
        return 1;
    }
    
    float* depth1 = tgls_get_depth_buffer(ctx1);
    float* depth2 = tgls_get_depth_buffer(ctx2);
    
    if (memcmp(depth1, depth2, 128 * 128 * sizeof(float)) != 0) {
        printf("FAIL: Depth buffers do not match\n");
        return 1;
    }
    
    tgls_destroy_context(ctx1);
    tgls_destroy_context(ctx2);
    
    printf("PASS: Determinism test passed\n");
    return 0;
}