#include "tinygl_synth/synth.h"
#include <stdio.h>
#include <stdlib.h>

static void write_ppm(const char* filename, uint8_t* rgb, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    
    for (int i = 0; i < width * height; i++) {
        uint32_t pixel = ((uint32_t*)rgb)[i];
        uint8_t r = (pixel >> 16) & 0xFF;
        uint8_t g = (pixel >> 8) & 0xFF;
        uint8_t b = pixel & 0xFF;
        fputc(r, f);
        fputc(g, f);
        fputc(b, f);
    }
    
    fclose(f);
}

int main(void) {
    printf("tinygl-synth: Rendering cube example\n");
    
    TGLSynthContext* ctx = tgls_create_context(640, 480);
    if (!ctx) {
        printf("Failed to create context\n");
        return 1;
    }
    
    float vertices[] = {
        -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
         1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
         1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
         1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
         1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
    };
    
    uint32_t indices[] = {
        0, 1, 2,  2, 3, 0,
        4, 5, 6,  6, 7, 4,
        0, 4, 7,  7, 3, 0,
        1, 5, 6,  6, 2, 1,
        0, 1, 5,  5, 4, 0,
        3, 2, 6,  6, 7, 3,
    };
    
    float view_matrix[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, -5.0f, 1.0f,
    };
    
    tgls_clear(ctx, 50, 50, 50, 1.0f);
    tgls_set_camera(ctx, 60.0f, 0.1f, 100.0f, view_matrix);
    tgls_add_mesh(ctx, vertices, 8, indices, 36, 1);
    tgls_render(ctx);
    
    uint8_t* rgb = tgls_get_rgb_buffer(ctx);
    write_ppm("cube.ppm", rgb, 640, 480);
    
    uint32_t* seg = tgls_get_segmentation_buffer(ctx);
    int seg_pixels = 0;
    for (int i = 0; i < 640 * 480; i++) {
        if (seg[i] == 1) seg_pixels++;
    }
    
    printf("Success! Rendered %d pixels\n", seg_pixels);
    printf("Output: cube.ppm\n");
    
    tgls_destroy_context(ctx);
    return 0;
}