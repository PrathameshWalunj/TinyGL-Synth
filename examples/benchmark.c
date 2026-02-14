#include "tinygl_synth/synth.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;

    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);

    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <time.h>
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}
#endif

#define WIDTH  1024
#define HEIGHT 1024

int main(void) {
    printf("tinygl-synth benchmark\n");

    TGLSynthContext* ctx = tgls_create_context(WIDTH, HEIGHT);
    if (!ctx) {
        printf("ERROR: Failed to create tinygl-synth context\n");
        return 1;
    }

    tgls_clear(ctx, 20, 20, 30, 1.0f);
    tgls_set_camera(ctx, 60.0f, 0.1f, 100.0f, NULL);

    const int num_cubes = 4166;
    printf("Adding %d cubes (~50k triangles)...\n", num_cubes);

    double t0 = get_time_ms();

    //
    // FIX: Put cubes inside the visible frustum
    // Instead of (-8..+8), we place them in (-2.4..+2.4)
    //
    for (int i = 0; i < num_cubes; i++) {

        float x = (float)(i % 64) * 0.35f - 11.0f;
        float y = (float)((i / 64) % 64) * 0.35f - 11.0f;
        float z = -15.0f;   // FIX: closer to camera so projection is valid

        float vertices[] = {
            // pos                    // normal
            x-0.1f, y-0.1f, z-0.1f,     0,0,-1,
            x+0.1f, y-0.1f, z-0.1f,     0,0,-1,
            x+0.1f, y+0.1f, z-0.1f,     0,0,-1,
            x-0.1f, y+0.1f, z-0.1f,     0,0,-1,

            x-0.1f, y-0.1f, z+0.1f,     0,0, 1,
            x+0.1f, y-0.1f, z+0.1f,     0,0, 1,
            x+0.1f, y+0.1f, z+0.1f,     0,0, 1,
            x-0.1f, y+0.1f, z+0.1f,     0,0, 1,
        };

        uint32_t indices[] = {
            0,1,2, 2,3,0,
            4,5,6, 6,7,4,
            0,4,7, 7,3,0,
            1,5,6, 6,2,1,
            0,1,5, 5,4,0,
            3,2,6, 6,7,3,
        };

        tgls_add_mesh(ctx, vertices, 8, indices, 36, (uint32_t)(i+1));
    }

    double t1 = get_time_ms();
    printf("Mesh upload time: %.2f ms\n", t1 - t0);

    printf("Rendering...\n");

    double t2 = get_time_ms();
    tgls_render(ctx);
    double t3 = get_time_ms();

    uint32_t* seg = tgls_get_segmentation_buffer(ctx);
    int pixel_count = 0;
    int max_id = 0;

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        if (seg[i] != 0) {
            pixel_count++;
            if (seg[i] > max_id)
                max_id = seg[i];
        }
    }

    printf("\n=== tinygl-synth Benchmark Results ===\n");
    const int total_triangles = num_cubes * 12;

    printf("Triangles: %d\n", total_triangles);
    printf("Render time: %.3f ms\n", t3 - t2);

    double tsec = (t3 - t2) / 1000.0;
    double throughput = tsec > 0 ? (double)total_triangles / tsec : 0.0;

    printf("Throughput: %.0f triangles/sec\n", throughput);
    printf("Visible pixels: %d (%.2f%% of screen)\n",
           pixel_count,
           pixel_count * 100.0 / (WIDTH * HEIGHT));

    printf("Highest object ID rendered: %d\n", max_id);

    tgls_destroy_context(ctx);
    return 0;
}
