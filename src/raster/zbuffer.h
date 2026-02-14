/**
 * ZBuffer - Framebuffer and Depth Buffer Management
 * 32-bit RGBA only
 */

#ifndef TINYGL_ZBUFFER_H
#define TINYGL_ZBUFFER_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TINYGL_PIXEL_FORMAT_RGBA32

typedef float ZDepth;
#define ZDEPTH_MAX INFINITY

typedef uint32_t Pixel;

typedef struct {

  int width;
  int height;
  int pitch; /* Row stride in bytes */

  /* Buffers */
  Pixel *pixels;   /* Color buffer (RGBA) */
  ZDepth *zbuffer; /* Depth buffer */

  /* Allocation flags */
  bool owns_pixels;  /* Have we allocate pixels? */
  bool owns_zbuffer; /* Have we allocate zbuffer? */

  /* Current state */
  Pixel *current_texture;
  int texture_width;
  int texture_height;
} ZBuffer;

typedef struct {

  int x, y;

  float z;

  float r, g, b, a;

  float s, t;

  float sz, tz;
} ZBufferPoint;

//! ZBuffer Functions

/**
 * Create a new ZBuffer
 * If pixels is NULL, allocates internal buffer
 * If pixels is provided, uses external buffer
 */
ZBuffer *zb_create(int width, int height, Pixel *pixels);

/**
 * Destroy ZBuffer and free resources
 */
void zb_destroy(ZBuffer *zb);

/**
 * Resize ZBuffer (recreates buffers if internally allocated)
 */
bool zb_resize(ZBuffer *zb, int width, int height);

/**
 * Clear buffers
 */
void zb_clear_color(ZBuffer *zb, uint8_t r, uint8_t g, uint8_t b, uint8_t a);
void zb_clear_depth(ZBuffer *zb, ZDepth depth);
void zb_clear(ZBuffer *zb, bool clear_color, bool clear_depth, uint8_t r,
              uint8_t g, uint8_t b, uint8_t a, ZDepth depth);

/**
 * Copy framebuffer to external buffer with stride
 */
void zb_copy_framebuffer(ZBuffer *zb, void *dest, int dest_pitch);

//! Triangle Rasterization

/**
 * Draw flat-shaded triangle (single color)
 */
void zb_fill_triangle_flat(ZBuffer *zb, ZBufferPoint *p0, ZBufferPoint *p1,
                           ZBufferPoint *p2);

/**
 * Draw Gouraud-shaded triangle (smooth color interpolation)
 */
void zb_fill_triangle_smooth(ZBuffer *zb, ZBufferPoint *p0, ZBufferPoint *p1,
                             ZBufferPoint *p2);

/**
 * Set current texture for mapping
 */
void zb_set_texture(ZBuffer *zb, Pixel *texture, int tex_width, int tex_height);

/**
 * Draw textured triangle with perspective-correct mapping
 */
void zb_fill_triangle_textured(ZBuffer *zb, ZBufferPoint *p0, ZBufferPoint *p1,
                               ZBufferPoint *p2);

/**
 * Draw line between two points with depth testing
 */
void zb_draw_line(ZBuffer *zb, ZBufferPoint *p0, ZBufferPoint *p1);

/**
 * Draw single pixel (utility)
 */
void zb_plot_pixel(ZBuffer *zb, int x, int y, float z, uint8_t r, uint8_t g,
                   uint8_t b, uint8_t a);

//! Utility Functions

//! Pack RGBA into 32-bit pixel

static inline Pixel zb_pack_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  return ((uint32_t)a << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) |
         (uint32_t)b;
}

//! Convert float color (0-1) to byte

static inline uint8_t zb_float_to_byte(float c) {
  if (c <= 0.0f)
    return 0;
  if (c >= 1.0f)
    return 255;
  return (uint8_t)(c * 255.0f);
}

#ifdef __cplusplus
}
#endif

#endif