/**
 * ZBuffer Implementation - Framebuffer and Depth Buffer
 * 32-bit RGBA, float depth
 */

#include "zbuffer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//! ZBuffer creation and destruction

ZBuffer *zb_create(int width, int height, Pixel *pixels) {
  ZBuffer *zb = (ZBuffer *)malloc(sizeof(ZBuffer));
  if (!zb) {
    fprintf(stderr, "ZBuffer: Failed to allocate ZBuffer structure\n");
    return NULL;
  }

  zb->width = width;
  zb->height = height;
  zb->pitch = width * sizeof(Pixel);

  /* Allocate or use external pixel buffer */
  if (pixels == NULL) {
    zb->pixels = (Pixel *)malloc(width * height * sizeof(Pixel));
    if (!zb->pixels) {
      fprintf(stderr, "ZBuffer: Failed to allocate pixel buffer\n");
      free(zb);
      return NULL;
    }
    zb->owns_pixels = true;
  } else {
    zb->pixels = pixels;
    zb->owns_pixels = false;
  }

  /* Alwaysallocate Z-buffer internally */
  zb->zbuffer = (ZDepth *)malloc(width * height * sizeof(ZDepth));
  if (!zb->zbuffer) {
    fprintf(stderr, "ZBuffer: Failed to allocate depth buffer\n");
    if (zb->owns_pixels)
      free(zb->pixels);
    free(zb);
    return NULL;
  }
  zb->owns_zbuffer = true;

  zb->current_texture = NULL;
  zb->texture_width = 0;
  zb->texture_height = 0;

  /* Clear to default values */
  zb_clear(zb, true, true, 0, 0, 0, 255, 1.0f);

  return zb;
}

void zb_destroy(ZBuffer *zb) {
  if (!zb)
    return;

  if (zb->owns_pixels && zb->pixels) {
    free(zb->pixels);
  }

  if (zb->owns_zbuffer && zb->zbuffer) {
    free(zb->zbuffer);
  }

  free(zb);
}

bool zb_resize(ZBuffer *zb, int width, int height) {
  if (!zb)
    return false;

  /* Only resize if we own the buffers */
  if (!zb->owns_pixels || !zb->owns_zbuffer) {
    fprintf(stderr, "ZBuffer: Cannot resize external buffers\n");
    return false;
  }

  /* Reallocate pixel buffer */
  Pixel *new_pixels =
      (Pixel *)realloc(zb->pixels, width * height * sizeof(Pixel));
  if (!new_pixels) {
    fprintf(stderr, "ZBuffer: Failed to resize pixel buffer\n");
    return false;
  }
  zb->pixels = new_pixels;

  /* Reallocate depth buffer */
  ZDepth *new_zbuffer =
      (ZDepth *)realloc(zb->zbuffer, width * height * sizeof(ZDepth));
  if (!new_zbuffer) {
    fprintf(stderr, "ZBuffer: Failed to resize depth buffer\n");
    return false;
  }
  zb->zbuffer = new_zbuffer;

  /* Update dimensions */
  zb->width = width;
  zb->height = height;
  zb->pitch = width * sizeof(Pixel);

  /* Clear new buffers */
  zb_clear(zb, true, true, 0, 0, 0, 255, 1.0f);

  return true;
}

//! Clear Operations

void zb_clear_color(ZBuffer *zb, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  if (!zb || !zb->pixels)
    return;

  Pixel color = zb_pack_color(r, g, b, a);
  int total_pixels = zb->width * zb->height;

  /* Fast fill using word-sized writes */
  for (int i = 0; i < total_pixels; i++) {
    zb->pixels[i] = color;
  }
}

void zb_clear_depth(ZBuffer *zb, ZDepth depth) {
  if (!zb || !zb->zbuffer)
    return;

  int total_pixels = zb->width * zb->height;

  /* Fill depth buffer */
  for (int i = 0; i < total_pixels; i++) {
    zb->zbuffer[i] = depth;
  }
}

void zb_clear(ZBuffer *zb, bool clear_color_flag, bool clear_depth_flag,
              uint8_t r, uint8_t g, uint8_t b, uint8_t a, ZDepth depth) {

  if (clear_color_flag) {
    zb_clear_color(zb, r, g, b, a);
  }

  if (clear_depth_flag) {
    zb_clear_depth(zb, depth);
  }
}

//! Framebuffer Copy

void zb_copy_framebuffer(ZBuffer *zb, void *dest, int dest_pitch) {
  if (!zb || !zb->pixels || !dest)
    return;

  uint8_t *dst = (uint8_t *)dest;
  uint8_t *src = (uint8_t *)zb->pixels;

  /* Copy row by row to handle different pitches */
  for (int y = 0; y < zb->height; y++) {
    memcpy(dst, src, zb->width * sizeof(Pixel));
    dst += dest_pitch;
    src += zb->pitch;
  }
}

//! Texture Management

void zb_set_texture(ZBuffer *zb, Pixel *texture, int tex_width,
                    int tex_height) {
  if (!zb)
    return;

  zb->current_texture = texture;
  zb->texture_width = tex_width;
  zb->texture_height = tex_height;
}