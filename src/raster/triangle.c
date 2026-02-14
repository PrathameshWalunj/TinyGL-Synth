/**
 * Triangle Rasterization - Extended for Multi-Buffer Output
 * Based on scanline algorithm with depth testing
 */

#include "triangle.h"
#include <stdlib.h>
#include <string.h>


  //! Helper: Sort vertices by Y coordinate (top to bottom)


static void sort_vertices_by_y(ZBufferPoint **p0, ZBufferPoint **p1,
                               ZBufferPoint **p2) {
  ZBufferPoint *temp;

  if ((*p0)->y > (*p1)->y) {
    temp = *p0;
    *p0 = *p1;
    *p1 = temp;
  }
  if ((*p1)->y > (*p2)->y) {
    temp = *p1;
    *p1 = *p2;
    *p2 = temp;
  }
  if ((*p0)->y > (*p1)->y) {
    temp = *p0;
    *p0 = *p1;
    *p1 = temp;
  }
}


 //! Flat-Shaded Triangle (Single Color)


void zb_fill_triangle_flat(ZBuffer *zb, ZBufferPoint *v0, ZBufferPoint *v1,
                           ZBufferPoint *v2) {

  ZBufferPoint *p0 = v0, *p1 = v1, *p2 = v2;
  sort_vertices_by_y(&p0, &p1, &p2);

  if (p0->y == p2->y)
    return;

  uint8_t r = zb_float_to_byte(p2->r);
  uint8_t g = zb_float_to_byte(p2->g);
  uint8_t b = zb_float_to_byte(p2->b);
  uint8_t a = zb_float_to_byte(p2->a);
  Pixel color = zb_pack_color(r, g, b, a);

  int total_height = p2->y - p0->y;

  for (int part = 0; part < 2; part++) {
    int segment_height = (part == 0) ? (p1->y - p0->y) : (p2->y - p1->y);
    if (segment_height == 0)
      continue;

    for (int y = 0; y < segment_height; y++) {
      int current_y = (part == 0) ? (p0->y + y) : (p1->y + y);

      if (current_y < 0 || current_y >= zb->height)
        continue;

      float alpha = (float)(current_y - p0->y) / total_height;
      float beta = (float)y / segment_height;

      ZBufferPoint *segment_start = (part == 0) ? p0 : p1;
      ZBufferPoint *segment_end = (part == 0) ? p1 : p2;

      int x_a = p0->x + (int)((p2->x - p0->x) * alpha);
      int x_b =
          segment_start->x + (int)((segment_end->x - segment_start->x) * beta);

      float z_a = p0->z + (p2->z - p0->z) * alpha;
      float z_b = segment_start->z + (segment_end->z - segment_start->z) * beta;

      if (x_a > x_b) {
        int temp_x = x_a;
        x_a = x_b;
        x_b = temp_x;
        float temp_z = z_a;
        z_a = z_b;
        z_b = temp_z;
      }

      int scanline_width = x_b - x_a;
      if (scanline_width == 0)
        scanline_width = 1;

      for (int x = x_a; x <= x_b; x++) {
        if (x < 0 || x >= zb->width)
          continue;

        float t = (float)(x - x_a) / scanline_width;
        float z = z_a + (z_b - z_a) * t;

        int index = current_y * zb->width + x;
        if (z < zb->zbuffer[index]) {
          zb->zbuffer[index] = z;
          zb->pixels[index] = color;
        }
      }
    }
  }
}


 //! Gouraud-Shaded Triangle (Smooth Color Interpolation)


void zb_fill_triangle_smooth(ZBuffer *zb, ZBufferPoint *v0, ZBufferPoint *v1,
                             ZBufferPoint *v2) {

  ZBufferPoint *p0 = v0, *p1 = v1, *p2 = v2;
  sort_vertices_by_y(&p0, &p1, &p2);

  if (p0->y == p2->y)
    return;

  int total_height = p2->y - p0->y;

  for (int part = 0; part < 2; part++) {
    int segment_height = (part == 0) ? (p1->y - p0->y) : (p2->y - p1->y);
    if (segment_height == 0)
      continue;

    for (int y = 0; y < segment_height; y++) {
      int current_y = (part == 0) ? (p0->y + y) : (p1->y + y);

      if (current_y < 0 || current_y >= zb->height)
        continue;

      float alpha = (float)(current_y - p0->y) / total_height;
      float beta = (float)y / segment_height;

      ZBufferPoint *segment_start = (part == 0) ? p0 : p1;
      ZBufferPoint *segment_end = (part == 0) ? p1 : p2;

      int x_a = p0->x + (int)((p2->x - p0->x) * alpha);
      int x_b =
          segment_start->x + (int)((segment_end->x - segment_start->x) * beta);

      float z_a = p0->z + (p2->z - p0->z) * alpha;
      float z_b = segment_start->z + (segment_end->z - segment_start->z) * beta;

      float r_a = p0->r + (p2->r - p0->r) * alpha;
      float r_b = segment_start->r + (segment_end->r - segment_start->r) * beta;

      float g_a = p0->g + (p2->g - p0->g) * alpha;
      float g_b = segment_start->g + (segment_end->g - segment_start->g) * beta;

      float b_a = p0->b + (p2->b - p0->b) * alpha;
      float b_b = segment_start->b + (segment_end->b - segment_start->b) * beta;

      float a_a = p0->a + (p2->a - p0->a) * alpha;
      float a_b = segment_start->a + (segment_end->a - segment_start->a) * beta;

      if (x_a > x_b) {
        int temp_x = x_a;
        x_a = x_b;
        x_b = temp_x;
        float temp;
        temp = z_a;
        z_a = z_b;
        z_b = temp;
        temp = r_a;
        r_a = r_b;
        r_b = temp;
        temp = g_a;
        g_a = g_b;
        g_b = temp;
        temp = b_a;
        b_a = b_b;
        b_b = temp;
        temp = a_a;
        a_a = a_b;
        a_b = temp;
      }

      int scanline_width = x_b - x_a;
      if (scanline_width == 0)
        scanline_width = 1;

      for (int x = x_a; x <= x_b; x++) {
        if (x < 0 || x >= zb->width)
          continue;

        float t = (float)(x - x_a) / scanline_width;
        float z = z_a + (z_b - z_a) * t;
        float r = r_a + (r_b - r_a) * t;
        float g = g_a + (g_b - g_a) * t;
        float b = b_a + (b_b - b_a) * t;
        float a = a_a + (a_b - a_a) * t;

        int index = current_y * zb->width + x;
        if (z < zb->zbuffer[index]) {
          zb->zbuffer[index] = z;

          uint8_t rb = zb_float_to_byte(r);
          uint8_t gb = zb_float_to_byte(g);
          uint8_t bb = zb_float_to_byte(b);
          uint8_t ab = zb_float_to_byte(a);

          zb->pixels[index] = zb_pack_color(rb, gb, bb, ab);
        }
      }
    }
  }
}