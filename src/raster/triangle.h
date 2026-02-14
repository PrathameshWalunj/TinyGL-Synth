#ifndef TINYGL_SYNTH_TRIANGLE_H
#define TINYGL_SYNTH_TRIANGLE_H

#include "zbuffer.h"
#include "../core/synth_context.h"

void zb_fill_triangle_flat(ZBuffer* zb,
                           ZBufferPoint* p0,
                           ZBufferPoint* p1,
                           ZBufferPoint* p2);

void zb_fill_triangle_smooth(ZBuffer* zb,
                             ZBufferPoint* p0,
                             ZBufferPoint* p1,
                             ZBufferPoint* p2);

void zb_fill_triangle_synth(ZBuffer *zb,
                            ZBufferPoint *v0,
                            ZBufferPoint *v1,
                            ZBufferPoint *v2,
                            uint32_t object_id,
                            uint32_t *id_buffer,
                            float *normal_buffer,
                            const Vec3 *n0,
                            const Vec3 *n1,
                            const Vec3 *n2);

#endif