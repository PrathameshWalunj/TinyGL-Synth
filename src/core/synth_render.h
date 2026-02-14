#ifndef TINYGL_SYNTH_RENDER_H
#define TINYGL_SYNTH_RENDER_H

#include "synth_context.h"

void tgls_render_triangle(TGLSynthContext* ctx, const TGLSynthTriangle* tri);
void tgls_flush_triangles(TGLSynthContext* ctx);

#endif