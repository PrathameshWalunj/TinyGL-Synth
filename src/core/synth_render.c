#include "synth_render.h"
#include "../../src/raster/triangle.h"
#include <math.h>
#include <stdio.h>

static void transform_vertex(const TGLSynthContext* ctx, 
                            const TGLSynthVertex* in,
                           ZBufferPoint* out) {

   
    Vec4 pos = {in->x, in->y, in->z, 1.0f};
    Vec4 view_pos, clip_pos;
    
    mat4_mul_vec4(&view_pos, &ctx->view_matrix, &pos);

    mat4_mul_vec4(&clip_pos, &ctx->projection_matrix, &view_pos);
    
    
       
    if (fabsf(clip_pos.w) < 1e-6f) {
        out->x = -1000;
        out->y = -1000;
        out->z = 10.0f;
        return;
    }
    
    float inv_w = 1.0f / clip_pos.w;
    float ndc_x = clip_pos.x * inv_w;
    float ndc_y = clip_pos.y * inv_w;
    float ndc_z = clip_pos.z * inv_w;

      
    
    out->x = (int)((ndc_x + 1.0f) * 0.5f * ctx->width);
    out->y = (int)((1.0f - ndc_y) * 0.5f * ctx->height);
    
    out->z = (ndc_z + 1.0f) * 0.5f;
    
    if (out->z < 0.0f) out->z = 0.0f;
    if (out->z > 1.0f) out->z = 1.0f;
    
    out->r = in->r;
    out->g = in->g;
    out->b = in->b;
    out->a = in->a;
    
    out->s = 0.0f;
    out->t = 0.0f;
    out->sz = 0.0f;
    out->tz = 0.0f;
}

static void render_triangle_extended(TGLSynthContext* ctx, const TGLSynthTriangle* tri) {
    ZBufferPoint p0, p1, p2;
    transform_vertex(ctx, &tri->v[0], &p0);
    transform_vertex(ctx, &tri->v[1], &p1);
    transform_vertex(ctx, &tri->v[2], &p2);
    
    if (p0.x < -1000 || p1.x < -1000 || p2.x < -1000) {
        return;
    }
    
    Vec3 n0 = {tri->v[0].nx, tri->v[0].ny, tri->v[0].nz};
    Vec3 n1 = {tri->v[1].nx, tri->v[1].ny, tri->v[1].nz};
    Vec3 n2 = {tri->v[2].nx, tri->v[2].ny, tri->v[2].nz};


    zb_fill_triangle_synth(ctx->zbuffer, &p0, &p1, &p2,
                          tri->object_id,
                          ctx->id_buffer,
                          ctx->normal_buffer,
                          &n0, &n1, &n2);
}

void tgls_render_triangle(TGLSynthContext* ctx, const TGLSynthTriangle* tri) {
    if (!ctx || !tri) return;
      printf("[DEBUG] tgls_render: triangle_count = %d\n", ctx->triangle_count);
    render_triangle_extended(ctx, tri);
}

void tgls_flush_triangles(TGLSynthContext* ctx)
{
    if (!ctx) return;

    for (int i = 0; i < ctx->triangle_count; i++) {

       

        // Directly render triangle
        render_triangle_extended(ctx, &ctx->triangle_buffer[i]);
    }

    // After rendering, reset count
    ctx->triangle_count = 0;
}
