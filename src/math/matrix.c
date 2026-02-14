/*
 * Matrix Operations Implementation
 */

#include "matrix.h"
#include <stdio.h>
#include <string.h>

/* ========================================================================
 * Identity Matrix
 * ======================================================================== */

void mat4_identity(Mat4 *TGL_RESTRICT m) {
  memset(m, 0, sizeof(Mat4));
  m->m[0][0] = 1.0f;
  m->m[1][1] = 1.0f;
  m->m[2][2] = 1.0f;
  m->m[3][3] = 1.0f;
  m->flags = MAT4_FLAG_IDENTITY | MAT4_FLAG_ORTHOGONAL | MAT4_FLAG_AFFINE |
             MAT4_FLAG_UNIFORM;
}

bool mat4_is_identity(const Mat4 *m) {
  /* Fast path: check flag first */
  if (m->flags & MAT4_FLAG_IDENTITY) {
    return true;
  }

  /* Check diagonal */
  if (m->m[0][0] != 1.0f || m->m[1][1] != 1.0f || m->m[2][2] != 1.0f ||
      m->m[3][3] != 1.0f) {
    return false;
  }

  /* Check off-diagonal elements */
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i != j && m->m[i][j] != 0.0f) {
        return false;
      }
    }
  }

  return true;
}

/* ========================================================================
 * Matrix Multiplication - Optimized
 * ======================================================================== */

/* Naive implementation for reference */
static void mat4_mul_naive(Mat4 *TGL_RESTRICT c, const Mat4 *TGL_RESTRICT a,
                           const Mat4 *TGL_RESTRICT b) {
  float temp[16];

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      float sum = 0.0f;
      for (int k = 0; k < 4; k++) {
        sum += a->m[k][i] * b->m[j][k]; /* Column-major */
      }
      temp[j * 4 + i] = sum;
    }
  }

  memcpy(c->data, temp, sizeof(temp));
  c->flags = MAT4_FLAG_DIRTY;
}

/* Optimized implementation with loop unrolling */
void mat4_mul(Mat4 *TGL_RESTRICT c, const Mat4 *TGL_RESTRICT a,
              const Mat4 *TGL_RESTRICT b) {

  /*Use SIMD if available*/
  /*
 #ifdef TINYGL_SIMD_ENABLED
     mat4_mul_simd(c, a, b);
     return;
 #endif */

  /* Manual loop unrolling for better performance */
  float temp[16];
  const float *pa = a->data;
  const float *pb = b->data;

  /* Column 0 */
  temp[0] = pa[0] * pb[0] + pa[4] * pb[1] + pa[8] * pb[2] + pa[12] * pb[3];
  temp[1] = pa[1] * pb[0] + pa[5] * pb[1] + pa[9] * pb[2] + pa[13] * pb[3];
  temp[2] = pa[2] * pb[0] + pa[6] * pb[1] + pa[10] * pb[2] + pa[14] * pb[3];
  temp[3] = pa[3] * pb[0] + pa[7] * pb[1] + pa[11] * pb[2] + pa[15] * pb[3];

  /* Column 1 */
  temp[4] = pa[0] * pb[4] + pa[4] * pb[5] + pa[8] * pb[6] + pa[12] * pb[7];
  temp[5] = pa[1] * pb[4] + pa[5] * pb[5] + pa[9] * pb[6] + pa[13] * pb[7];
  temp[6] = pa[2] * pb[4] + pa[6] * pb[5] + pa[10] * pb[6] + pa[14] * pb[7];
  temp[7] = pa[3] * pb[4] + pa[7] * pb[5] + pa[11] * pb[6] + pa[15] * pb[7];

  /* Column 2 */
  temp[8] = pa[0] * pb[8] + pa[4] * pb[9] + pa[8] * pb[10] + pa[12] * pb[11];
  temp[9] = pa[1] * pb[8] + pa[5] * pb[9] + pa[9] * pb[10] + pa[13] * pb[11];
  temp[10] = pa[2] * pb[8] + pa[6] * pb[9] + pa[10] * pb[10] + pa[14] * pb[11];
  temp[11] = pa[3] * pb[8] + pa[7] * pb[9] + pa[11] * pb[10] + pa[15] * pb[11];

  /* Column 3 */
  temp[12] = pa[0] * pb[12] + pa[4] * pb[13] + pa[8] * pb[14] + pa[12] * pb[15];
  temp[13] = pa[1] * pb[12] + pa[5] * pb[13] + pa[9] * pb[14] + pa[13] * pb[15];
  temp[14] =
      pa[2] * pb[12] + pa[6] * pb[13] + pa[10] * pb[14] + pa[14] * pb[15];
  temp[15] =
      pa[3] * pb[12] + pa[7] * pb[13] + pa[11] * pb[14] + pa[15] * pb[15];

  memcpy(c->data, temp, sizeof(temp));
  c->flags = MAT4_FLAG_DIRTY;
}

/* Fast multiplication with flag-based optimization */
void mat4_mul_fast(Mat4 *TGL_RESTRICT c, const Mat4 *TGL_RESTRICT a,
                   const Mat4 *TGL_RESTRICT b) {

  /* Identity shortcuts */
  if (a->flags & MAT4_FLAG_IDENTITY) {
    mat4_copy(c, b);
    return;
  }
  if (b->flags & MAT4_FLAG_IDENTITY) {
    mat4_copy(c, a);
    return;
  }

  /* Affine fast path (bottom row is [0 0 0 1]) */
  if ((a->flags & MAT4_FLAG_AFFINE) && (b->flags & MAT4_FLAG_AFFINE)) {
    float temp[12]; /* Only need 3x4 */
    const float *pa = a->data;
    const float *pb = b->data;

    /* First 3 rows only */
    for (int col = 0; col < 4; col++) {
      int idx = col * 4;
      temp[col * 3 + 0] = pa[0] * pb[idx] + pa[4] * pb[idx + 1] +
                          pa[8] * pb[idx + 2] + pa[12] * pb[idx + 3];
      temp[col * 3 + 1] = pa[1] * pb[idx] + pa[5] * pb[idx + 1] +
                          pa[9] * pb[idx + 2] + pa[13] * pb[idx + 3];
      temp[col * 3 + 2] = pa[2] * pb[idx] + pa[6] * pb[idx + 1] +
                          pa[10] * pb[idx + 2] + pa[14] * pb[idx + 3];
    }

    /* Copy results */
    for (int col = 0; col < 4; col++) {
      c->data[col * 4 + 0] = temp[col * 3 + 0];
      c->data[col * 4 + 1] = temp[col * 3 + 1];
      c->data[col * 4 + 2] = temp[col * 3 + 2];
      c->data[col * 4 + 3] = (col == 3) ? 1.0f : 0.0f;
    }

    c->flags = MAT4_FLAG_AFFINE | MAT4_FLAG_DIRTY;
    return;
  }

  /* General case */
  mat4_mul(c, a, b);
}

/* ========================================================================
 * Vector Transformation
 * ======================================================================== */

void mat4_mul_vec4(Vec4 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m,
                   const Vec4 *TGL_RESTRICT v) {
  /*
  #ifdef TINYGL_SIMD_ENABLED
      mat4_mul_vec4_simd(out, m, v);
      return;
  #endif */

  Vec4 temp;
  const float *md = m->data;

  temp.x = md[0] * v->x + md[4] * v->y + md[8] * v->z + md[12] * v->w;
  temp.y = md[1] * v->x + md[5] * v->y + md[9] * v->z + md[13] * v->w;
  temp.z = md[2] * v->x + md[6] * v->y + md[10] * v->z + md[14] * v->w;
  temp.w = md[3] * v->x + md[7] * v->y + md[11] * v->z + md[15] * v->w;

  *out = temp;
}

void mat4_mul_vec3_point(Vec3 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m,
                         const Vec3 *TGL_RESTRICT v) {
  Vec4 v4 = {.x = v->x, .y = v->y, .z = v->z, .w = 1.0f};
  Vec4 result;
  mat4_mul_vec4(&result, m, &v4);

  /* Perspective divide if needed */
  if (fabsf(result.w - 1.0f) > 1e-6f) {
    float inv_w = 1.0f / result.w;
    out->x = result.x * inv_w;
    out->y = result.y * inv_w;
    out->z = result.z * inv_w;
  } else {
    out->x = result.x;
    out->y = result.y;
    out->z = result.z;
  }
}

void mat4_mul_vec3_dir(Vec3 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m,
                       const Vec3 *TGL_RESTRICT v) {
  Vec3 temp;
  const float *md = m->data;

  temp.x = md[0] * v->x + md[4] * v->y + md[8] * v->z;
  temp.y = md[1] * v->x + md[5] * v->y + md[9] * v->z;
  temp.z = md[2] * v->x + md[6] * v->y + md[10] * v->z;

  *out = temp;
}

/* ========================================================================
 * Transpose
 * ======================================================================== */

void mat4_transpose(Mat4 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m) {

#ifdef TINYGL_SIMD_ENABLED
  mat4_transpose_simd(out, m);
  return;
#endif

  Mat4 temp;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      temp.m[i][j] = m->m[j][i];
    }
  }

  *out = temp;
  out->flags = MAT4_FLAG_DIRTY;
}

/* ========================================================================
 * Matrix Inversion
 * ======================================================================== */

/* Fast inversion for orthogonal matrices (rotation + translation) */
void mat4_invert_orthogonal(Mat4 *TGL_RESTRICT out,
                            const Mat4 *TGL_RESTRICT m) {
  /* Transpose the 3x3 rotation part */
  out->m[0][0] = m->m[0][0];
  out->m[0][1] = m->m[1][0];
  out->m[0][2] = m->m[2][0];
  out->m[1][0] = m->m[0][1];
  out->m[1][1] = m->m[1][1];
  out->m[1][2] = m->m[2][1];
  out->m[2][0] = m->m[0][2];
  out->m[2][1] = m->m[1][2];
  out->m[2][2] = m->m[2][2];

  /* Negate and transform translation */
  float tx = -m->m[3][0];
  float ty = -m->m[3][1];
  float tz = -m->m[3][2];

  out->m[3][0] = out->m[0][0] * tx + out->m[1][0] * ty + out->m[2][0] * tz;
  out->m[3][1] = out->m[0][1] * tx + out->m[1][1] * ty + out->m[2][1] * tz;
  out->m[3][2] = out->m[0][2] * tx + out->m[1][2] * ty + out->m[2][2] * tz;

  /* Bottom row */
  out->m[0][3] = 0.0f;
  out->m[1][3] = 0.0f;
  out->m[2][3] = 0.0f;
  out->m[3][3] = 1.0f;

  out->flags = MAT4_FLAG_ORTHOGONAL | MAT4_FLAG_AFFINE | MAT4_FLAG_DIRTY;
}

/* General matrix inversion using Gauss-Jordan elimination */
bool mat4_invert(Mat4 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m) {
  /* Fast path for orthogonal matrices */
  if (m->flags & MAT4_FLAG_ORTHOGONAL) {
    mat4_invert_orthogonal(out, m);
    return true;
  }

  float inv[16];
  float det;
  const float *src = m->data;

  /* Calculate cofactors */
  inv[0] = src[5] * src[10] * src[15] - src[5] * src[11] * src[14] -
           src[9] * src[6] * src[15] + src[9] * src[7] * src[14] +
           src[13] * src[6] * src[11] - src[13] * src[7] * src[10];
  inv[4] = -src[4] * src[10] * src[15] + src[4] * src[11] * src[14] +
           src[8] * src[6] * src[15] - src[8] * src[7] * src[14] -
           src[12] * src[6] * src[11] + src[12] * src[7] * src[10];
  inv[8] = src[4] * src[9] * src[15] - src[4] * src[11] * src[13] -
           src[8] * src[5] * src[15] + src[8] * src[7] * src[13] +
           src[12] * src[5] * src[11] - src[12] * src[7] * src[9];
  inv[12] = -src[4] * src[9] * src[14] + src[4] * src[10] * src[13] +
            src[8] * src[5] * src[14] - src[8] * src[6] * src[13] -
            src[12] * src[5] * src[10] + src[12] * src[6] * src[9];

  /* Calculate determinant */
  det = src[0] * inv[0] + src[1] * inv[4] + src[2] * inv[8] + src[3] * inv[12];

  if (fabsf(det) < 1e-8f) {
    return false; /* Singular matrix */
  }

  det = 1.0f / det;

  inv[1] = -src[1] * src[10] * src[15] + src[1] * src[11] * src[14] +
           src[9] * src[2] * src[15] - src[9] * src[3] * src[14] -
           src[13] * src[2] * src[11] + src[13] * src[3] * src[10];
  inv[5] = src[0] * src[10] * src[15] - src[0] * src[11] * src[14] -
           src[8] * src[2] * src[15] + src[8] * src[3] * src[14] +
           src[12] * src[2] * src[11] - src[12] * src[3] * src[10];
  inv[9] = -src[0] * src[9] * src[15] + src[0] * src[11] * src[13] +
           src[8] * src[1] * src[15] - src[8] * src[3] * src[13] -
           src[12] * src[1] * src[11] + src[12] * src[3] * src[9];
  inv[13] = src[0] * src[9] * src[14] - src[0] * src[10] * src[13] -
            src[8] * src[1] * src[14] + src[8] * src[2] * src[13] +
            src[12] * src[1] * src[10] - src[12] * src[2] * src[9];

  inv[2] = src[1] * src[6] * src[15] - src[1] * src[7] * src[14] -
           src[5] * src[2] * src[15] + src[5] * src[3] * src[14] +
           src[13] * src[2] * src[7] - src[13] * src[3] * src[6];
  inv[6] = -src[0] * src[6] * src[15] + src[0] * src[7] * src[14] +
           src[4] * src[2] * src[15] - src[4] * src[3] * src[14] -
           src[12] * src[2] * src[7] + src[12] * src[3] * src[6];
  inv[10] = src[0] * src[5] * src[15] - src[0] * src[7] * src[13] -
            src[4] * src[1] * src[15] + src[4] * src[3] * src[13] +
            src[12] * src[1] * src[7] - src[12] * src[3] * src[5];
  inv[14] = -src[0] * src[5] * src[14] + src[0] * src[6] * src[13] +
            src[4] * src[1] * src[14] - src[4] * src[2] * src[13] -
            src[12] * src[1] * src[6] + src[12] * src[2] * src[5];

  inv[3] = -src[1] * src[6] * src[11] + src[1] * src[7] * src[10] +
           src[5] * src[2] * src[11] - src[5] * src[3] * src[10] -
           src[9] * src[2] * src[7] + src[9] * src[3] * src[6];
  inv[7] = src[0] * src[6] * src[11] - src[0] * src[7] * src[10] -
           src[4] * src[2] * src[11] + src[4] * src[3] * src[10] +
           src[8] * src[2] * src[7] - src[8] * src[3] * src[6];
  inv[11] = -src[0] * src[5] * src[11] + src[0] * src[7] * src[9] +
            src[4] * src[1] * src[11] - src[4] * src[3] * src[9] -
            src[8] * src[1] * src[7] + src[8] * src[3] * src[5];
  inv[15] = src[0] * src[5] * src[10] - src[0] * src[6] * src[9] -
            src[4] * src[1] * src[10] + src[4] * src[2] * src[9] +
            src[8] * src[1] * src[6] - src[8] * src[2] * src[5];

  /* Multiply by determinant */
  for (int i = 0; i < 16; i++) {
    out->data[i] = inv[i] * det;
  }

  out->flags = MAT4_FLAG_DIRTY;
  return true;
}

/* ========================================================================
 * Utility
 * ======================================================================== */

void mat4_print(const Mat4 *m) {
  printf("Matrix 4x4:\n");
  for (int row = 0; row < 4; row++) {
    printf("  [");
    for (int col = 0; col < 4; col++) {
      printf(" %8.4f", m->m[col][row]);
    }
    printf(" ]\n");
  }
  printf("  Flags: 0x%02X\n", m->flags);
}

/* ========================================================================
 * Projection Matrix Construction
 * ======================================================================== */

void mat4_frustum(Mat4 *m, float left, float right, float bottom, float top,
                  float near, float far) {
  mat4_identity(m);

  float width = right - left;
  float height = top - bottom;
  float depth = far - near;

  m->m[0][0] = (2.0f * near) / width;
  m->m[1][1] = (2.0f * near) / height;
  m->m[2][0] = (right + left) / width;
  m->m[2][1] = (top + bottom) / height;
  m->m[2][2] = -(far + near) / depth;
  m->m[2][3] = -1.0f;
  m->m[3][2] = -(2.0f * far * near) / depth;
  m->m[3][3] = 0.0f;

  m->flags = MAT4_FLAG_DIRTY;
}

void mat4_ortho(Mat4 *m, float left, float right, float bottom, float top,
                float near, float far) {
  mat4_identity(m);

  float width = right - left;
  float height = top - bottom;
  float depth = far - near;

  m->m[0][0] = 2.0f / width;
  m->m[1][1] = 2.0f / height;
  m->m[2][2] = -2.0f / depth;
  m->m[3][0] = -(right + left) / width;
  m->m[3][1] = -(top + bottom) / height;
  m->m[3][2] = -(far + near) / depth;

  m->flags = MAT4_FLAG_DIRTY;
}

void mat4_perspective(Mat4 *m, float fovy_rad, float aspect, float near,
                      float far) {
  float tan_half_fovy = tanf(fovy_rad / 2.0f);

  mat4_identity(m);

  m->m[0][0] = 1.0f / (aspect * tan_half_fovy);
  m->m[1][1] = 1.0f / tan_half_fovy;
  m->m[2][2] = -(far + near) / (far - near);
  m->m[2][3] = -1.0f;
  m->m[3][2] = -(2.0f * far * near) / (far - near);
  m->m[3][3] = 0.0f;

  m->flags = MAT4_FLAG_DIRTY;
}

void mat4_lookat(Mat4 *m, const Vec3 *eye, const Vec3 *center, const Vec3 *up) {
  Vec3 f = vec3_sub(center, eye);
  vec3_normalize(&f);

  Vec3 s = vec3_cross(&f, up);
  vec3_normalize(&s);

  Vec3 u = vec3_cross(&s, &f);

  mat4_identity(m);

  // ROW 0 = s
  m->data[0] = s.x;
  m->data[1] = s.y;
  m->data[2] = s.z;

  // ROW 1 = u
  m->data[4] = u.x;
  m->data[5] = u.y;
  m->data[6] = u.z;

  // ROW 2 = -f
  m->data[8] = -f.x;
  m->data[9] = -f.y;
  m->data[10] = -f.z;

  // ROW 3 = translation
  m->data[12] = -vec3_dot(&s, eye);
  m->data[13] = -vec3_dot(&u, eye);
  m->data[14] = vec3_dot(&f, eye);

  m->flags = MAT4_FLAG_DIRTY;
}