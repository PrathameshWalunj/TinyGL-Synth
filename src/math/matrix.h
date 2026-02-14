/*
MATHS x MATHS x MATHS
 */

#ifndef TINYGL_MATH_H
#define TINYGL_MATH_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Compiler Hints & Alignment
 * ======================================================================== */

#ifdef _MSC_VER
#define TGL_ALIGN(n) __declspec(align(n))
#define TGL_INLINE static __forceinline
#define TGL_RESTRICT __restrict
#else
#define TGL_ALIGN(n) __attribute__((aligned(n)))
#define TGL_INLINE static inline __attribute__((always_inline))
#define TGL_RESTRICT __restrict__
#endif

/* Cache line size for optimal alignment */
#define TGL_CACHE_LINE 64

/* ========================================================================
 * Vector & Matrix Types (16-byte aligned for SIMD)
 * ======================================================================== */

/* 3D Vector */
typedef TGL_ALIGN(16) struct {
  union {
    struct {
      float x, y, z, _pad;
    };
    struct {
      float r, g, b, _pad2;
    };
    float v[4];
  };
} Vec3;

/* 4D Vector */
typedef TGL_ALIGN(16) struct {
  union {
    struct {
      float x, y, z, w;
    };
    struct {
      float r, g, b, a;
    };
    float v[4];
  };
} Vec4;

/* 4x4 Matrix (column-major, OpenGL style) */
typedef TGL_ALIGN(16) struct {
  union {
    float m[4][4];  /* [column][row] */
    float data[16]; /* Flat array */
    Vec4 col[4];    /* Column vectors */
  };

  /* Matrix properties for fast-path optimization */
  uint8_t flags;
  uint8_t _pad[3];
} Mat4;

/* Matrix flags for fast-path detection */
#define MAT4_FLAG_IDENTITY 0x01   /* Is identity matrix */
#define MAT4_FLAG_ORTHOGONAL 0x02 /* Has orthogonal basis */
#define MAT4_FLAG_AFFINE 0x04     /* Bottom row is [0 0 0 1] */
#define MAT4_FLAG_UNIFORM 0x08    /* Uniform scale */
#define MAT4_FLAG_DIRTY 0x80      /* Needs recomputation */

/* ========================================================================
 * Vector Operations - Vec3
 * ======================================================================== */

TGL_INLINE Vec3 vec3_new(float x, float y, float z);

/* Create vector */
TGL_INLINE Vec3 vec3_new(float x, float y, float z) {
  Vec3 v = {.x = x, .y = y, .z = z, ._pad = 0.0f};
  return v;
}

/* Dot product */
TGL_INLINE float vec3_dot(const Vec3 *a, const Vec3 *b) {
  return a->x * b->x + a->y * b->y + a->z * b->z;
}

/* Cross product */
TGL_INLINE Vec3 vec3_cross(const Vec3 *a, const Vec3 *b) {
  return vec3_new(a->y * b->z - a->z * b->y, a->z * b->x - a->x * b->z,
                  a->x * b->y - a->y * b->x);
}

/* Length squared (faster, no sqrt) */
TGL_INLINE float vec3_length_sq(const Vec3 *v) { return vec3_dot(v, v); }

/* Length */
TGL_INLINE float vec3_length(const Vec3 *v) { return sqrtf(vec3_length_sq(v)); }

/* Normalize (returns true if successful) */
TGL_INLINE bool vec3_normalize(Vec3 *v) {
  float len_sq = vec3_length_sq(v);
  if (len_sq < 1e-8f)
    return false;

  float inv_len = 1.0f / sqrtf(len_sq);
  v->x *= inv_len;
  v->y *= inv_len;
  v->z *= inv_len;
  return true;
}

/* Add vectors */
TGL_INLINE Vec3 vec3_add(const Vec3 *a, const Vec3 *b) {
  return vec3_new(a->x + b->x, a->y + b->y, a->z + b->z);
}

/* Subtract vectors */
TGL_INLINE Vec3 vec3_sub(const Vec3 *a, const Vec3 *b) {
  return vec3_new(a->x - b->x, a->y - b->y, a->z - b->z);
}

/* Scale vector */
TGL_INLINE Vec3 vec3_scale(const Vec3 *v, float s) {
  return vec3_new(v->x * s, v->y * s, v->z * s);
}

/* ========================================================================
 * Vector Operations - Vec4
 * ======================================================================== */

/* Create vector */
TGL_INLINE Vec4 vec4_new(float x, float y, float z, float w) {
  Vec4 v = {.x = x, .y = y, .z = z, .w = w};
  return v;
}

/* Dot product */
TGL_INLINE float vec4_dot(const Vec4 *a, const Vec4 *b) {
  return a->x * b->x + a->y * b->y + a->z * b->z + a->w * b->w;
}

/* ========================================================================
 * Matrix Operations - Mat4
 * ======================================================================== */

/* Initialize identity matrix */
void mat4_identity(Mat4 *TGL_RESTRICT m);

/* Check if matrix is identity */
bool mat4_is_identity(const Mat4 *m);

/* Copy matrix */
TGL_INLINE void mat4_copy(Mat4 *TGL_RESTRICT dst,
                          const Mat4 *TGL_RESTRICT src) {
  memcpy(dst, src, sizeof(Mat4));
}

/* Matrix multiplication: C = A * B */
void mat4_mul(Mat4 *TGL_RESTRICT c, const Mat4 *TGL_RESTRICT a,
              const Mat4 *TGL_RESTRICT b);

/* Optimized multiply with flags */
void mat4_mul_fast(Mat4 *TGL_RESTRICT c, const Mat4 *TGL_RESTRICT a,
                   const Mat4 *TGL_RESTRICT b);

/* Transform vector by matrix: out = M * v */
void mat4_mul_vec4(Vec4 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m,
                   const Vec4 *TGL_RESTRICT v);

/* Transform point (w=1) by matrix */
void mat4_mul_vec3_point(Vec3 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m,
                         const Vec3 *TGL_RESTRICT v);

/* Transform direction (w=0) by matrix */
void mat4_mul_vec3_dir(Vec3 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m,
                       const Vec3 *TGL_RESTRICT v);

/* Transpose matrix */
void mat4_transpose(Mat4 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m);

/* Invert general matrix (returns false if singular) */
bool mat4_invert(Mat4 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m);

/* Fast invert for orthogonal matrices (rotation + translation) */
void mat4_invert_orthogonal(Mat4 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m);

/* Fast invert for affine matrices */
bool mat4_invert_affine(Mat4 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m);

/* ========================================================================
 * Matrix Construction
 * ======================================================================== */

/*
 * Planned API surface: declarations kept for upcoming helpers.
 * These are not implemented in matrix.c yet.
 */
/* Build transformation matrices */
void mat4_translate(Mat4 *m, float x, float y, float z);
void mat4_rotate(Mat4 *m, float angle_rad, float x, float y, float z);
void mat4_rotate_x(Mat4 *m, float angle_rad);
void mat4_rotate_y(Mat4 *m, float angle_rad);
void mat4_rotate_z(Mat4 *m, float angle_rad);
void mat4_scale(Mat4 *m, float x, float y, float z);

/* Projection matrices */
void mat4_frustum(Mat4 *m, float left, float right, float bottom, float top,
                  float near, float far);
void mat4_ortho(Mat4 *m, float left, float right, float bottom, float top,
                float near, float far);
void mat4_perspective(Mat4 *m, float fovy_rad, float aspect, float near,
                      float far);
void mat4_lookat(Mat4 *m, const Vec3 *eye, const Vec3 *center, const Vec3 *up);

/* ========================================================================
 * SIMD Variants (when TINYGL_SIMD_ENABLED)
 * ======================================================================== */

/*
 * 1. I don't need it (yet)
 * 2. I don't fully understand it (yet)
 * 3. The scalar code is surprisingly fast anyway
 */
#ifdef TINYGL_SIMD_ENABLED
/* SIMD implementations in matrix_simd.c */
void mat4_mul_simd(Mat4 *TGL_RESTRICT c, const Mat4 *TGL_RESTRICT a,
                   const Mat4 *TGL_RESTRICT b);

void mat4_mul_vec4_simd(Vec4 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m,
                        const Vec4 *TGL_RESTRICT v);

void mat4_transpose_simd(Mat4 *TGL_RESTRICT out, const Mat4 *TGL_RESTRICT m);
#endif

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

/* Planned helper: compute matrix flags for optimization (not implemented yet) */
void mat4_update_flags(Mat4 *m);

/* Print matrix (debugging) */
void mat4_print(const Mat4 *m);

/* Convert degrees to radians */
TGL_INLINE float deg_to_rad(float degrees) {
  return degrees * (3.14159265358979323846f / 180.0f);
}

/* Convert radians to degrees */
TGL_INLINE float rad_to_deg(float radians) {
  return radians * (180.0f / 3.14159265358979323846f);
}

/* Fast inverse square root (Quake III style) */
TGL_INLINE float fast_inv_sqrt(float x) {
  union {
    float f;
    uint32_t i;
  } u = {.f = x};
  u.i = 0x5f3759df - (u.i >> 1);
  float y = u.f;
  y = y * (1.5f - 0.5f * x * y * y); /* Newton iteration */
  return y;
}

#ifdef __cplusplus
}
#endif

#endif /* TINYGL_MATH_H */
