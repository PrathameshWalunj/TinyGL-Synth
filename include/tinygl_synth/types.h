#ifndef TINYGL_SYNTH_TYPES_H
#define TINYGL_SYNTH_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TGLSynthContext TGLSynthContext;

typedef enum {
    TGLS_SUCCESS = 0,
    TGLS_ERROR_INVALID_CONTEXT,
    TGLS_ERROR_INVALID_PARAMETER,
    TGLS_ERROR_OUT_OF_MEMORY,
    TGLS_ERROR_INVALID_OPERATION
} TGLSynthError;

typedef struct {
    int width;
    int height;
    bool enable_depth;
    bool enable_segmentation;
    bool enable_normals;
    bool enable_optical_flow;
} TGLSynthContextCreateInfo;

#ifdef __cplusplus
}
#endif

#endif