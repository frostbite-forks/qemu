/*
 * GLSL to MSL translation for NV20 PGRAPH Metal renderer.
 */
#ifndef HW_NV20_PGRAPH_METAL_GLSL_H
#define HW_NV20_PGRAPH_METAL_GLSL_H

#include "qemu/osdep.h"

void nv20_mtl_glsl_init(void);
void nv20_mtl_glsl_finalize(void);

char *nv20_mtl_glsl_to_msl_vertex(const char *glsl);
char *nv20_mtl_glsl_to_msl_fragment(const char *glsl);
char *nv20_mtl_glsl_to_msl_geometry(const char *glsl);

void *nv20_mtl_compile_msl_library(void *backend, const char *msl_source,
                                   const char *entry_point);

#endif /* HW_NV20_PGRAPH_METAL_GLSL_H */
