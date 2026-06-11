/*
 * NV20 PGRAPH Metal pixel format tables (replaces GL constants.h for native path).
 */
#ifndef HW_NV20_PGRAPH_METAL_FORMATS_H
#define HW_NV20_PGRAPH_METAL_FORMATS_H

#include "qemu/osdep.h"
#include "hw/display/nv20/nv20_regs.h"

/* Shader stage kinds (match GL enum values used in cache keys). */
enum {
    NV20_SHADER_VERTEX   = 0x8B31,
    NV20_SHADER_FRAGMENT = 0x8B30,
    NV20_SHADER_GEOMETRY = 0x8DD9,
};

typedef struct ColorFormatInfo {
    unsigned int bytes_per_pixel;
    bool linear;
    uint32_t mtl_pixel_format; /* MTLPixelFormat as integer */
    bool is_compressed;
    uint8_t swizzle[4];
} ColorFormatInfo;

typedef struct SurfaceFormatInfo {
    unsigned int bytes_per_pixel;
    uint32_t mtl_pixel_format;
    bool has_stencil;
} SurfaceFormatInfo;

extern const ColorFormatInfo kelvin_mtl_color_format_map[];
extern const SurfaceFormatInfo kelvin_surface_color_format_map[];
extern const SurfaceFormatInfo kelvin_surface_zeta_float_format_map[];
extern const SurfaceFormatInfo kelvin_surface_zeta_fixed_format_map[];

uint32_t nv20_mtl_blend_factor(unsigned int factor);
uint32_t nv20_mtl_blend_op(unsigned int equation);
uint32_t nv20_mtl_compare(unsigned int func);
uint32_t nv20_mtl_stencil_op(unsigned int op);
uint32_t nv20_mtl_sampler_address(unsigned int addr);
uint32_t nv20_mtl_sampler_min_filter(unsigned int filter);
uint32_t nv20_mtl_sampler_mag_filter(unsigned int filter);
uint32_t nv20_mtl_cull_mode(unsigned int mode);
uint32_t nv20_mtl_winding(bool front_cw);

#endif
