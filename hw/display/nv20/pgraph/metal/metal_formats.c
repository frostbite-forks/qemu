/*
 * NV20 PGRAPH Metal pixel format tables.
 */
#include "metal_formats.h"

/* MTLPixelFormat integer constants (Metal/MTLPixelFormat.h) */
enum {
    MTL_PF_INVALID = 0,
    MTL_PF_A8_UNORM = 1,
    MTL_PF_R8_UNORM = 10,
    MTL_PF_R16_UNORM = 20,
    MTL_PF_R16_FLOAT = 25,
    MTL_PF_RG8_UNORM = 30,
    MTL_PF_RG8_SNORM = 32,
    MTL_PF_B5G6R5_UNORM = 40,
    MTL_PF_A1BGR5_UNORM = 41,
    MTL_PF_ABGR4_UNORM = 60,
    MTL_PF_RGBA8_UNORM = 70,
    MTL_PF_RGBA8_SNORM = 72,
    MTL_PF_BGRA8_UNORM = 80,
    MTL_PF_RGB10A2_UNORM = 90,
    MTL_PF_BC1_RGBA = 130,
    MTL_PF_BC2_RGBA = 131,
    MTL_PF_BC3_RGBA = 132,
    MTL_PF_DEPTH16_UNORM = 250,
    MTL_PF_DEPTH32_FLOAT = 252,
    MTL_PF_DEPTH24_UNORM_STENCIL8 = 255,
};

const ColorFormatInfo kelvin_mtl_color_format_map[66] = {
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_Y8] =
        { 1, false, MTL_PF_R8_UNORM, false, { 0, 0, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_AY8] =
        { 1, false, MTL_PF_R8_UNORM, false, { 0, 0, 0, 0 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_A1R5G5B5] =
        { 2, false, MTL_PF_A1BGR5_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_X1R5G5B5] =
        { 2, false, MTL_PF_A1BGR5_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_A4R4G4B4] =
        { 2, false, MTL_PF_ABGR4_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_R5G6B5] =
        { 2, false, MTL_PF_B5G6R5_UNORM, false, { 0, 1, 2, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_A8R8G8B8] =
        { 4, false, MTL_PF_BGRA8_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_X8R8G8B8] =
        { 4, false, MTL_PF_BGRA8_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_I8_A8R8G8B8] =
        { 1, false, MTL_PF_BGRA8_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_L_DXT1_A1R5G5B5] =
        { 4, false, MTL_PF_BC1_RGBA, true, { 0, 1, 2, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_L_DXT23_A8R8G8B8] =
        { 4, false, MTL_PF_BC2_RGBA, true, { 0, 1, 2, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_L_DXT45_A8R8G8B8] =
        { 4, false, MTL_PF_BC3_RGBA, true, { 0, 1, 2, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_A1R5G5B5] =
        { 2, true, MTL_PF_A1BGR5_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_R5G6B5] =
        { 2, true, MTL_PF_B5G6R5_UNORM, false, { 0, 1, 2, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_A8R8G8B8] =
        { 4, true, MTL_PF_BGRA8_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_Y8] =
        { 1, true, MTL_PF_R8_UNORM, false, { 0, 0, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_G8B8] =
        { 2, true, MTL_PF_RG8_UNORM, false, { 0, 1, 0, 1 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_A8] =
        { 1, false, MTL_PF_R8_UNORM, false, { 3, 3, 3, 0 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_A8Y8] =
        { 2, false, MTL_PF_RG8_UNORM, false, { 0, 0, 0, 1 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_AY8] =
        { 1, true, MTL_PF_R8_UNORM, false, { 0, 0, 0, 0 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_X1R5G5B5] =
        { 2, true, MTL_PF_A1BGR5_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_A4R4G4B4] =
        { 2, true, MTL_PF_ABGR4_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_X8R8G8B8] =
        { 4, true, MTL_PF_BGRA8_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_A8] =
        { 1, true, MTL_PF_R8_UNORM, false, { 3, 3, 3, 0 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_A8Y8] =
        { 2, true, MTL_PF_RG8_UNORM, false, { 0, 0, 0, 1 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_R6G5B5] =
        { 2, false, MTL_PF_RGBA8_SNORM, false, { 0, 1, 2, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_G8B8] =
        { 2, false, MTL_PF_RG8_UNORM, false, { 0, 1, 0, 1 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_R8B8] =
        { 2, false, MTL_PF_RG8_UNORM, false, { 1, 0, 1, 0 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LC_IMAGE_CR8YB8CB8YA8] =
        { 2, true, MTL_PF_RGBA8_UNORM, false, { 0, 1, 2, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LC_IMAGE_YB8CR8YA8CB8] =
        { 2, true, MTL_PF_RGBA8_UNORM, false, { 0, 1, 2, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_DEPTH_Y16_FIXED] =
        { 2, false, MTL_PF_DEPTH16_UNORM, false, { 0, 3, 3, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_DEPTH_X8_Y24_FIXED] =
        { 4, true, MTL_PF_DEPTH24_UNORM_STENCIL8, false, { 0, 3, 3, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_DEPTH_X8_Y24_FLOAT] =
        { 4, true, MTL_PF_DEPTH24_UNORM_STENCIL8, false, { 0, 3, 3, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_DEPTH_Y16_FIXED] =
        { 2, true, MTL_PF_DEPTH16_UNORM, false, { 0, 3, 3, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_DEPTH_Y16_FLOAT] =
        { 2, true, MTL_PF_R16_FLOAT, false, { 0, 3, 3, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_Y16] =
        { 2, true, MTL_PF_R16_UNORM, false, { 0, 0, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_A8B8G8R8] =
        { 4, false, MTL_PF_RGBA8_UNORM, false, { 0, 1, 2, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_B8G8R8A8] =
        { 4, false, MTL_PF_BGRA8_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_SZ_R8G8B8A8] =
        { 4, false, MTL_PF_RGBA8_UNORM, false, { 0, 1, 2, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_A8B8G8R8] =
        { 4, true, MTL_PF_RGBA8_UNORM, false, { 0, 1, 2, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_B8G8R8A8] =
        { 4, true, MTL_PF_BGRA8_UNORM, false, { 2, 1, 0, 3 } },
    [NV097_SET_TEXTURE_FORMAT_COLOR_LU_IMAGE_R8G8B8A8] =
        { 4, true, MTL_PF_RGBA8_UNORM, false, { 0, 1, 2, 3 } },
};

const SurfaceFormatInfo kelvin_surface_color_format_map[] = {
    [NV097_SET_SURFACE_FORMAT_COLOR_LE_X1R5G5B5_Z1R5G5B5] =
        { 2, MTL_PF_A1BGR5_UNORM, false },
    [NV097_SET_SURFACE_FORMAT_COLOR_LE_R5G6B5] =
        { 2, MTL_PF_B5G6R5_UNORM, false },
    [NV097_SET_SURFACE_FORMAT_COLOR_LE_X8R8G8B8_Z8R8G8B8] =
        { 4, MTL_PF_BGRA8_UNORM, false },
    [NV097_SET_SURFACE_FORMAT_COLOR_LE_A8R8G8B8] =
        { 4, MTL_PF_BGRA8_UNORM, false },
    [NV097_SET_SURFACE_FORMAT_COLOR_LE_B8] =
        { 1, MTL_PF_R8_UNORM, false },
    [NV097_SET_SURFACE_FORMAT_COLOR_LE_G8B8] =
        { 2, MTL_PF_RG8_UNORM, false },
};

const SurfaceFormatInfo kelvin_surface_zeta_float_format_map[] = {
    [NV097_SET_SURFACE_FORMAT_ZETA_Z16] =
        { 2, MTL_PF_DEPTH32_FLOAT, false },
    [NV097_SET_SURFACE_FORMAT_ZETA_Z24S8] =
        { 4, MTL_PF_DEPTH24_UNORM_STENCIL8, true },
};

const SurfaceFormatInfo kelvin_surface_zeta_fixed_format_map[] = {
    [NV097_SET_SURFACE_FORMAT_ZETA_Z16] =
        { 2, MTL_PF_DEPTH16_UNORM, false },
    [NV097_SET_SURFACE_FORMAT_ZETA_Z24S8] =
        { 4, MTL_PF_DEPTH24_UNORM_STENCIL8, true },
};

/* MTLBlendFactor */
uint32_t nv20_mtl_blend_factor(unsigned int factor)
{
    static const uint32_t map[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14,
    };
    return factor < ARRAY_SIZE(map) ? map[factor] : 1;
}

uint32_t nv20_mtl_blend_op(unsigned int equation)
{
    static const uint32_t map[] = { 2, 3, 0, 4, 5, 3, 0 };
    return equation < ARRAY_SIZE(map) ? map[equation] : 0;
}

uint32_t nv20_mtl_compare(unsigned int func)
{
    static const uint32_t map[] = { 7, 0, 1, 2, 3, 4, 5, 6 };
    return func < ARRAY_SIZE(map) ? map[func] : 6;
}

uint32_t nv20_mtl_stencil_op(unsigned int op)
{
    static const uint32_t map[] = { 0, 0, 1, 2, 3, 4, 5, 6, 7 };
    return op < ARRAY_SIZE(map) ? map[op] : 0;
}

uint32_t nv20_mtl_sampler_address(unsigned int addr)
{
    static const uint32_t map[] = { 0, 2, 3, 1, 4, 1 };
    return addr < ARRAY_SIZE(map) ? map[addr] : 1;
}

uint32_t nv20_mtl_sampler_min_filter(unsigned int filter)
{
    static const uint32_t map[] = { 0, 0, 1, 2, 3, 4, 5, 1 };
    return filter < ARRAY_SIZE(map) ? map[filter] : 1;
}

uint32_t nv20_mtl_sampler_mag_filter(unsigned int filter)
{
    static const uint32_t map[] = { 0, 0, 1, 0, 1 };
    return filter < ARRAY_SIZE(map) ? map[filter] : 1;
}

uint32_t nv20_mtl_cull_mode(unsigned int mode)
{
    static const uint32_t map[] = { 0, 1, 2, 3 };
    return mode < ARRAY_SIZE(map) ? map[mode] : 0;
}

uint32_t nv20_mtl_winding(bool front_cw)
{
    return front_cw ? 1 : 0;
}
