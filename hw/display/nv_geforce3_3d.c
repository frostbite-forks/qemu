/*
 * QEMU NVIDIA GeForce3 (NV20) emulation -- the Celsius/Kelvin
 * fixed-function-and-vertex-shader 3D pipeline (PFIFO object classes
 * 0x0096, 0x0097 and 0x0497).
 *
 * Ported from Bochs' geforce.cc (see nv_geforce3_int.h). Every branch
 * in the original code keyed on a different Bochs "card_type" (NV1x/
 * NV3x/NV4x) has been resolved to this device's fixed NV20 behaviour;
 * in particular:
 *  - class 0x4097 (Curie/NV40) is never reachable on NV20 -- PFIFO's
 *    class mask is 12 bits here (real hardware limit, not a
 *    simplification), so a handful of Bochs' d3d_mh_* handlers that
 *    only that class ever dispatched to (surface pitch Z, the
 *    indexed vertex-data base, and the NV40 vertex-shader output
 *    lookup tables) have no port here at all.
 *  - d3d_scissor_clip()/d3d_viewport_clip()/d3d_window_clip() are
 *    NV3x-and-later features; on NV20 they are unconditional passes.
 *  - the vertex shader's instruction encoding is NV20's own (Bochs'
 *    "card_type == 0x20" decode), not the later NV3x/NV4x layouts.
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include <math.h>

#include "nv_geforce3_int.h"

/* ---------------------------------------------------------------- */
/* Small math helpers                                                */

static float nv_geforce3_uint32_as_float(uint32_t val)
{
    union {
        uint32_t u;
        float f;
    } conv;

    conv.u = val;
    return conv.f;
}

static double nv_geforce3_edge_function(const float v0[4], const float v1[4],
                                        const float v2[4])
{
    return ((double)v1[0] - v0[0]) * ((double)v2[1] - v0[1]) -
           ((double)v1[1] - v0[1]) * ((double)v2[0] - v0[0]);
}

static float nv_geforce3_dot3(const float x[3], const float y[3])
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

static float nv_geforce3_dot4(const float x[4], const float y[4])
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}

static void nv_geforce3_reflection(const float axis[3],
                                   const float direction[3],
                                   float refl_dir[3])
{
    float k = 2.0f * nv_geforce3_dot3(axis, direction) /
             nv_geforce3_dot3(axis, axis);

    refl_dir[0] = k * axis[0] - direction[0];
    refl_dir[1] = k * axis[1] - direction[1];
    refl_dir[2] = k * axis[2] - direction[2];
}

static void nv_geforce3_dot_map(uint32_t func, const float src[4],
                                float dst[3])
{
    int ci;

    switch (func) {
    default:
    case 0:
        for (ci = 0; ci < 3; ci++) {
            dst[ci] = src[ci];
        }
        break;
    case 1:
        for (ci = 0; ci < 3; ci++) {
            dst[ci] = (src[ci] * 255.0f - 128.0f) / 127.0f;
        }
        break;
    }
}

static float nv_geforce3_dot3_map(const float x[3], const float y[4],
                                  uint32_t map_func)
{
    float ym[3];

    nv_geforce3_dot_map(map_func, y, ym);
    return nv_geforce3_dot3(x, ym);
}

static float nv_geforce3_length2(const float v[2])
{
    return sqrtf(v[0] * v[0] + v[1] * v[1]);
}

static float nv_geforce3_length3(const float v[3])
{
    return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

static float nv_geforce3_normalize_ip(float v[3])
{
    float l = nv_geforce3_length3(v);
    float scale = 1.0f / l;

    v[0] *= scale;
    v[1] *= scale;
    v[2] *= scale;
    return l;
}

static void nv_geforce3_normalize(const float in[3], float out[3])
{
    float scale = 1.0f / nv_geforce3_length3(in);

    out[0] = in[0] * scale;
    out[1] = in[1] * scale;
    out[2] = in[2] * scale;
}

static void nv_geforce3_compute_partials_x(const float v0[4],
                                           const float v1[4], float ddx[4])
{
    uint32_t ci;

    for (ci = 0; ci < 4; ci++) {
        ddx[ci] = v1[ci] - v0[ci];
    }
}

static void nv_geforce3_compute_partials_y(const float v0[4],
                                           const float v2[4], float ddy[4])
{
    uint32_t ci;

    for (ci = 0; ci < 4; ci++) {
        ddy[ci] = v2[ci] - v0[ci];
    }
}

static void nv_geforce3_compute_partials(const float v0[3], const float v1[3],
                                         const float v2[3], float ddx[3],
                                         float ddy[3])
{
    uint32_t ci;

    for (ci = 0; ci < 3; ci++) {
        ddx[ci] = v1[ci] - v0[ci];
        ddy[ci] = v2[ci] - v0[ci];
    }
}

static void nv_geforce3_position_to_view3(NVGeForce3Channel *ch,
                                          const float p[4], float pt[3])
{
    const float *m = ch->d3d_model_view_matrix[0];

    pt[0] = p[0] * m[0] + p[1] * m[1] + p[2] * m[2] + p[3] * m[3];
    pt[1] = p[0] * m[4] + p[1] * m[5] + p[2] * m[6] + p[3] * m[7];
    pt[2] = p[0] * m[8] + p[1] * m[9] + p[2] * m[10] + p[3] * m[11];
}

static void nv_geforce3_position_to_view4(NVGeForce3Channel *ch,
                                          const float p[4], float pt[4])
{
    const float *m = ch->d3d_model_view_matrix[0];

    pt[0] = p[0] * m[0] + p[1] * m[1] + p[2] * m[2] + p[3] * m[3];
    pt[1] = p[0] * m[4] + p[1] * m[5] + p[2] * m[6] + p[3] * m[7];
    pt[2] = p[0] * m[8] + p[1] * m[9] + p[2] * m[10] + p[3] * m[11];
    pt[3] = p[0] * m[12] + p[1] * m[13] + p[2] * m[14] + p[3] * m[15];
}

static void nv_geforce3_normal_to_view(NVGeForce3Channel *ch,
                                       const float n[3], float nt[3])
{
    const float *m = ch->d3d_inverse_model_view_matrix;

    nt[0] = n[0] * m[0] + n[1] * m[1] + n[2] * m[2];
    nt[1] = n[0] * m[4] + n[1] * m[5] + n[2] * m[6];
    nt[2] = n[0] * m[8] + n[1] * m[9] + n[2] * m[10];
    if (ch->d3d_normalize_enable) {
        nv_geforce3_normalize_ip(nt);
    }
}

static void nv_geforce3_unpack_attribute(uint32_t value, bool d3d,
                                         float comp[4])
{
    if (d3d) {
        comp[0] = ((value >> 16) & 0xff) / 255.0f;
        comp[1] = ((value >> 8) & 0xff) / 255.0f;
        comp[2] = (value & 0xff) / 255.0f;
        comp[3] = ((value >> 24) & 0xff) / 255.0f;
    } else {
        int i;

        for (i = 0; i < 4; i++) {
            comp[i] = ((value >> (i * 8)) & 0xff) / 255.0f;
        }
    }
}

/* ---------------------------------------------------------------- */
/* Clipping: NV3x-and-later features, unconditional passes on NV20   */

static bool nv_geforce3_d3d_scissor_clip(NVGeForce3Channel *ch, uint32_t *x,
                                         uint32_t *y, uint32_t *width,
                                         uint32_t *height)
{
    return true;
}

static bool nv_geforce3_d3d_viewport_clip(NVGeForce3Channel *ch, uint32_t *x,
                                          uint32_t *y, uint32_t *width,
                                          uint32_t *height)
{
    return true;
}

static bool nv_geforce3_d3d_window_clip(NVGeForce3Channel *ch, uint32_t *x,
                                        uint32_t *y, uint32_t *width,
                                        uint32_t *height)
{
    return true;
}

uint32_t nv_geforce3_d3d_get_surface_pitch_z(NVGeForce3Channel *ch)
{
    return ch->d3d_surface_pitch_a >> 16;
}

void nv_geforce3_d3d_clear_surface(NVGeForce3State *s, NVGeForce3Channel *ch)
{
    uint32_t dx = ch->d3d_clip_horizontal & 0xFFFF;
    uint32_t dy = ch->d3d_clip_vertical & 0xFFFF;
    uint32_t width = ch->d3d_clip_horizontal >> 16;
    uint32_t height = ch->d3d_clip_vertical >> 16;
    bool depth_clear, stencil_clear;
    uint32_t x, y;

    if (!nv_geforce3_d3d_scissor_clip(ch, &dx, &dy, &width, &height)) {
        return;
    }
    if (ch->d3d_clear_surface & 0x000000F0) {
        uint32_t pitch = ch->d3d_surface_pitch_a & 0xFFFF;
        uint32_t draw_offset = ch->d3d_surface_color_offset +
                              dy * pitch + dx * ch->d3d_color_bytes;

        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                if (ch->d3d_color_bytes == 2) {
                    nv_geforce3_dma_write16(s, ch->d3d_color_obj,
                        draw_offset + x * 2,
                        (uint16_t)ch->d3d_color_clear_value);
                } else {
                    nv_geforce3_dma_write32(s, ch->d3d_color_obj,
                        draw_offset + x * 4, ch->d3d_color_clear_value);
                }
            }
            draw_offset += pitch;
        }
    }
    depth_clear = (ch->d3d_clear_surface & 0x00000001) != 0;
    stencil_clear = (ch->d3d_clear_surface & 0x00000002) != 0;
    if (depth_clear || stencil_clear) {
        uint32_t pitch = nv_geforce3_d3d_get_surface_pitch_z(ch);
        uint32_t draw_offset = ch->d3d_surface_zeta_offset +
                              dy * pitch + dx * ch->d3d_depth_bytes;

        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                if (ch->d3d_depth_bytes == 2) {
                    if (depth_clear) {
                        nv_geforce3_dma_write16(s, ch->d3d_zeta_obj,
                            draw_offset + x * 2,
                            (uint16_t)ch->d3d_zstencil_clear_value);
                    }
                } else if (depth_clear) {
                    if (stencil_clear) {
                        nv_geforce3_dma_write32(s, ch->d3d_zeta_obj,
                            draw_offset + x * 4,
                            ch->d3d_zstencil_clear_value);
                    } else {
                        nv_geforce3_dma_write8(s, ch->d3d_zeta_obj,
                            draw_offset + x * 4 + 1,
                            (uint8_t)(ch->d3d_zstencil_clear_value >> 8));
                        nv_geforce3_dma_write16(s, ch->d3d_zeta_obj,
                            draw_offset + x * 4 + 2,
                            (uint16_t)(ch->d3d_zstencil_clear_value >> 16));
                    }
                } else {
                    nv_geforce3_dma_write8(s, ch->d3d_zeta_obj,
                        draw_offset + x * 4,
                        (uint8_t)ch->d3d_zstencil_clear_value);
                }
            }
            draw_offset += pitch;
        }
    }
}

/* ---------------------------------------------------------------- */
/* Texture format decode / sampling                                  */

static void nv_geforce3_d3d_texture_process_format(NVGeForce3Texture *tex)
{
    tex->linear = false;
    tex->unnormalized = false;
    tex->compressed = false;
    tex->dxt_alpha_data = false;
    tex->dxt_alpha_explicit = false;
    if (tex->format & 0x80) {
        if (tex->format & 0x20) {
            tex->linear = true;
        }
        if (tex->format & 0x40) {
            tex->unnormalized = true;
        }
        tex->format &= 0x9f;
    } else if (tex->format == 0x12 || tex->format == 0x1b ||
               tex->format == 0x1e) {
        tex->linear = true;
        tex->unnormalized = true;
    }
    switch (tex->format) {
    case 0x0c:
    case 0x0e:
    case 0x0f:
    case 0x86:
    case 0x87:
    case 0x88:
        tex->compressed = true;
        tex->dxt_alpha_data = tex->format != 0x0c && tex->format != 0x86;
        tex->dxt_alpha_explicit = tex->format == 0x0e || tex->format == 0x87;
        tex->color_bytes = tex->dxt_alpha_data ? 16 : 8;
        break;
    case 0x02:
    case 0x03:
    case 0x04:
    case 0x05:
    case 0x27:
    case 0x28:
    case 0x82:
    case 0x83:
    case 0x84:
    case 0x8b:
    case 0x8f:
        tex->color_bytes = 2;
        break;
    case 0x06:
    case 0x07:
    case 0x12:
    case 0x1e:
    case 0x3a:
    case 0x85:
        tex->color_bytes = 4;
        break;
    case 0x00:
    case 0x01:
    case 0x0b:
    case 0x19:
    case 0x1b:
    case 0x81:
        tex->color_bytes = 1;
        break;
    default:
        qemu_log_mask(LOG_UNIMP, "nv-geforce3: unknown texture format "
                      "0x%02x\n", tex->format);
        tex->color_bytes = 1;
        break;
    }
}

static void nv_geforce3_texture_update_size(NVGeForce3Texture *tex,
                                            uint32_t cls)
{
    uint32_t lw, lh, ld, lod;

    if (tex->linear || cls >= 0x4097) {
        lw = tex->size_npot[0];
        lh = tex->dimensions > 1 ? tex->size_npot[1] : 1;
        ld = tex->dimensions > 2 ? tex->size_npot[2] : 1;
    } else {
        lw = 1u << tex->size_log[0];
        lh = 1u << tex->size_log[1];
        ld = 1u << tex->size_log[2];
    }
    tex->face_bytes = 0;
    for (lod = 0; lod < tex->levels; lod++) {
        uint32_t level_bytes;

        tex->sizes[lod][0] = lw;
        tex->sizes[lod][1] = lh;
        tex->sizes[lod][2] = ld;
        if (tex->linear) {
            level_bytes = tex->pitch * lh * ld;
        } else if (tex->compressed) {
            level_bytes = tex->color_bytes * QEMU_ALIGN_UP(lw, 4) *
                          QEMU_ALIGN_UP(lh, 4) / 16 * ld;
        } else {
            level_bytes = tex->color_bytes * lw * lh * ld;
        }
        tex->level_offset[lod] = tex->face_bytes;
        tex->face_bytes += level_bytes;
        lw /= 2;
        lh /= 2;
        ld /= 2;
        if (lw == 0) {
            lw = 1;
        }
        if (lh == 0) {
            lh = 1;
        }
        if (ld == 0) {
            ld = 1;
        }
    }
    tex->face_bytes = QEMU_ALIGN_UP(tex->face_bytes, 128);
}

static void nv_geforce3_d3d_sample_texel(NVGeForce3State *s,
                                         NVGeForce3Channel *ch,
                                         NVGeForce3Texture *tex,
                                         const int32_t coords_in[3],
                                         uint32_t face, uint32_t lod,
                                         float color[4])
{
    uint32_t coords[3] = { 0 };
    uint32_t *lod_size = tex->sizes[lod];
    uint32_t tex_ofs;
    int32_t color_int[4] = { 0 };
    float color_scale[4] = { 0 };
    uint32_t d, i;

    for (d = 0; d < tex->dimensions; d++) {
        int32_t c = coords_in[d];

        coords[d] = (uint32_t)c;
        if (c < 0 || (uint32_t)c >= lod_size[d]) {
            switch (tex->wrap[d]) {
            case 1:
                coords[d] = (c < 0) ? lod_size[d] - 1 : 0;
                break;
            case 4:
            case 5:
                for (i = 0; i < 4; i++) {
                    color[i] = tex->border_color[i];
                }
                return;
            default:
                color[0] = 1.0f;
                color[1] = 0.0f;
                color[2] = 0.0f;
                color[3] = 1.0f;
                return;
            }
        }
    }
    tex_ofs = tex->offset + tex->level_offset[lod] + face * tex->face_bytes;
    if (tex->compressed) {
        uint32_t pitch = lod_size[0] * (tex->dxt_alpha_data ? 4 : 2);
        uint32_t bx = coords[0] >> 2, by = coords[1] >> 2;

        tex_ofs += by * pitch + bx * tex->color_bytes;
    } else if (tex->linear) {
        tex_ofs += coords[1] * tex->pitch + coords[0] * tex->color_bytes;
    } else {
        tex_ofs += nv_geforce3_swizzle(coords[0], coords[1], coords[2],
                                       lod_size[0], lod_size[1],
                                       lod_size[2]) * tex->color_bytes;
    }

    switch (tex->format) {
    case 0x0c:
    case 0x0e:
    case 0x0f:
    case 0x86:
    case 0x87:
    case 0x88: {
        uint32_t ox = coords[0] & 3, oy = coords[1] & 3;
        uint64_t color_word;
        uint32_t color_index;
        uint16_t color0, color1;

        if (tex->dxt_alpha_data) {
            uint64_t alpha_word = nv_geforce3_dma_read64(s, tex->dma_obj,
                                                          tex_ofs);

            if (tex->dxt_alpha_explicit) {
                color_int[0] = (alpha_word >> (oy * 16 + ox * 4)) & 0xf;
                color_scale[0] = 1.0f / 15.0f;
            } else {
                uint8_t alpha0 = (uint8_t)alpha_word;
                uint8_t alpha1 = (uint8_t)(alpha_word >> 8);
                uint32_t alpha_index = (alpha_word >> (16 + oy * 12 +
                                                        ox * 3)) & 7;

                switch (alpha_index) {
                case 0:
                    color_int[0] = alpha0;
                    color_scale[0] = 1.0f / 255.0f;
                    break;
                case 1:
                    color_int[0] = alpha1;
                    color_scale[0] = 1.0f / 255.0f;
                    break;
                case 2:
                    if (alpha0 > alpha1) {
                        color_int[0] = 6 * alpha0 + alpha1;
                        color_scale[0] = 1.0f / 1785.0f;
                    } else {
                        color_int[0] = 4 * alpha0 + alpha1;
                        color_scale[0] = 1.0f / 1275.0f;
                    }
                    break;
                case 3:
                    if (alpha0 > alpha1) {
                        color_int[0] = 5 * alpha0 + 2 * alpha1;
                        color_scale[0] = 1.0f / 1785.0f;
                    } else {
                        color_int[0] = 3 * alpha0 + 2 * alpha1;
                        color_scale[0] = 1.0f / 1275.0f;
                    }
                    break;
                case 4:
                    if (alpha0 > alpha1) {
                        color_int[0] = 4 * alpha0 + 3 * alpha1;
                        color_scale[0] = 1.0f / 1785.0f;
                    } else {
                        color_int[0] = 2 * alpha0 + 3 * alpha1;
                        color_scale[0] = 1.0f / 1275.0f;
                    }
                    break;
                case 5:
                    if (alpha0 > alpha1) {
                        color_int[0] = 3 * alpha0 + 4 * alpha1;
                        color_scale[0] = 1.0f / 1785.0f;
                    } else {
                        color_int[0] = alpha0 + 4 * alpha1;
                        color_scale[0] = 1.0f / 1275.0f;
                    }
                    break;
                case 6:
                    if (alpha0 > alpha1) {
                        color_int[0] = 2 * alpha0 + 5 * alpha1;
                        color_scale[0] = 1.0f / 1785.0f;
                    } else {
                        color_int[0] = 0;
                        color_scale[0] = 1.0f;
                    }
                    break;
                case 7:
                    if (alpha0 > alpha1) {
                        color_int[0] = alpha0 + 6 * alpha1;
                        color_scale[0] = 1.0f / 1785.0f;
                    } else {
                        color_int[0] = 1;
                        color_scale[0] = 1.0f;
                    }
                    break;
                }
            }
        } else {
            color_int[0] = 1;
            color_scale[0] = 1.0f;
        }
        color_word = nv_geforce3_dma_read64(s, tex->dma_obj,
            tex_ofs + (tex->dxt_alpha_data ? 8 : 0));
        color_index = (color_word >> (32 + oy * 8 + ox * 2)) & 3;
        color0 = (uint16_t)color_word;
        color1 = (uint16_t)(color_word >> 16);
        switch (color_index) {
        case 0:
            color_int[1] = (color0 >> 11) & 0x1f;
            color_scale[1] = 1.0f / 31.0f;
            color_int[2] = (color0 >> 5) & 0x3f;
            color_scale[2] = 1.0f / 63.0f;
            color_int[3] = color0 & 0x1f;
            color_scale[3] = 1.0f / 31.0f;
            break;
        case 1:
            color_int[1] = (color1 >> 11) & 0x1f;
            color_scale[1] = 1.0f / 31.0f;
            color_int[2] = (color1 >> 5) & 0x3f;
            color_scale[2] = 1.0f / 63.0f;
            color_int[3] = color1 & 0x1f;
            color_scale[3] = 1.0f / 31.0f;
            break;
        case 2:
            if (color0 > color1) {
                color_int[1] = 2 * ((color0 >> 11) & 0x1f) +
                               ((color1 >> 11) & 0x1f);
                color_scale[1] = 1.0f / 93.0f;
                color_int[2] = 2 * ((color0 >> 5) & 0x3f) +
                               ((color1 >> 5) & 0x3f);
                color_scale[2] = 1.0f / 189.0f;
                color_int[3] = 2 * (color0 & 0x1f) + (color1 & 0x1f);
                color_scale[3] = 1.0f / 93.0f;
            } else {
                color_int[1] = ((color0 >> 11) & 0x1f) +
                               ((color1 >> 11) & 0x1f);
                color_scale[1] = 1.0f / 62.0f;
                color_int[2] = ((color0 >> 5) & 0x3f) +
                               ((color1 >> 5) & 0x3f);
                color_scale[2] = 1.0f / 126.0f;
                color_int[3] = (color0 & 0x1f) + (color1 & 0x1f);
                color_scale[3] = 1.0f / 62.0f;
            }
            break;
        case 3:
            if (color0 > color1) {
                color_int[1] = 2 * ((color1 >> 11) & 0x1f) +
                               ((color0 >> 11) & 0x1f);
                color_scale[1] = 1.0f / 93.0f;
                color_int[2] = 2 * ((color1 >> 5) & 0x3f) +
                               ((color0 >> 5) & 0x3f);
                color_scale[2] = 1.0f / 189.0f;
                color_int[3] = 2 * (color1 & 0x1f) + (color0 & 0x1f);
                color_scale[3] = 1.0f / 93.0f;
            } else {
                color_int[0] = 0;
                color_scale[0] = 1.0f;
                color_int[1] = 0;
                color_scale[1] = 1.0f;
                color_int[2] = 0;
                color_scale[2] = 1.0f;
                color_int[3] = 0;
                color_scale[3] = 1.0f;
            }
            break;
        }
        break;
    }
    case 0x04:
    case 0x83: {
        uint16_t value = nv_geforce3_dma_read16(s, tex->dma_obj, tex_ofs);

        color_int[0] = (value >> 12) & 0xf;
        color_scale[0] = 1.0f / 15.0f;
        color_int[1] = (value >> 8) & 0xf;
        color_scale[1] = 1.0f / 15.0f;
        color_int[2] = (value >> 4) & 0xf;
        color_scale[2] = 1.0f / 15.0f;
        color_int[3] = value & 0xf;
        color_scale[3] = 1.0f / 15.0f;
        break;
    }
    case 0x05:
    case 0x84: {
        uint16_t value = nv_geforce3_dma_read16(s, tex->dma_obj, tex_ofs);

        color_int[0] = 1;
        color_scale[0] = 1.0f;
        color_int[1] = (value >> 11) & 0x1f;
        color_scale[1] = 1.0f / 31.0f;
        color_int[2] = (value >> 5) & 0x3f;
        color_scale[2] = 1.0f / 63.0f;
        color_int[3] = value & 0x1f;
        color_scale[3] = 1.0f / 31.0f;
        break;
    }
    case 0x02:
    case 0x82: {
        uint16_t value = nv_geforce3_dma_read16(s, tex->dma_obj, tex_ofs);

        if ((tex->control0 & 3) != 0 && value == tex->key_color) {
            color_int[0] = 0;
        } else {
            color_int[0] = (value >> 15) & 1;
        }
        color_scale[0] = 1.0f;
        color_int[1] = (value >> 10) & 0x1f;
        color_scale[1] = 1.0f / 31.0f;
        color_int[2] = (value >> 5) & 0x1f;
        color_scale[2] = 1.0f / 31.0f;
        color_int[3] = value & 0x1f;
        color_scale[3] = 1.0f / 31.0f;
        break;
    }
    case 0x03: {
        uint16_t value = nv_geforce3_dma_read16(s, tex->dma_obj, tex_ofs);

        color_int[0] = 1;
        color_scale[0] = 1.0f;
        color_int[1] = (value >> 10) & 0x1f;
        color_scale[1] = 1.0f / 31.0f;
        color_int[2] = (value >> 5) & 0x1f;
        color_scale[2] = 1.0f / 31.0f;
        color_int[3] = value & 0x1f;
        color_scale[3] = 1.0f / 31.0f;
        break;
    }
    case 0x27:
    case 0x8f: {
        uint16_t value = nv_geforce3_dma_read16(s, tex->dma_obj, tex_ofs);

        color_int[0] = 1;
        color_scale[0] = 1.0f;
        color_int[1] = (value >> 10) & 0x3f;
        color_scale[1] = 1.0f / 63.0f;
        color_int[2] = (value >> 5) & 0x1f;
        color_scale[2] = 1.0f / 31.0f;
        color_int[3] = value & 0x1f;
        color_scale[3] = 1.0f / 31.0f;
        break;
    }
    case 0x28:
    case 0x8b: {
        uint16_t value = nv_geforce3_dma_read16(s, tex->dma_obj, tex_ofs);

        color_int[0] = 1;
        color_scale[0] = 1.0f;
        color_int[1] = 1;
        color_scale[1] = 1.0f;
        color_int[2] = (value >> 8) & 0xff;
        color_scale[2] = 1.0f / 255.0f;
        color_int[3] = value & 0xff;
        color_scale[3] = 1.0f / 255.0f;
        break;
    }
    case 0x06:
    case 0x12:
    case 0x85: {
        uint32_t value = nv_geforce3_dma_read32(s, tex->dma_obj, tex_ofs);

        color_int[0] = (value >> 24) & 0xff;
        color_scale[0] = 1.0f / 255.0f;
        color_int[1] = (value >> 16) & 0xff;
        color_scale[1] = 1.0f / 255.0f;
        color_int[2] = (value >> 8) & 0xff;
        color_scale[2] = 1.0f / 255.0f;
        color_int[3] = value & 0xff;
        color_scale[3] = 1.0f / 255.0f;
        break;
    }
    case 0x3a: {
        uint32_t value = nv_geforce3_dma_read32(s, tex->dma_obj, tex_ofs);

        color_int[0] = (value >> 24) & 0xff;
        color_scale[0] = 1.0f / 255.0f;
        color_int[1] = value & 0xff;
        color_scale[1] = 1.0f / 255.0f;
        color_int[2] = (value >> 8) & 0xff;
        color_scale[2] = 1.0f / 255.0f;
        color_int[3] = (value >> 16) & 0xff;
        color_scale[3] = 1.0f / 255.0f;
        break;
    }
    case 0x07:
    case 0x1e: {
        uint32_t value = nv_geforce3_dma_read32(s, tex->dma_obj, tex_ofs);

        color_int[0] = 1;
        color_scale[0] = 1.0f;
        color_int[1] = (value >> 16) & 0xff;
        color_scale[1] = 1.0f / 255.0f;
        color_int[2] = (value >> 8) & 0xff;
        color_scale[2] = 1.0f / 255.0f;
        color_int[3] = value & 0xff;
        color_scale[3] = 1.0f / 255.0f;
        break;
    }
    case 0x0b: {
        uint32_t pal_index = nv_geforce3_dma_read8(s, tex->dma_obj, tex_ofs);
        uint32_t value = nv_geforce3_dma_read32(s, tex->pal_dma_obj,
            tex->pal_ofs + pal_index * 4);

        color_int[0] = (value >> 24) & 0xff;
        color_scale[0] = 1.0f / 255.0f;
        color_int[1] = (value >> 16) & 0xff;
        color_scale[1] = 1.0f / 255.0f;
        color_int[2] = (value >> 8) & 0xff;
        color_scale[2] = 1.0f / 255.0f;
        color_int[3] = value & 0xff;
        color_scale[3] = 1.0f / 255.0f;
        break;
    }
    case 0x00:
    case 0x81: {
        uint8_t value = nv_geforce3_dma_read8(s, tex->dma_obj, tex_ofs);

        color_int[0] = 1;
        color_scale[0] = 1.0f;
        color_int[1] = value;
        color_scale[1] = 1.0f / 255.0f;
        color_int[2] = value;
        color_scale[2] = 1.0f / 255.0f;
        color_int[3] = value;
        color_scale[3] = 1.0f / 255.0f;
        break;
    }
    case 0x19: {
        uint8_t value = nv_geforce3_dma_read8(s, tex->dma_obj, tex_ofs);

        color_int[0] = value;
        color_scale[0] = 1.0f / 255.0f;
        color_int[1] = 1;
        color_scale[1] = 1.0f;
        color_int[2] = 1;
        color_scale[2] = 1.0f;
        color_int[3] = 1;
        color_scale[3] = 1.0f;
        break;
    }
    case 0x01:
    case 0x1b: {
        uint8_t value = nv_geforce3_dma_read8(s, tex->dma_obj, tex_ofs);

        color_int[0] = value;
        color_scale[0] = 1.0f / 255.0f;
        color_int[1] = value;
        color_scale[1] = 1.0f / 255.0f;
        color_int[2] = value;
        color_scale[2] = 1.0f / 255.0f;
        color_int[3] = value;
        color_scale[3] = 1.0f / 255.0f;
        break;
    }
    default:
        color_int[0] = 1;
        color_scale[0] = 0.8f;
        color_int[1] = 1;
        color_scale[1] = 0.8f + 0.2f * coords[0] / lod_size[0];
        color_int[2] = 1;
        color_scale[2] = 0.6f + 0.2f * coords[1] / lod_size[1];
        color_int[3] = 1;
        color_scale[3] = 0.6f + 0.2f * coords[2] / lod_size[2];
        break;
    }

    if (tex->signed_any) {
        for (i = 0; i < 4; i++) {
            if (tex->signed_comp[i]) {
                color_int[i] = (int8_t)color_int[i];
                color_scale[i] = 1.0f / 128.0f;
            }
        }
    }
    for (i = 0; i < 4; i++) {
        uint32_t j = (i + 3) & 3;

        switch (tex->s0[i]) {
        case 0:
            color[j] = 0.0f;
            break;
        case 1:
            color[j] = 1.0f;
            break;
        default: {
            uint32_t swz = tex->s1[i];

            color[j] = color_int[swz] * color_scale[swz];
            break;
        }
        }
    }
}

static void nv_geforce3_d3d_sample_texture(NVGeForce3State *s,
                                           NVGeForce3Channel *ch,
                                           NVGeForce3Texture *tex,
                                           const float coords_in[3],
                                           float lodf, float color[4])
{
    const float *coords;
    float coords_cubemap[3];
    uint32_t face;
    uint32_t filter, lod_samples = 1, li;
    uint32_t lodi[2] = { 0 };
    float lodf_frac = 0.0f;
    bool linear_coord;
    float colors[2][4] = { { 0.0f } };

    if (tex->cubemap) {
        float coords_abs[3];
        uint32_t i;

        for (i = 0; i < 3; i++) {
            coords_abs[i] = fabsf(coords_in[i]);
        }
        if (coords_abs[0] > coords_abs[1] && coords_abs[0] > coords_abs[2]) {
            coords_cubemap[0] = coords_cubemap[1] = 1.0f / coords_abs[0];
            coords_cubemap[1] *= -coords_in[1];
            if (coords_in[0] > 0.0f) {
                face = 0;
                coords_cubemap[0] *= -coords_in[2];
            } else {
                face = 1;
                coords_cubemap[0] *= coords_in[2];
            }
        } else if (coords_abs[1] > coords_abs[0] &&
                   coords_abs[1] > coords_abs[2]) {
            coords_cubemap[0] = coords_cubemap[1] = 1.0f / coords_abs[1];
            coords_cubemap[0] *= coords_in[0];
            if (coords_in[1] > 0.0f) {
                face = 2;
                coords_cubemap[1] *= coords_in[2];
            } else {
                face = 3;
                coords_cubemap[1] *= -coords_in[2];
            }
        } else {
            coords_cubemap[0] = coords_cubemap[1] = 1.0f / coords_abs[2];
            coords_cubemap[1] *= -coords_in[1];
            if (coords_in[2] > 0.0f) {
                face = 4;
                coords_cubemap[0] *= coords_in[0];
            } else {
                face = 5;
                coords_cubemap[0] *= -coords_in[0];
            }
        }
        coords_cubemap[0] = 0.5f * coords_cubemap[0] + 0.5f;
        coords_cubemap[1] = 0.5f * coords_cubemap[1] + 0.5f;
        coords_cubemap[2] = 0.0f;
        coords = coords_cubemap;
    } else {
        face = 0;
        coords = coords_in;
    }

    filter = lodf < 0.0f ? tex->filter_mag : tex->filter_min;
    linear_coord = filter == 2 || filter == 4 || filter == 6;
    if (filter > 2) {
        if (filter == 3 || filter == 4) {
            lodi[0] = (uint32_t)(lodf + 0.5f);
        } else if (filter == 5 || filter == 6) {
            lod_samples = 2;
            lodi[0] = (uint32_t)lodf;
            lodi[1] = (uint32_t)(lodf + 1.0f);
            lodf_frac = lodf - lodi[0];
        }
        for (li = 0; li < lod_samples; li++) {
            if (lodi[li] >= tex->levels) {
                lodi[li] = tex->levels - 1;
            }
        }
        if (lod_samples == 2 && lodi[0] == lodi[1]) {
            lod_samples = 1;
        }
    }

    for (li = 0; li < lod_samples; li++) {
        uint32_t *lod_size = tex->sizes[lodi[li]];
        float coords_w[3] = { 0.0f };
        uint32_t d;

        for (d = 0; d < tex->dimensions; d++) {
            float c = coords[d];

            if (!tex->unnormalized) {
                c *= lod_size[d];
            }
            if (c < 0.5f || c > lod_size[d] - 0.5f) {
                switch (tex->wrap[d]) {
                case 1:
                    c = fmodf(c, lod_size[d]);
                    if (c < 0.0f) {
                        c += lod_size[d];
                    }
                    break;
                case 4:
                    if (c < -0.5f) {
                        c = -0.5f;
                    }
                    if (c > lod_size[d] + 0.5f) {
                        c = lod_size[d] + 0.5f;
                    }
                    break;
                case 5:
                    if (c < 0.0f) {
                        c = 0.0f;
                    }
                    if (c >= lod_size[d]) {
                        c = nextafterf(lod_size[d], -INFINITY);
                    }
                    break;
                case 2:
                    c = fmodf(c, 2.0f * lod_size[d]);
                    if (c < 0.0f) {
                        c += 2.0f * lod_size[d];
                    }
                    if (c > lod_size[d]) {
                        c = 2.0f * lod_size[d] - c;
                    }
                    /* fallthrough */
                case 3:
                default:
                    if (c < 0.5f) {
                        c = 0.5f;
                    }
                    if (c > lod_size[d] - 0.5f) {
                        c = lod_size[d] - 0.5f;
                    }
                    break;
                }
            }
            coords_w[d] = c;
        }
        if (linear_coord) {
            float coords_sf[3];
            float coeffs[3][2];
            int32_t coords_i[3] = { 0 };
            float corner[4];

            for (d = 0; d < tex->dimensions; d++) {
                float coord_s = coords_w[d] - 0.5f;

                coords_sf[d] = floorf(coord_s);
                coeffs[d][1] = coord_s - coords_sf[d];
                coeffs[d][0] = 1.0f - coeffs[d][1];
            }
            switch (tex->dimensions) {
            case 1:
            default: {
                uint32_t cx;

                for (cx = 0; cx < 2; cx++) {
                    uint32_t ci;
                    float k0 = coeffs[0][cx];

                    coords_i[0] = (int32_t)coords_sf[0] + cx;
                    nv_geforce3_d3d_sample_texel(s, ch, tex, coords_i, face,
                                                 lodi[li], corner);
                    for (ci = 0; ci < 4; ci++) {
                        colors[li][ci] += k0 * corner[ci];
                    }
                }
                break;
            }
            case 2: {
                uint32_t cx, cy;

                for (cy = 0; cy < 2; cy++) {
                    float k1 = coeffs[1][cy];

                    coords_i[1] = (int32_t)coords_sf[1] + cy;
                    for (cx = 0; cx < 2; cx++) {
                        uint32_t ci;
                        float k0 = k1 * coeffs[0][cx];

                        coords_i[0] = (int32_t)coords_sf[0] + cx;
                        nv_geforce3_d3d_sample_texel(s, ch, tex, coords_i,
                                                     face, lodi[li], corner);
                        for (ci = 0; ci < 4; ci++) {
                            colors[li][ci] += k0 * corner[ci];
                        }
                    }
                }
                break;
            }
            case 3: {
                uint32_t cx, cy, cz;

                for (cz = 0; cz < 2; cz++) {
                    float k2 = coeffs[2][cz];

                    coords_i[2] = (int32_t)coords_sf[2] + cz;
                    for (cy = 0; cy < 2; cy++) {
                        float k1 = k2 * coeffs[1][cy];

                        coords_i[1] = (int32_t)coords_sf[1] + cy;
                        for (cx = 0; cx < 2; cx++) {
                            uint32_t ci;
                            float k0 = k1 * coeffs[0][cx];

                            coords_i[0] = (int32_t)coords_sf[0] + cx;
                            nv_geforce3_d3d_sample_texel(s, ch, tex,
                                coords_i, face, lodi[li], corner);
                            for (ci = 0; ci < 4; ci++) {
                                colors[li][ci] += k0 * corner[ci];
                            }
                        }
                    }
                }
                break;
            }
            }
        } else {
            int32_t coords_i[3] = { 0 };

            for (d = 0; d < tex->dimensions; d++) {
                coords_i[d] = (int32_t)floorf(coords_w[d]);
            }
            nv_geforce3_d3d_sample_texel(s, ch, tex, coords_i, face,
                                         lodi[li], colors[li]);
        }
    }
    if (lod_samples == 1) {
        uint32_t ci;

        for (ci = 0; ci < 4; ci++) {
            color[ci] = colors[0][ci];
        }
    } else {
        float omlf = 1.0f - lodf_frac;
        uint32_t ci;

        for (ci = 0; ci < 4; ci++) {
            color[ci] = colors[0][ci] * omlf + colors[1][ci] * lodf_frac;
        }
    }
}

static float nv_geforce3_compute_lod(NVGeForce3Texture *tex,
                                     const float coords[3],
                                     const float ddx[3], const float ddy[3])
{
    float rho_x, rho_y;

    if (tex->cubemap) {
        float ddx_sel[3], ddy_sel[3], coords_sel[3], coords_abs[3];
        float ddx_face[2], ddy_face[2], k;
        uint32_t ci;

        for (ci = 0; ci < 3; ci++) {
            coords_abs[ci] = fabsf(coords[ci]);
        }
        if (coords_abs[0] > coords_abs[1] && coords_abs[0] > coords_abs[2]) {
            coords_sel[2] = coords_abs[0];
            coords_sel[1] = -coords[1];
            ddx_sel[1] = -ddx[1];
            ddy_sel[1] = -ddy[1];
            if (coords[0] > 0.0f) {
                coords_sel[0] = -coords[2];
                ddx_sel[0] = -ddx[2];
                ddy_sel[0] = -ddy[2];
                ddx_sel[2] = ddx[0];
                ddy_sel[2] = ddy[0];
            } else {
                coords_sel[0] = coords[2];
                ddx_sel[0] = ddx[2];
                ddy_sel[0] = ddy[2];
                ddx_sel[2] = -ddx[0];
                ddy_sel[2] = -ddy[0];
            }
        } else if (coords_abs[1] > coords_abs[0] &&
                   coords_abs[1] > coords_abs[2]) {
            coords_sel[2] = coords_abs[1];
            coords_sel[0] = coords[0];
            ddx_sel[0] = ddx[0];
            ddy_sel[0] = ddy[0];
            if (coords[1] > 0.0f) {
                coords_sel[1] = coords[2];
                ddx_sel[1] = ddx[2];
                ddy_sel[1] = ddy[2];
                ddx_sel[2] = ddx[1];
                ddy_sel[2] = ddy[1];
            } else {
                coords_sel[1] = -coords[2];
                ddx_sel[1] = -ddx[2];
                ddy_sel[1] = -ddy[2];
                ddx_sel[2] = -ddx[1];
                ddy_sel[2] = -ddy[1];
            }
        } else {
            coords_sel[2] = coords_abs[2];
            coords_sel[1] = -coords[1];
            ddx_sel[1] = -ddx[1];
            ddy_sel[1] = -ddy[1];
            if (coords[2] > 0.0f) {
                coords_sel[0] = coords[0];
                ddx_sel[0] = ddx[0];
                ddy_sel[0] = ddy[0];
                ddx_sel[2] = ddx[2];
                ddy_sel[2] = ddy[2];
            } else {
                coords_sel[0] = -coords[0];
                ddx_sel[0] = -ddx[0];
                ddy_sel[0] = -ddy[0];
                ddx_sel[2] = -ddx[2];
                ddy_sel[2] = -ddy[2];
            }
        }
        k = 0.5f * tex->sizes[0][0] / (coords_sel[2] * coords_sel[2]);
        ddx_face[0] = k * (coords_sel[2] * ddx_sel[0] -
                          coords_sel[0] * ddx_sel[2]);
        ddy_face[0] = k * (coords_sel[2] * ddy_sel[0] -
                          coords_sel[0] * ddy_sel[2]);
        ddx_face[1] = k * (coords_sel[2] * ddx_sel[1] -
                          coords_sel[1] * ddx_sel[2]);
        ddy_face[1] = k * (coords_sel[2] * ddy_sel[1] -
                          coords_sel[1] * ddy_sel[2]);
        rho_x = nv_geforce3_length2(ddx_face);
        rho_y = nv_geforce3_length2(ddy_face);
    } else {
        float ddx_scale[3], ddy_scale[3];
        uint32_t *tex_sizes = tex->sizes[0];
        uint32_t ci;

        for (ci = 0; ci < 3; ci++) {
            ddx_scale[ci] = ddx[ci] * tex_sizes[ci];
            ddy_scale[ci] = ddy[ci] * tex_sizes[ci];
        }
        rho_x = nv_geforce3_length3(ddx_scale);
        rho_y = nv_geforce3_length3(ddy_scale);
    }
    return log2f(MAX(rho_x, rho_y));
}

/* ---------------------------------------------------------------- */
/* Vertex shader (NV20 instruction encoding)                         */

static void nv_geforce3_d3d_vertex_shader(NVGeForce3Channel *ch,
                                          float in[16][4], float out[16][4])
{
    int32_t addr_regs[2][4] = { { 0 } };
    float tmp_regs[32][4];
    uint32_t op_index, r, ci;

    for (r = 0; r < 16; r++) {
        out[r][0] = 0.0f;
        out[r][1] = 0.0f;
        out[r][2] = 0.0f;
        out[r][3] = 1.0f;
    }
    for (r = 0; r < ch->d3d_vs_temp_regs_count; r++) {
        for (ci = 0; ci < 4; ci++) {
            tmp_regs[r][ci] = 0.0f;
        }
    }

    for (op_index = ch->d3d_transform_program_start; op_index < 544;
         op_index++) {
        uint32_t *tokens = ch->d3d_transform_program[op_index];
        float params[3][4];
        uint32_t vec_op, sca_op;
        bool addr_write = false;
        float vec_result[4], sca_result[4];
        int p;

        for (p = 0; p < 3; p++) {
            uint32_t tmp_index = 0, reg_type = 0;
            bool negate = false;
            uint32_t swizzle[4] = { 0 };
            int comp_index;

            if (p == 0) {
                reg_type = (tokens[2] >> 26) & 3;
                tmp_index = (tokens[2] >> 28) & 0xf;
                negate = (tokens[1] >> 8) & 1;
                for (comp_index = 0; comp_index < 4; comp_index++) {
                    swizzle[comp_index] =
                        (tokens[1] >> (6 - comp_index * 2)) & 3;
                }
            } else if (p == 1) {
                reg_type = (tokens[2] >> 11) & 3;
                tmp_index = (tokens[2] >> 13) & 0xf;
                negate = (tokens[2] >> 25) & 1;
                for (comp_index = 0; comp_index < 4; comp_index++) {
                    swizzle[comp_index] =
                        (tokens[2] >> (23 - comp_index * 2)) & 3;
                }
            } else {
                reg_type = (tokens[3] >> 28) & 3;
                tmp_index = ((tokens[2] & 3) << 2) | ((tokens[3] >> 30) & 3);
                negate = (tokens[2] >> 10) & 1;
                for (comp_index = 0; comp_index < 4; comp_index++) {
                    swizzle[comp_index] =
                        (tokens[2] >> (8 - comp_index * 2)) & 3;
                }
            }
            for (comp_index = 0; comp_index < 4; comp_index++) {
                int comp_index_swizzle = swizzle[comp_index];

                if (reg_type == 1) {
                    if (tmp_index == 12) {
                        params[p][comp_index] = out[0][comp_index_swizzle];
                    } else {
                        params[p][comp_index] =
                            tmp_regs[tmp_index][comp_index_swizzle];
                    }
                } else if (reg_type == 2) {
                    uint32_t in_index = (tokens[1] >> 9) & 0xf;

                    params[p][comp_index] = in[in_index][comp_index_swizzle];
                } else if (reg_type == 3) {
                    uint32_t const_index = (tokens[1] >> 13) & 0xff;

                    if ((tokens[3] >> 1) & 1) {
                        const_index = (const_index +
                            addr_regs[0][0]) & 0x1ff;
                    }
                    params[p][comp_index] =
                        ch->d3d_transform_constant[const_index]
                                                   [comp_index_swizzle];
                } else {
                    params[p][comp_index] = 0.0f;
                }
                if (negate) {
                    params[p][comp_index] = -params[p][comp_index];
                }
            }
        }

        vec_op = (tokens[1] >> 21) & 0xf;
        switch (vec_op) {
        case 0:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = 0.0f;
            }
            break;
        case 1:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = params[0][ci];
            }
            break;
        case 2:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = params[0][ci] * params[1][ci];
            }
            break;
        case 3:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = params[0][ci] + params[2][ci];
            }
            break;
        case 4:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = params[0][ci] * params[1][ci] +
                                 params[2][ci];
            }
            break;
        case 5: {
            float dp3 = nv_geforce3_dot3(params[0], params[1]);

            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = dp3;
            }
            break;
        }
        case 6: {
            float dph = nv_geforce3_dot3(params[0], params[1]) +
                       params[1][3];

            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = dph;
            }
            break;
        }
        case 7: {
            float dp4 = nv_geforce3_dot4(params[0], params[1]);

            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = dp4;
            }
            break;
        }
        case 8:
            vec_result[0] = 1.0f;
            vec_result[1] = params[0][1] * params[1][1];
            vec_result[2] = params[0][2];
            vec_result[3] = params[1][3];
            break;
        case 9:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = MIN(params[0][ci], params[1][ci]);
            }
            break;
        case 0xa:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = MAX(params[0][ci], params[1][ci]);
            }
            break;
        case 0xb:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = params[0][ci] < params[1][ci] ? 1.0f : 0.0f;
            }
            break;
        case 0xc:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = params[0][ci] >= params[1][ci] ? 1.0f :
                                                                  0.0f;
            }
            break;
        case 0xd:
            addr_write = true;
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = floorf(params[0][ci]);
            }
            break;
        case 0xe:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = params[0][ci] - floorf(params[0][ci]);
            }
            break;
        case 0xf:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = floorf(params[0][ci]);
            }
            break;
        case 0x10:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = params[0][ci] == params[1][ci] ? 1.0f :
                                                                  0.0f;
            }
            break;
        case 0x11:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = 0.0f;
            }
            break;
        case 0x12:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = params[0][ci] > params[1][ci] ? 1.0f : 0.0f;
            }
            break;
        case 0x13:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = params[0][ci] <= params[1][ci] ? 1.0f :
                                                                  0.0f;
            }
            break;
        case 0x14:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = params[0][ci] != params[1][ci] ? 1.0f :
                                                                  0.0f;
            }
            break;
        case 0x15:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = 1.0f;
            }
            break;
        case 0x16:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = params[0][ci] == 0.0f ? 0.0f :
                    params[0][ci] < 0.0f ? -1.0f : 1.0f;
            }
            break;
        default:
            for (ci = 0; ci < 4; ci++) {
                vec_result[ci] = 0.5f;
            }
            qemu_log_mask(LOG_UNIMP, "nv-geforce3: vertex shader unknown "
                          "VEC opcode 0x%02x\n", vec_op);
            break;
        }

        sca_op = (tokens[1] >> 25) & 7;
        switch (sca_op) {
        case 0:
            for (ci = 0; ci < 4; ci++) {
                sca_result[ci] = 0.0f;
            }
            break;
        case 1:
            for (ci = 0; ci < 4; ci++) {
                sca_result[ci] = params[2][ci];
            }
            break;
        case 2:
        case 3: {
            float rcp = 1.0f / params[2][0];

            for (ci = 0; ci < 4; ci++) {
                sca_result[ci] = rcp;
            }
            break;
        }
        case 4: {
            float rsq = 1.0f / sqrtf(fabsf(params[2][0]));

            for (ci = 0; ci < 4; ci++) {
                sca_result[ci] = rsq;
            }
            break;
        }
        case 5: {
            float fl = floorf(params[2][0]);

            sca_result[0] = exp2f(fl);
            sca_result[1] = params[2][0] - fl;
            sca_result[2] = exp2f(params[2][0]);
            sca_result[3] = 1.0f;
            break;
        }
        case 6: {
            float fa = fabsf(params[2][0]);

            if (fa != 0.0f) {
                if (isinf(fa)) {
                    sca_result[0] = INFINITY;
                    sca_result[1] = 1.0f;
                    sca_result[2] = INFINITY;
                } else {
                    sca_result[0] = floorf(log2f(fa));
                    sca_result[1] = fa / exp2f(floorf(log2f(fa)));
                    sca_result[2] = log2f(fa);
                }
            } else {
                sca_result[0] = -INFINITY;
                sca_result[1] = 1.0f;
                sca_result[2] = -INFINITY;
            }
            sca_result[3] = 1.0f;
            break;
        }
        case 7: {
            float tmpx = params[2][0], tmpy = params[2][1];
            float tmpw = params[2][3];
            const float epsilon = 1.0e-6f;

            if (tmpx < 0.0f) {
                tmpx = 0.0f;
            }
            if (tmpy < 0.0f) {
                tmpy = 0.0f;
            }
            if (tmpw < -(128.0f - epsilon)) {
                tmpw = -(128.0f - epsilon);
            } else if (tmpw > 128.0f - epsilon) {
                tmpw = 128.0f - epsilon;
            }
            sca_result[0] = 1.0f;
            sca_result[1] = tmpx;
            sca_result[2] = (tmpx > 0.0f) ? powf(tmpy, tmpw) : 0.0f;
            sca_result[3] = 1.0f;
            break;
        }
        default:
            for (ci = 0; ci < 4; ci++) {
                sca_result[ci] = 0.5f;
            }
            qemu_log_mask(LOG_UNIMP, "nv-geforce3: vertex shader unknown "
                          "SCA opcode 0x%02x\n", sca_op);
            break;
        }

        {
            uint32_t dst_out_reg = (tokens[3] >> 3) & 0xf;
            uint32_t dst_vec_mask = (tokens[3] >> 24) & 0xf;
            uint32_t dst_sca_mask = (tokens[3] >> 16) & 0xf;
            uint32_t dst_out_mask = (tokens[3] >> 12) & 0xf;
            uint32_t dst_tmp_reg = (tokens[3] >> 20) & 0xf;
            bool dst_out_sca = (tokens[3] >> 2) & 1;
            bool paired_ops = vec_op != 0 && sca_op != 0;
            int comp_index;

            for (comp_index = 0; comp_index < 4; comp_index++) {
                if (addr_write) {
                    if (comp_index == 0) {
                        addr_regs[0][0] = (int32_t)vec_result[0];
                    }
                } else if (dst_vec_mask & (8 >> comp_index)) {
                    tmp_regs[dst_tmp_reg][comp_index] =
                        vec_result[comp_index];
                }
                if (dst_sca_mask & (8 >> comp_index)) {
                    tmp_regs[paired_ops ? 1 : dst_tmp_reg][comp_index] =
                        sca_result[comp_index];
                }
                if (dst_out_mask & (8 >> comp_index)) {
                    out[dst_out_reg][comp_index] = dst_out_sca ?
                        sca_result[comp_index] : vec_result[comp_index];
                }
            }
        }

        if (tokens[3] & 1) {
            break;
        }
    }
}

/* ---------------------------------------------------------------- */
/* Register combiners / pixel shader                                 */

static float nv_geforce3_rc_get_var(uint32_t cw, uint32_t shift,
                                    float regs[16][4], uint32_t civ)
{
    uint32_t x = cw >> shift;
    uint32_t reg = x & 0xf;
    uint32_t pir = (x >> 4) & 1;
    uint32_t map = (x >> 5) & 7;
    uint32_t cir = pir ? 3 : civ;
    float value = regs[reg][cir];

    switch (map) {
    case 0:
        return MAX(0.0f, value);
    case 1:
        return 1.0f - MIN(MAX(value, 0.0f), 1.0f);
    case 2:
        return 2.0f * MAX(0.0f, value) - 1.0f;
    case 3:
        return -2.0f * MAX(0.0f, value) + 1.0f;
    case 4:
        return MAX(0.0f, value) - 0.5f;
    case 5:
        return -MAX(0.0f, value) + 0.5f;
    default:
    case 6:
        return value;
    case 7:
        return -value;
    }
}

static void nv_geforce3_d3d_register_combiners(NVGeForce3Channel *ch,
                                               float regs[16][4],
                                               float out[4])
{
    float vars_final[6][3];
    uint32_t s, ci, civ;

    for (s = 0; s < ch->d3d_combiner_control_num_stages; s++) {
        uint32_t icws[2] = {
            ch->d3d_combiner_color_icw[s], ch->d3d_combiner_alpha_icw[s]
        };
        float vars[4][4];
        uint32_t color_ocw, color_cd, color_ab, color_muxsum;
        bool color_cd_dot, color_ab_dot;
        uint32_t alpha_ocw, alpha_cd, alpha_ab, alpha_muxsum;

        if (icws[0] == 0 && icws[1] == 0) {
            continue;
        }
        for (ci = 0; ci < 4; ci++) {
            regs[1][ci] = ch->d3d_combiner_const_color[s][0][ci];
            regs[2][ci] = ch->d3d_combiner_const_color[s][1][ci];
        }
        for (civ = 0; civ < 4; civ++) {
            uint32_t icw = icws[civ == 3];

            vars[0][civ] = nv_geforce3_rc_get_var(icw, 24, regs, civ);
            vars[1][civ] = nv_geforce3_rc_get_var(icw, 16, regs, civ);
            vars[2][civ] = nv_geforce3_rc_get_var(icw, 8, regs, civ);
            vars[3][civ] = nv_geforce3_rc_get_var(icw, 0, regs, civ);
        }
        color_ocw = ch->d3d_combiner_color_ocw[s];
        color_cd = color_ocw & 0xf;
        color_ab = (color_ocw >> 4) & 0xf;
        color_muxsum = (color_ocw >> 8) & 0xf;
        color_cd_dot = (color_ocw & 0x00001000) != 0;
        color_ab_dot = (color_ocw & 0x00002000) != 0;
        if (color_ab != 0) {
            if (color_ab_dot) {
                float ab_dot = vars[0][0] * vars[1][0] +
                    vars[0][1] * vars[1][1] + vars[0][2] * vars[1][2];

                for (ci = 0; ci < 3; ci++) {
                    regs[color_ab][ci] = ab_dot;
                }
            } else {
                for (ci = 0; ci < 3; ci++) {
                    regs[color_ab][ci] = vars[0][ci] * vars[1][ci];
                }
            }
        }
        if (color_cd != 0) {
            if (color_cd_dot) {
                float cd_dot = vars[2][0] * vars[3][0] +
                    vars[2][1] * vars[3][1] + vars[2][2] * vars[3][2];

                for (ci = 0; ci < 3; ci++) {
                    regs[color_cd][ci] = cd_dot;
                }
            } else {
                for (ci = 0; ci < 3; ci++) {
                    regs[color_cd][ci] = vars[2][ci] * vars[3][ci];
                }
            }
        }
        if (color_muxsum != 0) {
            for (ci = 0; ci < 3; ci++) {
                regs[color_muxsum][ci] = vars[0][ci] * vars[1][ci] +
                                         vars[2][ci] * vars[3][ci];
            }
        }
        alpha_ocw = ch->d3d_combiner_alpha_ocw[s];
        alpha_cd = alpha_ocw & 0xf;
        alpha_ab = (alpha_ocw >> 4) & 0xf;
        alpha_muxsum = (alpha_ocw >> 8) & 0xf;
        if (alpha_ab != 0) {
            regs[alpha_ab][3] = vars[0][3] * vars[1][3];
        }
        if (alpha_cd != 0) {
            regs[alpha_cd][3] = vars[2][3] * vars[3][3];
        }
        if (alpha_muxsum != 0) {
            regs[alpha_muxsum][3] = vars[0][3] * vars[1][3] +
                                    vars[2][3] * vars[3][3];
        }
    }

    for (civ = 0; civ < 3; civ++) {
        vars_final[4][civ] = nv_geforce3_rc_get_var(
            ch->d3d_combiner_final[1], 24, regs, civ);
        vars_final[5][civ] = nv_geforce3_rc_get_var(
            ch->d3d_combiner_final[1], 16, regs, civ);
    }
    for (ci = 0; ci < 3; ci++) {
        regs[0xe][ci] = regs[5][ci] + regs[0xc][ci];
        regs[0xf][ci] = vars_final[4][ci] * vars_final[5][ci];
    }
    for (civ = 0; civ < 3; civ++) {
        vars_final[0][civ] = nv_geforce3_rc_get_var(
            ch->d3d_combiner_final[0], 24, regs, civ);
        vars_final[1][civ] = nv_geforce3_rc_get_var(
            ch->d3d_combiner_final[0], 16, regs, civ);
        vars_final[2][civ] = nv_geforce3_rc_get_var(
            ch->d3d_combiner_final[0], 8, regs, civ);
        vars_final[3][civ] = nv_geforce3_rc_get_var(
            ch->d3d_combiner_final[0], 0, regs, civ);
    }
    out[3] = nv_geforce3_rc_get_var(ch->d3d_combiner_final[1], 8, regs, 2);
    for (civ = 0; civ < 3; civ++) {
        out[civ] = vars_final[0][civ] * vars_final[1][civ] +
            (1.0f - vars_final[0][civ]) * vars_final[2][civ] +
            vars_final[3][civ];
    }
}

static void nv_geforce3_d3d_pixel_quad_shader(NVGeForce3State *s,
                                              NVGeForce3Channel *ch,
                                              float in[4][16][4],
                                              bool discard[4],
                                              float tmp_regs16[4][64][4],
                                              float tmp_regs32[4][64][4])
{
    uint32_t ps_offset = ch->d3d_shader_offset;
    uint32_t cc[4][4] = { { 0 } };

    for (;;) {
        uint32_t dst_word = nv_geforce3_dma_read32(s, ch->d3d_shader_obj,
                                                    ps_offset);
        uint32_t src_words[3];
        float cnst[4] = { 0.0f };
        float paramsq[4][3][4];
        bool const_loaded = false;
        uint32_t cond;
        bool execute[4];
        float op_results[4][4];
        uint32_t op;
        int pi;
        uint32_t fi;

        ps_offset += 4;
        for (pi = 0; pi < 3; pi++) {
            src_words[pi] = nv_geforce3_dma_read32(s, ch->d3d_shader_obj,
                                                    ps_offset);
            ps_offset += 4;
        }
        for (pi = 0; pi < 3; pi++) {
            uint32_t reg_type = src_words[pi] & 3;
            uint32_t swizzle[4];
            bool negate = (src_words[pi] >> 17) & 1;
            bool src_abs = ((pi == 0 ? src_words[0] >> 29 :
                                       src_words[pi] >> 18)) & 1;
            int i, comp_index;

            if (reg_type == 2 && !const_loaded) {
                for (comp_index = 0; comp_index < 4; comp_index++) {
                    cnst[comp_index] = nv_geforce3_uint32_as_float(
                        nv_geforce3_dma_read32(s, ch->d3d_shader_obj,
                                               ps_offset));
                    ps_offset += 4;
                }
                const_loaded = true;
            }
            for (i = 0; i < 4; i++) {
                swizzle[i] = (src_words[pi] >> (9 + i * 2)) & 3;
            }
            for (fi = 0; fi < 4; fi++) {
                for (comp_index = 0; comp_index < 4; comp_index++) {
                    float *paramc = &paramsq[fi][pi][comp_index];
                    int comp_index_swizzle = swizzle[comp_index];

                    if (reg_type == 0) {
                        uint32_t tmp_index = (src_words[pi] >> 2) & 0x3f;
                        bool fp16 = (src_words[pi] >> 8) & 1;

                        *paramc = fp16 ?
                            tmp_regs16[fi][tmp_index][comp_index_swizzle] :
                            tmp_regs32[fi][tmp_index][comp_index_swizzle];
                    } else if (reg_type == 1) {
                        uint32_t in_index = (dst_word >> 13) & 0xf;

                        *paramc = in[fi][in_index][comp_index_swizzle];
                    } else if (reg_type == 2) {
                        *paramc = cnst[comp_index_swizzle];
                    } else {
                        *paramc = 0.0f;
                    }
                    if (src_abs) {
                        *paramc = fabsf(*paramc);
                    }
                    if (negate) {
                        *paramc = -*paramc;
                    }
                }
            }
        }

        cond = (src_words[0] >> 18) & 7;
        if (cond == 7) {
            for (fi = 0; fi < 4; fi++) {
                execute[fi] = true;
            }
        } else {
            for (fi = 0; fi < 4; fi++) {
                int i;

                execute[fi] = false;
                for (i = 0; i < 4; i++) {
                    uint32_t cond_swizzle = (src_words[0] >>
                                             (21 + i * 2)) & 3;

                    if (cc[fi][cond_swizzle] & cond) {
                        execute[fi] = true;
                        break;
                    }
                }
            }
        }

        op = (dst_word >> 24) & 0x3f;
        switch (op) {
        case 0x15: {
            float ddx[4];

            nv_geforce3_compute_partials_x(paramsq[0][0], paramsq[1][0],
                                           ddx);
            for (fi = 0; fi < 4; fi++) {
                uint32_t ci;

                for (ci = 0; ci < 4; ci++) {
                    op_results[fi][ci] = ddx[ci];
                }
            }
            break;
        }
        case 0x16: {
            float ddy[4];

            nv_geforce3_compute_partials_y(paramsq[0][0], paramsq[2][0],
                                           ddy);
            for (fi = 0; fi < 4; fi++) {
                uint32_t ci;

                for (ci = 0; ci < 4; ci++) {
                    op_results[fi][ci] = ddy[ci];
                }
            }
            break;
        }
        case 0x18:
            for (fi = 0; fi < 4; fi++) {
                float winv = 1.0f / paramsq[fi][0][3];

                paramsq[fi][0][0] *= winv;
                paramsq[fi][0][1] *= winv;
                paramsq[fi][0][2] *= winv;
            }
            /* fallthrough */
        case 0x2f:
        case 0x31:
        case 0x19:
        case 0x17: {
            uint32_t tex_unit = (dst_word >> 17) & 0xf;
            NVGeForce3Texture *tex = &ch->d3d_texture[tex_unit];
            float lodq[4];

            if (op == 0x2f) {
                for (fi = 0; fi < 4; fi++) {
                    lodq[fi] = paramsq[fi][1][0];
                }
            } else if (op == 0x19) {
                for (fi = 0; fi < 4; fi++) {
                    lodq[fi] = nv_geforce3_compute_lod(tex, paramsq[fi][0],
                        paramsq[fi][1], paramsq[fi][2]);
                }
            } else {
                float ddx[3], ddy[3], lambda_base;

                nv_geforce3_compute_partials(paramsq[0][0], paramsq[1][0],
                                             paramsq[2][0], ddx, ddy);
                lambda_base = nv_geforce3_compute_lod(tex, paramsq[0][0],
                                                      ddx, ddy);
                if (op == 0x31) {
                    for (fi = 0; fi < 4; fi++) {
                        lodq[fi] = lambda_base + paramsq[fi][1][0];
                    }
                } else {
                    for (fi = 0; fi < 4; fi++) {
                        lodq[fi] = lambda_base;
                    }
                }
            }
            for (fi = 0; fi < 4; fi++) {
                if (execute[fi]) {
                    nv_geforce3_d3d_sample_texture(s, ch, tex,
                        paramsq[fi][0], lodq[fi], op_results[fi]);
                    if ((dst_word >> 21) & 1) {
                        uint32_t ci;

                        for (ci = 0; ci < 4; ci++) {
                            op_results[fi][ci] =
                                op_results[fi][ci] * 2.0f - 1.0f;
                        }
                    }
                }
            }
            break;
        }
        case 0x34:
            for (fi = 0; fi < 4; fi++) {
                paramsq[fi][0][0] /= paramsq[fi][0][3];
                paramsq[fi][0][1] /= paramsq[fi][0][3];
            }
            /* fallthrough */
        case 0x33: {
            float coords[4][3];
            uint32_t tex_unit = (dst_word >> 17) & 0xf;
            NVGeForce3Texture *tex = &ch->d3d_texture[tex_unit];
            float ddx[3], ddy[3], lod;

            for (fi = 0; fi < 4; fi++) {
                coords[fi][0] = paramsq[fi][0][0] +
                    paramsq[fi][1][0] * paramsq[fi][2][0] +
                    paramsq[fi][1][1] * paramsq[fi][2][1];
                coords[fi][1] = paramsq[fi][0][1] +
                    paramsq[fi][1][0] * paramsq[fi][2][2] +
                    paramsq[fi][1][1] * paramsq[fi][2][3];
                coords[fi][2] = 0.0f;
            }
            nv_geforce3_compute_partials(coords[0], coords[1], coords[2],
                                         ddx, ddy);
            lod = nv_geforce3_compute_lod(tex, coords[0], ddx, ddy);
            for (fi = 0; fi < 4; fi++) {
                if (execute[fi]) {
                    nv_geforce3_d3d_sample_texture(s, ch, tex, coords[fi],
                                                   lod, op_results[fi]);
                    if ((dst_word >> 21) & 1) {
                        int ci;

                        for (ci = 0; ci < 4; ci++) {
                            op_results[fi][ci] =
                                op_results[fi][ci] * 2.0f - 1.0f;
                        }
                    }
                }
            }
            break;
        }
        default:
            for (fi = 0; fi < 4; fi++) {
                float (*params)[4] = paramsq[fi];
                float *op_result = op_results[fi];
                int comp_index;

                if (!execute[fi]) {
                    continue;
                }
                switch (op) {
                case 0:
                    break;
                case 1:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = params[0][comp_index];
                    }
                    break;
                case 2:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = params[0][comp_index] *
                                                params[1][comp_index];
                    }
                    break;
                case 3:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = params[0][comp_index] +
                                                params[1][comp_index];
                    }
                    break;
                case 4:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = params[0][comp_index] *
                            params[1][comp_index] + params[2][comp_index];
                    }
                    break;
                case 5: {
                    float dp3 = nv_geforce3_dot3(params[0], params[1]);

                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = dp3;
                    }
                    break;
                }
                case 6: {
                    float dp4 = nv_geforce3_dot4(params[0], params[1]);

                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = dp4;
                    }
                    break;
                }
                case 8:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = MIN(params[0][comp_index],
                                                    params[1][comp_index]);
                    }
                    break;
                case 9:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = MAX(params[0][comp_index],
                                                    params[1][comp_index]);
                    }
                    break;
                case 0xa:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] =
                            params[0][comp_index] < params[1][comp_index] ?
                            1.0f : 0.0f;
                    }
                    break;
                case 0xb:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] =
                            params[0][comp_index] >= params[1][comp_index] ?
                            1.0f : 0.0f;
                    }
                    break;
                case 0xc:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] =
                            params[0][comp_index] <= params[1][comp_index] ?
                            1.0f : 0.0f;
                    }
                    break;
                case 0xd:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] =
                            params[0][comp_index] > params[1][comp_index] ?
                            1.0f : 0.0f;
                    }
                    break;
                case 0xe:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] =
                            params[0][comp_index] != params[1][comp_index] ?
                            1.0f : 0.0f;
                    }
                    break;
                case 0xf:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] =
                            params[0][comp_index] == params[1][comp_index] ?
                            1.0f : 0.0f;
                    }
                    break;
                case 0x10:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = params[0][comp_index] -
                            floorf(params[0][comp_index]);
                    }
                    break;
                case 0x11:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] =
                            floorf(params[0][comp_index]);
                    }
                    break;
                case 0x12:
                    discard[fi] = true;
                    break;
                case 0x1a: {
                    float rcp = 1.0f / params[0][0];

                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = rcp;
                    }
                    break;
                }
                case 0x1c: {
                    float ex2 = exp2f(params[0][0]);

                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = ex2;
                    }
                    break;
                }
                case 0x1d: {
                    float lg2 = log2f(params[0][0]);

                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = lg2;
                    }
                    break;
                }
                case 0x1f:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] =
                            params[0][comp_index] * params[1][comp_index] +
                            (1.0f - params[0][comp_index]) *
                            params[2][comp_index];
                    }
                    break;
                case 0x22: {
                    float cosv = cosf(params[0][0]);

                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = cosv;
                    }
                    break;
                }
                case 0x23: {
                    float sinv = sinf(params[0][0]);

                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = sinv;
                    }
                    break;
                }
                case 0x26: {
                    float powv = powf(params[0][0], params[1][0]);

                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = powv;
                    }
                    break;
                }
                case 0x2e: {
                    float dp2a = params[0][0] * params[1][0] +
                                params[0][1] * params[1][1] + params[2][0];

                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = dp2a;
                    }
                    break;
                }
                case 0x36:
                    nv_geforce3_reflection(params[0], params[1], op_result);
                    op_result[3] = 0.0f;
                    break;
                case 0x38: {
                    float dp2 = params[0][0] * params[1][0] +
                               params[0][1] * params[1][1];

                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = dp2;
                    }
                    break;
                }
                case 0x39:
                    nv_geforce3_normalize(params[0], op_result);
                    op_result[3] = 0.0f;
                    break;
                case 0x3a:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = params[0][comp_index] /
                                                params[1][0];
                    }
                    break;
                default:
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        op_result[comp_index] = 0.5f;
                    }
                    qemu_log_mask(LOG_UNIMP, "nv-geforce3: pixel shader "
                                  "unknown opcode 0x%02x\n", op);
                    break;
                }
            }
            break;
        }

        for (fi = 0; fi < 4; fi++) {
            float *op_result = op_results[fi];
            bool set_cc, no_dst;

            if (!execute[fi]) {
                continue;
            }
            set_cc = (dst_word >> 8) & 1;
            if (set_cc) {
                int comp_index;

                for (comp_index = 0; comp_index < 4; comp_index++) {
                    if (op_result[comp_index] < 0.0f) {
                        cc[fi][comp_index] = 1;
                    } else if (op_result[comp_index] == 0.0f) {
                        cc[fi][comp_index] = 2;
                    } else {
                        cc[fi][comp_index] = 4;
                    }
                }
            }
            no_dst = (dst_word >> 30) & 1;
            if (op != 0 && !no_dst) {
                static const float dst_scales[] = {
                    1.0f, 2.0f, 4.0f, 8.0f, 1.0f, 0.5f, 0.25f, 0.125f
                };
                uint32_t mask = (dst_word >> 9) & 0xf;
                uint32_t dst_tmp_reg = (dst_word >> 1) & 0x3f;
                uint32_t dst_scale = (src_words[1] >> 28) & 7;
                bool dst_fp16 = (dst_word >> 7) & 1;
                bool saturate = (dst_word >> 31) & 1;
                int comp_index;

                for (comp_index = 0; comp_index < 4; comp_index++) {
                    if (mask & (1 << comp_index)) {
                        float value = op_result[comp_index] *
                                     dst_scales[dst_scale];

                        if (saturate) {
                            value = MIN(MAX(value, 0.0f), 1.0f);
                        }
                        if (dst_fp16) {
                            tmp_regs16[fi][dst_tmp_reg][comp_index] = value;
                        } else {
                            tmp_regs32[fi][dst_tmp_reg][comp_index] = value;
                        }
                    }
                }
            }
        }

        if (dst_word & 1) {
            break;
        }
    }
}

/* ---------------------------------------------------------------- */
/* Blend / compare helpers                                           */

static float nv_geforce3_blend_equation(uint16_t equation, float src,
                                        float src_factor, float dst,
                                        float dst_factor)
{
    switch (equation) {
    case 0x0001:
    case 0x8006:
    default:
        return src * src_factor + dst * dst_factor;
    case 0x0002:
    case 0x800a:
        return src * src_factor - dst * dst_factor;
    case 0x0003:
    case 0x800b:
        return dst * dst_factor - src * src_factor;
    case 0x0004:
    case 0x8007:
        return MIN(src, dst);
    case 0x0005:
    case 0x8008:
        return MAX(src, dst);
    }
}

static float nv_geforce3_blend_factor(uint16_t factor, float src_rgb,
                                      float src_a, float dst_rgb,
                                      float dst_a, float const_rgb,
                                      float const_a)
{
    switch (factor) {
    case 0x0000:
    case 0x1001:
        return 0.0f;
    case 0x0001:
    case 0x1002:
        return 1.0f;
    case 0x0300:
    case 0x1003:
        return src_rgb;
    case 0x0301:
    case 0x1004:
        return 1.0f - src_rgb;
    case 0x0302:
    case 0x1005:
        return src_a;
    case 0x0303:
    case 0x1006:
        return 1.0f - src_a;
    case 0x0304:
    case 0x1007:
        return dst_a;
    case 0x0305:
    case 0x1008:
        return 1.0f - dst_a;
    case 0x0306:
    case 0x1009:
        return dst_rgb;
    case 0x0307:
    case 0x100a:
        return 1.0f - dst_rgb;
    case 0x0308:
    case 0x100b:
        return MIN(src_a, 1.0f - dst_a);
    case 0x8001:
    case 0x100e:
        return const_rgb;
    case 0x8002:
    case 0x100f:
        return 1.0f - const_rgb;
    case 0x8003:
        return const_a;
    case 0x8004:
        return 1.0f - const_a;
    default:
        return 0.5f;
    }
}

static bool nv_geforce3_compare(uint32_t func, uint32_t val1, uint32_t val2)
{
    switch (func) {
    case 1:
    case 0x200:
        return false;
    case 2:
    case 0x201:
    default:
        return val1 < val2;
    case 3:
    case 0x202:
        return val1 == val2;
    case 4:
    case 0x203:
        return val1 <= val2;
    case 5:
    case 0x204:
        return val1 > val2;
    case 6:
    case 0x205:
        return val1 != val2;
    case 7:
    case 0x206:
        return val1 >= val2;
    case 8:
    case 0x207:
        return true;
    }
}

/* ---------------------------------------------------------------- */
/* Triangle setup / rasterisation                                    */

static void nv_geforce3_d3d_clip_to_screen(NVGeForce3Channel *ch,
                                           const float pos_clip[4],
                                           float pos_screen[4])
{
    pos_screen[3] = 1.0f / pos_clip[3];
    if ((ch->d3d_transform_execution_mode & 3) == 0) {
        int i;

        for (i = 0; i < 3; i++) {
            pos_screen[i] = pos_clip[i] * pos_screen[3];
            if (ch->d3d_view_matrix_enable & 1) {
                pos_screen[i] *= ch->d3d_model_view_matrix[1][i];
                pos_screen[i] += ch->d3d_model_view_matrix[1][i + 4];
            } else {
                pos_screen[i] *= ch->d3d_viewport_scale[i];
                pos_screen[i] += ch->d3d_viewport_offset[i];
            }
        }
        pos_screen[0] += ch->d3d_window_offset_x;
        pos_screen[1] += ch->d3d_window_offset_y;
    } else {
        int i;

        for (i = 0; i < 3; i++) {
            pos_screen[i] = pos_clip[i];
        }
    }
}

static void nv_geforce3_d3d_triangle_clipped(NVGeForce3State *s,
                                             NVGeForce3Channel *ch,
                                             float v0[16][4], float v1[16][4],
                                             float v2[16][4])
{
    float sp0[4], sp1[4], sp2[4];
    double b012, b012inv;
    bool front_face_cw, clockwise, front_face;
    uint32_t surf_x1, surf_y1, surf_x2, surf_y2;
    int32_t tri_x1, tri_y1, tri_x2, tri_y2;
    uint32_t draw_x1, draw_y1, draw_x2, draw_y2, draw_width, draw_height;
    uint32_t pitch, pitch_zeta, draw_offset_base, draw_offset_zeta;
    bool interpolate[16];
    float ps_in[4][16][4];
    float rc_regs[4][16][4];
    float fog_factor = 1.0f;
    bool stencil_test_enable, zstencil_enable, ps_enable, rc_enable;
    float ps_tmp_regs16[4][64][4];
    float ps_tmp_regs32[4][64][4];
    float (*ps_tmp_regs_exp)[64][4];
    uint16_t qy, qx;
    int a, i;
    uint32_t fi, ci;

    nv_geforce3_d3d_clip_to_screen(ch, v0[0], sp0);
    nv_geforce3_d3d_clip_to_screen(ch, v1[0], sp1);
    nv_geforce3_d3d_clip_to_screen(ch, v2[0], sp2);
    b012 = nv_geforce3_edge_function(sp0, sp1, sp2);
    front_face_cw = ch->d3d_front_face == 0x00000900;
    clockwise = b012 > 0.0;
    front_face = (clockwise != ch->d3d_triangle_flip) == front_face_cw;
    if (ch->d3d_cull_face_enable) {
        if ((ch->d3d_cull_face == 0x00000405 && !front_face) ||
            (ch->d3d_cull_face == 0x00000404 && front_face) ||
            ch->d3d_cull_face == 0x00000408) {
            return;
        }
    }

    surf_x1 = ch->d3d_clip_horizontal & 0xFFFF;
    surf_y1 = ch->d3d_clip_vertical & 0xFFFF;
    surf_x2 = surf_x1 + (ch->d3d_clip_horizontal >> 16);
    surf_y2 = surf_y1 + (ch->d3d_clip_vertical >> 16);
    tri_x1 = (int32_t)MIN(MIN(sp0[0], sp1[0]), sp2[0]);
    tri_y1 = (int32_t)MIN(MIN(sp0[1], sp1[1]), sp2[1]);
    tri_x2 = (int32_t)MAX(MAX(sp0[0], sp1[0]), sp2[0]);
    tri_y2 = (int32_t)MAX(MAX(sp0[1], sp1[1]), sp2[1]);
    draw_x1 = MIN(MAX(tri_x1, (int32_t)surf_x1), (int32_t)surf_x2);
    draw_y1 = MIN(MAX(tri_y1, (int32_t)surf_y1), (int32_t)surf_y2);
    draw_x2 = MIN(MAX(tri_x2 + 1, (int32_t)surf_x1), (int32_t)surf_x2);
    draw_y2 = MIN(MAX(tri_y2 + 1, (int32_t)surf_y1), (int32_t)surf_y2);
    if (draw_x2 < draw_x1 || draw_y2 < draw_y1) {
        return;
    }
    draw_width = draw_x2 - draw_x1;
    draw_height = draw_y2 - draw_y1;
    if (!nv_geforce3_d3d_window_clip(ch, &draw_x1, &draw_y1, &draw_width,
                                     &draw_height)) {
        return;
    }
    if (!nv_geforce3_d3d_viewport_clip(ch, &draw_x1, &draw_y1, &draw_width,
                                       &draw_height)) {
        return;
    }
    if (!nv_geforce3_d3d_scissor_clip(ch, &draw_x1, &draw_y1, &draw_width,
                                      &draw_height)) {
        return;
    }

    pitch = ch->d3d_surface_pitch_a & 0xFFFF;
    pitch_zeta = nv_geforce3_d3d_get_surface_pitch_z(ch);
    draw_offset_base = ch->d3d_surface_color_offset + draw_y1 * pitch +
                       draw_x1 * ch->d3d_color_bytes;
    draw_offset_zeta = ch->d3d_surface_zeta_offset + draw_y1 * pitch_zeta +
                       draw_x1 * ch->d3d_depth_bytes;

    for (a = 0; a < 16; a++) {
        bool result = false;

        for (ci = 0; ci < 4; ci++) {
            result |= v0[a][ci] != v1[a][ci];
            result |= v1[a][ci] != v2[a][ci];
        }
        interpolate[a] = result;
    }
    for (fi = 0; fi < 4; fi++) {
        ps_in[fi][3][1] = fog_factor;
        rc_regs[fi][3][3] = fog_factor;
        for (ci = 0; ci < 3; ci++) {
            rc_regs[fi][3][ci] = ch->d3d_fog_color[ci];
        }
    }
    for (i = 0; i < 2; i++) {
        if (!interpolate[ch->d3d_attrib_out_color[i]]) {
            for (fi = 0; fi < 4; fi++) {
                for (ci = 0; ci < 4; ci++) {
                    ps_in[fi][i + 1][ci] =
                        v0[ch->d3d_attrib_out_color[i]][ci];
                }
            }
        }
    }
    for (i = 0; i < (int)ch->d3d_tex_coord_count; i++) {
        if (!interpolate[ch->d3d_attrib_out_tex_coord[i]]) {
            for (fi = 0; fi < 4; fi++) {
                for (ci = 0; ci < 4; ci++) {
                    ps_in[fi][i + 4][ci] =
                        v0[ch->d3d_attrib_out_tex_coord[i]][ci];
                }
            }
        }
    }

    b012inv = 1.0 / b012;
    stencil_test_enable = ch->d3d_stencil_test_enable &&
                          ch->d3d_depth_bytes != 2;
    zstencil_enable = ch->d3d_depth_test_enable || stencil_test_enable;
    ps_enable = ch->d3d_shader_obj != 0;
    rc_enable = ch->d3d_combiner_control_num_stages != 0;
    ps_tmp_regs_exp = ps_tmp_regs16;
    if (ps_enable && (ch->d3d_shader_control & 0x00000040)) {
        ps_tmp_regs_exp = ps_tmp_regs32;
    }

    for (qy = 0; qy < draw_height; qy += 2) {
        for (qx = 0; qx < draw_width; qx += 2) {
            float xy[4][2];
            double b0[4], b1[4], b2[4];
            float z[4] = { 0.0f };
            bool discard[4];
            uint32_t z_new[4] = { 0 };
            uint8_t stencils[4] = { 0 };

            for (fi = 0; fi < 4; fi++) {
                uint32_t x = qx + (fi & 1);
                uint32_t y = qy + (fi >> 1);

                discard[fi] = false;
                xy[fi][0] = draw_x1 + x + 0.5f;
                xy[fi][1] = draw_y1 + y + 0.5f;
                if (x >= draw_width || y >= draw_height) {
                    discard[fi] = true;
                }
                b0[fi] = nv_geforce3_edge_function(sp1, sp2, xy[fi]);
                b1[fi] = nv_geforce3_edge_function(sp2, sp0, xy[fi]);
                b2[fi] = nv_geforce3_edge_function(sp0, sp1, xy[fi]);
                if (clockwise) {
                    if (b0[fi] < 0.0 || b1[fi] < 0.0 || b2[fi] < 0.0) {
                        discard[fi] = true;
                    }
                } else if (b0[fi] > 0.0 || b1[fi] > 0.0 || b2[fi] > 0.0) {
                    discard[fi] = true;
                }
                b0[fi] *= b012inv;
                b1[fi] *= b012inv;
                b2[fi] *= b012inv;
                z[fi] = sp0[2] * b0[fi] + sp1[2] * b1[fi] + sp2[2] * b2[fi];
                if (z[fi] > ch->d3d_clip_max) {
                    discard[fi] = true;
                }
            }
            if (discard[0] && discard[1] && discard[2] && discard[3]) {
                continue;
            }

            for (fi = 0; fi < 4; fi++) {
                uint8_t stencil = 0x00;
                uint32_t x = qx + (fi & 1);
                uint32_t y = qy + (fi >> 1);
                double winv;
                int ii;

                if (zstencil_enable && !discard[fi]) {
                    uint32_t z_prev;
                    bool depth_test_pass;

                    if (ch->d3d_depth_bytes == 2) {
                        z_prev = nv_geforce3_dma_read16(s, ch->d3d_zeta_obj,
                            draw_offset_zeta + y * pitch_zeta + x * 2);
                    } else {
                        uint32_t zstencil = nv_geforce3_dma_read32(s,
                            ch->d3d_zeta_obj,
                            draw_offset_zeta + y * pitch_zeta + x * 4);

                        z_prev = zstencil >> 8;
                        stencil = (uint8_t)zstencil;
                    }
                    if (ch->d3d_depth_test_enable) {
                        if (ch->d3d_depth_bytes == 2) {
                            z_new[fi] = (uint32_t)(z[fi] * 65535.0f);
                        } else {
                            z_new[fi] = (uint32_t)(z[fi] * 16777215.0f);
                        }
                        depth_test_pass = nv_geforce3_compare(
                            ch->d3d_depth_func, z_new[fi], z_prev);
                    } else {
                        depth_test_pass = true;
                    }
                    if (stencil_test_enable) {
                        bool stencil_test_pass = nv_geforce3_compare(
                            ch->d3d_stencil_func,
                            ch->d3d_stencil_func_ref & ch->d3d_stencil_func_mask,
                            stencil & ch->d3d_stencil_func_mask);
                        uint32_t stencil_op;

                        if (stencil_test_pass) {
                            stencil_op = depth_test_pass ?
                                ch->d3d_stencil_op_dppass :
                                ch->d3d_stencil_op_dpfail;
                        } else {
                            stencil_op = ch->d3d_stencil_op_sfail;
                        }
                        switch (stencil_op) {
                        case 0x1e00:
                        default:
                            break;
                        case 0x0000:
                            stencil = 0x00;
                            break;
                        case 0x1e01:
                            stencil = (uint8_t)ch->d3d_stencil_func_ref;
                            break;
                        case 0x1e02:
                            if (stencil < 0xff) {
                                stencil++;
                            }
                            break;
                        case 0x1e03:
                            if (stencil > 0x00) {
                                stencil--;
                            }
                            break;
                        case 0x150a:
                            stencil = ~stencil;
                            break;
                        case 0x8507:
                            stencil++;
                            break;
                        case 0x8508:
                            stencil--;
                            break;
                        }
                        if (stencil_op != 0x1e00) {
                            stencil &= ch->d3d_stencil_mask;
                            nv_geforce3_dma_write8(s, ch->d3d_zeta_obj,
                                draw_offset_zeta + y * pitch_zeta + x * 4,
                                stencil);
                        }
                        stencils[fi] = stencil;
                        if (!stencil_test_pass) {
                            discard[fi] = true;
                        }
                    }
                    if (!depth_test_pass) {
                        discard[fi] = true;
                    }
                }

                ps_in[fi][0][3] = sp0[3] * b0[fi] + sp1[3] * b1[fi] +
                                  sp2[3] * b2[fi];
                winv = 1.0 / ps_in[fi][0][3];
                b0[fi] *= sp0[3] * winv;
                b1[fi] *= sp1[3] * winv;
                b2[fi] *= sp2[3] * winv;
                for (ii = 0; ii < 2; ii++) {
                    if (interpolate[ch->d3d_attrib_out_color[ii]]) {
                        uint32_t attr = ch->d3d_attrib_out_color[ii];
                        int comp_index;

                        for (comp_index = 0; comp_index < 4; comp_index++) {
                            ps_in[fi][ii + 1][comp_index] =
                                v0[attr][comp_index] * b0[fi] +
                                v1[attr][comp_index] * b1[fi] +
                                v2[attr][comp_index] * b2[fi];
                        }
                    }
                }
                for (ii = 0; ii < (int)ch->d3d_tex_coord_count; ii++) {
                    uint32_t attr = ch->d3d_attrib_out_tex_coord[ii];

                    if (interpolate[attr]) {
                        int comp_index;

                        for (comp_index = 0; comp_index < 4; comp_index++) {
                            ps_in[fi][ii + 4][comp_index] =
                                v0[attr][comp_index] * b0[fi] +
                                v1[attr][comp_index] * b1[fi] +
                                v2[attr][comp_index] * b2[fi];
                        }
                    }
                }
                {
                    int comp_index;

                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        ps_tmp_regs16[fi][0][comp_index] =
                            ps_in[fi][1][comp_index];
                    }
                }
                if (ch->d3d_fog_enable) {
                    float fog_dist = v0[ch->d3d_attrib_out_fogc][0] * b0[fi] +
                        v1[ch->d3d_attrib_out_fogc][0] * b1[fi] +
                        v2[ch->d3d_attrib_out_fogc][0] * b2[fi];

                    switch (ch->d3d_fog_mode) {
                    case 0x2601:
                        fog_factor = ch->d3d_fog_params[1] * fog_dist +
                                    ch->d3d_fog_params[0] - 1.0f;
                        break;
                    case 0x804:
                        fog_factor = ch->d3d_fog_params[1] *
                            fabsf(fog_dist) + ch->d3d_fog_params[0] - 1.0f;
                        break;
                    case 0x800:
                        fog_factor = exp2f(16.0f * (ch->d3d_fog_params[1] *
                            fog_dist + ch->d3d_fog_params[0] - 1.5f));
                        break;
                    case 0x802:
                        fog_factor = exp2f(16.0f * (ch->d3d_fog_params[1] *
                            fabsf(fog_dist) + ch->d3d_fog_params[0] - 1.5f));
                        break;
                    case 0x801:
                        fog_factor = expf(-powf(4.709f *
                            (ch->d3d_fog_params[1] * fog_dist +
                             ch->d3d_fog_params[0] - 1.5f), 2.0f));
                        break;
                    case 0x803:
                        fog_factor = expf(-powf(4.709f *
                            (ch->d3d_fog_params[1] * fabsf(fog_dist) +
                             ch->d3d_fog_params[0] - 1.5f), 2.0f));
                        break;
                    default:
                        fog_factor = 0.5f;
                        break;
                    }
                    fog_factor = MIN(MAX(fog_factor, 0.0f), 1.0f);
                    if (ps_enable) {
                        ps_in[fi][3][1] = fog_factor;
                    }
                    if (rc_enable) {
                        rc_regs[fi][3][3] = fog_factor;
                    }
                }
            }

            if (ps_enable) {
                for (fi = 0; fi < 4; fi++) {
                    uint32_t ci2;

                    ps_in[fi][0][0] = xy[fi][0] - ch->d3d_window_offset_x;
                    ps_in[fi][0][1] = ch->d3d_viewport_height -
                        (xy[fi][1] - ch->d3d_window_offset_y);
                    ps_in[fi][0][2] = 0.0f;
                    ps_in[fi][15][0] = ps_in[fi][5][3];
                    ps_in[fi][15][1] = ps_in[fi][6][3];
                    ps_in[fi][15][2] = ps_in[fi][7][3];
                    for (ci2 = 0; ci2 < 4; ci2++) {
                        ps_tmp_regs32[fi][0][ci2] = 0.0f;
                    }
                }
                nv_geforce3_d3d_pixel_quad_shader(s, ch, ps_in, discard,
                    ps_tmp_regs16, ps_tmp_regs32);
                if (rc_enable) {
                    for (fi = 0; fi < 4; fi++) {
                        uint32_t ri;

                        for (ri = 0; ri < 8; ri++) {
                            for (ci = 0; ci < 4; ci++) {
                                rc_regs[fi][ri + 8][ci] =
                                    ps_tmp_regs16[fi][ri][ci];
                            }
                        }
                    }
                }
            }
            if (rc_enable) {
                for (fi = 0; fi < 4; fi++) {
                    for (ci = 0; ci < 4; ci++) {
                        rc_regs[fi][0][ci] = 0.0f;
                        rc_regs[fi][4][ci] = ps_in[fi][1][ci];
                        rc_regs[fi][5][ci] = ps_in[fi][2][ci];
                    }
                    rc_regs[fi][0xe][3] = 0.0f;
                    rc_regs[fi][0xf][3] = 0.0f;
                }
                if (!ps_enable) {
                    float uv[4][2] = { { 0.0f } };
                    uint32_t t;

                    for (t = 0; t < ch->d3d_tex_coord_count; t++) {
                        switch (ch->d3d_tex_shader_op[t]) {
                        case 0x00:
                            break;
                        case 0x01:
                        case 0x02:
                        case 0x03: {
                            NVGeForce3Texture *tex = &ch->d3d_texture[t];
                            float ddx[3], ddy[3], lod;

                            nv_geforce3_compute_partials(ps_in[0][4 + t],
                                ps_in[1][4 + t], ps_in[2][4 + t], ddx, ddy);
                            lod = nv_geforce3_compute_lod(tex, ps_in[0][4 + t],
                                                          ddx, ddy);
                            for (fi = 0; fi < 4; fi++) {
                                nv_geforce3_d3d_sample_texture(s, ch, tex,
                                    ps_in[fi][4 + t], lod, rc_regs[fi][8 + t]);
                            }
                            break;
                        }
                        case 0x06: {
                            float coords[4][3];
                            NVGeForce3Texture *tex = &ch->d3d_texture[t];
                            float ddx[3], ddy[3], lod;

                            for (fi = 0; fi < 4; fi++) {
                                float *in_coords = ps_in[fi][4 + t];
                                float *prev_color = rc_regs[fi][8 +
                                    ch->d3d_tex_shader_previous[t]];

                                coords[fi][0] = in_coords[0] / in_coords[3] +
                                    tex->offset_matrix[0] * prev_color[2] +
                                    tex->offset_matrix[3] * prev_color[1];
                                coords[fi][1] = in_coords[1] / in_coords[3] +
                                    tex->offset_matrix[1] * prev_color[2] +
                                    tex->offset_matrix[2] * prev_color[1];
                                coords[fi][2] = 0.0f;
                            }
                            nv_geforce3_compute_partials(coords[0], coords[1],
                                coords[2], ddx, ddy);
                            lod = nv_geforce3_compute_lod(tex, coords[0], ddx,
                                                          ddy);
                            for (fi = 0; fi < 4; fi++) {
                                nv_geforce3_d3d_sample_texture(s, ch, tex,
                                    coords[fi], lod, rc_regs[fi][8 + t]);
                            }
                            break;
                        }
                        case 0x0c: {
                            float rv[4][3];
                            NVGeForce3Texture *tex = &ch->d3d_texture[t];
                            float ddx[3], ddy[3], lod;

                            for (fi = 0; fi < 4; fi++) {
                                float *input_tex = rc_regs[fi][8 +
                                    ch->d3d_tex_shader_previous[t]];
                                float w = nv_geforce3_dot3_map(
                                    ps_in[fi][4 + t], input_tex,
                                    ch->d3d_tex_shader_dotmapping[t]);
                                float n[3] = { uv[fi][0], uv[fi][1], w };
                                float e[3] = {
                                    ps_in[fi][5][3], ps_in[fi][6][3],
                                    ps_in[fi][7][3]
                                };

                                nv_geforce3_reflection(n, e, rv[fi]);
                            }
                            nv_geforce3_compute_partials(rv[0], rv[1], rv[2],
                                                         ddx, ddy);
                            lod = nv_geforce3_compute_lod(tex, rv[0], ddx,
                                                          ddy);
                            for (fi = 0; fi < 4; fi++) {
                                nv_geforce3_d3d_sample_texture(s, ch, tex,
                                    rv[fi], lod, rc_regs[fi][8 + t]);
                            }
                            break;
                        }
                        case 0x11:
                            for (fi = 0; fi < 4; fi++) {
                                float *input_tex = rc_regs[fi][8 +
                                    ch->d3d_tex_shader_previous[t]];

                                uv[fi][t == 1 ? 0 : 1] =
                                    nv_geforce3_dot3_map(ps_in[fi][4 + t],
                                        input_tex,
                                        ch->d3d_tex_shader_dotmapping[t]);
                            }
                            break;
                        default:
                            for (fi = 0; fi < 4; fi++) {
                                float *color = rc_regs[fi][8 + t];

                                color[0] = 0.0f;
                                color[1] = 0.5f;
                                color[2] = 0.5f;
                                color[3] = 1.0f;
                            }
                            break;
                        }
                    }
                }
                for (fi = 0; fi < 4; fi++) {
                    nv_geforce3_d3d_register_combiners(ch, rc_regs[fi],
                        ps_tmp_regs_exp[fi][0]);
                }
            }

            for (fi = 0; fi < 4; fi++) {
                float a_val, r, g, b;
                uint32_t x, y, draw_offset;

                if (discard[fi]) {
                    continue;
                }
                a_val = MIN(MAX(ps_tmp_regs_exp[fi][0][3], 0.0f), 1.0f);
                if (ch->d3d_alpha_test_enable &&
                    !nv_geforce3_compare(ch->d3d_alpha_func,
                        (uint32_t)(a_val * 255.0f), ch->d3d_alpha_ref)) {
                    continue;
                }
                x = qx + (fi & 1);
                y = qy + (fi >> 1);
                draw_offset = ch->d3d_swizzled ?
                    ch->d3d_surface_color_offset +
                    nv_geforce3_swizzle(x + draw_x1, y + draw_y1, 0,
                        ch->swzs_width, ch->swzs_height, 1) *
                    ch->d3d_color_bytes :
                    draw_offset_base + y * pitch + x * ch->d3d_color_bytes;
                r = MIN(MAX(ps_tmp_regs_exp[fi][0][0], 0.0f), 1.0f);
                g = MIN(MAX(ps_tmp_regs_exp[fi][0][1], 0.0f), 1.0f);
                b = MIN(MAX(ps_tmp_regs_exp[fi][0][2], 0.0f), 1.0f);
                if (ch->d3d_blend_enable) {
                    float sr = r, sg = g, sb = b, sa = a_val;
                    float dr, dg, db, da;

                    if (ch->d3d_color_bytes == 2) {
                        uint16_t color = nv_geforce3_dma_read16(s,
                            ch->d3d_color_obj, draw_offset);

                        dr = ((color >> 11) & 0x1f) / 31.0f;
                        dg = ((color >> 5) & 0x3f) / 63.0f;
                        db = (color & 0x1f) / 31.0f;
                        da = 1.0f;
                    } else if (ch->d3d_color_bytes == 4) {
                        uint32_t color = nv_geforce3_dma_read32(s,
                            ch->d3d_color_obj, draw_offset);

                        dr = ((color >> 16) & 0xff) / 255.0f;
                        dg = ((color >> 8) & 0xff) / 255.0f;
                        db = (color & 0xff) / 255.0f;
                        da = ((color >> 24) & 0xff) / 255.0f;
                    } else {
                        uint8_t color = nv_geforce3_dma_read8(s,
                            ch->d3d_color_obj, draw_offset);

                        dr = 0.0f;
                        dg = 0.0f;
                        db = color / 255.0f;
                        da = 1.0f;
                    }
                    r = nv_geforce3_blend_equation(ch->d3d_blend_equation_rgb,
                        sr, nv_geforce3_blend_factor(ch->d3d_blend_sfactor_rgb,
                            sr, sa, dr, da, ch->d3d_blend_color[0],
                            ch->d3d_blend_color[3]),
                        dr, nv_geforce3_blend_factor(ch->d3d_blend_dfactor_rgb,
                            sr, sa, dr, da, ch->d3d_blend_color[0],
                            ch->d3d_blend_color[3]));
                    g = nv_geforce3_blend_equation(ch->d3d_blend_equation_rgb,
                        sg, nv_geforce3_blend_factor(ch->d3d_blend_sfactor_rgb,
                            sg, sa, dg, da, ch->d3d_blend_color[1],
                            ch->d3d_blend_color[3]),
                        dg, nv_geforce3_blend_factor(ch->d3d_blend_dfactor_rgb,
                            sg, sa, dg, da, ch->d3d_blend_color[1],
                            ch->d3d_blend_color[3]));
                    b = nv_geforce3_blend_equation(ch->d3d_blend_equation_rgb,
                        sb, nv_geforce3_blend_factor(ch->d3d_blend_sfactor_rgb,
                            sb, sa, db, da, ch->d3d_blend_color[2],
                            ch->d3d_blend_color[3]),
                        db, nv_geforce3_blend_factor(ch->d3d_blend_dfactor_rgb,
                            sb, sa, db, da, ch->d3d_blend_color[2],
                            ch->d3d_blend_color[3]));
                    a_val = nv_geforce3_blend_equation(
                        ch->d3d_blend_equation_alpha, sa,
                        nv_geforce3_blend_factor(ch->d3d_blend_sfactor_alpha,
                            sa, sa, da, da, ch->d3d_blend_color[3],
                            ch->d3d_blend_color[3]),
                        da, nv_geforce3_blend_factor(
                            ch->d3d_blend_dfactor_alpha, sa, sa, da, da,
                            ch->d3d_blend_color[3], ch->d3d_blend_color[3]));
                    r = MIN(MAX(r, 0.0f), 1.0f);
                    g = MIN(MAX(g, 0.0f), 1.0f);
                    b = MIN(MAX(b, 0.0f), 1.0f);
                    a_val = MIN(MAX(a_val, 0.0f), 1.0f);
                }
                if (ch->d3d_color_mask != 0) {
                    if (ch->d3d_color_bytes == 2) {
                        uint8_t r5 = (uint8_t)(r * 31.0f + 0.5f);
                        uint8_t g6 = (uint8_t)(g * 63.0f + 0.5f);
                        uint8_t b5 = (uint8_t)(b * 31.0f + 0.5f);
                        uint16_t color = b5 | (g6 << 5) | (r5 << 11);

                        if (ch->d3d_color_mask == 0x01010101) {
                            nv_geforce3_dma_write16(s, ch->d3d_color_obj,
                                draw_offset, color);
                        } else {
                            uint16_t dstcolor = nv_geforce3_dma_read16(s,
                                ch->d3d_color_obj, draw_offset);

                            dstcolor &= ~ch->d3d_color_mask_565;
                            dstcolor |= color & ch->d3d_color_mask_565;
                            nv_geforce3_dma_write16(s, ch->d3d_color_obj,
                                draw_offset, dstcolor);
                        }
                    } else if (ch->d3d_color_bytes == 4) {
                        uint8_t r8 = (uint8_t)(r * 255.0f + 0.5f);
                        uint8_t g8 = (uint8_t)(g * 255.0f + 0.5f);
                        uint8_t b8 = (uint8_t)(b * 255.0f + 0.5f);
                        uint8_t a8 = (uint8_t)(a_val * 255.0f + 0.5f);
                        uint32_t color = (uint32_t)b8 | ((uint32_t)g8 << 8) |
                            ((uint32_t)r8 << 16) | ((uint32_t)a8 << 24);

                        if (ch->d3d_color_mask == 0x01010101) {
                            nv_geforce3_dma_write32(s, ch->d3d_color_obj,
                                draw_offset, color);
                        } else {
                            uint32_t dstcolor = nv_geforce3_dma_read32(s,
                                ch->d3d_color_obj, draw_offset);

                            dstcolor &= ~ch->d3d_color_mask_8888;
                            dstcolor |= color & ch->d3d_color_mask_8888;
                            nv_geforce3_dma_write32(s, ch->d3d_color_obj,
                                draw_offset, dstcolor);
                        }
                    } else {
                        nv_geforce3_dma_write8(s, ch->d3d_color_obj,
                            draw_offset, (uint8_t)(b * 255.0f + 0.5f));
                    }
                }
                if (ch->d3d_depth_test_enable && ch->d3d_depth_write_enable) {
                    if (ch->d3d_depth_bytes == 2) {
                        nv_geforce3_dma_write16(s, ch->d3d_zeta_obj,
                            draw_offset_zeta + y * pitch_zeta + x * 2,
                            (uint16_t)z_new[fi]);
                    } else {
                        nv_geforce3_dma_write32(s, ch->d3d_zeta_obj,
                            draw_offset_zeta + y * pitch_zeta + x * 4,
                            (z_new[fi] << 8) | stencils[fi]);
                    }
                }
            }
        }
    }
    if (!ch->d3d_swizzled) {
        nv_geforce3_redraw_area(s, 0, draw_width, draw_height);
    }
}

static void nv_geforce3_d3d_triangle(NVGeForce3State *s,
                                     NVGeForce3Channel *ch, uint32_t base)
{
    float vs_out[3][16][4];
    bool clipped[3];
    uint32_t clip_count = 0;
    float clip_thresh = ch->d3d_viewport_offset[2] - ch->d3d_clip_min;
    uint32_t vi, v, i;
    int comp_index;

    if (ch->d3d_shade_mode == 0x00001d00) {
        float (*v_pr)[4] = ch->d3d_vertex_data[ch->d3d_vertex_index - 1];

        for (vi = 0; vi < 3; vi++) {
            float (*v_in)[4] = ch->d3d_vertex_data[(vi + base) & 3];
            uint32_t ci;

            for (ci = 0; ci < 4; ci++) {
                v_in[ch->d3d_attrib_in_color[0]][ci] =
                    v_pr[ch->d3d_attrib_in_color[0]][ci];
                v_in[ch->d3d_attrib_in_normal][ci] =
                    v_pr[ch->d3d_attrib_in_normal][ci];
            }
        }
    }

    for (vi = 0; vi < 3; vi++) {
        float (*v_out)[4] = vs_out[vi];
        float (*v_in)[4] = ch->d3d_vertex_data[(vi + base) & 3];

        if (ch->d3d_transform_execution_mode & 3) {
            nv_geforce3_d3d_vertex_shader(ch, v_in, v_out);
        } else {
            float *p = v_out[0];
            float *color_out[2] = {
                v_out[ch->d3d_attrib_out_color[0]],
                v_out[ch->d3d_attrib_out_color[1]]
            };
            float *color_in[2] = {
                v_in[ch->d3d_attrib_in_color[0]],
                v_in[ch->d3d_attrib_in_color[1]]
            };
            uint32_t ci;

            for (ci = 0; ci < 4; ci++) {
                p[ci] = v_in[0][ci];
            }
            if (ch->d3d_lighting_enable) {
                float nt[3], pt[3];
                float *n = v_in[ch->d3d_attrib_in_normal];
                uint32_t light_index;

                color_out[0][3] = ch->d3d_material_factor[3];
                for (ci = 0; ci < 3; ci++) {
                    switch (ch->d3d_color_material_ambient) {
                    case 0:
                    default:
                        color_out[0][ci] = ch->d3d_scene_ambient_color[ci];
                        break;
                    case 1:
                        color_out[0][ci] =
                            color_in[0][ci] * ch->d3d_material_factor[ci];
                        break;
                    case 2:
                        color_out[0][ci] =
                            color_in[1][ci] * ch->d3d_material_factor[ci];
                        break;
                    }
                    color_out[1][ci] = 0.0f;
                }
                nv_geforce3_normal_to_view(ch, n, nt);
                nv_geforce3_position_to_view3(ch, p, pt);
                for (light_index = 0; light_index < 8; light_index++) {
                    uint32_t light_type =
                        (ch->d3d_light_enable_mask >> (light_index * 2)) & 3;
                    NVGeForce3Light *light = &ch->d3d_light[light_index];
                    float n_dot_ld, n_dot_hv, att;

                    if (light_type == 0) {
                        continue;
                    }
                    if (light_type == 1) {
                        n_dot_ld = nv_geforce3_dot3(nt, light->inf_direction);
                        if (ch->d3d_local_viewer) {
                            float ed[3], hv[3];

                            for (ci = 0; ci < 3; ci++) {
                                ed[ci] = ch->d3d_eye_position[ci] - pt[ci];
                            }
                            nv_geforce3_normalize_ip(ed);
                            hv[0] = light->inf_direction[0] + ed[0];
                            hv[1] = light->inf_direction[1] + ed[1];
                            hv[2] = light->inf_direction[2] + ed[2];
                            nv_geforce3_normalize_ip(hv);
                            n_dot_hv = nv_geforce3_dot3(nt, hv);
                        } else {
                            n_dot_hv = nv_geforce3_dot3(nt,
                                light->inf_half_vector);
                        }
                        att = 1.0f;
                    } else {
                        float ld[3], hv[3], d;

                        for (ci = 0; ci < 3; ci++) {
                            ld[ci] = light->local_position[ci] - pt[ci];
                        }
                        d = nv_geforce3_normalize_ip(ld);
                        n_dot_ld = nv_geforce3_dot3(nt, ld);
                        if (ch->d3d_local_viewer) {
                            float ed[3];

                            for (ci = 0; ci < 3; ci++) {
                                ed[ci] = ch->d3d_eye_position[ci] - pt[ci];
                            }
                            nv_geforce3_normalize_ip(ed);
                            hv[0] = ld[0] + ed[0];
                            hv[1] = ld[1] + ed[1];
                            hv[2] = ld[2] + ed[2];
                        } else {
                            hv[0] = ld[0];
                            hv[1] = ld[1];
                            hv[2] = ld[2] + 1.0f;
                        }
                        nv_geforce3_normalize_ip(hv);
                        n_dot_hv = nv_geforce3_dot3(nt, hv);
                        att = 1.0f / (light->local_attenuation[0] +
                            light->local_attenuation[1] * d +
                            light->local_attenuation[2] * d * d);
                        if (light_type == 3) {
                            float rho = -nv_geforce3_dot3(
                                light->spot_direction, ld);

                            if (rho > light->spot_direction[3]) {
                                continue;
                            }
                        }
                    }
                    if (n_dot_ld < 0.0f) {
                        n_dot_ld = 0.0f;
                    }
                    for (ci = 0; ci < 3; ci++) {
                        float ambient = light->ambient_color[ci];
                        float diffuse = att * light->diffuse_color[ci] *
                                       n_dot_ld;

                        if (ch->d3d_color_material_ambient == 1) {
                            ambient *= color_in[0][ci];
                        } else if (ch->d3d_color_material_ambient == 2) {
                            ambient *= color_in[1][ci];
                        }
                        if (ch->d3d_color_material_diffuse == 1) {
                            diffuse *= color_in[0][ci];
                        } else if (ch->d3d_color_material_diffuse == 2) {
                            diffuse *= color_in[1][ci];
                        }
                        color_out[0][ci] += ambient + diffuse;
                    }
                    if (n_dot_hv < 0.0f) {
                        n_dot_hv = 0.0f;
                    }
                    if (n_dot_hv != 0.0f) {
                        float pf = powf(n_dot_hv, ch->d3d_specular_power);

                        for (ci = 0; ci < 3; ci++) {
                            color_out[ch->d3d_separate_specular][ci] +=
                                att * light->specular_color[ci] * pf;
                        }
                    }
                }
            } else {
                for (ci = 0; ci < 4; ci++) {
                    color_out[0][ci] = color_in[0][ci];
                    color_out[1][ci] = 0.0f;
                }
            }
            for (i = 0; i < ch->d3d_tex_coord_count; i++) {
                float *tc = v_out[ch->d3d_attrib_out_tex_coord[i]];
                int comp_index;

                for (comp_index = 0; comp_index < 4; comp_index++) {
                    uint32_t texgen = ch->d3d_texgen[i][comp_index];

                    switch (texgen) {
                    case 0x0000:
                        tc[comp_index] =
                            v_in[ch->d3d_attrib_in_tex_coord[i]][comp_index];
                        break;
                    case 0x2400: {
                        float pt4[4];

                        nv_geforce3_position_to_view4(ch, p, pt4);
                        tc[comp_index] = nv_geforce3_dot4(
                            ch->d3d_texgen_plane[i][comp_index], pt4);
                        break;
                    }
                    case 0x2401:
                        tc[comp_index] = nv_geforce3_dot4(
                            ch->d3d_texgen_plane[i][comp_index], p);
                        break;
                    case 0x2402:
                    case 0x8512: {
                        float nt[3], pt[3], u[3], r[3];
                        float *n = v_in[ch->d3d_attrib_in_normal];
                        float ntu;

                        nv_geforce3_normal_to_view(ch, n, nt);
                        nv_geforce3_position_to_view3(ch, p, pt);
                        nv_geforce3_normalize(pt, u);
                        ntu = nv_geforce3_dot3(nt, u);
                        r[0] = u[0] - 2 * nt[0] * ntu;
                        r[1] = u[1] - 2 * nt[1] * ntu;
                        r[2] = u[2] - 2 * nt[2] * ntu;
                        if (texgen == 0x2402) {
                            float m = 2 * sqrtf(r[0] * r[0] + r[1] * r[1] +
                                (r[2] + 1.0f) * (r[2] + 1.0f));

                            tc[comp_index] = comp_index < 2 ?
                                r[comp_index] / m + 0.5f : 0.0f;
                        } else {
                            tc[comp_index] = comp_index < 3 ?
                                r[comp_index] : 0.0f;
                        }
                        break;
                    }
                    case 0x8511:
                        if (comp_index < 3) {
                            float *n = v_in[ch->d3d_attrib_in_normal];
                            float *r =
                                &ch->d3d_inverse_model_view_matrix[
                                    comp_index * 4];

                            tc[comp_index] = nv_geforce3_dot3(n, r);
                        } else {
                            tc[comp_index] = 0.0f;
                        }
                        break;
                    default:
                        tc[comp_index] = 0.5f;
                        break;
                    }
                }
                if (ch->d3d_texture_matrix_enable[i]) {
                    float ttc[4];
                    float *m = ch->d3d_texture_matrix[i];

                    ttc[0] = tc[0] * m[0] + tc[1] * m[1] + tc[2] * m[2] +
                            tc[3] * m[3];
                    ttc[1] = tc[0] * m[4] + tc[1] * m[5] + tc[2] * m[6] +
                            tc[3] * m[7];
                    ttc[2] = tc[0] * m[8] + tc[1] * m[9] + tc[2] * m[10] +
                            tc[3] * m[11];
                    ttc[3] = tc[0] * m[12] + tc[1] * m[13] + tc[2] * m[14] +
                            tc[3] * m[15];
                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        tc[comp_index] = ttc[comp_index];
                    }
                }
            }
            if (ch->d3d_fog_enable) {
                float fog_dist;

                switch (ch->d3d_fog_gen_mode) {
                case 0:
                    fog_dist = v_in[ch->d3d_attrib_in_color[1]][3];
                    break;
                case 1: {
                    float pt[3];

                    nv_geforce3_position_to_view3(ch, p, pt);
                    fog_dist = nv_geforce3_length3(pt);
                    break;
                }
                case 2:
                case 3: {
                    float *m = ch->d3d_model_view_matrix[0];

                    fog_dist = p[0] * m[8] + p[1] * m[9] + p[2] * m[10] +
                              p[3] * m[11];
                    if (ch->d3d_fog_gen_mode == 3) {
                        fog_dist = fabsf(fog_dist);
                    }
                    break;
                }
                default:
                    fog_dist = 3.0f;
                    break;
                }
                v_out[ch->d3d_attrib_out_fogc][0] = fog_dist;
            }
            if (ch->d3d_view_matrix_enable == 0 ||
                ch->d3d_view_matrix_enable == 2 ||
                ch->d3d_view_matrix_enable == 6) {
                float tp[4];
                float *m = ch->d3d_composite_matrix;

                tp[0] = p[0] * m[0] + p[1] * m[1] + p[2] * m[2] +
                       p[3] * m[3];
                tp[1] = p[0] * m[4] + p[1] * m[5] + p[2] * m[6] +
                       p[3] * m[7];
                tp[2] = p[0] * m[8] + p[1] * m[9] + p[2] * m[10] +
                       p[3] * m[11];
                tp[3] = p[0] * m[12] + p[1] * m[13] + p[2] * m[14] +
                       p[3] * m[15];
                for (comp_index = 0; comp_index < 4; comp_index++) {
                    p[comp_index] = tp[comp_index];
                }
            }
        }
        for (i = 0; i < 2; i++) {
            if (ch->d3d_attrib_out_enable[i]) {
                float *color = v_out[ch->d3d_attrib_out_color[i]];
                uint32_t ci;

                for (ci = 0; ci < 4; ci++) {
                    color[ci] = MIN(MAX(color[ci], 0.0f), 1.0f);
                }
            }
        }
    }

    for (v = 0; v < 3; v++) {
        clipped[v] = vs_out[v][0][2] < -vs_out[v][0][3] * clip_thresh;
        if (clipped[v]) {
            clip_count++;
        }
    }
    if (clip_count == 0) {
        nv_geforce3_d3d_triangle_clipped(s, ch, vs_out[0], vs_out[1],
                                         vs_out[2]);
    } else if (clip_count != 3) {
        uint32_t intersection_index = 0;
        float intersections[2][16][4];
        int v0;

        for (v0 = 0; v0 < 3; v0++) {
            uint32_t v1 = (v0 + 1) % 3;

            if (clipped[v0] != clipped[v1]) {
                float k = vs_out[v1][0][2] + vs_out[v1][0][3] * clip_thresh;
                float t = k / (k - vs_out[v0][0][2] -
                              vs_out[v0][0][3] * clip_thresh);
                float omt = 1.0f - t;
                int a2;

                for (a2 = 0; a2 < 16; a2++) {
                    int comp_index;

                    for (comp_index = 0; comp_index < 4; comp_index++) {
                        intersections[intersection_index][a2][comp_index] =
                            t * vs_out[v0][a2][comp_index] +
                            omt * vs_out[v1][a2][comp_index];
                    }
                }
                intersection_index++;
            }
        }
        if (clip_count == 2) {
            if (!clipped[0]) {
                nv_geforce3_d3d_triangle_clipped(s, ch, vs_out[0],
                    intersections[0], intersections[1]);
            } else if (!clipped[1]) {
                nv_geforce3_d3d_triangle_clipped(s, ch, intersections[0],
                    vs_out[1], intersections[1]);
            } else {
                nv_geforce3_d3d_triangle_clipped(s, ch, intersections[1],
                    intersections[0], vs_out[2]);
            }
        } else {
            if (clipped[0]) {
                nv_geforce3_d3d_triangle_clipped(s, ch, intersections[0],
                    vs_out[1], vs_out[2]);
                nv_geforce3_d3d_triangle_clipped(s, ch, intersections[1],
                    intersections[0], vs_out[2]);
            } else if (clipped[1]) {
                nv_geforce3_d3d_triangle_clipped(s, ch, vs_out[0],
                    intersections[0], vs_out[2]);
                nv_geforce3_d3d_triangle_clipped(s, ch, intersections[0],
                    intersections[1], vs_out[2]);
            } else {
                nv_geforce3_d3d_triangle_clipped(s, ch, vs_out[0], vs_out[1],
                    intersections[0]);
                nv_geforce3_d3d_triangle_clipped(s, ch, vs_out[0],
                    intersections[0], intersections[1]);
            }
        }
    }
}

void nv_geforce3_d3d_process_vertex(NVGeForce3State *s, NVGeForce3Channel *ch,
                                    bool immediate)
{
    if (immediate) {
        uint32_t ai;

        for (ai = 0; ai < ch->d3d_attrib_count; ai++) {
            uint32_t ci;

            for (ci = 0; ci < 4; ci++) {
                ch->d3d_vertex_data[ch->d3d_vertex_index][ai][ci] =
                    ch->d3d_vertex_data_imm[ai][ci];
            }
        }
    }
    if (ch->d3d_vertex_data_array_format_homogeneous[0]) {
        float *p = ch->d3d_vertex_data[ch->d3d_vertex_index][0];

        p[3] = 1.0f / p[3];
        p[0] *= p[3];
        p[1] *= p[3];
        p[2] *= p[3];
    }
    ch->d3d_vertex_index++;
    switch (ch->d3d_begin_end) {
    case 5:
    case 0x1012:
    case 0x101a:
        if (ch->d3d_vertex_index == 3) {
            nv_geforce3_d3d_triangle(s, ch, 0);
            ch->d3d_vertex_index = 0;
        }
        break;
    case 6:
        if (ch->d3d_vertex_index == 3 || ch->d3d_primitive_done) {
            nv_geforce3_d3d_triangle(s, ch, 0);
            ch->d3d_primitive_done = true;
            ch->d3d_triangle_flip = !ch->d3d_triangle_flip;
            if (ch->d3d_vertex_index == 3) {
                ch->d3d_vertex_index = 0;
            }
        }
        break;
    case 7:
    case 0xa:
    case 0x1015:
    case 0x1017:
        if (ch->d3d_vertex_index == 3 || ch->d3d_primitive_done) {
            nv_geforce3_d3d_triangle(s, ch, 0);
            ch->d3d_primitive_done = true;
            ch->d3d_triangle_flip = !ch->d3d_triangle_flip;
            if (ch->d3d_vertex_index == 3) {
                ch->d3d_vertex_index = 1;
            }
        }
        break;
    case 8:
        if (ch->d3d_vertex_index == 4) {
            nv_geforce3_d3d_triangle(s, ch, 0);
            nv_geforce3_d3d_triangle(s, ch, 2);
            ch->d3d_vertex_index = 0;
        }
        break;
    case 9:
        if (ch->d3d_vertex_index == 4 ||
            (ch->d3d_vertex_index == 2 && ch->d3d_primitive_done)) {
            if (ch->d3d_vertex_index == 4) {
                nv_geforce3_d3d_triangle(s, ch, 0);
                ch->d3d_triangle_flip = true;
                nv_geforce3_d3d_triangle(s, ch, 1);
                ch->d3d_triangle_flip = false;
                ch->d3d_primitive_done = true;
                ch->d3d_vertex_index = 0;
            } else {
                nv_geforce3_d3d_triangle(s, ch, 2);
                ch->d3d_triangle_flip = true;
                nv_geforce3_d3d_triangle(s, ch, 3);
                ch->d3d_triangle_flip = false;
            }
        }
        break;
    default:
        ch->d3d_vertex_index = 0;
        break;
    }
}

void nv_geforce3_d3d_load_vertex(NVGeForce3State *s, NVGeForce3Channel *ch,
                                 uint32_t index)
{
    uint32_t index_adj = ch->d3d_vertex_data_base_index + index;
    uint32_t ai;

    for (ai = 0; ai < ch->d3d_attrib_count; ai++) {
        uint32_t comp_count = ch->d3d_vertex_data_array_format_size[ai];

        if (comp_count != 0) {
            uint32_t array_offset = ch->d3d_vertex_data_array_offset[ai];
            uint32_t array_obj = (array_offset & 0x80000000) ?
                ch->d3d_vertex_b_obj : ch->d3d_vertex_a_obj;
            uint32_t attrib_stride =
                ch->d3d_vertex_data_array_format_stride[ai];
            uint32_t format_type = ch->d3d_vertex_data_array_format_type[ai];

            array_offset &= 0x7fffffff;
            array_offset -= nv_geforce3_ramin_read32(s, array_obj) >> 20;
            array_offset += index_adj * attrib_stride;
            ch->d3d_vertex_data[ch->d3d_vertex_index][ai][2] = 0.0f;
            ch->d3d_vertex_data[ch->d3d_vertex_index][ai][3] = 1.0f;
            if ((format_type == 0 || format_type == 4) && comp_count == 4) {
                uint32_t value = nv_geforce3_dma_read32(s, array_obj,
                                                        array_offset);

                nv_geforce3_unpack_attribute(value, format_type == 0,
                    ch->d3d_vertex_data[ch->d3d_vertex_index][ai]);
            } else if (format_type == 5 && comp_count == 2) {
                uint32_t value = nv_geforce3_dma_read32(s, array_obj,
                                                        array_offset);

                ch->d3d_vertex_data[ch->d3d_vertex_index][ai][0] =
                    (float)(int16_t)value;
                ch->d3d_vertex_data[ch->d3d_vertex_index][ai][1] =
                    (float)(int16_t)(value >> 16);
            } else {
                uint32_t ci;

                for (ci = 0; ci < comp_count; ci++) {
                    uint32_t ui32 = nv_geforce3_dma_read32(s, array_obj,
                        array_offset + ci * 4);

                    ch->d3d_vertex_data[ch->d3d_vertex_index][ai][ci] =
                        nv_geforce3_uint32_as_float(ui32);
                }
            }
        } else {
            uint32_t ci;

            for (ci = 0; ci < 4; ci++) {
                ch->d3d_vertex_data[ch->d3d_vertex_index][ai][ci] =
                    ch->d3d_vertex_data_imm[ai][ci];
            }
        }
    }
    nv_geforce3_d3d_process_vertex(s, ch, false);
}

/* ---------------------------------------------------------------- */
/* Method handlers                                                   */

static void nv_geforce3_d3d_mh_object(NVGeForce3State *s,
                                      NVGeForce3Channel *ch, uint32_t cls,
                                      uint32_t method, uint32_t param)
{
    uint32_t j;

    if (cls == 0x0096) {
        ch->d3d_window_offset_x = 2048;
        ch->d3d_window_offset_y = 2048;
        ch->d3d_attrib_count = 8;
    } else {
        ch->d3d_window_offset_x = 0;
        ch->d3d_window_offset_y = 0;
        ch->d3d_attrib_count = 16;
    }
    for (j = 0; j < ch->d3d_attrib_count; j++) {
        ch->d3d_vertex_data_array_format_type[j] = 0;
        ch->d3d_vertex_data_array_format_size[j] = 0;
        ch->d3d_vertex_data_array_format_stride[j] = 0;
        ch->d3d_vertex_data_array_format_dx[j] = false;
        ch->d3d_vertex_data_array_format_homogeneous[j] = false;
    }
    if (cls == 0x0096) {
        ch->d3d_vs_temp_regs_count = 0;
    } else if (cls == 0x0097) {
        ch->d3d_vs_temp_regs_count = 12;
    } else {
        ch->d3d_vs_temp_regs_count = 16;
    }
    if (cls == 0x0096) {
        ch->d3d_combiner_control_num_stages = 2;
        ch->d3d_tex_coord_count = 2;
    } else if (cls == 0x0097) {
        ch->d3d_tex_coord_count = 4;
    } else {
        ch->d3d_tex_coord_count = 8;
    }
    if (cls == 0x0096) {
        ch->d3d_attrib_in_color[0] = 1;
        ch->d3d_attrib_in_color[1] = 2;
        ch->d3d_attrib_in_normal = 5;
    } else {
        ch->d3d_attrib_in_color[0] = 3;
        ch->d3d_attrib_in_color[1] = 4;
        ch->d3d_attrib_in_normal = 2;
    }
    ch->d3d_attrib_out_color[0] = 3;
    ch->d3d_attrib_out_color[1] = 4;
    ch->d3d_attrib_out_fogc = 5;
    for (j = 0; j < 32; j++) {
        ch->d3d_attrib_out_enable[j] = true;
    }
    for (j = 0; j < 16; j++) {
        ch->d3d_attrib_in_tex_coord[j] = 0xf;
        ch->d3d_attrib_out_tex_coord[j] = 0xf;
    }
    for (j = 0; j < ch->d3d_tex_coord_count; j++) {
        if (cls == 0x0096) {
            ch->d3d_attrib_in_tex_coord[j] = j + 3;
        } else if (cls == 0x0097) {
            ch->d3d_attrib_in_tex_coord[j] = j + 9;
        } else {
            ch->d3d_attrib_in_tex_coord[j] = j + 8;
        }
        if (cls <= 0x0097) {
            ch->d3d_attrib_out_tex_coord[j] = j + 9;
        } else {
            ch->d3d_attrib_out_tex_coord[j] = j + 8;
        }
    }
    for (j = 0; j < 4; j++) {
        ch->d3d_vertex_data_imm[ch->d3d_attrib_in_color[0]][j] = 1.0f;
    }
}

static void nv_geforce3_d3d_mh_flip_read(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    s->graph_flip_read = param;
}

static void nv_geforce3_d3d_mh_flip_write(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    s->graph_flip_write = param;
}

static void nv_geforce3_d3d_mh_flip_modulo(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    s->graph_flip_modulo = param;
}

static void nv_geforce3_d3d_mh_flip_incr(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    s->graph_flip_write++;
    if (s->graph_flip_modulo) {
        s->graph_flip_write %= s->graph_flip_modulo;
    }
}

static void nv_geforce3_d3d_mh_fifo_wait(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    if (s->graph_flip_read == s->graph_flip_write) {
        s->fifo_wait_flip = true;
        s->fifo_wait = true;
    }
}

static void nv_geforce3_d3d_mh_a_obj(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_a_obj = param;
}

static void nv_geforce3_d3d_mh_b_obj(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_b_obj = param;
}

static void nv_geforce3_d3d_mh_vertex_obj(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_vertex_a_obj = param;
    ch->d3d_vertex_b_obj = param;
}

static void nv_geforce3_d3d_mh_color_obj(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_color_obj = param;
}

static void nv_geforce3_d3d_mh_zeta_obj(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_zeta_obj = param;
}

static void nv_geforce3_d3d_mh_vertex_a_obj(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_vertex_a_obj = param;
}

static void nv_geforce3_d3d_mh_vertex_b_obj(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_vertex_b_obj = param;
}

static void nv_geforce3_d3d_mh_semaphore_obj(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_semaphore_obj = param;
}

static void nv_geforce3_d3d_mh_report_obj(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_report_obj = param;
}

static void nv_geforce3_d3d_mh_clip_horizontal(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_clip_horizontal = param;
}

static void nv_geforce3_d3d_mh_clip_vertical(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_clip_vertical = param;
}

static void nv_geforce3_d3d_mh_surface_format(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t format_color, format_depth;

    ch->d3d_surface_format = param;
    ch->d3d_swizzled = ((param >> 8) & 0xf) == 2;
    if (cls <= 0x0097) {
        format_color = param & 0x0000000F;
        format_depth = (param >> 4) & 0x0000000F;
    } else {
        format_color = param & 0x0000001F;
        format_depth = (param >> 5) & 0x00000007;
    }
    if (format_color == 0x9) {
        ch->d3d_color_bytes = 1;
    } else if (format_color == 0x3) {
        ch->d3d_color_bytes = 2;
    } else if (format_color == 0x4 || format_color == 0x5 ||
               format_color == 0x8) {
        ch->d3d_color_bytes = 4;
    } else {
        qemu_log_mask(LOG_UNIMP, "nv-geforce3: unknown D3D color format "
                      "0x%01x\n", format_color);
    }
    if (format_depth == 0) {
        ch->d3d_depth_bytes = ch->d3d_color_bytes;
    } else if (format_depth == 1) {
        ch->d3d_depth_bytes = 2;
    } else if (format_depth == 2) {
        ch->d3d_depth_bytes = 4;
    } else {
        qemu_log_mask(LOG_UNIMP, "nv-geforce3: unknown D3D depth format "
                      "0x%01x\n", format_depth);
    }
    if (cls == 0x0096) {
        ch->d3d_viewport_scale[2] = ch->d3d_depth_bytes == 2 ? 32767.0f :
                                                               8388607.0f;
    }
}

static void nv_geforce3_d3d_mh_surface_pitch_a(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_surface_pitch_a = param;
}

static void nv_geforce3_d3d_mh_surface_color_offset(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_surface_color_offset = param;
}

static void nv_geforce3_d3d_mh_surface_zeta_offset(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_surface_zeta_offset = param;
}

static void nv_geforce3_d3d_mh_combiner_alpha_icw(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_combiner_alpha_icw[method - 0x098] = param;
}

static void nv_geforce3_d3d_mh_combiner_final(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t i = method - (cls <= 0x0097 ? 0x0a2 : 0x23d);

    ch->d3d_combiner_final[i] = param;
}

static void nv_geforce3_d3d_mh_local_viewer(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_local_viewer = (param & 0x00010000) != 0;
}

static void nv_geforce3_d3d_mh_color_material(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    if (cls == 0x0096) {
        ch->d3d_color_material_emission = (param >> 0) & 1;
        ch->d3d_color_material_ambient = (param >> 1) & 1;
        ch->d3d_color_material_diffuse = (param >> 2) & 1;
        ch->d3d_color_material_specular = (param >> 3) & 1;
    } else {
        ch->d3d_color_material_emission = (param >> 0) & 3;
        ch->d3d_color_material_ambient = (param >> 2) & 3;
        ch->d3d_color_material_diffuse = (param >> 4) & 3;
        ch->d3d_color_material_specular = (param >> 6) & 3;
    }
}

static void nv_geforce3_d3d_mh_fog_mode(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_fog_mode = param;
}

static void nv_geforce3_d3d_mh_fog_gen_mode(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_fog_gen_mode = param;
}

static void nv_geforce3_d3d_mh_fog_params(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_fog_params[method & 3] = nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_fog_enable(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_fog_enable = param;
}

static void nv_geforce3_d3d_mh_fog_color(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t ci;

    for (ci = 0; ci < 4; ci++) {
        ch->d3d_fog_color[ci] = ((param >> (ci * 8)) & 0xff) / 255.0f;
    }
}

static void nv_geforce3_d3d_mh_window_offset(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_window_offset_x = (int16_t)param;
    ch->d3d_window_offset_y = (int16_t)(param >> 16);
}

static void nv_geforce3_d3d_mh_window_clip(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t index = (method >> 1) & 7;

    if ((method & 1) == 0) {
        ch->d3d_window_clip_x1[index] = param & 0x0000ffff;
        ch->d3d_window_clip_x2[index] = param >> 16;
    } else {
        ch->d3d_window_clip_y1[index] = param & 0x0000ffff;
        ch->d3d_window_clip_y2[index] = param >> 16;
    }
}

#define NV_GEFORCE3_SIMPLE_D3D_HANDLER(name, field) \
    static void nv_geforce3_d3d_mh_##name(NVGeForce3State *s, \
        NVGeForce3Channel *ch, uint32_t cls, uint32_t method, \
        uint32_t param) \
    { \
        ch->field = param; \
    }

NV_GEFORCE3_SIMPLE_D3D_HANDLER(alpha_test_enable, d3d_alpha_test_enable)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(alpha_func, d3d_alpha_func)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(alpha_ref, d3d_alpha_ref)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(blend_enable, d3d_blend_enable)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(cull_face_enable, d3d_cull_face_enable)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(depth_test_enable, d3d_depth_test_enable)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(lighting_enable, d3d_lighting_enable)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(stencil_test_enable, d3d_stencil_test_enable)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(depth_func, d3d_depth_func)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(depth_write_enable, d3d_depth_write_enable)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(stencil_mask, d3d_stencil_mask)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(stencil_func, d3d_stencil_func)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(stencil_func_ref, d3d_stencil_func_ref)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(stencil_func_mask, d3d_stencil_func_mask)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(stencil_op_sfail, d3d_stencil_op_sfail)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(stencil_op_dpfail, d3d_stencil_op_dpfail)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(stencil_op_dppass, d3d_stencil_op_dppass)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(shade_mode, d3d_shade_mode)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(cull_face, d3d_cull_face)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(front_face, d3d_front_face)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(normalize_enable, d3d_normalize_enable)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(light_enable_mask, d3d_light_enable_mask)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(view_matrix_enable, d3d_view_matrix_enable)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(shader_control, d3d_shader_control)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(semaphore_offset, d3d_semaphore_offset)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(zstencil_clear_value, d3d_zstencil_clear_value)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(color_clear_value, d3d_color_clear_value)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(transform_execution_mode,
                               d3d_transform_execution_mode)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(transform_program_load,
                               d3d_transform_program_load)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(transform_program_start,
                               d3d_transform_program_start)
NV_GEFORCE3_SIMPLE_D3D_HANDLER(transform_constant_load,
                               d3d_transform_constant_load)

static void nv_geforce3_d3d_mh_blend_sfactor_0096(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_blend_sfactor_rgb = (uint16_t)param;
    ch->d3d_blend_sfactor_alpha = (uint16_t)param;
}

static void nv_geforce3_d3d_mh_blend_dfactor_0096(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_blend_dfactor_rgb = (uint16_t)param;
    ch->d3d_blend_dfactor_alpha = (uint16_t)param;
}

static void nv_geforce3_d3d_mh_blend_equation_0096(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_blend_equation_rgb = (uint16_t)param;
    ch->d3d_blend_equation_alpha = (uint16_t)param;
}

static void nv_geforce3_d3d_mh_blend_sfactor_0497(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_blend_sfactor_rgb = (uint16_t)param;
    ch->d3d_blend_sfactor_alpha = param >> 16;
}

static void nv_geforce3_d3d_mh_blend_dfactor_0497(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_blend_dfactor_rgb = (uint16_t)param;
    ch->d3d_blend_dfactor_alpha = param >> 16;
}

static void nv_geforce3_d3d_mh_blend_equation_0497(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_blend_equation_rgb = (uint16_t)param;
    ch->d3d_blend_equation_alpha = param >> 16;
}

static void nv_geforce3_d3d_mh_blend_color(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_blend_color[0] = ((param >> 16) & 0xff) / 255.0f;
    ch->d3d_blend_color[1] = ((param >> 8) & 0xff) / 255.0f;
    ch->d3d_blend_color[2] = (param & 0xff) / 255.0f;
    ch->d3d_blend_color[3] = ((param >> 24) & 0xff) / 255.0f;
}

static void nv_geforce3_d3d_mh_color_mask(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_color_mask = param;
    ch->d3d_color_mask_565 = 0;
    ch->d3d_color_mask_8888 = 0;
    if ((param >> 0) & 1) {
        ch->d3d_color_mask_565 |= 0x001f;
        ch->d3d_color_mask_8888 |= 0x000000ff;
    }
    if ((param >> 8) & 1) {
        ch->d3d_color_mask_565 |= 0x07e0;
        ch->d3d_color_mask_8888 |= 0x0000ff00;
    }
    if ((param >> 16) & 1) {
        ch->d3d_color_mask_565 |= 0xf800;
        ch->d3d_color_mask_8888 |= 0x00ff0000;
    }
    if ((param >> 24) & 1) {
        ch->d3d_color_mask_8888 |= 0xff000000;
    }
}

static void nv_geforce3_d3d_mh_clip_min(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_clip_min = nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_clip_max(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_clip_max = nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_material_factor(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_material_factor[method - 0x0ea] =
        nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_separate_specular(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_separate_specular = param & 1;
}

static void nv_geforce3_d3d_mh_texgen(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t method_offset = method - (cls <= 0x0097 ? 0x0f0 : 0x100);

    ch->d3d_texgen[method_offset >> 2][method_offset & 3] = param;
}

static void nv_geforce3_d3d_mh_texture_matrix_enable(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t i = method - (cls == 0x0096 ? 0x0f8 :
                           (cls == 0x0097 ? 0x108 : 0x090));

    ch->d3d_texture_matrix_enable[i] = param;
}

static void nv_geforce3_d3d_mh_model_view_matrix(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t i = method & 0x00f;
    uint32_t m = (method >> 4) & 1;

    ch->d3d_model_view_matrix[m][i] = nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_inverse_model_view_matrix(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_inverse_model_view_matrix[method & 0x00f] =
        nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_composite_matrix(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_composite_matrix[method & 0x00f] =
        nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_texture_matrix(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t method_offset = method - (cls == 0x0096 ? 0x150 : 0x1b0);
    uint32_t tex_index = method_offset >> 4;
    uint32_t i = method_offset & 0x00f;

    ch->d3d_texture_matrix[tex_index][i] = nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_texgen_plane(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t method_offset = method - (cls == 0x0096 ? 0x180 :
                                       (cls == 0x0097 ? 0x210 : 0x380));
    uint32_t tex_index = method_offset >> 4;
    uint32_t tex_coord = (method_offset >> 2) & 3;
    uint32_t i = method_offset & 0x003;

    ch->d3d_texgen_plane[tex_index][tex_coord][i] =
        nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_scissor_x_width(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_scissor_x = param & 0x0000ffff;
    ch->d3d_scissor_width = param >> 16;
}

static void nv_geforce3_d3d_mh_scissor_y_height(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_scissor_y = param & 0x0000ffff;
    ch->d3d_scissor_height = param >> 16;
}

static void nv_geforce3_d3d_mh_shader_program(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t location = param & 3;

    ch->d3d_shader_program = param;
    ch->d3d_shader_offset = ch->d3d_shader_program & ~3u;
    if (location == 1) {
        ch->d3d_shader_obj = ch->d3d_a_obj;
    } else if (location == 2) {
        ch->d3d_shader_obj = ch->d3d_b_obj;
    } else {
        ch->d3d_shader_obj = 0;
    }
}

static void nv_geforce3_d3d_mh_0497_240(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t stage = (method >> 3) & 7;
    uint32_t rc_method = method & 7;

    if (rc_method == 0) {
        ch->d3d_combiner_alpha_icw[stage] = param;
    } else if (rc_method == 1) {
        ch->d3d_combiner_color_icw[stage] = param;
    } else if (rc_method == 2 || rc_method == 3) {
        uint32_t i = rc_method - 2;

        ch->d3d_combiner_const_color[stage][i][0] =
            ((param >> 16) & 0xff) / 255.0f;
        ch->d3d_combiner_const_color[stage][i][1] =
            ((param >> 8) & 0xff) / 255.0f;
        ch->d3d_combiner_const_color[stage][i][2] = (param & 0xff) / 255.0f;
        ch->d3d_combiner_const_color[stage][i][3] =
            ((param >> 24) & 0xff) / 255.0f;
    } else if (rc_method == 4) {
        ch->d3d_combiner_alpha_ocw[stage] = param;
    } else if (rc_method == 5) {
        ch->d3d_combiner_color_ocw[stage] = param;
    }
}

static void nv_geforce3_d3d_mh_viewport_x_width(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_viewport_x = param & 0x0000ffff;
    ch->d3d_viewport_width = param >> 16;
}

static void nv_geforce3_d3d_mh_viewport_y_height(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_viewport_y = param & 0x0000ffff;
    ch->d3d_viewport_height = param >> 16;
}

static void nv_geforce3_d3d_mh_specular_params(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t i = method & 7;

    ch->d3d_specular_params[i] = nv_geforce3_uint32_as_float(param);
    if (i == 5) {
        if (ch->d3d_specular_params[0] > -0.2f) {
            ch->d3d_specular_power = ch->d3d_specular_params[2];
        } else {
            ch->d3d_specular_power = 1.0f / (1.0f + ch->d3d_specular_params[0]);
            ch->d3d_specular_power = ch->d3d_specular_power *
                (2.7f + 0.25f * logf(ch->d3d_specular_power)) - 1.0f;
        }
    }
}

static void nv_geforce3_d3d_mh_scene_ambient_color(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t i = method - (cls == 0x0096 ? 0x1b1 : 0x284);

    ch->d3d_scene_ambient_color[i] = nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_viewport_offset(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t i = method - (cls == 0x0096 ? 0x1ba : 0x288);

    ch->d3d_viewport_offset[i] = nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_eye_position(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_eye_position[method - 0x294] = nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_0096_09c(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t i = method & 1;
    uint32_t s_idx;

    for (s_idx = 0; s_idx < 2; s_idx++) {
        ch->d3d_combiner_const_color[s_idx][i][0] =
            ((param >> 16) & 0xff) / 255.0f;
        ch->d3d_combiner_const_color[s_idx][i][1] =
            ((param >> 8) & 0xff) / 255.0f;
        ch->d3d_combiner_const_color[s_idx][i][2] = (param & 0xff) / 255.0f;
        ch->d3d_combiner_const_color[s_idx][i][3] =
            ((param >> 24) & 0xff) / 255.0f;
    }
}

static void nv_geforce3_d3d_mh_0097_298(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t method_offset = method - 0x298;
    uint32_t s_idx = method_offset & 7;
    uint32_t i = method_offset >> 3;

    ch->d3d_combiner_const_color[s_idx][i][0] =
        ((param >> 16) & 0xff) / 255.0f;
    ch->d3d_combiner_const_color[s_idx][i][1] =
        ((param >> 8) & 0xff) / 255.0f;
    ch->d3d_combiner_const_color[s_idx][i][2] = (param & 0xff) / 255.0f;
    ch->d3d_combiner_const_color[s_idx][i][3] =
        ((param >> 24) & 0xff) / 255.0f;
}

static void nv_geforce3_d3d_mh_combiner_alpha_ocw(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_combiner_alpha_ocw[method - (cls == 0x0096 ? 0x09e : 0x2a8)] =
        param;
}

static void nv_geforce3_d3d_mh_combiner_color_icw(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_combiner_color_icw[method - (cls == 0x0096 ? 0x09a : 0x2b0)] =
        param;
}

static void nv_geforce3_d3d_mh_texture_key_color(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t texture_index = method - (cls == 0x0097 ? 0x2b8 : 0x740);

    ch->d3d_texture[texture_index].key_color = param;
}

static void nv_geforce3_d3d_mh_viewport_scale(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_viewport_scale[method & 3] = nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_transform_program(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t i = method & 0x003;

    ch->d3d_transform_program[ch->d3d_transform_program_load][i] = param;
    if (i == 3) {
        ch->d3d_transform_program_load++;
    }
}

static void nv_geforce3_d3d_mh_transform_constant(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t i = method & 0x003;

    ch->d3d_transform_constant[ch->d3d_transform_constant_load][i] =
        nv_geforce3_uint32_as_float(param);
    if (i == 3) {
        ch->d3d_transform_constant_load++;
    }
}

static void nv_geforce3_d3d_mh_light(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t light_index, light_method;
    NVGeForce3Light *light;

    if (cls <= 0x0097) {
        light_index = (method >> 5) & 7;
        light_method = method & 0x01f;
    } else {
        light_index = (method >> 4) & 7;
        light_method = (method & 0x00f) | ((method & 0x080) >> 3);
    }
    light = &ch->d3d_light[light_index];
    if (light_method <= 0x02) {
        light->ambient_color[light_method] = nv_geforce3_uint32_as_float(param);
    } else if (light_method >= 0x03 && light_method <= 0x05) {
        light->diffuse_color[light_method - 0x03] =
            nv_geforce3_uint32_as_float(param);
    } else if (light_method >= 0x06 && light_method <= 0x08) {
        light->specular_color[light_method - 0x06] =
            nv_geforce3_uint32_as_float(param);
    } else if (light_method >= 0x0a && light_method <= 0x0c) {
        light->inf_half_vector[light_method - 0x0a] =
            nv_geforce3_uint32_as_float(param);
    } else if (light_method >= 0x0d && light_method <= 0x0f) {
        light->inf_direction[light_method - 0x0d] =
            nv_geforce3_uint32_as_float(param);
    } else if (light_method >= 0x13 && light_method <= 0x16) {
        light->spot_direction[light_method - 0x13] =
            nv_geforce3_uint32_as_float(param);
    } else if (light_method >= 0x17 && light_method <= 0x19) {
        light->local_position[light_method - 0x17] =
            nv_geforce3_uint32_as_float(param);
    } else if (light_method >= 0x1a && light_method <= 0x1c) {
        light->local_attenuation[light_method - 0x1a] =
            nv_geforce3_uint32_as_float(param);
    }
}

static void nv_geforce3_d3d_mh_0096_300(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t comp_index = method & 0x003;

    ch->d3d_vertex_data_imm[0][comp_index] = nv_geforce3_uint32_as_float(param);
    if (comp_index == 2) {
        ch->d3d_vertex_data_imm[0][3] = 1.0f;
        nv_geforce3_d3d_process_vertex(s, ch, true);
    }
}

static void nv_geforce3_d3d_mh_0497_540(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t comp_index = method & 0x003;
    uint32_t attrib_index;

    if (comp_index == 3) {
        return;
    }
    attrib_index = (method >> 2) & 0xf;
    ch->d3d_vertex_data_imm[attrib_index][comp_index] =
        nv_geforce3_uint32_as_float(param);
    if (comp_index == 2) {
        ch->d3d_vertex_data_imm[attrib_index][3] = 1.0f;
        if (attrib_index == 0) {
            nv_geforce3_d3d_process_vertex(s, ch, true);
        }
    }
}

static void nv_geforce3_d3d_mh_0096_306(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t i = method - (cls == 0x0096 ? 0x306 : 0x546);

    ch->d3d_vertex_data_imm[0][i] = nv_geforce3_uint32_as_float(param);
    if (i == 3) {
        nv_geforce3_d3d_process_vertex(s, ch, true);
    }
}

static void nv_geforce3_d3d_mh_0096_30c(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_vertex_data_imm[ch->d3d_attrib_in_normal][method & 0x003] =
        nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_0096_314(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_vertex_data_imm[ch->d3d_attrib_in_color[0]][method & 0x003] =
        nv_geforce3_uint32_as_float(param);
}

static void nv_geforce3_d3d_mh_0096_318(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t i = method & 0x003;

    ch->d3d_vertex_data_imm[ch->d3d_attrib_in_color[0]][i] =
        nv_geforce3_uint32_as_float(param);
    if (i == 2) {
        ch->d3d_vertex_data_imm[ch->d3d_attrib_in_color[0]][3] = 1.0f;
    }
}

static void nv_geforce3_d3d_mh_0096_31b(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    nv_geforce3_unpack_attribute(param, false,
        ch->d3d_vertex_data_imm[ch->d3d_attrib_in_color[0]]);
}

static void nv_geforce3_d3d_mh_texcoord(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t method_offset = method - (cls == 0x0096 ? 0x324 : 0x564);
    uint32_t texcoord_index = method_offset / 10;
    uint32_t texcoord_method = method_offset % 10;
    float *texcoord =
        ch->d3d_vertex_data_imm[ch->d3d_attrib_in_tex_coord[texcoord_index]];

    if (texcoord_method <= 1) {
        if (texcoord_method == 1) {
            texcoord[2] = 0.0f;
            texcoord[3] = 1.0f;
        }
        texcoord[texcoord_method] = nv_geforce3_uint32_as_float(param);
    } else if (texcoord_method == 2) {
        texcoord[0] = (int16_t)(param & 0xffff);
        texcoord[1] = (int16_t)(param >> 16);
        texcoord[2] = 0.0f;
        texcoord[3] = 1.0f;
    } else if (texcoord_method >= 4 && texcoord_method <= 7) {
        texcoord[texcoord_method - 4] = nv_geforce3_uint32_as_float(param);
    }
}

static void nv_geforce3_d3d_mh_0097_5c8(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_vertex_data_array_offset[
        method - (cls == 0x0097 ? 0x5c8 : 0x5a0)] = param;
}

static void nv_geforce3_d3d_mh_vertex_data_array_format(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t i;

    if (cls == 0x0096) {
        uint32_t method_offset = method - 0x340;

        i = method_offset >> 1;
        if ((method_offset & 1) == 0) {
            ch->d3d_vertex_data_array_offset[i] = param;
            return;
        }
    } else {
        i = method - (cls == 0x0097 ? 0x5d8 : 0x5d0);
    }
    ch->d3d_vertex_data_array_format_stride[i] = (param >> 8) & 0xff;
    ch->d3d_vertex_data_array_format_dx[i] = (param & 0x00010000) != 0;
    ch->d3d_vertex_data_array_format_homogeneous[i] =
        (param & 0x01000000) != 0;
    if (!ch->d3d_vertex_data_array_format_dx[i]) {
        ch->d3d_vertex_data_array_format_type[i] = param & 0xf;
        ch->d3d_vertex_data_array_format_size[i] = (param >> 4) & 0xf;
    } else {
        uint32_t dxtype = param & 0xff;

        if (dxtype == 0x44) {
            ch->d3d_vertex_data_array_format_type[i] = 4;
            ch->d3d_vertex_data_array_format_size[i] = 4;
        } else if (dxtype == 0x88) {
            ch->d3d_vertex_data_array_format_type[i] = 2;
            ch->d3d_vertex_data_array_format_size[i] = 1;
        } else if (dxtype == 0x99) {
            ch->d3d_vertex_data_array_format_type[i] = 2;
            ch->d3d_vertex_data_array_format_size[i] = 2;
        } else if (dxtype == 0xaa) {
            ch->d3d_vertex_data_array_format_type[i] = 2;
            ch->d3d_vertex_data_array_format_size[i] = 3;
        } else if (dxtype == 0xbb) {
            ch->d3d_vertex_data_array_format_type[i] = 2;
            ch->d3d_vertex_data_array_format_size[i] = 4;
        } else if (dxtype == 0xcc) {
            ch->d3d_vertex_data_array_format_type[i] = 0;
            ch->d3d_vertex_data_array_format_size[i] = 4;
        } else if (dxtype == 0xee) {
            ch->d3d_vertex_data_array_format_type[i] = 5;
            ch->d3d_vertex_data_array_format_size[i] = 2;
        }
    }
}

static void nv_geforce3_d3d_mh_get_report(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t offset = param & 0x00ffffff;

    nv_geforce3_dma_write64(s, ch->d3d_report_obj, offset,
                            nv_geforce3_get_current_time(s));
    nv_geforce3_dma_write32(s, ch->d3d_report_obj, offset + 0x8, 0);
    nv_geforce3_dma_write32(s, ch->d3d_report_obj, offset + 0xC, 0);
}

static void nv_geforce3_d3d_mh_begin_end(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    if (param != 0) {
        ch->d3d_primitive_done = false;
        ch->d3d_triangle_flip = false;
        ch->d3d_vertex_index = 0;
        ch->d3d_attrib_index = cls == 0x0096 ? 7 : 0;
        ch->d3d_comp_index = 0;
    }
    ch->d3d_begin_end = param;
}

static void nv_geforce3_d3d_mh_array_element16(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    nv_geforce3_d3d_load_vertex(s, ch, param & 0x0000ffff);
    nv_geforce3_d3d_load_vertex(s, ch, param >> 16);
}

static void nv_geforce3_d3d_mh_array_element32(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    nv_geforce3_d3d_load_vertex(s, ch, param);
}

static void nv_geforce3_d3d_mh_draw_arrays(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t vertex_first = param & 0x00ffffff;
    uint32_t vertex_last = vertex_first + (param >> 24);
    uint32_t v;

    for (v = vertex_first; v <= vertex_last; v++) {
        nv_geforce3_d3d_load_vertex(s, ch, v);
    }
}

static void nv_geforce3_d3d_mh_inline_array(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t format_type;
    bool process = false;

    if (cls == 0x0096) {
        while (ch->d3d_vertex_data_array_format_size[ch->d3d_attrib_index]
               == 0) {
            uint32_t ci;

            for (ci = 0; ci < 4; ci++) {
                ch->d3d_vertex_data[ch->d3d_vertex_index][
                    ch->d3d_attrib_index][ci] =
                    ch->d3d_vertex_data_imm[ch->d3d_attrib_index][ci];
            }
            ch->d3d_attrib_index--;
        }
    }
    if (ch->d3d_comp_index == 0) {
        ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index][2] =
            0.0f;
        ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index][3] =
            1.0f;
    }
    format_type = ch->d3d_vertex_data_array_format_type[ch->d3d_attrib_index];
    if ((format_type == 0 || format_type == 4) &&
        ch->d3d_vertex_data_array_format_size[ch->d3d_attrib_index] == 4) {
        nv_geforce3_unpack_attribute(param, format_type == 0,
            ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index]);
        ch->d3d_comp_index = 4;
    } else if (format_type == 5 &&
        ch->d3d_vertex_data_array_format_size[ch->d3d_attrib_index] == 2) {
        ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index][0] =
            (float)(int16_t)param;
        ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index][1] =
            (float)(int16_t)(param >> 16);
        ch->d3d_comp_index = 2;
    } else {
        ch->d3d_vertex_data[ch->d3d_vertex_index][ch->d3d_attrib_index][
            ch->d3d_comp_index++] = nv_geforce3_uint32_as_float(param);
    }
    while (ch->d3d_comp_index ==
           ch->d3d_vertex_data_array_format_size[ch->d3d_attrib_index]) {
        if (ch->d3d_comp_index == 0) {
            uint32_t ci;

            for (ci = 0; ci < 4; ci++) {
                ch->d3d_vertex_data[ch->d3d_vertex_index][
                    ch->d3d_attrib_index][ci] =
                    ch->d3d_vertex_data_imm[ch->d3d_attrib_index][ci];
            }
        } else {
            ch->d3d_comp_index = 0;
        }
        if (cls == 0x0096) {
            if (ch->d3d_attrib_index == 0) {
                ch->d3d_attrib_index = 7;
                process = true;
                break;
            }
            ch->d3d_attrib_index--;
        } else {
            if (ch->d3d_attrib_index == 15) {
                ch->d3d_attrib_index = 0;
                process = true;
                break;
            }
            ch->d3d_attrib_index++;
        }
    }
    if (process) {
        nv_geforce3_d3d_process_vertex(s, ch, false);
    }
}

static void nv_geforce3_d3d_mh_index_array_offset(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_index_array_offset = param;
}

static void nv_geforce3_d3d_mh_index_array_dma(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_index_array_dma = (param & 1) != 0;
    ch->d3d_index_array_type_16 = ((param >> 4) & 1) != 0;
}

static void nv_geforce3_d3d_mh_0497_609(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t vertex_first = param & 0x00ffffff;
    uint32_t vertex_last = vertex_first + (param >> 24);
    uint32_t index_array_obj = ch->d3d_index_array_dma ? ch->d3d_vertex_b_obj :
                                                         ch->d3d_vertex_a_obj;
    uint32_t v;

    for (v = vertex_first; v <= vertex_last; v++) {
        uint32_t vertex_array_index;

        if (ch->d3d_index_array_type_16) {
            vertex_array_index = nv_geforce3_dma_read16(s, index_array_obj,
                ch->d3d_index_array_offset + v * 2);
        } else {
            vertex_array_index = nv_geforce3_dma_read32(s, index_array_obj,
                ch->d3d_index_array_offset + v * 4);
        }
        nv_geforce3_d3d_load_vertex(s, ch, vertex_array_index);
    }
}

static void nv_geforce3_d3d_mh_0097_60a(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    if (ch->d3d_vertex_index != 2) {
        uint32_t ai;

        for (ai = 0; ai < ch->d3d_attrib_count; ai++) {
            uint32_t ci;

            for (ci = 0; ci < 4; ci++) {
                ch->d3d_vertex_data[ch->d3d_vertex_index][ai][ci] =
                    ch->d3d_vertex_data[2 - (param & 1)][ai][ci];
            }
        }
    }
    nv_geforce3_d3d_process_vertex(s, ch, false);
}

static void nv_geforce3_d3d_mh_0497_610(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t texture_index = method & 0x00f;
    NVGeForce3Texture *tex = &ch->d3d_texture[texture_index];

    if (cls == 0x0497) {
        tex->pal_dma_obj = (param & 1) == 1 ? ch->d3d_b_obj : ch->d3d_a_obj;
        tex->pal_ofs = param & 0xffffffc0;
    } else {
        tex->pitch = param & 0x000fffff;
        tex->size_npot[2] = param >> 20;
        nv_geforce3_texture_update_size(tex, cls);
    }
}

static void nv_geforce3_d3d_mh_0097_620(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t comp_index = method & 1;
    uint32_t attrib_index = (method >> 1) & 0xf;

    ch->d3d_vertex_data_imm[attrib_index][comp_index] =
        nv_geforce3_uint32_as_float(param);
    if (comp_index == 1) {
        ch->d3d_vertex_data_imm[attrib_index][2] = 0.0f;
        ch->d3d_vertex_data_imm[attrib_index][3] = 1.0f;
        if (attrib_index == 0) {
            nv_geforce3_d3d_process_vertex(s, ch, true);
        }
    }
}

static void nv_geforce3_d3d_mh_0097_640(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t attrib_index = method & 0xf;

    ch->d3d_vertex_data_imm[attrib_index][0] = (int16_t)(param & 0xffff);
    ch->d3d_vertex_data_imm[attrib_index][1] = (int16_t)(param >> 16);
    ch->d3d_vertex_data_imm[attrib_index][2] = 0.0f;
    ch->d3d_vertex_data_imm[attrib_index][3] = 1.0f;
    if (attrib_index == 0) {
        nv_geforce3_d3d_process_vertex(s, ch, true);
    }
}

static void nv_geforce3_d3d_mh_0097_650(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t attrib_index = method & 0xf;

    nv_geforce3_unpack_attribute(param, false,
                                 ch->d3d_vertex_data_imm[attrib_index]);
    if (attrib_index == 0) {
        nv_geforce3_d3d_process_vertex(s, ch, true);
    }
}

static void nv_geforce3_d3d_mh_0097_680(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t comp_index = method & 3;
    uint32_t attrib_index = (method >> 2) & 0xf;

    ch->d3d_vertex_data_imm[attrib_index][comp_index] =
        nv_geforce3_uint32_as_float(param);
    if (comp_index == 3 && attrib_index == 0) {
        nv_geforce3_d3d_process_vertex(s, ch, true);
    }
}

static void nv_geforce3_d3d_mh_texture(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    uint32_t method_offset = method -
        (cls == 0x0096 ? 0x086 : (cls == 0x0097 ? 0x6c0 : 0x680));
    uint32_t texture_index, texture_method;
    NVGeForce3Texture *tex;

    if (cls == 0x0096) {
        texture_index = method_offset & 1;
        texture_method = method_offset >> 1;
    } else {
        texture_index = method_offset >> (cls == 0x0097 ? 4 : 3);
        texture_method = method_offset & (cls == 0x0097 ? 0xf : 7);
    }
    tex = &ch->d3d_texture[texture_index];
    if (texture_method == 0) {
        tex->offset = param;
    } else if (texture_method == 1) {
        tex->dma_obj = (param & 3) == 1 ? ch->d3d_a_obj : ch->d3d_b_obj;
        tex->cubemap = (param & 4) != 0;
        if (cls == 0x0096) {
            tex->dimensions = 2;
            tex->format = (param >> 7) & 0x1f;
            tex->levels = (param >> 12) & 0xf;
            tex->size_log[0] = (param >> 16) & 0xf;
            tex->size_log[1] = (param >> 20) & 0xf;
            tex->size_log[2] = 0;
            tex->wrap[0] = (param >> 24) & 0xf;
            tex->wrap[1] = (param >> 28) & 0xf;
            tex->wrap[2] = 1;
        } else {
            tex->dimensions = (param >> 4) & 0xf;
            if (tex->dimensions < 1 || tex->dimensions > 3) {
                qemu_log_mask(LOG_GUEST_ERROR,
                              "nv-geforce3: texture with %u dimensions\n",
                              tex->dimensions);
            }
            tex->format = (param >> 8) & 0xff;
            tex->levels = (param >> 16) & 0xf;
            tex->size_log[0] = (param >> 20) & 0xf;
            tex->size_log[1] = (param >> 24) & 0xf;
            tex->size_log[2] = (param >> 28) & 0xf;
        }
        nv_geforce3_d3d_texture_process_format(tex);
        nv_geforce3_texture_update_size(tex, cls);
    } else if (texture_method == 2 && cls != 0x0096) {
        tex->wrap[0] = param & 0xf;
        tex->wrap[1] = (param >> 8) & 0xf;
        tex->wrap[2] = (param >> 16) & 0xf;
    } else if ((texture_method == 2 && cls == 0x0096) ||
               (texture_method == 3 && cls != 0x0096)) {
        tex->control0 = param;
        tex->enabled = (param >> 30) & 1;
        if (cls == 0x0096) {
            ch->d3d_tex_shader_op[texture_index] = tex->enabled ? 0x01 : 0x00;
        }
    } else if ((texture_method == 3 && cls == 0x0096) ||
               (texture_method == 4 && cls != 0x0096)) {
        int i;

        for (i = 0; i < 4; i++) {
            tex->s0[i] = (param >> (8 + i * 2)) & 3;
            tex->s1[i] = (param >> (i * 2)) & 3;
        }
        if (cls <= 0x0497) {
            tex->pitch = param >> 16;
            nv_geforce3_texture_update_size(tex, cls);
        }
    } else if ((texture_method == 6 && cls == 0x0096) ||
               (texture_method == 5 && cls != 0x0096)) {
        tex->filter_min = (param >> 16) & 7;
        tex->filter_mag = (param >> 24) & 7;
        if (cls != 0x0096) {
            uint32_t signed_argb = param >> 28;
            int i;

            tex->signed_any = signed_argb != 0;
            for (i = 0; i < 4; i++) {
                tex->signed_comp[i] = (signed_argb & (1 << i)) != 0;
            }
        } else {
            tex->signed_any = false;
        }
    } else if ((texture_method == 5 && cls == 0x0096) ||
               (texture_method == 7 && cls == 0x0097) ||
               (texture_method == 6 && cls >= 0x0497)) {
        tex->size_npot[0] = param >> 16;
        tex->size_npot[1] = param & 0x0000ffff;
        nv_geforce3_texture_update_size(tex, cls);
    } else if ((texture_method == 7 && cls == 0x0096) ||
               (texture_method == 8 && cls == 0x0097)) {
        tex->pal_dma_obj = (param & 1) == 1 ? ch->d3d_b_obj : ch->d3d_a_obj;
        tex->pal_ofs = param & 0xffffffc0;
    } else if ((texture_method == 9 && cls == 0x0097) ||
               (texture_method == 7 && cls >= 0x0497)) {
        tex->border_color[0] = ((param >> 16) & 0xff) / 255.0f;
        tex->border_color[1] = ((param >> 8) & 0xff) / 255.0f;
        tex->border_color[2] = (param & 0xff) / 255.0f;
        tex->border_color[3] = ((param >> 24) & 0xff) / 255.0f;
    } else if (texture_method >= 10 && texture_method <= 13 &&
               cls == 0x0097) {
        tex->offset_matrix[texture_method - 10] =
            nv_geforce3_uint32_as_float(param);
    }
}

static void nv_geforce3_d3d_mh_clear_surface(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_clear_surface = param;
    nv_geforce3_d3d_clear_surface(s, ch);
}

static void nv_geforce3_d3d_mh_combiner_color_ocw(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_combiner_color_ocw[method - (cls == 0x0096 ? 0x0a0 : 0x790)] =
        param;
}

static void nv_geforce3_d3d_mh_combiner_control(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_combiner_control = param;
    ch->d3d_combiner_control_num_stages = param & 0xf;
}

static void nv_geforce3_d3d_mh_tex_shader_op(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    int i;

    for (i = 0; i < 4; i++) {
        ch->d3d_tex_shader_op[i] = (param >> (i * 5)) & 0x1f;
    }
}

static void nv_geforce3_d3d_mh_tex_shader_dotmapping(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_tex_shader_dotmapping[1] = param & 0xf;
    ch->d3d_tex_shader_dotmapping[2] = (param >> 4) & 0xf;
    ch->d3d_tex_shader_dotmapping[3] = (param >> 8) & 0xf;
}

static void nv_geforce3_d3d_mh_tex_shader_previous(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
    ch->d3d_tex_shader_previous[2] = (param >> 16) & 3;
    ch->d3d_tex_shader_previous[3] = (param >> 20) & 3;
}

/* ---------------------------------------------------------------- */
/* Method-handler table setup                                        */

static void nv_geforce3_empty_method_handler(NVGeForce3State *s,
    NVGeForce3Channel *ch, uint32_t cls, uint32_t method, uint32_t param)
{
}

static void nv_geforce3_set_method_handler(NVGeForce3State *s, uint32_t cls,
                                           uint32_t method,
                                           NVGeForce3MethodHandler handler)
{
    NVGeForce3MethodHandler *table;

    g_assert(cls < NV_GEFORCE3_CLASS_COUNT);
    g_assert(method < NV_GEFORCE3_METHOD_COUNT);
    table = s->class_method_handlers[cls];
    /* a double-set here is a bug in this table, not a guest condition */
    g_assert(table[method] == nv_geforce3_empty_method_handler);
    table[method] = handler;
}

static void nv_geforce3_set_method_handler_range(NVGeForce3State *s,
    uint32_t cls, uint32_t method_start, uint32_t method_end,
    NVGeForce3MethodHandler handler)
{
    uint32_t i;

    g_assert(method_start <= method_end);
    for (i = method_start; i <= method_end; i++) {
        nv_geforce3_set_method_handler(s, cls, i, handler);
    }
}

static void nv_geforce3_set_d3d_method_handler(NVGeForce3State *s,
    bool cl0096, bool cl0097, bool cl0497, uint32_t method_start,
    uint32_t method_end, NVGeForce3MethodHandler handler)
{
    if (cl0096) {
        nv_geforce3_set_method_handler_range(s, 0x0096, method_start,
                                             method_end, handler);
    }
    if (cl0097) {
        nv_geforce3_set_method_handler_range(s, 0x0097, method_start,
                                             method_end, handler);
    }
    if (cl0497) {
        nv_geforce3_set_method_handler_range(s, 0x0497, method_start,
                                             method_end, handler);
    }
}

#define SET1(cl0096, cl0097, cl0497, method, handler) \
    nv_geforce3_set_d3d_method_handler(s, cl0096, cl0097, cl0497, method, \
                                       method, handler)
#define SETR(cl0096, cl0097, cl0497, start, end, handler) \
    nv_geforce3_set_d3d_method_handler(s, cl0096, cl0097, cl0497, start, \
                                       end, handler)

void nv_geforce3_init_method_handlers(NVGeForce3State *s)
{
    int i;

    for (i = 0; i < NV_GEFORCE3_METHOD_COUNT; i++) {
        s->empty_method_handlers[i] = nv_geforce3_empty_method_handler;
        s->cl0096_method_handlers[i] = nv_geforce3_empty_method_handler;
        s->cl0097_method_handlers[i] = nv_geforce3_empty_method_handler;
        s->cl0497_method_handlers[i] = nv_geforce3_empty_method_handler;
    }
    for (i = 0; i < NV_GEFORCE3_CLASS_COUNT; i++) {
        s->class_method_handlers[i] = s->empty_method_handlers;
    }
    s->class_method_handlers[0x0096] = s->cl0096_method_handlers;
    s->class_method_handlers[0x0097] = s->cl0097_method_handlers;
    s->class_method_handlers[0x0497] = s->cl0497_method_handlers;

    SET1(1, 1, 1, 0x000, nv_geforce3_d3d_mh_object);
    SET1(1, 1, 1, 0x048, nv_geforce3_d3d_mh_flip_read);
    SET1(1, 1, 1, 0x049, nv_geforce3_d3d_mh_flip_write);
    SET1(1, 1, 1, 0x04a, nv_geforce3_d3d_mh_flip_modulo);
    SET1(1, 1, 1, 0x04b, nv_geforce3_d3d_mh_flip_incr);
    SET1(1, 1, 1, 0x04c, nv_geforce3_d3d_mh_fifo_wait);
    SET1(1, 1, 1, 0x061, nv_geforce3_d3d_mh_a_obj);
    SET1(1, 1, 1, 0x062, nv_geforce3_d3d_mh_b_obj);
    SET1(1, 0, 0, 0x063, nv_geforce3_d3d_mh_vertex_obj);
    SET1(1, 1, 1, 0x065, nv_geforce3_d3d_mh_color_obj);
    SET1(1, 1, 1, 0x066, nv_geforce3_d3d_mh_zeta_obj);
    SET1(1, 1, 1, 0x067, nv_geforce3_d3d_mh_vertex_a_obj);
    SET1(1, 1, 1, 0x068, nv_geforce3_d3d_mh_vertex_b_obj);
    SET1(1, 1, 1, 0x069, nv_geforce3_d3d_mh_semaphore_obj);
    SET1(1, 1, 1, 0x06a, nv_geforce3_d3d_mh_report_obj);
    SET1(1, 1, 1, 0x080, nv_geforce3_d3d_mh_clip_horizontal);
    SET1(1, 1, 1, 0x081, nv_geforce3_d3d_mh_clip_vertical);
    SET1(1, 1, 1, 0x082, nv_geforce3_d3d_mh_surface_format);
    SET1(1, 1, 1, 0x083, nv_geforce3_d3d_mh_surface_pitch_a);
    SET1(1, 1, 1, 0x084, nv_geforce3_d3d_mh_surface_color_offset);
    SET1(1, 1, 1, 0x085, nv_geforce3_d3d_mh_surface_zeta_offset);
    SETR(1, 0, 0, 0x098, 0x099, nv_geforce3_d3d_mh_combiner_alpha_icw);
    SETR(0, 1, 0, 0x098, 0x09f, nv_geforce3_d3d_mh_combiner_alpha_icw);
    SETR(1, 1, 0, 0x0a2, 0x0a3, nv_geforce3_d3d_mh_combiner_final);
    SETR(0, 0, 1, 0x23d, 0x23e, nv_geforce3_d3d_mh_combiner_final);
    SET1(1, 1, 0, 0x0a5, nv_geforce3_d3d_mh_local_viewer);
    SET1(0, 0, 1, 0x509, nv_geforce3_d3d_mh_local_viewer);
    SET1(1, 1, 0, 0x0a6, nv_geforce3_d3d_mh_color_material);
    SET1(0, 0, 1, 0x0e4, nv_geforce3_d3d_mh_color_material);
    SET1(1, 1, 0, 0x0a7, nv_geforce3_d3d_mh_fog_mode);
    SET1(0, 0, 1, 0x233, nv_geforce3_d3d_mh_fog_mode);
    SET1(1, 1, 0, 0x0a8, nv_geforce3_d3d_mh_fog_gen_mode);
    SET1(0, 0, 1, 0x232, nv_geforce3_d3d_mh_fog_gen_mode);
    SETR(1, 0, 0, 0x1a0, 0x1a2, nv_geforce3_d3d_mh_fog_params);
    SETR(0, 1, 0, 0x270, 0x272, nv_geforce3_d3d_mh_fog_params);
    SETR(0, 0, 1, 0x234, 0x236, nv_geforce3_d3d_mh_fog_params);
    SET1(1, 1, 0, 0x0a9, nv_geforce3_d3d_mh_fog_enable);
    SET1(0, 0, 1, 0x0db, nv_geforce3_d3d_mh_fog_enable);
    SET1(1, 1, 0, 0x0aa, nv_geforce3_d3d_mh_fog_color);
    SET1(0, 0, 1, 0x0dc, nv_geforce3_d3d_mh_fog_color);
    SET1(0, 0, 1, 0x0ae, nv_geforce3_d3d_mh_window_offset);
    SETR(0, 0, 1, 0x0b0, 0x0bf, nv_geforce3_d3d_mh_window_clip);
    SET1(1, 1, 0, 0x0c0, nv_geforce3_d3d_mh_alpha_test_enable);
    SET1(0, 0, 1, 0x0c1, nv_geforce3_d3d_mh_alpha_test_enable);
    SET1(1, 1, 0, 0x0cf, nv_geforce3_d3d_mh_alpha_func);
    SET1(0, 0, 1, 0x0c2, nv_geforce3_d3d_mh_alpha_func);
    SET1(1, 1, 0, 0x0d0, nv_geforce3_d3d_mh_alpha_ref);
    SET1(0, 0, 1, 0x0c3, nv_geforce3_d3d_mh_alpha_ref);
    SET1(1, 1, 0, 0x0c1, nv_geforce3_d3d_mh_blend_enable);
    SET1(0, 0, 1, 0x0c4, nv_geforce3_d3d_mh_blend_enable);
    SET1(1, 1, 0, 0x0c2, nv_geforce3_d3d_mh_cull_face_enable);
    SET1(0, 0, 1, 0x60f, nv_geforce3_d3d_mh_cull_face_enable);
    SET1(1, 1, 0, 0x0c3, nv_geforce3_d3d_mh_depth_test_enable);
    SET1(0, 0, 1, 0x29d, nv_geforce3_d3d_mh_depth_test_enable);
    SET1(1, 1, 0, 0x0c5, nv_geforce3_d3d_mh_lighting_enable);
    SET1(0, 0, 1, 0x516, nv_geforce3_d3d_mh_lighting_enable);
    SET1(1, 1, 0, 0x0cb, nv_geforce3_d3d_mh_stencil_test_enable);
    SET1(0, 0, 1, 0x0ca, nv_geforce3_d3d_mh_stencil_test_enable);
    SET1(1, 1, 0, 0x0d1, nv_geforce3_d3d_mh_blend_sfactor_0096);
    SET1(1, 1, 0, 0x0d2, nv_geforce3_d3d_mh_blend_dfactor_0096);
    SET1(1, 1, 0, 0x0d4, nv_geforce3_d3d_mh_blend_equation_0096);
    SET1(0, 0, 1, 0x0c5, nv_geforce3_d3d_mh_blend_sfactor_0497);
    SET1(0, 0, 1, 0x0c6, nv_geforce3_d3d_mh_blend_dfactor_0497);
    SET1(0, 0, 1, 0x0c8, nv_geforce3_d3d_mh_blend_equation_0497);
    SET1(1, 1, 0, 0x0d3, nv_geforce3_d3d_mh_blend_color);
    SET1(0, 0, 1, 0x0c7, nv_geforce3_d3d_mh_blend_color);
    SET1(1, 1, 0, 0x0d5, nv_geforce3_d3d_mh_depth_func);
    SET1(0, 0, 1, 0x29b, nv_geforce3_d3d_mh_depth_func);
    SET1(1, 1, 0, 0x0d6, nv_geforce3_d3d_mh_color_mask);
    SET1(0, 0, 1, 0x0c9, nv_geforce3_d3d_mh_color_mask);
    SET1(1, 1, 0, 0x0d7, nv_geforce3_d3d_mh_depth_write_enable);
    SET1(0, 0, 1, 0x29c, nv_geforce3_d3d_mh_depth_write_enable);
    SET1(1, 1, 0, 0x0d8, nv_geforce3_d3d_mh_stencil_mask);
    SET1(0, 0, 1, 0x0cb, nv_geforce3_d3d_mh_stencil_mask);
    SET1(1, 1, 0, 0x0d9, nv_geforce3_d3d_mh_stencil_func);
    SET1(0, 0, 1, 0x0cc, nv_geforce3_d3d_mh_stencil_func);
    SET1(1, 1, 0, 0x0da, nv_geforce3_d3d_mh_stencil_func_ref);
    SET1(0, 0, 1, 0x0cd, nv_geforce3_d3d_mh_stencil_func_ref);
    SET1(1, 1, 0, 0x0db, nv_geforce3_d3d_mh_stencil_func_mask);
    SET1(0, 0, 1, 0x0ce, nv_geforce3_d3d_mh_stencil_func_mask);
    SET1(1, 1, 0, 0x0dc, nv_geforce3_d3d_mh_stencil_op_sfail);
    SET1(0, 0, 1, 0x0cf, nv_geforce3_d3d_mh_stencil_op_sfail);
    SET1(1, 1, 0, 0x0dd, nv_geforce3_d3d_mh_stencil_op_dpfail);
    SET1(0, 0, 1, 0x0d0, nv_geforce3_d3d_mh_stencil_op_dpfail);
    SET1(1, 1, 0, 0x0de, nv_geforce3_d3d_mh_stencil_op_dppass);
    SET1(0, 0, 1, 0x0d1, nv_geforce3_d3d_mh_stencil_op_dppass);
    SET1(1, 1, 0, 0x0df, nv_geforce3_d3d_mh_shade_mode);
    SET1(0, 0, 1, 0x0da, nv_geforce3_d3d_mh_shade_mode);
    SET1(1, 1, 1, 0x0e5, nv_geforce3_d3d_mh_clip_min);
    SET1(1, 1, 1, 0x0e6, nv_geforce3_d3d_mh_clip_max);
    SET1(1, 1, 0, 0x0e7, nv_geforce3_d3d_mh_cull_face);
    SET1(0, 0, 1, 0x60c, nv_geforce3_d3d_mh_cull_face);
    SET1(1, 1, 0, 0x0e8, nv_geforce3_d3d_mh_front_face);
    SET1(0, 0, 1, 0x60d, nv_geforce3_d3d_mh_front_face);
    SET1(1, 1, 0, 0x0e9, nv_geforce3_d3d_mh_normalize_enable);
    SET1(0, 0, 1, 0x0df, nv_geforce3_d3d_mh_normalize_enable);
    SETR(1, 1, 0, 0x0ea, 0x0ed, nv_geforce3_d3d_mh_material_factor);
    SET1(0, 0, 1, 0x0ed, nv_geforce3_d3d_mh_material_factor);
    SET1(1, 1, 0, 0x0ee, nv_geforce3_d3d_mh_separate_specular);
    SET1(0, 0, 1, 0x50a, nv_geforce3_d3d_mh_separate_specular);
    SET1(1, 1, 0, 0x0ef, nv_geforce3_d3d_mh_light_enable_mask);
    SET1(0, 0, 1, 0x508, nv_geforce3_d3d_mh_light_enable_mask);
    SETR(1, 0, 0, 0x0f0, 0x0f7, nv_geforce3_d3d_mh_texgen);
    SETR(0, 1, 0, 0x0f0, 0x0ff, nv_geforce3_d3d_mh_texgen);
    SETR(0, 0, 1, 0x100, 0x11f, nv_geforce3_d3d_mh_texgen);
    SETR(1, 0, 0, 0x0f8, 0x0f9, nv_geforce3_d3d_mh_texture_matrix_enable);
    SETR(0, 1, 0, 0x108, 0x10b, nv_geforce3_d3d_mh_texture_matrix_enable);
    SETR(0, 0, 1, 0x090, 0x097, nv_geforce3_d3d_mh_texture_matrix_enable);
    SET1(1, 0, 0, 0x0fa, nv_geforce3_d3d_mh_view_matrix_enable);
    SETR(1, 0, 0, 0x100, 0x11f, nv_geforce3_d3d_mh_model_view_matrix);
    SETR(0, 1, 1, 0x120, 0x13f, nv_geforce3_d3d_mh_model_view_matrix);
    SETR(1, 0, 0, 0x120, 0x12b, nv_geforce3_d3d_mh_inverse_model_view_matrix);
    SETR(0, 1, 1, 0x160, 0x16b, nv_geforce3_d3d_mh_inverse_model_view_matrix);
    SETR(1, 0, 0, 0x140, 0x14f, nv_geforce3_d3d_mh_composite_matrix);
    SETR(0, 1, 1, 0x1a0, 0x1af, nv_geforce3_d3d_mh_composite_matrix);
    SETR(1, 0, 0, 0x150, 0x16f, nv_geforce3_d3d_mh_texture_matrix);
    SETR(0, 1, 0, 0x1b0, 0x1ef, nv_geforce3_d3d_mh_texture_matrix);
    SETR(0, 0, 1, 0x1b0, 0x22f, nv_geforce3_d3d_mh_texture_matrix);
    SETR(1, 0, 0, 0x180, 0x19f, nv_geforce3_d3d_mh_texgen_plane);
    SETR(0, 1, 0, 0x210, 0x24f, nv_geforce3_d3d_mh_texgen_plane);
    SETR(0, 0, 1, 0x380, 0x3ff, nv_geforce3_d3d_mh_texgen_plane);
    SET1(0, 0, 1, 0x230, nv_geforce3_d3d_mh_scissor_x_width);
    SET1(0, 0, 1, 0x231, nv_geforce3_d3d_mh_scissor_y_height);
    SET1(0, 0, 1, 0x239, nv_geforce3_d3d_mh_shader_program);
    SETR(0, 0, 1, 0x240, 0x27f, nv_geforce3_d3d_mh_0497_240);
    SET1(0, 0, 1, 0x280, nv_geforce3_d3d_mh_viewport_x_width);
    SET1(0, 0, 1, 0x281, nv_geforce3_d3d_mh_viewport_y_height);
    SETR(1, 0, 0, 0x1a8, 0x1ad, nv_geforce3_d3d_mh_specular_params);
    SETR(0, 1, 0, 0x278, 0x27d, nv_geforce3_d3d_mh_specular_params);
    SETR(0, 0, 1, 0x500, 0x505, nv_geforce3_d3d_mh_specular_params);
    SETR(1, 0, 0, 0x1b1, 0x1b3, nv_geforce3_d3d_mh_scene_ambient_color);
    SETR(0, 1, 1, 0x284, 0x286, nv_geforce3_d3d_mh_scene_ambient_color);
    SETR(1, 0, 0, 0x1ba, 0x1bd, nv_geforce3_d3d_mh_viewport_offset);
    SETR(0, 1, 1, 0x288, 0x28b, nv_geforce3_d3d_mh_viewport_offset);
    SETR(0, 1, 1, 0x294, 0x297, nv_geforce3_d3d_mh_eye_position);
    SETR(1, 0, 0, 0x09c, 0x09d, nv_geforce3_d3d_mh_0096_09c);
    SETR(0, 1, 0, 0x298, 0x2a7, nv_geforce3_d3d_mh_0097_298);
    SETR(1, 0, 0, 0x09e, 0x09f, nv_geforce3_d3d_mh_combiner_alpha_ocw);
    SETR(0, 1, 0, 0x2a8, 0x2af, nv_geforce3_d3d_mh_combiner_alpha_ocw);
    SETR(1, 0, 0, 0x09a, 0x09b, nv_geforce3_d3d_mh_combiner_color_icw);
    SETR(0, 1, 0, 0x2b0, 0x2b7, nv_geforce3_d3d_mh_combiner_color_icw);
    SETR(0, 1, 0, 0x2b8, 0x2bb, nv_geforce3_d3d_mh_texture_key_color);
    SETR(0, 0, 1, 0x740, 0x74f, nv_geforce3_d3d_mh_texture_key_color);
    SETR(0, 1, 0, 0x2bc, 0x2bf, nv_geforce3_d3d_mh_viewport_scale);
    SETR(0, 0, 1, 0x28c, 0x28f, nv_geforce3_d3d_mh_viewport_scale);
    SETR(0, 1, 0, 0x2c0, 0x2c3, nv_geforce3_d3d_mh_transform_program);
    SETR(0, 0, 1, 0x2e0, 0x2e3, nv_geforce3_d3d_mh_transform_program);
    SETR(0, 1, 0, 0x2e0, 0x2e3, nv_geforce3_d3d_mh_transform_constant);
    SETR(0, 0, 1, 0x7c0, 0x7cf, nv_geforce3_d3d_mh_transform_constant);
    SETR(1, 0, 0, 0x200, 0x2ff, nv_geforce3_d3d_mh_light);
    SETR(0, 1, 1, 0x400, 0x4ff, nv_geforce3_d3d_mh_light);
    SETR(1, 0, 0, 0x300, 0x302, nv_geforce3_d3d_mh_0096_300);
    SETR(0, 1, 0, 0x540, 0x542, nv_geforce3_d3d_mh_0096_300);
    SETR(0, 0, 1, 0x540, 0x57f, nv_geforce3_d3d_mh_0497_540);
    SETR(1, 0, 0, 0x306, 0x309, nv_geforce3_d3d_mh_0096_306);
    SETR(0, 1, 0, 0x546, 0x549, nv_geforce3_d3d_mh_0096_306);
    SETR(1, 0, 0, 0x30c, 0x30e, nv_geforce3_d3d_mh_0096_30c);
    SETR(0, 1, 0, 0x54c, 0x54e, nv_geforce3_d3d_mh_0096_30c);
    SETR(1, 0, 0, 0x314, 0x317, nv_geforce3_d3d_mh_0096_314);
    SETR(0, 1, 0, 0x554, 0x557, nv_geforce3_d3d_mh_0096_314);
    SETR(1, 0, 0, 0x318, 0x31a, nv_geforce3_d3d_mh_0096_318);
    SETR(0, 1, 0, 0x558, 0x55a, nv_geforce3_d3d_mh_0096_318);
    SET1(1, 0, 0, 0x31b, nv_geforce3_d3d_mh_0096_31b);
    SET1(0, 1, 0, 0x55b, nv_geforce3_d3d_mh_0096_31b);
    SETR(1, 0, 0, 0x324, 0x337, nv_geforce3_d3d_mh_texcoord);
    SETR(0, 1, 0, 0x564, 0x58b, nv_geforce3_d3d_mh_texcoord);
    SETR(0, 1, 0, 0x5c8, 0x5d7, nv_geforce3_d3d_mh_0097_5c8);
    SETR(0, 0, 1, 0x5a0, 0x5af, nv_geforce3_d3d_mh_0097_5c8);
    SETR(1, 0, 0, 0x340, 0x34f, nv_geforce3_d3d_mh_vertex_data_array_format);
    SETR(0, 1, 0, 0x5d8, 0x5e7, nv_geforce3_d3d_mh_vertex_data_array_format);
    SETR(0, 0, 1, 0x5d0, 0x5df, nv_geforce3_d3d_mh_vertex_data_array_format);
    SET1(0, 1, 0, 0x5f4, nv_geforce3_d3d_mh_get_report);
    SET1(0, 0, 1, 0x600, nv_geforce3_d3d_mh_get_report);
    SET1(1, 0, 0, 0x37f, nv_geforce3_d3d_mh_begin_end);
    SET1(1, 0, 0, 0x4ff, nv_geforce3_d3d_mh_begin_end);
    SET1(1, 1, 0, 0x5ff, nv_geforce3_d3d_mh_begin_end);
    SET1(0, 0, 1, 0x602, nv_geforce3_d3d_mh_begin_end);
    SET1(1, 0, 0, 0x380, nv_geforce3_d3d_mh_array_element16);
    SET1(0, 1, 0, 0x600, nv_geforce3_d3d_mh_array_element16);
    SET1(0, 0, 1, 0x603, nv_geforce3_d3d_mh_array_element16);
    SET1(1, 0, 0, 0x440, nv_geforce3_d3d_mh_array_element32);
    SET1(0, 1, 0, 0x602, nv_geforce3_d3d_mh_array_element32);
    SET1(0, 0, 1, 0x604, nv_geforce3_d3d_mh_array_element32);
    SET1(1, 0, 0, 0x500, nv_geforce3_d3d_mh_draw_arrays);
    SET1(0, 1, 0, 0x604, nv_geforce3_d3d_mh_draw_arrays);
    SET1(0, 0, 1, 0x605, nv_geforce3_d3d_mh_draw_arrays);
    SETR(1, 0, 0, 0x600, 0x6ff, nv_geforce3_d3d_mh_inline_array);
    SET1(0, 1, 1, 0x606, nv_geforce3_d3d_mh_inline_array);
    SET1(0, 0, 1, 0x607, nv_geforce3_d3d_mh_index_array_offset);
    SET1(0, 0, 1, 0x608, nv_geforce3_d3d_mh_index_array_dma);
    SET1(0, 0, 1, 0x609, nv_geforce3_d3d_mh_0497_609);
    SET1(0, 1, 0, 0x60a, nv_geforce3_d3d_mh_0097_60a);
    SET1(0, 0, 1, 0x0e7, nv_geforce3_d3d_mh_0097_60a);
    SETR(0, 0, 1, 0x610, 0x61f, nv_geforce3_d3d_mh_0497_610);
    SETR(0, 1, 1, 0x620, 0x63f, nv_geforce3_d3d_mh_0097_620);
    SETR(0, 1, 1, 0x640, 0x64f, nv_geforce3_d3d_mh_0097_640);
    SETR(0, 1, 1, 0x650, 0x65f, nv_geforce3_d3d_mh_0097_650);
    SETR(0, 1, 0, 0x680, 0x6bf, nv_geforce3_d3d_mh_0097_680);
    SETR(0, 0, 1, 0x700, 0x73f, nv_geforce3_d3d_mh_0097_680);
    SETR(1, 0, 0, 0x086, 0x095, nv_geforce3_d3d_mh_texture);
    SETR(0, 1, 0, 0x6c0, 0x6ff, nv_geforce3_d3d_mh_texture);
    SETR(0, 0, 1, 0x680, 0x6ff, nv_geforce3_d3d_mh_texture);
    SET1(1, 1, 1, 0x758, nv_geforce3_d3d_mh_shader_control);
    SET1(1, 1, 1, 0x75b, nv_geforce3_d3d_mh_semaphore_offset);
    SET1(1, 1, 1, 0x763, nv_geforce3_d3d_mh_zstencil_clear_value);
    SET1(1, 1, 1, 0x764, nv_geforce3_d3d_mh_color_clear_value);
    SET1(1, 1, 1, 0x765, nv_geforce3_d3d_mh_clear_surface);
    SETR(1, 0, 0, 0x0a0, 0x0a1, nv_geforce3_d3d_mh_combiner_color_ocw);
    SETR(0, 1, 0, 0x790, 0x797, nv_geforce3_d3d_mh_combiner_color_ocw);
    SET1(0, 1, 0, 0x798, nv_geforce3_d3d_mh_combiner_control);
    SET1(0, 0, 1, 0x23f, nv_geforce3_d3d_mh_combiner_control);
    SET1(0, 1, 0, 0x79c, nv_geforce3_d3d_mh_tex_shader_op);
    SET1(0, 1, 0, 0x79d, nv_geforce3_d3d_mh_tex_shader_dotmapping);
    SET1(0, 1, 0, 0x79e, nv_geforce3_d3d_mh_tex_shader_previous);
    SET1(1, 1, 1, 0x7a5, nv_geforce3_d3d_mh_transform_execution_mode);
    SET1(1, 1, 1, 0x7a7, nv_geforce3_d3d_mh_transform_program_load);
    SET1(1, 1, 1, 0x7a8, nv_geforce3_d3d_mh_transform_program_start);
    SET1(0, 1, 0, 0x7a9, nv_geforce3_d3d_mh_transform_constant_load);
    SET1(0, 0, 1, 0x7bf, nv_geforce3_d3d_mh_transform_constant_load);
}

#undef SET1
#undef SETR

void nv_geforce3_execute_d3d(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t cls, uint32_t method, uint32_t param)
{
    s->class_method_handlers[cls][method](s, ch, cls, method, param);
}
