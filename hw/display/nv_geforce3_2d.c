/*
 * QEMU NVIDIA GeForce3 (NV20) emulation -- 2D ("software object")
 * engine: the PFIFO-driven GDI/blit/rectangle/image-transfer classes
 * and the M2MF copy engine.
 *
 * Ported from Bochs' geforce.cc (see nv_geforce3_int.h). The one
 * substantive departure is the ROP3 (ternary raster operation)
 * engine: rather than port bitblt.h's ~30 hand-specialised forward
 * ROP functions plus its separate general-case evaluator, this uses
 * a single generic bit-level ROP3 evaluator for all 256 codes (the
 * same technique ati_rage128_2d.c's ati_rage128_apply_rop3() uses),
 * which makes the per-code dispatch table (Bochs' rop_handler[]/
 * rop_flags[]) and its setup unnecessary.
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"

#include "nv_geforce3_int.h"

static uint32_t nv_geforce3_apply_rop3(uint8_t rop, uint32_t src,
                                       uint32_t dst, uint32_t pat)
{
    uint32_t result = 0;
    int bit;

    switch (rop) {
    case 0x00:
        return 0;
    case 0xff:
        return 0xffffffffu;
    case 0xcc: /* S */
        return src;
    case 0xaa: /* D */
        return dst;
    case 0xf0: /* P */
        return pat;
    case 0x55: /* Dn */
        return ~dst;
    case 0x66: /* DSx */
        return src ^ dst;
    case 0x88: /* DSa */
        return src & dst;
    case 0xee: /* DSo */
        return src | dst;
    case 0x33: /* Sn */
        return ~src;
    default:
        break;
    }

    for (bit = 0; bit < 32; bit++) {
        uint32_t mask = 1u << bit;
        int sb = (src & mask) ? 1 : 0;
        int db = (dst & mask) ? 1 : 0;
        int pb = (pat & mask) ? 1 : 0;
        /*
         * Standard ROP3 truth-table indexing: bit (P<<2)|(S<<1)|D of
         * the ROP code selects the output for that (P,S,D) input.
         */
        int idx = (pb << 2) | (sb << 1) | db;

        if (rop & (1 << idx)) {
            result |= mask;
        }
    }
    return result;
}

static uint32_t nv_geforce3_color_565_to_888(uint16_t value)
{
    uint8_t r = ((value >> 11) & 0x1f) << 3;
    uint8_t g = ((value >> 5) & 0x3f) << 2;
    uint8_t b = (value & 0x1f) << 3;

    r |= r >> 5;
    g |= g >> 6;
    b |= b >> 5;
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
}

static uint16_t nv_geforce3_color_888_to_565(uint32_t value)
{
    return (uint16_t)((((value >> 19) & 0x1F) << 11) |
                      (((value >> 10) & 0x3F) << 5) |
                      ((value >> 3) & 0x1F));
}

uint32_t nv_geforce3_swizzle(uint32_t x, uint32_t y, uint32_t z,
                             uint32_t width, uint32_t height, uint32_t depth)
{
    bool xleft = true;
    bool yleft = height != 1;
    bool zleft = depth != 1;
    uint32_t xbit = 1, ybit = 1, zbit = 1, rbit = 1, r = 0;

    do {
        if (xleft) {
            if (x & xbit) {
                r |= rbit;
            }
            rbit <<= 1;
            xbit <<= 1;
            xleft = xbit < width;
        }
        if (yleft) {
            if (y & ybit) {
                r |= rbit;
            }
            rbit <<= 1;
            ybit <<= 1;
            yleft = ybit < height;
        }
        if (zleft) {
            if (z & zbit) {
                r |= rbit;
            }
            rbit <<= 1;
            zbit <<= 1;
            zleft = zbit < depth;
        }
    } while (xleft || yleft || zleft);
    return r;
}

uint32_t nv_geforce3_get_pixel(NVGeForce3State *s, uint32_t obj, uint32_t ofs,
                               uint32_t x, uint32_t cb)
{
    if (cb == 1) {
        return nv_geforce3_dma_read8(s, obj, ofs + x);
    } else if (cb == 2) {
        return nv_geforce3_dma_read16(s, obj, ofs + x * 2);
    }
    return nv_geforce3_dma_read32(s, obj, ofs + x * 4);
}

void nv_geforce3_put_pixel(NVGeForce3State *s, NVGeForce3Channel *ch,
                           uint32_t ofs, uint32_t x, uint32_t value)
{
    if (ch->s2d_color_bytes == 1) {
        nv_geforce3_dma_write8(s, ch->s2d_img_dst, ofs + x, (uint8_t)value);
    } else if (ch->s2d_color_bytes == 2) {
        nv_geforce3_dma_write16(s, ch->s2d_img_dst, ofs + x * 2,
                                (uint16_t)value);
    } else if (ch->s2d_color_fmt == 6) {
        nv_geforce3_dma_write32(s, ch->s2d_img_dst, ofs + x * 4,
                                value & 0x00FFFFFF);
    } else {
        nv_geforce3_dma_write32(s, ch->s2d_img_dst, ofs + x * 4, value);
    }
}

static void nv_geforce3_put_pixel_swzs(NVGeForce3State *s,
                                       NVGeForce3Channel *ch, uint32_t ofs,
                                       uint32_t value)
{
    if (ch->swzs_color_bytes == 1) {
        nv_geforce3_dma_write8(s, ch->swzs_img_obj, ofs, (uint8_t)value);
    } else if (ch->swzs_color_bytes == 2) {
        nv_geforce3_dma_write16(s, ch->swzs_img_obj, ofs, (uint16_t)value);
    } else {
        nv_geforce3_dma_write32(s, ch->swzs_img_obj, ofs, value);
    }
}

static uint8_t nv_geforce3_alpha_wrap(int value)
{
    return (uint8_t)(-(value >> 8) ^ value);
}

void nv_geforce3_pixel_operation(NVGeForce3State *s, NVGeForce3Channel *ch,
                                 uint32_t op, uint32_t *dstcolor,
                                 const uint32_t *srccolor, uint32_t cb,
                                 uint32_t px, uint32_t py)
{
    if (op == 1) {
        uint32_t patt_color;
        uint32_t i = (py % 8) * 8 + (px % 8);

        if (ch->patt_type_color) {
            patt_color = ch->patt_data_color[i];
        } else {
            patt_color = ch->patt_data_mono[i] ? ch->patt_fg_color :
                                                 ch->patt_bg_color;
        }
        *dstcolor = nv_geforce3_apply_rop3(ch->rop, *srccolor, *dstcolor,
                                           patt_color);
        if (cb < 4) {
            *dstcolor &= (1u << (cb * 8)) - 1;
        }
    } else if (op == 5) {
        if (cb == 4) {
            if (*srccolor) {
                uint8_t sb = *srccolor;
                uint8_t sg = *srccolor >> 8;
                uint8_t sr = *srccolor >> 16;
                uint8_t sa = *srccolor >> 24;
                uint32_t beta = ch->beta;

                if (beta != 0xFFFFFFFF) {
                    uint8_t bb = beta, bg = beta >> 8, br = beta >> 16;
                    uint8_t ba = beta >> 24;

                    sb = sb * bb / 0xFF;
                    sg = sg * bg / 0xFF;
                    sr = sr * br / 0xFF;
                    sa = sa * ba / 0xFF;
                }
                {
                    uint8_t db = *dstcolor, dg = *dstcolor >> 8;
                    uint8_t dr = *dstcolor >> 16, da = *dstcolor >> 24;
                    uint8_t isa = 0xFF - sa;
                    uint8_t b = nv_geforce3_alpha_wrap(db * isa / 0xFF + sb);
                    uint8_t g = nv_geforce3_alpha_wrap(dg * isa / 0xFF + sg);
                    uint8_t r = nv_geforce3_alpha_wrap(dr * isa / 0xFF + sr);
                    uint8_t a = nv_geforce3_alpha_wrap(da * isa / 0xFF + sa);

                    *dstcolor = (uint32_t)b | ((uint32_t)g << 8) |
                               ((uint32_t)r << 16) | ((uint32_t)a << 24);
                }
            }
        } else {
            uint32_t beta = ch->beta;
            uint8_t bb = beta, bg = beta >> 8, br = beta >> 16;
            uint8_t iba = 0xFF - (beta >> 24);
            uint8_t sb = *srccolor & 0x1F;
            uint8_t sg = (*srccolor >> 5) & 0x3F;
            uint8_t sr = (*srccolor >> 11) & 0x1F;
            uint8_t db = *dstcolor & 0x1F;
            uint8_t dg = (*dstcolor >> 5) & 0x3F;
            uint8_t dr = (*dstcolor >> 11) & 0x1F;
            uint8_t b = (db * iba + sb * bb) / 0xFF;
            uint8_t g = (dg * iba + sg * bg) / 0xFF;
            uint8_t r = (dr * iba + sr * br) / 0xFF;

            *dstcolor = (uint32_t)b | ((uint32_t)g << 5) |
                       ((uint32_t)r << 11);
        }
    } else {
        *dstcolor = *srccolor;
    }
}

static void nv_geforce3_gdi_fillrect(NVGeForce3State *s,
                                     NVGeForce3Channel *ch, bool clipped)
{
    int16_t clipx0 = 0, clipy0 = 0, clipx1 = 0, clipy1 = 0;
    int16_t dx, dy;
    uint16_t width, height;
    uint32_t pitch = ch->s2d_pitch_dst;
    uint32_t srccolor = ch->gdi_rect_color;
    uint32_t draw_offset;
    uint16_t x, y;

    if (clipped) {
        clipx0 = ch->gdi_clip_yx0 & 0xFFFF;
        clipy0 = ch->gdi_clip_yx0 >> 16;
        clipx1 = ch->gdi_clip_yx1 & 0xFFFF;
        clipy1 = ch->gdi_clip_yx1 >> 16;
        dx = ch->gdi_rect_yx0 & 0xFFFF;
        dy = ch->gdi_rect_yx0 >> 16;
        clipx0 -= dx;
        clipy0 -= dy;
        clipx1 -= dx;
        clipy1 -= dy;
        width = (ch->gdi_rect_yx1 & 0xFFFF) - dx;
        height = (ch->gdi_rect_yx1 >> 16) - dy;
    } else {
        dx = ch->gdi_rect_xy >> 16;
        dy = ch->gdi_rect_xy & 0xFFFF;
        width = ch->gdi_rect_wh >> 16;
        height = ch->gdi_rect_wh & 0xFFFF;
    }

    draw_offset = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            if (!clipped || (x >= clipx0 && x < clipx1 && y >= clipy0 &&
                             y < clipy1)) {
                uint32_t dstcolor = nv_geforce3_get_pixel(s, ch->s2d_img_dst,
                    draw_offset, x, ch->s2d_color_bytes);

                nv_geforce3_pixel_operation(s, ch, ch->gdi_operation,
                    &dstcolor, &srccolor, ch->s2d_color_bytes, dx + x,
                    dy + y);
                nv_geforce3_put_pixel(s, ch, draw_offset, x, dstcolor);
            }
        }
        draw_offset += pitch;
    }
}

static void nv_geforce3_gdi_blit(NVGeForce3State *s, NVGeForce3Channel *ch,
                                 uint32_t type)
{
    int16_t dx = ch->gdi_image_xy & 0xFFFF;
    int16_t dy = ch->gdi_image_xy >> 16;
    int16_t clipx0 = (ch->gdi_clip_yx0 & 0xFFFF) - dx;
    int16_t clipy0 = (ch->gdi_clip_yx0 >> 16) - dy;
    int16_t clipx1 = (ch->gdi_clip_yx1 & 0xFFFF) - dx;
    int16_t clipy1 = (ch->gdi_clip_yx1 >> 16) - dy;
    uint32_t swidth = ch->gdi_image_swh & 0xFFFF;
    uint32_t dwidth = type ? ch->gdi_image_dwh & 0xFFFF : swidth;
    uint32_t height = ch->gdi_image_swh >> 16;
    uint32_t pitch = ch->s2d_pitch_dst;
    uint32_t bg_color = ch->gdi_bg_color;
    uint32_t fg_color = ch->gdi_fg_color;
    uint32_t draw_offset;
    uint32_t bit_index = 0;
    uint16_t x, y;

    if (ch->s2d_color_bytes == 4 && ch->gdi_color_fmt != 3) {
        bg_color = nv_geforce3_color_565_to_888(bg_color);
        fg_color = nv_geforce3_color_565_to_888(fg_color);
    }
    draw_offset = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;
    for (y = 0; y < height; y++) {
        for (x = 0; x < dwidth; x++) {
            if (x >= clipx0 && x < clipx1 && y >= clipy0 && y < clipy1) {
                uint32_t word_offset = bit_index / 32;
                uint32_t bit_offset = bit_index % 32;
                bool pixel;

                if (ch->gdi_mono_fmt == 1) {
                    bit_offset ^= 7;
                }
                pixel = (ch->gdi_words[word_offset] >> bit_offset) & 1;
                if (type || (!type && pixel)) {
                    uint32_t dstcolor = nv_geforce3_get_pixel(s,
                        ch->s2d_img_dst, draw_offset, x, ch->s2d_color_bytes);
                    uint32_t srccolor = pixel ? fg_color : bg_color;

                    nv_geforce3_pixel_operation(s, ch, ch->gdi_operation,
                        &dstcolor, &srccolor, ch->s2d_color_bytes, dx + x,
                        dy + y);
                    nv_geforce3_put_pixel(s, ch, draw_offset, x, dstcolor);
                }
            }
            bit_index++;
        }
        bit_index += swidth - dwidth;
        draw_offset += pitch;
    }
}

static void nv_geforce3_lin(NVGeForce3State *s, NVGeForce3Channel *ch)
{
    int32_t x0 = ch->lin_x0, y0 = ch->lin_y0;
    int32_t x1 = ch->lin_x1, y1 = ch->lin_y1;
    uint32_t srccolor = ch->lin_color;
    int32_t clipx0 = ch->clip_x, clipy0 = ch->clip_y;
    int32_t clipx1 = clipx0 + (int32_t)ch->clip_width;
    int32_t clipy1 = clipy0 + (int32_t)ch->clip_height;
    int32_t dx = abs(x1 - x0);
    int32_t incx = x0 < x1 ? 1 : -1;
    int32_t dy = -abs(y1 - y0);
    int32_t incy = y0 < y1 ? 1 : -1;
    int32_t error = dx + dy;
    int32_t x = x0, y = y0;

    for (;;) {
        if (x >= clipx0 && x < clipx1 && y >= clipy0 && y < clipy1 &&
            (x != x1 || y != y1)) {
            uint32_t draw_offset = ch->s2d_ofs_dst + y * ch->s2d_pitch_dst;
            uint32_t dstcolor = nv_geforce3_get_pixel(s, ch->s2d_img_dst,
                draw_offset, x, ch->s2d_color_bytes);

            nv_geforce3_pixel_operation(s, ch, ch->lin_operation, &dstcolor,
                &srccolor, ch->s2d_color_bytes, x, y);
            nv_geforce3_put_pixel(s, ch, draw_offset, x, dstcolor);
        }
        {
            int32_t e2 = error * 2;

            if (e2 >= dy) {
                if (x == x1) {
                    break;
                }
                error += dy;
                x += incx;
            }
            if (e2 <= dx) {
                if (y == y1) {
                    break;
                }
                error += dx;
                y += incy;
            }
        }
    }
}

static void nv_geforce3_rect(NVGeForce3State *s, NVGeForce3Channel *ch)
{
    int32_t x0 = (int16_t)ch->rect_yx;
    int32_t y0 = (int16_t)(ch->rect_yx >> 16);
    uint32_t width = ch->rect_hw & 0xFFFF;
    uint32_t height = ch->rect_hw >> 16;
    uint32_t srccolor = ch->rect_color;
    uint32_t draw_offset = ch->s2d_ofs_dst + y0 * ch->s2d_pitch_dst +
                           x0 * ch->s2d_color_bytes;
    uint32_t x, y;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            uint32_t dstcolor = nv_geforce3_get_pixel(s, ch->s2d_img_dst,
                draw_offset, x, ch->s2d_color_bytes);

            nv_geforce3_pixel_operation(s, ch, ch->rect_operation, &dstcolor,
                &srccolor, ch->s2d_color_bytes, x0 + x, y0 + y);
            nv_geforce3_put_pixel(s, ch, draw_offset, x, dstcolor);
        }
        draw_offset += ch->s2d_pitch_dst;
    }
}

static void nv_geforce3_ifc(NVGeForce3State *s, NVGeForce3Channel *ch,
                            uint32_t word)
{
    uint32_t chromacolor = 0;
    bool chroma_enabled = false;
    uint32_t i;

    if (ch->ifc_color_key_enable) {
        if (ch->ifc_color_bytes == 4) {
            chromacolor = ch->chroma_color & 0x00FFFFFF;
            chroma_enabled = (ch->chroma_color & 0xFF000000) != 0;
        } else if (ch->ifc_color_bytes == 2) {
            chromacolor = ch->chroma_color & 0x0000FFFF;
            chroma_enabled = (ch->chroma_color & 0xFFFF0000) != 0;
        } else {
            chromacolor = ch->chroma_color & 0x000000FF;
            chroma_enabled = (ch->chroma_color & 0xFFFFFF00) != 0;
        }
    }
    for (i = 0; i < ch->ifc_pixels_per_word; i++) {
        if (ch->ifc_x >= ch->ifc_clip_x0 && ch->ifc_x < ch->ifc_clip_x1 &&
            ch->ifc_y >= ch->ifc_clip_y0 && ch->ifc_y < ch->ifc_clip_y1) {
            uint32_t srccolor;

            if (ch->ifc_color_bytes == 4) {
                srccolor = word;
            } else if (ch->ifc_color_bytes == 2) {
                srccolor = (i == 0) ? (word & 0xffff) : (word >> 16);
            } else {
                srccolor = (word >> (i * 8)) & 0xff;
            }
            if (!chroma_enabled || srccolor != chromacolor) {
                uint32_t dstcolor = nv_geforce3_get_pixel(s, ch->s2d_img_dst,
                    ch->ifc_draw_offset, ch->ifc_x, ch->s2d_color_bytes);

                if (ch->ifc_color_bytes == 4 && ch->s2d_color_bytes == 2) {
                    dstcolor = nv_geforce3_color_565_to_888(dstcolor);
                }
                nv_geforce3_pixel_operation(s, ch, ch->ifc_operation,
                    &dstcolor, &srccolor, ch->ifc_color_bytes,
                    ch->ifc_ofs_x + ch->ifc_x, ch->ifc_ofs_y + ch->ifc_y);
                if (ch->ifc_color_bytes == 4 && ch->s2d_color_bytes == 2) {
                    dstcolor = nv_geforce3_color_888_to_565(dstcolor);
                }
                nv_geforce3_put_pixel(s, ch, ch->ifc_draw_offset, ch->ifc_x,
                                      dstcolor);
            }
        }
        ch->ifc_x++;
        if (ch->ifc_x >= ch->ifc_src_width) {
            ch->ifc_draw_offset += ch->s2d_pitch_dst;
            ch->ifc_x = 0;
            ch->ifc_y++;
        }
    }
}

static void nv_geforce3_iifc(NVGeForce3State *s, NVGeForce3Channel *ch)
{
    int16_t dx = ch->iifc_yx & 0xFFFF;
    int16_t dy = ch->iifc_yx >> 16;
    int16_t clipx0 = ch->clip_x - dx, clipy0 = ch->clip_y - dy;
    int16_t clipx1 = clipx0 + ch->clip_width;
    int16_t clipy1 = clipy0 + ch->clip_height;
    uint32_t swidth = ch->iifc_shw & 0xFFFF;
    uint32_t dwidth = ch->iifc_dhw & 0xFFFF;
    uint32_t height = ch->iifc_dhw >> 16;
    uint32_t pitch = ch->s2d_pitch_dst;
    uint32_t draw_offset = ch->s2d_ofs_dst + dy * pitch +
                           dx * ch->s2d_color_bytes;
    uint32_t symbol_index = 0;
    uint16_t x, y;

    for (y = 0; y < height; y++) {
        for (x = 0; x < dwidth; x++) {
            if (x >= clipx0 && x < clipx1 && y >= clipy0 && y < clipy1) {
                uint8_t symbol;
                uint32_t dstcolor;

                if (ch->iifc_bpp4) {
                    uint32_t word_offset = symbol_index / 8;
                    uint32_t symbol_offset = ((symbol_index % 8) ^ 1) * 4;

                    symbol = (ch->iifc_words[word_offset] >> symbol_offset) &
                             0xF;
                } else {
                    uint32_t word_offset = symbol_index / 4;
                    uint32_t symbol_offset = (symbol_index % 4) * 8;

                    symbol = (ch->iifc_words[word_offset] >> symbol_offset) &
                             0xFF;
                }
                dstcolor = nv_geforce3_get_pixel(s, ch->s2d_img_dst,
                    draw_offset, x, ch->s2d_color_bytes);
                if (ch->iifc_color_bytes == 4) {
                    uint32_t srccolor = nv_geforce3_dma_read32(s,
                        ch->iifc_palette, ch->iifc_palette_ofs + symbol * 4);

                    if (ch->s2d_color_bytes == 2) {
                        dstcolor = nv_geforce3_color_565_to_888(dstcolor);
                    }
                    nv_geforce3_pixel_operation(s, ch, ch->iifc_operation,
                        &dstcolor, &srccolor, 4, dx + x, dy + y);
                    if (ch->s2d_color_bytes == 2) {
                        dstcolor = nv_geforce3_color_888_to_565(dstcolor);
                    }
                } else if (ch->iifc_color_bytes == 2) {
                    uint32_t srccolor = nv_geforce3_dma_read16(s,
                        ch->iifc_palette, ch->iifc_palette_ofs + symbol * 2);

                    nv_geforce3_pixel_operation(s, ch, ch->iifc_operation,
                        &dstcolor, &srccolor, 2, dx + x, dy + y);
                }
                nv_geforce3_put_pixel(s, ch, draw_offset, x, dstcolor);
            }
            symbol_index++;
        }
        symbol_index += swidth - dwidth;
        draw_offset += pitch;
    }
}

static void nv_geforce3_sifc(NVGeForce3State *s, NVGeForce3Channel *ch)
{
    uint16_t dx = ch->sifc_clip_yx & 0xFFFF;
    uint16_t dy = ch->sifc_clip_yx >> 16;
    uint32_t dsdx = (uint32_t)(UINT64_C(1099511627776) / ch->sifc_dxds);
    uint32_t dtdy = (uint32_t)(UINT64_C(1099511627776) / ch->sifc_dydt);
    uint32_t swidth = ch->sifc_shw & 0xFFFF;
    uint32_t dwidth = ch->sifc_clip_hw & 0xFFFF;
    uint32_t height = ch->sifc_clip_hw >> 16;
    uint32_t pitch = ch->s2d_pitch_dst;
    uint32_t draw_offset = ch->s2d_ofs_dst + dy * pitch +
                           dx * ch->s2d_color_bytes;
    int32_t sx0 = ((ch->sifc_syx & 0xFFFF) << 16) - (dx << 20) - 0x80000;
    int32_t sy = (ch->sifc_syx & 0xFFFF0000) - (dy << 20) - 0x80000;
    uint32_t symbol_offset_y = 0;
    uint16_t x, y;

    if (sx0 < 0) {
        sx0 = 0;
    }
    if (sy < 0) {
        sy = 0;
    }
    for (y = 0; y < height; y++) {
        uint32_t sx = sx0;

        for (x = 0; x < dwidth; x++) {
            uint32_t dstcolor = nv_geforce3_get_pixel(s, ch->s2d_img_dst,
                draw_offset, x, ch->s2d_color_bytes);
            uint32_t srccolor;
            uint32_t symbol_offset = symbol_offset_y + (sx >> 20);

            if (ch->sifc_color_bytes == 4) {
                srccolor = ch->sifc_words[symbol_offset];
            } else if (ch->sifc_color_bytes == 2) {
                uint16_t *w16 = (uint16_t *)ch->sifc_words;

                srccolor = w16[symbol_offset];
            } else {
                uint8_t *w8 = (uint8_t *)ch->sifc_words;

                srccolor = w8[symbol_offset];
            }
            if (ch->sifc_color_bytes == 4 && ch->s2d_color_bytes == 2) {
                dstcolor = nv_geforce3_color_565_to_888(dstcolor);
            }
            nv_geforce3_pixel_operation(s, ch, ch->sifc_operation, &dstcolor,
                &srccolor, ch->sifc_color_bytes, dx + x, dy + y);
            if (ch->sifc_color_bytes == 4 && ch->s2d_color_bytes == 2) {
                dstcolor = nv_geforce3_color_888_to_565(dstcolor);
            }
            nv_geforce3_put_pixel(s, ch, draw_offset, x, dstcolor);
            sx += dsdx;
        }
        sy += dtdy;
        symbol_offset_y = (sy >> 20) * swidth;
        draw_offset += pitch;
    }
}

static void nv_geforce3_copyarea(NVGeForce3State *s, NVGeForce3Channel *ch)
{
    uint16_t sx = ch->blit_syx & 0xFFFF, sy = ch->blit_syx >> 16;
    uint16_t dx = ch->blit_dyx & 0xFFFF, dy = ch->blit_dyx >> 16;
    uint16_t width = ch->blit_hw & 0xFFFF, height = ch->blit_hw >> 16;
    uint32_t spitch = ch->s2d_pitch_src, dpitch = ch->s2d_pitch_dst;
    uint32_t src_offset = ch->s2d_ofs_src, draw_offset = ch->s2d_ofs_dst;
    bool xdir = dx > sx, ydir = dy > sy;
    uint32_t chromacolor = 0;
    bool chroma_enabled = false;
    uint16_t x, y;

    src_offset += (sy + ydir * (height - 1)) * spitch +
                 sx * ch->s2d_color_bytes;
    draw_offset += (dy + ydir * (height - 1)) * dpitch +
                  dx * ch->s2d_color_bytes;
    if (ch->blit_color_key_enable) {
        if (ch->s2d_color_bytes == 4) {
            chromacolor = ch->chroma_color & 0x00FFFFFF;
            chroma_enabled = (ch->chroma_color & 0xFF000000) != 0;
        } else if (ch->s2d_color_bytes == 2) {
            chromacolor = ch->chroma_color & 0x0000FFFF;
            chroma_enabled = (ch->chroma_color & 0xFFFF0000) != 0;
        } else {
            chromacolor = ch->chroma_color & 0x000000FF;
            chroma_enabled = (ch->chroma_color & 0xFFFFFF00) != 0;
        }
    }
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            uint16_t xa = xdir ? width - x - 1 : x;
            uint32_t srccolor = nv_geforce3_get_pixel(s, ch->s2d_img_src,
                src_offset, xa, ch->s2d_color_bytes);

            if (!chroma_enabled || srccolor != chromacolor) {
                uint32_t dstcolor = nv_geforce3_get_pixel(s, ch->s2d_img_dst,
                    draw_offset, xa, ch->s2d_color_bytes);

                nv_geforce3_pixel_operation(s, ch, ch->blit_operation,
                    &dstcolor, &srccolor, ch->s2d_color_bytes, dx + x,
                    dy + y);
                nv_geforce3_put_pixel(s, ch, draw_offset, xa, dstcolor);
            }
        }
        src_offset += spitch * (1 - 2 * ydir);
        draw_offset += dpitch * (1 - 2 * ydir);
    }
}

void nv_geforce3_execute_m2mf(NVGeForce3State *s, NVGeForce3Channel *ch,
                              uint32_t subc, uint32_t method, uint32_t param)
{
    if (method == 0x061) {
        ch->m2mf_src = param;
    } else if (method == 0x062) {
        ch->m2mf_dst = param;
    } else if (method == 0x0c3) {
        ch->m2mf_src_offset = param;
    } else if (method == 0x0c4) {
        ch->m2mf_dst_offset = param;
    } else if (method == 0x0c5) {
        ch->m2mf_src_pitch = param;
    } else if (method == 0x0c6) {
        ch->m2mf_dst_pitch = param;
    } else if (method == 0x0c7) {
        ch->m2mf_line_length = param;
    } else if (method == 0x0c8) {
        ch->m2mf_line_count = param;
    } else if (method == 0x0c9) {
        ch->m2mf_format = param;
    } else if (method == 0x0ca) {
        uint32_t src_offset = ch->m2mf_src_offset;
        uint32_t dst_offset = ch->m2mf_dst_offset;
        uint16_t y;

        ch->m2mf_buffer_notify = param;
        for (y = 0; y < ch->m2mf_line_count; y++) {
            nv_geforce3_dma_copy(s, ch->m2mf_dst, dst_offset, ch->m2mf_src,
                                 src_offset, ch->m2mf_line_length);
            src_offset += ch->m2mf_src_pitch;
            dst_offset += ch->m2mf_dst_pitch;
        }
        if ((nv_geforce3_ramin_read32(s, ch->schs[subc].notifier) & 0xFF) !=
            0x30) {
            nv_geforce3_dma_write64(s, ch->schs[subc].notifier, 0x10,
                                    nv_geforce3_get_current_time(s));
            nv_geforce3_dma_write32(s, ch->schs[subc].notifier, 0x18, 0);
            nv_geforce3_dma_write32(s, ch->schs[subc].notifier, 0x1C, 0);
        }
    }
}

void nv_geforce3_execute_rop(NVGeForce3Channel *ch, uint32_t method,
                             uint32_t param)
{
    if (method == 0x0c0) {
        ch->rop = (uint8_t)param;
    }
}

void nv_geforce3_execute_patt(NVGeForce3Channel *ch, uint32_t method,
                              uint32_t param)
{
    if (method == 0x0c2) {
        ch->patt_shape = param;
    } else if (method == 0x0c3) {
        ch->patt_type_color = param == 2;
    } else if (method == 0x0c4) {
        ch->patt_bg_color = param;
    } else if (method == 0x0c5) {
        ch->patt_fg_color = param;
    } else if (method == 0x0c6 || method == 0x0c7) {
        uint32_t i;

        for (i = 0; i < 32; i++) {
            ch->patt_data_mono[i + (method & 1) * 32] =
                ((1u << (i ^ 7)) & param) != 0;
        }
    } else if (method >= 0x100 && method < 0x110) {
        uint32_t i = (method - 0x100) * 4;

        ch->patt_data_color[i] = param & 0xFF;
        ch->patt_data_color[i + 1] = (param >> 8) & 0xFF;
        ch->patt_data_color[i + 2] = (param >> 16) & 0xFF;
        ch->patt_data_color[i + 3] = param >> 24;
    } else if (method >= 0x140 && method < 0x160) {
        uint32_t i = (method - 0x140) * 2;

        ch->patt_data_color[i] = param & 0xFFFF;
        ch->patt_data_color[i + 1] = param >> 16;
    } else if (method >= 0x1c0 && method < 0x200) {
        ch->patt_data_color[method - 0x1c0] = param;
    }
}

void nv_geforce3_execute_gdi(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t cls, uint32_t method, uint32_t param)
{
    if (method == 0x0bf) {
        ch->gdi_operation = param;
    } else if (method == 0x0c0) {
        ch->gdi_color_fmt = param;
    } else if (method == 0x0c1) {
        ch->gdi_mono_fmt = param;
    } else if (method == 0x0ff) {
        ch->gdi_rect_color = param;
    } else if (method >= 0x100 && method < 0x140) {
        if (method & 1) {
            ch->gdi_rect_wh = param;
            nv_geforce3_gdi_fillrect(s, ch, false);
        } else {
            ch->gdi_rect_xy = param;
        }
    } else if (method == 0x17d) {
        ch->gdi_clip_yx0 = param;
    } else if (method == 0x17e) {
        ch->gdi_clip_yx1 = param;
    } else if (method == 0x17f) {
        ch->gdi_rect_color = param;
    } else if (method >= 0x180 && method < 0x1c0) {
        if (method & 1) {
            ch->gdi_rect_yx1 = param;
            nv_geforce3_gdi_fillrect(s, ch, true);
        } else {
            ch->gdi_rect_yx0 = param;
        }
    } else if ((method == 0x1fb && cls == 0x004a) ||
               (method == 0x2fb && cls == 0x004b)) {
        ch->gdi_clip_yx0 = param;
    } else if ((method == 0x1fc && cls == 0x004a) ||
               (method == 0x2fc && cls == 0x004b)) {
        ch->gdi_clip_yx1 = param;
    } else if ((method == 0x1fd && cls == 0x004a) ||
               (method == 0x2fd && cls == 0x004b)) {
        ch->gdi_fg_color = param;
    } else if ((method == 0x1fe && cls == 0x004a) ||
               (method == 0x2fe && cls == 0x004b)) {
        ch->gdi_image_swh = param;
    } else if ((method == 0x1ff && cls == 0x004a) ||
               (method == 0x2ff && cls == 0x004b)) {
        uint32_t width = ch->gdi_image_swh & 0xFFFF;
        uint32_t height = ch->gdi_image_swh >> 16;
        uint32_t word_count = QEMU_ALIGN_UP(width * height, 32) >> 5;

        ch->gdi_image_xy = param;
        g_free(ch->gdi_words);
        ch->gdi_words_ptr = 0;
        ch->gdi_words_left = word_count;
        ch->gdi_words = g_new0(uint32_t, word_count);
    } else if ((method >= 0x200 && method < 0x280 && cls == 0x004a) ||
               (method >= 0x300 && method < 0x380 && cls == 0x004b)) {
        ch->gdi_words[ch->gdi_words_ptr++] = param;
        ch->gdi_words_left--;
        if (!ch->gdi_words_left) {
            nv_geforce3_gdi_blit(s, ch, 0);
            g_free(ch->gdi_words);
            ch->gdi_words = NULL;
        }
    } else if ((method == 0x2f9 && cls == 0x004a) ||
               (method == 0x4f9 && cls == 0x004b)) {
        ch->gdi_clip_yx0 = param;
    } else if ((method == 0x2fa && cls == 0x004a) ||
               (method == 0x4fa && cls == 0x004b)) {
        ch->gdi_clip_yx1 = param;
    } else if ((method == 0x2fb && cls == 0x004a) ||
               (method == 0x4fb && cls == 0x004b)) {
        ch->gdi_bg_color = param;
    } else if ((method == 0x2fc && cls == 0x004a) ||
               (method == 0x4fc && cls == 0x004b)) {
        ch->gdi_fg_color = param;
    } else if ((method == 0x2fd && cls == 0x004a) ||
               (method == 0x4fd && cls == 0x004b)) {
        ch->gdi_image_swh = param;
    } else if ((method == 0x2fe && cls == 0x004a) ||
               (method == 0x4fe && cls == 0x004b)) {
        ch->gdi_image_dwh = param;
    } else if ((method == 0x2ff && cls == 0x004a) ||
               (method == 0x4ff && cls == 0x004b)) {
        uint32_t width = ch->gdi_image_swh & 0xFFFF;
        uint32_t height = ch->gdi_image_swh >> 16;
        uint32_t word_count = QEMU_ALIGN_UP(width * height, 32) >> 5;

        ch->gdi_image_xy = param;
        g_free(ch->gdi_words);
        ch->gdi_words_ptr = 0;
        ch->gdi_words_left = word_count;
        ch->gdi_words = g_new0(uint32_t, word_count);
    } else if ((method >= 0x300 && method < 0x380 && cls == 0x004a) ||
               (method >= 0x500 && method < 0x580 && cls == 0x004b)) {
        ch->gdi_words[ch->gdi_words_ptr++] = param;
        ch->gdi_words_left--;
        if (!ch->gdi_words_left) {
            nv_geforce3_gdi_blit(s, ch, 1);
            g_free(ch->gdi_words);
            ch->gdi_words = NULL;
        }
    } else if (method == 0x3fd) {
        ch->gdi_clip_yx0 = param;
    } else if (method == 0x3fe) {
        ch->gdi_clip_yx1 = param;
    } else if (method == 0x3ff) {
        ch->gdi_fg_color = param;
    }
}

void nv_geforce3_execute_swzsurf(NVGeForce3Channel *ch, uint32_t method,
                                 uint32_t param)
{
    if (method == 0x061) {
        ch->swzs_img_obj = param;
    } else if (method == 0x0c0) {
        uint32_t color_fmt = param & 0xffff;

        ch->swzs_fmt = param;
        ch->swzs_width = 1u << ((param >> 16) & 0xff);
        ch->swzs_height = 1u << (param >> 24);
        if (color_fmt == 1) {
            ch->swzs_color_bytes = 1;
        } else if (color_fmt == 2 || color_fmt == 4) {
            ch->swzs_color_bytes = 2;
        } else if (color_fmt == 0x6 || color_fmt == 0xA ||
                   color_fmt == 0xB) {
            ch->swzs_color_bytes = 4;
        } else {
            qemu_log_mask(LOG_UNIMP, "nv-geforce3: unknown swizzled surface "
                          "color format 0x%02x\n", color_fmt);
        }
    } else if (method == 0x0c1) {
        ch->swzs_ofs = param;
    }
}

void nv_geforce3_execute_chroma(NVGeForce3Channel *ch, uint32_t method,
                                uint32_t param)
{
    if (method == 0x0c0) {
        ch->chroma_color_fmt = param;
    } else if (method == 0x0c1) {
        ch->chroma_color = param;
    }
}

void nv_geforce3_execute_clip(NVGeForce3Channel *ch, uint32_t method,
                              uint32_t param)
{
    if (method == 0x0c0) {
        ch->clip_x = (uint16_t)param;
        ch->clip_y = param >> 16;
    } else if (method == 0x0c1) {
        ch->clip_width = (uint16_t)param;
        ch->clip_height = param >> 16;
    }
}

void nv_geforce3_execute_lin(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t method, uint32_t param)
{
    if (method == 0x0bf) {
        ch->lin_operation = param;
    } else if (method == 0x0c0) {
        ch->lin_color_fmt = param;
    } else if (method == 0x0c1) {
        ch->lin_color = param;
    } else if (method >= 0x100 && method < 0x120) {
        if (method & 1) {
            ch->lin_x1 = (int16_t)param;
            ch->lin_y1 = (int16_t)(param >> 16);
            nv_geforce3_lin(s, ch);
        } else {
            ch->lin_x0 = (int16_t)param;
            ch->lin_y0 = (int16_t)(param >> 16);
        }
    } else if (method >= 0x120 && method < 0x140) {
        switch (method & 3) {
        case 0:
            ch->lin_x0 = (int32_t)param;
            break;
        case 1:
            ch->lin_y0 = (int32_t)param;
            break;
        case 2:
            ch->lin_x1 = (int32_t)param;
            break;
        case 3:
            ch->lin_y1 = (int32_t)param;
            nv_geforce3_lin(s, ch);
            break;
        }
    } else if (method >= 0x140 && method < 0x160) {
        ch->lin_x0 = ch->lin_x1;
        ch->lin_y0 = ch->lin_y1;
        ch->lin_x1 = (int16_t)param;
        ch->lin_y1 = (int16_t)(param >> 16);
        nv_geforce3_lin(s, ch);
    } else if (method >= 0x160 && method < 0x180) {
        if (method & 1) {
            ch->lin_y0 = ch->lin_y1;
            ch->lin_y1 = (int32_t)param;
            nv_geforce3_lin(s, ch);
        } else {
            ch->lin_x0 = ch->lin_x1;
            ch->lin_x1 = (int32_t)param;
        }
    }
}

void nv_geforce3_execute_rect(NVGeForce3State *s, NVGeForce3Channel *ch,
                              uint32_t method, uint32_t param)
{
    if (method == 0x0bf) {
        ch->rect_operation = param;
    } else if (method == 0x0c0) {
        ch->rect_color_fmt = param;
    } else if (method == 0x0c1) {
        ch->rect_color = param;
    } else if (method >= 0x100 && method < 0x120) {
        if (method & 1) {
            ch->rect_hw = param;
            nv_geforce3_rect(s, ch);
        } else {
            ch->rect_yx = param;
        }
    }
}

void nv_geforce3_execute_imageblit(NVGeForce3State *s, NVGeForce3Channel *ch,
                                   uint32_t method, uint32_t param)
{
    if (method == 0x061) {
        ch->blit_color_key_enable =
            (nv_geforce3_ramin_read32(s, param) & 0xFF) != 0x30;
    } else if (method == 0x0bf) {
        ch->blit_operation = param;
    } else if (method == 0x0c0) {
        ch->blit_syx = param;
    } else if (method == 0x0c1) {
        ch->blit_dyx = param;
    } else if (method == 0x0c2) {
        ch->blit_hw = param;
        nv_geforce3_copyarea(s, ch);
    }
}

void nv_geforce3_execute_ifc(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t method, uint32_t param)
{
    if (method == 0x061) {
        ch->ifc_color_key_enable =
            (nv_geforce3_ramin_read32(s, param) & 0xFF) != 0x30;
    } else if (method == 0x062) {
        ch->ifc_clip_enable =
            (nv_geforce3_ramin_read32(s, param) & 0xFF) != 0x30;
    } else if (method == 0x0bf) {
        ch->ifc_operation = param;
    } else if (method == 0x0c0) {
        ch->ifc_color_fmt = param;
        nv_geforce3_update_color_bytes_ifc(ch);
        ch->ifc_pixels_per_word = 4 / ch->ifc_color_bytes;
    } else if (method == 0x0c1) {
        ch->ifc_x = 0;
        ch->ifc_y = 0;
        ch->ifc_ofs_x = param & 0xFFFF;
        ch->ifc_ofs_y = param >> 16;
        ch->ifc_draw_offset = ch->s2d_ofs_dst +
            ch->ifc_ofs_y * ch->s2d_pitch_dst +
            ch->ifc_ofs_x * ch->s2d_color_bytes;
    } else if (method == 0x0c2) {
        ch->ifc_dst_width = param & 0xFFFF;
        ch->ifc_dst_height = param >> 16;
        ch->ifc_clip_x0 = 0;
        ch->ifc_clip_y0 = 0;
        ch->ifc_clip_x1 = ch->ifc_dst_width;
        ch->ifc_clip_y1 = ch->ifc_dst_height;
        if (ch->ifc_clip_enable) {
            int32_t clipx0 = ch->clip_x - ch->ifc_ofs_x;
            int32_t clipy0 = ch->clip_y - ch->ifc_ofs_y;
            int32_t clipx1 = clipx0 + ch->clip_width;
            int32_t clipy1 = clipy0 + ch->clip_height;

            ch->ifc_clip_x0 = MAX((int32_t)ch->ifc_clip_x0, clipx0);
            ch->ifc_clip_y0 = MAX((int32_t)ch->ifc_clip_y0, clipy0);
            ch->ifc_clip_x1 = MIN((int32_t)ch->ifc_clip_x1, clipx1);
            ch->ifc_clip_y1 = MIN((int32_t)ch->ifc_clip_y1, clipy1);
        }
    } else if (method == 0x0c3) {
        ch->ifc_src_width = param & 0xFFFF;
        ch->ifc_src_height = param >> 16;
    } else if (method >= 0x100 && method < 0x800) {
        nv_geforce3_ifc(s, ch, param);
    }
}

void nv_geforce3_execute_surf2d(NVGeForce3State *s, NVGeForce3Channel *ch,
                                uint32_t method, uint32_t param)
{
    ch->s2d_locked = true;
    if (method == 0x061) {
        ch->s2d_img_src = param;
    } else if (method == 0x062) {
        ch->s2d_img_dst = param;
    } else if (method == 0x0c0) {
        uint32_t prev = ch->s2d_color_bytes;

        ch->s2d_color_fmt = param;
        nv_geforce3_update_color_bytes_s2d(ch);
        if (ch->s2d_color_bytes != prev &&
            (ch->s2d_color_bytes == 1 || prev == 1)) {
            nv_geforce3_update_color_bytes_ifc(ch);
            nv_geforce3_update_color_bytes_sifc(ch);
            nv_geforce3_update_color_bytes_tfc(ch);
        }
    } else if (method == 0x0c1) {
        ch->s2d_pitch_src = param & 0xFFFF;
        ch->s2d_pitch_dst = param >> 16;
    } else if (method == 0x0c2) {
        ch->s2d_ofs_src = param;
    } else if (method == 0x0c3) {
        ch->s2d_ofs_dst = param;
    }
}

void nv_geforce3_update_color_bytes_s2d(NVGeForce3Channel *ch)
{
    if (ch->s2d_color_fmt == 0x1) {
        ch->s2d_color_bytes = 1;
    } else if (ch->s2d_color_fmt == 0x2 || ch->s2d_color_fmt == 0x4 ||
               ch->s2d_color_fmt == 0x5) {
        ch->s2d_color_bytes = 2;
    } else if (ch->s2d_color_fmt == 0x6 || ch->s2d_color_fmt == 0x7 ||
               ch->s2d_color_fmt == 0xA || ch->s2d_color_fmt == 0xB) {
        ch->s2d_color_bytes = 4;
    } else {
        qemu_log_mask(LOG_UNIMP, "nv-geforce3: unknown 2d surface color "
                      "format 0x%02x\n", ch->s2d_color_fmt);
    }
}

static void nv_geforce3_update_color_bytes(uint32_t s2d_color_fmt,
                                           uint32_t color_fmt,
                                           uint32_t *color_bytes)
{
    if (s2d_color_fmt == 1) {
        *color_bytes = 1;
    } else if (color_fmt == 1 || color_fmt == 2 || color_fmt == 3) {
        *color_bytes = 2;
    } else if (color_fmt == 4 || color_fmt == 5) {
        *color_bytes = 4;
    } else {
        qemu_log_mask(LOG_UNIMP,
                      "nv-geforce3: unknown color format 0x%02x\n",
                      color_fmt);
    }
}

void nv_geforce3_update_color_bytes_ifc(NVGeForce3Channel *ch)
{
    nv_geforce3_update_color_bytes(ch->s2d_color_fmt, ch->ifc_color_fmt,
                                   &ch->ifc_color_bytes);
}

void nv_geforce3_update_color_bytes_sifc(NVGeForce3Channel *ch)
{
    nv_geforce3_update_color_bytes(ch->s2d_color_fmt, ch->sifc_color_fmt,
                                   &ch->sifc_color_bytes);
}

void nv_geforce3_update_color_bytes_tfc(NVGeForce3Channel *ch)
{
    nv_geforce3_update_color_bytes(ch->s2d_color_fmt, ch->tfc_color_fmt,
                                   &ch->tfc_color_bytes);
}

void nv_geforce3_update_color_bytes_iifc(NVGeForce3Channel *ch)
{
    nv_geforce3_update_color_bytes(0, ch->iifc_color_fmt,
                                   &ch->iifc_color_bytes);
}

void nv_geforce3_execute_iifc(NVGeForce3State *s, NVGeForce3Channel *ch,
                              uint32_t method, uint32_t param)
{
    if (method == 0x061) {
        ch->iifc_palette = param;
    } else if (method == 0x0f9) {
        ch->iifc_operation = param;
    } else if (method == 0x0fa) {
        ch->iifc_color_fmt = param;
        nv_geforce3_update_color_bytes_iifc(ch);
    } else if (method == 0x0fb) {
        ch->iifc_bpp4 = param;
    } else if (method == 0x0fc) {
        ch->iifc_palette_ofs = param;
    } else if (method == 0x0fd) {
        ch->iifc_yx = param;
    } else if (method == 0x0fe) {
        ch->iifc_dhw = param;
    } else if (method == 0x0ff) {
        uint32_t width, height, word_count;

        ch->iifc_shw = param;
        width = ch->iifc_shw & 0xFFFF;
        height = ch->iifc_shw >> 16;
        word_count = QEMU_ALIGN_UP(width * height *
                                   (ch->iifc_bpp4 ? 4 : 8), 32) >> 5;
        g_free(ch->iifc_words);
        ch->iifc_words_ptr = 0;
        ch->iifc_words_left = word_count;
        ch->iifc_words = g_new0(uint32_t, word_count);
    } else if (method >= 0x100 && method < 0x800) {
        ch->iifc_words[ch->iifc_words_ptr++] = param;
        ch->iifc_words_left--;
        if (!ch->iifc_words_left) {
            nv_geforce3_iifc(s, ch);
            g_free(ch->iifc_words);
            ch->iifc_words = NULL;
        }
    }
}

void nv_geforce3_execute_sifc(NVGeForce3State *s, NVGeForce3Channel *ch,
                              uint32_t method, uint32_t param)
{
    if (method == 0x0bf) {
        ch->sifc_operation = param;
    } else if (method == 0x0c0) {
        ch->sifc_color_fmt = param;
        nv_geforce3_update_color_bytes_sifc(ch);
    } else if (method == 0x0c1) {
        ch->sifc_shw = param;
    } else if (method == 0x0c2) {
        ch->sifc_dxds = param;
    } else if (method == 0x0c3) {
        ch->sifc_dydt = param;
    } else if (method == 0x0c4) {
        ch->sifc_clip_yx = param;
    } else if (method == 0x0c5) {
        ch->sifc_clip_hw = param;
    } else if (method == 0x0c6) {
        uint32_t width, height, word_count;

        ch->sifc_syx = param;
        width = ch->sifc_shw & 0xFFFF;
        height = ch->sifc_shw >> 16;
        word_count = QEMU_ALIGN_UP(width * height * ch->sifc_color_bytes,
                                   4) >> 2;
        g_free(ch->sifc_words);
        ch->sifc_words_ptr = 0;
        ch->sifc_words_left = word_count;
        ch->sifc_words = g_new0(uint32_t, word_count);
    } else if (method >= 0x100 && method < 0x800) {
        ch->sifc_words[ch->sifc_words_ptr++] = param;
        ch->sifc_words_left--;
        if (!ch->sifc_words_left) {
            nv_geforce3_sifc(s, ch);
            g_free(ch->sifc_words);
            ch->sifc_words = NULL;
        }
    }
}

void nv_geforce3_execute_beta(NVGeForce3Channel *ch, uint32_t method,
                              uint32_t param)
{
    if (method == 0x0c0) {
        ch->beta = param;
    }
}

static void nv_geforce3_tfc(NVGeForce3State *s, NVGeForce3Channel *ch)
{
    uint16_t dx = ch->tfc_yx & 0xFFFF, dy = ch->tfc_yx >> 16;
    int16_t clipx0 = (ch->tfc_clip_wx & 0xFFFF) - dx;
    int16_t clipy0 = (ch->tfc_clip_hy & 0xFFFF) - dy;
    int16_t clipx1 = clipx0 + (ch->tfc_clip_wx >> 16);
    int16_t clipy1 = clipy0 + (ch->tfc_clip_hy >> 16);
    uint32_t width = ch->tfc_hw & 0xFFFF, height = ch->tfc_hw >> 16;
    uint32_t word_offset = 0;
    uint16_t x, y;

    if (ch->tfc_swizzled) {
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                if (x >= clipx0 && x < clipx1 && y >= clipy0 &&
                    y < clipy1) {
                    uint32_t srccolor;

                    if (ch->tfc_color_bytes == 4) {
                        srccolor = ch->tfc_words[word_offset];
                    } else if (ch->tfc_color_bytes == 2) {
                        uint16_t *w16 = (uint16_t *)ch->tfc_words;

                        srccolor = w16[word_offset];
                    } else {
                        uint8_t *w8 = (uint8_t *)ch->tfc_words;

                        srccolor = w8[word_offset];
                    }
                    nv_geforce3_put_pixel_swzs(s, ch, ch->swzs_ofs +
                        nv_geforce3_swizzle(x + dx, y + dy, 0,
                            ch->swzs_width, ch->swzs_height, 1) *
                        ch->swzs_color_bytes, srccolor);
                }
                word_offset++;
            }
        }
    } else {
        uint32_t pitch = ch->s2d_pitch_dst;
        uint32_t draw_offset = ch->s2d_ofs_dst + dy * pitch +
                               dx * ch->s2d_color_bytes;

        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                if (x >= clipx0 && x < clipx1 && y >= clipy0 &&
                    y < clipy1) {
                    uint32_t srccolor;

                    if (ch->tfc_color_bytes == 4) {
                        srccolor = ch->tfc_words[word_offset];
                    } else if (ch->tfc_color_bytes == 2) {
                        uint16_t *w16 = (uint16_t *)ch->tfc_words;

                        srccolor = w16[word_offset];
                    } else {
                        uint8_t *w8 = (uint8_t *)ch->tfc_words;

                        srccolor = w8[word_offset];
                    }
                    nv_geforce3_put_pixel(s, ch, draw_offset, x, srccolor);
                }
                word_offset++;
            }
            draw_offset += pitch;
        }
    }
}

void nv_geforce3_execute_tfc(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t method, uint32_t param)
{
    if (method == 0x061) {
        uint8_t cls8 = (uint8_t)nv_geforce3_ramin_read32(s, param);

        ch->tfc_swizzled = cls8 == 0x52 || cls8 == 0x9e;
    } else if (method == 0x0c0) {
        ch->tfc_color_fmt = param;
        nv_geforce3_update_color_bytes_tfc(ch);
    } else if (method == 0x0c1) {
        ch->tfc_yx = param;
    } else if (method == 0x0c2) {
        ch->tfc_hw = param;
        ch->tfc_upload = param == 0x01000100 && ch->tfc_yx == 0 &&
            ch->tfc_color_fmt == 4 && ch->s2d_color_fmt == 0xA &&
            ch->s2d_pitch_src == 0x0400 && ch->s2d_pitch_dst == 0x0400;
        if (ch->tfc_upload) {
            ch->tfc_upload_offset = ch->s2d_ofs_dst;
        } else {
            uint32_t width = ch->tfc_hw & 0xFFFF;
            uint32_t height = ch->tfc_hw >> 16;
            uint32_t word_count = QEMU_ALIGN_UP(width * height *
                                                ch->tfc_color_bytes, 4) >> 2;

            g_free(ch->tfc_words);
            ch->tfc_words_ptr = 0;
            ch->tfc_words_left = word_count;
            ch->tfc_words = g_new0(uint32_t, word_count);
        }
    } else if (method == 0x0c3) {
        ch->tfc_clip_wx = param;
    } else if (method == 0x0c4) {
        ch->tfc_clip_hy = param;
    } else if (method >= 0x100 && method < 0x800) {
        if (ch->tfc_upload) {
            nv_geforce3_dma_write32(s, ch->s2d_img_dst, ch->tfc_upload_offset,
                                    param);
            ch->tfc_upload_offset += 4;
        } else if (ch->tfc_words != NULL) {
            ch->tfc_words[ch->tfc_words_ptr++] = param;
            ch->tfc_words_left--;
            if (!ch->tfc_words_left) {
                nv_geforce3_tfc(s, ch);
                g_free(ch->tfc_words);
                ch->tfc_words = NULL;
            }
        }
    }
}

static void nv_geforce3_sifm(NVGeForce3State *s, NVGeForce3Channel *ch,
                             bool swizzled)
{
    uint16_t dx = ch->sifm_dyx & 0xFFFF, dy = ch->sifm_dyx >> 16;
    uint16_t dwidth = ch->sifm_dhw & 0xFFFF, dheight = ch->sifm_dhw >> 16;
    uint32_t spitch = ch->sifm_sfmt & 0xFFFF;
    uint16_t x, y;

    if (ch->sifm_dudx == 0x00100000 && ch->sifm_dvdy == 0x00100000) {
        uint16_t sx = (ch->sifm_syx & 0xFFFF) >> 4;
        uint16_t sy = (ch->sifm_syx >> 16) >> 4;
        uint32_t src_offset = ch->sifm_sofs + sy * spitch +
                              sx * ch->sifm_color_bytes;

        if (swizzled) {
            for (y = 0; y < dheight; y++) {
                for (x = 0; x < dwidth; x++) {
                    uint32_t srccolor = nv_geforce3_get_pixel(s,
                        ch->sifm_src, src_offset, x, ch->sifm_color_bytes);

                    if (ch->sifm_color_bytes == 2 &&
                        ch->swzs_color_bytes == 4) {
                        srccolor = nv_geforce3_color_565_to_888(srccolor);
                    }
                    nv_geforce3_put_pixel_swzs(s, ch, ch->swzs_ofs +
                        nv_geforce3_swizzle(x + dx, y + dy, 0,
                            ch->swzs_width, ch->swzs_height, 1) *
                        ch->swzs_color_bytes, srccolor);
                }
                src_offset += spitch;
            }
        } else {
            uint32_t dpitch = ch->s2d_pitch_dst;
            uint32_t draw_offset = ch->s2d_ofs_dst + dy * dpitch +
                                   dx * ch->s2d_color_bytes;

            for (y = 0; y < dheight; y++) {
                for (x = 0; x < dwidth; x++) {
                    uint32_t dstcolor = nv_geforce3_get_pixel(s,
                        ch->s2d_img_dst, draw_offset, x, ch->s2d_color_bytes);
                    uint32_t srccolor = nv_geforce3_get_pixel(s,
                        ch->sifm_src, src_offset, x, ch->sifm_color_bytes);

                    if (ch->sifm_color_fmt == 4) {
                        srccolor |= 0xFF000000;
                    }
                    nv_geforce3_pixel_operation(s, ch, ch->sifm_operation,
                        &dstcolor, &srccolor, ch->s2d_color_bytes, dx + x,
                        dy + y);
                    nv_geforce3_put_pixel(s, ch, draw_offset, x, dstcolor);
                }
                src_offset += spitch;
                draw_offset += dpitch;
            }
        }
    } else {
        int32_t sx0 = ((ch->sifm_syx & 0xFFFF) << 16) - 0x80000;
        int32_t sy = (ch->sifm_syx & 0xFFFF0000) +
                    ((int32_t)ch->sifm_dvdy < 0 ? 0x80000 : -0x80000);

        if (sx0 < 0) {
            sx0 = 0;
        }
        if (sy < 0) {
            sy = 0;
        }
        if (swizzled) {
            for (y = 0; y < dheight; y++) {
                uint32_t sx = sx0;
                uint32_t src_offset = ch->sifm_sofs + (sy >> 20) * spitch;

                for (x = 0; x < dwidth; x++) {
                    uint32_t srccolor = nv_geforce3_get_pixel(s,
                        ch->sifm_src, src_offset, sx >> 20,
                        ch->sifm_color_bytes);

                    if (ch->sifm_color_bytes == 2 &&
                        ch->swzs_color_bytes == 4) {
                        srccolor = nv_geforce3_color_565_to_888(srccolor);
                    }
                    nv_geforce3_put_pixel_swzs(s, ch, ch->swzs_ofs +
                        nv_geforce3_swizzle(x + dx, y + dy, 0,
                            ch->swzs_width, ch->swzs_height, 1) *
                        ch->swzs_color_bytes, srccolor);
                    sx += ch->sifm_dudx;
                }
                sy += ch->sifm_dvdy;
            }
        } else {
            uint32_t dpitch = ch->s2d_pitch_dst;
            uint32_t draw_offset = ch->s2d_ofs_dst + dy * dpitch +
                                   dx * ch->s2d_color_bytes;

            for (y = 0; y < dheight; y++) {
                uint32_t sx = sx0;
                uint32_t src_offset = ch->sifm_sofs + (sy >> 20) * spitch;

                for (x = 0; x < dwidth; x++) {
                    uint32_t dstcolor = nv_geforce3_get_pixel(s,
                        ch->s2d_img_dst, draw_offset, x, ch->s2d_color_bytes);
                    uint32_t srccolor = nv_geforce3_get_pixel(s,
                        ch->sifm_src, src_offset, sx >> 20,
                        ch->sifm_color_bytes);

                    if (ch->sifm_color_fmt == 4) {
                        srccolor |= 0xFF000000;
                    }
                    nv_geforce3_pixel_operation(s, ch, ch->sifm_operation,
                        &dstcolor, &srccolor, ch->s2d_color_bytes, dx + x,
                        dy + y);
                    nv_geforce3_put_pixel(s, ch, draw_offset, x, dstcolor);
                    sx += ch->sifm_dudx;
                }
                sy += ch->sifm_dvdy;
                draw_offset += dpitch;
            }
        }
    }
}

void nv_geforce3_execute_sifm(NVGeForce3State *s, NVGeForce3Channel *ch,
                              uint32_t cls, uint32_t method, uint32_t param)
{
    if (method == 0x061) {
        ch->sifm_src = param;
    } else if (method == 0x066) {
        uint8_t surf_cls8 = (uint8_t)nv_geforce3_ramin_read32(s, param);
        bool swizzled = surf_cls8 == 0x52 || surf_cls8 == 0x9e;

        if (cls == 0x0389) {
            ch->sifm_swizzled_0389 = swizzled;
        } else {
            ch->sifm_swizzled = swizzled;
        }
    } else if (method == 0x0c0) {
        ch->sifm_color_fmt = param;
        if (ch->sifm_color_fmt == 8) {
            ch->sifm_color_bytes = 1;
        } else if (ch->sifm_color_fmt == 1 || ch->sifm_color_fmt == 2 ||
                   ch->sifm_color_fmt == 7) {
            ch->sifm_color_bytes = 2;
        } else if (ch->sifm_color_fmt == 3 || ch->sifm_color_fmt == 4) {
            ch->sifm_color_bytes = 4;
        } else {
            qemu_log_mask(LOG_UNIMP,
                          "nv-geforce3: unknown sifm color format 0x%02x\n",
                          ch->sifm_color_fmt);
        }
    } else if (method == 0x0c1) {
        ch->sifm_operation = param;
    } else if (method == 0x0c4) {
        ch->sifm_dyx = param;
    } else if (method == 0x0c5) {
        ch->sifm_dhw = param;
    } else if (method == 0x0c6) {
        ch->sifm_dudx = param;
    } else if (method == 0x0c7) {
        ch->sifm_dvdy = param;
    } else if (method == 0x100) {
        ch->sifm_shw = param;
    } else if (method == 0x101) {
        ch->sifm_sfmt = param;
    } else if (method == 0x102) {
        ch->sifm_sofs = param;
    } else if (method == 0x103) {
        ch->sifm_syx = param;
        nv_geforce3_sifm(s, ch,
            cls == 0x0389 ? ch->sifm_swizzled_0389 : ch->sifm_swizzled);
    }
}

/* ---------------------------------------------------------------- */
/* Software-object save/restore on PFIFO subchannel object bind      */

void nv_geforce3_object_save(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t subc)
{
    uint32_t object = ch->schs[subc].object;
    uint8_t engine = ch->schs[subc].engine;
    uint32_t word0, word1;
    uint8_t cls8;

    if (engine != 0x01) {
        return;
    }
    if (object == 0) {
        qemu_log_mask(LOG_GUEST_ERROR, "nv-geforce3: object_save: empty "
                      "object\n");
        return;
    }
    word1 = nv_geforce3_ramin_read32(s, object + 0x4);
    word1 = (word1 & 0x0000FFFF) | (ch->schs[subc].notifier >> 4 << 16);
    word0 = nv_geforce3_ramin_read32(s, object);
    cls8 = (uint8_t)word0;
    if (cls8 == 0x4a || cls8 == 0x4b) {
        word0 = (word0 & 0xFFFC7FFF) | (ch->gdi_operation << 15);
        word1 = (word1 & 0xFFFFFFFC) | ch->gdi_mono_fmt;
        nv_geforce3_ramin_write32(s, object, word0);
    } else if (cls8 == 0x62) {
        nv_geforce3_ramin_write32(s, object + 0x8,
            (ch->s2d_img_src >> 4) | (ch->s2d_img_dst >> 4 << 16));
    } else if (cls8 == 0x64) {
        nv_geforce3_ramin_write32(s, object + 0x8, ch->iifc_palette >> 4);
        word0 = (word0 & 0xFFFC7FFF) | (ch->iifc_operation << 15);
        nv_geforce3_ramin_write32(s, object, word0);
        word1 = (word1 & 0xFFFF00FF) | ((ch->iifc_color_fmt + 9) << 8);
    }
    nv_geforce3_ramin_write32(s, object + 0x4, word1);
}

void nv_geforce3_object_load(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t subc)
{
    uint32_t object = ch->schs[subc].object;
    uint32_t word0, word1;
    uint8_t cls8;

    if (object == 0) {
        qemu_log_mask(LOG_GUEST_ERROR, "nv-geforce3: object_load: empty "
                      "object\n");
        return;
    }
    word1 = nv_geforce3_ramin_read32(s, object + 0x4);
    ch->schs[subc].notifier = word1 >> 16 << 4;
    word0 = nv_geforce3_ramin_read32(s, object);
    cls8 = (uint8_t)word0;
    if (cls8 == 0x48) {
        /* compatibility hack for old X11 drivers */
        if (!ch->s2d_locked) {
            uint32_t srcdst = nv_geforce3_ramin_read32(s, object + 0x8);

            ch->s2d_img_src = (srcdst & 0xFFFF) << 4;
            ch->s2d_img_dst = (srcdst >> 16) << 4;
            ch->s2d_color_fmt = s->graph_bpixel & 0xf;
            nv_geforce3_update_color_bytes_s2d(ch);
            ch->s2d_pitch_src = s->graph_pitch0 & 0xffff;
            ch->s2d_pitch_dst = ch->s2d_pitch_src;
            ch->s2d_ofs_src = s->graph_offset0;
            ch->s2d_ofs_dst = s->graph_offset0;
        }
    } else if (cls8 == 0x4a || cls8 == 0x4b) {
        ch->gdi_operation = (word0 >> 15) & 7;
        ch->gdi_mono_fmt = word1 & 3;
    } else if (cls8 == 0x62) {
        uint32_t srcdst = nv_geforce3_ramin_read32(s, object + 0x8);

        ch->s2d_img_src = (srcdst & 0xFFFF) << 4;
        ch->s2d_img_dst = (srcdst >> 16) << 4;
    } else if (cls8 == 0x64) {
        ch->iifc_palette = nv_geforce3_ramin_read32(s, object + 0x8) << 4;
        ch->iifc_operation = (word0 >> 15) & 7;
        ch->iifc_color_fmt = ((word1 >> 8) & 0xFF) - 9;
        nv_geforce3_update_color_bytes_iifc(ch);
    } else if (cls8 == 0x96 || cls8 == 0x97) {
        nv_geforce3_execute_d3d(s, ch, word0 & 0xFFF, 0, 0);
    }
}
