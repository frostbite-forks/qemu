/*
 * NV20 PGRAPH Metal texture management.
 */
#include "qemu/fast-hash.h"
#include "hw/display/nv20/nv20_int.h"
#include "hw/display/nv20/pgraph/swizzle.h"
#include "hw/display/nv20/pgraph/s3tc.h"
#include "hw/display/nv20/pgraph/texture.h"
#include "debug.h"
#include "renderer.h"

static TextureBinding *generate_texture(const TextureShape s,
                                        const uint8_t *texture_data,
                                        const uint8_t *palette_data);
static void texture_binding_destroy(gpointer data);

static void apply_texture_parameters(PGRAPHMetalState *r,
                                     TextureBinding *binding,
                                     const ColorFormatInfo *f,
                                     unsigned int dimensionality,
                                     unsigned int filter,
                                     unsigned int address,
                                     bool is_bordered,
                                     uint32_t border_color,
                                     uint32_t max_anisotropy)
{
    unsigned int min_filter = GET_MASK(filter, NV_PGRAPH_TEXFILTER0_MIN);
    unsigned int mag_filter = GET_MASK(filter, NV_PGRAPH_TEXFILTER0_MAG);
    unsigned int lod_bias = GET_MASK(filter, NV_PGRAPH_TEXFILTER0_MIPMAP_LOD_BIAS);
    unsigned int addru = GET_MASK(address, NV_PGRAPH_TEXADDRESS0_ADDRU);
    unsigned int addrv = GET_MASK(address, NV_PGRAPH_TEXADDRESS0_ADDRV);
    unsigned int addrp = GET_MASK(address, NV_PGRAPH_TEXADDRESS0_ADDRP);

    if (binding->mtl_sampler) {
        nv20_mtl_sampler_release(binding->mtl_sampler);
    }
    binding->min_filter = min_filter;
    binding->mag_filter = mag_filter;
    binding->lod_bias = lod_bias;
    binding->addru = addru;
    binding->addrv = addrv;
    binding->addrp = addrp;
    binding->mtl_sampler = nv20_mtl_sampler_get(
        r->backend, min_filter, mag_filter, addru, addrv, addrp,
        pgraph_convert_lod_bias_to_float(lod_bias), max_anisotropy,
        border_color, is_bordered);
    (void)f; (void)dimensionality;
}

static bool texture_cache_compare(Lru *lru, LruNode *node, const void *key)
{
    TextureLruNode *tnode = container_of(node, TextureLruNode, node);
    return memcmp(&tnode->key, key, sizeof(TextureKey));
}

static void texture_cache_init(Lru *lru, LruNode *node, const void *key)
{
    TextureLruNode *tnode = container_of(node, TextureLruNode, node);
    memcpy(&tnode->key, key, sizeof(TextureKey));
    tnode->binding = NULL;
    tnode->possibly_dirty = false;
}

struct tex_dirty_ctx { hwaddr addr, end; };

static void mark_dirty_visitor(Lru *lru, LruNode *node, void *opaque)
{
    struct tex_dirty_ctx *test = opaque;
    TextureLruNode *tnode = container_of(node, TextureLruNode, node);
    if (!tnode->binding || tnode->possibly_dirty) {
        return;
    }
    hwaddr k_end = tnode->key.texture_vram_offset + tnode->key.texture_length - 1;
    tnode->possibly_dirty = !(test->addr > k_end ||
                              tnode->key.texture_vram_offset > test->end);
}

void pgraph_metal_mark_textures_possibly_dirty(NV20State *d,
                                               hwaddr addr, hwaddr size)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    hwaddr end = TARGET_PAGE_ALIGN(addr + size) - 1;
    addr &= TARGET_PAGE_MASK;
    struct tex_dirty_ctx test = { .addr = addr, .end = end };
    lru_visit_active(&r->texture_cache, mark_dirty_visitor, &test);
}

static TextureBinding *generate_texture(const TextureShape s,
                                        const uint8_t *texture_data,
                                        const uint8_t *palette_data)
{
    ColorFormatInfo f = kelvin_mtl_color_format_map[s.color_format];
    PGRAPHMetalState *r = g_nv20->pgraph.metal_renderer_state;

    unsigned width = s.width, height = s.height;
    if (!f.linear && s.border) {
        width = MAX(16, width * 2);
        height = MAX(16, height * 2);
    }

    NV20MtlTexture *tex = nv20_mtl_texture_create(
        r->backend, f.mtl_pixel_format, width, height,
        MAX(s.depth, 1), MAX(s.levels, 1), s.cubemap);

    if (f.linear) {
        uint8_t *converted = pgraph_convert_texture_data(
            s, texture_data, palette_data, width, height, 1,
            s.pitch, 0, NULL);
        nv20_mtl_texture_upload_level(r->backend, tex, 0, width, height, 1,
                                      s.pitch, converted ? converted : texture_data,
                                      f.bytes_per_pixel, true);
        g_free(converted);
    } else if (f.is_compressed) {
        enum S3TC_DECOMPRESS_FORMAT fmt = S3TC_DECOMPRESS_FORMAT_DXT1;
        if (f.mtl_pixel_format == 131) {
            fmt = S3TC_DECOMPRESS_FORMAT_DXT3;
        } else if (f.mtl_pixel_format == 132) {
            fmt = S3TC_DECOMPRESS_FORMAT_DXT5;
        }
        uint8_t *converted = s3tc_decompress_2d(fmt, texture_data, width, height);
        nv20_mtl_texture_upload_level(r->backend, tex, 0, width, height, 1,
                                      width * 4, converted, 4, true);
        g_free(converted);
    } else {
        nv20_mtl_texture_upload_swizzled(r->backend, tex, 0, width, height,
                                         s.pitch, texture_data,
                                         f.bytes_per_pixel);
    }

    TextureBinding *ret = g_malloc0(sizeof(*ret));
    ret->mtl_texture = tex;
    ret->mtl_target = s.cubemap ? 6 : s.dimensionality;
    ret->refcnt = 1;
    return ret;
}

static void texture_binding_destroy(gpointer data)
{
    TextureBinding *binding = data;
    if (!binding) {
        return;
    }
    if (--binding->refcnt > 0) {
        return;
    }
    if (binding->mtl_sampler) {
        nv20_mtl_sampler_release(binding->mtl_sampler);
    }
    nv20_mtl_texture_destroy(binding->mtl_texture);
    g_free(binding);
}

void pgraph_metal_bind_textures(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    for (int i = 0; i < NV20_MAX_TEXTURES; i++) {
        if (!pgraph_is_texture_enabled(pg, i)) {
            continue;
        }

        TextureShape state = pgraph_get_texture_shape(pg, i);
        hwaddr texture_vram_offset = pgraph_get_texture_phys_addr(pg, i);
        size_t length = pgraph_get_texture_length(pg, &state);

        TextureKey key = { .state = state,
                           .texture_vram_offset = texture_vram_offset,
                           .texture_length = length };
        uint64_t hash = fast_hash((uint8_t *)&key, sizeof(key));
        LruNode *found = lru_lookup(&r->texture_cache, hash, &key);
        TextureLruNode *node = container_of(found, TextureLruNode, node);

        if (!node->binding) {
            void *data = d->vram_ptr + texture_vram_offset;
            node->binding = generate_texture(state, data, NULL);
        }

        TextureBinding *binding = node->binding;
        binding->refcnt++;
        uint32_t filter = pgraph_reg_r(pg, NV_PGRAPH_TEXFILTER0 + i * 4);
        uint32_t address = pgraph_reg_r(pg, NV_PGRAPH_TEXADDRESS0 + i * 4);
        uint32_t border_color = pgraph_reg_r(pg, NV_PGRAPH_BORDERCOLOR0 + i * 4);
        apply_texture_parameters(r, binding,
                                 &kelvin_mtl_color_format_map[state.color_format],
                                 state.dimensionality, filter, address,
                                 state.border, border_color, 1);

        if (r->texture_binding[i]) {
            texture_binding_destroy(r->texture_binding[i]);
        }
        r->texture_binding[i] = binding;
        nv20_mtl_set_fragment_texture(r->backend, i, binding->mtl_texture,
                                      binding->mtl_sampler);
        pg->texture_dirty[i] = false;
    }
}

bool pgraph_metal_check_surface_to_texture_compatibility(
    const SurfaceBinding *surface, const TextureShape *shape)
{
    if (surface->width != shape->width || surface->height != shape->height) {
        return false;
    }
    return surface->color;
}

void pgraph_metal_render_surface_to_texture(NV20State *d,
                                            SurfaceBinding *surface,
                                            TextureBinding *texture,
                                            TextureShape *texture_shape,
                                            int texture_unit)
{
    (void)d; (void)surface; (void)texture; (void)texture_shape; (void)texture_unit;
}

void pgraph_metal_init_textures(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;
    const size_t cache_size = 50 * 1024;

    lru_init(&r->texture_cache);
    r->texture_cache_entries = g_malloc_n(cache_size, sizeof(TextureLruNode));
    for (int i = 0; i < cache_size; i++) {
        lru_add_free(&r->texture_cache, &r->texture_cache_entries[i].node);
    }
    r->texture_cache.init_node = texture_cache_init;
    r->texture_cache.compare_nodes = texture_cache_compare;
}

void pgraph_metal_finalize_textures(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;
    for (int i = 0; i < NV20_MAX_TEXTURES; i++) {
        texture_binding_destroy(r->texture_binding[i]);
        r->texture_binding[i] = NULL;
    }
    lru_flush(&r->texture_cache);
    g_free(r->texture_cache_entries);
    r->texture_cache_entries = NULL;
}
