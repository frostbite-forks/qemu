/*
 * NV20 PGRAPH Metal surface management.
 */
#include "hw/display/nv20/pgraph/pgraph.h"
#include "hw/display/nv20/nv20_pgraph_settings.h"
#include "hw/display/nv20/nv20_int.h"
#include "hw/display/nv20/pgraph/swizzle.h"
#include "debug.h"
#include "renderer.h"

static void surface_download(NV20State *d, SurfaceBinding *surface, bool force);
static void surface_get_dimensions(PGRAPHState *pg, unsigned int *width,
                                   unsigned int *height);

void pgraph_metal_set_surface_scale_factor(NV20State *d, unsigned int scale)
{
    g_config.display.quality.surface_scale = scale < 1 ? 1 : scale;
}

unsigned int pgraph_metal_get_surface_scale_factor(NV20State *d)
{
    return d->pgraph.surface_scale_factor;
}

void pgraph_metal_reload_surface_scale_factor(PGRAPHState *pg)
{
    int factor = g_config.display.quality.surface_scale;
    pg->surface_scale_factor = factor < 1 ? 1 : factor;
}

void pgraph_metal_set_surface_dirty(PGRAPHState *pg, bool color, bool zeta)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    color = color && pgraph_color_write_enabled(pg);
    zeta = zeta && pgraph_zeta_write_enabled(pg);
    pg->surface_color.draw_dirty |= color;
    pg->surface_zeta.draw_dirty |= zeta;

    if (r->color_binding) {
        r->color_binding->draw_dirty |= color;
        r->color_binding->frame_time = pg->frame_time;
        r->color_binding->cleared = false;
    }
    if (r->zeta_binding) {
        r->zeta_binding->draw_dirty |= zeta;
        r->zeta_binding->frame_time = pg->frame_time;
        r->zeta_binding->cleared = false;
    }
}

SurfaceBinding *pgraph_metal_surface_get(NV20State *d, hwaddr addr)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    SurfaceBinding *surface;

    QTAILQ_FOREACH(surface, &r->surfaces, entry) {
        if (surface->vram_addr == addr) {
            return surface;
        }
    }
    return NULL;
}

SurfaceBinding *pgraph_metal_surface_get_within(NV20State *d, hwaddr addr)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    SurfaceBinding *surface;

    QTAILQ_FOREACH(surface, &r->surfaces, entry) {
        if (addr >= surface->vram_addr &&
            addr < surface->vram_addr + surface->size) {
            return surface;
        }
    }
    return NULL;
}

void pgraph_metal_surface_invalidate(NV20State *d, SurfaceBinding *surface)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    if (surface == r->color_binding) {
        pgraph_metal_unbind_surface(d, true);
    }
    if (surface == r->zeta_binding) {
        pgraph_metal_unbind_surface(d, false);
    }

    nv20_mtl_texture_destroy(surface->mtl_texture);
    surface->mtl_texture = NULL;
    QTAILQ_REMOVE(&r->surfaces, surface, entry);
    g_free(surface);
}

void pgraph_metal_unbind_surface(NV20State *d, bool color)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    if (color) {
        r->color_binding = NULL;
    } else {
        r->zeta_binding = NULL;
    }
}

static SurfaceBinding *surface_create(NV20State *d, hwaddr addr,
                                      const SurfaceShape *shape, bool color)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;
    SurfaceBinding *surface = g_malloc0(sizeof(*surface));

    surface->vram_addr = addr;
    surface->shape = *shape;
    surface->color = color;
    surface->swizzle = pg->surface_type == NV097_SET_SURFACE_FORMAT_TYPE_SWIZZLE;

    surface_get_dimensions(pg, &surface->width, &surface->height);
    pgraph_apply_scaling_factor(pg, &surface->width, &surface->height);

    if (color) {
        surface->fmt = kelvin_surface_color_format_map[shape->color_format];
    } else if (shape->zeta_format) {
        if (shape->z_format) {
            surface->fmt = kelvin_surface_zeta_float_format_map[shape->zeta_format];
        } else {
            surface->fmt = kelvin_surface_zeta_fixed_format_map[shape->zeta_format];
        }
    }

    surface->pitch = color ? pg->surface_color.pitch : pg->surface_zeta.pitch;
    surface->size = surface->pitch * surface->height;
    surface->mtl_texture = nv20_mtl_texture_create(
        r->backend, surface->fmt.mtl_pixel_format,
        surface->width, surface->height, 1, 1, false);

    QTAILQ_INSERT_TAIL(&r->surfaces, surface, entry);
    return surface;
}

static void surface_get_dimensions(PGRAPHState *pg, unsigned int *width,
                                   unsigned int *height)
{
    if (pg->surface_type == NV097_SET_SURFACE_FORMAT_TYPE_SWIZZLE) {
        *width = 1 << pg->surface_shape.log_width;
        *height = 1 << pg->surface_shape.log_height;
    } else {
        *width = pg->surface_shape.clip_width;
        *height = pg->surface_shape.clip_height;
    }
}

void pgraph_metal_surface_update(NV20State *d, bool upload, bool color_write,
                                 bool zeta_write)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    if (color_write && pg->surface_color.buffer_dirty) {
        hwaddr addr = pg->surface_color.offset;
        if (r->color_binding && r->color_binding->vram_addr != addr) {
            pgraph_metal_surface_invalidate(d, r->color_binding);
        }
        if (!r->color_binding) {
            r->color_binding = surface_create(d, addr, &pg->surface_shape, true);
        }
        pg->surface_color.buffer_dirty = false;
    }

    if (zeta_write && pg->surface_zeta.buffer_dirty) {
        hwaddr addr = pg->surface_zeta.offset;
        if (r->zeta_binding && r->zeta_binding->vram_addr != addr) {
            pgraph_metal_surface_invalidate(d, r->zeta_binding);
        }
        if (!r->zeta_binding) {
            r->zeta_binding = surface_create(d, addr, &pg->surface_shape, false);
        }
        pg->surface_zeta.buffer_dirty = false;
    }

    if (upload) {
        if (r->color_binding && r->color_binding->upload_pending) {
            pgraph_metal_upload_surface_data(d, r->color_binding, false);
        }
        if (r->zeta_binding && r->zeta_binding->upload_pending) {
            pgraph_metal_upload_surface_data(d, r->zeta_binding, false);
        }
    }
}

void pgraph_metal_upload_surface_data(NV20State *d, SurfaceBinding *surface,
                                      bool force)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    const uint8_t *src = d->vram_ptr + surface->vram_addr;

    if (!force && !surface->upload_pending) {
        return;
    }

    if (surface->swizzle) {
        nv20_mtl_texture_upload_swizzled(r->backend, surface->mtl_texture, 0,
                                         surface->width, surface->height,
                                         surface->pitch, src,
                                         surface->fmt.bytes_per_pixel);
    } else {
        nv20_mtl_texture_upload_level(r->backend, surface->mtl_texture, 0,
                                      surface->width, surface->height, 1,
                                      surface->pitch, src,
                                      surface->fmt.bytes_per_pixel, true);
    }
    surface->upload_pending = false;
}

static void surface_download(NV20State *d, SurfaceBinding *surface, bool force)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    uint8_t *dst = d->vram_ptr + surface->vram_addr;

    if (!force && !surface->draw_dirty) {
        return;
    }

    nv20_mtl_wait_idle(r->backend);
    nv20_mtl_texture_read_pixels(r->backend, surface->mtl_texture, dst,
                                 surface->width, surface->height,
                                 surface->pitch, surface->fmt.bytes_per_pixel);

    if (surface->swizzle) {
        uint8_t *swizzled = g_malloc(surface->size);
        swizzle_rect(dst, surface->width, surface->height, swizzled,
                     surface->pitch, surface->fmt.bytes_per_pixel);
        memcpy(dst, swizzled, surface->size);
        g_free(swizzled);
    }

    surface->draw_dirty = false;
    surface->download_pending = false;
}

void pgraph_metal_surface_download_if_dirty(NV20State *d, SurfaceBinding *surface)
{
    if (surface->draw_dirty) {
        surface_download(d, surface, true);
    }
}

void pgraph_metal_download_dirty_surfaces(NV20State *d)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    SurfaceBinding *surface;

    QTAILQ_FOREACH(surface, &r->surfaces, entry) {
        pgraph_metal_surface_download_if_dirty(d, surface);
    }
    qatomic_set(&r->download_dirty_surfaces_pending, false);
    qemu_event_set(&r->dirty_surfaces_download_complete);
}

void pgraph_metal_process_pending_downloads(NV20State *d)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    SurfaceBinding *surface;

    QTAILQ_FOREACH(surface, &r->surfaces, entry) {
        if (surface->download_pending) {
            surface_download(d, surface, true);
        }
    }
    qatomic_set(&r->downloads_pending, false);
    qemu_event_set(&r->downloads_complete);
}

void pgraph_metal_surface_flush(NV20State *d)
{
    nv20_mtl_wait_idle(d->pgraph.metal_renderer_state->backend);
}

void pgraph_metal_init_surfaces(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;
    QTAILQ_INIT(&r->surfaces);
    qemu_event_init(&r->downloads_complete, false);
    qemu_event_init(&r->dirty_surfaces_download_complete, false);
}

void pgraph_metal_finalize_surfaces(PGRAPHState *pg)
{
    NV20State *d = container_of(pg, NV20State, pgraph);
    PGRAPHMetalState *r = pg->metal_renderer_state;
    SurfaceBinding *surface, *next;

    QTAILQ_FOREACH_SAFE(surface, &r->surfaces, entry, next) {
        pgraph_metal_surface_invalidate(d, surface);
    }
}
