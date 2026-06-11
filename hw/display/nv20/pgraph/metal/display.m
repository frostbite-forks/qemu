/*
 * NV20 PGRAPH Metal display composite.
 */
#include "qemu/osdep.h"
#include "hw/display/vga_int.h"
#include "hw/display/nv20/nv20_int.h"
#include "hw/display/nv20/pgraph/util.h"
#include "renderer.h"
#include <math.h>

void pgraph_metal_init_display(NV20State *d)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    r->mtl_display_width = 0;
    r->mtl_display_height = 0;
    r->mtl_display_buffer = NULL;
    r->disp_rndr.pvideo_tex = nv20_mtl_texture_create(r->backend, 70, 64, 64, 1, 1, false);
}

void pgraph_metal_finalize_display(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;
    if (r->mtl_display_buffer) {
        nv20_mtl_texture_destroy(r->mtl_display_buffer);
        r->mtl_display_buffer = NULL;
    }
    if (r->disp_rndr.pvideo_tex) {
        nv20_mtl_texture_destroy(r->disp_rndr.pvideo_tex);
        r->disp_rndr.pvideo_tex = NULL;
    }
}

static void render_display(NV20State *d, SurfaceBinding *surface)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;
    VGADisplayParams vga_display_params;

    d->vga.get_params(&d->vga, &vga_display_params);
    unsigned width, height;
    d->vga.get_resolution(&d->vga, (int *)&width, (int *)&height);
    if (d->vga.cr[NV_PRMCIO_INTERLACE_MODE] != NV_PRMCIO_INTERLACE_MODE_DISABLED) {
        height *= 2;
    }
    pgraph_apply_scaling_factor(pg, &width, &height);

    r->mtl_display_buffer = nv20_mtl_display_texture_get(r->backend, 80, width, height);

    bool pvideo_enable = false;
    float pvideo_params[16] = { 0 };
    float line_offset = (float)vga_display_params.line_offset;

    nv20_mtl_display_blit(r->backend, surface->mtl_texture,
                          surface->width, surface->height,
                          line_offset, pvideo_enable,
                          r->disp_rndr.pvideo_tex, pvideo_params);
}

void pgraph_metal_sync(NV20State *d)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    VGADisplayParams vga_display_params;
    d->vga.get_params(&d->vga, &vga_display_params);

    SurfaceBinding *surface = pgraph_metal_surface_get_within(
        d, d->pcrtc.start + vga_display_params.line_offset);
    if (surface == NULL || !surface->color || !surface->width || !surface->height) {
        qatomic_set(&d->pgraph.sync_pending, false);
        qemu_event_set(&d->pgraph.sync_complete);
        return;
    }

    pgraph_metal_upload_surface_data(d, surface, !tcg_enabled());
    nv20_mtl_wait_idle(r->backend);

    unsigned int width, height;
    d->vga.get_resolution(&d->vga, (int *)&width, (int *)&height);
    if (d->vga.cr[NV_PRMCIO_INTERLACE_MODE] != NV_PRMCIO_INTERLACE_MODE_DISABLED) {
        height *= 2;
    }
    pgraph_apply_scaling_factor(&d->pgraph, &width, &height);

    render_display(d, surface);
    pgraph_metal_sync_native(d, surface, width, height);

    qatomic_set(&d->pgraph.sync_pending, false);
    qemu_event_set(&d->pgraph.sync_complete);
}

NV20MtlTexture *pgraph_metal_get_framebuffer_surface(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    qemu_mutex_lock(&d->pfifo.lock);
    VGADisplayParams vga_display_params;
    d->vga.get_params(&d->vga, &vga_display_params);

    SurfaceBinding *surface = pgraph_metal_surface_get_within(
        d, d->pcrtc.start + vga_display_params.line_offset);
    if (surface == NULL || !surface->color) {
        qemu_mutex_unlock(&d->pfifo.lock);
        return NULL;
    }

    surface->frame_time = pg->frame_time;
    qemu_event_reset(&d->pgraph.sync_complete);
    qatomic_set(&pg->sync_pending, true);
    pfifo_kick(d);
    qemu_mutex_unlock(&d->pfifo.lock);
    qemu_event_wait(&d->pgraph.sync_complete);

    return r->mtl_display_buffer;
}
