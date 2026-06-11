/*
 * Native Metal display composite for NV20 PGRAPH.
 */
#include "hw/display/nv20/nv20_int.h"
#include "hw/display/nv20/pgraph/util.h"
#include "renderer.h"

void pgraph_metal_init_native(NV20State *d)
{
    (void)d;
}

void pgraph_metal_finalize_native(PGRAPHState *pg)
{
    (void)pg;
}

void pgraph_metal_sync_native(NV20State *d, SurfaceBinding *surface,
                              unsigned int width, unsigned int height)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    VGADisplayParams vga_display_params;

    if (!r->mtl_native) {
        return;
    }

    d->vga.get_params(&d->vga, &vga_display_params);

    uint32_t bpp = surface->fmt.bytes_per_pixel ? surface->fmt.bytes_per_pixel : 4;
    uint32_t dst_offset = d->pcrtc.start + vga_display_params.line_offset;

    nv20_mtl_composite_surface(r->backend,
                               surface->vram_addr, surface->pitch,
                               surface->width, surface->height,
                               dst_offset, surface->pitch,
                               width, height, bpp);
}

void pgraph_metal_clear_surface_native(NV20State *d, SurfaceBinding *surface,
                                       uint32_t rgba)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    if (!r->mtl_native || !surface) {
        return;
    }

    uint32_t bpp = surface->fmt.bytes_per_pixel ? surface->fmt.bytes_per_pixel : 4;

    nv20_mtl_clear_region(r->backend, surface->vram_addr, surface->width,
                          surface->height, surface->pitch, bpp, rgba);
}
