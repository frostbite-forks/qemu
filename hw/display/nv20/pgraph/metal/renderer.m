/*
 * Geforce NV2A PGRAPH Metal Renderer
 */
#include "hw/display/nv20/nv20_int.h"
#include "hw/display/nv20/pgraph/pgraph.h"
#include "debug.h"
#include "renderer.h"

static void early_context_init(void)
{
    pgraph_metal_determine_gpu_properties();
}

static void pgraph_metal_init(NV20State *d, Error **errp)
{
    PGRAPHState *pg = &d->pgraph;

    pg->metal_renderer_state = g_malloc0(sizeof(*pg->metal_renderer_state));
    PGRAPHMetalState *r = pg->metal_renderer_state;

    r->backend = nv20_mtl_backend_create(d->vram_ptr,
                                         memory_region_size(&d->vram_mr));
    r->mtl_native = nv20_mtl_backend_is_available(r->backend);
    if (!r->mtl_native) {
        error_setg(errp, "Metal backend unavailable");
        return;
    }

    nv20_mtl_glsl_init();

    r->supported_aliased_line_width_range[0] = 1.f;
    r->supported_aliased_line_width_range[1] = 1.f;
    r->supported_smooth_line_width_range[0] = 1.f;
    r->supported_smooth_line_width_range[1] = 1.f;
    r->supported_extensions.texture_filter_anisotropic = true;

    pgraph_metal_init_surfaces(pg);
    pgraph_metal_init_reports(d);
    pgraph_metal_init_textures(d);
    pgraph_metal_init_buffers(d);
    pgraph_metal_init_shaders(pg);
    pgraph_metal_init_display(d);

    pgraph_metal_update_entire_memory_buffer(d);

    pg->uniform_attrs = 0;
    pg->swizzle_attrs = 0;
}

static void pgraph_metal_finalize(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    pgraph_metal_finalize_surfaces(pg);
    pgraph_metal_finalize_shaders(pg);
    pgraph_metal_finalize_textures(pg);
    pgraph_metal_finalize_reports(pg);
    pgraph_metal_finalize_buffers(pg);
    pgraph_metal_finalize_display(pg);

    nv20_mtl_glsl_finalize();
    nv20_mtl_backend_destroy(r->backend);
    r->backend = NULL;
    r->mtl_native = false;

    g_free(pg->metal_renderer_state);
    pg->metal_renderer_state = NULL;
}

static void pgraph_metal_flip_stall(NV20State *d)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    nv20_mtl_wait_idle(r->backend);
}

static void pgraph_metal_flush(NV20State *d)
{
    pgraph_metal_surface_flush(d);
    pgraph_metal_mark_textures_possibly_dirty(d, 0, memory_region_size(&d->vram_mr));
    pgraph_metal_update_entire_memory_buffer(d);

    qatomic_set(&d->pgraph.flush_pending, false);
    qemu_event_set(&d->pgraph.flush_complete);
}

static void pgraph_metal_process_pending(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    if (qatomic_read(&r->downloads_pending) ||
        qatomic_read(&r->download_dirty_surfaces_pending) ||
        qatomic_read(&d->pgraph.sync_pending) ||
        qatomic_read(&d->pgraph.flush_pending) ||
        qatomic_read(&r->shader_cache_writeback_pending)) {
        qemu_mutex_unlock(&d->pfifo.lock);
        qemu_mutex_lock(&d->pgraph.lock);
        if (qatomic_read(&r->downloads_pending)) {
            pgraph_metal_process_pending_downloads(d);
        }
        if (qatomic_read(&r->download_dirty_surfaces_pending)) {
            pgraph_metal_download_dirty_surfaces(d);
        }
        if (qatomic_read(&d->pgraph.sync_pending)) {
            pgraph_metal_sync(d);
        }
        if (qatomic_read(&d->pgraph.flush_pending)) {
            pgraph_metal_flush(d);
        }
        if (qatomic_read(&r->shader_cache_writeback_pending)) {
            pgraph_metal_shader_write_cache_reload_list(&d->pgraph);
        }
        qemu_mutex_unlock(&d->pgraph.lock);
        qemu_mutex_lock(&d->pfifo.lock);
    }
}

static void pgraph_metal_pre_savevm_trigger(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    qatomic_set(&r->download_dirty_surfaces_pending, true);
    qemu_event_reset(&r->dirty_surfaces_download_complete);
}

static void pgraph_metal_pre_savevm_wait(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    qemu_event_wait(&r->dirty_surfaces_download_complete);
}

static void pgraph_metal_pre_shutdown_trigger(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    qatomic_set(&r->shader_cache_writeback_pending, true);
    qemu_event_reset(&r->shader_cache_writeback_complete);
}

static void pgraph_metal_pre_shutdown_wait(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    qemu_event_wait(&r->shader_cache_writeback_complete);
}

static PGRAPHRenderer pgraph_metal_renderer = {
    .type = CONFIG_DISPLAY_RENDERER_METAL,
    .name = "Metal",
    .ops = {
        .init = pgraph_metal_init,
        .early_context_init = early_context_init,
        .finalize = pgraph_metal_finalize,
        .clear_report_value = pgraph_metal_clear_report_value,
        .clear_surface = pgraph_metal_clear_surface,
        .draw_begin = pgraph_metal_draw_begin,
        .draw_end = pgraph_metal_draw_end,
        .flip_stall = pgraph_metal_flip_stall,
        .flush_draw = pgraph_metal_flush_draw,
        .get_report = pgraph_metal_get_report,
        .image_blit = pgraph_metal_image_blit,
        .pre_savevm_trigger = pgraph_metal_pre_savevm_trigger,
        .pre_savevm_wait = pgraph_metal_pre_savevm_wait,
        .pre_shutdown_trigger = pgraph_metal_pre_shutdown_trigger,
        .pre_shutdown_wait = pgraph_metal_pre_shutdown_wait,
        .process_pending = pgraph_metal_process_pending,
        .process_pending_reports = pgraph_metal_process_pending_reports,
        .surface_update = pgraph_metal_surface_update,
        .set_surface_scale_factor = pgraph_metal_set_surface_scale_factor,
        .get_surface_scale_factor = pgraph_metal_get_surface_scale_factor,
        .get_framebuffer_surface = (int (*)(NV20State *))pgraph_metal_get_framebuffer_surface,
        .get_gpu_properties = pgraph_metal_get_gpu_properties,
    }
};

static void __attribute__((constructor)) register_renderer(void)
{
    pgraph_renderer_register(&pgraph_metal_renderer);
}
