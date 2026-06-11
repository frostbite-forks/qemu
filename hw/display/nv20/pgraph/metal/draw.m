/*
 * NV20 PGRAPH Metal draw path.
 */
#include "qemu/fast-hash.h"
#include "hw/display/nv20/nv20_int.h"
#include "debug.h"
#include "renderer.h"

static NV20MtlPrimitive get_mtl_primitive_mode(enum ShaderPolygonMode polygon_mode,
                                               enum ShaderPrimitiveMode primitive_mode)
{
    switch (primitive_mode) {
    case PRIM_TYPE_POINTS: return NV20_MTL_PRIM_POINTS;
    case PRIM_TYPE_LINES: return NV20_MTL_PRIM_LINES;
    case PRIM_TYPE_LINE_LOOP: return NV20_MTL_PRIM_LINE_LOOP;
    case PRIM_TYPE_LINE_STRIP: return NV20_MTL_PRIM_LINE_STRIP;
    case PRIM_TYPE_TRIANGLES: return NV20_MTL_PRIM_TRIANGLES;
    case PRIM_TYPE_TRIANGLE_STRIP: return NV20_MTL_PRIM_TRIANGLE_STRIP;
    case PRIM_TYPE_TRIANGLE_FAN: return NV20_MTL_PRIM_TRIANGLE_FAN;
    case PRIM_TYPE_QUADS: return NV20_MTL_PRIM_LINES_ADJACENCY;
    case PRIM_TYPE_QUAD_STRIP: return NV20_MTL_PRIM_LINE_STRIP_ADJACENCY;
    case PRIM_TYPE_POLYGON:
        return polygon_mode == POLY_MODE_LINE ? NV20_MTL_PRIM_LINE_LOOP
                                              : NV20_MTL_PRIM_TRIANGLE_FAN;
    default:
        return NV20_MTL_PRIM_TRIANGLES;
    }
}

static void begin_render_pass(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;
    NV20MtlRenderPassDesc desc = { 0 };

    desc.color = r->color_binding ? r->color_binding->mtl_texture : NULL;
    desc.depth = r->zeta_binding ? r->zeta_binding->mtl_texture : NULL;
    desc.load_color = r->color_binding && !r->color_binding->cleared;
    desc.load_depth = r->zeta_binding && !r->zeta_binding->cleared;
    if (r->color_binding) {
        desc.width = r->color_binding->width;
        desc.height = r->color_binding->height;
    } else if (r->zeta_binding) {
        desc.width = r->zeta_binding->width;
        desc.height = r->zeta_binding->height;
    }
    nv20_mtl_begin_render_pass(r->backend, &desc);
}

void pgraph_metal_clear_surface(NV20State *d, uint32_t parameter)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    pg->clearing = true;

    bool write_color = parameter & NV097_CLEAR_SURFACE_COLOR;
    bool write_zeta = parameter & (NV097_CLEAR_SURFACE_Z | NV097_CLEAR_SURFACE_STENCIL);

    pgraph_metal_surface_update(d, true, write_color, write_zeta);

    unsigned int xmin = GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_CLEARRECTX),
                                 NV_PGRAPH_CLEARRECTX_XMIN);
    unsigned int xmax = GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_CLEARRECTX),
                                 NV_PGRAPH_CLEARRECTX_XMAX);
    unsigned int ymin = GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_CLEARRECTY),
                                 NV_PGRAPH_CLEARRECTY_YMIN);
    unsigned int ymax = GET_MASK(pgraph_reg_r(pg, NV_PGRAPH_CLEARRECTY),
                                 NV_PGRAPH_CLEARRECTY_YMAX);

    unsigned int scissor_width = xmax - xmin + 1;
    unsigned int scissor_height = ymax - ymin + 1;
    pgraph_apply_anti_aliasing_factor(pg, &xmin, &ymin);
    pgraph_apply_anti_aliasing_factor(pg, &scissor_width, &scissor_height);
    pgraph_apply_scaling_factor(pg, &xmin, &ymin);
    pgraph_apply_scaling_factor(pg, &scissor_width, &scissor_height);

    NV20MtlRenderPassDesc desc = { 0 };
    desc.color = write_color && r->color_binding ? r->color_binding->mtl_texture : NULL;
    desc.depth = write_zeta && r->zeta_binding ? r->zeta_binding->mtl_texture : NULL;
    if (write_color) {
        pgraph_get_clear_color(pg, desc.clear_color);
    }
    if (write_zeta) {
        float depth;
        int stencil;
        pgraph_get_clear_depth_stencil_value(pg, &depth, &stencil);
        desc.clear_depth = depth;
        desc.clear_stencil = stencil;
    }
    if (r->color_binding) {
        desc.width = r->color_binding->width;
        desc.height = r->color_binding->height;
    }
    nv20_mtl_begin_render_pass(r->backend, &desc);
    nv20_mtl_set_scissor(r->backend, xmin, ymin, scissor_width, scissor_height);
    nv20_mtl_end_render_pass(r->backend);

    pgraph_metal_set_surface_dirty(pg, write_color, write_zeta);
    pg->clearing = false;
}

void pgraph_metal_draw_begin(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    uint32_t control_0 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_0);
    bool color_write = (control_0 & (NV_PGRAPH_CONTROL_0_ALPHA_WRITE_ENABLE |
                                      NV_PGRAPH_CONTROL_0_RED_WRITE_ENABLE |
                                      NV_PGRAPH_CONTROL_0_GREEN_WRITE_ENABLE |
                                      NV_PGRAPH_CONTROL_0_BLUE_WRITE_ENABLE));
    bool depth_test = control_0 & NV_PGRAPH_CONTROL_0_ZENABLE;
    bool stencil_test = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_1) &
                        NV_PGRAPH_CONTROL_1_STENCIL_TEST_ENABLE;
    bool is_nop_draw = !(color_write || depth_test || stencil_test);

    pgraph_metal_surface_update(d, true, true, depth_test || stencil_test);
    if (is_nop_draw) {
        return;
    }

    pgraph_metal_bind_textures(d);
    pgraph_metal_bind_shaders(pg);
    begin_render_pass(d);

    unsigned int vp_width = pg->surface_binding_dim.width;
    unsigned int vp_height = pg->surface_binding_dim.height;
    pgraph_apply_scaling_factor(pg, &vp_width, &vp_height);
    nv20_mtl_set_viewport(r->backend, 0, 0, vp_width, vp_height);

    unsigned int xmin = pg->surface_shape.clip_x;
    unsigned int ymin = pg->surface_shape.clip_y;
    unsigned int scissor_width = pg->surface_shape.clip_width;
    unsigned int scissor_height = pg->surface_shape.clip_height;
    pgraph_apply_anti_aliasing_factor(pg, &xmin, &ymin);
    pgraph_apply_anti_aliasing_factor(pg, &scissor_width, &scissor_height);
    pgraph_apply_scaling_factor(pg, &xmin, &ymin);
    pgraph_apply_scaling_factor(pg, &scissor_width, &scissor_height);
    nv20_mtl_set_scissor(r->backend, xmin, ymin, scissor_width, scissor_height);

    if (r->shader_binding && r->shader_binding->mtl_pipeline) {
        nv20_mtl_set_pipeline(r->backend, r->shader_binding->mtl_pipeline);
    }
}

void pgraph_metal_draw_end(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    uint32_t control_0 = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_0);
    bool color_write = (control_0 & (NV_PGRAPH_CONTROL_0_ALPHA_WRITE_ENABLE |
                                      NV_PGRAPH_CONTROL_0_RED_WRITE_ENABLE |
                                      NV_PGRAPH_CONTROL_0_GREEN_WRITE_ENABLE |
                                      NV_PGRAPH_CONTROL_0_BLUE_WRITE_ENABLE));
    bool depth_test = control_0 & NV_PGRAPH_CONTROL_0_ZENABLE;
    bool stencil_test = pgraph_reg_r(pg, NV_PGRAPH_CONTROL_1) &
                        NV_PGRAPH_CONTROL_1_STENCIL_TEST_ENABLE;
    if (!(color_write || depth_test || stencil_test)) {
        return;
    }

    pgraph_metal_flush_draw(d);
    nv20_mtl_end_render_pass(r->backend);

    pg->draw_time++;
    if (r->color_binding && pgraph_color_write_enabled(pg)) {
        r->color_binding->draw_time = pg->draw_time;
    }
    if (r->zeta_binding && pgraph_zeta_write_enabled(pg)) {
        r->zeta_binding->draw_time = pg->draw_time;
    }
    pgraph_metal_set_surface_dirty(pg, color_write, depth_test || stencil_test);
}

void pgraph_metal_flush_draw(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    if (!(r->color_binding || r->zeta_binding) || !r->shader_binding) {
        return;
    }

    NV20MtlPrimitive prim = r->shader_binding->mtl_primitive_mode;

    if (pg->draw_arrays_length) {
        pgraph_metal_bind_vertex_attributes(d, pg->draw_arrays_min_start,
                                            pg->draw_arrays_max_count - 1,
                                            false, 0,
                                            pg->draw_arrays_max_count - 1);
        for (unsigned int i = 0; i < pg->draw_arrays_length; i++) {
            nv20_mtl_draw(r->backend, prim, pg->draw_arrays_start[i],
                          pg->draw_arrays_count[i]);
        }
    } else if (pg->inline_elements_length) {
        uint32_t min_element = (uint32_t)-1, max_element = 0;
        for (int i = 0; i < pg->inline_elements_length; i++) {
            max_element = MAX(pg->inline_elements[i], max_element);
            min_element = MIN(pg->inline_elements[i], min_element);
        }
        pgraph_metal_bind_vertex_attributes(d, min_element, max_element, false, 0,
                                          pg->inline_elements[pg->inline_elements_length - 1]);

        VertexKey k = { .count = pg->inline_elements_length,
                        .stride = sizeof(uint32_t),
                        .mtl_type = NV20_MTL_INDEX_UINT32 };
        uint64_t h = fast_hash((uint8_t *)pg->inline_elements,
                               pg->inline_elements_length * 4);
        LruNode *node = lru_lookup(&r->element_cache, h, &k);
        VertexLruNode *found = container_of(node, VertexLruNode, node);
        if (!found->initialized) {
            nv20_mtl_buffer_write(found->mtl_buffer, 0, pg->inline_elements,
                                  pg->inline_elements_length * 4);
            found->initialized = true;
        }
        nv20_mtl_draw_indexed(r->backend, prim, NV20_MTL_INDEX_UINT32,
                              found->mtl_buffer, pg->inline_elements_length, 0);
    } else if (pg->inline_buffer_length) {
        for (int i = 0; i < NV20_VERTEXSHADER_ATTRIBUTES; i++) {
            VertexAttribute *attr = &pg->vertex_attributes[i];
            if (attr->inline_buffer_populated) {
                nv20_mtl_buffer_write(r->mtl_inline_buffer[i], 0,
                                      attr->inline_buffer,
                                      pg->inline_buffer_length * sizeof(float) * 4);
                nv20_mtl_set_vertex_buffer(r->backend, i, r->mtl_inline_buffer[i], 0);
                attr->inline_buffer_populated = false;
            }
        }
        nv20_mtl_draw(r->backend, prim, 0, pg->inline_buffer_length);
    } else if (pg->inline_array_length) {
        unsigned int index_count = pgraph_metal_bind_inline_array(d);
        nv20_mtl_draw(r->backend, prim, 0, index_count);
    }
}
