/*
 * NV20 PGRAPH Metal shader management.
 */
#include "qemu/osdep.h"
#include "qemu/fast-hash.h"
#include "qemu/mstring.h"
#include "hw/display/nv20/pgraph/util.h"
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

static void *compile_shader_module(PGRAPHMetalState *r,
                                   const ShaderModuleCacheKey *key)
{
    MString *code = NULL;
    const char *kind_str;
    char *msl = NULL;
    void *library = NULL;

    GenVshGlslOptions vsh_opts = key->vsh.glsl_opts;
    GenGeomGlslOptions geom_opts = key->geom.glsl_opts;
    GenPshGlslOptions psh_opts = key->psh.glsl_opts;
    vsh_opts.vulkan = true;
    geom_opts.vulkan = true;
    psh_opts.vulkan = true;

    switch (key->kind) {
    case NV20_SHADER_VERTEX:
        kind_str = "vertex";
        code = pgraph_glsl_gen_vsh(&key->vsh.state, vsh_opts);
        msl = nv20_mtl_glsl_to_msl_vertex(mstring_get_str(code));
        break;
    case NV20_SHADER_GEOMETRY:
        kind_str = "geometry";
        code = pgraph_glsl_gen_geom(&key->geom.state, geom_opts);
        msl = nv20_mtl_glsl_to_msl_geometry(mstring_get_str(code));
        break;
    case NV20_SHADER_FRAGMENT:
        kind_str = "fragment";
        code = pgraph_glsl_gen_psh(&key->psh.state, psh_opts);
        msl = nv20_mtl_glsl_to_msl_fragment(mstring_get_str(code));
        break;
    default:
        return NULL;
    }

    library = nv20_mtl_compile_library(r->backend, msl);
    if (!library) {
        fprintf(stderr, "nv2a: %s MSL compilation failed\n", kind_str);
    }

    g_free(msl);
    mstring_unref(code);
    return library;
}

static void shader_module_cache_entry_init(Lru *lru, LruNode *node,
                                           const void *key)
{
    ShaderModuleCacheEntry *module =
        container_of(node, ShaderModuleCacheEntry, node);
    PGRAPHMetalState *r = g_nv20->pgraph.metal_renderer_state;

    memcpy(&module->key, key, sizeof(ShaderModuleCacheKey));
    module->mtl_library = compile_shader_module(r, &module->key);
    module->entry_point = module->key.kind == NV20_SHADER_VERTEX ? "nv20_vsh_main" :
                          module->key.kind == NV20_SHADER_FRAGMENT ? "nv20_psh_main" :
                          "nv20_geom_main";
}

static void shader_module_cache_entry_post_evict(Lru *lru, LruNode *node)
{
    ShaderModuleCacheEntry *module =
        container_of(node, ShaderModuleCacheEntry, node);
    if (module->mtl_library) {
        nv20_mtl_release_library(g_nv20->pgraph.metal_renderer_state->backend,
                                 module->mtl_library);
        module->mtl_library = NULL;
    }
}

static bool shader_module_cache_entry_compare(Lru *lru, LruNode *node,
                                              const void *key)
{
    ShaderModuleCacheEntry *module =
        container_of(node, ShaderModuleCacheEntry, node);
    return memcmp(&module->key, key, sizeof(ShaderModuleCacheKey));
}

static void *get_shader_module_for_key(PGRAPHMetalState *r,
                                       const ShaderModuleCacheKey *key)
{
    uint64_t hash = fast_hash((void *)key, sizeof(ShaderModuleCacheKey));
    LruNode *node = lru_lookup(&r->shader_module_cache, hash, key);
    ShaderModuleCacheEntry *module =
        container_of(node, ShaderModuleCacheEntry, node);
    return module->mtl_library;
}

static void generate_shaders(PGRAPHMetalState *r, ShaderBinding *binding)
{
    ShaderState *state = &binding->state;
    ShaderModuleCacheKey key;
    void *vs_lib = NULL, *fs_lib = NULL, *gs_lib = NULL;

    if (pgraph_glsl_need_geom(&state->geom)) {
        memset(&key, 0, sizeof(key));
        key.kind = NV20_SHADER_GEOMETRY;
        key.geom.state = state->geom;
        key.geom.glsl_opts.vulkan = true;
        gs_lib = get_shader_module_for_key(r, &key);
    }

    memset(&key, 0, sizeof(key));
    key.kind = NV20_SHADER_VERTEX;
    key.vsh.state = state->vsh;
    key.vsh.glsl_opts.vulkan = true;
    key.vsh.glsl_opts.prefix_outputs = gs_lib != NULL;
    vs_lib = get_shader_module_for_key(r, &key);

    memset(&key, 0, sizeof(key));
    key.kind = NV20_SHADER_FRAGMENT;
    key.psh.state = state->psh;
    key.psh.glsl_opts.vulkan = true;
    fs_lib = get_shader_module_for_key(r, &key);

    NV20MtlPipelineKey pkey = {
        .vertex_lib = vs_lib,
        .fragment_lib = fs_lib,
        .geometry_lib = gs_lib,
        .vs_entry = "nv20_vsh_main",
        .fs_entry = "nv20_psh_main",
        .gs_entry = "nv20_geom_main",
        .color_format = r->color_binding ?
            r->color_binding->fmt.mtl_pixel_format : 80,
        .depth_format = r->zeta_binding ?
            r->zeta_binding->fmt.mtl_pixel_format : 0,
    };

    binding->mtl_pipeline = nv20_mtl_pipeline_get(r->backend, &pkey);
    binding->mtl_primitive_mode = get_mtl_primitive_mode(
        state->geom.polygon_front_mode, state->geom.primitive_mode);
    binding->initialized = true;
}

static void shader_cache_entry_init(Lru *lru, LruNode *node, const void *state)
{
    ShaderBinding *binding = container_of(node, ShaderBinding, node);
    memcpy(&binding->state, state, sizeof(ShaderState));
    binding->initialized = false;
    binding->cached = false;
    binding->program = NULL;
    binding->save_thread = NULL;
}

static void shader_cache_entry_post_evict(Lru *lru, LruNode *node)
{
    ShaderBinding *binding = container_of(node, ShaderBinding, node);
    if (binding->mtl_pipeline) {
        nv20_mtl_pipeline_release(binding->mtl_pipeline);
        binding->mtl_pipeline = NULL;
    }
    g_free(binding->program);
    binding->program = NULL;
    memset(&binding->state, 0, sizeof(ShaderState));
}

static bool shader_cache_entry_compare(Lru *lru, LruNode *node, const void *key)
{
    ShaderBinding *binding = container_of(node, ShaderBinding, node);
    return memcmp(&binding->state, key, sizeof(ShaderState));
}

void pgraph_metal_init_shaders(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;
    const size_t shader_cache_size = 50 * 1024;
    const size_t module_cache_size = 50 * 1024;

    qemu_mutex_init(&r->shader_cache_lock);
    qemu_event_init(&r->shader_cache_writeback_complete, false);

    lru_init(&r->shader_cache);
    r->shader_cache_entries = g_malloc_n(shader_cache_size, sizeof(ShaderBinding));
    for (int i = 0; i < shader_cache_size; i++) {
        lru_add_free(&r->shader_cache, &r->shader_cache_entries[i].node);
    }
    r->shader_cache.init_node = shader_cache_entry_init;
    r->shader_cache.compare_nodes = shader_cache_entry_compare;
    r->shader_cache.post_node_evict = shader_cache_entry_post_evict;

    lru_init(&r->shader_module_cache);
    r->shader_module_cache_entries =
        g_malloc_n(module_cache_size, sizeof(ShaderModuleCacheEntry));
    for (int i = 0; i < module_cache_size; i++) {
        lru_add_free(&r->shader_module_cache,
                     &r->shader_module_cache_entries[i].node);
    }
    r->shader_module_cache.init_node = shader_module_cache_entry_init;
    r->shader_module_cache.compare_nodes = shader_module_cache_entry_compare;
    r->shader_module_cache.post_node_evict = shader_module_cache_entry_post_evict;
}

void pgraph_metal_finalize_shaders(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;
    lru_flush(&r->shader_cache);
    g_free(r->shader_cache_entries);
    r->shader_cache_entries = NULL;
    lru_flush(&r->shader_module_cache);
    g_free(r->shader_module_cache_entries);
    r->shader_module_cache_entries = NULL;
    qemu_mutex_destroy(&r->shader_cache_lock);
}

void pgraph_metal_bind_shaders(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;
    ShaderState state = pgraph_glsl_get_shader_state(pg);
    uint64_t hash = fast_hash((uint8_t *)&state, sizeof(state));
    LruNode *node = lru_lookup(&r->shader_cache, hash, &state);
    ShaderBinding *binding = container_of(node, ShaderBinding, node);

    if (!binding->initialized) {
        generate_shaders(r, binding);
    }
    r->shader_binding = binding;
    nv20_mtl_set_pipeline(r->backend, binding->mtl_pipeline);
}

void *pgraph_metal_compile_shader(const char *vs_src, const char *fs_src)
{
    (void)vs_src; (void)fs_src;
    return NULL;
}

bool pgraph_metal_shader_load_from_memory(ShaderBinding *binding)
{
    (void)binding;
    return false;
}

void pgraph_metal_shader_cache_to_disk(ShaderBinding *snode)
{
    (void)snode;
}

void pgraph_metal_shader_write_cache_reload_list(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;
    qatomic_set(&r->shader_cache_writeback_pending, false);
    qemu_event_set(&r->shader_cache_writeback_complete);
}
