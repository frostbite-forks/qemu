/*
 * Geforce NV2A PGRAPH Metal Renderer
 *
 * Copyright (c) 2012 espes
 * Copyright (c) 2015 Jannik Vogel
 * Copyright (c) 2018-2025 Matt Borgerson
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */

#ifndef HW_NV20_PGRAPH_METAL_RENDERER_H
#define HW_NV20_PGRAPH_METAL_RENDERER_H

#include "qemu/osdep.h"
#include "qemu/thread.h"
#include "qemu/queue.h"
#include "qemu/lru.h"

#include "hw/display/nv20/nv20_int.h"
#include "hw/display/nv20/nv20_regs.h"
#include "hw/display/nv20/nv20_pgraph_config.h"
#include "hw/display/nv20/pgraph/surface.h"
#include "hw/display/nv20/pgraph/texture.h"
#include "hw/display/nv20/pgraph/glsl/shaders.h"

#if CONFIG_METAL_NATIVE
#include "metal_backend.h"
#include "metal_formats.h"
#include "metal_glsl.h"
#else
#include "gloffscreen.h"
#include "constants.h"
#endif

typedef struct NV20MetalBackend NV20MetalBackend;

typedef struct SurfaceBinding {
    QTAILQ_ENTRY(SurfaceBinding) entry;
    MemAccessCallback *access_cb;

    hwaddr vram_addr;

    SurfaceShape shape;
    uintptr_t dma_addr;
    uintptr_t dma_len;
    bool color;
    bool swizzle;

    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    size_t size;

    bool cleared;
    int frame_time;
    int draw_time;
    bool draw_dirty;
    bool download_pending;
    bool upload_pending;

#if CONFIG_METAL_NATIVE
    NV20MtlTexture *mtl_texture;
#else
    GLuint gl_buffer;
#endif
    SurfaceFormatInfo fmt;
} SurfaceBinding;

typedef struct TextureBinding {
    unsigned int refcnt;
    int draw_time;
    uint64_t data_hash;
    unsigned int scale;
    unsigned int min_filter;
    unsigned int mag_filter;
    uint32_t lod_bias;
    unsigned int addru;
    unsigned int addrv;
    unsigned int addrp;
    uint32_t border_color;
    bool border_color_set;
#if CONFIG_METAL_NATIVE
    unsigned int mtl_target;
    NV20MtlTexture *mtl_texture;
    NV20MtlSampler *mtl_sampler;
#else
    GLenum gl_target;
    GLuint gl_texture;
#endif
} TextureBinding;

typedef struct ShaderModuleCacheKey {
#if CONFIG_METAL_NATIVE
    uint32_t kind;
#else
    GLenum kind;
#endif
    union {
        struct {
            VshState state;
            GenVshGlslOptions glsl_opts;
        } vsh;
        struct {
            GeomState state;
            GenGeomGlslOptions glsl_opts;
        } geom;
        struct {
            PshState state;
            GenPshGlslOptions glsl_opts;
        } psh;
    };
} ShaderModuleCacheKey;

typedef struct ShaderModuleCacheEntry {
    LruNode node;
    ShaderModuleCacheKey key;
#if CONFIG_METAL_NATIVE
    void *mtl_library;
    const char *entry_point;
#else
    GLuint gl_shader;
#endif
} ShaderModuleCacheEntry;

typedef struct ShaderBinding {
    LruNode node;
    bool initialized;

    bool cached;
    void *program;
    size_t program_size;
#if CONFIG_METAL_NATIVE
    uint32_t program_format;
#else
    GLenum program_format;
#endif
    ShaderState state;
    QemuThread *save_thread;

#if CONFIG_METAL_NATIVE
    NV20MtlPipeline *mtl_pipeline;
    NV20MtlPrimitive mtl_primitive_mode;
#else
    GLuint gl_program;
    GLenum gl_primitive_mode;
#endif

    struct {
        PshUniformLocs psh;
        VshUniformLocs vsh;
    } uniform_locs;
} ShaderBinding;

typedef struct VertexKey {
    size_t count;
    size_t stride;
    hwaddr addr;

#if CONFIG_METAL_NATIVE
    bool mtl_normalize;
    uint32_t mtl_type;
#else
    GLboolean gl_normalize;
    GLuint gl_type;
#endif
} VertexKey;

typedef struct VertexLruNode {
    LruNode node;
    VertexKey key;
    bool initialized;

#if CONFIG_METAL_NATIVE
    NV20MtlBuffer *mtl_buffer;
#else
    GLuint gl_buffer;
#endif
} VertexLruNode;

typedef struct TextureKey {
    TextureShape state;
    hwaddr texture_vram_offset;
    hwaddr texture_length;
    hwaddr palette_vram_offset;
    hwaddr palette_length;
} TextureKey;

typedef struct TextureLruNode {
    LruNode node;
    TextureKey key;
    TextureBinding *binding;
    bool possibly_dirty;
} TextureLruNode;

typedef struct QueryReport {
    QSIMPLEQ_ENTRY(QueryReport) entry;
    bool clear;
    uint32_t parameter;
    unsigned int query_count;
#if CONFIG_METAL_NATIVE
    uint32_t *query_results;
#else
    GLuint *queries;
#endif
} QueryReport;

typedef struct PGRAPHMetalState {
#if CONFIG_METAL_NATIVE
    NV20MetalBackend *backend;
    NV20MtlTexture *mtl_display_buffer;
    NV20MtlTexture *mtl_framebuffer_color;
    NV20MtlTexture *mtl_framebuffer_depth;
    unsigned mtl_display_width;
    unsigned mtl_display_height;
    NV20MtlBuffer *mtl_memory_buffer;
    NV20MtlBuffer *mtl_inline_array_buffer;
    NV20MtlBuffer *mtl_inline_buffer[NV20_VERTEXSHADER_ATTRIBUTES];
    struct {
        NV20MtlPipeline *pipeline;
        NV20MtlTexture *pvideo_tex;
    } disp_rndr;
    struct {
        NV20MtlPipeline *pipeline;
    } s2t_rndr;
#else
    GLuint gl_framebuffer;
    GLuint gl_display_buffer;
    GLint gl_display_buffer_internal_format;
    GLsizei gl_display_buffer_width;
    GLsizei gl_display_buffer_height;
    GLenum gl_display_buffer_format;
    GLenum gl_display_buffer_type;
    GLuint gl_inline_array_buffer;
    GLuint gl_memory_buffer;
    GLuint gl_vertex_array;
    GLuint gl_inline_buffer[NV20_VERTEXSHADER_ATTRIBUTES];
    struct s2t_rndr {
        GLuint fbo, vao, vbo, prog;
        GLuint tex_loc, surface_size_loc;
    } s2t_rndr;
    struct disp_rndr {
        GLuint fbo, vao, vbo, prog;
        GLuint display_size_loc;
        GLuint line_offset_loc;
        GLuint tex_loc;
        GLuint pvideo_tex;
        GLint pvideo_enable_loc;
        GLint pvideo_tex_loc;
        GLint pvideo_in_pos_loc;
        GLint pvideo_pos_loc;
        GLint pvideo_scale_loc;
        GLint pvideo_color_key_enable_loc;
        GLint pvideo_color_key_loc;
        GLint palette_loc[256];
    } disp_rndr;
    unsigned int gl_zpass_pixel_count_query_count;
    GLuint *gl_zpass_pixel_count_queries;
#endif

    Lru element_cache;
    VertexLruNode *element_cache_entries;

    QTAILQ_HEAD(, SurfaceBinding) surfaces;
    SurfaceBinding *color_binding, *zeta_binding;
    bool downloads_pending;
    QemuEvent downloads_complete;
    bool download_dirty_surfaces_pending;
    QemuEvent dirty_surfaces_download_complete;

    TextureBinding *texture_binding[NV20_MAX_TEXTURES];
    Lru texture_cache;
    TextureLruNode *texture_cache_entries;

    Lru shader_cache;
    ShaderBinding *shader_cache_entries;
    ShaderBinding *shader_binding;
    QemuMutex shader_cache_lock;
    QemuThread shader_disk_thread;

    Lru shader_module_cache;
    ShaderModuleCacheEntry *shader_module_cache_entries;

    unsigned int zpass_pixel_count_result;
#if CONFIG_METAL_NATIVE
    unsigned int mtl_zpass_query_count;
    uint32_t *mtl_zpass_query_results;
#else
    unsigned int gl_zpass_pixel_count_query_count;
    GLuint *gl_zpass_pixel_count_queries;
#endif
    QSIMPLEQ_HEAD(, QueryReport) report_queue;

    bool shader_cache_writeback_pending;
    QemuEvent shader_cache_writeback_complete;

    float supported_aliased_line_width_range[2];
    float supported_smooth_line_width_range[2];

    struct supported_extensions {
        bool texture_filter_anisotropic;
    } supported_extensions;

    bool mtl_native;
} PGRAPHMetalState;

#if !CONFIG_METAL_NATIVE
extern GloContext *g_nv20_context_render;
extern GloContext *g_nv20_context_display;
#endif

unsigned int pgraph_metal_bind_inline_array(NV20State *d);
void pgraph_metal_bind_shaders(PGRAPHState *pg);
void pgraph_metal_bind_textures(NV20State *d);
void pgraph_metal_bind_vertex_attributes(NV20State *d, unsigned int min_element, unsigned int max_element, bool inline_data, unsigned int inline_stride, unsigned int provoking_element);
bool pgraph_metal_check_surface_to_texture_compatibility(const SurfaceBinding *surface, const TextureShape *shape);
#if CONFIG_METAL_NATIVE
void *pgraph_metal_compile_shader(const char *vs_src, const char *fs_src);
#else
GLuint pgraph_metal_compile_shader(const char *vs_src, const char *fs_src);
#endif
void pgraph_metal_download_dirty_surfaces(NV20State *d);
void pgraph_metal_clear_report_value(NV20State *d);
void pgraph_metal_clear_surface(NV20State *d, uint32_t parameter);
void pgraph_metal_draw_begin(NV20State *d);
void pgraph_metal_draw_end(NV20State *d);
void pgraph_metal_flush_draw(NV20State *d);
void pgraph_metal_get_report(NV20State *d, uint32_t parameter);
void pgraph_metal_image_blit(NV20State *d);
void pgraph_metal_mark_textures_possibly_dirty(NV20State *d, hwaddr addr, hwaddr size);
void pgraph_metal_process_pending_reports(NV20State *d);
void pgraph_metal_surface_flush(NV20State *d);
void pgraph_metal_surface_update(NV20State *d, bool upload, bool color_write, bool zeta_write);
void pgraph_metal_sync(NV20State *d);
void pgraph_metal_update_entire_memory_buffer(NV20State *d);
void pgraph_metal_init_display(NV20State *d);
void pgraph_metal_finalize_display(PGRAPHState *pg);
#if CONFIG_METAL_NATIVE
void pgraph_metal_init_native(NV20State *d);
void pgraph_metal_finalize_native(PGRAPHState *pg);
void pgraph_metal_sync_native(NV20State *d, SurfaceBinding *surface,
                              unsigned int width, unsigned int height);
void pgraph_metal_clear_surface_native(NV20State *d, SurfaceBinding *surface,
                                       uint32_t rgba);
#endif
void pgraph_metal_init_reports(NV20State *d);
void pgraph_metal_finalize_reports(PGRAPHState *pg);
void pgraph_metal_init_shaders(PGRAPHState *pg);
void pgraph_metal_finalize_shaders(PGRAPHState *pg);
void pgraph_metal_init_surfaces(PGRAPHState *pg);
void pgraph_metal_finalize_surfaces(PGRAPHState *pg);
void pgraph_metal_init_textures(NV20State *d);
void pgraph_metal_finalize_textures(PGRAPHState *pg);
void pgraph_metal_init_buffers(NV20State *d);
void pgraph_metal_finalize_buffers(PGRAPHState *pg);
void pgraph_metal_process_pending_downloads(NV20State *d);
void pgraph_metal_reload_surface_scale_factor(PGRAPHState *pg);
void pgraph_metal_render_surface_to_texture(NV20State *d, SurfaceBinding *surface, TextureBinding *texture, TextureShape *texture_shape, int texture_unit);
void pgraph_metal_set_surface_dirty(PGRAPHState *pg, bool color, bool zeta);
void pgraph_metal_surface_download_if_dirty(NV20State *d, SurfaceBinding *surface);
SurfaceBinding *pgraph_metal_surface_get(NV20State *d, hwaddr addr);
SurfaceBinding *pgraph_metal_surface_get_within(NV20State *d, hwaddr addr);
void pgraph_metal_surface_invalidate(NV20State *d, SurfaceBinding *e);
void pgraph_metal_unbind_surface(NV20State *d, bool color);
void pgraph_metal_upload_surface_data(NV20State *d, SurfaceBinding *surface, bool force);
void pgraph_metal_shader_cache_to_disk(ShaderBinding *snode);
bool pgraph_metal_shader_load_from_memory(ShaderBinding *snode);
void pgraph_metal_shader_write_cache_reload_list(PGRAPHState *pg);
void pgraph_metal_set_surface_scale_factor(NV20State *d, unsigned int scale);
unsigned int pgraph_metal_get_surface_scale_factor(NV20State *d);
#if CONFIG_METAL_NATIVE
NV20MtlTexture *pgraph_metal_get_framebuffer_surface(NV20State *d);
#else
int pgraph_metal_get_framebuffer_surface(NV20State *d);
#endif
void pgraph_metal_determine_gpu_properties(void);
GPUProperties *pgraph_metal_get_gpu_properties(void);

#endif
