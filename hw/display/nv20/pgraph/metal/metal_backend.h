/*
 * Native Metal backend for NV20 PGRAPH (Darwin).
 */
#ifndef HW_NV20_PGRAPH_METAL_BACKEND_H
#define HW_NV20_PGRAPH_METAL_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct NV20MetalBackend NV20MetalBackend;

typedef struct NV20MtlTexture NV20MtlTexture;
typedef struct NV20MtlBuffer NV20MtlBuffer;
typedef struct NV20MtlPipeline NV20MtlPipeline;
typedef struct NV20MtlSampler NV20MtlSampler;

typedef enum NV20MtlPrimitive {
    NV20_MTL_PRIM_POINTS = 0,
    NV20_MTL_PRIM_LINES,
    NV20_MTL_PRIM_LINE_LOOP,
    NV20_MTL_PRIM_LINE_STRIP,
    NV20_MTL_PRIM_TRIANGLES,
    NV20_MTL_PRIM_TRIANGLE_STRIP,
    NV20_MTL_PRIM_TRIANGLE_FAN,
    NV20_MTL_PRIM_LINES_ADJACENCY,
    NV20_MTL_PRIM_LINE_STRIP_ADJACENCY,
} NV20MtlPrimitive;

typedef enum NV20MtlIndexType {
    NV20_MTL_INDEX_UINT16 = 0,
    NV20_MTL_INDEX_UINT32,
} NV20MtlIndexType;

typedef struct NV20MtlRenderPassDesc {
    NV20MtlTexture *color;
    NV20MtlTexture *depth;
    uint32_t load_color;
    uint32_t load_depth;
    float clear_color[4];
    float clear_depth;
    uint32_t clear_stencil;
    unsigned width;
    unsigned height;
} NV20MtlRenderPassDesc;

typedef struct NV20MtlPipelineKey {
    void *vertex_lib;
    void *fragment_lib;
    void *geometry_lib;
    const char *vs_entry;
    const char *fs_entry;
    const char *gs_entry;
    uint32_t color_format;
    uint32_t depth_format;
    uint32_t blend_enabled;
    uint32_t blend_src;
    uint32_t blend_dst;
    uint32_t blend_eq;
    uint32_t depth_test;
    uint32_t depth_write;
    uint32_t depth_func;
    uint32_t stencil_test;
    uint32_t stencil_ref;
    uint32_t stencil_read_mask;
    uint32_t stencil_write_mask;
    uint32_t stencil_fail;
    uint32_t stencil_zfail;
    uint32_t stencil_zpass;
    uint32_t stencil_func;
    uint32_t cull_mode;
    uint32_t front_face_cw;
    uint32_t color_mask;
    uint32_t primitive_type;
} NV20MtlPipelineKey;

typedef struct NV20MtlVertexAttr {
    NV20MtlBuffer *buffer;
    unsigned offset;
    unsigned format; /* MTLVertexFormat as integer */
    unsigned buffer_index;
} NV20MtlVertexAttr;

NV20MetalBackend *nv20_mtl_backend_create(void *vram_ptr, size_t vram_size);
void nv20_mtl_backend_destroy(NV20MetalBackend *be);
bool nv20_mtl_backend_is_available(const NV20MetalBackend *be);
NV20MetalBackend *nv20_mtl_backend_from_context(void *ctx);

/* VRAM compute helpers (display composite / clear) */
void nv20_mtl_composite_surface(NV20MetalBackend *be,
                                uint32_t src_offset, uint32_t src_pitch,
                                uint32_t src_width, uint32_t src_height,
                                uint32_t dst_offset, uint32_t dst_pitch,
                                uint32_t dst_width, uint32_t dst_height,
                                uint32_t bytes_per_pixel);
void nv20_mtl_clear_region(NV20MetalBackend *be, uint32_t offset,
                           uint32_t width, uint32_t height,
                           uint32_t pitch, uint32_t bytes_per_pixel,
                           uint32_t rgba);

/* Textures */
NV20MtlTexture *nv20_mtl_texture_create(NV20MetalBackend *be,
                                        uint32_t pixel_format,
                                        unsigned width, unsigned height,
                                        unsigned depth, unsigned levels,
                                        bool cube);
void nv20_mtl_texture_destroy(NV20MtlTexture *tex);
void nv20_mtl_texture_upload_level(NV20MetalBackend *be, NV20MtlTexture *tex,
                                   unsigned level, unsigned width,
                                   unsigned height, unsigned depth,
                                   unsigned pitch, const void *data,
                                   unsigned bytes_per_pixel, bool linear);
void nv20_mtl_texture_upload_swizzled(NV20MetalBackend *be, NV20MtlTexture *tex,
                                      unsigned level, unsigned width,
                                      unsigned height, unsigned pitch,
                                      const void *data,
                                      unsigned bytes_per_pixel);
bool nv20_mtl_texture_read_pixels(NV20MetalBackend *be, NV20MtlTexture *tex,
                                  void *dst, unsigned width, unsigned height,
                                  unsigned dst_pitch,
                                  unsigned bytes_per_pixel);
void *nv20_mtl_texture_get_handle(NV20MtlTexture *tex);
uint32_t nv20_mtl_texture_get_format(NV20MtlTexture *tex);
unsigned nv20_mtl_texture_get_width(NV20MtlTexture *tex);
unsigned nv20_mtl_texture_get_height(NV20MtlTexture *tex);

/* Samplers */
NV20MtlSampler *nv20_mtl_sampler_get(NV20MetalBackend *be,
                                     unsigned min_filter, unsigned mag_filter,
                                     unsigned addru, unsigned addrv,
                                     unsigned addrp, float lod_bias,
                                     float max_anisotropy, uint32_t border_color,
                                     bool border_color_set);
void nv20_mtl_sampler_release(NV20MtlSampler *sampler);

/* Buffers */
NV20MtlBuffer *nv20_mtl_buffer_create(NV20MetalBackend *be, size_t size,
                                      bool cpu_access);
void nv20_mtl_buffer_destroy(NV20MtlBuffer *buf);
void *nv20_mtl_buffer_map(NV20MtlBuffer *buf);
void nv20_mtl_buffer_unmap(NV20MtlBuffer *buf);
void nv20_mtl_buffer_write(NV20MtlBuffer *buf, size_t offset,
                           const void *data, size_t size);
void *nv20_mtl_buffer_get_handle(NV20MtlBuffer *buf);

/* Shader libraries / pipelines */
void *nv20_mtl_compile_library(NV20MetalBackend *be, const char *msl_source);
void nv20_mtl_release_library(NV20MetalBackend *be, void *library);
NV20MtlPipeline *nv20_mtl_pipeline_get(NV20MetalBackend *be,
                                       const NV20MtlPipelineKey *key);
void nv20_mtl_pipeline_release(NV20MtlPipeline *pipeline);
void *nv20_mtl_pipeline_get_handle(NV20MtlPipeline *pipeline);

/* Render pass / draw */
void nv20_mtl_begin_frame(NV20MetalBackend *be);
void nv20_mtl_end_frame(NV20MetalBackend *be);
void nv20_mtl_begin_render_pass(NV20MetalBackend *be,
                                const NV20MtlRenderPassDesc *desc);
void nv20_mtl_end_render_pass(NV20MetalBackend *be);
void nv20_mtl_set_pipeline(NV20MetalBackend *be, NV20MtlPipeline *pipeline);
void nv20_mtl_set_viewport(NV20MetalBackend *be, double x, double y,
                           double w, double h);
void nv20_mtl_set_scissor(NV20MetalBackend *be, unsigned x, unsigned y,
                          unsigned w, unsigned h);
void nv20_mtl_set_vertex_buffer(NV20MetalBackend *be, unsigned index,
                                NV20MtlBuffer *buf, unsigned offset);
void nv20_mtl_set_vertex_bytes(NV20MetalBackend *be, unsigned index,
                               const void *data, size_t size);
void nv20_mtl_set_fragment_texture(NV20MetalBackend *be, unsigned index,
                                   NV20MtlTexture *tex, NV20MtlSampler *sampler);
void nv20_mtl_set_fragment_bytes(NV20MetalBackend *be, unsigned index,
                                 const void *data, size_t size);
void nv20_mtl_set_vertex_bytes_uniform(NV20MetalBackend *be, unsigned index,
                                       const void *data, size_t size);
void nv20_mtl_draw(NV20MetalBackend *be, NV20MtlPrimitive prim,
                   unsigned first, unsigned count);
void nv20_mtl_draw_indexed(NV20MetalBackend *be, NV20MtlPrimitive prim,
                           NV20MtlIndexType index_type,
                           NV20MtlBuffer *index_buf, unsigned index_count,
                           unsigned index_offset);
void nv20_mtl_draw_indexed_cpu(NV20MetalBackend *be, NV20MtlPrimitive prim,
                               NV20MtlIndexType index_type,
                               const void *indices, unsigned index_count);
void nv20_mtl_wait_idle(NV20MetalBackend *be);

/* Display texture for QEMU UI */
NV20MtlTexture *nv20_mtl_display_texture_get(NV20MetalBackend *be,
                                             uint32_t pixel_format,
                                             unsigned width, unsigned height);
void nv20_mtl_display_blit(NV20MetalBackend *be, NV20MtlTexture *src,
                           unsigned src_w, unsigned src_h,
                           float line_offset, bool pvideo_enable,
                           NV20MtlTexture *pvideo_tex,
                           float pvideo_params[16]);

#endif /* HW_NV20_PGRAPH_METAL_BACKEND_H */
