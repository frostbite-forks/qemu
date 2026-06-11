/*
 * Native Metal backend for NV20 PGRAPH (Darwin).
 */
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include "qemu/osdep.h"
#include "hw/display/nv20/pgraph/swizzle.h"
#include "metal_backend.h"
#include "metal_formats.h"
#include <string.h>

struct NV20MtlTexture {
    id<MTLTexture> tex;
    uint32_t pixel_format;
    unsigned width;
    unsigned height;
};

struct NV20MtlBuffer {
    id<MTLBuffer> buf;
    size_t size;
};

struct NV20MtlSampler {
    id<MTLSamplerState> state;
    int refcnt;
};

struct NV20MtlPipeline {
    id<MTLRenderPipelineState> pso;
    id<MTLDepthStencilState> dss;
    int refcnt;
};

struct NV20MetalBackend {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> cmd;
    id<MTLRenderCommandEncoder> enc;
    id<MTLLibrary> util_library;
    id<MTLComputePipelineState> composite_pso;
    id<MTLComputePipelineState> clear_pso;
    id<MTLBuffer> vram_buf;
    void *vram_ptr;
    size_t vram_size;

    NV20MtlTexture *display_tex;
    id<MTLRenderPipelineState> display_pso;
    id<MTLLibrary> display_lib;

    GHashTable *pipeline_cache;
    GHashTable *sampler_cache;
    NV20MtlPipeline *current_pipeline;
};

static NSString *const kNV20UtilShaderSrc = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct CompositeParams {\n"
"    uint src_offset; uint src_pitch; uint src_width; uint src_height;\n"
"    uint dst_offset; uint dst_pitch; uint dst_width; uint dst_height; uint bpp;\n"
"};\n"
"kernel void nv20_composite_vram(device uchar *vram [[buffer(0)]],\n"
"    constant CompositeParams &p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {\n"
"    if (gid.x >= p.dst_width || gid.y >= p.dst_height) return;\n"
"    uint sy = gid.y * p.src_height / max(p.dst_height, 1u);\n"
"    uint sx = gid.x * p.src_width / max(p.dst_width, 1u);\n"
"    uint src_idx = p.src_offset + sy * p.src_pitch + sx * p.bpp;\n"
"    uint dst_idx = p.dst_offset + gid.y * p.dst_pitch + gid.x * p.bpp;\n"
"    for (uint i = 0; i < p.bpp; i++) vram[dst_idx + i] = vram[src_idx + i];\n"
"}\n"
"struct ClearParams { uint offset, pitch, width, height, bpp, rgba; };\n"
"kernel void nv20_clear_vram(device uchar *vram [[buffer(0)]],\n"
"    constant ClearParams &p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {\n"
"    if (gid.x >= p.width || gid.y >= p.height) return;\n"
"    uint idx = p.offset + gid.y * p.pitch + gid.x * p.bpp;\n"
"    for (uint i = 0; i < p.bpp; i++) vram[idx + i] = (p.rgba >> (8u * i)) & 0xffu;\n"
"}\n";

static NSString *const kNV20DisplayShaderSrc = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct DispParams {\n"
"    float2 display_size; float line_offset; float pvideo_enable;\n"
"    float4 pvideo_pos; float3 pvideo_scale; float2 pvideo_in_pos;\n"
"    float pvideo_color_key_enable; float3 pvideo_color_key;\n"
"};\n"
"struct VOut { float4 position [[position]]; float2 texCoord; };\n"
"vertex VOut nv20_disp_vs(uint vid [[vertex_id]]) {\n"
"    float2 pos[3] = { float2(-1,-1), float2(3,-1), float2(-1,3) };\n"
"    VOut o; o.position = float4(pos[vid], 0, 1); o.texCoord = pos[vid]*0.5+0.5; return o;\n"
"}\n"
"fragment float4 nv20_disp_fs(VOut in [[stage_in]],\n"
"    texture2d<float> tex [[texture(0)]], sampler s [[sampler(0)]],\n"
"    texture2d<float> pvideo_tex [[texture(1)]], sampler ps [[sampler(1)]],\n"
"    constant DispParams &p [[buffer(0)]]) {\n"
"    float2 tc = in.texCoord;\n"
"    float rel = p.display_size.y / float(tex.get_height()) / p.line_offset;\n"
"    tc.y = rel * (1.0 - tc.y);\n"
"    float4 col = tex.sample(s, tc);\n"
"    if (p.pvideo_enable > 0.5) {\n"
"        float2 sc = in.texCoord * p.display_size - 0.5;\n"
"        float4 reg = float4(p.pvideo_pos.xy, p.pvideo_pos.xy + p.pvideo_pos.zw);\n"
"        if (all(sc >= reg.xy) && all(sc <= reg.zw)) {\n"
"            if (p.pvideo_color_key_enable < 0.5 || all(col.rgb == p.pvideo_color_key)) {\n"
"                float2 out_xy = (sc - p.pvideo_pos.xy) * p.pvideo_scale.z;\n"
"                float2 in_st = (p.pvideo_in_pos + out_xy * p.pvideo_scale.xy) / float2(pvideo_tex.get_width(), pvideo_tex.get_height());\n"
"                in_st.y *= -1.0; col = pvideo_tex.sample(ps, in_st);\n"
"            }\n"
"        }\n"
"    }\n"
"    return col;\n"
"}\n";

static MTLPixelFormat nv20_mtl_pf(uint32_t fmt)
{
    return (MTLPixelFormat)fmt;
}

static void nv20_mtl_end_encoder(NV20MetalBackend *be)
{
    if (be->enc) {
        [be->enc endEncoding];
        be->enc = nil;
    }
}

static void nv20_mtl_dispatch_compute(NV20MetalBackend *be,
                                      id<MTLComputePipelineState> pso,
                                      const void *params, size_t params_size,
                                      uint32_t w, uint32_t h)
{
    id<MTLCommandBuffer> cmd = [be->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:be->vram_buf offset:0 atIndex:0];
    [enc setBytes:params length:params_size atIndex:1];
    MTLSize threads = MTLSizeMake(w, h, 1);
    MTLSize tg = MTLSizeMake(16, 16, 1);
    [enc dispatchThreads:threads threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

static BOOL nv20_mtl_build_util(NV20MetalBackend *be)
{
    NSError *err = nil;
    be->util_library = [be->device newLibraryWithSource:kNV20UtilShaderSrc
                                                options:nil error:&err];
    if (!be->util_library) {
        return NO;
    }
    be->composite_pso = [be->device
        newComputePipelineStateWithFunction:[be->util_library newFunctionWithName:@"nv20_composite_vram"]
                                      error:&err];
    be->clear_pso = [be->device
        newComputePipelineStateWithFunction:[be->util_library newFunctionWithName:@"nv20_clear_vram"]
                                      error:&err];
    return be->composite_pso && be->clear_pso;
}

static BOOL nv20_mtl_build_display(NV20MetalBackend *be)
{
    NSError *err = nil;
    be->display_lib = [be->device newLibraryWithSource:kNV20DisplayShaderSrc
                                               options:nil error:&err];
    if (!be->display_lib) {
        return NO;
    }
    id<MTLFunction> vs = [be->display_lib newFunctionWithName:@"nv20_disp_vs"];
    id<MTLFunction> fs = [be->display_lib newFunctionWithName:@"nv20_disp_fs"];
    MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = vs;
    desc.fragmentFunction = fs;
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    be->display_pso = [be->device newRenderPipelineStateWithDescriptor:desc
                                                                 error:&err];
    return be->display_pso != nil;
}

NV20MetalBackend *nv20_mtl_backend_create(void *vram_ptr, size_t vram_size)
{
    NV20MetalBackend *be = g_malloc0(sizeof(*be));
    be->device = MTLCreateSystemDefaultDevice();
    if (!be->device) {
        g_free(be);
        return NULL;
    }
    be->queue = [be->device newCommandQueue];
    be->vram_ptr = vram_ptr;
    be->vram_size = vram_size;
    be->vram_buf = [be->device newBufferWithBytesNoCopy:vram_ptr
                                                 length:vram_size
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
    if (!be->vram_buf || !nv20_mtl_build_util(be) || !nv20_mtl_build_display(be)) {
        nv20_mtl_backend_destroy(be);
        return NULL;
    }
    be->pipeline_cache = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    be->sampler_cache = g_hash_table_new_full(g_str_hash, g_str_equal, g_free, NULL);
    return be;
}

void nv20_mtl_backend_destroy(NV20MetalBackend *be)
{
    if (!be) {
        return;
    }
    nv20_mtl_end_encoder(be);
    be->cmd = nil;
    if (be->display_tex) {
        nv20_mtl_texture_destroy(be->display_tex);
    }
    g_hash_table_destroy(be->pipeline_cache);
    g_hash_table_destroy(be->sampler_cache);
    be->display_pso = nil;
    be->display_lib = nil;
    be->composite_pso = nil;
    be->clear_pso = nil;
    be->util_library = nil;
    be->vram_buf = nil;
    be->queue = nil;
    be->device = nil;
    g_free(be);
}

bool nv20_mtl_backend_is_available(const NV20MetalBackend *be)
{
    return be && be->device && be->composite_pso;
}

void nv20_mtl_composite_surface(NV20MetalBackend *be, uint32_t src_offset,
                                uint32_t src_pitch, uint32_t src_width,
                                uint32_t src_height, uint32_t dst_offset,
                                uint32_t dst_pitch, uint32_t dst_width,
                                uint32_t dst_height, uint32_t bpp)
{
    struct {
        uint32_t src_offset, src_pitch, src_width, src_height;
        uint32_t dst_offset, dst_pitch, dst_width, dst_height, bpp;
    } params = { src_offset, src_pitch, src_width, src_height,
                 dst_offset, dst_pitch, dst_width, dst_height, bpp };
    if (!nv20_mtl_backend_is_available(be) || !dst_width || !dst_height) {
        return;
    }
    nv20_mtl_dispatch_compute(be, be->composite_pso, &params, sizeof(params),
                              dst_width, dst_height);
}

void nv20_mtl_clear_region(NV20MetalBackend *be, uint32_t offset,
                           uint32_t width, uint32_t height, uint32_t pitch,
                           uint32_t bpp, uint32_t rgba)
{
    struct { uint32_t offset, pitch, width, height, bpp, rgba; } params =
        { offset, pitch, width, height, bpp, rgba };
    if (!nv20_mtl_backend_is_available(be) || !width || !height) {
        return;
    }
    nv20_mtl_dispatch_compute(be, be->clear_pso, &params, sizeof(params),
                              width, height);
}

NV20MtlTexture *nv20_mtl_texture_create(NV20MetalBackend *be, uint32_t pixel_format,
                                        unsigned width, unsigned height,
                                        unsigned depth, unsigned levels,
                                        bool cube)
{
    NV20MtlTexture *t = g_malloc0(sizeof(*t));
    MTLTextureDescriptor *desc = [MTLTextureDescriptor new];
    desc.pixelFormat = nv20_mtl_pf(pixel_format);
    desc.width = width;
    desc.height = height;
    desc.depth = MAX(depth, 1);
    desc.mipmapLevelCount = MAX(levels, 1);
    desc.textureType = cube ? MTLTextureTypeCube : MTLTextureType2D;
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModeShared;
    t->tex = [be->device newTextureWithDescriptor:desc];
    t->pixel_format = pixel_format;
    t->width = width;
    t->height = height;
    return t;
}

void nv20_mtl_texture_destroy(NV20MtlTexture *tex)
{
    if (!tex) {
        return;
    }
    tex->tex = nil;
    g_free(tex);
}

void nv20_mtl_texture_upload_level(NV20MetalBackend *be, NV20MtlTexture *tex,
                                   unsigned level, unsigned width,
                                   unsigned height, unsigned depth,
                                   unsigned pitch, const void *data,
                                   unsigned bytes_per_pixel, bool linear)
{
    MTLRegion region = { {0, 0, 0}, {width, height, depth} };
    [tex->tex replaceRegion:region mipmapLevel:level slice:0
                withBytes:data bytesPerRow:pitch bytesPerImage:pitch * height];
    (void)be; (void)bytes_per_pixel; (void)linear;
}

void nv20_mtl_texture_upload_swizzled(NV20MetalBackend *be, NV20MtlTexture *tex,
                                      unsigned level, unsigned width,
                                      unsigned height, unsigned pitch,
                                      const void *data, unsigned bytes_per_pixel)
{
    size_t size = width * height * bytes_per_pixel;
    uint8_t *linear = g_malloc(size);
    unswizzle_rect(data, width, height, linear, pitch, bytes_per_pixel);
    nv20_mtl_texture_upload_level(be, tex, level, width, height, 1, width * bytes_per_pixel,
                                  linear, bytes_per_pixel, true);
    g_free(linear);
}

bool nv20_mtl_texture_read_pixels(NV20MetalBackend *be, NV20MtlTexture *tex,
                                  void *dst, unsigned width, unsigned height,
                                  unsigned dst_pitch, unsigned bytes_per_pixel)
{
    nv20_mtl_end_encoder(be);
    if (be->cmd) {
        [be->cmd commit];
        [be->cmd waitUntilCompleted];
        be->cmd = nil;
    }
    MTLRegion region = { {0, 0, 0}, {width, height, 1} };
    [tex->tex getBytes:dst bytesPerRow:dst_pitch
          fromRegion:region mipmapLevel:0];
    (void)bytes_per_pixel;
    return true;
}

void *nv20_mtl_texture_get_handle(NV20MtlTexture *tex)
{
    return tex ? (__bridge void *)tex->tex : NULL;
}

uint32_t nv20_mtl_texture_get_format(NV20MtlTexture *tex)
{
    return tex ? tex->pixel_format : 0;
}

unsigned nv20_mtl_texture_get_width(NV20MtlTexture *tex)
{
    return tex ? tex->width : 0;
}

unsigned nv20_mtl_texture_get_height(NV20MtlTexture *tex)
{
    return tex ? tex->height : 0;
}

static char *nv20_mtl_sampler_key(unsigned min_f, unsigned mag_f,
                                  unsigned addru, unsigned addrv,
                                  unsigned addrp, float lod_bias,
                                  float max_aniso, uint32_t border_color,
                                  bool border_set)
{
    return g_strdup_printf("%u:%u:%u:%u:%u:%.4f:%.2f:%08x:%d",
                           min_f, mag_f, addru, addrv, addrp,
                           lod_bias, max_aniso, border_color, border_set);
}

NV20MtlSampler *nv20_mtl_sampler_get(NV20MetalBackend *be,
                                     unsigned min_filter, unsigned mag_filter,
                                     unsigned addru, unsigned addrv,
                                     unsigned addrp, float lod_bias,
                                     float max_anisotropy, uint32_t border_color,
                                     bool border_color_set)
{
    g_autofree char *key = nv20_mtl_sampler_key(min_filter, mag_filter,
                                              addru, addrv, addrp, lod_bias,
                                              max_anisotropy, border_color,
                                              border_color_set);
    NV20MtlSampler *s = g_hash_table_lookup(be->sampler_cache, key);
    if (s) {
        s->refcnt++;
        return s;
    }
    s = g_malloc0(sizeof(*s));
    MTLSamplerDescriptor *desc = [MTLSamplerDescriptor new];
    desc.minFilter = (MTLSamplerMinMagFilter)nv20_mtl_sampler_min_filter(min_filter);
    desc.magFilter = (MTLSamplerMinMagFilter)nv20_mtl_sampler_mag_filter(mag_filter);
    desc.sAddressMode = (MTLSamplerAddressMode)nv20_mtl_sampler_address(addru);
    desc.tAddressMode = (MTLSamplerAddressMode)nv20_mtl_sampler_address(addrv);
    desc.rAddressMode = (MTLSamplerAddressMode)nv20_mtl_sampler_address(addrp);
    desc.lodMinClamp = lod_bias;
    desc.maxAnisotropy = MAX(1, (NSUInteger)max_anisotropy);
    if (border_color_set) {
        desc.borderColor = MTLSamplerBorderColorOpaqueWhite;
    }
    s->state = [be->device newSamplerStateWithDescriptor:desc];
    s->refcnt = 1;
    g_hash_table_insert(be->sampler_cache, g_strdup(key), s);
    return s;
}

void nv20_mtl_sampler_release(NV20MtlSampler *sampler)
{
    if (!sampler) {
        return;
    }
    if (--sampler->refcnt <= 0) {
        sampler->state = nil;
        g_free(sampler);
    }
}

NV20MtlBuffer *nv20_mtl_buffer_create(NV20MetalBackend *be, size_t size,
                                      bool cpu_access)
{
    NV20MtlBuffer *b = g_malloc0(sizeof(*b));
    b->size = size;
    MTLResourceOptions opts = cpu_access ? MTLResourceStorageModeShared
                                         : MTLResourceStorageModePrivate;
    b->buf = [be->device newBufferWithLength:size options:opts];
    return b;
}

void nv20_mtl_buffer_destroy(NV20MtlBuffer *buf)
{
    if (!buf) {
        return;
    }
    buf->buf = nil;
    g_free(buf);
}

void *nv20_mtl_buffer_map(NV20MtlBuffer *buf)
{
    return buf ? [buf->buf contents] : NULL;
}

void nv20_mtl_buffer_unmap(NV20MtlBuffer *buf)
{
    (void)buf;
}

void nv20_mtl_buffer_write(NV20MtlBuffer *buf, size_t offset,
                           const void *data, size_t size)
{
    memcpy((char *)[buf->buf contents] + offset, data, size);
}

void *nv20_mtl_buffer_get_handle(NV20MtlBuffer *buf)
{
    return buf ? (__bridge void *)buf->buf : NULL;
}

void *nv20_mtl_compile_library(NV20MetalBackend *be, const char *msl_source)
{
    NSError *err = nil;
    id<MTLLibrary> lib =
        [be->device newLibraryWithSource:[NSString stringWithUTF8String:msl_source]
                                 options:nil error:&err];
    if (!lib) {
        NSLog(@"nv20: MSL compile: %@", err);
        return NULL;
    }
    return (__bridge void *)lib;
}

void nv20_mtl_release_library(NV20MetalBackend *be, void *library)
{
    (void)be;
    (void)library;
}

static MTLPrimitiveType nv20_mtl_prim(NV20MtlPrimitive prim)
{
    static const MTLPrimitiveType map[] = {
        MTLPrimitiveTypePoint, MTLPrimitiveTypeLine, MTLPrimitiveTypeLine,
        MTLPrimitiveTypeLineStrip, MTLPrimitiveTypeTriangle,
        MTLPrimitiveTypeTriangleStrip, MTLPrimitiveTypeTriangle,
        MTLPrimitiveTypeLine, MTLPrimitiveTypeLineStrip,
    };
    return prim < ARRAY_SIZE(map) ? map[prim] : MTLPrimitiveTypeTriangle;
}

static char *nv20_mtl_pipeline_key_str(const NV20MtlPipelineKey *key)
{
    return g_strdup_printf("%p:%p:%p:%u:%u:%u:%u:%u",
                           key->vertex_lib, key->fragment_lib, key->geometry_lib,
                           key->color_format, key->depth_format, key->blend_enabled,
                           key->depth_test, key->cull_mode);
}

NV20MtlPipeline *nv20_mtl_pipeline_get(NV20MetalBackend *be,
                                       const NV20MtlPipelineKey *key)
{
    g_autofree char *k = nv20_mtl_pipeline_key_str(key);
    NV20MtlPipeline *p = g_hash_table_lookup(be->pipeline_cache, k);
    if (p) {
        p->refcnt++;
        return p;
    }
    p = g_malloc0(sizeof(*p));
    NSError *err = nil;
    id<MTLLibrary> vs_lib = (__bridge id<MTLLibrary>)key->vertex_lib;
    id<MTLLibrary> fs_lib = (__bridge id<MTLLibrary>)key->fragment_lib;
    id<MTLFunction> vs = [vs_lib newFunctionWithName:@(key->vs_entry ?: "nv20_vsh_main")];
    id<MTLFunction> fs = [fs_lib newFunctionWithName:@(key->fs_entry ?: "nv20_psh_main")];
    MTLRenderPipelineDescriptor *desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = vs;
    desc.fragmentFunction = fs;
    desc.colorAttachments[0].pixelFormat = nv20_mtl_pf(key->color_format);
    if (key->depth_format) {
        desc.depthAttachmentPixelFormat = nv20_mtl_pf(key->depth_format);
    }
    if (key->blend_enabled) {
        desc.colorAttachments[0].blendingEnabled = YES;
        desc.colorAttachments[0].sourceRGBBlendFactor =
            (MTLBlendFactor)nv20_mtl_blend_factor(key->blend_src);
        desc.colorAttachments[0].destinationRGBBlendFactor =
            (MTLBlendFactor)nv20_mtl_blend_factor(key->blend_dst);
        desc.colorAttachments[0].rgbBlendOperation =
            (MTLBlendOperation)nv20_mtl_blend_op(key->blend_eq);
        desc.colorAttachments[0].sourceAlphaBlendFactor =
            (MTLBlendFactor)nv20_mtl_blend_factor(key->blend_src);
        desc.colorAttachments[0].destinationAlphaBlendFactor =
            (MTLBlendFactor)nv20_mtl_blend_factor(key->blend_dst);
        desc.colorAttachments[0].alphaBlendOperation =
            (MTLBlendOperation)nv20_mtl_blend_op(key->blend_eq);
    }
    p->pso = [be->device newRenderPipelineStateWithDescriptor:desc error:&err];
    if (!p->pso) {
        NSLog(@"nv20: pipeline: %@", err);
        g_free(p);
        return NULL;
    }
    MTLDepthStencilDescriptor *dsd = [MTLDepthStencilDescriptor new];
    dsd.depthCompareFunction = key->depth_test ?
        (MTLCompareFunction)nv20_mtl_compare(key->depth_func) : MTLCompareFunctionAlways;
    dsd.depthWriteEnabled = key->depth_write;
    p->dss = [be->device newDepthStencilStateWithDescriptor:dsd];
    p->refcnt = 1;
    g_hash_table_insert(be->pipeline_cache, g_strdup(k), p);
    return p;
}

void nv20_mtl_pipeline_release(NV20MtlPipeline *pipeline)
{
    if (!pipeline) {
        return;
    }
    if (--pipeline->refcnt <= 0) {
        pipeline->pso = nil;
        pipeline->dss = nil;
        g_free(pipeline);
    }
}

void *nv20_mtl_pipeline_get_handle(NV20MtlPipeline *pipeline)
{
    return pipeline ? (__bridge void *)pipeline->pso : NULL;
}

void nv20_mtl_begin_frame(NV20MetalBackend *be)
{
    if (!be->cmd) {
        be->cmd = [be->queue commandBuffer];
    }
}

void nv20_mtl_end_frame(NV20MetalBackend *be)
{
    nv20_mtl_end_encoder(be);
    if (be->cmd) {
        [be->cmd commit];
        be->cmd = nil;
    }
}

void nv20_mtl_begin_render_pass(NV20MetalBackend *be,
                                const NV20MtlRenderPassDesc *desc)
{
    nv20_mtl_end_encoder(be);
    if (!be->cmd) {
        be->cmd = [be->queue commandBuffer];
    }
    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    if (desc->color) {
        rpd.colorAttachments[0].texture = desc->color->tex;
        rpd.colorAttachments[0].loadAction = desc->load_color ?
            MTLLoadActionLoad : MTLLoadActionClear;
        rpd.colorAttachments[0].clearColor = MTLClearColorMake(
            desc->clear_color[0], desc->clear_color[1],
            desc->clear_color[2], desc->clear_color[3]);
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    if (desc->depth) {
        rpd.depthAttachment.texture = desc->depth->tex;
        rpd.depthAttachment.loadAction = desc->load_depth ?
            MTLLoadActionLoad : MTLLoadActionClear;
        rpd.depthAttachment.clearDepth = desc->clear_depth;
        if (desc->clear_stencil || nv20_mtl_texture_get_format(desc->depth) == 255) {
            rpd.stencilAttachment.texture = desc->depth->tex;
            rpd.stencilAttachment.loadAction = rpd.depthAttachment.loadAction;
            rpd.stencilAttachment.storeAction = MTLStoreActionStore;
            rpd.stencilAttachment.clearStencil = desc->clear_stencil;
        }
        rpd.depthAttachment.storeAction = MTLStoreActionStore;
    }
    be->enc = [be->cmd renderCommandEncoderWithDescriptor:rpd];
}

void nv20_mtl_end_render_pass(NV20MetalBackend *be)
{
    nv20_mtl_end_encoder(be);
}

void nv20_mtl_set_pipeline(NV20MetalBackend *be, NV20MtlPipeline *pipeline)
{
    be->current_pipeline = pipeline;
    if (be->enc && pipeline) {
        [be->enc setRenderPipelineState:pipeline->pso];
        [be->enc setDepthStencilState:pipeline->dss];
    }
}

void nv20_mtl_set_viewport(NV20MetalBackend *be, double x, double y,
                           double w, double h)
{
    if (be->enc) {
        [be->enc setViewport:(MTLViewport){ x, y, w, h, 0, 1 }];
    }
}

void nv20_mtl_set_scissor(NV20MetalBackend *be, unsigned x, unsigned y,
                          unsigned w, unsigned h)
{
    if (be->enc) {
        [be->enc setScissorRect:(MTLScissorRect){ x, y, w, h }];
    }
}

void nv20_mtl_set_vertex_buffer(NV20MetalBackend *be, unsigned index,
                                NV20MtlBuffer *buf, unsigned offset)
{
    if (be->enc && buf) {
        [be->enc setVertexBuffer:buf->buf offset:offset atIndex:index];
    }
}

void nv20_mtl_set_vertex_bytes(NV20MetalBackend *be, unsigned index,
                               const void *data, size_t size)
{
    if (be->enc) {
        [be->enc setVertexBytes:data length:size atIndex:index];
    }
}

void nv20_mtl_set_fragment_texture(NV20MetalBackend *be, unsigned index,
                                   NV20MtlTexture *tex, NV20MtlSampler *sampler)
{
    if (be->enc && tex) {
        [be->enc setFragmentTexture:tex->tex atIndex:index];
        if (sampler) {
            [be->enc setFragmentSamplerState:sampler->state atIndex:index];
        }
    }
}

void nv20_mtl_set_fragment_bytes(NV20MetalBackend *be, unsigned index,
                                 const void *data, size_t size)
{
    if (be->enc) {
        [be->enc setFragmentBytes:data length:size atIndex:index];
    }
}

void nv20_mtl_set_vertex_bytes_uniform(NV20MetalBackend *be, unsigned index,
                                       const void *data, size_t size)
{
    nv20_mtl_set_vertex_bytes(be, index, data, size);
}

void nv20_mtl_draw(NV20MetalBackend *be, NV20MtlPrimitive prim,
                   unsigned first, unsigned count)
{
    if (be->enc) {
        [be->enc drawPrimitives:nv20_mtl_prim(prim)
                    vertexStart:first vertexCount:count];
    }
}

void nv20_mtl_draw_indexed(NV20MetalBackend *be, NV20MtlPrimitive prim,
                           NV20MtlIndexType index_type,
                           NV20MtlBuffer *index_buf, unsigned index_count,
                           unsigned index_offset)
{
    if (be->enc && index_buf) {
        MTLIndexType it = index_type == NV20_MTL_INDEX_UINT32 ?
            MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
        [be->enc drawIndexedPrimitives:nv20_mtl_prim(prim)
                            indexCount:index_count
                             indexType:it
                           indexBuffer:index_buf->buf
                     indexBufferOffset:index_offset * (it == MTLIndexTypeUInt32 ? 4 : 2)];
    }
}

void nv20_mtl_draw_indexed_cpu(NV20MetalBackend *be, NV20MtlPrimitive prim,
                               NV20MtlIndexType index_type,
                               const void *indices, unsigned index_count)
{
    size_t sz = index_count * (index_type == NV20_MTL_INDEX_UINT32 ? 4 : 2);
    NV20MtlBuffer *tmp = nv20_mtl_buffer_create(be, sz, true);
    nv20_mtl_buffer_write(tmp, 0, indices, sz);
    nv20_mtl_draw_indexed(be, prim, index_type, tmp, index_count, 0);
    nv20_mtl_buffer_destroy(tmp);
}

void nv20_mtl_wait_idle(NV20MetalBackend *be)
{
    nv20_mtl_end_encoder(be);
    if (be->cmd) {
        [be->cmd commit];
        [be->cmd waitUntilCompleted];
        be->cmd = nil;
    }
}

NV20MtlTexture *nv20_mtl_display_texture_get(NV20MetalBackend *be,
                                             uint32_t pixel_format,
                                             unsigned width, unsigned height)
{
    if (be->display_tex &&
        be->display_tex->width == width &&
        be->display_tex->height == height &&
        be->display_tex->pixel_format == pixel_format) {
        return be->display_tex;
    }
    if (be->display_tex) {
        nv20_mtl_texture_destroy(be->display_tex);
    }
    be->display_tex = nv20_mtl_texture_create(be, pixel_format, width, height, 1, 1, false);
    return be->display_tex;
}

void nv20_mtl_display_blit(NV20MetalBackend *be, NV20MtlTexture *src,
                           unsigned src_w, unsigned src_h,
                           float line_offset, bool pvideo_enable,
                           NV20MtlTexture *pvideo_tex,
                           float pvideo_params[16])
{
    NV20MtlTexture *dst = be->display_tex;
    if (!dst || !src) {
        return;
    }
    nv20_mtl_end_encoder(be);
    if (!be->cmd) {
        be->cmd = [be->queue commandBuffer];
    }
    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = dst->tex;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc = [be->cmd renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:be->display_pso];
    [enc setFragmentTexture:src->tex atIndex:0];
    if (pvideo_tex) {
        [enc setFragmentTexture:pvideo_tex->tex atIndex:1];
    }
    struct {
        float display_size[2];
        float line_offset;
        float pvideo_enable;
        float pvideo_pos[4];
        float pvideo_scale[3];
        float pvideo_in_pos[2];
        float pvideo_color_key_enable;
        float pvideo_color_key[3];
    } params = {
        .display_size = { (float)dst->width, (float)dst->height },
        .line_offset = line_offset,
        .pvideo_enable = pvideo_enable ? 1.f : 0.f,
    };
    if (pvideo_params) {
        memcpy(&params.pvideo_pos, pvideo_params, 4 * sizeof(float));
        memcpy(&params.pvideo_scale, pvideo_params + 4, 3 * sizeof(float));
        memcpy(&params.pvideo_in_pos, pvideo_params + 7, 2 * sizeof(float));
    }
    [enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
}
