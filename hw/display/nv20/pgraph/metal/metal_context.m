/*
 * Native Metal rendering context for NV20 PGRAPH (Darwin).
 */
#import <Metal/Metal.h>
#include "qemu/osdep.h"
#include "metal_context.h"
#include <string.h>

struct NV20MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> composite_pso;
    id<MTLComputePipelineState> clear_pso;
    id<MTLBuffer> vram_buf;
    void *vram_ptr;
    size_t vram_size;
};

static NSString *const kNV20MetalShaderSrc = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct CompositeParams {\n"
"    uint src_offset;\n"
"    uint src_pitch;\n"
"    uint src_width;\n"
"    uint src_height;\n"
"    uint dst_offset;\n"
"    uint dst_pitch;\n"
"    uint dst_width;\n"
"    uint dst_height;\n"
"    uint bpp;\n"
"};\n"
"\n"
"kernel void nv20_composite_vram(\n"
"    device uchar *vram [[buffer(0)]],\n"
"    constant CompositeParams &p [[buffer(1)]],\n"
"    uint2 gid [[thread_position_in_grid]])\n"
"{\n"
"    if (gid.x >= p.dst_width || gid.y >= p.dst_height) {\n"
"        return;\n"
"    }\n"
"    uint sy = gid.y * p.src_height / max(p.dst_height, 1u);\n"
"    uint sx = gid.x * p.src_width / max(p.dst_width, 1u);\n"
"    uint src_idx = p.src_offset + sy * p.src_pitch + sx * p.bpp;\n"
"    uint dst_idx = p.dst_offset + gid.y * p.dst_pitch + gid.x * p.bpp;\n"
"    for (uint i = 0; i < p.bpp; i++) {\n"
"        vram[dst_idx + i] = vram[src_idx + i];\n"
"    }\n"
"}\n"
"\n"
"struct ClearParams {\n"
"    uint offset;\n"
"    uint pitch;\n"
"    uint width;\n"
"    uint height;\n"
"    uint bpp;\n"
"    uint rgba;\n"
"};\n"
"\n"
"kernel void nv20_clear_vram(\n"
"    device uchar *vram [[buffer(0)]],\n"
"    constant ClearParams &p [[buffer(1)]],\n"
"    uint2 gid [[thread_position_in_grid]])\n"
"{\n"
"    if (gid.x >= p.width || gid.y >= p.height) {\n"
"        return;\n"
"    }\n"
"    uint idx = p.offset + gid.y * p.pitch + gid.x * p.bpp;\n"
"    for (uint i = 0; i < p.bpp; i++) {\n"
"        vram[idx + i] = (p.rgba >> (8u * i)) & 0xffu;\n"
"    }\n"
"}\n";

static BOOL nv20_metal_build_pipelines(NV20MetalContext *ctx)
{
    NSError *err = nil;

    ctx->library = [ctx->device newLibraryWithSource:kNV20MetalShaderSrc
                                             options:nil
                                               error:&err];
    if (!ctx->library) {
        NSLog(@"nv20: Metal shader compile failed: %@", err);
        return NO;
    }

    id<MTLFunction> composite_fn =
        [ctx->library newFunctionWithName:@"nv20_composite_vram"];
    id<MTLFunction> clear_fn =
        [ctx->library newFunctionWithName:@"nv20_clear_vram"];

    ctx->composite_pso =
        [ctx->device newComputePipelineStateWithFunction:composite_fn
                                                   error:&err];
    if (!ctx->composite_pso) {
        NSLog(@"nv20: Metal composite PSO failed: %@", err);
        return NO;
    }

    ctx->clear_pso =
        [ctx->device newComputePipelineStateWithFunction:clear_fn error:&err];
    if (!ctx->clear_pso) {
        NSLog(@"nv20: Metal clear PSO failed: %@", err);
        return NO;
    }

    return YES;
}

NV20MetalContext *nv20_metal_context_create(void *vram_ptr, size_t vram_size)
{
    NV20MetalContext *ctx = g_malloc0(sizeof(*ctx));

    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        g_free(ctx);
        return NULL;
    }

    ctx->queue = [ctx->device newCommandQueue];
    ctx->vram_ptr = vram_ptr;
    ctx->vram_size = vram_size;

    ctx->vram_buf = [ctx->device newBufferWithBytesNoCopy:vram_ptr
                                                   length:vram_size
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    if (!ctx->vram_buf) {
        nv20_metal_context_destroy(ctx);
        return NULL;
    }

    if (!nv20_metal_build_pipelines(ctx)) {
        nv20_metal_context_destroy(ctx);
        return NULL;
    }

    return ctx;
}

void nv20_metal_context_destroy(NV20MetalContext *ctx)
{
    if (!ctx) {
        return;
    }

    ctx->composite_pso = nil;
    ctx->clear_pso = nil;
    ctx->library = nil;
    ctx->vram_buf = nil;
    ctx->queue = nil;
    ctx->device = nil;
    g_free(ctx);
}

bool nv20_metal_context_is_available(const NV20MetalContext *ctx)
{
    return ctx && ctx->device && ctx->composite_pso && ctx->clear_pso;
}

static void nv20_metal_dispatch(NV20MetalContext *ctx,
                                id<MTLComputePipelineState> pso,
                                const void *params, size_t params_size,
                                uint32_t width, uint32_t height)
{
    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:pso];
    [enc setBuffer:ctx->vram_buf offset:0 atIndex:0];
    [enc setBytes:params length:params_size atIndex:1];

    MTLSize threads = MTLSizeMake(width, height, 1);
    MTLSize tg = MTLSizeMake(16, 16, 1);
    [enc dispatchThreads:threads threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

void nv20_metal_composite_surface(NV20MetalContext *ctx,
                                  uint32_t src_offset, uint32_t src_pitch,
                                  uint32_t src_width, uint32_t src_height,
                                  uint32_t dst_offset, uint32_t dst_pitch,
                                  uint32_t dst_width, uint32_t dst_height,
                                  uint32_t bytes_per_pixel)
{
    struct {
        uint32_t src_offset, src_pitch, src_width, src_height;
        uint32_t dst_offset, dst_pitch, dst_width, dst_height;
        uint32_t bpp;
    } params = {
        src_offset, src_pitch, src_width, src_height,
        dst_offset, dst_pitch, dst_width, dst_height,
        bytes_per_pixel,
    };

    if (!nv20_metal_context_is_available(ctx) || !dst_width || !dst_height) {
        return;
    }

    nv20_metal_dispatch(ctx, ctx->composite_pso, &params, sizeof(params),
                        dst_width, dst_height);
}

void nv20_metal_clear_region(NV20MetalContext *ctx, uint32_t offset,
                             uint32_t width, uint32_t height,
                             uint32_t pitch, uint32_t bytes_per_pixel,
                             uint32_t rgba)
{
    struct {
        uint32_t offset, pitch, width, height, bpp, rgba;
    } params = { offset, pitch, width, height, bytes_per_pixel, rgba };

    if (!nv20_metal_context_is_available(ctx) || !width || !height) {
        return;
    }

    nv20_metal_dispatch(ctx, ctx->clear_pso, &params, sizeof(params),
                        width, height);
}
