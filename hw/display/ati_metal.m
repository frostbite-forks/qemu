/*
 * ati_metal.m — vgpu3d Metal rendering backend
 *
 * Compiled only on Darwin (aarch64-apple-darwin).
 * Called from ati_vgpu3d_thread (a QemuThread / POSIX thread).
 *
 * Presentation model:
 *   Renders directly into guest VRAM via a MTLTexture whose backing
 *   MTLBuffer wraps s->vga.vram_ptr (Apple Silicon unified memory,
 *   zero copy). After [commandBuffer waitUntilCompleted] the GPU has
 *   written the rendered frame into VRAM. QEMU's periodic VGA refresh
 *   picks it up and updates the display in the existing QEMU window.
 *   No AppKit. No NSWindow. No CAMetalLayer. No second window.
 */

#import <Metal/Metal.h>
#include "ati_vgpu3d.h"
#include <string.h>
#include <unistd.h>

/* ---- Metal MSL shaders -------------------------------------------------- */

static NSString *const kGouraudShaderSrc = @""
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct VIn {\n"
"    float3 pos   [[attribute(0)]];\n"
"    float4 color [[attribute(1)]];\n"
"};\n"
"\n"
"struct VOut {\n"
"    float4 pos [[position]];\n"
"    float4 color;\n"
"};\n"
"\n"
"struct Viewport { float2 size; };\n"
"\n"
"vertex VOut vgpu3d_vert_gouraud(\n"
"    VIn in [[stage_in]],\n"
"    constant Viewport &vp [[buffer(1)]])\n"
"{\n"
"    VOut out;\n"
"    out.pos = float4(\n"
"         2.0f * in.pos.x / vp.size.x - 1.0f,\n"
"         1.0f - 2.0f * in.pos.y / vp.size.y,\n"
"         in.pos.z,\n"
"         1.0f);\n"
"    out.color = in.color;\n"
"    return out;\n"
"}\n"
"\n"
"fragment float4 vgpu3d_frag_gouraud(VOut in [[stage_in]]) {\n"
"    return in.color;\n"
"}\n";

/* ---- Internal state ----------------------------------------------------- */

typedef struct {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLRenderPipelineState> gouraud_pso;
    id<MTLBuffer>              vram_buf;      /* MTLBuffer over guest VRAM  */
    id<MTLTexture>             vram_texture;  /* texture backed by vram_buf */
    void                      *ring_mem;
    uint32_t                   viewport_w;
    uint32_t                   viewport_h;
    vgpu3d_vtx_gouraud        *vtx_buf;
    uint32_t                   vtx_count;
    uint32_t                   vtx_cap;
} VGpu3dMetal;

static VGpu3dMetal g_metal;

/* ---- Vertex buffer helpers ---------------------------------------------- */

static void vtx_reset(void) { g_metal.vtx_count = 0; }

static void vtx_push(const vgpu3d_vtx_gouraud *v)
{
    if (g_metal.vtx_count >= g_metal.vtx_cap) {
        uint32_t newcap = g_metal.vtx_cap ? g_metal.vtx_cap * 2 : 4096;
        g_metal.vtx_buf = realloc(g_metal.vtx_buf,
                                  newcap * sizeof(vgpu3d_vtx_gouraud));
        g_metal.vtx_cap = newcap;
    }
    g_metal.vtx_buf[g_metal.vtx_count++] = *v;
}

/* ---- Pipeline creation -------------------------------------------------- */

static BOOL build_pipeline(void)
{
    NSError *err = nil;
    id<MTLLibrary> lib =
        [g_metal.device newLibraryWithSource:kGouraudShaderSrc
                                     options:nil
                                       error:&err];
    if (!lib) {
        NSLog(@"vgpu3d: shader compile error: %@", err);
        return NO;
    }

    MTLRenderPipelineDescriptor *desc =
        [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction   = [lib newFunctionWithName:@"vgpu3d_vert_gouraud"];
    desc.fragmentFunction = [lib newFunctionWithName:@"vgpu3d_frag_gouraud"];
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    desc.colorAttachments[0].blendingEnabled             = YES;
    desc.colorAttachments[0].sourceRGBBlendFactor        = MTLBlendFactorSourceAlpha;
    desc.colorAttachments[0].destinationRGBBlendFactor   = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].sourceAlphaBlendFactor      = MTLBlendFactorOne;
    desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

    MTLVertexDescriptor *vdesc = [[MTLVertexDescriptor alloc] init];
    vdesc.attributes[0].format      = MTLVertexFormatFloat3;
    vdesc.attributes[0].offset      = 0;
    vdesc.attributes[0].bufferIndex = 0;
    vdesc.attributes[1].format      = MTLVertexFormatFloat4;
    vdesc.attributes[1].offset      = 12;
    vdesc.attributes[1].bufferIndex = 0;
    vdesc.layouts[0].stride         = sizeof(vgpu3d_vtx_gouraud);
    desc.vertexDescriptor = vdesc;

    g_metal.gouraud_pso =
        [g_metal.device newRenderPipelineStateWithDescriptor:desc error:&err];
    if (!g_metal.gouraud_pso) {
        NSLog(@"vgpu3d: PSO error: %@", err);
        return NO;
    }
    return YES;
}

/* ---- Public C interface ------------------------------------------------- */

void vgpu3d_metal_init(void *ring_mem, void *vram_ptr)
{
    memset(&g_metal, 0, sizeof(g_metal));
    g_metal.ring_mem   = ring_mem;
    g_metal.viewport_w = 1680;
    g_metal.viewport_h = 1050;

    g_metal.device = MTLCreateSystemDefaultDevice();
    if (!g_metal.device) {
        NSLog(@"vgpu3d: no Metal device");
        return;
    }
    g_metal.queue = [g_metal.device newCommandQueue];

    /* Wrap guest VRAM in a Metal shared buffer (unified memory, zero copy). */
    NSUInteger fb_bytes = (NSUInteger)(g_metal.viewport_w *
                                       g_metal.viewport_h * 4);
    NSUInteger page    = (NSUInteger)getpagesize();
    NSUInteger aligned = (fb_bytes + page - 1) & ~(page - 1);
    g_metal.vram_buf =
        [g_metal.device newBufferWithBytesNoCopy:vram_ptr
                                          length:aligned
                                         options:MTLResourceStorageModeShared
                                     deallocator:nil];
    if (!g_metal.vram_buf) {
        NSLog(@"vgpu3d: failed to wrap VRAM as MTLBuffer");
        return;
    }

    /* Texture backed by vram_buf — rendering writes straight to guest VRAM. */
    MTLTextureDescriptor *td =
        [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                         width:g_metal.viewport_w
                                        height:g_metal.viewport_h
                                     mipmapped:NO];
    td.usage       = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModeShared;
    g_metal.vram_texture =
        [g_metal.vram_buf newTextureWithDescriptor:td
                                            offset:0
                                       bytesPerRow:g_metal.viewport_w * 4];
    if (!g_metal.vram_texture) {
        NSLog(@"vgpu3d: failed to create VRAM-backed texture");
        return;
    }

    if (!build_pipeline()) {
        NSLog(@"vgpu3d: pipeline build failed");
    }
}

void vgpu3d_metal_dispatch(void *ring_mem)
{
    if (!g_metal.device || !g_metal.vram_texture) {
        return;
    }

    uint8_t  *data = vgpu3d_ring_data(ring_mem);
    uint32_t  head = vgpu3d_ring_head(ring_mem);
    uint32_t  tail = vgpu3d_ring_tail(ring_mem);

    vtx_reset();

    while (tail != head) {
        const vgpu3d_pkt_hdr *hdr =
            (const vgpu3d_pkt_hdr *)(data + tail);
        uint16_t type = be16_to_cpu(hdr->type);
        uint16_t len  = be16_to_cpu(hdr->len);

        if (len < sizeof(vgpu3d_pkt_hdr)) {
            tail = head;
            break;
        }

        if (type == VGPU3D_PKT_NOP) {
            tail = 0;
            head = vgpu3d_ring_head(ring_mem);
            continue;
        }

        if (type == VGPU3D_PKT_SET_VIEWPORT) {
            const vgpu3d_pkt_set_viewport *pkt =
                (const vgpu3d_pkt_set_viewport *)(data + tail);
            g_metal.viewport_w = be16_to_cpu(pkt->width);
            g_metal.viewport_h = be16_to_cpu(pkt->height);

        } else if (type == VGPU3D_PKT_DRAW_TRIS) {
            const vgpu3d_pkt_draw_tris *pkt =
                (const vgpu3d_pkt_draw_tris *)(data + tail);
            uint16_t nv = be16_to_cpu(pkt->vertex_count);
            const vgpu3d_vtx_gouraud *vsrc =
                (const vgpu3d_vtx_gouraud *)(pkt + 1);

            for (uint16_t i = 0; i < nv; i++) {
                vgpu3d_vtx_gouraud v;
                uint32_t tmp;
#define BEFL(dst, src) \
    memcpy(&tmp, &(src), 4); tmp = be32_to_cpu(tmp); memcpy(&(dst), &tmp, 4)
                BEFL(v.x, vsrc[i].x); BEFL(v.y, vsrc[i].y);
                BEFL(v.z, vsrc[i].z); BEFL(v.r, vsrc[i].r);
                BEFL(v.g, vsrc[i].g); BEFL(v.b, vsrc[i].b);
                BEFL(v.a, vsrc[i].a);
#undef BEFL
                v.fog = 1.0f;
                vtx_push(&v);
            }

        } else if (type == VGPU3D_PKT_PRESENT) {

            @autoreleasepool {
                MTLRenderPassDescriptor *rpd =
                    [MTLRenderPassDescriptor renderPassDescriptor];
                rpd.colorAttachments[0].texture     = g_metal.vram_texture;
                rpd.colorAttachments[0].loadAction  = MTLLoadActionClear;
                rpd.colorAttachments[0].clearColor  =
                    MTLClearColorMake(0, 0, 0, 1);
                rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

                id<MTLCommandBuffer> cb = [g_metal.queue commandBuffer];
                id<MTLRenderCommandEncoder> enc =
                    [cb renderCommandEncoderWithDescriptor:rpd];

                if (g_metal.vtx_count > 0) {
                    [enc setRenderPipelineState:g_metal.gouraud_pso];
                    id<MTLBuffer> vbuf =
                        [g_metal.device
                            newBufferWithBytes:g_metal.vtx_buf
                                       length:g_metal.vtx_count *
                                              sizeof(vgpu3d_vtx_gouraud)
                                      options:MTLResourceStorageModeShared];
                    float vp[2] = { (float)g_metal.viewport_w,
                                    (float)g_metal.viewport_h };
                    [enc setVertexBuffer:vbuf offset:0 atIndex:0];
                    [enc setVertexBytes:vp length:sizeof(vp) atIndex:1];
                    [enc drawPrimitives:MTLPrimitiveTypeTriangle
                           vertexStart:0
                           vertexCount:g_metal.vtx_count];
                }

                [enc endEncoding];
                [cb commit];
                /* Wait for GPU — vram_texture is backed by vram_buf which
                 * wraps guest VRAM, so after this the rendered frame is in
                 * VRAM and QEMU's VGA refresh shows it in the main window. */
                [cb waitUntilCompleted];
            }

            vtx_reset();
        }

        tail = vgpu3d_pkt_next(tail, hdr->len);
        vgpu3d_ring_set_tail(ring_mem, tail);
    }
}

void vgpu3d_metal_cleanup(void)
{
    free(g_metal.vtx_buf);
    memset(&g_metal, 0, sizeof(g_metal));
}
