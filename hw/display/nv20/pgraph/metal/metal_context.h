/*
 * Native Metal rendering context for NV20 PGRAPH (Darwin).
 */
#ifndef HW_NV20_PGRAPH_METAL_CONTEXT_H
#define HW_NV20_PGRAPH_METAL_CONTEXT_H

#include <stdbool.h>
#include <stdint.h>

typedef struct NV20MetalContext NV20MetalContext;

NV20MetalContext *nv20_metal_context_create(void *vram_ptr, size_t vram_size);
void nv20_metal_context_destroy(NV20MetalContext *ctx);

bool nv20_metal_context_is_available(const NV20MetalContext *ctx);

/*
 * Composite a linear surface region in guest VRAM into the CRTC framebuffer
 * region, applying basic scaling for the QEMU display path.
 */
void nv20_metal_composite_surface(NV20MetalContext *ctx,
                                  uint32_t src_offset, uint32_t src_pitch,
                                  uint32_t src_width, uint32_t src_height,
                                  uint32_t dst_offset, uint32_t dst_pitch,
                                  uint32_t dst_width, uint32_t dst_height,
                                  uint32_t bytes_per_pixel);

/*
 * Clear a VRAM region via Metal blit encoder (first native 3D primitive).
 */
void nv20_metal_clear_region(NV20MetalContext *ctx, uint32_t offset,
                             uint32_t width, uint32_t height,
                             uint32_t pitch, uint32_t bytes_per_pixel,
                             uint32_t rgba);

#endif /* HW_NV20_PGRAPH_METAL_CONTEXT_H */
