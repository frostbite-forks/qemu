/*
 * ati_vgpu3d.h — vgpu3d wire protocol v1
 *
 * Shared contract between the Mac OS 9 guest CFM extension (sources/) and
 * the QEMU Metal dispatch thread (ati_metal.m).
 *
 * Byte order: BIG-ENDIAN throughout (PPC-native).
 * Host reads use be32_to_cpu() / be16_to_cpu().
 * Guest writes naturally with PPC big-endian stores — no byte-swap needed.
 *
 * Ring buffer layout (BAR2 + VGPU3D_RING_OFFSET, i.e. BAR2+0x10000):
 *
 *   +0x00  uint32_t head     guest enqueue ptr  (byte offset into data[])
 *   +0x04  uint32_t _pad[15] cache-line fill    (64 bytes total)
 *   +0x40  uint32_t tail     host dequeue ptr   (byte offset into data[])
 *   +0x44  uint32_t _pad[15] cache-line fill
 *   +0x80  uint8_t  data[]   packet stream starts here
 *
 * head and tail are byte offsets into data[], wrapping at VGPU3D_RING_CAP.
 * Ring is empty when head == tail.
 * All packets start at 8-byte-aligned offsets within data[].
 * Wrap: if a packet does not fit before VGPU3D_RING_CAP, the writer emits
 * a PKT_NOP spanning the remaining bytes, then wraps head to 0.
 */

#ifndef ATI_VGPU3D_H
#define ATI_VGPU3D_H

#include <stdint.h>

/* ---- Ring layout --------------------------------------------------------- */
#define VGPU3D_RING_HEAD_OFF    0x00u
#define VGPU3D_RING_TAIL_OFF    0x40u
#define VGPU3D_RING_DATA_OFF    0x80u
#define VGPU3D_RING_BYTES       (320u * 1024u)
#define VGPU3D_RING_CAP         (VGPU3D_RING_BYTES - VGPU3D_RING_DATA_OFF)

/* ---- Packet types -------------------------------------------------------- */
#define VGPU3D_PKT_NOP          0x0000u
#define VGPU3D_PKT_DRAW_TRIS    0x0001u
#define VGPU3D_PKT_DRAW_TEX     0x0002u
#define VGPU3D_PKT_UPLOAD_TEX   0x0003u
#define VGPU3D_PKT_FREE_TEX     0x0004u
#define VGPU3D_PKT_SET_VIEWPORT 0x0005u
#define VGPU3D_PKT_PRESENT      0x0006u

/* ---- Packet header (8 bytes, 8-byte aligned) ----------------------------- */
typedef struct {
    uint16_t type;
    uint16_t len;   /* total bytes incl. header, rounded up to 8 */
    uint32_t seq;
} vgpu3d_pkt_hdr;

/* ---- PKT_DRAW_TRIS ------------------------------------------------------- */
/*
 * Gouraud-shaded triangle list.  Vertices in RAVE screen space:
 *   x, y  = screen pixels (origin top-left, y increases downward)
 *   z     = depth [0.0 near .. 1.0 far]
 *   r,g,b,a = RGBA [0.0..1.0]
 *   fog   = fog factor [1.0 = no fog] — reserved v1
 */
typedef struct {
    float x, y, z;
    float r, g, b, a;
    float fog;
} vgpu3d_vtx_gouraud;   /* 36 bytes */

typedef struct {
    vgpu3d_pkt_hdr hdr;
    uint16_t       vertex_count;  /* multiple of 3 */
    uint16_t       _pad;
    /* vgpu3d_vtx_gouraud verts[vertex_count] follow */
} vgpu3d_pkt_draw_tris;

/* ---- PKT_DRAW_TEX -------------------------------------------------------- */
typedef struct {
    float x, y, z;
    float uOverW, vOverW, invW;
    float fog;
} vgpu3d_vtx_tex;       /* 28 bytes */

typedef struct {
    vgpu3d_pkt_hdr hdr;
    uint32_t       tex_id;
    uint16_t       vertex_count;
    uint16_t       _pad;
    /* vgpu3d_vtx_tex verts[vertex_count] follow */
} vgpu3d_pkt_draw_tex;

/* ---- PKT_UPLOAD_TEX ------------------------------------------------------ */
#define VGPU3D_TEX_FMT_ARGB1555  0u
#define VGPU3D_TEX_FMT_ARGB32    1u
#define VGPU3D_TEX_FMT_RGB565    2u

typedef struct {
    vgpu3d_pkt_hdr hdr;
    uint32_t       tex_id;
    uint16_t       width;
    uint16_t       height;
    uint8_t        format;
    uint8_t        _pad[3];
    /* texel data follows */
} vgpu3d_pkt_upload_tex;

#define VGPU3D_TEX_MAX  256u

/* ---- PKT_FREE_TEX -------------------------------------------------------- */
typedef struct {
    vgpu3d_pkt_hdr hdr;
    uint32_t       tex_id;
} vgpu3d_pkt_free_tex;

/* ---- PKT_SET_VIEWPORT ---------------------------------------------------- */
typedef struct {
    vgpu3d_pkt_hdr hdr;
    uint16_t       width;
    uint16_t       height;
} vgpu3d_pkt_set_viewport;

/* ---- PKT_PRESENT --------------------------------------------------------- */
/* No payload beyond hdr. */

/* ---- Host-side ring accessors ------------------------------------------- */
#ifndef VGPU3D_GUEST_BUILD

/* be32_to_cpu / be16_to_cpu / cpu_to_be32 come from qemu/osdep.h in ati.c.
 * In ati_metal.m (Objective-C, no qemu/osdep.h) we fall back to libkern. */
#ifndef be32_to_cpu
#include <libkern/OSByteOrder.h>
#define be32_to_cpu(v)  OSSwapBigToHostInt32(v)
#define be16_to_cpu(v)  OSSwapBigToHostInt16(v)
#define cpu_to_be32(v)  OSSwapHostToBigInt32(v)
#endif

static inline uint32_t vgpu3d_ring_head(const void *ring_mem)
{
    const uint32_t *p = (const uint32_t *)ring_mem;
    return be32_to_cpu(p[0]);
}

static inline uint32_t vgpu3d_ring_tail(const void *ring_mem)
{
    const uint32_t *p =
        (const uint32_t *)((const uint8_t *)ring_mem + VGPU3D_RING_TAIL_OFF);
    return be32_to_cpu(p[0]);
}

static inline void vgpu3d_ring_set_tail(void *ring_mem, uint32_t tail)
{
    uint32_t *p =
        (uint32_t *)((uint8_t *)ring_mem + VGPU3D_RING_TAIL_OFF);
    p[0] = cpu_to_be32(tail);
}

static inline uint8_t *vgpu3d_ring_data(void *ring_mem)
{
    return (uint8_t *)ring_mem + VGPU3D_RING_DATA_OFF;
}

static inline const vgpu3d_pkt_hdr *
vgpu3d_pkt_at(const void *ring_mem, uint32_t off)
{
    const uint8_t *data =
        (const uint8_t *)ring_mem + VGPU3D_RING_DATA_OFF;
    return (const vgpu3d_pkt_hdr *)(data + off);
}

static inline uint32_t vgpu3d_pkt_next(uint32_t off, uint16_t pkt_len)
{
    uint32_t next = off + be16_to_cpu(pkt_len);
    if (next >= VGPU3D_RING_CAP) {
        next = 0;
    }
    return next;
}

#endif /* VGPU3D_GUEST_BUILD */

#endif /* ATI_VGPU3D_H */
