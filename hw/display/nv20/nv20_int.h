/*
 * QEMU NVIDIA GeForce3 (NV20) AGP — internal definitions
 *
 * Derived from xemu NV2A (LGPL-2.1+).
 */

#ifndef HW_NV20_INT_H
#define HW_NV20_INT_H

#include "qemu/osdep.h"
#include "qemu/units.h"
#include "qemu/thread.h"
#include "qemu/timer.h"
#include "hw/display/vga_int.h"
#include "hw/display/vga_regs.h"
#include "hw/pci/pci_device.h"
#include "hw/pci/pci_ids.h"
#include "hw/display/i2c-ddc.h"
#include "hw/i2c/bitbang_i2c.h"
#include "nv20_regs.h"
#include "debug.h"
#include "nv20_xemu_compat.h"
#include "hw/display/trace.h"
#include "pgraph/pgraph.h"
#include "qom/object.h"

#define TYPE_NV20_VGA "nv20-vga"
OBJECT_DECLARE_SIMPLE_TYPE(NV20State, NV20_VGA)

#define NV20_MMIO_SIZE          (16 * MiB)
#define NV20_VRAM_DEFAULT_MB    64
#define NV20_RAMIN_SIZE         0x100000

typedef struct NV20BlockInfo {
    const char *name;
    hwaddr offset;
    uint64_t size;
    MemoryRegionOps ops;
} NV20BlockInfo;

extern NV20State *g_nv20;

enum FIFOEngine {
    ENGINE_SOFTWARE = 0,
    ENGINE_GRAPHICS = 1,
    ENGINE_DVD = 2,
};

typedef struct DMAObject {
    unsigned int dma_class;
    unsigned int dma_target;
    hwaddr address;
    hwaddr limit;
} DMAObject;

struct NV20State {
    PCIDevice parent_obj;

    bool exiting;

    VGACommonState vga;
    GraphicHwOps hw_ops;
    QEMUTimer vblank_timer;

    MemoryRegion vram_mr;
    MemoryRegion vram_pci;
    MemoryRegion ramin;
    MemoryRegion mmio;
    MemoryRegion block_mmio[NV_NUM_BLOCKS];
    MemoryRegion vbe_compat;

    uint8_t *vram_ptr;
    uint8_t *ramin_ptr;

    struct {
        uint32_t pending_interrupts;
        uint32_t enabled_interrupts;
    } pmc;

    struct {
        uint32_t pending_interrupts;
        uint32_t enabled_interrupts;
        uint32_t regs[0x2000];
        QemuMutex lock;
        QemuThread thread;
        QemuCond fifo_cond;
        QemuCond fifo_idle_cond;
        bool fifo_kick;
        bool halt;
    } pfifo;

    struct {
        uint32_t regs[0x1000];
    } pvideo;

    struct {
        uint32_t pending_interrupts;
        uint32_t enabled_interrupts;
        uint32_t numerator;
        uint32_t denominator;
        uint32_t alarm_time;
    } ptimer;

    struct {
        uint32_t regs[0x1000];
    } pfb;

    PGRAPHState pgraph;

    struct {
        uint32_t pending_interrupts;
        uint32_t enabled_interrupts;
        hwaddr start;
        uint32_t raster;
    } pcrtc;

    struct {
        uint32_t core_clock_coeff;
        uint64_t core_clock_freq;
        uint32_t memory_clock_coeff;
        uint32_t video_clock_coeff;
        uint32_t general_control;
        uint32_t fp_vdisplay_end;
        uint32_t fp_vcrtc;
        uint32_t fp_vsync_end;
        uint32_t fp_vvalid_end;
        uint32_t fp_hdisplay_end;
        uint32_t fp_hcrtc;
        uint32_t fp_hvalid_end;
    } pramdac;

    struct {
        uint16_t write_mode_address;
        uint8_t palette[256 * 3];
    } puserdac;

    bitbang_i2c_interface bbi2c;
    I2CDDCState i2cddc;
};

void nv20_update_irq(NV20State *d);

DMAObject nv_dma_load(NV20State *d, hwaddr dma_obj_address);
void *nv_dma_map(NV20State *d, hwaddr dma_obj_address, hwaddr *len);
hwaddr nv_clip_gpu_tile_blit(NV20State *d, hwaddr blit_base_address,
                             hwaddr len);

void pfifo_kick(NV20State *d);
void *pfifo_thread(void *opaque);

#define DEFINE_PROTO(n) \
    uint64_t n##_read(void *opaque, hwaddr addr, unsigned int size); \
    void n##_write(void *opaque, hwaddr addr, uint64_t val, unsigned int size);

DEFINE_PROTO(pmc)
DEFINE_PROTO(pbus)
DEFINE_PROTO(pfifo)
DEFINE_PROTO(prma)
DEFINE_PROTO(pvideo)
DEFINE_PROTO(ptimer)
DEFINE_PROTO(pcounter)
DEFINE_PROTO(pvpe)
DEFINE_PROTO(ptv)
DEFINE_PROTO(prmfb)
DEFINE_PROTO(prmvio)
DEFINE_PROTO(pfb)
DEFINE_PROTO(pstraps)
DEFINE_PROTO(pgraph)
DEFINE_PROTO(pcrtc)
DEFINE_PROTO(prmcio)
DEFINE_PROTO(pramdac)
DEFINE_PROTO(prmdio)
DEFINE_PROTO(user)
#undef DEFINE_PROTO

void pgraph_init(NV20State *d);
void pgraph_destroy(PGRAPHState *pg);
void nv20_context_init(void);

#endif /* HW_NV20_INT_H */
