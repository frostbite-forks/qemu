/*
 * QEMU NVIDIA GeForce3 AGP (NV20) for PowerMac / OpenBIOS
 *
 * PCI 10de:0200 with Mac FCode ROM (gf3_mac.rom).
 * Full PGRAPH interpreter (from xemu NV2A) with Metal renderer for
 * native Mac OS 9 NVIDIA driver 3D — no guest vgpu3d shim.
 */

#include "qemu/osdep.h"
#include "qemu/units.h"
#include "qemu/thread.h"
#include "hw/core/qdev-properties.h"
#include "hw/display/edid.h"
#include "hw/pci/pci.h"
#include "hw/display/vga_regs.h"
#include "migration/vmstate.h"
#include "ui/console.h"
#include "qemu/module.h"
#include "qapi/error.h"
#include "nv20_int.h"
#include "nv20_pgraph_settings.h"
#include "debug.h"

NV20State *g_nv20;

const NV20BlockInfo nv20_blocktable[NV_NUM_BLOCKS] = {
#define ENTRY(NAME, LNAME, OFFSET, SIZE) [NV_##NAME] = { \
        .name = #NAME, \
        .offset = OFFSET, \
        .size = SIZE, \
        .ops = { .read = LNAME##_read, .write = LNAME##_write }, \
    }
    ENTRY(PMC,      pmc,      0x000000, 0x001000),
    ENTRY(PBUS,     pbus,     0x001000, 0x001000),
    ENTRY(PFIFO,    pfifo,    0x002000, 0x002000),
    ENTRY(PRMA,     prma,     0x007000, 0x001000),
    ENTRY(PVIDEO,   pvideo,   0x008000, 0x001000),
    ENTRY(PTIMER,   ptimer,   0x009000, 0x001000),
    ENTRY(PCOUNTER, pcounter, 0x00a000, 0x001000),
    ENTRY(PVPE,     pvpe,     0x00b000, 0x001000),
    ENTRY(PTV,      ptv,      0x00d000, 0x001000),
    ENTRY(PRMFB,    prmfb,    0x0a0000, 0x020000),
    ENTRY(PRMVIO,   prmvio,   0x0c0000, 0x001000),
    ENTRY(PFB,      pfb,      0x100000, 0x001000),
    ENTRY(PSTRAPS,  pstraps,  0x101000, 0x001000),
    ENTRY(PGRAPH,   pgraph,   0x400000, 0x002000),
    ENTRY(PCRTC,    pcrtc,    0x600000, 0x001000),
    ENTRY(PRMCIO,   prmcio,   0x601000, 0x001000),
    ENTRY(PRAMDAC,  pramdac,  0x680000, 0x001000),
    ENTRY(PRMDIO,   prmdio,   0x681000, 0x001000),
    ENTRY(USER,     user,     0x800000, 0x800000),
#undef ENTRY
};

void nv20_update_irq(NV20State *d)
{
    if (d->pfifo.pending_interrupts & d->pfifo.enabled_interrupts) {
        d->pmc.pending_interrupts |= NV_PMC_INTR_0_PFIFO;
    } else {
        d->pmc.pending_interrupts &= ~NV_PMC_INTR_0_PFIFO;
    }

    if (d->pcrtc.pending_interrupts & d->pcrtc.enabled_interrupts) {
        d->pmc.pending_interrupts |= NV_PMC_INTR_0_PCRTC;
    } else {
        d->pmc.pending_interrupts &= ~NV_PMC_INTR_0_PCRTC;
    }

    if (d->pgraph.pending_interrupts & d->pgraph.enabled_interrupts) {
        d->pmc.pending_interrupts |= NV_PMC_INTR_0_PGRAPH;
    } else {
        d->pmc.pending_interrupts &= ~NV_PMC_INTR_0_PGRAPH;
    }

    if (d->pmc.pending_interrupts && d->pmc.enabled_interrupts) {
        pci_irq_assert(PCI_DEVICE(d));
    } else {
        pci_irq_deassert(PCI_DEVICE(d));
    }
}

DMAObject nv_dma_load(NV20State *d, hwaddr dma_obj_address)
{
    assert(dma_obj_address < memory_region_size(&d->ramin));

    uint32_t *dma_obj = (uint32_t *)(d->ramin_ptr + dma_obj_address);
    uint32_t flags = ldl_le_p(dma_obj);
    uint32_t limit = ldl_le_p(dma_obj + 1);
    uint32_t frame = ldl_le_p(dma_obj + 2);

    return (DMAObject){
        .dma_class  = GET_MASK(flags, NV_DMA_CLASS),
        .dma_target = GET_MASK(flags, NV_DMA_TARGET),
        .address    = (frame & NV_DMA_ADDRESS) | GET_MASK(flags, NV_DMA_ADJUST),
        .limit      = limit,
    };
}

void *nv_dma_map(NV20State *d, hwaddr dma_obj_address, hwaddr *len)
{
    DMAObject dma = nv_dma_load(d, dma_obj_address);

    dma.address &= 0x07ffffff;
    assert(dma.address < memory_region_size(&d->vram_mr));
    *len = dma.limit;
    return d->vram_ptr + dma.address;
}

hwaddr nv_clip_gpu_tile_blit(NV20State *d, hwaddr blit_base_address, hwaddr len)
{
    const uint32_t *regs = d->pfb.regs;
    hwaddr blit_end = blit_base_address + len;

    for (int i = 0; i < NV_NUM_GPU_TILES; ++i) {
        uint32_t base_and_flags = regs[NV_PFB_TILE_BASE_ADDRESS_AND_FLAGS(i)];
        if (!(base_and_flags & NV_PFB_TILE_FLAGS_VALID)) {
            continue;
        }
        uint32_t limit = regs[NV_PFB_TILE_LIMIT(i)];
        if (blit_base_address < limit && blit_end > limit) {
            return limit + 1 - blit_base_address;
        }
    }
    return len;
}

static int nv20_get_bpp(VGACommonState *s)
{
    NV20State *d = container_of(s, NV20State, vga);
    int depth = s->cr[0x28] & 3;
    int bpp;

    switch (depth) {
    case 0:
        bpp = 0;
        break;
    case 2:
        bpp = d->pramdac.general_control &
              NV_PRAMDAC_GENERAL_CONTROL_ALT_MODE_SEL ? 16 : 15;
        break;
    case 3:
        bpp = 32;
        break;
    default:
        bpp = depth * 8;
        break;
    }
    return bpp;
}

static void nv20_get_params(VGACommonState *s, VGADisplayParams *params)
{
    NV20State *d = container_of(s, NV20State, vga);

    params->line_offset = (s->cr[0x13] | ((s->cr[0x19] & 0xe0) << 3) |
                           ((s->cr[0x25] & 0x20) << 6)) << 3;
    params->start_addr = d->pcrtc.start / 4;
    params->line_compare = s->cr[VGA_CRTC_LINE_COMPARE] |
                           ((s->cr[VGA_CRTC_OVERFLOW] & 0x10) << 4) |
                           ((s->cr[VGA_CRTC_MAX_SCAN] & 0x40) << 3);
}

static bool nv20_vga_gfx_update(void *opaque)
{
    VGACommonState *vga = opaque;
    NV20State *d = container_of(vga, NV20State, vga);
    bool ret;

    ret = vga->hw_ops->gfx_update(vga);
    d->pcrtc.pending_interrupts |= NV_PCRTC_INTR_0_VBLANK;
    d->pcrtc.raster = 0;
    nv20_update_irq(d);
    return ret;
}

static void nv20_vblank_irq(void *opaque)
{
    NV20State *d = opaque;

    d->pcrtc.pending_interrupts |= NV_PCRTC_INTR_0_VBLANK;
    nv20_update_irq(d);
    timer_mod(&d->vblank_timer,
              qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + NANOSECONDS_PER_SECOND / 60);
}

static uint64_t nv20_vga_bochs_read(void *opaque, hwaddr addr, unsigned size)
{
    VGACommonState *s = opaque;

    vbe_ioport_write_index(s, 0, addr >> 1);
    return vbe_ioport_read_data(s, 0);
}

static void nv20_vga_bochs_write(void *opaque, hwaddr addr,
                                 uint64_t val, unsigned size)
{
    VGACommonState *s = opaque;

    vbe_ioport_write_index(s, 0, addr >> 1);
    vbe_ioport_write_data(s, 0, val);
}

static const MemoryRegionOps nv20_vga_bochs_ops = {
    .read = nv20_vga_bochs_read,
    .write = nv20_vga_bochs_write,
    .valid.min_access_size = 1,
    .valid.max_access_size = 4,
    .impl.min_access_size = 2,
    .impl.max_access_size = 2,
    .endianness = DEVICE_LITTLE_ENDIAN,
};

static void nv20_init_pgraph_hw(NV20State *d)
{
    memory_region_set_log(&d->vram_mr, true, DIRTY_MEMORY_VGA);
    memory_region_set_dirty(&d->vram_mr, 0, memory_region_size(&d->vram_mr));

    qemu_mutex_init(&d->pfifo.lock);
    qemu_cond_init(&d->pfifo.fifo_cond);
    qemu_cond_init(&d->pfifo.fifo_idle_cond);

    pgraph_init(d);
    nv20_context_init();

    d->pfifo.halt = false;
    qemu_thread_create(&d->pfifo.thread, "nv20.pfifo",
                       pfifo_thread, d, QEMU_THREAD_JOINABLE);
}

static void nv20_reset_hold(Object *obj, ResetType type)
{
    NV20State *d = NV20_VGA(obj);
    int i;

    d->pmc.pending_interrupts = 0;
    d->pfifo.pending_interrupts = 0;
    d->ptimer.pending_interrupts = 0;
    d->pcrtc.pending_interrupts = 0;
    d->pgraph.pending_interrupts = 0;

    memset(d->pfifo.regs, 0, sizeof(d->pfifo.regs));
    memset(d->pvideo.regs, 0, sizeof(d->pvideo.regs));

    d->pcrtc.start = 0;
    d->pramdac.core_clock_coeff = 0x00011c01;
    d->pramdac.core_clock_freq = 233333324;
    d->pramdac.memory_clock_coeff = 0;
    d->pramdac.video_clock_coeff = 0x0003c20d;
    d->pfifo.regs[NV_PFIFO_CACHE1_STATUS] |= NV_PFIFO_CACHE1_STATUS_LOW_MARK;

    vga_common_reset(&d->vga);
    d->vga.msr = VGA_MIS_COLOR;

    d->pgraph.waiting_for_nop = false;
    d->pgraph.waiting_for_flip = false;
    d->pgraph.waiting_for_context_switch = false;

    for (i = 0; i < 256; i++) {
        d->puserdac.palette[i * 3] = i;
        d->puserdac.palette[i * 3 + 1] = i;
        d->puserdac.palette[i * 3 + 2] = i;
    }
}

static void nv20_realize(PCIDevice *dev, Error **errp)
{
    NV20State *d = NV20_VGA(dev);
    VGACommonState *vga = &d->vga;
    I2CBus *i2cbus;
    int i;

    g_nv20 = d;

    pci_set_word(dev->config + PCI_DEVICE_ID, PCI_DEVICE_ID_NVIDIA_GEFORCE3);
    dev->config[PCI_INTERRUPT_PIN] = 1;

    if (!vga_common_init(vga, OBJECT(d), errp)) {
        return;
    }
    vga->get_bpp = nv20_get_bpp;
    vga->get_params = nv20_get_params;
    vga_init(vga, OBJECT(d), pci_address_space(dev),
             pci_address_space_io(dev), true);

    d->hw_ops = *vga->hw_ops;
    d->hw_ops.gfx_update = nv20_vga_gfx_update;
    vga->con = qemu_graphic_console_create(DEVICE(d), 0, &d->hw_ops, vga);

    memory_region_init_ram(&d->vram_mr, OBJECT(d), "nv20-vram",
                           (hwaddr)vga->vram_size_mb * MiB, errp);
    if (*errp) {
        return;
    }
    d->vram_ptr = memory_region_get_ram_ptr(&d->vram_mr);

    memory_region_init_alias(&vga->vram, OBJECT(d), "nv20-vga.vram",
                             &d->vram_mr, 0, memory_region_size(&d->vram_mr));
    vga->vram_ptr = d->vram_ptr;
    vga_dirty_log_start(vga);

    memory_region_init_alias(&d->vram_pci, OBJECT(d), "nv20-vram-pci",
                             &d->vram_mr, 0, memory_region_size(&d->vram_mr));
    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_MEM_PREFETCH, &d->vram_pci);

    memory_region_init(&d->mmio, OBJECT(dev), "nv20-mmio", NV20_MMIO_SIZE);
    pci_register_bar(dev, 1, PCI_BASE_ADDRESS_SPACE_MEMORY, &d->mmio);

    for (i = 0; i < NV_NUM_BLOCKS; i++) {
        if (!nv20_blocktable[i].name) {
            continue;
        }
        memory_region_init_io(&d->block_mmio[i], OBJECT(dev),
                              &nv20_blocktable[i].ops, d,
                              nv20_blocktable[i].name,
                              nv20_blocktable[i].size);
        memory_region_add_subregion(&d->mmio, nv20_blocktable[i].offset,
                                    &d->block_mmio[i]);
    }

    memory_region_init_ram(&d->ramin, OBJECT(d), "nv20-ramin",
                           NV20_RAMIN_SIZE, errp);
    if (*errp) {
        return;
    }
    d->ramin_ptr = memory_region_get_ram_ptr(&d->ramin);
    memory_region_add_subregion(&d->mmio, 0x700000, &d->ramin);

    memory_region_init_io(&d->vbe_compat, OBJECT(d), &nv20_vga_bochs_ops,
                          vga, "nv20-vbe-compat", PCI_VGA_BOCHS_SIZE);
    memory_region_add_subregion(&d->mmio, PCI_VGA_BOCHS_OFFSET, &d->vbe_compat);

    i2cbus = i2c_init_bus(DEVICE(d), "nv20-vga.ddc");
    bitbang_i2c_init(&d->bbi2c, i2cbus);
    i2c_slave_set_address(I2C_SLAVE(&d->i2cddc), 0x50);
    qdev_realize(DEVICE(&d->i2cddc), BUS(i2cbus), &error_abort);

    nv20_init_pgraph_hw(d);

    timer_init_ns(&d->vblank_timer, QEMU_CLOCK_VIRTUAL, nv20_vblank_irq, d);
}

static void nv20_exit(PCIDevice *dev)
{
    NV20State *d = NV20_VGA(dev);

    d->exiting = true;
    qemu_cond_broadcast(&d->pfifo.fifo_cond);
    qemu_thread_join(&d->pfifo.thread);

    pgraph_destroy(&d->pgraph);

    timer_del(&d->vblank_timer);
    qemu_graphic_console_close(d->vga.con);
    if (g_nv20 == d) {
        g_nv20 = NULL;
    }
}

static const Property nv20_vga_properties[] = {
    DEFINE_PROP_UINT32("vgamem_mb", NV20State, vga.vram_size_mb,
                       NV20_VRAM_DEFAULT_MB),
    DEFINE_EDID_PROPERTIES(NV20State, i2cddc.edid_info),
};

static void nv20_vga_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);
    ResettableClass *rc = RESETTABLE_CLASS(klass);

    device_class_set_props(dc, nv20_vga_properties);
    dc->hotpluggable = false;
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);

    k->class_id = PCI_CLASS_DISPLAY_VGA;
    k->vendor_id = PCI_VENDOR_ID_NVIDIA;
    k->device_id = PCI_DEVICE_ID_NVIDIA_GEFORCE3;
    k->revision = 0xa1;
    k->romfile = "gf3_mac.rom";
    k->realize = nv20_realize;
    k->exit = nv20_exit;

    rc->phases.hold = nv20_reset_hold;
}

static void nv20_vga_init(Object *o)
{
    NV20State *s = NV20_VGA(o);

    object_initialize_child(o, "edid", &s->i2cddc, TYPE_I2CDDC);
}

static const TypeInfo nv20_vga_info = {
    .name = TYPE_NV20_VGA,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(NV20State),
    .instance_init = nv20_vga_init,
    .class_init = nv20_vga_class_init,
    .interfaces = (const InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void nv20_register_types(void)
{
    type_register_static(&nv20_vga_info);
}

type_init(nv20_register_types);
