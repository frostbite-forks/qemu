/*
 * QEMU NVIDIA GeForce3 (NV20) emulation
 *
 * See nv_geforce3_int.h for background on the card identity, BAR
 * layout and the relationship to Bochs' geforce.cc/.h, which this
 * file's PCI/CRTC/RAMDAC/display, MMIO dispatch, PFIFO command
 * processing and RAMHT/RAMFC/DMA-object plumbing are ported from.
 *
 * A guest-triggerable condition that made Bochs abort the whole
 * emulator (BX_PANIC) is instead logged with qemu_log_mask() and
 * handled as safely as possible here -- QEMU devices must never let
 * guest-controlled register content crash the host process.
 *
 * Two features present in the Bochs source are deliberately not
 * ported, as dead weight for this specific device:
 *  - The RMA (Register/Memory Access) backdoor at legacy I/O ports
 *    0x3d0/0x3d2: real hardware only exposes it on literal ISA/PCI
 *    port I/O, which this device does not have (no I/O BAR, matching
 *    the real Mac Edition card) -- everything reachable on real
 *    hardware without port I/O is reachable here too, via the MMIO
 *    byte-addressed legacy-register windows (see
 *    nv_geforce3_legacy_read/write()).
 *  - The "disable ROM shadow to repurpose the expansion-ROM BAR as a
 *    PRAMIN backdoor" trick: QEMU's generic PCI option-ROM mechanism
 *    (driven by the standard "romfile" property -- no default romfile
 *    name is set, so nothing loads unless the user passes one) gives
 *    the ROM BAR its own independent, read-only MemoryRegion that
 *    this device's own MMIO callbacks have no hook into, so the
 *    trick cannot apply here regardless; no FCode or NDRV bring-up
 *    path needs it.
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "qemu/units.h"
#include "hw/pci/pci_device.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"
#include "qapi/error.h"
#include "system/memory.h"
#include "system/dma.h"
#include "ui/console.h"
#include "qom/object.h"

#include "nv_geforce3_int.h"

static uint32_t nv_geforce3_register_read32(NVGeForce3State *s,
                                            uint32_t address);
static void nv_geforce3_register_write32(NVGeForce3State *s,
                                         uint32_t address, uint32_t value);

/* ---------------------------------------------------------------- */
/* Time / VRAM / RAMIN / system-memory DMA access                   */

uint64_t nv_geforce3_get_current_time(NVGeForce3State *s)
{
    uint64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

    return (s->timer_inittime1 + now - s->timer_inittime2) & ~0x1FULL;
}

uint8_t nv_geforce3_vram_read8(NVGeForce3State *s, uint32_t addr)
{
    return s->vram_ptr[addr & s->memsize_mask];
}

uint16_t nv_geforce3_vram_read16(NVGeForce3State *s, uint32_t addr)
{
    return (uint16_t)nv_geforce3_vram_read8(s, addr) |
           ((uint16_t)nv_geforce3_vram_read8(s, addr + 1) << 8);
}

uint32_t nv_geforce3_vram_read32(NVGeForce3State *s, uint32_t addr)
{
    return (uint32_t)nv_geforce3_vram_read8(s, addr) |
           ((uint32_t)nv_geforce3_vram_read8(s, addr + 1) << 8) |
           ((uint32_t)nv_geforce3_vram_read8(s, addr + 2) << 16) |
           ((uint32_t)nv_geforce3_vram_read8(s, addr + 3) << 24);
}

uint64_t nv_geforce3_vram_read64(NVGeForce3State *s, uint32_t addr)
{
    return (uint64_t)nv_geforce3_vram_read32(s, addr) |
           ((uint64_t)nv_geforce3_vram_read32(s, addr + 4) << 32);
}

void nv_geforce3_vram_write8(NVGeForce3State *s, uint32_t addr, uint8_t val)
{
    s->vram_ptr[addr & s->memsize_mask] = val;
}

void nv_geforce3_vram_write16(NVGeForce3State *s, uint32_t addr, uint16_t val)
{
    nv_geforce3_vram_write8(s, addr, (uint8_t)val);
    nv_geforce3_vram_write8(s, addr + 1, (uint8_t)(val >> 8));
}

void nv_geforce3_vram_write32(NVGeForce3State *s, uint32_t addr, uint32_t val)
{
    nv_geforce3_vram_write8(s, addr, (uint8_t)val);
    nv_geforce3_vram_write8(s, addr + 1, (uint8_t)(val >> 8));
    nv_geforce3_vram_write8(s, addr + 2, (uint8_t)(val >> 16));
    nv_geforce3_vram_write8(s, addr + 3, (uint8_t)(val >> 24));
}

void nv_geforce3_vram_write64(NVGeForce3State *s, uint32_t addr, uint64_t val)
{
    nv_geforce3_vram_write32(s, addr, (uint32_t)val);
    nv_geforce3_vram_write32(s, addr + 4, (uint32_t)(val >> 32));
}

/*
 * PRAMIN (instance memory): the top of VRAM, addressed backwards via
 * an XOR flip -- real NV hardware behaviour, not an emulation
 * shortcut (see the reset-time comment on ramin_flip in realize()).
 */
uint8_t nv_geforce3_ramin_read8(NVGeForce3State *s, uint32_t addr)
{
    return nv_geforce3_vram_read8(s, addr ^ s->ramin_flip);
}

uint16_t nv_geforce3_ramin_read16(NVGeForce3State *s, uint32_t addr)
{
    return nv_geforce3_vram_read16(s, addr ^ s->ramin_flip);
}

uint32_t nv_geforce3_ramin_read32(NVGeForce3State *s, uint32_t addr)
{
    return nv_geforce3_vram_read32(s, addr ^ s->ramin_flip);
}

void nv_geforce3_ramin_write8(NVGeForce3State *s, uint32_t addr, uint8_t val)
{
    nv_geforce3_vram_write8(s, addr ^ s->ramin_flip, val);
}

void nv_geforce3_ramin_write32(NVGeForce3State *s, uint32_t addr, uint32_t val)
{
    nv_geforce3_vram_write32(s, addr ^ s->ramin_flip, val);
}

static uint8_t nv_geforce3_physical_read8(NVGeForce3State *s, uint32_t addr)
{
    uint8_t data = 0xff;

    pci_dma_read(&s->parent_obj, addr, &data, 1);
    return data;
}

static uint16_t nv_geforce3_physical_read16(NVGeForce3State *s, uint32_t addr)
{
    uint8_t data[2] = { 0xff, 0xff };

    pci_dma_read(&s->parent_obj, addr, data, 2);
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static uint32_t nv_geforce3_physical_read32(NVGeForce3State *s, uint32_t addr)
{
    uint8_t data[4] = { 0xff, 0xff, 0xff, 0xff };

    pci_dma_read(&s->parent_obj, addr, data, 4);
    return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
           ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static uint64_t nv_geforce3_physical_read64(NVGeForce3State *s, uint32_t addr)
{
    return (uint64_t)nv_geforce3_physical_read32(s, addr) |
           ((uint64_t)nv_geforce3_physical_read32(s, addr + 4) << 32);
}

static void nv_geforce3_physical_write8(NVGeForce3State *s, uint32_t addr,
                                        uint8_t val)
{
    pci_dma_write(&s->parent_obj, addr, &val, 1);
}

static void nv_geforce3_physical_write16(NVGeForce3State *s, uint32_t addr,
                                         uint16_t val)
{
    uint8_t data[2] = { (uint8_t)val, (uint8_t)(val >> 8) };

    pci_dma_write(&s->parent_obj, addr, data, 2);
}

static void nv_geforce3_physical_write32(NVGeForce3State *s, uint32_t addr,
                                         uint32_t val)
{
    uint8_t data[4] = {
        (uint8_t)val, (uint8_t)(val >> 8),
        (uint8_t)(val >> 16), (uint8_t)(val >> 24),
    };

    pci_dma_write(&s->parent_obj, addr, data, 4);
}

static void nv_geforce3_physical_write64(NVGeForce3State *s, uint32_t addr,
                                         uint64_t val)
{
    nv_geforce3_physical_write32(s, addr, (uint32_t)val);
    nv_geforce3_physical_write32(s, addr + 4, (uint32_t)(val >> 32));
}

static uint32_t nv_geforce3_dma_pt_lookup(NVGeForce3State *s, uint32_t object,
                                          uint32_t address)
{
    uint32_t address_adj = address + (nv_geforce3_ramin_read32(s, object) >> 20);
    uint32_t page_offset = address_adj & 0xFFF;
    uint32_t page_index = address_adj >> 12;
    uint32_t page = nv_geforce3_ramin_read32(s, object + 8 + page_index * 4) &
                    0xFFFFF000;

    return page | page_offset;
}

uint32_t nv_geforce3_dma_lin_lookup(NVGeForce3State *s, uint32_t object,
                                    uint32_t address)
{
    uint32_t adjust = nv_geforce3_ramin_read32(s, object) >> 20;
    uint32_t base = nv_geforce3_ramin_read32(s, object + 8) & 0xFFFFF000;

    return base + adjust + address;
}

static uint32_t nv_geforce3_dma_addr(NVGeForce3State *s, uint32_t object,
                                     uint32_t address, uint32_t *flags)
{
    *flags = nv_geforce3_ramin_read32(s, object);
    if (*flags & 0x00002000) {
        return nv_geforce3_dma_lin_lookup(s, object, address);
    }
    return nv_geforce3_dma_pt_lookup(s, object, address);
}

uint8_t nv_geforce3_dma_read8(NVGeForce3State *s, uint32_t object,
                              uint32_t address)
{
    uint32_t flags;
    uint32_t addr_abs = nv_geforce3_dma_addr(s, object, address, &flags);

    if (flags & 0x00020000) {
        return nv_geforce3_physical_read8(s, addr_abs);
    }
    return nv_geforce3_vram_read8(s, addr_abs);
}

uint16_t nv_geforce3_dma_read16(NVGeForce3State *s, uint32_t object,
                                uint32_t address)
{
    uint32_t flags;
    uint32_t addr_abs = nv_geforce3_dma_addr(s, object, address, &flags);

    if (flags & 0x00020000) {
        return nv_geforce3_physical_read16(s, addr_abs);
    }
    return nv_geforce3_vram_read16(s, addr_abs);
}

uint32_t nv_geforce3_dma_read32(NVGeForce3State *s, uint32_t object,
                                uint32_t address)
{
    uint32_t flags;
    uint32_t addr_abs = nv_geforce3_dma_addr(s, object, address, &flags);

    if (flags & 0x00020000) {
        return nv_geforce3_physical_read32(s, addr_abs);
    }
    return nv_geforce3_vram_read32(s, addr_abs);
}

uint64_t nv_geforce3_dma_read64(NVGeForce3State *s, uint32_t object,
                                uint32_t address)
{
    uint32_t flags;
    uint32_t addr_abs = nv_geforce3_dma_addr(s, object, address, &flags);

    if (flags & 0x00020000) {
        return nv_geforce3_physical_read64(s, addr_abs);
    }
    return nv_geforce3_vram_read64(s, addr_abs);
}

void nv_geforce3_dma_write8(NVGeForce3State *s, uint32_t object,
                            uint32_t address, uint8_t val)
{
    uint32_t flags;
    uint32_t addr_abs = nv_geforce3_dma_addr(s, object, address, &flags);

    if (flags & 0x00020000) {
        nv_geforce3_physical_write8(s, addr_abs, val);
    } else {
        nv_geforce3_vram_write8(s, addr_abs, val);
    }
}

void nv_geforce3_dma_write16(NVGeForce3State *s, uint32_t object,
                             uint32_t address, uint16_t val)
{
    uint32_t flags;
    uint32_t addr_abs = nv_geforce3_dma_addr(s, object, address, &flags);

    if (flags & 0x00020000) {
        nv_geforce3_physical_write16(s, addr_abs, val);
    } else {
        nv_geforce3_vram_write16(s, addr_abs, val);
    }
}

void nv_geforce3_dma_write32(NVGeForce3State *s, uint32_t object,
                             uint32_t address, uint32_t val)
{
    uint32_t flags;
    uint32_t addr_abs = nv_geforce3_dma_addr(s, object, address, &flags);

    if (flags & 0x00020000) {
        nv_geforce3_physical_write32(s, addr_abs, val);
    } else {
        nv_geforce3_vram_write32(s, addr_abs, val);
    }
}

void nv_geforce3_dma_write64(NVGeForce3State *s, uint32_t object,
                             uint32_t address, uint64_t val)
{
    uint32_t flags;
    uint32_t addr_abs = nv_geforce3_dma_addr(s, object, address, &flags);

    if (flags & 0x00020000) {
        nv_geforce3_physical_write64(s, addr_abs, val);
    } else {
        nv_geforce3_vram_write64(s, addr_abs, val);
    }
}

void nv_geforce3_dma_copy(NVGeForce3State *s, uint32_t dst_obj,
                          uint32_t dst_addr, uint32_t src_obj,
                          uint32_t src_addr, uint32_t byte_count)
{
    uint32_t dst_flags = nv_geforce3_ramin_read32(s, dst_obj);
    uint32_t src_flags = nv_geforce3_ramin_read32(s, src_obj);
    uint8_t buffer[4096];
    uint32_t bytes_left = byte_count;

    while (bytes_left) {
        uint32_t dst_addr_abs = (dst_flags & 0x00002000) ?
            nv_geforce3_dma_lin_lookup(s, dst_obj, dst_addr) :
            nv_geforce3_dma_pt_lookup(s, dst_obj, dst_addr);
        uint32_t src_addr_abs = (src_flags & 0x00002000) ?
            nv_geforce3_dma_lin_lookup(s, src_obj, src_addr) :
            nv_geforce3_dma_pt_lookup(s, src_obj, src_addr);
        uint32_t chunk_bytes = MIN(bytes_left,
            MIN(sizeof(buffer) - (dst_addr_abs & 0xFFF),
                sizeof(buffer) - (src_addr_abs & 0xFFF)));

        if (src_flags & 0x00020000) {
            pci_dma_read(&s->parent_obj, src_addr_abs, buffer, chunk_bytes);
        } else {
            memcpy(buffer, s->vram_ptr + (src_addr_abs & s->memsize_mask),
                   chunk_bytes);
        }
        if (dst_flags & 0x00020000) {
            pci_dma_write(&s->parent_obj, dst_addr_abs, buffer, chunk_bytes);
        } else {
            /*
             * Not masked against memsize_mask, matching the ported
             * model exactly (see the file comment on faithful
             * porting of hardware/model quirks).
             */
            memcpy(s->vram_ptr + dst_addr_abs, buffer, chunk_bytes);
        }
        dst_addr += chunk_bytes;
        src_addr += chunk_bytes;
        bytes_left -= chunk_bytes;
    }
}

static uint32_t nv_geforce3_ramfc_address(NVGeForce3State *s, uint32_t chid,
                                          uint32_t offset)
{
    uint32_t ramfc = (s->fifo_ramfc & 0xFFF) << 8;

    return ramfc + chid * 0x40 + offset;
}

static void nv_geforce3_ramfc_write32(NVGeForce3State *s, uint32_t chid,
                                      uint32_t offset, uint32_t value)
{
    nv_geforce3_ramin_write32(s, nv_geforce3_ramfc_address(s, chid, offset),
                              value);
}

static uint32_t nv_geforce3_ramfc_read32(NVGeForce3State *s, uint32_t chid,
                                         uint32_t offset)
{
    return nv_geforce3_ramin_read32(s, nv_geforce3_ramfc_address(s, chid,
                                                                 offset));
}

static void nv_geforce3_ramht_lookup(NVGeForce3State *s, uint32_t handle,
                                     uint32_t chid, uint32_t *object,
                                     uint8_t *engine)
{
    uint32_t ramht_addr = (s->fifo_ramht & 0xFFF) << 8;
    uint32_t ramht_bits = ((s->fifo_ramht >> 16) & 0xFF) + 9;
    uint32_t ramht_size = (1u << ramht_bits) << 3;
    uint32_t hash = 0;
    uint32_t x = handle;
    uint32_t it;

    while (x) {
        hash ^= (x & ((1u << ramht_bits) - 1));
        x >>= ramht_bits;
    }
    hash ^= (chid & 0xF) << (ramht_bits - 4);
    hash <<= 3;

    it = hash;
    do {
        if (nv_geforce3_ramin_read32(s, ramht_addr + it) == handle) {
            uint32_t context = nv_geforce3_ramin_read32(s, ramht_addr + it + 4);
            uint32_t ctx_chid = (context >> 24) & 0x1F;

            if (chid == ctx_chid) {
                if (object) {
                    *object = (context & 0xFFFF) << 4;
                }
                if (engine) {
                    *engine = (context >> 16) & 0xFF;
                }
                return;
            }
        }
        it += 8;
        if (it >= ramht_size) {
            it = 0;
        }
    } while (it != hash);

    qemu_log_mask(LOG_GUEST_ERROR,
                  "nv-geforce3: ramht_lookup failed for handle 0x%08x\n",
                  handle);
    if (object) {
        *object = 0;
    }
    if (engine) {
        *engine = 0;
    }
}

/* ---------------------------------------------------------------- */
/* PFIFO command processing                                         */

void nv_geforce3_update_fifo_wait(NVGeForce3State *s)
{
    s->fifo_wait = s->fifo_wait_soft || s->fifo_wait_notify ||
                   s->fifo_wait_flip || s->fifo_wait_acquire;
}

void nv_geforce3_fifo_process_chid(NVGeForce3State *s, uint32_t chid)
{
    uint32_t oldchid;
    NVGeForce3Channel *ch;

    if (s->fifo_wait) {
        return;
    }
    if ((s->fifo_mode & (1u << chid)) == 0) {
        return;
    }
    if ((s->fifo_cache1_push0 & 1) == 0) {
        return;
    }
    if ((s->fifo_cache1_pull0 & 1) == 0) {
        return;
    }

    oldchid = s->fifo_cache1_push1 & 0x1F;
    if (oldchid == chid) {
        if (s->fifo_cache1_dma_put == s->fifo_cache1_dma_get) {
            return;
        }
    } else if (nv_geforce3_ramfc_read32(s, chid, 0x0) ==
               nv_geforce3_ramfc_read32(s, chid, 0x4)) {
        return;
    }

    if (oldchid != chid) {
        nv_geforce3_ramfc_write32(s, oldchid, 0x0, s->fifo_cache1_dma_put);
        nv_geforce3_ramfc_write32(s, oldchid, 0x4, s->fifo_cache1_dma_get);
        nv_geforce3_ramfc_write32(s, oldchid, 0x8, s->fifo_cache1_ref_cnt);
        nv_geforce3_ramfc_write32(s, oldchid, 0xC,
                                  s->fifo_cache1_dma_instance);
        nv_geforce3_ramfc_write32(s, oldchid, 0x2C, s->fifo_cache1_semaphore);

        s->fifo_cache1_dma_put = nv_geforce3_ramfc_read32(s, chid, 0x0);
        s->fifo_cache1_dma_get = nv_geforce3_ramfc_read32(s, chid, 0x4);
        s->fifo_cache1_ref_cnt = nv_geforce3_ramfc_read32(s, chid, 0x8);
        s->fifo_cache1_dma_instance = nv_geforce3_ramfc_read32(s, chid, 0xC);
        s->fifo_cache1_semaphore = nv_geforce3_ramfc_read32(s, chid, 0x2C);

        s->fifo_cache1_push1 = (s->fifo_cache1_push1 & ~0x1Fu) | chid;
    }

    s->fifo_cache1_dma_push |= 0x100;
    if (s->fifo_cache1_dma_instance == 0) {
        qemu_log_mask(LOG_GUEST_ERROR, "nv-geforce3: fifo DMA instance = 0\n");
        return;
    }

    ch = &s->chs[chid];
    while (s->fifo_cache1_dma_get != s->fifo_cache1_dma_put) {
        uint32_t word = nv_geforce3_dma_read32(s,
            s->fifo_cache1_dma_instance << 4, s->fifo_cache1_dma_get);

        s->fifo_cache1_dma_get += 4;
        if (ch->dma_state.mcnt) {
            int cmd_result = nv_geforce3_execute_command(s, chid,
                ch->dma_state.subc, ch->dma_state.mthd, word);

            if (cmd_result <= 1) {
                if (!ch->dma_state.ni) {
                    ch->dma_state.mthd++;
                }
                ch->dma_state.mcnt--;
            } else {
                s->fifo_cache1_dma_get -= 4;
            }
            if (cmd_result != 0) {
                break;
            }
        } else if ((word & 0xe0000003) == 0x20000000) {
            s->fifo_cache1_dma_get = word & 0x1fffffff;
        } else if ((word & 3) == 1) {
            s->fifo_cache1_dma_get = word & 0xfffffffc;
        } else if ((word & 3) == 2) {
            if (ch->subr_active) {
                qemu_log_mask(LOG_GUEST_ERROR,
                    "nv-geforce3: fifo call with subroutine active\n");
            } else {
                ch->subr_return = s->fifo_cache1_dma_get;
                ch->subr_active = true;
            }
            s->fifo_cache1_dma_get = word & 0xfffffffc;
        } else if (word == 0x00020000) {
            if (!ch->subr_active) {
                qemu_log_mask(LOG_GUEST_ERROR,
                    "nv-geforce3: fifo return with subroutine inactive\n");
            } else {
                s->fifo_cache1_dma_get = ch->subr_return;
                ch->subr_active = false;
            }
        } else if ((word & 0xa0030003) == 0) {
            ch->dma_state.mthd = (word >> 2) & 0x7ff;
            ch->dma_state.subc = (word >> 13) & 7;
            ch->dma_state.mcnt = (word >> 18) & 0x7ff;
            ch->dma_state.ni = (word & 0x40000000) != 0;
        } else {
            qemu_log_mask(LOG_GUEST_ERROR,
                "nv-geforce3: fifo unexpected word 0x%08x\n", word);
            break;
        }
    }
}

void nv_geforce3_fifo_process(NVGeForce3State *s)
{
    uint32_t offset = (s->fifo_cache1_push1 & 0x1f) + 1;
    uint32_t i;

    for (i = 0; i < NV_GEFORCE3_CHANNEL_COUNT; i++) {
        nv_geforce3_fifo_process_chid(s, (i + offset) & 0x1f);
    }
}

int nv_geforce3_execute_command(NVGeForce3State *s, uint32_t chid,
                                uint32_t subc, uint32_t method,
                                uint32_t param)
{
    NVGeForce3Channel *ch = &s->chs[chid];
    int result = 0;
    bool software_method = false;

    if (method == 0x000) {
        uint32_t object = 0;
        uint8_t engine = 0;
        uint32_t subc_it;

        nv_geforce3_ramht_lookup(s, param, chid, &object, &engine);
        for (subc_it = 0; subc_it < NV_GEFORCE3_SUBCHANNEL_COUNT; subc_it++) {
            if (ch->schs[subc_it].object == object &&
                ch->schs[subc_it].engine == engine && subc_it != subc) {
                nv_geforce3_object_save(s, ch, subc_it);
            }
        }
        nv_geforce3_object_save(s, ch, subc);
        ch->schs[subc].object = object;
        ch->schs[subc].engine = engine;
        if (ch->schs[subc].engine == 0x01) {
            nv_geforce3_object_load(s, ch, subc);
        } else if (ch->schs[subc].engine == 0x00) {
            software_method = true;
        }
    } else if (method == 0x014) {
        s->fifo_cache1_ref_cnt = param;
    } else if (method == 0x018) {
        uint32_t semaphore_obj;

        nv_geforce3_ramht_lookup(s, param, chid, &semaphore_obj, NULL);
        s->fifo_cache1_semaphore = semaphore_obj >> 4;
    } else if (method == 0x019) {
        s->fifo_cache1_semaphore &= 0x000FFFFF;
        s->fifo_cache1_semaphore |= param << 20;
    } else if (method == 0x01a || method == 0x01b) {
        uint32_t semaphore_obj = (s->fifo_cache1_semaphore & 0x000FFFFF) << 4;
        uint32_t semaphore_offset = s->fifo_cache1_semaphore >> 20;

        if (method == 0x01a) {
            if (nv_geforce3_dma_read32(s, semaphore_obj, semaphore_offset) !=
                param) {
                s->fifo_wait_acquire = true;
                s->fifo_wait = true;
                result = 2;
            }
        } else {
            nv_geforce3_dma_write32(s, semaphore_obj, semaphore_offset,
                                    param);
        }
    } else if (method >= 0x040) {
        if (ch->schs[subc].engine == 0x01) {
            uint32_t cls;
            uint8_t cls8;

            if (method >= 0x060 && method < 0x080) {
                nv_geforce3_ramht_lookup(s, param, chid, &param, NULL);
            }
            cls = nv_geforce3_ramin_read32(s, ch->schs[subc].object) & 0xFFF;
            cls8 = (uint8_t)cls;
            switch (cls8) {
            case 0x19:
                nv_geforce3_execute_clip(ch, method, param);
                break;
            case 0x39:
                nv_geforce3_execute_m2mf(s, ch, subc, method, param);
                break;
            case 0x43:
                nv_geforce3_execute_rop(ch, method, param);
                break;
            case 0x44:
            case 0x18:
                nv_geforce3_execute_patt(ch, method, param);
                break;
            case 0x4a:
            case 0x4b:
                nv_geforce3_execute_gdi(s, ch, cls, method, param);
                break;
            case 0x52:
            case 0x9e:
                nv_geforce3_execute_swzsurf(ch, method, param);
                break;
            case 0x57:
                nv_geforce3_execute_chroma(ch, method, param);
                break;
            case 0x1c:
            case 0x5c:
                nv_geforce3_execute_lin(s, ch, method, param);
                break;
            case 0x1e:
            case 0x5e:
                nv_geforce3_execute_rect(s, ch, method, param);
                break;
            case 0x5f:
            case 0x9f:
                nv_geforce3_execute_imageblit(s, ch, method, param);
                break;
            case 0x21:
            case 0x65:
            case 0x8a:
                nv_geforce3_execute_ifc(s, ch, method, param);
                break;
            case 0x62:
                nv_geforce3_execute_surf2d(s, ch, method, param);
                break;
            case 0x64:
                nv_geforce3_execute_iifc(s, ch, method, param);
                break;
            case 0x66:
            case 0x76:
                nv_geforce3_execute_sifc(s, ch, method, param);
                break;
            case 0x72:
                nv_geforce3_execute_beta(ch, method, param);
                break;
            case 0x7b:
                nv_geforce3_execute_tfc(s, ch, method, param);
                break;
            case 0x89:
                nv_geforce3_execute_sifm(s, ch, cls, method, param);
                break;
            case 0x96:
            case 0x97:
                nv_geforce3_execute_d3d(s, ch, cls, method, param);
                if (s->fifo_wait_flip) {
                    result = 1;
                }
                break;
            default:
                break;
            }
            if (ch->notify_pending) {
                ch->notify_pending = false;
                if ((nv_geforce3_ramin_read32(s, ch->schs[subc].notifier) &
                     0xFF) != 0x30) {
                    nv_geforce3_dma_write64(s, ch->schs[subc].notifier, 0x0,
                                            nv_geforce3_get_current_time(s));
                    nv_geforce3_dma_write32(s, ch->schs[subc].notifier, 0x8,
                                            0);
                    nv_geforce3_dma_write32(s, ch->schs[subc].notifier, 0xC,
                                            0);
                }
                if (ch->notify_type) {
                    uint32_t notifier = ch->schs[subc].notifier >> 4;

                    s->graph_intr |= 0x00000001;
                    nv_geforce3_update_irq(s);
                    s->graph_nsource |= 0x00000001;
                    s->graph_notify = 0x00110000;
                    s->graph_ctx_switch2 = notifier << 16;
                    s->graph_ctx_switch4 = ch->schs[subc].object >> 4;
                    s->graph_trapped_addr = (method << 2) | (subc << 16) |
                                            (chid << 20);
                    s->graph_trapped_data = param;
                    s->fifo_wait_notify = true;
                    s->fifo_wait = true;
                }
            }
            if (method == 0x041) {
                ch->notify_pending = true;
                ch->notify_type = param;
            } else if (method == 0x060) {
                ch->schs[subc].notifier = param;
            }
        } else if (ch->schs[subc].engine == 0x00) {
            software_method = true;
        }
    }

    if (software_method) {
        s->fifo_wait_soft = true;
        s->fifo_wait = true;
        s->fifo_intr |= 0x00000001;
        nv_geforce3_update_irq(s);
        s->fifo_cache1_method[s->fifo_cache1_put / 4] =
            (method << 2) | (subc << 13);
        s->fifo_cache1_data[s->fifo_cache1_put / 4] = param;
        s->fifo_cache1_put += 4;
        if (s->fifo_cache1_put == NV_GEFORCE3_CACHE1_SIZE * 4) {
            s->fifo_cache1_put = 0;
        }
        result = 1;
    }
    return result;
}

/*
 * The ported 2D/3D engine code calls this after every draw operation,
 * mirroring Bochs' own tile-based partial-invalidation scheme. This
 * device instead redraws the whole visible mode every refresh (see
 * nv_geforce3_gfx_update()), which is simpler and already correct for
 * a device whose interesting workloads are 3D-bound rather than
 * blit-bound -- so selective invalidation buys nothing here, and this
 * is intentionally a no-op rather than reworking every draw routine's
 * control flow to drop the calls.
 */
void nv_geforce3_redraw_area(NVGeForce3State *s, uint32_t offset,
                             uint32_t width, uint32_t height)
{
}

/* ---------------------------------------------------------------- */
/* Interrupts                                                       */

static uint32_t nv_geforce3_get_mc_intr(NVGeForce3State *s)
{
    uint32_t value = 0;

    if (s->bus_intr & s->bus_intr_en) {
        value |= 0x10000000;
    }
    if (s->fifo_intr & s->fifo_intr_en) {
        value |= 0x00000100;
    }
    if (s->graph_intr & s->graph_intr_en) {
        value |= 0x00001000;
    }
    if (s->crtc_intr & s->crtc_intr_en) {
        value |= 0x01000000;
    }
    return value;
}

void nv_geforce3_update_irq(NVGeForce3State *s)
{
    bool level = ((nv_geforce3_get_mc_intr(s) != 0 && (s->mc_intr_en & 1)) ||
                  (s->mc_soft_intr && (s->mc_intr_en & 2)));

    pci_set_irq(&s->parent_obj, level);
}

/* ---------------------------------------------------------------- */
/* CRTC / hardware cursor                                            */

static void nv_geforce3_svga_write_crtc(NVGeForce3State *s, unsigned index,
                                        uint8_t value)
{
    bool update_cursor_addr = false;

    if (index == 0x1c) {
        if (!(s->crtc.reg[index] & 0x80) && (value & 0x80) != 0) {
            /* matches a documented real-hardware/driver quirk */
            s->crtc_intr_en = 0x00000000;
            nv_geforce3_update_irq(s);
        }
    } else if (index == 0x2f || index == 0x30 || index == 0x31) {
        update_cursor_addr = true;
    } else if (index == 0x58) {
        return;
    }

    if (index <= NV_GEFORCE3_CRTC_MAX) {
        s->crtc.reg[index] = value;
    } else {
        qemu_log_mask(LOG_UNIMP, "nv-geforce3: crtc write unknown index "
                      "0x%02x\n", index);
        return;
    }

    if (update_cursor_addr) {
        s->hw_cursor.enabled = (s->crtc.reg[0x31] & 0x01) ||
            (s->crtc_cursor_config & 0x00000001);
        s->hw_cursor.vram = (s->crtc.reg[0x30] & 0x80) ||
            (s->crtc_cursor_config & 0x00000100);
        s->hw_cursor.offset = ((uint32_t)(s->crtc.reg[0x31] >> 2) << 11) |
            ((uint32_t)(s->crtc.reg[0x30] & 0x7F) << 17) |
            ((uint32_t)s->crtc.reg[0x2f] << 24);
        s->hw_cursor.offset += s->crtc_cursor_offset;
    }
}

/*
 * Legacy VGA-compatible register file, reached as byte-addressed
 * windows inside the MMIO BAR (see the header comment on
 * nv_geforce3_legacy_read/write callers). Nothing in this device's
 * own extended/native display path consults any of this; it exists
 * purely so bring-up code that probes standard VGA state gets
 * consistent shadow values instead of "unknown register" noise.
 */
static uint8_t nv_geforce3_legacy_read(NVGeForce3State *s, uint32_t port)
{
    switch (port) {
    case 0x3b4:
    case 0x3d4:
        return (uint8_t)s->crtc.index;
    case 0x3b5:
    case 0x3d5:
        if (s->crtc.index <= NV_GEFORCE3_CRTC_MAX) {
            return s->crtc.reg[s->crtc.index];
        }
        qemu_log_mask(LOG_UNIMP,
                      "nv-geforce3: crtc read unknown index 0x%02x\n",
                      s->crtc.index);
        return 0xff;
    case 0x3c0:
        return s->attr_index;
    case 0x3c1:
        return s->attr_reg[s->attr_index & 0x1f];
    case 0x3c2:
        return 0x10; /* Input Status 0: monitor presence (DAC sensing) */
    case 0x3c3:
        return s->feature_ctl;
    case 0x3c4:
        return s->seq_index;
    case 0x3c5:
        return s->seq_reg[s->seq_index & 7];
    case 0x3c6:
        return s->dac_mask;
    case 0x3c7:
        return 0x00; /* DAC state: ready */
    case 0x3c8:
        return s->dac_wr_index;
    case 0x3c9: {
        uint8_t value = s->palette[s->dac_rd_index][s->dac_state];

        if (++s->dac_state == 3) {
            s->dac_state = 0;
            s->dac_rd_index++;
        }
        return value;
    }
    case 0x3cc:
        return s->misc_output;
    case 0x3ce:
        return s->gr_index;
    case 0x3cf:
        return s->gr_reg[s->gr_index & 0xf];
    case 0x3d8:
        return s->feature_ctl;
    case 0x3da:
        /* Input Status 1: bit 3 = vertical retrace, approximated */
        s->attr_flipflop = false;
        return (s->crtc_intr & 1) ? 0x08 : 0x00;
    default:
        qemu_log_mask(LOG_UNIMP,
                      "nv-geforce3: legacy read unknown port 0x%03x\n", port);
        return 0xff;
    }
}

static void nv_geforce3_legacy_write(NVGeForce3State *s, uint32_t port,
                                     uint8_t value)
{
    switch (port) {
    case 0x3b4:
    case 0x3d4:
        s->crtc.index = value;
        return;
    case 0x3b5:
    case 0x3d5:
        if (s->crtc.index <= VGA_CRTC_MAX) {
            s->crtc.reg[s->crtc.index] = value;
        } else {
            nv_geforce3_svga_write_crtc(s, s->crtc.index, value);
        }
        return;
    case 0x3c0:
        if (!s->attr_flipflop) {
            s->attr_index = value & 0x1f;
        } else {
            s->attr_reg[s->attr_index & 0x1f] = value;
        }
        s->attr_flipflop = !s->attr_flipflop;
        return;
    case 0x3c2:
        s->misc_output = value;
        return;
    case 0x3c3:
        s->feature_ctl = value;
        return;
    case 0x3c4:
        s->seq_index = value;
        return;
    case 0x3c5:
        s->seq_reg[s->seq_index & 7] = value;
        return;
    case 0x3c6:
        s->dac_mask = value;
        return;
    case 0x3c7:
        s->dac_rd_index = value;
        s->dac_state = 0;
        return;
    case 0x3c8:
        s->dac_wr_index = value;
        s->dac_state = 0;
        return;
    case 0x3c9:
        s->palette[s->dac_wr_index][s->dac_state] = value;
        if (++s->dac_state == 3) {
            s->dac_state = 0;
            s->dac_wr_index++;
        }
        return;
    case 0x3ce:
        s->gr_index = value;
        return;
    case 0x3cf:
        s->gr_reg[s->gr_index & 0xf] = value;
        return;
    case 0x3d8:
        s->feature_ctl = value;
        return;
    default:
        qemu_log_mask(LOG_UNIMP,
                      "nv-geforce3: legacy write unknown port 0x%03x "
                      "value 0x%02x\n", port, value);
        return;
    }
}

/* ---------------------------------------------------------------- */
/* Register-level model: PMC/PBUS/PFIFO/PTIMER/PFB/PGRAPH/PCRTC/     */
/* PRAMDAC/PSTRAPS, plus the legacy-compat windows and PRAMIN/USER   */

static uint8_t nv_geforce3_register_read8(NVGeForce3State *s, uint32_t address)
{
    if (address >= 0x1800 && address < 0x1900) {
        return s->parent_obj.config[address - 0x1800];
    }
    if ((address >= 0xc0300 && address < 0xc0400) ||
        (address >= 0xc2300 && address < 0xc2400)) {
        uint32_t head = (address >> 13) & 1;
        uint32_t offset = address & 0xfff;

        if (offset == 0x3c3 || offset == 0x3c4 || offset == 0x3c5 ||
            offset == 0x3cc || offset == 0x3cf) {
            return head ? 0x00 : nv_geforce3_legacy_read(s, offset);
        }
        qemu_log_mask(LOG_UNIMP,
                      "nv-geforce3: unknown register 0x%08x read\n", address);
        return 0xff;
    }
    if ((address >= 0x601300 && address < 0x601400) ||
        (address >= 0x603300 && address < 0x603400)) {
        uint32_t head = (address >> 13) & 1;
        uint32_t offset = address & 0xfff;

        if (offset == 0x3b4 || offset == 0x3b5 || offset == 0x3c0 ||
            offset == 0x3c1 || offset == 0x3c2 || offset == 0x3d4 ||
            offset == 0x3d5 || offset == 0x3d8 || offset == 0x3da) {
            return head ? 0x00 : nv_geforce3_legacy_read(s, offset);
        }
        qemu_log_mask(LOG_UNIMP,
                      "nv-geforce3: unknown register 0x%08x read\n", address);
        return 0xff;
    }
    if ((address >= 0x681300 && address < 0x681400) ||
        (address >= 0x683300 && address < 0x683400)) {
        uint32_t head = (address >> 13) & 1;
        uint32_t offset = address & 0xfff;

        if (offset >= 0x3c6 && offset <= 0x3c9) {
            return head ? 0x00 : nv_geforce3_legacy_read(s, offset);
        }
        qemu_log_mask(LOG_UNIMP,
                      "nv-geforce3: unknown register 0x%08x read\n", address);
        return 0xff;
    }
    if (address >= 0x700000 && address < 0x800000) {
        return s->vram_ptr[((address - 0x700000) ^ s->ramin_flip) &
                           s->memsize_mask];
    }
    return (uint8_t)nv_geforce3_register_read32(s, address & ~3u) >>
           ((address & 3) * 8);
}

static void nv_geforce3_register_write8(NVGeForce3State *s, uint32_t address,
                                        uint8_t value)
{
    if ((address >= 0xc0300 && address < 0xc0400) ||
        (address >= 0xc2300 && address < 0xc2400)) {
        uint32_t head = (address >> 13) & 1;
        uint32_t offset = address & 0xfff;

        if (offset == 0x3c2 || offset == 0x3c3 || offset == 0x3c4 ||
            offset == 0x3c5 || offset == 0x3ce || offset == 0x3cf) {
            if (!head) {
                nv_geforce3_legacy_write(s, offset, value);
            }
            return;
        }
        qemu_log_mask(LOG_UNIMP,
                      "nv-geforce3: unknown register 0x%08x write\n",
                      address);
        return;
    }
    if ((address >= 0x601300 && address < 0x601400) ||
        (address >= 0x603300 && address < 0x603400)) {
        uint32_t head = (address >> 13) & 1;
        uint32_t offset = address & 0xfff;

        if (offset == 0x3b4 || offset == 0x3b5 || offset == 0x3c0 ||
            offset == 0x3c1 || offset == 0x3c2 || offset == 0x3d4 ||
            offset == 0x3d5 || offset == 0x3da) {
            if (!head) {
                nv_geforce3_legacy_write(s, offset, value);
            }
            return;
        }
        qemu_log_mask(LOG_UNIMP,
                      "nv-geforce3: unknown register 0x%08x write\n",
                      address);
        return;
    }
    if ((address >= 0x681300 && address < 0x681400) ||
        (address >= 0x683300 && address < 0x683400)) {
        uint32_t head = (address >> 13) & 1;
        uint32_t offset = address & 0xfff;

        if (offset >= 0x3c6 && offset <= 0x3c9) {
            if (!head) {
                nv_geforce3_legacy_write(s, offset, value);
            }
            return;
        }
        qemu_log_mask(LOG_UNIMP,
                      "nv-geforce3: unknown register 0x%08x write\n",
                      address);
        return;
    }
    if (address >= 0x700000 && address < 0x800000) {
        s->vram_ptr[((address - 0x700000) ^ s->ramin_flip) &
                    s->memsize_mask] = value;
        return;
    }
    nv_geforce3_register_write32(s, address,
        deposit32(nv_geforce3_register_read32(s, address & ~3u),
                  (address & 3) * 8, 8, value));
}

static uint32_t nv_geforce3_register_read32(NVGeForce3State *s,
                                            uint32_t address)
{
    uint32_t value;

    if (address == 0x000000) {
        value = 0x020000A3; /* NV_PMC_BOOT_0: NV20 */
    } else if (address == 0x000100) {
        value = nv_geforce3_get_mc_intr(s);
        if (s->mc_soft_intr) {
            value |= 0x80000000;
        }
    } else if (address == 0x000140) {
        value = s->mc_intr_en;
    } else if (address == 0x000200) {
        value = s->mc_enable;
    } else if (address >= 0x001800 && address < 0x001900) {
        value = pci_get_long(s->parent_obj.config + (address - 0x1800));
    } else if (address == 0x001100) {
        value = s->bus_intr;
    } else if (address == 0x001140) {
        value = s->bus_intr_en;
    } else if (address == 0x002100) {
        value = s->fifo_intr;
    } else if (address == 0x002140) {
        value = s->fifo_intr_en;
    } else if (address == 0x002210) {
        value = s->fifo_ramht;
    } else if (address == 0x002214) {
        value = s->fifo_ramfc;
    } else if (address == 0x002218) {
        value = s->fifo_ramro;
    } else if (address == 0x002400) {
        value = (s->fifo_cache1_get == s->fifo_cache1_put) ? 0x00000010 : 0;
    } else if (address == 0x002504) {
        value = s->fifo_mode;
    } else if (address == 0x003200) {
        value = s->fifo_cache1_push0;
    } else if (address == 0x003204) {
        value = s->fifo_cache1_push1;
    } else if (address == 0x003210) {
        value = s->fifo_cache1_put;
    } else if (address == 0x003214) {
        value = (s->fifo_cache1_get == s->fifo_cache1_put) ? 0x00000010 : 0;
    } else if (address == 0x003220) {
        value = s->fifo_cache1_dma_push;
    } else if (address == 0x00322c) {
        value = s->fifo_cache1_dma_instance;
    } else if (address == 0x003230) {
        value = 0x80000000;
    } else if (address == 0x003240) {
        value = s->fifo_cache1_dma_put;
    } else if (address == 0x003244) {
        value = s->fifo_cache1_dma_get;
    } else if (address == 0x003248) {
        value = s->fifo_cache1_ref_cnt;
    } else if (address == 0x003250) {
        if (s->fifo_cache1_get != s->fifo_cache1_put) {
            s->fifo_cache1_pull0 |= 0x00000100;
        }
        value = s->fifo_cache1_pull0;
    } else if (address == 0x003270) {
        value = s->fifo_cache1_get;
    } else if (address == 0x0032e0) {
        value = s->fifo_grctx_instance;
    } else if (address == 0x003304) {
        value = 0x00000001;
    } else if (address >= 0x003800 && address < 0x004000) {
        uint32_t offset = address - 0x3800;
        uint32_t index = offset / 8;

        value = (offset % 8 == 0) ? s->fifo_cache1_method[index] :
                                     s->fifo_cache1_data[index];
    } else if (address == 0x009100) {
        value = s->timer_intr;
    } else if (address == 0x009140) {
        value = s->timer_intr_en;
    } else if (address == 0x009200) {
        value = s->timer_num;
    } else if (address == 0x009210) {
        value = s->timer_den;
    } else if (address == 0x009400) {
        value = (uint32_t)nv_geforce3_get_current_time(s);
    } else if (address == 0x009410) {
        value = (uint32_t)(nv_geforce3_get_current_time(s) >> 32);
    } else if (address == 0x009420) {
        value = s->timer_alarm;
    } else if ((address >= 0xc0300 && address < 0xc0400) ||
               (address >= 0xc2300 && address < 0xc2400)) {
        value = nv_geforce3_register_read8(s, address);
    } else if (address == 0x10020c) {
        value = s->vram_size;
    } else if (address == 0x100320) {
        value = 0x00007fff; /* PFB_ZCOMP_SIZE, NV20 */
    } else if (address == 0x101000) {
        value = s->straps0_primary;
    } else if (address == 0x400100) {
        value = s->graph_intr;
    } else if (address == 0x400108) {
        value = s->graph_nsource;
    } else if (address == 0x400140) {
        value = s->graph_intr_en;
    } else if (address == 0x40014c) {
        value = s->graph_ctx_switch1;
    } else if (address == 0x400150) {
        value = s->graph_ctx_switch2;
    } else if (address == 0x400158) {
        value = s->graph_ctx_switch4;
    } else if (address == 0x40032c) {
        value = s->graph_ctxctl_cur;
    } else if (address == 0x400700) {
        value = s->graph_status;
    } else if (address == 0x400704) {
        value = s->graph_trapped_addr;
    } else if (address == 0x400708) {
        value = s->graph_trapped_data;
    } else if (address == 0x400718) {
        value = s->graph_notify;
    } else if (address == 0x400720) {
        value = s->graph_fifo;
    } else if (address == 0x400724) {
        value = s->graph_bpixel;
    } else if (address == 0x400780) {
        value = s->graph_channel_ctx_table;
    } else if (address == 0x400820) {
        value = s->graph_offset0;
    } else if (address == 0x400850) {
        value = s->graph_pitch0;
    } else if (address == 0x600100) {
        value = s->crtc_intr;
    } else if (address == 0x600140) {
        value = s->crtc_intr_en;
    } else if (address == 0x600800) {
        value = s->crtc_start;
    } else if (address == 0x600804) {
        value = s->crtc_config;
    } else if (address == 0x600808) {
        s->crtc_raster_pos ^= 1;
        value = s->crtc_raster_pos;
    } else if (address == 0x60080c) {
        value = s->crtc_cursor_offset;
    } else if (address == 0x600810) {
        value = s->crtc_cursor_config;
    } else if (address == 0x60081c) {
        value = s->crtc_gpio_ext;
    } else if ((address >= 0x601300 && address < 0x601400) ||
               (address >= 0x603300 && address < 0x603400)) {
        value = nv_geforce3_register_read8(s, address);
    } else if (address == 0x680300) {
        value = s->ramdac_cu_start_pos;
    } else if (address == 0x680404) {
        value = 0x00000000;
    } else if (address == 0x680508) {
        value = s->ramdac_vpll;
    } else if (address == 0x68050c) {
        value = s->ramdac_pll_select;
    } else if (address == 0x680600) {
        value = s->ramdac_general_control;
    } else if (address == 0x680828) {
        value = 0x00000000; /* second head: no flat panel */
    } else if ((address >= 0x681300 && address < 0x681400) ||
               (address >= 0x683300 && address < 0x683400)) {
        value = nv_geforce3_register_read8(s, address);
    } else if (address >= 0x700000 && address < 0x800000) {
        uint32_t offset = address & 0x000fffff;

        if (offset & 3) {
            value = (uint32_t)nv_geforce3_ramin_read8(s, offset + 0) << 0 |
                    (uint32_t)nv_geforce3_ramin_read8(s, offset + 1) << 8 |
                    (uint32_t)nv_geforce3_ramin_read8(s, offset + 2) << 16 |
                    (uint32_t)nv_geforce3_ramin_read8(s, offset + 3) << 24;
        } else {
            value = nv_geforce3_ramin_read32(s, offset);
        }
    } else if ((address >= 0x800000 && address < 0xA00000) ||
               (address >= 0xC00000 && address < 0xE00000)) {
        uint32_t chid, offset, curchid;

        if (address >= 0x800000 && address < 0xA00000) {
            chid = (address >> 16) & 0x1F;
            offset = address & 0x1FFF;
        } else {
            chid = (address >> 12) & 0x1FF;
            if (chid >= NV_GEFORCE3_CHANNEL_COUNT) {
                qemu_log_mask(LOG_GUEST_ERROR,
                              "nv-geforce3: channel id >= 32\n");
                chid = 0;
            }
            offset = address & 0x1FF;
        }
        value = 0x00000000;
        curchid = s->fifo_cache1_push1 & 0x1F;
        if (offset == 0x54 && address >= 0xC00000 && address < 0xE00000) {
            if (s->chs[chid].subr_active) {
                value = s->chs[chid].subr_return;
            } else if (curchid == chid) {
                value = s->fifo_cache1_dma_get;
            } else {
                value = nv_geforce3_ramfc_read32(s, chid, 0x4);
            }
        } else if (offset == 0x10) {
            value = 0xffff;
        } else if (offset >= 0x40 && offset <= 0x48) {
            if (curchid == chid) {
                if (offset == 0x40) {
                    value = s->fifo_cache1_dma_put;
                } else if (offset == 0x44) {
                    value = s->fifo_cache1_dma_get;
                } else {
                    value = s->fifo_cache1_ref_cnt;
                }
            } else if (offset == 0x40) {
                value = nv_geforce3_ramfc_read32(s, chid, 0x0);
            } else if (offset == 0x44) {
                value = nv_geforce3_ramfc_read32(s, chid, 0x4);
            } else {
                value = nv_geforce3_ramfc_read32(s, chid, 0x8);
            }
        } else {
            qemu_log_mask(LOG_UNIMP,
                          "nv-geforce3: unknown FIFO offset 0x%08x\n",
                          offset);
        }
    } else {
        value = s->unk_regs[(address & (NV_GEFORCE3_MMIO_SIZE - 1)) / 4];
    }
    return value;
}

static void nv_geforce3_register_write32(NVGeForce3State *s,
                                         uint32_t address, uint32_t value)
{
    if (address == 0x000100) {
        s->mc_soft_intr = (value >> 31) != 0;
        nv_geforce3_update_irq(s);
    } else if (address == 0x000140) {
        s->mc_intr_en = value;
        nv_geforce3_update_irq(s);
    } else if (address == 0x000200) {
        s->mc_enable = value;
    } else if (address >= 0x001800 && address < 0x001900) {
        pci_default_write_config(&s->parent_obj, address - 0x1800, value, 4);
    } else if (address == 0x001100) {
        s->bus_intr &= ~value;
        nv_geforce3_update_irq(s);
    } else if (address == 0x001140) {
        s->bus_intr_en = value;
        nv_geforce3_update_irq(s);
    } else if (address == 0x002100) {
        s->fifo_intr &= ~value;
        nv_geforce3_update_irq(s);
    } else if (address == 0x002140) {
        s->fifo_intr_en = value;
        nv_geforce3_update_irq(s);
    } else if (address == 0x002210) {
        s->fifo_ramht = value;
    } else if (address == 0x002214) {
        s->fifo_ramfc = value;
    } else if (address == 0x002218) {
        s->fifo_ramro = value;
    } else if (address == 0x002504) {
        bool process = (s->fifo_mode | value) != s->fifo_mode;

        s->fifo_mode = value;
        if (process) {
            nv_geforce3_fifo_process(s);
        }
    } else if (address == 0x003200) {
        s->fifo_cache1_push0 = value;
        if (s->fifo_cache1_push0 & 1) {
            nv_geforce3_fifo_process(s);
        }
    } else if (address == 0x003204) {
        s->fifo_cache1_push1 = value;
    } else if (address == 0x003210) {
        s->fifo_cache1_put = value;
    } else if (address == 0x003220) {
        s->fifo_cache1_dma_push = value;
    } else if (address == 0x00322c) {
        s->fifo_cache1_dma_instance = value;
    } else if (address == 0x003240) {
        s->fifo_cache1_dma_put = value;
    } else if (address == 0x003244) {
        s->fifo_cache1_dma_get = value;
    } else if (address == 0x003248) {
        s->fifo_cache1_ref_cnt = value;
    } else if (address == 0x003250) {
        s->fifo_cache1_pull0 = value;
        if (s->fifo_cache1_pull0 & 1) {
            nv_geforce3_fifo_process(s);
        }
    } else if (address == 0x003270) {
        s->fifo_cache1_get = value & (NV_GEFORCE3_CACHE1_SIZE * 4 - 1);
        if (s->fifo_cache1_get != s->fifo_cache1_put) {
            s->fifo_intr |= 0x00000001;
        } else {
            s->fifo_intr &= ~0x00000001;
            s->fifo_cache1_pull0 &= ~0x00000100;
            if (s->fifo_wait_soft) {
                s->fifo_wait_soft = false;
                nv_geforce3_update_fifo_wait(s);
                nv_geforce3_fifo_process(s);
            }
        }
        nv_geforce3_update_irq(s);
    } else if (address == 0x0032e0) {
        s->fifo_grctx_instance = value;
    } else if (address == 0x009100) {
        s->timer_intr &= ~value;
    } else if (address == 0x009140) {
        s->timer_intr_en = value;
    } else if (address == 0x009200) {
        s->timer_num = value;
    } else if (address == 0x009210) {
        s->timer_den = value;
    } else if (address == 0x009400 || address == 0x009410) {
        s->timer_inittime2 = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
        if (address == 0x009400) {
            s->timer_inittime1 = (s->timer_inittime1 &
                                  0xFFFFFFFF00000000ULL) | value;
        } else {
            s->timer_inittime1 = (s->timer_inittime1 &
                                  0x00000000FFFFFFFFULL) |
                                  ((uint64_t)value << 32);
        }
    } else if (address == 0x009420) {
        s->timer_alarm = value;
    } else if ((address >= 0xc0300 && address < 0xc0400) ||
               (address >= 0xc2300 && address < 0xc2400)) {
        nv_geforce3_register_write8(s, address, (uint8_t)value);
    } else if (address == 0x101000) {
        s->straps0_primary = (value >> 31) ? value :
                             s->straps0_primary_original;
    } else if (address == 0x400100) {
        s->graph_intr &= ~value;
        nv_geforce3_update_irq(s);
        if (s->fifo_wait_notify && s->graph_intr == 0) {
            s->fifo_wait_notify = false;
            nv_geforce3_update_fifo_wait(s);
            nv_geforce3_fifo_process(s);
        }
    } else if (address == 0x400108) {
        s->graph_nsource = value;
    } else if (address == 0x400140) {
        s->graph_intr_en = value;
        nv_geforce3_update_irq(s);
    } else if (address == 0x40014c) {
        s->graph_ctx_switch1 = value;
    } else if (address == 0x400150) {
        s->graph_ctx_switch2 = value;
    } else if (address == 0x400158) {
        s->graph_ctx_switch4 = value;
    } else if (address == 0x40032c) {
        s->graph_ctxctl_cur = value;
    } else if (address == 0x400700) {
        s->graph_status = value;
    } else if (address == 0x400704) {
        s->graph_trapped_addr = value;
    } else if (address == 0x400708) {
        s->graph_trapped_data = value;
    } else if (address == 0x400718) {
        s->graph_notify = value;
    } else if (address == 0x40071c) {
        if (value & 0x00000002) {
            s->graph_flip_read++;
            if (s->graph_flip_modulo) {
                s->graph_flip_read %= s->graph_flip_modulo;
            }
            if (s->fifo_wait_flip &&
                s->graph_flip_read != s->graph_flip_write) {
                s->fifo_wait_flip = false;
                nv_geforce3_update_fifo_wait(s);
                nv_geforce3_fifo_process(s);
            }
        }
    } else if (address == 0x400720) {
        s->graph_fifo = value;
    } else if (address == 0x400724) {
        s->graph_bpixel = value;
    } else if (address == 0x400780) {
        s->graph_channel_ctx_table = value;
    } else if (address == 0x400820) {
        s->graph_offset0 = value;
    } else if (address == 0x400850) {
        s->graph_pitch0 = value;
    } else if (address == 0x600100) {
        s->crtc_intr &= ~value;
        nv_geforce3_update_irq(s);
    } else if (address == 0x600140) {
        s->crtc_intr_en = value;
        nv_geforce3_update_irq(s);
    } else if (address == 0x600800) {
        s->crtc_start = value;
        s->mode_dirty = true;
    } else if (address == 0x600804) {
        s->crtc_config = value;
    } else if (address == 0x60080c) {
        s->crtc_cursor_offset = value;
        s->hw_cursor.offset = s->crtc_cursor_offset;
    } else if (address == 0x600810) {
        s->crtc_cursor_config = value;
        s->hw_cursor.enabled = (s->crtc.reg[0x31] & 0x01) ||
                              (value & 0x00000001);
        s->hw_cursor.vram = (s->crtc.reg[0x30] & 0x80) ||
                           (value & 0x00000100);
        s->hw_cursor.size = (value & 0x00010000) ? 64 : 32;
        s->hw_cursor.bpp32 = (value & 0x00001000) != 0;
    } else if (address == 0x60081c) {
        s->crtc_gpio_ext = value;
    } else if ((address >= 0x601300 && address < 0x601400) ||
               (address >= 0x603300 && address < 0x603400)) {
        nv_geforce3_register_write8(s, address, (uint8_t)value);
    } else if (address == 0x680300) {
        s->ramdac_cu_start_pos = value;
        s->hw_cursor.x = (int32_t)(s->ramdac_cu_start_pos << 20) >> 20;
        s->hw_cursor.y = (int32_t)(s->ramdac_cu_start_pos << 4) >> 20;
    } else if (address == 0x680508) {
        s->ramdac_vpll = value;
    } else if (address == 0x68050c) {
        s->ramdac_pll_select = value;
    } else if (address == 0x680600) {
        s->ramdac_general_control = value;
    } else if ((address >= 0x681300 && address < 0x681400) ||
               (address >= 0x683300 && address < 0x683400)) {
        nv_geforce3_register_write8(s, address, (uint8_t)value);
    } else if (address >= 0x700000 && address < 0x800000) {
        nv_geforce3_ramin_write32(s, address - 0x700000, value);
    } else if ((address >= 0x800000 && address < 0xA00000) ||
               (address >= 0xC00000 && address < 0xE00000)) {
        uint32_t chid, offset;

        if (address >= 0x800000 && address < 0xA00000) {
            chid = (address >> 16) & 0x1F;
            offset = address & 0x1FFF;
        } else {
            chid = (address >> 12) & 0x1FF;
            if (chid >= NV_GEFORCE3_CHANNEL_COUNT) {
                qemu_log_mask(LOG_GUEST_ERROR,
                              "nv-geforce3: channel id >= 32\n");
                chid = 0;
            }
            offset = address & 0x1FF;
        }
        if (s->fifo_mode & (1u << chid)) {
            if (offset == 0x40) {
                uint32_t curchid = s->fifo_cache1_push1 & 0x1F;

                if (curchid == chid) {
                    s->fifo_cache1_dma_put = value;
                } else {
                    nv_geforce3_ramfc_write32(s, chid, 0x0, value);
                }
                nv_geforce3_fifo_process_chid(s, chid);
            }
        } else if (address >= 0x800000 && address < 0xA00000) {
            uint32_t subc = (address >> 13) & 7;

            nv_geforce3_execute_command(s, chid, subc, offset / 4, value);
        }
    } else {
        s->unk_regs[(address & (NV_GEFORCE3_MMIO_SIZE - 1)) / 4] = value;
    }
}

/* ---------------------------------------------------------------- */
/* MMIO (BAR0)                                                       */

static uint64_t nv_geforce3_mmio_read(void *opaque, hwaddr addr,
                                      unsigned size)
{
    NVGeForce3State *s = opaque;
    uint64_t val = 0;
    unsigned i;

    if (size == 1) {
        return nv_geforce3_register_read8(s, addr);
    }
    for (i = 0; i < size; i += 4) {
        uint32_t dw = nv_geforce3_register_read32(s, (addr + i) & ~3u);

        val |= (uint64_t)dw << (i * 8);
    }
    return extract64(val, 0, size * 8);
}

static void nv_geforce3_mmio_write(void *opaque, hwaddr addr, uint64_t data,
                                   unsigned size)
{
    NVGeForce3State *s = opaque;
    unsigned i;

    if (size == 1) {
        nv_geforce3_register_write8(s, addr, (uint8_t)data);
        return;
    }
    for (i = 0; i < size; i += 4) {
        nv_geforce3_register_write32(s, (addr + i) & ~3u,
                                     (uint32_t)(data >> (i * 8)));
    }
}

static const MemoryRegionOps nv_geforce3_mmio_ops = {
    .read = nv_geforce3_mmio_read,
    .write = nv_geforce3_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

/* ---------------------------------------------------------------- */
/* Display                                                           */

static bool nv_geforce3_get_mode(NVGeForce3State *s, NVGeForce3Mode *mode,
                                 uint32_t *fb_offset)
{
    uint8_t crtc28 = s->crtc.reg[0x28] & 0x7F;
    uint32_t top_offset, pitch, width, height;
    uint8_t bpp;

    memset(mode, 0, sizeof(*mode));
    if (crtc28 == 0) {
        return false;
    }

    top_offset = s->crtc.reg[0x0d] |
                ((uint32_t)s->crtc.reg[0x0c] << 8) |
                (((uint32_t)s->crtc.reg[0x19] & 0x1F) << 16);
    top_offset <<= 2;
    top_offset += s->crtc_start;

    pitch = s->crtc.reg[0x13] |
           (((uint32_t)s->crtc.reg[0x19] >> 5) << 8) |
           ((((uint32_t)s->crtc.reg[0x42] >> 6) & 1) << 11);
    pitch <<= 3;

    switch (crtc28) {
    case 0x01:
        bpp = 1;
        break;
    case 0x02:
        bpp = 2;
        break;
    case 0x03:
        bpp = 4;
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR, "nv-geforce3: unknown bpp code %u\n",
                      crtc28);
        return false;
    }

    width = ((uint32_t)s->crtc.reg[1] +
            (((uint32_t)s->crtc.reg[0x2D] & 0x02) << 7) + 1) * 8;
    height = (s->crtc.reg[18] |
             (((uint32_t)s->crtc.reg[7] & 0x02) << 7) |
             (((uint32_t)s->crtc.reg[7] & 0x40) << 3) |
             (((uint32_t)s->crtc.reg[0x25] & 0x02) << 9) |
             (((uint32_t)s->crtc.reg[0x41] & 0x04) << 9)) + 1;

    if (width < 16 || height < 16 || pitch < width * bpp) {
        return false;
    }
    if ((uint64_t)top_offset + (uint64_t)pitch * height > s->vram_size) {
        return false;
    }

    mode->width = width;
    mode->height = height;
    mode->pitch = pitch;
    mode->bpp = bpp;
    *fb_offset = top_offset;
    return true;
}

static void nv_geforce3_draw_8bpp(NVGeForce3State *s, DisplaySurface *ds,
                                  const NVGeForce3Mode *mode,
                                  uint32_t fb_offset)
{
    uint8_t *src = s->vram_ptr + fb_offset;
    int x, y;

    for (y = 0; y < mode->height; y++) {
        uint32_t *dst = (uint32_t *)(surface_data(ds) + y * surface_stride(ds));

        for (x = 0; x < mode->width; x++) {
            uint8_t idx = src[x];

            dst[x] = 0xff000000u | ((uint32_t)s->palette[idx][0] << 16) |
                     ((uint32_t)s->palette[idx][1] << 8) | s->palette[idx][2];
        }
        src += mode->pitch;
    }
}

static void nv_geforce3_draw_16bpp(NVGeForce3State *s, DisplaySurface *ds,
                                   const NVGeForce3Mode *mode,
                                   uint32_t fb_offset)
{
    uint8_t *src = s->vram_ptr + fb_offset;
    int x, y;

    for (y = 0; y < mode->height; y++) {
        uint32_t *dst = (uint32_t *)(surface_data(ds) + y * surface_stride(ds));

        for (x = 0; x < mode->width; x++) {
            uint16_t pixel = (uint16_t)src[2 * x] |
                             ((uint16_t)src[2 * x + 1] << 8);
            uint8_t r = ((pixel >> 11) & 0x1f) << 3;
            uint8_t g = ((pixel >> 5) & 0x3f) << 2;
            uint8_t b = (pixel & 0x1f) << 3;

            r |= r >> 5;
            g |= g >> 6;
            b |= b >> 5;
            dst[x] = 0xff000000u | ((uint32_t)r << 16) | ((uint32_t)g << 8) |
                     b;
        }
        src += mode->pitch;
    }
}

static void nv_geforce3_draw_32bpp(NVGeForce3State *s, DisplaySurface *ds,
                                   const NVGeForce3Mode *mode,
                                   uint32_t fb_offset)
{
    uint8_t *src = s->vram_ptr + fb_offset;
    int x, y;

    for (y = 0; y < mode->height; y++) {
        uint32_t *dst = (uint32_t *)(surface_data(ds) + y * surface_stride(ds));

        for (x = 0; x < mode->width; x++) {
            /* chip-native little-endian: B,G,R,X in VRAM */
            dst[x] = 0xff000000u | ((uint32_t)src[4 * x + 2] << 16) |
                     ((uint32_t)src[4 * x + 1] << 8) | src[4 * x];
        }
        src += mode->pitch;
    }
}

static uint16_t nv_geforce3_cursor_read16(NVGeForce3State *s, uint32_t addr)
{
    return s->hw_cursor.vram ? nv_geforce3_vram_read16(s, addr) :
                               nv_geforce3_ramin_read16(s, addr);
}

static uint32_t nv_geforce3_cursor_read32(NVGeForce3State *s, uint32_t addr)
{
    return s->hw_cursor.vram ? nv_geforce3_vram_read32(s, addr) :
                               nv_geforce3_ramin_read32(s, addr);
}

/*
 * Hardware cursor: a QEMUCursor sprite rebuilt from the current
 * cursor image every refresh. Real hardware's ARGB1555 format XORs
 * the destination when its "alpha" bit is clear (rather than simply
 * hiding the pixel); since that only matters for cursor images that
 * store a non-zero RGB in their transparent texels (an edge-outline
 * trick), and QEMUCursor has no XOR compositing mode, transparent
 * (alpha 0) is used there instead -- an exact match for the far more
 * common case of transparent texels also storing RGB 0.
 */
static void nv_geforce3_cursor_apply(NVGeForce3State *s)
{
    unsigned size = s->hw_cursor.size;
    QEMUCursor *c;
    unsigned x, y;

    if (!s->con) {
        return;
    }
    if (!s->hw_cursor.enabled) {
        qemu_console_set_mouse(s->con, 0, 0, false);
        return;
    }

    c = cursor_alloc(size, size);
    for (y = 0; y < size; y++) {
        for (x = 0; x < size; x++) {
            uint32_t pitch = size * (s->hw_cursor.bpp32 ? 4 : 2);
            uint32_t ofs = s->hw_cursor.offset + y * pitch;
            uint32_t argb;

            if (s->hw_cursor.bpp32) {
                uint32_t raw = nv_geforce3_cursor_read32(s, ofs + x * 4);
                uint8_t b = raw, g = raw >> 8, r = raw >> 16, a = raw >> 24;

                argb = ((uint32_t)a << 24) | ((uint32_t)r << 16) |
                       ((uint32_t)g << 8) | b;
            } else {
                uint16_t raw = nv_geforce3_cursor_read16(s, ofs + x * 2);
                uint8_t a = (raw & 0x8000) ? 0xff : 0x00;
                uint8_t r = ((raw >> 10) & 0x1f) << 3;
                uint8_t g = ((raw >> 5) & 0x1f) << 3;
                uint8_t b = (raw & 0x1f) << 3;

                argb = ((uint32_t)a << 24) | ((uint32_t)r << 16) |
                       ((uint32_t)g << 8) | b;
            }
            c->data[y * size + x] = argb;
        }
    }
    qemu_console_set_cursor(s->con, c);
    cursor_unref(c);
    qemu_console_set_mouse(s->con, s->hw_cursor.x, s->hw_cursor.y, true);
}

static bool nv_geforce3_gfx_update(void *opaque)
{
    NVGeForce3State *s = opaque;
    DisplaySurface *ds;
    NVGeForce3Mode mode;
    uint32_t fb_offset = 0;

    if (!nv_geforce3_get_mode(s, &mode, &fb_offset)) {
        if (s->mode.width != 0) {
            memset(&s->mode, 0, sizeof(s->mode));
            qemu_console_resize(s->con, 640, 480);
            ds = qemu_console_surface(s->con);
            memset(surface_data(ds), 0,
                   (size_t)surface_stride(ds) * surface_height(ds));
            qemu_console_update_full(s->con);
        }
        return true;
    }

    ds = qemu_console_surface(s->con);
    if (mode.width != s->mode.width || mode.height != s->mode.height ||
        surface_bits_per_pixel(ds) != 32) {
        qemu_console_resize(s->con, mode.width, mode.height);
        ds = qemu_console_surface(s->con);
    }
    s->mode = mode;
    s->fb_offset = fb_offset;

    switch (mode.bpp) {
    case 1:
        nv_geforce3_draw_8bpp(s, ds, &mode, fb_offset);
        break;
    case 2:
        nv_geforce3_draw_16bpp(s, ds, &mode, fb_offset);
        break;
    case 4:
        nv_geforce3_draw_32bpp(s, ds, &mode, fb_offset);
        break;
    default:
        break;
    }
    nv_geforce3_cursor_apply(s);
    qemu_console_update_full(s->con);
    return true;
}

static const GraphicHwOps nv_geforce3_gfx_ops = {
    .gfx_update = nv_geforce3_gfx_update,
};

/* ---------------------------------------------------------------- */
/* VBLANK                                                             */

static void nv_geforce3_vblank_timer_tick(void *opaque)
{
    NVGeForce3State *s = opaque;

    s->crtc_intr |= 0x00000001;
    nv_geforce3_update_irq(s);
    if (s->fifo_wait_acquire) {
        s->fifo_wait_acquire = false;
        nv_geforce3_update_fifo_wait(s);
        nv_geforce3_fifo_process(s);
    }
    timer_mod(s->vblank_timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +
              NV_GEFORCE3_VBLANK_PERIOD_NS);
}

/* ---------------------------------------------------------------- */
/* PCI device lifecycle                                              */

static void nv_geforce3_reset_hold(Object *obj, ResetType type)
{
    NVGeForce3State *s = NV_GEFORCE3(obj);
    unsigned i;

    memset(&s->crtc, 0, sizeof(s->crtc));
    s->crtc.index = NV_GEFORCE3_CRTC_MAX + 1;

    s->mc_soft_intr = false;
    s->mc_intr_en = 0;
    s->mc_enable = 0;
    s->bus_intr = 0;
    s->bus_intr_en = 0;
    s->fifo_wait = false;
    s->fifo_wait_soft = false;
    s->fifo_wait_notify = false;
    s->fifo_wait_flip = false;
    s->fifo_wait_acquire = false;
    s->fifo_intr = 0;
    s->fifo_intr_en = 0;
    s->fifo_ramht = 0;
    s->fifo_ramfc = 0;
    s->fifo_ramro = 0;
    s->fifo_mode = 0;
    s->fifo_cache1_push0 = 0;
    s->fifo_cache1_push1 = 0;
    s->fifo_cache1_put = 0;
    s->fifo_cache1_dma_push = 0;
    s->fifo_cache1_dma_instance = 0;
    s->fifo_cache1_dma_put = 0;
    s->fifo_cache1_dma_get = 0;
    s->fifo_cache1_ref_cnt = 0;
    s->fifo_cache1_pull0 = 0;
    s->fifo_cache1_semaphore = 0;
    s->fifo_cache1_get = 0;
    s->fifo_grctx_instance = 0;
    memset(s->fifo_cache1_method, 0, sizeof(s->fifo_cache1_method));
    memset(s->fifo_cache1_data, 0, sizeof(s->fifo_cache1_data));

    s->rma_addr = 0;
    s->timer_intr = 0;
    s->timer_intr_en = 0;
    s->timer_num = 0;
    s->timer_den = 0;
    s->timer_inittime1 = 0;
    s->timer_inittime2 = 0;
    s->timer_alarm = 0;

    s->graph_intr = 0;
    s->graph_nsource = 0;
    s->graph_intr_en = 0;
    s->graph_ctx_switch1 = 0;
    s->graph_ctx_switch2 = 0;
    s->graph_ctx_switch4 = 0;
    s->graph_ctxctl_cur = 0;
    s->graph_status = 0;
    s->graph_trapped_addr = 0;
    s->graph_trapped_data = 0;
    s->graph_flip_read = 0;
    s->graph_flip_write = 0;
    s->graph_flip_modulo = 0;
    s->graph_notify = 0;
    s->graph_fifo = 0;
    s->graph_bpixel = 0;
    s->graph_channel_ctx_table = 0;
    s->graph_offset0 = 0;
    s->graph_pitch0 = 0;

    s->crtc_intr = 0;
    s->crtc_intr_en = 0;
    s->crtc_start = 0;
    s->crtc_config = 0;
    s->crtc_raster_pos = 0;
    s->crtc_cursor_offset = 0;
    s->crtc_cursor_config = 0;
    s->crtc_gpio_ext = 0;

    s->ramdac_cu_start_pos = 0;
    s->ramdac_vpll = 0;
    s->ramdac_pll_select = 0;
    s->ramdac_general_control = 0;

    memset(s->chs, 0, sizeof(s->chs));
    for (i = 0; i < NV_GEFORCE3_CHANNEL_COUNT; i++) {
        s->chs[i].swzs_color_bytes = 1;
        s->chs[i].s2d_color_bytes = 1;
        s->chs[i].d3d_color_bytes = 1;
        s->chs[i].d3d_depth_bytes = 1;
    }

    memset(s->unk_regs, 0, NV_GEFORCE3_MMIO_SIZE);

    memset(&s->hw_cursor, 0, sizeof(s->hw_cursor));
    s->hw_cursor.size = 32;

    s->misc_output = 0;
    s->feature_ctl = 0;
    s->seq_index = 0;
    memset(s->seq_reg, 0, sizeof(s->seq_reg));
    s->gr_index = 0;
    memset(s->gr_reg, 0, sizeof(s->gr_reg));
    s->attr_index = 0;
    memset(s->attr_reg, 0, sizeof(s->attr_reg));
    s->attr_flipflop = false;
    s->dac_mask = 0xff;
    s->dac_wr_index = 0;
    s->dac_rd_index = 0;
    s->dac_state = 0;
    memset(s->palette, 0, sizeof(s->palette));

    /*
     * Real hardware straps: matches the real board with the
     * exception of disabled TV-out.
     */
    s->straps0_primary_original = 0x7FF86C6B | 0x00000180;
    s->straps0_primary = s->straps0_primary_original;
    s->ramin_flip = s->vram_size - 64;
    s->memsize_mask = s->vram_size - 1;

    memset(&s->mode, 0, sizeof(s->mode));
    s->mode_dirty = true;
}

static void nv_geforce3_realize(PCIDevice *dev, Error **errp)
{
    NVGeForce3State *s = NV_GEFORCE3(dev);
    Object *obj = OBJECT(dev);

    s->con = qemu_graphic_console_create(DEVICE(dev), 0,
                                         &nv_geforce3_gfx_ops, s);

    s->vram_size = NV_GEFORCE3_VRAM_SIZE;
    memory_region_init_ram(&s->vram, obj, "nv-geforce3-vram", s->vram_size,
                           &error_fatal);
    s->vram_ptr = memory_region_get_ram_ptr(&s->vram);

    /*
     * PRAMIN (instance memory) lives at the top of VRAM, addressed
     * backwards via an XOR flip against (vram_size - 64) -- confirmed
     * real NV hardware layout (RAMIN grows down from the top of VRAM
     * in reverse byte order), not an emulation shortcut.
     */
    s->ramin_flip = s->vram_size - 64;
    s->memsize_mask = s->vram_size - 1;

    s->unk_regs = g_new0(uint32_t, NV_GEFORCE3_MMIO_SIZE / 4);

    memory_region_init_io(&s->mmio, obj, &nv_geforce3_mmio_ops, s,
                          "nv-geforce3-mmio", NV_GEFORCE3_MMIO_SIZE);
    /*
     * PFIFO/PGRAPH command processing happens synchronously inside
     * these MMIO callbacks (no worker thread, matching the ported
     * model's own single-threaded design), so no BQL re-entrancy
     * concerns apply here.
     */

    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
    pci_register_bar(dev, 1, PCI_BASE_ADDRESS_SPACE_MEMORY |
                     PCI_BASE_ADDRESS_MEM_PREFETCH, &s->vram);

    pci_config_set_interrupt_pin(dev->config, 1);

    nv_geforce3_init_method_handlers(s);

    s->vblank_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,
                                   nv_geforce3_vblank_timer_tick, s);
    timer_mod(s->vblank_timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +
              NV_GEFORCE3_VBLANK_PERIOD_NS);
}

static void nv_geforce3_exit(PCIDevice *dev)
{
    NVGeForce3State *s = NV_GEFORCE3(dev);

    timer_free(s->vblank_timer);
    g_free(s->unk_regs);
}

static const VMStateDescription vmstate_nv_geforce3 = {
    .name = "nv-geforce3",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]) {
        VMSTATE_PCI_DEVICE(parent_obj, NVGeForce3State),
        VMSTATE_UINT16(crtc.index, NVGeForce3State),
        VMSTATE_UINT8_ARRAY(crtc.reg, NVGeForce3State,
                            NV_GEFORCE3_CRTC_MAX + 1),
        VMSTATE_UINT8(dac_mask, NVGeForce3State),
        VMSTATE_UINT8(dac_wr_index, NVGeForce3State),
        VMSTATE_UINT8(dac_rd_index, NVGeForce3State),
        VMSTATE_UINT8(dac_state, NVGeForce3State),
        VMSTATE_UINT8_2DARRAY(palette, NVGeForce3State, 256, 3),
        VMSTATE_UINT8(misc_output, NVGeForce3State),
        VMSTATE_UINT8(feature_ctl, NVGeForce3State),
        VMSTATE_UINT8(seq_index, NVGeForce3State),
        VMSTATE_UINT8_ARRAY(seq_reg, NVGeForce3State, 8),
        VMSTATE_UINT8(gr_index, NVGeForce3State),
        VMSTATE_UINT8_ARRAY(gr_reg, NVGeForce3State, 16),
        VMSTATE_UINT8(attr_index, NVGeForce3State),
        VMSTATE_UINT8_ARRAY(attr_reg, NVGeForce3State, 32),
        VMSTATE_BOOL(attr_flipflop, NVGeForce3State),
        VMSTATE_UINT32(rma_addr, NVGeForce3State),
        VMSTATE_BOOL(mc_soft_intr, NVGeForce3State),
        VMSTATE_UINT32(mc_intr_en, NVGeForce3State),
        VMSTATE_UINT32(mc_enable, NVGeForce3State),
        VMSTATE_UINT32(bus_intr, NVGeForce3State),
        VMSTATE_UINT32(bus_intr_en, NVGeForce3State),
        VMSTATE_BOOL(fifo_wait, NVGeForce3State),
        VMSTATE_BOOL(fifo_wait_soft, NVGeForce3State),
        VMSTATE_BOOL(fifo_wait_notify, NVGeForce3State),
        VMSTATE_BOOL(fifo_wait_flip, NVGeForce3State),
        VMSTATE_BOOL(fifo_wait_acquire, NVGeForce3State),
        VMSTATE_UINT32(fifo_intr, NVGeForce3State),
        VMSTATE_UINT32(fifo_intr_en, NVGeForce3State),
        VMSTATE_UINT32(fifo_ramht, NVGeForce3State),
        VMSTATE_UINT32(fifo_ramfc, NVGeForce3State),
        VMSTATE_UINT32(fifo_ramro, NVGeForce3State),
        VMSTATE_UINT32(fifo_mode, NVGeForce3State),
        VMSTATE_UINT32(fifo_cache1_push0, NVGeForce3State),
        VMSTATE_UINT32(fifo_cache1_push1, NVGeForce3State),
        VMSTATE_UINT32(fifo_cache1_put, NVGeForce3State),
        VMSTATE_UINT32(fifo_cache1_dma_push, NVGeForce3State),
        VMSTATE_UINT32(fifo_cache1_dma_instance, NVGeForce3State),
        VMSTATE_UINT32(fifo_cache1_dma_put, NVGeForce3State),
        VMSTATE_UINT32(fifo_cache1_dma_get, NVGeForce3State),
        VMSTATE_UINT32(fifo_cache1_ref_cnt, NVGeForce3State),
        VMSTATE_UINT32(fifo_cache1_pull0, NVGeForce3State),
        VMSTATE_UINT32(fifo_cache1_semaphore, NVGeForce3State),
        VMSTATE_UINT32(fifo_cache1_get, NVGeForce3State),
        VMSTATE_UINT32(fifo_grctx_instance, NVGeForce3State),
        VMSTATE_UINT32_ARRAY(fifo_cache1_method, NVGeForce3State,
                             NV_GEFORCE3_CACHE1_SIZE),
        VMSTATE_UINT32_ARRAY(fifo_cache1_data, NVGeForce3State,
                             NV_GEFORCE3_CACHE1_SIZE),
        VMSTATE_UINT32(timer_intr, NVGeForce3State),
        VMSTATE_UINT32(timer_intr_en, NVGeForce3State),
        VMSTATE_UINT32(timer_num, NVGeForce3State),
        VMSTATE_UINT32(timer_den, NVGeForce3State),
        VMSTATE_UINT64(timer_inittime1, NVGeForce3State),
        VMSTATE_UINT64(timer_inittime2, NVGeForce3State),
        VMSTATE_UINT32(timer_alarm, NVGeForce3State),
        VMSTATE_UINT32(straps0_primary, NVGeForce3State),
        VMSTATE_UINT32(straps0_primary_original, NVGeForce3State),
        VMSTATE_UINT32(graph_intr, NVGeForce3State),
        VMSTATE_UINT32(graph_nsource, NVGeForce3State),
        VMSTATE_UINT32(graph_intr_en, NVGeForce3State),
        VMSTATE_UINT32(graph_ctx_switch1, NVGeForce3State),
        VMSTATE_UINT32(graph_ctx_switch2, NVGeForce3State),
        VMSTATE_UINT32(graph_ctx_switch4, NVGeForce3State),
        VMSTATE_UINT32(graph_ctxctl_cur, NVGeForce3State),
        VMSTATE_UINT32(graph_status, NVGeForce3State),
        VMSTATE_UINT32(graph_trapped_addr, NVGeForce3State),
        VMSTATE_UINT32(graph_trapped_data, NVGeForce3State),
        VMSTATE_UINT32(graph_flip_read, NVGeForce3State),
        VMSTATE_UINT32(graph_flip_write, NVGeForce3State),
        VMSTATE_UINT32(graph_flip_modulo, NVGeForce3State),
        VMSTATE_UINT32(graph_notify, NVGeForce3State),
        VMSTATE_UINT32(graph_fifo, NVGeForce3State),
        VMSTATE_UINT32(graph_bpixel, NVGeForce3State),
        VMSTATE_UINT32(graph_channel_ctx_table, NVGeForce3State),
        VMSTATE_UINT32(graph_offset0, NVGeForce3State),
        VMSTATE_UINT32(graph_pitch0, NVGeForce3State),
        VMSTATE_UINT32(crtc_intr, NVGeForce3State),
        VMSTATE_UINT32(crtc_intr_en, NVGeForce3State),
        VMSTATE_UINT32(crtc_start, NVGeForce3State),
        VMSTATE_UINT32(crtc_config, NVGeForce3State),
        VMSTATE_UINT32(crtc_raster_pos, NVGeForce3State),
        VMSTATE_UINT32(crtc_cursor_offset, NVGeForce3State),
        VMSTATE_UINT32(crtc_cursor_config, NVGeForce3State),
        VMSTATE_UINT32(crtc_gpio_ext, NVGeForce3State),
        VMSTATE_UINT32(ramdac_cu_start_pos, NVGeForce3State),
        VMSTATE_UINT32(ramdac_vpll, NVGeForce3State),
        VMSTATE_UINT32(ramdac_pll_select, NVGeForce3State),
        VMSTATE_UINT32(ramdac_general_control, NVGeForce3State),
        VMSTATE_BUFFER_UNSAFE(chs, NVGeForce3State, 0,
                              sizeof(NVGeForce3Channel) *
                              NV_GEFORCE3_CHANNEL_COUNT),
        VMSTATE_END_OF_LIST()
    }
};

static const Property nv_geforce3_properties[] = {
    DEFINE_PROP_BOOL("monitor-connected", NVGeForce3State,
                     monitor_connected, true),
};

static void nv_geforce3_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    ResettableClass *rc = RESETTABLE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    k->class_id = PCI_CLASS_DISPLAY_VGA;
    k->vendor_id = PCI_VENDOR_ID_NVIDIA;
    /*
     * Must match the PCIR vendor/device of the OEM Mac ROM image
     * exactly, or Open Firmware refuses to bind the FCode to the
     * card (see nv_geforce3_int.h).
     */
    k->device_id = PCI_DEVICE_ID_NVIDIA_GEFORCE3;
    k->revision = 0xA3;
    k->realize = nv_geforce3_realize;
    k->exit = nv_geforce3_exit;
    dc->vmsd = &vmstate_nv_geforce3;
    device_class_set_props(dc, nv_geforce3_properties);
    rc->phases.hold = nv_geforce3_reset_hold;
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);
}

static const TypeInfo nv_geforce3_type_info = {
    .name = TYPE_NV_GEFORCE3,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(NVGeForce3State),
    .class_init = nv_geforce3_class_init,
    .interfaces = (const InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void nv_geforce3_register_types(void)
{
    type_register_static(&nv_geforce3_type_info);
}

type_init(nv_geforce3_register_types)
