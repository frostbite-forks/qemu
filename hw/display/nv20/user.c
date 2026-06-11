/*
 * QEMU GeForce3 USER — PFIFO DMA submission area (minimal stub)
 */

#include "nv20_int.h"

uint64_t user_read(void *opaque, hwaddr addr, unsigned int size)
{
    NV20State *d = opaque;
    unsigned int channel_id = addr >> 16;
    uint64_t r = 0;

    if (channel_id >= NV20_NUM_CHANNELS) {
        return 0;
    }

    switch (addr & 0xffff) {
    case NV_USER_DMA_PUT:
        r = d->pfifo.regs[NV_PFIFO_CACHE1_DMA_PUT];
        break;
    case NV_USER_DMA_GET:
        r = d->pfifo.regs[NV_PFIFO_CACHE1_DMA_GET];
        break;
    case NV_USER_REF:
        r = d->pfifo.regs[NV_PFIFO_CACHE1_REF];
        break;
    default:
        break;
    }
    return r;
}

void user_write(void *opaque, hwaddr addr, uint64_t val, unsigned int size)
{
    NV20State *d = opaque;
    unsigned int channel_id = addr >> 16;

    if (channel_id >= NV20_NUM_CHANNELS) {
        return;
    }

    switch (addr & 0xffff) {
    case NV_USER_DMA_PUT:
        d->pfifo.regs[NV_PFIFO_CACHE1_DMA_PUT] = val;
        break;
    case NV_USER_DMA_GET:
        d->pfifo.regs[NV_PFIFO_CACHE1_DMA_GET] = val;
        break;
    case NV_USER_REF:
        d->pfifo.regs[NV_PFIFO_CACHE1_REF] = val;
        break;
    default:
        break;
    }
}
