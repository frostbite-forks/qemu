/*
 * TCG memory access callbacks for GPU VRAM coherency (from xemu).
 */
#include "qemu/osdep.h"
#include "exec/mem-access-callback.h"
#include "exec/cpu-common.h"
#include "exec/cputlb.h"
#include "exec/target_page.h"
#include "hw/core/cpu.h"
#include "system/memory.h"

static bool access_callback_address_matches(MemAccessCallback *cb,
                                            hwaddr addr, hwaddr len)
{
    hwaddr watch_end = cb->addr + cb->len - 1;
    hwaddr access_end = addr + len - 1;

    return !(addr > watch_end || cb->addr > access_end);
}

int mem_access_callback_address_matches(CPUState *cpu, hwaddr addr, hwaddr len)
{
    int ret = 0;
    MemAccessCallback *cb;

    QTAILQ_FOREACH(cb, &cpu->mem_access_callbacks, entry) {
        if (access_callback_address_matches(cb, addr, len)) {
            ret |= BP_MEM_READ | BP_MEM_WRITE;
        }
    }

    return ret;
}

MemAccessCallback *mem_access_callback_insert(CPUState *cpu, MemoryRegion *mr,
                                              hwaddr offset, hwaddr len,
                                              MemAccessCallbackFunc func,
                                              void *opaque)
{
    MemAccessCallback *cb;

    if (len == 0) {
        return NULL;
    }

    cb = g_malloc(sizeof(*cb));
    cb->mr = mr;
    cb->addr = memory_region_get_ram_addr(mr) + offset;
    cb->len = len;
    cb->func = func;
    cb->opaque = opaque;
    QTAILQ_INSERT_TAIL(&cpu->mem_access_callbacks, cb, entry);

    tlb_flush_all_cpus_synced(cpu);

    return cb;
}

void mem_access_callback_remove_by_ref(CPUState *cpu, MemAccessCallback *cb)
{
    if (!cb) {
        return;
    }

    QTAILQ_REMOVE(&cpu->mem_access_callbacks, cb, entry);
    g_free(cb);

    tlb_flush_all_cpus_synced(cpu);
}

void mem_check_access_callback_ramaddr(CPUState *cpu, hwaddr ram_addr,
                                       vaddr len, int flags)
{
    MemAccessCallback *cb;

    QTAILQ_FOREACH(cb, &cpu->mem_access_callbacks, entry) {
        if (access_callback_address_matches(cb, ram_addr, len)) {
            ram_addr_t ram_addr_base = memory_region_get_ram_addr(cb->mr);
            ram_addr_t hit_addr = MAX(ram_addr, cb->addr);
            hwaddr mr_offset = hit_addr - ram_addr_base;
            bool is_write = (flags & BP_MEM_WRITE) != 0;

            cb->func(cb->opaque, cb->mr, mr_offset, len, is_write);
        }
    }
}

void mem_check_access_callback_vaddr(CPUState *cpu, vaddr addr, vaddr len,
                                     int flags, void *tlbentry)
{
    CPUTLBEntryFull *full = tlbentry;
    ram_addr_t ram_addr = addr + full->xlat_offset;

    mem_check_access_callback_ramaddr(cpu, ram_addr, len, flags);
}
