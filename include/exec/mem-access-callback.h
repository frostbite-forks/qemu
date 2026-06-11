/*
 * TCG memory access callbacks for GPU VRAM coherency (from xemu).
 */
#ifndef EXEC_MEM_ACCESS_CALLBACK_H
#define EXEC_MEM_ACCESS_CALLBACK_H

#include "exec/hwaddr.h"
#include "exec/vaddr.h"
#include "qemu/queue.h"

struct CPUState;
struct MemoryRegion;

typedef void (*MemAccessCallbackFunc)(void *opaque, MemoryRegion *mr,
                                      hwaddr addr, hwaddr len, bool write);

typedef struct MemAccessCallback {
    MemoryRegion *mr;
    hwaddr addr;
    hwaddr len;
    MemAccessCallbackFunc func;
    void *opaque;
    QTAILQ_ENTRY(MemAccessCallback) entry;
} MemAccessCallback;

MemAccessCallback *mem_access_callback_insert(CPUState *cpu, MemoryRegion *mr,
                                              hwaddr offset, hwaddr len,
                                              MemAccessCallbackFunc func,
                                              void *opaque);
void mem_access_callback_remove_by_ref(CPUState *cpu, MemAccessCallback *cb);
int mem_access_callback_address_matches(CPUState *cpu, hwaddr addr, hwaddr len);
void mem_check_access_callback_ramaddr(CPUState *cpu, hwaddr ram_addr,
                                       vaddr len, int flags);
void mem_check_access_callback_vaddr(CPUState *cpu, vaddr addr, vaddr len,
                                     int flags, void *tlbentry);

#endif /* EXEC_MEM_ACCESS_CALLBACK_H */
