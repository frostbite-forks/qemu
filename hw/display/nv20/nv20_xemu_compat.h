/*
 * xemu API compatibility shims for NV20 PGRAPH on upstream QEMU.
 */
#ifndef HW_NV20_XEMU_COMPAT_H
#define HW_NV20_XEMU_COMPAT_H

#include "qemu/osdep.h"
#include "exec/hwaddr.h"
#include "exec/target_page.h"
#include "exec/mem-access-callback.h"
#include "system/memory.h"
#include "system/tcg.h"

static inline bool memory_region_test_and_clear_dirty(MemoryRegion *mr,
                                                      hwaddr addr, hwaddr len,
                                                      unsigned client)
{
    DirtyBitmapSnapshot *snap;
    bool dirty;

    if (!mr || len == 0) {
        return false;
    }

    snap = memory_region_snapshot_and_clear_dirty(mr, addr, len, client);
    dirty = memory_region_snapshot_get_dirty(mr, snap, addr, len);
    g_free(snap);
    return dirty;
}

static inline void memory_region_set_client_dirty(MemoryRegion *mr, hwaddr addr,
                                                  hwaddr len, unsigned client)
{
    memory_region_set_dirty(mr, addr, len);
    (void)client;
}

const char *xemu_settings_get_base_path(void);
void xemu_settings_set_string(char **dest, const char *value);
void qemu_mkdir(const char *path);
FILE *qemu_fopen(const char *path, const char *mode);

#endif /* HW_NV20_XEMU_COMPAT_H */
