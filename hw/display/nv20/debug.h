/*
 * QEMU GeForce3 (NV20) profiling and debug helpers
 *
 * Derived from xemu NV2A debug.h (LGPL-2.1+).
 */

#ifndef HW_NV20_DEBUG_H
#define HW_NV20_DEBUG_H

#include "qemu/osdep.h"
#include "exec/hwaddr.h"

#define NV20_DPRINTF(...) do { } while (0)

#ifndef DEBUG_NV20_FEATURES
#define DEBUG_NV20_FEATURES 0
#endif

#if DEBUG_NV20_FEATURES
#define NV20_UNCONFIRMED(format, ...) do { \
    fprintf(stderr, "nv20: Warning unconfirmed feature: " format "\n", \
            ## __VA_ARGS__); \
} while (0)
#define NV20_UNIMPLEMENTED(format, ...) do { \
    fprintf(stderr, "nv20: Warning unimplemented feature: " format "\n", \
            ## __VA_ARGS__); \
} while (0)
#else
#define NV20_UNCONFIRMED(...) do { } while (0)
#define NV20_UNIMPLEMENTED(...) do { } while (0)
#endif

#define NV20_PROF_COUNTERS_XMAC \
    _X(NV20_PROF_FINISH_VERTEX_BUFFER_DIRTY) \
    _X(NV20_PROF_FINISH_SURFACE_CREATE) \
    _X(NV20_PROF_FINISH_SURFACE_DOWN) \
    _X(NV20_PROF_FINISH_NEED_BUFFER_SPACE) \
    _X(NV20_PROF_FINISH_FRAMEBUFFER_DIRTY) \
    _X(NV20_PROF_FINISH_PRESENTING) \
    _X(NV20_PROF_FINISH_FLIP_STALL) \
    _X(NV20_PROF_FINISH_FLUSH) \
    _X(NV20_PROF_FINISH_STALLED) \
    _X(NV20_PROF_CLEAR) \
    _X(NV20_PROF_QUEUE_SUBMIT) \
    _X(NV20_PROF_QUEUE_SUBMIT_AUX) \
    _X(NV20_PROF_PIPELINE_NOTDIRTY) \
    _X(NV20_PROF_PIPELINE_GEN) \
    _X(NV20_PROF_PIPELINE_BIND) \
    _X(NV20_PROF_PIPELINE_RENDERPASSES) \
    _X(NV20_PROF_BEGIN_ENDS) \
    _X(NV20_PROF_DRAW_ARRAYS) \
    _X(NV20_PROF_INLINE_BUFFERS) \
    _X(NV20_PROF_INLINE_ARRAYS) \
    _X(NV20_PROF_INLINE_ELEMENTS) \
    _X(NV20_PROF_QUERY) \
    _X(NV20_PROF_SHADER_GEN) \
    _X(NV20_PROF_SHADER_BIND) \
    _X(NV20_PROF_SHADER_BIND_NOTDIRTY) \
    _X(NV20_PROF_SHADER_UBO_DIRTY) \
    _X(NV20_PROF_SHADER_UBO_NOTDIRTY) \
    _X(NV20_PROF_ATTR_BIND) \
    _X(NV20_PROF_TEX_UPLOAD) \
    _X(NV20_PROF_GEOM_BUFFER_UPDATE_1) \
    _X(NV20_PROF_GEOM_BUFFER_UPDATE_2) \
    _X(NV20_PROF_GEOM_BUFFER_UPDATE_3) \
    _X(NV20_PROF_GEOM_BUFFER_UPDATE_4) \
    _X(NV20_PROF_GEOM_BUFFER_UPDATE_4_NOTDIRTY) \
    _X(NV20_PROF_SURF_SWIZZLE) \
    _X(NV20_PROF_SURF_CREATE) \
    _X(NV20_PROF_SURF_DOWNLOAD) \
    _X(NV20_PROF_SURF_UPLOAD) \
    _X(NV20_PROF_SURF_TO_TEX) \
    _X(NV20_PROF_SURF_TO_TEX_FALLBACK) \
    _X(NV20_PROF_QUEUE_SUBMIT_1) \
    _X(NV20_PROF_QUEUE_SUBMIT_2) \
    _X(NV20_PROF_QUEUE_SUBMIT_3) \
    _X(NV20_PROF_QUEUE_SUBMIT_4) \
    _X(NV20_PROF_QUEUE_SUBMIT_5)

enum NV20_PROF_COUNTERS_ENUM {
#define _X(x) x,
    NV20_PROF_COUNTERS_XMAC
#undef _X
    NV20_PROF__COUNT
};

#define NV20_PROF_NUM_FRAMES 300

typedef struct NV20Stats {
    int64_t last_flip_time;
    unsigned int frame_count;
    unsigned int increment_fps;
    struct {
        int mspf;
        int counters[NV20_PROF__COUNT];
    } frame_working, frame_history[NV20_PROF_NUM_FRAMES];
    unsigned int frame_ptr;
} NV20Stats;

#ifdef __cplusplus
extern "C" {
#endif

extern NV20Stats g_nv20_stats;

const char *nv20_profile_get_counter_name(unsigned int cnt);
int nv20_profile_get_counter_value(unsigned int cnt);
void nv20_profile_increment(void);
void nv20_profile_flip_stall(void);

static inline void nv20_profile_inc_counter(enum NV20_PROF_COUNTERS_ENUM cnt)
{
    g_nv20_stats.frame_working.counters[cnt] += 1;
}

static inline void nv20_reg_log_read(int block, hwaddr addr,
                                     unsigned int size, uint64_t val)
{
}

static inline void nv20_reg_log_write(int block, hwaddr addr,
                                      unsigned int size, uint64_t val)
{
}

#ifdef __cplusplus
}
#endif

#endif /* HW_NV20_DEBUG_H */
