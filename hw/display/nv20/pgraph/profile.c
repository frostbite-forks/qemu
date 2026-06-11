/*
 * QEMU GeForce3 (NV20) profiling helpers
 */

#include "hw/display/nv20/nv20_int.h"

NV20Stats g_nv20_stats;

void nv20_profile_increment(void)
{
    int64_t now = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    const int64_t fps_update_interval = 250000;
    g_nv20_stats.last_flip_time = now;

    static int64_t frame_count = 0;
    frame_count++;

    static int64_t ts = 0;
    int64_t delta = now - ts;
    if (delta >= fps_update_interval) {
        g_nv20_stats.increment_fps = frame_count * 1000000 / delta;
        ts = now;
        frame_count = 0;
    }
}

void nv20_profile_flip_stall(void)
{
    int64_t now = qemu_clock_get_us(QEMU_CLOCK_REALTIME);
    int64_t render_time = (now - g_nv20_stats.last_flip_time) / 1000;

    g_nv20_stats.frame_working.mspf = render_time;
    g_nv20_stats.frame_history[g_nv20_stats.frame_ptr] =
        g_nv20_stats.frame_working;
    g_nv20_stats.frame_ptr =
        (g_nv20_stats.frame_ptr + 1) % NV20_PROF_NUM_FRAMES;
    g_nv20_stats.frame_count++;
    memset(&g_nv20_stats.frame_working, 0, sizeof(g_nv20_stats.frame_working));
}

const char *nv20_profile_get_counter_name(unsigned int cnt)
{
    const char *default_names[NV20_PROF__COUNT] = {
#define _X(x) stringify(x),
        NV20_PROF_COUNTERS_XMAC
#undef _X
    };

    assert(cnt < NV20_PROF__COUNT);
    return default_names[cnt] + 10; /* 'NV20_PROF_' */
}

int nv20_profile_get_counter_value(unsigned int cnt)
{
    assert(cnt < NV20_PROF__COUNT);
    unsigned int idx = (g_nv20_stats.frame_ptr + NV20_PROF_NUM_FRAMES - 1) %
                       NV20_PROF_NUM_FRAMES;
    return g_nv20_stats.frame_history[idx].counters[cnt];
}
