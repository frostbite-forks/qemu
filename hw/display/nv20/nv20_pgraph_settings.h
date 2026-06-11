/*
 * NV20 PGRAPH runtime settings (replaces xemu-settings / xemu-notifications)
 */
#ifndef HW_NV20_PGRAPH_SETTINGS_H
#define HW_NV20_PGRAPH_SETTINGS_H

#include <stdio.h>
#include <stdbool.h>
#include "nv20_pgraph_config.h"

typedef struct Nv20Config {
    struct {
        CONFIG_DISPLAY_RENDERER renderer;
        struct {
            unsigned int surface_scale;
        } quality;
        struct {
            bool validation_layers;
            bool assert_on_validation_msg;
            bool debug_shaders;
            char *preferred_physical_device;
        } vulkan;
    } display;
    struct {
        bool cache_shaders;
    } perf;
} Nv20Config;

extern Nv20Config g_config;

static inline void nv20_queue_error_message(const char *msg)
{
    if (msg) {
        fprintf(stderr, "nv20-vga: %s\n", msg);
    }
}

static inline void nv20_queue_notification(const char *msg)
{
    if (msg) {
        fprintf(stderr, "nv20-vga: %s\n", msg);
    }
}

#define xemu_queue_error_message nv20_queue_error_message
#define xemu_queue_notification nv20_queue_notification

#endif /* HW_NV20_PGRAPH_SETTINGS_H */
