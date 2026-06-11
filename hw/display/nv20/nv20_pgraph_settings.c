#include "nv20_pgraph_settings.h"
#include "xemu-version.h"

const int xemu_version_major = 0;
const int xemu_version_minor = 0;
const int xemu_version_patch = 0;
const char *xemu_version = "qemu-nv20";
const char *xemu_commit = "qemu";
const char *xemu_date = __DATE__;

Nv20Config g_config = {
    .display = {
#if CONFIG_METAL
        .renderer = CONFIG_DISPLAY_RENDERER_METAL,
#elif CONFIG_OPENGL
        .renderer = CONFIG_DISPLAY_RENDERER_OPENGL,
#else
        .renderer = CONFIG_DISPLAY_RENDERER_NULL,
#endif
        .quality = {
            .surface_scale = 1,
        },
        .vulkan = {
            .validation_layers = false,
            .assert_on_validation_msg = false,
            .debug_shaders = false,
            .preferred_physical_device = NULL,
        },
    },
    .perf = {
        .cache_shaders = true,
    },
};
