#include "nv20_xemu_compat.h"

const char *xemu_settings_get_base_path(void)
{
    static char *base_path;

    if (!base_path) {
        base_path = g_strdup_printf("%s/nv20", g_get_user_cache_dir());
        g_mkdir_with_parents(base_path, 0755);
    }

    return base_path;
}

void xemu_settings_set_string(char **dest, const char *value)
{
    g_free(*dest);
    *dest = g_strdup(value);
}

void qemu_mkdir(const char *path)
{
    g_mkdir_with_parents(path, 0755);
}

FILE *qemu_fopen(const char *path, const char *mode)
{
    return fopen(path, mode);
}
