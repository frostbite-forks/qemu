/*
 * GLSL to MSL translation for NV20 PGRAPH Metal renderer.
 */
#import <Metal/Metal.h>
#include "qemu/osdep.h"
#include "metal_backend.h"
#include "metal_glsl.h"
#include <string.h>

static char *nv20_mtl_str_replace(const char *src, const char *from, const char *to);

void nv20_mtl_glsl_init(void)
{
}

void nv20_mtl_glsl_finalize(void)
{
}

static char *nv20_mtl_apply_common_transforms(const char *glsl, const char *stage)
{
    GString *out = g_string_new(
        "#include <metal_stdlib>\n"
        "using namespace metal;\n\n");

    const char *p = glsl;
    while (*p) {
        if (strncmp(p, "#version", 8) == 0) {
            while (*p && *p != '\n') {
                p++;
            }
            if (*p == '\n') {
                p++;
            }
            continue;
        }
        if (strncmp(p, "layout(location = ", 18) == 0) {
            int loc = 0;
            if (sscanf(p, "layout(location = %d)", &loc) == 1) {
                if (strstr(stage, "fragment")) {
                    g_string_append(out, "[[color(");
                    g_string_append_printf(out, "%d)]] ", loc);
                } else {
                    g_string_append(out, "[[user(loc");
                    g_string_append_printf(out, "%d)]] ", loc);
                }
                while (*p && *p != '\n') {
                    p++;
                }
                if (*p == '\n') {
                    p++;
                }
                continue;
            }
        }
        if (strncmp(p, "layout(set = ", 12) == 0) {
            while (*p && *p != '\n') {
                p++;
            }
            if (*p == '\n') {
                p++;
            }
            continue;
        }
        if (strncmp(p, "layout(binding = ", 17) == 0) {
            int binding = 0;
            if (sscanf(p, "layout(binding = %d)", &binding) == 1) {
                g_string_append(out, "[[buffer(");
                g_string_append_printf(out, "%d)]] ", binding);
                while (*p && *p != '\n') {
                    p++;
                }
                if (*p == '\n') {
                    p++;
                }
                continue;
            }
        }
        g_string_append_c(out, *p++);
    }

    char *result = g_string_free(out, false);

    struct {
        const char *from;
        const char *to;
    } subs[] = {
        { "texture2D", "texture2d" },
        { "texture3D", "texture3d" },
        { "textureCube", "texturecube" },
        { "sampler2D", "texture2d<float>" },
        { "sampler3D", "texture3d<float>" },
        { "samplerCube", "texturecube<float>" },
        { "gl_FragCoord", "in.position" },
        { "gl_VertexID", "vertex_id" },
        { "gl_InstanceID", "instance_id" },
        { "gl_Position", "out.position" },
        { "gl_in[", "in[" },
        { "gl_out[", "out[" },
        { "in vec4 gl_Position", "in float4 position" },
        { "vec4 gl_Position", "float4 position" },
        { "mat4", "float4x4" },
        { "mat3", "float3x3" },
        { "mat2", "float2x2" },
        { "vec4", "float4" },
        { "vec3", "float3" },
        { "vec2", "float2" },
        { "ivec4", "int4" },
        { "ivec3", "int3" },
        { "ivec2", "int2" },
        { "uvec4", "uint4" },
        { "uvec3", "uint3" },
        { "uvec2", "uint2" },
        { "uint", "uint32_t" },
    };

    for (size_t i = 0; i < ARRAY_SIZE(subs); i++) {
        char *next = nv20_mtl_str_replace(result, subs[i].from, subs[i].to);
        g_free(result);
        result = next;
    }

    return result;
}

static char *nv20_mtl_str_replace(const char *src, const char *from, const char *to)
{
    GString *out = g_string_new(NULL);
    size_t from_len = strlen(from);
    const char *p = src;

    while (*p) {
        const char *found = strstr(p, from);
        if (!found) {
            g_string_append(out, p);
            break;
        }
        g_string_append_len(out, p, found - p);
        g_string_append(out, to);
        p = found + from_len;
    }
    return g_string_free(out, false);
}

static char *nv20_mtl_wrap_stage(const char *glsl, const char *stage,
                                 const char *entry)
{
    g_autofree char *body = nv20_mtl_apply_common_transforms(glsl, stage);
    GString *out = g_string_new(NULL);

    if (!strcmp(stage, "vertex")) {
        g_string_append(out,
            "struct VOut {\n"
            "    float4 position [[position]];\n"
            "};\n\n"
            "vertex VOut ");
        g_string_append(out, entry);
        g_string_append(out,
            "(uint vertex_id [[vertex_id]],\n"
            " device const uchar *vram [[buffer(20)]],\n"
            " constant float *uniforms [[buffer(0)]])\n"
            "{\n"
            "    VOut out;\n"
            "    /* translated body follows */\n");
    } else if (!strcmp(stage, "fragment")) {
        g_string_append(out, "fragment float4 ");
        g_string_append(out, entry);
        g_string_append(out,
            "(VOut in [[stage_in]],\n"
            " constant float *uniforms [[buffer(1)]])\n"
            "{\n");
    } else {
        g_string_append(out, "/* geometry stage stub */\n");
        g_string_append(out, entry);
        g_string_append(out, "\n{\n");
    }

    g_string_append(out, body);
    g_string_append_c(out, '\n');
    g_string_append_c(out, '}');

    return g_string_free(out, false);
}

char *nv20_mtl_glsl_to_msl_vertex(const char *glsl)
{
    return nv20_mtl_wrap_stage(glsl, "vertex", "nv20_vsh_main");
}

char *nv20_mtl_glsl_to_msl_fragment(const char *glsl)
{
    return nv20_mtl_wrap_stage(glsl, "fragment", "nv20_psh_main");
}

char *nv20_mtl_glsl_to_msl_geometry(const char *glsl)
{
    return nv20_mtl_wrap_stage(glsl, "geometry", "nv20_geom_main");
}

void *nv20_mtl_compile_msl_library(void *backend, const char *msl_source,
                                   const char *entry_point)
{
    if (!backend || !msl_source) {
        return NULL;
    }

    void *library = nv20_mtl_compile_library(backend, msl_source);
    if (!library) {
        NSLog(@"nv20: MSL compile failed: %s", entry_point ? entry_point : "?");
    }
    return library;
}
