/*
 * ATI R300/R350 -- the Metal rendering backend.
 *
 * The second implementation of the contract in ati_r350_gl.h, and a
 * TRANSLATION of the first rather than a redesign of it: ati_r350_gl.c
 * is the OpenGL 3.3 core backend that phase 2's milestones measured
 * against the software rasterizer, and every decision that file
 * explains -- the integer colour buffer and the truncating pack, the
 * blend computed in the fragment stage, the host-supplied 1/area, the
 * host-built k/255 table, the flat per-triangle corners the fragment
 * stage rebuilds barycentrics from, the explicit fma() calls -- is
 * reproduced here for the same reasons, and the reasons are not
 * repeated. Read that file first; this one says only where Metal
 * differs and what was done about it.
 *
 * WHERE METAL DIFFERS.
 *
 *   - Origin. Metal's framebuffer has row 0 at the TOP: a fragment's
 *     [[position]] is (x + 0.5, y + 0.5) with y growing downward, and
 *     normalized device y = +1 is the first row of the attachment. GL's
 *     is the other way up, and the GL backend arranged its mapping so
 *     that target row k is GL texture row k with no flip anywhere. The
 *     same arrangement here needs the vertex stage to NEGATE the
 *     normalized y it would have handed GL, and then everything else --
 *     seed, fetch, scissor, the snapshot copies, gl_FragCoord's
 *     replacement -- is a device coordinate used as a texture
 *     coordinate, exactly as in the GL file. Negation is exact, so the
 *     geometry the rasterizer sees is the mirror image of GL's to the
 *     bit; and because the device's own coordinates and the software
 *     rasterizer's top-left rule (r300_top_left() in ati_r350_3d.c) are
 *     y-down too, Metal's own top-left rule now runs in the same
 *     orientation as the device's, which GL's could not.
 *
 *   - `precise`. MSL has no such qualifier. What the GLSL word buys is
 *     no re-association and no contraction of a*b+c into a fused
 *     multiply-add the source did not spell; here that is the whole
 *     translation unit compiled without fast math (MTLMathModeSafe, or
 *     fastMathEnabled = NO on an SDK that predates it) and
 *     `#pragma METAL fp contract(off)` at the top of every source. The
 *     explicit fma() calls stay fused because they are calls.
 *
 *   - The guest's fragment program arrives as GLSL. The translator in
 *     ati_r350_us_glsl.c emits a small, fixed subset -- vec3/vec4
 *     locals, swizzles, ternaries, fma/abs/floor/clamp/exp2/log2/
 *     inversesqrt, USK[] constant reads -- and every construct in it
 *     but four spells the same in MSL. r350_mtl_us_msl() below
 *     rewrites those four (the vector type names, `precise`,
 *     inversesqrt and the signature's `out` parameter) token by token
 *     and refuses anything whose shape it does not recognise, which
 *     the caller counts as a program that would not build and renders
 *     on the software path, exactly as a GLSL compile failure is
 *     treated. A translator emitting MSL beside the GLSL would be the
 *     other way to do this; it would be a second copy of an executable
 *     specification for the sake of eight identifiers.
 *
 *   - State lives in pipeline objects. A colour write mask is part of
 *     a MTLRenderPipelineState, not a call, so a linked program is a
 *     fragment function plus up to sixteen pipelines made lazily by
 *     mask. The blend variant is a pipeline property too, and it is
 *     the same two-variant split as the GL file: `main` writes bytes
 *     into the RGBA8Uint colour buffer with blending off, `add` writes
 *     a float source term into the RGBA8Unorm twin with ONE/ONE
 *     additive blending, bracketed by the same two exact conversions.
 *
 *   - Transfers are blits. Every byte that crosses the bus goes
 *     through a MTLBuffer in shared storage and a blit command
 *     encoder, in both directions, and the textures themselves are
 *     private. That is one code path for Apple silicon and for an
 *     Intel host with a discrete GPU, where a texture the CPU can
 *     address would need explicit synchronisation per direction. The
 *     aperture swapper's byte permutation happens on the CPU into and
 *     out of the staging buffer, as in the GL file and for the reason
 *     measured there.
 *
 *   - No context, no thread rule. Metal objects are safe to use from
 *     any thread; the device may reach this from a vCPU thread or the
 *     main thread and the big QEMU lock keeps two out. Command buffers
 *     from one queue execute in submission order, so a seed, a draw and
 *     a fetch compose without waiting -- only fetch, verify's readback
 *     and close wait for the GPU.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/atomic.h"
#include "ati_r350_gl.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/* the fragment-program cache, the same shape as the GL file's */
#define R350_MTL_PROGSLOTS 16
/* one pipeline per RGBA write-mask combination */
#define R350_MTL_WMASKS 16

/*
 * Buffer and texture bindings, shared between the MSL sources below
 * and the encoding in r350_mtl_draw(). The vertex stage sees the
 * vertex array and the uniforms; the fragment stage sees the uniforms,
 * the fragment program's constants, the k/255 table, the bound texture
 * and the destination snapshot.
 */
#define R350_MTL_VB_VERTS  0
#define R350_MTL_VB_UNI    1
#define R350_MTL_FB_UNI    0
#define R350_MTL_FB_USK    1
#define R350_MTL_FB_N255   2
#define R350_MTL_FT_TEX    0
#define R350_MTL_FT_DST    1

/*
 * The per-draw uniforms, as one constant buffer. Every member is a
 * four-byte scalar and the MSL struct is spelled member for member
 * the same, so the two layouts agree without either side padding --
 * a vector member would have sixteen-byte alignment in the constant
 * address space and none here.
 */
typedef struct R350MtlUni {
    float rect[4];
    float org[2];
    int32_t texsize[2];
    int32_t clamp[2];
    int32_t textured, alphatest, affunc;
    float afref;
    int32_t discard, blend, blendread;
    int32_t cfac[3];
    int32_t afac[3];
    float konst[4];
} R350MtlUni;

typedef struct R350MtlProg {
    uint64_t key;               /* 0 = empty */
    bool add;                   /* which blend variant this is */
    id<MTLFunction> fs;         /* nil: the program would not build */
    id<MTLRenderPipelineState> pso[R350_MTL_WMASKS];
} R350MtlProg;

typedef struct R350MtlCtx {
    R350GlCtx base;
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;         /* the vertex stage and the two conversions */
    id<MTLFunction> vs;
    id<MTLRenderPipelineState> ui2n, n2ui;
    id<MTLBuffer> n255;         /* the 256-entry k/255 table */
    id<MTLBuffer> back;         /* readback staging, grown on demand */
    size_t back_sz;
    id<MTLTexture> cbuf, dst, acc, white;
    id<MTLTexture> tex[R350_GL_TEXSLOTS + 1];
    int tex_w[R350_GL_TEXSLOTS + 1], tex_h[R350_GL_TEXSLOTS + 1];
    int fb_w, fb_h;
    R350MtlProg prog[R350_MTL_PROGSLOTS];
    unsigned prog_next;
    uint64_t prog_hits, prog_links, prog_failed;
    /*
     * Set from a command buffer's completion handler when the GPU
     * reported an error. Draws do not wait for the GPU, so the failure
     * of one is reported by the next entry point, which refuses -- the
     * caller then releases the target and renders in software, which
     * is the right response to a lost device.
     */
    int gpu_failed;
    char desc[128];
} R350MtlCtx;

/*
 * THE SHADERS. One prologue every source starts with: the standard
 * library, contraction off, the uniform block and the vertex-to-
 * fragment frame -- flat per-triangle corners, as in the GL file.
 */
static const char *msl_common =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"#pragma METAL fp contract(off)\n"
"\n"
"struct Uni {\n"
"    float rect_x, rect_y, rect_w, rect_h;\n"
"    float org_x, org_y;\n"
"    int texsize_x, texsize_y;\n"
"    int clamp_x, clamp_y;\n"
"    int textured, alphatest, affunc;\n"
"    float afref;\n"
"    int discard, blend, blendread;\n"
"    int cfac_x, cfac_y, cfac_z;\n"
"    int afac_x, afac_y, afac_z;\n"
"    float k_r, k_g, k_b, k_a;\n"
"};\n"
"\n"
"struct VOut {\n"
"    float4 pos [[position]];\n"
"    float2 p0 [[flat]];\n"
"    float2 p1 [[flat]];\n"
"    float2 p2 [[flat]];\n"
"    float4 c0 [[flat]];\n"
"    float4 c1 [[flat]];\n"
"    float4 c2 [[flat]];\n"
"    float2 t0 [[flat]];\n"
"    float2 t1 [[flat]];\n"
"    float2 t2 [[flat]];\n"
"    float inv [[flat]];\n"
"    float4 s0 [[flat]];\n"
"    float4 s1 [[flat]];\n"
"    float4 s2 [[flat]];\n"
"};\n";

/*
 * The vertex stage reads the request's vertex array straight out of a
 * buffer by vertex index -- no vertex descriptor, so the offsets are
 * derived from R350_GL_TEXCOORDS in C (r350_mtl_vs_src()) rather than
 * written into the text, for the reason the GL file's attribute table
 * gives. `%d` below are those offsets, in the order the header lists
 * the layout; the stride is last.
 */
static const char *msl_vs_fmt =
"#define VO_POS %d\n"
"#define VO_P0 %d\n"
"#define VO_P1 %d\n"
"#define VO_P2 %d\n"
"#define VO_C0 %d\n"
"#define VO_C1 %d\n"
"#define VO_C2 %d\n"
"#define VO_T0 %d\n"
"#define VO_T1 %d\n"
"#define VO_T2 %d\n"
"#define VO_INV %d\n"
"#define VO_S0 %d\n"
"#define VO_S1 %d\n"
"#define VO_S2 %d\n"
"#define VO_STRIDE %d\n"
"\n"
"vertex VOut vs_main(uint vid [[vertex_id]],\n"
"                    const device float *v [[buffer(0)]],\n"
"                    constant Uni &u [[buffer(1)]])\n"
"{\n"
"    const device float *a = v + vid * VO_STRIDE;\n"
"    VOut o;\n"
"    float nx = (a[VO_POS] - u.rect_x) / u.rect_w * 2.0 - 1.0;\n"
"    float ny = (a[VO_POS + 1] - u.rect_y) / u.rect_h * 2.0 - 1.0;\n"
/*
 * w = 1 as in the GL file, and y negated: see the origin note in the
 * file comment. Row 0 of the target is the top of Metal's attachment.
 */
"    o.pos = float4(nx, -ny, 0.0, 1.0);\n"
"    o.p0 = float2(a[VO_P0], a[VO_P0 + 1]);\n"
"    o.p1 = float2(a[VO_P1], a[VO_P1 + 1]);\n"
"    o.p2 = float2(a[VO_P2], a[VO_P2 + 1]);\n"
"    o.c0 = float4(a[VO_C0], a[VO_C0 + 1], a[VO_C0 + 2], a[VO_C0 + 3]);\n"
"    o.c1 = float4(a[VO_C1], a[VO_C1 + 1], a[VO_C1 + 2], a[VO_C1 + 3]);\n"
"    o.c2 = float4(a[VO_C2], a[VO_C2 + 1], a[VO_C2 + 2], a[VO_C2 + 3]);\n"
"    o.t0 = float2(a[VO_T0], a[VO_T0 + 1]);\n"
"    o.t1 = float2(a[VO_T1], a[VO_T1 + 1]);\n"
"    o.t2 = float2(a[VO_T2], a[VO_T2 + 1]);\n"
"    o.inv = a[VO_INV];\n"
"    o.s0 = float4(a[VO_S0], a[VO_S0 + 1], a[VO_S0 + 2], a[VO_S0 + 3]);\n"
"    o.s1 = float4(a[VO_S1], a[VO_S1 + 1], a[VO_S1 + 2], a[VO_S1 + 3]);\n"
"    o.s2 = float4(a[VO_S2], a[VO_S2 + 1], a[VO_S2 + 2], a[VO_S2 + 3]);\n"
"    return o;\n"
"}\n"
"\n"
/*
 * The add-blend path's two conversions and the full-viewport triangle
 * both are drawn with, as in the GL file: byte k out as (k + 0.25)/255,
 * which the normalized twin stores as k; and floor(v * 255 + 0.5) back.
 */
"vertex float4 vs_blit(uint vid [[vertex_id]])\n"
"{\n"
"    float2 p = float2(float((vid << 1) & 2), float(vid & 2));\n"
"    return float4(p * 2.0 - 1.0, 0.0, 1.0);\n"
"}\n"
"\n"
"fragment float4 fs_ui2n(float4 pos [[position]],\n"
"                        texture2d<uint, access::read> src [[texture(0)]])\n"
"{\n"
"    uint4 v = src.read(uint2(pos.xy));\n"
"    return (float4(v) + 0.25) / 255.0;\n"
"}\n"
"\n"
"fragment uint4 fs_n2ui(float4 pos [[position]],\n"
"                       texture2d<float, access::read> src [[texture(0)]])\n"
"{\n"
"    float4 v = src.read(uint2(pos.xy));\n"
"    return uint4(floor(v * 255.0 + 0.5));\n"
"}\n";

/*
 * The two fragment prologues, one per blend variant, and the body that
 * follows the spliced us_main(). `r350_out` is the attachment's type:
 * the device's truncating pack written literally into the integer
 * buffer, or the biased float source term for Metal's blender.
 */
static const char *msl_fs_head_main =
"typedef uint4 r350_out;\n";

static const char *msl_fs_head_add =
"#define R350_ADD 1\n"
"typedef float4 r350_out;\n";

static const char *msl_fs_src =
"\n"
"static float bf(int code, float sc, float sa, float dc, float da,\n"
"                float kc, float ka)\n"
"{\n"
"    if (code == 1 || code == 32) return 0.0;\n"
"    if (code == 2 || code == 33) return 1.0;\n"
"    if (code == 3 || code == 34) return sc;\n"
"    if (code == 4 || code == 35) return 1.0 - sc;\n"
"    if (code == 9 || code == 36) return dc;\n"
"    if (code == 10 || code == 37) return 1.0 - dc;\n"
"    if (code == 5 || code == 38) return sa;\n"
"    if (code == 6 || code == 39) return 1.0 - sa;\n"
"    if (code == 7 || code == 40) return da;\n"
"    if (code == 8 || code == 41) return 1.0 - da;\n"
"    if (code == 11 || code == 42) return min(sa, 1.0 - da);\n"
"    if (code == 43) return kc;\n"
"    if (code == 44) return 1.0 - kc;\n"
"    if (code == 45) return ka;\n"
"    if (code == 46) return 1.0 - ka;\n"
"    return 1.0;\n"
"}\n"
"\n"
"static float comb(int f, float s, float d)\n"
"{\n"
"    if (f == 2 || f == 3) return s - d;\n"
"    if (f == 4) return min(s, d);\n"
"    if (f == 5) return max(s, d);\n"
"    if (f == 6 || f == 7) return d - s;\n"
"    return s + d;\n"
"}\n"
"\n"
"fragment r350_out fs_main(VOut in [[stage_in]],\n"
"                          constant Uni &u [[buffer(0)]],\n"
"                          constant float4 *USK [[buffer(1)]],\n"
"                          constant float *n255 [[buffer(2)]],\n"
"                          texture2d<uint, access::read> tex [[texture(0)]],\n"
"                          texture2d<uint, access::read> dstt [[texture(1)]])\n"
"{\n"
"    float4 c;\n"
"    float ts, tt;\n"
/*
 * r300_raster_tri()'s own weights, expression for expression, fusion
 * included -- the GL file's text with in.pos for gl_FragCoord, which
 * on Metal is already the device pixel plus a half in both axes.
 */
"    float inv = in.inv;\n"
"    float px = in.pos.x + u.org_x;\n"
"    float py = in.pos.y + u.org_y;\n"
"    float q0 = (in.p2.y - in.p1.y) * (px - in.p1.x);\n"
"    float q1 = (in.p0.y - in.p2.y) * (px - in.p2.x);\n"
"    float d0 = fma(in.p2.x - in.p1.x, py - in.p1.y, -q0);\n"
"    float d1 = fma(in.p0.x - in.p2.x, py - in.p2.y, -q1);\n"
"    float w0 = d0 * inv;\n"
"    float w1 = d1 * inv;\n"
"    float w2 = 1.0 - w0 - w1;\n"
"    c = fma(float4(w2), in.c2, fma(float4(w1), in.c1, w0 * in.c0));\n"
"    float2 st = fma(float2(w2), in.t2, fma(float2(w1), in.t1, w0 * in.t0));\n"
"    ts = st.x; tt = st.y;\n"
"    float4 c1 = fma(float4(w2), in.s2, fma(float4(w1), in.s1, w0 * in.s0));\n"
"    float4 texel = float4(1.0);\n"
"    if (u.textured != 0) {\n"
"        int tx = int(ts);\n"
"        int ty = int(tt);\n"
"        if (u.clamp_x <= 1 && u.texsize_x > 0) {\n"
"            tx = tx % u.texsize_x;\n"
"            if (tx < 0) tx += u.texsize_x;\n"
"        } else {\n"
"            tx = clamp(tx, 0, u.texsize_x - 1);\n"
"        }\n"
"        if (u.clamp_y <= 1 && u.texsize_y > 0) {\n"
"            ty = ty % u.texsize_y;\n"
"            if (ty < 0) ty += u.texsize_y;\n"
"        } else {\n"
"            ty = clamp(ty, 0, u.texsize_y - 1);\n"
"        }\n"
"        uint4 tu = tex.read(uint2(tx, ty));\n"
"        texel = float4(n255[tu.r], n255[tu.g], n255[tu.b], n255[tu.a]);\n"
"    }\n"
"    {\n"
"        float4 shaded;\n"
"        us_main(texel, c, c1, shaded, USK);\n"
"        c = shaded;\n"
"    }\n"
/*
 * discard_fragment() marks the fragment dead but does not leave the
 * function, so the return is spelled out to stop the work there.
 */
"    if (u.alphatest != 0) {\n"
"        bool pass;\n"
"        if (u.affunc == 0)      pass = false;\n"
"        else if (u.affunc == 1) pass = c.a <  u.afref;\n"
"        else if (u.affunc == 2) pass = c.a == u.afref;\n"
"        else if (u.affunc == 3) pass = c.a <= u.afref;\n"
"        else if (u.affunc == 4) pass = c.a >  u.afref;\n"
"        else if (u.affunc == 5) pass = c.a != u.afref;\n"
"        else if (u.affunc == 6) pass = c.a >= u.afref;\n"
"        else                    pass = true;\n"
"        if (!pass) { discard_fragment(); return r350_out(0); }\n"
"    }\n"
"    if (u.discard != 0) {\n"
"        bool a0 = c.a == 0.0, a1 = c.a == 1.0;\n"
"        bool z0 = c.r == 0.0 && c.g == 0.0 && c.b == 0.0;\n"
"        bool z1 = c.r == 1.0 && c.g == 1.0 && c.b == 1.0;\n"
"        bool kill = false;\n"
"        if (u.discard == 1) kill = a0;\n"
"        else if (u.discard == 2) kill = z0;\n"
"        else if (u.discard == 3) kill = a0 && z0;\n"
"        else if (u.discard == 4) kill = a1;\n"
"        else if (u.discard == 5) kill = z1;\n"
"        else if (u.discard == 6) kill = a1 && z1;\n"
"        if (kill) { discard_fragment(); return r350_out(0); }\n"
"    }\n"
"#ifdef R350_ADD\n"
"    float sr = c.r * bf(u.cfac_x, c.r, c.a, 0.0, 0.0, u.k_r, u.k_a);\n"
"    float sg = c.g * bf(u.cfac_x, c.g, c.a, 0.0, 0.0, u.k_g, u.k_a);\n"
"    float sb = c.b * bf(u.cfac_x, c.b, c.a, 0.0, 0.0, u.k_b, u.k_a);\n"
"    float sa = c.a * bf(u.afac_x, c.a, c.a, 0.0, 0.0, u.k_a, u.k_a);\n"
"    float4 t = clamp(float4(sr, sg, sb, sa), 0.0, 1.0);\n"
"    return (floor(t * 255.0) + 0.25) / 255.0;\n"
"#else\n"
"    if (u.blend != 0) {\n"
"        uint4 du = dstt.read(uint2(in.pos.xy));\n"
"        float4 d = u.blendread != 0\n"
"                   ? float4(n255[du.r], n255[du.g], n255[du.b], n255[du.a])\n"
"                   : float4(0.0);\n"
"        float nr = comb(u.cfac_z,\n"
"            c.r * bf(u.cfac_x, c.r, c.a, d.r, d.a, u.k_r, u.k_a),\n"
"            d.r * bf(u.cfac_y, c.r, c.a, d.r, d.a, u.k_r, u.k_a));\n"
"        float ng = comb(u.cfac_z,\n"
"            c.g * bf(u.cfac_x, c.g, c.a, d.g, d.a, u.k_g, u.k_a),\n"
"            d.g * bf(u.cfac_y, c.g, c.a, d.g, d.a, u.k_g, u.k_a));\n"
"        float nb = comb(u.cfac_z,\n"
"            c.b * bf(u.cfac_x, c.b, c.a, d.b, d.a, u.k_b, u.k_a),\n"
"            d.b * bf(u.cfac_y, c.b, c.a, d.b, d.a, u.k_b, u.k_a));\n"
"        c.a = comb(u.afac_z,\n"
"            c.a * bf(u.afac_x, c.a, c.a, d.a, d.a, u.k_a, u.k_a),\n"
"            d.a * bf(u.afac_y, c.a, c.a, d.a, d.a, u.k_a, u.k_a));\n"
"        c.r = nr; c.g = ng; c.b = nb;\n"
"    }\n"
"    return uint4(floor(clamp(c, 0.0, 1.0) * 255.0));\n"
"#endif\n"
"}\n";

/*
 * The guest's fragment program, GLSL to MSL. The input is the text of
 * r300_us_glsl(): a `void us_main(vec4 tex0, vec4 col0, vec4 col1, out
 * vec4 outc)` whose body uses a fixed vocabulary (see that file). The
 * signature is replaced whole -- MSL has no `out`, and a function
 * cannot see a buffer argument it was not handed, so the constants
 * become a parameter -- and the body is copied token by token with the
 * four spellings that differ mapped. Anything else passes through
 * unchanged, which is safe precisely because the vocabulary is closed:
 * a construct the translator does not emit cannot appear.
 *
 * Returns false when the text does not start the way the translator
 * starts it; the caller treats that as a program that would not build.
 */
static bool r350_mtl_us_msl(const char *glsl, GString *out)
{
    static const char sig[] = "void us_main(";
    static const struct {
        const char *from, *to;
    } map[] = {
        { "vec4", "float4" },
        { "vec3", "float3" },
        { "vec2", "float2" },
        { "precise", "" },
        { "inversesqrt", "rsqrt" },
    };
    const char *p;

    if (strncmp(glsl, sig, sizeof(sig) - 1)) {
        return false;
    }
    p = strchr(glsl, ')');
    if (!p) {
        return false;
    }
    g_string_append(out,
                    "static void us_main(float4 tex0, float4 col0, "
                    "float4 col1,\n"
                    "                    thread float4 &outc, "
                    "constant float4 *USK)");
    p++;
    while (*p) {
        if (g_ascii_isalpha(*p) || *p == '_') {
            const char *q = p;
            size_t n, k;

            while (g_ascii_isalnum(*q) || *q == '_') {
                q++;
            }
            n = q - p;
            for (k = 0; k < ARRAY_SIZE(map); k++) {
                if (n == strlen(map[k].from) && !memcmp(p, map[k].from, n)) {
                    g_string_append(out, map[k].to);
                    break;
                }
            }
            if (k == ARRAY_SIZE(map)) {
                g_string_append_len(out, p, n);
            }
            p = q;
        } else {
            g_string_append_c(out, *p++);
        }
    }
    return true;
}

/*
 * The compile options every source is built with: no fast math, for
 * the reason in the file comment. MTLMathModeSafe is the current
 * spelling and fastMathEnabled the one older SDKs and hosts have; both
 * say the same thing.
 */
static MTLCompileOptions *r350_mtl_options(void)
{
    MTLCompileOptions *o = [[[MTLCompileOptions alloc] init] autorelease];

#if defined(MAC_OS_VERSION_15_0) && \
    MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_VERSION_15_0
    if (@available(macOS 15.0, *)) {
        o.mathMode = MTLMathModeSafe;
        return o;
    }
#endif
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    o.fastMathEnabled = NO;
#pragma clang diagnostic pop
    return o;
}

static id<MTLLibrary> r350_mtl_compile(R350MtlCtx *g, const char *src,
                                       const char **err)
{
    NSError *e = nil;
    id<MTLLibrary> lib;

    lib = [g->dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                               options:r350_mtl_options()
                                 error:&e];
    if (!lib) {
        *err = "MSL shader would not compile";
        return nil;
    }
    return lib;
}

static char *r350_mtl_vs_src(void)
{
    const int C = R350_GL_TEXCOORDS;

    return g_strdup_printf(msl_vs_fmt,
                           0,
                           6 + 2 * C, 8 + 2 * C, 10 + 2 * C,
                           12 + 2 * C, 16 + 2 * C, 20 + 2 * C,
                           24 + 2 * C, 24 + 4 * C, 24 + 6 * C,
                           24 + 8 * C,
                           25 + 8 * C, 29 + 8 * C, 33 + 8 * C,
                           R350_GL_VSTRIDE);
}

static id<MTLTexture> r350_mtl_texture(R350MtlCtx *g, MTLPixelFormat fmt,
                                       int w, int h, bool target)
{
    MTLTextureDescriptor *d;

    d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt
                                                           width:w
                                                          height:h
                                                       mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageShaderRead |
              (target ? MTLTextureUsageRenderTarget : 0);
    return [g->dev newTextureWithDescriptor:d];
}

/*
 * Queue an upload of w x h packed RGBA8 into a texture's top-left
 * corner, through a staging buffer the command buffer keeps alive
 * until it has finished with it.
 */
static bool r350_mtl_upload(R350MtlCtx *g, id<MTLCommandBuffer> cb,
                            id<MTLTexture> t, int x0, int y0, int w, int h,
                            const void *rgba)
{
    size_t n = (size_t)w * h * 4;
    id<MTLBuffer> st = [[g->dev newBufferWithBytes:rgba
                                            length:n
                                           options:MTLResourceStorageModeShared]
                        autorelease];
    id<MTLBlitCommandEncoder> bl;

    if (!st) {
        return false;
    }
    bl = [cb blitCommandEncoder];
    [bl copyFromBuffer:st
          sourceOffset:0
     sourceBytesPerRow:(NSUInteger)w * 4
   sourceBytesPerImage:n
            sourceSize:MTLSizeMake(w, h, 1)
             toTexture:t
      destinationSlice:0
      destinationLevel:0
     destinationOrigin:MTLOriginMake(x0, y0, 0)];
    [bl endEncoding];
    return true;
}

/* the readback staging buffer, grown on demand and kept */
static id<MTLBuffer> r350_mtl_back(R350MtlCtx *g, size_t need)
{
    if (need > g->back_sz) {
        [g->back release];
        g->back = [g->dev newBufferWithLength:need
                                      options:MTLResourceStorageModeShared];
        g->back_sz = g->back ? need : 0;
    }
    return g->back;
}

/*
 * Copy a rectangle of the colour buffer into the readback buffer and
 * WAIT for it: this is the one transfer whose result the CPU reads.
 */
static bool r350_mtl_readback(R350MtlCtx *g, id<MTLCommandBuffer> cb,
                              int x0, int y0, int w, int h)
{
    id<MTLBuffer> bk = r350_mtl_back(g, (size_t)w * h * 4);
    id<MTLBlitCommandEncoder> bl;

    if (!bk) {
        return false;
    }
    bl = [cb blitCommandEncoder];
    [bl copyFromTexture:g->cbuf
            sourceSlice:0
            sourceLevel:0
           sourceOrigin:MTLOriginMake(x0, y0, 0)
             sourceSize:MTLSizeMake(w, h, 1)
               toBuffer:bk
      destinationOffset:0
 destinationBytesPerRow:(NSUInteger)w * 4
destinationBytesPerImage:(NSUInteger)w * h * 4];
    [bl endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return cb.status == MTLCommandBufferStatusCompleted;
}

/* the fragment function for one guest program and one blend variant */
static R350MtlProg *r350_mtl_prog_for(R350MtlCtx *g, const R350GlReq *r,
                                      bool add)
{
    const char *err = NULL;
    R350MtlProg *sl;
    GString *src;
    id<MTLLibrary> lib = nil;
    id<MTLFunction> fs = nil;
    unsigned k;

    for (k = 0; k < R350_MTL_PROGSLOTS; k++) {
        if (g->prog[k].key == r->us_key && g->prog[k].add == add) {
            g->prog_hits++;
            return g->prog[k].fs ? &g->prog[k] : NULL;
        }
    }

    src = g_string_new(msl_common);
    g_string_append(src, add ? msl_fs_head_add : msl_fs_head_main);
    if (r350_mtl_us_msl(r->us_glsl, src)) {
        g_string_append(src, msl_fs_src);
        lib = r350_mtl_compile(g, src->str, &err);
        if (lib) {
            fs = [lib newFunctionWithName:@"fs_main"];
            [lib release];
        }
    }
    g_string_free(src, TRUE);

    sl = &g->prog[g->prog_next];
    g->prog_next = (g->prog_next + 1) % R350_MTL_PROGSLOTS;
    for (k = 0; k < R350_MTL_WMASKS; k++) {
        [sl->pso[k] release];
    }
    [sl->fs release];
    memset(sl, 0, sizeof(*sl));
    sl->key = r->us_key;
    sl->add = add;
    sl->fs = fs;
    if (!fs) {
        g->prog_failed++;
        return NULL;
    }
    g->prog_links++;
    return sl;
}

/*
 * The pipeline for one program and one colour write mask, made on
 * first use. `wm` is the mask as four bits, red lowest.
 */
static id<MTLRenderPipelineState> r350_mtl_pso_for(R350MtlCtx *g,
                                                   R350MtlProg *p,
                                                   unsigned wm)
{
    MTLRenderPipelineDescriptor *d;
    MTLRenderPipelineColorAttachmentDescriptor *ca;
    NSError *e = nil;

    if (p->pso[wm]) {
        return p->pso[wm];
    }
    d = [[[MTLRenderPipelineDescriptor alloc] init] autorelease];
    d.vertexFunction = g->vs;
    d.fragmentFunction = p->fs;
    ca = d.colorAttachments[0];
    ca.writeMask = ((wm & 1) ? MTLColorWriteMaskRed : 0) |
                   ((wm & 2) ? MTLColorWriteMaskGreen : 0) |
                   ((wm & 4) ? MTLColorWriteMaskBlue : 0) |
                   ((wm & 8) ? MTLColorWriteMaskAlpha : 0);
    if (p->add) {
        ca.pixelFormat = MTLPixelFormatRGBA8Unorm;
        ca.blendingEnabled = YES;
        ca.rgbBlendOperation = MTLBlendOperationAdd;
        ca.alphaBlendOperation = MTLBlendOperationAdd;
        ca.sourceRGBBlendFactor = MTLBlendFactorOne;
        ca.destinationRGBBlendFactor = MTLBlendFactorOne;
        ca.sourceAlphaBlendFactor = MTLBlendFactorOne;
        ca.destinationAlphaBlendFactor = MTLBlendFactorOne;
    } else {
        ca.pixelFormat = MTLPixelFormatRGBA8Uint;
    }
    p->pso[wm] = [g->dev newRenderPipelineStateWithDescriptor:d error:&e];
    return p->pso[wm];
}

/* one of the two conversion pipelines, at open */
static id<MTLRenderPipelineState> r350_mtl_blit_pso(R350MtlCtx *g,
                                                    const char *fs,
                                                    MTLPixelFormat fmt)
{
    MTLRenderPipelineDescriptor *d;
    id<MTLFunction> vf, ff;
    id<MTLRenderPipelineState> pso;
    NSError *e = nil;

    vf = [g->lib newFunctionWithName:@"vs_blit"];
    ff = [g->lib newFunctionWithName:[NSString stringWithUTF8String:fs]];
    if (!vf || !ff) {
        [vf release];
        [ff release];
        return nil;
    }
    d = [[[MTLRenderPipelineDescriptor alloc] init] autorelease];
    d.vertexFunction = vf;
    d.fragmentFunction = ff;
    d.colorAttachments[0].pixelFormat = fmt;
    pso = [g->dev newRenderPipelineStateWithDescriptor:d error:&e];
    [vf release];
    [ff release];
    return pso;
}

static void r350_mtl_close(R350GlCtx *ctx);

static R350GlCtx *r350_mtl_open(const char **err)
{
    R350MtlCtx *g;
    bool ok = false;

    *err = NULL;
    g = g_new0(R350MtlCtx, 1);
    g->base.ops = &r350_mtl_ops;

    @autoreleasepool {
        char *vs_src;
        GString *src;
        id<MTLCommandBuffer> cb;
        float n255[256];
        static const uint8_t white[4] = { 255, 255, 255, 255 };
        unsigned k;

        g->dev = MTLCreateSystemDefaultDevice();
        if (!g->dev) {
            *err = "no Metal device on this host";
            goto out;
        }
        g->queue = [g->dev newCommandQueue];
        if (!g->queue) {
            *err = "Metal would not create a command queue";
            goto out;
        }

        vs_src = r350_mtl_vs_src();
        src = g_string_new(msl_common);
        g_string_append(src, vs_src);
        g_free(vs_src);
        g->lib = r350_mtl_compile(g, src->str, err);
        g_string_free(src, TRUE);
        if (!g->lib) {
            goto out;
        }
        g->vs = [g->lib newFunctionWithName:@"vs_main"];
        g->ui2n = r350_mtl_blit_pso(g, "fs_ui2n", MTLPixelFormatRGBA8Unorm);
        g->n2ui = r350_mtl_blit_pso(g, "fs_n2ui", MTLPixelFormatRGBA8Uint);
        if (!g->vs || !g->ui2n || !g->n2ui) {
            *err = "Metal would not build the backend's own pipelines";
            goto out;
        }

        for (k = 0; k < 256; k++) {
            n255[k] = k / 255.0f;
        }
        g->n255 = [g->dev newBufferWithBytes:n255
                                      length:sizeof(n255)
                                     options:MTLResourceStorageModeShared];
        /*
         * What an untextured draw samples, specified once: see the GL
         * file. Waited for, so that it is there before the first draw
         * of a device that never uploads a texture.
         */
        g->white = r350_mtl_texture(g, MTLPixelFormatRGBA8Uint, 1, 1, false);
        cb = [g->queue commandBuffer];
        if (!g->n255 || !g->white || !cb ||
            !r350_mtl_upload(g, cb, g->white, 0, 0, 1, 1, white)) {
            *err = "Metal would not allocate the backend's resources";
            goto out;
        }
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            *err = "Metal reported an error while setting the backend up";
            goto out;
        }

        snprintf(g->desc, sizeof(g->desc), "Metal offscreen, %s",
                 [[g->dev name] UTF8String]);
        ok = true;
out:
        ;
    }
    if (!ok) {
        r350_mtl_close(&g->base);
        return NULL;
    }
    return &g->base;
}

static void r350_mtl_close(R350GlCtx *ctx)
{
    R350MtlCtx *g = (R350MtlCtx *)ctx;
    unsigned k, m;

    @autoreleasepool {
        /*
         * Drain the queue before anything a command buffer might still
         * be reading goes away, and before a completion handler could
         * write to freed memory.
         */
        if (g->queue) {
            id<MTLCommandBuffer> cb = [g->queue commandBuffer];

            [cb commit];
            [cb waitUntilCompleted];
        }
        for (k = 0; k < R350_MTL_PROGSLOTS; k++) {
            for (m = 0; m < R350_MTL_WMASKS; m++) {
                [g->prog[k].pso[m] release];
            }
            [g->prog[k].fs release];
        }
        for (k = 0; k < R350_GL_TEXSLOTS + 1; k++) {
            [g->tex[k] release];
        }
        [g->cbuf release];
        [g->dst release];
        [g->acc release];
        [g->white release];
        [g->back release];
        [g->n255 release];
        [g->ui2n release];
        [g->n2ui release];
        [g->vs release];
        [g->lib release];
        [g->queue release];
        [g->dev release];
    }
    g_free(g);
}

static const char *r350_mtl_describe(R350GlCtx *ctx)
{
    return ((R350MtlCtx *)ctx)->desc;
}

static void r350_mtl_prog_stats(R350GlCtx *ctx, uint64_t *hits,
                                uint64_t *links, uint64_t *failed)
{
    R350MtlCtx *g = (R350MtlCtx *)ctx;

    *hits = g->prog_hits;
    *links = g->prog_links;
    *failed = g->prog_failed;
}

static bool r350_mtl_target(R350GlCtx *ctx, int w, int h, bool *lost)
{
    R350MtlCtx *g = (R350MtlCtx *)ctx;
    id<MTLTexture> cbuf, dst, acc;

    *lost = false;
    if (w <= 0 || h <= 0) {
        return false;
    }
    if (w <= g->fb_w && h <= g->fb_h) {
        return true;
    }
    /* grow only, as in the GL file, and the caller is told what it cost */
    w = MAX(w, g->fb_w);
    h = MAX(h, g->fb_h);
    if (w > 16384 || h > 16384) {
        return false;
    }
    @autoreleasepool {
        cbuf = r350_mtl_texture(g, MTLPixelFormatRGBA8Uint, w, h, true);
        dst = r350_mtl_texture(g, MTLPixelFormatRGBA8Uint, w, h, false);
        acc = r350_mtl_texture(g, MTLPixelFormatRGBA8Unorm, w, h, true);
    }
    if (!cbuf || !dst || !acc) {
        [cbuf release];
        [dst release];
        [acc release];
        g->fb_w = g->fb_h = 0;
        return false;
    }
    [g->cbuf release];
    [g->dst release];
    [g->acc release];
    g->cbuf = cbuf;
    g->dst = dst;
    g->acc = acc;
    g->fb_w = w;
    g->fb_h = h;
    *lost = true;
    return true;
}

static bool r350_mtl_seed(R350GlCtx *ctx, int x0, int y0, int w, int h,
                          const uint8_t *base, unsigned pitch, unsigned xr)
{
    R350MtlCtx *g = (R350MtlCtx *)ctx;
    bool ok = false;

    if (w <= 0 || h <= 0 ||
        x0 < 0 || y0 < 0 || x0 + w > g->fb_w || y0 + h > g->fb_h ||
        qatomic_read(&g->gpu_failed)) {
        return false;
    }
    @autoreleasepool {
        /*
         * Permuted straight into the staging buffer: it is the
         * command buffer's from here on, and nothing waits.
         */
        id<MTLBuffer> st = [[g->dev newBufferWithLength:(size_t)w * h * 4
                                                options:MTLResourceStorageModeShared]
                            autorelease];
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        id<MTLBlitCommandEncoder> bl;
        uint8_t *o;
        int x, y;

        if (!st || !cb) {
            return false;
        }
        o = [st contents];
        for (y = 0; y < h; y++) {
            const uint8_t *p = base + (size_t)(y0 + y) * pitch +
                               (size_t)x0 * 4;

            for (x = 0; x < w; x++, p += 4, o += 4) {
                o[0] = p[2 ^ xr];           /* R */
                o[1] = p[1 ^ xr];           /* G */
                o[2] = p[0 ^ xr];           /* B */
                o[3] = p[3 ^ xr];           /* A */
            }
        }
        bl = [cb blitCommandEncoder];
        [bl copyFromBuffer:st
              sourceOffset:0
         sourceBytesPerRow:(NSUInteger)w * 4
       sourceBytesPerImage:(NSUInteger)w * h * 4
                sourceSize:MTLSizeMake(w, h, 1)
                 toTexture:g->cbuf
          destinationSlice:0
          destinationLevel:0
         destinationOrigin:MTLOriginMake(x0, y0, 0)];
        [bl endEncoding];
        [cb commit];
        ok = true;
    }
    return ok;
}

static bool r350_mtl_fetch(R350GlCtx *ctx, int x0, int y0, int w, int h,
                           uint8_t *base, unsigned pitch, unsigned xr)
{
    R350MtlCtx *g = (R350MtlCtx *)ctx;
    bool ok = false;

    if (w <= 0 || h <= 0 ||
        x0 < 0 || y0 < 0 || x0 + w > g->fb_w || y0 + h > g->fb_h ||
        qatomic_read(&g->gpu_failed)) {
        return false;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g->queue commandBuffer];
        const uint8_t *i;
        int x, y;

        if (!cb || !r350_mtl_readback(g, cb, x0, y0, w, h)) {
            return false;
        }
        i = [g->back contents];
        for (y = 0; y < h; y++) {
            uint8_t *p = base + (size_t)(y0 + y) * pitch + (size_t)x0 * 4;

            for (x = 0; x < w; x++, p += 4, i += 4) {
                p[2 ^ xr] = i[0];
                p[1 ^ xr] = i[1];
                p[0 ^ xr] = i[2];
                p[3 ^ xr] = i[3];
            }
        }
        ok = true;
    }
    return ok;
}

/* the destination snapshot the shader-side blend samples, on the GPU */
static void r350_mtl_snapshot(R350MtlCtx *g, id<MTLCommandBuffer> cb,
                              const R350GlReq *r)
{
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];

    [bl copyFromTexture:g->cbuf
            sourceSlice:0
            sourceLevel:0
           sourceOrigin:MTLOriginMake(r->x0, r->y0, 0)
             sourceSize:MTLSizeMake(r->w, r->h, 1)
              toTexture:g->dst
       destinationSlice:0
       destinationLevel:0
      destinationOrigin:MTLOriginMake(r->x0, r->y0, 0)];
    [bl endEncoding];
}

/*
 * Begin a render pass on a target the pass must leave intact outside
 * what it draws (load and store), or on the twin whose contents
 * outside the pass are never read (don't care).
 */
static id<MTLRenderCommandEncoder> r350_mtl_pass(id<MTLCommandBuffer> cb,
                                                 id<MTLTexture> t, bool keep,
                                                 const R350GlReq *r,
                                                 MTLScissorRect sc)
{
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    MTLRenderPassColorAttachmentDescriptor *ca = d.colorAttachments[0];
    id<MTLRenderCommandEncoder> enc;
    MTLViewport vp = { 0.0, 0.0, (double)r->surf_w, (double)r->surf_h,
                       0.0, 1.0 };

    ca.texture = t;
    ca.loadAction = keep ? MTLLoadActionLoad : MTLLoadActionDontCare;
    ca.storeAction = MTLStoreActionStore;
    enc = [cb renderCommandEncoderWithDescriptor:d];
    [enc setViewport:vp];
    [enc setScissorRect:sc];
    return enc;
}

/* the draw pipeline's bindings, once per encoder that runs it */
static void r350_mtl_bind(id<MTLRenderCommandEncoder> enc,
                          id<MTLRenderPipelineState> pso, R350MtlCtx *g,
                          id<MTLBuffer> vb, id<MTLBuffer> ub,
                          id<MTLBuffer> kb, id<MTLTexture> tex)
{
    [enc setRenderPipelineState:pso];
    [enc setVertexBuffer:vb offset:0 atIndex:R350_MTL_VB_VERTS];
    [enc setVertexBuffer:ub offset:0 atIndex:R350_MTL_VB_UNI];
    [enc setFragmentBuffer:ub offset:0 atIndex:R350_MTL_FB_UNI];
    [enc setFragmentBuffer:kb offset:0 atIndex:R350_MTL_FB_USK];
    [enc setFragmentBuffer:g->n255 offset:0 atIndex:R350_MTL_FB_N255];
    [enc setFragmentTexture:tex atIndex:R350_MTL_FT_TEX];
    [enc setFragmentTexture:g->dst atIndex:R350_MTL_FT_DST];
}

static bool r350_mtl_draw(R350GlCtx *ctx, const R350GlReq *r)
{
    R350MtlCtx *g = (R350MtlCtx *)ctx;
    static const float zero_konst[R350_GL_USK * 4];
    bool ok = false;

    if (r->w <= 0 || r->h <= 0 || !r->nvert ||
        r->surf_w > g->fb_w || r->surf_h > g->fb_h ||
        qatomic_read(&g->gpu_failed)) {
        return false;
    }
    @autoreleasepool {
        R350MtlProg *p;
        id<MTLRenderPipelineState> pso;
        id<MTLCommandBuffer> cb;
        id<MTLBuffer> vb, ub, kb;
        id<MTLTexture> tex;
        id<MTLRenderCommandEncoder> enc;
        R350MtlUni u;
        MTLScissorRect sc;
        unsigned wm;
        int sx0, sy0, sx1, sy1;
        bool empty;

        p = r350_mtl_prog_for(g, r, r->add_blend);
        if (!p) {
            /* the program would not build: fall back */
            return false;
        }
        wm = (!!(r->wmask & 0x00ff0000)) | (!!(r->wmask & 0x0000ff00)) << 1 |
             (!!(r->wmask & 0x000000ff)) << 2 | (!!(r->wmask & 0xff000000)) << 3;
        pso = r350_mtl_pso_for(g, p, wm);
        if (!pso) {
            return false;
        }

        cb = [g->queue commandBuffer];
        if (!cb) {
            return false;
        }

        /*
         * The bound texture, as in the GL file: a fresh slot is
         * uploaded (re-specified if its size changed), a stale slot
         * is bound as it is, and a slot the caller says is current but
         * is not is refused rather than rendered from.
         */
        if ((r->textured & 1) && r->tex_slot[0] <= R350_GL_TEXSLOTS) {
            unsigned sl = r->tex_slot[0];

            if (r->tex_fresh[0] && r->tex[0]) {
                if (!g->tex[sl] || g->tex_w[sl] != r->tex_w[0] ||
                    g->tex_h[sl] != r->tex_h[0]) {
                    id<MTLTexture> t = r350_mtl_texture(g,
                                                        MTLPixelFormatRGBA8Uint,
                                                        r->tex_w[0],
                                                        r->tex_h[0], false);

                    if (!t) {
                        return false;
                    }
                    [g->tex[sl] release];
                    g->tex[sl] = t;
                    g->tex_w[sl] = r->tex_w[0];
                    g->tex_h[sl] = r->tex_h[0];
                }
                if (!r350_mtl_upload(g, cb, g->tex[sl], 0, 0, r->tex_w[0],
                                     r->tex_h[0], r->tex[0])) {
                    return false;
                }
            } else if (!g->tex[sl] || g->tex_w[sl] != r->tex_w[0] ||
                       g->tex_h[sl] != r->tex_h[0]) {
                return false;
            }
            tex = g->tex[sl];
        } else {
            tex = g->white;
        }

        vb = [[g->dev newBufferWithBytes:r->verts
                                  length:sizeof(float) * R350_GL_VSTRIDE *
                                         r->nvert
                                 options:MTLResourceStorageModeShared]
              autorelease];
        kb = [[g->dev newBufferWithBytes:r->us_konst ? r->us_konst
                                                     : zero_konst
                                  length:sizeof(zero_konst)
                                 options:MTLResourceStorageModeShared]
              autorelease];
        /*
         * The whole target, not the draw's rectangle: device
         * coordinates are target coordinates throughout, so u.rect
         * maps them to NDC without an offset and u.org is zero. See
         * the GL file.
         */
        memset(&u, 0, sizeof(u));
        u.rect[2] = (float)r->surf_w;
        u.rect[3] = (float)r->surf_h;
        u.texsize[0] = r->tex_w[0];
        u.texsize[1] = r->tex_h[0];
        u.clamp[0] = r->clamp_s[0];
        u.clamp[1] = r->clamp_t[0];
        u.textured = r->textured & 1;
        u.alphatest = r->alpha_test;
        u.affunc = r->af_func;
        u.afref = r->af_ref;
        u.discard = r->discard;
        u.blend = r->blend;
        u.blendread = r->blend_read;
        u.cfac[0] = r->src_factor;
        u.cfac[1] = r->dst_factor;
        u.cfac[2] = r->comb_fcn;
        u.afac[0] = r->a_src_factor;
        u.afac[1] = r->a_dst_factor;
        u.afac[2] = r->a_comb_fcn;
        u.konst[0] = r->k_r;
        u.konst[1] = r->k_g;
        u.konst[2] = r->k_b;
        u.konst[3] = r->k_a;
        ub = [[g->dev newBufferWithBytes:&u
                                  length:sizeof(u)
                                 options:MTLResourceStorageModeShared]
              autorelease];
        if (!vb || !kb || !ub) {
            return false;
        }

        /*
         * Scissor, the one cliprect it absorbed, AND the draw's
         * rectangle, clipped to the attachment because Metal validates
         * that it lies within one. An empty scissor draws nothing and
         * says so by skipping the passes, not by encoding a zero-sized
         * one.
         */
        sx0 = MAX(MAX(r->sx0, r->x0), 0);
        sy0 = MAX(MAX(r->sy0, r->y0), 0);
        sx1 = MIN(MIN(r->sx1, r->x0 + r->w), g->fb_w);
        sy1 = MIN(MIN(r->sy1, r->y0 + r->h), g->fb_h);
        empty = sx1 <= sx0 || sy1 <= sy0;
        sc.x = empty ? 0 : sx0;
        sc.y = empty ? 0 : sy0;
        sc.width = empty ? 0 : sx1 - sx0;
        sc.height = empty ? 0 : sy1 - sy0;

        if (r->blend && r->blend_read && !r->add_blend && !empty) {
            r350_mtl_snapshot(g, cb, r);
        }

        if (empty) {
            /* nothing to draw; the readback below is still owed */
        } else if (r->add_blend) {
            /*
             * dst' = dst + f(src) in one pass with Metal's blender
             * adding, in the normalized twin: bytes in, blend, bytes
             * back, both conversions exact and both on the GPU. The
             * conversions run under the draw's scissor, so only the
             * rectangle the draw may touch round-trips.
             */
            enc = r350_mtl_pass(cb, g->acc, false, r, sc);
            [enc setRenderPipelineState:g->ui2n];
            [enc setFragmentTexture:g->cbuf atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:3];
            r350_mtl_bind(enc, pso, g, vb, ub, kb, tex);
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:r->nvert];
            [enc endEncoding];

            enc = r350_mtl_pass(cb, g->cbuf, true, r, sc);
            [enc setRenderPipelineState:g->n2ui];
            [enc setFragmentTexture:g->acc atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:3];
            [enc endEncoding];
        } else if (r->npass > 1) {
            unsigned k;

            /*
             * One encoder per pass, with the destination snapshot
             * refreshed between them by a blit -- the same ordering
             * the GL file reproduces with glCopyTexSubImage2D.
             */
            for (k = 0; k < r->npass; k++) {
                if (k) {
                    r350_mtl_snapshot(g, cb, r);
                }
                enc = r350_mtl_pass(cb, g->cbuf, true, r, sc);
                r350_mtl_bind(enc, pso, g, vb, ub, kb, tex);
                [enc drawPrimitives:MTLPrimitiveTypeTriangle
                        vertexStart:r->pass[k]
                        vertexCount:r->pass[k + 1] - r->pass[k]];
                [enc endEncoding];
            }
        } else {
            enc = r350_mtl_pass(cb, g->cbuf, true, r, sc);
            r350_mtl_bind(enc, pso, g, vb, ub, kb, tex);
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:r->nvert];
            [enc endEncoding];
        }

        if (r->out) {
            /* gl=verify only: the one draw that waits for the GPU */
            if (!r350_mtl_readback(g, cb, r->x0, r->y0, r->w, r->h)) {
                return false;
            }
            memcpy(r->out, [g->back contents], (size_t)r->w * r->h * 4);
        } else {
            [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
                if (done.status == MTLCommandBufferStatusError) {
                    qatomic_set(&g->gpu_failed, 1);
                }
            }];
            [cb commit];
        }
        ok = true;
    }
    return ok;
}

const R350GlOps r350_mtl_ops = {
    .name = "metal",
    .open = r350_mtl_open,
    .close = r350_mtl_close,
    .target = r350_mtl_target,
    .seed = r350_mtl_seed,
    .fetch = r350_mtl_fetch,
    .draw = r350_mtl_draw,
    .describe = r350_mtl_describe,
    .prog_stats = r350_mtl_prog_stats,
};
