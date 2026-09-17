/*
 * QEMU NVIDIA GeForce3 (NV20) emulation
 *
 * Models the NVIDIA GeForce3 (PCI vendor 0x10de, device 0x0200, "NV20")
 * as a PCI add-in graphics card. The GeForce3 was sold both as an AGP
 * card (PC) and, in a "GeForce3 Mac Edition" variant, as a PCI card for
 * Power Macs -- QEMU has no AGP bus, and the electrical difference
 * between the two is invisible to guest software beyond the (absent)
 * AGP capability block, so this device always looks like the PCI
 * variant, matching the real Mac Edition card.
 *
 * The card identity (0x10de:0x0200) and PCI class (0x030000, VGA
 * controller) are confirmed from the PCIR header of a real OEM Mac
 * FCode ROM dump (Open Firmware code type, single/last image, 45568
 * bytes) -- the same "match a real ROM's PCIR or Open Firmware won't
 * bind the FCode to the card" requirement documented in
 * ati_rage128_int.h.
 *
 * BAR layout matches the real GeForce3 Mac Edition card and the FCode
 * ROM's own mapping (no separate register-aperture BAR2, unlike the
 * AGP retail card and unlike this device's own later NV3x/NV4x
 * siblings -- Mac Edition boards only ever populate BAR0 and BAR1):
 *   BAR0: 16MB memory, MMIO register aperture (PMC/PBUS/PFIFO/PTIMER/
 *         PFB/PGRAPH/PCRTC/PRAMDAC and the PRAMIN/USER apertures)
 *   BAR1: memory, linear VRAM framebuffer aperture
 *   Expansion ROM: loaded via the generic PCI "romfile" property --
 *         no default romfile name, since there is no single "the"
 *         GeForce3 ROM the way there might be for an onboard chip;
 *         pass romfile=<path> on the -device option to point at a
 *         real card ROM dump.
 *
 * The register-level model (FIFO/PGRAPH command processing, the 2D
 * "software object" engine and the fixed-function/vertex-shader 3D
 * pipeline) is a port of the NV20 support in Bochs' geforce.cc/.h
 * (Copyright (C) 2025-2026 The Bochs Project, LGPL v2+), adapted to
 * QEMU's device model conventions and specialised to this device's
 * single supported chip generation (NV20 / "GeForce3 Ti 500"): every
 * branch in the original code that only applies to a different Bochs
 * "card_type" (NV1x GeForce2, NV3x GeForce FX, NV4x GeForce 6) has
 * been resolved at port time rather than carried as dead conditionals.
 *
 * This work is licensed under the GNU GPL license version 2 or later.
 */

#ifndef NV_GEFORCE3_INT_H
#define NV_GEFORCE3_INT_H

#include "hw/pci/pci_device.h"
#include "hw/display/edid.h"
#include "hw/i2c/bitbang_i2c.h"
#include "qemu/timer.h"
#include "qom/object.h"

#define PCI_VENDOR_ID_NVIDIA            0x10de
#define PCI_DEVICE_ID_NVIDIA_GEFORCE3   0x0200

#define TYPE_NV_GEFORCE3 "nv-geforce3"
OBJECT_DECLARE_SIMPLE_TYPE(NVGeForce3State, NV_GEFORCE3)

/* BAR0: 16MB MMIO register aperture (NV_PMC/PBUS/PFIFO/.../PRAMIN/USER) */
#define NV_GEFORCE3_MMIO_SIZE     (16 * 1024 * 1024)
/*
 * 64MB VRAM: what real retail GeForce3 Ti 500 / Mac Edition boards
 * shipped with, and what Bochs' own GEFORCE_3 model uses.
 */
#define NV_GEFORCE3_VRAM_SIZE     (64 * 1024 * 1024)

#define NV_GEFORCE3_CRTC_MAX      0xF0
/* 0x3b4/0x3d4, 0x3b5/0x3d5: legacy VGA-compatible CRTC index/data */
#define VGA_CRTC_MAX              0x18

#define NV_GEFORCE3_CHANNEL_COUNT    32
#define NV_GEFORCE3_SUBCHANNEL_COUNT 8
#define NV_GEFORCE3_CACHE1_SIZE      64

/*
 * NV20's PFIFO uses a 12-bit object class (class_mask = 0xFFF); classes
 * above that are an NV40-and-later feature this chip does not have.
 */
#define NV_GEFORCE3_CLASS_COUNT   0x1000
#define NV_GEFORCE3_METHOD_COUNT  0x800

typedef struct NVGeForce3State NVGeForce3State;
typedef struct NVGeForce3Channel NVGeForce3Channel;

typedef void (*NVGeForce3MethodHandler)(NVGeForce3State *s,
                                        NVGeForce3Channel *ch,
                                        uint32_t cls, uint32_t method,
                                        uint32_t param);

typedef struct NVGeForce3Texture {
    uint32_t offset;
    uint32_t dma_obj;
    uint32_t dimensions;
    uint32_t format;
    bool cubemap;
    bool linear;
    bool unnormalized;
    bool compressed;
    bool dxt_alpha_data;
    bool dxt_alpha_explicit;
    uint32_t color_bytes;
    uint32_t pitch;
    uint32_t levels;
    uint32_t filter_min;
    uint32_t filter_mag;
    uint32_t size_log[3];
    uint32_t size_npot[3];
    uint32_t sizes[16][3];
    uint32_t level_offset[16];
    uint32_t face_bytes;
    uint32_t wrap[3];
    uint32_t control0;
    bool enabled;
    uint32_t s0[4];
    uint32_t s1[4];
    bool signed_any;
    bool signed_comp[4];
    uint32_t pal_dma_obj;
    uint32_t pal_ofs;
    float border_color[4];
    uint32_t key_color;
    float offset_matrix[4];
} NVGeForce3Texture;

typedef struct NVGeForce3Light {
    float ambient_color[3];
    float diffuse_color[3];
    float specular_color[3];
    float inf_half_vector[3];
    float inf_direction[3];
    float spot_direction[4];
    float local_position[3];
    float local_attenuation[3];
} NVGeForce3Light;

/*
 * Per-PFIFO-channel state: everything a channel's currently bound
 * software objects (2D "GDI"/blit engine, the M2MF copy engine and the
 * Kelvin/Celsius 3D pipeline) need across method calls. Ported field
 * for field from Bochs' gf_channel (geforce.h).
 */
struct NVGeForce3Channel {
    uint32_t subr_return;
    bool subr_active;
    struct {
        uint32_t mthd;
        uint32_t subc;
        uint32_t mcnt;
        bool ni;
    } dma_state;
    struct {
        uint32_t object;
        uint8_t engine;
        uint32_t notifier;
    } schs[NV_GEFORCE3_SUBCHANNEL_COUNT];

    bool notify_pending;
    uint32_t notify_type;

    bool s2d_locked;
    uint32_t s2d_img_src;
    uint32_t s2d_img_dst;
    uint32_t s2d_color_fmt;
    uint32_t s2d_color_bytes;
    uint32_t s2d_pitch_src;
    uint32_t s2d_pitch_dst;
    uint32_t s2d_ofs_src;
    uint32_t s2d_ofs_dst;

    uint32_t swzs_img_obj;
    uint32_t swzs_fmt;
    uint32_t swzs_color_bytes;
    uint32_t swzs_width;
    uint32_t swzs_height;
    uint32_t swzs_ofs;

    bool ifc_color_key_enable;
    bool ifc_clip_enable;
    uint32_t ifc_operation;
    uint32_t ifc_color_fmt;
    uint32_t ifc_color_bytes;
    uint32_t ifc_pixels_per_word;
    uint32_t ifc_x;
    uint32_t ifc_y;
    uint32_t ifc_ofs_x;
    uint32_t ifc_ofs_y;
    uint32_t ifc_draw_offset;
    uint32_t ifc_redraw_offset;
    uint32_t ifc_dst_width;
    uint32_t ifc_dst_height;
    uint32_t ifc_src_width;
    uint32_t ifc_src_height;
    uint32_t ifc_clip_x0;
    uint32_t ifc_clip_y0;
    uint32_t ifc_clip_x1;
    uint32_t ifc_clip_y1;

    uint32_t iifc_palette;
    uint32_t iifc_palette_ofs;
    uint32_t iifc_operation;
    uint32_t iifc_color_fmt;
    uint32_t iifc_color_bytes;
    uint32_t iifc_bpp4;
    uint32_t iifc_yx;
    uint32_t iifc_dhw;
    uint32_t iifc_shw;
    uint32_t iifc_words_ptr;
    uint32_t iifc_words_left;
    uint32_t *iifc_words;

    uint32_t sifc_operation;
    uint32_t sifc_color_fmt;
    uint32_t sifc_color_bytes;
    uint32_t sifc_shw;
    uint32_t sifc_dxds;
    uint32_t sifc_dydt;
    uint32_t sifc_clip_yx;
    uint32_t sifc_clip_hw;
    uint32_t sifc_syx;
    uint32_t sifc_words_ptr;
    uint32_t sifc_words_left;
    uint32_t *sifc_words;

    bool blit_color_key_enable;
    uint32_t blit_operation;
    uint32_t blit_syx;
    uint32_t blit_dyx;
    uint32_t blit_hw;

    bool tfc_swizzled;
    uint32_t tfc_color_fmt;
    uint32_t tfc_color_bytes;
    uint32_t tfc_yx;
    uint32_t tfc_hw;
    uint32_t tfc_clip_wx;
    uint32_t tfc_clip_hy;
    uint32_t tfc_words_ptr;
    uint32_t tfc_words_left;
    uint32_t *tfc_words;
    bool tfc_upload;
    uint32_t tfc_upload_offset;

    uint32_t sifm_src;
    bool sifm_swizzled;
    bool sifm_swizzled_0389;
    uint32_t sifm_operation;
    uint32_t sifm_color_fmt;
    uint32_t sifm_color_bytes;
    uint32_t sifm_syx;
    uint32_t sifm_dyx;
    uint32_t sifm_shw;
    uint32_t sifm_dhw;
    int32_t sifm_dudx;
    int32_t sifm_dvdy;
    uint32_t sifm_sfmt;
    uint32_t sifm_sofs;

    uint32_t m2mf_src;
    uint32_t m2mf_dst;
    uint32_t m2mf_src_offset;
    uint32_t m2mf_dst_offset;
    uint32_t m2mf_src_pitch;
    uint32_t m2mf_dst_pitch;
    uint32_t m2mf_line_length;
    uint32_t m2mf_line_count;
    uint32_t m2mf_format;
    uint32_t m2mf_buffer_notify;

    uint32_t d3d_a_obj;
    uint32_t d3d_b_obj;
    uint32_t d3d_color_obj;
    uint32_t d3d_zeta_obj;
    uint32_t d3d_vertex_a_obj;
    uint32_t d3d_vertex_b_obj;
    uint32_t d3d_report_obj;
    uint32_t d3d_clip_horizontal;
    uint32_t d3d_clip_vertical;
    uint32_t d3d_surface_format;
    uint32_t d3d_color_bytes;
    uint32_t d3d_depth_bytes;
    bool d3d_swizzled;
    uint32_t d3d_surface_pitch_a;
    uint32_t d3d_surface_pitch_z;
    bool d3d_local_viewer;
    uint32_t d3d_color_material_emission;
    uint32_t d3d_color_material_ambient;
    uint32_t d3d_color_material_diffuse;
    uint32_t d3d_color_material_specular;
    uint32_t d3d_fog_mode;
    uint32_t d3d_fog_gen_mode;
    float d3d_fog_params[3];
    uint32_t d3d_fog_enable;
    float d3d_fog_color[4];
    int16_t d3d_window_offset_x;
    int16_t d3d_window_offset_y;
    uint32_t d3d_window_clip_x1[8];
    uint32_t d3d_window_clip_x2[8];
    uint32_t d3d_window_clip_y1[8];
    uint32_t d3d_window_clip_y2[8];
    uint32_t d3d_surface_color_offset;
    uint32_t d3d_surface_zeta_offset;
    uint32_t d3d_combiner_alpha_icw[8];
    uint32_t d3d_combiner_final[2];
    uint32_t d3d_alpha_test_enable;
    uint32_t d3d_alpha_func;
    uint32_t d3d_alpha_ref;
    uint32_t d3d_blend_enable;
    uint16_t d3d_blend_sfactor_rgb;
    uint16_t d3d_blend_sfactor_alpha;
    uint16_t d3d_blend_dfactor_rgb;
    uint16_t d3d_blend_dfactor_alpha;
    uint16_t d3d_blend_equation_rgb;
    uint16_t d3d_blend_equation_alpha;
    float d3d_blend_color[4];
    uint32_t d3d_cull_face_enable;
    uint32_t d3d_depth_test_enable;
    uint32_t d3d_depth_write_enable;
    uint32_t d3d_stencil_mask;
    uint32_t d3d_stencil_func;
    uint32_t d3d_stencil_func_ref;
    uint32_t d3d_stencil_func_mask;
    uint32_t d3d_stencil_op_sfail;
    uint32_t d3d_stencil_op_dpfail;
    uint32_t d3d_stencil_op_dppass;
    uint32_t d3d_lighting_enable;
    uint32_t d3d_stencil_test_enable;
    uint32_t d3d_depth_func;
    uint32_t d3d_color_mask;
    uint32_t d3d_color_mask_565;
    uint32_t d3d_color_mask_8888;
    uint32_t d3d_shade_mode;
    float d3d_clip_min;
    float d3d_clip_max;
    uint32_t d3d_cull_face;
    uint32_t d3d_front_face;
    uint32_t d3d_normalize_enable;
    float d3d_material_factor[4];
    uint32_t d3d_separate_specular;
    uint32_t d3d_light_enable_mask;
    uint32_t d3d_texgen[8][4];
    uint32_t d3d_texture_matrix_enable[16];
    uint32_t d3d_view_matrix_enable;
    float d3d_model_view_matrix[2][16];
    float d3d_inverse_model_view_matrix[12];
    float d3d_composite_matrix[16];
    float d3d_texture_matrix[8][16];
    float d3d_texgen_plane[8][4][4];
    uint32_t d3d_scissor_x;
    uint32_t d3d_scissor_width;
    uint32_t d3d_scissor_y;
    uint32_t d3d_scissor_height;
    uint32_t d3d_shader_program;
    uint32_t d3d_shader_obj;
    uint32_t d3d_shader_offset;
    float d3d_specular_params[6];
    float d3d_specular_power;
    float d3d_scene_ambient_color[4];
    uint32_t d3d_viewport_x;
    uint32_t d3d_viewport_width;
    uint32_t d3d_viewport_y;
    uint32_t d3d_viewport_height;
    float d3d_viewport_offset[4];
    float d3d_eye_position[4];
    float d3d_combiner_const_color[8][2][4];
    uint32_t d3d_combiner_alpha_ocw[8];
    uint32_t d3d_combiner_color_icw[8];
    float d3d_viewport_scale[4];
    uint32_t d3d_transform_program[544][4];
    float d3d_transform_constant[512][4];
    NVGeForce3Light d3d_light[8];
    uint32_t d3d_vs_temp_regs_count;
    uint32_t d3d_attrib_count;
    uint32_t d3d_vertex_data_base_index;
    uint32_t d3d_vertex_data_array_offset[16];
    uint32_t d3d_vertex_data_array_format_type[16];
    uint32_t d3d_vertex_data_array_format_size[16];
    uint32_t d3d_vertex_data_array_format_stride[16];
    bool d3d_vertex_data_array_format_dx[16];
    bool d3d_vertex_data_array_format_homogeneous[16];
    uint32_t d3d_begin_end;
    bool d3d_primitive_done;
    bool d3d_triangle_flip;
    uint32_t d3d_vertex_index;
    uint32_t d3d_attrib_index;
    uint32_t d3d_comp_index;
    float d3d_vertex_data[4][16][4];
    float d3d_vertex_data_imm[16][4];
    uint32_t d3d_index_array_offset;
    bool d3d_index_array_dma;
    bool d3d_index_array_type_16;
    NVGeForce3Texture d3d_texture[16];
    uint32_t d3d_shader_control;
    uint32_t d3d_semaphore_obj;
    uint32_t d3d_semaphore_offset;
    uint32_t d3d_zstencil_clear_value;
    uint32_t d3d_color_clear_value;
    uint32_t d3d_clear_surface;
    uint32_t d3d_combiner_color_ocw[8];
    uint32_t d3d_combiner_control;
    uint32_t d3d_combiner_control_num_stages;
    uint32_t d3d_tex_shader_op[4];
    uint32_t d3d_tex_shader_dotmapping[4];
    uint32_t d3d_tex_shader_previous[4];
    uint32_t d3d_transform_execution_mode;
    uint32_t d3d_transform_program_load;
    uint32_t d3d_transform_program_start;
    uint32_t d3d_transform_constant_load;
    uint32_t d3d_attrib_in_normal;
    uint32_t d3d_attrib_in_color[2];
    uint32_t d3d_attrib_out_color[2];
    uint32_t d3d_attrib_out_fogc;
    uint32_t d3d_attrib_in_tex_coord[16];
    uint32_t d3d_attrib_out_tex_coord[16];
    bool d3d_attrib_out_enable[32];
    uint32_t d3d_tex_coord_count;

    uint8_t rop;
    uint32_t beta;

    uint16_t clip_x;
    uint16_t clip_y;
    uint16_t clip_width;
    uint16_t clip_height;

    uint32_t chroma_color_fmt;
    uint32_t chroma_color;

    uint32_t patt_shape;
    bool patt_type_color;
    uint32_t patt_bg_color;
    uint32_t patt_fg_color;
    bool patt_data_mono[64];
    uint32_t patt_data_color[64];

    uint32_t gdi_operation;
    uint32_t gdi_color_fmt;
    uint32_t gdi_mono_fmt;
    uint32_t gdi_clip_yx0;
    uint32_t gdi_clip_yx1;
    uint32_t gdi_rect_color;
    uint32_t gdi_rect_xy;
    uint32_t gdi_rect_yx0;
    uint32_t gdi_rect_yx1;
    uint32_t gdi_rect_wh;
    uint32_t gdi_bg_color;
    uint32_t gdi_fg_color;
    uint32_t gdi_image_swh;
    uint32_t gdi_image_dwh;
    uint32_t gdi_image_xy;
    uint32_t gdi_words_ptr;
    uint32_t gdi_words_left;
    uint32_t *gdi_words;

    uint32_t lin_operation;
    uint32_t lin_color_fmt;
    uint32_t lin_color;
    int32_t lin_x0;
    int32_t lin_y0;
    int32_t lin_x1;
    int32_t lin_y1;

    uint32_t rect_operation;
    uint32_t rect_color_fmt;
    uint32_t rect_color;
    uint32_t rect_yx;
    uint32_t rect_hw;
};

typedef struct NVGeForce3Mode {
    uint32_t width;
    uint32_t height;
    uint32_t pitch;     /* bytes per scanline */
    uint32_t bpp;       /* bytes per pixel; 0 = mode not usable */
} NVGeForce3Mode;

struct NVGeForce3State {
    PCIDevice parent_obj;

    MemoryRegion mmio;          /* BAR0: 16MB register aperture */
    MemoryRegion vram;          /* BAR1: linear VRAM */
    uint8_t *vram_ptr;
    uint32_t vram_size;
    uint32_t memsize_mask;
    uint32_t ramin_flip;

    /*
     * Fallback backing for anything this model does not decode --
     * mirrors Bochs' unk_regs[]. Sized to the whole 16MB MMIO
     * aperture; allocated in realize(), freed in exit().
     */
    uint32_t *unk_regs;

    QemuConsole *con;

    /* 0x3b4-5/0x3d4-5: legacy-VGA-compatible index plus NV20's own
     * extended CRTC register block sharing the same index space */
    struct {
        uint16_t index;
        uint8_t reg[NV_GEFORCE3_CRTC_MAX + 1];
    } crtc;

    /* DAC palette (ports 0x3c6-0x3c9) */
    uint8_t dac_mask;
    uint8_t dac_wr_index;
    uint8_t dac_rd_index;
    uint8_t dac_state;      /* 0 = expect red, 1 = green, 2 = blue */
    uint8_t palette[256][3];

    /*
     * Legacy VGA-compatible register file, reached (per real NV20
     * hardware, confirmed against this model's own MMIO byte-access
     * dispatch) as byte-addressed windows inside the MMIO BAR rather
     * than through any I/O BAR -- see nv_geforce3_legacy_read/write().
     * Nothing in this device's own rendering path consults these
     * (the native CRTC/RAMDAC registers above do the real work);
     * they exist so guest software that probes standard VGA state
     * during bring-up gets consistent shadow values back instead of
     * an unimplemented-register log storm.
     */
    uint8_t misc_output;
    uint8_t feature_ctl;
    uint8_t seq_index;
    uint8_t seq_reg[8];
    uint8_t gr_index;
    uint8_t gr_reg[16];
    uint8_t attr_index;
    uint8_t attr_reg[32];
    bool attr_flipflop;

    uint32_t rma_addr;

    /* NV_PMC */
    bool mc_soft_intr;
    uint32_t mc_intr_en;
    uint32_t mc_enable;

    /* NV_PBUS */
    uint32_t bus_intr;
    uint32_t bus_intr_en;

    /* NV_PFIFO */
    bool fifo_wait;
    bool fifo_wait_soft;
    bool fifo_wait_notify;
    bool fifo_wait_flip;
    bool fifo_wait_acquire;
    uint32_t fifo_intr;
    uint32_t fifo_intr_en;
    uint32_t fifo_ramht;
    uint32_t fifo_ramfc;
    uint32_t fifo_ramro;
    uint32_t fifo_mode;
    uint32_t fifo_cache1_push0;
    uint32_t fifo_cache1_push1;
    uint32_t fifo_cache1_put;
    uint32_t fifo_cache1_dma_push;
    uint32_t fifo_cache1_dma_instance;
    uint32_t fifo_cache1_dma_put;
    uint32_t fifo_cache1_dma_get;
    uint32_t fifo_cache1_ref_cnt;
    uint32_t fifo_cache1_pull0;
    uint32_t fifo_cache1_semaphore;
    uint32_t fifo_cache1_get;
    uint32_t fifo_grctx_instance;
    uint32_t fifo_cache1_method[NV_GEFORCE3_CACHE1_SIZE];
    uint32_t fifo_cache1_data[NV_GEFORCE3_CACHE1_SIZE];

    /* NV_PTIMER */
    uint32_t timer_intr;
    uint32_t timer_intr_en;
    uint32_t timer_num;
    uint32_t timer_den;
    uint64_t timer_inittime1;
    uint64_t timer_inittime2;
    uint32_t timer_alarm;

    /* NV_PSTRAPS */
    uint32_t straps0_primary;
    uint32_t straps0_primary_original;

    /* NV_PGRAPH */
    uint32_t graph_intr;
    uint32_t graph_nsource;
    uint32_t graph_intr_en;
    uint32_t graph_ctx_switch1;
    uint32_t graph_ctx_switch2;
    uint32_t graph_ctx_switch4;
    uint32_t graph_ctxctl_cur;
    uint32_t graph_status;
    uint32_t graph_trapped_addr;
    uint32_t graph_trapped_data;
    uint32_t graph_flip_read;
    uint32_t graph_flip_write;
    uint32_t graph_flip_modulo;
    uint32_t graph_notify;
    uint32_t graph_fifo;
    uint32_t graph_bpixel;
    uint32_t graph_channel_ctx_table;
    uint32_t graph_offset0;
    uint32_t graph_pitch0;

    /* NV_PCRTC */
    uint32_t crtc_intr;
    uint32_t crtc_intr_en;
    uint32_t crtc_start;
    uint32_t crtc_config;
    uint32_t crtc_raster_pos;
    uint32_t crtc_cursor_offset;
    uint32_t crtc_cursor_config;
    uint32_t crtc_gpio_ext;

    /* NV_PRAMDAC */
    uint32_t ramdac_cu_start_pos;
    uint32_t ramdac_vpll;
    uint32_t ramdac_pll_select;
    uint32_t ramdac_general_control;

    NVGeForce3MethodHandler empty_method_handlers[NV_GEFORCE3_METHOD_COUNT];
    NVGeForce3MethodHandler cl0096_method_handlers[NV_GEFORCE3_METHOD_COUNT];
    NVGeForce3MethodHandler cl0097_method_handlers[NV_GEFORCE3_METHOD_COUNT];
    NVGeForce3MethodHandler cl0497_method_handlers[NV_GEFORCE3_METHOD_COUNT];
    NVGeForce3MethodHandler *class_method_handlers[NV_GEFORCE3_CLASS_COUNT];

    NVGeForce3Channel chs[NV_GEFORCE3_CHANNEL_COUNT];

    /* display state, derived from crtc.reg[] by nv_geforce3_update_mode() */
    bool mode_dirty;
    bool extended_mode;   /* crtc.reg[0x28] != 0: native linear FB active */
    NVGeForce3Mode mode;
    uint32_t fb_offset;   /* byte offset into VRAM of the visible surface */

    /* hardware cursor (NV_PCRTC/NV_PRAMDAC cursor registers) */
    struct {
        bool enabled;
        bool vram;      /* cursor image lives in VRAM vs. PRAMIN */
        bool bpp32;
        uint32_t offset;
        int16_t x, y;
        uint8_t size;   /* 32 or 64 */
    } hw_cursor;

    bool monitor_connected;
    QEMUTimer *vblank_timer;

    /* DDC (CRTC extended index 0x3f bit-banged EEPROM bus) */
    uint8_t edid[128];
    qemu_edid_info edid_info;
    bitbang_i2c_interface ddc_i2c;
};

#define NV_GEFORCE3_VBLANK_PERIOD_NS (NANOSECONDS_PER_SECOND / 60)

/* nv_geforce3.c */
void nv_geforce3_update_irq(NVGeForce3State *s);
uint64_t nv_geforce3_get_current_time(NVGeForce3State *s);
uint8_t nv_geforce3_vram_read8(NVGeForce3State *s, uint32_t addr);
uint16_t nv_geforce3_vram_read16(NVGeForce3State *s, uint32_t addr);
uint32_t nv_geforce3_vram_read32(NVGeForce3State *s, uint32_t addr);
uint64_t nv_geforce3_vram_read64(NVGeForce3State *s, uint32_t addr);
void nv_geforce3_vram_write8(NVGeForce3State *s, uint32_t addr, uint8_t val);
void nv_geforce3_vram_write16(NVGeForce3State *s, uint32_t addr, uint16_t val);
void nv_geforce3_vram_write32(NVGeForce3State *s, uint32_t addr, uint32_t val);
void nv_geforce3_vram_write64(NVGeForce3State *s, uint32_t addr, uint64_t val);
uint8_t nv_geforce3_ramin_read8(NVGeForce3State *s, uint32_t addr);
uint16_t nv_geforce3_ramin_read16(NVGeForce3State *s, uint32_t addr);
uint32_t nv_geforce3_ramin_read32(NVGeForce3State *s, uint32_t addr);
void nv_geforce3_ramin_write8(NVGeForce3State *s, uint32_t addr, uint8_t val);
void nv_geforce3_ramin_write32(NVGeForce3State *s, uint32_t addr, uint32_t val);
uint8_t nv_geforce3_dma_read8(NVGeForce3State *s, uint32_t obj, uint32_t addr);
uint16_t nv_geforce3_dma_read16(NVGeForce3State *s, uint32_t obj, uint32_t addr);
uint32_t nv_geforce3_dma_read32(NVGeForce3State *s, uint32_t obj, uint32_t addr);
uint64_t nv_geforce3_dma_read64(NVGeForce3State *s, uint32_t obj, uint32_t addr);
void nv_geforce3_dma_write8(NVGeForce3State *s, uint32_t obj, uint32_t addr,
                            uint8_t val);
void nv_geforce3_dma_write16(NVGeForce3State *s, uint32_t obj, uint32_t addr,
                             uint16_t val);
void nv_geforce3_dma_write32(NVGeForce3State *s, uint32_t obj, uint32_t addr,
                             uint32_t val);
void nv_geforce3_dma_write64(NVGeForce3State *s, uint32_t obj, uint32_t addr,
                             uint64_t val);
uint32_t nv_geforce3_dma_lin_lookup(NVGeForce3State *s, uint32_t obj,
                                    uint32_t addr);
void nv_geforce3_dma_copy(NVGeForce3State *s, uint32_t dst_obj,
                          uint32_t dst_addr, uint32_t src_obj,
                          uint32_t src_addr, uint32_t byte_count);
void nv_geforce3_redraw_area(NVGeForce3State *s, uint32_t offset,
                             uint32_t width, uint32_t height);
void nv_geforce3_fifo_process(NVGeForce3State *s);
void nv_geforce3_fifo_process_chid(NVGeForce3State *s, uint32_t chid);
void nv_geforce3_update_fifo_wait(NVGeForce3State *s);
int nv_geforce3_execute_command(NVGeForce3State *s, uint32_t chid,
                                uint32_t subc, uint32_t method,
                                uint32_t param);

/* nv_geforce3_2d.c */
uint32_t nv_geforce3_swizzle(uint32_t x, uint32_t y, uint32_t z,
                             uint32_t width, uint32_t height, uint32_t depth);
uint32_t nv_geforce3_get_pixel(NVGeForce3State *s, uint32_t obj, uint32_t ofs,
                               uint32_t x, uint32_t cb);
void nv_geforce3_put_pixel(NVGeForce3State *s, NVGeForce3Channel *ch,
                           uint32_t ofs, uint32_t x, uint32_t value);
void nv_geforce3_pixel_operation(NVGeForce3State *s, NVGeForce3Channel *ch,
                                 uint32_t op, uint32_t *dstcolor,
                                 const uint32_t *srccolor, uint32_t cb,
                                 uint32_t px, uint32_t py);
void nv_geforce3_execute_clip(NVGeForce3Channel *ch, uint32_t method,
                              uint32_t param);
void nv_geforce3_execute_m2mf(NVGeForce3State *s, NVGeForce3Channel *ch,
                              uint32_t subc, uint32_t method, uint32_t param);
void nv_geforce3_execute_rop(NVGeForce3Channel *ch, uint32_t method,
                             uint32_t param);
void nv_geforce3_execute_patt(NVGeForce3Channel *ch, uint32_t method,
                              uint32_t param);
void nv_geforce3_execute_gdi(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t cls, uint32_t method, uint32_t param);
void nv_geforce3_execute_swzsurf(NVGeForce3Channel *ch, uint32_t method,
                                 uint32_t param);
void nv_geforce3_execute_chroma(NVGeForce3Channel *ch, uint32_t method,
                                uint32_t param);
void nv_geforce3_execute_lin(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t method, uint32_t param);
void nv_geforce3_execute_rect(NVGeForce3State *s, NVGeForce3Channel *ch,
                              uint32_t method, uint32_t param);
void nv_geforce3_execute_imageblit(NVGeForce3State *s, NVGeForce3Channel *ch,
                                   uint32_t method, uint32_t param);
void nv_geforce3_execute_ifc(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t method, uint32_t param);
void nv_geforce3_execute_surf2d(NVGeForce3State *s, NVGeForce3Channel *ch,
                                uint32_t method, uint32_t param);
void nv_geforce3_execute_iifc(NVGeForce3State *s, NVGeForce3Channel *ch,
                              uint32_t method, uint32_t param);
void nv_geforce3_execute_sifc(NVGeForce3State *s, NVGeForce3Channel *ch,
                              uint32_t method, uint32_t param);
void nv_geforce3_execute_beta(NVGeForce3Channel *ch, uint32_t method,
                              uint32_t param);
void nv_geforce3_execute_tfc(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t method, uint32_t param);
void nv_geforce3_execute_sifm(NVGeForce3State *s, NVGeForce3Channel *ch,
                              uint32_t cls, uint32_t method, uint32_t param);
void nv_geforce3_update_color_bytes_s2d(NVGeForce3Channel *ch);
void nv_geforce3_update_color_bytes_ifc(NVGeForce3Channel *ch);
void nv_geforce3_update_color_bytes_sifc(NVGeForce3Channel *ch);
void nv_geforce3_update_color_bytes_tfc(NVGeForce3Channel *ch);
void nv_geforce3_update_color_bytes_iifc(NVGeForce3Channel *ch);
void nv_geforce3_object_save(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t subc);
void nv_geforce3_object_load(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t subc);

/* nv_geforce3_3d.c */
void nv_geforce3_init_method_handlers(NVGeForce3State *s);
void nv_geforce3_execute_d3d(NVGeForce3State *s, NVGeForce3Channel *ch,
                             uint32_t cls, uint32_t method, uint32_t param);
void nv_geforce3_d3d_process_vertex(NVGeForce3State *s, NVGeForce3Channel *ch,
                                    bool immediate);
void nv_geforce3_d3d_load_vertex(NVGeForce3State *s, NVGeForce3Channel *ch,
                                 uint32_t index);
void nv_geforce3_d3d_clear_surface(NVGeForce3State *s, NVGeForce3Channel *ch);
uint32_t nv_geforce3_d3d_get_surface_pitch_z(NVGeForce3Channel *ch);

#endif /* NV_GEFORCE3_INT_H */
