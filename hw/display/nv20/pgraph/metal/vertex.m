/*
 * NV20 PGRAPH Metal vertex buffer management.
 */
#include "hw/display/nv20/nv20_regs.h"
#include "hw/display/nv20/nv20_int.h"
#include "debug.h"
#include "renderer.h"

static void update_memory_buffer(NV20State *d, hwaddr addr, hwaddr size,
                                 bool quick)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    hwaddr end = TARGET_PAGE_ALIGN(addr + size);
    addr &= TARGET_PAGE_MASK;
    assert(end < memory_region_size(&d->vram_mr));

    static hwaddr last_addr, last_end;
    if (quick && (addr >= last_addr) && (end <= last_end)) {
        return;
    }
    last_addr = addr;
    last_end = end;

    size = end - addr;
    if (memory_region_test_and_clear_dirty(&d->vram_mr, addr, size,
                                           DIRTY_MEMORY_VGA)) {
        nv20_mtl_buffer_write(r->mtl_memory_buffer, addr, d->vram_ptr + addr,
                              size);
        nv20_profile_inc_counter(NV20_PROF_GEOM_BUFFER_UPDATE_1);
    }
}

void pgraph_metal_update_entire_memory_buffer(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    nv20_mtl_buffer_write(r->mtl_memory_buffer, 0, d->vram_ptr,
                          memory_region_size(&d->vram_mr));
}

void pgraph_metal_bind_vertex_attributes(NV20State *d, unsigned int min_element,
                                         unsigned int max_element,
                                         bool inline_data,
                                         unsigned int inline_stride,
                                         unsigned int provoking_element)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;
    bool updated_memory_buffer = false;
    unsigned int num_elements = max_element - min_element + 1;

    pg->compressed_attrs = 0;

    for (int i = 0; i < NV20_VERTEXSHADER_ATTRIBUTES; i++) {
        VertexAttribute *attr = &pg->vertex_attributes[i];

        if (!attr->count) {
            continue;
        }

        hwaddr attrib_data_addr;
        size_t stride;

        if (inline_data) {
            attrib_data_addr = attr->inline_array_offset;
            stride = inline_stride;
            nv20_mtl_set_vertex_buffer(r->backend, i,
                                       r->mtl_inline_array_buffer,
                                       attrib_data_addr);
        } else {
            hwaddr dma_len;
            uint8_t *attr_data = (uint8_t *)nv_dma_map(
                d, attr->dma_select ? pg->dma_vertex_b : pg->dma_vertex_a,
                &dma_len);
            assert(attr->offset < dma_len);
            attrib_data_addr = attr_data + attr->offset - d->vram_ptr;
            stride = attr->stride;
            hwaddr start = attrib_data_addr + min_element * stride;
            update_memory_buffer(d, start, num_elements * stride,
                                 updated_memory_buffer);
            updated_memory_buffer = true;
            nv20_mtl_set_vertex_buffer(r->backend, i, r->mtl_memory_buffer,
                                       start);
        }

        if (!stride) {
            pgraph_update_inline_value(attr, inline_data ?
                (uint8_t *)pg->inline_array + attr->inline_array_offset :
                d->vram_ptr + attrib_data_addr);
            continue;
        }

        uint32_t provoking_element_index = provoking_element - min_element;
        const uint8_t *last_entry = inline_data ?
            (uint8_t *)pg->inline_array + attr->inline_array_offset :
            d->vram_ptr + attrib_data_addr + min_element * stride;
        last_entry += stride * provoking_element_index;
        pgraph_update_inline_value(attr, last_entry);
        (void)i;
    }
}

unsigned int pgraph_metal_bind_inline_array(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    unsigned int offset = 0;
    for (int i = 0; i < NV20_VERTEXSHADER_ATTRIBUTES; i++) {
        VertexAttribute *attr = &pg->vertex_attributes[i];
        if (attr->count == 0) {
            continue;
        }
        offset = ROUND_UP(offset, attr->size);
        attr->inline_array_offset = offset;
        offset += attr->size * attr->count;
        offset = ROUND_UP(offset, attr->size);
    }

    unsigned int vertex_size = offset;
    unsigned int index_count = pg->inline_array_length * 4 / vertex_size;

    nv20_profile_inc_counter(NV20_PROF_GEOM_BUFFER_UPDATE_2);
    nv20_mtl_buffer_write(r->mtl_inline_array_buffer, 0, pg->inline_array,
                          index_count * vertex_size);
    pgraph_metal_bind_vertex_attributes(d, 0, index_count - 1, true,
                                        vertex_size, index_count - 1);
    return index_count;
}

static PGRAPHMetalState *g_mtl_vertex_state;

static void vertex_cache_entry_init(Lru *lru, LruNode *node, const void *key)
{
    VertexLruNode *vnode = container_of(node, VertexLruNode, node);
    memcpy(&vnode->key, key, sizeof(VertexKey));
    vnode->initialized = false;
    if (!vnode->mtl_buffer && g_mtl_vertex_state) {
        vnode->mtl_buffer = nv20_mtl_buffer_create(g_mtl_vertex_state->backend,
                                                    4096, true);
    }
}

static bool vertex_cache_entry_compare(Lru *lru, LruNode *node, const void *key)
{
    VertexLruNode *vnode = container_of(node, VertexLruNode, node);
    return memcmp(&vnode->key, key, sizeof(VertexKey));
}

static const size_t element_cache_size = 50 * 1024;

void pgraph_metal_init_buffers(NV20State *d)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    g_mtl_vertex_state = r;
    lru_init(&r->element_cache);
    r->element_cache_entries = g_malloc_n(element_cache_size, sizeof(VertexLruNode));
    for (int i = 0; i < element_cache_size; i++) {
        lru_add_free(&r->element_cache, &r->element_cache_entries[i].node);
    }
    r->element_cache.init_node = vertex_cache_entry_init;
    r->element_cache.compare_nodes = vertex_cache_entry_compare;

    r->mtl_inline_array_buffer =
        nv20_mtl_buffer_create(r->backend, 1024 * 1024, true);
    r->mtl_memory_buffer =
        nv20_mtl_buffer_create(r->backend, memory_region_size(&d->vram_mr), true);
    nv20_mtl_buffer_write(r->mtl_memory_buffer, 0, d->vram_ptr,
                          memory_region_size(&d->vram_mr));

    for (int i = 0; i < NV20_VERTEXSHADER_ATTRIBUTES; i++) {
        r->mtl_inline_buffer[i] = nv20_mtl_buffer_create(r->backend, 65536, true);
    }
}

void pgraph_metal_finalize_buffers(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;

    for (int i = 0; i < element_cache_size; i++) {
        if (r->element_cache_entries[i].mtl_buffer) {
            nv20_mtl_buffer_destroy(r->element_cache_entries[i].mtl_buffer);
        }
    }
    lru_flush(&r->element_cache);
    g_free(r->element_cache_entries);
    r->element_cache_entries = NULL;

    for (int i = 0; i < NV20_VERTEXSHADER_ATTRIBUTES; i++) {
        nv20_mtl_buffer_destroy(r->mtl_inline_buffer[i]);
        r->mtl_inline_buffer[i] = NULL;
    }
    nv20_mtl_buffer_destroy(r->mtl_inline_array_buffer);
    r->mtl_inline_array_buffer = NULL;
    nv20_mtl_buffer_destroy(r->mtl_memory_buffer);
    r->mtl_memory_buffer = NULL;
}
