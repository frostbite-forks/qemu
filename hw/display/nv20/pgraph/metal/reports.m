/*
 * NV20 PGRAPH Metal query reports.
 */
#include "hw/display/nv20/nv20_int.h"
#include "renderer.h"

static void process_pending_report(NV20State *d, QueryReport *report)
{
    PGRAPHState *pg = &d->pgraph;
    PGRAPHMetalState *r = pg->metal_renderer_state;

    if (report->clear) {
        r->zpass_pixel_count_result = 0;
        return;
    }

    uint8_t type = GET_MASK(report->parameter, NV097_GET_REPORT_TYPE);
    assert(type == NV097_GET_REPORT_TYPE_ZPASS_PIXEL_CNT);

    for (int i = 0; i < report->query_count; i++) {
        uint32_t result = report->query_results ? report->query_results[i] : 0;
        result /= pg->surface_scale_factor * pg->surface_scale_factor;
        r->zpass_pixel_count_result += result;
    }

    g_free(report->query_results);

    pgraph_write_zpass_pixel_cnt_report(d, report->parameter,
                                        r->zpass_pixel_count_result);
}

void pgraph_metal_process_pending_reports(NV20State *d)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;
    QueryReport *report, *next;

    QSIMPLEQ_FOREACH_SAFE(report, &r->report_queue, entry, next) {
        process_pending_report(d, report);
        QSIMPLEQ_REMOVE_HEAD(&r->report_queue, entry);
        g_free(report);
    }
}

void pgraph_metal_clear_report_value(NV20State *d)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    g_free(r->mtl_zpass_query_results);
    r->mtl_zpass_query_results = NULL;
    r->mtl_zpass_query_count = 0;

    QueryReport *report = g_malloc(sizeof(*report));
    report->clear = true;
    QSIMPLEQ_INSERT_TAIL(&r->report_queue, report, entry);
}

void pgraph_metal_get_report(NV20State *d, uint32_t parameter)
{
    PGRAPHMetalState *r = d->pgraph.metal_renderer_state;

    QueryReport *report = g_malloc(sizeof(*report));
    report->clear = false;
    report->parameter = parameter;
    report->query_count = r->mtl_zpass_query_count;
    report->query_results = r->mtl_zpass_query_results;
    QSIMPLEQ_INSERT_TAIL(&r->report_queue, report, entry);

    r->mtl_zpass_query_count = 0;
    r->mtl_zpass_query_results = NULL;
}

void pgraph_metal_init_reports(NV20State *d)
{
    QSIMPLEQ_INIT(&d->pgraph.metal_renderer_state->report_queue);
}

void pgraph_metal_finalize_reports(PGRAPHState *pg)
{
    PGRAPHMetalState *r = pg->metal_renderer_state;
    QueryReport *report, *next;

    QSIMPLEQ_FOREACH_SAFE(report, &r->report_queue, entry, next) {
        g_free(report->query_results);
        QSIMPLEQ_REMOVE_HEAD(&r->report_queue, entry);
        g_free(report);
    }

    g_free(r->mtl_zpass_query_results);
    r->mtl_zpass_query_results = NULL;
}
