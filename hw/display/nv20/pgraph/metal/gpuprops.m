/*
 * NV20 PGRAPH Metal GPU properties.
 */
#include "debug.h"
#include "renderer.h"

static GPUProperties pgraph_metal_gpu_properties;

void pgraph_metal_determine_gpu_properties(void)
{
    memset(&pgraph_metal_gpu_properties, 0, sizeof(pgraph_metal_gpu_properties));
}

GPUProperties *pgraph_metal_get_gpu_properties(void)
{
    return &pgraph_metal_gpu_properties;
}
