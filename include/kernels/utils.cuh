#pragma once

#include "utils.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <vector_types.h>

__device__ inline float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

__device__ inline float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

__device__ inline float4 float4_add(const float4& f1, const float4& f2) {
    return float4({
        f1.x + f2.x,
        f1.y + f2.y,
        f1.z + f2.z,
        f1.w + f2.w
    });
}

void launch_print_kernel(const float* data, int size);

void assert_float_kernel_close(const float* data1, const float* data2, int size, float threshold);

void assert_int_kernel_close(const int* data1, const int* data2, int size);

// emb_offset==0: inp_emb
// emb_offset==1: k_cache. For k_cache, we will compare with kt_cache
// emb_offset==2: v_cache
void assert_page_table_close(
    const float** page_table, const float* to_compare_with, const int*lengths,
    int n_batch, int n_sequence, int emb_offset, int emb_dim, float threshold=1e-3);
