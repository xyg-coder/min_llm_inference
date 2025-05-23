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
