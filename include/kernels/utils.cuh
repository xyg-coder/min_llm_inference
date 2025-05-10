#pragma once

#include "utils.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <vector_types.h>

__device__ float vec_at(const float4& vec, int index);
__device__ float& vec_at(float4& vec, int index);
void launch_print_kernel(const float* data, int size);
