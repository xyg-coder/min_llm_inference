#pragma once

#include <cstddef>
#include <cuda_runtime.h>

const int TILE_SIZE = 16;

struct Stride3D {
    size_t batch_stride;
    size_t row_stride;
    size_t col_stride;
};

// Host wrapper function declarations
void launch_gemm_kernel(const float* s1, const float* s2, float* output,
                        size_t batch_size, size_t rows, size_t N, size_t cols);

void launch_gemm_bias_kernel(const float* s1, Stride3D s1_stride,
                            const float* s2, Stride3D s2_stride,
                            const float* bias, Stride3D bias_stride,
                            float* output, Stride3D output_stride,
                            size_t batch_size, size_t rows, size_t N, size_t cols);


