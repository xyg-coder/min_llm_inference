#pragma once

#include <cstddef>
#include <cuda_runtime.h>

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


/**
    s1: [batch_size, rows, N]
    s2: [batch_size, cols, N]
    output: [batch_size, rows, cols]
*/
void launch_gemm_transpose_kernel(const float* s1, const float* s2, float* output,
    size_t batch_size, size_t rows, size_t cols, size_t N);
