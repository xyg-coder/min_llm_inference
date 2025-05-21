#include "kernels/gemm.h"
#include "utils.h"
#include "constants.h"
#include <cstddef>


/**
    s1: [batch_size, rows, N]
    s2: [batch_size, cols, N]
    output: [batch_size, rows, cols]
*/
__global__ void gemm_transpose_kernel(
    const float* s1, const float* s2, float* output,
    size_t batch_size, size_t rows, size_t cols, size_t N) {

    __shared__ float s1_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float s2_shared[TILE_SIZE][TILE_SIZE];
    const float* base_s1 = s1 + blockIdx.z * rows * N;
    const float* base_s2 = s2 + blockIdx.z * cols * N;
    float result = 0;
    /**
     * writing to [blockIdx.z, blockIdx.y * TILE_SIZE + threadIdx.y, blockIdx.x * TILE_SIZE + threadIdx.x]
     * using s1[blockIdx.z, blockIdx.y * TILE_SIZE : blockIdx.y * TILE_SIZE + TILE_SIZE, :]
     * using s2[blockIdx.z, blockIdx.x * TILE_SIZE : blockIdx.x * TILE_SIZE + TILE_SIZE, :]
     */
    for (size_t i = 0; i < N; i += TILE_SIZE) {
        if (blockIdx.y * TILE_SIZE + threadIdx.y < rows && i + threadIdx.x < N) {
            s1_shared[threadIdx.y][threadIdx.x] = base_s1[(blockIdx.y * TILE_SIZE + threadIdx.y) * N + i + threadIdx.x];
        } else {
            s1_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (blockIdx.x * TILE_SIZE + threadIdx.y < cols && i + threadIdx.x < N) {
            // this is to ensure coalensce, always use threadIdx.x as the last dim
            s2_shared[threadIdx.y][threadIdx.x] = base_s2[(blockIdx.x * TILE_SIZE + threadIdx.y) * N + i + threadIdx.x];
        } else {
            s2_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        
        #pragma unroll
        for (size_t j = 0; j < TILE_SIZE; ++j) {
            result += (s1_shared[threadIdx.y][j] * s2_shared[threadIdx.x][j]);
        }
        __syncthreads();
    }

    if (blockIdx.y * TILE_SIZE + threadIdx.y < rows && blockIdx.x * TILE_SIZE + threadIdx.x < cols) {
        output[blockIdx.z * rows * cols + (blockIdx.y * TILE_SIZE + threadIdx.y) * cols + blockIdx.x * TILE_SIZE + threadIdx.x] = result;
    }
}

void launch_gemm_transpose_kernel(const float* s1, const float* s2, float* output,
    size_t batch_size, size_t rows, size_t cols, size_t N) {

    dim3 gridDim(ceil_div(cols, TILE_SIZE), ceil_div(rows, TILE_SIZE), batch_size);
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    gemm_transpose_kernel<<<gridDim, blockDim>>>(s1, s2, output, batch_size, rows, cols, N);
    CUDA_CHECK_LAST();
}

/**
    s1: [batch_size, rows, N]
    s2: [batch_size, N, cols]
    output: [batch_size, rows, cols]
*/
__global__ void gemm_bias_kernel(
    const float* s1, Stride3D s1_stride,
    const float* s2, Stride3D s2_stride,
    const float* bias, Stride3D bias_stride,
    float* output, Stride3D output_stride,
    size_t batch_size, size_t rows, size_t N, size_t cols) {
    __shared__ float s1_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float s2_shared[TILE_SIZE][TILE_SIZE];
    const float* base_s1 = s1 + (blockIdx.z * s1_stride.batch_stride);
    const float* base_s2 = s2 + (blockIdx.z * s2_stride.batch_stride);
    float default_bias = 0;
    if (bias == nullptr) {
        bias = &default_bias;
    }
    const float* base_bias = bias + (blockIdx.z * bias_stride.batch_stride);
    float result = 0;
    for (size_t i = 0; i < N; i += TILE_SIZE) {
        if ((blockIdx.y * TILE_SIZE + threadIdx.y) < rows && (i + threadIdx.x) < N) {
            s1_shared[threadIdx.y][threadIdx.x] = base_s1[(blockIdx.y * TILE_SIZE + threadIdx.y) * s1_stride.row_stride + (i + threadIdx.x) * s1_stride.col_stride];
        } else {
            s1_shared[threadIdx.y][threadIdx.x] = 0;
        }

        if ((blockIdx.x * TILE_SIZE + threadIdx.x) < cols && (i + threadIdx.y) < N) {
            s2_shared[threadIdx.y][threadIdx.x] = base_s2[(i + threadIdx.y) * s2_stride.row_stride + (blockIdx.x * TILE_SIZE + threadIdx.x) * s2_stride.col_stride];
        } else {
            s2_shared[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        #pragma unroll
        for (size_t j = 0; j < TILE_SIZE; ++j) {
            result += (s1_shared[threadIdx.y][j] * s2_shared[j][threadIdx.x]);
        }
        __syncthreads();
    }
    if (blockIdx.y * TILE_SIZE + threadIdx.y < rows && blockIdx.x * TILE_SIZE + threadIdx.x < cols) {
        output[blockIdx.z * output_stride.batch_stride
            + (blockIdx.y * TILE_SIZE + threadIdx.y) * output_stride.row_stride
            + (blockIdx.x * TILE_SIZE + threadIdx.x) * output_stride.col_stride]
            = result + base_bias[(blockIdx.y * TILE_SIZE + threadIdx.y) * bias_stride.row_stride
                + ((blockIdx.x * TILE_SIZE + threadIdx.x) * bias_stride.col_stride)];
    }
}


void launch_gemm_kernel(const float* s1, const float* s2, float* output, size_t batch_size, size_t rows, size_t N, size_t cols) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE, batch_size);
    const float* null_float = nullptr;
    gemm_bias_kernel<<<gridDim, blockDim>>>(
        s1, Stride3D{rows * N, N, 1},
        s2, Stride3D{N * cols, cols, 1},
        null_float, Stride3D{0, 0, 0},
        output, Stride3D{rows * cols, cols, 1},
        batch_size, rows, N, cols);
    CUDA_CHECK_LAST();
}

void launch_gemm_bias_kernel(
    const float* s1, Stride3D s1_stride,
    const float* s2, Stride3D s2_stride,
    const float* bias, Stride3D bias_stride,
    float* output, Stride3D output_stride,
    size_t batch_size, size_t rows, size_t N, size_t cols) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE, batch_size);
    gemm_bias_kernel<<<gridDim, blockDim>>>(
        s1, s1_stride,
        s2, s2_stride,
        bias, bias_stride,
        output, output_stride,
        batch_size, rows, N, cols);
    CUDA_CHECK_LAST();
}
