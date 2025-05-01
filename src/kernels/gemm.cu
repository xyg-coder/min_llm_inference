#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>
#include "kernels/gemm.h"

/**
    s1: [batch_size, rows, N]
    s2: [batch_size, N, cols]
    output: [batch_size, rows, cols]
*/
template<typename T>
__global__ void gemm_kernel(const T* s1, const T* s2, T* output, size_t batch_size, size_t rows, size_t N, size_t cols) {
    __shared__ T s1_shared[TILE_SIZE][TILE_SIZE];
    __shared__ T s2_shared[TILE_SIZE][TILE_SIZE];
    const T* base_s1 = s1 + (blockIdx.z * rows * N);
    const T* base_s2 = s2 + (blockIdx.z * N * cols);
    T result = 0;
    for (size_t i = 0; i < N; i += TILE_SIZE) {
        if ((blockIdx.y * TILE_SIZE + threadIdx.y) < rows && (i + threadIdx.x) < N) {
            s1_shared[threadIdx.y][threadIdx.x] = base_s1[(blockIdx.y * TILE_SIZE + threadIdx.y) * N + i + threadIdx.x];
        } else {
            s1_shared[threadIdx.y][threadIdx.x] = 0;
        }

        if ((blockIdx.x * TILE_SIZE + threadIdx.x) < cols && (i + threadIdx.y) < N) {
            s2_shared[threadIdx.y][threadIdx.x] = base_s2[(i + threadIdx.y) * cols + blockIdx.x * TILE_SIZE + threadIdx.x];
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
        output[blockIdx.z * rows * cols + (blockIdx.y * TILE_SIZE + threadIdx.y) * cols + blockIdx.x * TILE_SIZE + threadIdx.x] = result;
    }
}

template<typename T>
void launch_gemm_kernel(const T* s1, const T* s2, T* output, size_t batch_size, size_t rows, size_t N, size_t cols) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE, batch_size);
    gemm_kernel<<<gridDim, blockDim>>>(s1, s2, output, batch_size, rows, N, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw(std::runtime_error("Gemm kernel throws exception"));
    }
}

// Explicit template instantiation
template void launch_gemm_kernel<float>(const float*, const float*, float*, 
    size_t, size_t, size_t, size_t);

template void launch_gemm_kernel<int>(const int*, const int*, int*, 
    size_t, size_t, size_t, size_t);
