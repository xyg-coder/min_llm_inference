#include "kernels/gemm.cuh"
#include <cstddef>

/**
    s1: [batch_size, rows, N]
    s2: [batch_size, N, cols]
    output: [batch_size, rows, cols]
*/
template<typename T>
__global__ void gemm_bias_kernel(
    const T* s1, Stride3D s1_stride,
    const T* s2, Stride3D s2_stride,
    const T* bias, Stride3D bias_stride,
    T* output, Stride3D output_stride,
    size_t batch_size, size_t rows, size_t N, size_t cols) {
    __shared__ T s1_shared[TILE_SIZE][TILE_SIZE];
    __shared__ T s2_shared[TILE_SIZE][TILE_SIZE];
    const T* base_s1 = s1 + (blockIdx.z * s1_stride.batch_stride);
    const T* base_s2 = s2 + (blockIdx.z * s2_stride.batch_stride);
    T default_bias = 0;
    if (bias == nullptr) {
        bias = &default_bias;
    }
    const T* base_bias = bias + (blockIdx.z * bias_stride.batch_stride);
    T result = 0;
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

// Explicit instantiation for supported types
template __global__ void gemm_bias_kernel<float>(
    const float*, Stride3D,
    const float*, Stride3D,
    const float*, Stride3D,
    float*, Stride3D,
    size_t, size_t, size_t, size_t);


template<typename T>
void launch_gemm_kernel(const T* s1, const T* s2, T* output, size_t batch_size, size_t rows, size_t N, size_t cols) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE, batch_size);
    const float* null_float = nullptr;
    gemm_bias_kernel<<<gridDim, blockDim>>>(
        s1, Stride3D{rows * N, N, 1},
        s2, Stride3D{N * cols, cols, 1},
        null_float, Stride3D{0, 0, 0},
        output, Stride3D{rows * cols, cols, 1},
        batch_size, rows, N, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw(std::runtime_error("Gemm kernel throws exception"));
    }
}

template<typename T>
void launch_gemm_bias_kernel(
    const T* s1, Stride3D s1_stride,
    const T* s2, Stride3D s2_stride,
    const T* bias, Stride3D bias_stride,
    T* output, Stride3D output_stride,
    size_t batch_size, size_t rows, size_t N, size_t cols) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((cols + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE, batch_size);
    gemm_bias_kernel<<<gridDim, blockDim>>>(
        s1, s1_stride,
        s2, s2_stride,
        bias, bias_stride,
        output, output_stride,
        batch_size, rows, N, cols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw(std::runtime_error("Gemm kernel throws exception"));
    }
}


// Explicit instantiation for supported host functions
template void launch_gemm_kernel<float>(const float*, const float*, float*, size_t, size_t, size_t, size_t);

template void launch_gemm_bias_kernel<float>(
    const float*, Stride3D,
    const float*, Stride3D,
    const float*, Stride3D,
    float*, Stride3D,
    size_t, size_t, size_t, size_t);
