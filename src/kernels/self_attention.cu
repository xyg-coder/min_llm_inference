#include "tensor.h"
#include <cassert>
#include "kernels/gemm.h"
#include "kernels/softmax.h"
#include "utils.h"


/**
kqv: [n_batch, n_sequence, dims * 3]
kqt: [n_batch, n_sequence, n_sequence]
 */
__global__ void kqt_kernel(const float* kqv, float* output, int n_batch, int n_sequence, int dims) {
    __shared__ float k_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float q_shared[TILE_SIZE][TILE_SIZE];
    const float* base = kqv + (blockIdx.z * n_sequence * dims * 3);
    float result = 0;
    for (int i = 0; i < dims; i += TILE_SIZE) {
        if (blockIdx.y * TILE_SIZE + threadIdx.y < n_sequence && i + threadIdx.x < dims) {
            k_shared[threadIdx.y][threadIdx.x] = base[(blockIdx.y * TILE_SIZE + threadIdx.y) * dims * 3 + i + threadIdx.x];
        } else {
            k_shared[threadIdx.y][threadIdx.x] = 0;
        }
        if (blockIdx.x * TILE_SIZE + threadIdx.x < n_sequence && i + threadIdx.x < dims) {
            q_shared[threadIdx.y][threadIdx.x] = base[(blockIdx.x * TILE_SIZE + threadIdx.x) * dims * 3 + dims + i + threadIdx.x];
        } else {
            q_shared[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            result += (k_shared[threadIdx.y][j] * q_shared[threadIdx.x][j]);
        }
        __syncthreads();
    }

    if (blockIdx.y * TILE_SIZE + threadIdx.y < n_sequence && blockIdx.x * TILE_SIZE + threadIdx.x < n_sequence) {
        output[blockIdx.z * n_sequence * n_sequence + (blockIdx.y * TILE_SIZE + threadIdx.y) * n_sequence + blockIdx.x * TILE_SIZE + threadIdx.x] =
            result / sqrtf(dims);
    }
}

void launch_kqt_kernel(const float* kqv, float* output, int n_batch, int n_sequence, int dims) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(n_batch, ceil_div(n_sequence, TILE_SIZE), ceil_div(n_sequence, TILE_SIZE));
    kqt_kernel<<<gridDim, blockDim>>>(kqv, output, n_batch, n_sequence, dims);
    CUDA_CHECK_LAST();
}

/**
kqv: [n_batch, n_sequence, dims * 3]
softmax_result: [n_batch, n_sequence, n_sequence]
result: [n_batch, n_sequence, dims]
 */
__global__ void softmax_v_kernel(
    const float* softmax_result, const float* kqv, float* output,
    int n_batch, int n_sequence, int dims
) {
    __shared__ float softmax_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float v_shared[TILE_SIZE][TILE_SIZE];

    const float* softmax_base = softmax_result + (blockIdx.z * n_sequence * n_sequence);
    const float* kqv_base = kqv + (blockIdx.z * n_sequence * dims * 3);
    float result = 0;
    for (int i = 0; i < n_sequence; i += TILE_SIZE) {
        if (blockIdx.y * TILE_SIZE + threadIdx.y < n_sequence && i + threadIdx.x < n_sequence) {
            softmax_shared[threadIdx.y][threadIdx.x] = softmax_base[(blockIdx.y * TILE_SIZE + threadIdx.y) * n_sequence + i + threadIdx.x];
        } else {
            softmax_shared[threadIdx.y][threadIdx.x] = 0;
        }

        if (blockIdx.x * TILE_SIZE + threadIdx.x < dims && i + threadIdx.y < n_sequence) {
            v_shared[threadIdx.y][threadIdx.x] = kqv_base[(i + threadIdx.y) * dims * 3 + dims * 2 + blockIdx.x * TILE_SIZE + threadIdx.x];
        } else{
            v_shared[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < TILE_SIZE; ++j) {
            result += (softmax_shared[threadIdx.y][j] * v_shared[j][threadIdx.x]);
        }
        __syncthreads();
    }
    if (blockIdx.y * TILE_SIZE + threadIdx.y < n_sequence && blockIdx.x * TILE_SIZE + threadIdx.x < dims) {
        output[blockIdx.z * n_sequence * dims + (blockIdx.y * TILE_SIZE + threadIdx.y) * dims + blockIdx.x * TILE_SIZE + threadIdx.x] = result;
    }
}

void launch_softmax_v_kernel(
    const float* softmax_result, const float* kqv, float* output,
    int n_batch, int n_sequence, int dims) {

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(n_batch, ceil_div(n_sequence, TILE_SIZE), ceil_div(dims, TILE_SIZE));
    softmax_v_kernel<<<gridDim, blockDim>>>(softmax_result, kqv, output, n_batch, n_sequence, dims);
    CUDA_CHECK_LAST();
}


Tensor self_attention(const Tensor& inp, const Tensor& wk_wq_wv) {
    size_t n_batch = inp.shape()[0];
    size_t n_sequence = inp.shape()[1];
    size_t input_dims = inp.shape()[2];
    size_t output_dims_3 = wk_wq_wv.shape()[1];

    assert(wk_wq_wv.shape()[0] == input_dims);
    assert(output_dims_3 % 3 == 0);

    Tensor KQV({n_batch, n_sequence, output_dims_3}, DeviceType::DEVICE);
    launch_gemm_bias_kernel(
        inp.data(), Stride3D({n_sequence * input_dims, input_dims, 1}),
        KQV.data(), Stride3D({0, output_dims_3, 1}),
        nullptr, Stride3D(), 
        KQV.data(), Stride3D({n_sequence * output_dims_3, output_dims_3, 1}),
        n_batch, n_sequence, input_dims, output_dims_3);
    
    Tensor kqt({n_batch, n_sequence, n_sequence}, DeviceType::DEVICE);
    launch_kqt_kernel(KQV.data(), kqt.data(), n_batch, n_sequence, output_dims_3 / 3);
    launch_softmax_in_place_kernel(kqt.data(), n_batch, n_sequence);
    Tensor result({n_batch, n_sequence, output_dims_3 / 3}, DeviceType::DEVICE);
    launch_softmax_v_kernel(kqt.data(), KQV.data(), result.data(), n_batch, n_sequence, output_dims_3 / 3);
    return result;
}