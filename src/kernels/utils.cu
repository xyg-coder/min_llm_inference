#include "kernels/utils.cuh"

__device__ float vec_at(const float4& vec, int index) {
    return reinterpret_cast<const float*>(&vec)[index];
}

__device__ float& vec_at(float4& vec, int index) {
    return reinterpret_cast<float*>(&vec)[index];
}

__global__ void print_kernel(const float* data, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        printf("printing: %d = %f\n", idx, data[idx]);
    }
}

void launch_print_kernel(const float* data, int size) {
    constexpr int blockDim = 256;
    print_kernel<<<ceil_div(size, blockDim), blockDim>>>(data, size);
}