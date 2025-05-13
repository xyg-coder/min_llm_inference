#include "kernels/utils.cuh"
#include "utils.h"
#include <cuda_runtime.h>
#include <stdexcept>

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

__global__ void compare_float_array_kernel(const float* data1, const float* data2, int size, float threshold, int* exception_flag) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        if (fabsf(data1[idx] - data2[idx]) > threshold) {
            *exception_flag = 1;
        }
    }
}

void assert_float_kernel_close(const float* data1, const float* data2, int size, float threshold) {
    constexpr int blockDim = 256;
    int *d_exception_flag, h_exception_flag = 0;
    cudaMalloc(&d_exception_flag, sizeof(int));
    cudaMemcpy(d_exception_flag, &h_exception_flag, sizeof(int), cudaMemcpyHostToDevice);
    compare_float_array_kernel<<<ceil_div(size, blockDim), blockDim>>>(
        data1, data2, size, threshold, d_exception_flag);
    CUDA_CHECK_LAST();
    cudaMemcpy(&h_exception_flag, d_exception_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_exception_flag) {
        throw std::runtime_error("2 arrays are not close");
    }
}
