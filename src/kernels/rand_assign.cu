#include "kernels/rand_assign.h"
#include <ctime>
#include <curand_kernel.h>
#include <random>
#include <utils.h>

__global__ void randomizeArray(float *array, int N, unsigned long seed, float ratio) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state);
    if (idx < N) {
        // Generate random float between 0 and 1
        array[idx] = curand_uniform(&state) * ratio;
    }
}

__global__ void randomizeArray(int* array, int N, unsigned int maxValue, unsigned long seed) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        curandState state;
        curand_init(seed, idx, 0, &state); 
        unsigned int randomInt = curand(&state) % maxValue;
        array[idx] = randomInt;
    }
}

void launch_randn_kernel(float *out, int size, float ratio) {
    randomizeArray<<<ceil_div(size, 256), 256>>>(out, size, std::random_device{}(), ratio);
    CUDA_CHECK_LAST();
}

void launch_randn_kernel(int *out, int size, int max_value) {
    randomizeArray<<<ceil_div(size, 256), 256>>>(out, size, max_value, std::random_device{}());
    CUDA_CHECK_LAST();
}

