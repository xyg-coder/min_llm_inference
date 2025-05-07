#include "kernels/rand_assign.h"
#include <ctime>
#include <curand_kernel.h>
#include <utils.h>

__global__ void randomizeArray(float *array, int N, unsigned long seed) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state);
    if (idx < N) {
        // Generate random float between 0 and 1
        array[idx] = curand_uniform(&state);
    }
}

void launch_randn_kernel(float *out, int size) {
    randomizeArray<<<ceil_div(size, 256), 256>>>(out, size, time(NULL));
    cuda_check(cudaGetLastError(), __FILE__, __LINE__);
}
