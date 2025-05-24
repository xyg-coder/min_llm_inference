#include "utils.h"
#include <stdexcept>
#include <cuda_runtime.h>

void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
            cudaGetErrorString(error));
        throw(std::runtime_error("Cuda Failure"));
    }
}
