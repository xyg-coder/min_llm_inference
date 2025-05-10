#pragma once
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK_LAST() cuda_check(cudaGetLastError(), __FILE__, __LINE__)

void cuda_check(cudaError_t error, const char *file, int line);

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
