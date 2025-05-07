#include <cstdio>
#include <cuda_runtime.h>


void cuda_check(cudaError_t error, const char *file, int line);

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}
