#pragma once
#include "tensor.hpp"
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK_LAST() cuda_check(cudaGetLastError(), __FILE__, __LINE__)

void cuda_check(cudaError_t error, const char *file, int line);

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

class NonCopyableNonClonable {
protected:
    NonCopyableNonClonable() = default;
    ~NonCopyableNonClonable() = default;

    // Optional: If you want base class to only be abstract (cannot be instantiated)
    // NonCopyableNonClonable(const NonCopyableNonClonable&) = delete;
    // NonCopyableNonClonable& operator=(const NonCopyableNonClonable&) = delete;

public:
    // Delete copy constructor & copy assignment
    NonCopyableNonClonable(const NonCopyableNonClonable&) = delete;
    NonCopyableNonClonable& operator=(const NonCopyableNonClonable&) = delete;
};
