#pragma once
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK_LAST() cuda_check(cudaGetLastError(), __FILE__, __LINE__)

// TODO: synchronized version. maybe use cuda flag to control this:
// #define CUDA_CHECK_LAST()                                            \
//     do {                                                            \
//         cudaError_t err_sync = cudaDeviceSynchronize();             \
//         if (err_sync != cudaSuccess) {                              \
//             cuda_check(err_sync, __FILE__, __LINE__);               \
//         }                                                           \
//         cudaError_t err_async = cudaGetLastError();                 \
//         if (err_async != cudaSuccess) {                             \
//             cuda_check(err_async, __FILE__, __LINE__);              \
//         }                                                           \
//     } while (0)

void cuda_check(cudaError_t error, const char *file, int line);

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

class NonCopyableNonClonable {
protected:
    NonCopyableNonClonable() = default;
    ~NonCopyableNonClonable() = default;

    NonCopyableNonClonable(NonCopyableNonClonable&&) noexcept = default;
    NonCopyableNonClonable& operator=(NonCopyableNonClonable&&) noexcept = default;
    // Optional: If you want base class to only be abstract (cannot be instantiated)
    // NonCopyableNonClonable(const NonCopyableNonClonable&) = delete;
    // NonCopyableNonClonable& operator=(const NonCopyableNonClonable&) = delete;

public:
    // Delete copy constructor & copy assignment
    NonCopyableNonClonable(const NonCopyableNonClonable&) = delete;
    NonCopyableNonClonable& operator=(const NonCopyableNonClonable&) = delete;
};
