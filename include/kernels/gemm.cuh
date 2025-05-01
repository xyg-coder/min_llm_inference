#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_SIZE 16

struct Stride3D {
    size_t batch_stride;
    size_t row_stride;
    size_t col_stride;
};

template<typename T>
__global__ void gemm_bias_kernel(
    const T* s1, Stride3D s1_stride,
    const T* s2, Stride3D s2_stride,
    const T* bias, Stride3D bias_stride,
    T* output, Stride3D output_stride,
    size_t batch_size, size_t rows, size_t N, size_t cols);

// Host wrapper function declarations
template<typename T>
void launch_gemm_kernel(const T* s1, const T* s2, T* output,
                        size_t batch_size, size_t rows, size_t N, size_t cols);

template<typename T>
void launch_gemm_bias_kernel(const T* s1, Stride3D s1_stride,
                            const T* s2, Stride3D s2_stride,
                            const T* bias, Stride3D bias_stride,
                            T* output, Stride3D output_stride,
                            size_t batch_size, size_t rows, size_t N, size_t cols);


