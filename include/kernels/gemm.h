#pragma once
#include <cstddef>


#define TILE_SIZE 16

template<typename T>
void launch_gemm_kernel(const T* s1, const T* s2, T* output, 
                       size_t batch_size, size_t rows, size_t N, size_t cols);
