#pragma once
#include <cstdio>
#include <cuda_runtime.h>

#ifdef USE_SYNC_CUDA_CHECK
#define CUDA_CHECK_LAST()                                            \
    do {                                                            \
        cudaError_t err_sync = cudaDeviceSynchronize();             \
        if (err_sync != cudaSuccess) {                              \
            cuda_check(err_sync, __FILE__, __LINE__);               \
        }                                                           \
        cudaError_t err_async = cudaGetLastError();                 \
        if (err_async != cudaSuccess) {                             \
            cuda_check(err_async, __FILE__, __LINE__);              \
        }                                                           \
    } while (0)
#else
#define CUDA_CHECK_LAST() cuda_check(cudaGetLastError(), __FILE__, __LINE__)
#endif


// emb_offset == 0: inp_embedding
// emb_offset == 1: k_cache
// emb_offset == 2: v_cache
__device__ __inline__ float get_page_table_value(
    const float** page_table, int i_batch, int n_sequence, int i_sequence, int emb_dim, int page_block_size, int i_dim, int emb_offset) {
    
    int page_table_width = n_sequence / page_block_size;
    const float* page_pos = page_table[i_batch * page_table_width + i_sequence / page_block_size];
    return page_pos[(i_sequence % page_block_size) * emb_dim * 3 + emb_offset * emb_dim + i_dim];
}

__device__ __inline__ void set_page_table_value(
    float** page_table, int i_batch, int n_sequence, int i_sequence, int emb_dim, int page_block_size, int i_dim, int emb_offset,
    float value) {
    
    int page_table_width = n_sequence / page_block_size;
    float* page_pos = page_table[i_batch * page_table_width + i_sequence / page_block_size];
    page_pos[(i_sequence % page_block_size) * emb_dim * 3 + emb_dim * emb_offset + i_dim] = value;
}

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

void launch_clone_inp_embedding_k_v_cache(
    float** page_table, const float* inp_embedding, const float* kt_cache,
    const float* v_cache, const int* lengths, int n_batch, int n_sequence, int emb_dim);
