#include "constants.h"
#include "kernels/utils.cuh"
#include "utils.h"
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void print_kernel(const float* data, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        printf("printing: %d = %f\n", idx, data[idx]);
    }
}

void launch_print_kernel(const float* data, int size) {
    constexpr int blockDim = 256;
    print_kernel<<<ceil_div(size, blockDim), blockDim>>>(data, size);
}

__global__ void compare_float_array_kernel(const float* data1, const float* data2, int size, float threshold, int* exception_flag) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        if (fabsf(data1[idx] - data2[idx]) > threshold) {
            *exception_flag = 1;
        }
    }
}

__global__ void compare_int_array_kernel(const int* data1, const int* data2, int size, int* exception_flag) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        if (data1[idx] != data2[idx]) {
            *exception_flag = 1;
        }
    }
}

// page_table: [n_batch, n_sequence / page_block]
// to_compare_with: [n_batch, n_sequence, emb_dim]
__global__ void compare_page_table(const float** page_table, const float* to_compare_with, const int* lengths, int n_batch, int n_sequence, int page_block_size, int emb_offset, int emb_dim,
    float threshold, int* exception_flag) {

    int i_batch = blockIdx.z;
    int batch_length = lengths[i_batch];
    int i_sequence = blockIdx.y;
    int i_dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_sequence >= batch_length || i_dim >= emb_dim) {
        return;
    }
    if (fabsf(to_compare_with[i_batch * n_sequence * emb_dim + i_sequence * emb_dim + i_dim] -
        get_page_table_value(page_table, i_batch, n_sequence, i_sequence, emb_dim, page_block_size, i_dim, emb_offset)) > threshold) {

        *exception_flag = 1;
    }
}

// page_table: [n_batch, n_sequence / page_block]
// to_compare_with: [n_batch, emb_dim, n_sequence]
__global__ void compare_page_table_transpose(const float** page_table, const float* to_compare_with, const int* lengths, int n_batch, int n_sequence, int page_block_size, int emb_offset, int emb_dim,
    float threshold, int* exception_flag) {

    __shared__ float compare_with_shared[TILE_SIZE][TILE_SIZE];
    int i_batch = blockIdx.z;
    int batch_length = lengths[i_batch];
    int i_sequence = blockIdx.y * blockDim.y + threadIdx.y;
    int i_dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.y * blockDim.y >= batch_length || blockIdx.x * blockDim.x >= emb_dim) {
        return;
    }

    int to_compare_with_i_sequence = blockIdx.y * blockDim.y + threadIdx.x;
    int to_compare_with_i_emb_dim = blockIdx.x * blockDim.x + threadIdx.y;
    if (to_compare_with_i_sequence < batch_length && to_compare_with_i_emb_dim < emb_dim) {
        compare_with_shared[threadIdx.y][threadIdx.x] = to_compare_with[
            i_batch * n_sequence * emb_dim + to_compare_with_i_emb_dim * n_sequence + to_compare_with_i_sequence];
    }
    __syncthreads();
    if (i_sequence < batch_length && i_dim < emb_dim) {
        if (fabsf(compare_with_shared[threadIdx.x][threadIdx.y] - get_page_table_value(page_table, i_batch, n_sequence, i_sequence, emb_dim, page_block_size, i_dim, emb_offset)) > threshold) {
            *exception_flag = 1;
        }
    }
}

/**
 * Clone the data into the page_table
 * 
 * page_table: [n_batch, n_sequence / page_block]
 * inp_embedding, v_cache: [n_batch, n_sequence, emb_dim]
 * kt_cache: [n_batch, emb_dim, n_sequence]
 */
__global__ void clone_inp_embedding_k_v_cache(
    float** page_table, const float* inp_embedding, const float* kt_cache, const float* v_cache, const int* lengths, 
    int n_batch, int n_sequence, int emb_dim, int page_block_size) {

    int i_batch = blockIdx.z;
    int i_sequence = blockIdx.y * blockDim.y + threadIdx.y;
    int i_dim_4 = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_length = lengths[i_batch];
    // never allocated
    if (batch_length == 0) {
        return;
    }
    // For decoder, it will generate inp_embedding for the next position.
    // So here we clone one more
    int max_to_clone_index = min(batch_length, n_sequence - 1);
    __shared__ float4 kt_cache_shared[TILE_SIZE][TILE_SIZE];
    assert(emb_dim % 4 == 0);
    int emb_dim_4 = emb_dim / 4;
    if (blockIdx.y * blockDim.y > max_to_clone_index || blockIdx.x * blockDim.x >= emb_dim_4) {
        return;
    }
    const float4* kt_cache_4 = reinterpret_cast<const float4*>(kt_cache);
    const float4* inp_embedding_4 = reinterpret_cast<const float4*>(inp_embedding);
    const float4* v_cache_4 = reinterpret_cast<const float4*>(v_cache);

    if (blockIdx.y * blockDim.y + threadIdx.x <= max_to_clone_index && blockIdx.x * blockDim.x + threadIdx.y < emb_dim_4) {
        kt_cache_shared[threadIdx.y][threadIdx.x] = kt_cache_4[i_batch * emb_dim_4 * n_sequence +
            (blockIdx.x * blockDim.x + threadIdx.y) * n_sequence + blockIdx.y * blockDim.y + threadIdx.x];
    }
    __syncthreads();
    if (i_sequence <= max_to_clone_index && i_dim_4 < emb_dim_4) {
        set_page_table_value_float4(page_table, i_batch, n_sequence, i_sequence, emb_dim_4, page_block_size, i_dim_4, INP_EMB_EMB_OFFSET, 
            inp_embedding_4[i_batch * emb_dim_4 * n_sequence + i_sequence * emb_dim_4 + i_dim_4]);
        set_page_table_value_float4(page_table, i_batch, n_sequence, i_sequence, emb_dim_4, page_block_size, i_dim_4, V_CACHE_EMB_OFFSET, 
            v_cache_4[i_batch * emb_dim_4 * n_sequence + i_sequence * emb_dim_4 + i_dim_4]);
        set_page_table_value_float4(page_table, i_batch, n_sequence, i_sequence, emb_dim_4, page_block_size, i_dim_4, K_CACHE_EMB_OFFSET, 
            kt_cache_shared[threadIdx.x][threadIdx.y]);
    }
}

void assert_float_kernel_close(const float* data1, const float* data2, int size, float threshold) {
    constexpr int blockDim = 256;
    int *d_exception_flag, h_exception_flag = 0;
    cudaMalloc(&d_exception_flag, sizeof(int));
    cudaMemcpy(d_exception_flag, &h_exception_flag, sizeof(int), cudaMemcpyHostToDevice);
    compare_float_array_kernel<<<ceil_div(size, blockDim), blockDim>>>(
        data1, data2, size, threshold, d_exception_flag);
    CUDA_CHECK_LAST();
    cudaMemcpy(&h_exception_flag, d_exception_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_exception_flag) {
        throw std::runtime_error("2 arrays are not close");
    }
}

void assert_int_kernel_close(const int* data1, const int* data2, int size) {
    constexpr int blockDim = 256;
    int *d_exception_flag, h_exception_flag = 0;
    cudaMalloc(&d_exception_flag, sizeof(int));
    cudaMemcpy(d_exception_flag, &h_exception_flag, sizeof(int), cudaMemcpyHostToDevice);
    compare_int_array_kernel<<<ceil_div(size, blockDim), blockDim>>>(
        data1, data2, size, d_exception_flag);
    CUDA_CHECK_LAST();
    cudaMemcpy(&h_exception_flag, d_exception_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_exception_flag) {
        throw std::runtime_error("2 arrays are not close");
    }
}

void assert_page_table_close(
    const float** page_table, const float* to_compare_with, const int*lengths,
    int n_batch, int n_sequence, int emb_offset, int emb_dim, float threshold) {
    
    assert(emb_offset == K_CACHE_EMB_OFFSET || emb_offset == V_CACHE_EMB_OFFSET || emb_offset == INP_EMB_EMB_OFFSET);
    int *d_exception_flag, h_exception_flag = 0;
    cudaMalloc(&d_exception_flag, sizeof(int));
    cudaMemcpy(d_exception_flag, &h_exception_flag, sizeof(int), cudaMemcpyHostToDevice);
    if (emb_offset == K_CACHE_EMB_OFFSET) {
        dim3 gridDim(ceil_div(emb_dim, TILE_SIZE), ceil_div(n_sequence, TILE_SIZE), n_batch);
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        compare_page_table_transpose<<<gridDim, blockDim>>>(page_table, to_compare_with, lengths, n_batch, n_sequence, PAGE_BLOCK_SIZE,
            emb_offset, emb_dim, threshold, d_exception_flag);

    } else {
        dim3 gridDim(ceil_div(emb_dim, TILE_SIZE_SQUARE), n_sequence, n_batch);
        compare_page_table<<<gridDim, TILE_SIZE_SQUARE>>>(
            page_table, to_compare_with, lengths, n_batch, n_sequence, PAGE_BLOCK_SIZE,
            emb_offset, emb_dim, threshold, d_exception_flag);
    }
    CUDA_CHECK_LAST();
    cudaMemcpy(&h_exception_flag, d_exception_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_exception_flag) {
        throw std::runtime_error("2 arrays are not close");
    }
}

void launch_clone_inp_embedding_k_v_cache(
    float** page_table, const float* inp_embedding, const float* kt_cache,
    const float* v_cache, const int* lengths, int n_batch, int n_sequence, int emb_dim) {

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(ceil_div(emb_dim, TILE_SIZE), ceil_div(n_sequence, TILE_SIZE), n_batch);
    clone_inp_embedding_k_v_cache<<<gridDim, blockDim>>>(page_table, inp_embedding, kt_cache, v_cache, lengths, n_batch, n_sequence, emb_dim, PAGE_BLOCK_SIZE);
    CUDA_CHECK_LAST();
}
