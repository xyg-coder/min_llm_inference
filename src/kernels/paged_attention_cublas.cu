#include "constants.h"
#include "tensor.hpp"
#include "utils.h"
#include "kernels/self_attention_inference_optimized.h"
#include <cassert>
#include <cstdlib>
#include <cublas_v2.h>
#include "kernels/paged_attention.h"
#include "kernels/gemm.h"

/**
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * lengths: [n_batch]
 * latest_embs: [n_batch, embs]
 */
__global__ void get_latest_batch_embs(
    const float** page_table, const int* lengths,
    float* latest_embs,
    int n_batch, int n_sequence, int emb_dim) {

    int i_batch = blockIdx.y;
    int cur_length = lengths[i_batch];
    if (cur_length == 0) {
        return;
    }
    int i_sequence = cur_length - 1;
    int i_dim = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ const float* page_pos;
    if (threadIdx.x == 0) {
        page_pos = page_table[i_batch * (n_sequence / PAGE_BLOCK_SIZE) + i_sequence / PAGE_BLOCK_SIZE];
    }
    __syncthreads();
    if (i_dim < emb_dim) {
        latest_embs[i_batch * emb_dim + i_dim] = get_page_table_value(page_pos, i_batch, n_sequence, i_sequence, emb_dim, PAGE_BLOCK_SIZE, i_dim, INP_EMB_EMB_OFFSET);
    }
}

/**
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * lengths: [n_batch]
 * latest_k, latest_v: [n_batch, embs]
 */
__global__ void save_to_page_table(
    float** page_table, const int* lengths, 
    const float* latest_k, const float* latest_v,
    int n_batch, int n_sequence, int emb_dim) {
    
    int i_batch = blockIdx.y;
    int cur_length = lengths[i_batch];
    if (cur_length == 0) {
        return;
    }
    int i_sequence = cur_length - 1;
    int i_dim = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float* page_pos;
    if (threadIdx.x == 0) {
        page_pos = page_table[i_batch * (n_sequence / PAGE_BLOCK_SIZE) + i_sequence / PAGE_BLOCK_SIZE];
    }
    __syncthreads();
    if (i_dim < emb_dim) {
        set_page_table_value(page_pos, i_batch, n_sequence, i_sequence, emb_dim, PAGE_BLOCK_SIZE, i_dim, K_CACHE_EMB_OFFSET, latest_k[i_batch * emb_dim + i_dim]);
        set_page_table_value(page_pos, i_batch, n_sequence, i_sequence, emb_dim, PAGE_BLOCK_SIZE, i_dim, V_CACHE_EMB_OFFSET, latest_v[i_batch * emb_dim + i_dim]);
    }
}

/**
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * lengths: [n_batch]
 * latest_k, latest_v: [n_batch, embs]
 */
__global__ void assert_emb_equal(
    const float* q_to_check, const int* lengths, 
    const float* latest_emb,
    int n_batch, int n_sequence, int emb_dim) {
    
    int i_batch = blockIdx.y;
    int cur_length = lengths[i_batch];
    if (cur_length == 0) {
        return;
    }
    int i_sequence = cur_length - 1;
    int i_dim = blockIdx.x * blockDim.x + threadIdx.x;

    // __shared__ float* page_pos;
    // if (threadIdx.x == 0) {
    //     page_pos = page_table[i_batch * (n_sequence / PAGE_BLOCK_SIZE) + i_sequence / PAGE_BLOCK_SIZE];
    // }
    __syncthreads();
    if (i_dim < emb_dim) {
        assert(abs(latest_emb[i_batch * emb_dim + i_dim] - q_to_check[i_batch * emb_dim + i_dim]) < 0.001);
    }
}

/**
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * lengths: [n_batch]
 * wq, wk, wv: [emb_dim, emb_dim]
 * q_output: [n_batch, emb_dim]
 * temp_placeholder: [n_batch, emb_dim]
 */
void launch_get_latest_k_q_v_paged_attention_cublas(
    TensorFloatPoint& page_table, const TensorInt& lengths,
    TensorFloat& latest_emb,
    const TensorFloat& wk, const TensorFloat& wq,
    const TensorFloat& wv, TensorFloat& q_output, TensorFloat& temp_placeholder,
    cublasHandle_t& handle, int n_sequence, const TensorFloat& q_to_check) {

    int n_batch = page_table.shape()[0];
    int emb_dim = wq.shape()[0];
    
    dim3 gridDim(ceil_div(emb_dim, TILE_SIZE_SQUARE), n_batch);
    get_latest_batch_embs<<<gridDim, TILE_SIZE_SQUARE>>>((const float**)page_table.data(), lengths.data(), latest_emb.data(), n_batch, n_sequence, emb_dim);
    CUDA_CHECK_LAST();

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, emb_dim, n_batch, emb_dim, &alpha, wk.data(),
        emb_dim, latest_emb.data(), emb_dim, &beta, q_output.data(), emb_dim));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, emb_dim, n_batch, emb_dim, &alpha, wv.data(),
        emb_dim, latest_emb.data(), emb_dim, &beta, temp_placeholder.data(), emb_dim));
    save_to_page_table<<<gridDim, TILE_SIZE_SQUARE>>>(page_table.data(), lengths.data(), q_output.data(), temp_placeholder.data(), n_batch, n_sequence, emb_dim);
    CUDA_CHECK_LAST();
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, emb_dim, n_batch, emb_dim, &alpha, wq.data(),
        emb_dim, latest_emb.data(), emb_dim, &beta, q_output.data(), emb_dim));
    // launch_gemm_kernel(latest_emb.data(), wq.data(), q_output.data(), 1, n_batch, emb_dim, emb_dim);
    assert_emb_equal<<<gridDim, TILE_SIZE_SQUARE>>>(q_output.data(), lengths.data(), q_to_check.data(), n_batch, n_sequence, emb_dim);
    CUDA_CHECK_LAST();
}


/**
 * - page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE], input_embedding + k_cache + v_cache
 *      Each pointer points to a memory block of (3 * embedding_dim * PAGE_BLOCK_SIZE)
 * - lengths: [n_batch], the token lengths for each batch
 * - wk, wq, wv: [emb_dim, emb_dim]: since we don't have the following feed-forward layer, 
 * input_dim should be equal to output_dim
 * - new_batch_idx: [n_batch], but only n_new_items are used
 * - q_output: [n_batch, emb_dim]
 * - qkt_output: [n_batch, n_sequence]
 * - attention_result: [n_batch, emb_dim]
 */
void paged_attention_with_cublas(
    TensorFloatPoint& page_table,
    const TensorInt& lengths,
    const TensorFloat& wk,
    const TensorFloat& wq,
    const TensorFloat& wv,
    const TensorInt& new_batch_idx,
    TensorFloat& q_output, TensorFloat& qkt_output, TensorFloat& attention_result,
    TensorFloat& latest_emb, TensorFloat& temp_placeholder,
    int n_new_items, int n_sequence, cublasHandle_t& handle) {

    launch_fill_new_k_v_cache_paged_attention(page_table, new_batch_idx, lengths, wk, wv, n_new_items, n_sequence);

    // launch_get_latest_k_q_v_paged_attention_cublas(page_table, lengths, latest_emb, wk, wq, wv, q_output, temp_placeholder, handle, n_sequence);

    launch_qkt_paged_attention(q_output, page_table, lengths, qkt_output);

    launch_softmax_in_place_with_lengths(qkt_output, lengths);

    launch_softmax_v_paged_attention(qkt_output, page_table, attention_result, lengths);
}
