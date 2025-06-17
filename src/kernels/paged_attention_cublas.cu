#include "constants.h"
#include "tensor.hpp"
#include "utils.h"
#include "kernels/self_attention_inference_optimized.h"
#include <cassert>
#include <cublas_v2.h>


__global__ void get_latest_batch_embs(
    const float** page_table, const int* lengths,
    float* latest_embs,
    int n_batch, int n_sequence, int emb_dim) {

}

__global__ void save_to_page_table() {}

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
    cublasHandle_t& handle, int n_sequence) {

    int n_batch = page_table.shape()[0];
    int emb_dim = wq.shape()[0];
    
    get_latest_batch_embs(const float **page_table, const int *lengths, float *latest_embs, int n_batch, int n_sequence, int emb_dim);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n_batch, emb_dim, emb_dim, &alpha, latest_emb.data(),
        n_batch, wk.data(), n_batch, &beta, q_output.data(), n_batch);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n_batch, emb_dim, emb_dim, &alpha, latest_emb.data(),
        n_batch, wv.data(), n_batch, &beta, temp_placeholder.data(), n_batch);
    save_to_page_table();
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n_batch, emb_dim, emb_dim, &alpha, latest_emb.data(),
        n_batch, wq.data(), n_batch, &beta, q_output.data(), n_batch);
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
    TensorFloat& latest_emb,
    int n_new_items, int n_sequence, cublasHandle_t& handle) {

    launch_fill_new_k_v_cache_paged_attention(page_table, new_batch_idx, lengths, wk, wv, n_new_items, n_sequence);

    launch_get_latest_k_q_v_paged_attention_cublas(page_table, lengths, wk, wq, wv, q_output, n_sequence);

    launch_qkt_paged_attention(q_output, page_table, lengths, qkt_output);

    launch_softmax_in_place_with_lengths(qkt_output, lengths);

    launch_softmax_v_paged_attention(qkt_output, page_table, attention_result, lengths);
}
