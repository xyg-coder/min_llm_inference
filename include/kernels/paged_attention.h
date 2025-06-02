#pragma once

#include "tensor.hpp"

/**
 * inp_embedding: [n_batch, n_sequence / PAGE_BLOCK_SIZE], each element is one float*
 * lengths: [n_batch]
 * wk, wq, wv: [input_dim, input_dim]
 * new_batch_idx: [n_batch], but only [0 : n_new_items - 1] are valid
 * k_cache: [n_batch, n_sequence / PAGE_BLOCK_SIZE], each element is one float* or nullptr
 * v_cache: [n_batch, n_sequence / PAGE_BLOCK_SIZE], each element is one float* or nullptr
 * q_output: [n_batch, input_dim]
 * qkt_output: [n_batch, n_sequence]
 * attention_result: [n_batch, input_dim]
 */
void launch_paged_attention_kernels(
    const TensorFloatPoint& inp_embedding, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wq, const TensorFloat& wv,
    const TensorInt& new_batch_idx, TensorFloatPoint& k_cache,
    TensorFloatPoint v_cache,
    TensorFloat& q_output, TensorFloat& qkt_output, TensorFloat& attention_result,
    int n_new_items, int n_batch);
