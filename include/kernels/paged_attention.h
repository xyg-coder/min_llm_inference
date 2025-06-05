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
void paged_attention(
    TensorFloatPoint& page_table,
    const TensorInt& lengths,
    const TensorFloat& wk,
    const TensorFloat& wq,
    const TensorFloat& wv,
    const TensorInt& new_batch_idx,
    TensorFloat& q_output, TensorFloat& qkt_output, TensorFloat& attention_result,
    int n_new_items, int n_sequence);


void launch_fill_new_k_v_cache_paged_attention(
    TensorFloatPoint page_table, const TensorInt& new_batch_idx, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wv, int n_new_items, int n_sequence);

void launch_get_latest_k_q_v_paged_attention(
    TensorFloatPoint& page_table, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wq,
    const TensorFloat& wv, TensorFloat& q_output, int n_sequence);

void launch_qkt_paged_attention(
    const TensorFloat& q_output, const TensorFloatPoint& page_table, const TensorInt& lengths,
    TensorFloat& qkt_output);

void launch_softmax_v_paged_attention(
    const TensorFloat& softmax_result, const TensorFloatPoint& page_table, TensorFloat& attention_result,
    const TensorInt& lengths);
