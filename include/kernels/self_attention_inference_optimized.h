#pragma once

#include "tensor.hpp"

void launch_fill_new_kt_v_cache(
    const TensorFloat& inp_embedding, const TensorInt& new_batch_idx, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wv, TensorFloat& kt_cache,
    TensorFloat& v_cache, int n_new_items);


void launch_get_latest_kt_q_v(
    const TensorFloat& inp_embedding, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wq,
    const TensorFloat& wv, TensorFloat& kt_cache,
    TensorFloat& v_cache, TensorFloat& q_output);

void launch_qkt(
    const TensorFloat& q_output, const TensorFloat& kt_cache, const TensorInt& lengths,
    TensorFloat& qkt_output);

void launch_softmax_in_place_with_lengths(
    TensorFloat& qkt_output, const TensorInt& lengths);

void launch_softmax_v(
    const TensorFloat& softmax_result, const TensorFloat& v_cache, TensorFloat& attention_result,
    const TensorInt& lengths);

/**
 * - inp_embedding: [n_batch, n_sequence, input_dim], input_embedding
 * - lengths: [n_batch], the token lengths for each batch
 * - wk, wq, wv: [input_dim, input_dim]: since we don't have the following feed-forward layer, 
 * input_dim should be equal to output_dim
 * - new_batch_idx: [n_batch], but only n_new_items are used
 * - kt_cache: [n_batch, dim, n_sequence]
 * - v_cache: [n_batch, n_sequence, dim]
 * - q_output: [n_batch, input_dim]
 * - qkt_output: [n_batch, n_sequence]
 * - attention_result: [n_batch, output_dim]
 */
void inference_self_attention(
    const TensorFloat& inp_embedding, const TensorInt& lengths,
    const TensorFloat& wk,
    const TensorFloat& wq,
    const TensorFloat& wv,
    const TensorInt& new_batch_idx, TensorFloat& kt_cache, TensorFloat& v_cache,
    // avoid frequent creation of tensors
    TensorFloat& q_output, TensorFloat& qkt_output, TensorFloat& attention_result,
    int n_new_items);