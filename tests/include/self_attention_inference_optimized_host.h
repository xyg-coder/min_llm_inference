#pragma once

#include "tensor.hpp"

void fill_new_kt_v_cache(
    const TensorFloat& inp, const TensorInt& new_batch_idx, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wv, TensorFloat& kt_cache,
    TensorFloat& v_cache, int n_new_batch);

void get_latest_kt_q_v(
    const TensorFloat& inp, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wq,
    const TensorFloat& wv, TensorFloat& kt_cache,
    TensorFloat& v_cache, TensorFloat& q_output);

void qkt_host(
    const TensorFloat& q_output, const TensorFloat& kt_cache, const TensorInt& lengths,
    TensorFloat& qkt_output);

void softmax_in_place_with_lengths_host(
    TensorFloat& qkt_output, const TensorInt& lengths);

void softmax_v_host(
    const TensorFloat& softmax_result, const TensorFloat& v_cache, TensorFloat& attention_result,
    const TensorInt& lengths);

void self_attention_inference_host(const TensorFloat& inp, const TensorInt& lengths,
    const TensorFloat& wk,
    const TensorFloat& wq,
    const TensorFloat& wv,
    const TensorInt& new_batch_idx, TensorFloat& kt_cache, TensorFloat& v_cache,
    // avoid frequent creation of tensors
    TensorFloat& q_output, TensorFloat& qkt_output, TensorFloat& attention_result, int n_new_batch);