#pragma once

#include "tensor.hpp"

void launch_fill_new_kt_v_cache(
    const TensorFloat& inp, const TensorInt& new_batch_idx, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wv, TensorFloat& kt_cache,
    TensorFloat& v_cache);


void launch_get_latest_kt_q_v(
    const TensorFloat& inp, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wq,
    const TensorFloat& wv, TensorFloat& kt_cache,
    TensorFloat& v_cache, TensorFloat& q_output);

void launch_qkt(
    const TensorFloat& q_output, const TensorFloat& kt_cache, const TensorInt& lengths,
    TensorFloat& qkt_output);

void launch_softmax_in_place_with_lengths(
    TensorFloat& qkt_output, const TensorInt& lengths);

void launch_softmax_v(
    const TensorFloat& qkt_output, const TensorFloat& v_cache, TensorFloat& attention_result,
    const TensorInt& lengths);
