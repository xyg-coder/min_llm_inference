#include "test_utils.h"
#include "kernels/self_attention_inference_optimized.h"
#include "self_attention_inference_optimized_host.h"
#include <gtest/gtest.h>

TEST(InferenceOptimizedSelfAttentionTest, FillNewKtVCache) {
    // use strange number to test
    auto device_to_host_tensors = generate_random_shape_device_and_host_tensors();
    TensorWrapForInferenceOptimizedSelfAttention device_tensors = device_to_host_tensors.first;
    TensorWrapForInferenceOptimizedSelfAttention host_tensors = device_to_host_tensors.second;
    launch_fill_new_kt_v_cache(
        device_tensors.inp,
        device_tensors.new_batch_idx,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wv,
        device_tensors.kt_cache,
        device_tensors.v_cache);
    fill_new_kt_v_cache(
        host_tensors.inp,
        host_tensors.new_batch_idx,
        host_tensors.lengths,
        host_tensors.wk,
        host_tensors.wv,
        host_tensors.kt_cache,
        host_tensors.v_cache);
    assert_near(device_tensors.kt_cache, host_tensors.kt_cache);
    assert_near(device_tensors.v_cache, host_tensors.v_cache);
}

TEST(InferenceOptimizedSelfAttentionTest, LatestKtQVTest) {
    auto device_to_host_tensors = generate_random_shape_device_and_host_tensors();
    TensorWrapForInferenceOptimizedSelfAttention device_tensors = device_to_host_tensors.first;
    TensorWrapForInferenceOptimizedSelfAttention host_tensors = device_to_host_tensors.second;

    launch_get_latest_kt_q_v(
        device_tensors.inp,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wq,
        device_tensors.wv,
        device_tensors.kt_cache,
        device_tensors.v_cache,
        device_tensors.q_output);

    get_latest_kt_q_v(
        host_tensors.inp,
        host_tensors.lengths,
        host_tensors.wk,
        host_tensors.wq,
        host_tensors.wv,
        host_tensors.kt_cache,
        host_tensors.v_cache,
        host_tensors.q_output);
    
    assert_near(device_tensors.kt_cache, host_tensors.kt_cache);
    assert_near(device_tensors.v_cache, host_tensors.v_cache);
    assert_near(device_tensors.q_output, host_tensors.q_output);
}

TEST(InferenceOptimizedSelfAttentionTest, QKtTest) {
    auto device_to_host_tensors = generate_random_shape_device_and_host_tensors();
    TensorWrapForInferenceOptimizedSelfAttention device_tensors = device_to_host_tensors.first;
    TensorWrapForInferenceOptimizedSelfAttention host_tensors = device_to_host_tensors.second;

    launch_qkt(
        device_tensors.q_output,
        device_tensors.kt_cache,
        device_tensors.lengths,
        device_tensors.qkt_output);

    qkt_host(
        host_tensors.q_output,
        host_tensors.kt_cache,
        host_tensors.lengths,
        host_tensors.qkt_output);
    
    assert_near(device_tensors.qkt_output, host_tensors.qkt_output);
}

TEST(InferenceOptimizedSelfAttentionTest, SoftmaxInPlaceWithLengthsTest) {
    auto device_to_host_tensors = generate_random_shape_device_and_host_tensors();
    TensorWrapForInferenceOptimizedSelfAttention device_tensors = device_to_host_tensors.first;
    TensorWrapForInferenceOptimizedSelfAttention host_tensors = device_to_host_tensors.second;

    launch_softmax_in_place_with_lengths(
        device_tensors.qkt_output, device_tensors.lengths);

    softmax_in_place_with_lengths_host(
        host_tensors.qkt_output, host_tensors.lengths);
    
    assert_near(device_tensors.qkt_output, host_tensors.qkt_output);
}


TEST(InferenceOptimizedSelfAttentionTest, SoftmaxVTest) {
    auto device_to_host_tensors = generate_random_shape_device_and_host_tensors();
    TensorWrapForInferenceOptimizedSelfAttention device_tensors = device_to_host_tensors.first;
    TensorWrapForInferenceOptimizedSelfAttention host_tensors = device_to_host_tensors.second;

    launch_softmax_v(
        device_tensors.qkt_output,
        device_tensors.v_cache,
        device_tensors.attention_result,
        device_tensors.lengths);

    softmax_v_host(
        host_tensors.qkt_output,
        host_tensors.v_cache,
        host_tensors.attention_result,
        host_tensors.lengths);
    
    assert_near(device_tensors.attention_result, host_tensors.attention_result);
}

TEST(InferenceOptimizedSelfAttentionTest, InferenceOptimizedSelfAttentionTest) {
    auto device_to_host_tensors = generate_random_shape_device_and_host_tensors();

    TensorWrapForInferenceOptimizedSelfAttention device_tensors = device_to_host_tensors.first;
    TensorWrapForInferenceOptimizedSelfAttention host_tensors = device_to_host_tensors.second;
    inference_self_attention(
        device_tensors.inp,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wq,
        device_tensors.wv,
        device_tensors.new_batch_idx,
        device_tensors.kt_cache,
        device_tensors.v_cache,
        device_tensors.q_output,
        device_tensors.qkt_output,
        device_tensors.attention_result);

    self_attention_inference_host(
        host_tensors.inp,
        host_tensors.lengths,
        host_tensors.wk,
        host_tensors.wq,
        host_tensors.wv,
        host_tensors.new_batch_idx,
        host_tensors.kt_cache,
        host_tensors.v_cache,
        host_tensors.q_output,
        host_tensors.qkt_output,
        host_tensors.attention_result);
    
    assert_near(device_tensors.attention_result, host_tensors.attention_result);
}

TEST(InferenceOptimizedSelfAttentionTest, InferenceOptimizedSelfAttentionZeroLengthTest) {
    // TO FILL
    ASSERT_FALSE(true);
}
