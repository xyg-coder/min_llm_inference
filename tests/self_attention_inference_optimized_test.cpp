#include "test_utils.h"
#include "kernels/self_attention_inference_optimized.h"
#include "self_attention_inference_optimized_host.h"
#include <gtest/gtest.h>

TEST(InferenceOptimizedSelfAttentionTest, FillNewKtVCache) {
    auto device_to_host_tensors = generate_device_and_host_tensors();
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
