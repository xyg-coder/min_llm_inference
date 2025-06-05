#include "constants.h"
#include "kernels/self_attention_inference_optimized.h"
#include "kernels/paged_attention.h"
#include "kernels/utils.cuh"
#include "tensor.hpp"
#include "test_utils.h"
#include <gtest/gtest.h>

TEST(PagedAttentionKernelTest, FillNewKtVCache) {
    size_t n_batch = get_random_number(128, 256);
    size_t n_sequence = get_random_number(4, 16) * PAGE_BLOCK_SIZE;
    size_t emb_dim = get_random_number(128, 256) * 4;
    auto device_tensors = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim);

    launch_fill_new_kt_v_cache(
        device_tensors.inp_embedding,
        device_tensors.new_batch_idx,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wv,
        device_tensors.kt_cache,
        device_tensors.v_cache, device_tensors.n_new_batches);

    launch_fill_new_k_v_cache_paged_attention(
        device_tensors.page_table,
        device_tensors.new_batch_idx,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wv, device_tensors.n_new_batches, n_sequence);

    assert_page_table_close(
        (const float**)device_tensors.page_table.data(), device_tensors.kt_cache.data(),
        device_tensors.lengths.data(), n_batch, n_sequence, K_CACHE_EMB_OFFSET, emb_dim);
    assert_page_table_close(
        (const float**)device_tensors.page_table.data(), device_tensors.v_cache.data(),
        device_tensors.lengths.data(), n_batch, n_sequence, V_CACHE_EMB_OFFSET, emb_dim);
}

TEST(PagedAttentionKernelTest, LatestKtQVTest) {
    size_t n_batch = get_random_number(128, 256);
    size_t n_sequence = get_random_number(4, 16) * PAGE_BLOCK_SIZE;
    size_t emb_dim = get_random_number(128, 256) * 4;

    auto device_tensors = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim);
    auto q_output_to_compare = get_random_device_tensor({n_batch, emb_dim});
    q_output_to_compare.copy_from(device_tensors.q_output);
    launch_get_latest_kt_q_v(
        device_tensors.inp_embedding,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wq,
        device_tensors.wv,
        device_tensors.kt_cache,
        device_tensors.v_cache,
        q_output_to_compare); 
    launch_get_latest_k_q_v_paged_attention(
        device_tensors.page_table, device_tensors.lengths,
        device_tensors.wk, device_tensors.wq, device_tensors.wv,
        device_tensors.q_output, n_sequence);

    assert_float_kernel_close(
        q_output_to_compare.data(), device_tensors.q_output.data(), q_output_to_compare.get_total_size());
    assert_page_table_close(
        (const float**)device_tensors.page_table.data(), device_tensors.kt_cache.data(),
        device_tensors.lengths.data(), n_batch, n_sequence, K_CACHE_EMB_OFFSET, emb_dim);
    assert_page_table_close(
        (const float**)device_tensors.page_table.data(), device_tensors.v_cache.data(),
        device_tensors.lengths.data(), n_batch, n_sequence, V_CACHE_EMB_OFFSET, emb_dim);
}

TEST(PagedAttentionKernelTest, QKtTest) {
    size_t n_batch = get_random_number(128, 256);
    size_t n_sequence = get_random_number(4, 16) * PAGE_BLOCK_SIZE;
    size_t emb_dim = get_random_number(128, 256) * 4;

    auto device_tensors = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim);
    auto qkt_to_compare = get_random_device_tensor({n_batch, n_sequence});
    qkt_to_compare.copy_from(device_tensors.qkt_output);
    launch_qkt(
        device_tensors.q_output,
        device_tensors.kt_cache,
        device_tensors.lengths,
        qkt_to_compare);
    launch_qkt_paged_attention(
        device_tensors.q_output, device_tensors.page_table, device_tensors.lengths, device_tensors.qkt_output);
    assert_float_kernel_close(
        qkt_to_compare.data(), device_tensors.qkt_output.data(), qkt_to_compare.get_total_size());
}

TEST(PagedAttentionKernelTest, SoftmaxVTest) {
    size_t n_batch = get_random_number(128, 256);
    size_t n_sequence = get_random_number(4, 16) * PAGE_BLOCK_SIZE;
    size_t emb_dim = get_random_number(128, 256) * 4;

    auto device_tensors = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim);
    auto attention_result_to_compare = get_random_device_tensor({n_batch, emb_dim});
    attention_result_to_compare.copy_from(device_tensors.attention_result);
    launch_softmax_v(
        device_tensors.qkt_output,
        device_tensors.v_cache,
        attention_result_to_compare,
        device_tensors.lengths);
    launch_softmax_v_paged_attention(
       device_tensors.qkt_output,
       device_tensors.page_table,
       device_tensors.attention_result, device_tensors.lengths);
    assert_float_kernel_close(
        attention_result_to_compare.data(), device_tensors.attention_result.data(), attention_result_to_compare.get_total_size());
}

TEST(PagedAttentionKernelTest, InferenceOptimizedSelfAttentionTest) {
    size_t n_batch = get_random_number(128, 256);
    size_t n_sequence = get_random_number(4, 16) * PAGE_BLOCK_SIZE;
    size_t emb_dim = get_random_number(128, 256) * 4;

    auto device_tensors = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim);

    auto q_output_to_compare = get_random_device_tensor({n_batch, emb_dim});
    q_output_to_compare.copy_from(device_tensors.q_output);

    auto qkt_to_compare = get_random_device_tensor({n_batch, n_sequence});
    qkt_to_compare.copy_from(device_tensors.qkt_output);

    auto attention_result_to_compare = get_random_device_tensor({n_batch, emb_dim});
    attention_result_to_compare.copy_from(device_tensors.attention_result);

    inference_self_attention(
        device_tensors.inp_embedding,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wq,
        device_tensors.wv,
        device_tensors.new_batch_idx,
        device_tensors.kt_cache,
        device_tensors.v_cache,
        q_output_to_compare,
        qkt_to_compare,
        attention_result_to_compare, device_tensors.n_new_batches);

    paged_attention(
        device_tensors.page_table,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wq,
        device_tensors.wv,
        device_tensors.new_batch_idx,
        device_tensors.q_output,
        device_tensors.qkt_output,
        device_tensors.attention_result,
        device_tensors.n_new_batches, n_sequence);
    
    assert_float_kernel_close(
        q_output_to_compare.data(), device_tensors.q_output.data(), q_output_to_compare.get_total_size());
    assert_page_table_close(
        (const float**)device_tensors.page_table.data(), device_tensors.kt_cache.data(),
        device_tensors.lengths.data(), n_batch, n_sequence, K_CACHE_EMB_OFFSET, emb_dim);
    assert_page_table_close(
        (const float**)device_tensors.page_table.data(), device_tensors.v_cache.data(),
        device_tensors.lengths.data(), n_batch, n_sequence, V_CACHE_EMB_OFFSET, emb_dim);
    assert_float_kernel_close(
        qkt_to_compare.data(), device_tensors.qkt_output.data(), qkt_to_compare.get_total_size());
    assert_float_kernel_close(
        attention_result_to_compare.data(), device_tensors.attention_result.data(), attention_result_to_compare.get_total_size());
}

TEST(PagedAttentionKernelTest, InferenceOptimizedSelfAttentionZeroLengthTest) {
    size_t n_batch = get_random_number(128, 256);
    size_t n_sequence = get_random_number(4, 16) * PAGE_BLOCK_SIZE;
    size_t emb_dim = get_random_number(128, 256) * 4;

    auto device_tensors = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim);

    auto q_output_to_compare = get_random_device_tensor({n_batch, emb_dim});
    q_output_to_compare.copy_from(device_tensors.q_output);

    auto qkt_to_compare = get_random_device_tensor({n_batch, n_sequence});
    qkt_to_compare.copy_from(device_tensors.qkt_output);

    auto attention_result_to_compare = get_random_device_tensor({n_batch, emb_dim});
    attention_result_to_compare.copy_from(device_tensors.attention_result);

    TensorInt lengths_host({n_batch}, DeviceType::HOST);
    lengths_host.copy_from(device_tensors.lengths);
    int* host_lengths_data = lengths_host.data();
    for (int i = 0; i < n_batch; i += 5) {
        host_lengths_data[i] = 0;
    }
    device_tensors.lengths.copy_from(lengths_host);

    inference_self_attention(
        device_tensors.inp_embedding,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wq,
        device_tensors.wv,
        device_tensors.new_batch_idx,
        device_tensors.kt_cache,
        device_tensors.v_cache,
        q_output_to_compare,
        qkt_to_compare,
        attention_result_to_compare, device_tensors.n_new_batches);

    paged_attention(
        device_tensors.page_table,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wq,
        device_tensors.wv,
        device_tensors.new_batch_idx,
        device_tensors.q_output,
        device_tensors.qkt_output,
        device_tensors.attention_result,
        device_tensors.n_new_batches, n_sequence);
    
    assert_float_kernel_close(
        q_output_to_compare.data(), device_tensors.q_output.data(), q_output_to_compare.get_total_size());
    assert_page_table_close(
        (const float**)device_tensors.page_table.data(), device_tensors.kt_cache.data(),
        device_tensors.lengths.data(), n_batch, n_sequence, K_CACHE_EMB_OFFSET, emb_dim);
    assert_page_table_close(
        (const float**)device_tensors.page_table.data(), device_tensors.v_cache.data(),
        device_tensors.lengths.data(), n_batch, n_sequence, V_CACHE_EMB_OFFSET, emb_dim);
    assert_float_kernel_close(
        qkt_to_compare.data(), device_tensors.qkt_output.data(), qkt_to_compare.get_total_size());
    assert_float_kernel_close(
        attention_result_to_compare.data(), device_tensors.attention_result.data(), attention_result_to_compare.get_total_size());
}