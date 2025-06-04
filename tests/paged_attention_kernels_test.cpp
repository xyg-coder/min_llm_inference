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
