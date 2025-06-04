#include "constants.h"
#include "kernels/self_attention_inference_optimized.h"
#include "kernels/paged_attention.h"
#include "kernels/utils.cuh"
#include "test_utils.h"
#include <gtest/gtest.h>

TEST(PagedAttentionKernelTest, FillNewKtVCache) {
    size_t n_batch = get_random_number(128, 256);
    size_t n_sequence = get_random_number(4, 16) * PAGE_BLOCK_SIZE;
    size_t emb_dim = get_random_number(128, 256) * 4;
    auto device_tensors = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim);

    launch_fill_new_kt_v_cache(
        device_tensors.inp,
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
