#include <cstddef>
#include <gtest/gtest.h>
#include "test_utils.h"
#include "kernels/utils.cuh"
#include "kernels/self_attention_inference_optimized.h"
#include "kernels/templated_kernels.cuh"
#include "kernels/paged_attention.h"
#include "constants.h"

TEST(WarpTilingTest, WarpTilingFunctionality) {
    size_t n_batch = get_random_number(128, 256);
    size_t n_sequence = get_random_number(16, 32) * PAGE_BLOCK_SIZE;
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

    launch_fill_new_k_v_cache_paged_attention_warp_tiling(
        device_tensors.page_table,
        device_tensors.new_batch_idx,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wv, device_tensors.n_new_batches, n_sequence);

    // TODO: do we change to lengths considered assertion? Maybe not
    assert_page_table_close(
        (const float**)device_tensors.page_table.data(), device_tensors.kt_cache.data(),
        device_tensors.lengths.data(), n_batch, n_sequence, K_CACHE_EMB_OFFSET, emb_dim);
    assert_page_table_close(
        (const float**)device_tensors.page_table.data(), device_tensors.v_cache.data(),
        device_tensors.lengths.data(), n_batch, n_sequence, V_CACHE_EMB_OFFSET, emb_dim);
}
