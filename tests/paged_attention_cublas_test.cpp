#include "include/test_utils.h"
#include <cublas_v2.h>
#include <gtest/gtest.h>
#include "constants.h"
#include "kernels/paged_attention.h"
#include "kernels/self_attention_inference_optimized.h"
#include "test_utils.h"
#include "kernels/utils.cuh"

TEST(PagedAttentionCublasTest, LatestKtQVTest) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    size_t n_batch = get_random_number(128, 256);
    size_t n_sequence = get_random_number(4, 16) * PAGE_BLOCK_SIZE;
    size_t emb_dim = get_random_number(128, 256) * 4;

    auto device_tensors = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim);
    auto q_output_to_compare = get_random_device_tensor({n_batch, emb_dim});
    q_output_to_compare.copy_from(device_tensors.q_output);
    auto latest_emb = get_random_device_tensor({n_batch, emb_dim});
    auto tmp_placeholder = get_random_device_tensor({n_batch, emb_dim});
    launch_get_latest_kt_q_v(
        device_tensors.inp_embedding,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wq,
        device_tensors.wv,
        device_tensors.kt_cache,
        device_tensors.v_cache,
        q_output_to_compare); 
    launch_get_latest_k_q_v_paged_attention_cublas(
        device_tensors.page_table, device_tensors.lengths,
        latest_emb,
        device_tensors.wk, device_tensors.wq, device_tensors.wv,
        device_tensors.q_output, tmp_placeholder, handle, n_sequence, q_output_to_compare);

    // assert_float_kernel_close(
    //     q_output_to_compare.data(), device_tensors.q_output.data(), q_output_to_compare.get_total_size());
    // assert_page_table_close(
    //     (const float**)device_tensors.page_table.data(), device_tensors.kt_cache.data(),
    //     device_tensors.lengths.data(), n_batch, n_sequence, K_CACHE_EMB_OFFSET, emb_dim);
    // assert_page_table_close(
    //     (const float**)device_tensors.page_table.data(), device_tensors.v_cache.data(),
    //     device_tensors.lengths.data(), n_batch, n_sequence, V_CACHE_EMB_OFFSET, emb_dim);
    cublasDestroy(handle); 
}
