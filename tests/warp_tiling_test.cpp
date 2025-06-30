#include <cstddef>
#include <gtest/gtest.h>
#include "test_utils.h"
#include "kernels/utils.cuh"
#include "kernels/self_attention_inference_optimized.h"
#include "kernels/templated_kernels.cuh"
#include "kernels/paged_attention.h"
#include "constants.h"
#include <chrono>
#include <iostream>

TEST(WarpTilingTest, WarpTilingFunctionality) {
    size_t n_batch = get_random_number(128, 512);
    size_t n_sequence = get_random_number(16, 32) * PAGE_BLOCK_SIZE;
    size_t emb_dim = get_random_number(128, 256) * 4;
    auto device_tensors = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim);

    // First, run the reference implementation 
    launch_fill_new_kt_v_cache(
        device_tensors.inp_embedding,
        device_tensors.new_batch_idx,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wv,
        device_tensors.kt_cache,
        device_tensors.v_cache, device_tensors.n_new_batches);

    // Then run the warp tiling implementation
    launch_fill_new_k_v_cache_paged_attention_warp_tiling(
        device_tensors.page_table,
        device_tensors.new_batch_idx,
        device_tensors.lengths,
        device_tensors.wk,
        device_tensors.wv, device_tensors.n_new_batches, n_sequence);

    // Verify correctness against reference
    assert_page_table_close(
        (const float**)device_tensors.page_table.data(), device_tensors.kt_cache.data(),
        device_tensors.lengths.data(), n_batch, n_sequence, K_CACHE_EMB_OFFSET, emb_dim);
    assert_page_table_close(
        (const float**)device_tensors.page_table.data(), device_tensors.v_cache.data(),
        device_tensors.lengths.data(), n_batch, n_sequence, V_CACHE_EMB_OFFSET, emb_dim);
}


TEST(WarpTilingTest, CompareTimeWithPagedAttention) {
    // Use larger problem sizes for meaningful timing comparison
    size_t n_batch = 512;
    size_t n_sequence = 32 * PAGE_BLOCK_SIZE; // 512 sequence length
    size_t emb_dim = 512; // Must be multiple of 4
    
    auto device_tensors = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim);

    // Warm-up runs to avoid cold start overhead
    for (int i = 0; i < 3; i++) {
        launch_fill_new_k_v_cache_paged_attention(
            device_tensors.page_table,
            device_tensors.new_batch_idx,
            device_tensors.lengths,
            device_tensors.wk,
            device_tensors.wv, device_tensors.n_new_batches, n_sequence);
        cudaDeviceSynchronize();
        
        launch_fill_new_k_v_cache_paged_attention_warp_tiling(
            device_tensors.page_table,
            device_tensors.new_batch_idx,
            device_tensors.lengths,
            device_tensors.wk,
            device_tensors.wv, device_tensors.n_new_batches, n_sequence);
        cudaDeviceSynchronize();
    }

    const int num_iterations = 10;
    
    // Time standard paged attention
    cudaEvent_t start_standard, stop_standard;
    cudaEventCreate(&start_standard);
    cudaEventCreate(&stop_standard);
    
    cudaEventRecord(start_standard);
    for (int i = 0; i < num_iterations; i++) {
        launch_fill_new_k_v_cache_paged_attention(
            device_tensors.page_table,
            device_tensors.new_batch_idx,
            device_tensors.lengths,
            device_tensors.wk,
            device_tensors.wv, device_tensors.n_new_batches, n_sequence);
    }
    cudaEventRecord(stop_standard);
    cudaDeviceSynchronize(); // Critical for accurate timing
    
    float standard_time;
    cudaEventElapsedTime(&standard_time, start_standard, stop_standard);
    
    // Time warp tiling version
    cudaEvent_t start_warp, stop_warp;
    cudaEventCreate(&start_warp);
    cudaEventCreate(&stop_warp);
    
    cudaEventRecord(start_warp);
    for (int i = 0; i < num_iterations; i++) {
        launch_fill_new_k_v_cache_paged_attention_warp_tiling(
            device_tensors.page_table,
            device_tensors.new_batch_idx,
            device_tensors.lengths,
            device_tensors.wk,
            device_tensors.wv, device_tensors.n_new_batches, n_sequence);
    }
    cudaEventRecord(stop_warp);
    cudaDeviceSynchronize(); // Critical for accurate timing
    
    float warp_time;
    cudaEventElapsedTime(&warp_time, start_warp, stop_warp);
    
    // Print timing results
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "Problem size: [" << n_batch << ", " << n_sequence << ", " << emb_dim << "]" << std::endl;
    std::cout << "Standard paged attention: " << standard_time / num_iterations << " ms/iteration" << std::endl;
    std::cout << "Warp tiling version: " << warp_time / num_iterations << " ms/iteration" << std::endl;
    std::cout << "Speedup: " << (standard_time / warp_time) << "x" << std::endl;
    std::cout << "============================\n" << std::endl;
    
    // Clean up CUDA events
    cudaEventDestroy(start_standard);
    cudaEventDestroy(stop_standard);
    cudaEventDestroy(start_warp);
    cudaEventDestroy(stop_warp);
    
    // The test passes if both implementations complete without error
    // Performance comparison is informational
    EXPECT_GT(standard_time, 0.0f);
    EXPECT_GT(warp_time, 0.0f);
}
