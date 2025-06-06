#include "constants.h"
#include "include/test_utils.h"
#include "kernels/encoder.h"
#include "kernels/utils.cuh"
#include "tensor.hpp"
#include "test_utils.h"
#include <algorithm>
#include <gtest/gtest.h>

TEST(EncoderTest, EncoderTest) {
    size_t n_batch = 128;
    size_t n_sequence = 512;
    size_t max_sequence = 512;
    size_t n_vocab = 2048;
    size_t embedding_dim = 512;

    auto wte_device_host = get_random_device_host_tensor({n_vocab, embedding_dim});
    auto wpe_device_host = get_random_device_host_tensor({max_sequence, embedding_dim});
    auto inp_device_host = get_random_device_host_tensor_int({n_batch, n_sequence}, n_vocab - 1);

    TensorFloat wte_device = wte_device_host.first;
    TensorFloat wpe_device = wpe_device_host.first;
    TensorInt inp_device = inp_device_host.first;
    TensorFloat output_device({n_batch, n_sequence, embedding_dim}, DeviceType::DEVICE);

    launch_encoder_kernel(
        wte_device.data(), wpe_device.data(), inp_device.data(),
        output_device.data(), n_batch, n_sequence, embedding_dim);

    TensorFloat output_host = encoder_host(wte_device_host.second, wpe_device_host.second, inp_device_host.second, n_batch, n_sequence, embedding_dim);

    assert_near(output_device, output_host);
}

TEST(EncoderTest, EncoderInferenceOptimizedTest) {
    size_t n_vocab = get_random_number(1024, 3096);
    size_t embedding_dim = get_random_number(128, 128 * 3) / 4 * 4;
    size_t n_sequence = get_random_number(1024, 3096);
    size_t n_batch = get_random_number(128, 128 * 3);
    size_t n_new_items = get_random_number(1, n_batch);
    
    auto emb_table_device_host = get_random_device_host_tensor({n_vocab, embedding_dim});
    auto wpe_table_device_host = get_random_device_host_tensor({n_sequence, embedding_dim});
    auto inp_device_host = get_random_device_host_tensor_int({n_batch, n_sequence}, n_vocab - 1);
    auto output_device_host = get_random_device_host_tensor({n_batch, n_sequence, embedding_dim});
    auto lengths_device_host = get_random_device_host_tensor_int({n_batch}, n_sequence);
    auto new_item_indices_device_host = get_random_device_host_tensor_int({n_new_items}, 100);

    std::vector<int> new_item_indices_number = get_unique_num_array(0, n_batch - 1, n_new_items);
    std::copy(
        new_item_indices_number.begin(), new_item_indices_number.end(), new_item_indices_device_host.second.data());
    new_item_indices_device_host.first.copy_from(new_item_indices_device_host.second);

    launch_inference_optimized_encoder_kernel(
        emb_table_device_host.first.data(),
        wpe_table_device_host.first.data(),
        inp_device_host.first.data(),
        output_device_host.first.data(),
        lengths_device_host.first.data(),
        new_item_indices_device_host.first.data(),
        n_batch, n_sequence, embedding_dim, n_new_items);

    inference_optimized_encoder_host(
        emb_table_device_host.second.data(),
        wpe_table_device_host.second.data(),
        inp_device_host.second.data(),
        output_device_host.second.data(),
        lengths_device_host.second.data(),
        new_item_indices_device_host.second.data(),
        n_batch, n_sequence, embedding_dim, n_new_items);
    
    assert_near(output_device_host.first, output_device_host.second);
}

TEST(EncoderTest, PagedAttentionEncoderTest) {
    size_t n_vocab = get_random_number(1024, 3096);
    size_t emb_dim = get_random_number(128, 128 * 3) / 4 * 4;
    size_t n_sequence = get_random_number(1024, 3096) / PAGE_BLOCK_SIZE * PAGE_BLOCK_SIZE;
    size_t n_batch = get_random_number(128, 128 * 3);
    auto emb_table_device = get_random_device_tensor({n_vocab, emb_dim});
    auto wpe_table_device = get_random_device_tensor({n_sequence, emb_dim});
    auto inp_device = get_random_device_tensor_int({n_batch, n_sequence}, n_vocab - 1);

    TensorWrapperForPagedAttention wrapper = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim
    );

    launch_inference_optimized_encoder_kernel(
        emb_table_device.data(),
        wpe_table_device.data(),
        inp_device.data(),
        wrapper.inp_embedding.data(),
        wrapper.lengths.data(),
        wrapper.new_batch_idx.data(),
        n_batch, n_sequence, emb_dim, wrapper.n_new_batches); 

    launch_paged_attention_encoder_kernel(
        emb_table_device.data(),
        wpe_table_device.data(),
        inp_device.data(),
        wrapper.page_table.data(),
        wrapper.lengths.data(),
        wrapper.new_batch_idx.data(),
        n_batch, n_sequence, emb_dim, wrapper.n_new_batches);

    assert_page_table_close(
        (const float**)wrapper.page_table.data(), wrapper.inp_embedding.data(),
        wrapper.lengths.data(), n_batch, n_sequence, INP_EMB_EMB_OFFSET, emb_dim);
}
