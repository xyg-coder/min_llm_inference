#include "constants.h"
#include "include/test_utils.h"
#include "kernels/decoder.h"
#include "kernels/utils.cuh"
#include <gtest/gtest.h>
#include <vector>

TEST(DecoderTest, DecoderKernelTest) {
    size_t n_batch = get_random_number(128, 128 * 3);
    size_t embedding_dim = get_random_number(128, 128 * 3) / 4 * 4;
    size_t n_sequence = get_random_number(1024, 3096);
    size_t n_vocab = get_random_number(1024, 3096);

    auto emb_table_device_host = get_random_device_host_tensor({n_vocab, embedding_dim});
    auto batch_emb_device_host = get_random_device_host_tensor({n_batch, embedding_dim});
    auto emb_score_device_host = get_random_device_host_tensor({n_batch, n_vocab});
    auto wpe_table_device_host = get_random_device_host_tensor({n_sequence, embedding_dim});
    auto inp_table_device_host = get_random_device_host_tensor({n_batch, n_sequence, embedding_dim});
    auto lengths_device_host = get_random_device_host_tensor_int({n_batch}, n_sequence - 1);
    auto decoder_result_device_host = get_random_device_host_tensor_int({n_batch}, 100);

    launch_decoder(
        batch_emb_device_host.first,
        emb_table_device_host.first,
        emb_score_device_host.first,
        wpe_table_device_host.first,
        inp_table_device_host.first,
        lengths_device_host.first,
        decoder_result_device_host.first);
    
    decoder_host(
        batch_emb_device_host.second,
        emb_table_device_host.second,
        emb_score_device_host.second,
        wpe_table_device_host.second,
        inp_table_device_host.second,
        lengths_device_host.second,
        decoder_result_device_host.second);
    
    assert_near(emb_score_device_host.first, emb_score_device_host.second);
    assert_near(inp_table_device_host.first, inp_table_device_host.second);
    assert_near(lengths_device_host.first, lengths_device_host.second);
    assert_near(decoder_result_device_host.first, decoder_result_device_host.second);
}

TEST(DecoderTest, MaxLengthTest) {
    size_t n_batch = get_random_number(128, 128 * 3);
    size_t embedding_dim = get_random_number(128, 128 * 3) / 4 * 4;
    size_t n_sequence = get_random_number(1024, 3096);
    size_t n_vocab = get_random_number(1024, 3096);

    auto emb_table_device_host = get_random_device_host_tensor({n_vocab, embedding_dim});
    auto batch_emb_device_host = get_random_device_host_tensor({n_batch, embedding_dim});
    auto emb_score_device_host = get_random_device_host_tensor({n_batch, n_vocab});
    auto wpe_table_device_host = get_random_device_host_tensor({n_sequence, embedding_dim});
    auto inp_table_device_host = get_random_device_host_tensor({n_batch, n_sequence, embedding_dim});
    auto lengths_device_host = get_random_device_host_tensor_int({n_batch}, n_sequence - 1);
    auto decoder_result_device_host = get_random_device_host_tensor_int({n_batch}, 100);

    auto zero_and_max_length_indices = get_unique_num_array(0, n_batch - 1, get_random_number(2, n_batch - 3));
    int num_max_length = get_random_number(1, zero_and_max_length_indices.size() - 1);
    int* lengths_data_host = lengths_device_host.second.data();
    for (int i = 0; i < zero_and_max_length_indices.size(); ++i) {
        if (i < num_max_length) {
            lengths_data_host[zero_and_max_length_indices[i]] = n_sequence - 1;
        } else {
            lengths_data_host[zero_and_max_length_indices[i]] = 0;
        }
    }
    lengths_device_host.first.copy_from(lengths_device_host.second);

    launch_decoder(
        batch_emb_device_host.first,
        emb_table_device_host.first,
        emb_score_device_host.first,
        wpe_table_device_host.first,
        inp_table_device_host.first,
        lengths_device_host.first,
        decoder_result_device_host.first);
    
    decoder_host(
        batch_emb_device_host.second,
        emb_table_device_host.second,
        emb_score_device_host.second,
        wpe_table_device_host.second,
        inp_table_device_host.second,
        lengths_device_host.second,
        decoder_result_device_host.second);
    
    assert_near(emb_score_device_host.first, emb_score_device_host.second);
    assert_near(inp_table_device_host.first, inp_table_device_host.second);
    assert_near(lengths_device_host.first, lengths_device_host.second);
    assert_near(decoder_result_device_host.first, decoder_result_device_host.second);
}

TEST(DecoderTest, PagedAttentionDecoderKernelTest) {
    size_t n_batch = get_random_number(128, 128 * 3);
    size_t emb_dim = get_random_number(128, 128 * 3) / 4 * 4;
    size_t n_sequence = get_random_number(1024, 3096) / PAGE_BLOCK_SIZE * PAGE_BLOCK_SIZE;
    size_t n_vocab = get_random_number(1024, 3096);

    auto emb_table_device = get_random_device_tensor({n_vocab, emb_dim});
    auto wpe_table_device = get_random_device_tensor({n_sequence, emb_dim});

    TensorWrapperForPagedAttention wrapper = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim
    );

    auto batch_emb_to_compare = get_random_device_tensor({n_batch, emb_dim});
    auto paged_attention_batch_emb = get_random_device_tensor({n_batch, emb_dim});
    paged_attention_batch_emb.copy_from(batch_emb_to_compare);

    auto emb_score_to_compare = get_random_device_tensor({n_batch, n_vocab});
    auto paged_attention_emb_score = get_random_device_tensor({n_batch, n_vocab});
    paged_attention_emb_score.copy_from(emb_score_to_compare);

    auto decoder_result_to_compare = get_random_device_tensor_int({n_batch}, 100);
    auto paged_attention_decoder_result = get_random_device_tensor_int({n_batch}, 100);
    paged_attention_decoder_result.copy_from(decoder_result_to_compare);

    auto lengths_to_compare = get_random_device_tensor_int({n_batch}, n_sequence - 1);
    lengths_to_compare.copy_from(wrapper.lengths);

    launch_decoder(
        batch_emb_to_compare,
        emb_table_device,
        emb_score_to_compare,
        wpe_table_device,
        wrapper.inp_embedding,
        lengths_to_compare,
        decoder_result_to_compare);
    
    launch_paged_attention_decoder_multi_rounds(
        paged_attention_batch_emb,
        emb_table_device,
        paged_attention_emb_score,
        wpe_table_device,
        wrapper.page_table,
        wrapper.lengths,
        paged_attention_decoder_result, 0);
    
    assert_page_table_close(
        (const float**)wrapper.page_table.data(), wrapper.inp_embedding.data(),
        wrapper.lengths.data(), n_batch, n_sequence, INP_EMB_EMB_OFFSET, emb_dim);
    assert_float_kernel_close(emb_score_to_compare.data(), paged_attention_emb_score.data(), emb_score_to_compare.get_total_size());
    assert_int_kernel_close(lengths_to_compare.data(), wrapper.lengths.data(), lengths_to_compare.get_total_size());
    assert_int_kernel_close(decoder_result_to_compare.data(), paged_attention_decoder_result.data(), decoder_result_to_compare.get_total_size());
}

TEST(DecoderTest, PagedAttentionMaxLengthTest) {
    size_t n_batch = get_random_number(128, 128 * 3);
    size_t emb_dim = get_random_number(128, 128 * 3) / 4 * 4;
    size_t n_sequence = get_random_number(1024, 3096) / PAGE_BLOCK_SIZE * PAGE_BLOCK_SIZE;
    size_t n_vocab = get_random_number(1024, 3096);

    auto emb_table_device = get_random_device_tensor({n_vocab, emb_dim});
    auto wpe_table_device = get_random_device_tensor({n_sequence, emb_dim});

    auto lengths_to_compare_device_host = get_random_device_host_tensor_int({n_batch}, n_sequence - 1);

    auto zero_and_max_length_indices = get_unique_num_array(0, n_batch - 1, get_random_number(2, n_batch - 3));
    int num_max_length = get_random_number(1, zero_and_max_length_indices.size() - 1);
    int* lengths_data_host = lengths_to_compare_device_host.second.data();
    for (int i = 0; i < zero_and_max_length_indices.size(); ++i) {
        if (i < num_max_length) {
            lengths_data_host[zero_and_max_length_indices[i]] = n_sequence - 1;
        } else {
            lengths_data_host[zero_and_max_length_indices[i]] = 0;
        }
    }

    lengths_to_compare_device_host.first.copy_from(lengths_to_compare_device_host.second);
    TensorWrapperForPagedAttention wrapper = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim, lengths_to_compare_device_host.first);

    auto batch_emb_to_compare = get_random_device_tensor({n_batch, emb_dim});
    auto paged_attention_batch_emb = get_random_device_tensor({n_batch, emb_dim});
    paged_attention_batch_emb.copy_from(batch_emb_to_compare);

    auto emb_score_to_compare = get_random_device_tensor({n_batch, n_vocab});
    auto paged_attention_emb_score = get_random_device_tensor({n_batch, n_vocab});
    paged_attention_emb_score.copy_from(emb_score_to_compare);

    auto decoder_result_to_compare = get_random_device_tensor_int({n_batch}, 100);
    auto paged_attention_decoder_result = get_random_device_tensor_int({n_batch}, 100);
    paged_attention_decoder_result.copy_from(decoder_result_to_compare);


    launch_decoder(
        batch_emb_to_compare,
        emb_table_device,
        emb_score_to_compare,
        wpe_table_device,
        wrapper.inp_embedding,
        lengths_to_compare_device_host.first,
        decoder_result_to_compare);
    
    launch_paged_attention_decoder_multi_rounds(
        paged_attention_batch_emb,
        emb_table_device,
        paged_attention_emb_score,
        wpe_table_device,
        wrapper.page_table,
        wrapper.lengths,
        paged_attention_decoder_result, 0);
    
    assert_page_table_close(
        (const float**)wrapper.page_table.data(), wrapper.inp_embedding.data(),
        wrapper.lengths.data(), n_batch, n_sequence, INP_EMB_EMB_OFFSET, emb_dim);
    assert_float_kernel_close(emb_score_to_compare.data(), paged_attention_emb_score.data(), emb_score_to_compare.get_total_size());
    assert_int_kernel_close(lengths_to_compare_device_host.first.data(), wrapper.lengths.data(),
        lengths_to_compare_device_host.first.get_total_size());
    assert_int_kernel_close(decoder_result_to_compare.data(), paged_attention_decoder_result.data(), decoder_result_to_compare.get_total_size());
}


TEST(DecoderTest, PagedAttentionCublasDecoderKernelTest) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    size_t n_batch = get_random_number(128, 128 * 3);
    size_t emb_dim = get_random_number(128, 128 * 3) / 4 * 4;
    size_t n_sequence = get_random_number(1024, 3096) / PAGE_BLOCK_SIZE * PAGE_BLOCK_SIZE;
    size_t n_vocab = get_random_number(1024, 3096);

    auto emb_table_device = get_random_device_tensor({n_vocab, emb_dim});
    auto wpe_table_device = get_random_device_tensor({n_sequence, emb_dim});

    TensorWrapperForPagedAttention wrapper = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim
    );

    auto batch_emb_to_compare = get_random_device_tensor({n_batch, emb_dim});
    auto paged_attention_batch_emb = get_random_device_tensor({n_batch, emb_dim});
    paged_attention_batch_emb.copy_from(batch_emb_to_compare);

    auto emb_score_to_compare = get_random_device_tensor({n_batch, n_vocab});
    auto paged_attention_emb_score = get_random_device_tensor({n_batch, n_vocab});
    paged_attention_emb_score.copy_from(emb_score_to_compare);

    auto decoder_result_to_compare = get_random_device_tensor_int({n_batch}, 100);
    auto paged_attention_decoder_result = get_random_device_tensor_int({n_batch}, 100);
    paged_attention_decoder_result.copy_from(decoder_result_to_compare);

    auto lengths_to_compare = get_random_device_tensor_int({n_batch}, n_sequence - 1);
    lengths_to_compare.copy_from(wrapper.lengths);

    launch_decoder(
        batch_emb_to_compare,
        emb_table_device,
        emb_score_to_compare,
        wpe_table_device,
        wrapper.inp_embedding,
        lengths_to_compare,
        decoder_result_to_compare);
    
    launch_paged_attention_cublas_decoder_multi_rounds(
        paged_attention_batch_emb,
        emb_table_device,
        paged_attention_emb_score,
        wpe_table_device,
        wrapper.page_table,
        wrapper.lengths,
        paged_attention_decoder_result, 0, handle);
    
    assert_page_table_close(
        (const float**)wrapper.page_table.data(), wrapper.inp_embedding.data(),
        wrapper.lengths.data(), n_batch, n_sequence, INP_EMB_EMB_OFFSET, emb_dim);
    assert_float_kernel_close(emb_score_to_compare.data(), paged_attention_emb_score.data(), emb_score_to_compare.get_total_size());
    assert_int_kernel_close(lengths_to_compare.data(), wrapper.lengths.data(), lengths_to_compare.get_total_size());
    assert_int_kernel_close(decoder_result_to_compare.data(), paged_attention_decoder_result.data(), decoder_result_to_compare.get_total_size());
    cublasDestroy(handle);
}

TEST(DecoderTest, PagedAttentionCublasMaxLengthTest) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    size_t n_batch = get_random_number(128, 128 * 3);
    size_t emb_dim = get_random_number(128, 128 * 3) / 4 * 4;
    size_t n_sequence = get_random_number(1024, 3096) / PAGE_BLOCK_SIZE * PAGE_BLOCK_SIZE;
    size_t n_vocab = get_random_number(1024, 3096);

    auto emb_table_device = get_random_device_tensor({n_vocab, emb_dim});
    auto wpe_table_device = get_random_device_tensor({n_sequence, emb_dim});

    auto lengths_to_compare_device_host = get_random_device_host_tensor_int({n_batch}, n_sequence - 1);

    auto zero_and_max_length_indices = get_unique_num_array(0, n_batch - 1, get_random_number(2, n_batch - 3));
    int num_max_length = get_random_number(1, zero_and_max_length_indices.size() - 1);
    int* lengths_data_host = lengths_to_compare_device_host.second.data();
    for (int i = 0; i < zero_and_max_length_indices.size(); ++i) {
        if (i < num_max_length) {
            lengths_data_host[zero_and_max_length_indices[i]] = n_sequence - 1;
        } else {
            lengths_data_host[zero_and_max_length_indices[i]] = 0;
        }
    }

    lengths_to_compare_device_host.first.copy_from(lengths_to_compare_device_host.second);
    TensorWrapperForPagedAttention wrapper = generate_paged_attention_wrapper_device_tensors(
        n_batch, n_sequence, emb_dim, lengths_to_compare_device_host.first);

    auto batch_emb_to_compare = get_random_device_tensor({n_batch, emb_dim});
    auto paged_attention_batch_emb = get_random_device_tensor({n_batch, emb_dim});
    paged_attention_batch_emb.copy_from(batch_emb_to_compare);

    auto emb_score_to_compare = get_random_device_tensor({n_batch, n_vocab});
    auto paged_attention_emb_score = get_random_device_tensor({n_batch, n_vocab});
    paged_attention_emb_score.copy_from(emb_score_to_compare);

    auto decoder_result_to_compare = get_random_device_tensor_int({n_batch}, 100);
    auto paged_attention_decoder_result = get_random_device_tensor_int({n_batch}, 100);
    paged_attention_decoder_result.copy_from(decoder_result_to_compare);


    launch_decoder(
        batch_emb_to_compare,
        emb_table_device,
        emb_score_to_compare,
        wpe_table_device,
        wrapper.inp_embedding,
        lengths_to_compare_device_host.first,
        decoder_result_to_compare);
    
    launch_paged_attention_cublas_decoder_multi_rounds(
        paged_attention_batch_emb,
        emb_table_device,
        paged_attention_emb_score,
        wpe_table_device,
        wrapper.page_table,
        wrapper.lengths,
        paged_attention_decoder_result, 0, handle);
    
    assert_page_table_close(
        (const float**)wrapper.page_table.data(), wrapper.inp_embedding.data(),
        wrapper.lengths.data(), n_batch, n_sequence, INP_EMB_EMB_OFFSET, emb_dim);
    assert_float_kernel_close(emb_score_to_compare.data(), paged_attention_emb_score.data(), emb_score_to_compare.get_total_size());
    assert_int_kernel_close(lengths_to_compare_device_host.first.data(), wrapper.lengths.data(),
        lengths_to_compare_device_host.first.get_total_size());
    assert_int_kernel_close(decoder_result_to_compare.data(), paged_attention_decoder_result.data(), decoder_result_to_compare.get_total_size());
    cublasDestroy(handle);
}

