#include "include/test_utils.h"
#include "kernels/decoder.h"
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
    auto lengths_device_host = get_random_device_host_tensor_int({n_batch}, n_sequence);
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
    auto lengths_device_host = get_random_device_host_tensor_int({n_batch}, n_sequence);
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
