#include "constants.h"
#include "items_storage.h"
#include "test_utils.h"
#include <gtest/gtest.h>
#include <utility>
#include <vector>

TEST(ItemStorageTest, ProcessDecoderResultTest) {
    size_t n_batch = get_random_number(128, 1024);
    size_t n_sequence = get_random_number(128, 1024);
    size_t n_vocab = get_random_number(EOF_TOKEN_ID + 1, EOF_TOKEN_ID + 1024);
    ItemStorage item_storage;
    ProcessingStorage processing_storage;
    // so the generated decoder results are not having EOF_TOKEN_ID
    auto decoder_result_device_host = get_random_device_host_tensor_int(
        {n_batch}, EOF_TOKEN_ID - 1);

    int n_finishes = get_random_number(4, n_batch - 1);
    int n_finish_due_to_tokens = get_random_number(1, n_finishes - 1);
    std::vector<int> finish_indices = get_unique_num_array(0, n_batch - 1, n_finishes);
    int* decoder_host_data = decoder_result_device_host.second.data();
    for (int i = 0; i < n_batch; ++i) {
        processing_storage.get_processing_items().push_back(
                std::make_pair(i, create_random_vector(get_random_number(1, n_sequence - 3), 0, EOF_TOKEN_ID - 1))
            );
    }

    for (int i = 0; i < finish_indices.size(); ++i) {
        if (i < n_finish_due_to_tokens) {
            decoder_host_data[finish_indices[i]] = EOF_TOKEN_ID;
        } else { 
            processing_storage.get_processing_items()[finish_indices[i]] = std::make_pair(
                finish_indices[i], create_random_vector(n_sequence - 1, 0, EOF_TOKEN_ID - 1));
        }
    }
    decoder_result_device_host.first.copy_from(decoder_result_device_host.second);

    std::vector<int> finished_indices_result = process_decoder_result(
        decoder_result_device_host.first, decoder_result_device_host.second, item_storage, processing_storage, n_sequence);
    std::sort(finished_indices_result.begin(), finished_indices_result.end());
    std::sort(finish_indices.begin(), finish_indices.end());
    ASSERT_EQ(finish_indices.size(), finished_indices_result.size());
    for (int i = 0; i < finish_indices.size(); ++i) {
        ASSERT_EQ(finish_indices[i], finished_indices_result[i]);
    }
    ASSERT_EQ(item_storage.finish_count(), finished_indices_result.size());
}

TEST(ItemStorageTest, InsertNewItemsTest) {
}
