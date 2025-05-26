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

    int n_finishes = get_random_number(50, n_batch - 1);
    int n_finish_due_to_tokens = get_random_number(1, n_finishes - 10);
    int n_finished_due_to_empty_token_id = get_random_number(1, n_finishes - n_finish_due_to_tokens - 10);
    std::vector<int> finish_indices = get_unique_num_array(0, n_batch - 1, n_finishes);
    int* decoder_host_data = decoder_result_device_host.second.data();
    for (int i = 0; i < n_batch; ++i) {
        processing_storage.put(i, std::make_pair(i, create_random_vector(get_random_number(1, n_sequence - 3), 0, EOF_TOKEN_ID - 1)));
    }

    for (int i = 0; i < finish_indices.size(); ++i) {
        if (i < n_finish_due_to_tokens) {
            decoder_host_data[finish_indices[i]] = EOF_TOKEN_ID;
        } else if (i < n_finished_due_to_empty_token_id + n_finish_due_to_tokens) { 
            decoder_host_data[finish_indices[i]] = EMPTY_ROW_TOKEN_ID;
            processing_storage.remove(finish_indices[i]);
        } else {
            processing_storage.put(finish_indices[i], std::make_pair(
                finish_indices[i], create_random_vector(n_sequence - 1, 0, EOF_TOKEN_ID - 1)));
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
    ASSERT_EQ(item_storage.finish_count() + n_finished_due_to_empty_token_id, finished_indices_result.size());
    ASSERT_EQ(item_storage.finish_count() + processing_storage.size() + n_finished_due_to_empty_token_id, n_batch);
}

TEST(ItemStorageTest, InsertNewItemsWithEnoughNewItemsTest) {
    // The same as ProcessDecoderResultTest
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
        processing_storage.put(i, std::make_pair(i, create_random_vector(get_random_number(1, n_sequence - 3), 0, EOF_TOKEN_ID - 1)));
    }

    for (int i = 0; i < finish_indices.size(); ++i) {
        if (i < n_finish_due_to_tokens) {
            decoder_host_data[finish_indices[i]] = EOF_TOKEN_ID;
        } else { 
            processing_storage.put(finish_indices[i], std::make_pair(finish_indices[i], create_random_vector(get_random_number(1, n_sequence - 3), 0, EOF_TOKEN_ID - 1)));
        }
    }
    decoder_result_device_host.first.copy_from(decoder_result_device_host.second);

    std::vector<int> finished_indices_result = process_decoder_result(
        decoder_result_device_host.first, 
        decoder_result_device_host.second, item_storage, processing_storage, n_sequence);
    
    for (int new_id = n_batch; new_id < n_batch + n_finishes + 1; new_id++) {
        item_storage.add_new_item(std::make_pair(new_id, create_random_vector(get_random_number(1, n_sequence - 3), 0, EOF_TOKEN_ID - 1)));
    }
    
    // insert_new_items
    auto inp_device_host = get_random_device_host_tensor_int({n_batch, n_sequence}, n_vocab);
    auto lengths_device_host = get_random_device_host_tensor_int({n_batch}, n_sequence);
    auto new_items_indices_device_host = get_random_device_host_tensor_int({n_batch}, n_batch - 1);

    insert_new_items(finished_indices_result,
        inp_device_host.first, inp_device_host.second,
        lengths_device_host.first, lengths_device_host.second,
        new_items_indices_device_host.first, new_items_indices_device_host.second, item_storage, processing_storage);
    
    inp_device_host.second.copy_from(inp_device_host.first);
    lengths_device_host.second.copy_from(lengths_device_host.first);
    new_items_indices_device_host.second.copy_from(new_items_indices_device_host.first);

    const int* inp_data = inp_device_host.second.data();
    const int* lengths_data = lengths_device_host.second.data();
    const int* new_item_indices_data = new_items_indices_device_host.second.data();

    for (int i = 0; i < finished_indices_result.size(); ++i) {
        int index = finished_indices_result[i];
        ASSERT_EQ(processing_storage.get_token(index).first, i + n_batch);    // assert this is new item
        ASSERT_EQ(lengths_data[index], processing_storage.get_token(index).second.size());
        ASSERT_EQ(new_item_indices_data[i], index);
        for (int j = 0; j < lengths_data[index]; ++j) {
            ASSERT_EQ(inp_data[index * n_sequence + j], processing_storage.get_token(index).second[j]);
        }
    }
}

TEST(ItemStorageTest, InsertNewItemsWithoutEnoughNewItemsTest) {
    // The same as ProcessDecoderResultTest
    size_t n_batch = get_random_number(128, 1024);
    size_t n_sequence = get_random_number(128, 1024);
    size_t n_vocab = get_random_number(EOF_TOKEN_ID + 1, EOF_TOKEN_ID + 1024);
    ItemStorage item_storage;
    ProcessingStorage processing_storage;
    // so the generated decoder results are not having EOF_TOKEN_ID
    auto decoder_result_device_host = get_random_device_host_tensor_int(
        {n_batch}, EOF_TOKEN_ID - 1);

    int n_finishes = get_random_number(40, n_batch - 1);
    int n_finish_due_to_tokens = get_random_number(1, n_finishes - 1);
    std::vector<int> finish_indices = get_unique_num_array(0, n_batch - 1, n_finishes);
    int* decoder_host_data = decoder_result_device_host.second.data();
    for (int i = 0; i < n_batch; ++i) {
        processing_storage.put(i, std::make_pair(i, create_random_vector(get_random_number(1, n_sequence - 3), 0, EOF_TOKEN_ID - 1)));
    }

    for (int i = 0; i < finish_indices.size(); ++i) {
        if (i < n_finish_due_to_tokens) {
            decoder_host_data[finish_indices[i]] = EOF_TOKEN_ID;
        } else { 
            processing_storage.put(finish_indices[i], std::make_pair(finish_indices[i], create_random_vector(get_random_number(1, n_sequence - 3), 0, EOF_TOKEN_ID - 1)));
        }
    }
    decoder_result_device_host.first.copy_from(decoder_result_device_host.second);

    std::vector<int> finished_indices_result = process_decoder_result(
        decoder_result_device_host.first, 
        decoder_result_device_host.second, item_storage, processing_storage, n_sequence);
    
    int n_new_items = get_random_number(10, n_finishes - 10);
    for (int new_id = n_batch; new_id < n_batch + n_new_items; new_id++) {
        item_storage.add_new_item(std::make_pair(new_id, create_random_vector(get_random_number(1, n_sequence - 3), 0, EOF_TOKEN_ID - 1)));
    }
    
    // insert_new_items
    auto inp_device_host = get_random_device_host_tensor_int({n_batch, n_sequence}, n_vocab);
    auto lengths_device_host = get_random_device_host_tensor_int({n_batch}, n_sequence);
    auto new_items_indices_device_host = get_random_device_host_tensor_int({n_batch}, n_batch - 1);

    insert_new_items(finished_indices_result,
        inp_device_host.first, inp_device_host.second,
        lengths_device_host.first, lengths_device_host.second,
        new_items_indices_device_host.first, new_items_indices_device_host.second, item_storage, processing_storage);
    
    inp_device_host.second.copy_from(inp_device_host.first);
    lengths_device_host.second.copy_from(lengths_device_host.first);
    new_items_indices_device_host.second.copy_from(new_items_indices_device_host.first);

    const int* inp_data = inp_device_host.second.data();
    const int* lengths_data = lengths_device_host.second.data();
    const int* new_item_indices_data = new_items_indices_device_host.second.data();

    for (int i = 0; i < finished_indices_result.size(); ++i) {
        int index = finished_indices_result[i];
        if (i < n_new_items) {
            ASSERT_EQ(processing_storage.get_token(index).first, i + n_batch);    // assert this is new item
            ASSERT_EQ(lengths_data[index], processing_storage.get_token(index).second.size());
            ASSERT_EQ(new_item_indices_data[i], index);
            for (int j = 0; j < lengths_data[index]; ++j) {
                ASSERT_EQ(inp_data[index * n_sequence + j], processing_storage.get_token(index).second[j]);
            }
        } else {
            ASSERT_FALSE(processing_storage.batch_id_processing(index));
            ASSERT_EQ(lengths_data[index], 0);
            ASSERT_EQ(new_item_indices_data[i], index);
        }
    }
}
