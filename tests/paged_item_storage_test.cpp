#include "constants.h"
#include "include/test_utils.h"
#include "paged_item_storage.h"
#include "tensor.hpp"
#include "test_utils.h"
#include <gtest/gtest.h>
#include <utility>
#include <vector>

/**
 * what do we want to test:
 * 1. Insert all new possible items. The lengths for each should be correct.
 * 2. Process decoder result. The finished indices are correct. And the memories are returned. And new memories are allocated.
 * 3. If memories are not enough, there will be blocks freed. Test if this item is the last one.
 */

TEST(PagedItemStorageTest, InsertAllItemsTest) {
    size_t max_batches = get_random_number(128, 512);
    size_t n_blocks = DEFAULT_INIT_NUM_BLOCKS * max_batches;
    size_t n_sequence = PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS * get_random_number(2, 5);
    size_t n_dims = get_random_number(64, 128) * 4;
    std::vector<int> new_item_lengths = create_random_vector(max_batches * 2, 1, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS - 1);
    PagedAttentionTestWrapper wrapper = mock_paged_attention_test_wrapper(
        max_batches, n_sequence, n_dims, n_blocks, new_item_lengths);

    auto inp_device_host = get_random_device_host_tensor_int({max_batches, n_sequence}, EOF_TOKEN_ID - 1);
    auto lengths_device_host = get_random_device_host_tensor_int({max_batches}, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS);
    auto new_item_indices_device_host = get_random_device_host_tensor_int({max_batches}, max_batches - 1);
    std::vector<int> new_item_indices = insert_new_items(
        inp_device_host.first, inp_device_host.second,
        lengths_device_host.first, lengths_device_host.second,
        new_item_indices_device_host.first, new_item_indices_device_host.second,
        wrapper.item_storage, wrapper.processing_storage, wrapper.memory_block_manager, wrapper.paged_attention_manager);

    ASSERT_EQ(new_item_indices.size(), max_batches);
    for (int i = 0; i < new_item_indices.size(); ++i) {
        ASSERT_EQ(new_item_indices[i], i);
    }

    ASSERT_EQ(wrapper.item_storage.new_count(), max_batches);
    ASSERT_EQ(wrapper.memory_block_manager.free_blocks_size(), 0);

    inp_device_host.second.copy_from(inp_device_host.first);
    lengths_device_host.second.copy_from(lengths_device_host.first);
    new_item_indices_device_host.second.copy_from(new_item_indices_device_host.first);
    const int* inp_data = inp_device_host.second.data();
    const int* lengths_data = lengths_device_host.second.data();
    const int* new_item_indices_data = new_item_indices_device_host.second.data();
    for (int i = 0; i < max_batches; ++i) {
        ASSERT_EQ(lengths_data[i], new_item_lengths[i]);
        ASSERT_EQ(new_item_indices_data[i], i);
        for (int j = 0; j < new_item_lengths[i]; ++j) {
            ASSERT_EQ(wrapper.tokens[i].second[j], inp_data[i * n_sequence + j]);
        }
    }
}

TEST(PagedItemStorageTest, InsertNewItemsTest) {
    size_t max_batches = get_random_number(128, 512);
    size_t n_blocks = DEFAULT_INIT_NUM_BLOCKS * max_batches;
    size_t n_sequence = PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS * get_random_number(2, 5);
    size_t n_dims = get_random_number(64, 128) * 4;
    std::vector<int> new_item_lengths = create_random_vector(max_batches - 1, 1, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS - 1);
    PagedAttentionTestWrapper wrapper = mock_paged_attention_test_wrapper(
        max_batches, n_sequence, n_dims, n_blocks, new_item_lengths);

    auto inp_device_host = get_random_device_host_tensor_int({max_batches, n_sequence}, EOF_TOKEN_ID - 1);
    auto lengths_device_host = get_random_device_host_tensor_int({max_batches}, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS);
    auto new_item_indices_device_host = get_random_device_host_tensor_int({max_batches}, max_batches - 1);
    std::vector<int> new_item_indices = insert_new_items(
        inp_device_host.first, inp_device_host.second,
        lengths_device_host.first, lengths_device_host.second,
        new_item_indices_device_host.first, new_item_indices_device_host.second,
        wrapper.item_storage, wrapper.processing_storage, wrapper.memory_block_manager, wrapper.paged_attention_manager);

    ASSERT_EQ(new_item_indices.size(), max_batches - 1);

    int new_length = get_random_number(1, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS - 1);
    new_item_lengths.push_back(new_length);
    new_item_lengths.push_back(new_length);
    // We should only insert one
    wrapper.item_storage.add_new_item(std::make_pair(max_batches - 1, create_random_vector(new_length, 0, EOF_TOKEN_ID - 1)));
    wrapper.item_storage.add_new_item(std::make_pair(max_batches, create_random_vector(new_length, 0, EOF_TOKEN_ID - 1)));

    new_item_indices = insert_new_items(
        inp_device_host.first, inp_device_host.second,
        lengths_device_host.first, lengths_device_host.second,
        new_item_indices_device_host.first, new_item_indices_device_host.second,
        wrapper.item_storage, wrapper.processing_storage, wrapper.memory_block_manager, wrapper.paged_attention_manager);

    ASSERT_EQ(new_item_indices.size(), 1);
    ASSERT_EQ(new_item_indices[0], max_batches - 1);

    ASSERT_EQ(wrapper.item_storage.new_count(), 1);
    ASSERT_EQ(wrapper.memory_block_manager.free_blocks_size(), 0);

    inp_device_host.second.copy_from(inp_device_host.first);
    lengths_device_host.second.copy_from(lengths_device_host.first);
    new_item_indices_device_host.second.copy_from(new_item_indices_device_host.first);
    const int* inp_data = inp_device_host.second.data();
    const int* lengths_data = lengths_device_host.second.data();
    const int* new_item_indices_data = new_item_indices_device_host.second.data();
    for (int i = 0; i < max_batches; ++i) {
        ASSERT_EQ(lengths_data[i], new_item_lengths[i]);
        if (i < max_batches - 1) {
            for (int j = 0; j < new_item_lengths[i]; ++j) {
                ASSERT_EQ(wrapper.tokens[i].second[j], inp_data[i * n_sequence + j]);
            }
        }
    }
    ASSERT_EQ(new_item_indices_data[0], max_batches - 1);
}

// Assuming we have finished items
TEST(PagedItemStorageTest, ReturnFreeBlocksTest) {
    size_t max_batches = get_random_number(128, 512);
    size_t n_blocks = DEFAULT_INIT_NUM_BLOCKS * max_batches;
    size_t n_sequence = PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS * get_random_number(2, 5);
    size_t n_dims = get_random_number(64, 128) * 4;
    std::vector<int> new_item_lengths = create_random_vector(max_batches * 2, 1, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS - 1);
    PagedAttentionTestWrapper wrapper = mock_paged_attention_test_wrapper(
        max_batches, n_sequence, n_dims, n_blocks, new_item_lengths);

    auto inp_device_host = get_random_device_host_tensor_int({max_batches, n_sequence}, EOF_TOKEN_ID - 1);
    auto lengths_device_host = get_random_device_host_tensor_int({max_batches}, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS);
    auto new_item_indices_device_host = get_random_device_host_tensor_int({max_batches}, max_batches - 1);
    std::vector<int> new_item_indices = insert_new_items(
        inp_device_host.first, inp_device_host.second,
        lengths_device_host.first, lengths_device_host.second,
        new_item_indices_device_host.first, new_item_indices_device_host.second,
        wrapper.item_storage, wrapper.processing_storage, wrapper.memory_block_manager, wrapper.paged_attention_manager);

    ASSERT_EQ(wrapper.memory_block_manager.free_blocks_size(), 0);
    int n_finishes = get_random_number(2, max_batches - 10);
    std::vector<int> finish_indices = get_unique_num_array(0, max_batches - 1, n_finishes);
    auto decoder_result_device_host = get_random_device_host_tensor_int(
        {max_batches}, EOF_TOKEN_ID - 1);
    int* decoder_host_data = decoder_result_device_host.second.data();
    for (int i = 0; i < finish_indices.size(); ++i) {
        decoder_host_data[finish_indices[i]] = EOF_TOKEN_ID;
    }
    decoder_result_device_host.first.copy_from(decoder_result_device_host.second);
    std::vector<int> finished_indices_result = process_decoder_result(
        decoder_result_device_host.first, decoder_result_device_host.second,
        wrapper.item_storage,
        wrapper.processing_storage, n_sequence);
    allocate_or_free_memory_blocks_if_needed(
        wrapper.paged_attention_manager, wrapper.memory_block_manager,
        wrapper.processing_storage, wrapper.item_storage, finished_indices_result);
    ASSERT_EQ(wrapper.memory_block_manager.free_blocks_size(), n_finishes * DEFAULT_INIT_NUM_BLOCKS);
    ASSERT_EQ(wrapper.item_storage.finish_count(), n_finishes);
    new_item_indices = insert_new_items(
        inp_device_host.first, inp_device_host.second,
        lengths_device_host.first, lengths_device_host.second,
        new_item_indices_device_host.first, new_item_indices_device_host.second,
        wrapper.item_storage, wrapper.processing_storage, wrapper.memory_block_manager, wrapper.paged_attention_manager);
    ASSERT_EQ(new_item_indices.size(), n_finishes);
}

TEST(PagedItemStorageTest, AllocateMoreBlocksTest) {
    size_t max_batches = get_random_number(128, 512);
    size_t n_blocks = DEFAULT_INIT_NUM_BLOCKS * max_batches;
    size_t n_sequence = PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS * get_random_number(2, 5);
    size_t n_dims = get_random_number(64, 128) * 4;
    std::vector<int> new_item_lengths = create_random_vector(max_batches / 2, 1, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS - 2);
    int n_to_allocate = get_random_number(2, max_batches / 2);
    std::vector<int> to_allocate_indices = get_unique_num_array(0, max_batches / 2 - 1, n_to_allocate);
    for (int index : to_allocate_indices) {
        new_item_lengths[index] = PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS - 1;
    }

    PagedAttentionTestWrapper wrapper = mock_paged_attention_test_wrapper(
        max_batches, n_sequence, n_dims, n_blocks, new_item_lengths);

    auto inp_device_host = get_random_device_host_tensor_int({max_batches, n_sequence}, EOF_TOKEN_ID - 1);
    auto lengths_device_host = get_random_device_host_tensor_int({max_batches}, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS);
    auto new_item_indices_device_host = get_random_device_host_tensor_int({max_batches}, max_batches - 1);
    std::vector<int> new_item_indices = insert_new_items(
        inp_device_host.first, inp_device_host.second,
        lengths_device_host.first, lengths_device_host.second,
        new_item_indices_device_host.first, new_item_indices_device_host.second,
        wrapper.item_storage, wrapper.processing_storage, wrapper.memory_block_manager, wrapper.paged_attention_manager);

    ASSERT_EQ(wrapper.memory_block_manager.free_blocks_size(), n_blocks - max_batches / 2 * DEFAULT_INIT_NUM_BLOCKS);

    auto decoder_result_device_host = get_random_device_host_tensor_int(
        {max_batches}, EOF_TOKEN_ID - 1);
    std::vector<int> finished_indices_result = process_decoder_result(
        decoder_result_device_host.first, decoder_result_device_host.second,
        wrapper.item_storage,
        wrapper.processing_storage, n_sequence);
    allocate_or_free_memory_blocks_if_needed(
        wrapper.paged_attention_manager, wrapper.memory_block_manager,
        wrapper.processing_storage, wrapper.item_storage, finished_indices_result);
    ASSERT_EQ(finished_indices_result.size(), 0);
    ASSERT_EQ(wrapper.memory_block_manager.free_blocks_size(), n_blocks - max_batches / 2 * DEFAULT_INIT_NUM_BLOCKS - n_to_allocate);
}

TEST(PagedItemStorageTest, FreeBlocksTest) {

}
