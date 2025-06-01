#include "constants.h"
#include "paged_item_storage.h"
#include "tensor.hpp"
#include "test_utils.h"
#include <gtest/gtest.h>
#include <vector>

/**
 * what do we want to test:
 * 1. Insert all new possible items. The lengths for each should be correct.
 * 2. Process decoder result. The finished indices are correct. And the memories are returned. And new memories are allocated.
 * 3. If memories are not enough, there will be blocks freed. Test if this item is the last one.
 */

TEST(PagedItemStorageTest, InsertNewItemsTest) {
    size_t max_batches = get_random_number(56, 128);
    size_t n_blocks = DEFAULT_INIT_NUM_BLOCKS * max_batches;
    size_t n_sequence = PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS * get_random_number(2, 5);
    size_t n_dims = get_random_number(64, 128) * 4;
    std::vector<int> new_item_lengths = create_random_vector(max_batches * 2, 1, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS);
    PagedAttentionTestWrapper wrapper = mock_paged_attention_test_wrapper(
        PAGE_BLOCK_SIZE, n_sequence, n_dims, n_blocks, new_item_lengths);

    auto inp_device_host = get_random_device_host_tensor_int({max_batches, n_sequence}, EOF_TOKEN_ID - 1);
    auto lengths_device_host = get_random_device_host_tensor_int({max_batches}, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS);
    auto new_item_indices_device_host = get_random_device_host_tensor_int({max_batches}, max_batches - 1);
    std::vector<int> new_item_indices = insert_new_items(
        inp_device_host.first, inp_device_host.second,
        lengths_device_host.first, lengths_device_host.second,
        new_item_indices_device_host.first, new_item_indices_device_host.second,
        wrapper.item_storage, wrapper.processing_storage, wrapper.memory_block_manager, wrapper.paged_attention_manager);

    ASSERT_EQ(new_item_indices.size(), max_batches);
    // for (int i = 0; i < new_item_indices.size(); ++i) {
    //     ASSERT_EQ(new_item_indices[i], i);
    // }

    // ASSERT_EQ(wrapper.item_storage.new_count(), max_batches);
    // ASSERT_EQ(wrapper.memory_block_manager.free_blocks_size(), 0);

    // inp_device_host.second.copy_from(inp_device_host.first);
    // lengths_device_host.second.copy_from(lengths_device_host.first);
    // new_item_indices_device_host.second.copy_from(new_item_indices_device_host.first);
    // const int* inp_data = inp_device_host.second.data();
    // const int* lengths_data = lengths_device_host.second.data();
    // const int* new_item_indices_data = new_item_indices_device_host.second.data();
    // for (int i = 0; i < max_batches; ++i) {
    //     ASSERT_EQ(lengths_data[i], new_item_lengths[i]);
    //     ASSERT_EQ(new_item_indices_data[i], i);
    //     for (int j = 0; j < new_item_lengths[i]; ++j) {
    //         ASSERT_EQ(wrapper.tokens[i].second[j], inp_data[i * n_sequence + j]);
    //     }
    // }
}

// Test if we corretly allocate memories
TEST(PagedItemStorageTest, ReturnFreeBlocksTest) {
}
