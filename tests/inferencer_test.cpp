#include "constants.h"
#include "inference_model.h"
#include "inferencer.h"
#include "layers.h"
#include "test_utils.h"
#include <gtest/gtest.h>

TEST(InferencerTest, InferencerTest) {
    size_t n_batch = get_random_number(32, 128);
    size_t n_sequence = get_random_number(32, 100) * 4;
    size_t n_vocab = get_random_number(EOF_TOKEN_ID + 1, EOF_TOKEN_ID + 1024);
    size_t emb_dims = get_random_number(256, 512) * 4;

    ItemStorage item_storage;
    ProcessingStorage processing_storage;

    TensorFloat emb_table = get_random_device_tensor({n_vocab, emb_dims});
    TensorFloat pos_table = get_random_device_tensor({n_sequence, emb_dims});

    int n_items = get_random_number(n_batch * 2, n_batch * 3);
    for (int i = 0; i < n_items; ++i) {
        item_storage.add_new_item(std::make_pair(i, create_random_vector(get_random_number(1, n_sequence / 2), 0, EOF_TOKEN_ID - 1)));
    }

    InferenceModel model(
        SelfAttentionLayer(
            get_random_device_tensor({emb_dims, emb_dims}),
            get_random_device_tensor({emb_dims, emb_dims}),
            get_random_device_tensor({emb_dims, emb_dims}),
            n_batch, emb_dims, n_sequence),
        EncoderLayer(),
        DecoderLayer(n_batch, n_vocab),
        n_batch, n_sequence, emb_dims);
    start_inference_engine(
        emb_table, pos_table, item_storage,
        processing_storage,
        model, n_batch, n_sequence);
    
    ASSERT_EQ(item_storage.finish_count(), n_items);
}

TEST(InferenceTest, PagedAttentionInferenceTest) {
    size_t max_batches = get_random_number(128, 512);
    size_t n_blocks = DEFAULT_INIT_NUM_BLOCKS * max_batches;
    size_t n_sequence = PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS * get_random_number(2, 5);
    size_t emb_dims = get_random_number(64, 128) * 4;
    size_t n_vocab = get_random_number(EOF_TOKEN_ID + 1, EOF_TOKEN_ID + 1024);
    std::vector<int> new_item_lengths = create_random_vector(max_batches * 2, 1, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS - 1);
    PagedAttentionTestWrapper wrapper = mock_paged_attention_test_wrapper(
        max_batches, n_sequence, emb_dims, n_blocks, new_item_lengths);

    TensorFloat emb_table = get_random_device_tensor({n_vocab, emb_dims});
    TensorFloat pos_table = get_random_device_tensor({n_sequence, emb_dims});

    PagedAttentionInferenceModel model(
        PagedAttentionLayer(
            get_random_device_tensor({emb_dims, emb_dims}),
            get_random_device_tensor({emb_dims, emb_dims}),
            get_random_device_tensor({emb_dims, emb_dims}),
            max_batches, emb_dims, n_sequence),
        PagedEncoderLayer(),
        PagedDecoderLayer(max_batches, n_vocab),
        max_batches, n_sequence, emb_dims);
    
    start_paged_attention_inference_engine(
        emb_table, pos_table, wrapper.item_storage, wrapper.processing_storage, wrapper.memory_block_manager, wrapper.paged_attention_manager,
        model, max_batches, n_sequence);

    ASSERT_EQ(wrapper.item_storage.finish_count(), max_batches * 2);
}
