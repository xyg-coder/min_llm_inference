#include "constants.h"
#include "inference_model.h"
#include "inferencer.h"
#include "item_storage.h"
#include "layers.h"
#include "test_utils.h"
#include <cassert>
#include <gtest/gtest.h>
#include <iostream>
#include <unordered_map>

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
        max_batches, n_sequence, emb_dims, 1);
    
    start_paged_attention_inference_engine(
        emb_table, pos_table, wrapper.item_storage, wrapper.processing_storage, wrapper.memory_block_manager, wrapper.paged_attention_manager,
        model, max_batches, n_sequence, 1);

    ASSERT_EQ(wrapper.item_storage.finish_count(), max_batches * 2);
}

TEST(InferenceTest, Compare2Inferences) {
    int n_block_ratio = get_random_number(2, 5);
    size_t max_batches = get_random_number(128, 512) / n_block_ratio * n_block_ratio;
    size_t n_blocks = DEFAULT_INIT_NUM_BLOCKS * max_batches;
    size_t n_sequence = PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS * n_block_ratio;
    size_t emb_dims = get_random_number(64, 128) * 4;
    size_t n_vocab = get_random_number(EOF_TOKEN_ID + 1, EOF_TOKEN_ID + 1024);

    std::vector<int> new_item_lengths = create_random_vector(max_batches * 2, 1, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS / 2);
    PagedAttentionTestWrapper wrapper = mock_paged_attention_test_wrapper(
        max_batches, n_sequence, emb_dims, n_blocks, new_item_lengths);

    TensorFloat emb_table = get_random_device_emb_table(emb_dims, n_vocab, 1.3);
    TensorFloat pos_table = get_random_device_tensor({n_sequence, emb_dims});

    TensorFloat wk = get_random_device_tensor({emb_dims, emb_dims});
    TensorFloat wq = get_random_device_tensor({emb_dims, emb_dims});
    TensorFloat wv = get_random_device_tensor({emb_dims, emb_dims});

    TensorFloat wk_2 = get_random_device_tensor({emb_dims, emb_dims});
    wk_2.copy_from(wk);
    TensorFloat wq_2 = get_random_device_tensor({emb_dims, emb_dims});
    wq_2.copy_from(wq);
    TensorFloat wv_2 = get_random_device_tensor({emb_dims, emb_dims});
    wv_2.copy_from(wv);

    ItemStorage item_storage_2;
    ProcessingStorage processing_storage_2;

    for (const IdTokensPair& id_tokens : wrapper.tokens) {
        item_storage_2.add_new_item(IdTokensPair(id_tokens));
    }

    PagedAttentionInferenceModel model(
        PagedAttentionLayer(
            std::move(wk),
            std::move(wq),
            std::move(wv),
            max_batches, emb_dims, n_sequence),
        PagedEncoderLayer(),
        PagedDecoderLayer(max_batches, n_vocab),
        max_batches, n_sequence, emb_dims, 1);
    
    auto paged_attention_start = std::chrono::high_resolution_clock::now();
    start_paged_attention_inference_engine(
        emb_table, pos_table, wrapper.item_storage, wrapper.processing_storage, wrapper.memory_block_manager, wrapper.paged_attention_manager,
        model, max_batches, n_sequence, 1);
    auto paged_attention_end = std::chrono::high_resolution_clock::now();
    auto paged_attention_duration = std::chrono::duration_cast<std::chrono::milliseconds>(paged_attention_end - paged_attention_start);

    int allocated_memory = n_blocks * PAGE_BLOCK_SIZE * 3 * emb_dims;
    assert(allocated_memory % (3 * emb_dims * n_sequence) == 0);
    int n_batch = allocated_memory / (3 * emb_dims * n_sequence);

    InferenceModel model_2(
        SelfAttentionLayer(
            std::move(wk_2),
            std::move(wq_2),
            std::move(wv_2),
            n_batch, emb_dims, n_sequence),
        EncoderLayer(),
        DecoderLayer(n_batch, n_vocab),
        n_batch, n_sequence, emb_dims);
    auto pre_allocate_start = std::chrono::high_resolution_clock::now();
    start_inference_engine(
        emb_table, pos_table, item_storage_2,
        processing_storage_2,
        model_2, n_batch, n_sequence);
    auto pre_allocate_end = std::chrono::high_resolution_clock::now();
    auto pre_allocate_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pre_allocate_end - pre_allocate_start);
    std::cout << "Paged attention takes " << paged_attention_duration.count() << " ms" << std::endl;
    std::cout << "Pre-allocate takes " << pre_allocate_duration.count() << " ms" << std::endl;
    ASSERT_EQ(wrapper.item_storage.finish_count(), max_batches * 2);
    ASSERT_EQ(item_storage_2.finish_count(), max_batches * 2);

    const std::list<IdTokensPair>& paged_finished_results = wrapper.item_storage.get_finished_items();
    const std::list<IdTokensPair>& finished_results_to_compare = wrapper.item_storage.get_finished_items();
    std::unordered_map<int, std::vector<int>> paged_result_map;
    for (const IdTokensPair& pair : paged_finished_results) {
        paged_result_map[pair.first] = pair.second;
    }
    for (const IdTokensPair& pair : finished_results_to_compare) {
        const std::vector<int>& paged_result = paged_result_map[pair.first];
        ASSERT_EQ(paged_result.size(), pair.second.size());
        for (int i = 0; i < paged_result.size(); ++i) {
            ASSERT_EQ(paged_result[i], pair.second[i]);
        }
    }
}
