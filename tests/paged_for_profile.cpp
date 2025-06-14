
#include "constants.h"
#include "inference_model.h"
#include "inferencer.h"
#include "test_utils.h"
#include <chrono>
#include <nvtx3/nvToolsExt.h>


int main() {
    int n_block_ratio = 2;
    size_t max_batches = 512 / n_block_ratio * n_block_ratio * 2;
    size_t n_blocks = DEFAULT_INIT_NUM_BLOCKS * max_batches;
    size_t n_sequence = PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS * n_block_ratio;
    size_t emb_dims = 512 * 4;
    size_t n_vocab = EOF_TOKEN_ID + 1;

    std::vector<int> new_item_lengths = create_random_vector(max_batches * 2, 1, PAGE_BLOCK_SIZE * DEFAULT_INIT_NUM_BLOCKS);
    PagedAttentionTestWrapper wrapper = mock_paged_attention_test_wrapper(
        max_batches, n_sequence, emb_dims, n_blocks, new_item_lengths);

    TensorFloat emb_table = get_random_device_emb_table(emb_dims, n_vocab, 1.0001);
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
        max_batches, n_sequence, emb_dims);
    
    auto paged_attention_start = std::chrono::high_resolution_clock::now();
    nvtxRangePushA("page_attention_inference_engine");
    start_paged_attention_inference_engine(
        emb_table, pos_table, wrapper.item_storage, wrapper.processing_storage, wrapper.memory_block_manager, wrapper.paged_attention_manager,
        model, max_batches, n_sequence);
    nvtxRangePop();
    auto paged_attention_end = std::chrono::high_resolution_clock::now();
    auto paged_attention_duration = std::chrono::duration_cast<std::chrono::milliseconds>(paged_attention_end - paged_attention_start);
}