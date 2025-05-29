#include "paged_item_storage.h"
#include "constants.h"
#include <cassert>

void process_decoder_result(
    const TensorInt& decoder_result_device, TensorInt& decoder_result_host,
    ItemStorage& item_storage, ProcessingStorage& processing_storage, int n_sequence,
    MemoryBlockManager& memory_block_manager,
    PagedAttentionsManager& page_attentions_manager) {

    decoder_result_host.copy_from(decoder_result_device);
    const int* decoder_result_data = decoder_result_host.data();
    std::vector<int> finished_indices;
    int n_batch = page_attentions_manager.get_n_batch();
    // 1. firstly free out the finished indices
    for (int i = 0; i < n_batch; ++i) {
        if (decoder_result_data[i] == EMPTY_ROW_TOKEN_ID) {
            assert(!processing_storage.batch_id_processing(i));
            finished_indices.push_back(i);
        } else {
            append_token_to_id_string_pair(processing_storage.get_token(i), decode_result_data[i]);
            if (processing_storage.get_token(i).second.size() >= n_sequence
                || decoder_result_data[i] == EOF_TOKEN_ID) {

                finished_indices.push_back(i);
                processing_storage.move_to_finished(i, item_storage);
                return_memory_blocks(page_attentions_manager, memory_block_manager, i);
            }
        }
    }
    // 2. For processing items, check length and find free blocks if needed. If no free blocks,
    // we have to pop processing items back to new_items
}
