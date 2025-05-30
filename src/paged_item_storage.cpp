#include "paged_item_storage.h"
#include "constants.h"
#include "items_storage.h"
#include "utils.h"
#include <cassert>
#include <iterator>
#include <unordered_set>
#include <utility>
#include <vector>

void allocate_or_free_memory_blocks_if_needed(
    PagedAttentionsManager& paged_attention_manager,
    MemoryBlockManager& memory_block_manager,
    ProcessingStorage& processing_storage,
    ItemStorage& item_storage, const std::vector<int>& finished_indices) {

    // 1. iterate over the finished indices and free them
    std::unordered_set<int> finished_indices_set(finished_indices.begin(), finished_indices.end());
    auto used_blocks = paged_attention_manager.get_used_block_list();
    for (auto it = used_blocks.begin(); it != used_blocks.end();) {
        if (finished_indices_set.find(it->first) != finished_indices_set.end()) {
            return_memory_blocks(memory_block_manager, std::move(*it));
            it = used_blocks.erase(it);
        } else {
            ++it;
        }
    }


    // 2. allocate new blocks if needed
    for (auto it = used_blocks.begin(); it != used_blocks.end();) {
        int batch_index = it->first;
        assert(processing_storage.batch_id_processing(batch_index));
        const IdTokensPair& token_pair = processing_storage.get_token(batch_index);
        if (token_pair.second.size() == it->second.size() * PAGE_BLOCK_SIZE) {
            if (memory_block_manager.free_blocks_size() > 0) {
                allocate_memory_block(memory_block_manager, paged_attention_manager, *it);
            } else if (std::next(it) == used_blocks.end()) {
                // put this element to new_items
                move_to_new(it->first, processing_storage, item_storage);
                return_memory_blocks(memory_block_manager, std::move(*it));
                it = used_blocks.erase(it);
                continue;
            } else {
                // free the list's tail
                BatchIdMemoryBlocksPair to_remove(std::move(used_blocks.back()));
                used_blocks.pop_back();
                move_to_new(to_remove.first, processing_storage, item_storage);
                return_memory_blocks(memory_block_manager, std::move(to_remove));
            }
        }
    }
}

std::vector<int> insert_new_items(
    TensorInt& inp_device, TensorInt& inp_host,
    TensorInt& lengths_device, TensorInt& lengths_host,
    TensorInt& new_items_indices_device, TensorInt& new_items_indices_host,
    ItemStorage& item_storage, ProcessingStorage& processing_storage,
    MemoryBlockManager& memory_block_manager, PagedAttentionsManager& paged_attention_manager) {

    int max_batch = inp_device.shape()[0];
    int n_sequence = inp_device.shape()[1];
    int* inp_data = inp_host.data();
    int* lengths_data = lengths_host.data();
    int* new_items_indices_data = new_items_indices_host.data();
    std::unordered_set<int> occupied_batch_indices;
    std::vector<int> new_item_indices;
    for (auto it = paged_attention_manager.get_used_block_list().cbegin(); it != paged_attention_manager.get_used_block_list().cend(); ++it) {
        
        occupied_batch_indices.insert(it->first);
    }
    bool need_copy = false;
    for (int i = 0; i < max_batch; ++i) {
        if (occupied_batch_indices.find(i) != occupied_batch_indices.end()) {
            continue;
        }

        if (memory_block_manager.free_blocks_size() > DEFAULT_INIT_NUM_BLOCKS && item_storage.new_count() > 0 
            && ceil_div(std::min(item_storage.head_length() + 1, n_sequence), PAGE_BLOCK_SIZE) 
                > memory_block_manager.free_blocks_size()) {
            
            IdTokensPair popped = item_storage.pop_new_items(1)[0];
            lengths_data[i] = popped.second.size();
            processing_storage.put(i, std::move(popped));
            std::copy(
                popped.second.begin(), popped.second.end(),
                inp_data + i * n_sequence);
            int n_blocks = std::max(
                ceil_div(std::min(item_storage.head_length() + 1, n_sequence), PAGE_BLOCK_SIZE),
                DEFAULT_INIT_NUM_BLOCKS);

            paged_attention_manager.add_batch_block_pair(std::make_pair(
                i, std::move(memory_block_manager.get_free_blocks(n_blocks))));
            new_item_indices.push_back(i);
            need_copy = true;
        } else {
            lengths_data[i] = 0;
            need_copy = true;
        }
    }
    if (need_copy) {
        inp_device.copy_from(inp_host);
        lengths_device.copy_from(lengths_host);
        new_items_indices_device.copy_from(new_items_indices_host);
    }

    if (!new_item_indices.empty()) {
        paged_attention_manager.flush_changes();
    }

    return new_item_indices;
}
