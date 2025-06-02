#include "paged_item_storage.h"
#include "constants.h"
#include "items_storage.h"
#include "tensor.hpp"
#include "utils.h"
#include <cassert>
#include <iterator>
#include <list>
#include <stdexcept>
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
    std::list<BatchIdMemoryBlocksPair>& used_blocks = paged_attention_manager.get_used_block_list();
    for (auto it = used_blocks.begin(); it != used_blocks.end();) {
        if (finished_indices_set.find(it->first) != finished_indices_set.end()) {
            memory_block_manager.return_free_blocks(std::move(it->second));
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
                processing_storage.move_to_new(it->first, item_storage);
                memory_block_manager.return_free_blocks(std::move(it->second));
                it = used_blocks.erase(it);
                continue;
            } else {
                // free the list's tail
                BatchIdMemoryBlocksPair to_remove(std::move(used_blocks.back()));
                used_blocks.pop_back();
                processing_storage.move_to_new(to_remove.first, item_storage);
                memory_block_manager.return_free_blocks(std::move(to_remove.second));
            }
        } else {
            it++;
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
    int insert_index = 0;
    for (int i = 0; i < max_batch; ++i) {
        if (occupied_batch_indices.find(i) != occupied_batch_indices.end()) {
            continue;
        }

        if (memory_block_manager.free_blocks_size() >= DEFAULT_INIT_NUM_BLOCKS && item_storage.new_count() > 0 
            && memory_block_manager.free_blocks_size() 
                >= ceil_div(item_storage.head_length() + 1, PAGE_BLOCK_SIZE)) {
            
            IdTokensPair popped = item_storage.pop_new_items(1)[0];
            assert(popped.second.size() + 1 <= n_sequence);
            lengths_data[i] = popped.second.size();
            std::copy(
                popped.second.begin(), popped.second.end(),
                inp_data + i * n_sequence);
            new_items_indices_data[insert_index++] = i;
            int n_blocks = std::max(
                ceil_div(popped.second.size() + 1, PAGE_BLOCK_SIZE),
                DEFAULT_INIT_NUM_BLOCKS);
            processing_storage.put(i, std::move(popped));

            paged_attention_manager.add_batch_block_pair(std::make_pair(
                i, memory_block_manager.pop_free_blocks(n_blocks)));
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
    paged_attention_manager.maybe_flush_changes();

    return new_item_indices;
}

// each block size should be 3 * embedding_dim * PAGE_BLOCK_SIZE
MemoryBlockManager::MemoryBlockManager(int n_blocks, size_t each_block_size):
    block_memory_(TensorFloat({n_blocks * each_block_size}, DeviceType::DEVICE)) {
    
    free_blocks_ = std::list<float*>();
    float* data = block_memory_.data();
    for (int i = 0; i < n_blocks; ++i) {
        free_blocks_.push_back(data + i * each_block_size);
    }
}

int MemoryBlockManager::free_blocks_size() const { 
    return free_blocks_.size();
}


void MemoryBlockManager::return_free_blocks(std::list<float*>&& to_be_returned) {
    free_blocks_.splice(free_blocks_.end(), to_be_returned);
}

std::list<float*> MemoryBlockManager::pop_free_blocks(int size) {
    if (free_blocks_size() < size) {
        throw std::runtime_error("No enough block memories to return");
    }
    auto split_iter = free_blocks_.begin();
    std::advance(split_iter, size);
    std::list<float*> to_return;
    to_return.splice(to_return.end(), free_blocks_, free_blocks_.begin(), split_iter);
    return to_return;
}

PagedAttentionsManager::PagedAttentionsManager(
    size_t max_batches, size_t n_sequence, size_t emb_dim):
    page_table_device({max_batches, n_sequence / PAGE_BLOCK_SIZE}, DeviceType::DEVICE),
    page_table_host({max_batches, n_sequence / PAGE_BLOCK_SIZE}, DeviceType::HOST),
    needs_sync_(false), width_(n_sequence / PAGE_BLOCK_SIZE) { 
        assert(n_sequence % PAGE_BLOCK_SIZE == 0);
}

std::list<BatchIdMemoryBlocksPair>& PagedAttentionsManager::get_used_block_list() {
    return used_blocks_;
}

void PagedAttentionsManager::maybe_flush_changes() {
    if (needs_sync_) {
        page_table_device.copy_from(page_table_host); 
    }
    needs_sync_ = false;
}

void PagedAttentionsManager::set_block_pos(int batch_id, int i_block, float* memory_pos) {
    page_table_host.data()[batch_id * width_ + i_block] = memory_pos;
    needs_sync_ = true;
}

void PagedAttentionsManager::add_batch_block_pair(BatchIdMemoryBlocksPair&& pair) {
    int batch_id = pair.first;
    const std::list<float*>& blocks = pair.second;
    float** base = page_table_host.data() + batch_id * width_;
    int i = 0;
    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
        base[i++] = *it;
    }

    used_blocks_.push_back(std::move(pair));
    needs_sync_ = true;
}

void allocate_memory_block(
    MemoryBlockManager& memory_block_manager, PagedAttentionsManager& paged_attention_manager,
    BatchIdMemoryBlocksPair& pair) {
    
    float* allocated_memory = memory_block_manager.pop_free_blocks(1).front();
    pair.second.push_front(allocated_memory);
    paged_attention_manager.set_block_pos(pair.first, pair.second.size() - 1, allocated_memory);
}
