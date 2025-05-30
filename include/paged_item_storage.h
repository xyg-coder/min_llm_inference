#pragma once

#include "items_storage.h"
#include "tensor.hpp"
#include <cstddef>
#include <list>
#include <utility>
#include <vector>

class MemoryBlockManager {
public:
    // Allocate n_blocks * each_block_size * sizeof(float) cuda memory in total
    MemoryBlockManager(int n_blocks, size_t each_block_size);
    int free_blocks_size() const;
    // get size free_blocks, if not enough, throw exception
    std::list<float*> get_free_blocks(int size);
    void return_free_blocks(std::list<float*>&&);
private:
    TensorFloat block_memory_;
    std::list<float*> free_blocks_;
};

using BatchIdMemoryBlocksPair = std::pair<int, std::list<float*>>;

class PagedAttentionsManager {
public:
    int get_n_batch() const;
    void return_blocks(int index, MemoryBlockManager&);
    std::list<BatchIdMemoryBlocksPair>& get_used_block_list();
    void flush_changes();
    void add_batch_block_pair(BatchIdMemoryBlocksPair&&);
private:
    TensorFloatPoint inp_embedding_device;
    TensorFloatPoint k_cache_device;
    TensorFloatPoint v_cache_device;
    TensorFloatPoint inp_embedding_host;
    TensorFloatPoint k_cache_host;
    TensorFloatPoint v_cache_host;
    // a list of <batch_index, memory-blocks>
    std::list<BatchIdMemoryBlocksPair> used_blocks_;
};

void return_memory_blocks(MemoryBlockManager&, BatchIdMemoryBlocksPair&&);

// This function will also handle put the float* to the hosts devices
void allocate_memory_block(MemoryBlockManager&, PagedAttentionsManager&, BatchIdMemoryBlocksPair&);

void allocate_or_free_memory_blocks_if_needed(PagedAttentionsManager&, MemoryBlockManager&,
    ProcessingStorage&, std::vector<int>& finished_indices);

std::vector<int> insert_new_items(
    TensorInt& inp_device, TensorInt& inp_host,
    TensorInt& lengths_device, TensorInt& lengths_host,
    TensorInt& new_items_indices_device, TensorInt& new_items_indices_host,
    ItemStorage& item_storage, ProcessingStorage& processing_storage,
    MemoryBlockManager& memory_block_manager, PagedAttentionsManager& paged_attention_manager);
