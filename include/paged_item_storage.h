#pragma once

#include "items_storage.h"
#include "tensor.hpp"
#include <cstddef>
#include <list>
#include <vector>

class MemoryBlockManager {
public:
    // Allocate n_blocks * each_block_size * sizeof(float) cuda memory in total
    MemoryBlockManager(int n_blocks, size_t each_block_size);
    int free_blocks_size() const;
    // get size free_blocks, if not enough, throw exception
    std::vector<float*> get_free_blocks(int size);
    void return_free_blocks(std::list<float*>&&);
private:
    TensorFloat block_memory_;
    std::list<float*> free_blocks_;
};

class PagedAttentionsManager {
public:
    int get_n_batch() const;
    void return_blocks(int index, MemoryBlockManager&);
private:
    TensorFloatPoint inp_embedding_device;
    TensorFloatPoint k_cache_device;
    TensorFloatPoint v_cache_device;
    TensorFloatPoint inp_embedding_host;
    TensorFloatPoint k_cache_host;
    TensorFloatPoint v_cache_host;
    std::vector<std::list<float*>> used_blocks_;
    int n_batch;
};

void return_memory_blocks(PagedAttentionsManager&, MemoryBlockManager&, int index);

void process_decoder_result(
    const TensorInt& decoder_result_device, TensorInt& decoder_result_host,
    ItemStorage& item_storage, ProcessingStorage& processing_storage, int n_sequence,
    MemoryBlockManager& memory_block_manager,
    PagedAttentionsManager& page_attentions_manager);

int insert_new_items();
