#include "inferencer.h"
#include "inference_model.h"
#include "item_storage.h"
#include "paged_item_storage.h"
#include "tensor.hpp"
#include "throughput_counter.h"
#include <vector>
#include <nvtx3/nvToolsExt.h>


void start_inference_engine(const TensorFloat& emb_table, const TensorFloat& pos_table,
    ItemStorage& item_storage, ProcessingStorage& processing_storage,
    InferenceModel& inference_model, size_t n_batch_size, size_t n_sequence) {

    std::vector<int> finished_indices;
    for (int i = 0; i < n_batch_size; ++i) {
        finished_indices.push_back(i);
    }
    TensorInt inp_device({n_batch_size, n_sequence}, DeviceType::DEVICE);
    TensorInt inp_host({n_batch_size, n_sequence}, DeviceType::HOST);
    TensorInt lengths_device({n_batch_size}, DeviceType::DEVICE);
    TensorInt lengths_host({n_batch_size}, DeviceType::HOST);
    TensorInt new_items_indices_device({n_batch_size}, DeviceType::DEVICE);
    TensorInt new_items_indices_host({n_batch_size}, DeviceType::HOST);
    int n_new_items = insert_new_items(
        finished_indices, inp_device, inp_host, lengths_device,
        lengths_host, new_items_indices_device, new_items_indices_host, item_storage, processing_storage);
    TensorInt decoder_result_device({n_batch_size}, DeviceType::DEVICE);
    TensorInt decoder_result_host({n_batch_size}, DeviceType::HOST);
    
    while (!is_done(item_storage, processing_storage)) {
        inference_model.forward(inp_device, lengths_device, new_items_indices_device, decoder_result_device, n_new_items,
            emb_table, pos_table);

        finished_indices = process_decoder_result(
            decoder_result_device, decoder_result_host, item_storage, processing_storage, n_sequence);
        n_new_items = insert_new_items(
            finished_indices, inp_device, inp_host, lengths_device,
            lengths_host, new_items_indices_device, new_items_indices_host, item_storage, processing_storage);
    }
}

void start_paged_attention_inference_engine(const TensorFloat& emb_table, const TensorFloat& pos_table,
    ItemStorage& item_storage, ProcessingStorage& processing_storage,
    MemoryBlockManager& memory_block_manager, PagedAttentionsManager& paged_attention_manager,
    PagedAttentionInferenceModel& inference_model, size_t n_batch_size, size_t n_sequence, int n_forward_rounds) {

    TensorInt inp_device({n_batch_size, n_sequence}, DeviceType::DEVICE);
    TensorInt inp_host({n_batch_size, n_sequence}, DeviceType::HOST);
    TensorInt lengths_device({n_batch_size}, DeviceType::DEVICE);
    TensorInt lengths_host({n_batch_size}, DeviceType::HOST);
    TensorInt new_items_indices_device({n_batch_size}, DeviceType::DEVICE);
    TensorInt new_items_indices_host({n_batch_size}, DeviceType::HOST);

    nvtxRangePushA("insert_new_items");
    get_global_throughput_counter().start_record();
    std::vector<int> new_item_indices = insert_new_items(
        inp_device, inp_host, lengths_device, lengths_host, new_items_indices_device, new_items_indices_host,
        item_storage, processing_storage, memory_block_manager, paged_attention_manager, n_forward_rounds);
    nvtxRangePop();
    
    TensorInt decoder_result_device({n_batch_size, 1}, DeviceType::DEVICE);
    TensorInt decoder_result_host({n_batch_size, 1}, DeviceType::HOST);

    std::vector<int> finished_indices;
    while (!is_done(item_storage, processing_storage)) {
        nvtxRangePushA("forward");
        inference_model.forward(inp_device, lengths_device, new_items_indices_device, decoder_result_device,
            new_item_indices.size(), emb_table, pos_table, paged_attention_manager.get_page_table_device());
        nvtxRangePop();
        nvtxRangePushA("process_decoder_result");
        finished_indices = process_decoder_result(
            decoder_result_device, decoder_result_host, item_storage, processing_storage, n_sequence);
        nvtxRangePop();
        nvtxRangePushA("allocate_or_free_memory_blocks_if_needed");
        allocate_or_free_memory_blocks_if_needed(paged_attention_manager, memory_block_manager, processing_storage, item_storage, finished_indices, n_forward_rounds);
        nvtxRangePop();
        nvtxRangePushA("insert_new_items");
        new_item_indices = insert_new_items(
            inp_device, inp_host, lengths_device, lengths_host, new_items_indices_device, new_items_indices_host,
            item_storage, processing_storage, memory_block_manager, paged_attention_manager, n_forward_rounds);
        nvtxRangePop();
    }
    get_global_throughput_counter().print_throughput();
}
