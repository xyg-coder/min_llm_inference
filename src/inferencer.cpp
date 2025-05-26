#include "inferencer.h"
#include "inference_model.h"
#include "items_storage.h"
#include "tensor.hpp"
#include <vector>


void start_inference_engine(
    const TensorFloat& emb_table, const TensorFloat& pos_table,
    ItemStorage& item_storage, ProcessingStorage& processing_storage,
    InferenceModel& inference_model,
    size_t n_batch_size, size_t n_sequence) {

    std::vector<int> finished_indices;
    for (int i = 0; i < n_batch_size; ++i) {
        finished_indices.push_back(i);
    }
    TensorInt inp_device({n_batch_size, n_sequence}, DeviceType::DEVICE);
    TensorInt inp_host({n_batch_size, n_sequence}, DeviceType::DEVICE);
    TensorInt lengths_device({n_batch_size}, DeviceType::DEVICE);
    TensorInt lengths_host({n_batch_size}, DeviceType::HOST);
    TensorInt new_items_indices_device({n_batch_size}, DeviceType::DEVICE);
    TensorInt new_items_indices_host({n_batch_size}, DeviceType::HOST);
    insert_new_items(
        finished_indices, inp_device, inp_host, lengths_device,
        lengths_host, new_items_indices_device, new_items_indices_host, item_storage, processing_storage);
    TensorInt decoder_result_device({n_batch_size}, DeviceType::DEVICE);
    TensorInt decoder_result_host({n_batch_size}, DeviceType::HOST);
    
    while (!is_done(item_storage, processing_storage)) {
        inference_model.forward(inp_device, lengths_device, new_items_indices_device, decoder_result_device);
        finished_indices = process_decoder_result(
            decoder_result_device, decoder_result_host, item_storage, processing_storage, n_sequence);
        insert_new_items(
            finished_indices, inp_device, inp_host, lengths_device,
            lengths_host, new_items_indices_device, new_items_indices_host, item_storage, processing_storage);
    }

}
