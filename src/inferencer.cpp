#include "inferencer.h"
#include "inference_model.h"
#include "items_storage.h"
#include "kernels/rand_assign.h"
#include "tensor.hpp"
#include <vector>

constexpr int N_VOCAB = 20480;
constexpr int DIMS = 1024;
constexpr int N_SEQUENCE = 1024;
constexpr int N_BATCH_SIZE = 2048;

TensorFloat mock_emb_table() {
    TensorFloat device_tensor({N_VOCAB, DIMS}, DeviceType::DEVICE);
    launch_randn_kernel(device_tensor.data(), device_tensor.get_total_size());
    return device_tensor;
}

TensorFloat mock_pos_table() {
    TensorFloat device_tensor({N_SEQUENCE, DIMS}, DeviceType::DEVICE);
    launch_randn_kernel(device_tensor.data(), device_tensor.get_total_size());
    return device_tensor;
}

void start_inference_engine(
    const TensorFloat& emb_table, const TensorFloat& pos_table,
    ItemStorage& item_storage, ProcessingStorage& processing_storage,
    InferenceModel& inference_model) {

    std::vector<int> finished_indices;
    for (int i = 0; i < N_BATCH_SIZE; ++i) {
        finished_indices.push_back(i);
    }
    TensorInt inp_device({N_BATCH_SIZE, N_SEQUENCE}, DeviceType::DEVICE);
    TensorInt inp_host({N_BATCH_SIZE, N_SEQUENCE}, DeviceType::DEVICE);
    TensorInt lengths_device({N_BATCH_SIZE}, DeviceType::DEVICE);
    TensorInt lengths_host({N_BATCH_SIZE}, DeviceType::HOST);
    TensorInt new_items_indices_device({N_BATCH_SIZE}, DeviceType::DEVICE);
    TensorInt new_items_indices_host({N_BATCH_SIZE}, DeviceType::HOST);
    insert_new_items(
        finished_indices, inp_device, inp_host, lengths_device,
        lengths_host, new_items_indices_device, new_items_indices_host, item_storage);
    TensorInt decoder_result_device({N_BATCH_SIZE}, DeviceType::DEVICE);
    TensorInt decoder_result_host({N_BATCH_SIZE}, DeviceType::HOST);
    
    while (!is_done(item_storage, processing_storage)) {
        inference_model.forward(inp_device, lengths_device, new_items_indices_device, decoder_result_device);
        finished_indices = process_decoder_result(
            decoder_result_device, decoder_result_host, item_storage, processing_storage);
        insert_new_items(
            finished_indices, inp_device, inp_host, lengths_device,
            lengths_host, new_items_indices_device, new_items_indices_host, item_storage);
    }

}
