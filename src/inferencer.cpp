#include "inferencer.h"
#include "items_storage.h"
#include "kernels/rand_assign.h"
#include "tensor.hpp"
#include <string>
#include <vector>

constexpr int N_VOCAB = 20480;
constexpr int DIMS = 1024;
constexpr int N_SEQUENCE = 1024;

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

void start_inference_engine() {
    // ItemsHandler itemsHandler;
    // TensorFloat embedding_table({N_VOCAB, DIMS}, DeviceType::DEVICE);
    // std::vector<std::string> id_to_string;
    // std::vector<int> finished_indices;  // fill from 0 to n_batches



    // while (!itemsHandler.is_finished()) {
    //     insert_new_items
    //     encoder_launch
    //     result = model.forward()
    //     decoder_result = decoder
    //     finished_indices = process_decoder_result
    // }

}
