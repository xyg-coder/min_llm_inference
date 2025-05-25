#include "items_storage.h"
#include "constants.h"
#include <cassert>
#include <vector>

std::vector<IdTokensPair> ItemStorage::pop_finished_items(int size) {
    return finished_items_.pop_pairs(size);
}

std::vector<IdTokensPair> ItemStorage::pop_new_items(int size) {
    return new_items_.pop_pairs(size);
}

void append_token_to_id_string_pair(IdTokensPair& id_string_pair, int to_add) {
    id_string_pair.second.push_back(to_add);
}

void ItemStorage::add_finished_items(const std::vector<IdTokensPair>& finished_items) {
    finished_items_.add(finished_items);
}

void ItemStorage::add_new_items(const std::vector<IdTokensPair>& new_items) {
    new_items_.add(new_items);
}

int ItemStorage::finish_count() const {
    return finished_items_.size();
}

std::vector<IdTokensPair> Storage::pop_pairs(int size) {
    std::vector<IdTokensPair> to_pop;
    for (int i = 0; i < size && !data_.empty(); ++i) {
        to_pop.push_back(data_.front());
        data_.erase(data_.begin());
    }
    return to_pop;
}

void Storage::add(const std::vector<IdTokensPair>& to_add) {
    for (const IdTokensPair& item : to_add) {
        data_.push_back(item);
    }
}

std::vector<int> process_decoder_result(
    const TensorInt& decoder_result_device, TensorInt& decoder_result_host,
    ItemStorage& item_storage, ProcessingStorage& processing_storage, int n_sequence) {
    
    decoder_result_host.copy_from(decoder_result_device);
    const int* decode_result_data = decoder_result_host.data();
    std::vector<IdTokensPair>& processing_items = processing_storage.get_processing_items();
    std::vector<int> finished_indices;
    std::vector<IdTokensPair> to_move_to_finished;
    for (int i = 0; i < decoder_result_host.shape()[0]; ++i) {
        if (decode_result_data[i] == EMPTY_ROW_TOKEN_ID) {
            finished_indices.push_back(i);
        } else if (decode_result_data[i] == EOF_TOKEN_ID) {
            append_token_to_id_string_pair(processing_items[i], decode_result_data[i]);
            finished_indices.push_back(i);
            to_move_to_finished.push_back(processing_items[i]);
        } else {
            append_token_to_id_string_pair(processing_items[i], decode_result_data[i]);
            if (processing_items[i].second.size() >= n_sequence) {
                finished_indices.push_back(i);
                to_move_to_finished.push_back(processing_items[i]);
            }
        }
    }
    item_storage.add_finished_items(to_move_to_finished);
    return finished_indices;
}

void insert_new_items(
    const std::vector<int>& finished_indices, 
    TensorInt& inp_device, TensorInt& inp_host,
    TensorInt& lengths_device, TensorInt& lengths_host,
    TensorInt& new_items_indices_device, TensorInt& new_items_indices_host,
    ItemStorage& item_storage) {
    
    if (finished_indices.empty()) {
        return;
    }
    
    std::vector<IdTokensPair> new_item_pairs = item_storage.pop_new_items(finished_indices.size());
    inp_host.copy_from(inp_device);
    lengths_host.copy_from(lengths_device);
    int n_sequence = inp_host.shape()[1];
    int* inp_data = inp_host.data();
    int* lengths_data = lengths_host.data();
    int* new_items_indices_data = new_items_indices_host.data();

    for (int i = 0; i < finished_indices.size(); ++i) {
        new_items_indices_data[i] = finished_indices[i];

        if (i >= new_item_pairs.size()) {
            lengths_data[i] = 0;
        } else {
            lengths_data[i] = new_item_pairs[i].second.size();
            std::copy(
                new_item_pairs[i].second.begin(), new_item_pairs[i].second.end(),
                new_items_indices_data + finished_indices[i] * n_sequence);
        }
    }
    inp_device.copy_from(inp_host);
    lengths_device.copy_from(lengths_host);
    new_items_indices_device.copy_from(new_items_indices_host);
}
