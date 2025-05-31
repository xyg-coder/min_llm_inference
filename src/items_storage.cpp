#include "items_storage.h"
#include "constants.h"
#include <cassert>
#include <utility>
#include <vector>

int Storage::size() const {
    return data_.size();
}

std::vector<IdTokensPair> ItemStorage::pop_finished_items(int size) {
    return finished_items_.pop_pairs(size);
}

std::vector<IdTokensPair> ItemStorage::pop_new_items(int size) {
    return new_items_.pop_pairs(size);
}

void append_token_to_id_string_pair(IdTokensPair& id_string_pair, int to_add) {
    id_string_pair.second.push_back(to_add);
}

void ItemStorage::add_finished_item(IdTokensPair&& finished_item) {
    finished_items_.add(std::move(finished_item));
}

void ItemStorage::add_new_item(IdTokensPair&& new_item) {
    new_items_.add(std::move(new_item));
}

int ItemStorage::finish_count() const {
    return finished_items_.size();
}

int ItemStorage::new_count() const {
    return new_items_.size();
}

std::vector<IdTokensPair> Storage::pop_pairs(int size) {
    std::vector<IdTokensPair> to_pop;
    for (int i = 0; i < size && !data_.empty(); ++i) {
        to_pop.push_back(data_.front());
        data_.erase(data_.begin());
    }
    return to_pop;
}

void ProcessingStorage::put(int batch_index, IdTokensPair&& tokens) {
    batch_id_to_token_pairs_[batch_index] = std::move(tokens);
}

void ProcessingStorage::remove(int batch_index) {
    auto it = batch_id_to_token_pairs_.find(batch_index);
    if (it != batch_id_to_token_pairs_.end()) {
        batch_id_to_token_pairs_.erase(batch_index);
    } 
}

IdTokensPair& ProcessingStorage::get_token(int batch_id) {
    return batch_id_to_token_pairs_[batch_id];
}

void ProcessingStorage::move_to_finished(int batch_id, ItemStorage& item_storage) {
    auto it = batch_id_to_token_pairs_.find(batch_id);
    item_storage.add_finished_item(std::move(it->second));
    batch_id_to_token_pairs_.erase(it);
}

void ProcessingStorage::move_to_new(int batch_id, ItemStorage& item_storage) {
    auto it = batch_id_to_token_pairs_.find(batch_id);
    item_storage.add_new_item_to_head(std::move(it->second));
    batch_id_to_token_pairs_.erase(it); 
}

void Storage::add(IdTokensPair&& to_add) {
    data_.push_back(std::move(to_add));
}

int Storage::head_length() const {
    return data_.begin()->second.size();
}

int ItemStorage::head_length() const {
    return new_items_.head_length();
}

bool ProcessingStorage::batch_id_processing(int batch_id) {
    return batch_id_to_token_pairs_.find(batch_id) != batch_id_to_token_pairs_.end();
}

std::vector<int> process_decoder_result(
    const TensorInt& decoder_result_device, TensorInt& decoder_result_host,
    ItemStorage& item_storage, ProcessingStorage& processing_storage, int n_sequence) {
    
    decoder_result_host.copy_from(decoder_result_device);
    const int* decode_result_data = decoder_result_host.data();
    std::vector<int> finished_indices;
    for (int i = 0; i < decoder_result_host.shape()[0]; ++i) {
        if (decode_result_data[i] == EMPTY_ROW_TOKEN_ID) {
            assert(!processing_storage.batch_id_processing(i));
            finished_indices.push_back(i);
        } else {
            append_token_to_id_string_pair(processing_storage.get_token(i), decode_result_data[i]);
            if (processing_storage.get_token(i).second.size() >= n_sequence
                || decode_result_data[i] == EOF_TOKEN_ID) {

                finished_indices.push_back(i);
                processing_storage.move_to_finished(i, item_storage);
            }
        }
    }
    return finished_indices;
}

int insert_new_items(
    const std::vector<int>& finished_indices, 
    TensorInt& inp_device, TensorInt& inp_host,
    TensorInt& lengths_device, TensorInt& lengths_host,
    TensorInt& new_items_indices_device, TensorInt& new_items_indices_host,
    ItemStorage& item_storage, ProcessingStorage& processing_storage) {
    
    if (finished_indices.empty()) {
        return 0;
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
        int batch_idx = finished_indices[i];

        if (i >= new_item_pairs.size()) {
            lengths_data[batch_idx] = 0;
        } else {
            assert(new_item_pairs[i].second.size() + 1 <= n_sequence);
            lengths_data[batch_idx] = new_item_pairs[i].second.size();
            std::copy(
                new_item_pairs[i].second.begin(), new_item_pairs[i].second.end(),
                inp_data + finished_indices[i] * n_sequence);
            processing_storage.put(batch_idx, std::move(new_item_pairs[i]));
        }
    }
    inp_device.copy_from(inp_host);
    lengths_device.copy_from(lengths_host);
    new_items_indices_device.copy_from(new_items_indices_host);

    return new_item_pairs.size();
}

int ProcessingStorage::size() const {
    return batch_id_to_token_pairs_.size();
}

bool is_done(ItemStorage& item_storage, ProcessingStorage& processing_storage) {
    return processing_storage.size() + item_storage.new_count() == 0;
}

void Storage::add_to_front(IdTokensPair&& pair) {
    data_.push_front(std::move(pair));
}

void ItemStorage::add_new_item_to_head(IdTokensPair&& pair) {
    new_items_.add_to_front(std::move(pair));
}
