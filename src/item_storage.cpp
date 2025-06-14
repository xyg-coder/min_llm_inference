#include "item_storage.h"
#include "constants.h"
#include "throughput_counter.h"
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
    if (size > data_.size()) {
        size = data_.size();
    }
    std::vector<IdTokensPair> result;
    auto it = data_.begin();
    for (int i = 0; i < size; ++i) {
        result.push_back(std::move(*it));
        it = data_.erase(it);
    }
    return result;
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
    
    assert(decoder_result_host.shape().size() == 2);
    int n_batch = decoder_result_host.shape()[0];
    int n_decode_results = decoder_result_host.shape()[1];
    decoder_result_host.copy_from(decoder_result_device);
    const int* decode_result_data = decoder_result_host.data();
    std::vector<int> finished_indices;
    int total_processed_tokens = 0;
    for (int i = 0; i < n_batch; ++i) {
        bool empty = false;
        bool finished = false;
        for (int j = 0; j < n_decode_results; ++j) {
            int decoder_result = decode_result_data[i * n_decode_results + j];
            if (decoder_result == EMPTY_ROW_TOKEN_ID) {
                empty = true;
            } else {
                append_token_to_id_string_pair(processing_storage.get_token(i), decoder_result);
                total_processed_tokens++;
                if (processing_storage.get_token(i).second.size() >= n_sequence
                    || decoder_result == EOF_TOKEN_ID) {

                    finished = true;
                }
            }
            if (finished || empty) {
                break;
            }
        }
        if (finished || empty) {
            finished_indices.push_back(i);
        }
        if (finished) {
            processing_storage.move_to_finished(i, item_storage);
        }
    }
    get_global_throughput_counter().add_record_if_recording(total_processed_tokens);
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

const IdTokensPair& ItemStorage::get_top() const {
    return new_items_.get_top();
}

const IdTokensPair& Storage::get_top() const {
    return data_.front();
}

const std::list<IdTokensPair>& ItemStorage::get_finished_items() const {
    return finished_items_.get_data();
}

const std::list<IdTokensPair>& Storage::get_data() const {
    return data_;
}
