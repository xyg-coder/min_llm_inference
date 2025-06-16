#pragma once

#include "tensor.hpp"
#include "utils.h"
#include <list>
#include <unordered_map>
#include <vector>

using IdTokensPair = std::pair<int, std::vector<int>>;


class Storage : public NonCopyableNonClonable {
public:
    Storage() = default;
    std::vector<IdTokensPair> pop_pairs(int size);
    void add(IdTokensPair&&);
    void add_to_front(IdTokensPair&&);
    int size() const;
    int head_length() const;
    const IdTokensPair& get_top() const;
    const std::list<IdTokensPair>& get_data() const;
private:
    std::list<IdTokensPair> data_;
};


class ItemStorage : public NonCopyableNonClonable {
public:
    ItemStorage() = default;
    // return at most size finished items
    std::vector<IdTokensPair> pop_finished_items(int size);
    // return at most size new items
    std::vector<IdTokensPair> pop_new_items(int size);
    const IdTokensPair& get_top() const;
    void add_finished_item(IdTokensPair&&);
    void add_new_item(IdTokensPair&&);
    // add new items to head, this is for paged attention preemption
    void add_new_item_to_head(IdTokensPair&&);
    int finish_count() const;
    int new_count() const;
    // the length of the new_items_.begin()
    int head_length() const;
    const std::list<IdTokensPair>& get_finished_items() const;
private:
    Storage finished_items_;
    Storage new_items_; 
};


class ProcessingStorage : public NonCopyableNonClonable {
public:
    ProcessingStorage() = default;
    void put(int batch_id, IdTokensPair&&);
    void remove(int batch_id);
    bool batch_id_processing(int batch_id);
    IdTokensPair& get_token(int batch_id);
    void move_to_new(int batch_id, ItemStorage& item_storage);
    int size() const;
    void move_to_finished(int batch_id, ItemStorage& item_storage);
private:
    std::unordered_map<int, IdTokensPair> batch_id_to_token_pairs_;
};

void append_token_to_id_string_pair(IdTokensPair& id_string_pair, int to_add);

/**
 * 1. Clone the token_ids from decoder_result_device to host, and query the global token_mapping (list<string>) to fine the next tokens for each row
 * 2. Check the decoder result to see if this row is finished or not.
 * 3. return the indices that the new items can be inserted to
 */
std::vector<int> process_decoder_result(
    const TensorInt& decoder_result_device, TensorInt& decoder_result_host,
    ItemStorage& item_storage, ProcessingStorage& processing_storage, int n_sequence);


/**
 * 1. fetch new_items with finished_indices.size()
 * 2. fill new_tokens and new_lengths. Note, we have to fill n_finished_items even there isn't enough new_items. But padding it with length==0
 * 
 *  return n_new_items
 */
int insert_new_items(
    const std::vector<int>& finished_indices, 
    TensorInt& inp_device, TensorInt& inp_host,
    TensorInt& lengths_device, TensorInt& lengths_host,
    TensorInt& new_items_indices_device, TensorInt& new_items_indices_host,
    ItemStorage& item_storage, ProcessingStorage& processing_storage);
    

bool is_done(ItemStorage& item_storage, ProcessingStorage& processing_storage);
