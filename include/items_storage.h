#pragma once

#include "tensor.hpp"
#include <list>
#include <string>
#include <vector>

using IdTokensPair = std::pair<int, std::vector<int>>;


class ItemsStorage {
public:
    ItemsStorage() = default;
    ItemsStorage(const ItemsStorage&) = delete;
    ItemsStorage& operator=(const ItemsStorage&) = delete;
    std::vector<IdTokensPair> get_pairs(int size);
    void insert_one(const IdTokensPair&);
private:
    std::list<IdTokensPair> data_;
};


// global singleton items
class GlobalSingleton {
public:
    static GlobalSingleton& get_instance();
    // return at most size finished items
    std::vector<IdTokensPair> pop_finished_items(int size);
    // return at most size new items
    std::vector<IdTokensPair> pop_new_items(int size);
    void add_finished_items(const std::vector<IdTokensPair>&);
    void add_new_items(const std::vector<IdTokensPair>&);
private:
    GlobalSingleton() = default;
    GlobalSingleton(const GlobalSingleton&) = delete;
    GlobalSingleton& operator=(const GlobalSingleton&) = delete;
    ItemsStorage finished_items_;
    ItemsStorage new_items_; 
};


// There will be one single instance of ItemsHandler
class ThreadSingleton {
public:
    static ThreadSingleton& get_instance();
    std::vector<IdTokensPair>& get_processing_items();
private:
    ThreadSingleton() = default;
    ThreadSingleton(const ThreadSingleton&) = delete;
    ThreadSingleton& operator=(const ThreadSingleton&) = delete;
    // processing_items_has size [n_batch]
    std::vector<IdTokensPair> processing_items_;
};


/**
 * 1. Clone the token_ids from decoder_result_device to host, and query the global token_mapping (list<string>) to fine the next tokens for each row
 * 2. Check the decoder result to see if this row is finished or not.
 * 3. return the indices that the new items can be inserted to
 */
std::vector<int> process_decoder_result(
    const TensorInt& decoder_result_device,
    TensorInt& decoder_result_host,
    const std::vector<std::string>& decoder_result);


/**
 * 1. fetch new_items with finished_indices.size()
 * 2. fill new_tokens and new_lengths. Note, we have to fill n_finished_items even there isn't enough new_items. But padding it with length==0
 * After this step, we are ready to call encoder
 */
void insert_new_items(
    const std::vector<int>& finished_indices, 
    TensorInt& inp_device, TensorInt& inp_host,
    TensorInt& lengths_device, TensorInt& lengths_host,
    TensorInt& new_items_indices_device, TensorInt& new_items_indices_host);
    
