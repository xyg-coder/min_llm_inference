#pragma once

#include "tensor.hpp"
#include <list>
#include <string>
#include <vector>

using IdStringPair = std::pair<int, std::string>;


class ItemsStorage {
public:
    ItemsStorage(const ItemsStorage&) = delete;
    ItemsStorage& operator=(const ItemsStorage&) = delete;
    std::vector<IdStringPair> get_pairs(int size);
    void insert_one(const IdStringPair&);
private:
    std::list<IdStringPair> data_;
};


// There will be one single instance of ItemsHandler
class ItemsHandler {
public:
    ItemsHandler(const ItemsHandler&) = delete;
    ItemsHandler& operator=(const ItemsHandler&) = delete;
    // return at most size finished items
    std::vector<IdStringPair> get_finished_items(int size);
    // return at most size new items
    std::vector<IdStringPair> get_new_items(int size);
    void add_one_finished_item(const IdStringPair&);
    bool is_finished() const;

    /**
     * assert decoder_result.size() == processing_items_.size()
     * 1. for each processing_items_, append decoder_result to the end
     * 2. check the length of processing_items_ or if meets the ending token or special_token(-1 means this is empty item)
     * 3. insert those finished items to finished_items_
     * 4. returns the indices ([0, n_batch-1]) of the finished results
     */
    std::vector<int> process_decoder_result_and_check_finished(const std::vector<std::string>& decoder_result);
private:
    ItemsStorage finished_items_;
    ItemsStorage new_items_;
    // processing_items_has size [n_batch]
    std::vector<IdStringPair> processing_items_;
};


/**
 * 1. Clone the token_ids from decoder_result_device to host, and query the global token_mapping (list<string>) to fine the next tokens for each row
 * 2. get the finished_items_indices by calling itemsHandler.process_decoder_result_and_check_finished
 * 3. call itemsHandler.get_new_items to get at most new_items whose size == finished_items_indices.size() 
 * 4. return the indices of finished_results
 */
std::vector<int> process_decoder_result(
    const TensorInt& decoder_result_device,
    TensorInt& decoder_result_host,
    std::vector<std::string>& decoder_result);


/**
 * 1. fetch new_items with finished_indices.size()
 * 2. fill new_tokens and new_lengths. Note, we have to fill n_finished_items even there isn't enough new_items. But padding it with length==0
 * After this step, we are ready to call encoder
 */
void insert_new_items(
    const std::vector<int>& finished_indices, TensorInt& new_items, TensorInt& new_lengths,
    TensorInt& new_items_indices);
    
