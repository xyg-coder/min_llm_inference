#pragma once

#include "inference_model.h"
#include "item_storage.h"
#include "paged_item_storage.h"
#include "tensor.hpp"

/**
 * @brief Runs the main inference engine loop.
 *
 * This function orchestrates the inference process by:
 * - Loading new items from the item storage.
 * - Sending these items to the inference model for processing.
 * - Identifying which items have completed inference and marking them as finished.
 * - Refilling the processing queue with new items to replace those that have finished.
 * - Repeating the process until all items are processed or a stopping condition is met.
 */
void start_inference_engine(const TensorFloat& emb_table, const TensorFloat& pos_table,
    ItemStorage& item_storage, ProcessingStorage& processing_storage,
    InferenceModel& inference_model, size_t n_batch_size, size_t n_sequence);


void start_paged_attention_inference_engine(const TensorFloat& emb_table, const TensorFloat& pos_table,
    ItemStorage& item_storage, ProcessingStorage& processing_storage,
    MemoryBlockManager& memory_block_manager, PagedAttentionsManager& paged_attention_manager,
    PagedAttentionInferenceModel& inference_model, size_t n_batch_size, size_t n_sequence, int n_forward_rounds);
