#pragma once

#include "inference_model.h"
#include "items_storage.h"
#include "tensor.hpp"

void start_inference_engine(const TensorFloat& emb_table, const TensorFloat& pos_table,
    ItemStorage& item_storage, ProcessingStorage& processing_storage,
    InferenceModel& inference_model, size_t n_batch_size, size_t n_sequence);
