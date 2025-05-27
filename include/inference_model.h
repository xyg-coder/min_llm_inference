#pragma once

#include "layers.h"
#include "tensor.hpp"
#include "utils.h"


class InferenceModel : NonCopyableNonClonable {
public:
    InferenceModel(
        SelfAttentionLayer&&,
        EncoderLayer&&,
        DecoderLayer&&,
        const TensorFloat& emb_table,
        const TensorFloat& pos_table,
        size_t n_batch, size_t n_sequence, size_t emb_dim);

    void forward(const TensorInt& inp, TensorInt& lengths, const TensorInt& new_item_indices, TensorInt& decoder_result, int n_new_items);
private:
    SelfAttentionLayer attention_layer_;
    EncoderLayer encoder_layer_;
    DecoderLayer decoder_layer_;
    const TensorFloat& emb_table_;
    const TensorFloat& pos_table_;
    size_t n_batch_;
    size_t n_sequence_;
    size_t emb_dim_;
    TensorFloat inp_embedding_;
    TensorFloat attention_result_;
};
