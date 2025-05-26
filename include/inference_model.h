#pragma once

#include "layers.h"
#include "tensor.hpp"


class InferenceModel {
public:
    InferenceModel(
        const SelfAttentionLayer&,
        const EncoderLayer&,
        const DecoderLayer&,
        const TensorFloat& emb_table,
        const TensorFloat& pos_table);

    void forward(const TensorInt& inp, TensorInt& lengths, const TensorInt& new_item_indices, TensorInt& decoder_result, int n_new_items);
private:
    SelfAttentionLayer attention_layer_;
    EncoderLayer encoder_layer_;
    DecoderLayer decoder_layer_;
    TensorFloat emb_table_;
    TensorFloat pos_table_;
    int n_batch_;
    int n_sequence_;
    int emb_dim_;
    TensorFloat inp_embedding_;
    TensorFloat attention_result_;
};
