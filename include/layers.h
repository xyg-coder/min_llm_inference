#pragma once

#include "tensor.hpp"
#include "utils.h"
#include <optional>


class FeedForward : public NonCopyableNonClonable {
public:
    FeedForward(const TensorFloat& w, std::optional<const TensorFloat> b = std::nullopt);
    void forward(const TensorFloat& input, TensorFloat& output);
private:
    TensorFloat weight_;
    std::optional<TensorFloat> bias_;
};


class SelfAttentionLayer : public NonCopyableNonClonable {
public:
    SelfAttentionLayer(const TensorFloat& wk, const TensorFloat& wq, const TensorFloat& wv);
    void forward(const TensorFloat& inp, const TensorInt& lengths,
        const TensorInt& new_batch_idx, TensorFloat& attention_result);
private:
    TensorFloat wk;
    TensorFloat wq;
    TensorFloat wv;
    TensorFloat kt_cache;
    TensorFloat v_cache;
    // pre-allocated memories
    TensorFloat q_output_;
    TensorFloat qkt_output_;
};


/**
 * emb_table: [n_vocab, inputDim]
 * wpe: [n_sequence, inputDim]
 * inp: [n_batch, n_sequence]
 * output: [n_batch, n_sequence, inputDim]
 * lengths: [n_batch]
 * new_item_indices: [n_new_items]
 */
class EncoderLayer : public NonCopyableNonClonable {
public:
    void forward(
        const TensorFloat& emb_table, const TensorFloat& pos_emb, const TensorInt& inp,
        TensorFloat& output, const TensorInt& lengths,
        const TensorInt& new_item_indices);
};

/**
 * batch_embs: [n_batch, input_dim]
 * emb_table: [n_vocab, input_dim]
 * emb_score: [n_batch, n_vocab]
 * inp: [n_batch, n_sequence, input_dim]
 * lengths: [n_batch]
 * decoder_result: [n_batch]
 */
class DecoderLayer : public NonCopyableNonClonable {
public:
    void forward(
        const TensorFloat& batch_embs, const TensorFloat& emb_table,
        const TensorFloat& wpe_table,
        TensorFloat& inp, TensorInt& lengths, TensorInt& decoder_result);
private:
    TensorFloat emb_score_;
};
