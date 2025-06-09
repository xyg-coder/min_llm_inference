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
    SelfAttentionLayer(TensorFloat&& wk, TensorFloat&& wq, TensorFloat&& wv,
        size_t n_batch, size_t input_dim, size_t n_sequence);
    void forward(const TensorFloat& inp_embedding, const TensorInt& lengths,
        const TensorInt& new_batch_idx, TensorFloat& attention_result, int n_new_items);
private:
    TensorFloat wk_;
    TensorFloat wq_;
    TensorFloat wv_;
    // pre-allocated memories
    TensorFloat kt_cache_;
    TensorFloat v_cache_;
    TensorFloat q_output_;
    TensorFloat qkt_output_;
};


class PagedAttentionLayer : public NonCopyableNonClonable {
public:
    PagedAttentionLayer(TensorFloat&& wk, TensorFloat&& wq, TensorFloat&& wv,
        size_t n_batch, size_t input_dim, size_t n_sequence);
    void forward(TensorFloatPoint& page_table, const TensorInt& lengths,
        const TensorInt& new_batch_idx, TensorFloat& attention_result, int n_new_items);
private:
    TensorFloat wk_;
    TensorFloat wq_;
    TensorFloat wv_;
    TensorFloat q_output_;
    TensorFloat qkt_output_;
};


/**
 * emb_table: [n_vocab, inputDim]
 * wpe: [n_sequence, inputDim]
 * inp: [n_batch, n_sequence]
 * inp_embedding: [n_batch, n_sequence, inputDim]
 * lengths: [n_batch]
 * new_item_indices: [n_new_items]
 */
class EncoderLayer : public NonCopyableNonClonable {
public:
    void forward(
        const TensorFloat& emb_table, const TensorFloat& pos_emb, const TensorInt& inp,
        TensorFloat& inp_embedding, const TensorInt& lengths,
        const TensorInt& new_item_indices, int n_new_items);
};

/**
 * emb_table: [n_vocab, emb_dim]
 * wpe: [n_sequence, emb_dim]
 * inp: [n_batch, n_sequence]
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * lengths: [n_batch]
 * new_item_indices: [n_new_items]
 */
class PagedEncoderLayer : public NonCopyableNonClonable {
public:
    void forward(const TensorFloat& emb_table, const TensorFloat& pos_emb, const TensorInt& inp,
        TensorFloatPoint& page_table, const TensorInt& lengths,
        const TensorInt& new_item_indices, int n_new_items);
};

/**
 * batch_result: [n_batch, input_dim]
 * emb_table: [n_vocab, input_dim]
 * emb_score: [n_batch, n_vocab]
 * inp_embedding: [n_batch, n_sequence, input_dim]
 * lengths: [n_batch]
 * decoder_result: [n_batch]
 */
class DecoderLayer : public NonCopyableNonClonable {
public:
    DecoderLayer(size_t n_batch, size_t n_vocab);
    void forward(
        const TensorFloat& batch_result, const TensorFloat& emb_table,
        const TensorFloat& wpe_table,
        TensorFloat& inp_embedding, TensorInt& lengths, TensorInt& decoder_result);
private:
    TensorFloat emb_score_;
};

/**
 * batch_result: [n_batch, input_dim]
 * emb_table: [n_vocab, input_dim]
 * emb_score: [n_batch, n_vocab]
 * inp_embedding: [n_batch, n_sequence, input_dim]
 * lengths: [n_batch]
 * decoder_result: [n_batch]
 */
class PagedDecoderLayer : public NonCopyableNonClonable {
public:
    PagedDecoderLayer(size_t n_batch, size_t n_vocab);
    void forward(
        const TensorFloat& batch_result, const TensorFloat& emb_table,
        const TensorFloat& wpe_table,
        TensorFloatPoint& page_table, TensorInt& lengths, TensorInt& decoder_result);
private:
    TensorFloat emb_score_;
};
