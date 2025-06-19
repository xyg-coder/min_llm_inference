#pragma once

#include "layers.h"
#include "tensor.hpp"
#include "utils.h"


class InferenceModel : public NonCopyableNonClonable {
public:
    InferenceModel(
        SelfAttentionLayer&&,
        EncoderLayer&&,
        DecoderLayer&&,
        size_t n_batch, size_t n_sequence, size_t emb_dim);

    void forward(
        const TensorInt& inp, TensorInt& lengths, const TensorInt& new_item_indices, TensorInt& decoder_result, int n_new_items,
        const TensorFloat& emb_table, const TensorFloat& pos_emb_table);
private:
    SelfAttentionLayer attention_layer_;
    EncoderLayer encoder_layer_;
    DecoderLayer decoder_layer_;
    size_t n_batch_;
    size_t n_sequence_;
    size_t emb_dim_;
    TensorFloat inp_embedding_;
    TensorFloat attention_result_;
};


class PagedAttentionInferenceModel : public NonCopyableNonClonable {
public:
    PagedAttentionInferenceModel(
        PagedAttentionLayer&&,
        PagedEncoderLayer&&,
        PagedDecoderLayer&&,
        size_t n_batch, size_t n_sequence, size_t emb_dim, int n_forward_rounds);

    void forward(
        const TensorInt& inp, TensorInt& lengths, const TensorInt& new_item_indices, TensorInt& decoder_result, int n_new_items,
        const TensorFloat& emb_table, const TensorFloat& pos_emb_table, TensorFloatPoint& page_table);
private:
    PagedAttentionLayer paged_attention_layer_;
    PagedEncoderLayer paged_encoder_layer_;
    PagedDecoderLayer paged_decoder_layer_;
    size_t n_batch_;
    size_t n_sequence_;
    size_t emb_dim_;
    TensorFloat attention_result_;
    int n_forward_rounds_;
};


class PagedAttentionCublasInferenceModel : public NonCopyableNonClonable {
public:
    PagedAttentionCublasInferenceModel(
        PagedAttentionCublasLayer&&,
        PagedEncoderLayer&&,
        PagedCublasDecoderLayer&&,
        size_t n_batch, size_t n_sequence, size_t emb_dim, int n_forward_rounds);

    void forward(
        const TensorInt& inp, TensorInt& lengths, const TensorInt& new_item_indices, TensorInt& decoder_result, int n_new_items,
        const TensorFloat& emb_table, const TensorFloat& pos_emb_table, TensorFloatPoint& page_table, cublasHandle_t handle);
private:
    PagedAttentionCublasLayer paged_attention_layer_;
    PagedEncoderLayer paged_encoder_layer_;
    PagedCublasDecoderLayer paged_decoder_layer_;
    size_t n_batch_;
    size_t n_sequence_;
    size_t emb_dim_;
    TensorFloat attention_result_;
    int n_forward_rounds_;
};
