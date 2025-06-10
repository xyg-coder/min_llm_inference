#include "inference_model.h"
#include "tensor.hpp"

InferenceModel::InferenceModel(
    SelfAttentionLayer&& attention_layer,
    EncoderLayer&& encoder_layer,
    DecoderLayer&& decoder_layer,
    size_t n_batch, size_t n_sequence, size_t emb_dim): attention_layer_(std::move(attention_layer)),
        encoder_layer_(std::move(encoder_layer)), decoder_layer_(std::move(decoder_layer)),
        n_batch_(n_batch), n_sequence_(n_sequence), emb_dim_(emb_dim),
        inp_embedding_(TensorFloat({n_batch, n_sequence, emb_dim}, DeviceType::DEVICE)),
        attention_result_(TensorFloat({n_batch, emb_dim}, DeviceType::DEVICE)) { }

void InferenceModel::forward(
    const TensorInt& inp, TensorInt& lengths, const TensorInt& new_item_indices, TensorInt& decoder_result, int n_new_items,
    const TensorFloat& emb_table, const TensorFloat& pos_emb_table) {
    
    encoder_layer_.forward(
        emb_table,
        pos_emb_table,
        inp,
        inp_embedding_,
        lengths,
        new_item_indices, n_new_items);
    
    attention_layer_.forward(
        inp_embedding_,
        lengths,
        new_item_indices,
        attention_result_, n_new_items);
    
    decoder_layer_.forward(
        attention_result_,
        emb_table,
        pos_emb_table,
        inp_embedding_,
        lengths,
        decoder_result);
}


PagedAttentionInferenceModel::PagedAttentionInferenceModel(
    PagedAttentionLayer&& attention_layer,
    PagedEncoderLayer&& encoder_layer,
    PagedDecoderLayer&& decoder_layer,
    size_t n_batch, size_t n_sequence, size_t emb_dim): paged_attention_layer_(std::move(attention_layer)),
        paged_encoder_layer_(std::move(encoder_layer)), paged_decoder_layer_(std::move(decoder_layer)),
        n_batch_(n_batch), n_sequence_(n_sequence), emb_dim_(emb_dim),
        attention_result_(TensorFloat({n_batch, emb_dim}, DeviceType::DEVICE)) { }


void PagedAttentionInferenceModel::forward(
    const TensorInt& inp, TensorInt& lengths, const TensorInt& new_item_indices, TensorInt& decoder_result, int n_new_items,
    const TensorFloat& emb_table, const TensorFloat& pos_emb_table, TensorFloatPoint& page_table) {
    
    paged_encoder_layer_.forward(
        emb_table,
        pos_emb_table,
        inp,
        page_table,
        lengths,
        new_item_indices, n_new_items);
    
    paged_attention_layer_.forward(
        page_table,
        lengths,
        new_item_indices,
        attention_result_, n_new_items);
    
    paged_decoder_layer_.forward(
        attention_result_,
        emb_table,
        pos_emb_table,
        page_table,
        lengths,
        decoder_result);
}