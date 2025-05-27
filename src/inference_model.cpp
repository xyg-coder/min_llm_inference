#include "inference_model.h"

InferenceModel::InferenceModel(
    SelfAttentionLayer&& attention_layer,
    EncoderLayer&& encoder_layer,
    DecoderLayer&& decoder_layer,
    const TensorFloat& emb_table,
    const TensorFloat& pos_table,
    size_t n_batch, size_t n_sequence, size_t emb_dim): attention_layer_(std::move(attention_layer)),
        encoder_layer_(std::move(encoder_layer)), decoder_layer_(std::move(decoder_layer)),
        emb_table_(emb_table), pos_table_(pos_table),
        n_batch_(n_batch), n_sequence_(n_sequence), emb_dim_(emb_dim),
        inp_embedding_(TensorFloat({n_batch, n_sequence, emb_dim}, DeviceType::DEVICE)),
        attention_result_(TensorFloat({n_batch, emb_dim}, DeviceType::DEVICE)) { }

void InferenceModel::forward(
    const TensorInt& inp, TensorInt& lengths, const TensorInt& new_item_indices, TensorInt& decoder_result, int n_new_items) {
    
    encoder_layer_.forward(
        emb_table_,
        pos_table_,
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
        emb_table_,
        pos_table_,
        inp_embedding_,
        lengths,
        decoder_result);
}
