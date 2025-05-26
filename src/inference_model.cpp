#include "inference_model.h"

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
