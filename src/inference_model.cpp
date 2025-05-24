#include "inference_model.h"

void InferenceModel::forward(
    const TensorInt& inp, TensorInt& lengths, const TensorInt& new_item_indices, TensorInt& decoder_result) {
    
    // TODO: add comments for each part
    encoder_layer_.forward(
        emb_table_,
        pos_table_,
        inp,
        batch_embs_,
        lengths,
        new_item_indices);
    
    attention_layer_.forward(
        batch_embs_,
        lengths,
        new_item_indices,
        attention_result_);
    
    decoder_layer_.forward(
        attention_result_,
        emb_table_,
        pos_table_,
        batch_embs_,
        lengths,
        decoder_result);
}
