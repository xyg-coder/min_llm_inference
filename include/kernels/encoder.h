#pragma once

void launch_encoder_kernel(
    const float* wte, const float* wpe, const int* inp, float* output, int batch_size, int n_sequence, int embedding_dim);

/**
 * only calculates the embedding for new batches.
 * 
 * emb_table: [n_vocab, inputDim]
 * wpe: [n_sequence, inputDim]
 * inp: [n_batch, n_sequence]
 * output: [n_batch, n_sequence, inputDim]
 * lengths: [n_batch]
 * new_item_indices: [n_new_items]
 */
void launch_inference_optimized_encoder_kernel(
    const float* emb_table, const float* wpe, const int* inp, float* output, const int* lengths,
    const int* new_item_indices,
    int batch_size, int n_sequence, int embedding_dim, int n_new_items);
