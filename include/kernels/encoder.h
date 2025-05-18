#pragma once

void launch_encoder_kernel(
    const float* wte, const float* wpe, const int* inp, float* output, int batch_size, int n_sequence, int embedding_dim);

/**
 * only calculates the embedding for new batches. And will set the corresponding new_lengths to total_lengths.
 * 
 * emb_table: [n_vocab, inputDim]
 * wpe: [n_sequence, inputDim]
 * inp: [n_batch, n_sequence]
 * output: [total_batch_size, n_sequence, inputDim]
 * new_lengths: [n_batch]
 * total_lengths: [total_batch_size]
 * target_batch_indices: [n_batch]
 */
void launch_inference_optimized_encoder_kernel(
    const float* emb_table, const float* wpe, const int* inp, float* output, const int* new_lengths,
    const int* target_batch_indices, int* total_lengths,
    int batch_size, int n_sequence, int embedding_dim);
