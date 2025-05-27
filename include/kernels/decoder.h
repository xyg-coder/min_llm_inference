#pragma once

#include "tensor.hpp"

/**
 * batch_result: [n_batch, input_dim]
 * emb_table: [n_vocab, input_dim]
 * emb_score: [n_batch, n_vocab]
 * inp_embedding: [n_batch, n_sequence, input_dim]
 * lengths: [n_batch]
 * decoder_result: [n_batch]
 *
 * 1. Multiply the batch_embs with emb_table
 * 2. Find the maximum of each batch, and save to decoder_result.
 * 3. Copy the emb_table of that index to inp[i_batch, lengths[i_batch], dim] (remember to add positional embedding)
 * 4. increase lengths[i_batch] by 1
 */
void launch_decoder(
    const TensorFloat& batch_result, const TensorFloat& emb_table,
    TensorFloat& emb_score,
    const TensorFloat& wpe_table,
    TensorFloat& inp_embedding, TensorInt& lengths, TensorInt& decoder_result);
