#include "constants.h"
#include "kernels/encoder.h"
#include <cassert>
#include <vector_types.h>
#include "kernels/utils.cuh"
#include "utils.h"


// use of float4 leads to using 128-bit LDG / STG instructions in SASS,
// very helpful in memory-bound kernels
__global__ void encoder_kernel(
    const float *wte, const float *wpe, const int* inp, float* output,
    int batch_size, int n_sequence, int embedding_dim) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int embedding_dim_4 = embedding_dim / 4;
    if (idx >= batch_size * n_sequence * embedding_dim_4) {
        return;
    }

    int embedding_dim_idx = idx % embedding_dim_4;
    int batch_sequence = idx / embedding_dim_4;
    int sequence_index = batch_sequence % n_sequence;
    int inp_index = inp[batch_sequence];
    // output[idx] = wte[inp_index * embedding_dim_4 + inp % embedding_dim_4] + wpe[i_sequence + inp % embedding_dim_4]
    float4* output_4 = reinterpret_cast<float4*>(output);
    const float4* wte_4 = reinterpret_cast<const float4*>(wte);
    const float4* wpe_4 = reinterpret_cast<const float4*>(wpe);
    output_4[idx] = float4_add(wte_4[inp_index * embedding_dim_4 + embedding_dim_idx], wpe_4[sequence_index * embedding_dim_4 + embedding_dim_idx]);
}

/** 
 * wte: [vocab_size, embedding_dim]
 * wpe: [max_sequence, embedding_dim]
 * inp: [batch_size, n_sequence]
 * output: [batch_size, n_sequence, embedding_dim]
 */
void launch_encoder_kernel(
    const float *wte, const float *wpe, const int* inp, float* output,
    int batch_size, int n_sequence, int embedding_dim) {

    assert(embedding_dim % 4 == 0);
    int grid_dim = ceil_div(batch_size * n_sequence * embedding_dim / 4, BLOCK_DIM);
    encoder_kernel<<<grid_dim, BLOCK_DIM>>>(wte, wpe, inp, output, batch_size, n_sequence, embedding_dim);
    CUDA_CHECK_LAST();
}

/**
 * emb_table: [n_vocab, inputDim]
 * wpe: [n_sequence, inputDim]
 * inp: [n_batch, n_sequence]
 * output: [n_batch, n_sequence, inputDim]
 * lengths: [n_batch]
 * new_item_indices: [n_new_items]
 */
__global__ void inference_optimized_encoder(
    const float* emb_table, const float* wpe, const int* inp, float* output, const int* lengths,
    const int* new_item_indices,
    int batch_size, int n_sequence, int embedding_dim, int n_new_items) {

    assert(embedding_dim % 4 == 0);
    int i_batch = new_item_indices[blockIdx.z];
    int batch_len = lengths[i_batch];
    if (blockIdx.y >= batch_len) {
        return;
    }
    int token_id = inp[i_batch * n_sequence + blockIdx.y];
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim_idx >= embedding_dim / 4) {
        return;
    }

    const float4* emb_table_start = reinterpret_cast<const float4*>(emb_table + token_id * embedding_dim);
    const float4* wpe_table_start = reinterpret_cast<const float4*>(wpe + blockIdx.y * embedding_dim);
    float4* output_4 = reinterpret_cast<float4*>(output + i_batch * embedding_dim * n_sequence + blockIdx.y * embedding_dim);
    output_4[dim_idx] = float4_add(emb_table_start[dim_idx], wpe_table_start[dim_idx]);
}


void launch_inference_optimized_encoder_kernel(
    const float* emb_table, const float* wpe, const int* inp, float* output, const int* lengths,
    const int* new_item_indices,
    int batch_size, int n_sequence, int embedding_dim, int n_new_items) {

    assert(embedding_dim % 4 == 0);
    dim3 gridDim(ceil_div(embedding_dim / 4, BLOCK_DIM), n_sequence, n_new_items);
    inference_optimized_encoder<<<gridDim, BLOCK_DIM>>>(emb_table, wpe, inp, output, lengths, new_item_indices, batch_size, n_sequence, embedding_dim, n_new_items);
    CUDA_CHECK_LAST();
}
