#include "constants.h"
#include "kernels/decoder.h"
#include "kernels/gemm.h"
#include "kernels/utils.cuh"
#include "tensor.hpp"
#include "utils.h"
#include <cassert>
#include <cfloat>


/**
 * emb_score: [n_batch, n_vocab]
 * decoder_result: [n_batch]
 * lengths: [n_batch]
 * inp_embedding: [n_batch, n_sequence, input_dim]
 * wpe_table: [n_sequence, input_dim]
 * emb_table: [n_vocab, input_dim]
 *
 * 1. Each block handles one batch, get the maximum index among n_vocab
 * 2. Save the index to decoder_result
 * 3. lengths += 1
 * 4. calculate the new embedding and save to inp
 */
__global__ void decoder_kernel(
    const float* emb_score, int* decoder_result,
    int* lengths, float* inp_embedding, const float* wpe_table,
    const float* emb_table,
    int n_batch, int n_vocab, int n_sequence, int input_dim) {

    int i_batch = blockIdx.y;
    int cur_length = lengths[i_batch];
    // cur_length == 0 means invalid row
    if (cur_length == 0) {
        if (threadIdx.x == 0) {
            decoder_result[i_batch] = EMPTY_ROW_TOKEN_ID;
        }
        return;
    }
    
    // 1. get the maximum index among n_vocab
    __shared__ float max_value[BLOCK_DIM];
    __shared__ int max_index[BLOCK_DIM];
    const float* emb_base = emb_score + i_batch * n_vocab;
    float local_max = -FLT_MAX;
    int local_index = -1;
    int idx = threadIdx.x;
    for (int i = idx; i < n_vocab; i += BLOCK_DIM) {
        float val = emb_base[i];
        if (val > local_max) {
            local_max = val;
            local_index = i;
        }
    }
    max_value[idx] = local_max;
    max_index[idx] = local_index;
    __syncthreads();

    // BLOCK_DIM = 2^8
    for (int gap = BLOCK_DIM / 2; gap > 0; gap >>= 1) {
        if (idx < gap) {
            if (max_value[idx + gap] > max_value[idx]) {
                max_value[idx] = max_value[idx + gap];
                max_index[idx] = max_index[idx + gap];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        decoder_result[i_batch] = max_index[0];
        lengths[i_batch] = cur_length + 1;
    }


    // 2. calculate the embedding. Every block handles one batch
    assert(input_dim % 4 == 0);
    int embedding_dim_4 = input_dim / 4;
    if (idx >= embedding_dim_4 || cur_length >= n_sequence - 1 || max_index[0] == EOF_TOKEN_ID) {
        return;
    }

    const float4* emb_4 = reinterpret_cast<const float4*>(emb_table + max_index[0] * input_dim);
    const float4* wpe_4 = reinterpret_cast<const float4*>(wpe_table + cur_length * input_dim);
    float4* output_4 = reinterpret_cast<float4*>(inp_embedding + i_batch * n_sequence * input_dim + cur_length * input_dim);
    for (int i = idx; i < embedding_dim_4; i += BLOCK_DIM) {
        output_4[i] = float4_add(emb_4[i], wpe_4[i]);
    }
}


void launch_decoder(
    const TensorFloat& batch_result, const TensorFloat& emb_table,
    TensorFloat& emb_score,
    const TensorFloat& wpe_table,
    TensorFloat& inp_embedding, TensorInt& lengths, TensorInt& decoder_result) {

    int batch_size = batch_result.shape()[0];
    int emb_dim = batch_result.shape()[1];
    int n_vocab = emb_table.shape()[0];
    int n_sequence = wpe_table.shape()[0];

    launch_gemm_transpose_kernel(
        batch_result.data(), emb_table.data(), emb_score.data(), 1, batch_size, n_vocab, emb_dim);

    dim3 gridDim(1, batch_size);
    decoder_kernel<<<gridDim, BLOCK_DIM>>>(emb_score.data(), decoder_result.data(), lengths.data(), inp_embedding.data(), wpe_table.data(),
        emb_table.data(), batch_size, n_vocab, n_sequence, emb_dim);
    CUDA_CHECK_LAST();
}


/**
 * emb_score: [n_batch, n_vocab]
 * decoder_result: [n_batch]
 * lengths: [n_batch]
 * page_table: [n_batch, n_sequence / page_block_size]
 * wpe_table: [n_sequence, input_dim]
 * emb_table: [n_vocab, input_dim]
 *
 * 1. Each block handles one batch, get the maximum index among n_vocab
 * 2. Save the index to decoder_result
 * 3. lengths += 1
 * 4. calculate the new embedding and save to inp
 */
__global__ void paged_attention_decoder_kernel(
    const float* emb_score, int* decoder_result,
    int* lengths, float** page_table, const float* wpe_table,
    const float* emb_table,
    int n_batch, int n_vocab, int n_sequence, int emb_dim, int page_block_size) {

    int i_batch = blockIdx.y;
    int cur_length = lengths[i_batch];
    // cur_length == 0 means invalid row
    if (cur_length == 0) {
        if (threadIdx.x == 0) {
            decoder_result[i_batch] = EMPTY_ROW_TOKEN_ID;
        }
        return;
    }
    
    // 1. get the maximum index among n_vocab
    __shared__ float max_value[BLOCK_DIM];
    __shared__ int max_index[BLOCK_DIM];
    const float* emb_base = emb_score + i_batch * n_vocab;
    float local_max = -FLT_MAX;
    int local_index = -1;
    int idx = threadIdx.x;
    for (int i = idx; i < n_vocab; i += BLOCK_DIM) {
        float val = emb_base[i];
        if (val > local_max) {
            local_max = val;
            local_index = i;
        }
    }
    max_value[idx] = local_max;
    max_index[idx] = local_index;
    __syncthreads();

    // BLOCK_DIM = 2^8
    for (int gap = BLOCK_DIM / 2; gap > 0; gap >>= 1) {
        if (idx < gap) {
            if (max_value[idx + gap] > max_value[idx]) {
                max_value[idx] = max_value[idx + gap];
                max_index[idx] = max_index[idx + gap];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        decoder_result[i_batch] = max_index[0];
        lengths[i_batch] = cur_length + 1;
    }


    // 2. calculate the embedding. Every block handles one batch
    assert(emb_dim % 4 == 0);
    int embedding_dim_4 = emb_dim / 4;
    if (idx >= embedding_dim_4 || cur_length >= n_sequence - 1 || max_index[0] == EOF_TOKEN_ID) {
        return;
    }

    const float4* emb_4 = reinterpret_cast<const float4*>(emb_table + max_index[0] * emb_dim);
    const float4* wpe_4 = reinterpret_cast<const float4*>(wpe_table + cur_length * emb_dim);
    assert(n_sequence % page_block_size == 0);
    int width = n_sequence / page_block_size;
    float* page_pos = page_table[i_batch * width + cur_length / page_block_size];
    float* emb_pos = page_pos + (cur_length % page_block_size) * emb_dim * 3 + emb_dim * INP_EMB_EMB_OFFSET;

    float4* output_4 = reinterpret_cast<float4*>(emb_pos);
    for (int i = idx; i < embedding_dim_4; i += BLOCK_DIM) {
        output_4[i] = float4_add(emb_4[i], wpe_4[i]);
    }
}


void launch_paged_attention_decoder(
    const TensorFloat& batch_result, const TensorFloat& emb_table,
    TensorFloat& emb_score,
    const TensorFloat& wpe_table,
    TensorFloatPoint& page_table, TensorInt& lengths, TensorInt& decoder_result) {

    int batch_size = batch_result.shape()[0];
    int emb_dim = batch_result.shape()[1];
    int n_vocab = emb_table.shape()[0];
    int n_sequence = wpe_table.shape()[0];

    launch_gemm_transpose_kernel(
        batch_result.data(), emb_table.data(), emb_score.data(), 1, batch_size, n_vocab, emb_dim);

    dim3 gridDim(1, batch_size);
    paged_attention_decoder_kernel<<<gridDim, BLOCK_DIM>>>(emb_score.data(), decoder_result.data(), lengths.data(),
        page_table.data(), wpe_table.data(), emb_table.data(), batch_size, n_vocab, n_sequence, emb_dim, PAGE_BLOCK_SIZE);
    CUDA_CHECK_LAST();
}
