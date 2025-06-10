#include "constants.h"
#include "tensor.hpp"
#include "utils.h"
#include "kernels/self_attention_inference_optimized.h"
#include <cassert>


/**
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * new_batch_idx: [n_new_batch]
 * lengths: [n_batch]
 * wk: [emb_dim, emb_dim]
 * wv: [emb_dim, emb_dim]
 * 
 * Total number of threads: n_new_batch * num_sequence * emb_dim
 * outputting to [new_batch_idx[blockIdx.z], blockIdx.y * TILE_SIZE + threadIdx.y, blockIdx.x * TILE_SIZE + threadIdx.x]
 * 1. inp[new_batch_idx] * wk or wv -> [n_new_batch, n_sequence, emb_dim]
 * 2. if i_sequence < lenghs[batch_idx]: copy to to v_cache/k_cache
 */
__global__ void fill_new_k_v_cache_paged_attention(
    float** page_table, const int* new_batch_idx,
    const int* lengths,
    const float* wk, const float* wv,
    int n_sequence, int emb_dim) {
    __shared__ float inp_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float kv_shared[TILE_SIZE][TILE_SIZE];
    int batch_idx = new_batch_idx[blockIdx.z];
    int output_row_idx = blockIdx.y * TILE_SIZE + threadIdx.y;
    int output_col_idx = blockIdx.x * TILE_SIZE + threadIdx.x;
    // the max_i_sequence needed
    int cur_batch_length = lengths[batch_idx];
    // There is no thread in this block working, so we can exit
    if (blockIdx.y * TILE_SIZE >= cur_batch_length) {
        return;
    }
    float k_result = 0;
    float v_result = 0;
    int page_table_width = n_sequence / PAGE_BLOCK_SIZE;
    float** base_table = page_table + batch_idx * page_table_width;

    float* page_pos = page_table[batch_idx * (n_sequence / PAGE_BLOCK_SIZE) + output_row_idx / PAGE_BLOCK_SIZE];
    for (int i = 0; i < emb_dim; i += TILE_SIZE) {
        if (output_row_idx < cur_batch_length && i + threadIdx.x < emb_dim) {
            inp_shared[threadIdx.y][threadIdx.x] = get_page_table_value(
                page_pos, batch_idx, n_sequence, output_row_idx, emb_dim, PAGE_BLOCK_SIZE,
                i + threadIdx.x, INP_EMB_EMB_OFFSET);
        } else {
            inp_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (i + threadIdx.y < emb_dim && output_col_idx < emb_dim) {
            kv_shared[threadIdx.y][threadIdx.x] = wk[(i + threadIdx.y) * emb_dim + output_col_idx];
        } else {
            kv_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        for (int j = 0; j < TILE_SIZE; ++j) {
            k_result += (inp_shared[threadIdx.y][j] * kv_shared[j][threadIdx.x]);
        }

        __syncthreads();
        if (i + threadIdx.y < emb_dim && output_col_idx < emb_dim) {
            kv_shared[threadIdx.y][threadIdx.x] = wv[(i + threadIdx.y) * emb_dim + output_col_idx];
        } else {
            kv_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        for (int j = 0; j < TILE_SIZE; ++j) {
            v_result += (inp_shared[threadIdx.y][j] * kv_shared[j][threadIdx.x]);
        }
        __syncthreads();
    }
    if (output_row_idx < cur_batch_length && output_col_idx < emb_dim) {
        set_page_table_value(page_pos, batch_idx, n_sequence, output_row_idx, emb_dim,
            PAGE_BLOCK_SIZE, output_col_idx, K_CACHE_EMB_OFFSET, k_result);
        set_page_table_value(page_pos, batch_idx, n_sequence, output_row_idx, emb_dim,
            PAGE_BLOCK_SIZE, output_col_idx, V_CACHE_EMB_OFFSET, v_result);
    }
}

/**
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * new_batch_idx: [n_new_batch]
 * lengths: [n_batch]
 * wk: [emb_dim, emb_dim]
 * wv: [emb_dim, emb_dim]
 */
void launch_fill_new_k_v_cache_paged_attention(
    TensorFloatPoint page_table, const TensorInt& new_batch_idx, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wv, int n_new_items, int n_sequence) {

    if (n_new_items == 0) {
        return;
    }

    int n_batch = page_table.shape()[0];
    assert(page_table.shape()[1] == n_sequence / PAGE_BLOCK_SIZE && n_sequence % PAGE_BLOCK_SIZE == 0);
    int emb_dim = wk.shape()[0];
    assert(wk.shape()[0] == wk.shape()[1]);
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (emb_dim + TILE_SIZE - 1) / TILE_SIZE, (n_sequence + TILE_SIZE - 1) / TILE_SIZE, n_new_items);
    fill_new_k_v_cache_paged_attention<<<gridDim, blockDim>>>(
        page_table.data(), new_batch_idx.data(),
        lengths.data(), wk.data(), wv.data(), n_sequence, emb_dim);
    CUDA_CHECK_LAST();
}

/**
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * lengths: [n_batch]
 * wq, wk, wv: [input_dim, output_dim]
 * q_output: [n_batch, 1, output_dim]

 * 1. use the last embedding for each batch to multiply with wq, wk, wv -> [n_batch, 1, onput_dim]
 * 2. save to k_cache, v_cache and q_output
 */
__global__ void get_latest_k_q_v_paged_attention(
    float** page_table, const int* lengths,
    const float* wk, const float* wq, const float* wv,
    float* q_output,
    int n_batch, int n_sequence, int emb_dim) {
    __shared__ float inp_shared[TILE_SIZE_SQUARE];
    int i_batch = blockIdx.y;
    int cur_length = lengths[i_batch];
    // cur_length == 0 means empty row
    if (cur_length == 0) {
        return;
    }
    int i_sequence = cur_length - 1;
    int output_col_idx = blockIdx.x * TILE_SIZE_SQUARE + threadIdx.x;
    float k_result = 0;
    float q_result = 0;
    float v_result = 0;
    float* page_pos = page_table[i_batch * (n_sequence / PAGE_BLOCK_SIZE) + i_sequence / PAGE_BLOCK_SIZE];

    for (int i = 0; i < emb_dim; i += TILE_SIZE_SQUARE) {
        if (i + threadIdx.x < emb_dim) {
            inp_shared[threadIdx.x] = get_page_table_value(page_pos, i_batch, n_sequence, i_sequence, emb_dim, PAGE_BLOCK_SIZE, i + threadIdx.x, INP_EMB_EMB_OFFSET);
        } else {
            inp_shared[threadIdx.x] = 0.0f;
        }

        __syncthreads();
        if (output_col_idx < emb_dim) {
            for (int j = 0; j < TILE_SIZE_SQUARE && i + j < emb_dim; ++j) {
                k_result += (inp_shared[j] * wk[(i + j) * emb_dim + output_col_idx]);
                v_result += (inp_shared[j] * wv[(i + j) * emb_dim + output_col_idx]);
                q_result += (inp_shared[j] * wq[(i + j) * emb_dim + output_col_idx]);
            }
        }
        __syncthreads();
    }
    if (output_col_idx < emb_dim) {
        set_page_table_value(page_pos, i_batch, n_sequence, i_sequence, emb_dim, PAGE_BLOCK_SIZE, output_col_idx, K_CACHE_EMB_OFFSET, k_result);
        set_page_table_value(page_pos, i_batch, n_sequence, i_sequence, emb_dim, PAGE_BLOCK_SIZE, output_col_idx, V_CACHE_EMB_OFFSET, v_result);
        q_output[i_batch * emb_dim + output_col_idx] = q_result;
    }
}

/**
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * lengths: [n_batch]
 * wq, wk, wv: [emb_dim, emb_dim]
 * q_output: [n_batch, 1, emb_dim]
 */
void launch_get_latest_k_q_v_paged_attention(
    TensorFloatPoint& page_table, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wq,
    const TensorFloat& wv, TensorFloat& q_output, int n_sequence) {

    int n_batch = page_table.shape()[0];
    int emb_dim = wq.shape()[0];
    dim3 gridDim(ceil_div(emb_dim, TILE_SIZE_SQUARE), n_batch);
    get_latest_k_q_v_paged_attention<<<gridDim, TILE_SIZE_SQUARE>>>(
        page_table.data(), lengths.data(), wk.data(), wq.data(), wv.data(), q_output.data(), n_batch, n_sequence, emb_dim);
    CUDA_CHECK_LAST();
}

/**
 * q: [n_batch, emb_dim]
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * qkt: [n_batch, n_sequence]
 * 
 * One block handles (blockIdx.y, blockIdx.x*TILE_SIZE : blockIdx.x*TILE_SIZE + TILE_SIZE)
 */
__global__ void qkt_paged_attention(
    const float* q, const float** page_table, const int* lengths,
    float* qkt,
    int n_batch, int n_sequence, int emb_dim) {
    
    __shared__ float q_shared[TILE_SIZE];
    __shared__ float k_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float result_shared[TILE_SIZE];
    int batch_i = blockIdx.y;
    int result_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int cur_batch_length = lengths[batch_i];
    // all threads in this block exceed the cur_batch_lenght, no need to calculate
    if (blockIdx.x * TILE_SIZE >= cur_batch_length) {
        return;
    }
    result_shared[threadIdx.x] = 0.0f;

    const float* base_q = q + batch_i * emb_dim;

    for (int i = 0; i < emb_dim; i += TILE_SIZE) {
        if (i + threadIdx.x < emb_dim) {
            q_shared[threadIdx.x] = base_q[i + threadIdx.x];
        } else {
            q_shared[threadIdx.x] = 0.0f;
        }
        
        int i_block = -1;
        const float* page_pos = nullptr;
        for (int i_sequence = blockIdx.x * TILE_SIZE; i_sequence < blockIdx.x * TILE_SIZE + TILE_SIZE && i_sequence < cur_batch_length; i_sequence++) {
            if (i + threadIdx.x < emb_dim) {
                if (i_sequence / PAGE_BLOCK_SIZE != i_block) {
                    i_block = i_sequence / PAGE_BLOCK_SIZE;
                    page_pos = page_table[batch_i * (n_sequence / PAGE_BLOCK_SIZE) + i_block];
                }
                k_shared[i_sequence - blockIdx.x * TILE_SIZE][threadIdx.x] = get_page_table_value(
                    page_pos, batch_i, n_sequence, i_sequence, emb_dim, PAGE_BLOCK_SIZE, i + threadIdx.x, K_CACHE_EMB_OFFSET);
            } else {
                k_shared[i_sequence - blockIdx.x * TILE_SIZE][threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();
        for (int j = 0; result_col < cur_batch_length && j < TILE_SIZE; ++j) {
            result_shared[threadIdx.x] += (q_shared[j] * k_shared[threadIdx.x][j]);
        }
        __syncthreads();
    }

    if (result_col < cur_batch_length) {
        qkt[batch_i * n_sequence + result_col] = result_shared[threadIdx.x] / sqrtf(emb_dim);
    }
}

/**
 * q_output: [n_batch, dim]
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * qkt: [n_batch, n_sequence]
 */
void launch_qkt_paged_attention(
    const TensorFloat& q_output, const TensorFloatPoint& page_table, const TensorInt& lengths,
    TensorFloat& qkt_output) {
    
    int n_batch = q_output.shape()[0];
    int n_sequence = qkt_output.shape()[1];
    int emb_dim = q_output.shape()[1];
    dim3 gridDim(ceil_div(n_sequence, TILE_SIZE), n_batch);
    qkt_paged_attention<<<gridDim, TILE_SIZE>>>(q_output.data(), (const float**)page_table.data(), lengths.data(), qkt_output.data(), n_batch, n_sequence, emb_dim);
    CUDA_CHECK_LAST();
}

/**
 * softmax_result: [n_batch, n_sequence] 
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * attention_result: [n_batch, emb_dim]
 */
__global__ void softmax_v_paged_attention(
    const float* softmax_result, const float** page_table,
    float* attention_result,
    const int* lengths,
    int n_batch, int n_sequence, int emb_dim) {

    int i_batch = blockIdx.y;
    __shared__ float softmax_res_share[TILE_SIZE_SQUARE];
    float result = 0.0;

    int cur_batch_length = lengths[i_batch];
    const float* softmax_result_base = softmax_result + i_batch * n_sequence;
    int write_col = blockIdx.x * TILE_SIZE_SQUARE + threadIdx.x;

    int i_block = -1;
    const float* page_pos = nullptr;
    for (int i = 0; i < cur_batch_length; i += TILE_SIZE_SQUARE) {
        if (i + threadIdx.x < cur_batch_length) {
            softmax_res_share[threadIdx.x] = softmax_result_base[i + threadIdx.x];
        } else {
            softmax_res_share[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        for (int j = 0; write_col < emb_dim && j < TILE_SIZE_SQUARE && i + j < cur_batch_length; ++j) {
            if (i_block != (i + j) / PAGE_BLOCK_SIZE) {
                i_block = (i + j) / PAGE_BLOCK_SIZE;
                page_pos = page_table[i_batch * (n_sequence / PAGE_BLOCK_SIZE) + i_block];
            }
            result += (softmax_res_share[j] * get_page_table_value(page_pos, i_batch, n_sequence, i + j, emb_dim, PAGE_BLOCK_SIZE, write_col, V_CACHE_EMB_OFFSET));
        }
        __syncthreads();
    }
    if (write_col < emb_dim) {
        attention_result[i_batch * emb_dim + write_col] = result;
    }
}

/**
 * softmax_result: [n_batch, n_sequence] 
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * attention_result: [n_batch, emb_dim]
 */
void launch_softmax_v_paged_attention(
    const TensorFloat& softmax_result, const TensorFloatPoint& page_table, TensorFloat& attention_result,
    const TensorInt& lengths) {
    
    int n_batch = softmax_result.shape()[0];
    int n_sequence = softmax_result.shape()[1];
    int emb_dim = attention_result.shape()[1];
    dim3 gridDim(ceil_div(emb_dim, TILE_SIZE_SQUARE), n_batch);
    softmax_v_paged_attention<<<gridDim, TILE_SIZE_SQUARE>>>(
        softmax_result.data(), (const float**)page_table.data(),
        attention_result.data(), lengths.data(), n_batch, n_sequence, emb_dim);
    CUDA_CHECK_LAST();
}

/**
 * - page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE], input_embedding + k_cache + v_cache
 *      Each pointer points to a memory block of (3 * embedding_dim * PAGE_BLOCK_SIZE)
 * - lengths: [n_batch], the token lengths for each batch
 * - wk, wq, wv: [emb_dim, emb_dim]: since we don't have the following feed-forward layer, 
 * input_dim should be equal to output_dim
 * - new_batch_idx: [n_batch], but only n_new_items are used
 * - q_output: [n_batch, emb_dim]
 * - qkt_output: [n_batch, n_sequence]
 * - attention_result: [n_batch, emb_dim]
 */
void paged_attention(
    TensorFloatPoint& page_table,
    const TensorInt& lengths,
    const TensorFloat& wk,
    const TensorFloat& wq,
    const TensorFloat& wv,
    const TensorInt& new_batch_idx,
    TensorFloat& q_output, TensorFloat& qkt_output, TensorFloat& attention_result,
    int n_new_items, int n_sequence) {

    launch_fill_new_k_v_cache_paged_attention(page_table, new_batch_idx, lengths, wk, wv, n_new_items, n_sequence);

    launch_get_latest_k_q_v_paged_attention(page_table, lengths, wk, wq, wv, q_output, n_sequence);

    launch_qkt_paged_attention(q_output, page_table, lengths, qkt_output);

    launch_softmax_in_place_with_lengths(qkt_output, lengths);

    launch_softmax_v_paged_attention(qkt_output, page_table, attention_result, lengths);
}
