#include "tensor.hpp"
#include "kernels/utils.cuh"
#include "kernels/self_attention_inference_optimized.h"
#include <cfloat>
#include <vector_types.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cassert>
#include <cmath>
#include "constants.h"

namespace cg = cooperative_groups;
/**
 * inp_embedding: [n_batch, n_sequence, input_dim]
 * new_batch_idx: [n_new_batch]
 * lengths: [n_batch]
 * wk: [input_dim, output_dim]
 * wv: [input_dim, output_dim]
 * kt_cache: [n_batch_size, output_dim, n_sequence]
 * v_cache: [n_batch_size, n_sequence, output_dim]
 * 
 * Total number of threads: n_new_batch * num_sequence * output_dim
 * outputting to [new_batch_idx[blockIdx.z], blockIdx.y * TILE_SIZE + threadIdx.y, blockIdx.x * TILE_SIZE + threadIdx.x]
 * 1. inp[new_batch_idx] * wk or wv -> [n_new_batch, n_sequence, output_dim]
 * 2. if i_sequence < lenghs[batch_idx]: copy to to v_cache/k_cache
 */
__global__ void fill_new_kt_v_cache(
    const float* inp_embedding, const int* new_batch_idx,
    const int* lengths,
    const float* wk, const float* wv,
    float* kt_cache, float* v_cache,
    int n_sequence,
    int input_dim, int output_dim) {
    
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
    const float* base_inp = inp_embedding + batch_idx * n_sequence * input_dim;
    for (int i = 0; i < input_dim; i += TILE_SIZE) {
        if (output_row_idx < cur_batch_length && i + threadIdx.x < input_dim) {
            inp_shared[threadIdx.y][threadIdx.x] = base_inp[output_row_idx * input_dim + i + threadIdx.x];
        } else {
            inp_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (i + threadIdx.y < input_dim && output_col_idx < output_dim) {
            kv_shared[threadIdx.y][threadIdx.x] = wk[(i + threadIdx.y) * output_dim + output_col_idx];
        } else {
            kv_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j) {
            k_result += (inp_shared[threadIdx.y][j] * kv_shared[j][threadIdx.x]);
        }

        __syncthreads();
        if (i + threadIdx.y < input_dim && output_col_idx < output_dim) {
            kv_shared[threadIdx.y][threadIdx.x] = wv[(i + threadIdx.y) * output_dim + output_col_idx];
        } else {
            kv_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        for (int j = 0; j < TILE_SIZE; ++j) {
            v_result += (inp_shared[threadIdx.y][j] * kv_shared[j][threadIdx.x]);
        }
        __syncthreads();
    }
    if (output_row_idx < cur_batch_length && output_col_idx < output_dim) {
        kt_cache[batch_idx * n_sequence * output_dim + output_col_idx * n_sequence + output_row_idx] = k_result;
        v_cache[batch_idx * n_sequence * output_dim + output_row_idx * output_dim + output_col_idx] = v_result;
    }
}



/**
 * inp_embedding: [n_batch, n_sequence, input_dim]
 * lengths: [n_batch]
 * wq, wk, wv: [input_dim, output_dim]
 * v_cache: [n_batch, n_sequence, output_dim]
 * kt_cache: [n_batch, output_dim, n_sequence]
 * q_output: [n_batch, 1, output_dim]

 * 1. use the last embedding for each batch to multiply with wq, wk, wv -> [n_batch, 1, onput_dim]
 * 2. save to k_cache, v_cache and q_output
 */
__global__ void get_latest_kt_q_v(
    const float* inp_embedding, const int* lengths,
    const float* wk, const float* wq, const float* wv,
    float* kt_cache, float* v_cache,
    float* q_output,
    int n_batch, int n_sequence, int input_dim, int output_dim) {
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
    const float* base_inp = inp_embedding + i_batch * n_sequence * input_dim + i_sequence * input_dim;

    for (int i = 0; i < input_dim; i += TILE_SIZE_SQUARE) {
        if (i + threadIdx.x < input_dim) {
            inp_shared[threadIdx.x] = base_inp[i + threadIdx.x];
        } else {
            inp_shared[threadIdx.x] = 0.0f;
        }

        __syncthreads();
        if (output_col_idx < output_dim) {

            for (int j = 0; j < TILE_SIZE_SQUARE; ++j) {
                k_result += (inp_shared[j] * wk[(i + j) * output_dim + output_col_idx]);
                v_result += (inp_shared[j] * wv[(i + j) * output_dim + output_col_idx]);
                q_result += (inp_shared[j] * wq[(i + j) * output_dim + output_col_idx]);
            }
        }
        __syncthreads();
    }
    if (output_col_idx < output_dim) {
        kt_cache[i_batch * n_sequence * output_dim + output_col_idx * n_sequence + i_sequence] = k_result;
        v_cache[i_batch * n_sequence * output_dim + i_sequence * output_dim + output_col_idx] = v_result;
        q_output[i_batch * output_dim + output_col_idx] = q_result;
    }
}

/**
 * q: [n_batch, dim]
 * kt: [n_batch, dim, n_sequence]
 * qkt: [n_batch, n_sequence]
 */
__global__ void qkt(
    const float* q, const float* kt, const int* lengths,
    float* qkt,
    int n_batch, int n_sequence, int dim) {
    
    __shared__ float q_shared[TILE_SIZE_SQUARE];
    int batch_i = blockIdx.y;
    int result_col = blockIdx.x * TILE_SIZE_SQUARE + threadIdx.x;
    int cur_batch_length = lengths[batch_i];
    // all threads in this block exceed the cur_batch_lenght, no need to calculate
    if (blockIdx.x * TILE_SIZE_SQUARE >= cur_batch_length) {
        return;
    }
    const float* base_q = q + batch_i * dim;
    const float* base_kt = kt + batch_i * dim * n_sequence;
    float result = 0;

    for (int i = 0; i < dim; i += TILE_SIZE_SQUARE) {
        if (i + threadIdx.x < dim) {
            q_shared[threadIdx.x] = base_q[i + threadIdx.x];
        } else {
            q_shared[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int j = 0; result_col < cur_batch_length && j < TILE_SIZE_SQUARE; ++j) {
            result += (q_shared[j] * base_kt[(i + j) * n_sequence + result_col]);
        }
        __syncthreads();
    }

    if (result_col < cur_batch_length) {
        qkt[batch_i * n_sequence + result_col] = result / sqrtf(dim);
    }
}

/**
 * qkt: [n_batch, n_sequence]
 * result is written to qkt, with the same shape
 * Any element exceeding the lengths is 0
 */
__global__ void softmax_in_place_with_lengths(
    float* qkt, int n_batch, int n_sequence,
    const int* lengths) {

    assert(n_sequence % 4 == 0);
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    int row_id = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (row_id >= n_batch) {
        return;
    }

    const float* qkt_base = qkt + row_id * n_sequence;
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const float4* qkt_vec4 = reinterpret_cast<const float4*>(qkt_base);
    int cur_length = lengths[row_id];

    for (int i = warp.thread_rank(); i < (cur_length + 4  - 1) / 4; i += warp.num_threads()) {
        float4 v = qkt_vec4[i];
        float old_max_val = maxval;
        for (int j = 0; j < 4 && i * 4 + j < cur_length; ++j) {
            maxval = fmax(maxval, vec_at(v, j));
        }
        sumval *= expf(old_max_val - maxval);
        for (int j = 0; j < 4 && i * 4 + j < cur_length; ++j) {
            sumval += expf(vec_at(v, j) - maxval);
        }
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(maxval - global_maxval);
    float global_sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / global_sum;
    float temp[4];
    float4* out_vec = reinterpret_cast<float4*>(qkt + row_id * n_sequence);

    for (int i = warp.thread_rank(); i < n_sequence / 4; i += warp.num_threads()) {
        float4 v = out_vec[i];

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (i * 4 + j < cur_length) {
                temp[j] = expf(vec_at(v, j) - global_maxval) * norm;
            } else {
                temp[j] = 0.0f;
            }
        }
        out_vec[i] = float4({temp[0], temp[1], temp[2], temp[3]});
    }
}

/**
 * softmax_result: [n_batch, n_sequence] 
 * v_cache: [n_batch, n_sequence, output_dim]
 * softmax_v_result: [n_batch, output_dim]
 */
__global__ void softmax_v(
    const float* softmax_result, const float* v_cache,
    float* softmax_v_result,
    const int* lengths,
    int n_batch, int n_sequence, int output_dim) {

    int i_batch = blockIdx.y;
    __shared__ float softmax_res_share[TILE_SIZE_SQUARE];
    int cur_batch_length = lengths[i_batch];
    const float* softmax_result_base = softmax_result + i_batch * n_sequence;
    const float* v_cache_base = v_cache + i_batch * n_sequence * output_dim;
    float result = 0;
    int write_col = blockIdx.x * TILE_SIZE_SQUARE + threadIdx.x;

    for (int i = 0; i < cur_batch_length; i += TILE_SIZE_SQUARE) {
        if (i + threadIdx.x < cur_batch_length) {
            softmax_res_share[threadIdx.x] = softmax_result_base[i + threadIdx.x];
        } else {
            softmax_res_share[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int j = 0; write_col < output_dim && j < TILE_SIZE_SQUARE && i + j < cur_batch_length; ++j) {
            result += (softmax_res_share[j] * v_cache_base[(i + j) * output_dim + write_col]);
        }
        __syncthreads();
    }
    if (write_col < output_dim) {
        softmax_v_result[i_batch * output_dim + write_col] = result;
    }
}


void inference_self_attention(
    const TensorFloat& inp_embedding, const TensorInt& lengths,
    const TensorFloat& wk,
    const TensorFloat& wq,
    const TensorFloat& wv,
    const TensorInt& new_batch_idx, TensorFloat& kt_cache, TensorFloat& v_cache,
    // avoid frequent creation of tensors
    TensorFloat& q_output, TensorFloat& qkt_output, TensorFloat& attention_result,
    int n_new_items) {
    
    launch_fill_new_kt_v_cache(inp_embedding, new_batch_idx, lengths, wk, wv, kt_cache, v_cache, n_new_items);

    launch_get_latest_kt_q_v(inp_embedding, lengths, wk, wq, wv, kt_cache, v_cache, q_output);

    launch_qkt(q_output, kt_cache, lengths, qkt_output);

    launch_softmax_in_place_with_lengths(qkt_output, lengths);

    launch_softmax_v(qkt_output, v_cache, attention_result, lengths);
}

void launch_fill_new_kt_v_cache(
    const TensorFloat& inp_embedding, const TensorInt& new_batch_idx, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wv, TensorFloat& kt_cache,
    TensorFloat& v_cache, int n_new_items) {
    
    if (n_new_items == 0) {
        return;
    }
    
    int n_batch = inp_embedding.shape()[0];
    int n_sequence = inp_embedding.shape()[1];
    int input_dim = inp_embedding.shape()[2];
    int output_dim = wk.shape()[1];
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (output_dim + TILE_SIZE - 1) / TILE_SIZE, (n_sequence + TILE_SIZE - 1) / TILE_SIZE, n_new_items);
    fill_new_kt_v_cache<<<gridDim, blockDim>>>(
        inp_embedding.data(), new_batch_idx.data(), lengths.data(), wk.data(), wv.data(), kt_cache.data(),
        v_cache.data(), n_sequence, input_dim, output_dim);
    CUDA_CHECK_LAST();
}

void launch_get_latest_kt_q_v(
    const TensorFloat& inp_embedding, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wq,
    const TensorFloat& wv, TensorFloat& kt_cache,
    TensorFloat& v_cache, TensorFloat& q_output) {

    int n_batch = inp_embedding.shape()[0];
    int n_sequence = inp_embedding.shape()[1];
    int input_dim = inp_embedding.shape()[2];
    int output_dim = wk.shape()[1];

    dim3 gridDim(ceil_div(output_dim, TILE_SIZE_SQUARE), n_batch);
    get_latest_kt_q_v<<<gridDim, TILE_SIZE_SQUARE>>>(
        inp_embedding.data(), lengths.data(),
        wk.data(), wq.data(), wv.data(), kt_cache.data(),
        v_cache.data(), q_output.data(), n_batch,
        n_sequence, input_dim, output_dim);
    CUDA_CHECK_LAST();
}

void launch_qkt(
    const TensorFloat& q_output, const TensorFloat& kt_cache, const TensorInt& lengths,
    TensorFloat& qkt_output) {

    int n_batch = q_output.shape()[0];
    int n_sequence = kt_cache.shape()[2];
    int output_dim = q_output.shape()[1];

    dim3 gridDim(ceil_div(n_sequence, TILE_SIZE_SQUARE), n_batch);
    qkt<<<gridDim, TILE_SIZE_SQUARE>>>(
        q_output.data(), kt_cache.data(), lengths.data(), qkt_output.data(),
        n_batch, n_sequence, output_dim);
    CUDA_CHECK_LAST();
}

void launch_softmax_in_place_with_lengths(
    TensorFloat& qkt_output, const TensorInt& lengths) {

    int n_batch = qkt_output.shape()[0];
    int n_sequence = qkt_output.shape()[1];
    softmax_in_place_with_lengths<<<ceil_div(n_batch * WARP_SIZE, TILE_SIZE_SQUARE), TILE_SIZE_SQUARE>>>(
        qkt_output.data(), n_batch, n_sequence, lengths.data());
    CUDA_CHECK_LAST();
}

void launch_softmax_v(
    const TensorFloat& softmax_result, const TensorFloat& v_cache, TensorFloat& attention_result,
    const TensorInt& lengths) {
    
    int n_batch = v_cache.shape()[0];
    int n_sequence = v_cache.shape()[1];
    int output_dim = v_cache.shape()[2];

    dim3 gridDim(ceil_div(output_dim, TILE_SIZE_SQUARE), n_batch);
    softmax_v<<<gridDim, TILE_SIZE_SQUARE>>>(
        softmax_result.data(), v_cache.data(), attention_result.data(),
        lengths.data(), n_batch, n_sequence, output_dim);
}
