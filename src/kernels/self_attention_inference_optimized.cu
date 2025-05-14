#include "kernels/gemm.h"
#include "tensor.hpp"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cassert>
#include <cmath>

namespace cg = cooperative_groups;
constexpr int WARP_SIZE = 32;
/**
 * inp: [n_batch, n_sequence, input_dim]
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
    const float* inp, const int* new_batch_idx,
    const int* lengths,
    const float* wk, const float* wv,
    float* kt_cache, float* v_cache,
    int num_new_batches, int n_sequence,
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
    const float* base_inp = inp + batch_idx * n_sequence * input_dim;
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
        #pragma unroll
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
        #pragma unroll
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


constexpr int TILE_SIZE_SQUARE = TILE_SIZE * TILE_SIZE;

/**
 * inp: [n_batch, n_sequence, input_dim]
 * lengths: [n_batch]
 * wq, wk, wv: [input_dim, output_dim]
 * v_cache: [n_batch, n_sequence, output_dim]
 * kt_cache: [n_batch, output_dim, n_sequence]
 * q_output: [n_batch, 1, output_dim]

 * 1. use the last embedding for each batch to multiply with wq, wk, wv -> [n_batch, 1, onput_dim]
 * 2. save to k_cache, v_cache and q_output
 */
__global__ void get_latest_kt_q_v(
    const float* inp, const int* lengths,
    const float* wk, const float* wq, const float* wv,
    float* kt_cache, float* v_cache,
    float* q_output,
    int n_batch, int n_sequence, int input_dim, int output_dim) {
    __shared__ float inp_shared[TILE_SIZE_SQUARE];
    int i_batch = blockIdx.y;
    int i_sequence = lengths[i_batch] - 1;
    int output_col_idx = blockIdx.x * TILE_SIZE_SQUARE + threadIdx.x;
    float k_result = 0;
    float q_result = 0;
    float v_result = 0;
    const float* base_inp = inp + i_batch * n_sequence * input_dim + i_sequence * input_dim;

    for (int i = 0; i < input_dim; i += TILE_SIZE_SQUARE) {
        if (i + threadIdx.x < input_dim) {
            inp_shared[threadIdx.x] = base_inp[i + threadIdx.x];
        } else {
            inp_shared[threadIdx.x] = 0.0f;
        }

        __syncthreads();
        if (output_col_idx < output_dim) {
            for (int j = 0; j < TILE_SIZE_SQUARE; ++j) {
                k_result += (base_inp[j] * wk[(i + j) * output_dim + output_col_idx]);
                v_result += (base_inp[j] * wv[(i + j) * output_dim + output_col_idx]);
                q_result += (base_inp[j] * wq[(i + j) * output_dim + output_col_idx]);
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
    int result_col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int cur_batch_length = lengths[batch_i];
    // all threads in this block exceed the cur_batch_lenght, no need to calculate
    if (blockIdx.x * TILE_SIZE >= cur_batch_length) {
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
        for (int j = 0; j < TILE_SIZE_SQUARE; ++j) {
            result += (q_shared[j] * base_kt[(i + j) * n_sequence + result_col]);
        }
        __syncthreads();
    }

    if (result_col < n_sequence) {
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
    if (row_id >= n_batch)
}


TensorFloat inference_self_attention(
    const TensorFloat& inp, const TensorInt& lengths,
    const TensorFloat& wk,
    const TensorFloat& wq,
    const TensorFloat& wv,
    const TensorInt& new_batch_idx, TensorFloat& k_cache, TensorFloat& v_cache) {
    
    
    // read from new_batch_idx, [b, s, dim] * [wk_wq_wv] -> [b, s, dim] -> k_cache, v_cache
    // this is one matrix, multiplication kernel
    fill_kt_v_cache();

    // for each batch, fetch the latest sequence, [b, 1, dim] * [wk_wq_wv](1, dim, dim *3) -> [b, 1, dim * 3]
    // meanwhile, put k, v to the kv_cache
    get_latest_kt_q_v();

    // [b, 1, dim] * [b, dim, n_sequence] -> [b, 1, n_sequence]
    qkt();

    // softmax with the mask considered. Other elements are 0 -> [b, 1, n_sequence]
    softmax_with_lengths();

    // [b,1,n_sequence]*[b, n_sequence,dim]
    softmax_v();
    return;
}