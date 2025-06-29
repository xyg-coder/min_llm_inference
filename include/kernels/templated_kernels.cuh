#pragma once

#include <cuda_runtime.h>
#include "constants.h"

template<const int N_THREADS, const int ROW_STRIDE_INP,
         const int BM>
__device__ void load_from_page_table(
    const float** page_table,
    int n_sequence, int emb_dim,
    float* shared_inp,
    int inner_row_inp,
    int inner_col_inp, int row_offset, int k_offset, int cur_batch_length) {

    for (int i_row_inp = inner_row_inp; i_row_inp + ROW_STRIDE_INP <= BM; i_row_inp += ROW_STRIDE_INP) {
        float4 tmp;
        if (i_row_inp >= cur_batch_length) {
            tmp = 0;
        } else {
            int page_idx = (row_offset + i_row_inp) / PAGE_BLOCK_SIZE;
            const float* page_pos = page_table[page_idx];
            int inp_offset = ((row_offset + i_row_inp) % PAGE_BLOCK_SIZE) * emb_dim * 4 + emb_dim * INP_EMB_EMB_OFFSET + k_offset + inner_col_inp * 4;
            tmp = reinterpret_cast<const float4*>(page_pos + inp_offset)[0];
        }
    }

    shared_inp[inner_col_inp * 4 * BM + inner_row_inp] = tmp.x;
    shared_inp[(inner_col_inp * 4 + 1) * BM + inner_row_inp] = tmp.y;
    shared_inp[(inner_col_inp * 4 + 2) * BM + inner_row_inp] = tmp.z;
    shared_inp[(inner_col_inp * 4 + 3) * BM + inner_row_inp] = tmp.w;

}

template<const int N_THREADS, const int ROW_STRIDE_WK_WV,
         const int BN, const int BK>
__device__ void load_from_kv(
    const float* kv, int emb_dim,
    float* shared_wk_wv,
    int inner_row_wk_wv, int inner_col_wk_wv, int k_offset, int col_offset) {

    for (int i_row_wk_wv = inner_row_wk_wv; i_row_wk_wv + ROW_STRIDE_WK_WV <= BK; i_row_wk_wv += ROW_STRIDE_WK_WV) {
        float4 tmp;
        if (i_row_wk_wv >= emb_dim) {
            tmp = 0;
        } else {
            tmp = reinterpret_cast<const float4*>(kv + (k_offset + i_row_wk_wv) * emb_dim + col_offset + inner_col_wk_wv * 4)[0];
        }
        reinterpret_cast<float4*>(shared_wk_wv + inner_row_wk_wv * 4 * BN + inner_col_wk_wv)[0] = tmp;
    }
}

template<const int WM, const int WN, const int WMITER, const int WNITER,
         const int TM, const int TN, const int BK, const int WSUBM, const int WSUBN>
__device__ void process_result(
    const float* shared_inp, const float* shared_wk, const float* shared_wv,
    float* reg_inp, float* reg_k, float* reg_v, float* thread_result_k, float* thread_result_v,
    int warp_row_id, int warp_col_id, int row_in_warp, int col_in_warp) {

    for (int i_k = 0; i_k < BK; ++i_k) {
        for (int i_wm_iter = 0; i_wm_iter < WMITER; ++i_wm_iter) {
            for (int i_tm = 0; i_tm < TM; ++i_tm) {
                // as long as number_of_rows_in_warp * TM <= 32, we can avoid bank conflicts
                reg_inp[i_wm_iter * TM + i_tm] = shared_inp[i_k * BM + warp_row_id * WM + i_wm_iter * WSUMBM + row_in_warp * TM + i_tm];
            }
        }
        for (int i_wn_iter = 0; i_wn_iter < WNITER; ++i_wn_iter) {
            for (int i_tn = 0; i_tn < TN; ++i_tn) {
                // as long as number_of_cols_in_warp * TN <= 32, we can avoid bank conflicts
                reg_k[i_wn_iter * TN + i_tn] = shared_wk[i_k * BN + warp_col_id * WN + i_wn_iter * WSUBN + col_in_warp * TN + i_tn];
                reg_v[i_wn_iter * TN + i_tn] = shared_wv[i_k * BN + warp_col_id * WN + i_wn_iter * WSUBN + col_in_warp * TN + i_tn];
            }
        }
        for (int i_tm = 0; i_tm < TM; ++i_tm) {
            for (int i_tn = 0; i_tn < TN; ++i_tn) {
                for (int i_wm_iter = 0; i_wm_iter < WMITER; ++i_wm_iter) {
                    for (int i_wn_iter = 0; i_wn_iter < WNITER; ++i_wn_iter) {
                        thread_result_k[(i_wm_iter * TM + i_tm) * WNITER * TN + i_wn_iter * TN + i_tn] +=
                            reg_inp[i_wm_iter * TM + i_tm] * reg_k[i_wn_iter * TN + i_tn];
                        thread_result_v[i_tm * WMITER * TN * WNITER + i_tn * WMITER * WNITER + i_wm_iter * WNITER + i_wn_iter] +=
                            reg_inp[i_wm_iter * TM + i_tm] * reg_v[i_wn_iter * TN + i_tn];
                    }
                }
            }
        }
    }
}

/**
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * new_batch_idx: [n_new_batch]
 * lengths: [n_batch]
 * wk: [emb_dim, emb_dim]
 * wv: [emb_dim, emb_dim]
 *
 * Try to use warp tiling optimization
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int TM, const int TN, const int WNITER, const int N_THREADS>
__global__ void fill_new_k_v_cache_paged_attention_warp_tiling(
    float** page_table, const int* new_batch_idx,
    const int* lengths,
    const float* wk, const float* wv,
    int n_sequence, int emb_dim) {

    int batch_idx = new_batch_idx[blockIdx.z];
    int cur_length = lengths[batch_idx];

    // The starting position is [output_block_row_id * BM, output_block_col_id * BN]
    int output_block_row_id = blockIdx.y;
    int output_block_col_id = blockIdx.x;

    constexpr int WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER);
    
    __shared__ float shared_inp[BM * BK];
    __shared__ float shared_wk[BK * BN];
    __shared__ float shared_wv[BK * BN];

    if (output_block_row_id * BM >= cur_length) {
        return;
    }
    page_table += batch_idx * (n_sequence / PAGE_BLOCK_SIZE);

    constexpr int WSUBN = WN / WNITER;
    constexpr int WSUBM = WM / WMITER;

    constexpr int ROW_STRIDE_INP = N_THREADS * 4 / BK;
    constexpr int ROW_STRIDE_WK_WV = N_THREADS * 4 / BN;

    int inner_row_inp = (threadIdx.x / (BK / 4));
    int inner_col_inp = (threadIdx.x % (BK / 4));
    int inner_row_wk_wv = (threadIdx.x / (BN / 4));
    int inner_col_wk_wv = (threadIdx.x % (BN / 4));

    float reg_inp[TM * WMITER] = {0};
    float reg_wk[TN * WNITER] = {0};
    float reg_wv[TN * WNITER] = {0};
    float thread_result_k[TM * WMITER * TN * WNITER] = {0};
    float thread_result_v[TM * WMITER * TN * WNITER] = {0};

    for (int bk_idx = 0; bk_idx < emb_dim; bk_idx += BK) {
        load_from_page_table<N_THREADS, ROW_STRIDE_INP, BM>(
            page_table, n_sequence, emb_dim,
            shared_inp, inner_row_inp, inner_col_inp, output_block_row_id * BM, bk_idx, cur_length);
        load_from_kv<N_THREADS, ROW_STRIDE_WK_WV, BN, BK>(
            wk, emb_dim,
            shared_wk, inner_row_wk_wv, inner_col_wk_wv, bk_idx, output_block_col_id * BN);
        load_from_kv<N_THREADS, ROW_STRIDE_WK_WV, BN, BK>(
            wv, emb_dim,
            shared_wv, inner_row_wk_wv, inner_col_wk_wv, bk_idx, output_block_col_id * BN);
        __syncthreads();

        process_result<WM, WN, WMITER, WNITER, TM, TN, BK, WSUBM, WSUBN>(
            shared_inp, shared_wk, shared_wv,
            reg_inp, reg_wk, reg_wv,
            thread_result_k, thread_result_v,
            threadIdx.y, threadIdx.x / (BK / 4), inner_row_inp, inner_col_inp);
    }

    int index_in_warp = threadIdx.x % WARP_SIZE;
    int warp_row_id = index_in_warp / (WN / WMITER);
    int warp_col_id = index_in_warp % (WM * WMITER);
    
    for (int i_wm_iter = 0; i_wm_iter < WMITER; ++i_wm_iter) {
        for (int i_tm = 0; i_tm < TM; ++i_tm) {
            for (int i_wn_iter = 0; i_wn_iter < WNITER; ++i_wn_iter) {
                for (int i_tn = 0; i_tn < TN; ++i_tn) {
                    int output_row = output_block_row_id * BM + i_wm_iter * WSUBM +  * TM + ;
                    int output_col = output_block_col_id * BN + i_tn + i_wn_iter * TN;
                    if (output_row < cur_length && output_col < emb_dim) {
                        int k_offset = (output_row * emb_dim + output_col) * 4;
                        int v_offset = (output_row * emb_dim + output_col) * 4;
                        reinterpret_cast<float4*>(wk + k_offset)[0] += thread_result_k[i_wm_iter * TM * WNITER * TN + i_wn_iter * TN + i_tn];
                        reinterpret_cast<float4*>(wv + v_offset)[0] += thread_result_v[i_tm * WMITER * TN * WNITER + i_tn * WMITER * WNITER + i_wm_iter * WNITER + i_wn_iter];
                    }
                }
            }
        }

    }
}
