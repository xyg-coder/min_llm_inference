#pragma once

#include <cuda_runtime.h>
#include "constants.h"

template<const int N_THREADS, const int ROW_STRIDE_INP, const int ROW_STRIDE_WK_WV,
         const int BM, const int BN>
__device__ void load_from_page_table(
    const float** page_table, const float* kv,
    int n_sequence, int emb_dim,
    float* shared_inp, float* shared_wk_wv,
    int inner_row_inp, int inner_row_wk_wv,
    int inner_col_inp, int inner_col_wk_wv, int cur_batch_length) {

    for (int i_row_inp = inner_row_inp; i_row_inp + ROW_STRIDE_INP <= BM; i_row_inp += ROW_STRIDE_INP) {
        float4 tmp;
        if (i_row_inp >= cur_batch_length) {
            tmp = 0;
        } else {
            int page_idx = i_row_inp / PAGE_BLOCK_SIZE;
            const float* page_pos = page_table[page_idx];
            int inp_offset = i_row_inp % PAGE_BLOCK_SIZE * emb_dim + inner_col_inp * 4;
            tmp = reinterpret_cast<const float4*>(page_pos + inp_offset)[0];
        }
    }

    shared_inp[inner_col_inp * 4 * BM + inner_row_inp] = tmp.x;
    shared_inp[(inner_col_inp * 4 + 1) * BM + inner_row_inp] = tmp.y;
    shared_inp[(inner_col_inp * 4 + 2) * BM + inner_row_inp] = tmp.z;
    shared_inp[(inner_col_inp * 4 + 3) * BM + inner_row_inp] = tmp.w;

    for (int i_row_wk_wv = inner_row_wk_wv; i_row_wk_wv + ROW_STRIDE_WK_WV <= BN; i_row_wk_wv += ROW_STRIDE_WK_WV) {
        float4 tmp;
        if (i_row_wk_wv >= emb_dim) {
            tmp = 0;
        } else {
            tmp = reinterpret_cast<const float4*>(kv + i_row_wk_wv * emb_dim + inner_col_wk_wv * 4)[0];
        }
        reinterpret_cast<float4*>(shared_wk_wv + inner_row_wk_wv * 4 * BN + inner_col_wk_wv)[0] = tmp;
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
    __shared__ float shared_wk_wv[BK * BN];

    if (output_block_row_id * BM >= cur_length) {
        return;
    }
    page_table += batch_idx * (n_sequence / PAGE_BLOCK_SIZE);

    constexpr int WSUBN = WN / WNITER;
    constexpr int WSUBM = WM / WMITER;

    

    for (int bk_idx = 0; bk_idx < emb_dim; bk_idx += BK) {

    }
}
