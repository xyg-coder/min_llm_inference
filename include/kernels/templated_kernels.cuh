#pragma once

#include <cuda_runtime.h>
#include "constants.h"

template<const int N_THREADS, const int rowStrideA,
         const int BM>
__device__ void load_from_page_table(
    float** page_table,
    int n_sequence, int emb_dim,
    float* shared_inp,
    int innerRowA,
    int innerColA, int row_offset, int k_offset, int cur_batch_length) {
    int cached_paged_idx = -1;
    float* cached_page_pos = nullptr;

    for (int row_in_block = innerRowA; row_in_block < BM; row_in_block += rowStrideA) {
        float4 tmp;
        if (row_offset + row_in_block >= cur_batch_length || k_offset + innerColA * 4 >= emb_dim) {
            tmp = {0.0f, 0.0f, 0.0f, 0.0f};
        } else {
            int page_idx = (row_offset + row_in_block) / PAGE_BLOCK_SIZE;
            if (cached_paged_idx != page_idx) {
                cached_paged_idx = page_idx;
                cached_page_pos = page_table[page_idx];
            }
            int inp_offset = ((row_offset + row_in_block) % PAGE_BLOCK_SIZE) * emb_dim * 3 + emb_dim * INP_EMB_EMB_OFFSET + k_offset + innerColA * 4;
            tmp = reinterpret_cast<const float4*>(cached_page_pos + inp_offset)[0];
        }
        shared_inp[innerColA * 4 * BM + row_in_block] = tmp.x;
        shared_inp[(innerColA * 4 + 1) * BM + row_in_block] = tmp.y;
        shared_inp[(innerColA * 4 + 2) * BM + row_in_block] = tmp.z;
        shared_inp[(innerColA * 4 + 3) * BM + row_in_block] = tmp.w;
    }
}

template<const int N_THREADS, const int rowStrideB,
         const int BN, const int BK>
__device__ void load_from_kv(
    const float* kv, int emb_dim,
    float* shared_wk_wv,
    int innerRowB, int innerColB, int k_offset, int col_offset) {

    for (int row_in_block = innerRowB; row_in_block < BK; row_in_block += rowStrideB) {
        float4 tmp;
        if (k_offset + row_in_block >= emb_dim || col_offset + innerColB * 4 >= emb_dim) {
            tmp = {0.0f, 0.0f, 0.0f, 0.0f};
        } else {
            tmp = reinterpret_cast<const float4*>(kv + (k_offset + innerRowB) * emb_dim + col_offset + innerColB * 4)[0];
        }
        reinterpret_cast<float4*>(shared_wk_wv + innerRowB * BN + innerColB * 4)[0] = tmp;
    }
}

template<const int WM, const int WN, const int WMITER, const int WNITER,
         const int TM, const int TN, const int BM, const int BN, const int BK, const int WSUBM, const int WSUBN>
__device__ void process_result(
    const float* shared_inp, const float* shared_wk, const float* shared_wv,
    float* reg_inp, float* reg_k, float* reg_v, float* thread_result_k, float* thread_result_v,
    int warpRow, int warpCol, int threadRowInWarp, int threadColInWarp) {

  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // populate registers for whole warptile
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        reg_inp[wSubRowIdx * TM + i] =
            shared_inp[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        reg_k[wSubColIdx * TN + i] =
            shared_wk[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
        reg_v[wSubColIdx * TN + i] =
            shared_wv[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // execute warptile matmul
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // calculate per-thread results
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            thread_result_k[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                reg_inp[wSubRowIdx * TM + resIdxM] *
                reg_k[wSubColIdx * TN + resIdxN];
            thread_result_v[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                reg_inp[wSubRowIdx * TM + resIdxM] *
                reg_v[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

// Forward declaration of kernel function (implemented in paged_attention_cublas.cu)
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int TM, const int TN, const int WNITER, const int N_THREADS>
__global__ void fill_new_k_v_cache_paged_attention_warp_tiling(
    float** page_table, const int* new_batch_idx,
    const int* lengths,
    const float* wk, const float* wv,
    int n_sequence, int emb_dim);
