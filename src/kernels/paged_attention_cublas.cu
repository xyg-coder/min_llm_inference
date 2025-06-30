#include "constants.h"
#include "tensor.hpp"
#include "utils.h"
#include "kernels/self_attention_inference_optimized.h"
#include <cassert>
#include <cstdlib>
#include <cublas_v2.h>
#include "kernels/paged_attention.h"
#include "kernels/templated_kernels.cuh"

/**
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * lengths: [n_batch]
 * latest_embs: [n_batch, embs]
 */
__global__ void get_latest_batch_embs(
    const float** page_table, const int* lengths,
    float* latest_embs,
    int n_batch, int n_sequence, int emb_dim) {

    int i_batch = blockIdx.y;
    int cur_length = lengths[i_batch];
    if (cur_length == 0) {
        return;
    }
    int i_sequence = cur_length - 1;
    int i_dim = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ const float* page_pos;
    if (threadIdx.x == 0) {
        page_pos = page_table[i_batch * (n_sequence / PAGE_BLOCK_SIZE) + i_sequence / PAGE_BLOCK_SIZE];
    }
    __syncthreads();
    if (i_dim < emb_dim) {
        latest_embs[i_batch * emb_dim + i_dim] = get_page_table_value(page_pos, i_batch, n_sequence, i_sequence, emb_dim, PAGE_BLOCK_SIZE, i_dim, INP_EMB_EMB_OFFSET);
    }
}


/**
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * lengths: [n_batch]
 * latest_k, latest_v: [n_batch, embs]
 */
__global__ void save_to_page_table(
    float** page_table, const int* lengths, 
    const float* latest_k, const float* latest_v,
    int n_batch, int n_sequence, int emb_dim) {
    
    int i_batch = blockIdx.y;
    int cur_length = lengths[i_batch];
    if (cur_length == 0) {
        return;
    }
    int i_sequence = cur_length - 1;
    int i_dim = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float* page_pos;
    if (threadIdx.x == 0) {
        page_pos = page_table[i_batch * (n_sequence / PAGE_BLOCK_SIZE) + i_sequence / PAGE_BLOCK_SIZE];
    }
    __syncthreads();
    if (i_dim < emb_dim) {
        set_page_table_value(page_pos, i_batch, n_sequence, i_sequence, emb_dim, PAGE_BLOCK_SIZE, i_dim, K_CACHE_EMB_OFFSET, latest_k[i_batch * emb_dim + i_dim]);
        set_page_table_value(page_pos, i_batch, n_sequence, i_sequence, emb_dim, PAGE_BLOCK_SIZE, i_dim, V_CACHE_EMB_OFFSET, latest_v[i_batch * emb_dim + i_dim]);
    }
}

/**
 * page_table: [n_batch, n_sequence / PAGE_BLOCK_SIZE]
 * lengths: [n_batch]
 * wq, wk, wv: [emb_dim, emb_dim]
 * q_output: [n_batch, emb_dim]
 * temp_placeholder: [n_batch, emb_dim]
 */
void launch_get_latest_k_q_v_paged_attention_cublas(
    TensorFloatPoint& page_table, const TensorInt& lengths,
    TensorFloat& latest_emb,
    const TensorFloat& wk, const TensorFloat& wq,
    const TensorFloat& wv, TensorFloat& q_output, TensorFloat& temp_placeholder,
    cublasHandle_t& handle, int n_sequence) {

    int n_batch = page_table.shape()[0];
    int emb_dim = wq.shape()[0];
    
    dim3 gridDim(ceil_div(emb_dim, TILE_SIZE_SQUARE), n_batch);
    get_latest_batch_embs<<<gridDim, TILE_SIZE_SQUARE>>>((const float**)page_table.data(), lengths.data(), latest_emb.data(), n_batch, n_sequence, emb_dim);
    CUDA_CHECK_LAST();

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, emb_dim, n_batch, emb_dim, &alpha, wk.data(),
        emb_dim, latest_emb.data(), emb_dim, &beta, q_output.data(), emb_dim));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, emb_dim, n_batch, emb_dim, &alpha, wv.data(),
        emb_dim, latest_emb.data(), emb_dim, &beta, temp_placeholder.data(), emb_dim));
    save_to_page_table<<<gridDim, TILE_SIZE_SQUARE>>>(page_table.data(), lengths.data(), q_output.data(), temp_placeholder.data(), n_batch, n_sequence, emb_dim);
    CUDA_CHECK_LAST();
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, emb_dim, n_batch, emb_dim, &alpha, wq.data(),
        emb_dim, latest_emb.data(), emb_dim, &beta, q_output.data(), emb_dim));
}

constexpr int WARPSIZE = 32;

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

    // place the warp in the block
    int warpIdx = threadIdx.x / WARPSIZE;
    int warpRow = warpIdx / (BN / WN);
    int warpCol = warpIdx % (BN / WN);

    // The starting position is [output_block_row_id * BM, output_block_col_id * BN]
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");
    static_assert(N_THREADS * 4 % BK == 0, "N_THREADS * 4 must be divisible by BK");
    static_assert(N_THREADS * 4 % BN == 0, "N_THREADS * 4 must be divisible by BK");
    static_assert(WSUBN % TN == 0, "WSUBN must be divisible by TN");
    static_assert(WSUBM % TM == 0, "WSUBM must be divisible by TM");
    static_assert(WSUBM <= WARP_SIZE, "WSUBM must be less than or equal to WARP_SIZE to avoid bank conflicts");
    static_assert(WSUBN <= WARP_SIZE, "WSUBN must be less than or equal to WARP_SIZE to avoid bank conflicts");
    static_assert(TN >= 4, "TN must be greater than or equal to 4 to ensure we can use float4");
    
    __shared__ float shared_inp[BM * BK];
    __shared__ float shared_wk[BK * BN];
    __shared__ float shared_wv[BK * BN];

    if (cRow * BM >= cur_length) {
        return;
    }
    page_table += batch_idx * (n_sequence / PAGE_BLOCK_SIZE);

    constexpr int rowStrideA = (N_THREADS * 4) / BK;
    constexpr int rowStrideB = (N_THREADS * 4) / BN;

    // Place the thread in the block
    int innerRowA = (threadIdx.x / (BK / 4));
    int innerColA = (threadIdx.x % (BK / 4));
    int innerRowB = (threadIdx.x / (BN / 4));
    int innerColB = (threadIdx.x % (BN / 4));

    float reg_inp[TM * WMITER] = {0};
    float reg_wk[TN * WNITER] = {0};
    float reg_wv[TN * WNITER] = {0};
    float thread_result_k[TM * WMITER * TN * WNITER] = {0};
    float thread_result_v[TM * WMITER * TN * WNITER] = {0};


    // place the thread in the warp
    int threadIdxInWarp = threadIdx.x % WARPSIZE;
    int threadRowInWarp = threadIdxInWarp / (WSUBN / TN);
    int threadColInWarp = threadIdxInWarp % (WSUBN / TN);

    for (int bk_idx = 0; bk_idx < emb_dim; bk_idx += BK) {
        load_from_page_table<N_THREADS, rowStrideA, BM>(
            page_table, n_sequence, emb_dim,
            shared_inp, innerRowA, innerColA, cRow * BM, bk_idx, cur_length);
        load_from_kv<N_THREADS, rowStrideB, BN, BK>(
            wk, emb_dim,
            shared_wk, innerRowB, innerColB, bk_idx, cCol * BN);
        load_from_kv<N_THREADS, rowStrideB, BN, BK>(
            wv, emb_dim,
            shared_wv, innerRowB, innerColB, bk_idx, cCol * BN);
        __syncthreads();

        process_result<WM, WN, WMITER, WNITER, TM, TN, BM, BN, BK, WSUBM, WSUBN>(
            shared_inp, shared_wk, shared_wv,
            reg_inp, reg_wk, reg_wv,
            thread_result_k, thread_result_v,
            warpRow, warpCol, threadRowInWarp, threadColInWarp);
        __syncthreads();
    }

    int cached_paged_idx = -1;
    float* cached_page_pos = nullptr;
    for (int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (int resIdxM = 0; resIdxM < TM; ++resIdxM) {
            int output_row = cRow * BM + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + resIdxM;
            if (output_row < cur_length) {
                int page_idx = output_row / PAGE_BLOCK_SIZE;
                if (cached_paged_idx != page_idx) {
                    cached_paged_idx = page_idx;
                    cached_page_pos = page_table[page_idx];
                }

                for (int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    for (int resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                        int output_col = cCol * BN + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + resIdxN;
                        if (output_col < emb_dim) {
                            assert(output_col + 3 < emb_dim);
                            int k_offset = (output_row % PAGE_BLOCK_SIZE) * emb_dim * 3 + emb_dim * K_CACHE_EMB_OFFSET + output_col;
                            int v_offset = (output_row % PAGE_BLOCK_SIZE) * emb_dim * 3 + emb_dim * V_CACHE_EMB_OFFSET + output_col;
                            reinterpret_cast<float4*>(cached_page_pos + k_offset)[0] = reinterpret_cast<float4*>(thread_result_k + (wSubRowIdx * TM + resIdxM) * WNITER * TN + wSubColIdx * TN + resIdxN)[0];
                            reinterpret_cast<float4*>(cached_page_pos + v_offset)[0] = reinterpret_cast<float4*>(thread_result_v + (wSubRowIdx * TM + resIdxM) * WNITER * TN + wSubColIdx * TN + resIdxN)[0];
                        }
                    }
                }
            }
        }
    }
}

void launch_fill_new_k_v_cache_paged_attention_warp_tiling(
    TensorFloatPoint page_table, const TensorInt& new_batch_idx, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wv, int n_new_items, int n_sequence) {

    if (n_new_items == 0) {
        return;
    }

    int n_batch = page_table.shape()[0];
    assert(page_table.shape()[1] == n_sequence / PAGE_BLOCK_SIZE && n_sequence % PAGE_BLOCK_SIZE == 0);
    int emb_dim = wk.shape()[0];
    assert(wk.shape()[0] == wk.shape()[1]);
    constexpr int N_THREADS = 128;
    constexpr int BM = 64, BN = 64, BK = 64, WM = 32, WN = 32, TM = 4, TN = 4, WNITER = 2;

    fill_new_k_v_cache_paged_attention_warp_tiling<BM, BN, BK, WM, WN, TM, TN, WNITER, N_THREADS>
        <<<dim3(ceil_div(emb_dim, BN), ceil_div(n_sequence, BM), n_new_items), N_THREADS>>>(
            page_table.data(), new_batch_idx.data(), lengths.data(),
            wk.data(), wv.data(), n_sequence, emb_dim);
    
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
void paged_attention_with_cublas(
    TensorFloatPoint& page_table,
    const TensorInt& lengths,
    const TensorFloat& wk,
    const TensorFloat& wq,
    const TensorFloat& wv,
    const TensorInt& new_batch_idx,
    TensorFloat& q_output, TensorFloat& qkt_output, TensorFloat& attention_result,
    TensorFloat& latest_emb, TensorFloat& temp_placeholder,
    int n_new_items, int n_sequence, cublasHandle_t& handle) {

    launch_fill_new_k_v_cache_paged_attention(page_table, new_batch_idx, lengths, wk, wv, n_new_items, n_sequence);

    launch_get_latest_k_q_v_paged_attention_cublas(page_table, lengths, latest_emb, wk, wq, wv, q_output, temp_placeholder, handle, n_sequence);

    launch_qkt_paged_attention(q_output, page_table, lengths, qkt_output);

    launch_softmax_in_place_with_lengths(qkt_output, lengths);

    launch_softmax_v_paged_attention(qkt_output, page_table, attention_result, lengths);
}
