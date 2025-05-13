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
    constexpr int block_dim = 256;
    int grid_dim = ceil_div(batch_size * n_sequence * embedding_dim / 4, block_dim);
    encoder_kernel<<<grid_dim, block_dim>>>(wte, wpe, inp, output, batch_size, n_sequence, embedding_dim);
    CUDA_CHECK_LAST();
}
