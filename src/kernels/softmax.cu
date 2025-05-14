#include <cfloat>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <assert.h>
#include <vector_types.h>
#include <kernels/utils.cuh>
#include <kernels/softmax.h>
#include <utils.h>

const int WARP_SIZE = 32;
const int SOFTMAX_BLOCK_SIZE = 256;

namespace cg = cooperative_groups;
/** calculate the softmax for inp:[N, T, T]
The out shoudl be [N, T, T]
*/
__global__ void softmax(float* out, const float* inp, int N, int T) {
    assert(T % 4 == 0);
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    int row_id = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (row_id >= N * T) {
        return;
    }

    const float* x = inp + row_id * T;
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    // It's faster to load 4 floats altogether
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = warp.thread_rank(); i < T / 4; i += warp.num_threads()) {
        float4 v = x_vec[i];
        float old_maxval = maxval;
        for (int k = 0; k < 4; k++) {
            maxval = fmax(old_maxval, vec_at(v, k));
        }
        sumval *= expf(old_maxval - maxval);
        for (int k = 0; k < 4; k++) {
            sumval += expf(vec_at(v, k) - maxval);
        }
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(maxval - global_maxval);
    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;
    float4* out_vec = reinterpret_cast<float4*>(out + row_id * T);
    float output_temp[4];
    for (int i = warp.thread_rank(); i < T / 4; i += warp.num_threads()) {
        float4 v = x_vec[i];
        for (int k = 0; k < 4; k++) {
            output_temp[k] = expf(vec_at(v, k) - global_maxval) * norm;
        }
        out_vec[i] = *reinterpret_cast<float4*>(output_temp);
    }
}

__global__ void softmax_in_place(float* inp, int N, int T) {
    assert(T % 4 == 0);
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    int row_id = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (row_id >= N * T) {
        return;
    }

    const float* x = inp + row_id * T;
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    // It's faster to load 4 floats altogether
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    for (int i = warp.thread_rank(); i < T / 4; i += warp.num_threads()) {
        float4 v = x_vec[i];
        float old_maxval = maxval;
        for (int k = 0; k < 4; k++) {
            maxval = fmax(old_maxval, vec_at(v, k));
        }
        sumval *= expf(old_maxval - maxval);
        for (int k = 0; k < 4; k++) {
            sumval += expf(vec_at(v, k) - maxval);
        }
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(maxval - global_maxval);
    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;
    float4* out_vec = reinterpret_cast<float4*>(inp + row_id * T);
    float output_temp[4];
    for (int i = warp.thread_rank(); i < T / 4; i += warp.num_threads()) {
        float4 v = x_vec[i];
        for (int k = 0; k < 4; k++) {
            output_temp[k] = expf(vec_at(v, k) - global_maxval) * norm;
        }
        out_vec[i] = *reinterpret_cast<float4*>(output_temp);
    }
}

void launch_softmax_kernel(const float* inp, float* out, int N, int T) {
    int grid_size = ceil_div(N * T * WARP_SIZE, SOFTMAX_BLOCK_SIZE);
    softmax<<<grid_size, SOFTMAX_BLOCK_SIZE>>>(out, inp, N, T);
    CUDA_CHECK_LAST();
}

void launch_softmax_in_place_kernel(float* inp, int N, int T) {
    int grid_size = ceil_div(N * T * WARP_SIZE, SOFTMAX_BLOCK_SIZE);
    softmax_in_place<<<grid_size, SOFTMAX_BLOCK_SIZE>>>(inp, N, T);
    CUDA_CHECK_LAST();
}
