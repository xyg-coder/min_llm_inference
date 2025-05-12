#include "tensor.hpp"
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>
#include "kernels/self_attention.h"


TensorFloat get_kqt_host(const TensorFloat& kqv) {
    const std::vector<size_t>& shape = kqv.shape();
    size_t n_batch = shape[0];
    size_t n_sequence = shape[1];
    size_t dims_3 = shape[2];
    size_t dim = dims_3 / 3;

    TensorFloat k_host({n_batch, n_sequence, dim}, DeviceType::HOST);
    TensorFloat qt_host({n_batch, dim, n_sequence}, DeviceType::HOST);
    float* k_host_ptr = k_host.data();
    const float* kqv_ptr = kqv.data();
    float* qt_host_ptr = qt_host.data();
    for (int i = 0; i < n_batch; ++i) {
        for (int j = 0; j < n_sequence; ++j) {
            for (int k = 0; k < dim; ++k) {
                k_host_ptr[i * n_sequence * dim + j * dim + k] = kqv_ptr[
                    i * n_sequence * dims_3 + j * dims_3 + k] / std::sqrt(dim);
                // qt[i][k][j] = kqv[i][j][dim + k]
                qt_host_ptr[i * n_sequence * dim + k * n_sequence + j] = kqv_ptr[
                    i * n_sequence * dims_3 + j * dims_3 + dim + k];
            }
        }
    }
    return host_matrix_multiply(k_host, qt_host);
}


TEST(SelfAttentionTest, kqtKernelTest) {
    size_t n_batch = 10;
    size_t n_sequence = 100;
    size_t dim = 100;
    auto kqv_device_host = get_random_device_host_tensor({
        n_batch, n_sequence, dim * 3});
    TensorFloat kqt_host = get_kqt_host(kqv_device_host.second);
    TensorFloat kqt_device({n_batch, n_sequence, n_sequence}, DeviceType::DEVICE);
    launch_kqt_kernel(kqv_device_host.first.data(), kqt_device.data(), n_batch, n_sequence, dim);
    assert_near(kqt_device, kqt_host);
}

TensorFloat duplicate_batch(const TensorFloat& inp, int n_batch) {
    assert(inp.shape().size() == 2);
    size_t rows = inp.shape()[0];
    size_t cols = inp.shape()[1];
    TensorFloat duplicated({(size_t)n_batch, rows, cols}, DeviceType::HOST);
    const float* inp_ptr = inp.data();
    float* duplicated_ptr = duplicated.data();
    for (int i = 0; i < n_batch; ++i) {
        std::copy(inp_ptr, inp_ptr + rows * cols, duplicated_ptr + i * rows * cols);
    }
    return duplicated;
}

// [n_batch, n_sequence, outdim * 3] -> [n_batch, n_sequence, output_dim]
TensorFloat get_batched_v(const TensorFloat& kqv) {
    TensorFloat result({kqv.shape()[0], kqv.shape()[1], kqv.shape()[2] / 3}, DeviceType::HOST);
    int n_rows = kqv.shape()[0] * kqv.shape()[1];
    int cols = kqv.shape()[2] / 3;
    const float* kqv_ptr = kqv.data();
    float* result_ptr = result.data();
    for (int i = 0; i < n_rows; ++i) {
        std::copy(kqv_ptr + i * cols * 3 + cols * 2, kqv_ptr + i * cols * 3 + cols * 3, result_ptr + i * cols);
    }
    return result;
}

TensorFloat self_attention_host(const TensorFloat& inp_host, const TensorFloat& wk_wq_wv_host) {
    TensorFloat duplicated_wk_wq_wv = duplicate_batch(wk_wq_wv_host, inp_host.shape()[0]);
    // kqv: [n_batch, n_sequence, out_dim & 3]
    TensorFloat kqv = host_matrix_multiply(inp_host, duplicated_wk_wq_wv);
    TensorFloat kqt = get_kqt_host(kqv);
    TensorFloat softmax_kqt = softmax(kqt);
    TensorFloat result = host_matrix_multiply(softmax_kqt, get_batched_v(kqv));
    return result;
}

TEST(SelfAttentionTest, SelfAttentionTest) {
    size_t n_batch = 10;
    size_t n_sequence = 100;
    size_t in_dim = 100;
    size_t out_dim = 200;
    // need to have this ratio, otherwise the number might explode
    auto wk_wq_wv_device_host = get_random_device_host_tensor({
        in_dim, out_dim * 3}, 0.1);
    auto inp_device_host = get_random_device_host_tensor({
        n_batch, n_sequence, in_dim});
    TensorFloat self_attention_device_tensor = self_attention(inp_device_host.first, wk_wq_wv_device_host.first);
    TensorFloat self_attention_host_tensor = self_attention_host(inp_device_host.second, wk_wq_wv_device_host.second);
    assert_near(self_attention_device_tensor, self_attention_host_tensor);
}
