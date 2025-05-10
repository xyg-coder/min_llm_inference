#include "tensor.h"
#include "test_utils.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>
#include "kernels/self_attention.h"


Tensor get_kqt_host(const Tensor& kqv) {
    const std::vector<size_t>& shape = kqv.shape();
    size_t n_batch = shape[0];
    size_t n_sequence = shape[1];
    size_t dims_3 = shape[2];
    size_t dim = dims_3 / 3;

    Tensor k_host({n_batch, n_sequence, dim}, DeviceType::HOST);
    Tensor qt_host({n_batch, dim, n_sequence}, DeviceType::HOST);
    for (int i = 0; i < n_batch; ++i) {
        for (int j = 0; j < n_sequence; ++j) {
            for (int k = 0; k < dim; ++k) {
                k_host.data()[i * n_sequence * dim + j * dim + k] = kqv.data()[
                    i * n_sequence * dims_3 + j * dims_3 + k] / std::sqrt(dim);
                // qt[i][k][j] = kqv[i][j][dim + k]
                qt_host.data()[i * n_sequence * dim + k * n_sequence + j] = kqv.data()[
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
    Tensor kqt_host = get_kqt_host(kqv_device_host.second);
    Tensor kqt_device({n_batch, n_sequence, n_sequence}, DeviceType::DEVICE);
    launch_kqt_kernel(kqv_device_host.first.data(), kqt_device.data(), n_batch, n_sequence, dim);
    assert_near(kqt_device, kqt_host);
}

Tensor duplicate_batch(const Tensor& inp, int n_batch) {
    assert(inp.shape().size() == 2);
    size_t rows = inp.shape()[0];
    size_t cols = inp.shape()[1];
    Tensor duplicated({(size_t)n_batch, rows, cols}, DeviceType::HOST);
    for (int i = 0; i < n_batch; ++i) {
        std::copy(inp.data(), inp.data() + rows * cols, duplicated.data() + i * rows * cols);
    }
    return duplicated;
}

// [n_batch, n_sequence, outdim * 3] -> [n_batch, n_sequence, output_dim]
Tensor get_batched_v(const Tensor& kqv) {
    Tensor result({kqv.shape()[0], kqv.shape()[1], kqv.shape()[2] / 3}, DeviceType::HOST);
    int n_rows = kqv.shape()[0] * kqv.shape()[1];
    int cols = kqv.shape()[2] / 3;
    for (int i = 0; i < n_rows; ++i) {
        std::copy(kqv.data() + i * cols * 3 + cols * 2, kqv.data() + i * cols * 3 + cols * 3, result.data() + i * cols);
    }
    return result;
}

Tensor self_attention_host(const Tensor& inp_host, const Tensor& wk_wq_wv_host) {
    Tensor duplicated_wk_wq_wv = duplicate_batch(wk_wq_wv_host, inp_host.shape()[0]);
    // kqv: [n_batch, n_sequence, out_dim & 3]
    Tensor kqv = host_matrix_multiply(inp_host, duplicated_wk_wq_wv);
    Tensor kqt = get_kqt_host(kqv);
    Tensor softmax_kqt = softmax(kqt);
    Tensor result = host_matrix_multiply(softmax_kqt, get_batched_v(kqv));
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
    Tensor self_attention_device_tensor = self_attention(inp_device_host.first, wk_wq_wv_device_host.first);
    Tensor self_attention_host_tensor = self_attention_host(inp_device_host.second, wk_wq_wv_device_host.second);
    assert_near(self_attention_device_tensor, self_attention_host_tensor);
}
