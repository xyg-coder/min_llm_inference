#include "tensor.h"
#include "test_utils.h"
#include <cmath>
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
