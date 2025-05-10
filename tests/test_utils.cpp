#include "test_utils.h"
#include "kernels/rand_assign.h"
#include "tensor.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <utility>
#include <gtest/gtest.h>
#include <vector>

std::pair<Tensor, Tensor> get_random_device_host_tensor(const std::vector<size_t> &shape, float ratio) {
    Tensor device_tensor(shape, DeviceType::DEVICE);    
    Tensor host_tensor(shape, DeviceType::HOST);
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    launch_randn_kernel(device_tensor.data(), total_size, ratio);
    host_tensor.copy_from(device_tensor);
    return std::make_pair(device_tensor, host_tensor);
}

void assert_near(const Tensor &tensor_device, const Tensor &tensor_host, float threshold) {
    size_t total_size = 1;
    for (int i = 0; i < tensor_device.shape().size(); ++i) {
        ASSERT_EQ(tensor_device.shape()[i], tensor_host.shape()[i]);
        total_size *= tensor_device.shape()[i];
    }
    Tensor copy_from_device(tensor_device.shape(), DeviceType::HOST);
    copy_from_device.copy_from(tensor_device);
    for (size_t i = 0; i < total_size; ++i) {
        ASSERT_NEAR(copy_from_device.data()[i], tensor_host.data()[i], threshold)
            << "Mismatch at (" << i <<  ")";
    }
}

Tensor host_matrix_multiply(const Tensor& inp1, const Tensor& inp2) {
    const std::vector<size_t>& shape1 = inp1.shape();
    const std::vector<size_t>& shape2 = inp2.shape();
    assert(shape1.size() == shape2.size());
    assert(shape1.size() == 2 || shape1.size() == 3);
    if (shape1.size() == 3) {
        size_t n_batch = shape1[0];
        size_t rows = shape1[1];
        size_t N = shape1[2];
        assert(shape2[0] == n_batch);
        assert(shape2[1] == N);
        size_t cols = shape2[2];
        Tensor result_tensor({n_batch, rows, cols}, DeviceType::HOST);
        for (int i = 0; i < n_batch; ++i) {
            for (int j = 0; j < rows; ++j) {
                for (int k = 0; k < cols; ++k) {
                    float result = 0;
                    for (int n = 0; n < N; ++n) {
                        result += (inp1.data()[i * rows * N + j * N + n] * inp2.data()[i * N * cols + n * cols + k]);
                    }
                    result_tensor.data()[i * rows * cols + j * cols + k] = result;
                }
            }
        }
        return result_tensor;
    } else {
        size_t rows = shape1[0];
        size_t N = shape1[1];
        assert(shape2[0] == N);
        size_t cols = shape2[1];
        Tensor result_tensor({rows, cols}, DeviceType::HOST);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float result = 0;
                for (int n = 0; n < N; ++n) {
                    result += (inp1.data()[i * N + n] * inp2.data()[n * cols + j]);
                }
                result_tensor.data()[i * cols + j] = result;
            }
        }
        return result_tensor;
    }
}

Tensor softmax(const Tensor &inp) {
    Tensor result(inp.shape());
    int cols = inp.shape()[inp.shape().size() - 1];
    int total_size = inp.get_total_size();
    int rows = total_size / cols;
    for (int y = 0; y < rows; ++y) {
        float maxval = -std::numeric_limits<float>::infinity();
        for (int x = 0; x < cols; ++x) {
            maxval = std::max(maxval, inp.data()[y * cols + x]);
        }
        float sum = 0;
        for (int x = 0; x < cols; ++x) {
            sum += std::exp(inp.data()[y * cols + x] - maxval);
        }
        for (int x = 0; x < cols; ++x) {
            result.data()[y * cols + x] = std::exp(inp.data()[y * cols + x] - maxval) / sum;
        }
    }
    return result;
}

void print_host(const float *data, int size) {
    for (int i = 0; i < size; ++i) {
        printf("printing host: %d = %f\n", i, data[i]);
    }
}
