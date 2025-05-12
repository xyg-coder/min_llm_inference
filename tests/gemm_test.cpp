#include <cstdio>
#include <gtest/gtest.h>
#include "tensor.hpp"
#include "kernels/gemm.h"
#include "test_utils.h"

TEST(GemmKernelTest, MatrixMultiplication) {
    // Define matrix dimensions
    size_t batch_size = 1;
    size_t rows = 400;
    size_t N = 400;
    size_t cols = 400;

    auto s1_device_host = get_random_device_host_tensor({
        batch_size, rows, N});
    auto s2_device_host = get_random_device_host_tensor({
        batch_size, N, cols});

    TensorFloat d_output({batch_size, rows, cols}, DeviceType::DEVICE);
    // Launch the kernel
    launch_gemm_kernel(s1_device_host.first.data(), s2_device_host.first.data(), d_output.data(), batch_size, rows, N, cols);

    TensorFloat h_output = host_matrix_multiply(s1_device_host.second, s2_device_host.second);

    assert_near(d_output, h_output);
}

TEST(GemmKernelTest, MultiBatchesMatrixMultiplication) {
    // Define matrix dimensions
    size_t batch_size = 5;
    size_t rows = 400;
    size_t N = 400;
    size_t cols = 400;

    auto s1_device_host = get_random_device_host_tensor({
        batch_size, rows, N});
    auto s2_device_host = get_random_device_host_tensor({
        batch_size, N, cols});

    TensorFloat d_output({batch_size, rows, cols}, DeviceType::DEVICE);
    // Launch the kernel
    launch_gemm_kernel(s1_device_host.first.data(), s2_device_host.first.data(), d_output.data(), batch_size, rows, N, cols);

    TensorFloat h_output = host_matrix_multiply(s1_device_host.second, s2_device_host.second);

    assert_near(d_output, h_output);
}

TEST(GemmKernelTest, MultiBatchesBiasMatrixMultiplication) {
    // Define matrix dimensions
    size_t batch_size = 5;
    size_t rows = 40;
    size_t N = 40;
    size_t cols = 40;

    auto s1_device_host = get_random_device_host_tensor({
        batch_size, rows, N});
    auto s2_device_host = get_random_device_host_tensor({
        batch_size, N, cols});
    auto bias_device_host = get_random_device_host_tensor({
        batch_size, rows, cols});

    TensorFloat d_output({batch_size, rows, cols}, DeviceType::DEVICE);

    // Launch the kernel
    launch_gemm_bias_kernel(
        s1_device_host.first.data(), Stride3D{rows * N, N, 1},
        s2_device_host.first.data(), Stride3D{N * cols, cols, 1},
        bias_device_host.first.data(), Stride3D{rows * cols, cols, 1},
        d_output.data(), Stride3D{rows * cols, cols, 1},
        batch_size, rows, N, cols);

    TensorFloat h_output({batch_size, rows, cols}, DeviceType::HOST);

    // Copy the result back to the host
    h_output.copy_from(d_output);
    const float* s1_data = s1_device_host.second.data();
    const float* s2_data = s2_device_host.second.data();
    const float* bias_data = bias_device_host.second.data();

    // Verify the result
    float* output_data = h_output.data();
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                float expected = 0;
                for (size_t k = 0; k < N; ++k) {
                    expected += s1_data[b * rows * N + i * N + k] * s2_data[b * N * cols + k * cols + j];
                }
                expected += bias_data[b * rows * cols + i * cols + j];
                ASSERT_NEAR(output_data[b * rows * cols + i * cols + j], expected, 1e-3)
                    << "Mismatch at (" << b << ", " << i << ", " << j << ")";
            }
        }
    }
}

TEST(GemmKernelTest, MultiBatchesBiasWithStrideMatrixMultiplication) {
    // Define matrix dimensions
    size_t batch_size = 5;
    size_t rows = 40;
    size_t N = 40;
    size_t cols = 40;

    auto s1_device_host = get_random_device_host_tensor({
        batch_size, rows, N});
    auto s2_device_host = get_random_device_host_tensor({
        batch_size, N, cols});
    auto bias_device_host = get_random_device_host_tensor({
        cols});

    TensorFloat d_output({batch_size, rows, cols}, DeviceType::DEVICE);

    // Launch the kernel
    launch_gemm_bias_kernel(
        s1_device_host.first.data(), Stride3D{rows * N, N, 1},
        s2_device_host.first.data(), Stride3D{N * cols, cols, 1},
        bias_device_host.first.data(), Stride3D{0, 0, 1},
        d_output.data(), Stride3D{rows * cols, cols, 1},
        batch_size, rows, N, cols);

    TensorFloat h_output({batch_size, rows, cols}, DeviceType::HOST);

    // Copy the result back to the host
    h_output.copy_from(d_output);
    const float* s1_data = s1_device_host.second.data();
    const float* s2_data = s2_device_host.second.data();
    const float* bias_data = bias_device_host.second.data();

    // Verify the result
    float* output_data = h_output.data();
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                float expected = 0;
                for (size_t k = 0; k < N; ++k) {
                    expected += s1_data[b * rows * N + i * N + k] * s2_data[b * N * cols + k * cols + j];
                }
                expected += bias_data[j];
                ASSERT_NEAR(output_data[b * rows * cols + i * cols + j], expected, 1e-3)
                    << "Mismatch at (" << b << ", " << i << ", " << j << ")";
            }
        }
    }
}
