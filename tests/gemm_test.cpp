#include <cstdio>
#include <gtest/gtest.h>
#include <random>
#include "tensor.hpp"
#include "kernels/gemm.h"

TEST(GemmKernelTest, MatrixMultiplication) {
    // Define matrix dimensions
    size_t batch_size = 1;
    size_t rows = 400;
    size_t N = 400;
    size_t cols = 400;

    // Create input tensors on the host
    Tensor<float> s1({batch_size, rows, N}, DeviceType::HOST);
    Tensor<float> s2({batch_size, N, cols}, DeviceType::HOST);
    Tensor<float> output({batch_size, rows, cols}, DeviceType::HOST);

    // Initialize input matrices
    float* s1_data = s1.data();
    float* s2_data = s2.data();
    for (size_t i = 0; i < rows * N; ++i) {
        s1_data[i] = static_cast<float>(1); // Fill with 1, 2, 3, ...
    }
    for (size_t i = 0; i < N * cols; ++i) {
        s2_data[i] = static_cast<float>(1); // Fill with 1, 2, 3, ...
    }

    // Copy input tensors to the device
    Tensor<float> d_s1({batch_size, rows, N}, DeviceType::DEVICE);
    Tensor<float> d_s2({batch_size, N, cols}, DeviceType::DEVICE);
    Tensor<float> d_output({batch_size, rows, cols}, DeviceType::DEVICE);
    d_s1.copy_from(s1);
    d_s2.copy_from(s2);

    // Launch the kernel
    launch_gemm_kernel<float>(d_s1.data(), d_s2.data(), d_output.data(), batch_size, rows, N, cols);

    // Copy the result back to the host
    output.copy_from(d_output);

    // Verify the result
    float* output_data = output.data();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float expected = 0;
            for (size_t k = 0; k < N; ++k) {
                expected += s1_data[i * N + k] * s2_data[k * cols + j];
            }
            ASSERT_NEAR(output_data[i * cols + j], expected, 1e-5)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(GemmKernelTest, MultiBatchesMatrixMultiplication) {
    // Define matrix dimensions
    size_t batch_size = 5;
    size_t rows = 400;
    size_t N = 400;
    size_t cols = 400;

    // Create input tensors on the host
    Tensor<float> s1({batch_size, rows, N}, DeviceType::HOST);
    Tensor<float> s2({batch_size, N, cols}, DeviceType::HOST);
    Tensor<float> output({batch_size, rows, cols}, DeviceType::HOST);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); // Range: [0.0, 1.0)

    // Initialize input matrices
    float* s1_data = s1.data();
    float* s2_data = s2.data();
    for (size_t i = 0; i < rows * N; ++i) {
        s1_data[i] = dis(gen);
    }
    for (size_t i = 0; i < N * cols; ++i) {
        s2_data[i] = dis(gen);
    }

    // Copy input tensors to the device
    Tensor<float> d_s1({batch_size, rows, N}, DeviceType::DEVICE);
    Tensor<float> d_s2({batch_size, N, cols}, DeviceType::DEVICE);
    Tensor<float> d_output({batch_size, rows, cols}, DeviceType::DEVICE);
    d_s1.copy_from(s1);
    d_s2.copy_from(s2);

    // Launch the kernel
    launch_gemm_kernel<float>(d_s1.data(), d_s2.data(), d_output.data(), batch_size, rows, N, cols);

    // Copy the result back to the host
    output.copy_from(d_output);

    // Verify the result
    float* output_data = output.data();
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                float expected = 0;
                for (size_t k = 0; k < N; ++k) {
                    expected += s1_data[b * rows * N + i * N + k] * s2_data[b * N * cols + k * cols + j];
                }
                ASSERT_NEAR(output_data[b * rows * cols + i * cols + j], expected, 1e-3)
                    << "Mismatch at (" << b << ", " << i << ", " << j << ")";
            }
        }
    }
}
