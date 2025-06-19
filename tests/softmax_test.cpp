#include <cmath>
#include <gtest/gtest.h>
#include "kernels/rand_assign.h"
#include "kernels/softmax.h"
#include "tensor.hpp"
#include <cuda_runtime.h>

TEST(SoftmaxTest, SoftMaxTest) {
    constexpr int N = 8, T = 1024;
    TensorFloat input_h({N, T, T}, DeviceType::HOST);
    TensorFloat input_d({N, T, T}, DeviceType::DEVICE);

    launch_randn_kernel(input_d.data(), N * T * T);
    input_h.copy_from(input_d);
    float* input_data = input_h.data();
    

    TensorFloat output_h({N, T, T}, DeviceType::HOST);
    float* output_data = output_h.data();

    // Compute expected softmax results on CPU for each row
    for (int row = 0; row < N; ++row) {
        float maxval = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < T; ++col) {
            maxval = std::max(maxval, input_data[row*T + col]);
        } 
        float sum = 0.0f;
        for (int col = 0; col < T; ++col) {
            sum += std::exp(input_data[row*T + col] - maxval);
        }
        for (int col = 0; col < T; ++col) {
            output_data[row*T + col] = std::exp(input_data[row*T + col] - maxval) / sum;
        }
    }


    TensorFloat output_d({N, T, T}, DeviceType::DEVICE);

    // Call the kernel launcher
    launch_softmax_kernel(input_d.data(), output_d.data(), N, T);

    TensorFloat output_copy_h({N, T, T}, DeviceType::HOST);
    output_copy_h.copy_from(output_d);


    // Check results
    for (int i = 0; i < N * T; ++i) {
        ASSERT_NEAR(output_copy_h.data()[i], output_data[i], 1e-4f);
    }
}

TEST(SoftmaxTest, SoftMaxInPlaceTest) {
    constexpr int N = 8, T = 1024;
    TensorFloat input_h({N, T, T}, DeviceType::HOST);
    TensorFloat input_d({N, T, T}, DeviceType::DEVICE);

    launch_randn_kernel(input_d.data(), N * T * T);
    input_h.copy_from(input_d);
    float* input_data = input_h.data();
    

    TensorFloat output_h({N, T, T}, DeviceType::HOST);
    float* output_data = output_h.data();

    // Compute expected softmax results on CPU for each row
    for (int row = 0; row < N; ++row) {
        float maxval = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < T; ++col) {
            maxval = std::max(maxval, input_data[row*T + col]);
        } 
        float sum = 0.0f;
        for (int col = 0; col < T; ++col) {
            sum += std::exp(input_data[row*T + col] - maxval);
        }
        for (int col = 0; col < T; ++col) {
            output_data[row*T + col] = std::exp(input_data[row*T + col] - maxval) / sum;
        }
    }


    // Call the kernel launcher
    launch_softmax_in_place_kernel(input_d.data(), N, T);

    input_h.copy_from(input_d);


    // Check results
    const float* input_h_ptr = input_h.data();
    for (int i = 0; i < N * T; ++i) {
        ASSERT_NEAR(input_h_ptr[i], output_data[i], 1e-4f);
    }
}

// Just to test that the cuda can run large scale softmax
TEST(SoftmaxTest, SoftMaxLargeTest) {
    constexpr int N = 400, T = 512;
    TensorFloat input_h({N, T, T}, DeviceType::HOST);
    TensorFloat input_d({N, T, T}, DeviceType::DEVICE);

    launch_randn_kernel(input_d.data(), N * T * T);

    input_h.copy_from(input_d);

    TensorFloat output_h({N, T, T}, DeviceType::HOST);
    float* output_data = output_h.data();

    const int CPU_N = 20;
    float* input_data = input_h.data();
    // Compute expected softmax results on CPU for each row
    for (int row = 0; row < CPU_N; ++row) {
        float maxval = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < T; ++col) {
            maxval = std::max(maxval, input_data[row*T + col]);
        } 
        float sum = 0.0f;
        for (int col = 0; col < T; ++col) {
            sum += std::exp(input_data[row*T + col] - maxval);
        }
        for (int col = 0; col < T; ++col) {
            output_data[row*T + col] = std::exp(input_data[row*T + col] - maxval) / sum;
        }
    }

    TensorFloat output_d({N, T, T}, DeviceType::DEVICE);

    // Call the kernel launcher
    launch_softmax_kernel(input_d.data(), output_d.data(), N, T);

    TensorFloat output_copy_h({N, T, T}, DeviceType::HOST);
    output_copy_h.copy_from(output_d);


    // Check results
    const float* output_copy_h_ptr = output_copy_h.data();
    for (int i = 0; i < CPU_N * T; ++i) {
        ASSERT_NEAR(output_copy_h_ptr[i], output_data[i], 1e-4f);
    }
}
