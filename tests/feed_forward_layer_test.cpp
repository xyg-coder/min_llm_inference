#include "feed_forward.h"
#include "model.h"
#include "tensor.h"
#include <gtest/gtest.h>
#include <random>

TEST(FeedForwardLayerTest, FeedForwardTest) {
    size_t batch_size = 1024;
    size_t in_features = 1024;
    size_t out_features = 1024;

    Tensor weights({in_features, out_features}, DeviceType::HOST);
    Tensor bias({out_features}, DeviceType::HOST);
    Tensor input({batch_size, in_features}, DeviceType::HOST);
    Tensor output({batch_size, out_features}, DeviceType::HOST);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); // Range: [0.0, 1.0)

    // Initialize input matrices
    float* weight_data = weights.data();
    float* bias_data = bias.data();
    float* input_data = input.data();

    for (size_t i = 0; i < in_features * out_features; ++i) {
        weight_data[i] = dis(gen);
    }
    for (size_t i = 0; i < out_features; ++i) {
        bias_data[i] = dis(gen);
    }
    for (size_t i = 0; i < batch_size * in_features; ++i) {
        input_data[i] = dis(gen);
    }

    Tensor weights_device({in_features, out_features}, DeviceType::DEVICE);
    Tensor bias_device({out_features}, DeviceType::DEVICE);
    Tensor input_device({batch_size, in_features}, DeviceType::DEVICE);
    weights_device.copy_from(weights);
    bias_device.copy_from(bias);
    input_device.copy_from(input);

    FeedForward layer(std::move(weights_device), std::move(bias_device));

    auto output_tensor = layer.forward(input_device);

    const Tensor& output_d = std::get<Tensor>(output_tensor);
    output.copy_from(output_d);
    float* output_data = output.data();
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            float expected = 0;
            for (size_t k = 0; k < in_features; ++k) {
                expected += input_data[i * in_features + k] * weight_data[k * out_features + j];
            }
            expected += bias_data[j];
            ASSERT_NEAR(output_data[i * out_features + j], expected, 1e-3) << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(FeedForwardLayerTest, FeedForwardNoBiasTest) {
    size_t batch_size = 1024;
    size_t in_features = 1024;
    size_t out_features = 1024;

    Tensor weights({in_features, out_features}, DeviceType::HOST);
    Tensor input({batch_size, in_features}, DeviceType::HOST);
    Tensor output({batch_size, out_features}, DeviceType::HOST);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0); // Range: [0.0, 1.0)

    // Initialize input matrices
    float* weight_data = weights.data();
    float* input_data = input.data();

    for (size_t i = 0; i < in_features * out_features; ++i) {
        weight_data[i] = dis(gen);
    }
    for (size_t i = 0; i < batch_size * in_features; ++i) {
        input_data[i] = dis(gen);
    }

    Tensor weights_device({in_features, out_features}, DeviceType::DEVICE);
    Tensor input_device({batch_size, in_features}, DeviceType::DEVICE);
    weights_device.copy_from(weights);
    input_device.copy_from(input);

    FeedForward layer(std::move(weights_device));

    auto output_tensor = layer.forward(input_device);

    const Tensor& output_d = std::get<Tensor>(output_tensor);
    output.copy_from(output_d);
    float* output_data = output.data();
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            float expected = 0;
            for (size_t k = 0; k < in_features; ++k) {
                expected += input_data[i * in_features + k] * weight_data[k * out_features + j];
            }
            ASSERT_NEAR(output_data[i * out_features + j], expected, 1e-3) << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}
