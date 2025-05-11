#include "feed_forward.h"
#include "model.h"
#include "tensor.h"
#include "test_utils.h"
#include <gtest/gtest.h>

TEST(FeedForwardLayerTest, FeedForwardTest) {
    size_t batch_size = 1024;
    size_t in_features = 1024;
    size_t out_features = 1024;

    auto weights = get_random_device_host_tensor({in_features, out_features});
    Tensor weights_device = weights.first;
    Tensor weights_host = weights.second;

    auto inputs = get_random_device_host_tensor({batch_size, in_features});
    Tensor inputs_device = inputs.first;
    Tensor inputs_host = inputs.second;

    auto bias = get_random_device_host_tensor({out_features});
    Tensor bias_device = bias.first;
    Tensor bias_host = bias.second;

    Tensor output({batch_size, out_features}, DeviceType::HOST);

    FeedForward layer(std::move(weights_device), std::move(bias_device));

    auto output_tensor = layer.forward(inputs_device);

    const Tensor& output_d = std::get<Tensor>(output_tensor);
    output.copy_from(output_d);
    float* output_data = output.data();
    float* input_host_data = inputs_host.data();
    float* weight_host_data = weights_host.data();
    float* bias_host_data = bias_host.data();
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            float expected = 0;
            for (size_t k = 0; k < in_features; ++k) {
                expected += input_host_data[i * in_features + k] * weight_host_data[k * out_features + j];
            }
            expected += bias_host_data[j];
            ASSERT_NEAR(output_data[i * out_features + j], expected, 1e-3) << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

TEST(FeedForwardLayerTest, FeedForwardNoBiasTest) {
    size_t batch_size = 1024;
    size_t in_features = 1024;
    size_t out_features = 1024;

    auto weights = get_random_device_host_tensor({in_features, out_features});
    Tensor weights_device = weights.first;
    Tensor weights_host = weights.second;

    auto inputs = get_random_device_host_tensor({batch_size, in_features});
    Tensor inputs_device = inputs.first;
    Tensor inputs_host = inputs.second;

    auto bias = get_random_device_host_tensor({out_features});
    Tensor bias_device = bias.first;
    Tensor bias_host = bias.second;

    FeedForward layer(std::move(weights_device));

    auto output_tensor = layer.forward(inputs_device);

    Tensor output({batch_size, out_features}, DeviceType::HOST);
    const Tensor& output_d = std::get<Tensor>(output_tensor);
    output.copy_from(output_d);
    float* input_host_data = inputs_host.data();
    float* weight_host_data = weights_host.data();
    float* output_data = output.data();
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            float expected = 0;
            for (size_t k = 0; k < in_features; ++k) {
                expected += input_host_data[i * in_features + k] * weight_host_data[k * out_features + j];
            }
            ASSERT_NEAR(output_data[i * out_features + j], expected, 1e-3) << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}
