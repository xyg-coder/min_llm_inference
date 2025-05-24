#include "tensor.hpp"
#include "layers.h"
#include <stdexcept>
#include "kernels/gemm.h"

FeedForward::FeedForward(const TensorFloat& w, std::optional<const TensorFloat> b): weight_(w), bias_(std::move(b)) {
    auto weight_shape = weight_.shape();
    if (weight_shape.size() != 2) {
        throw std::invalid_argument("[FeedForward] Weight must be 2D (got shape size " +
            std::to_string(weight_shape.size()) + ")");
    }
    if (weight_shape[0] == 0 || weight_shape[1] == 0) {
        throw std::invalid_argument("[FeedForward] Weight dimensions must be greater than zero.");
    }

    if (bias_) {
        const auto bias_shape = bias_->shape();
        if (bias_shape.size() != 1) {
            throw std::invalid_argument("[FeedForward] Bias must be 1D (got shape size " +
                std::to_string(bias_shape.size()) + ")");
        }
        if (bias_shape[0] != weight_shape[1]) {
            throw std::invalid_argument("[FeedForward] Bias size must match weight's out_features (expected " +
                std::to_string(weight_shape[1]) + ", got " + std::to_string(bias_shape[0]) + ")");
        }
    }
}

void FeedForward::forward(const TensorFloat& input, TensorFloat& output) {
    size_t n_batch = input.shape()[0];
    size_t in_features = input.shape()[1];
    size_t out_features = weight_.shape()[1];

    float* bias_data_ptr = nullptr;
    Stride3D bias_stride({0, 0, 0});
    if (bias_) {
        bias_data_ptr = bias_->data();
        bias_stride = Stride3D({0, 0, 1});
    }

    launch_gemm_bias_kernel(
        input.data(), Stride3D{0, in_features, 1},
        weight_.data(), Stride3D{0, out_features, 1},
        bias_data_ptr, bias_stride,
        output.data(), Stride3D{0, out_features, 1},
        1, n_batch, in_features, out_features
    );
}
