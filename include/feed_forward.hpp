#pragma once
#include "model.h"
#include "tensor.hpp"
#include <optional>
#include <stdexcept>
#include <variant>
#include "kernels/gemm.cuh"

template <typename T>
class FeedForward : public Layer<T> {
public:
    FeedForward(Tensor<T> w, std::optional<Tensor<T>> b = std::nullopt): weight_(std::move(w)), bias_(std::move(b)) {
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
    };
    
    ModelIO<T> forward(const ModelIO<T>& input) override;
private:
    Tensor<T> weight_;
    std::optional<Tensor<T>> bias_;
};

template <typename T>
ModelIO<T> FeedForward<T>::forward(const ModelIO<T>& input) {
    if (!std::holds_alternative<Tensor<T>>(input)) {
        throw std::invalid_argument("[FeedForward] Input must be a single Tensor (not a list or dict)");
    }

    const Tensor<T>& input_t = std::get<Tensor<T>>(input);
    size_t n_batch = input_t.shape()[0];
    size_t in_features = input_t.shape()[1];
    size_t out_features = weight_.shape()[1];

    Tensor<T> output_tensor({n_batch, out_features}, DeviceType::DEVICE);
    T* bias_data_ptr = nullptr;
    Stride3D bias_stride({0, 0, 0});
    if (bias_) {
        bias_data_ptr = bias_->data();
        bias_stride = Stride3D({0, 0, 1});
    }

    launch_gemm_bias_kernel(
        input_t.data(), Stride3D{0, in_features, 1},
        weight_.data(), Stride3D{0, out_features, 1},
        bias_data_ptr, bias_stride,
        output_tensor.data(), Stride3D{0, out_features, 1},
        1, n_batch, in_features, out_features
    );

    return output_tensor;
}
