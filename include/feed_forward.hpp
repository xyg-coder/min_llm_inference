#pragma once
#include "model.h"
#include "tensor.hpp"
#include <memory>
#include <stdexcept>
#include <variant>

template <typename T>
class FeedForward : public Layer<T> {
public:
    FeedForward(std::unique_ptr<Tensor<T>> w, std::unique_ptr<Tensor<T>> b = nullptr): weight_(std::move(w)), bias_(std::move(b)) {
        auto weight_shape = weight_->shape();
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
    std::unique_ptr<Tensor<T>> weight_;
    std::unique_ptr<Tensor<T>> bias_;
};

template <typename T>
ModelIO<T> FeedForward<T>::forward(const ModelIO<T>& input) {
    if (!std::holds_alternative<Tensor<T>>(input)) {
        throw std::invalid_argument("[FeedForward] Input must be a single Tensor (not a list or dict)");
    }

    const Tensor<T>& input_t = std::get<Tensor<T>>(input);
    // TODO: to add the feedforward logic

}
