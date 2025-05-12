#pragma once

#include "model.h"
#include "tensor.hpp"
#include <optional>


class FeedForward : public Layer {
public:
    FeedForward(const TensorFloat& w, std::optional<const TensorFloat> b = std::nullopt);
    ModelIO forward(const ModelIO& input) override;
private:
    TensorFloat weight_;
    std::optional<TensorFloat> bias_;
};
