#pragma once

#include "model.h"
#include "tensor.h"
#include <optional>


class FeedForward : public Layer {
public:
    FeedForward(const Tensor& w, std::optional<const Tensor> b = std::nullopt);
    ModelIO forward(const ModelIO& input) override;
private:
    Tensor weight_;
    std::optional<Tensor> bias_;
};
