#pragma once
#include "tensor.h"
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

using TensorList = std::vector<Tensor>;

using TensorDict = std::map<std::string, Tensor>;

using ModelIO = std::variant<Tensor, TensorDict, TensorList>;

class Layer {
public:
    virtual ~Layer() {}
    virtual ModelIO forward(const ModelIO& input) = 0;
};

class Model {
public:
    void add_layer(Layer* layer);
    ModelIO forward(const ModelIO& input);
private:
    std::vector<std::unique_ptr<Layer>> layers;
};
