#pragma once
#include "tensor.hpp"
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

using TensorList = std::vector<TensorFloat>;

using TensorDict = std::map<std::string, TensorFloat>;

using ModelIO = std::variant<TensorFloat, TensorDict, TensorList>;

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
