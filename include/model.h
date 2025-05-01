#pragma once
#include "tensor.hpp"
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

template <typename T>
using TensorList = std::vector<Tensor<T>>;

template <typename T>
using TensorDict = std::map<std::string, Tensor<T>>;

template <typename T>
using ModelIO = std::variant<Tensor<T>, TensorDict<T>, TensorList<T>>;

template <typename T>
class Layer {
public:
    virtual ~Layer() {}
    virtual ModelIO<T> forward(const ModelIO<T>& input) = 0;
};

template <typename T>
class Model {
public:
    void add_layer(Layer<T>* layer);
    ModelIO<T> forward(const ModelIO<T>& input);
private:
    std::vector<std::unique_ptr<Layer<T>>> layers;
};
