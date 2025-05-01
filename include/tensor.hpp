#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

enum class DeviceType {
    HOST,
    DEVICE
};





template <typename T>
class Tensor {
public:
    Tensor(const std::vector<size_t>& shape, DeviceType device = DeviceType::HOST);
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    ~Tensor();

    const std::vector<size_t>& shape() const { return shape_; }
    DeviceType device() const { return device_; }
    T* data() { return static_cast<T*>(data_); }
    void copy_from(const Tensor& other);
    const T* data() const { return static_cast<const T*>(data_); }

private:
    std::vector<size_t> shape_;
    size_t size_;
    DeviceType device_;
    void* data_;
};


template<typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, DeviceType device): shape_(shape), device_(device) {
    size_ = 1;
    for (auto d : shape_) {
        size_ *= d;
    }
    if (device_ == DeviceType::HOST) {
        data_ = (void*)new T[size_];
    } else {
        cudaError_t err = cudaMalloc(&data_, sizeof(T) * size_);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed!");
        }
    }
}

template<typename T>
Tensor<T>::~Tensor() {
    if (device_ == DeviceType::HOST) {
        delete[] (static_cast<T*>(data_));
    } else {
        cudaFree(data_);
    }
}

template<typename T>
void Tensor<T>::copy_from(const Tensor& other) {
    if (other.size_ != size_ || other.device_ == device_)
        throw std::runtime_error("Copy from: shape or device mismatch");

    if (device_ == DeviceType::HOST && other.device_ == DeviceType::DEVICE) {
        cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    }
    else if (device_ == DeviceType::DEVICE && other.device_ == DeviceType::HOST) {
        cudaMemcpy(data_, other.data_, size_ * sizeof(T), cudaMemcpyHostToDevice);
    }
    else {
        throw std::runtime_error("Unsupported copy direction");
    }
}
