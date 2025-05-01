#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
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
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    ~Tensor() = default;

    const std::vector<size_t>& shape() const { return shape_; }
    DeviceType device() const { return device_; }
    T* data() { return reinterpret_cast<T*>(data_.get()); }
    void copy_from(const Tensor& other);
    const T* data() const { return reinterpret_cast<const T*>(data_.get()); }

private:
    std::vector<size_t> shape_;
    size_t size_;
    DeviceType device_;

    struct Deleter {
        DeviceType device;
        void operator()(void* p) const {
            if (!p) return;
            if (device == DeviceType::HOST) delete[] reinterpret_cast<T*>(p);
            else cudaFree(p);
        }
    };

    std::shared_ptr<void> data_; // points to allocated memory (host or device)
};


template<typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, DeviceType device): shape_(shape), device_(device) {
    size_ = 1;
    for (auto d : shape_) {
        size_ *= d;
    }
    void* ptr;
    if (device_ == DeviceType::HOST) {
        ptr = (void*)new T[size_];
    } else {
        cudaError_t err = cudaMalloc(&ptr, sizeof(T) * size_);
        if (err != cudaSuccess) {
            std::cerr << "CudaMalloc failure: " << cudaGetErrorString(err)
                << ", trying to allocate " << sizeof(T) * size_ << std::endl;
            throw std::runtime_error("cudaMalloc failed!");
        }
    }
    data_ = std::shared_ptr<void>(ptr, Deleter{device});
}

template<typename T>
void Tensor<T>::copy_from(const Tensor& other) {
    if (other.size_ != size_ || other.device_ == device_)
        throw std::runtime_error("Copy from: shape or device mismatch, and we only support copy from different devices");

    cudaError err;
    if (device_ == DeviceType::HOST && other.device_ == DeviceType::DEVICE) {
        err = cudaMemcpy(data(), other.data(), size_ * sizeof(T), cudaMemcpyDeviceToHost);
    } else if (device_ == DeviceType::DEVICE && other.device_ == DeviceType::HOST) {
        err = cudaMemcpy(data(), other.data(), size_ * sizeof(T), cudaMemcpyHostToDevice);
    } else {
        throw std::runtime_error("Unsupported copy direction");
    }

    if (err != cudaSuccess) {
        std::cerr << "Memory copy error: " << cudaGetErrorString(err) << std::endl;
    }
}
