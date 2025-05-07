#include "tensor.h"
#include <iostream>
#include <vector>

struct Deleter {
    DeviceType device;
    void operator()(void* p) const {
        if (!p) return;
        if (device == DeviceType::HOST) delete[] reinterpret_cast<float*>(p);
        else cudaFree(p);
    }
};

Tensor::Tensor(const std::vector<size_t>& shape, DeviceType device): shape_(shape), device_(device) {
    size_ = 1;
    for (auto d : shape_) {
        size_ *= d;
    }
    void* ptr;
    if (device_ == DeviceType::HOST) {
        ptr = (void*)new float[size_];
    } else {
        cudaError_t err = cudaMalloc(&ptr, sizeof(float) * size_);
        if (err != cudaSuccess) {
            std::cerr << "CudaMalloc failure: " << cudaGetErrorString(err)
                << ", trying to allocate " << sizeof(float) * size_ << std::endl;
            throw std::runtime_error("cudaMalloc failed!");
        }
    }
    data_ = std::shared_ptr<void>(ptr, Deleter{device});
}

void Tensor::copy_from(const Tensor& other) {
    if (other.size_ != size_ || other.device_ == device_)
        throw std::runtime_error("Copy from: shape or device mismatch, and we only support copy from different devices");

    cudaError err;
    if (device_ == DeviceType::HOST && other.device_ == DeviceType::DEVICE) {
        err = cudaMemcpy(data(), other.data(), size_ * sizeof(float), cudaMemcpyDeviceToHost);
    } else if (device_ == DeviceType::DEVICE && other.device_ == DeviceType::HOST) {
        err = cudaMemcpy(data(), other.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        throw std::runtime_error("Unsupported copy direction");
    }

    if (err != cudaSuccess) {
        std::cerr << "Memory copy error: " << cudaGetErrorString(err) << std::endl;
    }
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

DeviceType Tensor::device() const {
    return device_;
}

float* Tensor::data() {
    return reinterpret_cast<float*>(data_.get());
}

const float* Tensor::data() const {
    return reinterpret_cast<const float*>(data_.get());
}
