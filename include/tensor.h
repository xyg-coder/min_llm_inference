#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

enum class DeviceType {
    HOST,
    DEVICE
};


class Tensor {
public:
    Tensor(const std::vector<size_t>& shape, DeviceType device = DeviceType::HOST);
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    ~Tensor() = default;

    const std::vector<size_t>& shape() const;
    DeviceType device() const;
    float* data();
    void copy_from(const Tensor& other);
    const float* data() const;
    size_t get_total_size() const;

private:
    std::vector<size_t> shape_;
    size_t size_;
    DeviceType device_;
    std::shared_ptr<void> data_; // points to allocated memory (host or device)
};
