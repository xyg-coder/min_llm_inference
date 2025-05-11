#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

enum class DeviceType {
    HOST,
    DEVICE
};


class TensorData {
public:
    TensorData(size_t size, DeviceType device);
    TensorData(const TensorData&) = delete;
    TensorData& operator=(const TensorData&) = delete;
    void copy_from(const TensorData&);
    float* data();
    const float* data() const;
    ~TensorData();
private:
    float* data_;
    size_t size_;
    mutable bool is_ready;
    DeviceType device_;
    cudaEvent_t latest_event;
    mutable cudaEvent_t last_event_ = nullptr;
    void wait_for_data_readiness() const;
    void destroy_last_event() const;
    void record_event() const;
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
    const float* data() const;
    void copy_from(const Tensor& other);
    size_t get_total_size() const;

private:
    std::vector<size_t> shape_;
    size_t size_;
    DeviceType device_;
    std::shared_ptr<TensorData> data_; // points to allocated memory (host or device)
};
