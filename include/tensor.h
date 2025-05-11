#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

enum class DeviceType {
    HOST,
    DEVICE
};

enum class TensorDataType {
    SYNC,
    ASYNC
};

constexpr TensorDataType DEFAULT_TENSOR_DATA_TYPE = TensorDataType::SYNC;

class TensorData {
public:
    TensorData(size_t size, DeviceType device);
    TensorData(const TensorData&) = delete;
    TensorData& operator=(const TensorData&) = delete;
    virtual void copy_from(const TensorData&) = 0;
    virtual float* data() = 0;
    virtual const float* data() const = 0;
    virtual ~TensorData();
protected:
    float* data_;
    size_t size_;
    DeviceType device_;
};

///////////////////////////////////////////////////////////////////////////
// For now, the 2 implementations don't show much difference:
// sync
// Test project /root/min_llm_inference/build
//     Start 1: feed_forward_layer_test
// 1/4 Test #1: feed_forward_layer_test ..........   Passed    8.78 sec
//     Start 2: gemm_test
// 2/4 Test #2: gemm_test ........................   Passed    2.06 sec
//     Start 3: self_attention_test
// 3/4 Test #3: self_attention_test ..............   Passed    1.08 sec
//     Start 4: softmax_test
// 4/4 Test #4: softmax_test .....................   Passed   11.54 sec
// 
// 
// async
// 1/4 Test #1: feed_forward_layer_test ..........   Passed    8.71 sec
//     Start 2: gemm_test
// 2/4 Test #2: gemm_test ........................   Passed    2.08 sec
//     Start 3: self_attention_test
// 3/4 Test #3: self_attention_test ..............   Passed    1.08 sec
//     Start 4: softmax_test
// 4/4 Test #4: softmax_test .....................   Passed   12.82 sec
// Assuming future with more complicated calculations, we can see more value from async version.
//////////////////////////////////////////////////////////////////////////

class AsyncTensorData : public TensorData {
public:
    AsyncTensorData(size_t size, DeviceType device);
    virtual void copy_from(const TensorData&) override;
    virtual float* data() override;
    virtual const float* data() const override;
    ~AsyncTensorData() override;
private:
    mutable bool is_ready;
    cudaEvent_t latest_event;
    mutable cudaEvent_t last_event_ = nullptr;
    void wait_for_data_readiness() const;
    void destroy_last_event() const;
    void record_event() const;
};

class SyncTensorData : public TensorData {
public:
    SyncTensorData(size_t size, DeviceType device);
    virtual void copy_from(const TensorData&) override;
    virtual float* data() override;
    virtual const float* data() const override;
    ~SyncTensorData() override;
};

class Tensor {
public:
    Tensor(const std::vector<size_t>& shape, DeviceType device = DeviceType::HOST, TensorDataType tensor_data_type = DEFAULT_TENSOR_DATA_TYPE);
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    ~Tensor() = default;

    const std::vector<size_t>& shape() const;
    DeviceType device() const;
    // WARNING: avoid keep calling the data() inside loop. Instead, save the float* from data() once
    // We are seeing obvoius slow down when keep calling data in large for-loop
    // Especially when the data() involves virtual table search (different implementations of TensorData)
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
