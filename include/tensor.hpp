#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>
#include "utils.h"

enum class DeviceType {
    HOST,
    DEVICE
};

enum class TensorDataType {
    SYNC_ALLOCATE = 0,
    ASYNC_ALLOCATE = 1
};

#ifndef DEFAULT_ALLOC_METHOD
#define DEFAULT_ALLOC_METHOD 0
#endif

constexpr TensorDataType DEFAULT_TENSOR_DATA_TYPE = static_cast<TensorDataType>(DEFAULT_ALLOC_METHOD);

template<typename T>
class TensorData {
public:
    TensorData(size_t size, DeviceType device);
    TensorData(const TensorData&) = delete;
    TensorData& operator=(const TensorData&) = delete;
    virtual void copy_from(const TensorData&) = 0;
    DeviceType device() const { return device_; }
    virtual T* data() = 0;
    virtual const T* data() const = 0;
    virtual ~TensorData();
protected:
    T* data_;
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

template<typename T>
class AsyncTensorData : public TensorData<T> {
public:
    AsyncTensorData(size_t size, DeviceType device);
    virtual void copy_from(const TensorData<T>&) override;
    virtual T* data() override;
    virtual const T* data() const override;
    ~AsyncTensorData() override;
private:
    mutable bool is_ready;
    cudaEvent_t latest_event;
    mutable cudaEvent_t last_event_ = nullptr;
    void wait_for_data_readiness() const;
    void destroy_last_event() const;
    void record_event() const;
};

template<typename T>
class SyncTensorData : public TensorData<T> {
public:
    SyncTensorData(size_t size, DeviceType device);
    virtual void copy_from(const TensorData<T>&) override;
    virtual T* data() override;
    virtual const T* data() const override;
    ~SyncTensorData() override;
};

template<typename T>
class Tensor {
public:
    Tensor(const std::vector<size_t>& shape, DeviceType device = DeviceType::HOST, TensorDataType tensor_data_type = DEFAULT_TENSOR_DATA_TYPE);
    // Tensor(const std::vector<size_t>& shape, DeviceType device = DeviceType::HOST, TensorDataType tensor_data_type = TensorDataType::ASYNC_ALLOCATE);
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    ~Tensor() = default;

    const std::vector<size_t>& shape() const;
    DeviceType device() const;
    // WARNING: avoid keep calling the data() inside loop. Instead, save the T* from data() once
    // We are seeing obvoius slow down when keep calling data in large for-loop
    // Especially when the data() involves virtual table search (different implementations of TensorData)
    T* data();
    const T* data() const;
    void copy_from(const Tensor& other);
    size_t get_total_size() const;

private:
    std::vector<size_t> shape_;
    size_t size_;
    DeviceType device_;
    std::shared_ptr<TensorData<T>> data_; // points to allocated memory (host or device)
};

// This stream is leaked. Should be fine
inline cudaStream_t& tensor_creation_free_cuda_stream() {
    static cudaStream_t s = []() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return stream;
    }();
    return s;
}


template <typename T>
Tensor<T>::Tensor(const std::vector<size_t>& shape, DeviceType device, TensorDataType tensor_data_type): shape_(shape), device_(device) {
    size_ = 1;
    for (auto d : shape_) {
        size_ *= d;
    }
    if (tensor_data_type == TensorDataType::SYNC_ALLOCATE) {
        data_ = std::make_shared<SyncTensorData<T>>(size_, device);
    } else {
        data_ = std::make_shared<AsyncTensorData<T>>(size_, device);
    }
}

template <typename T>
void Tensor<T>::copy_from(const Tensor<T>& other) {
    data_->copy_from(*other.data_.get());
}

template <typename T>
const std::vector<size_t>& Tensor<T>::shape() const {
    return shape_;
}

template <typename T>
DeviceType Tensor<T>::device() const {
    return device_;
}

template <typename T>
T* Tensor<T>::data() {
    return data_->data();
}

template <typename T>
const T* Tensor<T>::data() const {
    return data_->data();
}

template <typename T>
size_t Tensor<T>::get_total_size() const {
    return size_;
}

template <typename T>
TensorData<T>::TensorData(size_t size, DeviceType device): size_(size), device_(device) {}

template <typename T>
TensorData<T>::~TensorData() {}

template <typename T>
AsyncTensorData<T>::AsyncTensorData(size_t size, DeviceType device): TensorData<T>(size, device) {
    if (device == DeviceType::HOST) {
        is_ready = true;
        cudaHostAlloc(&this->data_, this->size_ * sizeof(T), cudaHostAllocDefault);
    } else {
        cudaMallocAsync(&this->data_, sizeof(T) * this->size_, tensor_creation_free_cuda_stream());
        CUDA_CHECK_LAST();
        is_ready = false;
        record_event();
    }
}

template <typename T>
AsyncTensorData<T>::~AsyncTensorData() {
    if (this->device_ == DeviceType::HOST) {
        wait_for_data_readiness();
        cudaFreeHost(this->data_);
    } else {
        cudaFreeAsync(this->data_, tensor_creation_free_cuda_stream());
    }
    CUDA_CHECK_LAST();
    destroy_last_event();
}

template <typename T>
void AsyncTensorData<T>::copy_from(const TensorData<T>& other) {
    const AsyncTensorData<T>* async_other_ptr = dynamic_cast<const AsyncTensorData<T>*>(&other);
    if (!async_other_ptr) {
        throw std::runtime_error("Copy from: source is not an AsyncTensorData");
    }
    const AsyncTensorData<T>& async_other = *async_other_ptr;
    if (async_other.size_ != this->size_) {
        throw std::runtime_error("Copy from: shape or device mismatch");
    }

    // copy from DEVICE to HOST
    if (this->device_ == DeviceType::HOST && other.device() == DeviceType::DEVICE) {
        cudaMemcpyAsync(data(), async_other.data(), this->size_ * sizeof(T), cudaMemcpyDeviceToHost, tensor_creation_free_cuda_stream());
    } else if (this->device_ == DeviceType::DEVICE && other.device() == DeviceType::HOST) {
        cudaMemcpyAsync(data(), async_other.data(), this->size_ * sizeof(T), cudaMemcpyHostToDevice, tensor_creation_free_cuda_stream());
    } else if (this->device_ == DeviceType::DEVICE && other.device() == DeviceType::DEVICE) {
        cudaMemcpyAsync(data(), async_other.data(), this->size_ * sizeof(T), cudaMemcpyDeviceToDevice, tensor_creation_free_cuda_stream());
    } else {
        cudaMemcpyAsync(data(), async_other.data(), this->size_ * sizeof(T), cudaMemcpyHostToHost, tensor_creation_free_cuda_stream());
    }
    is_ready = false;
    CUDA_CHECK_LAST();
    record_event();
}

template <typename T>
T* AsyncTensorData<T>::data() {
    wait_for_data_readiness();
    return this->data_;
}

template <typename T>
const T* AsyncTensorData<T>::data() const {
    wait_for_data_readiness();
    return this->data_;
}

template <typename T>
void AsyncTensorData<T>::wait_for_data_readiness() const {
    if (is_ready) {
        return;
    }
    is_ready = true;
    if (last_event_) {
        cudaEventSynchronize(last_event_);
    }
}

template <typename T>
void AsyncTensorData<T>::destroy_last_event() const {
    if (last_event_) {
        cudaEventDestroy(last_event_);
        last_event_ = nullptr;
    }
}

template <typename T>
void AsyncTensorData<T>::record_event() const {
    destroy_last_event();
    cudaEventCreateWithFlags(&last_event_, cudaEventDisableTiming);
    cudaEventRecord(last_event_, tensor_creation_free_cuda_stream());
}


template <typename T>
SyncTensorData<T>::SyncTensorData(size_t size, DeviceType device): TensorData<T>(size, device) {
    if (device == DeviceType::HOST) {
        cudaHostAlloc(&this->data_, this->size_ * sizeof(T), cudaHostAllocDefault);
    } else {
        cudaMalloc(&this->data_, sizeof(T) * this->size_);
        CUDA_CHECK_LAST();
    }
}

template <typename T>
void SyncTensorData<T>::copy_from(const TensorData<T>& other) {
    const SyncTensorData<T>* sync_other_ptr = dynamic_cast<const SyncTensorData<T>*>(&other);
    if (!sync_other_ptr) {
        throw std::runtime_error("Copy from: source is not an syncTensorData");
    }
    const SyncTensorData<T>& sync_other = *sync_other_ptr;

    if (sync_other.size_ != this->size_)
        throw std::runtime_error("Copy from: shape or device mismatch");

    if (this->device_ == DeviceType::HOST && other.device() == DeviceType::DEVICE) {
        cudaMemcpy(data(), other.data(), this->size_ * sizeof(T), cudaMemcpyDeviceToHost);
    } else if (this->device_ == DeviceType::DEVICE && other.device() == DeviceType::HOST) {
        cudaMemcpy(data(), other.data(), this->size_ * sizeof(T), cudaMemcpyHostToDevice);
    } else if (this->device_ == DeviceType::DEVICE && other.device() == DeviceType::DEVICE) {
        cudaMemcpy(data(), other.data(), this->size_ * sizeof(T), cudaMemcpyDeviceToDevice);
    } else {
        cudaMemcpy(data(), other.data(), this->size_ * sizeof(T), cudaMemcpyHostToHost);
    }
    CUDA_CHECK_LAST();
}

template <typename T>
T* SyncTensorData<T>::data() {
    return this->data_;
}

template <typename T>
const T* SyncTensorData<T>::data() const {
    return this->data_;
}

template <typename T>
SyncTensorData<T>::~SyncTensorData() {
    if (this->device_ == DeviceType::HOST) {
        cudaFreeHost(this->data_);
    } else {
        cudaFree(this->data_);
    }
}

typedef Tensor<float> TensorFloat;
typedef Tensor<int> TensorInt;
typedef Tensor<float*> TensorFloatPoint;
