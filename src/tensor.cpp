#include "tensor.h"
#include "utils.h"
#include <memory>
#include <stdexcept>
#include <vector>


// This stream is leaked. Should be fine
cudaStream_t& tensor_creation_free_cuda_stream() {
    static cudaStream_t s = []() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return stream;
    }();
    return s;
}


Tensor::Tensor(const std::vector<size_t>& shape, DeviceType device, TensorDataType tensor_data_type): shape_(shape), device_(device) {
    size_ = 1;
    for (auto d : shape_) {
        size_ *= d;
    }
    if (tensor_data_type == TensorDataType::SYNC) {
        data_ = std::make_shared<SyncTensorData>(size_, device);
    } else {
        data_ = std::make_shared<AsyncTensorData>(size_, device);
    }
}

void Tensor::copy_from(const Tensor& other) {
    data_->copy_from(*other.data_.get());
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

DeviceType Tensor::device() const {
    return device_;
}

float* Tensor::data() {
    return data_->data();
}

const float* Tensor::data() const {
    return data_->data();
}

size_t Tensor::get_total_size() const {
    return size_;
}

TensorData::TensorData(size_t size, DeviceType device): size_(size), device_(device) {}

TensorData::~TensorData() {}

AsyncTensorData::AsyncTensorData(size_t size, DeviceType device): TensorData(size, device) {
    if (device == DeviceType::HOST) {
        is_ready = true;
        cudaHostAlloc(&data_, size_ * sizeof(float), cudaHostAllocDefault);
    } else {
        cudaMallocAsync(&data_, sizeof(float) * size, tensor_creation_free_cuda_stream());
        CUDA_CHECK_LAST();
        is_ready = false;
        record_event();
    }
}

AsyncTensorData::~AsyncTensorData() {
    if (device_ == DeviceType::HOST) {
        wait_for_data_readiness();
        cudaFreeHost(data_);
    } else {
        cudaFreeAsync(data_, tensor_creation_free_cuda_stream());
    }
    CUDA_CHECK_LAST();
    destroy_last_event();
}

void AsyncTensorData::copy_from(const TensorData& other) {
    const AsyncTensorData& async_other = *dynamic_cast<const AsyncTensorData*>(&other);
    if (async_other.size_ != size_ || async_other.device_ == device_) {
        throw std::runtime_error("Copy from: shape or device mismatch, and we only support copy from different devices");
    }

    // copy from DEVICE to HOST
    if (device_ == DeviceType::HOST) {
        cudaMemcpyAsync(data(), async_other.data(), size_ * sizeof(float), cudaMemcpyDeviceToHost, tensor_creation_free_cuda_stream());
    } else {    // copy from HOST to DEVICE
        cudaMemcpyAsync(data(), async_other.data(), size_ * sizeof(float), cudaMemcpyHostToDevice, tensor_creation_free_cuda_stream());
    }
    is_ready = false;
    CUDA_CHECK_LAST();
    record_event();
}

float* AsyncTensorData::data() {
    wait_for_data_readiness();
    return data_;
}

const float* AsyncTensorData::data() const {
    wait_for_data_readiness();
    return data_;
}

void AsyncTensorData::wait_for_data_readiness() const {
    if (is_ready) {
        return;
    }
    is_ready = true;
    if (last_event_) {
        cudaEventSynchronize(last_event_);
    }
}

void AsyncTensorData::destroy_last_event() const {
    if (last_event_) {
        cudaEventDestroy(last_event_);
        last_event_ = nullptr;
    }
}

void AsyncTensorData::record_event() const {
    destroy_last_event();
    cudaEventCreateWithFlags(&last_event_, cudaEventDisableTiming);
    cudaEventRecord(last_event_, tensor_creation_free_cuda_stream());
}


SyncTensorData::SyncTensorData(size_t size, DeviceType device): TensorData(size, device) {
    if (device == DeviceType::HOST) {
        data_ = new float[size_];
    } else {
        cudaMalloc(&data_, sizeof(float) * size_);
        CUDA_CHECK_LAST();
    }
}

void SyncTensorData::copy_from(const TensorData& other) {
    const SyncTensorData& sync_other = *dynamic_cast<const SyncTensorData*>(&other);

    if (sync_other.size_ != size_ || sync_other.device_ == device_)
        throw std::runtime_error("Copy from: shape or device mismatch, and we only support copy from different devices");

    cudaError err;
    if (device_ == DeviceType::HOST) {
        cudaMemcpy(data(), other.data(), size_ * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(data(), other.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);
    }
    CUDA_CHECK_LAST();
}

float* SyncTensorData::data() {
    return data_;
}

const float* SyncTensorData::data() const {
    return data_;
}

SyncTensorData::~SyncTensorData() {
    if (device_ == DeviceType::HOST) {
        delete[] data_;
    } else {
        cudaFree(data_);
    }
}
