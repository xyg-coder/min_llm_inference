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


Tensor::Tensor(const std::vector<size_t>& shape, DeviceType device): shape_(shape), device_(device) {
    size_ = 1;
    for (auto d : shape_) {
        size_ *= d;
    }
    data_ = std::make_shared<TensorData>(size_, device);
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

TensorData::TensorData(size_t size, DeviceType device): size_(size), device_(device) {
    if (device == DeviceType::HOST) {
        is_ready = true;
        data_ = new float[size];
    } else {
        cudaMallocAsync(&data_, sizeof(float) * size, tensor_creation_free_cuda_stream());
        CUDA_CHECK_LAST();
        is_ready = false;
        record_event();
    }
}

TensorData::~TensorData() {
    if (device_ == DeviceType::HOST) {
        wait_for_data_readiness();
        delete[] data_;
    } else {
        cudaFreeAsync(data_, tensor_creation_free_cuda_stream());
        CUDA_CHECK_LAST();
    }
    destroy_last_event();
}

void TensorData::copy_from(const TensorData& other) {
    if (other.size_ != size_ || other.device_ == device_) {
        throw std::runtime_error("Copy from: shape or device mismatch, and we only support copy from different devices");
    }

    // copy from DEVICE to HOST
    if (device_ == DeviceType::HOST) {
        cudaMemcpyAsync(data_, other.data_, size_ * sizeof(float), cudaMemcpyDeviceToHost, tensor_creation_free_cuda_stream());
    } else {    // copy from HOST to DEVICE
        cudaMemcpyAsync(data_, other.data_, size_ * sizeof(float), cudaMemcpyDeviceToHost, tensor_creation_free_cuda_stream());
    }
    is_ready = false;
    CUDA_CHECK_LAST();
    record_event();
}

float* TensorData::data() {
    wait_for_data_readiness();
    return data_;
}

const float* TensorData::data() const {
    wait_for_data_readiness();
    return data_;
}

void TensorData::wait_for_data_readiness() const {
    if (is_ready) {
        return;
    }
    is_ready = true;
    if (last_event_) {
        cudaEventSynchronize(last_event_);
    }
}

void TensorData::destroy_last_event() const {
    if (last_event_) {
        cudaEventDestroy(last_event_);
        last_event_ = nullptr;
    }
}

void TensorData::record_event() const {
    destroy_last_event();
    cudaEventCreateWithFlags(&last_event_, cudaEventDisableTiming);
    cudaEventRecord(last_event_, tensor_creation_free_cuda_stream());
}
