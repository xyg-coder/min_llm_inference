
#pragma once

#include "tensor.h"
#include <utility>
#include <vector>

std::pair<Tensor, Tensor> get_random_device_host_tensor(const std::vector<size_t>& shape, float ratio = 1);

Tensor host_matrix_multiply(const Tensor& inp1, const Tensor& inp2);

Tensor softmax(const Tensor& inp);

Tensor transpose(const Tensor& inp);

void assert_near(const Tensor& tensor_device, const Tensor& tensor_host, float threashold = 1e-3);

void print_host(const float* data, int size);
