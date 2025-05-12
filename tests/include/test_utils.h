
#pragma once

#include "tensor.hpp"
#include <utility>
#include <vector>

std::pair<TensorFloat, TensorFloat> get_random_device_host_tensor(const std::vector<size_t>& shape, float ratio = 1);

TensorFloat host_matrix_multiply(const TensorFloat& inp1, const TensorFloat& inp2);

TensorFloat softmax(const TensorFloat& inp);

TensorFloat transpose(const TensorFloat& inp);

void assert_near(const TensorFloat& tensor_device, const TensorFloat& tensor_host, float threashold = 1e-3);

void print_host(const float* data, int size);
