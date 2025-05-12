#pragma once

#include "tensor.hpp"

void launch_kqt_kernel(const float* kqv, float* output, int n_batch, int n_sequence, int dims);
TensorFloat self_attention(const TensorFloat& inp, const TensorFloat& wk_wq_wv);
