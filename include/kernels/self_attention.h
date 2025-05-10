#pragma once

#include "tensor.h"

void launch_kqt_kernel(const float* kqv, float* output, int n_batch, int n_sequence, int dims);
Tensor self_attention(const Tensor& inp, const Tensor& wk_wq_wv);
