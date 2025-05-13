#pragma once

void launch_encoder_kernel(
    const float* wte, const float* wpe, const int* inp, float* output, int batch_size, int n_sequence, int embedding_dim);
