#pragma once

void launch_softmax_kernel(const float* inp, float* out, int N, int T);

void launch_softmax_in_place_kernel(float* inp, int N, int T);
