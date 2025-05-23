
#pragma once

#include "tensor.hpp"
#include <utility>
#include <vector>

class TensorWrapForInferenceOptimizedSelfAttention {
public:
    TensorFloat inp;
    TensorInt lengths;
    TensorFloat wk;
    TensorFloat wq;
    TensorFloat wv;
    TensorInt new_batch_idx;
    TensorFloat kt_cache;
    TensorFloat v_cache;
    TensorFloat q_output;
    TensorFloat qkt_output;
    TensorFloat attention_result;
};

std::pair<TensorFloat, TensorFloat> get_random_device_host_tensor(const std::vector<size_t>& shape, float ratio = 1);

std::pair<TensorInt, TensorInt> get_random_device_host_tensor_int(const std::vector<size_t>& shape, int max_val);

TensorFloat host_matrix_multiply(const TensorFloat& inp1, const TensorFloat& inp2);

TensorFloat softmax(const TensorFloat& inp);

TensorFloat transpose_host(const TensorFloat& inp_host);

void assert_near(const TensorFloat& tensor_device, const TensorFloat& tensor_host, float threashold = 1e-3);
void assert_near_on_host(const TensorFloat &tensor_device, const TensorFloat &tensor_host, float threshold=1e-3);

void print_host(const float* data, int size);

TensorFloat encoder_host(const TensorFloat& wte, const TensorFloat& wpe, const TensorInt& inp, size_t batch_size, size_t n_sequence, size_t embedding_dim);

int get_random_number(int min, int max);

std::vector<int> get_unique_num_array(int min, int max, int size);

std::pair<TensorWrapForInferenceOptimizedSelfAttention, TensorWrapForInferenceOptimizedSelfAttention> generate_device_and_host_tensors(
    size_t n_batch=1024, size_t n_sequence=1024, size_t input_dim=32, size_t output_dim=32);


std::pair<TensorWrapForInferenceOptimizedSelfAttention, TensorWrapForInferenceOptimizedSelfAttention> generate_random_shape_device_and_host_tensors();

void inference_optimized_encoder_host(const float* emb_table, const float* wpe, const int* inp, float* output, const int* lengths,
    const int* new_item_indices,
    int batch_size, int n_sequence, int embedding_dim, int n_new_items);
