#include "test_utils.h"
#include "include/test_utils.h"
#include "kernels/rand_assign.h"
#include "kernels/utils.cuh"
#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <gtest/gtest.h>
#include <utility>
#include <vector>
#include "self_attention_inference_optimized_host.h"

/**
 * inp: [n_batch, n_sequence, input_dim]
 * new_batch_idx: [n_new_batch]
 * lengths: [n_batch]
 * wk: [input_dim, output_dim]
 * wv: [input_dim, output_dim]
 * kt_cache: [n_batch_size, output_dim, n_sequence]
 * v_cache: [n_batch_size, n_sequence, output_dim]
 */
void fill_new_kt_v_cache(
    const TensorFloat& inp, const TensorInt& new_batch_idx, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wv, TensorFloat& kt_cache,
    TensorFloat& v_cache) {

    int n_batch = inp.shape()[0];
    int n_sequence = inp.shape()[1];
    int input_dim = inp.shape()[2];
    int output_dim = wk.shape()[1];
    int n_new_batch = new_batch_idx.shape()[0];
    
    const float* inp_data = inp.data();
    const int* new_batch_idx_data = new_batch_idx.data();
    const int* lengths_data = lengths.data();
    const float* wk_data = wk.data();
    const float* wv_data = wv.data();
    float* kt_cache_data = kt_cache.data();
    float* v_cache_data = v_cache.data();

    // inp * wk, transpose -> [n_batch_size, output_dim, n_sequence]
    // inp * wv -> [n_batch_size, n_sequence, output_dim]
    for (int i = 0; i < n_new_batch; ++i) {
        int batch_index = new_batch_idx_data[i];
        int cur_length = lengths_data[batch_index];
        const float* inp_base = inp_data + batch_index * n_sequence * input_dim;
        for (int j = 0; j < cur_length; ++j) {
            for (int k = 0; k < output_dim; ++k) {
                float k_result = 0;
                float v_result = 0;
                for (int w = 0; w < input_dim; ++w) {
                    k_result += (inp_base[j * input_dim + w] * wk_data[w * output_dim + k]);
                    v_result += (inp_base[j * input_dim + w] * wv_data[w * output_dim + k]);
                }
                kt_cache_data[batch_index * output_dim * n_sequence + k * n_sequence + j] = k_result;
                v_cache_data[batch_index * output_dim * n_sequence + j * output_dim + k] = v_result;
            }
        }
    }
}

std::pair<TensorFloat, TensorFloat> get_random_device_host_tensor(const std::vector<size_t> &shape, float ratio) {
    TensorFloat device_tensor(shape, DeviceType::DEVICE);    
    TensorFloat host_tensor(shape, DeviceType::HOST);
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    launch_randn_kernel(device_tensor.data(), total_size, ratio);
    host_tensor.copy_from(device_tensor);
    return std::make_pair(device_tensor, host_tensor);
}

std::pair<TensorInt, TensorInt> get_random_device_host_tensor_int(const std::vector<size_t>& shape, int max_val) {
    TensorInt device_tensor(shape, DeviceType::DEVICE);    
    TensorInt host_tensor(shape, DeviceType::HOST);
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    launch_randn_kernel(device_tensor.data(), total_size, max_val);
    host_tensor.copy_from(device_tensor);
    return std::make_pair(device_tensor, host_tensor);
}

// This function is slower but can be used to find where the similarity begins
void assert_near_on_host(const TensorFloat &tensor_device, const TensorFloat &tensor_host, float threshold) {
    size_t total_size = tensor_device.get_total_size();
    TensorFloat copy_from_device(tensor_device.shape(), DeviceType::HOST);
    copy_from_device.copy_from(tensor_device);
    const float* copy_from_device_ptr = copy_from_device.data();
    const float* host_tensor_ptr = tensor_host.data();
    for (size_t i = 0; i < total_size; ++i) {
        ASSERT_NEAR(copy_from_device_ptr[i], host_tensor_ptr[i], threshold)
            << "Mismatch at (" << i <<  ")";
    }
}

void assert_near(const TensorFloat &tensor_device, const TensorFloat &tensor_host, float threshold) {
    size_t total_size = tensor_device.get_total_size();
    TensorFloat copy_from_host(tensor_device.shape(), DeviceType::DEVICE);
    copy_from_host.copy_from(tensor_host);
    assert_float_kernel_close(tensor_device.data(), copy_from_host.data(), total_size, threshold);
}

TensorFloat host_matrix_multiply(const TensorFloat& inp1, const TensorFloat& inp2) {
    const std::vector<size_t>& shape1 = inp1.shape();
    const std::vector<size_t>& shape2 = inp2.shape();
    assert(shape1.size() == shape2.size());
    assert(shape1.size() == 2 || shape1.size() == 3);
    if (shape1.size() == 3) {
        size_t n_batch = shape1[0];
        size_t rows = shape1[1];
        size_t N = shape1[2];
        assert(shape2[0] == n_batch);
        assert(shape2[1] == N);
        size_t cols = shape2[2];
        TensorFloat result_tensor({n_batch, rows, cols}, DeviceType::HOST);
        const float* inp1_ptr = inp1.data();
        const float* inp2_ptr = inp2.data();
        float* result_ptr = result_tensor.data();
        for (int i = 0; i < n_batch; ++i) {
            for (int j = 0; j < rows; ++j) {
                for (int k = 0; k < cols; ++k) {
                    float result = 0;
                    for (int n = 0; n < N; ++n) {
                        result += (inp1_ptr[i * rows * N + j * N + n] * inp2_ptr[i * N * cols + n * cols + k]);
                    }
                    result_ptr[i * rows * cols + j * cols + k] = result;
                }
            }
        }
        return result_tensor;
    } else {
        size_t rows = shape1[0];
        size_t N = shape1[1];
        assert(shape2[0] == N);
        size_t cols = shape2[1];
        TensorFloat result_tensor({rows, cols}, DeviceType::HOST);
        const float* inp1_ptr = inp1.data();
        const float* inp2_ptr = inp2.data();
        float* result_ptr = result_tensor.data();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float result = 0;
                for (int n = 0; n < N; ++n) {
                    result += (inp1_ptr[i * N + n] * inp2_ptr[n * cols + j]);
                }
                result_ptr[i * cols + j] = result;
            }
        }
        return result_tensor;
    }
}

TensorFloat softmax(const TensorFloat &inp) {
    TensorFloat result(inp.shape());
    int cols = inp.shape()[inp.shape().size() - 1];
    int total_size = inp.get_total_size();
    int rows = total_size / cols;
    const float* inp_ptr = inp.data();
    float* result_ptr = result.data();
    for (int y = 0; y < rows; ++y) {
        float maxval = -std::numeric_limits<float>::infinity();
        for (int x = 0; x < cols; ++x) {
            maxval = std::max(maxval, inp_ptr[y * cols + x]);
        }
        float sum = 0;
        for (int x = 0; x < cols; ++x) {
            sum += std::exp(inp_ptr[y * cols + x] - maxval);
        }
        for (int x = 0; x < cols; ++x) {
            result_ptr[y * cols + x] = std::exp(inp_ptr[y * cols + x] - maxval) / sum;
        }
    }
    return result;
}

void print_host(const float *data, int size) {
    for (int i = 0; i < size; ++i) {
        printf("printing host: %d = %f\n", i, data[i]);
    }
}

TensorFloat encoder_host(const TensorFloat& wte, const TensorFloat& wpe, const TensorInt& inp, size_t batch_size, size_t n_sequence, size_t embedding_dim) {
    TensorFloat output({batch_size, n_sequence, embedding_dim});
    const float* wte_ptr = wte.data();
    const float* wpe_ptr = wpe.data();
    const int* inp_ptr = inp.data();
    float* output_ptr = output.data();
    for (int i = 0; i < n_sequence; ++i) {
        const float* wpe_ptr_base = wpe_ptr + i * embedding_dim;
        for (int j = 0; j < batch_size; ++j) {
            int idx = inp_ptr[j * n_sequence + i];
            const float* wte_ptr_base = wte_ptr + idx * embedding_dim;
            for (int k = 0; k < embedding_dim; ++k) {
                output_ptr[j * n_sequence * embedding_dim + i * embedding_dim + k] = wpe_ptr_base[k] + wte_ptr_base[k];
            }
        }
    }
    return output;
}

// both inclusive
int get_random_number(int min, int max) {
    std::random_device rd;                          // Obtain a random number from hardware
    std::mt19937 gen(rd());                         // Seed the generator
    std::uniform_int_distribution<> distr(min, max);   // Define the range [0, 74]

    return distr(gen); 
}

std::vector<int> get_unique_num_array(int min, int max, int size) {
    // Prepare a list of available numbers
    std::vector<int> numbers;
    for (int i = min; i <= max; ++i) {
        numbers.push_back(i);
    }

    // Shuffle to randomize the order
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(numbers.begin(), numbers.end(), gen);

    // Take the first n numbers as the result
    std::vector<int> result(numbers.begin(), numbers.begin() + size);
    return result;
}

std::pair<TensorWrapForInferenceOptimizedSelfAttention, TensorWrapForInferenceOptimizedSelfAttention> generate_device_and_host_tensors(
    size_t n_batch, size_t n_sequence, size_t input_dim, size_t output_dim) {

    size_t n_new_batches = get_random_number(1, n_batch);
    std::vector<int> new_batch_indices = get_unique_num_array(0, n_batch - 1, n_new_batches);
    auto inp_device_host = get_random_device_host_tensor({n_batch, n_sequence, input_dim});
    auto lengths_device_host = get_random_device_host_tensor_int({n_batch}, n_sequence);
    auto wk_device_host = get_random_device_host_tensor({input_dim, output_dim});
    auto wq_device_host = get_random_device_host_tensor({input_dim, output_dim});
    auto wv_device_host = get_random_device_host_tensor({input_dim, output_dim});
    TensorInt new_batch_idx_host({new_batch_indices.size()}, DeviceType::HOST);
    std::copy(new_batch_indices.begin(), new_batch_indices.end(), new_batch_idx_host.data());
    TensorInt new_batch_idx_device({new_batch_indices.size()}, DeviceType::DEVICE);
    new_batch_idx_device.copy_from(new_batch_idx_host);

    auto kt_cache_device_host = get_random_device_host_tensor({n_batch, output_dim, n_sequence});
    auto v_cache_device_host = get_random_device_host_tensor({n_batch, n_sequence, output_dim});
    auto q_output_device_host = get_random_device_host_tensor({n_batch, output_dim});
    auto qkt_output_device_host = get_random_device_host_tensor({n_batch, n_sequence});
    auto attention_result_device_host = get_random_device_host_tensor({n_batch, output_dim});
    return std::make_pair(
        TensorWrapForInferenceOptimizedSelfAttention{
            inp_device_host.first,
            lengths_device_host.first,
            wk_device_host.first,
            wq_device_host.first,
            wv_device_host.first,
            new_batch_idx_device,
            kt_cache_device_host.first,
            v_cache_device_host.first,
            q_output_device_host.first,
            qkt_output_device_host.first,
            attention_result_device_host.first
        },
        TensorWrapForInferenceOptimizedSelfAttention{
            inp_device_host.second,
            lengths_device_host.second,
            wk_device_host.second,
            wq_device_host.second,
            wv_device_host.second,
            new_batch_idx_host,
            kt_cache_device_host.second,
            v_cache_device_host.second,
            q_output_device_host.second,
            qkt_output_device_host.second,
            attention_result_device_host.second
        });
}

std::pair<TensorWrapForInferenceOptimizedSelfAttention, TensorWrapForInferenceOptimizedSelfAttention> generate_random_shape_device_and_host_tensors() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist_dims(100, 257);
    std::uniform_int_distribution<size_t> dist_batch(1, 100);
    std::uniform_int_distribution<size_t> dist_sequence(100, 200);
    return generate_device_and_host_tensors(
        dist_batch(rng), dist_sequence(rng) * 4, dist_dims(rng), dist_dims(rng));
}

/**
 * inp: [n_batch, n_sequence, input_dim]
 * lengths: [n_batch]
 * wq, wk, wv: [input_dim, output_dim]
 * v_cache: [n_batch, n_sequence, output_dim]
 * kt_cache: [n_batch, output_dim, n_sequence]
 * q_output: [n_batch, output_dim]

 * 1. use the last embedding for each batch to multiply with wq, wk, wv -> [n_batch, 1, onput_dim]
 * 2. save to k_cache, v_cache and q_output
 */
void get_latest_kt_q_v(
    const TensorFloat& inp, const TensorInt& lengths,
    const TensorFloat& wk, const TensorFloat& wq,
    const TensorFloat& wv, TensorFloat& kt_cache,
    TensorFloat& v_cache, TensorFloat& q_output) {

    int n_batch = inp.shape()[0];
    int n_sequence = inp.shape()[1];
    int input_dim = inp.shape()[2];
    int output_dim = wk.shape()[1];

    const float* inp_data = inp.data();
    const int* lengths_data = lengths.data();
    const float* wk_data = wk.data();
    const float* wq_data = wq.data();
    const float* wv_data = wv.data();
    float* kt_cache_data = kt_cache.data();
    float* v_cache_data = v_cache.data();
    float* q_output_data = q_output.data();
    
    for (int i = 0; i < n_batch; ++i) {
        if (lengths_data[i] == 0) {
            continue;;
        }
        int i_sequence = lengths_data[i] - 1;
        const float* inp_data_sequence = inp_data + i * n_sequence * input_dim + i_sequence * input_dim;
        for (int j = 0; j < output_dim; ++j) {
            float kt_result = 0;
            float v_result = 0;
            float q_result = 0;
            for (int w = 0; w < input_dim; ++w) {
                kt_result += (inp_data_sequence[w] * wk_data[w * output_dim + j]);
                v_result += (inp_data_sequence[w] * wv_data[w * output_dim + j]);
                q_result += (inp_data_sequence[w] * wq_data[w * output_dim + j]);
            }
            q_output_data[i * output_dim + j] = q_result;
            v_cache_data[i * n_sequence * output_dim + i_sequence * output_dim + j] = v_result;
            kt_cache_data[i * n_sequence * output_dim + j * n_sequence + i_sequence] = kt_result;
        }
    }
}

/**
 * q: [n_batch, dim]
 * kt: [n_batch, dim, n_sequence]
 * qkt: [n_batch, n_sequence]
 */
void qkt_host(
    const TensorFloat& q_output, const TensorFloat& kt_cache, const TensorInt& lengths,
    TensorFloat& qkt_output) {
    
    int n_batch = q_output.shape()[0];
    int output_dim = q_output.shape()[1];
    int n_sequence = kt_cache.shape()[2];

    const float* q_output_data = q_output.data();
    const float* kt_cache_data = kt_cache.data();
    const int* lengths_data = lengths.data();
    float* qkt_output_data = qkt_output.data();

    for (int i = 0; i < n_batch; ++i) {
        int cur_length = lengths_data[i];
        const float* q_output_batch_data = q_output_data + i * output_dim;
        const float* kt_cache_batch_data = kt_cache_data + i * output_dim * n_sequence;
        float* qkt_output_batch_data = qkt_output_data + i * n_sequence;
        for (int j = 0; j < cur_length; ++j) {
            float result = 0;
            for (int k = 0; k < output_dim; ++k) {
                result += (q_output_batch_data[k] * kt_cache_batch_data[k * n_sequence + j]);
            }
            qkt_output_batch_data[j] = result / std::sqrt(output_dim);
        }
    }
}

/**
 * qkt: [n_batch, n_sequence]
 * result is written to qkt, with the same shape
 * Any element exceeding the lengths is 0
 */
void softmax_in_place_with_lengths_host(
    TensorFloat& qkt_output, const TensorInt& lengths) {

    int n_batch = qkt_output.shape()[0];
    int n_sequence = qkt_output.shape()[1];
    float* qkt_data = qkt_output.data();
    const int* lengths_data = lengths.data();
    for (int i = 0; i < n_batch; ++i) {
        float* qkt_batch_data = qkt_data + i * n_sequence;
        int cur_length = lengths_data[i];
        float maxVal =  -std::numeric_limits<float>::infinity();
        for (int j = 0; j < cur_length; ++j) {
            maxVal = std::max(maxVal, qkt_batch_data[j]);
        }
        float sum = 0;
        for (int j = 0; j < cur_length; ++j) {
            sum += std::exp(qkt_batch_data[j] - maxVal);
        }
        for (int j = 0; j < n_sequence; ++j) {
            if (j >= cur_length) {
                qkt_batch_data[j] = 0.0f;
            } else {
                qkt_batch_data[j] = std::exp(qkt_batch_data[j] - maxVal) / sum;
            }
        }
    }
}

/**
 * softmax_result: [n_batch, n_sequence] 
 * v_cache: [n_batch, n_sequence, output_dim]
 * attention_result: [n_batch, output_dim]
 */
void softmax_v_host(
    const TensorFloat& softmax_result, const TensorFloat& v_cache, TensorFloat& attention_result,
    const TensorInt& lengths) {
    
    int n_batch = softmax_result.shape()[0];
    int n_sequence = softmax_result.shape()[1];
    int output_dim = v_cache.shape()[2];
    const float* softmax_result_data = softmax_result.data();
    const float* v_cache_data = v_cache.data();
    float* attention_result_data = attention_result.data();
    const int* lengths_data = lengths.data();
    for (int i = 0; i < n_batch; ++i) {
        const float* softmax_result_batch_data = softmax_result_data + i * n_sequence;
        const float* v_cache_batch_data = v_cache_data + i * n_sequence * output_dim;
        float* attention_result_batch_data = attention_result_data + i * output_dim;
        int cur_length = lengths_data[i];
        for (int j = 0; j < output_dim; ++j) {
            float result = 0;
            for (int k = 0; k < cur_length; ++k) {
                result += (softmax_result_batch_data[k] * v_cache_batch_data[k * output_dim + j]);
            }
            attention_result_batch_data[j] = result;
        }
    }
}

void self_attention_inference_host(const TensorFloat& inp, const TensorInt& lengths,
    const TensorFloat& wk,
    const TensorFloat& wq,
    const TensorFloat& wv,
    const TensorInt& new_batch_idx, TensorFloat& kt_cache, TensorFloat& v_cache,
    // avoid frequent creation of tensors
    TensorFloat& q_output, TensorFloat& qkt_output, TensorFloat& attention_result) {

    fill_new_kt_v_cache(inp, new_batch_idx, lengths, wk, wv, kt_cache, v_cache);

    get_latest_kt_q_v(inp, lengths, wk, wq, wv, kt_cache, v_cache, q_output);

    qkt_host(q_output, kt_cache, lengths, qkt_output);

    softmax_in_place_with_lengths_host(qkt_output, lengths);

    softmax_v_host(qkt_output, v_cache, attention_result, lengths);
}

TensorFloat transpose_host(const TensorFloat& inp_host) {
    int batch_size;
    int rows;
    int cols;
    TensorFloat result({1});
    if (inp_host.shape().size() == 3) {
        batch_size = inp_host.shape()[0];
        rows = inp_host.shape()[1];
        cols = inp_host.shape()[2];
        result = TensorFloat({batch_size, cols, rows}, DeviceType::HOST);
    } else {
        assert(inp_host.shape().size() == 2);
        batch_size = 1;
        rows = inp_host.shape()[0];
        cols = inp_host.shape()[1];
        result = TensorFloat({cols, rows}, DeviceType::HOST);
    }

    const float* inp_host_data = inp_host.data();
    float* output_data = result.data();
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < rows; ++j) {
            for (int k = 0; k < cols; ++k) {
                output_data[i * rows * cols + k * rows + j] = inp_host_data[i * rows * cols + j * cols + k];
            }
        }
    }
    return result;
}

/**
 * emb_table: [n_vocab, inputDim]
 * wpe: [n_sequence, inputDim]
 * inp: [n_batch, n_sequence]
 * output: [n_batch, n_sequence, inputDim]
 * lengths: [n_batch]
 * new_item_indices: [n_new_items]
 */
void inference_optimized_encoder_host(const float* emb_table, const float* wpe, const int* inp, float* output, const int* lengths,
    const int* new_item_indices,
    int batch_size, int n_sequence, int embedding_dim, int n_new_items) {

    for (int i_new_index = 0; i_new_index < n_new_items; ++i_new_index) {
        int i_batch = new_item_indices[i_new_index];
        for (int i_seq = 0; i_seq < lengths[i_batch]; ++i_seq) {
            int token_id = inp[i_batch * n_sequence + i_seq];
            for (int i_dim = 0; i_dim < embedding_dim; ++i_dim) {
                output[i_batch * n_sequence * embedding_dim + i_seq * embedding_dim + i_dim] =
                    emb_table[token_id * embedding_dim + i_dim] + wpe[i_seq * embedding_dim + i_dim];
            }
        }
    }
}
