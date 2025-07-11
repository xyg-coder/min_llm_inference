
#pragma once

#include "item_storage.h"
#include "paged_item_storage.h"
#include "tensor.hpp"
#include <optional>
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
    int n_new_batches;
};

std::pair<TensorFloat, TensorFloat> get_random_device_host_tensor(const std::vector<size_t>& shape, float ratio = 1);
TensorFloat get_random_device_tensor(const std::vector<size_t>& shape, float ratio = 1);
TensorFloat get_random_device_tensor(const std::vector<size_t>& shape, int max_val);
TensorInt get_random_device_tensor_int(const std::vector<size_t> &shape, int max_val);

std::pair<TensorInt, TensorInt> get_random_device_host_tensor_int(const std::vector<size_t>& shape, int max_val);

TensorFloat host_matrix_multiply(const TensorFloat& inp1, const TensorFloat& inp2);

TensorFloat softmax(const TensorFloat& inp);

TensorFloat transpose_host(const TensorFloat& inp_host);

void assert_near(const TensorFloat& tensor_device, const TensorFloat& tensor_host, float threashold = 1e-3);
void assert_near(const TensorInt& tensor_device, const TensorInt& tensor_host);
void assert_near_on_host(const TensorFloat &tensor_device, const TensorFloat &tensor_host, float threshold=1e-3);
void assert_near_on_host(const TensorInt &tensor_device, const TensorInt &tensor_host);

void print_host(const float* data, int size);

TensorFloat encoder_host(const TensorFloat& wte, const TensorFloat& wpe, const TensorInt& inp, size_t batch_size, size_t n_sequence, size_t embedding_dim);

int get_random_number(int min, int max);

std::vector<int> get_unique_num_array(int min, int max, int size);

std::vector<int> create_random_vector(size_t size, int min, int max);

std::pair<TensorWrapForInferenceOptimizedSelfAttention, TensorWrapForInferenceOptimizedSelfAttention> generate_device_and_host_tensors(
    size_t n_batch=1024, size_t n_sequence=1024, size_t input_dim=32, size_t output_dim=32);


std::pair<TensorWrapForInferenceOptimizedSelfAttention, TensorWrapForInferenceOptimizedSelfAttention> generate_random_shape_device_and_host_tensors();

void inference_optimized_encoder_host(const float* emb_table, const float* wpe, const int* inp, float* output, const int* lengths,
    const int* new_item_indices,
    int batch_size, int n_sequence, int embedding_dim, int n_new_items);

TensorFloat gemm_transpose_host(const TensorFloat& s1, const TensorFloat& s2);

void decoder_host(
    const TensorFloat& batch_embs, const TensorFloat& emb_table,
    TensorFloat& emb_score,
    const TensorFloat& wpe_table,
    TensorFloat& inp, TensorInt& lengths, TensorInt& decoder_result);

TensorFloat mock_emb_table(int n_vocab, int embedding_dims);

TensorFloat mock_pos_table(int n_sequence, int embedding_dims);

class PagedAttentionTestWrapper {
public:
    PagedAttentionTestWrapper(
        PagedAttentionsManager&&, MemoryBlockManager&&, ProcessingStorage&&, ItemStorage&&, std::vector<IdTokensPair>&&);
    PagedAttentionsManager paged_attention_manager;
    MemoryBlockManager memory_block_manager;
    ProcessingStorage processing_storage;
    ItemStorage item_storage;
    std::vector<IdTokensPair> tokens;
};

PagedAttentionTestWrapper mock_paged_attention_test_wrapper(
    size_t max_batches, size_t n_sequence, size_t emb_dim, int n_blocks,
    const std::vector<int>& new_items_lengths);

class TensorWrapperForPagedAttention {
public:
    TensorFloat inp_embedding;
    TensorInt lengths;
    TensorFloat wk;
    TensorFloat wq;
    TensorFloat wv;
    TensorInt new_batch_idx;
    TensorFloat page_table_memory;
    TensorFloatPoint page_table;
    TensorFloat q_output;
    TensorFloat qkt_output;
    TensorFloat attention_result;
    TensorFloat kt_cache;
    TensorFloat v_cache;
    int n_new_batches; 
};

TensorWrapperForPagedAttention generate_paged_attention_wrapper_device_tensors(size_t n_batch=1024, size_t n_sequence=1024, size_t emb_dim=32, std::optional<TensorInt> lengths_override=std::nullopt);

TensorFloat get_random_device_emb_table(size_t emb_dim, size_t n_vocab, float eof_larger_ratio=1.1);
