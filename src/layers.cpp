#include "tensor.hpp"
#include "layers.h"
#include <stdexcept>
#include "kernels/gemm.h"
#include "kernels/paged_attention.h"
#include "kernels/self_attention_inference_optimized.h"
#include "kernels/decoder.h"
#include "kernels/encoder.h"

FeedForward::FeedForward(const TensorFloat& w, std::optional<const TensorFloat> b): weight_(w), bias_(std::move(b)) {
    auto weight_shape = weight_.shape();
    if (weight_shape.size() != 2) {
        throw std::invalid_argument("[FeedForward] Weight must be 2D (got shape size " +
            std::to_string(weight_shape.size()) + ")");
    }
    if (weight_shape[0] == 0 || weight_shape[1] == 0) {
        throw std::invalid_argument("[FeedForward] Weight dimensions must be greater than zero.");
    }

    if (bias_) {
        const auto bias_shape = bias_->shape();
        if (bias_shape.size() != 1) {
            throw std::invalid_argument("[FeedForward] Bias must be 1D (got shape size " +
                std::to_string(bias_shape.size()) + ")");
        }
        if (bias_shape[0] != weight_shape[1]) {
            throw std::invalid_argument("[FeedForward] Bias size must match weight's out_features (expected " +
                std::to_string(weight_shape[1]) + ", got " + std::to_string(bias_shape[0]) + ")");
        }
    }
}

void FeedForward::forward(const TensorFloat& input, TensorFloat& output) {
    size_t n_batch = input.shape()[0];
    size_t in_features = input.shape()[1];
    size_t out_features = weight_.shape()[1];

    float* bias_data_ptr = nullptr;
    Stride3D bias_stride({0, 0, 0});
    if (bias_) {
        bias_data_ptr = bias_->data();
        bias_stride = Stride3D({0, 0, 1});
    }

    launch_gemm_bias_kernel(
        input.data(), Stride3D{0, in_features, 1},
        weight_.data(), Stride3D{0, out_features, 1},
        bias_data_ptr, bias_stride,
        output.data(), Stride3D{0, out_features, 1},
        1, n_batch, in_features, out_features
    );
}

SelfAttentionLayer::SelfAttentionLayer(TensorFloat&& wk, TensorFloat&& wq, TensorFloat&& wv,
    size_t n_batch, size_t input_dim, size_t n_sequence): wk_(std::move(wk)), wq_(std::move(wq)), wv_(std::move(wv)),
    kt_cache_(TensorFloat({n_batch, input_dim, n_sequence}, DeviceType::DEVICE)),
    v_cache_(TensorFloat({n_batch, n_sequence, input_dim}, DeviceType::DEVICE)),
    q_output_({n_batch, input_dim}, DeviceType::DEVICE),
    qkt_output_({n_batch, n_sequence}, DeviceType::DEVICE) { }


void SelfAttentionLayer::forward(const TensorFloat& inp_embedding, const TensorInt& lengths,
    const TensorInt& new_batch_idx, TensorFloat& attention_result, int n_new_items) {
    
    inference_self_attention(inp_embedding, lengths, wk_, wq_, wv_, new_batch_idx, kt_cache_,
        v_cache_, q_output_, qkt_output_, attention_result, n_new_items);
}


PagedAttentionLayer::PagedAttentionLayer(TensorFloat&& wk, TensorFloat&& wq, TensorFloat&& wv,
    size_t n_batch, size_t emb_dim, size_t n_sequence): wk_(std::move(wk)), wq_(std::move(wq)), wv_(std::move(wv)),
    q_output_({n_batch, emb_dim}, DeviceType::DEVICE),
    qkt_output_({n_batch, n_sequence}, DeviceType::DEVICE) { }


PagedAttentionCublasLayer::PagedAttentionCublasLayer(TensorFloat&& wk, TensorFloat&& wq, TensorFloat&& wv,
    size_t n_batch, size_t emb_dim, size_t n_sequence): wk_(std::move(wk)), wq_(std::move(wq)), wv_(std::move(wv)),
    q_output_({n_batch, emb_dim}, DeviceType::DEVICE),
    qkt_output_({n_batch, n_sequence}, DeviceType::DEVICE),
    latest_emb_({n_batch, emb_dim}, DeviceType::DEVICE),
    temp_placeholder_({n_batch, emb_dim}, DeviceType::DEVICE) { }


void PagedAttentionLayer::forward(TensorFloatPoint& page_table, const TensorInt& lengths,
        const TensorInt& new_batch_idx, TensorFloat& attention_result, int n_new_items) {

    int n_sequence = qkt_output_.shape()[1];
    paged_attention(page_table, lengths, wk_, wq_, wv_, new_batch_idx, q_output_, qkt_output_,
        attention_result, n_new_items, n_sequence);
}

void PagedAttentionCublasLayer::forward(TensorFloatPoint& page_table, const TensorInt& lengths,
    const TensorInt& new_batch_idx, TensorFloat& attention_result, int n_new_items,
    cublasHandle_t& handle) {

    int n_sequence = qkt_output_.shape()[1];
    paged_attention_with_cublas(
        page_table, lengths, wk_, wq_, wv_, new_batch_idx, q_output_, qkt_output_, attention_result, latest_emb_, temp_placeholder_,
        n_new_items, n_sequence, handle);
}


void EncoderLayer::forward(const TensorFloat& emb_table, const TensorFloat& pos_emb, const TensorInt& inp,
    TensorFloat& inp_embedding, const TensorInt& lengths,
    const TensorInt& new_item_indices, int n_new_items) {

    int n_batch = inp_embedding.shape()[0];
    int n_sequence = inp_embedding.shape()[1];
    int embedding_dim = inp_embedding.shape()[2];

    launch_inference_optimized_encoder_kernel(
        emb_table.data(), pos_emb.data(), inp.data(), inp_embedding.data(),
        lengths.data(), new_item_indices.data(), n_batch, n_sequence, embedding_dim, n_new_items);
}

void PagedEncoderLayer::forward(const TensorFloat& emb_table, const TensorFloat& pos_emb, const TensorInt& inp,
    TensorFloatPoint& page_table, const TensorInt& lengths,
    const TensorInt& new_item_indices, int n_new_items) {

    int n_batch = inp.shape()[0];
    int n_sequence = inp.shape()[1];
    int emb_dim = emb_table.shape()[1];

    launch_paged_attention_encoder_kernel(
        emb_table.data(), pos_emb.data(), inp.data(), page_table.data(), lengths.data(),
        new_item_indices.data(), n_batch, n_sequence, emb_dim, n_new_items);
}

DecoderLayer::DecoderLayer(size_t n_batch, size_t n_vocab): emb_score_(TensorFloat({n_batch, n_vocab}, DeviceType::DEVICE)) { }

void DecoderLayer::forward(const TensorFloat& batch_result, const TensorFloat& emb_table,
    const TensorFloat& wpe_table,
    TensorFloat& inp_embedding, TensorInt& lengths, TensorInt& decoder_result) {

    launch_decoder(batch_result, emb_table, emb_score_, wpe_table, inp_embedding, lengths, decoder_result);
}

PagedDecoderLayer::PagedDecoderLayer(size_t n_batch, size_t n_vocab): emb_score_(TensorFloat({n_batch, n_vocab}, DeviceType::DEVICE)) { }

void PagedDecoderLayer::forward(const TensorFloat& batch_result, const TensorFloat& emb_table,
    const TensorFloat& wpe_table,
    TensorFloatPoint& page_table, TensorInt& lengths, TensorInt& decoder_result, int i_decoder_round) {

    launch_paged_attention_decoder_multi_rounds(batch_result, emb_table, emb_score_, wpe_table, page_table, lengths, decoder_result, i_decoder_round);
}
