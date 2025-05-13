#include "kernels/encoder.h"
#include "tensor.hpp"
#include "test_utils.h"
#include <gtest/gtest.h>

TEST(EncoderTest, EncoderTest) {
    size_t n_batch = 1024;
    size_t n_sequence = 512;
    size_t max_sequence = 1024;
    size_t n_vocab = 4096;
    size_t embedding_dim = 1024;

    auto wte_device_host = get_random_device_host_tensor({n_vocab, embedding_dim});
    auto wpe_device_host = get_random_device_host_tensor({max_sequence, embedding_dim});
    auto inp_device_host = get_random_device_host_tensor_int({n_batch, n_sequence}, n_vocab);

    TensorFloat wte_device = wte_device_host.first;
    TensorFloat wpe_device = wpe_device_host.first;
    TensorInt inp_device = inp_device_host.first;
    TensorFloat output_device({n_batch, n_sequence, embedding_dim}, DeviceType::DEVICE);

    launch_encoder_kernel(
        wte_device.data(), wpe_device.data(), inp_device.data(),
        output_device.data(), n_batch, n_sequence, embedding_dim);

    TensorFloat output_host = encoder_host(wte_device_host.second, wpe_device_host.second, inp_device_host.second, n_batch, n_sequence, embedding_dim);

    assert_near(output_device, output_host);
}
