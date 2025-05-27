#include "constants.h"
#include "inferencer.h"
#include "test_utils.h"
#include <gtest/gtest.h>

TEST(InferencerTest, InferencerTest) {
    size_t n_batch = get_random_number(32, 100) * 4;
    size_t n_sequence = get_random_number(32, 100) * 4;
    size_t n_vocab = get_random_number(EOF_TOKEN_ID + 1, EOF_TOKEN_ID + 1024);
    size_t emb_dims = get_random_number(256, 512) * 4;

    ItemStorage item_storage;
    ProcessingStorage processing_storage;

    TensorFloat emb_table = get_random_device_tensor({n_vocab, emb_dims});
    TensorFloat pos_table = get_random_device_tensor({n_sequence, emb_dims});

    int n_items = get_random_number(n_batch * 2, n_batch * 3);
    for (int i = 0; i < n_items; ++i) {
        item_storage.add_new_item(std::make_pair(i, create_random_vector(get_random_number(1, n_sequence / 2), 0, EOF_TOKEN_ID - 1)));
    }

    InferenceModel model(
        SelfAttentionLayer(
            get_random_device_tensor({emb_dims, emb_dims}),
            get_random_device_tensor({emb_dims, emb_dims}),
            get_random_device_tensor({emb_dims, emb_dims}),
            n_batch, emb_dims, n_sequence),
        EncoderLayer(),
        DecoderLayer(n_batch, n_vocab),
        emb_table,
        pos_table,
        n_batch, n_sequence, emb_dims);
    start_inference_engine(
        emb_table, pos_table, item_storage,
        processing_storage,
        model, n_batch, n_sequence);
    
    ASSERT_EQ(item_storage.finish_count(), n_items);
}
