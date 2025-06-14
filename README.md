# min_llm_inference

A toy project to learn about cuda and language inference.

## High Level Design

* See [inferencer.cpp](./src/inferencer.cpp) for the basic structure.
* The inference has 3 steps:
    * Given the indices of the finished items. And insert new rows into the input.
    * Call `inference_model.forward` to get the next tokens for the batch.
    * Process the decoder results, to know the indices of the finished items.

## Model

For learning purpose, it is only one single self-attention block. But I did some optimizations for inference. See [code](./src/kernels/self_attention_inference_optimized.cu) for details.

```
# calculate the kt, v cache for the new batches
launch_fill_new_kt_v_cache

# for each batch, calculate the kt,q,v for the latest token
launch_get_latest_kt_q_v

# calculate q * kt
launch_qkt

# in-place softmax(qkt / sqrt(dim))
launch_softmax_in_place_with_lengths

# softmax(qkt) * v
launch_softmax_v
```

Note: we keep track of `lengths` of each batch, which is the sequence length of each batch, to avoid unnecessary calculations.

## Test

We implement 2 memory allocation modes: sync and async.

```
# -DUSE_ASYNC_ALLOC enabled async tensor creation / deletion
make all_test   # run all tests in 2 memory allocation modes
make # build and run tests
```

## Debug

* See [debug_in_vscode.md](./doc/debug_in_vscode.md) to see how to use breakpoint in vscode.

## Paged Attention

* One simple paged attention is also implemented. And from experiments, it seems the gains are large when the results are short.

## Profiling Result

`./build/to_profile`

### Non-pinned memory

Total tokens: 197013, seconds: 18.812, throughput: 10472.7

In `process_decoder_result`, `copy_from` takes 91ms.

### Pinned memory

Total tokens: 196684, seconds: 18.987, throughput: 10358.9

In `process_decoder_result`, `copy_from` takes 91ms.

## Plan

* `decoder_result` should be larger. So one time, we can clone much larger. And this can make insert_new less called.
* How to use multi-gpus to accelerate
