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

The `get_latest_k_q_v_paged_attention` is talking the longest time.

### Putting page_pos to shared memory to avoid memory scatter visit

[PR](https://github.com/xyg-coder/min_llm_inference/commit/14c48bf5b0a26d4166dd6abb95c973ecc4b38922)

Before putting to shared:
Total tokens: 194834, seconds: 18.861, throughput: 10330
Compute throughput: 12%

After putting to shared:
Total tokens: 197136, seconds: 8.761, throughput: 22501.5
Compute throughput: 24%

Use cublas inside the paged attention:
Total tokens: 195540, seconds: 3.835, throughput: 50988.3

## Plan

* How to use multi-gpus to accelerate
