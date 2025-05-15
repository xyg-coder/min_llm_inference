# min_llm_inference

## How to run

```
# -DUSE_ASYNC_ALLOC enabled async tensor creation / deletion
make all_test
```

## Text generation design

* Self-attention layer initializes with empty kv-cache, and new_batch_idx is all batch_idx.
* Self-attention layer will keep track of the kv-cache.
* Generation returns the new_tokens of each. Use greedy for now, it should be able to be done using gpu.
* With the new tokens, we can generate a new new_batch_idx using gpu (can be done using prefix scan).

## Plan

* Demo inference
* A batch. Keep generating the next token.
    * Possible optimizations:
        * dynamic batching
    * Decoder
* How to use multi-gpus to accelerate
