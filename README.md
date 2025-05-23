# min_llm_inference

## How to run

```
# -DUSE_ASYNC_ALLOC enabled async tensor creation / deletion
make all_test
```

## Plan

* ItemStorage unittest
* Demo inference
* How to use multi-gpus to accelerate
* self_attention test is flaky (seems cudaMalloc failure).
* Change the design from pdf to readme
