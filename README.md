# min_llm_inference

## How to run

```
# -DUSE_ASYNC_ALLOC enabled async tensor creation / deletion
cd build
cmake -DUSE_ASYNC_ALLOC=ON ..
make ..
ctest
```

## Plan

* Embedding
* Demo inference
