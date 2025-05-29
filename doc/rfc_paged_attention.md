# RFC: Paged Attention Implementation

## Main features

* The `inp_embedding`, `kt_cache` and `v_cache` are all of shape `[n_batch, n_sequence, embedding_dim]`. It will be a lot of waste, use paged attention to replace those so we can support more batches.
* Define one `AVERAGE_LEN`. The initial `n_batch` should be `max_batch / AVERAGE_LEN`.
* When there are no blocks for ongoing batches, we need to free some batches.

## Paged Attention

* At most `max_batch / AVERAGE_LEN` batches to avoid throttle.
* If no remaining memory blocks, put the last ones to new_items, so we free memories.
