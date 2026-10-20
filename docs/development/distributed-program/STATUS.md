# Native captured-program distribution

Implementation freeze 1 (2026-10-20), qualification pending. This is engineering
status, not a published support claim.

- `Program.distribute(devices, logical_shards=..., optimizer=...)` copies a
  user-composed captured graph and its existing optimizer state into native
  workers. `fit` owns every epoch, minibatch, graph walk, backward pass, gradient
  fold and update in Mojo. No Python worker or training loop is involved.
- Logical microbatches are independent of physical device count. Each physical
  GPU evaluates one shard per wave concurrently through `host_parallelize`.
  Shard gradients are weighted by sample count and folded in ascending logical
  order. Dropout counters use `(optimizer_step, logical_shard, graph_node)`;
  row permutations use `(seed, epoch)`. Comparison requires the same logical
  shard count. It does not promise the rounding of an unsplit minibatch.
- Model parameters are replicated. Optimizer moment buffers and accumulated
  gradients are divided into disjoint contiguous element ranges on the physical
  GPUs. `residency()` returns actual ownership ranges. A parameter smaller than
  the world has empty owners. This is optimizer sharding, not model sharding.
- The portable checkpoint includes graph, full parameter and moment bytes,
  optimizer hyperparameters and scalar powers, trainability, logical shard
  count, seed, epoch and step. A changed physical device count repartitions
  these canonical bytes. Complete-step saves only; a failed computation poisons
  the handle until a checkpoint is restored.
- Inference currently executes on the first worker. Transfers use native host
  staging on all vendors; no communication or throughput speedup is claimed.

The frozen executable check entry is
`python/mojolearn/tests/test_distributed_program.py`. It contains an independent
analytic SGD oracle, uninterrupted/resumed Adam with dropout and scheduling,
optimizer-state migration, corrupted-file refusal and physical multi-GPU
ownership/replay. Set `MOJOLEARN_REQUIRE_DISTRIBUTED_PROGRAM=1` so an old wheel
fails; set `MOJOLEARN_DISTRIBUTED_PROGRAM_DEVICES=0,1` for physical coverage.
One-device results do not qualify multi-device behavior. CPU replay accepts
`devices=(0,)` and preserves multiple logical shards.

Pending: integration compilation, installed CPU/Apple/NVIDIA/AMD executions,
physical two-GPU NVIDIA/AMD coverage, whole-workload performance, tensor
parallelism, model-state partitioning and overlapped pipeline execution.
The latter three are not represented as completed by this data-parallel path.
