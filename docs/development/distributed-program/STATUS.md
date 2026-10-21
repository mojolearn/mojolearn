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

## Model, tensor and pipeline executor (qualification pending)

`Program.partition` builds a separate immutable logical graph. Eligible
single-use rank-two matrix weights are packed into independent output-column
slices before device upload. Every parameter slice and its optimizer buffers
have one physical owner. Shared tensors retain one authoritative owner and
are copied transiently to consumers. The checkpoint retains the transformed
graph, logical placement, full parameter/moment bytes, optimizer configuration,
epoch, step, shuffle seed and dropout coordinates; a load may change physical
device count while preserving this logical graph.

Native worker tasks schedule DAG levels across pipeline microbatches. Independent
tensor branches and different microbatch stages can run concurrently. Every
operation interval records node, microbatch, logical stage and physical device.
Activations are staged through native host buffers; local VJPs recompute each
node. This is a correctness-first executor, with substantial transfer and
recomputation cost. No memory-limit or throughput improvement is admitted until
physical-device residency, overlap and full-workload measurements establish it.

The initial native data-parallel debug run found a transfer-buffer lifetime bug:
converting a List pointer to an untracked FP let Mojo destroy the List before
the copy. Retained native disassembly proves free-before-write; the repair keeps
the List owner alive through the final transfer. The failing run remains part
of the evidence.
