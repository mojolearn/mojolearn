# Byte parallel replicated-gradient staging reuse

Mode: `MOJOLEARN_NUMERIC_IDENTICAL`; branch base `b2f9a1a00`.

This change is deliberately limited to `pool_optimizer=False`.  The first
physical replica computes the first logical shard in every wave.  Once that
gradient has been copied/added to the distinct `total` accumulator it is dead,
so the replica's full `grad` allocation can receive later replicas' bytes.  A
new wave computes rank zero before any remote transfer overwrites it.  The
logical copy/add choice, `_ordered_add_kernel`, rank order, and accumulator do
not change.  The default pooled optimizer is untouched: its owner-local
gradient ranges are not dead early enough to provide this alias safely.

## GPT-3-small-like capacity

For the repository target in
`2026-09-20_gpt3_small_multigpu_audit.md` (B1, L2048, d_model 768, 12
heads/layers, intermediate 2048, vocabulary 50,257), `n_total` is
162,147,840.  Removing one FP32 `n_total` allocation saves exactly
648,591,360 bytes (618.55 MiB) on device zero.  Reduction scratch changes
from two full gradients to one; model, optimizer, and activation allocations
are unchanged.

## Local exactness and timing

Hardware: Apple M4, 16 GiB, Darwin arm64.  Builds used at most two jobs and
explicitly passed `-D MOJOLEARN_NUMERIC_IDENTICAL`.

* `training/checks/ordered_gradient_check.mojo`: PASS, including cancellation,
  signed zero, and subnormal seams.
* `training/checks/byte_lm_offload_check.mojo` with `RUNPOD_POD_ID=local`:
  PASS for logical-shard cases `(layers, d_model, length, devices, shards)`
  `(1,16,7,1,1)` and twice for `(2,16,7,1,3)`.  These compare every loss,
  parameter, first/second moment, gradient, rollback, and replay bit against
  the independently stored offload implementation.  The executable then
  refused its expected cloud-only device-1 case on this one-GPU host.
* Interleaved whole-executable wall seconds after build were baseline
  `3.98, 2.75, 2.40` and candidate `2.69, 2.41, 2.49`.  This toy check is
  dominated by setup/export and is not a throughput claim; it establishes no
  repeatable regression while the capacity saving is exact.

No cloud resource was rented.  Physical peer-transfer qualification remains
desirable, but the remote transfer primitive and synchronization sequence are
unchanged; only its target allocation is reused storage on the same target
context.
