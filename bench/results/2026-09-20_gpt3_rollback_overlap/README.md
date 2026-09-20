# GPT-3-small pooled rollback snapshot overlap

Hardware: one guarded RunPod with two NVIDIA L40S GPUs. Target shape was
`B=1, L=2048, d_model=768, heads=12, layers=12, intermediate=2048,
vocab=50257` (162,147,840 parameters), IDENTICAL FP32.

The candidate removes only the final wait after each owner's three rollback
snapshot copies. Owners are disjoint. The existing first-moment scan at the
start of each owner update drains that same in-order context before update;
rollback restoration also queues after snapshot copies and drains the context.

The strict fault matrix passed baseline-versus-candidate for logical shards
2, 3, and 5. It covers full state/gradient hashes, optimizer ownership,
post-update Python failure, native `grad_nonfinite`, `opt_refuse`,
`after_nonfinite`, and `after_negative` faults, rollback, and replay.

The rotated target matrix also passed complete loss and final
parameter/moment/flag/gradient hashes. Two-device median seconds and speedups:

| round | shards | baseline | candidate | speedup |
|---:|---:|---:|---:|---:|
| 1 | 2 | 0.304252 | 0.297264 | 1.02351x |
| 1 | 3 | 0.495092 | 0.489707 | 1.01100x |
| 1 | 5 | 0.707413 | 0.713140 | 0.99197x |
| 2 | 2 | 0.303113 | 0.298179 | 1.01655x |
| 2 | 3 | 0.495973 | 0.489819 | 1.01256x |
| 2 | 5 | 0.714019 | 0.701765 | 1.01746x |

One-device ratios stayed between 0.9935x and 1.0028x, consistent with the
candidate exposing concurrency only across owner contexts. Shards 2 and 3
improved in both rotations; shard 5 had one noisy regression and one gain.
The source is retained as a small, arithmetic-free, broadly non-regressing
multi-device improvement.

The first attempted target matrix used logical shard 1 with two devices and
correctly refused `devices > shards`. The committed harness now uses valid
shards 2, 3, and 5. The optimizer-pool gate's old replicated-reduction
allocation assertion was also corrected from `8*n` to `4*n`: the previously
landed staging-reuse implementation retains only the distinct total buffer.

Pod `rc7n2k02a5as12` was terminated at 2026-09-20 15:40:07 EDT. DELETE
returned HTTP 204 and the immediate verification GET returned HTTP 404.
