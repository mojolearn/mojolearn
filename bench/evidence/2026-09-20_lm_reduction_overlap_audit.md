# Ordered LM reduction overlap audit

Branch base: `77b1162b0`. The production candidate is intentionally not part
of this evidence commit; two-device NVIDIA qualification is still pending.

## Dependency proof

For one logical rank, each optimizer owner has a disjoint parameter interval
and a distinct context, incoming buffer, and total buffer. Within an owner's
interval the required sequence remains:

1. finish that rank's gradient;
2. transfer the owner's slice into its incoming buffer;
3. copy it for logical shard zero, otherwise apply `_ordered_add_kernel`;
4. finish the fold before the next rank can reuse the incoming buffer.

The candidate changes only scheduling between disjoint owners. It visits
remote owners before the rank-local owner, queues every owner fold, then joins
every owner before advancing to the next rank. Thus every parameter retains
the same ascending logical-shard fold and no incoming buffer is reused early.

On NVIDIA, `transfer_bytes` drains the source context after queueing a peer
copy. With the shipped local-owner-first order, that source drain immediately
serializes local-owner work. Remote-first ordering and a deferred all-owner
join permit remote target folds to overlap later source work. On AMD, the
correctness-required host-staging path drains both source and target, so the
candidate is safe but is not expected to create the same overlap.

## Local build and exactness gate

Both baseline and candidate byte-LM bindings compiled independently on an
Apple M4 with IDENTICAL mode and at most two compiler jobs. SHA-256:

- baseline: `fe3f51baf1acf6fa73dde999fac8ae970ca19706a676c03a9b791eba66c519db`
- candidate: `671ba9c689b096a54bb3635116a11471ae6b4bea9e83e1c01134b7912a44b85f`

The one-device control shape ran three steps at logical-shard counts 1, 2 and
4. Baseline and candidate matched every loss and the complete parameter,
first moment, second moment, optimizer-flag, and gradient hashes at each shard
count. The ordered-gradient cancellation, signed-zero, and subnormal seam gate
also passed. These local timings are setup-dominated and are not a performance
claim.

## Larger effective batches

Logical shard count may increase through 1024 without allocating another
trainer per shard. The implementation executes waves of at most the physical
device count and reuses trainer activation storage, incoming buffers, and
owner totals. Therefore memory is approximately constant in logical shard
count, while compute and tokens per optimizer step grow with it.

Increasing the shard count is deterministic for that chosen count and retains
the ascending fold, independent of whether one or two physical devices execute
the waves. It is not state-equivalent to a smaller count: additional
microbatches contribute additional summed gradients, and the public contract
explicitly sums rather than averages them. No learning-rate rescaling is
silently applied. Exact comparisons must hold shard count and ordered input
microbatches fixed while varying physical device count or implementation.

`tools/lm_shards_ab_matrix.sh` is the physical gate. It runs baseline and
candidate in forward and reverse order on one and two devices, at logical
shard counts 1, 2 and 4, with eight steps per arm. It refuses output reuse and
uses `lm_shards_ab_compare.py` to require all loss and full-state hashes before
reporting timing ratios.
