# GPT-3-small-like IDENTICAL multi-GPU training audit

Target shape: batch 1, sequence 2048, d_model 768, 12 attention heads, 12
layers, intermediate 2048, vocabulary 50,257; 162,147,840 FP32 parameters.

## Production ordering

`ByteParallelTrainer.step` assigns logical shards to physical trainers in
ascending logical-shard order. Each wave joins all gradient producers before
reduction. For every parameter, the reduction is the same left fold: shard 0
is copied, then shards 1 through K-1 are added by `_ordered_add_kernel` using
the IDENTICAL FTZ/FMA primitive. Owner parameter ranges are disjoint, so their
concurrent execution introduces no cross-owner floating-point reduction.
NCCL and topology-selected collectives are not used. Changing physical device
count while retaining logical shard count/order is therefore intended to
leave the arithmetic tree unchanged.

The pooled optimizer owns disjoint parameter/moment ranges, broadcasts only
updated parameter bytes, and validates every replica before publication.
Failure rolls back every owner. Device list order controls ownership and work
placement but not logical reduction order.

## Qualification status

Small-shape `par-byte-lm` records exist on Apple, NVIDIA, AMD, and two-H100
columns. They do not qualify this target shape's 1-vs-2-device performance,
memory, or state hashes. The production parallel binding has no CPU execution
path, so CPU/vendor equality is not established by a CPU fallback and must be
reported as not applicable rather than inferred.

`tools/lm_shards_probe.py` now accepts explicit physical device ids and emits
loss hashes plus final parameter, first-moment, second-moment, flag, and
gradient hashes. `tools/lm_shards_compare.py` refuses any experiment-field,
loss, or final-state mismatch before reporting speedup. Required qualification
is two separate target-shape runs from the same seed and logical shards, first
with `--devices 0`, then `--devices 0 1`, including device memory receipts.

## Candidate awaiting physical qualification

The current uncommitted candidate skips the pooled reducer's copy of an
owner's already-local gradient slice into its incoming scratch. It folds that
slice directly, retaining the same copy/add choice and logical position;
remote slices and all synchronizations remain unchanged. With two devices and
two logical shards this removes one complete model's worth of D2D slice copies
per optimizer step (648,591,360 bytes for this shape), without changing memory
capacity or any FP32 operation. It compiles and the ordered-gradient
cancellation/FTZ gate passes locally. It is not mergeable until physical
1-vs-2 baseline/candidate hashes, timing, and memory are recorded.

Cloud execution is pending because the live fleet was owned by three GEMM
pods and a CPU parity pod at audit time. None was touched and no fifth pod was
rented.
