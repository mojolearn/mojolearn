# Chunked LM-head v2 dWeight probability reuse

The prior `_chunk_dweight_kernel` nested features outside rows. It therefore
recomputed the token/row shifted logit, exponential, probability, and mean
gradient once per feature. The new loop computes that shared `dlogit` once per
row/token, then updates all feature cells. Every individual dWeight cell still
executes the same zero initialization and row-ascending
`identical_mul_add(dlogit, hidden, accumulator)` chain, followed by `ftz`.

Local Metal, IDENTICAL, B=1/L=64/DM=64/V=8192 end-to-end ByteTrainer step:

- merged pre-change evidence: 122.362, 128.988, 149.995, 157.341 ms, with a
  later post-merge witness of 147.358 ms;
- dWeight-reuse candidate: 119.393, 98.759, 93.247 ms;
- same candidate-run V1: 41.730, 64.587, 22.403 ms.

Later samples taken while several concurrent local build/controllers were
active were noisy (candidate 105.916, 162.275, and 207.297 ms). They are
retained here rather than filtered. Therefore the arithmetic optimization is
locally exact and promising, but NVIDIA promotion remains required before a
cross-vendor speed claim; the shared RunPod fleet was occupied by another
live session and this lane did not add a sixth rental.

The direct 7x513x17 device gate crosses the chunk boundary and matches the CPU
oracle bit-for-bit for loss, maxima, denominators, every dHidden cell, and
every dWeight cell. Independent end-to-end V2 trainers also produce identical
updated parameter bits. Storage is unchanged: one rows-by-256 logits chunk and
the existing row statistics.
