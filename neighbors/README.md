# Neighbors

GPU exact, radius, and ball-cover neighbor search. Provenance and the deliberately unsupported
surface are in the two TSV ledgers.

```bash
pixi run check-knn-identity
pixi run check-ball-cover-knn
pixi run check-radius
pixi run check-metric
```

Distance evaluation, candidate ordering, and equal-distance ties are part of the output contract.

The small-k selector and transposed-index optimizations are compile-time
choices inside IDENTICAL mode. Enable both when building a calling program:

```bash
pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
  -D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1 \
  -D MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL=1 \
  bench/knn_layout_adversarial_check.mojo -o /tmp/knn-flags-check
```

“Opt-in” means supplying these defines explicitly. Omit either define to
disable that optimization: these are presence flags, so `=0` still enables
them. An environment variable alone does not add a compiler define, and
released Python wheels are not changed by these source-build flags. FAST and
DETERMINISTIC ignore the two optimizations. The IDENTICAL flag remains the
arithmetic contract; the optimization flags must preserve its output bits.

The adversarial gate exercises non-dyadic values, duplicate rows, large
offsets, mixed feature scales, dimensions 1/3/17/33/65, query-tile tails, and
L2/rooted-L2 plus L1/cosine fallback. All 5,440 distance/index pairs must match
across the four flag combinations and named GPUs. See the
[experiment record](checks/SMALLK_DISPATCH_EXPERIMENT.md) for earlier broader
dispatch coverage and scoped timing evidence. These correctness checks do
not establish that either optimization is faster for every workload/device.

DEVIATION 2629 (2026-09-11, lane/knn-speed): the exact-chain admission. The
transposed register-tile distance (`checks/pinned_distance_tile.mojo`) flushes
every FMA step to zero when subnormal. One launch per matrix per request now
records each row's minimum nonzero and maximum biased exponent, with
nonfinite rows marked. A tile whose rows and columns give a minimum sum of at
least 174 and a maximum sum plus ceil(log2 d) of at most 376 cannot produce a
subnormal or an overflow anywhere in its chains, so it runs the same rounded
FMA chain without the flush; every other tile keeps the flushed chain. The
bits are equal by the proof written above `vector_exponent_admission_kernel`.
Measured NEUTRAL on the H100 on 2026-09-11 (400k x 4k x d32, three
interleaved pairs): request k10 23.65 to 23.85 ms and k15 26.24 to 26.19 ms,
distance class 15.31 to 15.41 ms, every dumped distance and index equal to
origin/main, sabotage reached. So kernel-matrix row
`knn_distance_exact_chain_for` is OFF on every column and the path is opt-in
through `-D MOJOLEARN_EXPERIMENTAL_KNN_EXACT_CHAIN=1`;
`-D MOJOLEARN_KNN_IDENTICAL_FLUSHED_CHAIN=1` forces the flushed chain and
`-D MOJOLEARN_KNN_EXACT_CHAIN_SABOTAGE=1` (never shipped) perturbs each
admitted step so a reached path cannot return clean bits. Evidence:
`bench/results/knn_speed_2026-09-11/`.

At frozen source `6dd44ac5`, AMD MI325X and NVIDIA RTX 4090 pass all four
arms and match every one of those 5,440 records. The driver explicitly scopes
its device context and synchronizes host allocation before pointer writes;
this fixed a NVIDIA stall in the first stress-driver version. See the
[retained comparison](../bench/results/resume/2026-09-06-ordered-mamba-knn/cross-device-continued.json).
