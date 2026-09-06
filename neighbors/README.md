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

At frozen source `6dd44ac5`, AMD MI325X and NVIDIA RTX 4090 pass all four
arms and match every one of those 5,440 records. The driver explicitly scopes
its device context and synchronizes host allocation before pointer writes;
this fixed a NVIDIA stall in the first stress-driver version. See the
[retained comparison](../bench/results/resume/2026-09-06-ordered-mamba-knn/cross-device-continued.json).
