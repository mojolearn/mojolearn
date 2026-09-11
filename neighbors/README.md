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

DEVIATION 2631 (2026-09-11, lane/knn-finish): the wider NVIDIA query tile,
and the radix scratch the small-k selector never reads. The IDENTICAL tiled
arm took 512 queries at a time, so a 400,000 x 4,000 request ran 56 distance
launches, 56 selection launches and 48 partial merges. Kernel-matrix row
`knn_query_tile_for` carries the default tile per column and
`neighbors/estimator.mojo` reads it; the workspace budget admits that tile's
bounded distance tile (the index axis is tiled at 65,536 columns, so 2,048
queries is 512 MiB and 4,096 is 1 GiB) instead of halving it. Tiling cannot
move a bit: every cell's chain is a function of its own query row and index
column, and every row's selection and partial merges run in the same
column-tile order whatever the query tile. Row `knn_radix_scratch_shrink_for`
adds the second half: a k <= 64 IDENTICAL request never reaches the radix
selector, so its scratch is `k` pairs per query row rather than
`n_index // 8`, which is 1.6 GB not allocated at tile 2,048. Measured on the
H200 2026-09-11 (pod `zwmta1li2twxx2`, 400k x 4k x d32, three interleaved
runs per arm): request k10 23.53 ms at 512 to 22.39 at 1024, 21.69 at 2048
and 21.38 at 4096 (0.909), k15 25.98 to 23.52 (0.905), the scratch shrink
alone flat at 0.999; distance 15.30 to 14.43 ms, selection 7.33 to 5.99 ms,
merges 0.45 to 0.14 ms, launches 56/56/48 to 14/14/12. Every dumped index
and distance equals the 512 build's.
`-D MOJOLEARN_KNN_QUERY_TILE_ARM_512` / `_1024` / `_2048` / `_4096` are the
A/B arms and `-D MOJOLEARN_KNN_IDENTICAL_FULL_RADIX_SCRATCH=1` keeps the old
scratch. Evidence: `bench/results/knn_finish_2026-09-11/`.

DEVIATION 2667 (2026-09-11, lane/knn-finish): the fused distance and small-k
selection, MEASURED NEGATIVE. `neighbors/checks/fused_distance_select_identical.mojo`
computes each candidate's distance inside the selector's scan with the
register tile's own per-cell chain and writes no distance matrix, which is
the shape of cuVS's `fusedL2Knn` (`knn_brute_force.cuh:447-451`). Same bits
(every dumped index and distance equal to the two-launch form on both k;
`-D MOJOLEARN_KNN_FUSED_SELECT_SABOTAGE=1` moves them, so the scan is
reached), and 1.41x to 1.48x the request time at the best tile on the H200
(k10 33.26 ms against 21.38, k15 38.34 against 23.52). The distance matrix
was not the cost: the fused block owns ONE query row, so it reads 1.125
operands per cell per feature where the register tile's 8x4 cells per thread
read 0.375, and the fused launch class (33.3 ms at tile 2048) is larger than
the distance and selection classes it replaces (14.4 + 6.0 ms). So the row
`knn_fused_distance_select_for` is OFF on every column and the path is opt-in
through `-D MOJOLEARN_EXPERIMENTAL_KNN_FUSED_SELECT=1`;
`-D MOJOLEARN_KNN_IDENTICAL_UNFUSED_SELECT=1` forces the two-launch form.
A fused block that owns several query rows (one k-deep register list per row
per thread) is the open item.

At frozen source `6dd44ac5`, AMD MI325X and NVIDIA RTX 4090 pass all four
arms and match every one of those 5,440 records. The driver explicitly scopes
its device context and synchronizes host allocation before pointer writes;
this fixed a NVIDIA stall in the first stress-driver version. See the
[retained comparison](../bench/results/resume/2026-09-06-ordered-mamba-knn/cross-device-continued.json).
