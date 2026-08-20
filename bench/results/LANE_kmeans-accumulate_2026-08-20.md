# LANE_kmeans-accumulate 2026-08-20: the accumulate's X read goes `veclen`-wide

Brief: the accumulate half (`cluster/mojo_only/reduce_by_key.mojo`) is the
widest bracket after the assignment flip -- 54 ms/iter at 4M x 32 float32,
~9.5 GB/s effective on a ~120 GB/s device -- and it is FAITHFUL to
upstream (LANE_kmeans-kernel audit row 6), so any win is a deliberate
deviation beyond upstream, priced as one. No timings were run in this
lane; the orchestrator re-verdicts.

## Access-pattern analysis (before)

`accumulate_centroid_sums_privatized_kernel`, the shipped arm at the fit's
shape, was a flat grid-stride over `n_rows * n_features` cells, one CELL
per thread step:

- per cell: one scalar 4-byte load of `x[gid]`, PLUS one label load and
  one weight load re-derived per cell (`labels[row]`, `weights[row]` --
  three loads per 4 bytes of X), then one shared-memory Int32 atomic.
- across a simdgroup the 32 consecutive `gid`s span exactly one row at
  d=32, so the reads are adjacent -- the pattern upstream's warp
  coalescing turns into full transactions on NVIDIA, and exactly the
  pattern the assignment kernel measured 3x slower than vector loads on
  this device.
- grid: 240 blocks x 128 threads (10 cores x 3072/128 from the matrix;
  Apple threadgroup memory is dynamically cached, so the 16 KB
  `PRIVATE_ACC_CELLS` allocation walls nothing and occupancy was NOT the
  lever -- `hardware_matrix.mojo:113-127`, measured in
  SCALING_2026-08-19.md. Verified before touching anything; the veclen
  story is the only structural lever found).

The weight kernel (`accumulate_weight_per_cluster_privatized_kernel`)
reads NO X: 8 bytes per row (weight + label), consecutive threads on
consecutive rows -- 1/16 of the sums kernel's traffic at d=32. Left
untouched. The direct scatter-add kernel is the upstream-faithful arm and
the checks' oracle; left verbatim (it can only be the oracle while it
stays scalar).

## The change (after)

The privatized sums kernel takes a comptime `veclen` and walks the same
flat cell order `veclen` cells per step: ONE `SIMD[float32, veclen]` X
load, ONE label and ONE weight load per chunk (the scalar body re-read
both per cell), then `veclen` shared atomics -- same multiset of adds.
`veclen = 1` is the old body verbatim and is the fallback arm.

Selection is REUSED, not duplicated: `launch_accumulate_centroid_sums`
routes on `fused_veclen_for` (the assignment port's transcription of
upstream's own 16/8/1-byte ladder, `fused_distance_nn-inl.cuh:107-110`),
fed x's base address for both pointer terms since this kernel reads one
matrix. The ladder's `4*n_features % 16 == 0` term doubles as the safety
contract: chunks never straddle a row and aligned in-bounds starts give
in-bounds ends, so no tail arm exists, exactly as in the assignment port.
Runtime `vl` meets the comptime parameter through
`_enqueue_privatized_sums[4|2|1]`, the `_launch_fused` shape.

## Upstream citation (the deviation)

`raft/cpp/include/raft/linalg/detail/reduce_rows_by_key.cuh:285`:
`SumsT val = d_A[j + lda * i];` -- one scalar element per thread in
`sum_rows_by_key_large_nkeys_kernel_rowmajor` (gid at `:280`, atomic at
`:287`); no `TxN_t`/`ldg` anywhere in that file or in
`reduce_cols_by_key.cuh`. Upstream loses nothing by this on NVIDIA
(warp-coalesced scalar reads are full transactions there); the premise
fails on Apple, where the identical swap on the assignment kernel
measured 63 -> 21 ms/iter (re-verdict ef0c4ba). Full pricing in
PORTING.md 46.

## Pricing

- Cost: one comptime parameter and a 3-arm dispatch ladder on machinery
  that already existed (`fused_veclen_for`); no new selection logic, no
  new tunables, no numerics change anywhere.
- Risk: none to the model -- all three arms are bit-identical to the
  scalar direct oracle (integer accumulator; vector loads return the same
  bits; per-lane quantization is the identical scalar fp32 expression in
  the identical order), and that is CHECKED per cell with `!=`, not
  argued.
- Reach: every arm is exercised -- veclen=4 pinned on the shipped-shape
  fixture, veclen=1 and 2 run through the real dispatch at d=33/d=34
  against the oracle with a nonzero-oracle guard, and the dropped-flush
  sabotage still registers on the vectorized body (the check-local
  sabotage copy mirrors the `veclen` parameter so it stays identical to
  the shipped kernel).

## Check output (build/kmeans_main, all green)

```
check_reach_by_sabotage OK: centroid_norm moved 384/512 labels; x_norm moved 512 distances and 0 labels, which is the predicted shape
check_kmeans_fit OK: 4/4 centroids matched as a permutation, 0/512 rows misassigned, inertia 170.703125 vs expected 171.19361649473004 (rel 0.0028651272446548032), 2 iterations
check_device_inclusive_scan OK: 20000 entries, worst relative error 0.0, past one block's worth
check_kmeans_plus_plus_init OK: 4/4 centroids recovered as a permutation through the k-means++ path, inertia 170.703125, 2 iterations
check_fused_reduction_across_lanes OK: 512 rows x 40 clusters match a host argmin, winners spread over 16 owner lanes
check_assignment_arm_dispatch OK: fused arm proved (0/32768 tile cells written; unfused sabotage overwrote 32768); arms agree on all labels, min_dist worst rel 0.0
  veclen=1 arm at d=33: 8256 labels correct (grid.y 120 of 129 tiles, 576 rows past the resident grid), fused == unfused
  veclen=2 arm at d=34: 8256 labels correct (grid.y 120 of 129 tiles, 576 rows past the resident grid), fused == unfused
check_fused_policy_dispatch OK: selection pinned (4/2/1 by k and alignment, skinny at k<32), bench alignment k=32 takes veclen 4 on real buffers, scalar and 2-wide arms correct through the launcher with the m grid-stride exercised
check_privatized_accumulate OK: 2048 sum cells + 64 weight cells bit-identical direct vs privatized, run-twice bitwise equal, dropped flush moved 352 cells, veclen=4 read arm pinned
  veclen=1 accumulate arm at d=33: 2112 sum cells + 64 weight cells bit-identical to the direct oracle (2112 nonzero)
  veclen=2 accumulate arm at d=34: 2176 sum cells + 64 weight cells bit-identical to the direct oracle (2176 nonzero)
check_accumulate_veclen_dispatch OK: scalar read arm reached-and-correct at d=33, 2-wide arm at d=34, 4-wide arm pinned in check_privatized_accumulate
```

(The dropped-flush count moved 128 -> 352 because the sabotage grid is now
sized in chunks, so block 0 owns a different slice of the fixture; the
sabotage registering is the assertion, the count is scheduling.)

## Suggested SCOREBOARD sentence (I do not edit the scoreboard)

> k-means accumulate now reads X `veclen`-wide through the same
> `fused_veclen_for` ladder as the assignment kernel (PORTING.md 46, a
> priced deviation beyond upstream's scalar reads: one SIMD chunk + one
> label/weight read per `veclen` cells), bit-identical on all three arms
> by check; awaiting re-timing -- the 54 ms/iter accumulate figure
> predates this.

## Commits (`git log -1 --format='%h parent %p'`)

```
ee05664 parent 5a62d4e
```
plus the report commit that follows it.

Files: `cluster/mojo_only/reduce_by_key.mojo`,
`cluster/mojo_only/kmeans_check.mojo`, `cluster/kmeans_main.mojo`,
`PORTING.md`.
