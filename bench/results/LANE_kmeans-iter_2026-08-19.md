# LANE kmeans-iter: assignment dispatch audited (already fused), accumulation privatized

2026-08-19/20. Scope: SCOREBOARD item 1's two phase brackets, km.assignment
(63 ms) and km.accumulate (56 ms), at 4M x 32, k=64. No timing was run in
this lane; the orchestrator times.

## A. Assignment: their dispatch, and which arm our fit takes

### Their dispatch conditions, file:line

- `cuvs/cpp/src/cluster/detail/kmeans_common.cuh:378-379` -- the WHOLE
  selector: `bool is_fused = metric == L2Expanded || metric ==
  L2SqrtExpanded;`. A metric test with nothing else in it: no shape term,
  no hardware term.
- `:380` -- on the fused arm `dataBatchSize = n_samples` (tiling is the
  other arm's property; `detail/kmeans.cuh:837-846` logs that
  `batch_samples` is ignored).
- `:383-389` -- fused arm resizes `L2NormBuf_OR_DistBuf` to `n_clusters`
  and rowNorms the centroids into it.
- `:430-449` -- fused arm calls `fusedDistanceNNMinReduce`; **no distance
  tile is ever materialized**.
- `:450-491` -- unfused arm: `pairwise_distance_kmeans` into the tile, then
  `coalescedReduction` with `raft::argmin_op` (`:474-489`).
- Metrics other than the two L2s never reach this dispatch in k-means:
  `pairwise_distance_kmeans` RAFT_FAILs on them (`kmeans_common.cuh:
  305-321`), and our `KMeansParams.validate()` ports that refusal.

### Tie rule, verified in their source

- Their FUSED comparator `KVPMinReduceImpl`
  (`cuvs/cpp/src/distance/detail/fused_distance_nn/helper_structs.cuh:
  39-44`) is `b.value < a.value ? b : a` -- VALUE ONLY; on a tie the
  incumbent wins, so their fused answer is reduction-shape dependent.
  `MinAndDistanceReduceOpImpl` (`:47-97`) is value-only strict `<` too.
  **The brief's premise "their kvp min picks lowest index" is true only of
  their UNFUSED arm**: `raft::argmin_op`
  (`raft/cpp/include/raft/core/operators.hpp:198-205`) breaks value ties
  toward the LOWER key.
- Our port uses `argmin_op`'s total order in BOTH arms (documented in
  `simt_kernel.mojo`, archive/reference/PORTING.md 14): that is what makes the two arms
  diffable and the fused answer reduction-shape independent. Unchanged by
  this lane.

### Which arm our fit takes: FUSED, before and after. No dispatch fix needed.

- `cluster/gbdt/cluster/detail/kmeans.mojo:606` (Lloyd loop), `:292`
  (init path) and `:755` (final inertia assignment) all call
  `min_cluster_and_distance_compute`
  (`cluster/gbdt/cluster/detail/min_cluster_distance_compute.mojo`),
  whose body enqueues `fused_distance_nn_kernel` unconditionally. Since
  only the two L2 metrics pass `validate()`, "unconditionally" coincides
  with their `is_fused` on every reachable input. Nothing in the tree calls
  `min_cluster_and_distance_compute_unfused` except the new check.
- Proved by sentinel, not by reading: `check_assignment_arm_dispatch`
  poisons `dist_buf` (the only buffer the unfused arm writes that the fused
  arm does not), runs the fit's entry at d=32, k=64, L2Expanded (rows
  scaled to 512; rows appear in neither selector), and 0/32,768 tile cells
  moved. The sabotage half runs the unfused arm on the same inputs and it
  overwrote all 32,768 -- so the sentinel demonstrably CAN register a
  materializing arm. Arms agree on all 512 labels; min_dist worst rel 0.0.

### Consequence the orchestrator should know (SCOREBOARD correction needed)

SCOREBOARD_2026-08-19 item 1 says "Assignment streams a materialized
distance tile where cuVS's k-means dispatches to its FUSED L2-NN kernel",
and commit 692834c's subject says "(unfused dispatch)". **Both are
falsified by the sentinel check**: the fused arm was already wired and
taken at the measured commit (`b8b604d` landed it). The 63 ms bracket is
the FUSED SIMT kernel's own efficiency (~16.4 GFLOP of dot products per
iteration; 63 ms is ~260 GFLOP/s effective against this box's multi-TFLOP
peak, with x read once at ~512 MB), not a dispatch miss. This lane was
barred from editing the SCOREBOARD; the false sentence is the
orchestrator's to delete. Assignment's next lever is kernel arithmetic
efficiency (per-thread tile work, vectorized smem loads), not dispatch.

## B. Accumulation: privatized into threadgroup memory

### Their dispatch, file:line

- Centroid sums: `cuvs/cpp/src/cluster/detail/kmeans.cuh:300-309` calls
  `raft::linalg::reduce_rows_by_key`. Its dispatch
  (`raft/cpp/include/raft/linalg/detail/reduce_rows_by_key.cuh:354-363`)
  gates the small-nkeys kernel at `nkeys <= 4`; at nkeys=64 it takes
  `sum_rows_by_key_large_nkeys_kernel_rowmajor` (`:272-288`) -- the direct
  one-global-atomic-per-cell kernel our
  `accumulate_centroid_sums_kernel` mirrors, launched one thread per cell
  (`:304-307`).
- Weights: `detail/kmeans.cuh:312-318` calls `reduce_cols_by_key` with
  nrows=1, ncols=n_samples, nkeys=k. Its dispatch
  (`reduce_cols_by_key.cuh:125-139`) takes the CACHED shared-memory kernel
  (`:51-81`) whenever `cache_size <= 49152 && nrows*ncols >= 8192` -- true
  at 4M x k=64 -- with grid `min(4 * numSMs, ceildiv(work, 256))`.

So: for the WEIGHT reduction, privatization IS the arm their dispatch
takes, and porting it removes a deviation. For the SUMS reduction their
taken arm is the direct scatter-add we already had, measured here at 56 ms
against a 15-20 ms traffic floor (128M global atomics into 2,048 cells).
The privatized sums kernel is therefore a recorded DEVIATION from their
taken arm -- but its STRUCTURE is still theirs, lifted from the two
privatized kernels RAFT ships in the same files
(`sum_rows_by_key_large_nkeys_kernel_colmajor`,
`reduce_rows_by_key.cuh:196-242`, and `reduce_cols_by_key_cached_kernel`,
`reduce_cols_by_key.cuh:51-81`): zero a `__shared__` cache, accumulate with
shared-memory atomics, flush non-zero cells to global atomics once per
block.

### Design (cluster/mojo_only/reduce_by_key.mojo)

- `accumulate_centroid_sums_privatized_kernel` /
  `accumulate_weight_per_cluster_privatized_kernel`: k*d (resp. k) Int32
  partials in threadgroup memory, flat grid-stride accumulate, one flush
  per block skipping zero cells (their `:235` / `:78`). At the fit's shape:
  8 KB of the comptime 16 KB allocation used; global atomics drop from
  128M to at most grid x 2,048 (~0.5M at the 240-block grid), the 128M land
  on threadgroup memory instead.
- Dispatch `launch_accumulate_centroid_sums` /
  `launch_accumulate_weight_per_cluster` mirrors
  `reduce_cols_by_key.cuh:125-139`'s shape: privatized iff the cache fits
  (`k*d <= PRIVATE_ACC_CELLS`) and the input is large (`work >= 8192`,
  their constant); direct kernel otherwise. Their 49,152-byte guard is the
  full CUDA default shared budget; ours is pinned at comptime (Mojo
  `stack_allocation` cannot size dynamically) to HALF the target column's
  threadgroup limit, read from `kernel_matrix.column_shared_limit` --
  4,096 Int32 cells / 16 KB on Apple's 32 KB.
- Grid sizing: the magic `min(1024, ...)` cap at the launch site is GONE.
  `accumulate_grid_blocks` computes `min(gpu_cores_for[TARGET_COLUMN] *
  max_active_blocks_for[TARGET_COLUMN](TPB, smem), ceildiv(work, TPB))` --
  RAFT's own rule shape (`target_nblks = 4 * getMultiProcessorCount()`,
  `reduce_cols_by_key.cuh:127-131`) with their hardcoded 4-blocks-per-SM
  guess replaced by the hardware matrix's occupancy computation (Apple: 10
  cores x 24 = 240 blocks). Their direct sums arm launches one thread per
  cell with no cap; our direct arm keeps its grid-stride launch-shape
  deviation, now sized from the same matrix (smem=0), as recorded in the
  function's docstring.

### Determinism argument (also in the kernel comment)

The addends are the identical quantized Int32 values the direct kernel
forms. Int32 addition is associative and commutative, so grouping them into
per-block partials and atomically adding the partials is a re-association
of the same multiset of adds: totals are BIT-IDENTICAL to the direct
kernel's and to themselves run to run, at any grid size, under any
scheduling. The k-means fixed-point accumulator keeps its property intact.
(Verified, not just argued: see the check below -- 2,048 + 64 cells
bitwise-equal across arms and across runs.)

Because the arms are bit-identical, the dispatch is SCHEDULING: it can
change the time, never the model. Per the switches-must-flip rule the
privatized default lands in this same session.

## Checks (cluster/kmeans_main.mojo, all seven)

New: `check_assignment_arm_dispatch` (sentinel/poison arm proof, above) and
`check_privatized_accumulate` (hashed values, hashed SCATTERED and SKEWED
labels -- every third row forced onto cluster 7 -- non-unit weights;
direct-vs-privatized bitwise equality on Int32; run-twice bitwise
determinism; check-local dropped-flush sabotage kernel that must move the
totals). Output, this tree:

```
check_reach_by_sabotage OK: centroid_norm moved 384/512 labels; x_norm moved 512 distances and 0 labels, which is the predicted shape
check_kmeans_fit OK: 4/4 centroids matched as a permutation, 0/512 rows misassigned, inertia 170.703125 vs expected 171.19361649473004 (rel 0.0028651272446548032), 2 iterations
check_device_inclusive_scan OK: 20000 entries, worst relative error 0.0, past one block's worth
check_kmeans_plus_plus_init OK: 4/4 centroids recovered as a permutation through the k-means++ path, inertia 170.703125, 2 iterations
check_fused_reduction_across_lanes OK: 512 rows x 40 clusters match a host argmin, winners spread over 10 lane groups
check_assignment_arm_dispatch OK: fused arm proved (0/32768 tile cells written; unfused sabotage overwrote 32768); arms agree on all labels, min_dist worst rel 0.0
check_privatized_accumulate OK: 2048 sum cells + 64 weight cells bit-identical direct vs privatized, run-twice bitwise equal, dropped flush moved 128 cells
```

Also probed before designing around it: threadgroup-memory Int32
`Atomic.fetch_add` works on this M4 (256 threads into 8 shared cells, all
exact).

## What the orchestrator should time

- The kmeans 4M x 32, k=64, 20-iter bench arm (`bench/bench_main.mojo`,
  unchanged by this lane), plus the PHASE km.assignment / km.accumulate
  brackets (orchestrator's harness).
- Expected: km.accumulate moves from 56 ms toward the 15-20 ms traffic
  bound (global atomic count drops ~260x); km.assignment is UNCHANGED --
  no assignment code moved, and its 63 ms was never a dispatch miss (see
  A). Totals are bit-identical, so any output digest must not move.
- Doc debt for the orchestrator: delete SCOREBOARD item 1's "streams a
  materialized distance tile" sentence (falsified above; this lane was
  barred from that file), and fold the sums-privatization deviation into
  archive/reference/PORTING.md's numbered list (archive/reference/PORTING.md was mid-edit by a peer session, so
  the deviation is recorded in the module docstrings and here instead).

## Files

- `cluster/mojo_only/reduce_by_key.mojo` -- privatized kernels, dispatch,
  matrix-sized grids, corrected upstream citation on the direct weight
  kernel.
- `cluster/gbdt/cluster/detail/kmeans.mojo` -- Lloyd loop calls the
  dispatch helpers; magic 1024 cap deleted.
- `cluster/mojo_only/kmeans_check.mojo`, `cluster/kmeans_main.mojo` -- the
  two new checks.
