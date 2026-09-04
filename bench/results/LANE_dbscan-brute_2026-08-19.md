# LANE dbscan-brute, 2026-08-19

Verdict: **righter, and structurally much cheaper.** Six identified
divergences confirmed against upstream and fixed, plus three more found by
reading (D7). The distance step is now one fused kernel writing one byte per
pair instead of three kernels moving sixteen; the exclusive scan is
device-wide instead of one threadgroup; the CSR build has no barriers in its
inner loop instead of two per 256 columns; labels now match scikit-learn's
numbering. **Not measured** -- no timing benchmarks were run, per the brief.
Strongest remaining gap is complexity, not constant: this is still
`O(n^2)` and the ball-cover lane owns that.

---

## 1. DIVERGENCES FOUND

| # | What upstream does (file:line) | What ours did | Fixed? |
|---|---|---|---|
| D1 | `vertexdeg/algo.cuh:229` calls `epsUnexpL2SqNeighborhood`, a FUSED `Contractions_NT` tile kernel: `acc[i][j]` in registers (`epsilon_neighborhood.cuh:41`), `acc <= eps` in `epilog()` (`:106`), writes bool `adj` + reduces `vd` with `logicalWarpReduce`+`blockReduce`+atomics in the SAME kernel (`:137-160`). Never materializes a distance matrix. | `runner.mojo` ran `gemm_nt` (MAX matmul) into an `m x N` float32 `dist`, then `expand_distances_kernel` over it, then `eps_neighborhood_kernel` reading it back. 16 bytes/pair of traffic against their 1, three launches against one. | **FIXED.** New `dbscan/gbdt/neighbors/epsilon_neighborhood.mojo`. |
| D2 | `accumulate()` (`epsilon_neighborhood.cuh:129-130`) is `diff = regx - regy; acc += diff*diff` -- UNEXPANDED, straight from coordinates. | Expanded identity `\|\|x\|\|^2+\|\|y\|\|^2-2xy` through a GEMM. An ARITHMETIC change: cancels catastrophically in float32 when norms dominate distances (`archive/reference/PORTING.md 21`). | **FIXED.** Unexpanded. `row_norm_kernel`, `x_norm`, `x_alias`, `xn_alias` and `dist` all leave the DBSCAN path. |
| D3 | `adjgraph/algo.cuh:65` `thrust::exclusive_scan` -> `cub::DeviceScan`, device-wide, single pass with decoupled lookback. | `exclusive_scan_kernel` launched at `grid_dim=(1,1,1)`: ONE threadgroup scanning the whole array serially, once per batch, twice per fit. | **FIXED.** Three-launch scan-then-propagate (deviation 32). Verified at 2,000,000 entries / 977 blocks. |
| D4 | `runner.cuh:245-400`: two batch loops. Loop 1 REVERSED (`:247`, their comment copied verbatim) does vertexdeg -> read `vd[n_points]` -> corepoints. Loop 2 does vertexdeg (`i>0`) -> adjgraph -> `weak_cc_batched` into `labels` (`i==0`) or `labels_temp` -> `MergeLabels::run`. `adj_graph` is a LOCAL `rmm::device_uvector` resized to the largest batch (`:230`, `:317`). | One global `weak_cc` over a CSR built from every row, `MergeLabels` never ported, `col_ind` sized `N x N` by the caller. Correct, but gave back exactly the memory batching exists to save. | **FIXED.** `mergelabels/runner.mojo` + `label/merge_labels.mojo` ported; `col_ind` now allocated inside `dbscan_fit` for the largest batch, as theirs is. |
| D4b | *(brief said this "halves the arithmetic")* `runner.cuh` still recomputes vertexdeg for every batch except batch 0 (`:328-350`). | -- | **The brief's premise is wrong and it is worth recording why.** The second pass is unavoidable in EITHER structure: `weak_cc`'s `filter_op` reads `core[j]` for neighbours `j` in every other batch, so the whole core mask must exist before any batch is labelled. Upstream does `2*n_batches - 1` neighborhood passes; we now do the same, where we did `2*n_batches`. The saving is one pass out of `2*n_batches`, not a halving. |
| D5 | `dbscan.cuh:34` `compute_batch_size`: `est_mem_per_row = n_rows*sizeof(bool) + (neigh_per_row+2)*sizeof(Index_)`, `est_mem_fixed = n_rows*(sizeof(Index_)+sizeof(bool))`, budget = 80% of TOTAL device memory minus the dataset (`:157`), then the overflow guard `batch_size <= MAX_LABEL / n_rows`. | Not ported. `batch_size` was a hand-passed argument defaulting to "one batch", so `bench/bench_main.mojo` capped DBSCAN at 4,000 rows and `scaling_main.mojo` guessed 2048. | **FIXED.** `dbscan/gbdt/dbscan/dbscan.mojo`, including the overflow guard and `DeviceContext.get_memory_info()` in place of `cudaMemGetInfo`. Both bench harnesses now call `dbscan_fit_impl` and allocate nothing. |
| D6 | `runner.cuh:410-416`: `final_relabel` (`raft::label::make_monotonic`) then `relabelForSkl` (`MAX_LABEL -> -1`, else `--`). `dbscanFitImpl` hardcodes `algo_ccl = 2` (`dbscan.cuh:122`), so this is not optional. | Neither ported. `UNPORTED.tsv` excused it: "labels are arbitrary up to permutation, so the check compares the PARTITION". True of the check, false of the API. | **FIXED.** `label/classlabels.mojo` + `relabel_for_skl_kernel`. Check now asserts ids are exactly `{0,1,2}` and noise is `-1`. |
| D7a | `raft/sparse/detail/csr.cuh:61-63`: `weak_cc_label_device` indexes the CSR by `tid` and `labels`/`filter_op` by `global_id = tid + start_vertex_id`. `weak_cc_batched` re-runs `weak_cc_init_all_kernel` over ALL `N` on every batch (`:149`). | Ours used `tid` for both and had no `start_vertex_id`. It was only correct because it was handed one global CSR; it could not have consumed a batch-local one. | **FIXED** as part of D4. |
| D7b | `raft/sparse/convert/detail/adj_to_csr.cuh:67-109`: 512 threads, a SHARED per-row cursor bumped with `atomicIncWarp`, the row read in `chunk_size = 16` bool vector loads, output deliberately unordered. | 256 threads and a `block.prefix_sum` per 256-column window with TWO barriers per window, to produce ascending column order. At `n_cols = 200,000` that is 782 block scans and 1,564 barriers per row against their zero. The old README defended it as worth more than "their last increment of throughput". | **FIXED.** Shared cursor + `Atomic.fetch_add` + 16-bool chunk loads, unordered like theirs. Deviation 34 prices what is still missing (warp aggregation, multi-block-per-row grid). |
| D7c | `corepoints/compute.cuh:50`: `mask[idx + start_vertex_id] = vd[idx] >= min_pts` -- `vd` indexed by BATCH, `mask` by DATASET. | `core_points_kernel` indexed both by the dataset, because `vd` was a global `n_rows` array rather than their `batch_size + 1`. | **FIXED.** `vd` is now `batch_size + 1` with `vd[n_points]` holding the batch's edge count, which is what `runner.cuh:281` reads back. |
| D7d | `vertexdeg/algo.cuh:234-257` `sample_weight`: `coalescedReduction` over `adj` with `adj_ij ? sample_weight[j] : 0` produces a WEIGHTED degree that `CorePoints::compute` thresholds instead of the integer degree. | Absent entirely. Reaches cuML's public API as `DBSCAN(sample_weight=)`. | **NOT FIXED.** New row in `UNPORTED.tsv`; it is a self-contained feature and out of this lane's scope. |
| D7e | `vertexdeg/algo.cuh:186-223` Cosine arm; `vertexdeg/precomputed.cuh` (`algo_vd == 2`). | Absent. | **NOT FIXED.** Both now in `UNPORTED.tsv` with the recipe. |
| D7f | `epsUnexpL2SqNeighborhood:228` memsets `vd` for `m+1` elements before the kernel, because the kernel ACCUMULATES. | N/A (the old kernel assigned). | New precondition, satisfied by `enqueue_memset` in the host wrapper. Deviation 31. |

### MAX primitives standing in for something inside a kernel (per the rule change)

Audited every one in `dbscan/`. After this lane:

- **`gemm_nt` (MAX `linalg.matmul`) -- REMOVED from the DBSCAN path.** This
  was the one that mattered and it is exactly the failure the rule change
  describes: a device-wide vendor call cannot be fused, so the distance
  matrix had to be materialized, so two more kernels had to read it back.
  It survives only inside `check_fused_eps_agrees_with_materialized`, where
  materializing is the point.
- `max.gpu.primitives.block.sum` for `raft::blockReduce`, and
  `block.prefix_sum` for `cub::BlockScan::ExclusiveSum`. **Kept.** These are
  INTRA-kernel block collectives: they run inside the fused kernel and inside
  each scan block, and they neither read nor write global memory, so they
  cannot freeze an unfused structure. CUB's `BlockReduce`/`BlockScan` are the
  same shape of thing upstream.
- `std.gpu.primitives.warp.shuffle_xor` for `raft::logicalWarpReduce` and for
  `atomicIncWarp`'s `g.shfl`. Same argument, and the first is a literal port.
- `ctx.enqueue_memset` for `cudaMemsetAsync`. A host-side call in both.
- `nn.cumsum` -- **not used, and confirmed unusable.** Its installed signature
  is `cumsum[dtype, exclusive, reverse, *, axis](output, input)`: no
  `ctx: DeviceContext`, no `target`, unlike `nn.argsort`/`nn.topk`/`nn.gather`
  which carry both. There is no way to enqueue it. `archive/reference/VENDOR_LIBS.md` was
  already right about this.

---

## 2. WHAT I CHANGED, file by file

**New**

- `dbscan/gbdt/neighbors/epsilon_neighborhood.mojo` (~350 lines) --
  `raft/spatial/knn/detail/epsilon_neighborhood.cuh:31-238`. The fused kernel
  and its host wrapper. Policy4x4<float> constants restated locally rather
  than imported from `core/gemm.mojo`: a kernel whose correctness depends on
  `AccThCols == 16` (the `shuffle_xor` group width) must not read that 16 out
  of a file another lane is free to retune. Deviations 30, 31.
- `dbscan/gbdt/label/merge_labels.mojo` --
  `raft/label/detail/merge_labels.cuh:36-154`.
- `dbscan/gbdt/label/classlabels.mojo` --
  `raft/label/detail/classlabels.cuh:122-166`. Deviation 33.
- `dbscan/gbdt/dbscan/mergelabels/runner.mojo` --
  `cuml/cpp/src/dbscan/mergelabels/runner.cuh:37-47`.
- `dbscan/gbdt/dbscan/dbscan.mojo` -- `cuml/cpp/src/dbscan/dbscan.cuh:34-98`
  (`compute_batch_size`) and `:101-217` (`dbscanFitImpl`).

**Rewritten**

- `dbscan/gbdt/dbscan/runner.mojo` -- `runner.cuh:109-447`. Both batch
  loops, the reversed first one with their comment copied verbatim, the
  `vd[n_points]` readback at `:281`, `adj_graph` allocated locally at `:317`,
  per-batch `weak_cc` + `MergeLabels` at `:374-400`, `final_relabel` +
  `relabelForSkl` at `:410-416`. Signature changed: `x_norm`, `dist`,
  `x_alias`, `xn_alias`, `col_ind` OUT; `labels_temp`, `work_buffer`,
  `block_sums` IN.
- `dbscan/gbdt/dbscan/adjgraph/algo.mojo` -- `adjgraph/algo.cuh:50-70` and
  `raft/sparse/convert/detail/adj_to_csr.cuh:67-172`. Three-kernel device
  scan, and the compaction rebuilt on their shared cursor. Deviations 32, 34.
- `dbscan/gbdt/sparse/detail/csr.mojo` -- `raft/sparse/detail/csr.cuh:50-168`.
  `weak_cc_label_device` gets `start_vertex_id`/`batch_size`/`global_id`;
  `weak_cc_batched` added as a host function with the init inside it.
- `dbscan/gbdt/dbscan/vertexdeg/algo.mojo` -- `vertexdeg/algo.cuh:170-232`.
  `vertex_deg_run` is the L2 brute-force arm of their `launcher`. The old
  `eps_neighborhood_kernel` is retained and relabelled as the check oracle.
- `dbscan/gbdt/dbscan/corepoints/compute.mojo` -- `compute.cuh:38-52`.
  Batch/dataset index split, plus a `core_points_compute` host wrapper.
- `dbscan/mojo_only/dbscan_check.mojo` -- all four checks updated, one added.
- `dbscan/README.md`, `dbscan/PORTED_MAP.tsv`, `dbscan/UNPORTED.tsv` --
  rewritten; the false sentences are listed in section 5.

**Touched outside `dbscan/` (both DBSCAN-only edits)**

- `bench/scaling_main.mojo` `_dbscan_at` -- drops eleven hand-allocated
  buffers and the guessed `batch = 2048`, calls `dbscan_fit_impl`.
- `bench/bench_main.mojo` -- same, drops the `N x N` `db_d`/`db_adj`/`db_ci`
  allocations that capped `db_rows` at 4,000.

---

## 3. PROPOSED ROWS

Already written into `dbscan/PORTED_MAP.tsv` and `dbscan/UNPORTED.tsv`, which
this lane owns. Nothing needs merging by hand unless the root
`PORTED_MAP.tsv` mirrors section rows.

---

## 4. PROPOSED `archive/reference/PORTING.md` DEVIATION ENTRIES (numbered from 30)

**30. `logicalWarpReduce<AccThCols>` -> `shuffle_xor` butterfly.**
RAFT reduces the 16 threads sharing an output row with `shfl_xor` inside a
width-16 logical warp. Mojo's `shuffle_idx` has no width argument;
`shuffle_xor` needs none, because XOR with an offset below 16 only flips the
low lane bits and therefore stays inside the same aligned 16-lane group. The
reducer is integer addition, so no reduction shape can change the result.
Relies on `AccThCols` being a power of two, `acccolid = tid % AccThCols`, and
a lane width of at least `AccThCols`. (`epsilon_neighborhood.mojo`)

**31. `cudaMemsetAsync(vd, 0, ...)` -> `ctx.enqueue_memset`.** Same
operation, host side in both. Recorded only because the fused kernel
ACCUMULATES into `vd` where its predecessor assigned, so this is a new
precondition. (`epsilon_neighborhood.mojo`)

**32. `thrust::exclusive_scan` -> scan-then-propagate, three launches.**
Thrust dispatches to `cub::DeviceScan`, a single pass with decoupled
lookback. Lookback needs an inter-block forward-progress guarantee and a
device-scope acquire/release fence; Metal promises neither and Mojo exposes
no such fence, so a literal port can deadlock rather than run slowly.
Scan-then-propagate is the same algorithm with the inter-block communication
moved into a second launch. Costs one extra read+write of `batch_size`
integers. (`adjgraph/algo.mojo`)

**33. `getUniquelabels` (radix sort + `DeviceSelect::Unique`) -> mark-and-scan.**
`weak_cc` labels are `min(vertex index) + 1`, so every non-noise label is an
integer in `1..N`. Flagging occupancy and exclusive-scanning the flags
computes the same rank their sorted-unique array plus `map_label_kernel`'s
linear search computes, in `O(N)` instead of `O(N * n_clusters)`, with no
device sort and no stream compaction. `MAX_LABEL` is excluded exactly as
their `filter_op` excludes it. (`label/classlabels.mojo`)

**34. `atomicIncWarp` -> `Atomic.fetch_add`, and one block per CSR row.**
`cooperative_groups::coalesced_threads()` has no Mojo counterpart; a
fixed-mask reduction is not equivalent, because the point is that only the
lanes that took the branch participate. Their own comment prices this at up
to 32x the atomic traffic. Separately, `cudaOccupancyMaxActiveBlocksPerMulti-
processor` has no MAX counterpart, so we cannot assign several blocks to one
row when rows are scarce; `compute_batch_size` picks thousands of rows, so
that arm is unreachable in practice. (`adjgraph/algo.mojo`)

---

## 5. FALSE DOC SENTENCES FOUND

**Fixed in files I own** (`dbscan/README.md`, `dbscan/UNPORTED.tsv`, and the
module docstrings):
- "It was cheap because the expensive half already existed... DBSCAN's
  distance step is the same `core/gemm.mojo` plus `core/expand_distances.mojo`
  that k-means and k-NN use. It differs from k-NN only in what it does with
  the tile." **False, and it is the root cause of the 37x.** cuML shares
  nothing with its k-NN path here.
- "Labels are arbitrary up to permutation, so the check compares the
  PARTITION" as a reason not to port the relabelling. True of the check, false
  of the API.
- "Ours emits in ascending order... a diffable CSR is worth more here than
  their last increment of throughput." It was worth 1,564 barriers per row.
- `PORTED_MAP.tsv` header pinned cuML at `7e29955c` and RAFT at `9aa17e5`;
  the checkout is `00094f7` / `661a3b8`.
- `dbscan/UNPORTED.tsv`: "mergelabels/ ... only needed to combine per-batch
  labelings, and there is one batch" -- there were five in the batching check
  at the time it was written.

**In files I may NOT edit** -- for the orchestrator:
- `archive/reference/VENDOR_LIBRARIES.md` and `archive/reference/VENDOR_LIBS.md` list `linalg.matmul` as the
  route for DBSCAN's distance step. DBSCAN no longer calls it, and per the
  new rule it should never have.
- `archive/reference/PORTING.md` mentions DBSCAN's distance path; whatever it says about
  `gemm_nt` + `expand_distances` there is now false for this section.
- Root `README.md` / `RESUME.md` / `HANDOFF.md` claims that DBSCAN "was
  cheap because the distance step already existed" (the same sentence as the
  section README) should go if present.
- `core/gemm.mojo`'s docstring says its rewrite was motivated by "we lost
  k-means, PCA and DBSCAN to scikit-learn... and all three are dominated by
  this file". DBSCAN is no longer dominated by it, or by any part of it.

---

## 6. BUILD / CHECK EVIDENCE

```
$ tools/with_build_lock.sh pixi run --manifest-path .../mojotrees/pixi.toml \
    mojo build -I . dbscan/dbscan_main.mojo -o /tmp/dbscan_probe
$ /tmp/dbscan_probe
check_fused_eps_agrees_with_materialized OK: 31950 cells identical to the
  materialized path AND to a float64 host oracle (15298 of them neighbours,
  so the pattern is irregular), 150 degrees identical, vd[m] = 15298
check_dbscan OK: 3/3 blobs each one whole cluster with ids exactly {0, 1, 2},
  0 merges, 12/12 isolated points labelled -1 as scikit-learn does, converged
  in 2 propagation passes
check_dbscan_eps_sensitivity OK: eps=2 gives 3 clusters and eps=12 gives 1,
  which is impossible without the neighbourhood kernel running on real
  distances
check_exclusive_scan_beyond_the_old_cap OK: 2000000 entries exact across 977
  blocks, total 12000001
check_dbscan_batching_agrees OK: 5 batches with 4 merge_labels folds give
  labels identical to one batch, in an adj buffer 4x smaller than the
  unbatched run needs

$ ... mojo build -I . bench/scaling_main.mojo -o /tmp/scaling_probe   # clean
```

`bench/bench_main.mojo` does **not** build, and not because of this lane: it
fails in `glm/gbdt/linalg/detail/lstsq.mojo` with
`constraint failed: Wrong number of arguments to enqueue`, a file another lane
last touched at 18:08. My edit to its DBSCAN section compiles past.

### Scale probe (correctness, NOT a benchmark)

`scratchpad/dbscan_scale_probe.mojo`, 50,000 rows, 8 features, `eps = 1.20`,
`min_pts = 5`, budget forced to 512 MB so it does not compete with the other
lanes for memory:

```
batch@50k/512MB  = 2046
batch@200k/512MB = 510
batch@200k/64GB  = 10737     <- the MAX_LABEL/n_rows guard biting, as theirs does
n=50000 d=8 eps=1.20: labels in [0, 14], 1617 noise, 15 cluster ids,
  597 total propagation passes
```

25 batches, 24 `merge_labels` folds, monotonic ids starting at 0. Whole
process 1.6 s wall including device init and host fixture generation --
recorded only as "it is not minutes"; the old path was 2,666 ms for the FIT
ALONE at 50,000 rows and a sparser radius. **That is not a comparison. Nobody
should quote it.** Interleaved arms in one thermal window are the
orchestrator's to run.

Note: the CSR reorder changed the propagation pass count from 591 to 597 and
left the labels, cluster count and noise count identical -- which is the
expected signature of an unordered CSR. A min over a set does not depend on
visitation order; how fast it converges does.

### SABOTAGE (reach, five of five predicted shapes)

| # | Sabotage | Predicted | Observed |
|---|---|---|---|
| 1 | fused epilog row map `startx + i*AccThRows` -> `startx + i` (blocked, not strided) | placement scrambles while the total roughly holds | `15250 of 31950 adjacency cells disagree`, first mismatches at (1,0),(1,1),(1,2),(1,4) -- both directions, so it is a permutation not a threshold shift |
| 2 | scan pass 3 drops `+ add` | block 0's 2048 entries stay right, everything after is wrong | `1997952 of 2000000 ... wrong` = exactly `n - 2048` |
| 3 | `merge_labels_run` removed from loop 2 | batched and unbatched disagree | `399 of 612 labels differ between one batch and 5 batches` |
| 4 | `make_monotonic` removed | labels stay at `weak_cc`'s `min index + 1`, minus one | `cluster label 200 is outside 0..2; final_relabel did not run` |
| 5 | CSR remainder loop `j1 = n_cols % 16` -> `0` (drops 4 of 612 columns) | edges vanish, clustering changes | `noise point 608 has label 0` |

Every sabotage was reverted and the suite re-run green afterwards.

---

## 7. WHAT I DID NOT DO

- **`sample_weight`, the Cosine metric, and `Precomputed`.** All three are
  separate arms of `vertexdeg/algo.cuh` reachable from cuML's public API.
  Recipes are in `UNPORTED.tsv`. `sample_weight` is the one a user is most
  likely to miss.
- **`core_indices`** (`runner.cuh:419-442`). `thrust::copy_if` over the core
  mask. Optional output, `nullptr` in cuML's default path.
- **`atomicIncWarp`'s warp aggregation** and the multi-block-per-row CSR grid.
  Deviation 34 says why; the first needs `coalesced_threads()`, the second an
  occupancy calculator.
- **`Veclen = 4` vectorized loads and double buffering** in the fused kernel.
  Same two gaps `core/gemm.mojo` already records for the contraction. The
  second is blocked by Metal's 32 KB threadgroup ceiling at Policy4x4; the
  first is not blocked by anything and is the cheapest remaining win in the
  kernel. At `k = 8` the k-loop already runs once with 24 of 32 lanes zero,
  so `Policy4x4Skinny` (`Kblk = 8`, 8x8 threads,
  `contractions.cuh:146`) is worth a look for DBSCAN's shapes -- **but note
  RAFT does NOT select it here**, `epsUnexpL2SqNeighImpl` always instantiates
  `Policy4x4`. Changing that would be improving, not copying.
- **Any timing benchmark.** Per the brief.
- **Any commit.** Working tree only.
- **The `O(n^2)`.** Ball-cover lane.
