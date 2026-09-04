# Upstream GPU tree-learner survey — RF speed at 1M–2M rows (2026-09-01)

Written to answer Andrew's three questions: **what does LightGBM do, what
does cuML do, and are we exhausting what we can do?** Every claim below was
read at the pins this round, by this lane and its read-only survey agents:

- cuML `v26.08.00` (`265b9da`), `~/CascadeProjects/upstream/cuml-v26.08.00`
  — OUR upstream; `B` = `cpp/src/decisiontree/batched-levelalgo/builder.cuh`,
  `KI` = `.../kernels/builder_kernels_impl.cuh`.
- LightGBM `@3d1cf30`, `~/CascadeProjects/upstream/lightgbm` — the CUDA
  learner (`src/treelearner/cuda/`) plus the OpenCL learner
  (`gpu_tree_learner.*`, `ocl/histogram16.cl`).
- CatBoost `@7055d33d`, `~/CascadeProjects/upstream/catboost` — cross-
  pollination only (`catboost/cuda/methods/`, `greedy_subsets_searcher/`).

Scope discipline: the timed subject is `ensemble/bench/rf_bench.mojo` at
`rf@1000000` / `rf@2000000` (Andrew's 1M-row floor). Sub-1M behavior is
not cited and does not vote. Tier language follows the three-tier doctrine:
a technique can be IDENTICAL-tier (bit-identical in the identical numeric
mode) or FAST-tier-only (a mode-gated technique, clearly labeled).

## 0. Where the time goes today (what any candidate must attack)

Launch-log attribution at the large bench shape: `histogram_shared` is
**85.3% of device time**, `find_best_splits` 8.0%, nothing else over 2%
(`bench/results/RF_2026-08-22_attribution.md`). The histogram kernel's
per-element cost is: one coalesced `row_ids[i]` read, one **random-index
gather** (`bin_of(row, col)`, 1 byte since DEV 314), one label read
(sequential since DEV 2001), one shared-memory integer atomic — plus, per
block, a `histogram_len` zero-init and a `histogram_len` global atomic
flush. The gather probe priced random vs sequential access on the M4 at
**6.7–9.0x** (extratrees/DEVIATIONS.md ~:2470). So the levers, in order:
(1) visit fewer elements, (2) make the gather sequential-ish, (3) pay less
per block, (4) pay less per shared-atomic collision.

## 1. Technique inventory

Verdicts: **HAVE** (ported/landed), **N/A** (not applicable — reason
given, per tier where it differs), **CANDIDATE(n)** (ranked; 1 = biggest
expected win at 1M–2M).

| # | Technique | Who does it | What it buys at 1M+ | Do we have it | Verdict |
|---|---|---|---|---|---|
| 1 | **Histogram subtraction (parent − smaller sibling)** | LightGBM CUDA (`cuda_single_gpu_tree_learner.cpp:201-210`, `:382-384`; `SubtractHistogramKernel`, `cuda_histogram_constructor.cu:726-739`; pointer-swap parent reuse `cuda_data_partition.cu:833-841`); LightGBM CPU (`serial_tree_learner.cpp:368-387`); CatBoost both paths (`split_properties_helpers.cuh:96-104` picks the smaller sibling **inside the kernel**; `histogram_utils.cu:288-371`). **cuML does NOT** (`B:591` unconditional memset + `B:598` rebuild; `NodeWorkItem` has no parent/sibling field, `builder_kernels.cuh:36-40`; grep zero hits) | Build only the smaller child; larger = parent − smaller. Caps per-level histogram row-visits at ~N instead of ~2N — roughly **halves the 85% kernel** across the fat levels | **NO** — we mirror cuML's rebuild | **CANDIDATE(1)** — and for us it is EXACT (§3). Design written, implementation deferred to a dedicated session; DEV **2013 reserved** |
| 2 | Pre-binned feature matrix (uint8 bins, no per-element bin search) | LightGBM (bin matrix uploaded once, `cuda_row_data.cpp:45-152`); CatBoost (compressed index, `grid_policy.h`); **cuML does NOT** (`lower_bound` per element per node, `KI:341`, `builder_kernels.cuh:118-133`) | Removes ~7-step dependent search + shrinks the gather 4B→1B | **YES** — DEV 314 (`bin_dataset_kernel`), ≤256 bins, both smem arms | **HAVE** |
| 3 | Sequential label/stat streams (no `labels[row]` gather) | LightGBM & XGBoost regather via partition indices; CatBoost moves stat columns | Kills one random gather per element | **YES** — DEV 2001 (`labels_s` re-permuted with `row_ids`), default flipped 2026-09-01 on the 1M/2M win | **HAVE** |
| 4 | **Ascending row ids at every depth** (sorted sample / contiguous leaf indices) | LightGBM: bagging emits ascending indices and the partition is stable-contiguous (`cuda_data_partition.cu:917-943`), so hist gathers walk ascending ids for the whole tree; CatBoost: goes further, pre-gathers bins into contiguous per-leaf buffers (`gather_bins.cu:11-110`, chosen at `split_properties_helper.cpp:1338-1340`). **cuML does NOT**: bootstrap draws are raw unsorted `uniformInt` (`randomforest.cuh:140-143`) and `row_ids` is never re-sorted, while the stable partition **preserves that random order to every depth** | Turns the dominant gather from random (6.7–9.0x penalty on this box) into an ascending gapped walk, at all depths, for one cheap per-tree sort | **NO** (until today) | **CANDIDATE(2)** — **IMPLEMENTED this round as DEV 2010**, flag `ROWS_SORTED_SAMPLE`, OFF |
| 5 | **Items-per-thread row blocking** in the histogram kernel | LightGBM CUDA: ~400 rows/thread (`NUM_DATA_PER_THREAD=400`, `cuda_histogram_constructor.hpp:22`); CatBoost: unroll-trait loops + `uint4` loads; **cuML does NOT**: one CTA per 128 rows (`B:398-399`), grid-stride loop runs ~1 iteration/thread, no `ITEMS_PER_THREAD` anywhere (grep zero hits) | R× fewer blocks per node ⇒ R× fewer per-block zero-inits and R× fewer global flush atomics (root at 1M ≈ 7.8k blocks × 512 cells today), plus a real loop to overlap gather latency | **NO** (until today) | **CANDIDATE(3)** — **IMPLEMENTED as DEV 2011**, flag `HIST_ITEMS_PER_THREAD` (candidate arm 4), OFF |
| 6 | **Privatized shared sub-histograms** (per-warp / sub-warp copies) | CatBoost: per-warp hists + sub-warp replication (`hist_one_byte.cu:22-24`, `:56-61`) + lane-rotated writes; LightGBM OpenCL: `NUM_BANKS=8` replicated copies (`histogram16.cl:26-32`) + grad/hess-first swapping; LightGBM CUDA: one copy per block but bounded contention by thread mapping; **cuML: one shared copy, all threads atomic into it** (`KI:322-333`) | Divides shared-atomic serialization depth on hot (bin,class) cells by P; compounds with #5 (fatter blocks = more contention) | **NO** (until today) | **CANDIDATE(4)** — **IMPLEMENTED as DEV 2012**, `SMEM_COPIES` (candidate arm 4), OFF |
| 7 | Quantized/discretized gradients (int16 packed grad+hess, per-leaf hist bit-width) | LightGBM CUDA (`cuda_gradient_discretizer.cu`; bit-width per leaf, `gradient_discretizer.cpp:165-200`) | Halves stat traffic, packs two stats into one atomic | N/A — RF has **no gradients**; our stats are already integer counts (classification) and fixed-point Int32 (regression, DEV 101), i.e. we already hold the endpoint this technique approximates | **N/A (both tiers)** — cuML likewise has none |
| 8 | Packed multi-feature bin words (4 features/`ui32`, 4-bit bins) | CatBoost (`grid_policy.h:8-48`); LightGBM 4-bit dense bins (`dense_bin.hpp:19`) + EFB GPU cap 256 (`dataset.cpp:142`) | Fewer bytes per gather; several features per load | Partial — DEV 314 is 1 byte/feature, unpacked | CANDIDATE(5), low: at 128 bins (our bench default) 8 bits/bin is already minimal; 4-bit packing needs `max_n_bins<=16`, a different operating point. Not pursued this round |
| 9 | EFB / exclusive feature bundling, sparse-aware histograms | LightGBM (`dataset.cpp:112+`) | Shrinks effective feature count on sparse data | NO | **N/A** — dense-only library today (cuML's Dataset is dense too, `dataset.h:15-45`); becomes relevant only with a sparse input surface (out of lane scope) |
| 10 | Leaf-wise (best-first) growth | LightGBM CUDA (`FindBestFromAllSplitsKernel`, `cuda_best_split_finder.cu:2142-2166`) | Reaches a loss with fewer leaves; smaller per-split row sets | NO — we mirror cuML's batched level algorithm | **N/A as a port** (it is a different learner, the surface LightGBM's CUDA learner exists to provide; PLAN.md already declined porting that learner). The lossguide/extratrees lanes own best-first shapes. NOTE: for RF **parity with cuML**, level-order IS the product |
| 11 | Launch amortization: fused setup, cached args, packed phase uploads, batched readbacks | LightGBM batches its 18-int/8-int D2H metadata (`cuda_data_partition.cu:1040-1057`); CatBoost batches zero/scan/subtract so launch count is dataset-independent (`compute_by_blocks_helper.h:87-92`) | Kills the per-batch host tax (our 94-launch/tree small-tree tax) | **YES** — DEV 1908/1909/1916/1917/1918/1919 family | **HAVE** (this vein is close to mined out; 1918's deliberate skips record why the rest stays) |
| 12 | Tree-level stream/pipeline concurrency | cuML `n_streams=4` OpenMP+streams (`randomforest.cuh:336-366`); LightGBM has streams but defeats them with ~8 device syncs/split (`cuda_best_split_finder.cu:1799` etc.) | Hides per-batch host round-trips | **YES** — DEV 117 K-way pipelining on one queue | **HAVE** |
| 13 | Occupancy tuning: `__launch_bounds__`, block-size sweeps | LightGBM 504-thread blocks + `min_grid_dim_y=160`; CatBoost `__launch_bounds__(384,2)`; **cuML: none** (grep zero hits), TPB=128 everywhere | Register/occupancy headroom | Partial — DEV 103a tiers built but DISABLED (A/B voided at canary 10.5x); TPB is cuML's 128 | CANDIDATE(6), low — the tier A/B is already built and queued; TPB sweeps are cheap but the workload map couples TPB to the partition scan, so a TPB flag is NOT a table-only change (unlike #5). Not pursued this round |
| 14 | Coalescing layout: row-major-within-partition (hist) + column-major (partition), two copies | LightGBM keeps the bin matrix in TWO layouts simultaneously (`cuda_row_data.cpp:274-297` vs `CUDAColumnData`) | Coalesced along the axis each phase walks | Partial — our binned matrix mirrors the input strides (col-major default = coalesced per column already) | **N/A mostly** — with col-major input + DEV 314's 1-byte cells + DEV 2010's ascending ids, the remaining gap to LightGBM's layout is small; a second layout costs 2x bin-matrix memory. Revisit only if 2010's A/B says locality is still the wall |
| 15 | Most-frequent-bin skip + FixHistogram reconstruction | LightGBM (`FixHistogramKernel`, `cuda_histogram_constructor.cu:741-768`) | Skips the densest bin's atomics; rebuilds it as leaf_sum − rest | NO | **N/A (identical tier)** — reconstruction reorders float accumulation in their design; in OUR integer bins it would be exact, but it only pays beside subtraction (#1) and EFB-style bin packing; folded into the #1 design note |
| 16 | Stochastic rounding, quantized-leaf refit | LightGBM (`RenewDiscretizedTreeLeaves`) | Accuracy recovery for quantized training | — | **N/A** — no quantization here (see #7) |
| 17 | Retry-round feature resampling cost (cuML's `B:419-457` multiplies hist work up to n_cols/max_features on pure nodes) | cuML-specific liability, not a technique | — | We transcribe it (rounds + retry compaction) | HAVE (transcribed); at the bench's `max_features=1.0` max_rounds=1 so it cannot bite there. A candidate to CAP retries would change outputs (fewer split candidates) — **N/A identical tier**, possible FAST-tier knob, unranked |
| 18 | Per-column real bin counts (dedup shrinks search + hist) | cuML (`quantiles.cuh:104-107`) | Smaller hists on low-cardinality columns | **YES** — quantiles.mojo | HAVE |
| 19 | Host-side pruning: min_samples/max_depth/max_leaves before enqueue | cuML `B:82-88` | Unsplittable nodes cost zero GPU work | **YES** — NodeQueue | HAVE |
| 20 | CUDA-graph-style whole-level capture | Nobody (LightGBM's most obvious unexploited win; grep `cudaGraph` = 0 there) | Removes per-launch host cost wholesale | NO | **N/A here** — no graph/indirect-command surface exposed by MAX on our targets today; if one appears, the 1908/1916 family is the shovel-ready consumer. Out-of-lane (toolchain) |

Sabotage/reach, identity gates and A/B commands for the three implemented
candidates are in `archive/plans/ensemble/PLAN.md` ("2026-09-01 candidate round") and in
each DEVIATION block (2010 in `randomforest.mojo`, 2011 in `builder.mojo`,
2012 in `kernels/builder_kernels_impl.mojo`), with
`ensemble/checks/rf_perf_candidates_check.mojo` as the mechanism check.
All three are **flag-guarded, OFF by default, and UNVERIFIED — every run
owed to the orchestrator.**

## 2. Ranked candidates (expected win at rf@1000000 / rf@2000000)

1. **DEV 2013 (reserved, unimplemented): histogram subtraction.** Halves
   the fat-level row visits of the 85% kernel; the one big algorithmic
   lever every competitor has and cuML lacks. See §3 for the design and
   why it is exact for us. **A dedicated session**, not a footnote.
2. **DEV 2010 (implemented, off): sorted sampled rows.** Attacks the
   6.7–9.0x random-gather penalty at every depth of every tree for the
   price of one ~2·log2(n) pass device sort per tree. Identity-free win
   candidate: pure locality.
3. **DEV 2011 (implemented, off): histogram items-per-thread (R=4).**
   Divides per-block zero-init and global-flush atomics by 4 and gives
   the grid-stride loop real iterations; costs the 1919 reuse (priced
   inside the A/B).
4. **DEV 2012 (implemented, off): privatized sub-histograms (P=4).**
   Divides hot-cell shared-atomic serialization by 4; the win depends on
   how contended Apple's threadgroup atomics actually are — exactly what
   the flag exists to measure. Compounds with 2011.

Interaction note for the measurement plan: 2010/2011/2012 are independent
flags. Measure each alone against baseline first (attribution per
mechanism), then the best-of composite; 2011+2012 are the natural pair.

## 3. The subtraction design (CANDIDATE 1, DEV 2013 reserved — not built)

Why it is EXACT here where LightGBM needs `max(0,·)` clamps and a refit:
our histogram cells are integer counts (classification) and fixed-point
Int32 (regression/weighted, DEV 101). The partition is exact
(`left ⊎ right = parent` as multisets), so
`hist_parent[cell] = hist_left[cell] + hist_right[cell]` holds in integer
arithmetic and `hist_larger = hist_parent − hist_smaller` is bit-identical
to a rebuild — **IDENTICAL-tier eligible**, unlike the float designs it
mirrors (CatBoost clamps stat 0 at `histogram_utils.cu:306-308` because
theirs is float; ours would not).

Sketch (constraints discovered this round, so the next session starts at
the design, not the archaeology):

- **Sibling pairing is free**: children are pushed adjacently
  (`RightChildId == LeftChildId + 1`), and the smaller side is known on
  the host at push time (`local_nLeft` vs `count − local_nLeft`) —
  LightGBM's `:382-384` rule, host-side.
- **Column matching is the real constraint.** Histograms are stored per
  (batch-node, round-column-slot) and the per-node feature sample is a
  per-node permutation. Parent and child histograms line up by COLUMN ID
  only when the sampled sets coincide — guaranteed at
  `max_features = 1.0` (the bench's own setting, and cuML's regression
  default); NOT at `sqrt`. So v1 is gated to the all-columns
  configuration, with the arena indexed by column id (a persist-copy
  kernel scattering `(nid, round-slot) → (arena-slot, col)`).
- **Lifetime**: a parent's round-workspace histogram is overwritten by
  the next round/batch, so nodes that split must have their histograms
  PERSISTED to an arena before the batch retires, and children must be
  processed while the arena entry lives. Under FIFO batching, children
  of batch B land in batch B+1 when levels fit `max_batch_size`; the v1
  rule is "subtract only when the parent's arena entry is live and the
  batch did not straddle" — else rebuild (always-correct fallback).
- **Memory**: arena = arena_nodes × n_cols × max_n_bins × n_classes ×
  size_of[Bin]. At 50 cols × 128 bins × 4 classes × 4B = 100KB/node;
  cap arena_nodes (e.g. 512 ⇒ 50MB) and rebuild past the cap — the
  capped levels (≤2^9 nodes) are exactly the fat levels where
  subtraction pays, so the cap costs little.
- **Retry rounds**: subtraction arm bypassed for retry rounds (they
  sample DIFFERENT columns by construction).
- **Gate**: fingerprints unchanged (subtraction is exact), a planted
  parent/sibling check per cell, and a sabotage that subtracts the WRONG
  sibling (must move cells in the larger child only).

## 4. Proposals outside this lane's ownership (not edited, per charter)

- **`core/segmented_sort.mojo`**: a multi-bit (radix-8) pass variant and
  a `keys-only, bit-bounded` entry would cut DEV 2010's pass count 8x
  and delete the duplicated driver in `randomforest.mojo`; also the
  serial `seg_scan_block_sums_kernel` (1 thread) becomes the sort's
  latency floor at 2M rows — a block-scan version is a core/ candidate.
- **`pixi.toml`**: still no task for any `ensemble/` check (re-checked
  2026-09-01); `check-rf-perf-candidates` belongs beside `check-if`
  when a slot opens.
- **Toolchain watch**: any MAX surface for command-buffer/graph capture
  or device-side enqueue directly serves items #11/#20.

## 5. Are we exhausting what we can do? (the direct answer)

**No — but the remaining headroom is now named and mostly staged.** The
launch-amortization vein (1908/1909/1916–1919) and the memory-shape vein
(314 binned matrix, 2001 sampled-order labels) are substantially mined:
on those axes we already do what cuML does, and in several places what
LightGBM does and cuML does not. What remains, in order of size:

1. **Histogram subtraction** — the one big algorithmic technique every
   maintained GPU tree learner has and we (mirroring cuML) do not. For
   our integer bins it is exact, so it is not even a tier trade-off.
   Until it lands, "exhausted" is false.
2. **Gather locality (2010) and block-shape economics (2011/2012)** —
   implemented today, off by default, each a measured question now
   rather than an open idea.
3. After those: the disabled 103a smem tiers (A/B still owed on a quiet
   box), and diminishing-return items (#8, #13, #14) that only earn a
   look if the A/Bs above say the wall is still where we think it is.

What we should NOT do: port LightGBM's leaf-wise learner (a different
product, already declined in PLAN.md), fake gradient quantization onto a
gradient-free algorithm, or let any of the above reorder float
accumulation into the identical arm — the three implemented candidates
were chosen precisely because integer/fixed-point accumulation makes them
tier-free.
