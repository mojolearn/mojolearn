# Upstream survey, September 2026: what cuML, LightGBM and sklearn do for tree-training throughput at 1M+ rows that this lane does not

Written 2026-09-01 by the ET perf-research lane (write-only: this lane ran
nothing; every verdict below that needs a number says RUN OWED and the exact
commands live in the deviation blocks and `PLAN.md`). Sources, and nothing
else: this repository's ledgers, and the pinned mirrors
`~/CascadeProjects/upstream/cuml-v26.08.00` (the v26.08 pin, NOT
`upstream/cuml`), `~/CascadeProjects/upstream/lightgbm` (3d1cf30),
`~/CascadeProjects/upstream/scikit-learn` (1.9.0). File/line citations were
read out of those mirrors this session.

## Why the bottleneck profile is not RF's, restated once

Extremely-randomized splits draw ONE threshold per (node, feature) inside the
node's own range -- no exact-split search, no bins, no histogram (DEVIATION
137). What remains at 1M-2M rows is: two scattered-gather passes over the
node's rows per level (range, then score -- the threshold cannot be drawn
before the range is known), a partition pass, and the per-block fixed costs
(collectives, publish atomics, workgroup dispatch). The lane's own measured
board: Apple M4 1M higgs = score 47% / range 28% / partition 15%
(`bench/results/et_profile/APPLE_M4_2026-09-01.md`), memory-LATENCY bound
within ~1.5x of the gather roofline (DEVIATION 208); MI325X = workgroup
DISPATCH-RATE bound (DEVIATION 1943: halving workgroups halved both hot
passes, twice); H100 = 4160 ms whole fit at 1M, bound unprofiled. Threshold
sampling itself is noise (a PCG draw per (node, feature) cell, recomputed
rather than stored -- DEVIATION 170's finalize).

The lane's standing lever-class rule (DEVIATION 212, 3-for-3 vs 0-for-2):
levers that REMOVE work or launches pay; levers that REARRANGE the same row
traffic wash. Every verdict below is scored against that rule.

## The technique table

Tiers: everything marked IMPLEMENTED or CANDIDATE here is **identical-tier
safe** (bit-free by integer/key-space accumulation) unless the row says
otherwise; no fast-tier-only candidate survived this survey (the one
fast-shaped idea, drawing from the parent's range to fuse the two passes, is
refused in DEVIATION 137 itself: it changes the algorithm, not the tier).

| # | technique | who | what it buys at 1M+ rows | have it? | verdict |
|---|---|---|---|---|---|
| 1 | Multi-row threads (real items-per-thread): tile `ceildiv(rows, TPB*R)` blocks per node so each thread folds R rows | LightGBM (`cuda_histogram_constructor.cu:30-32`, `NUM_DATA_PER_THREAD=400`); cuML does NOT have it at either pin (v26.08 `builder.cuh:393-408` is still one row/thread) | Hot-pass workgroup count /R at shallow levels -- the exact bound DEV 1943 measured on CDNA; collectives and publish atomics amortized xR everywhere | NO (we transcribed cuML's degenerate stride) | **CANDIDATE, RANK 1 -- IMPLEMENTED as DEVIATION 2020**, flag-guarded `MOJOLEARN_ET_SEARCH_RPT_{2,4,8,16}`, off by default. Both tiers, bit-identical (key-order per-thread range fold). RUNS OWED |
| 2 | Block width keyed on wavefront width; sweep to the CDNA max | cuML fixed 128; ours since DEV 1943 | Measured on MI325X: 17002 -> 6207 ms hot-kernel time (128 -> 512) | HAVE (1943) | HAVE; the owed past-512 sweep now has its arm (`MOJOLEARN_ET_TPB_1024`, 1943 AMENDED). RUN OWED, AMD leg |
| 3 | Per-thread accumulator width: don't size every fit's private arrays for 32 classes | cuML sizes its smem histogram at RUNTIME per launch (`extern __shared__`), so it never pays this; our comptime 32 is DEV 172's constant | 256 B/thread scratch -> 32 B at `_4`: occupancy divisor lifted, in the latency-bound regime (Apple) where residency hides gathers | PARTIAL (fixed 32) | **CANDIDATE, RANK 2 -- IMPLEMENTED as DEVIATION 2021**, `MOJOLEARN_ET_MAX_ACC_{4,8,16}`, off by default, loud refusal on overflow. Both tiers, bit-free. RUNS OWED |
| 4 | Small-node packing: several tiny nodes (or all k features of one node) per block at the deep frontier | NOBODY -- cuML v26.08 still burns a full block x 10 column-blocks on a 3-row node; LightGBM is leaf-wise and never holds a wide small-node frontier | Deep-level workgroup count /k or better, on the vendor where dispatch is the bound; RPT (row 1) cannot touch this regime (small nodes already have 1 block) | NO | **CANDIDATE, RANK 3 -- NOT implemented.** Requires sub-block (warp-level) collectives in the two hot kernels; too invasive to hand-write in a lane that cannot compile. Sketch: workload entries gain a `nodes_per_block` packing arm; per-node warp collectives via `split.mojo`'s width-free shuffle butterfly (DEV 168); integer/key-space accumulation keeps it bit-free. Needs a compiling session and its own deviation number from this band |
| 5 | Cross-tree overlap (their stream pool) | cuML `n_streams=4` (`randomforest.cuh:336-341`, unchanged at v26.08) | Grid never starves at small n / deep levels | HAVE, STRONGER (DEV 211: the frontier batch itself spans trees; measured 1.72-1.85x at 581k rows) | HAVE |
| 6 | Frontier batch width | cuML `max_batch_size=4096` (unchanged at v26.08) | -- | HAVE (DEV 213: 16384/32768 measured NO EFFECT; 4096 stays) | HAVE, measured |
| 7 | Occupancy floors for small work (`min_grid_dim_y_=160`, partition `min_num_blocks=80`) | LightGBM (`cuda_histogram_constructor.hpp:155`, `cuda_data_partition.cpp:262`) | Small leaves don't starve the device | HAVE structurally: >=1 block per (node, feature) AND the merged cross-tree frontier keeps deep grids wide | HAVE (the merged frontier is the floor) |
| 8 | Quantile binning + histograms, incl. v26.08's sampled quantiles (PR #8111: 512 sampled rows/column, startup cost independent of n_rows) and smem-staged quantiles (PR #8323) | cuML, LightGBM | Reads 1-byte bins instead of 4-byte floats; one pass per level instead of two | NO, BY DEFINITION | NOT APPLICABLE to this product: DEV 137 deletes the histogram deliberately; bin-space ET on the histogram builder is the OTHER product (`ensemble/` Step 2, LightGBM `USE_RAND` precedent). Proposal recorded below, not here |
| 9 | Sibling subtraction (build the smaller child's stats, derive the larger's) | LightGBM (`cuda_histogram_constructor.cu:726-739`) | ~2x on histogram work | NO | NOT APPLICABLE as-is: children sample DIFFERENT feature sets and draw NEW thresholds, so a child's (node, feature) cells are not derivable from the parent's. The one derivable quantity is per-node class TOTALS -- see row 10 |
| 10 | Scan-once-subtract-from-parent for left/right stats | LightGBM (`cuda_best_split_finder.cu:221-222`) | Halves the per-candidate accumulator work | PARTIAL: right = total - left is already how DEV 143's accumulators work, but each of the k fslots re-accumulates the SAME node totals | **CANDIDATE, RANK 4 -- NOT implemented.** Sketch: accumulate `acc_total`/`n_total` only at fslot 0 (unconditionally on rows, decoupled from fslot 0's own constant/missing early-outs), finalize reads totals from cell (nid, 0). Removes n_acc collectives + n_acc atomics from (k-1)/k of score blocks. Price that demoted it: the score kernel's early-return structure must be reworked, and every per-cell check (`score_kernel_check`, `regression_score_check`) asserts `acc_total` per cell bitwise, so the flag arm needs flag-aware check assertions -- a wide check surface for a win that overlaps row 1's. Re-rank after 2020's numbers land |
| 11 | Stable 3-kernel partition: predicate recovered FROM the scan (never re-read), uint16 per-row scratch | LightGBM (`cuda_data_partition.cu:53-74`, `:917-943`) | The scatter pass re-reads no feature values | UNKNOWN-UNTIL-READ: our multiblock partition (DEV 203) is cuML's swap scheme; whether our scatter re-gathers `data` is a property of `partition_multiblock.mojo` this survey did not audit | **CANDIDATE, RANK 5 -- NOT implemented.** Partition is 15% on Apple, 185 ms on MI325X (post-1943), so the ceiling is bounded; and DEV 212 presumes gather-for-scratch trades a wash. Audit our scatter's inputs first; only a removed READ (not a moved one) justifies the diff |
| 12 | Out-of-place segmented-scan partition (v26.08 PR #8257) | cuML v26.08 | Exact segment bounds for MNMG allreduce | NO | NOT TAKEN, deliberately: it reads the feature column TWICE plus a full-device scan plus a copy-back, where our in-place swap reads once. Their motivation is distributed training we don't have. Recording this here is the survey's job so nobody "catches us up" to a regression |
| 13 | O(1) per-draw feature sampler (`cuda::shuffle_iterator` + minstd, per-node FNV seed) | cuML v26.08 (`kernels/builder_kernels.cuh:66-94`) | Sampler cost independent of n_cols | HAVE-EQUIVALENT: DEV 215's keyed-hash uniform k-subset (which FIXED the old pin's 0.38x bias); 215 already records the rebase note | HAVE (different mechanism, same uniformity; theirs is the cleaner port target on a future rebase) |
| 14 | Re-sample rounds when a node finds no valid split (PR #8239, sklearn-matching) | cuML v26.08 (`builder.cuh:419-457`), sklearn (`_splitter.pyx:573-577`) | sklearn's "keep drawing past max_features while all-constant" guarantee | HAVE (DEV 205's rescue: one survey pass + host-chosen column, vs their full-pipeline replay per round with a host sync each) | HAVE, cheaper shape than theirs |
| 15 | Per-node feature-list COMPACTION (not masking) | ours; LightGBM only MASKS (`cuda_best_split_finder.cu:814`) and builds every feature's histogram regardless | Sampling shrinks the gather pass, which is our dominant cost | HAVE (`colids` is materialized per node, gridDim.y = k) | HAVE -- and LightGBM's own weakness here is worth a line in any comparison writeup |
| 16 | GOSS (gradient-based row sampling) | LightGBM | Fewer rows | NO | NOT APPLICABLE: ET has no gradients; `bootstrap`/`max_samples` (DEV 460) is the row-count lever this estimator legitimately owns |
| 17 | EFB (exclusive feature bundling) | LightGBM (`dataset.cpp:112-175`) | k sparse features -> one histogram column | NO | NOT APPLICABLE: premised on shared bin ranges; without bins a bundle saves zero gathers, and dense numeric data bundles nothing anyway |
| 18 | Packed atomics (grad+hess in one int32 `atomicAdd`) | LightGBM (`cuda_histogram_constructor.cu:266-297`) | Halves atomic traffic | NO | NOT APPLICABLE PORTABLY: our per-class pairs could pack two Int32 counts into one 64-bit atomic, but Metal has no 64-bit atomics (the Metal-gaps register), and per-block publish atomics are 3-7 per block, not per row -- not a cost center after row 1 |
| 19 | Launch-count reduction: fused seeders, folded memsets, skip-on-equal H2D | (no upstream; cuML enqueues per step) | Host enqueue cost per cycle | HAVE (DEV 470, 471, 472 -- gates green, cold-box timing owed) | HAVE, runs owed |
| 20 | Leaf-wise (best-first) growth with per-leaf split caching | LightGBM (`cuda_single_gpu_tree_learner.cpp:186`; cached per-leaf splits, only children recomputed) | max_leaf_nodes semantics; small frontiers | HAVE (DEV 466-469, landed 2026-09-01) | HAVE; their split-info caching maps to our heap keys, already cached per record |
| 21 | CUDA graphs / cooperative launch / persistent kernels / async pipelines | NOBODY (verified in both mirrors: zero hits) | -- | -- | NOT APPLICABLE, and a useful negative: neither incumbent found these worth it for tree training |
| 22 | Row-major direct input (v26.08 `Dataset::value` strided, PR #8324) | cuML v26.08 | Skips a transpose at ingest | NO | NOT TAKEN for perf (it makes the hot gather fully strided -- a coalescing LOSS at 1M rows, their own tradeoff); as a CAPABILITY it belongs to the estimator surface, not this survey |
| 23 | Threshold-interval midpoint (`split_start`/`split_end`, v26.08) | cuML v26.08 (`split.cuh:52-55`) | Accuracy for exact-split search | NO | NOT APPLICABLE: a random threshold has no training-equivalent interval to midpoint |
| 24 | 64-bit bin counts (`BinCountT = unsigned long long`) | cuML v26.08 | > 2^31 rows | different shape | NOT APPLICABLE as-is: our exactness bound is DEV 144/218's Int64 pair with the node-uniform shift; the >2^26 regime is priced there |
| 25 | Comptime monomorphization of inner-loop flags | LightGBM (6-bool template explosion, `cuda_data_partition.hpp:206-239`) | Registers/branches in the hot loop | HAVE (comptime params + `is_defined` arms are exactly this) | HAVE |
| 26 | Dataset-wide root-range dedup under `bootstrap=False` | NOBODY | Level-0 range pass costs 1/n_trees of today's: every tree's root ranges over the SAME rows | NO | **CANDIDATE, RANK 6 -- NOT implemented.** Bounded win: level 0 is ~1/12-1/16 of range traffic, so ~2% of an Apple fit; key-space folds make the cached cells bit-equal by construction under IDENTICAL, but FAST's float block fold could flip a -0.0/+0.0 sign vs the per-node compute, so the FAST arm needs the key fold too. Medium surface (cache + scatter kernel + bootstrap/rescue gating) for a small win: below the line this round |

### On LightGBM's `USE_RAND` "ExtraTrees", for the fast-lane board's honesty

LightGBM's ET (CUDA `cuda_best_split_finder.cu:161-166`, CPU
`feature_histogram.hpp:203-207`, 897-898) draws a random BIN and then runs
the FULL histogram build and full prefix scan anyway -- the randomness is a
gain-eligibility mask, not a cheaper search. Two consequences worth pinning:
(a) their ET price equals their exact-GBDT price, so `lgbm-et` rows on our
boards compare histogram-plus-mask against no-histogram -- state that beside
any et-vs-lgbm ratio; (b) there is still no incumbent GPU implementation of
the histogram-free formulation (re-verified this session at both pins), so
DEVIATION 137's novelty claim stands as of 2026-09-01.

## Proposals OUTSIDE this lane's ownership (proposals only, nothing edited)

1. **`tools/et_profile_leg.sh` (repo-root tools/, not ours): teach the leg
   the new arms.** It already builds per-arm with a `$DEF` slot; the ask is
   an `ET_PROFILE_ARMS` vocabulary for `rpt4/rpt8/acc4/tpb1024` so the
   AMD/NVIDIA legs can run DEVIATION 2020/2021's A/Bs unattended, and the
   `ET_PROFILE_OLD_COMMIT` env forwarding fix from DEVIATION 1945 rides the
   same edit.
2. **Bin-space ET (`ensemble/` Step 2).** DEVIATION 213's formulation-level
   note stands: on the vendor where the gather roofline binds (Apple), the
   only order-of-magnitude lever is reading bins, and that is the OTHER
   product on the ported cuML histogram builder with a random-bin flag
   (LightGBM `USE_RAND` is the precedent). Nothing in this lane's directory
   should grow toward it.
3. **`ensemble/` partition audit swap.** If row 11's audit of OUR scatter
   finds a re-gather, the same audit applies to the RF lane's partition,
   which shares the cuML swap ancestry.

## Are we exhausting ET? (the lane's standing question, answered from this table)

Within the histogram-free formulation: the incumbent-known levers are HAVE
rows except items-per-thread (now DEV 2020, runs owed) and the
nobody-has-it pair (small-node packing, rank 3; totals dedup, rank 4). On
Apple the formulation itself is within ~1.5x of the measured gather roofline
(DEV 208), so exhaustion there means DEV 2021's occupancy question and then
the formulation boundary -- past it lies only bin-space ET, a different
product. On the dispatch-bound vendor the table is NOT exhausted: 1943
proved workgroup count is worth 2.7x and rows 1/3/4 all attack it. The H100
column has never been profiled and its 4160 ms is far above any bandwidth
argument; one `et_profile` leg there decides which regime it is in.
