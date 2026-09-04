# Upstream technique survey, symmetric-tree GBDT speed at 1M-2M rows

Written 2026-09-01 by the symmetric performance lane, answering Andrew's
question: **"what techniques are left? are we exhausting symmetric trees?"**

Inputs, and nothing else: this repository; the pinned CatBoost mirror
`~/CascadeProjects/upstream/catboost` @7055d33d (`catboost/cuda/methods/` is
the learner our `gbdt/methods/` mirrors); `upstream/cuml-v26.08.00`;
`upstream/lightgbm` @3d1cf30 -- **technique reference only, LightGBM is never
benchmarked beside symmetric trees** ([[mojolearn-symmetric-vs-catboost-only]]).

Every claim below is a READ, not a measurement. Both implemented candidates
are FLAG-GUARDED, OFF BY DEFAULT, **UNVERIFIED, RUN OWED** -- the exact
commands are in `archive/plans/gbdt/PLAN.md`. All performance reasoning is at the 1M-2M-row
floor; nothing here cites or is motivated by a sub-1M number
([[mojolearn-large-data-timing-floor]]). The timed harness is
`checks/estimation_bench.mojo` (default greedy searcher, `border_count=254`
so the one-byte 8-bit family, depths 6 and 8, RMSE / Logloss-10 / Logloss-1
arms; bootstrap off, so the search stats' plane 0 is constant 1.0).

Deviation band claimed by this lane: **2030-2039**; 2030 and 2031 are used
below, 2032-2039 remain free (grepped clean 2026-09-01).

---

## 1. The table

Verdicts: **HAVE** (present, cited), **N/A** (not applicable, reason),
**CANDIDATE** (ranked in section 2), **NEGATIVE/CLOSED** (tried and settled).
Tier column: what the technique may touch -- `all` (bit-inert), `fast`
(fast tier only; upper tiers STABLE), `n/a`.

### 1a. Histogram computation and layout

| technique | who | what it buys at 1M+ | have it? | verdict | tier |
|---|---|---|---|---|---|
| Warp/sub-warp-private shared-memory histograms sized to fill smem, copy count traded per bit width | CatBoost `pointwise_hist2_one_byte_{5,6,7,8}bit.cu`, `hist_2_one_byte_base.cuh` | contention divided by up to 48 copies; accumulation never touches HBM | `kernel/hist_2_one_byte_base.mojo:164-199` via the `hist_smem_mode_for` matrix row; shared-Int32 arm measured 1.94x kernel, 1.33-1.51x tree | HAVE | all |
| Native threadgroup **integer** atomics + fixed-point i32 domain (Metal has no threadgroup float atomics) | ours (no CatBoost counterpart; [[metal-hardware-gaps]]) | real atomics instead of emulation; order-independent adds | `kernel/histogram_utils.mojo:100-155`, dither/quantize `:33-97`, device-side scale `:759-818` (DEV 95) | HAVE | all |
| Two-stat interleave, one pass per stat pair, odd stat peeled | CatBoost both stacks | halves doc/cindex traffic per stat | hist2 family z-axis; `hist_2_one_byte_base.mojo` | HAVE | all |
| Lock-step write turns (`tiled_partition<8/16/32>`) | CatBoost 5/6/7-bit | LD/ST + warp sync instead of atomics | `kernel/lane_sync.mojo:53-74` (`turn_sync`, DEVIATION 1947) | HAVE | all |
| 8-bit deferred-flush run-length register cache | CatBoost `pointwise_hist2_one_byte_8bit.cu:48-121` | a run of equal bins costs one atomic | pointwise family `kernel/pointwise_hist2_one_byte_8bit.mojo:20-58` (DEV 93); greedy H8 instead uses atomics-only shared slices (`hist_2_one_byte_8bit.mojo`, DEV 1906) -- our own >128-bin fused arm CatBoost never wrote | HAVE | all |
| Vector loads (64/128-bit) + head/tail alignment peel | CatBoost `compute_point_hist2_loop.cuh` | quarters LDG count on the streaming operands | `UNROLL/LOAD` traits per family, peel present (`hist_2_one_byte_base.mojo:53-59`) | HAVE | all |
| Inner-bits PASS trick (wide features paid in time, not smem) | CatBoost `hist_one_byte.cu:14-45` | 5-bit-sized smem for 6/7/8-bit features | `kernel/hist_one_byte.mojo:14-45` | HAVE | all |
| Binary packs 32 features/word, zero-side recovered by nibble sum | CatBoost `hist_binary.cu` | a binary feature costs one fold not two | `kernel/hist_binary.mojo:16-27` | HAVE | all |
| Last-bin skip (top bin recovered as `total - scanned[last]`) | CatBoost | one bin less smem per feature | ported with the scorers | HAVE | all |
| `abs(v)>1e-20` write guard | CatBoost | empty bins cost no global traffic | ported writebacks | HAVE | all |
| Two-level reduction of private copies; fused fixed-to-float bridge | CatBoost / ours | one full hist read+write per level deleted | `write_reduces_from_fixed_kernel`, `histogram_utils.mojo:663-756` (ours, beyond upstream) | HAVE | all |
| Hist zeroing elision (memset dead where every cell is overwritten) | ours | ~1 launch/level + one memset/tree | DEV 2008a/2008b (`greedy_search_helper.mojo:3569-3620`, `:3322-3337`), DEV 1892 scratch-dead | HAVE | all |
| Per-arch unroll/block tuning tables | CatBoost `__launch_bounds__` + arch traits | occupancy pinning | kernel-matrix rows per column; no `__launch_bounds__` equivalent surfaced by MAX -- **proposal**: ask the matrix lane whether register/occupancy pinning is expressible | HAVE (partial) | all |
| **Packed both-stats-in-one-atomic integer histogram** (grad<<16\|hess, one shared atomic/row) | LightGBM `cuda_histogram_constructor.cu:252-317` | halves shared atomic traffic and smem footprint | depthwise lane only: `hist_quantized_shared.mojo` (DEV 1911-1914) packs both planes into one UInt64 **load** but still two i32 atomics; the symmetric H8 does two atomics/point | **CANDIDATE #4** | fast |
| Per-leaf histogram bit-width selection (8/16/32-bit bins by `rows*quant_bins`) | LightGBM `gradient_discretizer.cpp:165-201` | 16-bit hist below ~16k rows/leaf | not present; at depth 6-8 over 1M rows leaves are 4k-31k, so it binds exactly at our shapes -- but it requires the packed-atomic family first | folds into CANDIDATE #4 | fast |
| Constant-hessian / unit-weight count plane (skip accumulating a constant plane; convert count once per bin) | LightGBM CPU only (`dense_bin.hpp:99-141`; never in their CUDA kernels) | removes one load + one add per (row, feature) on the plane-0=1.0 path -- exactly the bench's no-bootstrap first-order shape | not present; interacts with the dither (`hist2_quantize` docstring records unit-weight planes as the case that killed round-to-nearest, so a count plane CHANGES fixed-point bits) | **CANDIDATE #5** | fast |
| On-the-fly binning with quantiles cached in smem | cuML `builder_kernels_impl.cuh:286-352` | avoids storing a binned matrix | N/A -- strictly worse than the pre-binned compressed index at 1M+ (their own disadvantage); our cindex is the CatBoost contract | N/A | n/a |
| EFB feature bundling | LightGBM `dataset.cpp:112-247` | shrinks the bin matrix on sparse/one-hot data | N/A for this lane: an ingest/binning-layer change, off the CatBoost-mirror binning contract; worth a NOTE for a future ingest lane, never a kernel change | N/A (out of lane) | n/a |

### 1b. Partial pass, subtraction, scan

| technique | who | what it buys at 1M+ | have it? | verdict |
|---|---|---|---|---|
| Compute the smaller sibling, derive the larger by subtraction | CatBoost both stacks; LightGBM in-place in the parent slot | half the document scans per level, skewed splits up to 10x | `histograms_helper.mojo` state machine; planning fused into the split kernel (DEV 210, `split_points.mojo:250-345`); exact-tie takes RIGHT (archive/reference/PORTING.md 136) | HAVE |
| Subtraction kernel batched over all pairs, `max(v,0)` clamp on stat 0 | CatBoost `histogram_utils.cu:288-371` | O(binFeatures) replaces O(rows) | `substract_histograms_(vec4_)kernel`, vec4 dispatch `greedy_search_helper.mojo:3657-3678` | HAVE |
| Parent hist copied to right child at split | CatBoost `split_points.cpp:139-145` | subtraction bookkeeping becomes a size compare | `copy_histograms_vec4_kernel`, measured 11.0 -> 65.2 GB/s | HAVE |
| Prefix-scan of histograms so scoring reads O(1) per candidate | CatBoost | scorer never re-sums bins | `scan_histograms_kernel` (serial-scan substitution for `cub::WarpScan<double>`, recorded) | HAVE |
| Gather-histograms-by-leaves transpose | CatBoost multi-GPU only | reduce-scatter layout | N/A: single-device; their own single-device path shortcuts it (`histograms_helper.cpp:36-57`) | N/A |

### 1c. Streams and concurrency

| technique | who | what it buys | have it? | verdict |
|---|---|---|---|---|
| One stream per grouping policy (up to 6), lazy cross-stream sync | CatBoost `histograms_helper.h:370-376`, `pointwise_scores_calcer.h:43-60` | overlaps kernels with disjoint occupancy profiles | N/A: Metal has no streams ([[metal-hardware-gaps]]) and the lane is GPU-agnostic -- no vendor-branched stream arm ([[always-gpu-agnostic]]). What we have instead is stronger where it counts: the blind level loop (DEV 94/207) removes every per-level host block, which their stream machinery still pays twice per level | N/A |
| 3 streams for multiclass block round-robin | CatBoost `compute_by_blocks_helper.cpp:385-386` | hides launch latency for many small blocks | N/A, same reason; single-target stat_count=2 forces one block per policy on their side too | N/A |

### 1d. Scheduling, launch counts, partition/gather

| technique | who | what it buys at 1M+ | have it? | verdict |
|---|---|---|---|---|
| Zero host readbacks in the level loop; device-resolved winner | ours (CatBoost blocks twice per level: `greedy_search_helper.cpp:517`, `split_properties_helper.cpp:802`) | deletes 2 blocking syncs x depth per tree | DEV 94/207; two `wait_complete` per tree total | HAVE (beyond upstream) |
| Constant launch count per level, precomputed block mappings | CatBoost `compute_by_blocks_helper.h:87-92` | launch count independent of dataset | `build_layout`/`blocks_for`; census ~15-16 + 2B launches/level | HAVE |
| Blocks-per-feature multiplier with 10k-rows/block floor | CatBoost `split_properties_helpers.cuh:15-23` | occupancy for wide-tall data | `replication_for` (`greedy_search_helper.mojo:2044-2059`), sm pinned under IDENTICAL (DEV 354) | HAVE |
| `update_partition_props` one-block-per-partition reducer | CatBoost `pointwise_scores.cu:624-694` | 6 launches + 2 scratch -> 1; Split 17 -> 12 | CLOSED by measurement 2026-08-21 (archive/reference/PORTING.md 98, tie at 1 part, up to 4.6x after) -- **NOTE: `archive/plans/NEXT_TWO.md:338-346` still lists this as OPEN DECISION 1; stale, proposal below** | HAVE |
| One-bit reorder vs radix at the 500k crossover; single-pass decoupled-lookback partition per level | CatBoost `split_points.cu:737-770`; ours DEV 1907 | 3 launches -> 1 above 500k rows | `reorder_single_pass.mojo:501-551`, routed by matrix row (FAST NVIDIA/AMD) -- **proposal**: evaluate the Apple column for `reorder_single_pass_for` (matrix rows live in `checks/kernel_matrix.mojo`, outside this lane's ownership) | HAVE / proposal |
| Shared-memory in-place gather for leaves <= 1024 rows | CatBoost `split_points.cpp:113-137` | one launch replaces 4 at deep levels | ported (`gather_inplace_kernel`) but the symmetric driver's blind loop passes `max_live_rows = n_rows`, so it never fires -- N/A at 1M-2M x depth 6-8 anyway (leaves are 4k-31k, above the 1024 bound); would matter only at depth >= 11 | HAVE / N/A at bench shapes |
| **Stationary stat planes; a split permutes only the 4 B/row index** | ours, DEV 1902 (depthwise/lossguide only until now) | deletes 2 launches + ~2x2x8 B/row reorder traffic per level; ~128 MB/tree at 1M x depth 8 | depthwise: `greedy_search_helper_depthwise.mojo:2278-2285`, `:2434-2440`; **symmetric driver did NOT have it** | **IMPLEMENTED: DEVIATION 2031** (fast tier via the existing `ridx_only_splits_for` row) |
| Level-batched partition via one segmented scan-by-key | cuML `builder_kernels_impl.cuh:143-212` | one scan per level vs per-leaf kernels | equivalent-or-better already: per-level single-pass partition (DEV 1907) | HAVE-equivalent |
| GatherBins: materialize leaf-contiguous cindex once, amortize over stat pairs | CatBoost `gather_bins.cu`, dispatch `split_properties_helper.cpp:1338-1340` | multiclass: N random passes -> 1 random + N/2 sequential | not ported; **their own dispatch chooses LoadByIndexBins at statCount <= 2**, so it is N/A for every single-target cell; CANDIDATE only if a multiclass symmetric bench cell ever exists | N/A (single-target) / CANDIDATE #6 (multiclass) |
| Bootstrap zero-weight rows filtered before histogramming | CatBoost (their bootstrap makes the learn subset) | at `subsample=0.66` they stream 66% of rows, we stream 100% | DEVIATION 69 (`gpu_util/kernel/bootstrap.mojo:36-70`) records the gap, open | **CANDIDATE #3** (bootstrap cells only; the current bench runs bootstrap off, so it cannot move the 6 cells) |

### 1e. Leaves estimation (the Logloss arms)

| technique | who | what it buys at 1M+ | have it? | verdict |
|---|---|---|---|---|
| Flat `AddBinModelValue` MoveTo kernel (not partition-gridded) | CatBoost `add_model_value.cu:14-53` | 13.6 ms -> ~1 ms/call at 2M (measured, DEV 210b) | `kernel_add_model_value.mojo:114-183` | HAVE |
| Fused value+der+der2 single eval kernel (rowSize==1 fast path) | CatBoost `pointwise_oracle.cpp:73-78` | one pass computes all three | `launch_approximate[True]`, der2 cached | HAVE |
| One batched readback + one drain per walker evaluation | CatBoost `ReadReduce` | the floor CatBoost also pays | `pointwise_oracle.mojo` (DEV 1891 audit) | HAVE |
| **Fusing MoveTo INTO the evaluation kernel** (the pair is strictly move->eval, so the ABMV pass exists only to stage bytes the next launch reads) | nobody -- upstream pays both passes; [[ok-to-add-capability]] | deletes 1 full-row launch + ~12 B/row per evaluation; ~11 evaluations/tree at Logloss-10 (the bench priced one Newton iteration at ~8 ms/tree-iter at 1M, `bench/results/ab_large_2026-09-01/RESULTS.md`) | was absent | **IMPLEMENTED: DEVIATION 2030** (bit-identical, all tiers) |
| Device-resident Newton walker (no readback per try) | nobody | would delete ~11 blocking syncs/tree | NOT IMPLEMENTED: the accept/halve control flow is data-dependent (`AnyImprovement` compares function values), so blind-enqueueing requires speculative try budgets that diverge from CatBoost's tick semantics in the never-accepting tail. Revisit only if 2030's A/B shows the sync (not the pass) dominates `est.*` | CANDIDATE #7 (design risk) |

### 1f. Gradient quantization (Andrew's explicit question)

CatBoost's GPU has **no gradient quantization anywhere** in `cuda/methods/`
(stats are float; only reductions promote to double). LightGBM's quantized
training is the real art here: int16 grad+hess packed per row
(`cuda_gradient_discretizer.cu:88-126`), stochastic rounding, integer
histograms with one packed atomic, integer-exact subtraction, renormalization
deferred to split evaluation, and leaf renewal with true gradients. Our tree
already holds the middle ground: the fixed-point i32 histogram domain with
position-keyed dither is quantized ACCUMULATION (order-independent,
cross-vendor identical) without quantized GRADIENTS. The remaining
transplant -- quantized gradient STORAGE and the packed single atomic -- is
CANDIDATE #4: fast-tier only (it moves bits by construction), significant
build cost, and the depthwise lane's `hist_quantized_shared.mojo` is the
staging ground that already exists.

---

## 2. Ranked candidates

Ranking is expected-win-per-risk at the bench's 6 cells (1M/2M, d6/d8,
rmse/ll10/ll1), then breadth.

1. **DEVIATION 2030 -- fused MoveTo+eval in the Newton walker. IMPLEMENTED,
   OFF BY DEFAULT** (`-D MOJOLEARN_2030_FUSED_EST_MOVE=1`).
   Bit-identical in every tier (same ops, same operands, same order; the
   evaluation body is the existing kernel called unchanged). Targets the
   ll10/ll1 cells: deletes one full-row launch and ~12 B/row of the
   ~28 B/row each walker evaluation streams; ll10 pays ~11 evaluations/tree.
   Expected: several percent on ll cells, zero on rmse cells (DEV 64 skips
   the walker). Files: `gbdt/targets/kernel/pointwise_targets.mojo`
   (kernels + launcher), `gbdt/methods/leaves_estimation/pointwise_oracle.mojo`
   (flag, deferred shift, flush paths).
2. **DEVIATION 2031 -- ridx-only splits in the symmetric driver.
   IMPLEMENTED, OFF BY DEFAULT** (`-D MOJOLEARN_2031_SYM_RIDX_SPLITS=1`,
   AND-gated with the existing `ridx_only_splits_for` matrix row, so
   fast-tier only and inert under IDENTICAL by routing). Expected
   model-identical (DEV 1902's permutation argument), all 6 cells: deletes
   2 launches and ~32 B/row of reorder traffic per level (~128-256 MB/tree
   at 1M-2M x d8). File: `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo`
   (5 comptime-elided call sites riding DEV 1902's existing kernels).
3. **Bootstrap zero-row filtering (closes DEVIATION 69)** -- NOT implemented.
   Only moves fits with Bernoulli/Poisson bootstrap on; the current bench
   cells run bootstrap off, so it needs a new bench cell first (proposal in
   `archive/plans/gbdt/PLAN.md`). Mechanism: compact surviving rows once per tree after
   `launch_bootstrap`, hand the compacted count to the level loop. ~34%
   fewer rows streamed at their default subsample. Fast-tier question mark:
   compaction reorders accumulation -> gate before labeling.
4. **Packed integer histogram family for the symmetric one-byte path**
   (LightGBM 1b/1c/1d shape, staged on `hist_quantized_shared.mojo`) -- NOT
   implemented. Fast-tier only, biggest kernel-level headroom left
   (halves H8's shared-atomic traffic), largest build+gate cost; needs
   per-leaf bit-width policy to be sound at 4k-31k-row leaves.
5. **Unit-weight count plane** (constant-hessian analogue) -- NOT
   implemented. Fast-tier only (the dithered fixed-point plane-0 bits move
   by construction); wins only the no-bootstrap first-order shape; smaller
   than #4 and subsumed by it if #4 lands.
6. **GatherBins amortization** -- multiclass symmetric cells only; N/A for
   every current cell.
7. **Device-resident walker** -- design risk recorded in 1e; revisit after
   2030's numbers.

## 3. Proposals outside this lane's ownership (gbdt/** only)

* `archive/plans/NEXT_TWO.md:338-346` ("OPEN DECISIONS" item 1, the partition reducer) is
  STALE: archive/reference/PORTING.md 98 closed it by measurement 2026-08-21 and
  `update_subsets_stats` calls `update_partition_props` today
  (`pointwise_optimization_subsets.mojo:1278`). One-line fix owed by whoever
  next owns the root docs ([[fix-docs-on-discovery]] -- this lane's
  ownership stops at `gbdt/`).
* `checks/kernel_matrix.mojo`: (a) if DEV 2031's A/B wins and the
  fingerprint gate holds, the flag should become a named matrix row (or fold
  into `ridx_only_splits_for`'s consumers) instead of a `-D`; (b) evaluate
  the Apple column for `reorder_single_pass_for` (the single-pass partition
  currently routes FAST NVIDIA/AMD only); (c) occupancy/`__launch_bounds__`
  pinning expressibility question for the hist kernels.
* A bootstrap-on bench cell (Bernoulli 0.66, 1M/2M) is prerequisite to
  candidate #3; `checks/estimation_bench.mojo` is orchestrator-owned.

## 4. Answer: are we exhausting symmetric trees?

**The classic histogram playbook is nearly mined out; the lane is not.**
Of the ~40 concrete techniques CatBoost's CUDA learner uses, this tree
already carries essentially all that apply on one device (table 1a-1d), plus
five it improved on (blind level loop, memset elision, fused fixed-point
bridge, per-level single-pass partition, the >128-bin fused atomic arm) --
and the three structural N/As (streams, multi-GPU transpose, EFB) are N/A
for reasons that will not change. What is genuinely left, in order:

1. the two candidates now implemented (estimation fusion 2030, stationary
   stats 2031) -- the last large *bit-safe or routing-safe* wins visible
   from reading upstream;
2. one measurement-gated pool: the LightGBM-shaped quantized/packed integer
   family (#3-#5), all fast-tier, all bits-move-by-construction, worth one
   dedicated round IF the fast tier's speed mandate wants it;
3. after that, the honest answer is that symmetric-tree KERNEL speed at
   1M-2M rows approaches the memory-bandwidth floor on every vendor, and the
   remaining big numbers live in the lane's STRUCTURAL debt instead:
   ordered boosting (rung 3 -- CatBoost's shipped default we still cannot
   claim parity with), tree CTRs (rung 4), and the `use_pointwise_searcher`
   default question -- all blocked behind the DEV 134 embargo, whose soak
   remains the highest-value run this lane cannot perform itself.

No speed or parity number in this document is quotable until DEV 134 closes
(`archive/plans/NEXT_TWO.md`); everything here is code-read plus the already-recorded
2026-09-01 internal window.
