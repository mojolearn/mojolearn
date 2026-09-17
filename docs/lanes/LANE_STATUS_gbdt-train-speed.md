# LANE STATUS: lane/gbdt-train-speed

2026-09-17 afternoon, one of the five speed lanes of the Sep 17 fan-out. Goal:
GradientBoosting TRAINING at 1M+ rows on NVIDIA in IDENTICAL mode with no bit
moved, and a measured statement of what the FAST tier could drop.

Branch `lane/gbdt-train-speed` from main `86d33fcdf`. Worktree
`~/mojolearn-wt/gbdt-train-speed`. Evidence (every log, JSON and table quoted
here) under `~/mojolearn-evidence/gbdt-train-speed/leg_out/`. DEVIATION range
3040-3059; 3040 and 3041 are used.

## The box

RunPod pod `2ofug65rltppi5`, NVIDIA GeForce RTX 4090, driver 580.159.04, AMD
EPYC 7542 (64 threads visible, shared host, load average about 6 from other
tenants), Mojo 1.0.0 (ed45d567), CatBoost 1.2.10. Datasets staged from R2:
`taxi_speed.npz`, `istella_speed.npz`, `istella_rank.npz`. Every number below
is THIS box; none is comparable with the H100 board rows.

## What was measured first

`nsys profile -t cuda` over a whole fit (`tools/gbdt_train_probe.py fit` under
`tools/gbdt_train_body.sh pinning`), main at `86d33fcdf`, IDENTICAL, one
1-tree warm-up fit plus one 100-tree fit in the capture (101 trees; YetiRank's
first capture was 21 trees). `wall` is the 100-tree fit under the profiler.

| cell (rows x cols) | wall ms | GPU kernel ms | launches | launch API ms | cuMemAlloc+cuMemFree calls / ms | syncs / ms | H2D copies / ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| symmetric taxi (1M x 18) | 439 | 126 | 20,175 | 86 | 1,476 / 117 | 1,442 / 34 | 1,915 / 8 |
| depthwise taxi | 600 | 167 | 23,104 | 99 | 2,247 / 122 | 2,858 / 71 | 8,783 / 35 |
| lossguide taxi | 1,397 | 331 | 108,163 | 463 | 2,265 / 124 | 11,001 / 84 | 46,439 / 187 |
| symmetric Istella-S (1M x 220) | 2,045 | 303 | 21,517 | 95 | 1,476 / 119 | 1,534 / 278 | 3,387 / 15 |
| depthwise Istella-S | 2,644 | 364 | 23,840 | 100 | 2,235 / 124 | 2,950 / 259 | 10,255 / 41 |
| lossguide Istella-S | 3,460 | 627 | 114,776 | 496 | 2,235 / 127 | 11,306 / 266 | 48,913 / 196 |
| YetiRank Istella-S LETOR (2,043,304 x 220, 19,245 queries) | 10,843 | 6,405 | 18,491 | 79 | 1,520 / 159 | 532 / 6,245 | 2,395 / 14 |

What the table says:

1. **YetiRank's 85 ms a tree (65 on this box) is ONE kernel.**
   `yeti_rank_task_kernel` ran 29.07 ms a call (42 calls in 21 trees, min 28.8,
   max 32.0), two calls a tree (the search gradient at
   `gbdt/methods/doc_parallel_boosting.mojo`, the leaf estimation at
   `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:507-514`): 58.1 of
   58.2 ms a tree, 88.6 percent of all GPU kernel time. It is not pair
   sampling on the host, not a round trip per tree and not a synchronize per
   query. The kernel gave each of the 2,046 tasks ONE GPU thread
   (`block_dim=(1, 1, 1)`) that walked 1024 positions through ten rounds of
   draws, a ten-pass merge sort and 2,048 pair steps in device memory. The
   reference (`catboost/cuda/targets/kernel/yeti_rank_pointwise.cu:36-180`)
   runs a task on a 256-thread block with the sort in block shared memory.
2. **The pointwise fits are HOST bound, not kernel bound.** GPU kernels are 29
   percent of the symmetric taxi wall. `cuMemAlloc` plus `cuMemFree` were 117
   ms, 27 percent of that fit, about 16 device allocations a tree. With
   `--cuda-memory-usage=true` the sizes name the sites: per tree, three or
   four buffers of `n_rows * 4` bytes, one of `2 * n_rows * 4`, one of 15,872
   and two of 16,384 bytes, and eight of 512 bytes or less. Twelve of the
   sixteen are `make_bin_optimized_oracle`'s (one call per estimation task);
   five more are `compute_bins_for_model`'s.
3. **Lossguide is launch bound.** 1,071 kernel launches, 460 host-to-device
   copies and 109 synchronizes PER TREE (17 launches and 7.4 small uploads per
   expansion step): 463 + 187 ms of API time in a 1,397 ms fit. This, not
   kernel arithmetic, is the per-tree factor against XGBoost.

## What changed

### DEVIATION 3040: a YetiRank task runs on a 256-thread block (NVIDIA column)

`gbdt/targets/kernel/yeti_rank.mojo`: `yeti_rank_task_block_kernel`, grid
`(n_tasks, 1, 1)`, block `(256, 1, 1)`, four lanes a thread, 32 KiB of block
shared memory (composite sort keys and their ping-pong copy, `relev * weight`,
the exps, the two accumulators), a `barrier()` where the reference has
`__syncthreads()`. The row `yeti_block_parallel_for[column]` is True on the
NVIDIA column only; `-D MOJOLEARN_3040_YETI_SEQUENTIAL=1` is the kill switch,
`-D MOJOLEARN_3040_YETI_BLOCK=1` opts another column in. Apple (whose 32 KiB
threadgroup limit the kernel fills exactly) and AMD keep the sequential
kernel until their columns run this one.

Why no bit moves (the DEVIATION block in the file says the same):
- the draws: thread `tid` owns positions `tid + 256 * k` and advances one
  stream through its lanes in order every round, the stream the sequential
  kernel kept in `s_seed[tid]`; the float expressions are the same text;
- the sort computes no float. Its result is the unique ascending order of
  `(query << 42) | (~key << 10) | position`, which is (query ascending, key
  descending, position ascending), the order `_b_first` gives the stable
  merge. Every composite is distinct, so any correct sort gives this one
  permutation; here each element finds its slot by a binary search of the
  sibling run, ten passes;
- the pairs: `pairWeight` and `ll` are the same expressions on the same
  operands (`0.15 * pow(decay, j - begin - 1)` is hoisted out of the round
  loop; it depends on the position and its query begin only). Within one
  (round, lane, phase) step every thread writes a distinct document and the
  steps are separated by barriers, so each document's sums keep the one order
  (round, lane, phase);
- the accumulators start at 0.0 in shared memory and are stored once.

The identity fixtures' queries are at most 40 rows, so none crosses a 256-row
lane boundary. The A/B below is the check that does: Istella-S queries run to
hundreds of rows, and the 10-tree and 100-tree prediction digests over all
2,043,304 training rows are equal across arms.

### DEVIATION 3041: the oracle's twelve device buffers belong to the fit (NVIDIA column)

`gbdt/methods/leaves_estimation/pointwise_oracle.mojo`:
`OracleDeviceScratch`, `OracleScratchPool`, and an optional `scratch`
argument on `make_bin_optimized_oracle` (None allocates exactly as before, so
every check and every other column is unchanged).
`gbdt/methods/doc_parallel_boosting.mojo`: `TEstimationWorkspace` owns the
pool and `_estimate_and_apply` hands the oracle handle views. The precedent is
DEVIATION 1890 in the same function (the estimator's gathers), whose reason is
the reference's (`TCudaManager` hands these out of a per-device pool). The row
`oracle_scratch_pooled_for[column]` is True on NVIDIA only;
`-D MOJOLEARN_3041_ORACLE_ALLOC_PER_TREE=1` is the kill switch,
`-D MOJOLEARN_3041_ORACLE_POOL=1` the opt-in.

The pool has two halves because they have different keys: the `n_rows`-sized
buffers are a pool of one; the bin-sized buffers (a few KiB) are one entry per
exact `bin_count`, because they are whole-buffer copy endpoints and a
non-symmetric fit's trees do not all have the same leaf count (capped at 128
keys).

Why no bit moves: no kernel, grid, launch order, drain or host loop changes.
The oracle reads and writes the same cells through handle copies of buffers
the fit keeps. Every cell it reads it wrote earlier in the same task; the
negative control below is exactly the failure that would break this (a task
that reuses the pool skips the fill of `d_bins` and reads the previous
tree's).

After the change (taxi, 51 trees under nsys): symmetric 763 to 288 device
allocations, `cuMemAlloc` + `cuMemFree` 117 ms per 101 trees to 17 ms per 51
trees.

## Identity

`tools/identity_break.py`, the 22 `gbdt-*` lanes (the LANES default of
`tools/gbdt_resident_body.sh`), fixtures base, ties, odd, dupes, wide, two
repeats, on the pod. BEFORE is main `86d33fcdf` built on the pod; AFTER is
the branch tip. Two runs: DEVIATION 3040 alone (`identity.3040-only/`), and
both at the tip (`identity/`).

| diff | 3040 alone | tip (3040 + 3041) |
|---|---|---|
| before-cuda vs after-cuda | IDENTICAL=200 (infer/model), 105 (batch), N/A 20 and 5, DIVERGENT 0 | IDENTICAL=200, 105; N/A 20, 5; DIVERGENT 0 |
| before-cpu vs after-cpu | IDENTICAL=150, 95; REFUSED 40 (gbdt-multiclass, gbdt-onevsall and the two CTR-table lanes, by name, expected) | TIP_CPU |
| after-cuda vs after-cpu | IDENTICAL=150, 95; ONE-COLUMN 50, 10 (the refused lanes) | TIP_CROSS |
| after-cuda vs sabotage-cuda (`-D MOJOLEARN_GBDT_YETI_SABOTAGE=1`, phase 2 before phase 1) | DIVERGENT=10 and 5: all five `gbdt-yeti-rank` fixtures, `predict` and `seeded_predict`; every other cell IDENTICAL | DIVERGENT=10 and 5, the same five `gbdt-yeti-rank` fixtures; every other cell IDENTICAL |
| after-cuda vs sabotage2-cuda (`-D MOJOLEARN_GBDT_ORACLE_POOL_SABOTAGE=1`, a reusing task skips the `d_bins` fill) | not built yet | PARTIAL: the arm ran four lanes, DIVERGENT=30 (infer/model) and 15 (batch) against IDENTICAL=10 and 5, then ABORTED in `gbdt-ordered-rmse/base` with `CUDA_ERROR_ILLEGAL_ADDRESS` (stale bins index past a smaller leaf count). Seen divergent, but the arm must be made gentler before it can cover all 22 lanes (`diff.after-cuda.vs.sabotage2-cuda.PARTIAL.txt`) |

The control ran first in both runs (before vs after, and before-cuda vs
before-cpu, IDENTICAL). The CPU column cannot reach either change (both are
device code; the host oracle is `gbdt/host/`); it is recorded because the
merge rule asks for it.

Binding mtimes against the sources, read before any AFTER column
(`after_mtimes.txt`): AFTER binding built 20:25:22 UTC, sabotage 20:27:26,
sabotage2 20:29:20, BEFORE 19:44:06; no `gbdt/*.mojo` newer than the AFTER
binding.

## Speed: interleaved, three arms, one process per arm per round

`tools/gbdt_train_body.sh ab`, five rounds, the arm order flipped on even
rounds, each process: load, one untimed 1-tree fit, then a 10-tree and a
100-tree fit, each with a `cudaDeviceSynchronize` inside the clock. Median
(min..max), spread = max/min, `u` = outside the 1.10 gate (no ratio quoted
from a `u` side). `before` = main IDENTICAL, `after` = tip IDENTICAL, `fast` =
main's FAST build. Digests are sha256 of the predictions; FAST has no bit
promise and no digest is compared for it.

| cell | trees | before ms | after ms | before / after | digests equal | fast ms | fast / before |
|---|---:|---|---|---:|---|---|---:|
| YetiRank Istella-S LETOR 2,043,304 x 220 | 100 | 10856 (10817..11100) 1.026 | 5214 (5118..5231) 1.022 | **2.082** | yes `d4f04391f2ad81dc` | 11490 (11399..11575) 1.015 | 1.058 |
| same | 10 | 5059 (4882..5192) 1.064 | 4471 (4450..4476) 1.006 | 1.132 | yes `c6044355955eb760` | 5129 (5057..5215) 1.031 | 1.014 |
| symmetric taxi 1M x 18 | 100 | 404.8 (400.1..448.5) 1.121 u | 285.9 (280.3..296.0) 1.056 | not quoted (u) | yes `9d604972b94155d8` | 586.3 (580.1..596.2) 1.028 | not quoted (u) |
| same | 10 | 140.6 (139.9..157.8) 1.128 u | 130.6 (126.6..133.6) 1.055 | not quoted (u) | yes `0c31c08884daadda` | 159.4 (158.8..161.0) 1.014 | not quoted (u) |
| depthwise taxi | 100 | 523.8 (522.6..546.7) 1.046 | 418.0 (414.5..432.0) 1.042 | **1.253** | yes `5694af7699036c65` | 1232.7 (1214.7..1245.3) 1.025 | 2.353 |
| same | 10 | 157.8 (154.0..166.5) 1.081 | 150.8 (147.5..156.9) 1.063 | 1.046 | yes `dfa32e6222e1ec67` | 227.8 (221.6..237.0) 1.070 | 1.444 |
| lossguide taxi | 100 | 995.0 (991.4..1000.0) 1.009 | 899.4 (882.9..904.1) 1.024 | **1.106** | yes `b1761eecc6dfbc73` | 1725.8 (1692.6..1736.2) 1.026 | 1.734 |
| same | 10 | 203.5 (201.5..204.4) 1.014 | 200.4 (197.3..202.8) 1.028 | 1.015 | yes `4b3323f5ca6869c3` | 273.2 (270.9..276.6) 1.021 | 1.343 |
| symmetric Istella-S 1M x 220 | 100 | 1774 (1742..1832) 1.052 | 1714 (1651..1830) 1.108 u | not quoted (u) | yes `d5571c2a35ea06c8` | 1965 (1875..2026) 1.081 | 1.107 |
| same | 10 | 1509 (1455..1524) 1.047 | 1439 (1431..1494) 1.044 | 1.049 | yes `78c664534f58988e` | 1486 (1353..1518) 1.122 u | not quoted (u) |
| depthwise Istella-S | 100 | 2440 (2428..2580) 1.063 | 2384 (2265..2433) 1.074 | 1.023 | yes `e9c0f7e913af5a8a` | 3251 (3090..3309) 1.071 | 1.333 |
| same | 10 | 1561 (1466..1605) 1.095 | 1534 (1482..1575) 1.062 | 1.018 | yes `51d29f76ca04688f` | 1636 (1614..1670) 1.034 | 1.048 |
| lossguide Istella-S | 100 | 3025 (2956..3101) 1.049 | 2913 (2833..2945) 1.039 | 1.038 | yes `eb0d9510ee08a16f` | 3815 (3729..3953) 1.060 | 1.261 |
| same | 10 | 1615 (1533..1646) 1.074 | 1555 (1515..1630) 1.076 | 1.039 | yes `1af79aac126aa64e` | 1640 (1594..1719) 1.079 | 1.015 |

Two cells have a side outside the gate (symmetric taxi's BEFORE arm, one slow
round of five; symmetric Istella-S's AFTER arm at 100 trees). Their digests
are equal and their medians moved the same way as every other cell, but no
ratio is quoted from them; the rerun at 7 rounds was planned and NOT run (the
lane was paused). An earlier same-box, single-process look at symmetric taxi
(three repeats each, not interleaved) read 387.6 before and 277.5 after.

The Istella-S 1M cells move little because about 1.4 s of each fit is fixed
cost (880 MB of features quantized and uploaded) and their trees spend more in
histogram kernels; the change removes about 1 ms a tree whatever the width.

Per tree (slope between the 10-tree and 100-tree medians of each round):

| cell | before ms per tree | after ms per tree | fast ms per tree |
|---|---|---|---|
| YetiRank | 65.00 (63.74..66.37) | 8.20 (7.19..8.56) | 70.67 (70.05..71.03) |
| symmetric taxi | 2.92 (2.89..3.23) | 1.73 (1.70..1.80) | 4.75 (4.67..4.85) |
| depthwise taxi | 4.10 (4.07..4.22) | 2.97 (2.90..3.15) | 11.12 (11.00..11.37) |
| lossguide taxi | 8.82 (8.74..8.85) | 7.77 (7.58..7.79) | 16.11 (15.80..16.26) |
| symmetric Istella-S | 2.79 (2.52..4.06) | 3.06 (1.74..4.43) | 5.79 (4.08..6.41) |
| depthwise Istella-S | 10.41 (9.28..11.70) | 9.54 (8.13..10.02) | 17.94 (16.40..18.78) |
| lossguide Istella-S | 16.13 (14.90..16.30) | 14.94 (13.36..15.54) | 24.33 (23.20..24.82) |

A slope is a difference of two timings, so its spread is wide wherever the
fixed cost is large and noisy (every Istella-S row; the symmetric Istella-S
slopes span 1.6x to 2.5x and say nothing). The taxi and YetiRank slopes are
inside 1.20.

The first YetiRank A/B of this lane (`ab_yeti.void-both-arms-loaded-after/`)
is VOID and kept as a warning: the probe put its own tree's `python/` first
on `sys.path`, both arms ran the probe from the AFTER tree, both loaded the
AFTER binding, and the A/B read 1.00 with equal digests. Each arm now runs its
own copy of the probe and refuses a binding outside its tree
(`--expect-root`); the summary prints the binding each arm loaded.

### The opponent on the same box, measured once

No RTX 4090 CatBoost training row exists in `bench/OPPONENT_REFERENCE.md`
(origin/main read 2026-09-17; its 4090 section holds inference rows and the
2026-09-05 torch rows). CatBoost 1.2.10 GPU YetiRank, the config of
`tools/speed_gbdt_rank.py` (SymmetricTree, depth 6, lr 0.1, l2 1.0, 254
borders, no bootstrap, Plain, seed 7), host arrays in, `Pool` construction
inside the clock, five rounds: 100 trees 4,176 ms (4,133..4,215), 10 trees
3,029 ms (2,934..3,063), 12.74 ms a tree, 2,902 ms fixed
(`opponent/catboost-yeti.log`). Ours after this lane: 5,214 ms at 100 trees,
8.20 ms a tree, about 4,390 ms fixed. Our per-tree cost is now below
CatBoost's on this box and our total is still above it; all of the remaining
difference is the intercept. Before this lane: 10,856 ms and 65.0 ms a tree.
XGBoost was not timed (its H100 rows and slopes exist in
`bench/results/xgboost_gap_2026-09-12/README.md`; no ratio against them is
quoted because the GPU differs).

## The cost of pinning (the FAST question)

Method. The same commit (main `86d33fcdf`) built twice on the pod, IDENTICAL
and FAST; for every cell (a) the interleaved A/B above, (b) the stage ledger
(`MOJOLEARN_STAGE_TIMES=1`, which drains per stage: a SPLIT, never a timing)
of a 100-tree fit in both tiers, in ms per tree (`pinning/ledger.*.txt`), (c)
per-kernel GPU time from nsys in both tiers (`pinning/kernels.*.txt`). The
cost of a pinned seam is the stage's IDENTICAL time minus its FAST time, where
positive. The ceiling is the IDENTICAL time with every positive difference
removed: every pinned seam at zero cost, nothing else changed.

| cell | IDENTICAL 100 trees ms | FAST today ms | FAST / IDENTICAL | pinned seams, ms per tree (ledger) | share of the IDENTICAL fit loop | ceiling at zero pinning cost ms |
|---|---:|---:|---:|---:|---:|---:|
| YetiRank Istella-S LETOR | 10856 | 11490 | 1.058 | 0.30 | 0.5% (3.7% of this lane's 8.2 ms tree) | 10825 |
| symmetric taxi | 404.8 u | 586.3 | not quoted (u) | 0.09 | 3.0% | 396 |
| depthwise taxi | 523.8 | 1232.7 | 2.353 | 0.16 | 3.6% | 508 |
| lossguide taxi | 995.0 | 1725.8 | 1.734 | 0.65 | 6.4% | 930 |
| symmetric Istella-S | 1774 | 1965 | 1.107 | 0.18 | 3.7% | 1757 |
| depthwise Istella-S | 2440 | 3251 | 1.333 | 0.26 | 2.3% | 2414 |
| lossguide Istella-S | 3025 | 3815 | 1.261 | 0.77 | 4.2% | 2948 |

The share is the pinned milliseconds over the stage-timed fit loop's
milliseconds per tree (3.13, 4.37, 10.22, 4.79, 11.14, 18.29 and about 64);
against the whole 100-tree fit with its fixed cost it is smaller still. The
ceiling is the IDENTICAL median minus 100 trees of pinned milliseconds.

**The answer for the 25 / 10 percent rule: every cell is under 10 percent
(0.5 to 6.4 percent). There is no FAST prize in relaxing the pins. FAST today
is SLOWER than IDENTICAL on every cell on this box**, and not because of
arithmetic: FAST's GPU kernel time is higher than IDENTICAL's in all seven
cells (269 against 126 ms on symmetric taxi, 870 against 627 on lossguide
Istella-S).

Each seam FAST may legally relax and IDENTICAL may not, with its measured
cost. Ledger stages are ms per tree, IDENTICAL minus FAST, taxi / Istella-S;
negative means the pinned arm is the cheaper one.

| seam | where | IDENTICAL does | FAST does | measured |
|---|---|---|---|---|
| multi-block histogram flush | `greedy_subsets_searcher/kernel/hist_one_byte.mojo:833` (twin `:1325`), `hist_half_byte.mojo:509`, `hist_binary.mojo:490`, `hist_2_one_byte_base.mojo:540`; row `checks/kernel_matrix.mojo:834` | `Int32(val * scale)` into an integer atomic | float `atomicAdd` | inside `hist.build` / `sym.hist`: symmetric +0.071 / +0.177; depthwise -0.088 / -0.312; lossguide -0.322 / -0.619 (the whole stage, quantize and bridge included) |
| gradient quantize launch | `kernel/histogram_utils.mojo:100`, launched `greedy_search_helper.mojo:1906` | one launch per level | none | inside the `hist.build` row above |
| fixed-point to float bridge | `kernel/histogram_utils.mojo:615`, launched `greedy_search_helper.mojo:1408-1416`; fused form `:712`, `:3195` | one launch per level, zeroes `acc_i32` | omitted (DEVIATION 1892) | inside the `hist.build` row above; not among the 25 largest kernels of any cell |
| scale choice and magnitudes | `histogram_utils.mojo:808`, `greedy_search_helper.mojo:4942`; drain `doc_parallel_boosting.mojo:1972-1979` | one launch, a 2-float readback and a synchronize per tree | comptime dead | `iter_mags_drain` 0.011 / 0.008 ms per tree (the drain is billed where the tree's own drain would fall anyway) |
| leaf winner fold | `greedy_search_helper_depthwise.mojo:1809-1904` against `:1917-1995` (DEVIATION 1904) | reads back `2 * argmax_blocks` records per leaf, folds on the host | one fold kernel, a small readback | `score.read` + `score.hostreduce`: depthwise -0.001 / -0.082; lossguide -0.040 / -0.625 (FAST's device fold costs MORE at 220 features) |
| partition stats sweep every level | `greedy_search_helper_depthwise.mojo:1618` | sweep every level | sweep once, then an update kernel in the split chain | `partstats` +0.093 / +0.131 depthwise, +0.408 / +0.474 lossguide: the largest real pinning cost |
| zero all compute leaves | `greedy_search_helper_depthwise.mojo:1385` | zero every compute leaf | zero dirty slots only | `hist.zero` +0.018 / +0.024 depthwise, +0.147 / +0.174 lossguide |
| pinned partition chunk count | `gbdt/gpu_util/partitions_reduce.mojo:251`, `kernel_matrix.mojo:966` | SM pinned to 32 | the device's SM count | `est.pstats` -0.21 (every cell): the pinned count is the cheaper one here (grid 32x45 against 128x45, 17 against 40 ms of kernel time) |
| stable partition route | `greedy_search_helper.mojo:853`, `:1521`, `:4392`, `:5445`; row `kernel_matrix.mojo:914` (DEVIATION 1907) | the established partition chain | the single-pass reorder | `sym.split` -1.59 / -1.72, `split.chain` -1.24 / -1.56 depthwise, -1.51 / -1.37 lossguide: FAST's kernel is 233 to 290 microseconds a launch and is the largest single reason FAST is slower |
| device leaf partition | `leaves_estimation/doc_parallel_leaves_estimator.mojo:61` (DEVIATION 2551) | device radix sort, small readback | `partition_from_bins`: every row's leaf to the host, a host counting sort, re-upload | `iter_partition` -4.6 ms per tree on all four non-symmetric cells: FAST's arm is the slow one |
| flush to zero, `identical_mul_add`, software exp/log/pow, `pinned_block_sum` | `checks/numerics.mojo:73-90`, `kernel/compute_scores.mojo:49`, `targets/kernel/pointwise_targets.mojo:108-177` | software flush and fixed fold shape inside existing kernels | hardware fma, libm, `block.sum` | inside `sym.score` / `score.kernel` (+0.007 / -0.032, +0.002 / +0.009) and `est.approx` (+0.013 on every pointwise cell); YetiRank's `est.approx` holds its task kernel, -1.7 |
| drains kept byte for byte | `greedy_search_helper.mojo:327` (`IDENTICAL_DRAIN_SCHEDULE`) | every existing drain kept | may merge | not separable by this method: FAST today merges none the ledger can see (`sym.drain`, `split.sizes` differ by under 0.01) |

What FAST could usefully do is the opposite of relaxing: take IDENTICAL's
device leaf partition (4.6 ms a tree on non-symmetric fits) and IDENTICAL's
partition route on NVIDIA (1.2 to 1.7 ms a tree), and then the two changes of
this lane, both of which are tier independent. No FAST kernel was built here.

## Commands

    # rent, stage (from the worktree)
    export MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key TREES_LEG_STATE=$HOME/mojolearn-evidence/gbdt-train-speed/pod
    TREES_LEG_NAME=mojolearn-gbdt-train-speed TREES_LEG_CUDA_VERSIONS=13.0 \
      MOJOLEARN_STAGE_KEYS="gbm-bench/taxi/taxi_speed.npz gbm-bench/istella/istella_speed.npz gbm-bench/istella/istella_rank.npz" \
      sh tools/trees_leg.sh rent --gpu "NVIDIA GeForce RTX 4090" --minutes 150
    # on the pod
    bash tools/gbdt_train_body.sh setup          # main, the BEFORE copy, catboost
    bash tools/gbdt_train_body.sh build-fast     # main's FAST tier
    bash tools/gbdt_train_body.sh build-after    # after a push: AFTER, both sabotage arms, the CPU copy
    bash tools/gbdt_train_body.sh pinning gbdt-lossguide.taxi 100 --cell gbdt-lossguide --dataset taxi
    AB_DIR=ab_final AB_ROUNDS=5 AB_ARMS="before=/root/mojolearn-before=identical after=/root/mojolearn=identical fast=/root/mojolearn-before-fast=fast" \
      bash tools/gbdt_train_body.sh ab yeti --cell yeti --trees 10,100 --reps 1
    bash tools/gbdt_train_body.sh identity

## Rejected, and why

- A per-query rank sort for the YetiRank block kernel (count the same-query
  elements that come first): quadratic in the query size, and Istella-S has
  queries of hundreds of rows. The merge with a binary search is 55 key loads
  per element whatever the query sizes.
- One launch per sort pass over all tasks (no barriers): about 190 launches a
  call and the sort keys in device memory; the block kernel keeps them on
  chip.
- A capacity (not exact) key for the oracle pool: `d_leaves`, `d_shift`,
  `d_part_stats` and `d_multi_stats` are whole-buffer copy endpoints, so a
  wider buffer would copy past a host buffer's end.
- DEVIATION 2661 and 2636 were not re-proposed (the lane brief).

## Documents found false or incomplete

- `gbdt/methods/greedy_subsets_searcher/depthwise_stage_times.mojo:18-19`
  says one fit constructs one `StageTimes`. The non-symmetric searcher
  constructs and reports one PER TREE
  (`greedy_search_helper_depthwise.mojo:1145`, `:2524`): a 101-tree lossguide
  fit printed 101 searcher tables. A reader that takes the last table has one
  tree, not the fit; `tools/gbdt_train_probe.py ledger` sums them.
- `bench/results/xgboost_gap_2026-09-12/README.md:122-123` ("The split chain
  and the histogram build ... That is where a per-tree factor of two has to be
  found") and section 6 (both candidates "multi-hour kernel changes"): the
  55 percent of the fit it left as "un-wrapped host bookkeeping" holds the
  larger terms. On this box device allocation alone was 20 percent of the
  depthwise taxi fit, and lossguide spends 650 of 1,397 ms in launch and
  upload API calls (1,071 launches and 460 uploads a tree). DEVIATION 3041
  took 1.25x off depthwise taxi without touching a kernel.
- The lane brief's "FAST measured SLOWER than IDENTICAL on AMD" is true on
  NVIDIA as well, on every cell measured here (1.06x to 2.35x at 100 trees).

## Owed

- Apple and AMD columns for both rows (3040 and 3041 are NVIDIA-only by
  default). Apple: the block kernel claims exactly the 32 KiB threadgroup
  limit.
- The H100 board rows were not re-taken; this lane's numbers are RTX 4090.
- Opponent row for `bench/OPPONENT_REFERENCE.md` (a shared file; the
  orchestrator adds it), under "NVIDIA RTX 4090":

      ### RTX 4090 CatBoost YetiRank training, driver 580.159.04 (2026-09-17, lane gbdt-train-speed)
      | library | version | dataset | rows x cols | parameters | inputs | 100 trees ms, median (min..max) | 10 trees ms | per tree ms | rounds | pod | log |
      |---|---|---|---|---|---|---|---|---|---|---|---|
      | CatBoost GPU YetiRank | 1.2.10 | Istella-S LETOR, 19,245 queries | 2,043,304 x 220 | SymmetricTree, depth 6, lr 0.1, l2 1.0, border_count 254, bootstrap No, Plain, seed 7 | host arrays, Pool built inside the clock | 4176 (4133..4215) | 3029 (2934..3063) | 12.74 | 5 | 2ofug65rltppi5 | ~/mojolearn-evidence/gbdt-train-speed/leg_out/opponent/catboost-yeti.log |

## Next

1. Lossguide and depthwise: the per-step control plane. 17 launches, 7.4
   small host-to-device uploads and 1.75 synchronizes per expansion step; pack
   the per-step parameters into one upload and hoist
   `compute_non_symmetric_bins_for_model`'s seven buffers and its drain
   (`gbdt/models/add_non_symmetric_tree_doc_parallel.mojo:141-147`).
2. `compute_bins_for_model`'s five per-tree buffers
   (`doc_parallel_leaves_estimator.mojo:152-156`), the same repair as 3041.
3. The ranking intercept: ours is about 4.4 s against CatBoost's 2.9 s on this
   box, and it is now the whole YetiRank and QueryRMSE difference.
4. Symmetric fits are now about half launch overhead (200 launches a tree at
   4.3 microseconds).
