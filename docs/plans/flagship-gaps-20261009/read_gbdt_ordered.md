# gbdt-ordered (CatBoost-style Ordered boosting) vs catboost-gpu: read-only analysis, 2026-10-09

Read at main 42d1e42c6. No builds, no runs. All numbers quoted are from stored evidence (paths given).

## 0. The brief's numbers are stale: the 12x and the AUC gap are both gone on main

| cell | 0.8.25 board (brief) | main now (grid flips1, `~/mojolearn-evidence/p1-grid-flips/trees_bb_{nv_n0320,amd_499}.txt`) | catboost-gpu (L40S, stored) |
|---|---|---|---|
| taxi fit ms NV / AMD | 244,000 / 135,000 | **21,192 / 22,025** | 20,300 (NV only; CatBoost has no ROCm) |
| taxi auc / logloss | 0.620358 / 0.531519 | **0.629203 / 0.529007** | 0.629073 / 0.528987 |
| istella fit ms NV / AMD | 143,000 / 86,000 | **36,587 / 36,196** | 38,100 |
| istella auc / logloss | 0.972482 / 0.227811 | **0.979444 / 0.190928** | 0.979541 / 0.190146 |
| NV vs AMD output hash | - | 16bd87ff27b4893a = 16bd87ff27b4893a (taxi), 0e5986be1df21f54 = (istella) | - |

So the lane is at **1.04x (taxi) and 0.96x (istella)** of CatBoost GPU with quality equal. The 0.8.25 rows
(`bench/results/board-archive/nvidia-l40s/2026-09-29_nvidia/BOARD.md:318-321`) predate lane/ordered-speed (device
partition sort, batched lock-step estimation), IDN T4/T5 (one-step leaves and score scale on the device), T22
(fold-order compressed index, istella 54.4 -> 37.1 s) and ORD_STD_GRIDFOLD. The 0.8.25 AUC gap was NOT ordered-specific:
the same board shows plain gbdt-symmetric at 0.621310 vs CatBoost 0.630348 (`BOARD.md:334`), and main now has
gbdt-symmetric taxi at 0.630436 (`trees_bb_nv_n0320.txt:7`); whatever fixed the shared pipeline fixed both.

What remains is the ORDERED OVERHEAD RATIO. Per tree on taxi (500 trees): ours ordered 42.4 ms vs plain 7.3 ms
(`trees_bb_nv_n0320.txt:7`, 3,631 ms) = **5.8x**; CatBoost ordered 40.6 ms vs plain 15.2 ms = 2.7x. Our fold arm costs
35 ms/tree over the plain tree; CatBoost's costs 25 ms. The plain path got every optimization lane (HIST_REP_SM, fused
levels, SYM_* pools); the fold arm got T22 and the drains. That 35 ms is the target: closing it to CatBoost's 25 ms puts
taxi at ~16 s (0.8x) and istella at ~27 s (0.7x), the symmetric lane's ratio.

## 1. Shapes and parameters (board, `tools/speed_gbdt_arm.py`)

- taxi: `load_taxi("shipped")` = all plausible 2024-01/02 yellow-cab trips minus the last 500,000 test rows
  (`speed_gbdt_arm.py:986-988, 1077, 1120-1135`; ~4-5M train rows x 16 fp32 features; the 0.8.34 L40S profile ran 4.1M).
  The brief's "1.2M rows" is not the shipped size. istella: 2,043,304 x 220 (`:849`, `ISTELLA_FEATURES = 220`).
  The lane runs Logloss on taxi and RMSE on Istella-S (`TASK_LANES["gbdt-ordered"]`, `:1828-1850`).
- gbm-bench harness values (`bench_board_harness.py:143`, `lane_config` `:2031-2041`): **500 trees, depth 8 (256 leaves),
  lr 0.1, l2 1, border_count 254** (not 128), random_strength 1, bootstrap No, Newton 1 iteration, Cosine score,
  scale_pos_weight on the binary task. Ordered knobs set on both arms: permutation_count 4 (3 learn + 1 estimation),
  fold_len_multiplier 2, fold_permutation_block 64 (`:1836-1838`).
- Folds (`gbdt/methods/dynamic_boosting_folds.mojo:430-464, 472-578`): min_estimation 100, then the right edge doubles:
  F = ceil(log2(n/100)) folds = **16 on taxi, 15 on istella**. Fold f is learn `[0, 100*2^f)` + quality
  `[100*2^f, 100*2^(f+1))`. The concatenated fold layout has `total = sum_f R_f ~ 2n` positions and 2F = 32 (30) fold
  partitions. The geometric fold schedule, the per-permutation leaf values on ONE shared structure, and the
  structure-from-one-learn-permutation rule the brief asks about are ALL already here, as CatBoost's
  (`ordered_boosting.mojo:1-60` docstring; `fit_ordered` `:2287-3269`).

## 2. Fit path (file:line)

`mojolearn.ensemble.GradientBoosting.fit` (`python/mojolearn/ensemble.py:1850`, `_bind("_mojolearn_gbdt")` `:1928`)
-> `bindings/_mojolearn_gbdt.mojo::gbdt_fit_binding` -> `gbdt/train.mojo` (borders on the host from ALL rows,
`train.mojo:5-17`; device quantize `binarize_float_feature_kernel`; compressed index) -> `train.mojo:3081 fit_ordered`
(`gbdt/methods/ordered_boosting.mojo:2287`). Per tree: derivatives per fold -> score noise/scale on the device ->
`fit_oblivious_tree_structure_traced` with `folds` (`gbdt/methods/oblivious_tree_doc_parallel_structure_searcher.mojo:373`)
-> `compute_bins_for_model` -> 4 x `_PermPartition.enqueue_sort` (`OB:1311`) -> 3F+1 estimation tasks
(`_ordered_estimate_prepare` `OB:1611`, `_estimate_prepare(one_step_device=True)` `doc_parallel_boosting.mojo:2010`,
`_ordered_estimate_complete` `OB:1672`) -> learn loss drain (`OB:3139-3150`). The host column is
`gbdt/host/gbdt_oracle_ordered.mojo` (restates every fold order).

## 3. Per-tree stage table (IDENTICAL, NVIDIA/AMD; n rows, F folds, L = 3, P = 4, D = 8, total ~ 2n)

| stage | launches | rows touched | host syncs / copies | where |
|---|---|---|---|---|
| once per fit: 4 permutations built on the host (`ordered_permutations`, O(P n) host loops), 4 uploads + 4 drains; per-perm y/w gathers (2P); L x F fold cursors (L*2n floats = 24n B resident); 4 `_PermPartition`s | ~2P+2F | P n | P drains | `OB:2347-2460`, `ordered_plan.mojo:48-63` |
| 1 derivatives: `launch_approximate` + `_ord_scatter_planes_kernel` PER FOLD on the fold's prefix cursor | **2F = 32** | 2n (sum R_f) | 0 | `OB:2740-2775` |
| 2 score std + scale: `_ord_std_gridfold_kernel` (256x64 lanes over total) + combine (+T24 fused scale) + 1-thread `_ord_std_scale_kernel` + `enqueue_snap_plane_dev` | 4 | 2n x 3 | 0 (T5) | `OB:2778-2870, 2978-3010` |
| 3 structure: fold-order cindex gathered ONCE per learn perm and cached (T22; n_cols x 2n words per perm: taxi 4 cols = 0.13 GB/perm, istella 55 cols = 1.8 GB/perm, x3 resident); `write_fold_based_initial_bins` = fill + copy PER PARTITION (2F) + 1 drain; per level: hist (1-3 policy launches, grid (feature blocks x replication, parts, **grid.z = 2F folds**)), dynamic score (1), resolve/pack (1-2), `split_subsets_from_desc` (3-4) | 4F + 8 x ~8 = **~130** | per level ~n (partial pass) of the 2n layout; hist tensor 2F x 2^d x binFeat x 8 B | **2 drains** (initial bins `fold_tasks.mojo:182-205`; winners `searcher:1014`) | `searcher:536-580, 758-1014` |
| 4 bins + partitions: `compute_bins_for_model` (1) + P x `enqueue_sort` (7 kernels + 3 D2H each) + host `settle` loops (n_leaves x bounds) | 29 | n + P n | **1 drain**, 12 D2H | `OB:3014-3040, 1311-1390` |
| 5 estimation: L F + 1 = **49 (46) tasks**, each: `task_partition` (host 256-entry table, 1 H2D, `_ord_segment_rows_kernel`), `_ord_stage_in_p_kernel`, 2 H2D (leaf offsets/sizes), oracle build (`make_sequence`, leaves upload, `fill_bins_from_partition`), one-step walk (`_one_step_prepare_kernel`, `scaled_copy`, `newton_one_step_kernel`), `_ordered_apply_kernel` | ~11/task = **~540** | est prefixes sum ~ (L+1) n; apply over L x 2n + n | 0 drains (T4), ~150 small H2D | `OB:3058-3135, 1611-1740`; `doc_parallel_boosting.mojo:2170-2215`; `device_walker.mojo:630-680` |
| 6 learn loss: approximate + lanes sum + D2H | 2 | n | **1 drain** | `OB:3139-3150` |

Total ~**740 launches, 4 drains, ~165 host copies per tree** on the IDENTICAL path (the Apple FAST bundle ORD_ALL +
ORDERED_BATCH_EST is at ~225 launches/tree, `docs/apple-fast/notes/sym-ordered.md:14-33`). At 500 trees that is 370k
launches: at 5-10 us each (L40S enqueue through Mojo) **1.9-3.7 s of the 21.2 s**, before any kernel runs.

## 4. Cost model (taxi, n = 4.5M, 16 features x 254 borders = 4,064 bin features, L40S 800 GB/s)

- Measured split, 0.8.34 (pre-T22/GRIDFOLD), taxi 4.1M, 20 trees, 2,098 ms (`docs/apple-fast/notes/sym-ordered.md:36-38`):
  `pw.hist` 38%, `ord.score_std` 12% (now GRIDFOLD), `pw.split` 12%, `ord.fold_estimates` 10%, rest ~28%
  (derivatives, fold bins, partition sorts, learn loss, launch gaps). Per tree 105 ms then, 42 ms now.
- Streaming floor of the fold-arm structure search: docs pass 8 levels x ~n docs x (16 B index + 8 B target/weight +
  4 B partition) = 1.0 GB -> 1.3 ms; histogram tensor (zero, atomic flush, score read, subtraction) sum over levels
  2F x 2^d x 4,064 x 8 B = 266 MB x ~4 passes = 1.1 GB -> 1.3 ms. Fold-arm floor **~3 ms/tree vs ~35 ms measured
  overhead**: the fold arm is latency/occupancy bound, not bandwidth bound. Why: blocks are per (feature block, part
  pair, fold); the 10 smallest folds hold < 100k docs between them, so at depth 7 most of the 32 x 128 x featureblock
  blocks zero and flush a 32 KB shared histogram for < 100 docs; the dynamic Cosine score then reads every
  (fold-learn, fold-quality) pair of the tensor. CatBoost's GPU does the same algorithm (histograms over the fold
  layout with foldCount as gridDim.z, subtraction trick, 254 borders one-byte index); its only edge is kernel tuning,
  which is why the two are within 4%.
- Leaves: CatBoost estimates all tasks through one `TLeavesEstimator` (one concatenated point vector, one walk); ours
  runs 49 independent one-step tasks (T4), each with its own sort-partition lookup, gathers, oracle and apply: the
  launch count above, plus 49 x (3 H2D copies).
- istella (220 features, 55,880 bin features): the hist tensor at depth 7 is 30 x 128 x 55,880 x 8 B = **1.7 GB per
  level** (sum over levels 3.4 GB, x4 passes = 14 GB -> 17 ms/tree of pure traffic at 800 GB/s; AMD 2.3 ms at 6 TB/s),
  which is why istella is 73 ms/tree and why AMD equals NV there. This tensor is the istella lever.

## 5. Already here as switches (do not repeat)

`ORD_STD_GRIDFOLD` (promoted), `T22` fold-order storage (promoted), `HIST_REP_SM` (promoted, reaches the fold arm),
`T24` (score+scale one launch, off, neutral), `T25` (apply from the cached leaf table, off, neutral), `IDN_ORD_ONE_STEP_DEVICE`
(T4, default on), `IDN_ORD_STD_SCALE_DEVICE` (T5, on), `PW_FUSED_LEVEL`, `ORDERED_SABOTAGE` (negative control). Apple FAST
only: `ORD_ALL` bundle (fold bins in one launch, lean buffers, wide std reduce), `ORDERED_BATCH_EST` / `ORDERED_CAT_CURSORS`
(the batched fold-leaf reduce `_ord_bat_reduce_kernel` `OB:1856-1958`), `ORD_DOC_ID_STORAGE`. Roadmap rows T3-T7
(`docs/plans/IDENTICAL_ML_TREES_ROADMAP.md:35-39`): T6 (apply reads leaves in permutation order) = T25; T7 (per-leaf sums
from the chunk x leaf matrix instead of sort + gather) is idea B below. EXPERIMENTS.md rows: `ORDERED_BATCH_EST` (kept,
Apple), `ORDERED_FOLD_DERIVS` (dropped), `ORD_FOLD_BINS_ONE` / `ORD_FOLD_INDEX` / `ORD_STD_PARALLEL` / `ORD_TREE_LEAN`
(folded into ORD_ALL), `SYM_HIST_FAST` (dropped).

## 6. Ideas (each a `-D` define, default off, A/B on NV + AMD; bits per the identity contract)

**A. Sparse fold layout for the histogram: skip the empty prefix of the fold axis.** `-D MOJOLEARN_ORD_HIST_FOLD_SKIP`.
Fold f has 100 x 2^f docs; at level d a (fold, part) cell of the first ~log2(2^d) folds averages < 1 doc. Build the
small folds' histograms with ONE block per fold over all its parts (a "small-fold" kernel: shared histogram per part
pair is impossible, so one block loops the fold's docs and atomically adds into global for every part), and launch the
regular grid only over folds whose per-part doc count exceeds the block's doc budget. Grid.z shrinks from 32 to ~8-10
at every level on taxi. Expected: `pw.hist` -40..60% on taxi (the wave of empty 32 KB flushes is the cost), smaller on
istella (bandwidth bound). Bits: NONE if the small-fold kernel folds each cell in the same per-cell order as the
replicated kernel (the Int32 fixed-point histogram is order-free: `HIST_REP_SM` note says "associative atomics: no bit
moves"); the host column is untouched. Identity risk: low. Effort M. Opus-carriable from this text plus
`compute_hist2_dev` (`pointwise_kernels.mojo:1922`) and the one-byte kernel's `shift_part_and_bin_sums_ptr`
(`pointwise_hist2_one_byte_templ.mojo:265-272`).

**B. One batched leaf estimation for every task (port ORDERED_BATCH_EST to IDENTICAL).** `-D MOJOLEARN_IDN_ORD_BATCH_EST`.
Replace the 49 per-task pipelines (partition lookup, stage-in, oracle, one-step walk, apply: ~540 launches, ~150 H2D)
with: `_ord_bat_reduce_kernel`-style segmented sums over the per-permutation stable partition (grid = chunks x
(task x leaf)), one `_ord_bat_leaves_kernel`, and L+1 applies over the concatenated cursors (`ORDERED_CAT_CURSORS`), i.e.
~9 launches and 0 host copies. The Apple kernel exists (`OB:1856-2080`); for IDENTICAL replace its `+=` loop and
`pinned_block_sum` with the fixed chunk order (`ORDERED_PART_CHUNK`-sized chunks, `halving_block_sum`, then a fixed
chunk fold) and keep the Newton statement of `newton_one_step_kernel` (soft-float64). Expected: `ord.fold_estimates`
10% -> ~2%, and ~500 launches/tree gone (-1.5..2.5 s of 21 s, more on AMD where launches cost more). Bits: CHANGE (new
fold order of the leaf sums) -> the host column `gbdt_oracle_ordered` must fold in the same chunk order under the same
define; NV = AMD by construction. Identity risk: medium (two columns to change together). Effort M-L. Opus-carriable
with the Apple kernel as the template; the roadmap's T7 is the same idea.

**C. Fold derivatives and the fold layout in one launch.** `-D MOJOLEARN_IDN_ORD_CAT_PLANES`. With the L x F cursors
stored concatenated per learn permutation (as `ORDERED_CAT_CURSORS`), the 2F derivative launches + scatters become ONE
`launch_approximate` over `total` writing straight into `sw`/`sg` (no `der_stats` round trip: saves 2n x 16 B per
tree), and `write_fold_based_initial_bins`' 2F fill+copy pairs and their drain become the ORD_ALL one-launch
`fold_bins_from_table_kernel` (`searcher:195`, already written, Apple-gated). Expected: -2F-4F launches and one drain
per tree (-5..8% taxi). Bits: none (same per-row statements). Identity risk: low. Effort S. Opus-carriable.

**D. Histogram tensor as 16-bit fixed point on the fold axis for the dynamic score (istella lever).**
`-D MOJOLEARN_ORD_HIST_I16_FOLDS`. The dynamic Cosine score reads the whole 2F x 2^d x binFeat x 2 tensor per level;
on istella that is 1.7 GB/level. Keep the Int32 accumulate in shared memory but FLUSH the small folds' cells (whose
counts are bounded by 100 x 2^f docs) as packed Int16 pairs, and read them packed in `find_optimal_split_dev`. Halves
the tensor traffic for the folds that cannot overflow; a per-fold width table decides, from the fold's row count and
the fixed-point scale (a size rule, not a shape rule). Expected: istella -20..30% of `pw.hist`+`pw.split`, taxi small.
Bits: NONE if the Int16 value is the same integer (exact when it fits; the width rule must be provable from
`choose_scale`'s bound). Identity risk: low-medium (overflow proof). Effort L. Needs a Fable lane, not Opus.

**E. Pin the fold-order compressed index once per fit instead of per learn permutation, and drop the dead partition
readbacks.** `-D MOJOLEARN_IDN_ORD_INDEX_SHARED`. T22 keeps L = 3 fold-order copies of the index resident (istella
3 x 1.8 GB = 5.4 GB on a 48 GB L40S, alongside 3 x 2n cursors and 3 x 2n gathered planes), and the fold layout differs
between learn permutations only by the permutation (the fold boundaries are identical). Build the gather from ONE
permuted index per permutation of the ORIGINAL order (n rows, n_cols words: 3 x 0.45 GB) plus the fold-position map
(`make_fold_doc_indices_device`), and gather per level only the smaller sibling's docs (the partial pass already touches
~n). Also under this define: `enqueue_sort(readback=False)` + settle on the device (`OB:1367-1390`, the three D2H copies
per permutation and the host `settle` loop are dead once B lands). Expected: memory -4 GB on istella, L2/TLB
behavior better, -12 D2H/tree; time -0..5%. Bits: none. Identity risk: low. Effort M. Opus-carriable.

**F. Drop the learn permutation that is never searched on.** `-D MOJOLEARN_ORD_SKIP_UNSEARCHED_PERM`.
`learn_p = rng % (learn_count - 1)` (`OB:2560-2566`; the only readers of `cursors[learn_p]` are the derivative
launches `OB:2677-2684`) restates CatBoost's modulus (`dynamic_boosting.h:282-289`): at
4 permutations the structure comes from permutation 0 or 1 only. Permutation 2's F fold tasks (1/3 of stage 5), its
F cursors (2n floats) and its fold-order index copy (T22, 1.8 GB on istella) move only permutation 2's own cursors,
which no derivative, score or export reads (the exported leaves are `est_p = 3`'s, `OB:3104-3135`). Skipping them
changes no bit of the exported model; the trace loses its `*.perm.2.fold.*` records, so the ID check compares the
output hash (which is what the grid compares). A read-through of every consumer of `cursors[2][*]` and
`fold_cindex[2]` is the whole verification. Expected: -1/3 of stage 5 (~3% of the fit), -2n floats and one index
copy of memory. Effort S. Opus-carriable.

Ranking for a lane: A (taxi hist), B (launches), C (small, with B), then D (istella), E, F.
