# sym-iter: what one boosting iteration of the pointwise SymmetricTree arm does besides histograms

Lane `lane/apple-fast-sym-iter` (2026-10-03). Board lanes `gbdt-symmetric-1000` (deciding: taxi, 1000 trees, ~22 ms per
tree on the M3 Ultra FAST) and `gbdt-symmetric` (500 trees). Board configuration (`tools/speed_gbdt_arm.py`, gbm-bench
values): `grow_policy='SymmetricTree'`, depth 8 (256 leaves), learning rate 0.1, `l2_leaf_reg` 1, 254 borders,
`random_strength` 1.0, score function Cosine, Newton with one iteration, Plain boosting, one permutation, no eval set
(so no overfitting detector and no test cursor), Logloss on taxi, RMSE on Istella-S. Profile by reading, not by timing.

Code path: `gbdt/methods/doc_parallel_boosting.mojo::fit_with_test`, the `elif use_pointwise_searcher:` arm (the
`TDocParallelObliviousTreeSearcher` learner, `train.mojo:699-710`), with `need_estimation` False for RMSE (DEVIATION 64)
and True for Logloss; both end in `_estimate_and_apply` on the learn permutation. The histogram and score launches inside
`fit_oblivious_tree_structure_traced` are lane af-sym-hist's and are not listed per kernel here.

## Per tree, in order (main at 8897404da; D = host drain, A = Metal allocation, C = host copy)

| step | what | launches | D | A (device) | A (host) |
|---|---|---|---|---|---|
| 1 | gradient at the current cursor: `launch_approximate[False]` (`cross_entropy_kernel` / `pointwise_target_kernel`), fv fold `deterministic_sum_lanes_kernel[1]`, magnitude fold `[2]` (Apple quantizes the hist_2 shared stage, so `_needs_magnitudes` is True) | 3 | 0 | 0 | 0 |
| 2 | `enqueue_copy(h_fv <- fv)`; `launch_make_sequence(row_index)` | 1 | 0 | 0 | 0 |
| 3 | score-noise std dev (`random_strength` 1.0): `compute_std_dev` = partials kernel + fold + copy + **drain**, 2 device + 1 host buffer per call | 2 | **1** | 2 | 1 |
| 4 | fixed-point scale: host buffer `hm`, copy of `mags`, **drain**, `choose_scale` on the host (`compute_hist2` takes the scale as a host scalar) | 0 | **1** | 0 | 1 |
| 5 | `split_stat_planes`: two `n_rows` buffers per tree + one split launch; `enqueue_snap_plane` on the gradient plane | 2 | 0 | 2 (n_rows) | 0 |
| 6 | structure search (`fit_oblivious_tree_structure_traced`): per tree `d_doc_ids`(1), `d_fold_cindex`(1), `d_observations`(**n_rows**) allocated though the plain arm never reads them; pooled subsets/calcer reset; 8 levels of hist + score + fused resolve/pack/bins + one-bit sort + partition stats (af-sym-hist's); tail: 2 copies + **drain** (DEVIATION 207, one per tree) | per level: hist(1..3) + score(1..3) + pack(1) + sort(~4) + stats(3) | **1** | 3 (one n_rows) | 0 |
| 7 | `d_bins` (`n_rows`) + `compute_bins_for_model`: 5 host + 5 device staging buffers, 5 copies, 1 launch, **drain** (keep-alive only) | 1 | **1** | 6 (one n_rows) | 5 |
| 8 | `partition_from_bins`: a fresh `DeviceLeafPartitioner` (7 `n_rows`-sized buffers + bounds + host bounds, its ctor **drains**), copy kernel, `launch_make_sequence`, 8-bit LSD radix sort (~8 x 3 launches), bounds kernel, bounds copy, **drain**; then the caller's **drain** (keep-alive) | ~28 | **3** | 9 (7 n_rows) | 1 |
| 9 | `_estimate_and_apply` stage-in: gathers of target and cursor by `row_index` (2 launches), `h_po`/`h_ps` fill + 2 copies (pooled, DEVIATION 1890) | 2 | 0 | 0 | 0 |
| 10 | `make_bin_optimized_oracle`: device scratch pooled (DEVIATION 3041, Apple FAST on), host staging NOT pooled: `h_leaves`, `h_shift`, `h_fv`, `h_part_stats`, `h_multi_stats` per tree; `launch_make_sequence(d_identity)`, `h_leaves` copy, `fill_bins_from_partition_kernel` | 2 | 0 | 0 | 5 |
| 11 | Newton walker, one iteration: `move_to(0)` (`h_shift` fill + copy, shift deferred under DEVIATION 2030), `launch_approximate_move_eval[True]`, `compute_partition_stats` (2 launches), 2 copies, **drain**; host direction; `regularize` | 3 | **1** | 0 | 0 |
| 12 | tail: `h_est` fill + copy, `add_model_value_kernel`, **drain** (DEVIATION 1891) | 1 | **1** | 0 | 0 |
| 13 | host: model append (`leaf_values * learning_rate`), `h_fv` read into `losses` | 0 | 0 | 0 | 0 |

Totals per tree outside the histogram/score math: **9 host drains** (steps 3, 4, 6, 7, 8 x3, 11, 12), **~20 Metal
allocations** (6 of them `n_rows`-sized, 1 M rows x 4 B = 4 MB each on taxi), **12 host buffer allocations**, ~40
launches (28 of them the radix sort's). Every allocation is a live Metal buffer for the rest of the tree, and Apple's
launch cost grows ~0.25 us per launch per live buffer; every drain is ~0.2 ms of idle GPU plus the host's turnaround.
At 1000 trees that is 9000 waits and 32000 allocations the fit pays for bookkeeping.

What a tree needs the host for, irreducibly on this arm: the structure (the winners come back once, step 6) and the Newton
step (host walker, step 11) -- unless the leaves are computed on the device (SYM_LEAF_FROM_STATS below), when only the
structure drain and one closing drain remain.

Live Metal buffers across a fit (approximate, taxi): the fit-level set (~60: cindex, stats, cursor, fv/mag partials, the
pooled subsets (~17), the calcer's helpers (~10 per policy), the estimation workspace (9), the oracle scratch pool (12))
plus the per-tree transients above (~30), peaking inside step 8.

## The defines (details and risks in docs/apple-fast/ab/sym-iter.md)

- `MOJOLEARN_SYM_BUF_ARENA`: steps 4, 5, 6 (dummies), 7, 8 (partitioner), 10 (host staging) from a pool of one; removes
  the two keep-alive drains (7, 8's outer); allocations per tree -> 0.
- `MOJOLEARN_SYM_REUSE_PARTITION`: steps 7 and 8 skipped on full-depth trees; the searcher's partitions ride its drain.
- `MOJOLEARN_SYM_DERIV_FUSED`: steps 1-3 enqueued at the previous tree's step 12; drains 3 and 4 gone.
- `MOJOLEARN_SYM_LEAF_FROM_STATS`: steps 9-12 replaced by two launches + one readback (leaf values); drain 11 gone.
- `MOJOLEARN_SYM_ITER_ALL`: drains per tree 9 -> 2 (structure, tail), allocations -> 0, launches outside the search
  ~40 -> ~8.

Not done: the fixed-point scale as a device pointer into `compute_hist2` (would let the magnitude fold feed the search
without any host step; the kernel signature is af-sym-hist's), the overfitting detector / test metric batching (the board
lanes fit without an eval set, so there is nothing to batch on them), and a single fused add-model-value + gradient kernel
(the wait is what costs on Apple; the fused command buffer takes it without a new kernel).
