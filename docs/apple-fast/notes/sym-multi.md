# sym-multi: MultiClass, PairLogit and YetiRank on symmetric trees (Apple FAST), per-iteration profile

Lane af-sym-multi (branch lane/apple-fast-sym-multi, 2026-10-03). Board lanes gbdt-multiclass (taximc K=4,
istellamc K=5), gbdt-rank-pairlogit (istellarank), gbdt-rank-yetirank (istellarank); 500 symmetric trees, depth 6,
Newton, leaf_estimation_iterations 1 (tools/speed_gbdt_arm.py:2055). Everything below is read from the code on
main at 8897404da; nothing was measured on this laptop.

## Already on main (FAST + Apple defaults; do not redo)
- PairLogit: pairs enumerated on the device per call, one block per query group (`gbdt/targets/kernel/pair_logit_group.mojo`,
  `PAIRLOGIT_GROUP_FUSED`); the estimation's first evaluation scatters the search call's sums (`PAIRLOGIT_EST_REUSE`).
- MultiClass: the Hessian's K rows in one launch, one reduce, one copy, one wait (`MULTICLASS_HESSIAN_BATCH`,
  `pointwise_oracle.mojo:1284`); the value/der pass was already one launch over classes.
- YetiRank: one 256-thread block per task (`YETI_BLOCK_PARALLEL`), the estimation reuses the search gradient
  (`YETI_EST_REUSE_SEARCH`, `yeti_rank.mojo:162`).
- The symmetric level loop is BLIND (one drain per tree, `greedy_search_helper.mojo:5438` DEVIATION 94): no host
  readback per level for any of the three losses.

## MultiClass, per tree (K classes, approx_dim = K - 1, stat_count = K)
Search (`doc_parallel_boosting.mojo:2313`):
1. `multilogit_val_and_first_der_kernel[search=True]` (1 launch): per row reads the K-1 cursor planes THREE times
   (max, sum of exps, the der loop), K-1 exps for the sum and K-1 again for the ders; writes the weight plane and K-1
   der planes of `stats`.
2. two `deterministic_sum_lanes_kernel` (fv, mags), one copy of 2 floats, one `synchronize` (the fixed-point scale
   readback, `:2433`).
3. `run_tree_layout_traced`: per level one hist pass over the K stat planes (grid z = K walks of the one-byte
   kernel, `launch_one_byte` with `gz = stat_count`, `greedy_search_helper.mojo:2487`; every walk re-reads the
   compressed index), `compute_partition_stats` (2 launches, z = K), the score kernel, split, reorder of K planes
   (`launch_reorder_in_leaves`), partition update; no drain. One drain at the tree's tail (`:6118`).
Estimation (`_estimate_and_apply`, `doc_parallel_boosting.mojo:740`): 3 gathers (target, cursor planes, no
weights), 2 uploads (offsets, sizes), `make_bin_optimized_oracle` (identity fill, bins fill, leaves copy; no weight
fold since the board data has no weights), then the Newton walker (iterations 1 = move, value+der, der2, one step):
4. `move_to`: 1 upload (zero shift) + `add_bin_model_value_kernel` (1 launch).
5. `_write_multi_dim_value_and_first_derivatives` (`pointwise_oracle.mojo:760`): val/der kernel (1 launch, same
   3-fold plane reads), `compute_partition_stats` over K-1 columns (2 launches), 2 copies, 1 `synchronize`; host
   reconstructs the pinned class gradient.
6. `_write_blocked_second_derivatives` (`:1043`): `multilogit_second_der_all_rows_kernel` (1 launch; per row K-1
   planes read, then K rows x (row+1) exps recomputed = K(K+1)/2 + 2K exps per row, 15 at K=4, 25 at K=5),
   `compute_partition_stats` over K(K+1)/2 columns (2 launches), 1 copy, 1 `synchronize`; host mirrors the
   triangle, Cholesky per leaf on the host (64 leaves x K^2).
7. apply: 1 upload of the leaves + `add_model_value_kernel` (grid z = approx_dim, already all classes) +
   1 `synchronize` (the task's tail drain).
Total per tree: 4 waits (mags, tree tail, value/der, der2) + apply's tail = 5 drains; ~15 launches outside the
level loop. Host loops over K: the gradient reconstruction and the Hessian mirror (O(leaves x K^2), cheap), the
Cholesky; no host loop over rows. The class dimension is NOT a host loop of launches anywhere (grid z carries it).

Where the next layer is: (a) steps 5 and 6 evaluate the same point (iterations 1; and the walker always asks
der2 right after value/der at the same point): one fused launch, one reduce over (K-1) + K(K+1)/2 columns, one
copy, one wait removes 3 launches and 1 drain per tree; (b) every multilogit kernel recomputes the softmax per
class from global memory; caching the K-1 exps in registers turns K^2 exps and ~3K plane loads per row into K
exps and K-1 loads; (c) the hist pass's K walks each re-read the compressed index: a K-stat hist pass would read
it once, but the one-byte 8-bit kernel's shared-memory accumulator is sized for a stat PAIR at Apple's 32 KiB
threadgroup limit (`pointwise_hist2_one_byte_5bit.mojo:107`), so K stats per block do not fit at 254 borders;
not attempted in this lane (MC_HIST_MULTI dropped, see ab/sym-multi.md).

## PairLogit, per tree (group layout)
Search: `pair_logit_group_kernel[False, so, store_acc=True]` (1 launch, one 256-thread block per query group,
`pair_logit_group.mojo:180`): a thread per document, every OTHER document of the group evaluated against it (each
unordered pair is evaluated TWICE, once from each endpoint: exp, divide, clamp, and log on the winner side);
istella has ~103 documents per query so 60 percent of each block's lanes are idle; then 2 sum-lanes launches, the
mags copy and wait. Level loop as above with stat_count 2. Estimation: `launch_pair_logit_group_reuse` (1 launch
scatter), partition stats (2), 2 copies, 1 wait; move_to as above (1 upload + 1 launch); apply (1 upload +
1 launch + 1 wait); the zero-average host loop over leaves. One `launch_inverse_permutation` per tree.
Per tree: 4 drains, the pair kernel once. Next layer: evaluate each pair ONCE (halves the exp/log work, the
dominant term: ~1e8 pairs per call) and fit the block to the group size (128-thread blocks).

## YetiRank, per tree
Search (`launch_yeti_rank_with[False]`, `yeti_rank.mojo:816`): `launch_compute_group_ids` (1 launch, recomputes the
same row->query map every call), `launch_compute_group_means` (1), `yeti_rank_center_kernel` (1),
`yeti_rank_task_block_kernel` (1, the dominant kernel: 10 permutations x draws + 1024-key merge sort + pairs per
task), `yeti_rank_row_kernel` (1, scatters der/weight to the stats planes, zero fv partials, mags). Then sum-lanes
x2, mags copy + wait. Estimation: reuse scatter (1 launch), partition stats (2), copies, wait; move_to; apply + wait;
the non-reuse evaluation (iterations > 1 only) adds a gather launch before the means. Per tree: 5 target launches
of which 4 are plumbing around the task kernel; 4 drains. Next layer: fold the query means, the centering, the
scatter, the fv zero and the magnitudes into the task kernel (each task holds whole queries, so a per-query mean
is a segmented sum inside the block), and compute the group ids once per fit.
