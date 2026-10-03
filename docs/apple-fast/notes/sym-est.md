# sym-est: leaf value estimation for symmetric trees on the Apple GPU (profile)

Lane `lane/apple-fast-sym-est` (cut from origin/main 8897404da). Board lanes `gbdt-symmetric`, `gbdt-symmetric-1000` on
taxi and istella, `gbdt-ordered` second. The board pins `leaf_estimation_method=Newton`,
`leaf_estimation_iterations=1`, `score_function=Cosine`, `bootstrap_type=No` (tools/speed_gbdt_arm.py:2054-2056,
bench/speed/forest_speed_arm.py:218-234). taxi is RMSE; istella is BINARY (relevance > 0, Logloss).

## Which path each board cell takes (gbdt/methods/doc_parallel_boosting.mojo)

- `need_estimation = method != Simple`, EXCEPT RMSE + Newton + 1 iteration at one permutation (DEVIATION 64,
  `:2073-2087`): the searcher's own Newton step is the leaf. taxi (no categorical columns, so `perm_count == 1`)
  therefore NEVER runs the estimator; its leaves come from `run_tree_layout_traced`'s tail
  (greedy_search_helper.mojo:6090-6108 speculative tail, `:6240-6280` unspeculated): `compute_partition_stats`
  (2 launches: partial grid chunks x leaves x 2, finish grid 1 x leaves x 2) over the FINAL partition,
  `compute_leaf_values_kernel` (1 launch, f32 Newton step with l2 and the `RegularizeImpl` zero), the
  `h_leaf_values` readback riding the tree's one drain, then `add_model_value_kernel` after the gates.
  Nothing here is recomputed per Newton iteration (there is one). That tail is af-sym-hist/af-sym-iter territory;
  this lane does not touch it.
- istella (Logloss, Newton, 1 iteration): `need_estimation` is True, so every tree runs ONE `_estimate_and_apply`
  task (`:3010-3045`, the learn permutation, the searcher's partition `sizes`/`leaf_offsets`/`row_index`).
- `gbdt-ordered`: `ordered_boosting.mojo` has its own batched estimation (`MOJOLEARN_ORDERED_BATCH_EST`, default
  on); `_estimate_and_apply` is reached only for tasks `estimate_can_batch` refuses. Not a target of this lane.

## `_estimate_and_apply` per tree, single-dim pointwise loss, Newton, 1 iteration (`:738-1048`)

Inputs: `row_index` (bin order, position -> row), `sizes`, `leaf_offsets` (host lists), the live `cursor`.

| step | launches | h2d | d2h | drains | host allocs | full-row passes |
|---|---|---|---|---|---|---|
| gather target, (weights), cursor into bin order (`launch_gather_*`) | 2 (3 weighted) | | | | | 2 (3) |
| upload `d_p_off`, `d_p_sz` from `h_po`/`h_ps` (pool of one) | | 2 | | | | |
| `make_bin_optimized_oracle`: `launch_make_sequence(d_identity)` (never read on this path) | 1 | | | | | 1 |
| `fill_bins_from_partition_kernel` (`d_bins`, read only by MoveTo shifts) | 1 | | | | | 1 |
| `h_leaves` fill + upload; host buffers `h_leaves h_shift h_fv h_part_stats h_multi_stats` allocated PER TREE (no `host_scratch` on this path) | | 1 | | | 5 | |
| weights fold (weighted only): `compute_partition_stats` + readback + DRAIN | 2 | | 1 | 1 | | 1 |
| walker `move_to(zero)`: `h_shift` zero fill + upload, shift deferred (`FUSED_EST_MOVE_2030`, default on Apple FAST) | | 1 | | | | |
| evaluation: `launch_approximate_move_eval[True]` (adds the zero shift, writes der and der2 planes + fv partials) | 1 | | | | | 1 |
| `compute_partition_stats` on the two planes (partial + finish) | 2 | | | | | 1 (reads 2 planes) |
| readback `h_part_stats` (2 x leaves) and `h_fv` (one partial per 256 rows), DRAIN | | | 2 | 1 | | |
| host: fold fv in f32, gradient/Hessian in f64, direction `g / (h + 1e-20f)`, one full step from zero, `RegularizeImpl` | | | | | | |
| upload `d_est`, `add_model_value_kernel` (grid 2 x sm x leaves, cursor += lr * est), DRAIN | 1 | 1 | | 1 | | 1 |
| total (unweighted) | 8 | 5 | 2 | 2 | 5 | 7 |

So istella pays per tree: 8 launches, 5 small uploads, 2 readbacks, 2 full drains, 5 host allocations and frees,
and 7 passes over the rows, of which 4 (two gathers, identity, bins) exist only to feed an oracle that then
evaluates once at the zero point and never shifts. The second drain is structural (the oracle dies past it).

## `leaf_estimation_iterations > 1` (not on the board; the Logloss default when the user sets nothing)

Each try is `move_to` (host shift arithmetic, `h_shift` upload) + fused move-eval (1 pass) + partition stats
(2 launches) + 2 readbacks + DRAIN + host line-search decision (AnyImprovement on the f32 value). Ten iterations
= 11 evaluations, 11 drains, 33 launches per tree. Leaf statistics ARE recomputed from the rows at every
evaluation (that is the walker: the derivatives change with the point); nothing is reused across tries except the
cursor copy.

## Exact (MAE, Quantile, MAPE; not on the board)

`estimate_exact`: residual kernel, float keys, segmented radix sort over bits 10..32 (22 passes x 5 launches),
two gathers, flags fill + mark, segmented scan (several launches), need-weights, 16-step binary search, readback,
drain, then `move_to` + flush. ~130 launches per tree; one drain.

## What the candidates change (defines in `gbdt/methods/leaves_estimation/apple_fast_est.mojo`)

See docs/apple-fast/ab/sym-est.md for the per-define explanation. In short: `EST_STATS_FUSED` moves the
reduce-fold, Newton step and regularize into one device kernel (removes the mid-task drain, the two readbacks,
the host arithmetic and the `d_est` upload); `EST_REUSE_PART` evaluates in ROW order on the live buffers over
the searcher's partition (removes the gathers, the identity fill and the bins fill); `EST_ITERS_DEVICE` runs the
whole walk on the device (one drain per tree instead of one per evaluation); `EST_SHRINK_FUSED` fuses the cursor
add with the next iteration's derivative pass (removes the loop head's full-row pass).
