# lane/apple-fast-ordered: Ordered boosting (gbdt-ordered) under FAST on Apple

NEXT_PASS_TREES.md calls this family `trees-ordered`; the branch is `lane/apple-fast-ordered` (cut from origin/main
829c3fb4a). Written without a Mojo toolchain (cloud peer): the first M3 build of `gbdt` (`bindings/build_gbdt.sh`, FAST)
is the compile check. Every change is compiled under FAST + Apple only and defaults OFF; IDENTICAL compiles main's code
unchanged. Board lane `gbdt-ordered` (binding `gbdt`, `bench/speed/forest_speed_arm.py`), datasets taxi then istella;
the M3 FAST baseline is job `afb3-trees-fast`. Opponent: CatBoost CPU (taxi 99.3 s, istella over the 300 s cap).

Both switches live in `gbdt/methods/ordered_boosting.mojo`, in the "APPLE FAST" section before `fit_ordered`.

| switch | what it changes under FAST on Apple | stage of `fit_ordered` |
|---|---|---|
| `-D MOJOLEARN_ORDERED_FOLD_DERIVS` (`ORDERED_FOLD_DERIVS`) | step 2, the fold derivatives: ONE launch (`_ord_fold_planes_kernel`, `_launch_ord_fold_planes`) over the concatenated fold layout instead of `launch_approximate` + `_ord_scatter_planes_kernel` per fold (2 x n_folds launches a tree, n_folds ~ log2(rows / 100)). Per position: the fold by binary search on the device offset table, the learn permutation's own-order target and weight at the fold position, that fold's cursor, then `pointwise_target_kernel`'s / `cross_entropy_kernel`'s search-mode stores (plane 0 the weight, or `weight * Der2` under NewtonCosine; plane 1 `weight * Der`), through `_ord_point[objective]` (the same `target_der` / `target_der2` and cross-entropy statements, the same `ftz`). Also drops the per-iteration `getenv("MOJOLEARN_ORD_STD_SPLIT")` host read: the split noise fold is the only arm. | derivatives, score std |
| `-D MOJOLEARN_ORDERED_BATCH_EST` (`ORDERED_BATCH_EST`) | step 6, the fold models and the estimation model: every task of the tree (learn permutations x folds, plus the estimation task; 49 at the default 4 permutations and 16 folds) in one reduce launch per cursor buffer (`_ord_bat_reduce_kernel`, grid chunks x (tasks x leaves), reading the device partition's sorted runs and snapshots `_PermPartition.d_sorted / d_seg / d_snap` directly: no `task_partition`, no per-task row_index, no stage-in copies), one leaf kernel over every (task, leaf) (`_ord_bat_leaves_kernel`: the walker's one Newton / Gradient step, `leaf = G / (H + l2 + 1e-20)` when `H + l2 > 0` else 0, `H = sum w Der2` or `sum w` under Gradient, zeroed when `sum w < 1e-20`), one apply launch per cursor buffer (`_ord_bat_apply_kernel`). Skips `_PermPartition.settle` and its drain (the counts stay on the device), the per-task oracle construction and uploads, the lock-step walk drains and the host leaf arithmetic; the estimation task's leaves ride the learn-loss drain and the weak model is added after it. Runs when `estimate_can_batch` (Newton / Gradient, single-dim pointwise loss; `leaf_estimation_iterations` is 1 by `check()`) and the identity trace is off; otherwise the fit takes main's path. | bins/partition, fold estimates, estimation, learn loss |

Either switch keeps each learn permutation's fold cursors in ONE buffer (`ORDERED_CAT_CURSORS`, `_ord_fast_init_cursors`):
`cursors[lp][f]` becomes a `create_sub_buffer` view at the fold's concatenated offset, so every other site reads and
writes the same buffers as before.

Launches and drains per tree, default 4 permutations, depth 8, n_folds 16: main's step 2 is 32 launches, FOLD_DERIVS 1;
main's step 6 is ~8 launches + 4 host uploads per task (~400 a tree) and (walk rounds + 1) drains plus the partition
drain, BATCH_EST is 4 reduce + 1 leaves + 4 apply launches, one readback, no drain of its own. Drains per tree under both:
score-std/magnitudes, the structure search (fused level), the learn loss.

Risky compile sites (no toolchain here):

- `_ord_point[objective]` returns `SIMD[DType.float32, 2]` from a `comptime if` with both arms returning; it calls
  `target_der[objective]` / `target_der2[objective]` (imported from `gbdt/targets/kernel/pointwise_targets.mojo`, with
  `routed_exp` and `pinned_block_sum`) only in the non-cross-entropy arm.
- The objective switches `_launch_ord_fold_planes[second_order]` and `_launch_ord_bat_reduce` restate
  `launch_pointwise_target_kernel`'s `@parameter def _go[obj: Int]() raises` closure over `mut DeviceBuffer` arguments;
  `_launch_ord_bat_reduce` also captures a `mut _PermPartition` and launches a 2-D grid
  (`grid_dim=(ORDERED_BAT_CHUNKS, n_tasks * n_leaves, 1)`, `block_idx.y`).
- `_ord_bat_reduce_kernel` calls `pinned_block_sum[block_size=ORDERED_BLOCK]` three times in a row (every thread).
- `_ord_fast_init_cursors`: `cat.create_sub_buffer[DType.float32](offset, size)` views of a plain
  `enqueue_create_buffer` (the hierarchy/ idiom), the parent kept in `fast_cat` for the fit; `_ord_fast_init_tasks` makes
  one more such view (`fast_est_view`) of its own `fast_leaves` allocation for the readback copy
  (`enqueue_copy(dst_buf=HostBuffer, src_buf=view)`).
- The FAST state is nine lists local to `fit_ordered` (`fast_cat`, `fast_fold_off`, `fast_task_k`, `fast_est_off`,
  `fast_est_k`, `fast_partials`, `fast_leaves`, `fast_est_view`, `fast_h_leaves`), empty off the switches, passed as
  distinct `mut` arguments (main's `dys[lp]` / `parts[lp]` / `slots[slot]` idiom), released after the final drain.
- `fit_ordered`: `fast_on = batch and not trace.enabled` (field of `IdentityTrace`); `if fast_on:` blocks whose body is
  a `comptime if ORDERED_BATCH_EST:` (empty under IDENTICAL); the weak model held in a one-element
  `List[TObliviousTreeModel]` across the learn-loss drain (`weak^` moved in both arms of an if/else).
- `from std.sys.info import has_apple_gpu_accelerator`, `GLOBAL_NUMERIC_MODE` / `NUMERIC_FAST` / `identical_mul_add`
  from `checks.numerics`, `LEAF_ESTIMATION_GRADIENT` from `gbdt.options.catboost_options`.

Not changed: the structure search (already one fused level and one drain per tree under `PW_FUSED_LEVEL`), the device
partition sort (`_PermPartition.enqueue_sort`, already one launch set per permutation), the bootstrap, the score noise.
Next if both win: the score-std / magnitude readback deferred to the structure search's drain (needs `choose_scale` on
the device under FAST), and the per-permutation partition sorts as one launch set over every permutation.
