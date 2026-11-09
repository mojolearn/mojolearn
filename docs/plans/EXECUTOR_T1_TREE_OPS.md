# Executor T1 ops: the tree-builder ops (API note for T2)

Lane `t1-tree-builder-ops` (2026-11-08), plan rows "GBDT rounds" and "RandomForest / ExtraTrees / DecisionTree
levels" of docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 4 (Andrew 2026-11-08: tree builders are in the
refactor, the plan's "later" is overruled). This lane adds the ops; lane T2 moves the GBDT round loop
(`gbdt/methods/doc_parallel_boosting.mojo fit_with_test`, `for iteration in range(n_estimators)`) onto
`bounded_device_loop` and deletes the host round trips listed in section 5.

## 1. The contract

The N7 contract (docs/plans/EXECUTOR_N7_OPS.md) with one refinement for tree rounds:

- **Enqueue only.** No op reads device memory back or drains. Setup that drains (pool construction, the fold
  arm's first-tree caches in the searcher) happens before the loop.
- **The gate.** Every op takes `gate: DeviceBuffer[DType.int32]`, word 0 the loop's done word
  (`LoopRegisters.words` in a loop, `core.step_gate.step_gate_open_word` outside).
- **Carried vs round-local.** An op that writes carried state (bootstrap seeds, learn cursor, held-out cursor,
  loss history, detector state) is gated: past the stop it writes nothing. An op that writes only round-local
  scratch (subsets, histograms, winner slots, partition buffers) is NOT gated: a round enqueued past the stop
  recomputes scratch nobody reads. This keeps the gate off the searcher's ~15-25 + 6 depth launches per round
  without changing a word. `gbdt/tree_ops.mojo` records which is which (`TreeOpInfo.gated`).
- **Status words.** Ops that can refuse write a positive code in word 0 of their status buffer; join the loop
  with `step_fail_joins_loop(ctx, registers.words, status, it)` right after the op; after the loop
  `registers.failure()` is the code (raise by name, see section 3).
- **Bits.** Every op body is the existing kernel, called unchanged from a gated wrapper kernel (`if gate[0] != 0:
  return; body(...)`, block-uniform) or launched as before. The only new arithmetic is integer (gate walk,
  records, partition status) or soft binary64 (detector), section 4.

## 2. The table (`gbdt/tree_ops.mojo`)

`tree_op_info(code)` / `tree_op_name(code)`: one `comptime for` over the codes, no `elif` dispatch. Use
`tree_op_name` for `loop_lap` timer names.

| code | name | entry | gated | status | does |
|---|---|---|---|---|---|
| 0 `TREE_OP_BOOTSTRAP` | tree_bootstrap | `gbdt/gpu_util/kernel/bootstrap.mojo tree_op_bootstrap(ctx, gate, kind, seeds, stats, n_rows, param, mags, compute_mags, stat_count)` | yes (seeds) | no | `launch_bootstrap` behind the gate (`bootstrap_gated_kernel` calls `bootstrap_kernel`); the mags fold (`deterministic_sum_lanes_kernel[2]`) after it is round-local |
| 1 `TREE_OP_SEARCH` | tree_search | `fit_oblivious_tree_structure_traced(..., enqueue_only=True, gate=words)` (`oblivious_tree_doc_parallel_structure_searcher.mojo`) | no | yes (structure word 0) | the level loop: per level the histogram (segmented reduce by bin, `calcer.submit_compute*`), the split argmax (`compute_optimal_split_dev*` + `pw_resolve_pack_bins_kernel` / `launch_pw_pack_winner`), the stable partition (`split_subsets_from_desc`); then the gate walk; no drain; returns an empty list, structure in `pool[0].d_structure` |
| 2 `TREE_OP_GATE_WALK` | tree_gate_walk | `gbdt/methods/kernel/tree_gate_walk.mojo tree_op_gate_walk(ctx, gate, winners_ids, feat_table, max_depth, structure)` | yes | yes | the post-tree sentinel / `HasSplit` walk (the host loop it replaced, statement for statement) |
| 3 `TREE_OP_TREE_RECORDS` | tree_records | `tree_op_tree_records(ctx, gate, structure, feat_table, records)` | yes | no | the five per-level apply planes (`TreeRecords`) from the structure and a feature table built for the target's compressed index |
| 4 `TREE_OP_LEAF_BINS` | tree_leaf_bins | `tree_op_leaf_bins(ctx, gate, structure, records, cindex, n_rows, out_bins)` | yes | no | `compute_bins_kernel`, depth from the structure word |
| 5 `TREE_OP_PARTITION` | tree_partition | `doc_parallel_leaves_estimator.mojo tree_op_partition(ctx, gate, parts, n_rows, n_leaves, status)` | no (status gated) | yes | stable radix partition of `parts.bins` by leaf, bounds and sizes left on the device in `parts.d_part` (offsets `[0, nl)`, sizes `[nl, 2nl)`, total, out-of-range flag); status 1 out of range, 2 size sum |
| 6 `TREE_OP_APPLY` | tree_apply | `tree_op_apply(ctx, gate, structure, records, cindex, leaf_values, n_rows, cursor, approx_dim, cursor_stride)` | yes (cursor) | no | the tree gather: `compute_bins_and_add_kernel`, depth from the structure word; depth 0 adds nothing (as `_apply_last_tree_to_test`) |
| 7 `TREE_OP_HELDOUT_LOSS` | tree_heldout_loss | `doc_parallel_boosting.mojo tree_op_heldout_loss(ctx, gate, test, objective, num_classes, alpha, logloss_border, approx_dim, losses, it)` | yes (history) | no | `_test_loss_enqueue` at slot 0, then `-double(fv) / double(n)` in soft binary64 into `losses[it - 1]` (UInt64 bit pattern): the held-out loss as a program value |
| 8 `TREE_OP_OD_STEP` | tree_od_step | `overfitting_detector.mojo od_op_step(ctx, words, state, losses, it)` (+ `OdDeviceState(ctx)`, `od_op_init(ctx, state, detector)` once) | yes | criterion | `add_error` + `is_need_stop` on the device; sets `STEP_DONE`, `STEP_STOP_ITER = it` (the round's tree is kept, as the host `break` keeps it) |

Structure record (`tree_structure_words(max_depth)` Int32): word 0 status (level + 1 of a sentinel level),
word 1 kept depth, then (feature id, bin, split type) per split. Feature table: 4 words per feature
`(offset * n_rows_of_that_cindex, mask, shift, one_hot)`, the searcher's `PointwiseTreeWorkspace.d_feat_table`
for the learn index; T2 builds one more for the held-out index (`layout_for_test`, `test.n_rows`), once per fit.

## 3. One round as a `LoopStep` (what T2 writes)

```
enqueue_step(ctx, registers, it, timers):
    gradients / weak target            (existing launches; round-local stats; the learn loss into its slot)
    TREE_OP_BOOTSTRAP                  (when bootstrap is on)
    TREE_OP_SEARCH                     -> pool.d_structure;  step_fail_joins_loop(structure)
    TREE_OP_TREE_RECORDS (learn table) -> records_learn
    TREE_OP_LEAF_BINS                  -> parts.bins
    TREE_OP_PARTITION                  -> parts.vals / d_part;  step_fail_joins_loop(p_status)
    leaf estimation                    (NOT an op yet: section 5, item 3)
    TREE_OP_APPLY (learn cursor, leaf values x learning rate)
    TREE_OP_TREE_RECORDS (test table)  -> records_test
    TREE_OP_APPLY (test cursor)
    TREE_OP_HELDOUT_LOSS               -> losses[it - 1]
    TREE_OP_OD_STEP                    (criterion, last)
```

After the loop: read `registers.failure()` first. A structure failure raises the searcher's message ("best split
is undefined at depth k: every candidate scored non-finite"), with k from the structure word read once; a
partition failure raises `DeviceLeafPartitioner`'s two messages. Then download, once, the structure records and
leaf values of every committed round (keep one structure + leaf slab per round in a `max_rounds`-deep device
array so the model is built from one copy), the loss history, and the detector's `OD_W_BEST_ITERATION`.
`check_every` = the chunk the fused FAST eval used (1024 rounds) is the upper bound; IDENTICAL and FAST may use
the same chunk because nothing depends on it.

## 4. Bits

- No change for ops 0-7 (same kernels, same launches; integer walks). The searcher change (gate walk on the
  device) keeps every structure: identical walk, integer only.
- Detector: the host `OverfittingDetector._update_p_value` now takes `sf64_exp` (`portable_exp64`) instead of
  libm `exp`, so host and device decide the same stop. Only `od_type=IncToDec` can move (a p-value one ulp off
  the threshold); `Iter` and `None` stop on the same round as before. Identity lanes to re-record if any differ:
  the GBDT lanes fitted with an IncToDec detector and held-out rows (catboost-style regressor and classifier with
  `od_type="IncToDec"`), on every column together. The device op's expected-increment maximum is a block fold of
  exact maxima with first-index ties, so it is the host loop's word for any thread count.

## 5. What T2 deletes when it moves

1. The per-round host round trips: the drain of the structure record (`pw.drain` in the searcher, call it with
   `enqueue_only=True`), the host packing and five uploads of the split records (`sym_iter_fast.mojo
   sym_compute_bins_pooled`, `_apply_last_tree_to_test`'s host loops and six uploads), the per-tree
   `DeviceLeafPartitioner.partition` drain (`tree_op_partition` instead), `_test_loss`'s drain and the host
   `detector.add_error` / `is_need_stop` per round, and `_test_loss_flush` / `_EVAL_FV_SLOTS` (the device history
   replaces the FAST batching).
2. `fit_oblivious_tree_structure_traced`'s drain branch once every caller (doc_parallel, ordered, dynamic
   boosting, the checks) takes `enqueue_only=True`; then `h_structure`, `d_gate_open` and the `enqueue_only` flag
   itself go.
3. Owed, not in this lane: **leaf estimation as an op**. `_estimate_and_apply` / the walker
   (`walk_leaf_values`, `estimate_advance`) reads `h_est` on the host per Newton step (doc_parallel_boosting
   `:1652, :1771, :1864`). The plain-Newton arm with `leaf_estimation_iterations = 1` is a device reduce plus a
   divide per leaf and fits as an op over `parts.d_part`; the line-search and multi-step arms need the step
   loop as an inner `bounded_device_loop`. Until that op exists the round cannot leave the host, so T2 starts
   with it (or the orchestrator gives it its own lane, T1b).
4. RandomForest / ExtraTrees: no op here. Their per-tree RNG is already a stateless hash of (seed, tree id) and
   the rows / labels gather (`launch_bootstrap_rows_labels`) is round-local, so neither needs a gate; what keeps
   the cycle on the host is `advance_tree` building the sparse tree from the drained split records
   (`ensemble/randomforest.mojo` `flush_splits_downloads` + `ctx.synchronize()` per cycle). The plan rates the
   gain "little" (one drain per cycle shared by the in-flight trees); the next step is a device node-append op
   over the split records, sized like the gate walk. Listed as owed.

## 6. RUN OWED

- Compile: `checks/executor_t1_tree_ops_compile.mojo` on NVIDIA, AMD and Apple, IDENTICAL and FAST; the GBDT
  bindings (the searcher, leaves estimator, bootstrap, detector and doc_parallel_boosting changed).
- NVIDIA + AMD: the GBDT board rows (catboost-style regressor and classifier, symmetric, depthwise, lossguide,
  with and without held-out rows) once, against main: the searcher now drains the 2 + 3 depth structure words
  instead of the 4 depth winner words, and the partition zeroes its bounds with a memset instead of a host fill
  and an upload; expected equal or faster, same bits.
- Identity: the GBDT lanes, in the one global verify; only IncToDec early-stopping fits can move (section 4).
