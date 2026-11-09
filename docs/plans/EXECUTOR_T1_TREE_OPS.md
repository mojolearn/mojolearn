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
| 8 `TREE_OP_OD_STEP` | tree_od_step | `gbdt/overfitting_detector/od_device.mojo od_op_step(ctx, words, state, losses, it)` (+ `OdDeviceState(ctx)`, `od_op_init(ctx, state, detector)` once) | yes | criterion | `add_error` + `is_need_stop` on the device; sets `STEP_DONE`, `STEP_STOP_ITER = it` (the round's tree is kept, as the host `break` keeps it) |
| 9 `TREE_OP_LEAF_ESTIMATE` | tree_leaf_estimate | `doc_parallel_boosting.mojo tree_op_leaf_estimate(ctx, gate, state, parts, targets, weights, has_weights, cursor, objective, alpha, estimator_alpha, logloss_border, l2_leaf_reg, method, num_classes, iters, learning_rate, leaf_out, leaf_scaled, status)` (+ `LeafEstimateOpState(ctx, n_rows, 1 << max_depth, sm)` once; `tree_op_leaf_estimate_supported`) | no (outputs and status gated) | yes (`LEAF_STATUS_HOST_TAIL`) | lane T1b: `_estimate_and_apply`'s single-dimensional device walk from TREE_OP_PARTITION's device words: offsets / sizes to `d_p_off` / `d_p_sz` (`partition_words_u32_kernel`), the three gathers by `parts.vals`, the oracle enqueue-only (pooled device + host scratch, weights folded on the device), `device_walk_enqueue_bounded` (begin + the first `iters` line-search evaluations, steps and accept decisions in `walker_accept_kernel`'s control word), then `leaf_out` = the model's leaves and `leaf_scaled` = leaves x learning rate (correctly rounded, `scaled_copy_kernel`'s product) for TREE_OP_APPLY. Pointwise single-dim losses, Newton or Gradient, weighted or not, any `iters`; QueryRMSE / PairLogit / YetiRank / multiclass keep `_estimate_and_apply` |
| 10 `TREE_OP_SEED_SNAPSHOT` | tree_seed_snapshot | `gbdt/gpu_util/kernel/bootstrap.mojo tree_op_seed_snapshot(ctx, gate, seeds, snapshot)` | yes | no | the bootstrap seeds copied before TREE_OP_BOOTSTRAP draws, so a round stopped by `LEAF_STATUS_HOST_TAIL` can be rerun from the seeds it started with |
| 11 `TREE_OP_FOREST_BATCHES` | forest_batches | `ensemble/decisiontree/batched_levelalgo/builder.mojo Builder._enqueue_loop_batches` (RandomForest, DecisionTree); ExtraTrees `_et_run_depthwise`'s cycle body | queue (inert once empty) | overflow header word | lane T1b, registration only: the node append (`launch_loop_push`; ET `enqueue_push`) and the in-flight tree's bookkeeping (queue head / tail, node count, depth, overflow) were already device words (cpu4-forest); `LOOP_K` (ET `ET_LOOP_K`) batches between two 16-word header reads, which are the loop's chunk read; queue-empty is the criterion |
| 12 `TREE_OP_FOREST_FINISH` | forest_finish | `Builder._finish_tree_device` + `Builder.collect_finished` (+ randomforest `_take_finished`) | no | no | lane T1b: the finished tree's node table to `SparseTreeNode`s, the leaf pass and the two bulk downloads ENQUEUED; the host model is assembled after the caller's next drain (the forest cycle's one drain). The finish's own drain per tree is deleted |

Structure record (`tree_structure_words(max_depth)` Int32): word 0 status (level + 1 of a sentinel level),
word 1 kept depth, then (feature id, bin, split type) per split. Feature table: 4 words per feature
`(offset * n_rows_of_that_cindex, mask, shift, one_hot)`, the searcher's `PointwiseTreeWorkspace.d_feat_table`
for the learn index; T2 builds one more for the held-out index (`layout_for_test`, `test.n_rows`), once per fit.

## 3. One round as a `LoopStep` (what T2 writes)

```
enqueue_step(ctx, registers, it, timers):
    gradients / weak target            (existing launches; round-local stats; the learn loss into its slot)
    TREE_OP_SEED_SNAPSHOT              (when bootstrap is on: seeds -> seed_snapshot)
    TREE_OP_BOOTSTRAP                  (when bootstrap is on)
    TREE_OP_SEARCH                     -> pool.d_structure;  step_fail_joins_loop(structure)
    TREE_OP_TREE_RECORDS (learn table) -> records_learn
    TREE_OP_LEAF_BINS                  -> parts.bins
    TREE_OP_PARTITION                  -> parts.vals / d_part;  step_fail_joins_loop(p_status)
    TREE_OP_LEAF_ESTIMATE              -> leaf_out (into the round's slab) / leaf_scaled;  step_fail_joins_loop(l_status)
    TREE_OP_APPLY (learn cursor, leaf_scaled)
    TREE_OP_TREE_RECORDS (test table)  -> records_test
    TREE_OP_APPLY (test cursor)
    TREE_OP_HELDOUT_LOSS               -> losses[it - 1]
    TREE_OP_OD_STEP                    (criterion, last)
```

After the loop: read `registers.failure()` first. `LEAF_STATUS_HOST_TAIL` (1 << 20, never a structure or
partition code) is not an error: the walk of round `it = STEP_STOP_ITER` accepted none of its first `iters`
line-search evaluations, so the walker's never-accepted tail (`iteration < 100`, one drain per evaluation)
owns that round. Every carried op of round `it` after the leaf op, and every op of later rounds, was gated off,
so the cursors, the loss history and the detector are round `it - 1`'s; copy `seed_snapshot` back into the
bootstrap seeds, run round `it` on the host-driven path (`_estimate_and_apply`, `device_walk_estimate` owns the
tail), then restart `bounded_device_loop` at `it + 1`. Same bits as the host-driven fit. A structure failure
raises the searcher's message ("best split
is undefined at depth k: every candidate scored non-finite"), with k from the structure word read once; a
partition failure raises `DeviceLeafPartitioner`'s two messages. Then download, once, the structure records and
leaf values of every committed round (keep one structure + leaf slab per round in a `max_rounds`-deep device
array so the model is built from one copy), the loss history, and the detector's `OD_W_BEST_ITERATION`.
`check_every` = the chunk the fused FAST eval used (1024 rounds) is the upper bound; IDENTICAL and FAST may use
the same chunk because nothing depends on it.

## 4. Bits

- TREE_OP_LEAF_ESTIMATE: the walk's kernels, order and inputs are `_estimate_and_apply`'s device walk. Two
  differences, both bit-inert: the oracle is told every leaf may hold all `n_rows` rows (it sizes two grids from
  the widest leaf, the weight fold's `row_bound` and the shift-add width; the full grid gives the same bits,
  `compute_partition_stats`' docstring), and the walk runs over `1 << max_depth` leaves when the round's depth is
  a device word (the leaves past `1 << depth` are empty and come out 0; no fold crosses leaves). What does move
  bits is the PARTITION the walk sums in: TREE_OP_PARTITION orders each leaf's rows by ascending row id, the
  order `partition_from_bins` gives every Depthwise / Lossguide task already; the symmetric learn permutation
  sums in the searcher's partition order today, so when T2 feeds the learn task through TREE_OP_PARTITION the
  symmetric GBDT identity lanes move (every column together: the host column's symmetric task must then take the
  ascending order too). The learn cursor add moves from `add_model_value_kernel` (partition order, `est * lr`
  in the kernel) to TREE_OP_APPLY (bins order, the correctly rounded `est * lr`): T2 checks the two adds are
  the same float operation or takes the change on every column.
- TREE_OP_FOREST_FINISH: no change (the same launches and copies; the host assembly is the same statements
  after a later drain).
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
3. Done in lane T1b: **leaf estimation is TREE_OP_LEAF_ESTIMATE**. When T2 moves the round onto it, delete
   from the op path: the host pack of `leaf_offsets` / `sizes` and the `h_po` / `h_ps` uploads, the walk's drain
   and `h_ctl` read (`device_walk_estimate`'s GBDT_WALK_DEV_STEP arm), the `h_est` copy and the task's tail drain,
   `walk_leaf_values`, `estimate_advance` / `PendingEstimation` and `finish_deferred_estimation` for the tasks
   the op covers. `_estimate_and_apply` stays for the tasks the op refuses (grouped and multiclass losses, Exact)
   and as the host tail's round (section 3); when those tasks get their ops it goes too.
   The never-accepted tail stays host-driven on purpose: a loop whose trip count is a device word (up to 100
   evaluations, each ~7 launches) needs a device-side conditional launch, which Mojo / MAX does not expose
   (CUDA graph conditional nodes, HIP has none). **Ask for Modular:** a device-side conditional or while node
   in `DeviceContext` graphs (or an equivalent "launch if word" primitive) on NVIDIA, AMD and Apple. Emulating
   it with 93-99 gated evaluations per round (~650 empty launches) would cost more than the drain it removes,
   and capping the tail would change the walk, so neither is done.
4. RandomForest / DecisionTree / ExtraTrees: the premise above was wrong. On every fit the node append
   (`launch_loop_push`; ET `enqueue_push`) and the in-flight tree's bookkeeping (queue, node count, depth,
   overflow) already run on the device (cpu4-forest's device level loop); `advance_tree` builds a tree from
   drained split records only on the host-queue replay (check sabotage instantiations, identity traces). The
   host's part per cycle is the 16-word header read every `LOOP_K` batches, which is the loop's own chunk read
   (row 11). Lane T1b removed what was left: the finish's own drain per finished tree (row 12; the forest
   cycle's one drain settles it, `Builder.train` collects after its own). Owed for T2: the ExtraTrees cycle's
   host-grown FIFO (`lp.ensure`, a host bound per cycle) and its per-group finish are the same shape and were
   not touched here; the forest's slot scheduling (begin the next tree when one finishes) stays a host
   driver loop over trees.

## 6. RUN OWED

- Compile: `checks/executor_t1_tree_ops_compile.mojo` on NVIDIA, AMD and Apple, IDENTICAL and FAST; the GBDT
  bindings (the searcher, leaves estimator, bootstrap, detector and doc_parallel_boosting changed).
- NVIDIA + AMD: the GBDT board rows (catboost-style regressor and classifier, symmetric, depthwise, lossguide,
  with and without held-out rows) once, against main: the searcher now drains the 2 + 3 depth structure words
  instead of the 4 depth winner words, and the partition zeroes its bounds with a memset instead of a host fill
  and an upload; expected equal or faster, same bits.
- Identity: the GBDT lanes, in the one global verify; only IncToDec early-stopping fits can move (section 4).
- Lane T1b: compile `checks/executor_t1_tree_ops_compile.mojo` (now instantiates TREE_OP_SEED_SNAPSHOT and
  TREE_OP_LEAF_ESTIMATE) on NVIDIA, AMD and Apple, IDENTICAL and FAST; the GBDT binding (doc_parallel_boosting,
  device_walker, doc_parallel_leaves_estimator, bootstrap changed) and the RandomForest / DecisionTree bindings
  (builder.mojo, randomforest.mojo changed; ExtraTrees untouched). NVIDIA + AMD: the RandomForest and
  DecisionTree board rows (classifier and regressor) once against main: one drain fewer per finished tree,
  same bits expected; Apple identity unaffected. The leaf op has no caller until T2, so no GBDT row moves yet.

## 7. Lane T2 (`t2-trees-on-executor`, 2026-11-09): what moved, what stays and why

**Moved.** `gbdt/methods/doc_parallel_boosting.mojo`: `SymDeviceRounds` (a `LoopStep`) is section 3's round, and
`fit_sym_rounds_on_device` runs it on `bounded_device_loop` with the host-tail rerun (seed snapshot back into the
seeds, `run_host_tail_round` with `_estimate_and_apply`, restart at `s + 1`). `fit_with_test` routes every
`sym_device_rounds_eligible` fit there and runs its host loop zero times. Chunk: 16 rounds with an active detector,
128 without (cost rule in the comment). The fixed-point scale is a device word (`choose_scale_kernel` into the
searcher's `std_scale_word`), so the per-round magnitudes drain goes too. The model, the learn and held-out loss
histories and the detector's best iteration come home in one download after the loop.

**Eligible:** pointwise searcher, SymmetricTree, one permutation, `feature_fraction == 1`, `random_strength == 0`,
no identity trace, a single-device plan, and a loss TREE_OP_LEAF_ESTIMATE covers (single-dim pointwise, Newton or
Gradient). **Refused (they stay on the host loop, each owed its own op):** QueryRMSE / PairLogit / YetiRank, the
multiclass family and MultiRMSE, Exact / Simple, CTR estimation permutations, feature sampling (projection drains),
`random_strength` (host std-dev reduction), traced fits, multi-device plans, the greedy searcher (the default
SymmetricTree arm, `run_tree_layout`: no enqueue-only op), Depthwise / Lossguide, and FAST + Apple (no leaf-op arm,
its `apple_fast_est` / SYM_ITER pool / fused eval stay, as briefed).

**Bits.** The pointwise-searcher symmetric GBDT identity lanes move on every column together: leaf sums in the
partition op's ascending-row order, and the learn cursor add is `cursor + round(leaf * rate)` (TREE_OP_APPLY)
instead of `add_model_value_kernel`'s IDENTICAL `fma(leaf, rate, cursor)`. The fma note of section 4 was checked:
the two are not the same float operation, so the change is taken everywhere. The stored model leaves stay
`leaf * rate` on the host, as before. A/B define `-D MOJOLEARN_GBDT_DEVICE_ROUNDS_OFF` (host loop), row in
`docs/apple-fast/EXPERIMENTS.md`.

**Section 5 deletions, status.** Nothing in section 5.1 to 5.3 has lost its last caller, so nothing was deleted
there. Each one still serves a refused fit or another family:
- the searcher's drain branch, `h_structure`, `d_gate_open`, `enqueue_only` (5.2): the host loop's pointwise arm
  (the refused fits above), `ordered_boosting.mojo:3277`, `dynamic_boosting.mojo:389` and the checks still take
  the split list;
- `compute_bins_for_model` / `partition_from_bins` / `_apply_last_tree_to_test` / `_test_loss` / host
  `detector.add_error` (5.1): the greedy SymmetricTree arm (default), Depthwise / Lossguide, every refused fit;
- `_test_loss_flush` / `_EVAL_FV_SLOTS` (5.1): `GBDT_EVAL_FUSED` is Apple FAST, which keeps today's path;
- `estimate_advance` / `PendingEstimation` / `walk_leaf_values` / `finish_deferred_estimation` (5.3): the CTR
  permutation batch (`doc_parallel_boosting.mojo` `walk_leaf_values(perm_est_ws[...])`), ordered boosting
  (`_estimate_complete`, `estimate_workspace`), and `IDN_GBDT_SYM_EST_DEFER_DRAIN` on the greedy arm.
They go when the greedy searcher has an enqueue-only search op (then the default SymmetricTree fits move and
5.1's host packers lose their last oblivious caller) and when ordered boosting takes the same round step. After the
NVIDIA + AMD A/B, the `_OFF` arm is deleted (winner) or the device rounds block is (loser).

**ExtraTrees (task 2).** The cycle already runs on the device (cpu4-forest): pop / stage / search / partition /
push per batch, `ET_LOOP_K` batches per header read. Done here: `EtDeviceLoop.ensure` (the host-grown FIFO and node
arena) no longer synchronizes; a growth enqueues its copies and header upload and retires the replaced buffers to
the next header drain (three drains per growth deleted, same bits). The growth decision stays on the host: Mojo
has no device-side allocation, and a worst-case arena (about 2 x rows nodes per tree for the whole group) is
gigabytes at the group cap. The per-group finish (`download_trees`) stays: it is already one download and one
drain per group, a default forest is one group, and deferring it past the next group's first drain would keep two
groups' device arenas alive at once, which is the memory the grouping exists to bound.

**RandomForest / DecisionTree (task 3): left as they are.** A `LoopStep` wrapper removes no drain: node append
and bookkeeping are device words, the 16-word header read every `LOOP_K` batches is the loop's own chunk read, and
T1b already took the finish's per-tree drain out (row 12). The slot scheduling (begin the next tree when one
finishes) is a host driver loop over trees and needs the header to decide.

**RUN OWED (T2).**
- Compile once (orchestrator): the GBDT binding (`doc_parallel_boosting.mojo` now imports
  `autograd/program_loop_device.mojo`, `core/step_gate.mojo`, `gbdt/methods/kernel/tree_gate_walk.mojo`,
  `gbdt/overfitting_detector/od_device.mojo`), both with and without `-D MOJOLEARN_GBDT_DEVICE_ROUNDS_OFF`, NVIDIA,
  AMD and Apple, IDENTICAL and FAST; `checks/executor_t1_tree_ops_compile.mojo`; the ExtraTrees binding
  (`extratrees/impl/decisiontree/batched_levelalgo/builder.mojo`).
- NVIDIA + AMD: the pointwise-searcher GBDT board rows (with and without held-out rows, with an `Iter` and an
  `IncToDec` detector) A = device rounds vs B = `_OFF`; the ExtraTrees classifier and regressor rows vs main.
- Identity (the one global verify): the pointwise-searcher symmetric GBDT lanes move on every column.
