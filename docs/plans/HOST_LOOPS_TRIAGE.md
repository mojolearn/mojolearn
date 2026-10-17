# Triage of the GPU-path host loops (2026-10-17)

The rule is in `docs/plans/GPU_PATH_HOST_LOOPS.md`. On 2026-10-17
`python3 -I tools/hooks/no_host_routes.py --owed` listed 98 `mojo-host-loop`
findings. Each was read (its range and its body) and put in one class:

| class | meaning | count | what this lane did |
|---|---|---|---|
| 1 | driver loop over model or launch structure | 60 (+7 nested) | tagged `# driver-loop: over <what>` (comment only) |
| 2 | readback reduction or readback copy | 32 | listed with its fix below; still owed |
| 3 | data loop over rows or samples | 4 | listed with its fix below; still owed |
| D | device code the rule read as host code | 2 | `# device-helper:` note (comment only) |

The 7 nested class-1 loops were hidden under flagged parents before: the old
rule skipped the body of a flagged loop, and tagging the parent exposed them.
After tagging, `--owed` lists exactly the 36 class-2 and class-3 sites, and
`--tree` reports 67 accepted driver-loop tags. No kernel, host column or
Python changed, and no bits changed.

## Status after lane/host-loops-fix (2026-10-17)

24 of the 36 owed sites are fixed; `--owed` lists the 12 below.

- Copies: one `memcpy` (`doc_parallel_boosting` x3, `apple_fast_est:757`,
  `leaves_estimation_helper`), or widened on the device and copied in bulk
  (`fast_prep` counts and histogram, `ordered_boosting` snapshots).
- Integer reductions and scans on the device: first absent category
  (`Atomic.min`), the partition total, task offsets and task table
  (`ordered_boosting`), leaf sizes and offsets (`leaf_sizes_scan_kernel`).
- Float folds on the device in soft-float64, same order and same bits:
  the lossguide replay sums (`dw_lg_final_sums_kernel`), the deferred
  held-out losses (`_test_loss_div_kernel`), Huber's result words.
- `apple_fast_est:601`: the launch bound is `n_rows` (launch shape only).
- Data loops: `train.mojo` dense codes through `device_dense_codes`; the
  Lanczos start vector by `lanczos_v0_kernel`.
- `fa_fast` (FAST + Apple only): the sweep flag folded on the device; the
  log-likelihood sums use a fixed blocked soft-float64 order (FAST bits).

Closed by lane gbdt-host-loops-12 (2026-10-17); `no_host_routes.py --owed` lists no `mojo-host-loop` site:

- `pointwise_oracle.mojo` (11 sites): the multi-dimensional walk runs on the device
  (`device_walker.device_multi_walk_estimate`: MultiClass, MultiClassOneVsAll, MultiRMSE; Newton or Gradient; any
  iteration count; weighted, deferred weights included; IDENTICAL and FAST). Move, regularize and MoveTo, the gradient
  with MultiClass's pinned component, the value fold (`deterministic_sum_lanes_kernel[1]`'s order, as the
  single-dimensional walk), the AnyImprovement test, the blocked Cholesky (block per leaf in 32 KiB threadgroup memory;
  a wider system per thread over a global scratch, same chains) or the diagonal direction, all soft-float64. The
  oracle's multi-dimensional readbacks are deleted (TOMBSTONES.md "GPU oracle multi-dimensional host walk"); its arm is
  enqueue-only (`enqueue_multi_dim_evaluation`, `enqueue_multi_dim_hessian`). The single-dimensional host interface
  the checks' host walker reads (`:798`, `:1080`, `:1139`, `:1433`) runs its arithmetic in kernels and copies home in
  bulk; Exact's Regularize runs on the device. `_estimate_and_apply` refuses a walk no device walker takes: no GPU
  fit runs `descent_helpers`' host walker. Host column: `gbdt_oracle_multiclass._estimate_multi` (one Newton
  iteration) restates the same statements; unchanged.
- `resident_model.mojo:1024`: the row-major transpose and the CTR expansion (`expand_raw_columns`,
  `expand_for_model_apply`) run on the device together (`gbdt/models/kernel/resident_ctr.mojo`): the input goes up as
  handed over and the model columns land column-major in `d_raw`. Every float32 operation is the correctly rounded
  soft-float64 result narrowed once, so the host's words. Host column: `predict_floats`, unchanged.

Line numbers below are at this lane's head (tags add one line above each
tagged loop).

## Fix notes that apply to many rows

- **Readback copy (class 2, no compute).** The host downloads a per-leaf or
  per-coefficient buffer and copies it element by element into a `List`.
  Fix: one `memcpy` into the list storage (or keep the host buffer as the
  store). No kernel, no bit change.
- **Host Newton leaf step (class 2, `pointwise_oracle.mojo`).** The per-leaf
  derivative sums are read back and the Newton step (mirror the Hessian, add
  lambda, sum over dims, regularize by leaf weight) runs on the host. Fix: a
  device leaf-step kernel that reads the partition stats in place, applies
  the same per-leaf arithmetic in the same order (one add of lambda, the dim
  sum in dim order, the weight compare), and writes the leaf deltas; the
  host reads only the accepted leaf values once per step. The host column is
  the oracle's own CPU arm (`_mirror_second_der_row` and the
  `write_*_derivatives` CPU paths), which must keep the same order.
- **Partition scans (class 2).** Per-leaf sizes or bounds are read back and
  scanned into offsets on the host. Fix: an integer exclusive-scan kernel
  that writes offsets and sizes (and the task table) on the device. Integer
  only: no bit change.

## Class 2: readback reductions and copies

| file:line | range | class | reason | fix |
|---|---|---|---|---|
| `gbdt/ctrs/fast_prep.mojo:403` | `range(unique_values)` | 2 | scans the downloaded category counts for a missing category, copies them | a reduce kernel over `d_counts` returns the first absent category (or -1) as one int32; the counts stay on the device for the CTR kernels. Host column: the dense-code check in `train.mojo` (same error) |
| `gbdt/ctrs/fast_prep.mojo:451` | `range(hist_len)` | 2 | copy-out of the downloaded target histogram | keep `d_hist` on the device for the CTR table builder; if a host consumer remains, one memcpy. Host column: `build_ctr_tables` |
| `gbdt/methods/doc_parallel_boosting.mojo:703` | `range(pending)` | 2 | reads back per-iteration test-loss sums and divides by n on the host | the loss kernel writes `-sum / n` (float64) per iteration; the host copies the pending values and only feeds the detector (one value per iteration). Host column: the CPU test-loss evaluator does the same division |
| `gbdt/methods/doc_parallel_boosting.mojo:1234` | `range(tail.n_values)` | 2 | readback copy of `h_est` into the leaf values | one memcpy |
| `gbdt/methods/doc_parallel_boosting.mojo:1606` | `range(n_leaves)` | 2 | readback copy of `h_est` | one memcpy |
| `gbdt/methods/doc_parallel_boosting.mojo:1739` | `range(n_leaves * approx_dim)` | 2 | readback copy of `h_est` | one memcpy |
| `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:4062` | `range(num_leaves)` | 2 | folds downloaded leaf-slot stats into final-leaf sums (float64, left-first walk) and uploads them again | a kernel, one thread group per final leaf, walks the same `lg_node_*` tree left first in float64 and writes `d_sums64` directly; no readback and no upload. Host column: the CPU lossguide exact replay keeps the same walk |
| `gbdt/methods/leaves_estimation/apple_fast_est.mojo:601` | `range(n_leaves)` | 2 | max over leaf sizes read back from the searcher, used as a launch bound | a max-reduce kernel over `d_p_sz` into one word, or a launch bound that needs no readback (`n_rows`). Launch shape only: no bit change |
| `gbdt/methods/leaves_estimation/apple_fast_est.mojo:757` | `range(n_leaves)` | 2 | readback copy of `h_est` | one memcpy |
| `gbdt/methods/leaves_estimation/doc_parallel_leaves_estimator.mojo:359` | `range(n_leaves)` | 2 | scans downloaded leaf bounds into offsets and sizes | integer exclusive-scan kernel on the bounds; one copy of the sizes if a launch needs them |
| `gbdt/methods/leaves_estimation/doc_parallel_leaves_estimator.mojo:431` | `range(n_leaves)` | 2 | same scan (`partition_collect`) | same kernel |
| `gbdt/methods/leaves_estimation/leaves_estimation_helper.mojo:249` | `range(bin_count)` | 2 | readback copy of the quantile point `h_point` | one memcpy |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:256` | `range(column_count)` | 2 | mirrors the downloaded Hessian row into both triangles, adds lambda | a mirror kernel writes the full per-bin blocks on the device (see the Newton note). Host column: `_mirror_second_der_row` |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:798` | `range(self.bin_count)` | 2 | copies downloaded per-leaf der and der2 | device leaf-step kernel (Newton note) |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:915` | `range(self.bin_count * self.cursor_dim)` | 2 | copies downloaded multi-dim derivatives | device leaf-step kernel |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:931` | `range(self.bin_count * self.cursor_dim)` | 2 | copies the read-back derivatives into the gradient | device leaf-step kernel |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:934` | `range(self.bin_count)` | 2 | sums read-back derivatives over dims (the pinned plane) | the multi-dim stats kernel writes `-sum` over dims in dim order. Host column: the oracle CPU arm, same order |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:954` | `range(ml_blocks)` | 2 | float32 sum of downloaded block partials | a final-fold kernel over the partials in block order (float32), one scalar readback. Host column folds the same order |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:1080` | `range(self.bin_count)` | 2 | der2 from the leaf weights plus lambda on the host | device leaf-step kernel |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:1124` | `range(self.bin_count * self.cursor_dim)` | 2 | der2 from downloaded multi stats plus lambda | device leaf-step kernel |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:1139` | `range(self.bin_count)` | 2 | copies the cached read-back der2 | device leaf-step kernel |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:1295` | `range(hbs)` | 2 | blocked Hessian mirror from downloaded stats | the mirror kernel of `:256` |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:1433` | `range(self.bin_count)` | 2 | zeroes leaves whose read-back weight is under `min_leaf_weight` | the compare runs in the device leaf-step kernel |
| `gbdt/methods/ordered_boosting.mojo:1395` | `range(n_leaves)` | 2 | sums downloaded per-leaf totals to check every row landed | a reduce kernel (or `_ord_snap_kernel`) writes the total; one scalar readback for the check |
| `gbdt/methods/ordered_boosting.mojo:1404` | `range(n_leaves)` | 2 | readback copy of the segment starts | keep `d_seg` on the device; the task-table kernel reads it |
| `gbdt/methods/ordered_boosting.mojo:1408` | `range(nb * n_leaves)` | 2 | readback copy of the prefix snapshots | keep `d_snap` on the device |
| `gbdt/methods/ordered_boosting.mojo:1435` | `range(n_leaves)` | 2 | scans the read-back snapshot into offsets and the task table, then uploads it | an integer scan kernel builds `dtab` from `d_snap` and `d_seg`; one copy of the sizes if the estimator needs them |
| `x_decomp/fa_fast.mojo:900` | `range(h)` | 2 | scans one downloaded rotation flag per block for convergence | `rs_round_kernel` ORs into one flag word; one scalar readback per check |
| `x_decomp/fa_fast.mojo:915` | `range(nc)` | 2 | float64 sum of downloaded log terms (`_fa_ll`, slog) | a reduce kernel sums slog, unexp and plog in float64 in index order (the same fold on every vendor) and returns the log-likelihood as one value. Host column: `_dsum` folds in the same order |
| `x_decomp/fa_fast.mojo:918` | `range(nc, d)` | 2 | same, unexp | same kernel |
| `x_decomp/fa_fast.mojo:922` | `range(d, dd2)` | 2 | same, plog | same kernel |
| `x_linear/huber_fast.mojo:470` | `range(p)` | 2 | copies the downloaded theta into the result words | `hg_result_kernel` writes the whole result layout to a device buffer; one copy into `res` |

## Class 3: data loops

| file:line | range | class | reason | fix |
|---|---|---|---|---|
| `gbdt/resident_model.mojo:1022` | `range(n_rows)` | 3 | transposes the row-major predict input on the host for the CTR expansion | a transpose kernel; the CTR expansion (`expand_for_model_apply`, still host) must move to the device with it. Exact copy: no bit change. Host column: `predict_floats` |
| `gbdt/train.mojo:2187` | `range(n_rows)` | 3 | dense category codes and the seen flags per row on the host | route this branch through `device_dense_codes` (`code_histogram_kernel` plus the first-absent reduce of `fast_prep.mojo:403`). Host column: the CPU route's `dense_category_code` loop (exact ints) |
| `gbdt/train.mojo:2202` | `range(unique_values)` | 3 | the missing-category check over the per-row seen flags | goes with `:2187` |
| `spectral/impl/sparse/solver/detail/lanczos.mojo:2728` | `range(n)` | 3 | the hashed Lanczos start vector, one element per row, on the host | a kernel computes the same splitmix hash per index (integer ops and one float multiply: exact) into the device `v0`. Host column: `spectral/host/spectral_oracle.mojo:lanczos_v0` (same bits) |

## Class D: device code the rule read as host code

| file:line | range | class | reason | what changed |
|---|---|---|---|---|
| `neighbors/estimator.mojo:1003` | `range(d)` | D | `_knn_refine_sq` is a per-thread helper that only a kernel calls | `# device-helper:` on its def line (the hook checks every caller is a kernel) |
| `x_ann/ivf_scan_device.mojo:850` | `range(np)` | D | `_pq_fused_seq` runs inside the fused PQ kernel (`if t == 0`, NaN-candidate fallback) | `# device-helper:` on its def line. It is still a one-thread walk for one query when a candidate is NaN; that belongs to the serial-GPU rules, not this one |

## Class 1: driver loops (tagged)

| file:line | range | class | reason |
|---|---|---|---|
| `bindings/_mojolearn_resample.mojo:424` | `range(k)` | 1 | over the gathered arrays (argument spans) |
| `gbdt/ctrs/fast_prep.mojo:549` | `range(max_borders_count)` | 1 | over the CTR border slots (a config bound) |
| `gbdt/ctrs/fast_prep.mojo:984` | `range(len(configs))` | 1 | over the CTR configs |
| `gbdt/ctrs/fast_prep.mojo:988` | `range(len(configs))` | 1 | over the CTR configs |
| `gbdt/ctrs/fast_prep.mojo:993` | `range(i, len(configs))` | 1 | over the CTR configs |
| `gbdt/ctrs/fast_prep.mojo:1016` | `range(len(configs))` | 1 | over the CTR configs |
| `gbdt/ctrs/fast_prep.mojo:1025` | `range(len(configs))` | 1 | over the CTR configs |
| `gbdt/ctrs/fast_prep.mojo:1029` | `range(len(configs))` | 1 | over the CTR configs |
| `gbdt/ctrs/fast_prep.mojo:1036` | `range(len(groups[g]))` | 1 | over the configs of one CTR group |
| `gbdt/ctrs/fast_prep.mojo:1040` | `range(len(groups[g]))` | 1 | over the configs of one CTR group |
| `gbdt/ctrs/fast_prep.mojo:1048` | `range(len(configs))` | 1 | over the CTR configs |
| `gbdt/ctrs/fast_prep.mojo:1065` | `range(len(configs))` | 1 | over the CTR configs |
| `gbdt/ctrs/fast_prep.mojo:1069` | `range(len(configs))` | 1 | over the CTR configs |
| `gbdt/ctrs/fast_prep.mojo:1076` | `range(len(groups[g]))` | 1 | over the configs of one CTR group |
| `gbdt/ctrs/fast_prep.mojo:1080` | `range(len(groups[g]))` | 1 | over the configs of one CTR group |
| `gbdt/ctrs/fast_prep.mojo:1087` | `range(len(configs))` | 1 | over the CTR configs |
| `gbdt/methods/doc_parallel_boosting.mojo:483` | `range(depth)` | 1 | over the tree depth (split records) |
| `gbdt/methods/doc_parallel_boosting.mojo:501` | `range(p_values)` | 1 | over the last tree's leaf values (leaves x approx_dim) |
| `gbdt/methods/doc_parallel_boosting.mojo:555` | `range(n_values)` | 1 | over the last tree's leaf values (leaves x approx_dim) |
| `gbdt/methods/doc_parallel_boosting.mojo:1161` | `range(len(splits))` | 1 | over the tree splits (depth) |
| `gbdt/methods/doc_parallel_boosting.mojo:1166` | `range(len(leaf_values))` | 1 | over the leaf values (leaves x approx_dim) |
| `gbdt/methods/doc_parallel_boosting.mojo:1782` | `range(len(estimated))` | 1 | over the leaf values (leaves x approx_dim) |
| `gbdt/methods/doc_parallel_boosting.mojo:3967` | `range(len(pv))` | 1 | over the leaf values (leaves x approx_dim) |
| `gbdt/methods/doc_parallel_boosting.mojo:3980` | `range(len(leaf_values))` | 1 | over the leaf values (leaves x approx_dim) |
| `gbdt/methods/doc_parallel_boosting.mojo:4265` | `range(len(pv))` | 1 | over the leaf values (leaves x approx_dim) |
| `gbdt/methods/doc_parallel_boosting.mojo:4294` | `range(len(splits))` | 1 | over the tree splits (depth) |
| `gbdt/methods/doc_parallel_boosting.mojo:4311` | `range(len(leaf_values))` | 1 | over the leaf values (leaves x approx_dim) |
| `gbdt/methods/doc_parallel_boosting.mojo:4686` | `range(model.size())` | 1 | over the trees of the model |
| `gbdt/methods/doc_parallel_boosting.mojo:4690` | `range(depth)` | 1 | over the tree depth (split records) |
| `gbdt/methods/doc_parallel_boosting.mojo:4733` | `range(n_values)` | 1 | over the tree's leaf values (leaves x approx_dim) |
| `gbdt/methods/dynamic_boosting.mojo:416` | `range(n_leaves)` | 1 | over the leaves |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:492` | `range(len(new_point))` | 1 | over the Newton point's leaf values (bins x dims) |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:546` | `range(shift_len)` | 1 | over the Newton point's leaf values (bins x dims) |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:927` | `range(self.bin_count * self.single_bin_dim)` | 1 | over the leaf gradient slots (bins x dims) |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:1262` | `range(matrix_size * self.bin_count)` | 1 | over the per-leaf Hessian blocks (bins x matrix size) |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:1417` | `range(self.bin_count)` | 1 | over the leaf bins |
| `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:1419` | `range(self.cursor_dim)` | 1 | over the cursor dims of one leaf |
| `gbdt/methods/ordered_boosting.mojo:3323` | `range(n_leaves)` | 1 | over the leaves |
| `gbdt/methods/sym_iter_fast.mojo:261` | `range(depth)` | 1 | over the tree depth (split records) |
| `gbdt/resident_model.mojo:224` | `range(nw)` | 1 | over the packed words |
| `gbdt/resident_model.mojo:227` | `range(n_columns)` | 1 | over the feature columns |
| `gbdt/resident_model.mojo:234` | `range(nw)` | 1 | over the packed words |
| `gbdt/resident_model.mojo:237` | `range(len(per_word[w]))` | 1 | over the features of one packed word |
| `gbdt/resident_model.mojo:247` | `range(n_trees)` | 1 | over the trees |
| `gbdt/resident_model.mojo:253` | `range(n_trees)` | 1 | over the trees |
| `gbdt/resident_model.mojo:272` | `range(nw)` | 1 | over the packed words |
| `gbdt/resident_model.mojo:275` | `range(len(per_word[w]))` | 1 | over the features of one packed word |
| `gbdt/resident_model.mojo:290` | `range(n_trees)` | 1 | over the trees |
| `gbdt/train.mojo:593` | `range(n_words)` | 1 | over the packed words |
| `gbdt/train.mojo:596` | `range(n_features)` | 1 | over the features |
| `gbdt/train.mojo:606` | `range(n_words)` | 1 | over the packed words |
| `gbdt/train.mojo:611` | `range(len(per_word[w]))` | 1 | over the features of one packed word |
| `gbdt/train.mojo:661` | `range(n_features)` | 1 | over the features (border slabs) |
| `gbdt/train.mojo:875` | `range(n_features)` | 1 | over the features |
| `gbdt/train.mojo:1881` | `range(n_features)` | 1 | over the features |
| `gbdt/train.mojo:1900` | `range(n_features)` | 1 | over the features |
| `gbdt/train.mojo:2080` | `range(perm_count)` | 1 | over the permutations |
| `gbdt/train.mojo:2098` | `range(len(independent_configs))` | 1 | over the independent CTR configs |
| `gbdt/train.mojo:2111` | `range(perm_count)` | 1 | over the permutations |
| `gbdt/train.mojo:2116` | `range(len(configs))` | 1 | over the CTR configs |
| `gbdt/train.mojo:2164` | `range(unique_values - 1)` | 1 | over the one-hot categories (at most 255 borders) |
| `gbdt/train.mojo:2377` | `range(len(tables))` | 1 | over the CTR tables |
| `gbdt/train.mojo:2380` | `range(len(dependent_slots))` | 1 | over the dependent CTR slots |
| `gbdt/train.mojo:2570` | `range(n_columns)` | 1 | over the columns |
| `gbdt/train.mojo:2600` | `range(n_columns)` | 1 | over the columns |
| `gbdt/train.mojo:2605` | `range(len(symctr_pre_col))` | 1 | over the pre-quantized CTR columns |
| `resample/estimator.mojo:2372` | `range(len(widths))` | 1 | over the gathered arrays (argument spans) |
