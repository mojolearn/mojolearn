# lane/apple-fast-trees-depthwise: categorical CTR stages (PLAN-trees item 6)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch is compiled under FAST + Apple only and defaults OFF; IDENTICAL compiles the old code.
The depthwise split-chain fusion (PLAN-trees item 1) is the local session's
`origin/lane/apple-fast-depthwise` (f5be2f1d, `-D MOJOLEARN_GBDT_DW_FUSED_CHAIN`); nothing here duplicates it.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_GBDT_CTR_FAST_SCAN=1` | build define | `gbdt/gpu_util/kernel/scan.mojo` `launch_scan_vector_u32`; `gbdt/gpu_util/kernel/segmented_scan.mojo` `_run_segmented_scan` via `launch_segmented_scan_and_scatter_non_negative` | phase 2 of both scans (the per-block totals) on one 256-thread block (`scan_block_sums_u32_parallel_kernel`, `seg_scan_block_sums_parallel_kernel`) instead of the `grid_dim=1, block_dim=1` one-thread walk |
| `-D MOJOLEARN_GBDT_CTR_FAST_FREQ=1` | build define | `gbdt/train.mojo` `train`, the `compute_simple_ctrs(...)` site | the permutation-independent simple CTR (FeatureFreq / Counter, in the GPU default `simple_ctr`) through `compute_simple_ctrs_device` (`gbdt/ctrs/ctr_calcers.mojo:1065`) instead of the host calcer; `counter_calc_method=Full` keeps the host arm |

Cause. `gbdt-categorical` (Lossguide, taxi 5 declared columns, 2 above `one_hot_max_size`) builds its CTR columns
inside the fit (`train.mojo`, the `cat_features` loop before quantization), per cat feature and per permutation:
`TCtrBinBuilderGpu` (radix sort + `ScanVector`), `THistoryBasedCtrCalcerGpu` (two segmented scans per visit),
`TWeightedBinFreqCalcerGpu` (`ScanVector`). Each `ScanVector` and each segmented scan carried a serial one-thread
phase over `n / 512` and `n / 768` block totals (~8,000 and ~5,400 dependent global round trips at 4.1M rows,
`scan.mojo:101` `scan_block_sums_u32_kernel`, `segmented_scan.mojo:270` `seg_scan_block_sums_kernel`); the radix
sort already ran its block sums on one block (`reorder_one_bit.mojo` `scan_block_sums_parallel_kernel`, the
pattern copied). The independent half of the simple CTR was host work inside the fit: `compute_simple_ctrs`
(`ctr_calcers.mojo:81`) runs `TCtrBinBuilder`'s host stable sort of every row by category and the host frequency
calcer per cat feature; the device calcer existed, gated by `check-freq-ctr-device` (host == device cells,
bit for bit), and was unwired.

Bits. u32 scan: integer sums, exact. Segmented scan: taken only from the CTR entry point and only below 2^24 rows,
where the inputs are trivial weights and 0/1 target stats, so every partial sum is an integer below 2^24 and
exact in any association; `launch_segmented_scan_vector` (Exact leaf estimation, non-integer weights) keeps the
serial kernel. FeatureFreq on device: integer counts, the check above compares cells to the host column.

Keep rule: a switch becomes the FAST default when its arm is faster on the M3 and held-out quality stays within
FAST's run-to-run spread; then the define goes and the arm is the code. `tdw-cat-freqcheck` runs the existing
device-vs-host cell check at this head.

Not done (noted, not attempted blind): the depthwise per-level loop keeps two host waits per level after
f5be2f1d, `score.read` (the winner records feed `select_leaves_to_split` and the Lossguide replay on the host)
and `split.sizes` (`p_sz` feeds `is_terminal_leaf`, `build_necessary_histograms`' small-sibling choice and the
reorder dispatch). Folding `split.sizes` into the next level's `score.read` needs the plan on the device
(`update_partitions_and_plan_kernel`'s shape, DEVIATION 210) and the histogram build launched over device-resolved
child ids; that is the symmetric driver's blind level loop ported to the non-symmetric driver, too large to write
without a toolchain. Host loops that remain in the CTR prep: `dense_category_code` per row, the per-config
column read-back (`visit_cat_feature_ctr`), `build_ctr_tables` (model tables), the CTR column quantization.

## Settled overlap: the u32 `ScanVector` block-sums scan (NEXT_PASS item 6, 2026-10-02)

`lane/apple-fast-trees-scan` had its own copy of the one-block u32 block-sums scan in `scan.mojo`
(`-D MOJOLEARN_SCAN_U32_BLOCK`, guard `!= NUMERIC_IDENTICAL`). This copy is the one kept: the
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST` guard NEXT_PASS asks for, and the integer-only segmented-scan
arm beside it under the same define. trees-scan dropped its copy, so `MOJOLEARN_GBDT_CTR_FAST_SCAN`
is the only u32 scan switch and the two branches no longer touch the same file.

Neither copy was a multi-block scan: phase 2 is still one launch of one 256-thread block over
`n / 512` runtime totals (a stripe per thread, one `prefix_sum` for the carries). A two-level scan
would need a second-level `block_sums` buffer in all three callers (`ctr_bins_builder.mojo:494`,
`ctr_calcers.mojo:~1004`, `checks/ctr_device_check.mojo:210`), which size it at exactly
`ceil(n / SCAN_BLOCK)`; left for after this A/B.

Light form: three tags on `gbdt-categorical taxi` (scan, freq, both), 2 pairs each; the istella row
(`tdw-cat-both-istella`) is deferred until the taxi rows win.

## Pass 2: gbdt-categorical taxi 84.4 s vs LightGBM 58.9 s (NEXT_PASS_TREES item 4, family `trees-ctr`)

Read-profile first. The simple CTRs are computed ONCE before boosting (`train.mojo`, the `cat_features`
walk: `compute_simple_ctrs_gpu` per cat feature x permutation, read back, quantized), and tree CTRs are
not admitted (`max_ctr_complexity > 1` is refused), so there is no per-iteration CTR computation or
re-binarization to reuse: the borders are already built once per fit. What a CTR-bearing fit DOES pay
per tree is `permutation_count` = 4 column sets: the non-symmetric per-tree loop
(`doc_parallel_boosting.mojo`, `for p in range(perm_count)`) runs every permutation's leaf estimation
serially, each with its own partition drain, one drain per Newton evaluation
(`leaf_estimation_iterations` rounds plus the line search) and a tail drain: ~4 x (2 + N) host waits
per tree. Before boosting, each permutation's cindex is packed on the host (`flat.append` over
`n_columns x n_rows`, four 200M-element copies) into `_build_cindex_from_floats`.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_GBDT_CTR_PERM_BATCH=1` | build define | `gbdt/methods/doc_parallel_boosting.mojo` (`CTR_PERM_BATCH`, the per-tree permutation loop); `gbdt/methods/leaves_estimation/doc_parallel_leaves_estimator.mojo` (`DeviceLeafPartitioner.partition_enqueue` / `partition_collect`, the two halves of `partition`) | the four permutations' partitions enqueued back to back behind ONE drain, then the four Newton walks in lock step through the batched path `ordered_boosting.mojo` already uses (`_estimate_prepare`, `estimate_advance`, `_estimate_complete`): one drain per walker round for all four, one closing drain per tree. ~(2 + N) waits per tree instead of 4 x (2 + N). Taken when `estimate_can_batch` holds (Logloss: yes), `approx_dim == 1` and the device partitioner is on; else the serial loop |
| `-D MOJOLEARN_GBDT_CTR_PERM_PTRS=1` | build define | `gbdt/train.mojo` (`CTR_PERM_PTRS`, the per-permutation cindex loop) | each permutation's cindex from `_build_cindex_from_columns` with a pointer per column (dependent columns into `dep_by_perm[p]`), no host flat pack, one drain per staging ring instead of per feature |

Bits. PERM_BATCH: each task runs the same kernels on the same inputs in the same order as its serial
run (the ordered fit relies on the same property); only the drains are shared, and the estimation
permutation's leaf values are the ones the serial loop returned. PERM_PTRS: same columns, same borders,
same `binarize_float_feature_kernel`; the columns builder is documented bit-identical to the flat path.

Risky compile sites (no toolchain here): `_estimate_prepare(... perm_est_ws[p], perm_arena, stage_times,
iterations=...)` with a `List[List[TEstimationWorkspace]]` element as `mut est_ws`; `ref lp =
perm_leaf_parts[p]` then `lp.partition_enqueue` (mirrors `ref lp = leaf_parts[0]; lp.partition`);
`_ = parts^` / `_ = pend^` past the closing drain.

Not done (owed, read but not written blind): `build_ctr_tables` (the apply-time CTR tables) and the
`CTR_FAST_FREQ`-off `compute_simple_ctrs` are host passes over `n_rows` per cat feature inside the fit,
the plan's "one device pass per feature group (segmented sums)"; `visit_cat_feature_ctr` reads every
CTR column back (3 Borders priors x 2 features x 4 permutations) and `set_binarized_sample` /
`TCtrBinBuilderGpu(order)` re-upload the same target and order per feature; the CTR columns then go
host -> quantize -> device. A device-resident CTR column path (quantize from the calcer's `dst`)
removes those round trips; it needs `_quantize_training_columns` to take device columns.

Request lines: `tdw-cat-permbatch`, `tdw-cat-permptrs` (gbdt-categorical taxi, 2 pairs each).
