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
