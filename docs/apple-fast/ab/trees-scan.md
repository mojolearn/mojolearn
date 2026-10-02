# lane/apple-fast-trees-scan: one-thread scans in the tree sorts (PLAN-trees next-experiments 5)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch is compiled under FAST (`GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL`) + Apple
(`has_apple_gpu_accelerator()`) only and defaults OFF; IDENTICAL compiles the old launch.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_SEG_SCAN_BLOCK=1` | build define | `gbdt/gpu_util/kernel/segmented_sort.mojo` `_seg_radix_pass` (`seg_scan_block_sums_block_kernel`) | the per-segment block-sum scan of the Exact-leaf radix sort runs on one 256-thread block per segment (grid (n_segments) x 256) instead of one thread per segment (grid (n_segments) x 1) |
| same define | build define | `ensemble/randomforest.mojo` `sort_selected_rows` (`rows_sort_scan_block_sums_block_kernel`) | the bootstrap-row sort's block-sum scan (one segment) runs on one 256-thread block instead of grid 1 x block 1 |
| `-D MOJOLEARN_SCAN_U32_BLOCK=1` | build define | `gbdt/gpu_util/kernel/scan.mojo` `launch_scan_vector_u32` (`scan_block_sums_u32_block_kernel`) | the CTR path's unsegmented u32 scan of block totals runs on one 256-thread block instead of grid 1 x block 1 |
| `-D MOJOLEARN_REORDER_FLAGS_SCAN_BLOCK=1` | build define | `gbdt/gpu_util/kernel/reorder_one_bit.mojo` `launch_reorder_one_bit` | launches the existing `scan_block_sums_parallel_kernel` (256 threads) instead of the serial `scan_block_sums_kernel`; reached by `checks/reorder_check.mojo` only, no production driver |

## Cause and bits

Every scan here is an exclusive prefix over per-block totals, one Int32 / UInt32 per 512-row
block. The old kernels walk those totals with ONE thread (`seg_scan_block_sums_kernel`
`segmented_sort.mojo:~372` and `core/segmented_sort.mojo` via `randomforest.mojo:~1842`;
`scan_block_sums_u32_kernel` `scan.mojo:~174`), a chain of dependent global round trips per
radix pass: Istella RF `n_sampled / 512` entries x `sort_passes_for` (~22) passes per tree; the
Exact leaf sort 22 passes per estimation; the CTR scan `n_rows / 512` (~8,000 at taxi 4.1M) per
`compute_current_bins` / `visit_equal_up_to_prior_freq_ctrs` call.

The new kernels are the stripe form `scan_block_sums_parallel_kernel` already compiles for the
unsegmented sort (`reorder_one_bit.mojo:148`) and `_ord_leaf_scan_block_kernel` for ordered
boosting: thread t sums a contiguous stripe of the totals, one `prefix_sum[exclusive=True]`
hands each stripe its carry, the stripe writes its exclusive prefixes. Segments longer than the
block are covered by longer stripes, not by a tile loop, so every thread reaches the collective
exactly once. INTEGER sums (Int32; the u32 scan's totals are counts below `size`, carried
through Int32 as `scan_block_u32_kernel` already does), so every prefix equals the serial
scan's bit for bit: same offsets, same reorder, same sort, same hashes. No float is scanned.

## Which queued lane reaches which switch

- `rf istella`: reaches `MOJOLEARN_SEG_SCAN_BLOCK` (bootstrap sort runs only at
  `n_cols >= ROWS_SORTED_MIN_COLS = 64`); `rf taxi` (16 columns) does not sort: no-harm check.
- `gbdt-depthwise`/`gbdt-lossguide taxi`: the segmented sort runs only under
  `leaf_estimation_method == Exact`; with the board's Newton/Gradient estimation these two are
  no-harm checks for the three defines together (same hashes expected, same time).
- `gbdt-categorical taxi`: reaches `MOJOLEARN_SCAN_U32_BLOCK` (CTR scans) and is the row that
  can move.
- `et taxi`: ExtraTrees (`extratrees/impl/randomforest/`) uses neither sort: no-harm check.
- No queued lane reaches `MOJOLEARN_REORDER_FLAGS_SCAN_BLOCK` (check-only path); it is in the
  gbdt arm so the compile is proven.

## Other one-thread / one-block launches and host loops on the tree paths (not fixed here)

| site | shape | why not here |
|---|---|---|
| `gbdt/gpu_util/kernel/segmented_scan.mojo:~426` `seg_scan_block_sums_kernel` | grid 1 x block 1 over `n_rows / 768` Float32 totals with segment flags (Exact leaves, CTR `launch_segmented_scan_and_scatter_non_negative`) | Float32 running sum in sequential order; a block scan re-associates the adds, not bit-identical |
| `gbdt/methods/ordered_boosting.mojo:~965` `_ord_seg_start_kernel` | grid 1 x block 1 over `n_leaves` (<= 64) totals | 64 entries; nothing to gain |
| `gbdt/methods/kernel/pointwise_split_resolve.mojo:~238/259/286` seed sentinel, fold winner, pack winner | grid 1 x block 1; fold is <= 32 records with a tie rule that needs sequential order | split chain (trees-depthwise family); tiny |
| `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:~5015` `choose_scale_kernel`, `~5702` canary, `~6118` copy_scalar | grid 1 x block 1, 1-2 values | control plane |
| `greedy_search_helper.mojo:~800, ~1496, ~3980` score / resolve-and-pack | grid 1 x block SCORE_BLOCK_SIZE / RESOLVE_BLOCK_SIZE (one full block) | one block over per-level winners; symmetric trees |
| `greedy_search_helper.mojo:~280`, `random_score_helper.mojo:~205`, `ordered_boosting.mojo:~1667-1911`, `doc_parallel_boosting.mojo` | `deterministic_sum_lanes_kernel` grid 1 x 256 folds of block partials | fixed-order float folds; one block by design |
| `gbdt/targets/kernel/yeti_rank.mojo:~782`, `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:~1793` | grid 1 x block 1 | other families (trees-yeti) |
| `ensemble/randomforest.mojo:727 preprocess_labels`, `:771 postprocess_labels`; `extratrees/impl/randomforest/randomforest.mojo:274 class_ids_for` | host loops over `n_rows` (label class ids) | host step before the fit; needs a device label-encoding design (PLAN-trees 4 / trees-io) |
| `ensemble/randomforest.mojo` `predict`, `score`, `compute_oob_score` loops over `n_rows` | host | inference / OOB, not the timed fit |

Keep rule: a switch becomes the FAST Apple default when its arm's AFT-MEDIAN is faster and
hashes / held-out quality are unchanged (they must be: integer scans); then the define goes and
the block kernel is the launch. A row whose lane does not reach the switch is a no-harm check
and cannot decide it; `trees-scan-rf-istella` decides `MOJOLEARN_SEG_SCAN_BLOCK`,
`trees-scan-cat-taxi` decides `MOJOLEARN_SCAN_U32_BLOCK`. Owed afterwards: an IDENTICAL ID check
is NOT needed (IDENTICAL compiles no new code), but a FAST ID check with the defines on
(`lq add m3 ID ... rf istella`) proves the same-bits claim on the device.
