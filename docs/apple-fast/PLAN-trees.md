# Apple FAST trees: state and plan (from lane apple-fast-trees2, 2026-10-02)

Branch `lane/apple-fast-trees2`, head 50dfdcca0 (pushed), off origin/main 79cb08be2.
Worktree `~/mojolearn-wt/apple-fast-trees2`. Tools: `tools/aft_ab.sh` (alternating
FAST A/B of one binding, two define sets, board tree driver, prints AFT-MEDIAN with
hashes and held-out quality), `tools/aft_stage.sh` (MOJOLEARN_STAGE_TIMES triage),
`tools/aft_board_trees.py` (board tree rows), `tools/aft_if_refusal.py`.
Also fixed `~/mojolearn-evidence/lq/lq log` (zsh aborted on the unmatched race glob;
now `setopt nullglob`).

## Commits (all FAST + Apple comptime; IDENTICAL compiles the old path)

| commit | change | switch | measured (M3 Ultra, alternating A/B) |
|---|---|---|---|
| d88561e49 | GBDT Lossguide exact best-first in batches (trees-apple3 opt-in) and the inherited leaf partition are the FAST Apple defaults (`gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo:805,831`) | A/B arms `-D MOJOLEARN_GBDT_LG_EXACT_BATCH_OFF`, `-D MOJOLEARN_GBDT_NS_INHERIT_PARTITION_OFF` | aft-ab-lg1: Lossguide taxi 78,201 -> 16,722 ms; Istella 102,044 -> 23,084 ms; Depthwise taxi (inherit only) 17,260 -> 14,932 ms. Quality equal within FAST's run-to-run spread (logloss/AUC in commit message). PASSES speed + quality; IDENTICAL ID check still owed |
| 46f0cf09f | YetiRank: block kernel (one 256-thread block per task, DEVIATION 3040, same bits) on FAST Apple instead of one GPU thread per task (`gbdt/targets/kernel/yeti_rank.mojo:130`) | A arm `-D MOJOLEARN_3040_YETI_SEQUENTIAL` | A/B aft-ab-yeti1 queued, UNREAD. Stage run at this head: yetirank Istella fit 9.56 s (old board FAST 25.7 s, LightGBM 6.8 s) |
| 8d79e4d70 | iforest: fit's X stays row-major on device (raw upload, device finite scan `if_finite_scan_kernel`, row-major gather) (`isolation_forest/impl/isolation_forest.mojo` `_upload_rowmajor_fast`, `isolation_tree_builder.mojo` IF_FAST_ROWMAJOR) | `-D MOJOLEARN_IF_ROWMAJOR_OFF` | aft-ab-if1: FLAT (Istella 387.3 -> 383.3 ms, taxi 105.0 -> 105.4), same hash. Kept only as the base of the next commit |
| bfd1d7cc6 | iforest: Python fit skips its one-thread host `all_finite` over all of X when the binding scans on device (`python/mojolearn/_iforest_impl.py` fit; binding export `iforest_device_finite_scan`) | same define (binding answers 0) | aft-ab-if2 queued, UNREAD (also runs tools/aft_if_refusal.py: NaN/inf must raise ValueError) |

Rejected: threaded host `all_finite_f32` in the base binding: the no-host-routes push hook refuses host threads; dropped before push. Job aft-ab-fin1 (m3 #55) ran/runs at a head WITHOUT it: VOID, ignore.

## Queued on m3, unread (tags; `lq results m3 aft-`, `lq log m3 <tag> 'AFT-MEDIAN|Error'`)

| # | tag | what |
|---|---|---|
| 36 | aft-st-gbdt1 | stage triage: yetirank, depthwise, lossguide, categorical, ordered (yetirank read: tree_search 6.0 s, est.approx 2.7 s, sym.hist 1.75 s) |
| 38 | aft-ab-yeti1 | YetiRank block vs sequential, istellarank, 3 pairs |
| 43 | aft-ab-rf1 | RF `-D MOJOLEARN_RF_NODESPLIT_ZERO_AFTER_READ -D MOJOLEARN_RF_FAST_BATCH16K` vs default, taxi + istella |
| 45 | aft-ab-if2 | iforest device scan + skipped host scan, istella + taxi + refusal check |
| 46 | aft-ab-lgw64 | Lossguide exact batch width 64 vs 32 |
| 47 | aft-ab-lgw16 | width 16 vs 32 |
| 48 | aft-ab-smx8 | `-D MOJOLEARN_GBDT_SM_X8` depthwise + lossguide taxi |
| 49 | aft-ab-smx4 | `-D MOJOLEARN_GBDT_SM_X4` depthwise taxi |
| 50 | aft-ab-rfc8 | `-D MOJOLEARN_RF_HIST_COLUMNS8` rf taxi + istella |
| 51 | aft-ab-rfsn | `-D MOJOLEARN_RF_SMALL_NODE_1024` rf taxi + istella |
| 52 | aft-ab-etpr | `-D MOJOLEARN_ET_PART_ROWS` et taxi + istella |
| 53 | aft-ab-ettpb | `-D MOJOLEARN_ET_TPB_256` et taxi + istella |
| 54 | aft-ab-etb64 | `-D MOJOLEARN_ET_DEVICE_BATCH_65536` et taxi |
| 55 | aft-ab-fin1 | VOID (see above) |
| 80 | aft-ab-yreuse1 | lane/apple-fast-yetirank 098e89988: YetiRank estimation reuses the search derivatives (B, default) vs `-D MOJOLEARN_YETI_EST_REUSE_SEARCH_OFF` (A), istellarank, 3 pairs |
| 517 | aft-ab-ysort1 | lane/apple-fast-yetirank fc32c2a9c: YetiRank block-kernel sort as per-simdgroup register bitonic + 3 merge-path passes (B, `-D MOJOLEARN_YETI_FAST_SORT`) vs the ten-pass rank merge (A), istellarank, 3 pairs; same bits by construction (distinct composites) |

Owed: `lq add m3 ID lane/apple-fast-trees2 gbdt-lossguide,gbdt-depthwise,gbdt-rank-yetirank,iforest taxi,istella` (IDENTICAL device vs host at head; all changes are FAST-gated). Not queued.

## Worst FAST tree rows on the M3

The 0834 M3 board has only the algos family (no trees family rows). From it:
tree-shap Istella FAST 816 ms vs LightGBM 148 (5.5x), taxi 338 vs 73 (4.65x) — owned by lane gap-treeshap (GPU TreeSHAP in progress there). kernel-shap Istella 14,386 vs shap 7,695 (1.87x) and permutation-shap Istella 17,214 vs 12,335 (1.4x) explain a numpy ridge, not a tree. decision-tree 0.01-0.03x (we win), random-trees-embedding 0.11-0.44x.
Trees family, last M3 board (2026-09-29 m3ultra_checked, ours-ab = FAST, older params): yetirank 25,773 vs LightGBM 6,765 (3.8x); lossguide Istella 106,296 vs LightGBM 54,183 (1.96x, now ~23 s at board params); depthwise taxi 13,994 vs XGBoost 9,923 (1.41x); categorical taxi 63,398 vs LightGBM 46,900 (1.35x); iforest Istella 437 vs sklearn 282 (1.55x); lossguide taxi 44,556 vs 41,350 (1.08x). Needs a fresh trees-family M3 board at the new head.

## Next experiments, ranked

1. Depthwise per-level split chain (taxi 14.9 s vs XGBoost 9.9 s): stage triage (aft-st-gbdt1) shows the split chain stats/flags/partition/reorder/sizes each 20-35 ms per tree for categorical; fuse the chain kernels and drop host waits per level in `fit_non_symmetric_tree` (`greedy_search_helper_depthwise.mojo:1151`, split chain ~1800-2100).
2. YetiRank leaf estimation `est.approx` 2.7 s of 9.6 s: the estimation path relaunches the task kernel per Newton step (`gbdt/targets/kernel/yeti_rank.mojo:772` launch_yeti_rank_with[True], called from `gbdt/methods/leaves_estimation/pointwise_oracle.mojo:~600`); reuse the gradient pass's permutation pairs or fold estimation into one launch.
3. iforest Istella (383 ms vs 282): if aft-ab-if2 is still above sklearn, the 1.8 GB raw upload dominates; a device-side sample-index pass first, then upload only sampled rows (`isolation_forest/impl/isolation_forest.mojo` fit, `isolation_tree_builder.mojo:~700` gather) — needs a GPU-only gather design (no host gather: the hook refuses host steps).
4. RF/ET Python `all_finite` over all of X (one thread; `python/mojolearn/randomforest.py:481`, `extratrees.py:278`): move the refusal to a device scan in the binding's upload as done for iforest (bfd1d7cc6 pattern), not host threads (hook).
5. Single-thread scans in sorts: `seg_scan_block_sums_kernel` grid (n_segments) x block 1 (`gbdt/gpu_util/kernel/segmented_sort.mojo:372`, `ensemble/randomforest.mojo:1842`): a block-wide scan per segment.
6. Categorical taxi (1.35x): read aft-st-gbdt1 categorical table (per-tree 300-350 ms) for CTR stages.
7. Read lgw64/lgw16/smx8/smx4/rf*/et* A/Bs; flip any arm that wins with equal quality to the FAST Apple default (each define already exists).
