# Deleted arms worth running (2026-11-15)

Question (Andrew, 2026-11-15): of the arms deleted unmeasured by `EXPERIMENT_DELETIONS_V2_2026-11-15.md`, which have
merit and should be timed on NVIDIA instead of staying deleted? Reading and judgment only: nothing is restored here.

## Counts

Of the 1,390 defines in the V2 plan, 66 are not deleted at integration head `6906a59111` (still read outside the refusal
guards: the 36 restored owed B arms, KEEPs found by the lanes, the CPU-column switch and the one PROMOTE). The other
**1,324** were deleted:

| bin | defines | what is in it |
|---|---|---|
| WASTE | **1,122** | 597 old paths beside a default (`_OFF` / `QOLD` / `LEGACY`); 102 Apple FAST programme cards; 158 Apple-FAST-only, `FAST` or Metal arms (AFN*, AFN26*, ANN3*, BPE*, `X_*_FAST_*`, FAST+Apple); 57 tuning knobs (TPB, rows, chunk, tile, column counts); 34 toggles / diagnostics / controls; 36 recorded losers or duplicates; 138 by hand: umbrella switches over sub-arms (`TREES_T03..T30`), old forms re-offered as arms (`*_NAIVE`, `*_SERIAL`, `*_UNFUSED`, `XCNN_NO_*`, `ATTN_NO_*`, `ATTN_CACHE_WAIT`), forcing switches, more knobs and FAST arms the name rules missed |
| RUN | **40** | real `identical` kernel or pipeline work aimed at, or plainly on the path of, an NVIDIA row that is regressed against 0.8.55 or loses to an opponent, or that removes a fixed cost (launches, waits, copies, uploads). The ranked 25 are below; 15 more are listed after them. |
| UNSURE | **162** | real work with no NVIDIA board row attached (ExtraTrees, UMAP, ANN, CNN, decomposition, linear models other than OLS, ordered GBDT, time series, ...) or with an unclear path to one |

Method: every define of the V2 plan joined with its `docs/TOMBSTONES.md` section, its `experiments/removed/*.patch`
files and the files they touch; mechanical bins first (decision class, Apple/FAST markers, knob and diagnostic names,
the recorded verdict), then every remaining arm read by hand (tombstone, the patch's own comments, the
`docs/apple-fast/EXPERIMENTS.md` row) and matched to the NVIDIA rows of measurement `pc-80a7f8aca30a-nvidia-r2`
(`REGRESSIONS-grouped.txt`, and `compare-nvidia.json` for rows that lose to the best opponent). The 41 board lanes are
the only rows that count; an arm whose estimator has no board row is UNSURE, never RUN. Working files:
`~/mojolearn-evidence/briefs-20261115/deleted-arms-shortlist/` (`final.json` has every define's bin).

Answer to the question: most of what went was waste. 1,122 of 1,324 (85%) were old paths, Apple FAST work, knobs or
toggles. About 200 were real `identical` work, and 40 of those sit on a slow NVIDIA row. The best of them came from
deliberate NVIDIA speed lanes (the Oct 28 neural-perf ideas, trees S3/F4, iforest C48/T41-T43) that never got their
A/B. They are the list below.

## RUN, ranked (25)

Rows are NVIDIA L40S, ms at this head (0.8.55 in brackets where regressed; best opponent where we lose). "Restore"
says how the patch meets today's tree, checked with a 3-way apply into a throwaway index: **clean**, **3-way, N
files** (hand-merge against later tombstone and sweep edits in the same files), or **file gone** (restore that file
from the recoverable sha first). "Sweep" means `lane/expdel2-sweep` also removed helpers that only this arm reached,
so the patch alone will not compile: take those helpers back from the sha too (their Sweep paragraph in
`docs/TOMBSTONES.md` names them). All patch paths are under `experiments/removed/`.

| # | define | what it does | rows it should move | patch (bytes) | recoverable at | bits | restore |
|---|---|---|---|---|---|---|---|
| 1 | `TREES_LG_LEVEL_ROUNDS` | lossguide grows one round per level when `max_leaves >= 2**max_depth` (leaf budget cannot bind; the board sets 256 = 2**8), instead of best-first single-leaf rounds | gbdt-lossguide fit istella 898,292 (catboost 30,510), taxi 65,900 / gain0 66,115 (xgboost 11,422) | `MOJOLEARN_TREES_LG_LEVEL_ROUNDS.patch` + `.expdel2-43-trees.patch` (5,296) | `ce4ab6287e` | none (patch argues the same tree) | 3-way, 2 files |
| 2 | `IDN_GEMM_INT_SLICES` | fp32 GEMM as four int8 planes on the tuned IMMA kernel plus one-rounding epilogue (research profile fp32.v2-int) | gemm 75.8 [46.0] (torch 34.0) | `MOJOLEARN_IDN_GEMM_INT_SLICES.patch` (5,750; `gemm/int_slices*.mojo` still in tree) | `41a649b03c` | **changes bits and precision**, every column together; quality (max rel err vs fp64) must be read | 3-way, 1 file |
| 3 | `IDN_ATTN_FOLD_V2` | attention key axis folded in 128-key leaves plus the GEMM carry tree; fused forward split into row-tile x leaf blocks | transformer-forward 6.1 (torch 2.4); lm-forward 58.6 [44.9] (torch 31.1); lm-train-step 36.0 (torch 10.8) | `MOJOLEARN_IDN_ATTN_FOLD_V2.patch` (44,227; `attn_fold_v2.mojo` still in tree) | `c3d1c7865f` | **changes bits** (fold order, every column) | 3-way, 1 file; sweep |
| 4 | `IDN_SEQ_TILED_PREDICT` | Sequential predict in one chunk when the scratch fits 1 GiB, instead of 4,096-row chunks | sequential-predict 3.3 (torch 1.4) | `MOJOLEARN_IDN_SEQ_TILED_PREDICT.patch` (5,612) | `d19ca4f0c2` | none | clean |
| 5 | `IDN_SSD_OCCUPANCY` | Mamba-2 SSD: coalesced S10 discretize, 16x16 shared-tile S12 `G = C.B`, S18/S19 `Y_off` row tiles | mamba2-forward 6.5 [5.6] (mamba-ssm 2.8) | `MOJOLEARN_IDN_SSD_OCCUPANCY.patch` (18,362) | `44ec9b2ec9` | none (same chains) | 3-way, 1 file; sweep |
| 6 | `IDN_NORM_FIXED_LANES` | RMSNorm / LayerNorm statistics as one 32-lane cell per row (shuffle tree), chain depth d -> d/32 + 5 | transformer-forward, lm-forward, lm-train-step (as #3) | `MOJOLEARN_IDN_NORM_FIXED_LANES.patch` + `.expdel2-25-neural.patch` + `.expdel2-31-neural.patch` (21,700) | `c48e5cad76` | **changes bits** (fold order, every column) | 3-way, 5 files; sweep |
| 7 | `TREES_C48_IF_PARTITION` | IsolationForest node split as one fused stable partition (flag, block scan, destinations once; no atomics) | iforest fit istella 305.8 [197.2], taxi 202.8 [149.1] (cuML 113.6) | `MOJOLEARN_TREES_C48_IF_PARTITION.patch` (3,334) | `dc94980d4a` | none (stable order kept) | clean |
| 8 | `GBDT_ID_DEFER_COPY` | depthwise/lossguide on NVIDIA/AMD `identical` take Apple `identical`'s deferred histogram copy schedule (not T19's stat skip, which lost quality) | gbdt-depthwise fit istella 8,036 [7,262], prebinned 9,199 (xgboost 8,557), taxi prebinned 4,002 (xgboost 3,039); lossguide fit (#1) | `MOJOLEARN_GBDT_ID_DEFER_COPY.patch` (2,419) | `8c20167f4f` | none (Apple identical ships it) | 3-way, 1 file |
| 9 | `TREES_HIST_WARP_AGG` | one-byte GBDT histogram: when a whole wave holds the same bin, one shuffle-tree sum and one atomic instead of 32 | every boosted fit row (depthwise, lossguide, multiclass 14,734 vs catboost 14,624, pairlogit 3,872 [2,700] vs catboost 3,521) | `MOJOLEARN_TREES_HIST_WARP_AGG.patch` + `.expdel2-43-trees.patch` (7,630) | `1082bcb01c` | none (integer adds) | 3-way, 2 files |
| 10 | `TREES_HIST_PACKED_GH` | grad and hess of a cell added by one 64-bit atomic instead of two 32-bit atomics | as #9 | `MOJOLEARN_TREES_HIST_PACKED_GH.patch` + `.expdel2-43-trees.patch` (6,362) | `e312bd1c25` | none (exact integer pair) | 3-way, 2 files; sweep |
| 11 | `IDN_GBDT_APPLY_WIDE` | oblivious (symmetric) apply: an ensemble of one depth walks many trees per launch, one cursor store per launch | gbdt-symmetric infer istella 46.0 [33.1] / large 83.4 [60.4], taxi 7.3 [6.1] / 13.3 [11.1] | `MOJOLEARN_IDN_GBDT_APPLY_WIDE.patch` (4,267) | `805d22d33d` | none (same ordered adds) | 3-way, 2 files; sweep |
| 12 | `GBDT_NS_PREDICT_BITVEC` | depthwise/lossguide apply by bit vector, same per-row tree-order adds | gbdt-depthwise infer istella 43.7 [36.3] / large 83.9 [69.1] / prebinned 96.8 [73.3]; lossguide infer taxi gain0 10.2 / 17.4 (xgboost 7.9 / 14.3) | `MOJOLEARN_GBDT_NS_PREDICT_BITVEC.patch` (17,677) | `4cdd684efa` | none | file gone (`gbdt/models/ns_bitvector_predict.mojo`); sweep |
| 13 | `TREES_C50_IF_PACKED` | IsolationForest scoring on packed vector nodes (one load per node) | iforest infer istella test 79.1 [60.4] (cuML 78.0), taxi 3.9 [3.3], large 6.5 [5.6] | `MOJOLEARN_TREES_C50_IF_PACKED.patch` (4,134) | `931e7ca76a` | none (same ftz adds) | 3-way, 1 file; sweep |
| 14 | `TREES_T42_CORRECTION_CACHE` | c(n) path-length correction read from a table instead of recomputed per node | iforest fit and infer (as #7, #13) | `MOJOLEARN_TREES_T42_CORRECTION_CACHE.patch` (3,286) | `d30e6f5678` | none if the table is built by `compute_c_n` (check) | 3-way, 1 file; sweep (`if_correction_table_kernel`) |
| 15 | `TREES_T41_TREE_BATCH` | IsolationForest builds trees in batches per launch | iforest fit (as #7) | `MOJOLEARN_TREES_T41_TREE_BATCH.patch` (1,479) | `aa61101d80` | none | 3-way, 1 file |
| 16 | `IDN_M3_DECODE_WINDOW_REUSE` | Mamba-3 decode: one-token step reuses the previous step's stage window instead of rebuilding at the carried length | mamba3-decode 2,326 [1,815] | `MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE.patch` (4,327) | `74754fe939` | none (scheduling) | clean |
| 17 | `GAPLM_ZDOT_KREG` | attention backward zdot estash kernel with the key held in registers (NVIDIA only) | lm-train-step 36.0 (torch 10.8); samba-train-step 50.5 [45.7] (torch 11.8) | `MOJOLEARN_GAPLM_ZDOT_KREG.patch` (12,897) | `62df7ffb39` | none expected (same dy chain); read identity in the run | clean |
| 18 | `IDN_TRAIN_NO_DECODE_CACHE` | training prefill skips the persistent decode-cache copy it never reads | lm-train-step (as #17) | `MOJOLEARN_IDN_TRAIN_NO_DECODE_CACHE.patch` + `.expdel2-31-neural.patch` (15,152) | `311b50e05c` | none | 3-way, 3 files; sweep |
| 19 | `FOREST_SHARED_ROWS` (with `TREES_T32_SHARED_ROWS`, same code) | forest predict reads a node's four words in one 16-byte load and shares row features across the group | rf infer istella large 118.2 [105.1] (cuML 92.3), test 57.2 (cuML 47.8), taxi 9.3 [8.1] / 17.7 [16.0] | `MOJOLEARN_FOREST_SHARED_ROWS.patch` (9,801) | `a5502e594d` | none (same words fetched) | 3-way, 1 file |
| 20 | `TREES_T44_METADATA_CACHE` | forest predict keeps a retained metadata snapshot instead of re-reading model metadata per call | rf infer (as #19), a per-call fixed cost on the small rows | `MOJOLEARN_TREES_T44_METADATA_CACHE.patch` + `.expdel2-08-base.patch` (3,756) | `3e9cbbd919` | none | file gone (`core/forest_experiments.mojo`) |
| 21 | `C62_SAME_COMPONENT_SKIP` | HDBSCAN matrix-free Boruvka skips j tiles whose every point is in the block's own component | hdbscan fit taxi 583 [423] (cuML 365), istella 1,382 [1,193] (cuML 905) | `MOJOLEARN_C62_SAME_COMPONENT_SKIP.patch` + `.expdel2-11-cluster.patch` (5,641) | `774f01dc6b` | none (same cells, same minimum) | 3-way, 2 files |
| 22 | `CLASSICAL_C21_EXTREMA` | SVM SMO block solve: four-way tuple tree fuses the two extrema, half the barrier rounds | svc fit istella 43.1 [32.7], taxi 762.6 (cuML 383.4); svr fit taxi 64.3 (cuML 49.1) | `MOJOLEARN_CLASSICAL_C21_EXTREMA.patch` + `.expdel2-08-base.patch` (5,090) | `9aca9aa819` | none (total orders unchanged) | 3-way, 1 file |
| 23 | `CLASSICAL_C22_TRIANGLE` | symmetric kernel matrix (same dataset both sides) computed as one triangle | svc / svr fit (as #22); kernel-ridge fit istella 544 (cuML 341), taxi 533 (cuML 337) | `MOJOLEARN_CLASSICAL_C22_TRIANGLE.patch` + `.expdel2-20-linear.patch` (4,186) | `45379ff055` | none expected (mirror of the same dot); read identity | 3-way, 2 files |
| 24 | `2580_LEVEL_QUANT` | GBDT one-byte statistics quantized once per level, the histogram launch reading that plane | every boosted fit row (as #9) | `MOJOLEARN_2580_LEVEL_QUANT.patch` (6,443) | `7a9e6c5691` | none recorded; read identity | clean |
| 25 | `IDN_KM_RBF_CELL` | RBF kernel matrix at <= 64 features as one direct-distance launch, one thread per cell | kernel-ridge fit (as #23) if its kernel goes through `kernel_methods/` (check the call path first) | `MOJOLEARN_IDN_KM_RBF_CELL.patch` (8,225) | `0ade8f67d9` | **changes bits** (direct distance instead of the expanded form) | clean |

### RUN below the cap (15)

Real work on a slow row, ranked lower because the gain is smaller, the path is less certain, or a ranked entry covers
the same row: `TREES_T32_SHARED_ROWS` (one code with #19), `IDN_CE_GRAD_FUSED`, `IDN_TRAIN_SWIGLU`,
`IDN_CE_DENOM_ROWFOLD` (lm-train-step launches), `IDN_M2_YOFF_EXP`, `NN37_TRIANGLE_TASKS`, `NN35_CONV_PAIR`
(mamba2-forward), `CLASSICAL_C20_PAIR_LOAD` (svc kernel cache), `CLASSICAL_C05_OLS_PHASE` (OLS fit scratch, ols fit taxi
22.3 [16.5]), `TREES_C45_GBDT` (GBDT score grid), `TREES_C50_GB_PACKED` (GBDT predict groups; file gone),
`TREES_T37_WORKSPACE` (forest predict workspace reuse; file gone), `KDE_DIRECT_DISTANCE` (kde fit taxi 7.0 [6.0]),
`TREES_C47_IF_LIVE_TASKS`, `TREES_T43_RADIX_SELECT` (iforest fit).

### Running them

- Every RUN entry is an opt-in arm: restore it behind its define (default off, the four tombstone parts moved to
  `Restored`), build one wave sha, and time arm on vs off on NVIDIA in one measurement. Measurement is NVIDIA only
  (brief); AMD's vote, when a promotion is decided, follows the usual average rule.
- Bit-changing entries (#2, #3, #6, #25) need the host column and every vendor changed together before promotion;
  #2 also needs its quality read.
- These arms are candidates, not fixes: no row above is known to be fixed by them. Lossguide (#1, 29x behind
  catboost on istella) is the largest gap, and #1, #8, #9, #10 and #24 are the only deleted code aimed at it.

## DELETE-OLD-PATH whose surviving default has no board row (261)

The old path went and the default stays, but no NVIDIA board row times that default: nobody measured it against the old
path, and the board does not catch a regression in it. 117 more old paths sit beside FAST/Apple-only defaults and are
not listed (Apple FAST is stopped). Grouped by the first file the patch touches:

- x_decomp (31): `CLASSICAL_C24_ROWS2048_OFF`, `CLASSICAL_RSVD_TSQR_ORTHO_OFF`, `FA_GRAM_QOLD`, `FA_SVD_GROUP_OFF`, `IDN_CD_RESIDENT_OFF`, `IDN_CODE_RESIDENT_OFF`, `IDN_EIGH_RESIDENT_OFF`, `IDN_EIGH_TRIDIAG_OFF`, `IDN_ICA_LIM_DEV_OFF`, `IDN_LLE_DEV_F0_OFF`, `IDN_LLE_PAD_DEV_OFF`, `IDN_LU_GESV_OFF`, `IDN_LU_RESIDENT_OFF`, `IDN_NMF_L2_CELL_OFF`, `IDN_POLAR_MAX_DEV_OFF`, `IDN_QR_R_DIRECT_OFF`, `IDN_QR_R_INPLACE_OFF`, `IDN_QR_R_RESIDENT_OFF`, `IDN_RAND_RESIDENT_OFF`, `IDN_RES_MOVES_OFF`, `IDN_SVD_RESIDENT_OFF`, `IDN_TSQR_GRID_OFF`, `IDN_TSQR_NORM_OFF`, `IDN_XD_SVD_GROUP_OFF`, `IDN_XD_SWEEP_OFF`, `LDA_FUSED_SS_OFF`, `LDA_V_CAP_RULE_OFF`, `LEGACY_NARROW_LDA_FUSED_V`, `LU_QOLD`, `MCD_BATCH_MMA_OFF`, `MCD_SKIP_PINVH_OFF`
- x_prep (28): `CALIB_GNB_FOLDS_OFF`, `IDN_CLASS_ONEPASS_OFF`, `IDN_DRADIX_RANGE_PASSES_OFF`, `IDN_GRAM_BLOCKED_OFF`, `IDN_II_CONV_TREE_OFF`, `IDN_II_GRAM_TILE_OFF`, `IDN_LABEL_OFF`, `IDN_NB_CAT_ATOMIC_OFF`, `IDN_NB_CSR_DENSE_OFF`, `IDN_NB_CSR_OFF`, `IDN_NB_ONEPASS_OFF`, `IDN_PT_BLOCKED_OFF`, `IDN_RR_EIGH_OFF`, `IDN_SELECT_BLOCKED_OFF`, `IDN_TE_ENC_TREE_OFF`, `IDN_TE_GLOBAL_TREE_OFF`, `IDN_XPREP_RADIX_OFF`, `LABEL_DIRECT_OFF`, `LDAQDA_DEC_TILE_OFF`, `LDAQDA_PAR_STAGES_OFF`, `LDAQDA_RR_EIGH_OFF`, `MI_REG_TIES_OFF`, `PREP3_MAXABS_POOL_OFF`, `PT_SCORE_STABLE_OFF`, `SELECT_FCLS_OFF`, `SELECT_FREG_OFF`, `SI_ONEPASS_OFF`, `X_PREP_CLASS_COV_GRID_OFF`
- x_cnn (21): `BN_FOLD_BLOCK_OFF`, `DROPOUT2D_CH_MASK_OFF`, `IDN_ADAPT_BWD_BOUNDED_OFF`, `IDN_ADAPT_M_OFF`, `IDN_AVGPOOL_BWD_TILE_OFF`, `IDN_CNN_RESIDENT_GRADS_OFF`, `IDN_DIRECT_CONV_NVAMD_OFF`, `IDN_GAP_BLOCK_FOLD_OFF`, `IDN_GCN_LOOPS_DEV_OFF`, `IDN_GRAPH_M_OFF`, `IDN_GROUP_M_OFF`, `IDN_LAYER_DEV_IO_OFF`, `IDN_PAD_BWD_BOUNDED_OFF`, `IDN_PAD_M_OFF`, `IDN_PIN_WEIGHTS_OFF`, `IDN_XCNN_BIAS_FOLD_OFF`, `IDN_XCNN_SOFTMAX_ONE_EXP_OFF`, `IDN_XENT_DEV_FOLD_OFF`, `XCNN_ONES_CACHE_OFF`, `XCNN_RES_FREE_ASYNC_OFF`, `XCNN_RES_NEW_NOZERO_OFF`
- (no file recorded) (17): `CLASSICAL_C08_UNIQUE_SCAN_OFF`, `CLASSICAL_C25_PROJECTION_REUSE_OFF`, `CLASSICAL_ENETCV_FOLD_BLOCKS_OFF`, `CLASSICAL_ENETCV_SCORE_BLOCKS_OFF`, `DT_BINS_QOLD`, `IDN_GPC_OVR_OFF`, `IDN_GPR_PTR_OFF`, `IDN_MBK_RESIDENT_OFF`, `IDN_NC_CHUNKED_OFF`, `IDN_NC_FIT_LEAN_OFF`, `IDN_OPTICS_FUSED_STEP_OFF`, `IDN_SVGP_BLKCHOL_OFF`, `IDN_SVGP_RBFTILE_OFF`, `MBK_ZEROCOPY_OFF`, `PURITY_4_OFF`, `TRANSFORMER_LEGACY_FRESH_PREFILL`, `XN_IMPUTE_ROWDIST_OFF`
- arima (14): `ARIMA_FUSED_EVAL_TAIL_OFF`, `ARIMA_ID_BATCH_GRAD_OFF`, `ARIMA_ORDER_BATCH_OFF`, `IDN_ARIMA_EVAL_WS_EXOG_OFF`, `IDN_ARIMA_EVAL_WS_OFF`, `IDN_ARIMA_FUSED_TAIL_OFF`, `IDN_ARIMA_IC_DEVICE_OFF`, `IDN_ARIMA_LLONLY_OFF`, `IDN_ARIMA_ORDER_BATCH_OFF`, `IDN_ARIMA_ORDER_CAP_OFF`, `IDN_ARIMA_ORDER_DEVICE_OFF`, `IDN_ARIMA_ORDER_SEASONAL_OFF`, `IDN_KALMAN_RD_OFF`, `KALMAN_TIME_SCAN_OFF`
- sequence (13): `ETS_TEAM_OFF`, `IDN_AF_RESIDENT_OFF`, `IDN_ETS_TEAM_OFF`, `IDN_GARCH_COOP_OFF`, `IDN_GARCH_FOLD32_OFF`, `IDN_GARCH_LEAD_REG_OFF`, `IDN_MOE_GROUPED_GEMM_OFF`, `IDN_NM_SPEC_OFF`, `IDN_PROPHET_COOP_OFF`, `IDN_SEQ_AF_COOP_OFF`, `MOE_DEVGROUP_OFF`, `MOE_REGTILE_OFF`, `SEQ_CROSTON_REG_OFF`
- x_linear (13): `ARD_SIGMA_QOLD`, `BAYES_GRID_GUARD_OFF`, `LEGACY_NARROW_SGD_PS_SIMD`, `SGD_ADAPTIVE_BATCH_OFF`, `SGD_AUTO_STEPS_256_OFF`, `SGD_IDN_CHUNK_WIDE_OFF`, `SGD_IDN_DEV_FINITE_OFF`, `SGD_IDN_MB_FUSE_OFF`, `SGD_IDN_OVR_PAR_OFF`, `SGD_IDN_PS_WARP_OFF`, `SGD_OC_IDN_MB_DEFAULT_OFF`, `SGD_OC_MB_IMPLICIT_OFF`, `XLIN_IDN_DEV_FINITE_OFF`
- extratrees (11): `ET_BINNED_OFF`, `ET_PART_FLAGS_OFF`, `ET_RM_LINE_HW_OFF`, `ET_RM_NARROW_OFF`, `ET_TILED_SEARCH_IDENTICAL_OFF`, `ET_TILED_SEARCH_OFF`, `IDN_ET_PART_ROWS_OFF`, `IDN_ET_STAGE_LIVE_OFF`, `IDN_ET_TILED_SEARCH_OFF`, `LEGACY_NARROW_ET_BINNED`, `LEGACY_NARROW_ET_RM`
- gbdt (11): `GBDT_ORD_TREE_BUFS_OFF`, `IDN_GBDT_YETI_BLOCK_AMD_OFF`, `IDN_ORD_CAT_PLANES_OFF`, `IDN_ORD_ONE_STEP_DEVICE_OFF`, `IDN_ORD_STD_SCALE_DEVICE_OFF`, `LEGACY_NARROW_ORD_ALL`, `ORD_ALL_OFF`, `ORD_SKIP_UNSEARCHED_PERM_OFF`, `TREES_ORD_STD_GRIDFOLD_OFF`, `YETI_EST_REUSE_SEARCH_OFF`, `YETI_SEARCH_TASK16K_OFF`
- x_ann (10): `ANN_COARSE_TILED_OFF`, `IDN_PQ_DEVICE_CODEBOOKS_OFF`, `IDN_PQ_LUT_TILED_OFF`, `IDN_PQ_TRAINSET_STRIDED_OFF`, `IDN_TSNE_LANES_SPLIT_OFF`, `IDN_TSNE_LANE_FOLD_OFF`, `IDN_TSNE_REP_TILE_OFF`, `IVF_PQ_ONE_UPLOAD_OFF`, `IVF_PQ_RESIDENT_FIT_OFF`, `TSNE_SPLIT_OFF`
- ivf (9): `IDN_IVF_DEVICE_FINITE_OFF`, `IDN_IVF_DEVICE_SCALE_OFF`, `IDN_IVF_DEVICE_SQRT_OFF`, `IVF_COARSE_RANDOM_INIT_OFF`, `IVF_IDENTICAL_ANY_DIM_OFF`, `IVF_IDENTICAL_SCAN_OFF`, `IVF_IDN_RECLUSTER_CAP_OFF`, `IVF_IDN_STRIDED_INIT_OFF`, `IVF_TRAINSET_STRIDE_OFF`
- spectral (9): `IDN_LANCZOS_POOL_OFF`, `IDN_LAP_DEGREE_LANES_OFF`, `IDN_SPECTRAL_GRAPH_DEVICE_OFF`, `IDN_SPECTRAL_LABELS_DEVICE_OFF`, `IDN_SPECTRAL_VECS_DEVICE_OFF`, `IDN_SPMV_LANES_OFF`, `IDN_SYMMETRIZE_SORTED_OFF`, `LANCZOS_ID_DEV_FUSE_OFF`, `LANCZOS_ID_DEV_OFF`
- core (8): `IDN_CACHE_CTX_KEY_OFF`, `IDN_HPDEV_CAST_OFF`, `IDN_HPDEV_FOLDS_OFF`, `IDN_HPDEV_ISUM_OFF`, `IDN_HPDEV_LABELS_OFF`, `IDN_SCRATCH_POOL_OFF`, `IDN_SEG_RADIX8_OFF`, `XPREP_ARENA_POOL_OFF`
- glm (8): `MULTIOUT_RIDGE_OFF`, `PURITY2_2_OFF`, `PURITY_1_OFF`, `QN_DEV_ARGMAX_OFF`, `QN_IDN_FUSED_OFF`, `QN_IDN_SLIM_OFF`, `QN_OVR_ONE_UPLOAD_OFF`, `QN_TILED_OFF`
- training (8): `IDN_CE_PIPE_UP_OFF`, `IDN_MAXIMIZE_DEV_OFF`, `IDN_OPT_CLIP_BATCHED_OFF`, `IDN_OPT_GATE_SCAN_OFF`, `IDN_OPT_PARAMS_RESIDENT_OFF`, `IDN_OPT_SCRATCH_POOL_OFF`, `IDN_OPT_SGD_ONE_LAUNCH_OFF`, `IDN_TRAIN_DEV_TENSORS_OFF`
- xtrees (8): `AGN_IDN_SYN_POOL_OFF`, `DART_DEVICE_OFF`, `IDN_DART_DEVICE_OFF`, `IDN_PSHAP_RESIDENT_OFF`, `IDN_SHAP_DEVICE_MODEL_OFF`, `PSHAP_DELTA_OFF`, `SHAP_PERM_CACHE_OFF`, `TREES_PLATT_ONE_READ_OFF`
- bindings (7): `DART_ROUND_REF_OFF`, `IDN_HPDEV_NORMAL_OFF`, `IDN_XN_SKETCH_CTR_OFF`, `IDN_XN_UNIT_DEV_OFF`, `NEURAL_TABLE_ROUTE_OFF`, `PROBA64_QOLD`, `SEQ_MOE_DEVICE_IO_OFF`
- x_cluster (7): `IDN_AP_DEVICE_CONV_OFF`, `IDN_AP_R_TOP2_OFF`, `IDN_BGMM_MOMENTS_POOL_OFF`, `IDN_BGMM_NK_LEVELS_OFF`, `IDN_BISECT_DEVICE_CENTER_OFF`, `IDN_BISECT_DEVICE_SCORES_OFF`, `IDN_MEANSHIFT_GRID_OFF`
- solver (5): `CD_FUSED_OFF`, `CD_IDN_GRAM_COST_RULE_OFF`, `CD_IDN_GRAM_OFF`, `CD_THREE_LAUNCH_OFF`, `CD_TWO_STEP_OFF`
- gaussian_process (4): `IDN_GP_BULK_DOWNLOAD_OFF`, `IDN_GP_GRAD_ROWFOLD_OFF`, `IDN_GP_PREDICT_LAZY_L_OFF`, `IDN_GP_VAR_SEG_OFF`
- metrics (4): `IDN_METRIC_EPI_OFF`, `IDN_SIL_NO_WAIT_OFF`, `IDN_TRUST_DEV_OFF`, `IDN_TRUST_NO_STAGE_OFF`
- x_neighbors (3): `IDN_SVGP_B_FP32_GATED_OFF`, `IDN_SVGP_DOT2_OFF`, `PURITY_5_OFF`
- tsa (3): `IDN_KPSS_ONE_WAIT_OFF`, `IDN_KPSS_SCAN_OFF`, `IDN_SELECT_D_OFF`
- providers (2): `IDN_TREE_SHAPE_NATIVE_OFF`, `TE_NATIVE_SPLITS_OFF`
- embedding (2): `EMB_DEVICE_SCRATCH_OFF`, `EMB_RESIDENT_OFF`
- decomposition (1): `IDN_COLVAR_FUSED_OFF`
- cluster (1): `FUSED_STAGE_FTZ_OFF`
- mixture (1): `IDN_GMM_NK_LEVELS_OFF`
- autograd (1): `AUTOGRAD_CURVE_WORD_OFF`
- resample (1): `IDN_BOOT_SELECT_OFF`

## UNSURE (162)

Real work with no NVIDIA board row, or with an unclear path to one. Not ranked.

`2010_ROWS_SORTED`, `2635_LINEAR_BOUNDS`, `ATTN_DKDV_MFMA`, `ATTN_DQ_MFMA`, `ATTN_FWD_TQ_RULE`, `AUTOGRAD_HAS_INFERENCE`, `C31_DEVICE_BUCKETS`, `C32_COUNT_FUSION`, `C32_EMIT_FUSION`, `C34_PARALLEL_EDGES`, `C35_PACKED_LISTS`, `C39_RETAIN_STATE`, `C40_ACTIVE_SEEDS`, `C41_FUSED_MINIMA`, `C42_ACTIVE_TRIANGLE`, `C43_RESIDENT_NORMALIZATION`, `C44_SAMPLING_DESCRIPTORS`, `C52_PAIR_ROWS`, `C54_PREDICT_TILES`, `C57_CANDIDATE_STATE`, `C58_TEAM_MIB`, `C60_LAG4`, `CD_IDN_GRAM_WIDE`, `CLASSICAL_C02_LINEAR_PAIR`, `CLASSICAL_C03_FINITE_EXTREMA`, `CLASSICAL_C06_NORM_ROWS`, `CLASSICAL_C06_SMALL_D_THREAD`, `CLASSICAL_C07_RADIX_BITS`, `CLASSICAL_C07_RADIX_ROWS`, `CLASSICAL_C08_GROUPED_OUTPUT`, `CLASSICAL_C08_ONEHOT_FT`, `CLASSICAL_C08_ORDINAL_FT`, `CLASSICAL_C09_REG_BUNDLE`, `CLASSICAL_C10_RANK_REUSE`, `CLASSICAL_C11_DRAW_GATHER`, `CLASSICAL_C12_SPARSE_COUNTS`, `CLASSICAL_C13_LOGCV_WEIGHTS`, `CLASSICAL_C14_GROUP_RHS`, `CLASSICAL_C15_FACTOR_SOLVE`, `CLASSICAL_C16_GLM_FUSED`, `CLASSICAL_C17_OVR`, `CLASSICAL_C18_RESIDUAL_NEXT`, `CLASSICAL_C19_SGD_CHUNK`, `CLASSICAL_C23_MCD`, `CLASSICAL_C24_PANEL8`, `CLASSICAL_C24_TREE4`, `CLASSICAL_C26_PRODUCTS`, `CLASSICAL_C26_UPDATE_FUSED`, `CLASSICAL_C27_COMPONENTS`, `CLASSICAL_C27_FA_COMPONENTS`, `CLASSICAL_C27_NORM_VECTOR`, `CLASSICAL_C28_BUCKET_SOLVES`, `CLASSICAL_C56_LDA_INPUT`, `CLASSICAL_C56_QDA_PROJECT`, `CLASSICAL_ENETCV_SCORE_BLOCKS`, `CLASSICAL_RIDGECV_FF_BLOCKED`, `CLASSICAL_TE_BLOCKED_FOLD`, `EMB_AUTO_SORT_MIN_CELLS`, `ENTRY_TABLE_HEAP_OUTPUT`, `ET_RANGE_TILED`, `ET_ROW_MAJOR`, `ET_SCORE_TILED`, `EXPERIMENTAL_KMEANS_BLOCK_ACC`, `FA_GRAM_ONCE`, `FA_LIVEBUF`, `FOREST_ORDERED_RESIDENT`, `FOREST_PINNED_STAGE`, `GLM_CELLS_MIXED`, `GMM_INIT_FILL`, `IDN_AF_COL_CHUNK_FOLD`, `IDN_ATTN_TILE_ORDER`, `IDN_CACHE_CTX_KEY_DEVICE_ID`, `IDN_CNN_CONV_RELU`, `IDN_DENDRO_HOOK_CHUNK`, `IDN_GEMM_STAGE64`, `IDN_GRAM_ROWTILE`, `IDN_IF_FIT_EXPORT`, `IDN_LM_RESIDENT_GENERATE`, `IDN_LM_VIEWS`, `IDN_M2_GRAD_FOLD`, `IDN_M3_ANGLE_PARALLEL`, `IDN_M3_YINTRA_RESOURCE_TILE`, `IDN_MAMBA_BWD_FOLDS`, `IDN_MOE_STABLE_PACK`, `IDN_MST_SCAN_LANES`, `IDN_NEURAL_CHAINS`, `IDN_NEURAL_GEMM_SCHEDULE_ROLES`, `IDN_NEURAL_NN07`, `IDN_NEURAL_NN12`, `IDN_NEURAL_NN16`, `IDN_NORM`, `IDN_NYS_RR_DEV_STOP`, `IDN_ORD_INDEX_SHARED`, `IDN_RAGGED_FLOAT_RADIX`, `IDN_RAGGED_RADIX_NIBBLE`, `IDN_RAGGED_RADIX_SELECT`, `IDN_ROPE`, `IDN_SCRATCH_POOL`, `IDN_SEQ_LN_LEAF`, `IDN_SEQ_WGRAD`, `IDN_SL_SPARSE_MST`, `IDN_TRAIN_SCRATCH`, `IDN_TSQR_REUSE`, `IDN_X0_PAR_LS`, `KALMAN_TIME_SCAN`, `LDA_V_CAP_HALF_PAGE`, `MINIBATCH_ONE_PASS`, `MI_REG_RANKMAJOR`, `MLP_SESSIONS_PACKED`, `MSEL_UNITS_GPU_PARALLEL`, `NI01_TRAINING_WORKSPACE`, `NI07_GROUPED_PROJECTIONS`, `NI10_BUDGETED_CNN_TAPE`, `NI10_RECOMPUTE_CNN_TAPE`, `NI14_COL2IM`, `NI15_BIAS_NO_ONES`, `NI17_BALANCED_NORM`, `NI55_GRAPH_FEATURE4`, `NI57_GRAPH_TREE64`, `NI58_SAGE_FEATURE4`, `NI59_DROPOUT_CHANNEL`, `NI60_DROPOUT_APPLY4`, `NN21_SCALE_MASK`, `NN29_RING_PAIR`, `NN40_OWNED_WEIGHTS`, `NN40_WS_HEADROOM`, `NN42_INPUT_BIAS_FUSED`, `NN44_EXPERT_BISECT`, `NN46_GATHER_BOUNDS`, `NN47_APPLY_RUNNING`, `NN48_CSR_TILES`, `NN50_OWNED_RUNS`, `NN58_ACCUMULATE_STATUS`, `NN59_DROPOUT_RESIDUAL`, `NN63_CANONICAL_SHARD_MERGE`, `OPTICS_SIMD`, `ORD_HIST_FOLD_SKIP`, `PERM_COUNT_RANK`, `RF_FUSED_BOOTSTRAP_GATHER`, `SEQ_REC_STAGE64`, `TREESHAP_ACC_SHARED`, `TREES_CTR_SORTFREE_SUMS`, `TREES_FOREST_ROUTE`, `TREES_PREDICT_ROW_OUTPUTS`, `TREES_T14_EXACT`, `TREES_T28_PREFIX`, `TREES_T28_SORT`, `TREES_T30_ADABOOST`, `TREES_T33_COST_SCHEDULE`, `TREES_T34_CHUNK_FOLD`, `TREES_T38_FUSED_LABELS`, `TREES_T39_AUXILIARY_WALK`, `TREES_T40_ROWMAJOR`, `TREES_T45_LINEAR_UNWIND`, `UMAP_LIVE_BOTH_ARM`, `XCLUSTER_ROW_ASSIGN`, `XC_ALLOC`, `XN_IMPUTE_NO_SPLIT`, `XN_KNN_ROWWISE`, `XN_PCS_CELL`, `X_LINEAR_ARD_LEAD`, `X_LINEAR_NO_STAGED_FOLD`.

## Doubts

- The bins come from the tombstone text, the patch comments and the file paths, not from a build. An arm placed in
  RUN may not reach the board row's actual call path (#25 most of all; #14 needs the table built by the same function).
- `TREES_C50_GB_PACKED`, `TREES_PREDICT_ROW_OUTPUTS`, `FOREST_SHARED_ROWS` and other `core/forest_*` arms have no
  tombstone section under their own heading; their shas come from the TOMBSTONES index or EXPERIMENTS.md rows.
- A multi-lane define has one patch per lane (`<D>.<lane>.patch`); apply them all. The recoverable sha shown is the
  owner lane's; the others are in each TOMBSTONES section.
- "Sweep" helpers: `lane/expdel2-sweep` removed 380 symbols that were reachable only from deleted arms. A restore must
  bring back the ones its arm reached; the patch does not contain them.
- The FOLLOWUPS doubts: `IDN_GEMM_INT_SLICES` is RUN #2. `MSEL_UNITS_GPU_PARALLEL` is UNSURE (no board row, needs a
  thread-safety lane first). `IDN_GEMM_STAGE64` is UNSURE (AMD loser, its target is `gemm_dot` in `sequence/ops.mojo`,
  no slow NVIDIA row). `TREES_T29` is WASTE (its versioned arm measured 3.4x slower on NV, 3.8x on AMD).
  `HDB_ONE_SYNC` / `HDB_SELECT_DEVICE` are WASTE: Apple FAST routes whose `identical` versions already ship as the
  default (`IDN_HDB_ONE_SYNC`, `IDN_HDB_SELECT_ONE_READ`).
- The bins cover the 1,390 V2 defines only. The 108 deletions of expdel-01..06 (decided on records) are not re-binned.

