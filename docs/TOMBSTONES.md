# Tombstones: every deleted experiment

Every experiment define deleted from mojolearn, in one place. Each section says what the define tried, its verdict with the
NVIDIA / AMD ratios and the quality change as the guard message, `grid_controls` entry and `docs/apple-fast/EXPERIMENTS.md` rows
recorded them, the run ids, the sha where the code still exists, and a patch that restores it.

How to restore a deleted experiment:

1. `git apply experiments/removed/<DEFINE>.patch` from the repo root (use `git apply -3` when main has moved past the
   deletion; when a section says the patch conflicts, check the files out of the recoverable sha and merge by hand). When a
   section names an `_shared-<sha>.patch`, apply that first: it carries the deletion commit's hunks that name none of the
   defines it deleted.
2. Delete the define's refusal in `core/six_lane_experiment_guards.mojo` and move its entry from `removed` back to
   `controls` in `experiments/six_lane_integration/grid_controls/<lane>.json`.
3. Compile through the orchestrator (lanes never compile) and A/B it again: the old numbers are from the base they were
   measured on.

Patches are the reverse of the deletion commit restricted to code files (`.mojo`, `.py`, `.sh` outside `tools/`, `bench/` and
`docs/`; the guard file and the grid JSON are left out). A patch marked "all code hunks" is the whole deletion commit; the
others keep only the hunks that name the define. Patches generated 2026-10-09 against main `7f501620d`.

Sources: the `removed` refusals in `core/six_lane_experiment_guards.mojo`, the `removed` lists in
`experiments/six_lane_integration/grid_controls/*.json`, and the DROPPED / DELETED rows of `docs/apple-fast/EXPERIMENTS.md`.
Renamed or merged defines (code kept under a new name), promoted defines and DROP rows whose code is still on main are listed
in the tables after the sections.

## Index

| define | area | verdict | date | patch |
|---|---|---|---|---|
| [`MOJOLEARN_ET_DEVICE_BATCH_65536`](#mojolearn_et_device_batch_65536) | Trees | DROPPED-slower | 2026-10-03 | [MOJOLEARN_ET_DEVICE_BATCH_65536.patch](../experiments/removed/MOJOLEARN_ET_DEVICE_BATCH_65536.patch) |
| [`MOJOLEARN_ET_IDN_BINNED_ANY_WIDTH_OFF`](#mojolearn_et_idn_binned_any_width_off) | Trees | removed | 2026-10-07 | [MOJOLEARN_ET_IDN_BINNED_ANY_WIDTH_OFF.patch](../experiments/removed/MOJOLEARN_ET_IDN_BINNED_ANY_WIDTH_OFF.patch) |
| [`MOJOLEARN_ET_TPB_256`](#mojolearn_et_tpb_256) | Trees | DROPPED-slower | 2026-10-03 | [MOJOLEARN_ET_TPB_256.patch](../experiments/removed/MOJOLEARN_ET_TPB_256.patch) |
| [`MOJOLEARN_GBDT_CTR_FAST_SCAN`](#mojolearn_gbdt_ctr_fast_scan) | Trees | DROPPED-noise | 2026-10-03 | [MOJOLEARN_GBDT_CTR_FAST_SCAN.patch](../experiments/removed/MOJOLEARN_GBDT_CTR_FAST_SCAN.patch) |
| [`MOJOLEARN_GBDT_CTR_PERM_PTRS`](#mojolearn_gbdt_ctr_perm_ptrs) | Trees | DROPPED-noise | 2026-10-03 | [MOJOLEARN_GBDT_CTR_PERM_PTRS.patch](../experiments/removed/MOJOLEARN_GBDT_CTR_PERM_PTRS.patch) |
| [`MOJOLEARN_GBDT_DW2_COPY_ZERO`](#mojolearn_gbdt_dw2_copy_zero) | Trees | DROPPED-slower | 2026-10-03 | [MOJOLEARN_GBDT_DW2_COPY_ZERO.patch](../experiments/removed/MOJOLEARN_GBDT_DW2_COPY_ZERO.patch) |
| [`MOJOLEARN_GBDT_DW_FAST_DEV_SCALE`](#mojolearn_gbdt_dw_fast_dev_scale) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_GBDT_DW_FAST_SKIP_FINAL_STATS`](#mojolearn_gbdt_dw_fast_skip_final_stats) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_GBDT_DW_FLAT_GRID`](#mojolearn_gbdt_dw_flat_grid) | Trees | DROPPED-speed | 2026-10-04 | lane only |
| [`MOJOLEARN_GBDT_DW_TREE_SYNC`](#mojolearn_gbdt_dw_tree_sync) | Trees | DROPPED-noise | 2026-10-03 | [MOJOLEARN_GBDT_DW_TREE_SYNC.patch](../experiments/removed/MOJOLEARN_GBDT_DW_TREE_SYNC.patch) |
| [`MOJOLEARN_GBDT_DW_TREE_SYNC_CHECK`](#mojolearn_gbdt_dw_tree_sync_check) | Trees | DROPPED-noise | 2026-10-03 | [MOJOLEARN_GBDT_DW_TREE_SYNC.patch](../experiments/removed/MOJOLEARN_GBDT_DW_TREE_SYNC.patch) |
| [`MOJOLEARN_GBDT_LG_EXACT_BATCH16`](#mojolearn_gbdt_lg_exact_batch16) | Trees | DROPPED-slower | 2026-10-03 | [MOJOLEARN_GBDT_LG_EXACT_BATCH16.patch](../experiments/removed/MOJOLEARN_GBDT_LG_EXACT_BATCH16.patch) |
| [`MOJOLEARN_GBDT_QH_FAST_FUSED_Q`](#mojolearn_gbdt_qh_fast_fused_q) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_GBDT_SEG_SUMS_BLOCK`](#mojolearn_gbdt_seg_sums_block) | Trees | DROPPED-noise | 2026-10-03 | [MOJOLEARN_GBDT_SEG_SUMS_BLOCK.patch](../experiments/removed/MOJOLEARN_GBDT_SEG_SUMS_BLOCK.patch) |
| [`MOJOLEARN_GBDT_SM_X4`](#mojolearn_gbdt_sm_x4) | Trees | DROPPED-noise | 2026-10-03 | [MOJOLEARN_GBDT_SM_X4.patch](../experiments/removed/MOJOLEARN_GBDT_SM_X4.patch) |
| [`MOJOLEARN_GBDT_SM_X8`](#mojolearn_gbdt_sm_x8) | Trees | DROPPED-slower | 2026-10-03 | [MOJOLEARN_GBDT_SM_X8.patch](../experiments/removed/MOJOLEARN_GBDT_SM_X8.patch) |
| [`MOJOLEARN_IDN_ET_BINNED_U16`](#mojolearn_idn_et_binned_u16) | Trees | slower | 2026-10-07 | [MOJOLEARN_ET_IDN_BINNED_ANY_WIDTH_OFF.patch](../experiments/removed/MOJOLEARN_ET_IDN_BINNED_ANY_WIDTH_OFF.patch) |
| [`MOJOLEARN_IDN_GBDT_FRONTIER_RESIDENT`](#mojolearn_idn_gbdt_frontier_resident) | Trees | DROPPED | 2026-10-07 | [MOJOLEARN_IDN_GBDT_FRONTIER_RESIDENT.patch](../experiments/removed/MOJOLEARN_IDN_GBDT_FRONTIER_RESIDENT.patch) |
| [`MOJOLEARN_IDN_RF_DEVICE_LOOP_K1`](#mojolearn_idn_rf_device_loop_k1) | Trees | noise | 2026-10-07 | [MOJOLEARN_IDN_RF_DEVICE_LOOP_K1.patch](../experiments/removed/MOJOLEARN_IDN_RF_DEVICE_LOOP_K1.patch) |
| [`MOJOLEARN_IDN_RF_DEVICE_LOOP_K2`](#mojolearn_idn_rf_device_loop_k2) | Trees | noise | 2026-10-07 | [MOJOLEARN_IDN_RF_DEVICE_LOOP_K2.patch](../experiments/removed/MOJOLEARN_IDN_RF_DEVICE_LOOP_K2.patch) |
| [`MOJOLEARN_IDN_RF_DEVICE_LOOP_K8`](#mojolearn_idn_rf_device_loop_k8) | Trees | noise | 2026-10-07 | [MOJOLEARN_IDN_RF_DEVICE_LOOP_K2.patch](../experiments/removed/MOJOLEARN_IDN_RF_DEVICE_LOOP_K2.patch) |
| [`MOJOLEARN_IDN_RF_STREAM_REPLICAS`](#mojolearn_idn_rf_stream_replicas) | Trees | DROPPED | 2026-10-07 | [MOJOLEARN_IDN_RF_STREAM_REPLICAS.patch](../experiments/removed/MOJOLEARN_IDN_RF_STREAM_REPLICAS.patch) |
| [`MOJOLEARN_IDN_RF_TASK_ROWS256`](#mojolearn_idn_rf_task_rows256) | Trees | noise | 2026-10-07 | [MOJOLEARN_IDN_RF_TASK_ROWS256.patch](../experiments/removed/MOJOLEARN_IDN_RF_TASK_ROWS256.patch) |
| [`MOJOLEARN_IF_QUERY_RAW`](#mojolearn_if_query_raw) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_IF_SAMPLED_UPLOAD`](#mojolearn_if_sampled_upload) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_ORDERED_FOLD_DERIVS`](#mojolearn_ordered_fold_derivs) | Trees | DROPPED-slower | 2026-10-03 | [MOJOLEARN_ORDERED_FOLD_DERIVS.patch](../experiments/removed/MOJOLEARN_ORDERED_FOLD_DERIVS.patch) |
| [`MOJOLEARN_ORD_FOLD_BINS_ONE`](#mojolearn_ord_fold_bins_one) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_ORD_FOLD_INDEX`](#mojolearn_ord_fold_index) | Trees | DROPPED-quality |  | lane only |
| [`MOJOLEARN_ORD_STD_PARALLEL`](#mojolearn_ord_std_parallel) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_ORD_TREE_LEAN`](#mojolearn_ord_tree_lean) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_REORDER_FLAGS_SCAN_BLOCK`](#mojolearn_reorder_flags_scan_block) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_RF_FAST_BATCH16K`](#mojolearn_rf_fast_batch16k) | Trees | DROPPED-noise | 2026-10-03 | [MOJOLEARN_RF_FAST_BATCH16K.patch](../experiments/removed/MOJOLEARN_RF_FAST_BATCH16K.patch) |
| [`MOJOLEARN_RF_NODESPLIT_ZERO_AFTER_READ`](#mojolearn_rf_nodesplit_zero_after_read) | Trees | DROPPED-noise | 2026-10-03 | [MOJOLEARN_RF_NODESPLIT_ZERO_AFTER_READ.patch](../experiments/removed/MOJOLEARN_RF_NODESPLIT_ZERO_AFTER_READ.patch) |
| [`MOJOLEARN_RF_SMALL_NODE_1024`](#mojolearn_rf_small_node_1024) | Trees | DROPPED-noise | 2026-10-03 | [MOJOLEARN_RF_SMALL_NODE_1024.patch](../experiments/removed/MOJOLEARN_RF_SMALL_NODE_1024.patch) |
| [`MOJOLEARN_SEG_SCAN_BLOCK`](#mojolearn_seg_scan_block) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_SYM_CTR_PERM_BATCH`](#mojolearn_sym_ctr_perm_batch) | Trees | DROPPED-noise | 2026-10-04 | [MOJOLEARN_SYM_CTR_PERM_BATCH.patch](../experiments/removed/MOJOLEARN_SYM_CTR_PERM_BATCH.patch) |
| [`MOJOLEARN_SYM_DEVICE_LEAVES`](#mojolearn_sym_device_leaves) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_SYM_DEVICE_LEVEL`](#mojolearn_sym_device_level) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_SYM_DEVICE_PARTITION`](#mojolearn_sym_device_partition) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_SYM_HIST_FAST`](#mojolearn_sym_hist_fast) | Trees | DROPPED-noise | 2026-10-03 | [MOJOLEARN_SYM_HIST_FAST.patch](../experiments/removed/MOJOLEARN_SYM_HIST_FAST.patch) |
| [`MOJOLEARN_SYM_NO_TAIL_DRAIN`](#mojolearn_sym_no_tail_drain) | Trees | DROPPED-noise |  | lane only |
| [`MOJOLEARN_TREES_C47_GBDT`](#mojolearn_trees_c47_gbdt) | Trees | DROPPED | 2026-10-07 | [MOJOLEARN_TREES_C47_GBDT.patch](../experiments/removed/MOJOLEARN_TREES_C47_GBDT.patch) |
| [`MOJOLEARN_TREES_HIST_MULTISTAT`](#mojolearn_trees_hist_multistat) | Trees | slower | 2026-10-08 | [MOJOLEARN_TREES_HIST_MULTISTAT.patch](../experiments/removed/MOJOLEARN_TREES_HIST_MULTISTAT.patch) |
| [`MOJOLEARN_TREES_T01`](#mojolearn_trees_t01) | Trees | slower | 2026-10-07 | [MOJOLEARN_TREES_T01.patch](../experiments/removed/MOJOLEARN_TREES_T01.patch) |
| [`MOJOLEARN_TREES_T01_REPLICAS`](#mojolearn_trees_t01_replicas) | Trees | removed | 2026-10-07 | [MOJOLEARN_TREES_T01_REPLICAS.patch](../experiments/removed/MOJOLEARN_TREES_T01_REPLICAS.patch) |
| [`MOJOLEARN_TREES_T01_ROWS`](#mojolearn_trees_t01_rows) | Trees | removed | 2026-10-07 | [MOJOLEARN_TREES_T01_REPLICAS.patch](../experiments/removed/MOJOLEARN_TREES_T01_REPLICAS.patch) |
| [`MOJOLEARN_TREES_T09`](#mojolearn_trees_t09) | Trees | dead code | 2026-10-07 | [MOJOLEARN_TREES_T09.patch](../experiments/removed/MOJOLEARN_TREES_T09.patch) |
| [`MOJOLEARN_TREES_T16`](#mojolearn_trees_t16) | Trees | slower | 2026-10-07 | [MOJOLEARN_TREES_T16.patch](../experiments/removed/MOJOLEARN_TREES_T16.patch) |
| [`MOJOLEARN_TREES_T19`](#mojolearn_trees_t19) | Trees | quality loss | 2026-10-08 | [MOJOLEARN_TREES_T19.patch](../experiments/removed/MOJOLEARN_TREES_T19.patch) |
| [`MOJOLEARN_TREES_T20`](#mojolearn_trees_t20) | Trees | DROPPED | 2026-10-07 | [MOJOLEARN_TREES_T20.patch](../experiments/removed/MOJOLEARN_TREES_T20.patch) |
| [`MOJOLEARN_TREES_T27`](#mojolearn_trees_t27) | Trees | dead code | 2026-10-07 | [MOJOLEARN_TREES_T27.patch](../experiments/removed/MOJOLEARN_TREES_T27.patch) |
| [`MOJOLEARN_TREES_T29_VERSIONED`](#mojolearn_trees_t29_versioned) | Trees | slower | 2026-10-08 | [MOJOLEARN_TREES_T29_VERSIONED.patch](../experiments/removed/MOJOLEARN_TREES_T29_VERSIONED.patch) |
| [`MOJOLEARN_TREES_T29_YETI`](#mojolearn_trees_t29_yeti) | Trees | dead code | 2026-10-07 | [MOJOLEARN_TREES_T29_YETI.patch](../experiments/removed/MOJOLEARN_TREES_T29_YETI.patch) |
| [`MOJOLEARN_TREES_T31_PACKED_A`](#mojolearn_trees_t31_packed_a) | Trees | dead code | 2026-10-07 | [MOJOLEARN_TREES_T31_PACKED_A.patch](../experiments/removed/MOJOLEARN_TREES_T31_PACKED_A.patch) |
| [`MOJOLEARN_YETI_SYM_HIST_UNROLL8`](#mojolearn_yeti_sym_hist_unroll8) | Trees | DROPPED-slower | 2026-10-03 | [MOJOLEARN_YETI_SYM_HIST_UNROLL8.patch](../experiments/removed/MOJOLEARN_YETI_SYM_HIST_UNROLL8.patch) |
| [`MOJOLEARN_ARD_EQ_ONEPASS`](#mojolearn_ard_eq_onepass) | Linear | DROPPED-quality | 2026-10-05 | lane only |
| [`MOJOLEARN_ARD_EQ_ONEPASS_OFF`](#mojolearn_ard_eq_onepass_off) | Linear | DROPPED-quality | 2026-10-05 | lane only |
| [`MOJOLEARN_C13_FOLD_STATS`](#mojolearn_c13_fold_stats) | Linear | DROPPED | 2026-10-07 | lane only |
| [`MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS`](#mojolearn_classical_c13_cd_fold_stats) | Linear | quality loss | 2026-10-07 | [MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS.patch](../experiments/removed/MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS.patch) |
| [`MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF`](#mojolearn_classical_c13_fold_stats_off) | Linear | slower | 2026-10-08 | [MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF.patch](../experiments/removed/MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF.patch) |
| [`MOJOLEARN_CLASSICAL_C18_GRAM_PREFETCH`](#mojolearn_classical_c18_gram_prefetch) | Linear | unmeasured | 2026-10-07 | [MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS.patch](../experiments/removed/MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS.patch) |
| [`MOJOLEARN_CLASSICAL_C18_TILE64`](#mojolearn_classical_c18_tile64) | Linear | unmeasured | 2026-10-07 | [MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS.patch](../experiments/removed/MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS.patch) |
| [`MOJOLEARN_CLASSICAL_C20_ROW_CACHE`](#mojolearn_classical_c20_row_cache) | Linear | serial shape | 2026-10-07 | [MOJOLEARN_CLASSICAL_C20_ROW_CACHE.patch](../experiments/removed/MOJOLEARN_CLASSICAL_C20_ROW_CACHE.patch) |
| [`MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF`](#mojolearn_idn_ols_one_entry_off) | Linear | slower | 2026-10-08 | [MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF.patch](../experiments/removed/MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF.patch) |
| [`MOJOLEARN_IDN_SGD_EPOCH_KERNEL`](#mojolearn_idn_sgd_epoch_kernel) | Linear | noise | 2026-10-09 | [MOJOLEARN_IDN_SGD_EPOCH_KERNEL.patch](../experiments/removed/MOJOLEARN_IDN_SGD_EPOCH_KERNEL.patch) |
| [`MOJOLEARN_KERNEL_FAST_BAYES_JACOBI`](#mojolearn_kernel_fast_bayes_jacobi) | Linear | DROPPED-noise | 2026-10-03 | [MOJOLEARN_KERNEL_FAST_BAYES_JACOBI.patch](../experiments/removed/MOJOLEARN_KERNEL_FAST_BAYES_JACOBI.patch) |
| [`MOJOLEARN_KERNEL_FAST_BAYES_STATS`](#mojolearn_kernel_fast_bayes_stats) | Linear | DROPPED-slower | 2026-10-03 | [MOJOLEARN_KERNEL_FAST_BAYES_STATS.patch](../experiments/removed/MOJOLEARN_KERNEL_FAST_BAYES_STATS.patch) |
| [`MOJOLEARN_LSVR_DEVICE_CONVERGE`](#mojolearn_lsvr_device_converge) | Linear | DROPPED-noise |  | lane only |
| [`MOJOLEARN_LSVR_DUAL_CD`](#mojolearn_lsvr_dual_cd) | Linear | DROPPED-slower |  | lane only |
| [`MOJOLEARN_LSVR_EVAL_SLIM`](#mojolearn_lsvr_eval_slim) | Linear | DROPPED-noise |  | lane only |
| [`MOJOLEARN_LSVR_FASTPATH_FIX`](#mojolearn_lsvr_fastpath_fix) | Linear | DROPPED-slower |  | lane only |
| [`MOJOLEARN_LSVR_FUSED_GRAD`](#mojolearn_lsvr_fused_grad) | Linear | DROPPED-noise |  | lane only |
| [`MOJOLEARN_LSVR_LINESEARCH_BATCH`](#mojolearn_lsvr_linesearch_batch) | Linear | DROPPED-noise |  | lane only |
| [`MOJOLEARN_OLS_FAST_DEVICE_CENTER`](#mojolearn_ols_fast_device_center) | Linear | DROPPED-semantics | 2026-10-03 | [MOJOLEARN_OLS_FAST_DEVICE_CENTER.patch](../experiments/removed/MOJOLEARN_OLS_FAST_DEVICE_CENTER.patch) |
| [`MOJOLEARN_QN_FAST_GRID_SUMS`](#mojolearn_qn_fast_grid_sums) | Linear | DROPPED-noise | 2026-10-03 | [MOJOLEARN_QN_FAST_GRID_SUMS.patch](../experiments/removed/MOJOLEARN_QN_FAST_GRID_SUMS.patch) |
| [`MOJOLEARN_QN_IDN_DCONV`](#mojolearn_qn_idn_dconv) | Linear | slower | 2026-10-08 | [MOJOLEARN_QN_IDN_DCONV.patch](../experiments/removed/MOJOLEARN_QN_IDN_DCONV.patch) |
| [`MOJOLEARN_QN_IDN_DCONV_POLL_2`](#mojolearn_qn_idn_dconv_poll_2) | Linear | slower | 2026-10-08 | [MOJOLEARN_QN_IDN_DCONV.patch](../experiments/removed/MOJOLEARN_QN_IDN_DCONV.patch) |
| [`MOJOLEARN_QN_IDN_DCONV_POLL_8`](#mojolearn_qn_idn_dconv_poll_8) | Linear | slower | 2026-10-08 | [MOJOLEARN_QN_IDN_DCONV.patch](../experiments/removed/MOJOLEARN_QN_IDN_DCONV.patch) |
| [`MOJOLEARN_RIDGE_FAST_CLS1_PREDICT`](#mojolearn_ridge_fast_cls1_predict) | Linear | DROPPED-noise | 2026-10-03 | [MOJOLEARN_RIDGE_FAST_CLS1_PREDICT.patch](../experiments/removed/MOJOLEARN_RIDGE_FAST_CLS1_PREDICT.patch) |
| [`MOJOLEARN_SGD_PERC_QOLD`](#mojolearn_sgd_perc_qold) | Linear | DROPPED-quality | 2026-10-04 | [MOJOLEARN_SGD_PERC_QOLD.patch](../experiments/removed/MOJOLEARN_SGD_PERC_QOLD.patch) |
| [`MOJOLEARN_C29_STREAM_TOPK`](#mojolearn_c29_stream_topk) | Neighbors | serial shape | 2026-10-07 | [MOJOLEARN_C29_STREAM_TOPK.patch](../experiments/removed/MOJOLEARN_C29_STREAM_TOPK.patch) |
| [`MOJOLEARN_C29_TILE`](#mojolearn_c29_tile) | Neighbors | serial shape | 2026-10-07 | [MOJOLEARN_C29_TILE.patch](../experiments/removed/MOJOLEARN_C29_TILE.patch) |
| [`MOJOLEARN_CAGRA_FAST_DOT`](#mojolearn_cagra_fast_dot) | Neighbors | DROPPED-semantics | 2026-10-03 | [MOJOLEARN_CAGRA_FAST_DOT.patch](../experiments/removed/MOJOLEARN_CAGRA_FAST_DOT.patch) |
| [`MOJOLEARN_CAGRA_FAST_IVFG_P32`](#mojolearn_cagra_fast_ivfg_p32) | Neighbors | DROPPED-quality | 2026-10-03 | [MOJOLEARN_CAGRA_FAST_DOT.patch](../experiments/removed/MOJOLEARN_CAGRA_FAST_DOT.patch) |
| [`MOJOLEARN_CAGRA_FAST_TEAM`](#mojolearn_cagra_fast_team) | Neighbors | DROPPED-noise | 2026-10-02 | [MOJOLEARN_CAGRA_FAST_TEAM.patch](../experiments/removed/MOJOLEARN_CAGRA_FAST_TEAM.patch) |
| [`MOJOLEARN_CAGRA_FAST_WIDE`](#mojolearn_cagra_fast_wide) | Neighbors | DROPPED-semantics | 2026-10-03 | [MOJOLEARN_CAGRA_FAST_DOT.patch](../experiments/removed/MOJOLEARN_CAGRA_FAST_DOT.patch) |
| [`MOJOLEARN_IDN_PQ_SCAN_FUSED`](#mojolearn_idn_pq_scan_fused) | Neighbors | slower | 2026-10-09 | [MOJOLEARN_IDN_PQ_SCAN_FUSED.patch](../experiments/removed/MOJOLEARN_IDN_PQ_SCAN_FUSED.patch) |
| [`MOJOLEARN_IVFG_EXACTD`](#mojolearn_ivfg_exactd) | Neighbors | DROPPED-semantics |  | lane only |
| [`MOJOLEARN_IVFG_P8`](#mojolearn_ivfg_p8) | Neighbors | DROPPED-semantics |  | lane only |
| [`MOJOLEARN_IVF_FAST_DEVICE_CSR`](#mojolearn_ivf_fast_device_csr) | Neighbors | DROPPED-slower | 2026-10-02 | [MOJOLEARN_IVF_FAST_DEVICE_CSR.patch](../experiments/removed/MOJOLEARN_IVF_FAST_DEVICE_CSR.patch) |
| [`MOJOLEARN_IVF_FAST_DEVICE_TRAINSET`](#mojolearn_ivf_fast_device_trainset) | Neighbors | DROPPED-noise | 2026-10-02 | [MOJOLEARN_IVF_FAST_DEVICE_TRAINSET.patch](../experiments/removed/MOJOLEARN_IVF_FAST_DEVICE_TRAINSET.patch) |
| [`MOJOLEARN_IVF_FAST_SCAN_SELECT`](#mojolearn_ivf_fast_scan_select) | Neighbors | DROPPED-noise | 2026-10-02 | [MOJOLEARN_IVF_FAST_SCAN_SELECT.patch](../experiments/removed/MOJOLEARN_IVF_FAST_SCAN_SELECT.patch) |
| [`MOJOLEARN_IVF_LAYOUT_SCATTER`](#mojolearn_ivf_layout_scatter) | Neighbors | slower | 2026-10-09 | [MOJOLEARN_IVF_LAYOUT_SCATTER.patch](../experiments/removed/MOJOLEARN_IVF_LAYOUT_SCATTER.patch) |
| [`MOJOLEARN_KNN_FAST_CLS1_PRESEED`](#mojolearn_knn_fast_cls1_preseed) | Neighbors | DROPPED-noise | 2026-10-03 | [MOJOLEARN_KNN_FAST_CLS1_PRESEED.patch](../experiments/removed/MOJOLEARN_KNN_FAST_CLS1_PRESEED.patch) |
| [`MOJOLEARN_KNN_FAST_CLS1_SLICES2`](#mojolearn_knn_fast_cls1_slices2) | Neighbors | DROPPED-slower | 2026-10-03 | [MOJOLEARN_KNN_FAST_CLS1_PRESEED.patch](../experiments/removed/MOJOLEARN_KNN_FAST_CLS1_PRESEED.patch) |
| [`MOJOLEARN_LLE_FAST_KNN`](#mojolearn_lle_fast_knn) | Neighbors | DROPPED-noise |  | lane only |
| [`MOJOLEARN_LLE_SPARSE_EIG`](#mojolearn_lle_sparse_eig) | Neighbors | DROPPED-slower |  | lane only |
| [`MOJOLEARN_NC_FAST_CLS1_PREDICT`](#mojolearn_nc_fast_cls1_predict) | Neighbors | DROPPED-noise | 2026-10-03 | [MOJOLEARN_NC_FAST_CLS1_PREDICT.patch](../experiments/removed/MOJOLEARN_NC_FAST_CLS1_PREDICT.patch) |
| [`MOJOLEARN_RADIUS_FAST_REUSE_COUNT`](#mojolearn_radius_fast_reuse_count) | Neighbors | DROPPED-noise | 2026-10-03 | [MOJOLEARN_RADIUS_FAST_REUSE_COUNT.patch](../experiments/removed/MOJOLEARN_RADIUS_FAST_REUSE_COUNT.patch) |
| [`MOJOLEARN_XN_FAST_IMPUTE_TILED2`](#mojolearn_xn_fast_impute_tiled2) | Neighbors | DROPPED-slower |  | lane only |
| [`MOJOLEARN_XN_FAST_MMA_ROUTE`](#mojolearn_xn_fast_mma_route) | Neighbors | DROPPED-slower |  | lane only |
| [`MOJOLEARN_XN_FAST_TILED_RBF`](#mojolearn_xn_fast_tiled_rbf) | Neighbors | DROPPED-noise | 2026-10-03 | [MOJOLEARN_XN_FAST_TILED_RBF.patch](../experiments/removed/MOJOLEARN_XN_FAST_TILED_RBF.patch) |
| [`MOJOLEARN_XN_PCS_SPARSE`](#mojolearn_xn_pcs_sparse) | Neighbors | DROPPED-noise | 2026-10-03 | [MOJOLEARN_XN_PCS_SPARSE.patch](../experiments/removed/MOJOLEARN_XN_PCS_SPARSE.patch) |
| [`MOJOLEARN_CAGRA_FAST_IVFG_LOWD`](#mojolearn_cagra_fast_ivfg_lowd) | Neighbors | DROPPED-quality | 2026-10-09 | [MOJOLEARN_CAGRA_FAST_IVFG_LOWD.patch](../experiments/removed/MOJOLEARN_CAGRA_FAST_IVFG_LOWD.patch) |
| [`MOJOLEARN_CAGRA_FAST_SEEDS4`](#mojolearn_cagra_fast_seeds4) | Neighbors | DROPPED-semantics | 2026-10-09 | [MOJOLEARN_CAGRA_FAST_SEEDS4.patch](../experiments/removed/MOJOLEARN_CAGRA_FAST_SEEDS4.patch) |
| [`MOJOLEARN_ACHI2_FAST_DEVCHECK`](#mojolearn_achi2_fast_devcheck) | Prep | DROPPED | Oct 3 | lane only |
| [`MOJOLEARN_CLASSICAL_C55_CLASS_GROUP`](#mojolearn_classical_c55_class_group) | Prep | quality loss | 2026-10-07 | [MOJOLEARN_CLASSICAL_C55_CLASS_GROUP.patch](../experiments/removed/MOJOLEARN_CLASSICAL_C55_CLASS_GROUP.patch) |
| [`MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS`](#mojolearn_classical_c61_da_class_stats) | Prep | slower | 2026-10-08 | [MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS.patch](../experiments/removed/MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS.patch) |
| [`MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS`](#mojolearn_classical_c61_nb_class_stats) | Prep | slower | 2026-10-08 | [MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS.patch](../experiments/removed/MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS.patch) |
| [`MOJOLEARN_KSHAP_FAST_SIGNGRAM`](#mojolearn_kshap_fast_signgram) | Prep | DROPPED | Oct 3 | lane only |
| [`MOJOLEARN_PREP3_LABELS`](#mojolearn_prep3_labels) | Prep | DROPPED-noise |  | lane only |
| [`MOJOLEARN_PREP3_SPLINE`](#mojolearn_prep3_spline) | Prep | DROPPED-noise |  | lane only |
| [`MOJOLEARN_RESAMPLE_FAST_IDX_BULK`](#mojolearn_resample_fast_idx_bulk) | Prep | DROPPED-semantics |  | lane only |
| [`MOJOLEARN_RESAMPLE_FAST_TAKE`](#mojolearn_resample_fast_take) | Prep | DROPPED-semantics | 2026-10-04 | lane only |
| [`MOJOLEARN_SHAP_FAST_PIPE`](#mojolearn_shap_fast_pipe) | Prep | DROPPED-speed | 2026-10-04 | lane only |
| [`MOJOLEARN_XPREP_DEVICE_CODES`](#mojolearn_xprep_device_codes) | Prep | slower | 2026-10-09 | [MOJOLEARN_XPREP_DEVICE_CODES.patch](../experiments/removed/MOJOLEARN_XPREP_DEVICE_CODES.patch) |
| [`MOJOLEARN_XPREP_NO_SLOT_HOP`](#mojolearn_xprep_no_slot_hop) | Prep | slower | 2026-10-09 | [MOJOLEARN_XPREP_NO_SLOT_HOP.patch](../experiments/removed/MOJOLEARN_XPREP_NO_SLOT_HOP.patch) |
| [`MOJOLEARN_XPREP_PINNED_UPLOAD`](#mojolearn_xprep_pinned_upload) | Prep | slower | 2026-10-09 | [MOJOLEARN_XPREP_PINNED_UPLOAD.patch](../experiments/removed/MOJOLEARN_XPREP_PINNED_UPLOAD.patch) |
| [`MOJOLEARN_XPREP_PINNED_UPLOAD_OFF`](#mojolearn_xprep_pinned_upload_off) | Prep | slower | 2026-10-09 | [MOJOLEARN_XPREP_PINNED_UPLOAD.patch](../experiments/removed/MOJOLEARN_XPREP_PINNED_UPLOAD.patch) |
| [`MOJOLEARN_X_PREP_FAST_NONEG`](#mojolearn_x_prep_fast_noneg) | Prep | DROPPED-noise | 2026-10-03 | [MOJOLEARN_X_PREP_FAST_NONEG.patch](../experiments/removed/MOJOLEARN_X_PREP_FAST_NONEG.patch) |
| [`MOJOLEARN_X_PREP_PINNED_OUT`](#mojolearn_x_prep_pinned_out) | Prep | DROPPED | 2026-10-04 | lane only |
| [`MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE`](#mojolearn_classical_c25_projection_reuse) | Decomp | slower | 2026-10-08 | [MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE.patch](../experiments/removed/MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE.patch) |
| [`MOJOLEARN_CLASSICAL_PCA_COV=23`](#mojolearn_classical_pca_cov-arm23) | Decomp | slower | 2026-10-08 | [MOJOLEARN_CLASSICAL_PCA_COV-arm23.patch](../experiments/removed/MOJOLEARN_CLASSICAL_PCA_COV-arm23.patch) |
| [`MOJOLEARN_DECOMP_FAST_SMALL_EIGH_J2`](#mojolearn_decomp_fast_small_eigh_j2) | Decomp | DROPPED-semantics |  | lane only |
| [`MOJOLEARN_EIGH_TANGENT_CACHE`](#mojolearn_eigh_tangent_cache) | Decomp | DROPPED-speed | 2026-10-04 | lane only |
| [`MOJOLEARN_IDN_PCA_RR_ONE_BLOCK`](#mojolearn_idn_pca_rr_one_block) | Decomp | broken | 2026-10-09 | [MOJOLEARN_IDN_PCA_RR_ONE_BLOCK.patch](../experiments/removed/MOJOLEARN_IDN_PCA_RR_ONE_BLOCK.patch) |
| [`MOJOLEARN_IDN_PCA_RR_ONE_BLOCK_STEPS`](#mojolearn_idn_pca_rr_one_block_steps) | Decomp | broken | 2026-10-09 | [MOJOLEARN_IDN_PCA_RR_ONE_BLOCK.patch](../experiments/removed/MOJOLEARN_IDN_PCA_RR_ONE_BLOCK.patch) |
| [`MOJOLEARN_LU_FAST_TSLU`](#mojolearn_lu_fast_tslu) | Decomp | DROPPED-quality | 2026-10-04 | lane only |
| [`MOJOLEARN_PCA_FAST_EIG`](#mojolearn_pca_fast_eig) | Decomp | DROPPED-noise |  | lane only |
| [`MOJOLEARN_PCA_FAST_NO_ALIAS`](#mojolearn_pca_fast_no_alias) | Decomp | DROPPED-noise |  | lane only |
| [`MOJOLEARN_PCA_FAST_TOPK`](#mojolearn_pca_fast_topk) | Decomp | DROPPED-noise |  | lane only |
| [`MOJOLEARN_TSNE_FAST_SPLIT`](#mojolearn_tsne_fast_split) | Decomp | DROPPED-quality | 2026-09-28 | [MOJOLEARN_TSNE_FAST_SPLIT.patch](../experiments/removed/MOJOLEARN_TSNE_FAST_SPLIT.patch) |
| [`MOJOLEARN_CHOL_FAST_BLOCKED`](#mojolearn_chol_fast_blocked) | Decomp | DROPPED-slower+quality | 2026-10-09 | [MOJOLEARN_CHOL_FAST_BLOCKED.patch](../experiments/removed/MOJOLEARN_CHOL_FAST_BLOCKED.patch) |
| [`MOJOLEARN_DECOMP_FAST_GEMM_TILED`](#mojolearn_decomp_fast_gemm_tiled) | Decomp | DROPPED-slower | 2026-10-09 | [MOJOLEARN_DECOMP_FAST_GEMM_TILED.patch](../experiments/removed/MOJOLEARN_DECOMP_FAST_GEMM_TILED.patch) |
| [`MOJOLEARN_FA_ALL`](#mojolearn_fa_all) | Decomp | DROPPED-slower | 2026-10-09 | [MOJOLEARN_FA_ALL.patch](../experiments/removed/MOJOLEARN_FA_ALL.patch) |
| [`MOJOLEARN_FA_EIG_SMALL`](#mojolearn_fa_eig_small) | Decomp | DROPPED-slower | 2026-10-09 | [MOJOLEARN_FA_EIG_SMALL.patch](../experiments/removed/MOJOLEARN_FA_EIG_SMALL.patch) |
| [`MOJOLEARN_FA_LL_DEVICE`](#mojolearn_fa_ll_device) | Decomp | DROPPED-slower | 2026-10-09 | [MOJOLEARN_FA_LL_DEVICE.patch](../experiments/removed/MOJOLEARN_FA_LL_DEVICE.patch) |
| [`MOJOLEARN_AFFINITY_FAST_LOOP`](#mojolearn_affinity_fast_loop) | Cluster | DROPPED-noise | 2026-10-02 | [MOJOLEARN_AFFINITY_FAST_LOOP.patch](../experiments/removed/MOJOLEARN_AFFINITY_FAST_LOOP.patch) |
| [`MOJOLEARN_BGMM_ENT`](#mojolearn_bgmm_ent) | Cluster | DROPPED-noise | 2026-10-03 | [MOJOLEARN_BGMM_ENT.patch](../experiments/removed/MOJOLEARN_BGMM_ENT.patch) |
| [`MOJOLEARN_BISECT_FAST_RESIDENT`](#mojolearn_bisect_fast_resident) | Cluster | DROPPED-slower | 2026-10-02 | [MOJOLEARN_BISECT_FAST_RESIDENT.patch](../experiments/removed/MOJOLEARN_BISECT_FAST_RESIDENT.patch) |
| [`MOJOLEARN_C37_FUSED_ACCUMULATE`](#mojolearn_c37_fused_accumulate) | Cluster | slower | 2026-10-08 | [MOJOLEARN_C37_FUSED_ACCUMULATE.patch](../experiments/removed/MOJOLEARN_C37_FUSED_ACCUMULATE.patch) |
| [`MOJOLEARN_C37_FUSED_ROWS`](#mojolearn_c37_fused_rows) | Cluster | slower | 2026-10-08 | [MOJOLEARN_C37_FUSED_ACCUMULATE.patch](../experiments/removed/MOJOLEARN_C37_FUSED_ACCUMULATE.patch) |
| [`MOJOLEARN_C37_PANEL_128`](#mojolearn_c37_panel_128) | Cluster | DROPPED | 2026-10-07 | [MOJOLEARN_C37_PANEL_128.patch](../experiments/removed/MOJOLEARN_C37_PANEL_128.patch) |
| [`MOJOLEARN_C37_ROW_PANELS`](#mojolearn_c37_row_panels) | Cluster | serial shape | 2026-10-07 | [MOJOLEARN_C37_PANEL_128.patch](../experiments/removed/MOJOLEARN_C37_PANEL_128.patch) |
| [`MOJOLEARN_C38_DEVICE_POTENTIAL`](#mojolearn_c38_device_potential) | Cluster | dead code | 2026-10-07 | [MOJOLEARN_C37_PANEL_128.patch](../experiments/removed/MOJOLEARN_C37_PANEL_128.patch) |
| [`MOJOLEARN_C38_REUSE_NEAREST`](#mojolearn_c38_reuse_nearest) | Cluster | dead code | 2026-10-07 | [MOJOLEARN_C37_PANEL_128.patch](../experiments/removed/MOJOLEARN_C37_PANEL_128.patch) |
| [`MOJOLEARN_CC_FAST_OFF`](#mojolearn_cc_fast_off) | Cluster | removed | 2026-10-08 | [MOJOLEARN_CC_FAST_OFF.patch](../experiments/removed/MOJOLEARN_CC_FAST_OFF.patch) |
| [`MOJOLEARN_GMM_FAST_ESTEP_STACK`](#mojolearn_gmm_fast_estep_stack) | Cluster | DROPPED-slower | 2026-10-03 | [MOJOLEARN_GMM_FAST_ESTEP_STACK.patch](../experiments/removed/MOJOLEARN_GMM_FAST_ESTEP_STACK.patch) |
| [`MOJOLEARN_GMM_FAST_GRID_COV`](#mojolearn_gmm_fast_grid_cov) | Cluster | DROPPED-slower | 2026-10-03 | [MOJOLEARN_GMM_FAST_GRID_COV.patch](../experiments/removed/MOJOLEARN_GMM_FAST_GRID_COV.patch) |
| [`MOJOLEARN_GRAPH_DIRECT_DISTANCE`](#mojolearn_graph_direct_distance) | Cluster | slower | 2026-10-08 | [MOJOLEARN_GRAPH_DIRECT_DISTANCE.patch](../experiments/removed/MOJOLEARN_GRAPH_DIRECT_DISTANCE.patch) |
| [`MOJOLEARN_IDN_GMM_COV_SYM`](#mojolearn_idn_gmm_cov_sym) | Cluster | quality loss | 2026-10-08 | [MOJOLEARN_IDN_GMM_COV_SYM.patch](../experiments/removed/MOJOLEARN_IDN_GMM_COV_SYM.patch) |
| [`MOJOLEARN_KMEANS_DIRECT_DISTANCE`](#mojolearn_kmeans_direct_distance) | Cluster | slower | 2026-10-08 | [MOJOLEARN_KMEANS_DIRECT_DISTANCE.patch](../experiments/removed/MOJOLEARN_KMEANS_DIRECT_DISTANCE.patch) |
| [`MOJOLEARN_KMEANS_FAST_ROWNORM`](#mojolearn_kmeans_fast_rownorm) | Cluster | DROPPED-noise | 2026-10-03 | [MOJOLEARN_KMEANS_FAST_ROWNORM.patch](../experiments/removed/MOJOLEARN_KMEANS_FAST_ROWNORM.patch) |
| [`MOJOLEARN_KMEANS_FAST_SKIP_PREDICT`](#mojolearn_kmeans_fast_skip_predict) | Cluster | DROPPED-noise | 2026-10-03 | [MOJOLEARN_KMEANS_FAST_SKIP_PREDICT.patch](../experiments/removed/MOJOLEARN_KMEANS_FAST_SKIP_PREDICT.patch) |
| [`MOJOLEARN_OPTICS2_ALL`](#mojolearn_optics2_all) | Cluster | DROPPED |  | lane only |
| [`MOJOLEARN_OPTICS_CORE_SQ`](#mojolearn_optics_core_sq) | Cluster | DROPPED |  | lane only |
| [`MOJOLEARN_OPTICS_FRONTIER_DEVICE`](#mojolearn_optics_frontier_device) | Cluster | DROPPED |  | lane only |
| [`MOJOLEARN_OPTICS_LIVEBUF`](#mojolearn_optics_livebuf) | Cluster | DROPPED |  | lane only |
| [`MOJOLEARN_OPTICS_STEP_BATCH`](#mojolearn_optics_step_batch) | Cluster | DROPPED |  | lane only |
| [`MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_FIN`](#mojolearn_x_cluster_fast_cls2_mbk_fin) | Cluster | DROPPED | 2026-10-03 | [MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_FIN.patch](../experiments/removed/MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_FIN.patch) |
| [`MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_G128`](#mojolearn_x_cluster_fast_cls2_mbk_g128) | Cluster | DROPPED | 2026-10-03 | [MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_FIN.patch](../experiments/removed/MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_FIN.patch) |
| [`MOJOLEARN_BGMM_ESTEP1`](#mojolearn_bgmm_estep1) | Cluster | DROPPED-noise | 2026-10-09 | [MOJOLEARN_BGMM_ESTEP1.patch](../experiments/removed/MOJOLEARN_BGMM_ESTEP1.patch) |
| [`MOJOLEARN_ARIMA_FAST_LS_NOREAD`](#mojolearn_arima_fast_ls_noread) | Time series | DROPPED-noise | 2026-10-03 | [MOJOLEARN_ARIMA_FAST_LS_NOREAD.patch](../experiments/removed/MOJOLEARN_ARIMA_FAST_LS_NOREAD.patch) |
| [`MOJOLEARN_ARIMA_FAST_P_FIX`](#mojolearn_arima_fast_p_fix) | Time series | DROPPED-slower | 2026-10-03 | [MOJOLEARN_ARIMA_FAST_P_FIX.patch](../experiments/removed/MOJOLEARN_ARIMA_FAST_P_FIX.patch) |
| [`MOJOLEARN_C58_FORECAST4`](#mojolearn_c58_forecast4) | Time series | slower | 2026-10-08 | [MOJOLEARN_C58_FORECAST4.patch](../experiments/removed/MOJOLEARN_C58_FORECAST4.patch) |
| [`MOJOLEARN_C58_SHARED_PREP`](#mojolearn_c58_shared_prep) | Time series | slower | 2026-10-08 | [MOJOLEARN_C58_SHARED_PREP.patch](../experiments/removed/MOJOLEARN_C58_SHARED_PREP.patch) |
| [`MOJOLEARN_C60_DIFF_REUSE`](#mojolearn_c60_diff_reuse) | Time series | dead code | 2026-10-07 | [MOJOLEARN_C60_DIFF_REUSE.patch](../experiments/removed/MOJOLEARN_C60_DIFF_REUSE.patch) |
| [`MOJOLEARN_SEQ_FAST_THETA_HOIST`](#mojolearn_seq_fast_theta_hoist) | Time series | DROPPED-noise | 2026-10-03 | [MOJOLEARN_SEQ_FAST_THETA_HOIST.patch](../experiments/removed/MOJOLEARN_SEQ_FAST_THETA_HOIST.patch) |
| [`MOJOLEARN_SEQ_FAST_VAR_SPEC`](#mojolearn_seq_fast_var_spec) | Time series | DROPPED-noise | 2026-10-03 | [MOJOLEARN_SEQ_FAST_VAR_SPEC.patch](../experiments/removed/MOJOLEARN_SEQ_FAST_VAR_SPEC.patch) |
| [`MOJOLEARN_KERNEL_FAST_GPR_RESIDENT`](#mojolearn_kernel_fast_gpr_resident) | Kernel / GP | DROPPED-semantics | 2026-10-03 | [MOJOLEARN_KERNEL_FAST_GPR_RESIDENT.patch](../experiments/removed/MOJOLEARN_KERNEL_FAST_GPR_RESIDENT.patch) |
| [`MOJOLEARN_SVGP_FAST_GPU`](#mojolearn_svgp_fast_gpu) | Kernel / GP | DROPPED-noise |  | lane only |
| [`MOJOLEARN_IDN_AF_VEC_FUSED`](#mojolearn_idn_af_vec_fused) | Neural | noise | 2026-10-09 | [MOJOLEARN_IDN_AF_VEC_FUSED.patch](../experiments/removed/MOJOLEARN_IDN_AF_VEC_FUSED.patch) |
| [`MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE`](#mojolearn_idn_attn_gqa_head_reuse) | Neural | slower | 2026-10-07 | [MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE.patch](../experiments/removed/MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE.patch) |
| [`MOJOLEARN_IDN_ATTN_SOFTMAX=1`](#mojolearn_idn_attn_softmax-arm1) | Neural | broken | 2026-10-08 | [MOJOLEARN_IDN_ATTN_SOFTMAX-arm1.patch](../experiments/removed/MOJOLEARN_IDN_ATTN_SOFTMAX-arm1.patch) |
| [`MOJOLEARN_IDN_NEURAL_LEAF`](#mojolearn_idn_neural_leaf) | Neural | slower | 2026-10-07 | [MOJOLEARN_IDN_NEURAL_LEAF.patch](../experiments/removed/MOJOLEARN_IDN_NEURAL_LEAF.patch) |
| [`MOJOLEARN_IDN_NEURAL_NN05`](#mojolearn_idn_neural_nn05) | Neural | slower | 2026-10-08 | [MOJOLEARN_IDN_NEURAL_NN05.patch](../experiments/removed/MOJOLEARN_IDN_NEURAL_NN05.patch) |
| [`MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN`](#mojolearn_idn_seq_row_serial_scan) | Neural | serial shape | 2026-10-07 | [MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN.patch](../experiments/removed/MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN.patch) |
| [`MOJOLEARN_NI13_CNN_WEIGHT_GENERATION_CACHE`](#mojolearn_ni13_cnn_weight_generation_cache) | Neural | dead code | 2026-10-07 | [MOJOLEARN_NI13_CNN_WEIGHT_GENERATION_CACHE.patch](../experiments/removed/MOJOLEARN_NI13_CNN_WEIGHT_GENERATION_CACHE.patch) |
| [`MOJOLEARN_NN22_EAGER_DKDV_PAIR`](#mojolearn_nn22_eager_dkdv_pair) | Neural | unmeasured | 2026-10-07 | [MOJOLEARN_NN22_EAGER_DKDV_PAIR.patch](../experiments/removed/MOJOLEARN_NN22_EAGER_DKDV_PAIR.patch) |
| [`MOJOLEARN_NN23_ROWDOT_DS`](#mojolearn_nn23_rowdot_ds) | Neural | unmeasured | 2026-10-07 | [MOJOLEARN_NN23_ROWDOT_DS.patch](../experiments/removed/MOJOLEARN_NN23_ROWDOT_DS.patch) |
| [`MOJOLEARN_APPLE_FAST_GEMM_NT_TILED`](#mojolearn_apple_fast_gemm_nt_tiled) | GEMM | DROPPED-slower | 2026-10-03 | [MOJOLEARN_APPLE_FAST_GEMM_NT_TILED.patch](../experiments/removed/MOJOLEARN_APPLE_FAST_GEMM_NT_TILED.patch) |
| [`MOJOLEARN_APPLE_FAST_GEMM_PINNED`](#mojolearn_apple_fast_gemm_pinned) | GEMM | DROPPED-noise | 2026-10-03 | [MOJOLEARN_APPLE_FAST_GEMM_PINNED.patch](../experiments/removed/MOJOLEARN_APPLE_FAST_GEMM_PINNED.patch) |
| [`MOJOLEARN_BGMM_FAST_MAHAL_GEMM`](#mojolearn_bgmm_fast_mahal_gemm) | GEMM | DROPPED-slower |  | lane only |
| [`MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE`](#mojolearn_idn_gemm_compact_live_tile) | GEMM | slower | 2026-10-07 | [MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE.patch](../experiments/removed/MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE.patch) |
| [`MOJOLEARN_IDN_GEMM_FOLD_LEAF_64`](#mojolearn_idn_gemm_fold_leaf_64) | GEMM | slower | 2026-10-07 | [MOJOLEARN_IDN_GEMM_FOLD_LEAF_64.patch](../experiments/removed/MOJOLEARN_IDN_GEMM_FOLD_LEAF_64.patch) |
| [`MOJOLEARN_IDN_GEMM_FS2`](#mojolearn_idn_gemm_fs2) | GEMM | noise | 2026-10-07 | [MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE.patch](../experiments/removed/MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE.patch) |
| [`MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY`](#mojolearn_idn_gemm_group_tiles_body) | GEMM | noise | 2026-10-07 | [MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE.patch](../experiments/removed/MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE.patch) |
| [`MOJOLEARN_IDN_GEMM_OZAKI_LINALG`](#mojolearn_idn_gemm_ozaki_linalg) | GEMM | slower | 2026-10-08 | [MOJOLEARN_IDN_GEMM_OZAKI_LINALG.patch](../experiments/removed/MOJOLEARN_IDN_GEMM_OZAKI_LINALG.patch) |
| [`MOJOLEARN_IDN_GEMM_TILE_SHORT_K`](#mojolearn_idn_gemm_tile_short_k) | GEMM | slower | 2026-10-07 | [MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE.patch](../experiments/removed/MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE.patch) |
| [`MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE=1`](#mojolearn_idn_neural_gemm_epilogue-arm1) | GEMM | slower | 2026-10-09 | [MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE-arm1.patch](../experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE-arm1.patch) |
| [`MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE=3`](#mojolearn_idn_neural_gemm_schedule-arm3) | GEMM | slower | 2026-10-08 | [MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE-arm3.patch](../experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE-arm3.patch) |
| [`MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE=8`](#mojolearn_idn_neural_gemm_schedule-arm8) | GEMM | slower | 2026-10-07 | [MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE-arm8.patch](../experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE-arm8.patch) |
| [`MOJOLEARN_IDN_NN20_SPLIT_KV`](#mojolearn_idn_nn20_split_kv) | Other | slower | 2026-10-08 | [MOJOLEARN_IDN_NN20_SPLIT_KV.patch](../experiments/removed/MOJOLEARN_IDN_NN20_SPLIT_KV.patch) |
| [`MOJOLEARN_IDN_NN20_SPLIT_KV_LEAVES`](#mojolearn_idn_nn20_split_kv_leaves) | Other | slower | 2026-10-08 | [MOJOLEARN_IDN_NN20_SPLIT_KV_LEAVES.patch](../experiments/removed/MOJOLEARN_IDN_NN20_SPLIT_KV_LEAVES.patch) |


## Trees

### MOJOLEARN_ET_DEVICE_BATCH_65536

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `e242fb001` (extratrees: remove dropped ET_DEVICE_BATCH_65536 (DROPPED-slower; recover lane/apple-fast@269ffa57a)).
- Recoverable at `dd47c5df0` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_ET_DEVICE_BATCH_65536.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `extratrees/estimator.mojo`
- EXPERIMENTS.md:120 (Trees (101)): `ET_DEVICE_BATCH_65536` on et / taxi, lane/apple-fast @ 269ffa57a, A/B aft-ab-etb64, aft-ab-etb64b, on PART_ROWS: taxi 3,041 -> 3,080 ms, **DROPPED-slower**: +1.3%; code removed from main e242fb001; recover at lane/apple-fast@269ffa57a

### MOJOLEARN_ET_IDN_BINNED_ANY_WIDTH_OFF

- Verdict: removed. Deleted 2026-10-07 by `d631a057c` (trees: delete duplicate/loser defines (IDN_RF_ROWS_SORTED, IDN_RF_TASK_ROWS256, IDN_RF_DEVICE_LOOP_K*, IDN_ET_BINNED_U16 + host column, T29_YETI no-op)).
- Recoverable at `2394e18e1` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_ET_IDN_BINNED_ANY_WIDTH_OFF.patch` (applies cleanly to main at the time of writing); first apply `experiments/removed/_shared-d631a057c.patch`.
- Files the patch restores: `extratrees/impl/decisiontree/batched_levelalgo/builder.mojo`
- grid_controls/trees-small.json removed: sub-arm of deleted IDN_ET_BINNED_U16

### MOJOLEARN_ET_TPB_256

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `4a78e109a` (extratrees: remove dropped ET_TPB_256 (DROPPED-slower; recover lane/apple-fast@269ffa57a)).
- Recoverable at `e242fb001` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_ET_TPB_256.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `extratrees/impl/decisiontree/batched_levelalgo/builder.mojo`
- EXPERIMENTS.md:121 (Trees (101)): `ET_TPB_256` on et / taxi, istella, lane/apple-fast @ 269ffa57a, A/B aft-ab-ettpb, aft-ab-ettpb2, on PART_ROWS: taxi 3,030 -> 3,072; istella 4,133 -> 4,379 ms, **DROPPED-slower**: +1.4% / +6% (alone it was mixed); code removed from main 4a78e109a; recover at lane/apple-fast@269ffa57a

### MOJOLEARN_GBDT_CTR_FAST_SCAN

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `afda7b9ce` (gbdt scan: remove dropped GBDT_CTR_FAST_SCAN (DROPPED-noise; recover lane/apple-fast-trees-depthwise@f743edd60)).
- Recoverable at `4a78e109a` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_GBDT_CTR_FAST_SCAN.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gbdt/gpu_util/kernel/scan.mojo`, `gbdt/gpu_util/kernel/segmented_scan.mojo`
- EXPERIMENTS.md:122 (Trees (101)): `GBDT_CTR_FAST_FREQ + GBDT_CTR_FAST_SCAN` on categorical, lane/apple-fast-trees-depthwise @ f743edd60, A/B tdw-cat-both, categorical 32,739 -> 27,662 ms, **DROPPED-noise**: same as FREQ alone; SCAN adds nothing; code removed from main afda7b9ce; recover at lane/apple-fast-trees-depthwise@f743edd60
- EXPERIMENTS.md:123 (Trees (101)): `GBDT_CTR_FAST_SCAN` on categorical, lane/apple-fast-trees-depthwise @ f743edd60, A/B tdw-cat-scan, categorical 32,892 -> 33,044 ms, **DROPPED-noise**: +0.5%, overlap; code removed from main afda7b9ce; recover at lane/apple-fast-trees-depthwise@f743edd60

### MOJOLEARN_GBDT_CTR_PERM_PTRS

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `4722f0a88` (gbdt train: remove dropped GBDT_CTR_PERM_PTRS (DROPPED-noise; recover lane/apple-fast-trees-depthwise@f743edd60)).
- Recoverable at `afda7b9ce` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_GBDT_CTR_PERM_PTRS.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gbdt/train.mojo`
- EXPERIMENTS.md:124 (Trees (101)): `GBDT_CTR_PERM_PTRS` on categorical, lane/apple-fast-trees-depthwise @ f743edd60, A/B tdw-cat-permptrs, categorical taxicat 37,098 -> 36,618 ms, **DROPPED-noise**: -1.3%, B runs straddle A; code removed from main 4722f0a88; recover at lane/apple-fast-trees-depthwise@f743edd60

### MOJOLEARN_GBDT_DW2_COPY_ZERO

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `4037c6b9a` (gbdt depthwise: remove dropped GBDT_DW2_COPY_ZERO (DROPPED-slower; recover lane/apple-fast-dwgap2@23eca012b)).
- Recoverable at `4722f0a88` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_GBDT_DW2_COPY_ZERO.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo`, `gbdt/methods/greedy_subsets_searcher/kernel/dw2_level.mojo`
- EXPERIMENTS.md:125 (Trees (101)): `GBDT_DW2_COPY_ZERO` on depthwise / istella, taxi, lane/apple-fast-dwgap2 @ 23eca012b, A/B dw2-copy-zero-taxi, dw2-copy-zero-istella, depthwise taxi 13,051 -> 13,441 ms, **DROPPED-slower**: +3.0%, arms overlap; code removed from main 4037c6b9a; recover at lane/apple-fast-dwgap2@23eca012b

### MOJOLEARN_GBDT_DW_FAST_DEV_SCALE

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-gap-misc @ 5e2eec7a3`.
- EXPERIMENTS.md:126 (Trees (101)): `GBDT_DW_FAST_DEV_SCALE` on depthwise / taxi, lane/apple-fast-gap-misc @ 5e2eec7a3, A/B gapmisc-devscale-dwtaxi, 10,812 -> 10,780 ms, **DROPPED-noise**: -0.3%, overlap

### MOJOLEARN_GBDT_DW_FAST_SKIP_FINAL_STATS

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-gap-misc @ 5e2eec7a3`.
- EXPERIMENTS.md:127 (Trees (101)): `GBDT_DW_FAST_SKIP_FINAL_STATS` on depthwise / taxi, lane/apple-fast-gap-misc @ 5e2eec7a3, A/B gapmisc-skipfs-dwtaxi, 11,108 -> 11,048 ms, **DROPPED-noise**: -0.5%

### MOJOLEARN_GBDT_DW_FLAT_GRID

- Verdict: DROPPED-speed. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-w3-dw 1fb97706a`.
- EXPERIMENTS.md:897 (PT centered-score WIP checkpoint (2026-10-04)): `MOJOLEARN_GBDT_DW_FLAT_GRID` on gbdt-depthwise taxi, lane/apple-fast-w3-dw 1fb97706a, A/B w2-w3dw-*, quality PASS (AUC +0.000107); 11054.9 -> 11822.4 ms ms, **DROP-speed, opt-in only**: 

### MOJOLEARN_GBDT_DW_TREE_SYNC

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `951bf48da` (gbdt depthwise: remove dropped GBDT_DW_TREE_SYNC and GBDT_DW_TREE_SYNC_CHECK (DROPPED-noise; recover lane/apple-fast-depthwise@4547e0d14)).
- Recoverable at `4037c6b9a` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_GBDT_DW_TREE_SYNC.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo`, `gbdt/methods/greedy_subsets_searcher/kernel/dw_tree_sync.mojo`
- EXPERIMENTS.md:128 (Trees (101)): `GBDT_DW_FUSED_CHAIN + GBDT_DW_NO_LEVEL_SYNC + GBDT_DW_TREE_SYNC` on depthwise / taxi, lane/apple-fast-depthwise @ 4547e0d14, A/B dw-tree-taxi, depthwise taxi 14,453 -> 14,386 ms, **DROPPED-noise**: -0.5%, B runs straddle A; code removed from main 951bf48da; recover at lane/apple-fast-depthwise@4547e0d14
- EXPERIMENTS.md:129 (Trees (101)): `GBDT_DW_FUSED_CHAIN + GBDT_DW_NO_LEVEL_SYNC + GBDT_DW_TREE_SYNC + GBDT_DW_TREE_SYNC_CHECK` on depthwise / taxi, lane/apple-fast-depthwise @ 4547e0d14, A/B dw-tree-check-taxi, depthwise taxi 14,485 -> 14,581 ms, **DROPPED-noise**: +0.7%; GBDT_DW_TREE_SYNC: code removed from main 951bf48da; recover at lane/apple-fast-depthwise@4547e0d14; GBDT_DW_TREE_SYNC_CHECK: code removed from main 951bf48da; recover at lane/apple-fast-depthwise@4547e0d14

### MOJOLEARN_GBDT_DW_TREE_SYNC_CHECK

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `951bf48da` (gbdt depthwise: remove dropped GBDT_DW_TREE_SYNC and GBDT_DW_TREE_SYNC_CHECK (DROPPED-noise; recover lane/apple-fast-depthwise@4547e0d14)).
- Recoverable at `4037c6b9a` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_GBDT_DW_TREE_SYNC.patch` (same patch as above).
- Files the patch restores: `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo`, `gbdt/methods/greedy_subsets_searcher/kernel/dw_tree_sync.mojo`
- EXPERIMENTS.md:129 (Trees (101)): `GBDT_DW_FUSED_CHAIN + GBDT_DW_NO_LEVEL_SYNC + GBDT_DW_TREE_SYNC + GBDT_DW_TREE_SYNC_CHECK` on depthwise / taxi, lane/apple-fast-depthwise @ 4547e0d14, A/B dw-tree-check-taxi, depthwise taxi 14,485 -> 14,581 ms, **DROPPED-noise**: +0.7%; GBDT_DW_TREE_SYNC: code removed from main 951bf48da; recover at lane/apple-fast-depthwise@4547e0d14; GBDT_DW_TREE_SYNC_CHECK: code removed from main 951bf48da; recover at lane/apple-fast-depthwise@4547e0d14

### MOJOLEARN_GBDT_LG_EXACT_BATCH16

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `dc23f13c8` (gbdt lossguide: remove dropped GBDT_LG_EXACT_BATCH16 width arm (DROPPED-slower; recover lane/apple-fast-trees2@50dfdcca0)).
- Recoverable at `951bf48da` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_GBDT_LG_EXACT_BATCH16.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo`
- EXPERIMENTS.md:130 (Trees (101)): `GBDT_LG_EXACT_BATCH16` on lossguide / taxi, lane/apple-fast-trees2 @ 50dfdcca0, A/B aft-ab-lgw16, taxi 16,776 -> 18,093 ms, **DROPPED-slower**: +8%; code removed from main dc23f13c8; recover at lane/apple-fast-trees2@50dfdcca0

### MOJOLEARN_GBDT_QH_FAST_FUSED_Q

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-gap-misc @ 5e2eec7a3`.
- EXPERIMENTS.md:131 (Trees (101)): `GBDT_QH_FAST_FUSED_Q` on depthwise / taxi, lane/apple-fast-gap-misc @ 5e2eec7a3, A/B gapmisc-fusedq-dwtaxi, 10,673 -> 10,783 ms, **DROPPED-noise**: +1%, overlap

### MOJOLEARN_GBDT_SEG_SUMS_BLOCK

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `7ca37fd0b` (gbdt segmented sort: remove dropped GBDT_SEG_SUMS_BLOCK (DROPPED-noise; recover lane/apple-fast-rfet-scan@500168cfe)).
- Recoverable at `eccc9be7e` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_GBDT_SEG_SUMS_BLOCK.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `core/segmented_sort.mojo`, `gbdt/gpu_util/kernel/segmented_sort.mojo`
- EXPERIMENTS.md:132 (Trees (101)): `GBDT_SEG_SUMS_BLOCK` on depthwise / taxi, lane/apple-fast-rfet-scan @ 500168cfe, A/B aft-ab-gbseg, 10,325 -> 10,293 ms, **DROPPED-noise**: -0.3%, overlap; stays opt-in; code removed from main 7ca37fd0b; recover at lane/apple-fast-rfet-scan@500168cfe

### MOJOLEARN_GBDT_SM_X4

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `e1b520e88` (gbdt depthwise: remove dropped GBDT_SM_X4 (DROPPED-noise; recover lane/apple-fast-trees2@50dfdcca0)).
- Recoverable at `dc23f13c8` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_GBDT_SM_X4.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo`
- EXPERIMENTS.md:133 (Trees (101)): `GBDT_SM_X4` on depthwise / taxi, lane/apple-fast-trees2 @ 50dfdcca0, A/B aft-ab-smx4, taxi 14,881 -> 14,772 ms, **DROPPED-noise**: arms overlap; code removed from main e1b520e88; recover at lane/apple-fast-trees2@50dfdcca0

### MOJOLEARN_GBDT_SM_X8

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `eccc9be7e` (gbdt depthwise: remove dropped GBDT_SM_X8 (DROPPED-slower; recover lane/apple-fast-trees2@50dfdcca0)).
- Recoverable at `e1b520e88` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_GBDT_SM_X8.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo`
- EXPERIMENTS.md:134 (Trees (101)): `GBDT_SM_X8` on lossguide / taxi, lane/apple-fast-trees2 @ 50dfdcca0, A/B aft-ab-smx8, taxi 16,483 -> 19,577 ms, **DROPPED-slower**: +19%; code removed from main eccc9be7e; recover at lane/apple-fast-trees2@50dfdcca0

### MOJOLEARN_IDN_ET_BINNED_U16

- Verdict: slower. Deleted 2026-10-07 by `d631a057c` (trees: delete duplicate/loser defines (IDN_RF_ROWS_SORTED, IDN_RF_TASK_ROWS256, IDN_RF_DEVICE_LOOP_K*, IDN_ET_BINNED_U16 + host column, T29_YETI no-op)).
- Recoverable at `2394e18e1` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_ET_IDN_BINNED_ANY_WIDTH_OFF.patch` (same patch as above).
- Files the patch restores: `extratrees/impl/decisiontree/batched_levelalgo/builder.mojo`
- grid_controls/trees-small.json removed: measured loser: slower on NVIDIA and AMD (combined 1.044x), RMSE worse (forest-et-decision 2026-10-05); code and host column deleted
- EXPERIMENTS.md:1566 (IDENTICAL tree switch dedupe and losers (lane/trees-small, 2): `MOJOLEARN_IDN_ET_BINNED_U16` on ExtraTrees regression, Istella / Year, lane/trees-small @ 2394e18e1, A/B forest-et-decision 2026-10-05, ratio NVIDIA 0.969901 / 1.167335, AMD 1.033972 / 1.016499 (combined 1.044443, slower) ms, **DROP, code deleted with its host column (`HostBins`, host_binned.mojo, `node_feature_score_host_binned`, `et_identical_bins_wanted`)**: slower on both vendors and RMSE worse (Istella .5645249225 -> .5649949313, Year 9.3233870097 -> 9.3234698986). The FAST Apple binned search is kept. Evidence: experiments/identical_speed/results/20261005/forest-et-decision/board.json

### MOJOLEARN_IDN_GBDT_FRONTIER_RESIDENT

- Verdict: DROPPED. Deleted 2026-10-07 by `359df2d05` (trees: delete T16/I17 resident frontier (lost on NV+AMD) and duplicate T20; T17 batch and T21 streams become int sweeps).
- Recoverable at `bad9c74ce` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_GBDT_FRONTIER_RESIDENT.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-359df2d05.patch`.
- Files the patch restores: `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo`
- EXPERIMENTS.md:1484 (IDENTICAL tree switch cleanup (lane/trees-cleanup, 2026-10-0): `MOJOLEARN_IDN_GBDT_FRONTIER_RESIDENT` on gbdt-lossguide, 10000x17, 10001x18, 32769x9, 10 trees, main @ 5b467815b, A/B I17 overnight-ab-20261006, AMD candidate/base 1.041, 1.292, 1.083; NVIDIA 1.114, 1.111, 1.116 ms, **DROP, code deleted**: slower on both vendors

### MOJOLEARN_IDN_RF_DEVICE_LOOP_K1

- Verdict: noise. Deleted 2026-10-07 by `d631a057c` (trees: delete duplicate/loser defines (IDN_RF_ROWS_SORTED, IDN_RF_TASK_ROWS256, IDN_RF_DEVICE_LOOP_K*, IDN_ET_BINNED_U16 + host column, T29_YETI no-op)).
- Recoverable at `2394e18e1` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_RF_DEVICE_LOOP_K1.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-d631a057c.patch`.
- Files the patch restores: `ensemble/decisiontree/batched_levelalgo/builder.mojo`
- grid_controls/trees-small.json removed: no gain (forest-final-decisions 2026-10-05); sweep is MOJOLEARN_TREES_T11_LEVELS; deleted
- EXPERIMENTS.md:1565 (IDENTICAL tree switch dedupe and losers (lane/trees-small, 2): `MOJOLEARN_IDN_RF_DEVICE_LOOP_K1` on rf device level loop, Taxi + Istella, lane/trees-small @ 2394e18e1, A/B forest-final-decisions 2026-10-05, combined ratio K1 1.002993, K2 1.002050, K8 1.002461 ms, **DROP, defines deleted (no gain + duplicate)**: `MOJOLEARN_TREES_T11_LEVELS=1\

### MOJOLEARN_IDN_RF_DEVICE_LOOP_K2

- Verdict: noise. Deleted 2026-10-07 by `d631a057c` (trees: delete duplicate/loser defines (IDN_RF_ROWS_SORTED, IDN_RF_TASK_ROWS256, IDN_RF_DEVICE_LOOP_K*, IDN_ET_BINNED_U16 + host column, T29_YETI no-op)).
- Recoverable at `2394e18e1` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_RF_DEVICE_LOOP_K2.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-d631a057c.patch`.
- Files the patch restores: `ensemble/decisiontree/batched_levelalgo/builder.mojo`
- grid_controls/trees-small.json removed: no gain (forest-final-decisions 2026-10-05); sweep is MOJOLEARN_TREES_T11_LEVELS; deleted

### MOJOLEARN_IDN_RF_DEVICE_LOOP_K8

- Verdict: noise. Deleted 2026-10-07 by `d631a057c` (trees: delete duplicate/loser defines (IDN_RF_ROWS_SORTED, IDN_RF_TASK_ROWS256, IDN_RF_DEVICE_LOOP_K*, IDN_ET_BINNED_U16 + host column, T29_YETI no-op)).
- Recoverable at `2394e18e1` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_RF_DEVICE_LOOP_K2.patch` (same patch as above).
- Files the patch restores: `ensemble/decisiontree/batched_levelalgo/builder.mojo`
- grid_controls/trees-small.json removed: no gain (forest-final-decisions 2026-10-05); sweep is MOJOLEARN_TREES_T11_LEVELS; deleted

### MOJOLEARN_IDN_RF_STREAM_REPLICAS

- Verdict: DROPPED. Deleted 2026-10-07 by `e5ad4d891` (trees: delete T01/N07 streamed histogram replicas (lost on NV+AMD) and its experiment code).
- Recoverable at `359df2d05` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_RF_STREAM_REPLICAS.patch` (applies cleanly to main at the time of writing); first apply `experiments/removed/_shared-e5ad4d891.patch`.
- Files the patch restores: `ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`, `experiments/performance_ideas/N07/production_check.mojo`
- EXPERIMENTS.md:1483 (IDENTICAL tree switch cleanup (lane/trees-cleanup, 2026-10-0): `MOJOLEARN_IDN_RF_STREAM_REPLICAS` on rf fit, generated 100000x32 and 131071x17, main @ cbcc8dcd3303 (source of the measurement), A/B N07 2026-10-06 (measurements/20261006/index.json), AMD 335.85 -> 347.28 (1.034); NVIDIA L40S 124.76 -> 161.31 (1.293); NVIDIA 131071x17 107.52 -> 131.88 (1.227) ms, **DROP, code deleted**: slower on both vendors; T01_ROWS (task rows) is covered by T02's cost rule

### MOJOLEARN_IDN_RF_TASK_ROWS256

- Verdict: noise. Deleted 2026-10-07 by `d631a057c` (trees: delete duplicate/loser defines (IDN_RF_ROWS_SORTED, IDN_RF_TASK_ROWS256, IDN_RF_DEVICE_LOOP_K*, IDN_ET_BINNED_U16 + host column, T29_YETI no-op)).
- Recoverable at `2394e18e1` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_RF_TASK_ROWS256.patch` (applies cleanly to main at the time of writing); first apply `experiments/removed/_shared-d631a057c.patch`.
- Files the patch restores: `ensemble/decisiontree/batched_levelalgo/builder.mojo`, `experiments/performance_ideas/A07/production_check.mojo`
- grid_controls/trees-small.json removed: measured neutral on NVIDIA and AMD (A07, overnight-ab-20261006); task rows are MOJOLEARN_TREES_T02; define deleted
- EXPERIMENTS.md:1564 (IDENTICAL tree switch dedupe and losers (lane/trees-small, 2): `MOJOLEARN_IDN_RF_TASK_ROWS256` on rf histogram tasks; generated 100000x32, 100001x33, 65537x17, lane/trees-small @ 2394e18e1, A/B A07 (overnight-ab-20261006), L40S 124.910/126.939/92.868 -> 123.567/125.678/92.118; MI325X ratio 0.993/0.995/0.998 ms, **DROP, define deleted (neutral + duplicate)**: neutral on both vendors (<1.1%); the task-row knob is `MOJOLEARN_TREES_T02`'s cost rule (`histogram_task_rows`). Evidence: overnight-ab-20261006 amd/live/repair-summary.json, nvidia/default-repair-normalized-measurements.json

### MOJOLEARN_IF_QUERY_RAW

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-trees-io @ df6a77c21`.
- EXPERIMENTS.md:135 (Trees (101)): `IF_QUERY_RAW` on iforest, lane/apple-fast-trees-io @ df6a77c21, A/B trees-io-ifq-build, iforest taxi 328 -> 324; score istella 2,290 -> 2,289 ms, **DROPPED-noise**: A runs straddle; refusals ok

### MOJOLEARN_IF_SAMPLED_UPLOAD

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-trees-io @ df6a77c21`.
- EXPERIMENTS.md:136 (Trees (101)): `IF_SAMPLED_UPLOAD` on iforest / istella, lane/apple-fast-trees-io @ df6a77c21, A/B trees-io-if-istella, iforest istella 384 -> 385 ms, **DROPPED-noise**: same hash

### MOJOLEARN_ORDERED_FOLD_DERIVS

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `219ae194a` (gbdt ordered: remove dropped ORDERED_FOLD_DERIVS (DROPPED-slower; recover lane/apple-fast-ordered@5b8722353)).
- Recoverable at `dd83da010` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_ORDERED_FOLD_DERIVS.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gbdt/methods/ordered_boosting.mojo`
- EXPERIMENTS.md:137 (Trees (101)): `ORDERED_FOLD_DERIVS` on ordered / taxi, lane/apple-fast-ordered @ 5b8722353, A/B ord-fd-taxi, ordered taxi 282,245 -> 287,217 ms, **DROPPED-slower**: +1.8%; code removed from main 219ae194a; recover at lane/apple-fast-ordered@5b8722353

### MOJOLEARN_ORD_FOLD_BINS_ONE

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-sym-ordered @ 27b912397`.
- EXPERIMENTS.md:169 (Trees (101)): `ORD_FOLD_BINS_ONE` on ordered / taxi, lane/apple-fast-sym-ordered @ 27b912397, A/B sym-ordered-fbo-taxi, taxi -1.0% ms, **DROPPED-noise (standalone)**: define removed; the code stays as a piece of `ORD_ALL`

### MOJOLEARN_ORD_FOLD_INDEX

- Verdict: DROPPED-quality. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-sym-ordered @ 27b912397`.
- EXPERIMENTS.md:170 (Trees (101)): `ORD_FOLD_INDEX` on ordered / taxi, lane/apple-fast-sym-ordered @ 27b912397, A/B sym-ordered-fidx-taxi, taxi -5.9% ms, **DROPPED-quality (standalone)**: auc down on taxi; define removed; the code stays as a piece of `ORD_ALL` (wide data only)

### MOJOLEARN_ORD_STD_PARALLEL

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-sym-ordered @ 27b912397`.
- EXPERIMENTS.md:171 (Trees (101)): `ORD_STD_PARALLEL` on ordered / istella, lane/apple-fast-sym-ordered @ 27b912397, A/B sym-ordered-std-istella, istella -1.8% ms, **DROPPED-noise (standalone)**: define removed; the code stays as a piece of `ORD_ALL`

### MOJOLEARN_ORD_TREE_LEAN

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-sym-ordered @ 27b912397`.
- EXPERIMENTS.md:172 (Trees (101)): `ORD_TREE_LEAN` on ordered / istella, lane/apple-fast-sym-ordered @ 27b912397, A/B sym-ordered-lean-istella, istella ~-0.8% ms, **DROPPED-noise (standalone)**: define removed; the code stays as a piece of `ORD_ALL`

### MOJOLEARN_REORDER_FLAGS_SCAN_BLOCK

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-trees-scan @ 43430ca0f`.
- EXPERIMENTS.md:138 (Trees (101)): `REORDER_FLAGS_SCAN_BLOCK + SEG_SCAN_BLOCK` on depthwise / rf / taxi, lane/apple-fast-trees-scan @ 43430ca0f, A/B trees-scan-dw-taxi, depthwise taxi 12,631 -> 12,853 ms, **DROPPED-noise**: +1.8%, overlap

### MOJOLEARN_RF_FAST_BATCH16K

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `288809ef8` (rf: remove dropped RF_FAST_BATCH16K arm and the trees-apple3 one-shot A/B job scripts that request it (DROPPED-noise; recover lane/apple-fast-trees2@bfd1d7cc6)).
- Recoverable at `f9af6028e` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_RF_FAST_BATCH16K.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `bindings/_mojolearn_rf.mojo`
- EXPERIMENTS.md:139 (Trees (101)): `RF_NODESPLIT_ZERO_AFTER_READ + RF_FAST_BATCH16K` on rf / taxi, istella, lane/apple-fast-trees2 @ bfd1d7cc6, A/B aft-ab-rf1, taxi 11,502 -> 11,485; istella 14,324 -> 14,312 ms, **DROPPED-noise**: 0.1%; RF_FAST_BATCH16K: code removed from main 288809ef8; recover at lane/apple-fast-trees2@bfd1d7cc6; RF_NODESPLIT_ZERO_AFTER_READ: code removed from main 931143bf8; recover at lane/apple-fast-trees2@bfd1d7cc6

### MOJOLEARN_RF_NODESPLIT_ZERO_AFTER_READ

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `931143bf8` (rf: remove dropped RF_NODESPLIT_ZERO_AFTER_READ (DROPPED-noise; recover lane/apple-fast-trees2@bfd1d7cc6)).
- Recoverable at `288809ef8` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_RF_NODESPLIT_ZERO_AFTER_READ.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`
- EXPERIMENTS.md:139 (Trees (101)): `RF_NODESPLIT_ZERO_AFTER_READ + RF_FAST_BATCH16K` on rf / taxi, istella, lane/apple-fast-trees2 @ bfd1d7cc6, A/B aft-ab-rf1, taxi 11,502 -> 11,485; istella 14,324 -> 14,312 ms, **DROPPED-noise**: 0.1%; RF_FAST_BATCH16K: code removed from main 288809ef8; recover at lane/apple-fast-trees2@bfd1d7cc6; RF_NODESPLIT_ZERO_AFTER_READ: code removed from main 931143bf8; recover at lane/apple-fast-trees2@bfd1d7cc6

### MOJOLEARN_RF_SMALL_NODE_1024

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `baa477878` (rf: remove dropped RF_SMALL_NODE_1024 arm (DROPPED-noise; recover lane/apple-fast-trees2@50dfdcca0)).
- Recoverable at `931143bf8` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_RF_SMALL_NODE_1024.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`
- EXPERIMENTS.md:140 (Trees (101)): `RF_SMALL_NODE_1024` on rf / taxi, istella, lane/apple-fast-trees2 @ 50dfdcca0, A/B aft-ab-rfsn, taxi 11,491 -> 11,482; istella 14,320 -> 14,329 ms, **DROPPED-noise**: ; code removed from main baa477878; recover at lane/apple-fast-trees2@50dfdcca0

### MOJOLEARN_SEG_SCAN_BLOCK

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-trees-scan @ 43430ca0f`.
- EXPERIMENTS.md:138 (Trees (101)): `REORDER_FLAGS_SCAN_BLOCK + SEG_SCAN_BLOCK` on depthwise / rf / taxi, lane/apple-fast-trees-scan @ 43430ca0f, A/B trees-scan-dw-taxi, depthwise taxi 12,631 -> 12,853 ms, **DROPPED-noise**: +1.8%, overlap
- EXPERIMENTS.md:141 (Trees (101)): `SEG_SCAN_BLOCK` on depthwise / rf / istella, lane/apple-fast-trees-scan @ 43430ca0f, A/B trees-scan-rf-istella, rf istella 13,704 -> 10,146 ms, **DROPPED-semantics**: stale base: main already had the gain via SEG_SUMS_BLOCK_SCAN (69f7a41fd); duplicate

### MOJOLEARN_SYM_CTR_PERM_BATCH

- Verdict: DROPPED-noise. Deleted 2026-10-04 by `f4ac2ee28` (Revert "SYM_CTR_PERM_BATCH: the symmetric estimation loop's four permutations batched behind one drain per round").
- Recoverable at `d5d9d712c` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_SYM_CTR_PERM_BATCH.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gbdt/ctrs/fast_prep.mojo`, `gbdt/methods/doc_parallel_boosting.mojo`
- EXPERIMENTS.md:181 (Trees (101)): `SYM_CTR_PERM_BATCH` on categorical / taxicat, lane/apple-fast-sym-ctr @ 39c3c9daf, A/B sym-ctr-perm-batch-taxicat, - ms, **DROPPED-noise**: reconciled 2026-10-05: sym-ctr-perm-batch-taxicat-x 27161 -> 27196 (+0.1%), Oct 4 manager table (DROP-speed); not ported. Was OPEN: A/B queued (lane/apple-fast-batch prebuilt arms)
- EXPERIMENTS.md:205 (Trees (101)): `SYM_CTR_PERM_BATCH` on categorical / taxicat, lane/apple-fast-sym-ctr @ 39c3c9daf, A/B sym-ctr-perm-batch-taxicat, - ms, **DROPPED-speed (+0.1%)**: sym-ctr-perm-batch-taxicat-x 27,161 -> 27,196 ms (Oct 4 table); NOT ported to lane/apple-fast-rec-sym (reverted there), code stays at the source sha
- EXPERIMENTS.md:734 (Oct 4 manager takeover): `SYM_CTR_PERM_BATCH` on lane/apple-fast-batch @ 3150d75c1, sym-ctr-perm-batch-taxicat-x, A/B 27161 → 27196 (+0.1%), AUC .630994 → .631048, logloss .528561 → .528534 ms, **DROP-speed: no gain on old base**: 

### MOJOLEARN_SYM_DEVICE_LEAVES

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-trees-symmetric @ ce517b4b3`.
- EXPERIMENTS.md:142 (Trees (101)): `SYM_DEVICE_LEAVES` on symmetric / istella, lane/apple-fast-trees-symmetric @ ce517b4b3, A/B tsym-leaves-istella, symmetric istella 14,671 -> 14,643 ms, **DROPPED-noise**: -0.2%, overlap
- EXPERIMENTS.md:143 (Trees (101)): `SYM_DEVICE_LEAVES + SYM_DEVICE_LEVEL + SYM_DEVICE_PARTITION + SYM_NO_TAIL_DRAIN` on symmetric / istella, lane/apple-fast-trees-symmetric @ ce517b4b3, A/B tsym-all-1000-istella, symmetric-1000 istella 32,932 -> 33,146 ms, **DROPPED-noise**: +0.6%; no switch in this lane wins

### MOJOLEARN_SYM_DEVICE_LEVEL

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-trees-symmetric @ ce517b4b3`.
- EXPERIMENTS.md:143 (Trees (101)): `SYM_DEVICE_LEAVES + SYM_DEVICE_LEVEL + SYM_DEVICE_PARTITION + SYM_NO_TAIL_DRAIN` on symmetric / istella, lane/apple-fast-trees-symmetric @ ce517b4b3, A/B tsym-all-1000-istella, symmetric-1000 istella 32,932 -> 33,146 ms, **DROPPED-noise**: +0.6%; no switch in this lane wins
- EXPERIMENTS.md:144 (Trees (101)): `SYM_DEVICE_LEVEL` on symmetric / istella, lane/apple-fast-trees-symmetric @ ce517b4b3, A/B tsym-level-istella, symmetric istella 16,926 -> 16,852 ms, **DROPPED-noise**: -0.4%; auc equal

### MOJOLEARN_SYM_DEVICE_PARTITION

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-trees-symmetric @ ce517b4b3`.
- EXPERIMENTS.md:143 (Trees (101)): `SYM_DEVICE_LEAVES + SYM_DEVICE_LEVEL + SYM_DEVICE_PARTITION + SYM_NO_TAIL_DRAIN` on symmetric / istella, lane/apple-fast-trees-symmetric @ ce517b4b3, A/B tsym-all-1000-istella, symmetric-1000 istella 32,932 -> 33,146 ms, **DROPPED-noise**: +0.6%; no switch in this lane wins
- EXPERIMENTS.md:145 (Trees (101)): `SYM_DEVICE_PARTITION` on symmetric / istella, lane/apple-fast-trees-symmetric @ ce517b4b3, A/B tsym-part-istella, symmetric istella 16,869 -> 17,081 ms, **DROPPED-slower**: +1.3%

### MOJOLEARN_SYM_HIST_FAST

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `6f5ace7fa` (gbdt sym hist / kernel_matrix: remove dropped SYM_HIST_FAST 512 block on Apple FAST (DROPPED-noise; recover lane/apple-fast-trees-yeti@65f551e39)).
- Recoverable at `baa477878` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_SYM_HIST_FAST.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `checks/kernel_matrix.mojo`, `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_8bit.mojo`
- EXPERIMENTS.md:146 (Trees (101)): `SYM_HIST_FAST` on yetirank / symmetric / istella, taxi, lane/apple-fast-trees-yeti @ 65f551e39, A/B yeti-symhist, yeti-symhist-sym-taxi, yeti-symhist-sym-istella, yeti-symhist-ordered-taxi, symmetric istella 14,710 -> 14,685; ordered taxi 66,705 -> 67,150 ms, **DROPPED-noise**: -0.2% / +0.7%; code removed from main 6f5ace7fa; recover at lane/apple-fast-trees-yeti@65f551e39
- EXPERIMENTS.md:147 (Trees (101)): `SYM_HIST_FAST + YETI_SEARCH_TASK16K` on yetirank / symmetric, lane/apple-fast-trees-yeti @ 65f551e39, A/B yeti-both, - ms, **DROPPED-noise**: SYM_HIST_FAST part dropped (see above); code removed from main 6f5ace7fa; recover at lane/apple-fast-trees-yeti@65f551e39
- EXPERIMENTS.md:148 (Trees (101)): `SYM_HIST_FAST + YETI_SYM_HIST_UNROLL8` on yetirank, lane/apple-fast-yetirank @ c7b35fd7c, A/B yeti-h8unroll, yetirank 5,315 -> 5,373 ms, **DROPPED-slower**: +1.1%; SYM_HIST_FAST: code removed from main 6f5ace7fa; recover at lane/apple-fast-yetirank@c7b35fd7c; YETI_SYM_HIST_UNROLL8: code removed from main 4f634c5d0; recover at lane/apple-fast-yetirank@c7b35fd7c

### MOJOLEARN_SYM_NO_TAIL_DRAIN

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-trees-symmetric @ ce517b4b3`.
- EXPERIMENTS.md:143 (Trees (101)): `SYM_DEVICE_LEAVES + SYM_DEVICE_LEVEL + SYM_DEVICE_PARTITION + SYM_NO_TAIL_DRAIN` on symmetric / istella, lane/apple-fast-trees-symmetric @ ce517b4b3, A/B tsym-all-1000-istella, symmetric-1000 istella 32,932 -> 33,146 ms, **DROPPED-noise**: +0.6%; no switch in this lane wins
- EXPERIMENTS.md:149 (Trees (101)): `SYM_NO_TAIL_DRAIN` on symmetric / istella, lane/apple-fast-trees-symmetric @ ce517b4b3, A/B tsym-notail-istella, symmetric istella 16,998 -> 17,106 ms, **DROPPED-noise**: +0.6%

### MOJOLEARN_TREES_C47_GBDT

- Verdict: DROPPED. Deleted 2026-10-07 by `ec793e91c` (gbdt: delete T27 alias and C47_GBDT width cap; guards refuse both; EXPERIMENTS rows).
- Recoverable at `eb8efb83a` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_C47_GBDT.patch` (applies cleanly to main at the time of writing); first apply `experiments/removed/_shared-ec793e91c.patch`.
- Files the patch restores: `gbdt/trees_identical_switches.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:171): removed: MOJOLEARN_TREES_C47_GBDT (width cap subsumed by MOJOLEARN_TREES_T17_BATCH)
- grid_controls/trees-cleanup.json removed (control C47_GBDT): DELETED by lane/grid-prune 2026-10-07: width cap subsumed by T17_BATCH; guards refuse it.
- EXPERIMENTS.md:1595 (IDENTICAL grid prune: losers and dead arms deleted (lane/gri): `MOJOLEARN_TREES_C47_GBDT` on gbdt-lossguide exact batch, main @ ab554bb4a, A/B none, n/a ms, **DROP_REDUNDANT, define deleted**: only an 8 MiB cap on T17_BATCH's width: about 258 on taxi (a no-op) and about 18 on istella (narrower than the incumbent 32); a memory guard, not a speed arm. `MOJOLEARN_TREES_T17_BATCH` owns the width (greedy_search_helper_depthwise.mojo:3332-3340)

### MOJOLEARN_TREES_HIST_MULTISTAT

- Verdict: slower. Deleted 2026-10-08 by `9c6dccee7` (grid-act-2: delete hist_multistat (both arms; grid ge123e6f9 gbdt-multiclass 1.15x/1.13x combined slower, quality same); refuse the define; tombstones).
- Recoverable at `22187de78` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_HIST_MULTISTAT.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo`, `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_8bit_wide.mojo`, `gbdt/trees_hist_switches.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:199): removed: MOJOLEARN_TREES_HIST_MULTISTAT (=4|8) retired 2026-10-08: slower, gbdt-multiclass =4 NV 1.04x / AMD 1.41x istella, 1.07x / 1.11x taxi; =8 0.99x / 1.31x istella, 1.11x / 1.14x taxi; quality SAME (grid ge123e6f9); see EXPERIMENTS.md
- grid_controls/trees-hist-ideas.json removed (control hist_multistat): DELETED 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 loser, both arms. arm/off ms ratio NV/AMD on gbdt-multiclass: =4 istella 1.038/1.410, taxi 1.066/1.110 (1.147x combined); =8 istella 0.994/1.311, taxi 1.106/1.136 (1.131x combined); accuracy and mlogloss SAME. The MultiClass >128-bin blocks take launch_one_byte[8] again; the wide kernel stays for HIST_SYM_FEATURE_PARALLEL. Code recoverable at main 42d1e42c6; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1709 (IDENTICAL grid ge123e6f9: two promotions and eight loser arm): `MOJOLEARN_TREES_HIST_MULTISTAT=4` on trees:gbdt-multiclass / istella, taxi, main @ 42d1e42c6 (deleted on lane/grid-act-2), A/B ge123e6f9, =4: istella NV 14689.4 -> 15241.4 (1.038), AMD 10372.6 -> 14626.0 (1.410); taxi NV 7250.6 -> 7727.1 (1.066), AMD 6754.2 -> 7496.3 (1.110); 1.147x combined. =8: istella NV 14689.4 -> 14603.1 (0.994), AMD 10372.6 -> 13596.8 (1.311); taxi NV 7250.6 -> 8021.5 (1.106), AMD 6754.2 -> 7673.5 (1.136); 1.131x combined ms, **DROP (slower), code deleted**: one cindex walk per 4 or 8 MultiClass stat planes (launch_hist2_8bit_wide[NS, 1]) instead of one per plane: fewer reads but larger shared tables, slower on AMD at both widths and on NVIDIA taxi; accuracy and mlogloss SAME. Deleted the branch in greedy_search_helper.mojo and the define (gbdt/trees_hist_switches.mojo); the wide kernel stays for HIST_SYM_FEATURE_PARALLEL.

### MOJOLEARN_TREES_T01

- Verdict: slower. Deleted 2026-10-07 by `e5ad4d891` (trees: delete T01/N07 streamed histogram replicas (lost on NV+AMD) and its experiment code).
- Recoverable at `359df2d05` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_T01.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-e5ad4d891.patch`.
- Files the patch restores: `ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo`, `ensemble/tree_identical_ideas.mojo`
- grid_controls/trees-cleanup.json removed: N07 streamed replicas lost on NVIDIA (1.293, 1.227) and AMD (1.034); code deleted with MOJOLEARN_IDN_RF_STREAM_REPLICAS
- EXPERIMENTS.md:1483 (IDENTICAL tree switch cleanup (lane/trees-cleanup, 2026-10-0): `MOJOLEARN_TREES_T01` on rf fit, generated 100000x32 and 131071x17, main @ cbcc8dcd3303 (source of the measurement), A/B N07 2026-10-06 (measurements/20261006/index.json), AMD 335.85 -> 347.28 (1.034); NVIDIA L40S 124.76 -> 161.31 (1.293); NVIDIA 131071x17 107.52 -> 131.88 (1.227) ms, **DROP, code deleted**: slower on both vendors; T01_ROWS (task rows) is covered by T02's cost rule

### MOJOLEARN_TREES_T01_REPLICAS

- Verdict: removed. Deleted 2026-10-07 by `e5ad4d891` (trees: delete T01/N07 streamed histogram replicas (lost on NV+AMD) and its experiment code).
- Recoverable at `359df2d05` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_T01_REPLICAS.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-e5ad4d891.patch`.
- Files the patch restores: `ensemble/tree_identical_ideas.mojo`
- grid_controls/trees-cleanup.json removed: sub-parameter of deleted T01

### MOJOLEARN_TREES_T01_ROWS

- Verdict: removed. Deleted 2026-10-07 by `e5ad4d891` (trees: delete T01/N07 streamed histogram replicas (lost on NV+AMD) and its experiment code).
- Recoverable at `359df2d05` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_T01_REPLICAS.patch` (same patch as above).
- Files the patch restores: `ensemble/tree_identical_ideas.mojo`
- grid_controls/trees-cleanup.json removed: sub-parameter of deleted T01; T02 cost rule covers task rows

### MOJOLEARN_TREES_T09

- Verdict: dead code. Deleted 2026-10-07 by `eb8efb83a` (trees: delete ET T09 bootstrap sort (unreachable, host sync in fit); guard refuses MOJOLEARN_TREES_T09; EXPERIMENTS row).
- Recoverable at `225398f14` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_T09.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `ensemble/tree_identical_ideas.mojo`, `extratrees/impl/decisiontree/batched_levelalgo/builder.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:169): removed: MOJOLEARN_TREES_T09 (ExtraTrees bootstrap sort: unreachable at bootstrap=False, host sync in fit)
- grid_controls/trees-cleanup.json removed (control T09_ET): DELETED by lane/grid-prune 2026-10-07: unreachable on the board (et bootstrap=False) and a host sync inside the fit; guards refuse MOJOLEARN_TREES_T09. Recoverable at main ab554bb4a.
- EXPERIMENTS.md:1593 (IDENTICAL grid prune: losers and dead arms deleted (lane/gri): `MOJOLEARN_TREES_T09` on trees:et fit (ExtraTrees bootstrap-locality sort), main @ ab554bb4a, A/B none (unreachable), n/a ms, **ALWAYS_OFF_DELETE, define deleted**: the board et lane runs bootstrap=False (`tools/speed_gbdt_arm.py:2093`), so `fill_row_slots` returns before the T09 block (extratrees builder.mojo:2264-2300); reached, it put a host `ctx.synchronize()` inside the GPU fit (forbidden host step). RF's sorted bootstrap stays as `MOJOLEARN_TREES_RF_SAMPLE=1`

### MOJOLEARN_TREES_T16

- Verdict: slower. Deleted 2026-10-07 by `359df2d05` (trees: delete T16/I17 resident frontier (lost on NV+AMD) and duplicate T20; T17 batch and T21 streams become int sweeps).
- Recoverable at `bad9c74ce` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_T16.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-359df2d05.patch`.
- Files the patch restores: `gbdt/trees_identical_switches.mojo`
- grid_controls/trees-cleanup.json removed: I17 resident Lossguide frontier lost on AMD (1.041/1.292/1.083) and NVIDIA (1.114/1.111/1.116); code deleted with MOJOLEARN_IDN_GBDT_FRONTIER_RESIDENT
- EXPERIMENTS.md:1484 (IDENTICAL tree switch cleanup (lane/trees-cleanup, 2026-10-0): `MOJOLEARN_TREES_T16` on gbdt-lossguide, 10000x17, 10001x18, 32769x9, 10 trees, main @ 5b467815b, A/B I17 overnight-ab-20261006, AMD candidate/base 1.041, 1.292, 1.083; NVIDIA 1.114, 1.111, 1.116 ms, **DROP, code deleted**: slower on both vendors

### MOJOLEARN_TREES_T19

- Verdict: quality loss. Deleted 2026-10-08 by `e2c65c026` (grid-losers-1: delete T19 (grid ge123e6f9: gbdt-depthwise 0.47x/0.69x taxi but istella AUC -0.43%, logloss +20.7%); fused_scan_update_kernel back to its pre-T19).
- Recoverable at `9829cdb31` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_T19.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo`, `gbdt/methods/greedy_subsets_searcher/kernel/split_chain_fused.mojo`, `gbdt/trees_identical_switches.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:185): removed: MOJOLEARN_TREES_T19 retired 2026-10-08: quality loss, istella AUC -0.43% and logloss +20.7% despite NV 0.47x / AMD 0.69x on depthwise taxi (grid ge123e6f9); see EXPERIMENTS.md
- grid_controls/trees-cleanup.json removed (control T19): DELETED 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 quality loss, a loser despite the speed. on/off ms ratio NV/AMD: gbdt-depthwise taxi 0.470/0.692 (1964 vs 4180 ms NV, 2708 vs 3912 ms AMD), istella 0.826/0.813 (7495 vs 9070 ms NV, 6123 vs 7530 ms AMD); quality: istella AUC 0.97908 vs 0.98330 (-0.43%), logloss 0.18838 vs 0.15607 (+20.7%), taxi AUC 0.63023 vs 0.63221 (-0.31%), logloss +0.16%; all WORSE. gbdt-lossguide unaffected (T19 does not reach Lossguide: lossguide AUC/logloss SAME, timing noise NV 1.100/0.910, AMD 1.041/1.074). Revisit only with a design that keeps the IDENTICAL partition stats (the fused chain ran with PROPAGATE_STATS off). T19_DEFER is independent and kept. Code recoverable at main ad7ed2370; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1610 (IDENTICAL grid ge123e6f9 losers deleted (lane/grid-losers-1,): `MOJOLEARN_TREES_T19` on trees:gbdt-depthwise / taxi, istella, main @ ad7ed2370 (deleted on lane/grid-losers-1), A/B ge123e6f9, depthwise taxi NV 4180 -> 1964 (0.470), AMD 3912 -> 2708 (0.692); istella NV 9070 -> 7495 (0.826), AMD 7530 -> 6123 (0.813) ms, **DROP (quality loss), code deleted**: fast but worse models: istella AUC 0.98330 -> 0.97908 (-0.43%), logloss 0.15607 -> 0.18838 (+20.7%); taxi AUC 0.63221 -> 0.63023 (-0.31%), logloss +0.16%. T19 forced the row-index schedule and took the fused partition chain under IDENTICAL with the parent-stat propagation skipped (`fused_scan_update_kernel[..., PROPAGATE_STATS=False]`). Deleted T19, its `DW_FUSED_CHAIN` / `use_ridx` hooks and the PROPAGATE_STATS parameter (greedy_search_helper_depthwise.mojo, kernel/split_chain_fused.mojo, back to its pre-T19 body). gbdt-lossguide is not reached (AUC / logloss SAME, timing noise). Worth a revisit only with a design that keeps the IDENTICAL partition stats exact. `MOJOLEARN_TREES_T19_DEFER` is independent (DEFER_HIST_COPY_1903) and stays

### MOJOLEARN_TREES_T20

- Verdict: DROPPED. Deleted 2026-10-07 by `359df2d05` (trees: delete T16/I17 resident frontier (lost on NV+AMD) and duplicate T20; T17 batch and T21 streams become int sweeps).
- Recoverable at `bad9c74ce` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_T20.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-359df2d05.patch`.
- Files the patch restores: `gbdt/trees_identical_switches.mojo`
- grid_controls/trees-cleanup.json removed: same DEFER_HIST_COPY_1903 constant as T19_DEFER
- EXPERIMENTS.md:1485 (IDENTICAL tree switch cleanup (lane/trees-cleanup, 2026-10-0): `MOJOLEARN_TREES_T20` on gbdt depthwise / lossguide, lane/trees-cleanup, A/B none, n/a ms, **DROP, define deleted**: set the same `DEFER_HIST_COPY_1903` constant as `T19_DEFER`

### MOJOLEARN_TREES_T27

- Verdict: dead code. Deleted 2026-10-07 by `ec793e91c` (gbdt: delete T27 alias and C47_GBDT width cap; guards refuse both; EXPERIMENTS rows).
- Recoverable at `eb8efb83a` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_T27.patch` (applies cleanly to main at the time of writing); first apply `experiments/removed/_shared-ec793e91c.patch`.
- Files the patch restores: `gbdt/methods/leaves_estimation/pointwise_oracle.mojo`, `gbdt/trees_identical_switches.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:170): removed: MOJOLEARN_TREES_T27 (alias of MOJOLEARN_2030_FUSED_EST_MOVE; inert on board walks)
- grid_controls/trees-cleanup.json removed (control T27): DELETED by lane/grid-prune 2026-10-07: alias of MOJOLEARN_2030_FUSED_EST_MOVE, inert on board walks; guards refuse it.
- EXPERIMENTS.md:1594 (IDENTICAL grid prune: losers and dead arms deleted (lane/gri): `MOJOLEARN_TREES_T27` on gbdt leaf estimation (every gbdt-* lane), main @ ab554bb4a, A/B none (inert), n/a ms, **DROP_REDUNDANT, alias deleted**: alias of `MOJOLEARN_2030_FUSED_EST_MOVE` (which stays), and `_NO_FUSED_EST_MOVE` silently won over it; it acts only inside `move_to`, while board walks call `_launch_shift_abmv` directly (pointwise_oracle.mojo:195-207,523-531; device_walker.mojo:566)

### MOJOLEARN_TREES_T29_VERSIONED

- Verdict: slower. Deleted 2026-10-08 by `bd14ae97f` (grid-act-2: delete T29 versioned arm (MOJOLEARN_TREES_T29_VERSIONED; grid ge123e6f9 pairlogit istella 3.43x/3.76x slower); T29 on stays; refuse the define; tomb).
- Recoverable at `9c6dccee7` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_T29_VERSIONED.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `gbdt/host/gbdt_oracle_pair.mojo`, `gbdt/targets/kernel/pair_logit_group.mojo`, `gbdt/targets/kernel/tree_t29_pair.mojo`, `gbdt/targets/tree_t29_units.mojo`, `gbdt/trees_identical_switches.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:200): removed: MOJOLEARN_TREES_T29_VERSIONED (T29 arm 'versioned') retired 2026-10-08: slower, gbdt-rank-pairlogit istella NV 3.43x / AMD 3.76x (grid ge123e6f9); MOJOLEARN_TREES_T29 alone stays; see EXPERIMENTS.md
- grid_controls/trees-cleanup.json removed (control T29): DELETED 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 loser. versioned/off ms ratio NV/AMD: gbdt-rank-pairlogit istella 3.429/3.757 (2985.4 -> 10235.7 NV, 1655.7 -> 6219.9 AMD); 3.59x combined SLOWER. Deleted gbdt/targets/kernel/tree_t29_pair.mojo, gbdt/targets/tree_t29_units.mojo, the versioned launch in pair_logit_group.mojo and the host _group_values_t29 path in gbdt_oracle_pair.mojo; T29 'on' (unmeasured) stays. Code recoverable at main 42d1e42c6; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1710 (IDENTICAL grid ge123e6f9: two promotions and eight loser arm): `MOJOLEARN_TREES_T29_VERSIONED` on trees:gbdt-rank-pairlogit / istella, main @ 42d1e42c6 (deleted on lane/grid-act-2), A/B ge123e6f9, istella NV 2985.4 -> 10235.7 (3.429), AMD 1655.7 -> 6219.9 (3.757); 3.59x combined SLOWER ms, **DROP (slower), code deleted**: the generated V1 PairLogit objective (one 256-lane block per group, per-row partner chunks, lane carries) with its host oracle: a new bit profile and 3.4-3.8x slower on both vendors. Deleted gbdt/targets/kernel/tree_t29_pair.mojo, gbdt/targets/tree_t29_units.mojo, the versioned launch in gbdt/targets/kernel/pair_logit_group.mojo and `_group_values_t29` + the versioned magnitudes in gbdt/host/gbdt_oracle_pair.mojo. T29 `on` (scheduling only, unmeasured) stays.

### MOJOLEARN_TREES_T29_YETI

- Verdict: dead code. Deleted 2026-10-07 by `d631a057c` (trees: delete duplicate/loser defines (IDN_RF_ROWS_SORTED, IDN_RF_TASK_ROWS256, IDN_RF_DEVICE_LOOP_K*, IDN_ET_BINNED_U16 + host column, T29_YETI no-op)).
- Recoverable at `2394e18e1` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_T29_YETI.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-d631a057c.patch`.
- Files the patch restores: `gbdt/trees_identical_switches.mojo`
- grid_controls/trees-cleanup.json removed: no-op on NVIDIA and AMD (yeti_block_parallel_for already True under IDENTICAL); deleted by lane trees-small 2026-10-07
- EXPERIMENTS.md:1567 (IDENTICAL tree switch dedupe and losers (lane/trees-small, 2): `MOJOLEARN_TREES_T29_YETI` on gbdt rank-yetirank, lane/trees-small @ 2394e18e1, A/B none, n/a ms, **DROP, define deleted (no-op)**: `yeti_block_parallel_for` already selects the block kernel on NVIDIA and AMD under IDENTICAL, so the switch changed nothing; the negative arm stays `MOJOLEARN_IDN_GBDT_YETI_BLOCK_AMD_OFF`

### MOJOLEARN_TREES_T31_PACKED_A

- Verdict: dead code. Deleted 2026-10-07 by `bad9c74ce` (trees: drop no-op T31_PACKED_A and duplicate T31_PACKED_B (packed is the default)).
- Recoverable at `8be4d20d4` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TREES_T31_PACKED_A.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `core/forest_experiments.mojo`, `core/forest_inference.mojo`
- grid_controls/trees-cleanup.json removed: no-op: packed nodes are the default
- EXPERIMENTS.md:1487 (IDENTICAL tree switch cleanup (lane/trees-cleanup, 2026-10-0): `MOJOLEARN_TREES_T31_PACKED_A` on forest predict, lane/trees-cleanup, A/B none, n/a ms, **DROP, defines deleted**: A was a no-op (packed nodes are the default); B duplicated `MOJOLEARN_FOREST_SEPARATE_NODES`

### MOJOLEARN_YETI_SYM_HIST_UNROLL8

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `4f634c5d0` (gbdt sym hist: remove dropped YETI_SYM_HIST_UNROLL8 (DROPPED-slower; recover lane/apple-fast-yetirank@c7b35fd7c)).
- Recoverable at `6f5ace7fa` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_YETI_SYM_HIST_UNROLL8.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_8bit.mojo`
- EXPERIMENTS.md:148 (Trees (101)): `SYM_HIST_FAST + YETI_SYM_HIST_UNROLL8` on yetirank, lane/apple-fast-yetirank @ c7b35fd7c, A/B yeti-h8unroll, yetirank 5,315 -> 5,373 ms, **DROPPED-slower**: +1.1%; SYM_HIST_FAST: code removed from main 6f5ace7fa; recover at lane/apple-fast-yetirank@c7b35fd7c; YETI_SYM_HIST_UNROLL8: code removed from main 4f634c5d0; recover at lane/apple-fast-yetirank@c7b35fd7c


## Linear

### MOJOLEARN_ARD_EQ_ONEPASS

- Verdict: DROPPED-quality. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-general-speed @ d42318df5, 7ec8ec5ff`.
- EXPERIMENTS.md:1328 (lane/apple-fast-general-speed verdicts (2026-10-05, lane/app): `ARD_EQ_ONEPASS` on ard / board, lane/apple-fast-general-speed @ d42318df5, 7ec8ec5ff, A/B rab11-ardonepass, 1349.6 -> 34.1 ms, **DROPPED-quality**: r2 0.327 -> -0.00001; not merged, recoverable at 7ec8ec5ff

### MOJOLEARN_ARD_EQ_ONEPASS_OFF

- Verdict: DROPPED-quality. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-general-speed @ d42318df5, 7ec8ec5ff`.
- EXPERIMENTS.md:1328 (lane/apple-fast-general-speed verdicts (2026-10-05, lane/app): `ARD_EQ_ONEPASS_OFF` on ard / board, lane/apple-fast-general-speed @ d42318df5, 7ec8ec5ff, A/B rab11-ardonepass, 1349.6 -> 34.1 ms, **DROPPED-quality**: r2 0.327 -> -0.00001; not merged, recoverable at 7ec8ec5ff

### MOJOLEARN_C13_FOLD_STATS

- Verdict: DROPPED. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/ridgecv-c13@c59be1781, measured source 6fe3cfce38fd`.
- EXPERIMENTS.md:1530 (IDENTICAL CV coordinate descent (lane/classical-cv, 2026-10-): `C13_FOLD_STATS` on lasso-cv, enet-cv / taxi, istella (IDENTICAL, NV sm90 + AMD gfx942), lane/ridgecv-c13@c59be1781, measured source 6fe3cfce38fd, A/B T.C13.only (targeted-ab-20261007), candidate/baseline NV 4.01/1.11 (lasso taxi/istella), 0.79/1.11 (enet); AMD 1.40/1.13, 1.35/1.10 ms, **DROPPED, code deleted**: each fold cell rescans all n rows filtered by fold id (F-fold redundant reads) with a plain f32 serial sum; Taxi quality gate FAIL on both vendors (picked alpha moves). Replaced by `MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS`. RidgeCV half kept as `C13_FOLD_STATS` (default on).

### MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS

- Verdict: quality loss. Deleted 2026-10-07 by `f4de82db4` (Delete C13_CD_FOLD_STATS (measured loser) and unmeasured C18_TILE64/GRAM_PREFETCH; EXPERIMENTS rows).
- Recoverable at `5de5826c6` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `bindings/_mojolearn_x_linear_host.mojo`, `experiments/classical_identical_ideas/linear_controls.mojo`, `solver/impl/cd.mojo`, `x_linear/cd.mojo`, `x_linear/cd_grid.mojo`, `x_linear/classical_fold_stats.mojo`
- grid_controls/classical-cv.json removed: measured loser T.C13.only: lasso-cv taxi 4.01x NV / 1.40x AMD, enet-cv taxi 0.79x/1.35x, istella ~1.1x both; Taxi quality FAIL both vendors (F-fold full-n rescans, plain f32 serial sums). Replaced by enetcv_fold_blocks. EXPERIMENTS row added.
- EXPERIMENTS.md:1530 (IDENTICAL CV coordinate descent (lane/classical-cv, 2026-10-): `MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS` on lasso-cv, enet-cv / taxi, istella (IDENTICAL, NV sm90 + AMD gfx942), lane/ridgecv-c13@c59be1781, measured source 6fe3cfce38fd, A/B T.C13.only (targeted-ab-20261007), candidate/baseline NV 4.01/1.11 (lasso taxi/istella), 0.79/1.11 (enet); AMD 1.40/1.13, 1.35/1.10 ms, **DROPPED, code deleted**: each fold cell rescans all n rows filtered by fold id (F-fold redundant reads) with a plain f32 serial sum; Taxi quality gate FAIL on both vendors (picked alpha moves). Replaced by `MOJOLEARN_CLASSICAL_ENETCV_FOLD_BLOCKS`. RidgeCV half kept as `C13_FOLD_STATS` (default on).

### MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF

- Verdict: slower. Deleted 2026-10-08 by `3d12702e7` (grid-act-3: delete the c13_fold_stats_ridgecv off switch (MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF; grid ge123e6f9 off arm 1.19x/1.11x istella, 19.5x/18.0x taxi s).
- Recoverable at `d3f096a80` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `experiments/classical_identical_ideas/linear_controls.mojo`, `x_linear/ridgecv.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:207): removed: MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF retired 2026-10-08: the off arm is slower, ridge-cv NV 1.19x / AMD 1.11x istella, NV 19.5x / AMD 18.0x taxi, quality SAME (grid ge123e6f9); the RidgeCV fold cache is the only IDENTICAL route; see EXPERIMENTS.md
- grid_controls/classical-cv.json removed (control c13_fold_stats_ridgecv): DELETED 2026-10-08 (lane grid-act-3): IDENTICAL grid ge123e6f9, the off arm loses. off/on ms ratio NV/AMD: ridge-cv istella 1.19/1.11 (23110.5 -> 27500.3 NV, 62881.7 -> 69645.8 AMD), taxi 19.52/18.01 (152.2 -> 2970.3 NV, 287.9 -> 5183.8 AMD); quality SAME. The switch is gone: C13_FOLD_STATS = CLASSICAL_IDN (the fold cache is the only IDENTICAL RidgeCV route; FAST keeps the per-fold passes). Recoverable at main bc10b8b56; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1724 (IDENTICAL grid ge123e6f9: one promotion and six loser contro): `MOJOLEARN_CLASSICAL_C13_FOLD_STATS_OFF` on expanded:ridge-cv / istella, taxi, main @ bc10b8b56 (deleted on lane/grid-act-3), A/B ge123e6f9 (nv n0032, amd a0027), istella NV 23110.5 -> 27500.3 (1.19), AMD 62881.7 -> 69645.8 (1.11); taxi NV 152.2 -> 2970.3 (19.52), AMD 287.9 -> 5183.8 (18.01); 4.64x combined SLOWER (off vs on) ms, **DROP (off arm slower), switch deleted**: the per-fold passes in an IDENTICAL build lose to the compensated fold cache on every cell; quality SAME. `C13_FOLD_STATS = CLASSICAL_IDN`; FAST keeps the per-fold passes; the `_OFF` define is refused

### MOJOLEARN_CLASSICAL_C18_GRAM_PREFETCH

- Verdict: unmeasured. Deleted 2026-10-07 by `f4de82db4` (Delete C13_CD_FOLD_STATS (measured loser) and unmeasured C18_TILE64/GRAM_PREFETCH; EXPERIMENTS rows).
- Recoverable at `5de5826c6` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS.patch` (same patch as above).
- Files the patch restores: `bindings/_mojolearn_x_linear_host.mojo`, `experiments/classical_identical_ideas/linear_controls.mojo`, `solver/impl/cd.mojo`, `x_linear/cd.mojo`, `x_linear/cd_grid.mojo`, `x_linear/classical_fold_stats.mojo`
- grid_controls/classical-cv.json removed: per brief. Correction: implemented in solver/impl/cd.mojo (plain Lasso/ElasticNet Gram prefetch), never measured. Deleted; recoverable at c59be1781.
- EXPERIMENTS.md:1532 (IDENTICAL CV coordinate descent (lane/classical-cv, 2026-10-): `MOJOLEARN_CLASSICAL_C18_GRAM_PREFETCH` on plain Lasso/ElasticNet solver CD (`solver/impl/cd.mojo` Gram word prefetch), lane/ridgecv-c13@c59be1781, A/B none, unmeasured ms, **DROPPED (orchestrator brief L5), code deleted**: never measured; not on any CV route.

### MOJOLEARN_CLASSICAL_C18_TILE64

- Verdict: unmeasured. Deleted 2026-10-07 by `f4de82db4` (Delete C13_CD_FOLD_STATS (measured loser) and unmeasured C18_TILE64/GRAM_PREFETCH; EXPERIMENTS rows).
- Recoverable at `5de5826c6` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CLASSICAL_C13_CD_FOLD_STATS.patch` (same patch as above).
- Files the patch restores: `bindings/_mojolearn_x_linear_host.mojo`, `experiments/classical_identical_ideas/linear_controls.mojo`, `solver/impl/cd.mojo`, `x_linear/cd.mojo`, `x_linear/cd_grid.mojo`, `x_linear/classical_fold_stats.mojo`
- grid_controls/classical-cv.json removed: per brief. Correction: it DID have an implementation (solver/impl/cd.mojo CD_FUSED_STEPS 64 vs 128, plain Lasso/ElasticNet, not the CV route); never measured. Deleted; recoverable at lane/ridgecv-c13@c59be1781.
- EXPERIMENTS.md:1531 (IDENTICAL CV coordinate descent (lane/classical-cv, 2026-10-): `MOJOLEARN_CLASSICAL_C18_TILE64` on plain Lasso/ElasticNet solver CD (`solver/impl/cd.mojo` `CD_FUSED_STEPS` 64 vs 128), lane/ridgecv-c13@c59be1781, A/B none, unmeasured ms, **DROPPED (orchestrator brief L5), code deleted**: never measured; not on any CV route. A tile size belongs in an int sweep if revived.

### MOJOLEARN_CLASSICAL_C20_ROW_CACHE

- Verdict: serial shape. Deleted 2026-10-07 by `fc5c573f7` (svm: delete C20_ROW_CACHE (per-row host loop + 1x1 publish kernel); refuse the define; C22_TRIANGLE and C20_PAIR_LOAD square tile reachable again).
- Recoverable at `608a7cf4a` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CLASSICAL_C20_ROW_CACHE.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `experiments/classical_identical_ideas/linear_controls.mojo`, `svm/impl/classical_kernel_device.mojo`, `svm/impl/kernelcache.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:45): removed: C20_ROW_CACHE was a host loop of 3 launches per working-set row (one 1x1); forbidden serial shape, incumbent square tile is parallel
- grid_controls/classical-misc.json removed: forbidden serial shape; incumbent parallel. Host loop over n_ws working-set rows, 3 launches per row incl. a 1x1 publish kernel (kernelcache.mojo:327-338). Incumbent square tile is gather + identical GEMM. Code, kernels and slot buffers deleted; guard rejects the define; recoverable at origin/integration/switches-20261007 608a7cf4a (lane serial-cleanup, 2026-10-07)
- EXPERIMENTS.md:1551 (Serial-shape deletions (lane/serial-cleanup, 2026-10-07)): `MOJOLEARN_CLASSICAL_C20_ROW_CACHE` on SVC / SVR SMO square tile, integration/switches-20261007 @ 608a7cf4a, A/B none, - ms, **DROPPED (deleted, guard refuses)**: forbidden serial shape; incumbent parallel. Host loop of 3 launches per working-set row incl. a 1x1 publish kernel. Its early return hid `C22_TRIANGLE` and the square-tile half of `C20_PAIR_LOAD`: both reachable again and each OWES its own A/B

### MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF

- Verdict: slower. Deleted 2026-10-08 by `37b572c78` (grid-act-2: delete the ols_one_entry off switch (MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF; grid ge123e6f9 off arm 2.97x/1.74x istella, 6.17x/3.83x taxi slower, r2 same);).
- Recoverable at `3df299f27` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `python/mojolearn/linear_model.py`, `x_decomp/api.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:202): removed: MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF retired 2026-10-08: the off arm is slower, ols NV 2.97x / AMD 1.74x istella, NV 6.17x / AMD 3.83x taxi, r2 SAME (grid ge123e6f9); the one-entry route is the only IDENTICAL OLS route; see EXPERIMENTS.md
- grid_controls/classical-fixes.json removed (control ols_one_entry): DELETED 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 loser (the OFF arm). off/on ms ratio NV/AMD on ols: istella 2.968/1.738 (816.2 -> 2422.8 NV, 468.9 -> 815.1 AMD), taxi 6.165/3.828 (34.0 -> 209.9 NV, 14.5 -> 55.4 AMD); r2 SAME. The one-entry resident TSQR is the only IDENTICAL OLS route (x_decomp/api.mojo IDN_OLS_ONE_ENTRY = IDENTICAL; MOJOLEARN_IDN_ALL_OFF no longer clears it); the Python four-crossing sequence stays only for FAST builds and older bindings. Code recoverable at main 42d1e42c6; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1712 (IDENTICAL grid ge123e6f9: two promotions and eight loser arm): `MOJOLEARN_IDN_OLS_ONE_ENTRY_OFF` on classical:ols / istella, taxi, main @ 42d1e42c6 (switch deleted on lane/grid-act-2), A/B ge123e6f9, on -> off: istella NV 816.2 -> 2422.8 (2.968), AMD 468.9 -> 815.1 (1.738); taxi NV 34.0 -> 209.9 (6.165), AMD 14.5 -> 55.4 (3.828); 3.32x combined SLOWER ms, **DROP (the off arm is slower), switch deleted**: the old four-crossing sequence (_column_means + _center + _ols_tsqr) against the resident one-entry TSQR; r2 SAME. `IDN_OLS_ONE_ENTRY` (x_decomp/api.mojo) is now every IDENTICAL build (MOJOLEARN_IDN_ALL_OFF no longer clears bit 0); the Python sequence in linear_model.py stays only for FAST builds and older bindings.

### MOJOLEARN_IDN_SGD_EPOCH_KERNEL

- Verdict: noise. Deleted 2026-10-09 by `0c5b780af` (postmerge-act-2: delete fg-linear S1 (MOJOLEARN_IDN_SGD_EPOCH_KERNEL; post-merge A/B nv n0606->n0615, amd a1065->a1074: sgd-clf istella NV 0.99x / AMD 1.00x, ta).
- Recoverable at `5c137b55e` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_SGD_EPOCH_KERNEL.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `experiments/classical_identical_ideas/fg_linear_controls.mojo`, `x_linear/device.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:225): removed 2026-10-09 (post-merge A/B nv n0606->n0615, amd a1065->a1074, lane/postmerge-act-2): fg-linear S1 (large-batch SGD through the K-batch chunk kernel) was NOISE: sgd-clf istella NV 0.99x / AMD 1.00x, taxi NV 1.00x / AMD 1.00x, accuracy SAME, same digests; code at main 5c137b55e; see EXPERIMENTS.md
- grid_controls/fg-linear.json removed (control sgd_epoch_kernel): DELETED 2026-10-09 (lane postmerge-act-2): post-merge A/B on main (one run per arm) found S1 noise: candidate/default sgd-clf istella NV 1588.5 -> 1576.7 ms (0.99x) / AMD 3961.9 -> 3957.3 (1.00x), taxi NV 1036.6 -> 1033.1 (1.00x) / AMD 2320.4 -> 2325.9 (1.00x) (nv n0606 -> n0615, amd a1065 -> a1074); accuracy SAME (0.92033 / 0.75533), same digests. The define is refused in core/six_lane_experiment_guards.mojo; code recoverable at main 5c137b55e.
- EXPERIMENTS.md:1659 (Gap plan section 11: smaller family members (lane/gap-small-): `MOJOLEARN_IDN_SGD_EPOCH_KERNEL` on expanded:sgd-clf / taxi (istella keeps the grid form), lane/fg-linear @ 0f6fc775e, A/B owed, grid NV 1,040 / AMD 2,271-2,314 -> owed ms, **DELETED (noise, lane/postmerge-act-2: post-merge nv n0615, amd a1074, 0.99-1.00x)**: fg-linear S1: batch x (d+2) <= 2^17 at a host-schedule rate runs through the 64-batch chunk kernel (one block) instead of 2 launches a batch; same statements, no bit change expected (ID check owed).
- EXPERIMENTS.md:1764 (Post-merge A/B races: SGD epoch kernel, x_prep device codes ): `MOJOLEARN_IDN_SGD_EPOCH_KERNEL` on expanded:sgd-clf / istella, taxi, lane/postmerge-act-2 from main @ 5c137b55e, A/B post-merge (nv n0606 -> n0615, amd a1065 -> a1074), istella NV 1,588.5 -> 1,576.7 (0.99x), AMD 3,961.9 -> 3,957.3 (1.00x); taxi NV 1,036.6 -> 1,033.1 (1.00x), AMD 2,320.4 -> 2,325.9 (1.00x) ms, **DELETED (noise)**: accuracy SAME (0.92033 / 0.75533), same digests; the chunk kernel keeps its batch <= XG_TPB rule

### MOJOLEARN_KERNEL_FAST_BAYES_JACOBI

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `e302f0ee8` (x_linear: remove dropped KERNEL_FAST_BAYES_JACOBI skip threshold (DROPPED-noise; recover lane/apple-fast-kernel@9e851777c)).
- Recoverable at `e58326562` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_KERNEL_FAST_BAYES_JACOBI.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `x_linear/bayes.mojo`, `x_linear/device.mojo`, `x_linear/tops.mojo`
- EXPERIMENTS.md:244 (Linear (46)): `KERNEL_FAST_BAYES_JACOBI` on bayesian-ridge / istella, lane/apple-fast-kernel @ 9e851777c, A/B kernel-bayes-jacobi-ist, bayesian-ridge istella 2,741 -> 2,726 ms, **DROPPED-noise**: -0.6%; code removed from main e302f0ee8; recover at lane/apple-fast-kernel@9e851777c

### MOJOLEARN_KERNEL_FAST_BAYES_STATS

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `e33bb66e0` (x_linear: remove dropped KERNEL_FAST_BAYES_STATS (DROPPED-slower; recover lane/apple-fast-kernel@9e851777c)).
- Recoverable at `e302f0ee8` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_KERNEL_FAST_BAYES_STATS.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `x_linear/device.mojo`
- EXPERIMENTS.md:245 (Linear (46)): `KERNEL_FAST_BAYES_STATS` on bayesian-ridge / istella, lane/apple-fast-kernel @ 9e851777c, A/B kernel-bayes-stats-ist, bayesian-ridge istella 2,748 -> 3,143 ms, **DROPPED-slower**: +14.4%; define removed from main e33bb66e0 (code kept: KEPT BAYES_CLS1_STATS takes it); recover at lane/apple-fast-kernel@9e851777c

### MOJOLEARN_LSVR_DEVICE_CONVERGE

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-linsvr @ c649076a4`.
- EXPERIMENTS.md:273 (Linear (46)): `LSVR_DEVICE_CONVERGE` on linearsvr / taxi, lane/apple-fast-linsvr @ c649076a4, A/B linsvr-dconv-taxi, linearsvr taxi 26.2 vs board 27 ms, **DROPPED-noise**: rec-misc 2026-10-04: -3% at n=1 against a board that already has device convergence inside LSVR_ALL (d <= 32); not merged
- EXPERIMENTS.md:274 (Linear (46)): `LSVR_DEVICE_CONVERGE + LSVR_EVAL_SLIM + LSVR_LINESEARCH_BATCH` on linearsvr / taxi, lane/apple-fast-linsvr @ c649076a4, A/B linsvr-dconv-vs-batch-taxi, linearsvr taxi 26.2 vs board 27 ms, **DROPPED-noise**: rec-misc 2026-10-04: all three already on main via LSVR_ALL / EVAL_SLIM / LINESEARCH_BATCH; no separate gain

### MOJOLEARN_LSVR_DUAL_CD

- Verdict: DROPPED-slower. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-linsvr @ c649076a4`.
- EXPERIMENTS.md:275 (Linear (46)): `LSVR_DUAL_CD` on linearsvr / istella; linearsvr / taxi, lane/apple-fast-linsvr @ c649076a4, A/B linsvr-dualcd-taxi, linsvr-dualcd-istella, taxi 27 -> 57.1; istella 217 -> 218 ms, **DROPPED-slower**: rec-misc 2026-10-04: taxi 2.1x slower, istella inside noise; not merged

### MOJOLEARN_LSVR_EVAL_SLIM

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-linsvr @ c649076a4`.
- EXPERIMENTS.md:274 (Linear (46)): `LSVR_DEVICE_CONVERGE + LSVR_EVAL_SLIM + LSVR_LINESEARCH_BATCH` on linearsvr / taxi, lane/apple-fast-linsvr @ c649076a4, A/B linsvr-dconv-vs-batch-taxi, linearsvr taxi 26.2 vs board 27 ms, **DROPPED-noise**: rec-misc 2026-10-04: all three already on main via LSVR_ALL / EVAL_SLIM / LINESEARCH_BATCH; no separate gain

### MOJOLEARN_LSVR_FASTPATH_FIX

- Verdict: DROPPED-slower. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-linsvr @ c649076a4`.
- EXPERIMENTS.md:277 (Linear (46)): `LSVR_FASTPATH_FIX` on linearsvr / taxi; linearsvr / istella, lane/apple-fast-linsvr @ c649076a4, A/B linsvr-fix-taxi-x (M2); M3 istella, M2 taxi -0.2%; M3 istella +1.5% ms, **DROPPED-slower**: not merged

### MOJOLEARN_LSVR_FUSED_GRAD

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-linsvr @ c649076a4`.
- EXPERIMENTS.md:278 (Linear (46)): `LSVR_FUSED_GRAD` on linearsvr / taxi, lane/apple-fast-linsvr @ c649076a4, A/B M3 taxi, -1.9% ms, **DROPPED-noise**: not merged

### MOJOLEARN_LSVR_LINESEARCH_BATCH

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-linsvr @ c649076a4`.
- EXPERIMENTS.md:274 (Linear (46)): `LSVR_DEVICE_CONVERGE + LSVR_EVAL_SLIM + LSVR_LINESEARCH_BATCH` on linearsvr / taxi, lane/apple-fast-linsvr @ c649076a4, A/B linsvr-dconv-vs-batch-taxi, linearsvr taxi 26.2 vs board 27 ms, **DROPPED-noise**: rec-misc 2026-10-04: all three already on main via LSVR_ALL / EVAL_SLIM / LINESEARCH_BATCH; no separate gain

### MOJOLEARN_OLS_FAST_DEVICE_CENTER

- Verdict: DROPPED-semantics. Deleted 2026-10-03 by `dd83da010` (glm/OLS: remove dropped OLS_FAST_DEVICE_CENTER (DROPPED-semantics; recover lane/apple-fast-core@9a31ebb4c)).
- Recoverable at `a595988e8` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_OLS_FAST_DEVICE_CENTER.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `bindings/_mojolearn_estimators.mojo`, `glm/estimator.mojo`, `python/mojolearn/linear_model.py`
- EXPERIMENTS.md:246 (Linear (46)): `OLS_FAST_DEVICE_CENTER` on ols / taxi, lane/apple-fast-core @ 9a31ebb4c, A/B core-ols-dcenter-taxi, ols taxi 335.6 -> 157.6 ms, **DROPPED-semantics**: stale base: main OLS route changed (TSQR, then normal eq); code removed from main dd83da010; recover at lane/apple-fast-core@9a31ebb4c

### MOJOLEARN_QN_FAST_GRID_SUMS

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `5f1e86fd0` (glm/qn: remove dropped QN_FAST_GRID_SUMS (DROPPED-noise; recover lane/apple-fast-linear@1c7c213f8)).
- Recoverable at `219ae194a` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_QN_FAST_GRID_SUMS.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `glm/impl/qn/glm_base.mojo`
- EXPERIMENTS.md:248 (Linear (46)): `QN_FAST_GRID_SUMS` on linearsvc / istella; linearsvr / istella; logreg / istella, lane/apple-fast-linear @ 1c7c213f8, A/B linear-logreg-gs-istella, linear-svc-gs-istella, linear-svr-gs-istella, logreg 3,890 -> 4,235; svc 775 -> 802; svr 914 -> 213 ms, **DROPPED-noise**: mixed signs; code removed from main 5f1e86fd0; recover at lane/apple-fast-linear@1c7c213f8

### MOJOLEARN_QN_IDN_DCONV

- Verdict: slower. Deleted 2026-10-08 by `687938771` (grid-losers-1: delete qn_idn_dconv (grid ge123e6f9 neutral/slower on logreg/linearsvc/linearsvr); refuse MOJOLEARN_QN_IDN_DCONV and its POLL arms).
- Recoverable at `3978db7ce` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_QN_IDN_DCONV.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `glm/impl/qn/glm_base.mojo`, `glm/impl/qn/qn_dconv.mojo`, `glm/impl/qn/qn_solvers.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:183): removed: MOJOLEARN_QN_IDN_DCONV (and _POLL_2/_POLL_8) retired 2026-10-08: neutral/slower, NV 0.95-1.06x / AMD 0.98-1.08x on logreg, linearsvc, linearsvr (grid ge123e6f9); see EXPERIMENTS.md
- grid_controls/classical-fixes.json removed (control qn_idn_dconv): DELETED 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 neutral/slower (1.030x combined), a loser. on/off ms ratio NV/AMD: logreg istella 1.046/1.053, taxi 1.028/1.066; linearsvc istella 1.008/1.038, taxi 0.946/0.983; linearsvr istella 1.045/1.025, taxi 1.058/1.075 (SLOWER); accuracy/logloss/r2/rmse SAME. Code recoverable at main ad7ed2370; MOJOLEARN_QN_IDN_DCONV and its _POLL_2/_POLL_8 refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1608 (IDENTICAL grid ge123e6f9 losers deleted (lane/grid-losers-1,): `MOJOLEARN_QN_IDN_DCONV` on more:logreg, more:linearsvc, more:linearsvr / istella, taxi, main @ ad7ed2370 (deleted on lane/grid-losers-1), A/B ge123e6f9, logreg istella NV 1804 -> 1887 (1.046), AMD 2106 -> 2217 (1.053); taxi NV 22.4 -> 23.0 (1.028), AMD 21.6 -> 23.0 (1.066); linearsvc istella NV 503.6 -> 507.8 (1.008), AMD 457.5 -> 475.1 (1.038); taxi NV 109.7 -> 103.7 (0.946), AMD 46.1 -> 45.3 (0.983); linearsvr istella NV 121.6 -> 127.0 (1.045), AMD 88.1 -> 90.3 (1.025); taxi NV 34.8 -> 36.8 (1.058), AMD 39.1 -> 42.0 (1.075, SLOWER); 1.030x combined ms, **DROP (neutral / slower), code deleted**: the device step-1 Armijo + convergence loop costs more than the host line search it hands back to; accuracy / logloss / r2 / rmse SAME. Deleted the IDENTICAL device loop (`dconv_idn_*`, `dc_ls1_kernel`, glm/impl/qn/qn_dconv.mojo), its gated evaluation (three `*_gated_kernel`s and `enqueue_idn_dconv_eval`, glm/impl/qn/glm_base.mojo) and the call in `min_lbfgs` (glm/impl/qn/qn_solvers.mojo). The FAST `QN_FAST_DCONV` loop is untouched

### MOJOLEARN_QN_IDN_DCONV_POLL_2

- Verdict: slower. Deleted 2026-10-08 by `687938771` (grid-losers-1: delete qn_idn_dconv (grid ge123e6f9 neutral/slower on logreg/linearsvc/linearsvr); refuse MOJOLEARN_QN_IDN_DCONV and its POLL arms).
- Recoverable at `3978db7ce` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_QN_IDN_DCONV.patch` (same patch as above).
- Files the patch restores: `glm/impl/qn/glm_base.mojo`, `glm/impl/qn/qn_dconv.mojo`, `glm/impl/qn/qn_solvers.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:183): removed: MOJOLEARN_QN_IDN_DCONV (and _POLL_2/_POLL_8) retired 2026-10-08: neutral/slower, NV 0.95-1.06x / AMD 0.98-1.08x on logreg, linearsvc, linearsvr (grid ge123e6f9); see EXPERIMENTS.md

### MOJOLEARN_QN_IDN_DCONV_POLL_8

- Verdict: slower. Deleted 2026-10-08 by `687938771` (grid-losers-1: delete qn_idn_dconv (grid ge123e6f9 neutral/slower on logreg/linearsvc/linearsvr); refuse MOJOLEARN_QN_IDN_DCONV and its POLL arms).
- Recoverable at `3978db7ce` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_QN_IDN_DCONV.patch` (same patch as above).
- Files the patch restores: `glm/impl/qn/glm_base.mojo`, `glm/impl/qn/qn_dconv.mojo`, `glm/impl/qn/qn_solvers.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:183): removed: MOJOLEARN_QN_IDN_DCONV (and _POLL_2/_POLL_8) retired 2026-10-08: neutral/slower, NV 0.95-1.06x / AMD 0.98-1.08x on logreg, linearsvc, linearsvr (grid ge123e6f9); see EXPERIMENTS.md

### MOJOLEARN_RIDGE_FAST_CLS1_PREDICT

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `9decae29f` (gap-cls1: BAYES/ARD CLS1 STATS/PARTS/BATCH, RIDGE CLS1 CODES, NC CLS1 LABELS FAST+Apple default (_OFF); drop RIDGE/NC PREDICT and KNN PRESEED/SLICES2).
- Recoverable at `ebea91010` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_RIDGE_FAST_CLS1_PREDICT.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-9decae29f.patch`.
- Files the patch restores: `python/mojolearn/_expansion_linear.py`, `x_linear/cls1_fast.mojo`, `x_linear/device.mojo`
- EXPERIMENTS.md:249 (Linear (46)): `RIDGE_FAST_CLS1_CODES + RIDGE_FAST_CLS1_PREDICT` on ridge-clf / taxi, lane/apple-fast-gap-cls1 @ 4e341dc41, A/B gapcls1-rcall-taxi, ridge-clf taxi 120 -> 20.2 ms, **DROPPED-noise**: no better than CODES alone
- EXPERIMENTS.md:250 (Linear (46)): `RIDGE_FAST_CLS1_PREDICT` on ridge-clf / taxi, lane/apple-fast-gap-cls1 @ 4e341dc41, A/B gapcls1-rcpred-taxi, ridge-clf taxi -1.4% ms, **DROPPED-noise**: <5%

### MOJOLEARN_SGD_PERC_QOLD

- Verdict: DROPPED-quality. Deleted 2026-10-04 by `170cbedc0` (apple-fast verdicts 4: revert SGD_PERC averaging, IVF_COARSE_FAISS_INIT, KNN_FAST_REFINE to opt-in (rab5-perc, rab5-ivfinit, rab5-knnref)).
- Recoverable at `a0118d426` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_SGD_PERC_QOLD.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo`, `neighbors/estimator.mojo`, `x_linear/sgd.mojo`, `x_linear/sgd_avg.mojo`
- EXPERIMENTS.md:1228 (Quality fixes, classifiers (lane/apple-fast-q-clf, 2026-10-0): `MOJOLEARN_SGD_PERC_QOLD` on perceptron / taxi, istella (x_linear/sgd.mojo `SGD_PERC_AVG`, x_linear/sgd_avg.mojo), lane/apple-fast-q-clf @ a3bd71a65, A/B (owed), (owed) ms, **DROPPED-quality**: reconciled 2026-10-05: rab5-perc taxi 1401.12 -> 1400.29, accuracy .76219 -> .74097, Verdicts batch 4; reverted, opt-in MOJOLEARN_SGD_PERC_AVG. Was QUALITY-FIX, READY-AB: minibatch Perceptron returns the mean of its epoch-end iterates from epoch max_iter//2 on; audit accuracy 0.465 vs sklearn 0.751; float32 numpy model of the step (taxi 1M rows): last iterate 0.543/0.774/0.668/0.757 over seeds, mean 0.771/0.769/0.774; device grid + FAST host column


## Neighbors

### MOJOLEARN_C29_STREAM_TOPK

- Verdict: serial shape. Deleted 2026-10-07 by `58146b270` (knn: delete C29_STREAM_TOPK + C29_TILE (one thread per query over every index row); KNN_DIRECT_DISTANCE keeps only the row-major direct tile (host twin unchange).
- Recoverable at `fc5c573f7` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_C29_STREAM_TOPK.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-58146b270.patch`.
- Files the patch restores: `experiments/classical_identical_ideas/graph_controls.mojo`, `neighbors/impl/detail/knn_brute_force.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:50): removed: C29_STREAM_TOPK (and MOJOLEARN_C29_TILE) was one thread per query walking every index row; forbidden serial shape, incumbent kNN top-k is parallel
- grid_controls/classical-misc.json removed: forbidden serial shape; incumbent parallel. One thread per query walking every index row with O(k) insertion (classical_stream_topk.mojo). Incumbent top-k is block-per-row-per-column-tile + fixed merges. Gate (knn_brute_force.mojo:1946) and kernel file deleted; MOJOLEARN_KNN_DIRECT_DISTANCE no longer routes there: it keeps only the row-major direct tile pinned_distance_tile_direct_kernel (certified MMA and transposed-layout routes closed under it) and its unchanged host twin core/knn_host_predict.mojo; guard rejects the define; recoverable at origin/integration/switches-20261007 608a7cf4a (lane serial-cleanup, 2026-10-07)
- EXPERIMENTS.md:1552 (Serial-shape deletions (lane/serial-cleanup, 2026-10-07)): `MOJOLEARN_C29_STREAM_TOPK` on kNN brute force (KNN, kNN graphs), integration/switches-20261007 @ 608a7cf4a, A/B none, - ms, **DROPPED (deleted, guard refuses)**: forbidden serial shape; incumbent parallel. One thread per query over every index row. `MOJOLEARN_KNN_DIRECT_DISTANCE` no longer routes there: row-major direct tile only, host twin unchanged

### MOJOLEARN_C29_TILE

- Verdict: serial shape. Deleted 2026-10-07 by `58146b270` (knn: delete C29_STREAM_TOPK + C29_TILE (one thread per query over every index row); KNN_DIRECT_DISTANCE keeps only the row-major direct tile (host twin unchange).
- Recoverable at `fc5c573f7` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_C29_TILE.patch` (applies cleanly to main at the time of writing); first apply `experiments/removed/_shared-58146b270.patch`.
- Files the patch restores: `experiments/classical_identical_ideas/graph_controls.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:50): removed: C29_STREAM_TOPK (and MOJOLEARN_C29_TILE) was one thread per query walking every index row; forbidden serial shape, incumbent kNN top-k is parallel
- grid_controls/classical-misc.json removed: loop-blocking knob of the deleted C29 stream kernel only (C29_REFERENCE_TILE); guard rejects the define; recoverable at origin/integration/switches-20261007 608a7cf4a (lane serial-cleanup, 2026-10-07)
- EXPERIMENTS.md:1553 (Serial-shape deletions (lane/serial-cleanup, 2026-10-07)): `MOJOLEARN_C29_TILE` on kNN stream top-k tile knob, integration/switches-20261007 @ 608a7cf4a, A/B none, - ms, **DROPPED (deleted, guard refuses)**: forbidden serial shape; incumbent parallel. Loop knob of the deleted C29 kernel only

### MOJOLEARN_CAGRA_FAST_DOT

- Verdict: DROPPED-semantics. Deleted 2026-10-03 by `3852df59b` (gap-cagra: IVFG+EXACTD+SEEDS+ITERS FAST+Apple default (_OFF defines); delete DOT, WIDE, IVFG_P32, IVFG_P8 arms).
- Recoverable at `103923cba` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CAGRA_FAST_DOT.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `x_ann/cagra_device.mojo`, `x_ann/cagra_fast_knn.mojo`, `x_ann/fast_env.mojo`
- EXPERIMENTS.md:304 (Neighbors (42)): `CAGRA_FAST_DOT` on cagra / istella, lane/apple-fast-gap-cagra @ a3ebfc4a7, A/B gapcagra-dot-istella, cagra istella 21,239 -> ? ms, **DROPPED-semantics**: rec-misc 2026-10-04: arm deleted from main in 3852df59b; never judged alone; superseded by the IVFG + EXACTD + SEEDS + ITERS bundle

### MOJOLEARN_CAGRA_FAST_IVFG_P32

- Verdict: DROPPED-quality. Deleted 2026-10-03 by `3852df59b` (gap-cagra: IVFG+EXACTD+SEEDS+ITERS FAST+Apple default (_OFF defines); delete DOT, WIDE, IVFG_P32, IVFG_P8 arms).
- Recoverable at `103923cba` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CAGRA_FAST_DOT.patch` (same patch as above).
- Files the patch restores: `x_ann/cagra_device.mojo`, `x_ann/cagra_fast_knn.mojo`, `x_ann/fast_env.mojo`
- EXPERIMENTS.md:306 (Neighbors (42)): `CAGRA_FAST_IVFG + CAGRA_FAST_IVFG_P32` on cagra / istella, lane/apple-fast-gap-cagra @ a3ebfc4a7, A/B gapcagra-ivfg32-istella, cagra istella 21,239 -> 1,793 ms, **DROPPED-quality**: recall .9597; probes are not the loss

### MOJOLEARN_CAGRA_FAST_TEAM

- Verdict: DROPPED-noise. Deleted 2026-10-02 by `59afb759a` (x_ann/fast_env: every ann FAST switch a -D build define, no env reads; tools/afc_ab_def.sh; ab/ann.txt in the light define form).
- Recoverable at `e662dd863` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CAGRA_FAST_TEAM.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `x_ann/cagra_device.mojo`, `x_ann/fast_env.mojo`
- EXPERIMENTS.md:302 (Neighbors (42)): `CAGRA_FAST_TEAM` on cagra / istella, lane/apple-fast-ann @ 70833546a, A/B ann-cagra-team-istella, - ms, **DROPPED-noise**: reconciled 2026-10-05: ann-cagra-team-istella-b 172,845 -> 173,757 (+0.5%), LEDGER 2026-10-03; deleted at merge 8d43ec357. Was OPEN: A/B queued, no judged result yet

### MOJOLEARN_CAGRA_FAST_WIDE

- Verdict: DROPPED-semantics. Deleted 2026-10-03 by `3852df59b` (gap-cagra: IVFG+EXACTD+SEEDS+ITERS FAST+Apple default (_OFF defines); delete DOT, WIDE, IVFG_P32, IVFG_P8 arms).
- Recoverable at `103923cba` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CAGRA_FAST_DOT.patch` (same patch as above).
- Files the patch restores: `x_ann/cagra_device.mojo`, `x_ann/cagra_fast_knn.mojo`, `x_ann/fast_env.mojo`
- EXPERIMENTS.md:303 (Neighbors (42)): `CAGRA_FAST_WIDE` on cagra / istella, lane/apple-fast-gap-cagra @ a3ebfc4a7, A/B gapcagra-wide-istella, cagra istella 21,239 -> ? ms, **DROPPED-semantics**: rec-misc 2026-10-04: arm deleted from main in 3852df59b; never judged alone; superseded by the IVFG + EXACTD + SEEDS + ITERS bundle

### MOJOLEARN_IDN_PQ_SCAN_FUSED

- Verdict: slower. Deleted 2026-10-09 by `27fa9d4b1` (postmerge-act-3: delete fg-ivf A5 MOJOLEARN_IDN_PQ_SCAN_FUSED (ivf-pq istella NV 1.80x / AMD 1.00x, taxi NV 1.77x / AMD 1.00x slower, recall equal; nv n0669-n06).
- Recoverable at `0cc28e9bd` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_PQ_SCAN_FUSED.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `x_ann/ivf_scan_device.mojo`, `x_ann/vsearch_fast.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:230): removed 2026-10-09 (post-merge A/B nv n0669-n0671, amd a1131->a1132, lane/postmerge-act-3): fg-ivf A5 (pq_scan_fused_kernel as the IDENTICAL IVF-PQ score + top-k) was SLOWER: ivf-pq istella NV 1.80x / AMD 1.00x, taxi NV 1.77x / AMD 1.00x, recall equal; the tiled score + select chain stays (FAST Apple keeps PQ_SCAN_FUSED); see EXPERIMENTS.md, code at main a47bd9fb2
- grid_controls/fg-ivf.json removed (control idn_pq_scan_fused): DELETED 2026-10-09 (lane postmerge-act-3): post-merge A/B on main a47bd9fb2 (one run per arm) found fg-ivf A5 slower: ivf-pq istella NV 1.80x / AMD 1.00x (AMD 853.1 -> 852.4 ms), taxi NV 1.77x / AMD 1.00x (195.0 -> 195.5), recall@10 equal (nv n0669-n0671, amd a1131 -> a1132). The IDENTICAL scan keeps the tiled score + select chain; FAST Apple keeps PQ_SCAN_FUSED. Define refused by core/six_lane_experiment_guards.mojo. Code recoverable at main a47bd9fb2.
- EXPERIMENTS.md:1649 (Gap plan section 11: smaller family members (lane/gap-small-): `MOJOLEARN_IDN_PQ_SCAN_FUSED` on ann:ivf-pq / taxi, istella, lane/fg-ivf @ 1c24f289b, A/B post-merge (nv n0669-n0671, amd a1131 -> a1132), see lane/postmerge-act-3 section ms, **DELETED (loser, lane/postmerge-act-3)**: read_ivf.md A5: pq_scan_fused_kernel (score + register top-k, tiles its own table) as IDENTICAL's scan; no candidate buffer, no select chain; (distance, id) total order, same outputs and n_candidates_
- EXPERIMENTS.md:1781 (Post-merge A/B races: IVF strided init promoted; PQ fused sc): `MOJOLEARN_IDN_PQ_SCAN_FUSED` on ann:ivf-pq / istella, taxi, lane/postmerge-act-3 from main @ a47bd9fb2, A/B post-merge (nv n0669-n0671, amd a1131 -> a1132), istella NV 1.80x, AMD 853.1 -> 852.4 (1.00x); taxi NV 1.77x, AMD 195.0 -> 195.5 (1.00x) ms, **DELETED (loser)**: recall@10 equal, same digests on AMD; the tiled score + select chain stays IDENTICAL's scan

### MOJOLEARN_IVFG_EXACTD

- Verdict: DROPPED-semantics. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-gap-cagra @ 2b16b4322`.
- EXPERIMENTS.md:312 (Neighbors (42)): `CAGRA_FAST_IVFG + IVFG_EXACTD + IVFG_P8` on cagra / istella, lane/apple-fast-gap-cagra @ 2b16b4322, A/B gapcagra-ivfgx8-istella, 21,239 -> ? ms, **DROPPED-semantics**: rec-misc 2026-10-04: arm deleted from main in 3852df59b; never judged alone; superseded by the IVFG + EXACTD + SEEDS + ITERS bundle

### MOJOLEARN_IVFG_P8

- Verdict: DROPPED-semantics. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-gap-cagra @ 2b16b4322`.
- EXPERIMENTS.md:312 (Neighbors (42)): `CAGRA_FAST_IVFG + IVFG_EXACTD + IVFG_P8` on cagra / istella, lane/apple-fast-gap-cagra @ 2b16b4322, A/B gapcagra-ivfgx8-istella, 21,239 -> ? ms, **DROPPED-semantics**: rec-misc 2026-10-04: arm deleted from main in 3852df59b; never judged alone; superseded by the IVFG + EXACTD + SEEDS + ITERS bundle

### MOJOLEARN_IVF_FAST_DEVICE_CSR

- Verdict: DROPPED-slower. Deleted 2026-10-02 by `59afb759a` (x_ann/fast_env: every ann FAST switch a -D build define, no env reads; tools/afc_ab_def.sh; ab/ann.txt in the light define form).
- Recoverable at `e662dd863` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IVF_FAST_DEVICE_CSR.patch` (conflicts with main: restore by hand from the recoverable sha).
- Files the patch restores: `ivf/impl/neighbors/ivf_flat/fast_build_device.mojo`, `ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo`, `x_ann/fast_env.mojo`
- EXPERIMENTS.md:321 (Neighbors (42)): `IVF_FAST_DEVICE_CSR` on ivf / istella; ivf-pq / istella, lane/apple-fast-ann @ 70833546a, A/B ann-ivfpq-csr-istella, ann-ivf-csr-istella, - ms, **DROPPED-slower**: reconciled 2026-10-05: ann-ivfpq-csr-istella-b pq +0.4%, ivf +8% (LEDGER 2026-10-03 "ANN lane done"); deleted at merge 8d43ec357. Was OPEN: A/B queued, no judged result yet

### MOJOLEARN_IVF_FAST_DEVICE_TRAINSET

- Verdict: DROPPED-noise. Deleted 2026-10-02 by `59afb759a` (x_ann/fast_env: every ann FAST switch a -D build define, no env reads; tools/afc_ab_def.sh; ab/ann.txt in the light define form).
- Recoverable at `e662dd863` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IVF_FAST_DEVICE_TRAINSET.patch` (conflicts with main: restore by hand from the recoverable sha).
- Files the patch restores: `ivf/impl/neighbors/ivf_flat/fast_build_device.mojo`, `ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo`, `x_ann/fast_env.mojo`
- EXPERIMENTS.md:322 (Neighbors (42)): `IVF_FAST_DEVICE_TRAINSET` on ivf / istella; ivf-pq / istella; ivf-sq / istella, lane/apple-fast-ann @ 70833546a, A/B ann-ivfpq-trainset-istella, ann-ivfsq-trainset-istella, ann-ivf-trainset-istella, - ms, **DROPPED-noise**: reconciled 2026-10-05: ann-ivf*-trainset pq -1.9%, sq -5.1%, ivf -4.8%: <5% n=1, mixed (LEDGER 2026-10-03); deleted at merge 8d43ec357. Was OPEN: A/B queued, no judged result yet

### MOJOLEARN_IVF_FAST_SCAN_SELECT

- Verdict: DROPPED-noise. Deleted 2026-10-02 by `59afb759a` (x_ann/fast_env: every ann FAST switch a -D build define, no env reads; tools/afc_ab_def.sh; ab/ann.txt in the light define form).
- Recoverable at `e662dd863` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IVF_FAST_SCAN_SELECT.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `x_ann/fast_env.mojo`, `x_ann/ivf_scan_device.mojo`
- EXPERIMENTS.md:323 (Neighbors (42)): `IVF_FAST_SCAN_SELECT` on ivf-pq / istella; ivf-rabitq / istella; ivf-sq / istella, lane/apple-fast-ann @ 70833546a, A/B ann-ivfpq-select-istella, ann-ivfsq-select-istella, ann-ivfrq-select-istella, - ms, **DROPPED-noise**: reconciled 2026-10-05: ann-ivf*-select pq -0.3%, sq -0.5%, rq +0.1% (LEDGER 2026-10-03); deleted at merge 8d43ec357. Was OPEN: A/B queued, no judged result yet

### MOJOLEARN_IVF_LAYOUT_SCATTER

- Verdict: slower. Deleted 2026-10-09 by `27fa9d4b1` (postmerge-act-3: delete fg-ivf A5 MOJOLEARN_IDN_PQ_SCAN_FUSED (ivf-pq istella NV 1.80x / AMD 1.00x, taxi NV 1.77x / AMD 1.00x slower, recall equal; nv n0669-n06).
- Recoverable at `0cc28e9bd` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IVF_LAYOUT_SCATTER.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `ivf/impl/neighbors/ivf_flat/ivf_group_device.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:231): removed 2026-10-09 (post-merge A/B nv n0669-n0671, amd a1131->a1133, lane/postmerge-act-3): fg-ivf B3 (counting-sort IVF list layout) was SLOWER: ivf-pq istella NV 1.62x / AMD 1.00x, taxi NV 1.80x / AMD 1.00x, recall equal; the radix-sort layout stays; see EXPERIMENTS.md, code at main a47bd9fb2
- grid_controls/fg-ivf.json removed (control ivf_layout_scatter): DELETED 2026-10-09 (lane postmerge-act-3): post-merge A/B on main a47bd9fb2 (one run per arm) found fg-ivf B3 slower: ivf-pq istella NV 1.62x / AMD 1.00x (AMD 853.1 -> 851.6 ms), taxi NV 1.80x / AMD 1.00x (195.0 -> 195.3), recall@10 equal (nv n0669-n0671, amd a1131 -> a1133). The radix-sort layout stays. Define refused by core/six_lane_experiment_guards.mojo. Code recoverable at main a47bd9fb2.
- EXPERIMENTS.md:1653 (Gap plan section 11: smaller family members (lane/gap-small-): `MOJOLEARN_IVF_LAYOUT_SCATTER` on ann:ivf (and every IVF layout) / taxi, istella, lane/fg-ivf @ 1c24f289b, A/B post-merge (nv n0669-n0671, amd a1131 -> a1133), see lane/postmerge-act-3 section ms, **DELETED (loser, lane/postmerge-act-3)**: read_ivf.md B3: per-block rank + list-major table scan + scatter instead of the 32-bit radix sort + histogram + scan, taken while the table is <= 4 n words (n_lists <= 4 x 1024); same offsets, ids and words
- EXPERIMENTS.md:1782 (Post-merge A/B races: IVF strided init promoted; PQ fused sc): `MOJOLEARN_IVF_LAYOUT_SCATTER` on ann:ivf-pq / istella, taxi (every IVF layout), lane/postmerge-act-3 from main @ a47bd9fb2, A/B post-merge (nv n0669-n0671, amd a1131 -> a1133), istella NV 1.62x, AMD 853.1 -> 851.6 (1.00x); taxi NV 1.80x, AMD 195.0 -> 195.3 (1.00x) ms, **DELETED (loser)**: recall@10 equal, same digests on AMD; the radix-sort layout stays

### MOJOLEARN_KNN_FAST_CLS1_PRESEED

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `9decae29f` (gap-cls1: BAYES/ARD CLS1 STATS/PARTS/BATCH, RIDGE CLS1 CODES, NC CLS1 LABELS FAST+Apple default (_OFF); drop RIDGE/NC PREDICT and KNN PRESEED/SLICES2).
- Recoverable at `ebea91010` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_KNN_FAST_CLS1_PRESEED.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-9decae29f.patch`.
- Files the patch restores: `neighbors/impl/detail/fast_mma_knn.mojo`
- EXPERIMENTS.md:293 (Neighbors (42)): `KNN_FAST_CLS1_PRESEED` on knn / istella, lane/apple-fast-gap-cls1 @ 4e341dc41, A/B gapcls1-knnseed-istella, knn istella 357 -> 343 ms, **DROPPED-noise**: -3.9%, n=1, marginal
- EXPERIMENTS.md:294 (Neighbors (42)): `KNN_FAST_CLS1_PRESEED + KNN_FAST_CLS1_SLICES2` on knn / istella, lane/apple-fast-gap-cls1 @ 4e341dc41, A/B gapcls1-knnall-istella, knn istella 357 -> 385 ms, **DROPPED-slower**: slower

### MOJOLEARN_KNN_FAST_CLS1_SLICES2

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `9decae29f` (gap-cls1: BAYES/ARD CLS1 STATS/PARTS/BATCH, RIDGE CLS1 CODES, NC CLS1 LABELS FAST+Apple default (_OFF); drop RIDGE/NC PREDICT and KNN PRESEED/SLICES2).
- Recoverable at `ebea91010` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_KNN_FAST_CLS1_PRESEED.patch` (same patch as above).
- Files the patch restores: `neighbors/impl/detail/fast_mma_knn.mojo`
- EXPERIMENTS.md:294 (Neighbors (42)): `KNN_FAST_CLS1_PRESEED + KNN_FAST_CLS1_SLICES2` on knn / istella, lane/apple-fast-gap-cls1 @ 4e341dc41, A/B gapcls1-knnall-istella, knn istella 357 -> 385 ms, **DROPPED-slower**: slower
- EXPERIMENTS.md:295 (Neighbors (42)): `KNN_FAST_CLS1_SLICES2` on knn / istella, lane/apple-fast-gap-cls1 @ 4e341dc41, A/B gapcls1-knnsl2-istella, knn istella 357 -> 421 ms, **DROPPED-slower**: slower

### MOJOLEARN_LLE_FAST_KNN

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-isotonic-knn @ 7385fcfdd`.
- EXPERIMENTS.md:296 (Neighbors (42)): `LLE_FAST_KNN` on -, lane/apple-fast-isotonic-knn @ 7385fcfdd, A/B ik-lle-knn-taxi; M3 re-check (lane/apple-fast-m2b1 batch), lle taxi 4,266 -> 4,289; M3 -0.1% ms, **DROPPED-noise**: +0.5%; M2 ik-lle-knn-taxi-b +0.4%; M3 re-check -0.1% (inside spread): loser, never merges

### MOJOLEARN_LLE_SPARSE_EIG

- Verdict: DROPPED-slower. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-lle @ f2ea1ecb5`.
- EXPERIMENTS.md:409 (Decomp (34)): `LLE_SPARSE_EIG` on lle / taxi, lane/apple-fast-lle @ f2ea1ecb5, A/B lle-sparse-taxi, lle taxi 4,274 -> 9,174 ms, **DROPPED-slower**: +115%

### MOJOLEARN_NC_FAST_CLS1_PREDICT

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `9decae29f` (gap-cls1: BAYES/ARD CLS1 STATS/PARTS/BATCH, RIDGE CLS1 CODES, NC CLS1 LABELS FAST+Apple default (_OFF); drop RIDGE/NC PREDICT and KNN PRESEED/SLICES2).
- Recoverable at `ebea91010` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_NC_FAST_CLS1_PREDICT.patch` (conflicts with main: restore by hand from the recoverable sha); first apply `experiments/removed/_shared-9decae29f.patch`.
- Files the patch restores: `python/mojolearn/_expansion_neighbors.py`, `x_neighbors/nc_cls1.mojo`
- EXPERIMENTS.md:297 (Neighbors (42)): `NC_FAST_CLS1_LABELS + NC_FAST_CLS1_PREDICT` on nearest-centroid / taxi, lane/apple-fast-gap-cls1 @ 4e341dc41, A/B gapcls1-ncall-taxi, nearest-centroid taxi 105 -> 27 ms, **DROPPED-noise**: worse than LABELS alone
- EXPERIMENTS.md:298 (Neighbors (42)): `NC_FAST_CLS1_PREDICT` on nearest-centroid / taxi, lane/apple-fast-gap-cls1 @ 4e341dc41, A/B gapcls1-ncpred-taxi, nearest-centroid taxi -4% ms, **DROPPED-noise**: <5%

### MOJOLEARN_RADIUS_FAST_REUSE_COUNT

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `f9af6028e` (neighbors: remove dropped RADIUS_FAST_REUSE_COUNT (DROPPED-noise; recover lane/apple-fast-neighbors2@5fb6edd3f)).
- Recoverable at `5f1e86fd0` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_RADIUS_FAST_REUSE_COUNT.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `neighbors/estimator.mojo`
- EXPERIMENTS.md:299 (Neighbors (42)): `RADIUS_FAST_REUSE_COUNT` on -, lane/apple-fast-neighbors2 @ 5fb6edd3f, A/B n2-radius-reuse-taxi, radius-neighbors taxi 0.1 -> 0.1 ms, **DROPPED-noise**: ms-scale; code removed from main f9af6028e; recover at lane/apple-fast-neighbors2@5fb6edd3f

### MOJOLEARN_XN_FAST_IMPUTE_TILED2

- Verdict: DROPPED-slower. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-isotonic-knn @ 7385fcfdd`.
- EXPERIMENTS.md:344 (Neighbors (42)): `XN_FAST_IMPUTE_TILED2` on knn-imputer / taxi, lane/apple-fast-isotonic-knn @ 7385fcfdd, A/B ik-imp-t2-taxi-b (M2); M3 re-check, M2 +1.5%; M3 +5.6% ms, **DROPPED-slower**: slower on the M2 and the M3; never merged to main or lane/apple-fast-m2b1 (code only on its lane branch)

### MOJOLEARN_XN_FAST_MMA_ROUTE

- Verdict: DROPPED-slower. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-isotonic-knn @ 7385fcfdd`.
- EXPERIMENTS.md:345 (Neighbors (42)): `XN_FAST_MMA_ROUTE` on lof / taxi; lle / taxi, lane/apple-fast-isotonic-knn @ 7385fcfdd, A/B ik-lof-mma-taxi-b, ik-lle-mma-taxi-b (M2), lof taxi 120.7 -> 2,659 (22x slower); lle -1.8% ms, **DROPPED-slower**: never on main; not merged

### MOJOLEARN_XN_FAST_TILED_RBF

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `f103d7381` (x_neighbors: remove dropped XN_FAST_TILED_RBF kernel_tiled op and its env read (DROPPED-noise; recover lane/apple-fast-neighbors2@5fb6edd3f)).
- Recoverable at `7ff2caf99` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_XN_FAST_TILED_RBF.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `bindings/_mojolearn_x_neighbors.mojo`, `bindings/_mojolearn_x_neighbors_host.mojo`, `python/mojolearn/_expansion_neighbors.py`, `python/mojolearn/_surface_neighbors.py`, `x_neighbors/gen.py`, `x_neighbors/iter_device.mojo`, `x_neighbors/iter_host.mojo`
- EXPERIMENTS.md:559 (Kernel / GP (11)): `XN_FAST_TILED_RBF` on -, lane/apple-fast-neighbors2 @ 5fb6edd3f, A/B n2-ocsvm-tiled-istella, ocsvm istella 440 -> 445 ms, **DROPPED-noise**: +1.2%; code removed from main f103d7381; recover at lane/apple-fast-neighbors2@5fb6edd3f

### MOJOLEARN_XN_PCS_SPARSE

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `3d1c6bd73` (x_neighbors: remove dropped XN_PCS_SPARSE and x_neighbors/pcs_sparse.mojo (DROPPED-noise; recover lane/apple-fast-neighbors2@5fb6edd3f)).
- Recoverable at `f103d7381` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_XN_PCS_SPARSE.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `x_neighbors/iter_device.mojo`, `x_neighbors/pcs_sparse.mojo`
- EXPERIMENTS.md:560 (Kernel / GP (11)): `XN_PCS_SPARSE` on poly-count-sketch / taxi, lane/apple-fast-neighbors2 @ 5fb6edd3f, A/B n2-pcs-sparse-taxi, poly-count-sketch taxi 0.3 -> 0.3 ms, **DROPPED-noise**: no change; code removed from main 3d1c6bd73; recover at lane/apple-fast-neighbors2@5fb6edd3f


### MOJOLEARN_CAGRA_FAST_IVFG_LOWD

- Verdict: DROPPED-quality. Deleted 2026-10-09 by lane/owed-deletions-D3 (owed deletion, D3).
- Recoverable at `b639a2bd2` (main the lane branched from). Patch: `experiments/removed/MOJOLEARN_CAGRA_FAST_IVFG_LOWD.patch` (reverse of the lane's deletion commit; applies to the lane head).
- What it tried: the IVFG candidate graph for d <= 64 without the 4x search seeds, as an opt-in beside the LOWD_SEEDS4 default (only reachable with -D MOJOLEARN_CAGRA_FAST_IVFG_LOWD_SEEDS4_OFF).
- Files the patch restores: `x_ann/fast_env.mojo`
- EXPERIMENTS.md:314: `CAGRA_FAST_IVFG_LOWD` | cagra / taxi (istella must be identical) | lane/apple-fast-w2-cagra (base b2b1c22bc) | w2-cagra-lowd-q, w2-cagra-lowd-taxi | cagra taxi 2,900 -> ? | DROPPED-quality | reconciled 2026-10-05: w2-cagra-lowd-q taxi recall@10 .997925 -> .997125 (gate B >= A), Manager verdicts session 2; LOWD_SEEDS4 (row below) is the default. Was OPEN: IVFG graph for d <= 64 (taxi d = 11 still built the exact 1.6e11-pair graph); gate recall@10 B >= A (tools/cagra_lowd_pair.py)
- EXPERIMENTS.md:879: `MOJOLEARN_CAGRA_FAST_IVFG_LOWD` | cagra taxi | lane/apple-fast-w2-cagra 5d7d79cb5 | w2-cagra-lowd-q | taxi recall@10 A 0.997925 -> B 0.997125 (gate: B >= A); istella identical | DROP-quality; LOWD_SEEDS4 queued
- Guard refusal (core/six_lane_experiment_guards.mojo): removed 2026-10-09 (lane/owed-deletions-D3): CAGRA_FAST_IVFG_LOWD alone (low-d IVFG graph without SEEDS4, opt-in with _LOWD_SEEDS4_OFF) lost recall: cagra taxi recall@10 .997925 -> .997125 (gate B >= A); the low-d graph stays inside the LOWD_SEEDS4 FAST default; code at main b639a2bd2; see docs/TOMBSTONES.md

### MOJOLEARN_CAGRA_FAST_SEEDS4

- Verdict: DROPPED-semantics. Deleted 2026-10-09 by lane/owed-deletions-D3 (owed deletion, D3).
- Recoverable at `b639a2bd2` (main the lane branched from). Patch: `experiments/removed/MOJOLEARN_CAGRA_FAST_SEEDS4.patch` (reverse of the lane's deletion commit; applies to the lane head).
- What it tried: four times the CAGRA search seed work as a standalone opt-in define (taxi recall .9997).
- Files the patch restores: `x_ann/fast_env.mojo`
- EXPERIMENTS.md:308: `CAGRA_FAST_SEEDS4` | cagra / taxi | lane/apple-fast-gap-cagra @ 2b16b4322 | gapcagra-seeds4-taxi | 2,887 -> ? | DROPPED-semantics | rec-misc 2026-10-04: never judged alone; SEEDS (in the 3852df59b bundle) kept; `_CAGRA_SEEDS4` survives only inside the opt-in LOWD_SEEDS4 candidate
- Guard refusal (core/six_lane_experiment_guards.mojo): removed 2026-10-09 (lane/owed-deletions-D3): CAGRA_FAST_SEEDS4 alone (4x search seeds without the low-d graph) was never judged alone (DROPPED-semantics, gapcagra-seeds4-taxi); the 4x seeds stay inside the LOWD_SEEDS4 FAST default; code at main b639a2bd2; see docs/TOMBSTONES.md

## Prep

### MOJOLEARN_ACHI2_FAST_DEVCHECK

- Verdict: DROPPED. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-gap-kapprox2 @ 22aaa623e`.
- EXPERIMENTS.md:681 (Gap kapprox2 (lane/apple-fast-gap-kapprox2, Oct 3)): `ACHI2_FAST_DEVCHECK` on additive-chi2 / istella, taxi, lane/apple-fast-gap-kapprox2 @ 22aaa623e, A/B kap2-achi2-devcheck-{istella,taxi}, +1%, 0.9 -> 2.8 ms, **DROPPED**: device flag for the X < 0 check costs a launch + sync more than the host min pass at these sizes; not merged

### MOJOLEARN_CLASSICAL_C55_CLASS_GROUP

- Verdict: quality loss. Deleted 2026-10-07 by `213f349ed` (classical-nbda: delete C55, add C61 single-pass class stats (NB arms, DA), split C04_LDA, C56_QDA_PROJECT arms).
- Recoverable at `8be4d20d4` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CLASSICAL_C55_CLASS_GROUP.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `bindings/_mojolearn_x_prep.mojo`, `bindings/_mojolearn_x_prep_host.mojo`, `experiments/classical_identical_ideas/shared_controls.mojo`, `experiments/classical_identical_ideas/stats_controls.mojo`, `naive_bayes/da.mojo`, `python/mojolearn/_expansion_prep.py`, `x_prep/blocked.mojo`, `x_prep/host/program.mojo`, `x_prep/prims.mojo`, `x_prep/units.mojo`
- grid_controls/classical-nbda.json removed: measured loser: serial (class-group, column) walk of all n rows twice, plain f32 sums; gaussian-nb taxi 3.76x NV / 7.30x AMD and quality failed (gaussian-nb taxi, lda-clf istella). Code deleted; row in docs/apple-fast/EXPERIMENTS.md
- EXPERIMENTS.md:1522 (Classical IDENTICAL drops (lane classical-nbda, 2026-10-07)): `MOJOLEARN_CLASSICAL_C55_CLASS_GROUP` on GaussianNB, LDA, QDA, Multinomial/Bernoulli/Complement NB, feature-selection class stats / taxi, istella, main @ 8be4d20d4 (measured source 6fe3cfce38fd); deleted on lane/classical-nbda, A/B experiments/six_lane_integration/measurements/20261006/BOARD.md rows T.C55.only; ~/mojolearn-evidence/board-review-20261007/review_classical.md C55, NV sm90 gaussian-nb taxi/istella 3.76x/1.68x, lda-clf taxi/istella 2.39x/1.28x; AMD gfx942 7.30x/3.23x, 3.47x/1.69x ms, **DROPPED (code deleted)**: one thread per (class group, column) walking all n rows twice with a plain f32 running sum: serial, slower on both vendors, and quality failed (gaussian-nb taxi, lda-clf istella). Replaced by C61 (blocked single-pass, Chan merge, compensated)

### MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS

- Verdict: slower. Deleted 2026-10-08 by `d3f096a80` (grid-act-3: delete c61_da and c61_nb (both arms; the shared C61 csbm_* ops 190-192; grid ge123e6f9 lda-clf NV 1.38x/1.54x AMD 1.04x/1.11x, gaussian-nb NV 1.37-1).
- Recoverable at `c829cbe51` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `bindings/_mojolearn_x_prep.mojo`, `bindings/_mojolearn_x_prep_host.mojo`, `experiments/classical_identical_ideas/shared_controls.mojo`, `python/mojolearn/_expansion_prep.py`, `x_prep/blocked.mojo`, `x_prep/units.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:206): removed: MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS (=1|2) and MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS retired 2026-10-08: slower, gaussian-nb NV 1.37-1.58x / AMD 1.02-1.08x, lda-clf NV 1.38x/1.54x / AMD 1.04x/1.11x (istella/taxi), quality SAME (grid ge123e6f9); see EXPERIMENTS.md
- grid_controls/classical-nbda.json removed (control c61_da): DELETED 2026-10-08 (lane grid-act-3): IDENTICAL grid ge123e6f9 loser. on/off ms ratio NV/AMD: lda-clf istella 1.38/1.04 (201.0 -> 277.3 NV, 295.6 -> 307.2 AMD), taxi 1.54/1.11 (15.2 -> 23.4 NV, 16.4 -> 18.3 AMD); 1.25x combined SLOWER; quality SAME. Shares the deleted C61 csbm_* ops 190-192 (x_prep/blocked.mojo) with c61_nb; the Python LDA/QDA routes are the blocked two-pass class stats again. Code recoverable at main bc10b8b56; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1722 (IDENTICAL grid ge123e6f9: one promotion and six loser contro): `MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS` on expanded:lda-clf / istella, taxi, main @ bc10b8b56 (deleted on lane/grid-act-3), A/B ge123e6f9 (nv n0022, amd a0018), istella NV 201.0 -> 277.3 (1.38), AMD 295.6 -> 307.2 (1.04); taxi NV 15.2 -> 23.4 (1.54), AMD 16.4 -> 18.3 (1.11); 1.25x combined SLOWER ms, **DROP (slower), code deleted**: LDA svd within-class std from the pooled class M2 (csbm_pool) instead of X - mean[y] and two column-stat passes; quality SAME. Shares the C61 csbm_* ops with the NB arms below

### MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS

- Verdict: slower. Deleted 2026-10-08 by `d3f096a80` (grid-act-3: delete c61_da and c61_nb (both arms; the shared C61 csbm_* ops 190-192; grid ge123e6f9 lda-clf NV 1.38x/1.54x AMD 1.04x/1.11x, gaussian-nb NV 1.37-1).
- Recoverable at `c829cbe51` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS.patch` (same patch as above).
- Files the patch restores: `bindings/_mojolearn_x_prep.mojo`, `bindings/_mojolearn_x_prep_host.mojo`, `experiments/classical_identical_ideas/shared_controls.mojo`, `python/mojolearn/_expansion_prep.py`, `x_prep/blocked.mojo`, `x_prep/units.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:206): removed: MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS (=1|2) and MOJOLEARN_CLASSICAL_C61_DA_CLASS_STATS retired 2026-10-08: slower, gaussian-nb NV 1.37-1.58x / AMD 1.02-1.08x, lda-clf NV 1.38x/1.54x / AMD 1.04x/1.11x (istella/taxi), quality SAME (grid ge123e6f9); see EXPERIMENTS.md
- grid_controls/classical-nbda.json removed (control c61_nb): DELETED 2026-10-08 (lane grid-act-3): IDENTICAL grid ge123e6f9 loser, both arms. on/off ms ratio NV/AMD: gaussian-nb arm 1 istella 1.58/1.05 (57.3 -> 90.6 NV, 28.8 -> 30.2 AMD), taxi 1.42/1.08 (14.7 -> 20.9 NV, 12.7 -> 13.7 AMD), 1.26x combined; arm 2 istella 1.57/1.02, taxi 1.37/1.03, 1.22x combined SLOWER; quality SAME. The C61 csbm_* ops 190-192 (x_prep/blocked.mojo, units.mojo), the binding bits 16-64 and the Python _c61/_class_stats_m2 routes are gone. Code recoverable at main bc10b8b56; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1723 (IDENTICAL grid ge123e6f9: one promotion and six loser contro): `MOJOLEARN_CLASSICAL_C61_NB_CLASS_STATS=1` on expanded:gaussian-nb / istella, taxi, main @ bc10b8b56 (deleted on lane/grid-act-3), A/B ge123e6f9 (arm 1 nv n0022, amd a0018; arm 2 nv n0034, amd a0029), arm 1 istella NV 57.3 -> 90.6 (1.58), AMD 28.8 -> 30.2 (1.05); taxi NV 14.7 -> 20.9 (1.42), AMD 12.7 -> 13.7 (1.08); 1.26x combined; arm 2 istella NV 57.3 -> 89.8 (1.57), AMD 28.8 -> 29.3 (1.02); taxi NV 14.7 -> 20.1 (1.37), AMD 12.7 -> 13.1 (1.03); 1.22x combined SLOWER ms, **DROP (slower), code deleted**: single blocked pass (per-block Welford M2, Chan merge, compensated) and arm 2's epsilon from the class M2; quality SAME. csbm_part/fold/pool (x_prep ops 190-192), binding bits 16-64 and the Python `_c61` / `_class_stats_m2` routes deleted; both defines refused

### MOJOLEARN_KSHAP_FAST_SIGNGRAM

- Verdict: DROPPED. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-gap-kapprox2 @ 4fd464a43`.
- EXPERIMENTS.md:683 (Gap kapprox2 (lane/apple-fast-gap-kapprox2, Oct 3)): `KSHAP_FAST_SIGNGRAM` on kernel-shap / istella, lane/apple-fast-gap-kapprox2 @ 4fd464a43, A/B kap2-kshap-sign-istella (A = BATCH), -1.4% ms, **DROPPED**: noise; sign adds instead of soft-f64 products in the normal equations (same words); not merged

### MOJOLEARN_PREP3_LABELS

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-prep3 @ ec65873e3`.
- EXPERIMENTS.md:372 (Prep (42)): `PREP3_LABELS` on label-binarizer / taxi, lane/apple-fast-prep3 @ ec65873e3, A/B prep3-lb-taxi-x (M2); M3 re-check, M2 +1.0%; M3 -1.9% ms, **DROPPED-noise**: under 5% at n=1 on the M3, signs mixed with the M2; never merged to main or lane/apple-fast-m2b1 (code only on its lane branch)

### MOJOLEARN_PREP3_SPLINE

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-prep3 @ ec65873e3`.
- EXPERIMENTS.md:374 (Prep (42)): `PREP3_SPLINE` on spline / istella, lane/apple-fast-prep3 @ ec65873e3, A/B prep3-spline-istella, spline istella 11.8 -> 11.6 ms, **DROPPED-noise**: rec-misc 2026-10-04: -1.7% at n=1, under 5%; not merged

### MOJOLEARN_RESAMPLE_FAST_IDX_BULK

- Verdict: DROPPED-semantics. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-resample @ 50b96e795`.
- EXPERIMENTS.md:383 (Prep (42)): `RESAMPLE_FAST_IDX_BULK` on resample / taxi, lane/apple-fast-resample @ 50b96e795, A/B resample-rs-idxbulk-taxi, - ms, **DROPPED-semantics**: never compiled (parse error estimator.mojo:2096, a parameter named `out`); duplicate of main's KEPT `RESAMPLE_FAST_IDX_DIRECT` (gmp-rs-idx-*, same one-copy into the caller's buffer). Not ported

### MOJOLEARN_RESAMPLE_FAST_TAKE

- Verdict: DROPPED-semantics. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-s-shap (never pushed)`.
- EXPERIMENTS.md:1388 (Speed round 2: SHAP / manifold / resample (lane/apple-fast-s): `RESAMPLE_FAST_TAKE` on resample / istella, lane/apple-fast-s-shap (never pushed), A/B -, - ms, **DROPPED-semantics**: numpy.take with one intp conversion for the host gather: refused by tools/hooks/no_host_routes.py (new numpy compute in GPU-path Python); not kept

### MOJOLEARN_SHAP_FAST_PIPE

- Verdict: DROPPED-speed. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-w2-shap abd933572`.
- EXPERIMENTS.md:881 (Manager verdicts, 2026-10-04 session 2 (rejected or held; ca): `MOJOLEARN_SHAP_FAST_PIPE` on permutation-shap / kernel-shap istella, lane/apple-fast-w2-shap abd933572, A/B w2-shap-pipe-*-r1, quality PASS (phi byte-identical); pshap 28217.4 -> 28271.8 ms, kshap 15418.1 -> 15446.0 ms ms, **DROP-speed (no overlap gained), opt-in only**: 

### MOJOLEARN_XPREP_DEVICE_CODES

- Verdict: slower. Deleted 2026-10-09 by `08b1152e6` (postmerge-act-2: delete fg-knn-nb G5 (MOJOLEARN_XPREP_DEVICE_CODES; post-merge A/B nv n0668->n0631, amd a1066->a1091: gaussian-nb istella NV 1.29x / AMD 1.12x, ).
- Recoverable at `0c5b780af` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_XPREP_DEVICE_CODES.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `bindings/_mojolearn_x_prep.mojo`, `python/mojolearn/_expansion_prep.py`, `x_prep/device.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:226): removed 2026-10-09 (post-merge A/B nv n0668->n0631, amd a1066->a1091, lane/postmerge-act-2): fg-knn-nb G5 (GaussianNB.fit label codes from the device unique_inverse) was SLOWER: gaussian-nb istella NV 1.29x / AMD 1.12x, taxi NV 1.70x / AMD 1.35x, accuracy SAME, same digests; probe and Python route deleted; code at main 5c137b55e; see EXPERIMENTS.md
- grid_controls/fg-knn-nb.json removed (control xprep_device_codes): DELETED 2026-10-09 (lane postmerge-act-2): post-merge A/B on main (one run per arm) found G5 (GaussianNB.fit label codes from the device unique_inverse) SLOWER: default -> on ms gaussian-nb istella NV 92.1 -> 119.1 (1.29x) / AMD 46.4 -> 52.0 (1.12x), taxi NV 20.0 -> 34.0 (1.70x) / AMD 14.1 -> 19.1 (1.35x) (nv n0668 -> n0631, amd a1066 -> a1091); accuracy SAME, same digests. The x_prep_device_codes probe and the Python route (_fit_device_codes) are deleted; the define is refused in core/six_lane_experiment_guards.mojo; code recoverable at main 5c137b55e.
- EXPERIMENTS.md:1645 (Gap plan section 11: smaller family members (lane/gap-small-): `MOJOLEARN_XPREP_DEVICE_CODES` on expanded:gaussian-nb / istella, taxi (NV + AMD), lane/fg-knn-nb @ 6dc529c28, A/B post-merge (nv n0631, amd a1091), see lane/postmerge-act-2 section ms, **DELETED (loser, lane/postmerge-act-2)**: G5: fit classes and codes from the base binding's device unique_inverse instead of the host native sort (int/float labels); same classes and codes. Code-only lane (flagship gaps 2026-10-09, docs/plans/flagship-gaps-20261009/read_knn_nb.md)
- EXPERIMENTS.md:1765 (Post-merge A/B races: SGD epoch kernel, x_prep device codes ): `MOJOLEARN_XPREP_DEVICE_CODES` on expanded:gaussian-nb / istella, taxi, lane/postmerge-act-2 from main @ 5c137b55e, A/B post-merge (nv n0668 -> n0631, amd a1066 -> a1091), istella NV 92.1 -> 119.1 (1.29x), AMD 46.4 -> 52.0 (1.12x); taxi NV 20.0 -> 34.0 (1.70x), AMD 14.1 -> 19.1 (1.35x) ms, **DELETED (loser)**: accuracy SAME (0.87657 / 0.71982), same digests; the native host encoder (encode_labels) stays the fit route

### MOJOLEARN_XPREP_NO_SLOT_HOP

- Verdict: slower. Deleted 2026-10-09 by `dbfe642a7` (postmerge-act-2: delete fg-knn-nb G3 (MOJOLEARN_XPREP_NO_SLOT_HOP; post-merge A/B nv n0668->n0630, amd a1066->a1090: gaussian-nb istella NV 1.25x / AMD 1.00x, t).
- Recoverable at `08b1152e6` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_XPREP_NO_SLOT_HOP.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `bindings/_mojolearn_x_prep.mojo`, `core/arena_io.mojo`, `core/device_store.mojo`, `python/mojolearn/_arena_io.py`, `python/mojolearn/_expansion_prep.py`, `x_prep/device.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:227): removed 2026-10-09 (post-merge A/B nv n0668->n0630, amd a1066->a1090, lane/postmerge-act-2): fg-knn-nb G3 (large direct inputs as host spans straight into the x_prep arena) was SLOWER on the average: gaussian-nb istella NV 1.25x / AMD 1.00x, taxi NV 1.02x / AMD 1.03x, accuracy SAME, same digests; x_prep_run_ranges_host, upload_ranges_host and the _Prog.run glue deleted; code at main 5c137b55e; see EXPERIMENTS.md
- grid_controls/fg-knn-nb.json removed (control xprep_no_slot_hop): DELETED 2026-10-09 (lane postmerge-act-2): post-merge A/B on main (one run per arm) found G3 (large direct inputs as host spans straight into the arena) slower on the average: default -> on ms gaussian-nb istella NV 92.1 -> 114.7 (1.25x) / AMD 46.4 -> 46.3 (1.00x), taxi NV 20.0 -> 20.4 (1.02x) / AMD 14.1 -> 14.5 (1.03x) (nv n0668 -> n0630, amd a1066 -> a1090); accuracy SAME, same digests. x_prep_run_ranges_host, run_program_device_ranges_host, core/arena_io.mojo upload_ranges_host, DeviceStore.stage_upload and the _Prog.run glue are deleted; the define is refused in core/six_lane_experiment_guards.mojo; code recoverable at main 5c137b55e.
- EXPERIMENTS.md:1644 (Gap plan section 11: smaller family members (lane/gap-small-): `MOJOLEARN_XPREP_NO_SLOT_HOP` on expanded:gaussian-nb / istella, taxi (NV + AMD), lane/fg-knn-nb @ 6dc529c28, A/B post-merge (nv n0630, amd a1090), see lane/postmerge-act-2 section ms, **DELETED (loser on the average, lane/postmerge-act-2)**: G3: a large direct input goes host -> arena through the pinned stage (no slot, no D2D copy, no second X allocation); same words, same offsets. Code-only lane (flagship gaps 2026-10-09, docs/plans/flagship-gaps-20261009/read_knn_nb.md)
- EXPERIMENTS.md:1766 (Post-merge A/B races: SGD epoch kernel, x_prep device codes ): `MOJOLEARN_XPREP_NO_SLOT_HOP` on expanded:gaussian-nb / istella, taxi, lane/postmerge-act-2 from main @ 5c137b55e, A/B post-merge (nv n0668 -> n0630, amd a1066 -> a1090), istella NV 92.1 -> 114.7 (1.25x), AMD 46.4 -> 46.3 (1.00x); taxi NV 20.0 -> 20.4 (1.02x), AMD 14.1 -> 14.5 (1.03x) ms, **DELETED (loser on the average)**: accuracy SAME, same digests; large direct inputs keep the store-slot route (MOJOLEARN_XPREP_DIRECT)

### MOJOLEARN_XPREP_PINNED_UPLOAD

- Verdict: slower. Deleted 2026-10-09 by `fd7e86b46` (postmerge-act-1: delete the G1 DeviceStore pinned upload stage (MOJOLEARN_XPREP_PINNED_UPLOAD; post-merge A/B nv n0607->n0628, amd a1066->a1088: the pageable co).
- Recoverable at `3416a80d8` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_XPREP_PINNED_UPLOAD.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `core/arena_io.mojo`, `core/device_store.mojo`, `x_prep/device.mojo`
- What happened: G1 DeviceStore pinned upload stage (the on-define of XPREP_PINNED_UPLOAD_OFF); see that entry.
- Guard refusal (core/six_lane_experiment_guards.mojo:223): removed 2026-10-09 (post-merge A/B nv n0607->n0628, amd a1066->a1088, lane/postmerge-act-1): the G1 DeviceStore pinned upload stage (2 x 32 MB) was SLOWER than the pageable copy on both vendors: gaussian-nb istella NV 129.8 -> 88.7 ms with _OFF (0.68x), AMD 46.4 -> 27.9 (0.60x); taxi NV 20.9 -> 19.3 (0.92x), AMD 14.1 -> 12.9 (0.91x); same bits; the direct copy is the only path; code at main 432d6e8ff; see docs/apple-fast/EXPERIMENTS.md

### MOJOLEARN_XPREP_PINNED_UPLOAD_OFF

- Verdict: slower. Deleted 2026-10-09 by `fd7e86b46` (postmerge-act-1: delete the G1 DeviceStore pinned upload stage (MOJOLEARN_XPREP_PINNED_UPLOAD; post-merge A/B nv n0607->n0628, amd a1066->a1088: the pageable co).
- Recoverable at `3416a80d8` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_XPREP_PINNED_UPLOAD.patch` (same patch as above).
- Files the patch restores: `core/arena_io.mojo`, `core/device_store.mojo`, `x_prep/device.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:223): removed 2026-10-09 (post-merge A/B nv n0607->n0628, amd a1066->a1088, lane/postmerge-act-1): the G1 DeviceStore pinned upload stage (2 x 32 MB) was SLOWER than the pageable copy on both vendors: gaussian-nb istella NV 129.8 -> 88.7 ms with _OFF (0.68x), AMD 46.4 -> 27.9 (0.60x); taxi NV 20.9 -> 19.3 (0.92x), AMD 14.1 -> 12.9 (0.91x); same bits; the direct copy is the only path; code at main 432d6e8ff; see docs/apple-fast/EXPERIMENTS.md
- grid_controls/fg-knn-nb.json removed (control xprep_pinned_upload_off): DELETED 2026-10-09 (lane postmerge-act-1): post-merge A/B (one run per arm) found the G1 pinned upload stage SLOWER than the pageable copy it replaced; default -> _OFF ms gaussian-nb istella NV 129.8 -> 88.7 (0.68x) / AMD 46.4 -> 27.9 (0.60x), taxi NV 20.9 -> 19.3 (0.92x) / AMD 14.1 -> 12.9 (0.91x) (nv n0607 -> n0628, amd a1066 -> a1088); same digests. The stage code (core/device_store.mojo STORE_PINNED_UPLOAD) is deleted, the direct copy is the only path; both MOJOLEARN_XPREP_PINNED_UPLOAD_OFF and MOJOLEARN_XPREP_PINNED_UPLOAD are refused in core/six_lane_experiment_guards.mojo:223; code recoverable at main 432d6e8ff.
- EXPERIMENTS.md:1756 (Post-merge A/B races: IVF-PQ device codebooks promoted, pinn): `MOJOLEARN_XPREP_PINNED_UPLOAD_OFF` on expanded:gaussian-nb / istella, taxi (every DeviceStore binding), lane/postmerge-act-1 from main @ 432d6e8ff, A/B post-merge (nv n0607 -> n0628, amd a1066 -> a1088), default -> `_OFF`: istella NV 129.8 -> 88.7 (0.68x), AMD 46.4 -> 27.9 (0.60x); taxi NV 20.9 -> 19.3 (0.92x), AMD 14.1 -> 12.9 (0.91x) ms, **REVERTED + DELETED (loser; the old pageable copy is faster on both vendors)**: the host memcpy into the stage plus the pinned allocation per store cost more than the DMA gain; same digests (2eab934c8d istella, 6c24b800ea taxi); no bits change; code recoverable at main 432d6e8ff

### MOJOLEARN_X_PREP_FAST_NONEG

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `10a9ab8eb` (x_prep encoders: remove dropped X_PREP_FAST_NONEG env switch (DROPPED-noise; recover lane/apple-fast-prep@a11e43a5e)).
- Recoverable at `99da8a08f` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_X_PREP_FAST_NONEG.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `python/mojolearn/_expansion_prep.py`
- EXPERIMENTS.md:360 (Prep (42)): `X_PREP_FAST_NONEG` on -, lane/apple-fast-prep @ a11e43a5e, A/B prep-onehot-noneg-taxi, prep-ordinal-noneg-taxi, onehot -2.1%; ordinal -0.1% ms, **DROPPED-noise**: <5%; code removed from main 10a9ab8eb; recover at lane/apple-fast-prep@a11e43a5e

### MOJOLEARN_X_PREP_PINNED_OUT

- Verdict: DROPPED. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-w3-prep 92e4661e6`.
- EXPERIMENTS.md:899 (PT centered-score WIP checkpoint (2026-10-04)): `MOJOLEARN_X_PREP_PINNED_OUT` on label-binarizer / multilabel-binarizer / target-encoder taxi, lane/apple-fast-w3-prep 92e4661e6, A/B w2-pinned-*-r1, outputs sha-identical; timed call lb taxi 270.2 -> 64.5, mlb taxi 169.7 -> 87.7, te 201.6 -> 201.6 ms; BUT caller first read of the returned pinned array: lb taxi shape A 331.5 ms vs B 2153.2 ms, mlb 256.0 vs 1680.7 ms (call+read A ~604 ms vs B ~2199 ms) ms, **DROP: moves the cost to the caller's read (write-combined pinned memory); benchmark-only gain**: 


## Decomp

### MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE

- Verdict: slower. Deleted 2026-10-08 by `ab4e8e543` (grid-losers-1: delete C25_PROJECTION_REUSE (grid ge123e6f9 noise on nystroem/rbf-sampler); refuse the define).
- Recoverable at `ad7ed2370` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `experiments/classical_identical_ideas/linear_controls.mojo`, `kernel_methods/estimator.mojo`, `kernel_methods/rbf_fused.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:181): removed: MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE retired 2026-10-08: noise, NV 1.00-1.04x / AMD 0.88-0.98x on nystroem and rbf-sampler (grid ge123e6f9); see EXPERIMENTS.md
- grid_controls/classical-decomp.json removed (control C25_PROJECTION_REUSE): DELETED 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 noise, a loser. on/off ms ratio NV/AMD: nystroem istella 1.007/0.912, taxi 1.002/0.984; rbf-sampler istella 1.040/0.884, taxi 1.007/0.884; kernel_rel_error SAME. Code recoverable at main ad7ed2370; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1606 (IDENTICAL grid ge123e6f9 losers deleted (lane/grid-losers-1,): `MOJOLEARN_CLASSICAL_C25_PROJECTION_REUSE` on more:nystroem, more:rbf-sampler / istella, taxi, main @ ad7ed2370 (deleted on lane/grid-losers-1), A/B ge123e6f9, nystroem istella NV 118.2 -> 119.1 (1.007), AMD 63.5 -> 57.9 (0.912); taxi NV 141.6 -> 141.8 (1.002), AMD 149.8 -> 147.4 (0.984); rbf-sampler istella NV 95.2 -> 98.9 (1.040), AMD 38.1 -> 33.6 (0.884); taxi NV 89.3 -> 89.9 (1.007), AMD 33.9 -> 29.9 (0.884); 0.963x combined ms, **DROP (noise), code deleted**: inside the noise floor on both vendors (NVIDIA flat or slower, AMD gain not outside the floor); kernel_rel_error SAME. Deleted `classical_projection_kernel` (kernel_methods/rbf_fused.mojo) and its launch/route in kernel_methods/estimator.mojo; RBF_IDN_FUSED keeps its own d cut

### MOJOLEARN_CLASSICAL_PCA_COV=23

- Verdict: slower. Deleted 2026-10-08 by `3df299f27` (grid-act-2: delete PCA_COV=23 arm (grid ge123e6f9 pca istella 2.28x/1.27x slower, taxi 0.90x/0.78x; combined 1.195x slower); c04 stays; refuse =23; tombstones).
- Recoverable at `bd14ae97f` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CLASSICAL_PCA_COV-arm23.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `core/blocked_moments.mojo`, `core/blocked_moments_host.mojo`, `decomposition/host/pca_oracle.mojo`, `decomposition/impl/linalg/detail/pca.mojo`, `experiments/classical_identical_ideas/linear_controls.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:201): removed: MOJOLEARN_CLASSICAL_PCA_COV=23 (C23 one-pass covariance) retired 2026-10-08: slower, pca NV 2.28x / AMD 1.27x istella, NV 0.90x / AMD 0.78x taxi, combined 1.195x (grid ge123e6f9); =4 (c04) stays; see EXPERIMENTS.md
- grid_controls/classical-decomp.json removed (control PCA_COV): DELETED 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 loser. c23/off ms ratio NV/AMD on pca: istella 2.281/1.265 (154.0 -> 351.3 NV, 53.1 -> 67.2 AMD), taxi 0.903/0.782; combined 1.195x SLOWER (dimension-dependent: the one-pass d x d leaf Gram loses at wide d; lane grid-flips-1 refused to flip it). Deleted the C23 branch in decomposition/impl/linalg/detail/pca.mojo and pca_oracle.mojo, bm_onepass_covariance (core/blocked_moments.mojo) and its host twin; c04 stays (unmeasured), C23_MCD is separate. Code recoverable at main 42d1e42c6; =23 refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1711 (IDENTICAL grid ge123e6f9: two promotions and eight loser arm): `MOJOLEARN_CLASSICAL_PCA_COV=23` on classical:pca / istella, taxi, main @ 42d1e42c6 (deleted on lane/grid-act-2), A/B ge123e6f9, istella NV 154.0 -> 351.3 (2.281), AMD 53.1 -> 67.2 (1.265); taxi NV 14.9 -> 13.5 (0.903), AMD 8.2 -> 6.4 (0.782); 1.195x combined SLOWER ms, **DROP (slower on the average), code deleted**: one blocked pass with per-leaf centering and Chan merges: faster at narrow d (taxi), much slower at wide d (istella), so dimension-dependent; lane grid-flips-1 refused to flip it and the vendor average is slower. Deleted the C23 branch in decomposition/impl/linalg/detail/pca.mojo and decomposition/host/pca_oracle.mojo, `bm_onepass_covariance` (core/blocked_moments.mojo) and `host_bm_onepass_covariance` (core/blocked_moments_host.mojo); `=23` is refused.

### MOJOLEARN_DECOMP_FAST_SMALL_EIGH_J2

- Verdict: DROPPED-semantics. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-decomp-sparse @ 5fb1740cd`.
- EXPERIMENTS.md:408 (Decomp (34)): `DECOMP_FAST_SMALL_EIGH_J2` on fastica / istella, lane/apple-fast-decomp-sparse @ 5fb1740cd, A/B dsp-ica-istella, - ms, **DROPPED-semantics**: no-op after main removed _eigh2

### MOJOLEARN_EIGH_TANGENT_CACHE

- Verdict: DROPPED-speed. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-eigh-cache 14764dbb8`.
- EXPERIMENTS.md:878 (Manager verdicts, 2026-10-04 session 2 (rejected or held; ca): `MOJOLEARN_EIGH_TANGENT_CACHE` on eigh synthetic, lane/apple-fast-eigh-cache 14764dbb8, A/B gap26-eigh-cache-synthetic-ready, A 43721.3 -> B 44070.7 ms; quality pair PASS (B eigenvalue error 3.5e-7) ms, **DROP-speed, opt-in only**: 

### MOJOLEARN_IDN_PCA_RR_ONE_BLOCK

- Verdict: broken. Deleted 2026-10-09 by `16a2c0dd3` (postmerge-act-5: delete IDN_PCA_RR_ONE_BLOCK (broken on NVIDIA, nv2 v1030), tombstones).
- Recoverable at `a837c5d08` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_PCA_RR_ONE_BLOCK.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `decomposition/impl/linalg/detail/pca.mojo`, `decomposition/pca_rr_switch.mojo`, `x_decomp/rr.mojo`, `x_decomp/rr_one_block.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:238): removed 2026-10-09 (fg2 A/B nv2 v1030, lane/postmerge-act-5): fg-pca P1+P1b (one-launch one-block round-robin Jacobi, x_decomp/rr_one_block.mojo) was BROKEN on NVIDIA: pca and tsvd on taxi and istella REFUSED at decomposition/impl/linalg/detail/pca.mojo:734; code at main 0a7b206f1; see EXPERIMENTS.md
- grid_controls/fg-pca.json removed (control pca_rr_one_block): DELETED 2026-10-09 (lane postmerge-act-5): fg2 board-bridge A/B on main 0a7b206f1 found P1+P1b BROKEN on NVIDIA (nv2 v1030): pca and tsvd on taxi and istella REFUSED at decomposition/impl/linalg/detail/pca.mojo:734:55. x_decomp/rr_one_block.mojo and the pca.mojo launch deleted; define (and _STEPS) refused in core/six_lane_experiment_guards.mojo; code recoverable at main 0a7b206f1.
- EXPERIMENTS.md:1672 (Flagship gaps: PCA / TruncatedSVD / randomized_svd (lane/fg-): `MOJOLEARN_IDN_PCA_RR_ONE_BLOCK` on classical:pca, more:tsvd / istella, taxi, lane/fg-pca @ 385b276e8, A/B owed, owed ms, **DROPPED 2026-10-09 (broken on NVIDIA;see lane/postmerge-act-5 section)**: P1+P1b: the whole round-robin solve in one launch of one block (fused cs+update, in-kernel test and gate); same cells and order, no bit claimed to move
- EXPERIMENTS.md:1814 (Post-merge fg2 A/B: round-robin Jacobi and float-float Gram ): `MOJOLEARN_IDN_PCA_RR_ONE_BLOCK` on classical:pca, more:tsvd / istella, taxi, lane/fg-pca @ 385b276e8, on main @ 0a7b206f1, A/B fg2 (nv2 v1030), REFUSED on NVIDIA (no time) ms, **DROPPED**: BROKEN at runtime on NVIDIA: pca and tsvd on taxi and istella REFUSED with "decomposition/impl/linalg/detail/pca.mojo:734:55 failed calling ...". x_decomp/rr_one_block.mojo and the pca.mojo launch deleted; define refused; recoverable at main 0a7b206f1.

### MOJOLEARN_IDN_PCA_RR_ONE_BLOCK_STEPS

- Verdict: broken. Deleted 2026-10-09 by `16a2c0dd3` (postmerge-act-5: delete IDN_PCA_RR_ONE_BLOCK (broken on NVIDIA, nv2 v1030), tombstones).
- Recoverable at `a837c5d08` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_PCA_RR_ONE_BLOCK.patch` (same patch as above).
- Files the patch restores: `decomposition/impl/linalg/detail/pca.mojo`, `decomposition/pca_rr_switch.mojo`, `x_decomp/rr.mojo`, `x_decomp/rr_one_block.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:238): removed 2026-10-09 (fg2 A/B nv2 v1030, lane/postmerge-act-5): fg-pca P1+P1b (one-launch one-block round-robin Jacobi, x_decomp/rr_one_block.mojo) was BROKEN on NVIDIA: pca and tsvd on taxi and istella REFUSED at decomposition/impl/linalg/detail/pca.mojo:734; code at main 0a7b206f1; see EXPERIMENTS.md
- EXPERIMENTS.md:1672 (Flagship gaps: PCA / TruncatedSVD / randomized_svd (lane/fg-): `MOJOLEARN_IDN_PCA_RR_ONE_BLOCK_STEPS` on classical:pca, more:tsvd / istella, taxi, lane/fg-pca @ 385b276e8, A/B owed, owed ms, **DROPPED 2026-10-09 (broken on NVIDIA;see lane/postmerge-act-5 section)**: P1+P1b: the whole round-robin solve in one launch of one block (fused cs+update, in-kernel test and gate); same cells and order, no bit claimed to move

### MOJOLEARN_LU_FAST_TSLU

- Verdict: DROPPED-quality. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-w4-tslu 5867b9fbe`.
- EXPERIMENTS.md:900 (PT centered-score WIP checkpoint (2026-10-04)): `MOJOLEARN_LU_FAST_TSLU` on lu-factor / lu-solve, lane/apple-fast-w4-tslu 5867b9fbe, A/B w2-tslu-quality, gate PASS but hard matrices worse: plain1000 factor 2.54e-6 -> 3.12e-6, solve 1.20e-3 -> 1.66e-3; plain2051 factor 5.36e-6 -> 6.27e-6, solve 1.81e-3 -> 2.51e-3; board identical (no pivoting) ms, **DROP-quality (tournament pivoting less stable than partial pivoting); not promoted regardless of speed**: 

### MOJOLEARN_PCA_FAST_EIG

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-pca-eig @ 9819970b1`.
- EXPERIMENTS.md:410 (Decomp (34)): `PCA_FAST_EIG` on pca / tsvd / ipca / istella, taxi, lane/apple-fast-pca-eig @ 9819970b1, A/B pca-eig-eig-istella, pca-eig-eig-taxi, pca-eig-tsvd-eig-istella (+2), pca istella -2.2% / taxi +17%; tsvd 0%; ipca +0.4% / -1.2% ms, **DROPPED-noise**: mixed signs across datasets
- EXPERIMENTS.md:411 (Decomp (34)): `PCA_FAST_EIG,PCA_FAST_NO_ALIAS,PCA_FAST_TOPK` on pca / tsvd / ipca / istella, lane/apple-fast-pca-eig @ 9819970b1, A/B pca-eig-all-istella, pca istella -0.7% ms, **DROPPED-noise**: noise

### MOJOLEARN_PCA_FAST_NO_ALIAS

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-pca-eig @ 9819970b1`.
- EXPERIMENTS.md:411 (Decomp (34)): `PCA_FAST_EIG,PCA_FAST_NO_ALIAS,PCA_FAST_TOPK` on pca / tsvd / ipca / istella, lane/apple-fast-pca-eig @ 9819970b1, A/B pca-eig-all-istella, pca istella -0.7% ms, **DROPPED-noise**: noise
- EXPERIMENTS.md:412 (Decomp (34)): `PCA_FAST_NO_ALIAS` on pca / tsvd / ipca / istella, taxi, lane/apple-fast-pca-eig @ 9819970b1, A/B pca-eig-noalias-istella, pca-eig-noalias-taxi, pca istella +2.1% / taxi -8.5% ms, **DROPPED-noise**: mixed signs

### MOJOLEARN_PCA_FAST_TOPK

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-pca-eig @ 9819970b1`.
- EXPERIMENTS.md:411 (Decomp (34)): `PCA_FAST_EIG,PCA_FAST_NO_ALIAS,PCA_FAST_TOPK` on pca / tsvd / ipca / istella, lane/apple-fast-pca-eig @ 9819970b1, A/B pca-eig-all-istella, pca istella -0.7% ms, **DROPPED-noise**: noise
- EXPERIMENTS.md:413 (Decomp (34)): `PCA_FAST_TOPK` on pca / tsvd / ipca / istella, taxi, lane/apple-fast-pca-eig @ 9819970b1, A/B pca-eig-topk-istella, pca-eig-topk-taxi, pca-eig-tsvd-topk-istella, pca istella +1.5% / taxi -5.9%; tsvd +0.5% ms, **DROPPED-noise**: mixed signs

### MOJOLEARN_TSNE_FAST_SPLIT

- Verdict: DROPPED-quality. Deleted 2026-09-28 by `2c209a63f` (t-SNE device driver back to lane/apple-merged's (7483efa40): fixes the M3 Ultra iteration regression).
- Recoverable at `e3e17f59e` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_TSNE_FAST_SPLIT.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `x_ann/tsne_device.mojo`
- EXPERIMENTS.md:341 (Neighbors (42)): `TSNE_FAST_SPLIT` on tsne / istella, lane/apple-fast-ann @ 70833546a, A/B ann-tsne-split-istella, - ms, **DROPPED-quality**: reconciled 2026-10-05: ann-tsne-split-istella-b 2540.9 -> 2318.7 (-8.7%), trustworthiness .9923 -> .9921 (LEDGER 2026-10-03); deleted at merge 8d43ec357. Was OPEN: A/B queued, no judged result yet


### MOJOLEARN_CHOL_FAST_BLOCKED

- Verdict: DROPPED-slower+quality. Deleted 2026-10-09 by lane/owed-deletions-D3 (owed deletion, D3).
- Recoverable at `b639a2bd2` (main the lane branched from). Patch: `experiments/removed/MOJOLEARN_CHOL_FAST_BLOCKED.patch` (reverse of the lane's deletion commit; applies to the lane head).
- What it tried: a blocked right-looking Cholesky (x_decomp/fast_chol.mojo: diagonal block, panel TRSM, trailing SYRK; 3 n / 32 launches) for potrf_lower's defer_ok callers and x_decomp's chol / CholeskyQR Gram factor, FAST + Apple.
- Files the patch restores: `cholesky/checks/potrf.mojo`, `x_decomp/device.mojo`, `x_decomp/fast_chol.mojo`
- EXPERIMENTS.md:422: `CHOL_FAST_BLOCKED` | cholesky / synthetic | lane/apple-fast-decomp-linalg @ 74d52352b -> lane/apple-fast-rec-decomp | dlin-chol-blocked-synthetic | old-head B 476 vs board FAST 261 (no same-build A) | DROPPED-slower+quality 2026-10-04 (see verdicts batch 3) | ported to main (x_decomp/fast_chol.mojo, potrf_lower + DevExec.chol); default off; potrf route only for defer_ok callers with CHOL_FAST_NOSYNC on (LAPACK partial factor kept via the redo); awaiting M2 build + M3 A/B
- EXPERIMENTS.md:1241: `CHOL_FAST_BLOCKED` | cholesky / synthetic | lane/apple-fast-rec-ab3 @ 0ca521cc5 | afc_ab_def | 259.4 -> 330.7 | DROPPED-slower+quality | relative_residual 1.66e-7 -> 1.98e-6 (M3, full board, 1 run per arm, 2026-10-04); stays off
- Guard refusal (core/six_lane_experiment_guards.mojo): removed 2026-10-09 (lane/owed-deletions-D3): CHOL_FAST_BLOCKED (blocked right-looking Cholesky, x_decomp/fast_chol.mojo) was SLOWER and less accurate: cholesky synthetic 259.4 -> 330.7 ms, relative_residual 1.66e-7 -> 1.98e-6 (M3, 1 run per arm); code at main b639a2bd2; see docs/TOMBSTONES.md

### MOJOLEARN_DECOMP_FAST_GEMM_TILED

- Verdict: DROPPED-slower. Deleted 2026-10-09 by lane/owed-deletions-D3 (owed deletion, D3).
- Recoverable at `b639a2bd2` (main the lane branched from). Patch: `experiments/removed/MOJOLEARN_DECOMP_FAST_GEMM_TILED.patch` (reverse of this define's deletion commit on the lane; when a later deletion touched the same lines, use `git apply -3`).
- What it tried: x_decomp launch_gemm as a threadgroup-tiled kernel (32 x 32 output tile, 16-deep K slab, 2 x 2 micro-tiles, FOLD_BLOCK partials kept), FAST + Apple. The never-run AFCL-L08 knob (K slab 32, -D MOJOLEARN_AFCL_L08, which required this define in both arms) lived in the same file and is deleted with it.
- Files the patch restores: `x_decomp/device.mojo`, `x_decomp/fast_gemm.mojo`
- EXPERIMENTS.md:425: `DECOMP_FAST_GEMM_TILED` | als / taxi-zones; lstsq / istella; nmf / istella; randomized-svd / istella | lane/apple-fast-decomp-linalg @ 74d52352b -> lane/apple-fast-rec-decomp | dlin-lstsq-tiled-istella, dlin-rsvd-tiled-istella, dlin-nmf-tiled-istella, dlin-als-tiled-taxizones | old-head B rsvd istella 683 vs board FAST 533 (no same-build A) | DROPPED-slower | reconciled 2026-10-05: rab3-gemmtiled randomized-svd istella 483.56 -> 640.22 (+32.4%), reconstruction error equal (ab_all_latest.txt); stays off; lstsq/nmf/als callers not timed. Was READY-AB: ported to main (x_decomp/fast_gemm.mojo, launch_gemm ahead of DECOMP_FAST_GEMM_MMA, so the A/B is tiled vs MMA); default off; awaiting M2 build
- Guard refusal (core/six_lane_experiment_guards.mojo): removed 2026-10-09 (lane/owed-deletions-D3): DECOMP_FAST_GEMM_TILED (threadgroup-tiled x_decomp launch_gemm, x_decomp/fast_gemm.mojo) was SLOWER: randomized-svd istella 483.56 -> 640.22 ms (+32.4%), reconstruction error equal; its child knob MOJOLEARN_AFCL_L08 went with it; code at main b639a2bd2; see docs/TOMBSTONES.md

### MOJOLEARN_FA_ALL

- Verdict: DROPPED-slower. Deleted 2026-10-09 by lane/owed-deletions-D3 (owed deletion, D3).
- Recoverable at `b639a2bd2` (main the lane branched from). Patch: `experiments/removed/MOJOLEARN_FA_ALL.patch` (reverse of this define's deletion commit on the lane; when a later deletion touched the same lines, use `git apply -3`).
- What it tried: one -D that turned on every FactorAnalysis FAST define (GRAM_ONCE, ITER_DEVICE, EIG_SMALL, LIVEBUF, LL_DEVICE, TRANSFORM_FUSED).
- Files the patch restores: `x_decomp/fa_fast.mojo`
- EXPERIMENTS.md:430: `FA_ALL` | factor-analysis / taxi; istella | lane/apple-fast-fa @ 3efbce2af | fa-all-taxi, fa-all-istella | - | HOLD-quality 2026-10-04 (see verdicts batch 3) | ported to lane/apple-fast-rec-fa-robust; every FA define; awaiting M2 build + M3 A/B
- EXPERIMENTS.md:441: `FA_ALL` | factor-analysis / istella; factor-analysis / taxi | lane/apple-fast-fa @ 3efbce2af | fa-all-taxi, fa-all-istella | - | HOLD-quality 2026-10-04 (see verdicts batch 3) | A/B queued (lane/apple-fast-batch prebuilt arms)
- EXPERIMENTS.md:1244: `FA_ALL` | factor-analysis / istella; taxi | lane/apple-fast-rec-ab3 @ 0ca521cc5 | afc_ab_def | istella 10.4 s -> 9.54 s; taxi 358 -> 30.4 | HOLD-quality | same istella log-likelihood loss (M3, full board, 1 run per arm, 2026-10-04); stays off
- EXPERIMENTS.md:1264: `FA_ALL` (with FA_GRAM_DF) | factor-analysis / istella, taxi | lane/apple-fast-fa-quality | rab6-faqfix | istella 10300.89 -> 20530.81 (+99.3%); taxi 345.06 -> 34.19 | DROPPED-slower: stays off | istella slower (EIG_SMALL one-threadgroup eigh); quality noise
- Guard refusal (core/six_lane_experiment_guards.mojo): removed 2026-10-09 (lane/owed-deletions-D3): FA_ALL (every FactorAnalysis FAST define at once) was SLOWER: factor-analysis istella 10300.89 -> 20530.81 ms (+99.3%, rab6-faqfix), taxi 345.06 -> 34.19, quality noise; the slowdown is EIG_SMALL's one-threadgroup eigh; code at main b639a2bd2; see docs/TOMBSTONES.md

### MOJOLEARN_FA_EIG_SMALL

- Verdict: DROPPED-slower. Deleted 2026-10-09 by lane/owed-deletions-D3 (owed deletion, D3).
- Recoverable at `b639a2bd2` (main the lane branched from). Patch: `experiments/removed/MOJOLEARN_FA_EIG_SMALL.patch` (reverse of this define's deletion commit on the lane; when a later deletion touched the same lines, use `git apply -3`).
- What it tried: the FactorAnalysis EM loop's d x d eigh (fa_rr_eigh_block_kernel) and, under FA_GRAM_DF, its SVD (fa_rs_svd_block_kernel) as ONE launch of one threadgroup instead of the grid kernels and a sync per sweep.
- Files the patch restores: `x_decomp/fa_fast.mojo`
- EXPERIMENTS.md:431: `FA_EIG_SMALL + FA_ITER_DEVICE` | factor-analysis / taxi; istella | lane/apple-fast-fa @ 3efbce2af | fa-eig-taxi, fa-eig-istella | taxi 39.5, istella 5884 (board 308 / 10351) | DROPPED-slower | reconciled 2026-10-05: never timed alone on current main; FA_ALL (includes EIG_SMALL) rab6-faqfix istella 10300.89 -> 20530.81 (+99.3%), slowdown from the one-threadgroup EIG_SMALL eigh (Verdicts batch 4 FA_ALL row; x_decomp/fa_fast.mojo `#:`); FA_ITER_DEVICE + FA_GRAM_DF is the default (rab7-faiterfix). Was READY-AB: ported to lane/apple-fast-rec-fa-robust (x_decomp/fa_fast.mojo) with main's two-pass mean and cancellation-free psi; EIG_SMALL implies ITER_DEVICE; awaiting M2 build + M3 A/B
- EXPERIMENTS.md:442: `FA_EIG_SMALL + FA_ITER_DEVICE` | factor-analysis / istella; factor-analysis / taxi | lane/apple-fast-fa @ 3efbce2af | fa-eig-taxi, fa-eig-istella | - | DROPPED-slower | reconciled 2026-10-05: never timed alone on current main; FA_ALL (includes EIG_SMALL) rab6-faqfix istella 10300.89 -> 20530.81 (+99.3%), slowdown from the one-threadgroup EIG_SMALL eigh (Verdicts batch 4 FA_ALL row; x_decomp/fa_fast.mojo `#:`); FA_ITER_DEVICE + FA_GRAM_DF is the default (rab7-faiterfix). Was OPEN: A/B queued (lane/apple-fast-batch prebuilt arms)
- Guard refusal (core/six_lane_experiment_guards.mojo): removed 2026-10-09 (lane/owed-deletions-D3): FA_EIG_SMALL (one-threadgroup eigh / SVD inside the FactorAnalysis EM loop) was SLOWER: in FA_ALL factor-analysis istella 10300.89 -> 20530.81 ms (+99.3%), the slowdown from this one-threadgroup eigh on a 220 x 220; code at main b639a2bd2; see docs/TOMBSTONES.md

### MOJOLEARN_FA_LL_DEVICE

- Verdict: DROPPED-slower. Deleted 2026-10-09 by lane/owed-deletions-D3 (owed deletion, D3).
- Recoverable at `b639a2bd2` (main the lane branched from). Patch: `experiments/removed/MOJOLEARN_FA_LL_DEVICE.patch` (reverse of this define's deletion commit on the lane; when a later deletion touched the same lines, use `git apply -3`).
- What it tried: the FactorAnalysis EM convergence test on the device: fa_finish_kernel summed the 2 d log terms in double-float float32, tested (ll - old_ll) < tol and set a flag later kernels check; the host read the flag every 4 iterations and the ll pairs once at the end.
- Files the patch restores: `x_decomp/fa_fast.mojo`
- EXPERIMENTS.md:432: `FA_EIG_SMALL + FA_ITER_DEVICE + FA_LL_DEVICE` | factor-analysis / taxi; istella | lane/apple-fast-fa @ 3efbce2af | fa-lldev-taxi, fa-lldev-istella | taxi 30.2, istella 5879 | DROPPED-slower | reconciled 2026-10-05: contains EIG_SMALL: same evidence as the EIG_SMALL + ITER_DEVICE row (rab6-faqfix FA_ALL +99.3% istella); LL_DEVICE alone on the default loop not timed. Was READY-AB: ported to lane/apple-fast-rec-fa-robust; awaiting M2 build + M3 A/B
- EXPERIMENTS.md:443: `FA_EIG_SMALL + FA_ITER_DEVICE + FA_LL_DEVICE` | factor-analysis / istella; factor-analysis / taxi | lane/apple-fast-fa @ 3efbce2af | fa-lldev-taxi, fa-lldev-istella | - | DROPPED-slower | reconciled 2026-10-05: contains EIG_SMALL: same evidence as the EIG_SMALL + ITER_DEVICE row (rab6-faqfix FA_ALL +99.3% istella); LL_DEVICE alone on the default loop not timed. Was OPEN: A/B queued (lane/apple-fast-batch prebuilt arms)
- Guard refusal (core/six_lane_experiment_guards.mojo): removed 2026-10-09 (lane/owed-deletions-D3): FA_LL_DEVICE (FactorAnalysis EM convergence test on the device) was DROPPED-slower with its EIG_SMALL arm (FA_ALL istella +99.3%, rab6-faqfix); never timed alone on the default loop; code at main b639a2bd2; see docs/TOMBSTONES.md

## Cluster

### MOJOLEARN_AFFINITY_FAST_LOOP

- Verdict: DROPPED-noise. Deleted 2026-10-02 by `c5d0ef805` (x_cluster cluster2: the ap_loop comment names the define, not the old env switch).
- Recoverable at `50cb0d970` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_AFFINITY_FAST_LOOP.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `x_cluster/device_ops.mojo`
- EXPERIMENTS.md:478 (Cluster (38)): `AFFINITY_FAST_LOOP` on affinity-prop / istella, lane/apple-fast-cluster2 @ ded4ea07b, A/B cluster2-ap-loop-istella, - ms, **DROPPED-noise**: reconciled 2026-10-05: affinity-prop istella -6%, n=1, "marginal" (~/mojolearn-evidence/apple-fast/LEDGER.md 2026-10-03; ~/mojolearn-evidence/apple-fast/EXPERIMENTS.md); code deleted at cluster2 merge deee07721. Was OPEN: A/B queued, no judged result yet

### MOJOLEARN_BGMM_ENT

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `31b0cf94c` (bgmm: the k-sized float64 host updates move onto the device (float-float workspace, one scalar block per iteration); default priors on the device).
- Recoverable at `266bf8f33` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_BGMM_ENT.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `x_cluster/bgmm.mojo`
- EXPERIMENTS.md:481 (Cluster (38)): `BGMM_ENT` on bayesian-gmm / taxi, lane/apple-fast-cluster2 @ ded4ea07b, A/B cluster2-bgmm-ent-taxi, - ms, **DROPPED-noise**: reconciled 2026-10-05: cluster2-bgmm-ent-taxi-b 309.8 -> 310.5 (+0.2%), mean_log_likelihood same (LEDGER 2026-10-03); deleted at deee07721. Was OPEN: A/B queued, no judged result yet

### MOJOLEARN_BISECT_FAST_RESIDENT

- Verdict: DROPPED-slower. Deleted 2026-10-02 by `50cb0d970` (x_cluster cluster2: the OPTICS device order as one grid launch per step (no one-block launch)).
- Recoverable at `1e0012165` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_BISECT_FAST_RESIDENT.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `x_cluster/device_ops.mojo`
- EXPERIMENTS.md:485 (Cluster (38)): `BISECT_FAST_RESIDENT` on bisecting-kmeans / istella, lane/apple-fast-cluster2 @ ded4ea07b, A/B cluster2-bisect-resident-istella, - ms, **DROPPED-slower**: reconciled 2026-10-05: cluster2-bisect-resident-istella-b 2013.0 -> 2127.7 (+5.7%) (LEDGER 2026-10-03); deleted at deee07721; main took BISECT_FAST_ZEROCOPY (77da4a641). Was OPEN: A/B queued, no judged result yet

### MOJOLEARN_C37_FUSED_ACCUMULATE

- Verdict: slower. Deleted 2026-10-08 by `72716663a` (grid-act-2: delete c37_fused_accumulate + c37_fused_rows (grid ge123e6f9: kmeans NV 104.8x/68.6x slower, AMD 0.84x/0.90x; combined 9.4x/7.8x slower, inertia sam).
- Recoverable at `e448841ff` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_C37_FUSED_ACCUMULATE.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `cluster/impl/detail/kmeans.mojo`, `cluster/impl/detail/kmeans_fused_accumulate.mojo`, `experiments/classical_identical_ideas/graph_controls.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:195): removed: MOJOLEARN_C37_FUSED_ACCUMULATE (and _FUSED_ROWS) retired 2026-10-08: slower, kmeans NV 104.8x / AMD 0.84x istella, NV 68.6x / AMD 0.90x taxi (vendor split; combined 9.4x / 7.8x), inertia SAME (grid ge123e6f9); see EXPERIMENTS.md
- grid_controls/classical-kmeans.json removed (control c37_fused_accumulate): DELETED 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 loser. on/off ms ratio NV/AMD: kmeans istella 104.781/0.841 (64986 vs 620 ms NV), taxi 68.558/0.895; vendor split (AMD faster), combined 9.4x/7.8x SLOWER; inertia SAME. Code (cluster/impl/detail/kmeans_fused_accumulate.mojo and the Lloyd-loop hooks) recoverable at main 42d1e42c6; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1705 (IDENTICAL grid ge123e6f9: two promotions and eight loser arm): `MOJOLEARN_C37_FUSED_ACCUMULATE` on classical:kmeans / istella, taxi, main @ 42d1e42c6 (deleted on lane/grid-act-2), A/B ge123e6f9, istella NV 620.2 -> 64986.3 (104.781), AMD 541.0 -> 455.0 (0.841); taxi NV 319.0 -> 21870.9 (68.558), AMD 705.2 -> 631.5 (0.895); combined 9.4x / 7.8x SLOWER ms, **DROP (slower), code deleted**: vendor split: AMD faster, NVIDIA collapses (the shared-memory Int32 table pass); the average decides. inertia SAME. Deleted cluster/impl/detail/kmeans_fused_accumulate.mojo, the fused branches of the Lloyd loop in cluster/impl/detail/kmeans.mojo (min_cluster_and_distance_compute + blocked accumulation is the only path) and both defines.

### MOJOLEARN_C37_FUSED_ROWS

- Verdict: slower. Deleted 2026-10-08 by `72716663a` (grid-act-2: delete c37_fused_accumulate + c37_fused_rows (grid ge123e6f9: kmeans NV 104.8x/68.6x slower, AMD 0.84x/0.90x; combined 9.4x/7.8x slower, inertia sam).
- Recoverable at `e448841ff` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_C37_FUSED_ACCUMULATE.patch` (same patch as above).
- Files the patch restores: `cluster/impl/detail/kmeans.mojo`, `cluster/impl/detail/kmeans_fused_accumulate.mojo`, `experiments/classical_identical_ideas/graph_controls.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:195): removed: MOJOLEARN_C37_FUSED_ACCUMULATE (and _FUSED_ROWS) retired 2026-10-08: slower, kmeans NV 104.8x / AMD 0.84x istella, NV 68.6x / AMD 0.90x taxi (vendor split; combined 9.4x / 7.8x), inertia SAME (grid ge123e6f9); see EXPERIMENTS.md
- grid_controls/classical-kmeans.json removed (control c37_fused_rows): DELETED 2026-10-08 (lane grid-act-2) with c37_fused_accumulate (rows per fused block; only read by the deleted fused pass). DELETED 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 loser. on/off ms ratio NV/AMD: kmeans istella 104.781/0.841 (64986 vs 620 ms NV), taxi 68.558/0.895; vendor split (AMD faster), combined 9.4x/7.8x SLOWER; inertia SAME. Code (cluster/impl/detail/kmeans_fused_accumulate.mojo and the Lloyd-loop hooks) recoverable at main 42d1e42c6; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1705 (IDENTICAL grid ge123e6f9: two promotions and eight loser arm): `MOJOLEARN_C37_FUSED_ROWS` on classical:kmeans / istella, taxi, main @ 42d1e42c6 (deleted on lane/grid-act-2), A/B ge123e6f9, istella NV 620.2 -> 64986.3 (104.781), AMD 541.0 -> 455.0 (0.841); taxi NV 319.0 -> 21870.9 (68.558), AMD 705.2 -> 631.5 (0.895); combined 9.4x / 7.8x SLOWER ms, **DROP (slower), code deleted**: vendor split: AMD faster, NVIDIA collapses (the shared-memory Int32 table pass); the average decides. inertia SAME. Deleted cluster/impl/detail/kmeans_fused_accumulate.mojo, the fused branches of the Lloyd loop in cluster/impl/detail/kmeans.mojo (min_cluster_and_distance_compute + blocked accumulation is the only path) and both defines.

### MOJOLEARN_C37_PANEL_128

- Verdict: DROPPED. Deleted 2026-10-07 by `d7c9736f0` (graph_controls: split C30 per family, KMEANS_ASSIGN arms, rewritten C37 fused, C38 split).
- Recoverable at `8be4d20d4` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_C37_PANEL_128.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `experiments/classical_identical_ideas/graph_controls.mojo`
- grid_controls/classical-kmeans.json removed: sub-knob of the deleted C37; the new sweep is MOJOLEARN_C37_FUSED_ROWS
- EXPERIMENTS.md:1499 (Classical IDENTICAL KMeans and distance switches (lane/class): `C37_PANEL_128` on kmeans, minibatch-kmeans, gmm init / taxi, istella, main @ 8be4d20d4, A/B board-review-20261007 all-on, all-on ratio kmeans taxi 25x NV / 43x AMD, istella 5.3x NV; minibatch taxi 1.24 / 1.84; gmm istella 1.7 NV ms, **DROPPED (deleted)**: one thread per (cluster, feature) summed all n rows serially every Lloyd iteration (88 threads at taxi = 1 block), float panel sums (new bit contract); in MiniBatch it disabled the block-per-center kernel. Replaced by `C37_FUSED_ACCUMULATE` (fused assignment + Int32 row-block accumulation, incumbent bits), A/B owed

### MOJOLEARN_C37_ROW_PANELS

- Verdict: serial shape. Deleted 2026-10-07 by `d7c9736f0` (graph_controls: split C30 per family, KMEANS_ASSIGN arms, rewritten C37 fused, C38 split).
- Recoverable at `8be4d20d4` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_C37_PANEL_128.patch` (same patch as above).
- Files the patch restores: `experiments/classical_identical_ideas/graph_controls.mojo`
- grid_controls/classical-kmeans.json removed: per-(cluster,feature) serial sum over all rows (1 block at taxi), 25x/43x kmeans taxi; replaced by MOJOLEARN_C37_FUSED_ACCUMULATE (incumbent bits)
- EXPERIMENTS.md:1499 (Classical IDENTICAL KMeans and distance switches (lane/class): `C37_ROW_PANELS` on kmeans, minibatch-kmeans, gmm init / taxi, istella, main @ 8be4d20d4, A/B board-review-20261007 all-on, all-on ratio kmeans taxi 25x NV / 43x AMD, istella 5.3x NV; minibatch taxi 1.24 / 1.84; gmm istella 1.7 NV ms, **DROPPED (deleted)**: one thread per (cluster, feature) summed all n rows serially every Lloyd iteration (88 threads at taxi = 1 block), float panel sums (new bit contract); in MiniBatch it disabled the block-per-center kernel. Replaced by `C37_FUSED_ACCUMULATE` (fused assignment + Int32 row-block accumulation, incumbent bits), A/B owed

### MOJOLEARN_C38_DEVICE_POTENTIAL

- Verdict: dead code. Deleted 2026-10-07 by `d7c9736f0` (graph_controls: split C30 per family, KMEANS_ASSIGN arms, rewritten C37 fused, C38 split).
- Recoverable at `8be4d20d4` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_C37_PANEL_128.patch` (same patch as above).
- Files the patch restores: `experiments/classical_identical_ideas/graph_controls.mojo`
- grid_controls/classical-kmeans.json removed: no-op on NVIDIA/AMD (incumbent flags already on)
- EXPERIMENTS.md:1500 (Classical IDENTICAL KMeans and distance switches (lane/class): `C38_DEVICE_POTENTIAL` on kmeans / all, main @ 8be4d20d4, A/B none (no-op), - ms, **DROPPED (deleted)**: OR-ed into `IDN_KMEANS_INCR_INIT`, `KMEANS_FAST_PP_NOSYNC`, `IDN_KMEANS_INIT_PSI_DEVICE`, which NVIDIA/AMD IDENTICAL already set: no-op there. Its x_cluster half (k-means++ distances once per distinct candidate) is real and kept as `XCLUSTER_KPP_DISTINCT`

### MOJOLEARN_C38_REUSE_NEAREST

- Verdict: dead code. Deleted 2026-10-07 by `d7c9736f0` (graph_controls: split C30 per family, KMEANS_ASSIGN arms, rewritten C37 fused, C38 split).
- Recoverable at `8be4d20d4` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_C37_PANEL_128.patch` (same patch as above).
- Files the patch restores: `experiments/classical_identical_ideas/graph_controls.mojo`
- grid_controls/classical-kmeans.json removed: KMeans half no-op on NVIDIA/AMD; x_cluster half renamed MOJOLEARN_XCLUSTER_KPP_DISTINCT
- EXPERIMENTS.md:1500 (Classical IDENTICAL KMeans and distance switches (lane/class): `C38_REUSE_NEAREST` on kmeans / all, main @ 8be4d20d4, A/B none (no-op), - ms, **DROPPED (deleted)**: OR-ed into `IDN_KMEANS_INCR_INIT`, `KMEANS_FAST_PP_NOSYNC`, `IDN_KMEANS_INIT_PSI_DEVICE`, which NVIDIA/AMD IDENTICAL already set: no-op there. Its x_cluster half (k-means++ distances once per distinct candidate) is real and kept as `XCLUSTER_KPP_DISTINCT`

### MOJOLEARN_CC_FAST_OFF

- Verdict: removed. Deleted 2026-10-08 by `2851a6627` (PageRank/Louvain accept CSR (pointer handoff to xn_pr_csr/xn_louvain_csr); connected_components batched rounds + device relabel on every vendor, C33 CC chunking).
- Recoverable at `db707d03a` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_CC_FAST_OFF.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `python/mojolearn/_expansion_neighbors.py`, `x_neighbors/iter_device.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:48): removed: connected_components' batched rounds with the device relabel are the only path (lane gap-graph 2026-10-08); the per-round-wait path is gone

### MOJOLEARN_GMM_FAST_ESTEP_STACK

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `7b2638a38` (mixture estep: remove dropped GMM_FAST_ESTEP_STACK (DROPPED-slower; recover lane/apple-fast-linear@1c7c213f8)).
- Recoverable at `7ca37fd0b` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_GMM_FAST_ESTEP_STACK.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `mixture/checks/estep.mojo`
- EXPERIMENTS.md:473 (Cluster (38)): `GMM_FAST_BIG_CHOL + GMM_FAST_ESTEP_STACK + GMM_FAST_GRID_COV` on gmm / istella, lane/apple-fast-linear @ 1c7c213f8, A/B linear-gmm-all-istella, gmm istella 6,587 -> 9,046 ms, **DROPPED-slower**: +37.3%; GMM_FAST_ESTEP_STACK: code removed from main 7b2638a38; recover at lane/apple-fast-linear@1c7c213f8; GMM_FAST_GRID_COV: code removed from main dc1b4cc03; recover at lane/apple-fast-linear@1c7c213f8
- EXPERIMENTS.md:474 (Cluster (38)): `GMM_FAST_ESTEP_STACK` on gmm / istella, lane/apple-fast-linear @ 1c7c213f8, A/B linear-gmm-es-istella, gmm istella 6,566 -> 6,886 ms, **DROPPED-slower**: +4.9%; code removed from main 7b2638a38; recover at lane/apple-fast-linear@1c7c213f8

### MOJOLEARN_GMM_FAST_GRID_COV

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `dc1b4cc03` (mixture mstep: remove dropped GMM_FAST_GRID_COV (DROPPED-slower; recover lane/apple-fast-linear@1c7c213f8)).
- Recoverable at `7b2638a38` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_GMM_FAST_GRID_COV.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `mixture/checks/mstep.mojo`
- EXPERIMENTS.md:473 (Cluster (38)): `GMM_FAST_BIG_CHOL + GMM_FAST_ESTEP_STACK + GMM_FAST_GRID_COV` on gmm / istella, lane/apple-fast-linear @ 1c7c213f8, A/B linear-gmm-all-istella, gmm istella 6,587 -> 9,046 ms, **DROPPED-slower**: +37.3%; GMM_FAST_ESTEP_STACK: code removed from main 7b2638a38; recover at lane/apple-fast-linear@1c7c213f8; GMM_FAST_GRID_COV: code removed from main dc1b4cc03; recover at lane/apple-fast-linear@1c7c213f8
- EXPERIMENTS.md:475 (Cluster (38)): `GMM_FAST_GRID_COV` on gmm / istella, lane/apple-fast-linear @ 1c7c213f8, A/B linear-gmm-gc-istella, gmm istella 6,584 -> 9,858 ms, **DROPPED-slower**: +49.7%; n_iter 24 -> 42; code removed from main dc1b4cc03; recover at lane/apple-fast-linear@1c7c213f8

### MOJOLEARN_GRAPH_DIRECT_DISTANCE

- Verdict: slower. Deleted 2026-10-08 by `deae02e73` (grid-act-2: delete graph_direct (MOJOLEARN_GRAPH_DIRECT_DISTANCE; grid ge123e6f9 hdbscan 1.76x/1.70x istella, 1.14x/1.16x taxi slower); refuse the define; tombs).
- Recoverable at `a5eb8a916` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_GRAPH_DIRECT_DISTANCE.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `experiments/classical_identical_ideas/graph_controls.mojo`, `hdbscan/host/hdbscan_host_oracle.mojo`, `hdbscan/impl/cluster/detail/sparse_mr_mst.mojo`, `hierarchy/checks/linkage_oracle.mojo`, `hierarchy/impl/cluster/detail/connectivities.mojo`, `hierarchy/impl/cluster/detail/multi_gpu.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:197): removed: MOJOLEARN_GRAPH_DIRECT_DISTANCE retired 2026-10-08: slower, hdbscan NV 1.76x / AMD 1.70x istella, NV 1.14x / AMD 1.16x taxi (grid ge123e6f9); see EXPERIMENTS.md
- grid_controls/classical-kmeans.json removed (control graph_direct): DELETED 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 loser. on/off ms ratio NV/AMD: hdbscan istella 1.758/1.699 (1785.0 -> 3137.6 NV, 1413.6 -> 2402.2 AMD), taxi 1.141/1.158; combined 1.41x SLOWER. Deleted from sparse_mr_mst, the hdbscan host oracle, the linkage oracle and the connectivities/multi_gpu tile (expanded tile, [False]). Code recoverable at main 42d1e42c6; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1707 (IDENTICAL grid ge123e6f9: two promotions and eight loser arm): `MOJOLEARN_GRAPH_DIRECT_DISTANCE` on classical:hdbscan / istella, taxi (single-linkage Agglomerative shared the tile, unmeasured), main @ 42d1e42c6 (deleted on lane/grid-act-2), A/B ge123e6f9, istella NV 1785.0 -> 3137.6 (1.758), AMD 1413.6 -> 2402.2 (1.699); taxi NV 522.7 -> 596.5 (1.141), AMD 459.8 -> 532.5 (1.158); 1.41x combined SLOWER ms, **DROP (slower), code deleted**: (x-y)^2 in the mutual-reachability MST search/tile kernels and the linkage tile, host oracles following; slower on both vendors and both datasets. Deleted from hdbscan/impl/cluster/detail/sparse_mr_mst.mojo, hdbscan/host/hdbscan_host_oracle.mojo, hierarchy/checks/linkage_oracle.mojo; hierarchy connectivities and multi_gpu launch `pinned_distance_tile_direct_kernel[False]` (the expanded tile).

### MOJOLEARN_IDN_GMM_COV_SYM

- Verdict: quality loss. Deleted 2026-10-08 by `9829cdb31` (grid-losers-1: delete gmm_cov_sym (grid ge123e6f9: gmm taxi 1.84x/1.45x slower, mean log-likelihood -0.78%); mstep.mojo and gmm_host_oracle.mojo back to pre-b51).
- Recoverable at `687938771` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_GMM_COV_SYM.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `mixture/checks/mstep.mojo`, `mixture/cov_sym.mojo`, `mixture/host/gmm_host_oracle.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:184): removed: MOJOLEARN_IDN_GMM_COV_SYM retired 2026-10-08: slower, NV 1.84x / AMD 1.45x on gmm taxi, and mean log-likelihood -0.78% (grid ge123e6f9); see EXPERIMENTS.md
- grid_controls/classical-te-gmm.json removed (control gmm_cov_sym): DELETED 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 loser on speed AND quality. on/off ms ratio NV/AMD: gmm taxi 1.838/1.452 (SLOWER: 64.12 vs 34.88 ms NV, 92.24 vs 63.52 ms AMD), istella 0.966/0.978 (noise); mean_log_likelihood taxi 12.7082 vs 12.8076 (-0.78%, WORSE), istella SAME. The cover changes results, not a no-op. mixture/cov_sym.mojo, the sym kernels/launch in mixture/checks/mstep.mojo and the host mirror in mixture/host/gmm_host_oracle.mojo deleted (both files back to their pre-b512ddf29 text). Code recoverable at main ad7ed2370; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1609 (IDENTICAL grid ge123e6f9 losers deleted (lane/grid-losers-1,): `MOJOLEARN_IDN_GMM_COV_SYM` on more:gmm / taxi, istella, main @ ad7ed2370 (deleted on lane/grid-losers-1), A/B ge123e6f9, taxi NV 34.9 -> 64.1 (1.838), AMD 63.5 -> 92.2 (1.452); istella NV 579.6 -> 559.7 (0.966), AMD 542.9 -> 531.1 (0.978) ms, **DROP (slower AND quality), code deleted**: taxi mean log-likelihood 12.8076 -> 12.7082 (-0.78%, WORSE): the symmetric cover is not the no-op its lane claimed, it changes the fit; istella SAME and within noise. Deleted mixture/cov_sym.mojo, `center_scale_sym_kernel` / `cov_finish_sym_kernel` and the launch in mixture/checks/mstep.mojo (the plain d x d GEMM path is the only path) and the host mirror in mixture/host/gmm_host_oracle.mojo; both files are back to their text before b512ddf29

### MOJOLEARN_KMEANS_DIRECT_DISTANCE

- Verdict: slower. Deleted 2026-10-08 by `a5eb8a916` (grid-act-2: delete kmeans_assign direct arms (MOJOLEARN_KMEANS_DIRECT_DISTANCE; grid ge123e6f9 direct4 12.7x/6.2x istella, 1.78x/1.96x taxi slower, inertia same).
- Recoverable at `72716663a` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_KMEANS_DIRECT_DISTANCE.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `cluster/checks/plus_plus.mojo`, `cluster/host/kmeans_oracle.mojo`, `cluster/impl/detail/classical_assignment.mojo`, `cluster/impl/detail/kmeans.mojo`, `cluster/impl/detail/kmeans_transform.mojo`, `cluster/impl/detail/min_cluster_distance_compute.mojo`, `experiments/classical_identical_ideas/graph_controls.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:196): removed: MOJOLEARN_KMEANS_DIRECT_DISTANCE (kmeans_assign direct arms) retired 2026-10-08: slower, kmeans direct4 NV 12.7x / AMD 6.2x istella, NV 1.78x / AMD 1.96x taxi, inertia SAME (grid ge123e6f9); MOJOLEARN_KMEANS_ROW_ASSIGN=2|4 stays; see EXPERIMENTS.md
- grid_controls/classical-kmeans.json removed (control kmeans_assign): DELETED 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 loser. direct4/tiled ms ratio NV/AMD: kmeans istella 12.695/6.192 (620.2 -> 7873.6 NV, 541.0 -> 3349.7 AMD), taxi 1.777/1.961; combined 4.07x SLOWER; inertia SAME. The direct arms (direct2 shares the code, not a grid arm) are gone from the row kernel, k-means++, k-means||, transform and the host oracle; rows2/rows4 stay. Code recoverable at main 42d1e42c6; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1706 (IDENTICAL grid ge123e6f9: two promotions and eight loser arm): `MOJOLEARN_KMEANS_DIRECT_DISTANCE` on classical:kmeans / istella, taxi, main @ 42d1e42c6 (deleted on lane/grid-act-2), A/B ge123e6f9, istella NV 620.2 -> 7873.6 (12.695), AMD 541.0 -> 3349.7 (6.192); taxi NV 319.0 -> 566.9 (1.777), AMD 705.2 -> 1382.8 (1.961); 4.07x combined SLOWER ms, **DROP (slower), code deleted**: (x-c)^2 on the row-register kernel at every size (no tiled twin, so the k*d cost rule never applied) plus the direct k-means++/k-means

### MOJOLEARN_KMEANS_FAST_ROWNORM

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `073bd0029` (cluster kmeans: remove dropped KMEANS_FAST_ROWNORM (DROPPED-noise; recover lane/apple-fast-core@9a31ebb4c)).
- Recoverable at `cc300add8` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_KMEANS_FAST_ROWNORM.patch` (conflicts with main: restore by hand from the recoverable sha).
- Files the patch restores: `cluster/estimator.mojo`, `cluster/impl/detail/kmeans.mojo`, `cluster/impl/detail/kmeans_fast.mojo`
- EXPERIMENTS.md:476 (Cluster (38)): `KMEANS_FAST_ROWNORM` on kmeans / taxi, lane/apple-fast-core @ 9a31ebb4c, A/B core-kmeans-rownorm-taxi, kmeans taxi -3.1% ms, **DROPPED-noise**: <5% at n=1; inertia same; code removed from main 073bd0029; recover at lane/apple-fast-core@9a31ebb4c

### MOJOLEARN_KMEANS_FAST_SKIP_PREDICT

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `a595988e8` (cluster kmeans: remove dropped KMEANS_FAST_SKIP_PREDICT and detail/kmeans_fast.mojo (DROPPED-noise; recover lane/apple-fast-core@9a31ebb4c)).
- Recoverable at `073bd0029` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_KMEANS_FAST_SKIP_PREDICT.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `cluster/estimator.mojo`, `cluster/impl/detail/kmeans_fast.mojo`
- EXPERIMENTS.md:477 (Cluster (38)): `KMEANS_FAST_SKIP_PREDICT` on kmeans / taxi, lane/apple-fast-core @ 9a31ebb4c, A/B core-kmeans-skippred-taxi, kmeans taxi -2.2% ms, **DROPPED-noise**: <5% at n=1; code removed from main a595988e8; recover at lane/apple-fast-core@9a31ebb4c

### MOJOLEARN_OPTICS2_ALL

- Verdict: DROPPED. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-batch @ 3150d75c1`.
- EXPERIMENTS.md:497 (Cluster (38)): `OPTICS2_ALL` on optics / istella; optics / taxi, lane/apple-fast-batch @ 3150d75c1, A/B optics2-all-taxi-x, taxi 411 -> 34,474 (84x slower) ms, **DROP**: includes STEP_BATCH; never merged

### MOJOLEARN_OPTICS_CORE_SQ

- Verdict: DROPPED. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-batch @ 3150d75c1`.
- EXPERIMENTS.md:498 (Cluster (38)): `OPTICS_CORE_SQ` on optics / istella; optics / taxi, lane/apple-fast-batch @ 3150d75c1, A/B optics2-sq-taxi-x, taxi +6.1% ms, **DROP**: never merged

### MOJOLEARN_OPTICS_FRONTIER_DEVICE

- Verdict: DROPPED. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-opv @ 9a844772e`.
- EXPERIMENTS.md:500 (Cluster (38)): `OPTICS_FRONTIER_DEVICE` on optics / istella; optics / taxi, lane/apple-fast-opv @ 9a844772e, A/B opv-fd-istella, opv-fd-taxi (old base: optics2-fd-istella-x 433 -> 259), vs main istella 272.8 -> 272.0 (-0.3%); taxi 253.3 -> 250.5 (-1.1%) ms, **DROP**: main's OPTICS_FAST_DEVICE_ORDER (cluster2) already took the gain; clusters/silhouette identical; never merged

### MOJOLEARN_OPTICS_LIVEBUF

- Verdict: DROPPED. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-opv @ 9a844772e`.
- EXPERIMENTS.md:501 (Cluster (38)): `OPTICS_LIVEBUF` on optics / istella; optics / taxi, lane/apple-fast-opv @ 9a844772e, A/B opv-lb-istella, opv-lb-taxi (old base -8.6%), vs main istella 271.4 -> 405.8 (+50%); taxi 253.1 -> 383.4 (+52%) ms, **DROP**: turns on optics2's route, which bypasses main's faster device order; never merged

### MOJOLEARN_OPTICS_STEP_BATCH

- Verdict: DROPPED. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-batch @ 3150d75c1`.
- EXPERIMENTS.md:502 (Cluster (38)): `OPTICS_STEP_BATCH` on optics / istella; optics / taxi, lane/apple-fast-batch @ 3150d75c1, A/B optics2-sb-taxi-x, taxi 416 -> 22,564 (54x slower) ms, **DROP**: never merged

### MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_FIN

- Verdict: DROPPED. Deleted 2026-10-03 by `2b40bf184` (gap-cls2: MBK POOL FAST Apple default (M3 A/B istella 256.7->170.7 ms); drop losers G128 and FIN).
- Recoverable at `55bc92e33` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_FIN.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `x_cluster/minibatch_fast.mojo`
- EXPERIMENTS.md:506 (Cluster (38)): `MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_FIN` on minibatch-kmeans / istella, taxi, lane/apple-fast-gap-cls2@72602a339 (deleted before merge), A/B gapcls2-fin-mbk-{istella,taxi}, +5%, -3% ms, **DROP, deleted before merge**: noise; a single-block kernel (no-one-block rule)

### MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_G128

- Verdict: DROPPED. Deleted 2026-10-03 by `2b40bf184` (gap-cls2: MBK POOL FAST Apple default (M3 A/B istella 256.7->170.7 ms); drop losers G128 and FIN).
- Recoverable at `55bc92e33` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_FIN.patch` (same patch as above).
- Files the patch restores: `x_cluster/minibatch_fast.mojo`
- EXPERIMENTS.md:505 (Cluster (38)): `MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_G128` on minibatch-kmeans / istella, taxi, lane/apple-fast-gap-cls2@72602a339 (deleted before merge), A/B gapcls2-g128-mbk-{istella,taxi}, +14%, +22% ms, **DROP, deleted before merge**: slower


### MOJOLEARN_BGMM_ESTEP1

- Verdict: DROPPED-noise. Deleted 2026-10-09 by lane/owed-deletions-D3 (owed deletion, D3).
- Recoverable at `b639a2bd2` (main the lane branched from). Patch: `experiments/removed/MOJOLEARN_BGMM_ESTEP1.patch` (reverse of the lane's deletion commit; applies to the lane head).
- What it tried: the bayesian-gmm E-step's three kernels (gauss_q, resp, exp) as one row-per-thread launch (ops.estep / _estep_row_kernel), FAST only.
- Files the patch restores: `x_cluster/bgmm.mojo`, `x_cluster/device_ops.mojo`, `x_cluster/host/host_ops.mojo`, `x_cluster/ops.mojo`
- EXPERIMENTS.md:482: `BGMM_ESTEP1` | bayesian-gmm / taxi | lane/apple-fast-cluster2 @ ded4ea07b | cluster2-bgmm-estep1-taxi | - | DROPPED-noise | reconciled 2026-10-05: cluster2-bgmm-estep1-taxi-b 309.7 -> 311.1 (+0.5%), mean_log_likelihood same (LEDGER 2026-10-03); define stays opt-in on main (x_cluster/bgmm.mojo). Was OPEN: A/B queued, no judged result yet
- Guard refusal (core/six_lane_experiment_guards.mojo): removed 2026-10-09 (lane/owed-deletions-D3): BGMM_ESTEP1 (one-launch row-per-thread E-step, ops.estep) was NOISE: bayesian-gmm taxi 309.7 -> 311.1 ms (+0.5%), mean_log_likelihood same; code at main b639a2bd2; see docs/TOMBSTONES.md

## Time series

### MOJOLEARN_ARIMA_FAST_LS_NOREAD

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `569e80238` (arima: remove dropped ARIMA_FAST_LS_NOREAD (DROPPED-noise; recover lane/apple-fast-gap-arima@d967c0121)).
- Recoverable at `75c59ea26` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_ARIMA_FAST_LS_NOREAD.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `arima/impl/batched_fit.mojo`, `arima/impl/fast_lbfgs_async.mojo`
- EXPERIMENTS.md:537 (Time series (34)): `ARIMA_FAST_LS_NOREAD` on autoarima / synthetic, lane/apple-fast-gap-arima @ d967c0121, A/B gaparima-noread-synthetic, 28,578 -> 28,587 ms, **DROPPED-noise**: 0%; code removed from main 569e80238; recover at lane/apple-fast-gap-arima@d967c0121

### MOJOLEARN_ARIMA_FAST_P_FIX

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `dd47c5df0` (arima: remove dropped ARIMA_FAST_P_FIX (DROPPED-slower; recover lane/apple-fast-gap-arima@d967c0121)).
- Recoverable at `569e80238` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_ARIMA_FAST_P_FIX.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `arima/impl/batched_kalman.mojo`
- EXPERIMENTS.md:538 (Time series (34)): `ARIMA_FAST_P_FIX` on autoarima / taxi-hourly, lane/apple-fast-gap-arima @ d967c0121, A/B gaparima-pfix-taxi-b, gaparima-pfixasync, 39,550 -> 104,849; with ASYNC 39,570 -> 64,830 ms, **DROPPED-slower**: +165%; code removed from main dd47c5df0; recover at lane/apple-fast-gap-arima@d967c0121

### MOJOLEARN_C58_FORECAST4

- Verdict: slower. Deleted 2026-10-08 by `76967c4a4` (grid-act-3: delete c58_forecast4 (MOJOLEARN_C58_FORECAST4; grid ge123e6f9 theta/ets family 2.2-3.1x slower on both vendors, quality same); refuse the define; to).
- Recoverable at `68a8f4844` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_C58_FORECAST4.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `experiments/classical_identical_ideas/stats_controls.mojo`, `sequence/exec_device.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:205): removed: MOJOLEARN_C58_FORECAST4 retired 2026-10-08: slower, theta/ets family NV 2.53-3.15x / AMD 1.81-3.08x on synthetic and taxi-hourly (2.72x combined), quality SAME (grid ge123e6f9); see EXPERIMENTS.md
- grid_controls/classical-misc.json removed (control c58_forecast4): DELETED 2026-10-08 (lane grid-act-3): IDENTICAL grid ge123e6f9 loser. on/off ms ratio NV/AMD (synthetic, taxi-hourly): auto-theta 2.93/2.58, 2.81/2.13; damped-ets 2.59/2.39, 2.53/2.43; dynamic-optimized-theta 2.96/2.49, 2.66/1.81; dynamic-theta 3.15/3.00, 3.02/2.91; optimized-theta 2.72/2.60, 3.09/2.99; theta 2.93/2.86, 3.13/3.08 (2.72x combined SLOWER); quality SAME. Four series per thread cut the parallelism 4x. seq_kernel is one series per thread again (sequence/exec_device.mojo). Code recoverable at main bc10b8b56; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1721 (IDENTICAL grid ge123e6f9: one promotion and six loser contro): `MOJOLEARN_C58_FORECAST4` on expanded:auto-theta, damped-ets, dynamic-optimized-theta, dynamic-theta, optimized-theta, theta / synthetic, taxi-hourly, main @ bc10b8b56 (deleted on lane/grid-act-3), A/B ge123e6f9 (nv n0024, amd a0019), auto-theta synthetic NV 943.6 -> 2768.3 (2.93), AMD 2113.0 -> 5447.2 (2.58); taxi-hourly NV 3359.6 -> 9425.2 (2.81), AMD 8737.5 -> 18625.7 (2.13); damped-ets 2.59/2.39, 2.53/2.43; dynamic-optimized-theta 2.96/2.49, 2.66/1.81; dynamic-theta 3.15/3.00, 3.02/2.91; optimized-theta 2.72/2.60, 3.09/2.99; theta 2.93/2.86, 3.13/3.08 (NV/AMD, synthetic then taxi-hourly); 2.72x combined SLOWER ms, **DROP (slower), code deleted**: four theta/ets/garch series per thread: 4x fewer threads for the same serial optimizer work per series; quality SAME. seq_kernel is one series per thread again; define refused

### MOJOLEARN_C58_SHARED_PREP

- Verdict: slower. Deleted 2026-10-08 by `3978db7ce` (grid-losers-1: delete c58_shared_prep (grid ge123e6f9 noise on ets); refuse the define).
- Recoverable at `ab4e8e543` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_C58_SHARED_PREP.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `experiments/classical_identical_ideas/stats_controls.mojo`, `holtwinters/impl/internal/hw_estimate.mojo`, `holtwinters/impl/internal/hw_estimate_launch.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:182): removed: MOJOLEARN_C58_SHARED_PREP retired 2026-10-08: noise, NV 0.996x / AMD 0.980x on ets (grid ge123e6f9); see EXPERIMENTS.md
- grid_controls/classical-misc.json removed (control c58_shared_prep): DELETED 2026-10-08 (lane grid-losers-1): IDENTICAL grid ge123e6f9 noise, a loser. on/off ms ratio NV/AMD: ets synthetic 0.996/0.980; forecast_rmse and insample_rmse SAME. Code recoverable at main ad7ed2370; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1607 (IDENTICAL grid ge123e6f9 losers deleted (lane/grid-losers-1,): `MOJOLEARN_C58_SHARED_PREP` on more:ets / synthetic, main @ ad7ed2370 (deleted on lane/grid-losers-1), A/B ge123e6f9, NV 148.3 -> 147.8 (0.996), AMD 343.8 -> 337.0 (0.980); 0.988x combined ms, **DROP (noise), code deleted**: per-series scale once per series instead of per start block: inside the noise floor; forecast_rmse and insample_rmse SAME. Deleted `hw_classical_scale_kernel` and its launch and scratch slot (holtwinters/impl/internal/hw_estimate*.mojo)

### MOJOLEARN_C60_DIFF_REUSE

- Verdict: dead code. Deleted 2026-10-07 by `f7abb25cf` (classical-misc: C58 arms 16->5 and TEAM64 -> TEAM_MIB sweep; delete dead C60_DIFF_REUSE; add default-off C61 HDBSCAN same-component tile skip).
- Recoverable at `912c4560f` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_C60_DIFF_REUSE.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `experiments/classical_identical_ideas/graph_controls.mojo`, `experiments/classical_identical_ideas/stats_controls.mojo`, `hdbscan/impl/cluster/detail/sparse_mr_mst.mojo`, `sequence/fit_team_py.mojo`, `tsa/impl/select_d_fast.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:46): removed: C60_DIFF_REUSE was dead code (select_d never reaches d == 2)
- grid_controls/classical-misc.json removed: dead code: gated d_ == 2 branch unreachable (select_d loops d_ < d_max <= 2 - D). Code, arms (diff_reuse, combo, combined) deleted; guard rejects the define.

### MOJOLEARN_SEQ_FAST_THETA_HOIST

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `239fde86d` (sequence/theta: remove dropped SEQ_FAST_THETA_HOIST (DROPPED-noise on top of THETA_SPEC, gaptsa-spechoist-theta-taxi-hourly 21.8 -> 21.5; recover lane/apple-fas).
- Recoverable at `7c0839144` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_SEQ_FAST_THETA_HOIST.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `sequence/theta.mojo`
- EXPERIMENTS.md:547 (Time series (34)): `SEQ_FAST_THETA_HOIST` on theta / taxi-hourly, lane/apple-fast-gap-tsa @ e9da47064, A/B gaptsa-thetahoist-theta-taxi-hourly, theta taxi-hourly 218 -> 58 ms, **DROPPED-noise**: -73% alone (at 976a585a0); on top of THETA_SPEC (default) gaptsa-spechoist-theta-taxi-hourly 21.8 -> 21.5 (-1%, noise); code removed from main 239fde86d; recover at lane/apple-fast-gap-tsa@e9da47064

### MOJOLEARN_SEQ_FAST_VAR_SPEC

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `dfbdc7943` (apple-fast-gap-tsa: TSA_FAST_KPSS_PACK, TSA_FAST_SELD_FUSED, SEQ_FAST_VAR_ONECOPY, SEQ_FAST_THETA_SPEC default on FAST+Apple (M3 A/B n=1, quality identical: kps).
- Recoverable at `2575ed958` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_SEQ_FAST_VAR_SPEC.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `bindings/_mojolearn_x_sequence.mojo`, `python/mojolearn/_x_sequence_var.py`, `sequence/ops.mojo`, `sequence/pyapi.mojo`, `sequence/theta_spec.mojo`, `tsa/estimator.mojo`, `tsa/impl/timeSeries/kpss_fused.mojo`
- EXPERIMENTS.md:542 (Time series (34)): `SEQ_FAST_VAR_ONECOPY + SEQ_FAST_VAR_SPEC` on var / synthetic; var / taxi-hourly, lane/apple-fast-gap-tsa @ e9da47064, A/B gaptsa-varboth-var-taxi-hourly, gaptsa-varboth-var-synthetic, = VAR_ONECOPY alone ms, **DROPPED-noise**: SPEC adds nothing
- EXPERIMENTS.md:543 (Time series (34)): `SEQ_FAST_VAR_SPEC` on var / synthetic; var / taxi-hourly, lane/apple-fast-gap-tsa @ e9da47064, A/B gaptsa-varspec-var-taxi-hourly, gaptsa-varspec-var-synthetic, var +14% / -2% ms, **DROPPED-slower**: deleted from main at merge; recoverable at 5057bee75


## Kernel / GP

### MOJOLEARN_KERNEL_FAST_GPR_RESIDENT

- Verdict: DROPPED-semantics. Deleted 2026-10-03 by `cc300add8` (gaussian_process: remove dropped KERNEL_FAST_GPR_RESIDENT note (DROPPED-semantics, no code on main; recover lane/apple-fast-kernel@9e851777c)).
- Recoverable at `e33bb66e0` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_KERNEL_FAST_GPR_RESIDENT.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `gaussian_process/estimator.mojo`
- EXPERIMENTS.md:556 (Kernel / GP (11)): `KERNEL_FAST_GPR_RESIDENT` on gpr / istella, lane/apple-fast-kernel @ 9e851777c, A/B kernel-gpr-resident-ist, gpr istella 168 -> 133 ms, **DROPPED-semantics**: not made default: main resident GPR chain already covers it; lane arm folded on host; note removed from main cc300add8 (no code was on main); recover at lane/apple-fast-kernel@9e851777c

### MOJOLEARN_SVGP_FAST_GPU

- Verdict: DROPPED-noise. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-neighbors2 @ 5fb6edd3f`.
- EXPERIMENTS.md:558 (Kernel / GP (11)): `SVGP_FAST_GPU` on svgp / taxi, lane/apple-fast-neighbors2 @ 5fb6edd3f, A/B n2-svgp-gpu-taxi, svgp taxi 924 -> 903 ms, **DROPPED-noise**: -2.3% n=1; opt-in removed at merge


## Neural

### MOJOLEARN_IDN_AF_VEC_FUSED

- Verdict: noise. Deleted 2026-10-09 by `0b7bde4ca` (postmerge-act-5: delete IDN_AF_VEC_FUSED (noise, avg 0.98x), tombstones).
- Recoverable at `16a2c0dd3` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_AF_VEC_FUSED.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `sequence/adafactor.mojo`, `sequence/af_fused.mojo`, `sequence/dispatch.mojo`, `sequence/exec.mojo`, `sequence/exec_device.mojo`, `sequence/ops.mojo`, `sequence/pyapi.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:239): removed 2026-10-09 (neural A/B nv2 v1010/v1012, amd a1141/a1143, lane/postmerge-act-5): neural-io-2 Adafactor fused vector step (sequence/af_fused.mojo, OP_AF_VFUSE/OP_AF_VFIN) was NOISE: adafactor synthetic NV 13.671 -> 13.738 ms (1.00x), AMD 8.620 -> 8.186 ms (0.95x), avg 0.98x, digest aa99a3dc unchanged; code at main 0a7b206f1; see EXPERIMENTS.md
- grid_controls/neural-io-2.json removed (control idn_af_vec_fused): DELETED 2026-10-09 (lane postmerge-act-5): post-merge A/B, one run per arm, found it NOISE: adafactor synthetic NV 13.671 -> 13.738 ms (1.00x, nv2 v1010/v1012), AMD 8.620 -> 8.186 ms (0.95x, a1141/a1143), vendor average 0.98x; digest aa99a3dc unchanged. sequence/af_fused.mojo, OP_AF_VFUSE/OP_AF_VFIN and the pyapi route deleted; define refused in core/six_lane_experiment_guards.mojo; code recoverable at main 0a7b206f1.
- EXPERIMENTS.md:1773 (Post-merge A/B races: SGD epoch kernel, x_prep device codes ): `MOJOLEARN_IDN_AF_VEC_FUSED` on neural:adafactor / synthetic, lane/neural-io-2 @ bf06ffdb3, A/B owed (nv + amd RACE adafactor ON vs OFF; ID check nv + amd), 17.3 -> ? (NV) ms, **DROPPED 2026-10-09 (noise;see lane/postmerge-act-5 section)**: same coop_sumsq / af_fold_parts chains in the same order: bits expected unchanged; host twin built with the define
- EXPERIMENTS.md:1815 (Post-merge fg2 A/B: round-robin Jacobi and float-float Gram ): `MOJOLEARN_IDN_AF_VEC_FUSED` on neural:adafactor / synthetic, lane/neural-io-2 @ bf06ffdb3, on main @ 0a7b206f1, A/B nv2 v1010/v1012, amd a1141/a1143, NV 13.671 -> 13.738 (1.00x), AMD 8.620 -> 8.186 (0.95x); avg 0.98x ms, **DROPPED**: noise; digest aa99a3dc unchanged. sequence/af_fused.mojo, OP_AF_VFUSE/OP_AF_VFIN, the dispatch/exec entries and the pyapi route deleted; define refused; recoverable at main 0a7b206f1.

### MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE

- Verdict: slower. Deleted 2026-10-07 by `52aecf2f3` (attention head share: NN17/NI19 merged into MOJOLEARN_IDN_ATTN_HEAD_SHARE=4; delete I06/NI19 two-head loser).
- Recoverable at `f96f2f28b` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `experiments/performance_ideas/attention_time.mojo`, `transformer/impl/llama/fused_attention.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:104): MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json
- grid_controls/neural-gemm-attn-dedupe.json removed: I06/NI19 two-head GQA reuse: measured loser on MI325X (1.21x) and L40S (1.15/1.11x); deleted, EXPERIMENTS.md row
- EXPERIMENTS.md:1466 (IDENTICAL neural GEMM/attention switch dedupe (lane/neural-g): `MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE` on attention forward GQA ratio 2 / length 1024/1536, B1, heads8, kv4, hd64, measured @ e80a1d0 (fixed kvgrid schedule); deleted from lane/neural-gemm-attn-dedupe, A/B overnight-ab-20261006 I06, MI325X 2.434/3.581 -> 2.947/4.342 (1.211/1.212x); L40S 1.143/2.262 -> 1.311/2.506 (1.147/1.108x) ms, **DROP**: slower on both vendors. The four-head NN17 arm survives as `MOJOLEARN_IDN_ATTN_HEAD_SHARE=4`

### MOJOLEARN_IDN_ATTN_SOFTMAX=1

- Verdict: broken. Deleted 2026-10-08 by `89a183889` (grid-act-3: delete the attn_softmax summary_tree arm (MOJOLEARN_IDN_ATTN_SOFTMAX=1, NN20, and its split-KV controls; grid ge123e6f9 lm/samba/transformer 1.6-43x).
- Recoverable at `3d12702e7` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_ATTN_SOFTMAX-arm1.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `experiments/classical_identical_ideas/stats_controls.mojo`, `sequence/exec_device.mojo`, `training/byte_lm_host.mojo`, `training/neural_arithmetic_profile.mojo`, `training/neural_identical_experiments.mojo`, `transformer/checks/transformer_backward.mojo`, `transformer/checks/transformer_backward_oracle.mojo`, `transformer/checks/transformer_oracle.mojo`, `transformer/experiments/attention_summary_contract.mojo`, `transformer/experiments/attention_summary_split.mojo`, `transformer/experiments/attention_summary_tree.mojo`, `transformer/experiments/profile.mojo` ...
- grid_controls/neural-gemm-attn-dedupe.json removed (control attn_softmax): DELETED 2026-10-08 (lane grid-act-3): IDENTICAL grid ge123e6f9 loser. summary_tree/off ms ratio NV/AMD: lm-forward 4.86/16.79 (65.8 -> 319.9 NV, 34.2 -> 574.2 AMD), lm-train-step 42.89/40.18, samba-forward 1.79/4.79, samba-train-step 1.64/1.81, transformer-forward 5.48/21.89 (7.30x combined SLOWER); mean_nll not judged. The NN20 summary tree (transformer/experiments/attention_summary_contract/tree/split.mojo, summary_model*.mojo) and its split-KV controls (nn20_split_kv, nn20_split_kv_leaves) are deleted; online_tile32 (HOLD_BROKEN) stays. Code recoverable at main bc10b8b56; =1 refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1725 (IDENTICAL grid ge123e6f9: one promotion and six loser contro): `MOJOLEARN_IDN_ATTN_SOFTMAX=1` on neural:lm-forward, lm-train-step, samba-forward, samba-train-step, transformer-forward / bytes, gaussian, main @ bc10b8b56 (deleted on lane/grid-act-3), A/B ge123e6f9 (nv v0269/v0177/n0561, amd a0670/a0576/a0580), lm-forward NV 65.8 -> 319.9 (4.86), AMD 34.2 -> 574.2 (16.79); lm-train-step NV 36.0 -> 1542.8 (42.89), AMD 42.7 -> 1717.1 (40.18); samba-forward NV 12.0 -> 21.4 (1.79), AMD 8.8 -> 42.3 (4.79); samba-train-step NV 96.0 -> 157.6 (1.64), AMD 147.2 -> 266.5 (1.81); transformer-forward NV 7.3 -> 40.0 (5.48), AMD 3.2 -> 70.9 (21.89); 7.30x combined SLOWER ms, **DROP (slower), code deleted**: balanced 32-key summary-tree softmax: every cell slower on both vendors; mean_nll not judged. transformer/experiments/attention_summary_{contract,tree,split}.mojo and summary_model{,_contract,_host}.mojo deleted with their call sites (modeling_llama eager attention, transformer_backward, the two oracles, profile labels, byte_lm_host); `=1` and the split defines refused; online_tile32 (=2, HOLD_BROKEN) stays

### MOJOLEARN_IDN_NEURAL_LEAF

- Verdict: slower. Deleted 2026-10-07 by `870bcd1f9` (GEMM leaf: one switch MOJOLEARN_IDN_GEMM_LEAF (neural128/neural256/all256); delete I04 leaf64 loser).
- Recoverable at `25881f493` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_NEURAL_LEAF.patch` (applies cleanly to main at the time of writing); first apply `experiments/removed/_shared-870bcd1f9.patch`.
- Files the patch restores: `gemm/contract.mojo`, `gemm/experiments/neural_profile.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:97): MOJOLEARN_IDN_NEURAL_LEAF is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json
- grid_controls/neural-gemm-attn-dedupe.json removed: leaf-64 arm of NN03: same loser; legal set now 128/256
- grid_controls/neural-gemm-attn-dedupe.json removed: merged into gemm_leaf; the old define is refused by core/six_lane_experiment_guards.mojo
- EXPERIMENTS.md:1465 (IDENTICAL neural GEMM/attention switch dedupe (lane/neural-g): `IDN_NEURAL_LEAF` on IDENTICAL GEMM leaf 64 vs 128 / m1024 n1024 k2048 and neighbor m1023 n1025 k2049, measured @ cbcc8dcd3303; deleted from lane/neural-gemm-attn-dedupe (base 8be4d20d4), A/B overnight-ab-20261006 I04 (amd/normalized-measurements.json, nvidia/default-repair-normalized-measurements.json), MI325X 0.182 -> 0.231 (1.268x; neighbor 1.157x); L40S 0.280/0.255 -> 0.304/0.281 ms, **DROP**: slower on both voting vendors; leaf is a bit version, so no FAST-style keep. `MOJOLEARN_IDN_GEMM_LEAF` keeps arms neural128/neural256/all256

### MOJOLEARN_IDN_NEURAL_NN05

- Verdict: slower. Deleted 2026-10-08 by `ed0527dce` (grid-act-4: delete the neural_gemm_pair nn05 arm (MOJOLEARN_IDN_NEURAL_NN05; grid ge123e6f9 lm-forward 2.19x/2.80x, lm-train-step 3.72x/2.44x, transformer-forwa).
- Recoverable at `4495dac01` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_NEURAL_NN05.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `gemm/experiments/neural_grouped.mojo`, `gemm/neural_dispatch.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:217): removed 2026-10-08 (grid ge123e6f9, lane/grid-act-4): neural_gemm_pair=nn05 (fused gate/up pair kernel) was SLOWER on both vendors: lm-forward NV 2.19x / AMD 2.80x, lm-train-step 3.72x / 2.44x, transformer-forward 3.85x / 3.47x (3.0x combined); same bits; code at main 4e3da4282, see docs/apple-fast/EXPERIMENTS.md
- grid_controls/neural-gemm-attn-dedupe.json removed (control neural_gemm_pair): DELETED 2026-10-08 (lane grid-act-4): IDENTICAL grid ge123e6f9 loser. nn05/off ms NV/AMD: lm-forward 65.8 -> 144.4 / 34.2 -> 95.9 (2.19x / 2.80x), lm-train-step 36.0 -> 134.1 / 42.7 -> 104.2 (3.72x / 2.44x), transformer-forward 7.3 -> 28.1 / 3.2 -> 11.1 (3.85x / 3.47x); 3.0x combined SLOWER; hashes equal to the incumbent (no bits). _neural_pair_kernel (gemm/neural_dispatch.mojo) and neural_grouped_ab's shared-left / grouped arms (gemm/experiments/neural_grouped.mojo) deleted; nn07 stays. Code recoverable at main 4e3da4282; the define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1738 (IDENTICAL grid ge123e6f9: neural promotions, one split and o): `MOJOLEARN_IDN_NEURAL_NN05` on neural:lm-forward, neural:lm-train-step / bytes; neural:transformer-forward / gaussian, lane/grid-act-4 from main @ 4e3da4282, A/B ge123e6f9 (nv v0253/v0318, amd a0653/a0748), lm-forward NV 65.8 -> 144.4 (2.19), AMD 34.2 -> 95.9 (2.80); lm-train-step NV 36.0 -> 134.1 (3.72), AMD 42.7 -> 104.2 (2.44); transformer-forward NV 7.3 -> 28.1 (3.85), AMD 3.2 -> 11.1 (3.47); 3.0x combined ms, **DELETED (loser)**: gate/up projections as one thread-per-cell kernel sharing each A load loses the tiled identical GEMM on both vendors; same bits; `_neural_pair_kernel` and neural_grouped_ab's shared-left / grouped arms removed, tombstones at both sites

### MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN

- Verdict: serial shape. Deleted 2026-10-07 by `0a0c7205f` (L11: delete NI49 row-serial LSTM scan arm (serial per row); NN44 grouping merged into IDN_MOE_STABLE_PACK).
- Recoverable at `8fe3f695e` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `sequence/exec_device.mojo`, `sequence/moe_group.mojo`, `sequence/recurrent_scan.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:134): retired define MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN: use deleted (serial per-row scan)
- grid_controls/neural-seq-train-dedupe.json removed: DELETED: one GPU thread per batch row, serial over units/timesteps; breaks the IDENTICAL parallel rule (EXPERIMENTS row)
- EXPERIMENTS.md:1474 (IDENTICAL neural sequence/training switch dedupe (lane/neura): `IDN_SEQ_ROW_SERIAL_SCAN` on lstm-clf, lstm-reg (x_sequence), lane/neural-seq-train-dedupe @ 83b9bf20a (deleted; last present at 8be4d20d4), A/B none, - ms, **DROPPED-rule**: one GPU thread per batch row, serial over units and timesteps: breaks the IDENTICAL parallel-GPU rule; never measured, not a board lane. Recoverable at 8be4d20d4 sequence/recurrent_scan.mojo:71-83

### MOJOLEARN_NI13_CNN_WEIGHT_GENERATION_CACHE

- Verdict: dead code. Deleted 2026-10-07 by `8b90439d5` (L11: delete unreferenced rejected NI13 weight-cache helper; EXPERIMENTS row).
- Recoverable at `8a44c9587` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_NI13_CNN_WEIGHT_GENERATION_CACHE.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `x_cnn/neural_weight_cache.mojo`
- grid_controls/neural-seq-train-dedupe.json removed: DELETED: rejected at source review, x_cnn/neural_weight_cache.mojo had no importer (EXPERIMENTS row)
- EXPERIMENTS.md:1475 (IDENTICAL neural sequence/training switch dedupe (lane/neura): `NI13_CNN_WEIGHT_GENERATION_CACHE` on x_cnn (CNN forward), lane/neural-seq-train-dedupe (deleted; last present at 8be4d20d4), A/B none, - ms, **DROPPED-rejected**: rejected by source review (the shipped route performs no repack for a cache to remove); x_cnn/neural_weight_cache.mojo had no importer. Recoverable at 8be4d20d4

### MOJOLEARN_NN22_EAGER_DKDV_PAIR

- Verdict: unmeasured. Deleted 2026-10-07 by `a1488ca2d` (attention/norm/rope/swiglu/kv-cache: one switch per idea (IDN_NORM, IDN_ROPE, IDN_TRAIN_SWIGLU, IDN_TRAIN_NO_DECODE_CACHE); delete NN22/NN23).
- Recoverable at `8058b2554` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_NN22_EAGER_DKDV_PAIR.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-a1488ca2d.patch`.
- Files the patch restores: `transformer/checks/transformer_backward.mojo`, `transformer/experiments/attention_schedules.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:107): MOJOLEARN_NN22_EAGER_DKDV_PAIR is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json
- grid_controls/neural-gemm-attn-dedupe.json removed: eager-fallback backward only, unmeasured; deleted per brief
- EXPERIMENTS.md:1467 (IDENTICAL neural GEMM/attention switch dedupe (lane/neural-g): `MOJOLEARN_NN22_EAGER_DKDV_PAIR` on transformer eager attention backward dK/dV pairing, source @ 8be4d20d4; deleted, A/B -, - ms, **DROP (unmeasured)**: only the eager fallback backward (`transformer/checks/transformer_backward.mojo`) read it, never the fused backward the board trains through; deleted by the lane brief instead of wiring into `fused_bwd_*`

### MOJOLEARN_NN23_ROWDOT_DS

- Verdict: unmeasured. Deleted 2026-10-07 by `a1488ca2d` (attention/norm/rope/swiglu/kv-cache: one switch per idea (IDN_NORM, IDN_ROPE, IDN_TRAIN_SWIGLU, IDN_TRAIN_NO_DECODE_CACHE); delete NN22/NN23).
- Recoverable at `8058b2554` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_NN23_ROWDOT_DS.patch` (needs `git apply -3` (main moved on)); first apply `experiments/removed/_shared-a1488ca2d.patch`.
- Files the patch restores: `transformer/checks/transformer_backward.mojo`, `transformer/experiments/attention_schedules.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:108): MOJOLEARN_NN23_ROWDOT_DS is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json
- grid_controls/neural-gemm-attn-dedupe.json removed: eager-fallback backward only, unmeasured; deleted per brief
- EXPERIMENTS.md:1468 (IDENTICAL neural GEMM/attention switch dedupe (lane/neural-g): `MOJOLEARN_NN23_ROWDOT_DS` on transformer eager softmax backward row-dot + dS in one serial-per-row kernel, source @ 8be4d20d4; deleted, A/B -, - ms, **DROP (unmeasured)**: same reach as NN22; one thread walks every key of its row twice, which the split flat-grid incumbent already parallelizes


## GEMM

### MOJOLEARN_APPLE_FAST_GEMM_NT_TILED

- Verdict: DROPPED-slower. Deleted 2026-10-03 by `6f3e65746` (core/gemm: remove dropped APPLE_FAST_GEMM_NT_TILED (DROPPED-slower; recover lane/apple-fast-tier@95a09d1fd)).
- Recoverable at `d23f7c759` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_APPLE_FAST_GEMM_NT_TILED.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `core/gemm.mojo`
- EXPERIMENTS.md:407 (Decomp (34)): `APPLE_FAST_GEMM_NT_TILED` on kmeans, lane/apple-fast-tier @ 95a09d1fd, A/B tier-pca-nttiled, tier-ols-nttiled, tier-kmeans-nttiled, ols istella 2,179 -> 2,666; pca istella 599.6 -> 601.1; kmeans istella 1,474 -> 1,523 ms, **DROPPED-slower**: never wins; code removed from main 6f3e65746; recover at lane/apple-fast-tier@95a09d1fd

### MOJOLEARN_APPLE_FAST_GEMM_PINNED

- Verdict: DROPPED-noise. Deleted 2026-10-03 by `75c59ea26` (gemm_identical: remove dropped APPLE_FAST_GEMM_PINNED (DROPPED-noise; recover lane/apple-fast-tier@95a09d1fd)).
- Recoverable at `6f3e65746` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_APPLE_FAST_GEMM_PINNED.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gemm/checks/gemm_identical.mojo`
- EXPERIMENTS.md:554 (Kernel / GP (11)): `APPLE_FAST_GEMM_PINNED` on nys / taxi, lane/apple-fast-tier @ 95a09d1fd, A/B tier-rbf-pinned, tier-rbf-pinned-taxi, tier-nys-pinned, rbf taxi 65.0 -> 63.9; nystroem istella 525.6 -> 522.7 ms, **DROPPED-noise**: -1.6% / -0.6%; code removed from main 75c59ea26; recover at lane/apple-fast-tier@95a09d1fd

### MOJOLEARN_BGMM_FAST_MAHAL_GEMM

- Verdict: DROPPED-slower. The code never reached main as a live switch (no code line naming it was ever deleted from main); no patch.
- Recoverable on the lane: `lane/apple-fast-cluster2 @ ded4ea07b`.
- EXPERIMENTS.md:483 (Cluster (38)): `BGMM_FAST_MAHAL_GEMM` on bayesian-gmm / taxi, lane/apple-fast-cluster2 @ ded4ea07b, A/B cluster2-bgmm-mahal-taxi, - ms, **DROPPED-slower**: reconciled 2026-10-05: cluster2-bgmm-mahal-taxi-b 309.5 -> 558.0 (+80.3%) (LEDGER 2026-10-03); code deleted at deee07721. Was OPEN: A/B queued, no judged result yet

### MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE

- Verdict: slower. Deleted 2026-10-07 by `236996cda` (gemm: delete IDN_GEMM_TILE_SHORT_K define, FS2, COMPACT_LIVE_TILE, GROUP_TILES_BODY; narrow TILE_MIN_BLOCKS to {192,512,1024}; refuse in guards; EXPERIMENTS row).
- Recoverable at `e8a0960c6` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gemm/checks/gemm_identical.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:167): removed: MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE (OVN A05 slower on NVIDIA and AMD)
- grid_controls/neural-gemm-attn-dedupe.json removed (control gemm_compact_live_tile): DELETED by lane/grid-prune 2026-10-07: OVN A05 slower on NVIDIA and AMD. Recoverable at main ab554bb4a.
- EXPERIMENTS.md:1590 (IDENTICAL grid prune: losers and dead arms deleted (lane/gri): `MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE` on shipped IDENTICAL GEMM dispatch, NVIDIA + AMD, main @ ab554bb4a, A/B OVN A05 (`overnight-ab-20261006/{nvidia/default,amd}-normalized-measurements.json`), AMD 4096x1024x1024 op1/op2 0.392 -> 0.639, 0.412 -> 0.669; AMD 127x130x4097 0.064 -> 1.83; NV k=1025 m=511 0.064 -> 0.086 ms, **DROPPED-slower, define deleted**: slower on both vendors in nearly every fixture

### MOJOLEARN_IDN_GEMM_FOLD_LEAF_64

- Verdict: slower. Deleted 2026-10-07 by `870bcd1f9` (GEMM leaf: one switch MOJOLEARN_IDN_GEMM_LEAF (neural128/neural256/all256); delete I04 leaf64 loser).
- Recoverable at `25881f493` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_GEMM_FOLD_LEAF_64.patch` (applies cleanly to main at the time of writing); first apply `experiments/removed/_shared-870bcd1f9.patch`.
- Files the patch restores: `gemm/contract.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:101): MOJOLEARN_IDN_GEMM_FOLD_LEAF_64 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json
- grid_controls/neural-gemm-attn-dedupe.json removed: I04 leaf 64: measured loser on MI325X (1.268x) and L40S; deleted, EXPERIMENTS.md row
- EXPERIMENTS.md:1465 (IDENTICAL neural GEMM/attention switch dedupe (lane/neural-g): `MOJOLEARN_IDN_GEMM_FOLD_LEAF_64` on IDENTICAL GEMM leaf 64 vs 128 / m1024 n1024 k2048 and neighbor m1023 n1025 k2049, measured @ cbcc8dcd3303; deleted from lane/neural-gemm-attn-dedupe (base 8be4d20d4), A/B overnight-ab-20261006 I04 (amd/normalized-measurements.json, nvidia/default-repair-normalized-measurements.json), MI325X 0.182 -> 0.231 (1.268x; neighbor 1.157x); L40S 0.280/0.255 -> 0.304/0.281 ms, **DROP**: slower on both voting vendors; leaf is a bit version, so no FAST-style keep. `MOJOLEARN_IDN_GEMM_LEAF` keeps arms neural128/neural256/all256

### MOJOLEARN_IDN_GEMM_FS2

- Verdict: noise. Deleted 2026-10-07 by `236996cda` (gemm: delete IDN_GEMM_TILE_SHORT_K define, FS2, COMPACT_LIVE_TILE, GROUP_TILES_BODY; narrow TILE_MIN_BLOCKS to {192,512,1024}; refuse in guards; EXPERIMENTS row).
- Recoverable at `e8a0960c6` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE.patch` (same patch as above).
- Files the patch restores: `gemm/checks/gemm_identical.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:166): removed: MOJOLEARN_IDN_GEMM_FS2 (OVN N02 noise vs FS4)
- grid_controls/neural-small.json removed (control gemm_fs2): DELETED by lane/grid-prune 2026-10-07: OVN N02 FS2 vs FS4 noise (0.996/0.995). Recoverable at main ab554bb4a.
- EXPERIMENTS.md:1589 (IDENTICAL grid prune: losers and dead arms deleted (lane/gri): `MOJOLEARN_IDN_GEMM_FS2` on NVIDIA TUNED_128 kpack calls with <= 2 fold leaves, main @ ab554bb4a, A/B OVN N02 (`~/mojolearn-evidence/overnight-ab-20261006/nvidia/specific-normalized-measurements.json`), `bounded_fs2` median 0.996 vs `current_fs4` 0.995 over 120 cells ms, **DROPPED-noise, define deleted**: FS2 vs FS4 is noise on the L40S; NVIDIA-only, inert on AMD. The "N02 NEVER RUN" comment was stale

### MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY

- Verdict: noise. Deleted 2026-10-07 by `236996cda` (gemm: delete IDN_GEMM_TILE_SHORT_K define, FS2, COMPACT_LIVE_TILE, GROUP_TILES_BODY; narrow TILE_MIN_BLOCKS to {192,512,1024}; refuse in guards; EXPERIMENTS row).
- Recoverable at `e8a0960c6` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE.patch` (same patch as above).
- Files the patch restores: `gemm/checks/gemm_identical.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:168): removed: MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY (OVN N01 noise on the L40S)
- EXPERIMENTS.md:1591 (IDENTICAL grid prune: losers and dead arms deleted (lane/gri): `MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY` on ksplit group rule, NVIDIA packed body, main @ ab554bb4a, A/B OVN N01 `body_tiles` (`overnight-ab-20261006/nvidia/specific-normalized-measurements.json`), median 1.005, 46% of 72 L40S cells faster ms, **DROPPED-noise, define deleted (EXPERIMENTS row "GROUP_TILES_BODY OPEN" above closes)**: NVIDIA-only. Conflicting earlier hint: the 2026-10-05 RTX 4090 synthetic single-sample screen read geomean 0.78 over 18 cases (0.42..1.10, `experiments/identical_speed/results/20261005/nvidia-screen.json`); the same screen put slack2 at 0.86 with min 0.34, so it is a noisy screen; the L40S N01 read is the board box. Re-open from ab554bb4a if a board-shape A/B wants it

### MOJOLEARN_IDN_GEMM_OZAKI_LINALG

- Verdict: slower. Deleted 2026-10-08 by `22187de78` (grid-act-2: delete gemm_ozaki_linalg (both arms; grid ge123e6f9 gemm S=4 3.97x/2.06x, S=5 7.99x/2.51x slower); refuse the define; tombstones).
- Recoverable at `deae02e73` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_GEMM_OZAKI_LINALG.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `gemm/host_entry.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:198): removed: MOJOLEARN_IDN_GEMM_OZAKI_LINALG retired 2026-10-08: slower, gemm S=4 NV 3.97x / AMD 2.06x, S=5 NV 7.99x / AMD 2.51x (grid ge123e6f9); the neural Ozaki profile (MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES) is separate; see EXPERIMENTS.md
- grid_controls/neural-small.json removed (control gemm_ozaki_linalg): DELETED 2026-10-08 (lane grid-act-2): IDENTICAL grid ge123e6f9 loser, both arms. arm/off ms ratio NV/AMD on gemm gaussian: S=4 3.972/2.061 (69.6 -> 276.6 NV, 28.5 -> 58.8 AMD), S=5 7.986/2.510 (-> 556.2 NV, 71.6 AMD); combined 2.86x / 4.48x SLOWER. The linalg GEMM calls gemm_identical directly again (gemm/host_entry.mojo); MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES (the neural Ozaki profile) is untouched. Code recoverable at main 42d1e42c6; define refused in core/six_lane_experiment_guards.mojo.

### MOJOLEARN_IDN_GEMM_TILE_SHORT_K

- Verdict: slower. Deleted 2026-10-07 by `236996cda` (gemm: delete IDN_GEMM_TILE_SHORT_K define, FS2, COMPACT_LIVE_TILE, GROUP_TILES_BODY; narrow TILE_MIN_BLOCKS to {192,512,1024}; refuse in guards; EXPERIMENTS row).
- Recoverable at `e8a0960c6` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE.patch` (same patch as above).
- Files the patch restores: `gemm/checks/gemm_identical.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:165): removed: MOJOLEARN_IDN_GEMM_TILE_SHORT_K (arms equal measured pass62 losers / the NVIDIA default; inert on AMD)
- grid_controls/neural-small.json removed (control gemm_tile_short_k): DELETED by lane/grid-prune 2026-10-07 (code + define; refused in core/six_lane_experiment_guards.mojo): arms equal measured pass62 losers or the NVIDIA column default; inert on AMD. Recoverable at main ab554bb4a.
- EXPERIMENTS.md:1588 (IDENTICAL grid prune: losers and dead arms deleted (lane/gri): `MOJOLEARN_IDN_GEMM_TILE_SHORT_K` on every gemm_identical tuned-tile caller (gemm, neural, classical), NVIDIA + AMD, main @ ab554bb4a, A/B pass62 (gemm_identical.mojo:4042-4053), q/o 0.104/0.084 -> 0.051/0.049 ms a layer with the short rule; k 1024..2048 +0.2 ms a layer ms, **DROPPED, define deleted (arms equal measured losers / A/A)**: arm 512 = NVIDIA column default (A/A); arm 0 = pre-pass62 state and arm 1024 = long-k step-down, both measured slower by pass62; every arm inert on AMD (min k 0). Env knob `MOJOLEARN_GEMM_TILE_SHORT_K` and the column default stay

### MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE=1

- Verdict: slower. Deleted 2026-10-09 by `d48c5eaaf` (grid-act-5: delete the neural_gemm_epilogue mlp arm (MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE bit 1, NN06; grid ge123e6f9 mlp-train-step NV 1.9x / AMD 0.97x, 1.367x c).
- Recoverable at `f3d27d1e2` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE-arm1.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gemm/experiments/neural_epilogue.mojo`, `gemm/experiments/neural_tiled_v2.mojo`, `training/mlp_ops.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:220): removed 2026-10-08 (grid ge123e6f9, lane/grid-act-5): neural_gemm_epilogue=mlp (MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE bit 1, NN06 fused MLP GEMM + bias/ReLU) was SLOWER: mlp-train-step NV 1.9x (2.0 -> 3.8 ms) / AMD 0.97x (3.5 -> 3.4 ms), 1.367x combined; same bits; only =2 (cnn) remains; code at main 8ed94710a, see docs/apple-fast/EXPERIMENTS.md
- grid_controls/neural-gemm-attn-dedupe.json removed: removed 2026-10-08 (lane grid-act-5): neural_gemm_epilogue=mlp (NN06 fused MLP GEMM + bias/ReLU, gemm/experiments/neural_epilogue.mojo and training/mlp_ops.mojo _mlp_fused_projection) grid ge123e6f9 loser, mlp-train-step NV 2.0->3.8 / AMD 3.5->3.4 ms (1.367x combined slower), same bits; guard refuses bit 1; EXPERIMENTS.md row; code at main 8ed94710a
- EXPERIMENTS.md:1747 (IDENTICAL grid ge123e6f9: gemm_leaf split for the MLP step a): `MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE=1` on neural:mlp-train-step / gaussian, lane/grid-act-5 from main @ 8ed94710a, A/B ge123e6f9 (nv n0501, amd a0824), NV 2.0 -> 3.8 (1.90), AMD 3.5 -> 3.4 (0.97); 1.367x combined ms, **DELETED (loser; AMD flat, NV 1.9x)**: the forward's GEMM + bias (+ ReLU) as one thread-per-cell kernel loses the incumbent GEMM + separate bias launch on NVIDIA and gains nothing on AMD; same bits. `gemm/experiments/neural_epilogue.mojo` (its only caller was this arm) and `training/mlp_ops.mojo`'s `_mlp_gemm_epilogue_kernel` / `_mlp_fused_projection` removed; tombstones in mlp_ops.mojo and neural_tiled_v2.mojo

### MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE=3

- Verdict: slower. Deleted 2026-10-08 by `7663b5335` (grid-act-3: delete the neural_gemm_schedule stream_all arm (MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE=3, NI02; grid ge123e6f9 gemm 2.07x/1.78x, lm-train-step 1.11x/1.0).
- Recoverable at `89a183889` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE-arm3.patch` (applies cleanly to main at the time of writing).
- Files the patch restores: `gemm/checks/gemm_identical.mojo`, `gemm/experiments/neural_switches.mojo`, `x_cnn/device.mojo`
- What happened: stream_all arm (NI02): grid ge123e6f9 gemm NV 2.07x / AMD 1.78x slower (lane grid-act-3); see the commit message and EXPERIMENTS.md.
- grid_controls/neural-gemm-attn-dedupe.json removed (control neural_gemm_schedule): DELETED 2026-10-08 (lane grid-act-3): IDENTICAL grid ge123e6f9 loser. stream_all/off ms ratio NV/AMD: gemm gaussian 2.07/1.78 (61.9 -> 128.4 NV, 28.5 -> 50.7 AMD), lm-train-step 1.11/1.07 (36.0 -> 40.0 NV, 42.7 -> 45.7 AMD); 1.45x combined SLOWER; quality not judged. NI02 (gemm/checks/gemm_identical.mojo _ni02_* kernels, dispatch, workspace sizing, identical_gemm_streaming_applies and the x_cnn pre-check) deleted; geometry, stream, stream_exact, pages, cost, fold_exact, threadmap, pages_threadmap stay. Code recoverable at main bc10b8b56; =3 refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1726 (IDENTICAL grid ge123e6f9: one promotion and six loser contro): `MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE=3` on gemm:gemm / gaussian; neural:lm-train-step / bytes, main @ bc10b8b56 (deleted on lane/grid-act-3), A/B ge123e6f9 (nv n0458/v0189, amd a0722/a0592), gemm NV 61.9 -> 128.4 (2.07), AMD 28.5 -> 50.7 (1.78); lm-train-step NV 36.0 -> 40.0 (1.11), AMD 42.7 -> 45.7 (1.07); 1.45x combined SLOWER ms, **DROP (slower), arm deleted**: bounded 16-leaf partial-plane windows for every GEMM caller: twice the launches and a carry round trip per window; quality not judged. `_ni02_*` kernels, the dispatch and workspace sizing branches, `identical_gemm_streaming_applies` and the x_cnn pre-check deleted; =3 refused; the other schedule arms (unmeasured) stay

### MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE=8

- Verdict: slower. Deleted 2026-10-07 by `225398f14` (neural gemm: delete schedule arm 8 (async, NN08): SCHED_ASYNC, NN08, neural_async_ab; guards refuse 8; EXPERIMENTS row).
- Recoverable at `236996cda` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE-arm8.patch` (needs `git apply -3` (main moved on)).
- Files the patch restores: `gemm/experiments/neural_plans.mojo`, `gemm/experiments/neural_streaming.mojo`, `gemm/experiments/neural_switches.mojo`, `gemm/experiments/neural_tiled.mojo`, `gemm/neural_dispatch.mojo`
- What happened: async arm (NN08): OVN N03 NVIDIA async pipeline 2.2x/4.2x slower than its sync control (lane grid-prune).
- grid_controls/neural-gemm-attn-dedupe.json removed (control neural_gemm_schedule): DELETED by lane/grid-prune 2026-10-07: OVN N03 NVIDIA async pipeline 2.2x/4.2x slower than its sync control; guards refuse arm 8. Recoverable at main ab554bb4a.
- EXPERIMENTS.md:1592 (IDENTICAL grid prune: losers and dead arms deleted (lane/gri): `MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE=8` on neural GEMM callers (lm, transformer, mamba, samba, training), NVIDIA, main @ ab554bb4a, A/B OVN N03 `async_operand_pipeline_check` (`overnight-ab-20261006/nvidia/specific-repair-normalized-measurements.json`), 0.185 -> 0.411 ms and 2.76 -> 11.69 ms vs its synchronous control ms, **DROPPED-slower, arm deleted (`SCHED_ASYNC`, `NN08`, `neural_async_ab`)**: NVIDIA ran `pipeline_gemm[True]` 2.2x / 4.2x slower; AMD ran the sync control (A/A). `gemm/experiments/async_operand_pipeline.mojo` stays for its standalone check


## Other

### MOJOLEARN_IDN_NN20_SPLIT_KV

- Verdict: slower. Deleted 2026-10-08 by `89a183889` (grid-act-3: delete the attn_softmax summary_tree arm (MOJOLEARN_IDN_ATTN_SOFTMAX=1, NN20, and its split-KV controls; grid ge123e6f9 lm/samba/transformer 1.6-43x).
- Recoverable at `3d12702e7` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_NN20_SPLIT_KV.patch` (applies cleanly to main at the time of writing); first apply `experiments/removed/_shared-89a183889.patch`.
- Files the patch restores: `transformer/experiments/attention_summary_split.mojo`, `transformer/experiments/summary_model.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:208): removed: MOJOLEARN_IDN_NN20_SPLIT_KV (and _LEAVES) retired 2026-10-08 with the summary_tree arm it split (MOJOLEARN_IDN_ATTN_SOFTMAX=1, grid ge123e6f9 loser); see EXPERIMENTS.md
- grid_controls/neural-fusions.json removed (control nn20_split_kv): DELETED 2026-10-08 (lane grid-act-3) with its parent arm attn_softmax=summary_tree (MOJOLEARN_IDN_ATTN_SOFTMAX=1, IDENTICAL grid ge123e6f9 loser, 1.6-43x slower NV/AMD): the split-KV code lived in transformer/experiments/attention_summary_split.mojo and summary_model.mojo, both deleted. Never measured on its own. Code recoverable at main bc10b8b56; define refused in core/six_lane_experiment_guards.mojo.
- EXPERIMENTS.md:1725 (IDENTICAL grid ge123e6f9: one promotion and six loser contro): `MOJOLEARN_IDN_NN20_SPLIT_KV` on neural:lm-forward, lm-train-step, samba-forward, samba-train-step, transformer-forward / bytes, gaussian, main @ bc10b8b56 (deleted on lane/grid-act-3), A/B ge123e6f9 (nv v0269/v0177/n0561, amd a0670/a0576/a0580), lm-forward NV 65.8 -> 319.9 (4.86), AMD 34.2 -> 574.2 (16.79); lm-train-step NV 36.0 -> 1542.8 (42.89), AMD 42.7 -> 1717.1 (40.18); samba-forward NV 12.0 -> 21.4 (1.79), AMD 8.8 -> 42.3 (4.79); samba-train-step NV 96.0 -> 157.6 (1.64), AMD 147.2 -> 266.5 (1.81); transformer-forward NV 7.3 -> 40.0 (5.48), AMD 3.2 -> 70.9 (21.89); 7.30x combined SLOWER ms, **DROP (slower), code deleted**: balanced 32-key summary-tree softmax: every cell slower on both vendors; mean_nll not judged. transformer/experiments/attention_summary_{contract,tree,split}.mojo and summary_model{,_contract,_host}.mojo deleted with their call sites (modeling_llama eager attention, transformer_backward, the two oracles, profile labels, byte_lm_host); `=1` and the split defines refused; online_tile32 (=2, HOLD_BROKEN) stays

### MOJOLEARN_IDN_NN20_SPLIT_KV_LEAVES

- Verdict: slower. Deleted 2026-10-08 by `89a183889` (grid-act-3: delete the attn_softmax summary_tree arm (MOJOLEARN_IDN_ATTN_SOFTMAX=1, NN20, and its split-KV controls; grid ge123e6f9 lm/samba/transformer 1.6-43x).
- Recoverable at `3d12702e7` (the deletion commit's parent). Patch: `experiments/removed/MOJOLEARN_IDN_NN20_SPLIT_KV_LEAVES.patch` (applies cleanly to main at the time of writing); first apply `experiments/removed/_shared-89a183889.patch`.
- Files the patch restores: `transformer/experiments/attention_summary_split.mojo`
- Guard refusal (core/six_lane_experiment_guards.mojo:208): removed: MOJOLEARN_IDN_NN20_SPLIT_KV (and _LEAVES) retired 2026-10-08 with the summary_tree arm it split (MOJOLEARN_IDN_ATTN_SOFTMAX=1, grid ge123e6f9 loser); see EXPERIMENTS.md
- grid_controls/neural-fusions.json removed (control nn20_split_kv_leaves): DELETED 2026-10-08 (lane grid-act-3) with its parent arm attn_softmax=summary_tree (MOJOLEARN_IDN_ATTN_SOFTMAX=1, IDENTICAL grid ge123e6f9 loser, 1.6-43x slower NV/AMD): the split-KV code lived in transformer/experiments/attention_summary_split.mojo and summary_model.mojo, both deleted. Never measured on its own. Code recoverable at main bc10b8b56; define refused in core/six_lane_experiment_guards.mojo.

## Renamed or merged (code kept under the new name; no patch)

| old define | successor / reason | changed by |
|---|---|---|
| `MOJOLEARN_ATTN_V1_ALIAS_Y_ESTASH` | merged into attn_stash=alias_y; the old define is refused by core/six_lane_experiment_guards.mojo | `7d49e10d3` |
| `MOJOLEARN_ATTN_V1_PACKED_ESTASH` | merged into attn_stash=packed; the old define is refused by core/six_lane_experiment_guards.mojo | `7d49e10d3` |
| `MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD` | merged into attn_stash=recompute; the old define is refused by core/six_lane_experiment_guards.mojo | `7d49e10d3` |
| `MOJOLEARN_C29_TILE_128` | merged into MOJOLEARN_C29_TILE=128 | `58146b270` |
| `MOJOLEARN_C30_DIRECT_DISTANCE` | spanned ~12 algorithms; split into KMEANS_DIRECT_DISTANCE (arm of KMEANS_ROW_ASSIGN) and KNN/KDE/DBSCAN/GRAPH/IVF_DIRECT_DISTANCE | `d7c9736f0` |
| `MOJOLEARN_C30_ROWS_4` | same knob as C36_ROWS_4; now KMEANS_ROW_ASSIGN=4 | `d7c9736f0` |
| `MOJOLEARN_C36_CENTROID_TILES` | same kernel as C30 in kmeans; now KMEANS_ROW_ASSIGN (cluster) and XCLUSTER_ROW_ASSIGN (x_cluster) | `d7c9736f0` |
| `MOJOLEARN_C36_ROWS_4` | now the =4 arm of KMEANS_ROW_ASSIGN / XCLUSTER_ROW_ASSIGN | `d7c9736f0` |
| `MOJOLEARN_C52_PAIR_128` | merged into MOJOLEARN_C52_PAIR_ROWS=128 | `79aa4a19c` |
| `MOJOLEARN_C52_PAIR_512` | merged into MOJOLEARN_C52_PAIR_ROWS=512 | `79aa4a19c` |
| `MOJOLEARN_C56_QDA_PROJECT4` | renamed into the arms switch MOJOLEARN_CLASSICAL_C56_QDA_PROJECT=4 | `213f349ed` |
| `MOJOLEARN_C58_TEAM64` | merged into MOJOLEARN_C58_TEAM_MIB=64 | `f7abb25cf` |
| `MOJOLEARN_CLASSICAL_C01_LEAF128` | replaced by MOJOLEARN_CLASSICAL_C01_LEAF=128 and MOJOLEARN_CLASSICAL_C01_MEAN | `79aca50c9` |
| `MOJOLEARN_CLASSICAL_C01_LEAF64` | replaced by MOJOLEARN_CLASSICAL_C01_LEAF=64 (metrics) and MOJOLEARN_CLASSICAL_C01_MEAN (mean kernel); serial one-thread-per-column mean deleted | `79aca50c9` |
| `MOJOLEARN_CLASSICAL_C04_LOAD_CENTER` | split: PCA use is MOJOLEARN_CLASSICAL_PCA_COV=4 (rewritten row-parallel), LDA use is MOJOLEARN_CLASSICAL_C04_LDA | `79aca50c9` |
| `MOJOLEARN_CLASSICAL_C06_ROWS2` | now MOJOLEARN_CLASSICAL_C06_NORM_ROWS=2 | `ef89c6992` |
| `MOJOLEARN_CLASSICAL_C06_ROWS4` | now MOJOLEARN_CLASSICAL_C06_NORM_ROWS=4 | `ef89c6992` |
| `MOJOLEARN_CLASSICAL_C07_DIGIT4` | merged into int sweep MOJOLEARN_CLASSICAL_C07_RADIX_BITS=4 (silently overrode DIGIT6) | `0376bce1c` |
| `MOJOLEARN_CLASSICAL_C07_DIGIT6` | merged into int sweep MOJOLEARN_CLASSICAL_C07_RADIX_BITS=6 | `0376bce1c` |
| `MOJOLEARN_CLASSICAL_C07_KEYS1024` | merged into int sweep MOJOLEARN_CLASSICAL_C07_RADIX_ROWS=1024 | `0376bce1c` |
| `MOJOLEARN_CLASSICAL_C07_KEYS4096` | merged into int sweep MOJOLEARN_CLASSICAL_C07_RADIX_ROWS=4096 (silently overrode KEYS1024) | `0376bce1c` |
| `MOJOLEARN_CLASSICAL_C08_DICTIONARY` | spanned three routes; split into C08_TARGET_CODES / C08_ONEHOT_FT / C08_ORDINAL_FT; its one-thread-per-column binary-search device form is replaced by the parallel x_prep/ddict.mojo (C08_DICTIONARY survives only as a derived comptime, not a define) | `0376bce1c` |
| `MOJOLEARN_CLASSICAL_C19_ORDERED_128` | merged into MOJOLEARN_CLASSICAL_C19_SGD_CHUNK=128 (guard rejects the old name) | `79aa4a19c` |
| `MOJOLEARN_CLASSICAL_C19_ORDERED_32` | merged into MOJOLEARN_CLASSICAL_C19_SGD_CHUNK=32 | `79aa4a19c` |
| `MOJOLEARN_CLASSICAL_C23_CENTERED_PANELS` | split: PCA use is MOJOLEARN_CLASSICAL_PCA_COV=23 (one-pass Chan rewrite), MCD use is MOJOLEARN_CLASSICAL_C23_MCD; no LDA/QDA code ever read it; per-cell serial kernel deleted | `79aca50c9` |
| `MOJOLEARN_IDN_ATTENTION_V2` | merged into attn_softmax=online_tile32; the old define is refused by core/six_lane_experiment_guards.mojo | `8058b2554` |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_1` | five mutually overriding defines merged into the int sweep MOJOLEARN_IDN_KMEANS_CONV_CHUNK=1\|2\|4\|8\|16\|32 (control kmeans_conv_chunk); the guard refuses the old names (core/six_lane_experiment_guards.mojo:31). Same loop per value. | `68c97e338` |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_16` | five mutually overriding defines merged into the int sweep MOJOLEARN_IDN_KMEANS_CONV_CHUNK=1\|2\|4\|8\|16\|32 (control kmeans_conv_chunk); the guard refuses the old names (core/six_lane_experiment_guards.mojo:31). Same loop per value. | `68c97e338` |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_2` | five mutually overriding defines merged into the int sweep MOJOLEARN_IDN_KMEANS_CONV_CHUNK=1\|2\|4\|8\|16\|32 (control kmeans_conv_chunk); the guard refuses the old names (core/six_lane_experiment_guards.mojo:31). Same loop per value. | `68c97e338` |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_32` | five mutually overriding defines merged into the int sweep MOJOLEARN_IDN_KMEANS_CONV_CHUNK=1\|2\|4\|8\|16\|32 (control kmeans_conv_chunk); the guard refuses the old names (core/six_lane_experiment_guards.mojo:31). Same loop per value. | `68c97e338` |
| `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_4` | five mutually overriding defines merged into the int sweep MOJOLEARN_IDN_KMEANS_CONV_CHUNK=1\|2\|4\|8\|16\|32 (control kmeans_conv_chunk); the guard refuses the old names (core/six_lane_experiment_guards.mojo:31). Same loop per value. | `68c97e338` |
| `MOJOLEARN_IDN_LM_OWNED_TOKENS` | now MOJOLEARN_IDN_LM_RESIDENT_TOKENS=1 | `a87789e58` |
| `MOJOLEARN_IDN_LM_PARAM_VIEWS` | now MOJOLEARN_IDN_LM_VIEWS=3 | `a87789e58` |
| `MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2` | now MOJOLEARN_IDN_CE_TOKEN_FOLD=2 | `a87789e58` |
| `MOJOLEARN_IDN_M1_PERSISTENT_SCAN` | MERGED into m1_scan (-D MOJOLEARN_IDN_M1_SCAN=1\|2\|3) by lane/grid-prune 2026-10-07; old define refused in core/six_lane_experiment_guards.mojo | `a2b75bb65` |
| `MOJOLEARN_IDN_M1_STATE_WINDOW` | MERGED into m1_scan (-D MOJOLEARN_IDN_M1_SCAN=1\|2\|3) by lane/grid-prune 2026-10-07; old define refused in core/six_lane_experiment_guards.mojo | `a2b75bb65` |
| `MOJOLEARN_IDN_M2_GRAD_LEAF128` | now MOJOLEARN_IDN_M2_GRAD_FOLD=2 | `83b9bf20a` |
| `MOJOLEARN_IDN_M2_YOFF_EXP_CACHE` | now MOJOLEARN_IDN_M2_YOFF_EXP=2 | `83b9bf20a` |
| `MOJOLEARN_IDN_NEURAL_NN01` | merged into neural_gemm_schedule=geometry; the old define is refused by core/six_lane_experiment_guards.mojo | `426564404` |
| `MOJOLEARN_IDN_NEURAL_NN02` | merged into neural_gemm_schedule=stream; the old define is refused by core/six_lane_experiment_guards.mojo | `426564404` |
| `MOJOLEARN_IDN_NEURAL_NN03` | merged into gemm_leaf=neural128\|neural256; the old define is refused by core/six_lane_experiment_guards.mojo | `870bcd1f9` |
| `MOJOLEARN_IDN_NEURAL_NN04` | merged into neural_gemm_chains (define CHAINS); the old define is refused by core/six_lane_experiment_guards.mojo | `870bcd1f9` |
| `MOJOLEARN_IDN_NEURAL_NN06` | merged into neural_gemm_epilogue=mlp; the old define is refused by core/six_lane_experiment_guards.mojo | `f96f2f28b` |
| `MOJOLEARN_IDN_NEURAL_NN08` | merged into neural_gemm_schedule=async; the old define is refused by core/six_lane_experiment_guards.mojo | `426564404` |
| `MOJOLEARN_IDN_NEURAL_NN09` | merged into neural_gemm_schedule=pages; the old define is refused by core/six_lane_experiment_guards.mojo | `426564404` |
| `MOJOLEARN_IDN_NEURAL_NN10` | merged into neural_gemm_schedule=cost; the old define is refused by core/six_lane_experiment_guards.mojo | `426564404` |
| `MOJOLEARN_IDN_NEURAL_NN11` | merged into neural_gemm_schedule=fold_exact (with NN02: stream_exact); the old define is refused by core/six_lane_experiment_guards.mojo | `426564404` |
| `MOJOLEARN_IDN_NEURAL_NN15` | merged into neural_gemm_schedule=threadmap (with NN09: pages_threadmap); the old define is refused by core/six_lane_experiment_guards.mojo | `426564404` |
| `MOJOLEARN_IDN_RF_ROWS_SORTED` | duplicate of MOJOLEARN_TREES_RF_SAMPLE=1 (sorted bootstrap rows); define deleted | `d631a057c` |
| `MOJOLEARN_IDN_RMS_ROW_BLOCK` | merged into norm=row_block; the old define is refused by core/six_lane_experiment_guards.mojo | `a1488ca2d` |
| `MOJOLEARN_IDN_ROPE_CACHE` | merged into rope=k_cache; the old define is refused by core/six_lane_experiment_guards.mojo | `a1488ca2d` |
| `MOJOLEARN_IDN_SAMBA_FORWARD_TAPE` | now MOJOLEARN_IDN_ACT_RETAIN=3 (NI48; untangled from NI36) | `a87789e58` |
| `MOJOLEARN_IDN_SEQ_LN_LEAF32` | now MOJOLEARN_IDN_SEQ_LN_LEAF=32 | `8fe3f695e` |
| `MOJOLEARN_IDN_SEQ_WGRAD_LEAF256` | now MOJOLEARN_IDN_SEQ_WGRAD=1 | `8fe3f695e` |
| `MOJOLEARN_IDN_TRAIN_BACKWARD_SCRATCH` | now MOJOLEARN_IDN_TRAIN_SCRATCH=1 (NI36) | `a87789e58` |
| `MOJOLEARN_NI02_GEMM_STREAM_PARTIALS` | merged into neural_gemm_schedule=stream_all; the old define is refused by core/six_lane_experiment_guards.mojo | `426564404` |
| `MOJOLEARN_NI08_GEMM_LEAF_256` | merged into gemm_leaf=all256; the old define is refused by core/six_lane_experiment_guards.mojo | `870bcd1f9` |
| `MOJOLEARN_NI09_TILED_BIAS` | merged into neural_gemm_epilogue=cnn; the old define is refused by core/six_lane_experiment_guards.mojo | `f96f2f28b` |
| `MOJOLEARN_NI14_BOUNDED_COL2IM` | MERGED into ni14_col2im (-D MOJOLEARN_NI14_COL2IM=1\|2) by lane/grid-prune 2026-10-07; old define refused in core/six_lane_experiment_guards.mojo | `a2b75bb65` |
| `MOJOLEARN_NI14_TILED_COL2IM` | MERGED into ni14_col2im (-D MOJOLEARN_NI14_COL2IM=1\|2) by lane/grid-prune 2026-10-07; old define refused in core/six_lane_experiment_guards.mojo | `a2b75bb65` |
| `MOJOLEARN_NI16_CONV_RELU_FUSED` | now MOJOLEARN_IDN_CNN_CONV_RELU=2 | `33482aa39` |
| `MOJOLEARN_NN17_GQA_FOUR_HEADS` | merged into attn_head_share=4; the old define is refused by core/six_lane_experiment_guards.mojo | `52aecf2f3` |
| `MOJOLEARN_NN20_BALANCED_SUMMARY_TREE` | merged into attn_softmax=summary_tree; the old define is refused by core/six_lane_experiment_guards.mojo | `f29bde48d` |
| `MOJOLEARN_NN24_NORM_LANES8` | merged into norm=lanes8; the old define is refused by core/six_lane_experiment_guards.mojo | `a1488ca2d` |
| `MOJOLEARN_NN25_RMS_SPLIT_SCALE` | merged into norm=split_scale; the old define is refused by core/six_lane_experiment_guards.mojo | `a1488ca2d` |
| `MOJOLEARN_NN26_TRAIN_SWIGLU` | merged into train_swiglu; the old define is refused by core/six_lane_experiment_guards.mojo | `a1488ca2d` |
| `MOJOLEARN_NN27_QK_ROPE_PAIR` | merged into rope=qk_pair; the old define is refused by core/six_lane_experiment_guards.mojo | `a1488ca2d` |
| `MOJOLEARN_NN28_DEAD_TRAINING_CACHE` | merged into train_no_decode_cache; the old define is refused by core/six_lane_experiment_guards.mojo | `a1488ca2d` |
| `MOJOLEARN_NN31_BOUNDED_CHECKPOINTS` | now MOJOLEARN_IDN_ACT_RETAIN=2 | `a87789e58` |
| `MOJOLEARN_NN32_RETAIN_FORWARD` | now MOJOLEARN_IDN_ACT_RETAIN=1 | `a87789e58` |
| `MOJOLEARN_NN33_CSTATE_P16` | now MOJOLEARN_IDN_M2_CS_PT=16 | `83b9bf20a` |
| `MOJOLEARN_NN33_YDIAG_ROWS8` | now MOJOLEARN_IDN_M2_YD_ROWS=8 | `83b9bf20a` |
| `MOJOLEARN_NN34_AFFINE_PREFIX` | MERGED into m1_scan (-D MOJOLEARN_IDN_M1_SCAN=1\|2\|3) by lane/grid-prune 2026-10-07; old define refused in core/six_lane_experiment_guards.mojo | `a2b75bb65` |
| `MOJOLEARN_NN36_SHARED_DECAY` | now MOJOLEARN_IDN_M2_YOFF_EXP=1 | `0b8b1c664` |
| `MOJOLEARN_NN38_CACHE_SUFFIX_SEEDS` | duplicate of NI43; use MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE (guard refuses) | `83b9bf20a` |
| `MOJOLEARN_NN39_M2_GRAD_TREE` | now MOJOLEARN_IDN_M2_GRAD_FOLD=1 | `b4be5b1ba` |
| `MOJOLEARN_NN43_WGRAD_FIXED128` | now MOJOLEARN_IDN_SEQ_WGRAD=2 | `8fe3f695e` |
| `MOJOLEARN_NN44_STABLE_GROUP` | same kernel as NI53; use MOJOLEARN_IDN_MOE_STABLE_PACK | `0a0c7205f` |
| `MOJOLEARN_NN45_CONV_RELU` | now MOJOLEARN_IDN_CNN_CONV_RELU=1 | `33482aa39` |
| `MOJOLEARN_NN51_RESIDENT_TOKEN_VALIDATION` | now MOJOLEARN_IDN_LM_RESIDENT_TOKENS=2 | `a87789e58` |
| `MOJOLEARN_NN52_CE_WEIGHT_GRAD` | same fused kernel as NI33; use MOJOLEARN_IDN_CE_GRAD_FUSED | `a87789e58` |
| `MOJOLEARN_NN54_LOSS_PROFILE` | now MOJOLEARN_IDN_CE_TOKEN_FOLD=1 | `a87789e58` |
| `MOJOLEARN_NN55_BLOCK_STATUS` | now MOJOLEARN_IDN_LM_GROUPED_ADAM=2 (it always needed NN56) | `a87789e58` |
| `MOJOLEARN_NN56_GROUPED_ADAM` | now MOJOLEARN_IDN_LM_GROUPED_ADAM=1 | `a87789e58` |
| `MOJOLEARN_NN60_BLOCK_VIEWS` | now MOJOLEARN_IDN_LM_VIEWS=1 | `a87789e58` |
| `MOJOLEARN_NN60_EMB_HEAD_VIEWS` | now MOJOLEARN_IDN_LM_VIEWS=2 | `a87789e58` |
| `MOJOLEARN_NN62_LIFETIME_ARENA` | now MOJOLEARN_IDN_TRAIN_SCRATCH=2 | `a87789e58` |
| `MOJOLEARN_TREES_HIST_REP_BPSM` | MERGED into HIST_REP_SM by lane/grid-prune 2026-10-07 (arm device_bpsm4 = MOJOLEARN_TREES_HIST_REP_SM=1); the define is refused in core/six_lane_experiment_guards.mojo. | `c1fa5c7d3` |
| `MOJOLEARN_TREES_T10` | folded into MOJOLEARN_TREES_RF_SAMPLE=2 | `8c2545887` |
| `MOJOLEARN_TREES_T11` | T11 is on iff MOJOLEARN_TREES_T11_LEVELS is defined | `18695d4e2` |
| `MOJOLEARN_TREES_T17` | replaced by int sweep MOJOLEARN_TREES_T17_BATCH | `359df2d05` |
| `MOJOLEARN_TREES_T21_STREAM` | replaced by MOJOLEARN_TREES_T21_STREAMS=2 | `359df2d05` |
| `MOJOLEARN_TREES_T21_STREAM4` | replaced by MOJOLEARN_TREES_T21_STREAMS=4 | `359df2d05` |
| `MOJOLEARN_TREES_T31_PACKED_B` | duplicate of MOJOLEARN_FOREST_SEPARATE_NODES | `bad9c74ce` |
| `MOJOLEARN_TREES_T35_LEAF_REUSE` | MOJOLEARN_TREES_FOREST_ROUTE=1 | `3900e97e0` |
| `MOJOLEARN_TREES_T36_FINITE_STAGE` | MOJOLEARN_TREES_FOREST_ROUTE=2 | `3900e97e0` |

## Promoted (a DROP row was superseded; the define became a default)

| define | record | changed by |
|---|---|---|
| `MOJOLEARN_CAGRA_FAST_IVFG` | DROPPED-quality: recall .9838 -> .9595 | `3852df59b` |
| `MOJOLEARN_CC_FAST` | removed: connected_components' batched rounds with the device relabel are the only path (lane gap-graph 2026-10-08); the per-round-wait path is gone | `2851a6627` |
| `MOJOLEARN_GBDT_CTR_FAST_FREQ` | DROPPED-noise: same as FREQ alone; SCAN adds nothing; code removed from main afda7b9ce; recover at lane/apple-fast-trees-depthwise@f743edd60 | `2411879b6` |
| `MOJOLEARN_GBDT_DW_FUSED_CHAIN` | DROPPED-noise: -0.5%, B runs straddle A; code removed from main 951bf48da; recover at lane/apple-fast-depthwise@4547e0d14 | `4547e0d14` |
| `MOJOLEARN_GBDT_DW_NO_LEVEL_SYNC` | DROPPED-noise: -0.5%, B runs straddle A; code removed from main 951bf48da; recover at lane/apple-fast-depthwise@4547e0d14 | `951bf48da` |
| `MOJOLEARN_GMM_FAST_BIG_CHOL` | DROPPED-slower: +37.3%; GMM_FAST_ESTEP_STACK: code removed from main 7b2638a38; recover at lane/apple-fast-linear@1c7c213f8; GMM_FAST_GRID_COV: code removed from main dc1b4cc03; recover at lane/apple-fast-linear@1c7c213f8 | `1c7c213f8` |
| `MOJOLEARN_HDB_LINKAGE_DEVICE` | DROPPED-noise: rec-misc 2026-10-04: hdbscan taxi B 432.6 vs board 434 (-0.3%, inside noise); still opt-in on main (no default), so a dead toggle for the cleanup lane | `429622bef` |
| `MOJOLEARN_NC_FAST_CLS1_LABELS` | DROPPED-noise: worse than LABELS alone | `9decae29f` |
| `MOJOLEARN_OPT_FAST_STREAM` | DROPPED-slower (alone): kept only as the pair with OPT_RAW_UP (row above) | `52cd63cf8` |
| `MOJOLEARN_OPT_RAW_UP` | DROPPED-noise alone; KEEP 2026-10-04 as the pair with OPT_FAST_STREAM (see rec-optim table): stays opt-in | `52cd63cf8` |
| `MOJOLEARN_PQ_SCAN_FUSED` | DELETED (loser): recall@10 equal, same digests on AMD; the tiled score + select chain stays IDENTICAL's scan | `b6ad3fd1f` |
| `MOJOLEARN_SEQ_FAST_VAR_ONECOPY` | DROPPED-noise: SPEC adds nothing | `dfbdc7943` |

## Out of the grid, code kept on purpose

| define | note |
|---|---|
| `MOJOLEARN_ATTN_DKDV_MFMA` | not gridded on AMD (no-op at the AMD tile rule); code kept; Not gridded (AMD): engages only at BJ == 32 (fused_attention.mojo dkdv launch) while the AMD tile rule picks 16, so it silently no-ops on main; needs a BJ 16 body or a pairing first. |
| `MOJOLEARN_ATTN_DQ_MFMA` | not gridded on AMD (kernel aborts rc 134); code kept, repair before gridding; Not gridded (AMD): the dq MFMA kernel aborts (rc 134) on the #65 stack (bench/results/pr63-amd-mfma-20261001 SUMMARY.md:17-19); repair before gridding. |
| `MOJOLEARN_CLASSICAL_OLS_CENTER_RESIDENT` | never added (the resident centered TSQR is already the default); not added: the resident centered TSQR is already the IDENTICAL default (x_decomp/api.mojo:585 IDN_OLS_ONE_ENTRY); measured instead through its existing _OFF arm (control ols_one_entry). |
| `MOJOLEARN_FA_ITER_DEVICE` | a later EXPERIMENTS row keeps it (KEEP/KEPT/DEFAULT); the DROP row is superseded; DROPPED-slower: reconciled 2026-10-05: never timed alone on current main; FA_ALL (includes EIG_SMALL) rab6-faqfix istella 10300.89 -> 20530.81 (+99.3%), slowdown from the one-threadgroup EIG_SMALL eigh (Verdicts batch 4 FA_ALL row; x_decomp/fa_fast.mojo `#:`); |
| `MOJOLEARN_GEMM_KPACK_CPT4` | no-op on NVIDIA, inert on AMD; removed from the grid, code kept; No-op on NVIDIA (lib_gemm_kpack_narrow_for already selects it), inert on AMD. |
| `MOJOLEARN_IDN_ATTN_TILE_ORDER` | desc order is main; removed from the grid, code kept; desc is main (DEVIATION 2900 _bswz); paired not built (EXPERIMENTS.md). |
| `MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS` | no board effect; not registered in the grid, code kept; not registered: no board effect. The board HDBSCAN input is 100,000 rows (tools/classical_two_datasets.py:211 HDBSCAN_ROWS), above PAIRWISE_MAX_ROWS = 46340, so graph=auto already takes the sparse arm (single_linkage.mojo:201-205), and the define's legal range |
| `MOJOLEARN_KDE_DIMTILE` | a later EXPERIMENTS row keeps it (KEEP/KEPT/DEFAULT); the DROP row is superseded; DROP: inconclusive on taxi, no istella gain over DIMTILE; opt-in only |
| `MOJOLEARN_PL_GROUP_NARROW` | a later EXPERIMENTS row keeps it (KEEP/KEPT/DEFAULT); the DROP row is superseded; DROPPED-noise: reconciled 2026-10-05: old-base result, row `PL_GROUP_NARROW + PL_PAIRS_ONCE` DROPPED-noise (LEDGER 2026-10-03/04). Was OPEN: A/B queued (lane/apple-fast-batch prebuilt arms) |
| `MOJOLEARN_RIDGE_FAST_CLS1_CODES` | a later EXPERIMENTS row keeps it (KEEP/KEPT/DEFAULT); the DROP row is superseded; DROPPED-noise: no better than CODES alone |
| `MOJOLEARN_YETI_SEARCH_TASK16K` | a later EXPERIMENTS row keeps it (KEEP/KEPT/DEFAULT); the DROP row is superseded; DROPPED-noise: SYM_HIST_FAST part dropped (see above); code removed from main 6f5ace7fa; recover at lane/apple-fast-trees-yeti@65f551e39 |

## Owed deletions: DROP verdict recorded, code still on main

These defines have a DROPPED row in `docs/apple-fast/EXPERIMENTS.md` and no later KEEP/PROMOTED row, but code still reads them on
main ("a loser never reaches main"). Each needs a deletion lane (or a row that records why it stays). Sites are the first
non-comment reference at the time of writing.

| define | DROP row | site |
|---|---|---|
| `MOJOLEARN_AFN_OPT_FUSE_SCAN` | EXPERIMENTS.md:1415 DROPPED-noise (main) | `training/afn_optim.mojo:26` |
| `MOJOLEARN_AFN_OPT_RESIDENT_STATE` | EXPERIMENTS.md:1415 DROPPED-noise (main) | `training/afn_optim.mojo:51` |
| `MOJOLEARN_AFN_OPT_VEC4` | EXPERIMENTS.md:1415 DROPPED-noise (main) | `training/afn_optim.mojo:47` |
| `MOJOLEARN_ARIMA_FAST_CSS_SEARCH` | EXPERIMENTS.md:1416 DROPPED-quality (main a6ff25ff8 (arima-ics: search paths get device aic/bic)) | `arima/impl/fast_order_search.mojo:198` |
| `MOJOLEARN_ARIMA_FAST_D_CONCURRENT` | EXPERIMENTS.md:1409 DROPPED-slower (lane/apple-fast-s-ts) | `arima/impl/fast_order_search.mojo:140` |
| `MOJOLEARN_ARIMA_FAST_GROUPS_CONCURRENT` | EXPERIMENTS.md:1408 DROPPED-slower (lane/apple-fast-s-ts) | `arima/impl/fast_order_search.mojo:103` |
| `MOJOLEARN_ARIMA_FAST_STEPWISE` | EXPERIMENTS.md:1417 DROPPED-slower (main a6ff25ff8) | `arima/impl/fast_order_search.mojo:201` |
| `MOJOLEARN_DBSCAN_FAST_DENSEBALL` | EXPERIMENTS.md:488 DROPPED-slower (lane/apple-fast-dbscantaxi @ 1febff7df; ported lane/apple-fast-rec-misc) | `dbscan/impl/denseball.mojo:4` |
| `MOJOLEARN_EST_REUSE_PART` | EXPERIMENTS.md:156 DROPPED-BUG (auc .980 -> .930, logloss .186 -> 2.15) (lane/apple-fast-sym-est @ c8518eb52) | `gbdt/methods/leaves_estimation/apple_fast_est.mojo:20` |
| `MOJOLEARN_EST_SHRINK_FUSED` | EXPERIMENTS.md:157 DROPPED-inconclusive (-2.8% 1k old base) (lane/apple-fast-sym-est @ c8518eb52) | `gbdt/methods/leaves_estimation/apple_fast_est.mojo:32` |
| `MOJOLEARN_HDBSCAN2_ALL` | EXPERIMENTS.md:490 DROP (as a bundle) (lane/apple-fast-hdbscan2 @ 2fdb9114f) | `hdbscan/impl/detail/fast_apple.mojo:8` |
| `MOJOLEARN_HDB_CORE_TILE` | EXPERIMENTS.md:491 DROP (lane/apple-fast-batchv @ c7ede6e47) | `hdbscan/impl/detail/core_tile.mojo:5` |
| `MOJOLEARN_HDB_DEV_BORUVKA` | EXPERIMENTS.md:492 DROPPED-noise (lane/apple-fast-hdbscan2 @ 2fdb9114f) | `hdbscan/impl/cluster/detail/fast_mr_mst_device.mojo:4` |
| `MOJOLEARN_HDB_ONE_SYNC` | EXPERIMENTS.md:494 DROP (lane/apple-fast-batchv @ c8251211d) | `hdbscan/impl/detail/fast_apple.mojo:71` |
| `MOJOLEARN_HDB_SELECT_DEVICE` | EXPERIMENTS.md:495 DROPPED-noise (lane/apple-fast-hdbscan2 @ 2fdb9114f) | `hdbscan/impl/detail/fast_apple.mojo:95` |
| `MOJOLEARN_IVF_COARSE_FAISS_INIT` | EXPERIMENTS.md:1219 DROPPED-quality (lane/apple-fast-q-misc @ ab9acf0e8) | `ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo:183` |
| `MOJOLEARN_IVF_COARSE_INIT_QOLD` | EXPERIMENTS.md:1219 DROPPED-quality (lane/apple-fast-q-misc @ ab9acf0e8) | `ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo:186` |
| `MOJOLEARN_IVF_REFINE_TEAM` | EXPERIMENTS.md:326 DROPPED-noise (lane/apple-fast-batch @ 3150d75c1) | `x_ann/vsearch_fast.mojo:69` |
| `MOJOLEARN_KAPPROX_DEVICE` | EXPERIMENTS.md:555 DROPPED-quality (lane/apple-fast-kapprox @ 10d5a7970) | `x_neighbors/kapprox_dev.mojo:9` |
| `MOJOLEARN_KDE2_ALL` | EXPERIMENTS.md:327 DROP (lane/apple-fast-batch @ 3150d75c1) | `kde/impl/neighbors/kernel_density.mojo:2740` |
| `MOJOLEARN_KDE_KERNEL_VARIANTS` | EXPERIMENTS.md:330 DROP (lane/apple-fast-kde2 @ 659400b94) | `kde/impl/neighbors/kernel_density.mojo:2757` |
| `MOJOLEARN_KDE_LSE_FUSED` | EXPERIMENTS.md:331 DROP (lane/apple-fast-kde2 @ 659400b94) | `kde/impl/neighbors/kernel_density.mojo:2745` |
| `MOJOLEARN_KDE_NORM_FUSED` | EXPERIMENTS.md:332 DROP (lane/apple-fast-kde2 @ 659400b94) | `kde/impl/neighbors/kernel_density.mojo:2751` |
| `MOJOLEARN_KDE_SAMPLE_FUSED` | EXPERIMENTS.md:333 DROP (lane/apple-fast-kde2 @ 659400b94) | `kde/impl/neighbors/kernel_density.mojo:2763` |
| `MOJOLEARN_KMEANS_FAST_LAZY_SHIFT` | EXPERIMENTS.md:325 DROPPED-slower (lane/apple-fast-vsv-promote) | `cluster/impl/detail/kmeans.mojo:1498` |
| `MOJOLEARN_KMEANS_ROW_ASSIGN` | EXPERIMENTS.md:1706 DROP (slower), code deleted (main @ 42d1e42c6 (deleted on lane/grid-act-2)) | `cluster/impl/detail/classical_assignment.mojo:19` |
| `MOJOLEARN_KSHAP_FAST_OVERLAP` | EXPERIMENTS.md:1412 DROPPED-noise (lane/apple-fast-s-shap) | `python/mojolearn/_expansion_trees.py:3403` |
| `MOJOLEARN_LLE_FAST_NULL_CANON` | EXPERIMENTS.md:1414 DROPPED-quality (lane/apple-fast-s-shap) | `x_decomp/w4_fast.mojo:95` |
| `MOJOLEARN_MCD_DEVICE_CSTEPS` | EXPERIMENTS.md:451 DROPPED-quality (Oct 3; code kept opt-in `-D MOJOLEARN_MCD_DEVICE_CSTEPS` for a future correct parallel C-step) (lane/apple-fast-robust @ cfdb95e48) | `x_decomp/mcd_fast.mojo:13` |
| `MOJOLEARN_MC_CLASS_BATCH_DERIV` | EXPERIMENTS.md:165 DROPPED-noise (lane/apple-fast-sym-multi @ d2c832da0) | `gbdt/targets/kernel/multilogit.mojo:763` |
| `MOJOLEARN_MC_CLASS_BATCH_EST` | EXPERIMENTS.md:166 DROPPED-noise (lane/apple-fast-sym-multi @ d2c832da0) | `gbdt/targets/kernel/multilogit.mojo:774` |
| `MOJOLEARN_MI_ALL` | EXPERIMENTS.md:363 DROP (quality) (lane/apple-fast-miv @ 514401169) | `x_prep/device.mojo:201` |
| `MOJOLEARN_MI_FAST_FOLDS` | EXPERIMENTS.md:366 DROP (speed + quality) (lane/apple-fast-batch @ 3150d75c1) | `x_prep/device.mojo:216` |
| `MOJOLEARN_MOE_FAST_MMA` | EXPERIMENTS.md:1362 DROPPED-slower (bundle), toggles stay opt-in off (main @ 13246c64f) | `sequence/moe_mma.mojo:13` |
| `MOJOLEARN_MOE_FAST_MMA_KB32` | EXPERIMENTS.md:1362 DROPPED-slower (bundle), toggles stay opt-in off (main @ 13246c64f) | `sequence/moe_mma.mojo:45` |
| `MOJOLEARN_OPT_FAST_MAP_DOWN` | EXPERIMENTS.md:1193 DROPPED-slower (lane/apple-fast-gap-optim @ cf4513f8a (on main)) | `sequence/opt_resident.mojo:90` |
| `MOJOLEARN_OPT_FAST_PIPE_CH` | EXPERIMENTS.md:1193 DROPPED-slower (lane/apple-fast-gap-optim @ cf4513f8a (on main)) | `sequence/opt_resident.mojo:82` |
| `MOJOLEARN_OPT_FAST_RAW_DOWN` | EXPERIMENTS.md:1193 DROPPED-slower (lane/apple-fast-gap-optim @ cf4513f8a (on main)) | `sequence/opt_resident.mojo:91` |
| `MOJOLEARN_PL_PAIRS_ONCE` | EXPERIMENTS.md:174 DROPPED-noise (lane/apple-fast-sym-multi @ d2c832da0) | `gbdt/targets/kernel/pair_logit_group.mojo:130` |
| `MOJOLEARN_PSHAP_FAST_OVERLAP` | EXPERIMENTS.md:1413 DROPPED-slower (lane/apple-fast-s-shap) | `python/mojolearn/_expansion_trees.py:3530` |
| `MOJOLEARN_PTIMPUTE_ALL` | EXPERIMENTS.md:376 DROP (quality) (lane/apple-fast-batchv @ 77f1f5afb) | `x_prep/fastpt.mojo:24` |
| `MOJOLEARN_PT_COLBATCH` | EXPERIMENTS.md:377 DROP (quality) (lane/apple-fast-batchv @ 30aa43339) | `x_prep/fastpt.mojo:11` |
| `MOJOLEARN_PT_FOLD_NOX` | EXPERIMENTS.md:379 DROP (lane/apple-fast-ptimpute @ 9623cd7dc) | `x_prep/fastpt.mojo:8` |
| `MOJOLEARN_PT_FUSED_TRANSFORM` | EXPERIMENTS.md:380 DROP (quality, with COLBATCH) (lane/apple-fast-batchv) | `x_prep/fastpt.mojo:18` |
| `MOJOLEARN_PT_SPEC` | EXPERIMENTS.md:378 DROP (lane/apple-fast-batch @ 3150d75c1) | `x_prep/fastpt.mojo:14` |
| `MOJOLEARN_QN_FAST_COALESCED_OFF` | EXPERIMENTS.md:247 DROPPED-slower (lane/apple-fast-linear @ 1c7c213f8) | `glm/impl/qn/glm_base.mojo:112` |
| `MOJOLEARN_QR_FAST_DEV` | EXPERIMENTS.md:453 DROPPED-slower (lane/apple-fast-decomp-linalg @ 74d52352b -> lane/apple-fast-rec-decomp) | `python/mojolearn/_linalg_impl.py:1251` |
| `MOJOLEARN_RESAMPLE_FAST_ONE_FOLD` | EXPERIMENTS.md:384 DROPPED-slower (lane/apple-fast-resample @ 50b96e795; A/B ab1 d51f4b4bf) | `resample/estimator.mojo:221` |
| `MOJOLEARN_RESAMPLE_FAST_RANK_SORT` | EXPERIMENTS.md:386 DROPPED-slower (lane/apple-fast-resample @ 50b96e795; A/B ab1 d51f4b4bf) | `resample/estimator.mojo:205` |
| `MOJOLEARN_SEQ_FAST_LSTM_SCAN` | EXPERIMENTS.md:658 not measured alone; bundle DROPPED-quality 2026-10-04 (see rec-optim table) (lane/apple-fast-gap-lstm @ 0d6cbc821) | `sequence/recurrent_scan.mojo:17` |
| `MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM` | EXPERIMENTS.md:659 not measured alone; bundle DROPPED-quality 2026-10-04 (see rec-optim table) (lane/apple-fast-gap-lstm @ 0d6cbc821) | `sequence/recurrent_scan.mojo:18` |
| `MOJOLEARN_SEQ_FAST_LSTM_WGRAD` | EXPERIMENTS.md:660 DROPPED-quality 2026-10-04 (BROKEN; see rec-optim table) (lane/apple-fast-gap-lstm @ 0d6cbc821) | `sequence/recurrent_scan.mojo:23` |
| `MOJOLEARN_SEQ_FAST_MAP_DOWN` | EXPERIMENTS.md:662 DROPPED-slower 2026-10-04 on layernorm (see rec-optim table) (lane/apple-fast-gap-optim @ cf4513f8a) | `sequence/exec_device.mojo:139` |
| `MOJOLEARN_SEQ_FAST_RAW_DOWN` | EXPERIMENTS.md:1196 DROPPED-slower (lane/apple-fast-gap-optim @ cf4513f8a (on main)) | `sequence/exec_device.mojo:151` |
| `MOJOLEARN_SEQ_FAST_VAR_NODRAIN` | EXPERIMENTS.md:1411 DROPPED-slower (lane/apple-fast-s-ts) | `sequence/exec_device.mojo:150` |
| `MOJOLEARN_SPARSE_RP_DEVICE` | EXPERIMENTS.md:557 DROPPED-quality (lane/apple-fast-kapprox @ 10d5a7970) | `x_neighbors/kapprox_dev.mojo:10` |
| `MOJOLEARN_SVD_FAST_CHOLQR` | EXPERIMENTS.md:454 DROPPED-slower (lane/apple-fast-decomp-linalg @ 74d52352b -> lane/apple-fast-rec-decomp) | `python/mojolearn/_linalg_impl.py:1567` |
| `MOJOLEARN_SVD_QFIX` | EXPERIMENTS.md:1216 DROPPED-slower (lane/apple-fast-q-linalg @ aaebc0ab8) | `x_decomp/qfix.mojo:10` |
| `MOJOLEARN_SVD_QOLD` | EXPERIMENTS.md:1216 DROPPED-slower (lane/apple-fast-q-linalg @ aaebc0ab8) | `x_decomp/qfix.mojo:10` |
| `MOJOLEARN_SYM_DERIV_FUSED` | EXPERIMENTS.md:182 DROPPED-noise (lane/apple-fast-sym-iter @ 4956a2234) | `gbdt/methods/sym_iter_fast.mojo:30` |
| `MOJOLEARN_SYM_GATHER_FUSED` | EXPERIMENTS.md:185 DROPPED-noise (lane/apple-fast-sym-hist @ 3bb4db314) | `gbdt/methods/kernel/sym_fast.mojo:65` |
| `MOJOLEARN_SYM_HIST_MULT` | EXPERIMENTS.md:187 DROPPED-noise (lane/apple-fast-sym-hist @ 3bb4db314) | `gbdt/methods/kernel/sym_fast.mojo:106` |
| `MOJOLEARN_SYM_LEAF_FROM_STATS` | EXPERIMENTS.md:189 DROPPED-noise (lane/apple-fast-sym-iter @ 4956a2234) | `gbdt/methods/sym_iter_fast.mojo:36` |
| `MOJOLEARN_SYM_PART_STATS_PAR` | EXPERIMENTS.md:191 DROPPED-noise (lane/apple-fast-sym-hist @ 3bb4db314) | `gbdt/methods/kernel/pointwise_scores.mojo:1954` |
| `MOJOLEARN_SYM_REUSE_PARTITION` | EXPERIMENTS.md:193 DROPPED-noise (lane/apple-fast-sym-iter @ 4956a2234) | `gbdt/methods/sym_iter_fast.mojo:21` |
| `MOJOLEARN_SYM_SCAN_SUB_FUSED` | EXPERIMENTS.md:194 DROPPED-noise (lane/apple-fast-sym-hist @ 3bb4db314) | `gbdt/methods/kernel/split_properties_helpers.mojo:483` |
| `MOJOLEARN_TREES_T29` | EXPERIMENTS.md:1710 DROP (slower), code deleted (main @ 42d1e42c6 (deleted on lane/grid-act-2)) | `gbdt/trees_identical_switches.mojo:74` |
| `MOJOLEARN_TSA2_KPSS` | EXPERIMENTS.md:544 DROPPED-noise (lane/apple-fast-gap-tsa @ e9da47064) | `tsa/impl/timeSeries/kpss_fused.mojo:7` |
| `MOJOLEARN_TSVD_FAST_CHOLQR3` | EXPERIMENTS.md:1410 DROPPED-slower (lane/apple-fast-s-linalg) | `x_decomp/tsvd_fast.mojo:4` |
| `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_LABRG` | EXPERIMENTS.md:872 reconciled 2026-10-05: w2-mbk-labrg: quality PASS, istella 146.6 -> 144.0, taxi 45.3 -> 46.3 (Manager verdicts session 2: DROP-speed noise); opt-in only. Was OPEN, opt-in: Last labelling pass as the CLS3_ROWGRP 32-thread-per-row assignment; reorders distance sums (labrg tolerance mode). ((not promoted)) | `x_cluster/minibatch_fast.mojo:160` |
| `MOJOLEARN_YETI_TREE_SEARCH_SCORE_GRID` | EXPERIMENTS.md:150 DROPPED-noise (lane/apple-fast-yetirank @ c7b35fd7c) | `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo:2945` |
