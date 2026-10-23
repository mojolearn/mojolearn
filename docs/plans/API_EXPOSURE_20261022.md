# API exposure fixes, 2026-10-22 (lane/api-exposure-fixes)

Input: the API exposure audit of the 1508 native binding entries
(`~/mojolearn-evidence/audit-20261022/api_exposure.md` and `.tsv`, main 551dee1ce):
PUBLIC 1153, INTERNAL-SUPPORT 246, DEAD 109, plus one module-level finding (the
multi-GPU driver modules had no `__all__`, were not imported by `mojolearn` and were
not counted by `tools/public_runtime_inventory.py`).

## 1. Exposed names

Every module below now has a literal `__all__`, is imported by `mojolearn/__init__.py`
and listed in `mojolearn.__all__` (wired as `parallel_forecasting` /
`parallel_model_selection` are), has rows in `packaging/native_only/manifest.json`
(so `tools/public_runtime_inventory.py` counts it), carries user-voice docstrings (the
old engineering docstrings are `#` comments beside them), and has a real-path case in
`python/mojolearn/tests/test_parallel_public_drivers.py` (devices from
`MOJOLEARN_PAR_DEVICES`, default `0`; each case equals the plain one-device estimator
byte for byte).

Docs decision for all six modules: excluded in `docs-site/api_exclude.txt`
("multi-GPU: physical qualification pending (par-* lanes recorded on one device per
vendor)"), the same footing as `parallel_forecasting`, `parallel_gaussian_process`,
`parallel_ivf` and `parallel_model_selection`. Their par-* lanes have NVIDIA, AMD and
Apple reference records in `python/mojolearn/verify_reference/table.json`, but those
records are one-device columns (`tools/record_identity_column.sh`: par_devices 0);
`docs/multi_gpu/COVERAGE.md` still lists cross-vendor multi-device qualification as
remaining for every family. Assigned to the `multi_gpu` family in
`docs-site/tools/gen_api.py` for the day they qualify.

| module | public names | verifier lanes (table.json columns) |
|---|---|---|
| `parallel_classical` | `fit_logistic` | par-logistic (nv, amd, apple) |
| | `fit_exponential_smoothing` | par-holtwinters (nv, amd, apple, cpu) |
| | `fit_gram_estimator` | par-gram, par-gram-ols, par-gram-pca, par-gram-tsvd (nv, amd, apple) |
| | `fit_arima` | par-arima (nv, amd, apple, cpu) |
| | `fit_kmeans` | par-kmeans (nv, amd, apple) |
| | `fit_coordinate_descent` | par-cd, par-cd-elasticnet (nv, amd, apple) |
| | `fit_svm`, `predict_svm` | par-svm, par-svm-svr (nv, amd, apple) |
| | `fit_gaussian_process`, `predict_gaussian_process` | par-gp (nv, amd, apple) |
| | `fit_dbscan` | par-dbscan (nv, amd, apple) |
| | `fit_gaussian_mixture`, `predict_gaussian_mixture` | par-gmm (nv, amd, apple) |
| | `bootstrap`, `permutation_test`, `monte_carlo_integrate` | par-resample (nv, amd, apple, cpu) |
| | `fit_hdbscan` | par-hdbscan (nv, amd, apple) |
| | `fit_cholesky`, `solve_cholesky` | par-cholesky (nv, amd, apple) |
| | `fit_kernel_method`, `apply_kernel_method` | par-kernel-ridge, par-nystroem (nv, amd, apple) |
| | `transform_rbf_sampler` | par-rbf-sampler (nv, amd, apple, cpu) |
| `parallel_ensemble` | `fit_boosting` | par-boosting, par-boosting-clf, par-boosting-reg, par-boosting-pointwise, par-ordered, par-border-types (nv, amd, apple) |
| | `fit_forest` | par-forest, par-forest-et, par-forest-et-clf, par-forest-reg (nv, amd, apple, cpu) |
| | `fit_isolation_forest`, `score_isolation_forest` | par-iforest (nv, amd, apple) |
| | `fit_ordered_rmse` | par-ordered-rmse (nv, amd, apple) |
| | `fit_feature_freq` | par-feature-freq (nv, amd, apple) |
| | `ParallelForestPredictor` | par-forest-pool (nv, amd, apple) |
| `parallel_graph` | `fit_graph` | par-graph-agglomerative, par-graph-spectral, par-graph-umap (nv, amd, apple) |
| | `transform_umap` | par-graph-umap (nv, amd, apple) |
| `parallel_neighbors` | `ParallelQueries` | par-queries-knn, par-queries-nn, par-queries-radius, par-queries-kde (nv, amd, apple, cpu) |
| `parallel_neighbors_reference` | `ReferenceShardedNeighbors` | par-reference-knn, par-reference-knn-reg (nv, amd, apple, cpu) |
| `parallel_preprocessing` | `fit_scaler`, `transform_scaler` | par-scaler, par-scaler-minmax (nv, amd, apple, cpu) |
| `ensemble` | `catboost_auto_learning_rate` | none directly; reached through GradientBoosting fits (gbdt lanes) |

`parallel_neighbors_reference` is product: `docs/multi_gpu/README.md` documents it,
two verifier lanes and two capacity/transport tools call it.

`ensemble.catboost_auto_learning_rate`: `ensemble.py` gets a literal `__all__` that keeps
today's public set (the boosting estimators, their option tables, scikit-learn's lazily
resolved ensemble names) and adds this function. Docs decision: excluded
(`ensemble.catboost_auto_learning_rate`, "host scalar helper behind GradientBoosting's
automatic learning rate; no verifier lane calls it directly").

Also updated: `SUPPORT_MATRIX.md` (Multi-GPU row), `tools/qualify_verifier_wheel.py`
(expanded API import guard), `tools/public_runtime_inventory.py` (family names).

Inventory (`tools/public_runtime_inventory.py --summary`, TOTAL row): before 383 names /
996 methods (120 native, 367 orchestrated, 509 python); after 418 names / 1033 methods
(120 native, 368 orchestrated, 545 python). The 36 new python-verdict methods are the
drivers' worker-pool glue; the static walk does not follow `DevicePool` requests into
the worker. Baseline refreshed in its own commit; `--check` passes.

## 2. Removed DEAD entries (87)

Plain dead-code removals (no tombstone protocol): the `def_function` registration, the Mojo
wrapper and any impl left with no caller, the `host_surface.py` / `_surface_*.py` export rows,
and Python wrappers that existed only for the door. Merged from `lane/api-exposure-dead`.

| entry | files | commit |
|---|---|---|
| `adafactor_step` | bindings/_mojolearn_x_sequence.mojo,bindings/_mojolearn_x_sequence_host.mojo | 642b9d8f2 |
| `byte_lm_session_run` | bindings/_mojolearn_byte_lm.mojo | d7e1712a8 |
| `clip_grad_norm_multi` | bindings/_mojolearn_training.mojo | 7e88fbc71 |
| `compact_notnan_f32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `dot_rows_f32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `draw_rows_without_replacement_i32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `epoch_order_i32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `epoch_schedule` | bindings/_mojolearn_x_sequence_host.mojo | 642b9d8f2 |
| `gather_keep_neg_i32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `generation_initialize` | bindings/_mojolearn_tokenizer_host.mojo | 544358e1e |
| `gp_py2mojo` | bindings/_mojolearn_gp.mojo,bindings/_mojolearn_gp_host.mojo,bindings/_mojolearn_gp_infer_host.mojo | 8e77f1fa4 |
| `gp_restart_uniforms` | bindings/_mojolearn_gp.mojo,bindings/_mojolearn_gp_host.mojo | 8e77f1fa4 |
| `lr_decide` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `lr_onecycle_exact` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `lr_onecycle_exact_block` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `lr_onecycle_fill` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `lr_pow_values` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `lr_schedule_values` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `mean_std_f32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `mlp_resident_upload` | bindings/_mojolearn_training.mojo | 7e88fbc71 |
| `ocsvm_alpha_init_f32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `optimizer_resident_get` | bindings/_mojolearn_training.mojo | 7e88fbc71 |
| `optimizer_resident_put` | bindings/_mojolearn_training.mojo | 7e88fbc71 |
| `probability_rows_f32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `rf_classifier_fit_oob_export` | bindings/_mojolearn_rf.mojo,bindings/_mojolearn_rf_host.mojo | 003b59a67 |
| `rf_regressor_fit_oob_export` | bindings/_mojolearn_rf.mojo,bindings/_mojolearn_rf_host.mojo | 003b59a67 |
| `rnn_n_params` | bindings/_mojolearn_x_sequence.mojo,bindings/_mojolearn_x_sequence_host.mojo | 642b9d8f2 |
| `scale_shift_ftz_f32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `split_table_i32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `strat_group_assign_i32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `tokenizer_pretokenize` | bindings/_mojolearn_tokenizer_host.mojo | 544358e1e |
| `vocab_decode` | bindings/_mojolearn_tokenizer_host.mojo | 544358e1e |
| `vocab_encode` | bindings/_mojolearn_tokenizer_host.mojo | 544358e1e |
| `vocab_info` | bindings/_mojolearn_tokenizer_host.mojo | 544358e1e |
| `vocab_load_bytes` | bindings/_mojolearn_tokenizer_host.mojo | 544358e1e |
| `vocab_load_spelled` | bindings/_mojolearn_tokenizer_host.mojo | 544358e1e |
| `vocab_pretokenize` | bindings/_mojolearn_tokenizer_host.mojo | 544358e1e |
| `weighted_draw_rows_i32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `weighted_pick_i32` | bindings/_mojolearn.mojo,bindings/_mojolearn_core_host.mojo | 9a2a07c70 |
| `x_cnn_adam` | bindings/_mojolearn_x_cnn.mojo,bindings/_mojolearn_x_cnn_host.mojo | 4894f3a7f |
| `x_cnn_sgd` | bindings/_mojolearn_x_cnn.mojo,bindings/_mojolearn_x_cnn_host.mojo | 4894f3a7f |
| `x_cnn_softmax_xent` | bindings/_mojolearn_x_cnn.mojo,bindings/_mojolearn_x_cnn_host.mojo | 4894f3a7f |
| `x_decomp_chi2_cdf` | bindings/_mojolearn_x_decomp.mojo,bindings/_mojolearn_x_decomp_host.mojo | aaed358c2 |
| `x_decomp_dev_als_rows` | bindings/_mojolearn_x_decomp.mojo | aaed358c2 |
| `x_decomp_dev_knn_select` | bindings/_mojolearn_x_decomp.mojo | aaed358c2 |
| `x_decomp_dev_mds_disp` | bindings/_mojolearn_x_decomp.mojo | aaed358c2 |
| `x_decomp_dev_mds_setup` | bindings/_mojolearn_x_decomp.mojo | aaed358c2 |
| `x_decomp_iso_order` | bindings/_mojolearn_x_decomp.mojo,bindings/_mojolearn_x_decomp_host.mojo | aaed358c2 |
| `x_decomp_mds_disp` | bindings/_mojolearn_x_decomp_host.mojo | aaed358c2 |
| `x_decomp_mds_setup` | bindings/_mojolearn_x_decomp_host.mojo | aaed358c2 |
| `x_decomp_scatter` | bindings/_mojolearn_x_decomp.mojo,bindings/_mojolearn_x_decomp_host.mojo | aaed358c2 |
| `x_decomp_sign_labels` | bindings/_mojolearn_x_decomp.mojo,bindings/_mojolearn_x_decomp_host.mojo | aaed358c2 |
| `x_decomp_triu_nonzero` | bindings/_mojolearn_x_decomp.mojo,bindings/_mojolearn_x_decomp_host.mojo | aaed358c2 |
| `x_metrics_auc_xy` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_centroids` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_ch_extra` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_class_sums` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_contingency_stats` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_curve_det` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_curve_pr` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_curve_roc` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_db_score` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_encode_small_i64` | bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_expected_mi` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_expected_mi_tasks` | bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_mi_contingency` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_ndcg_mean` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_ovo_pair` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_row_sum_range` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_row_sum_range_tasks` | bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_metrics_scatter_rows` | bindings/_mojolearn_x_metrics.mojo,bindings/_mojolearn_x_metrics_host.mojo | b5261b94c |
| `x_prep_ii_conv` | bindings/_mojolearn_x_prep_host.mojo | 64a8b980d |
| `x_prep_ii_gather` | bindings/_mojolearn_x_prep_host.mojo | 64a8b980d |
| `x_prep_ii_rows` | bindings/_mojolearn_x_prep_host.mojo | 64a8b980d |
| `x_prep_ii_scatter` | bindings/_mojolearn_x_prep_host.mojo | 64a8b980d |
| `x_prep_kfold_folds` | bindings/_mojolearn_x_prep.mojo,bindings/_mojolearn_x_prep_host.mojo | 64a8b980d |
| `x_prep_strat_folds` | bindings/_mojolearn_x_prep.mojo,bindings/_mojolearn_x_prep_host.mojo | 64a8b980d |
| `xn_cc_step` | bindings/_mojolearn_x_neighbors.mojo,bindings/_mojolearn_x_neighbors_host.mojo | 96bf2ce89 |
| `xn_nan_sqdist` | bindings/_mojolearn_x_neighbors.mojo,bindings/_mojolearn_x_neighbors_host.mojo | 96bf2ce89 |
| `xn_p2m_negate` | bindings/_mojolearn_x_neighbors.mojo,bindings/_mojolearn_x_neighbors_host.mojo | 96bf2ce89 |
| `xn_pagerank_step` | bindings/_mojolearn_x_neighbors.mojo,bindings/_mojolearn_x_neighbors_host.mojo | 96bf2ce89 |
| `xn_pcs_conv` | bindings/_mojolearn_x_neighbors.mojo,bindings/_mojolearn_x_neighbors_host.mojo | 96bf2ce89 |
| `xn_pcs_copy0` | bindings/_mojolearn_x_neighbors.mojo,bindings/_mojolearn_x_neighbors_host.mojo | 96bf2ce89 |
| `xn_pcs_sketch` | bindings/_mojolearn_x_neighbors.mojo,bindings/_mojolearn_x_neighbors_host.mojo | 96bf2ce89 |
| `xn_pr_iterate` | bindings/_mojolearn_x_neighbors.mojo,bindings/_mojolearn_x_neighbors_host.mojo | 96bf2ce89 |
| `xn_skew_weights` | bindings/_mojolearn_x_neighbors.mojo,bindings/_mojolearn_x_neighbors_host.mojo | 96bf2ce89 |
| `xn_svd_flip` | bindings/_mojolearn_x_neighbors.mojo,bindings/_mojolearn_x_neighbors_host.mojo | 96bf2ce89 |

## 3. DEAD entries kept (22)

The audit's `_byte_lm_trainer_host.py` is not orphaned: `host_surface.ADAPTED_MODULES` imports it
(`python/mojolearn/_backend.py:1889-1891`), so its three session doors are live.

| entry | reference that keeps it |
|---|---|
| `adam_hyper_f64` | tools/identical_wave_quality.py:138 calls it |
| `byte_lm_session_logits` | tools/byte_lm_gpu_logits_sweep.py:138; composed by TrainerOwnerEntries (_byte_lm_trainer_host.py:316), which is live via host_surface.ADAPTED_MODULES -> _backend.py:1889-1891 |
| `byte_lm_session_next_bytes` | _byte_lm_trainer_host.py:315 TrainerOwnerEntries.byte_lm_trainer_next_bytes (live adapter, _backend.py:1889-1891); test_byte_lm_trainer_logits.py:86-108 |
| `byte_lm_session_rollback` | _byte_lm_trainer_host.py:220 (live adapter); test_cpu_training_embedding_ivf.py:357 |
| `clip_grad_norm` | tools/parallel_clip_check.py:27; test_cpu_training_misc.py:65 NEURAL_EXPORTS |
| `from_f16` | verifier lane safetensors-widen-f16 (tools/identity_lanes/coverage.py:273 via models/safetensors.widen_f16) |
| `lamb_step` | tools/identity_lanes/coverage_sabotage.tsv:37 (sequence-lamb sabotage arm) |
| `optimizer_resident_copy_dev` | whole body of experiment switch IDN_OPT_PARAMS_RESIDENT (bindings/_mojolearn_training.mojo:1417-1418) |
| `sched_exp_block` | whole body of experiment switch MOJOLEARN_SCHED_FAST_TABLE (bindings/_mojolearn_x_sequence.mojo:862-863); removal = experiment deletion |
| `transformer_flash_call_count` | experiments/performance_ideas/F07/caller.py:25 |
| `x_cluster_mbk_policy` | experiments/performance_ideas/F15/caller.py:17 |
| `x_cnn_conv_block_backward` | python/mojolearn/tests/test_x_cnn_repeat.py:106 oracle for x_cnn_conv_block_backward_r; host wrapper also called at bindings/_mojolearn_x_cnn_host.mojo:774 |
| `x_cnn_conv_block_forward` | python/mojolearn/tests/test_x_cnn_repeat.py:102 oracle for x_cnn_conv_block_forward_r |
| `x_decomp_argsort_f32` | tools/hooks/test_no_host_routes.py:429 asserts it is a host export |
| `x_decomp_dev_cd_rows` | python/mojolearn/tests/test_x_decomp_repeat.py:66 via _Kit.cd_rows (_expansion_decomp.py:1178) |
| `x_decomp_dev_dict_update` | whole body of experiment switch DECOMP_FAST_DICT_DEV (bindings/_mojolearn_x_decomp.mojo:288-290); _expansion_decomp._dict_dev reads it |
| `x_decomp_dev_maxabs` | whole body of experiment switch IDN_DEV_MAXABS (bindings/_mojolearn_x_decomp.mojo:278-281) |
| `x_decomp_fast_defines` | restore patches experiments/removed/MOJOLEARN_QR_FAST_DEV.patch:280, MOJOLEARN_SVD_FAST_CHOLQR.patch:99; docs/TOMBSTONES.md:2012 |
| `x_linear_cls1_flags` | experiments/removed/_shared-9decae29f.patch:50 (restore patch context) |
| `x_neighbors_kfeat_schi2_draw` | host export that keeps _schi2_draw on the host column; removing it makes no_host_routes refuse x_neighbors/kfeat_dev.mojo:230,232 (removed in 96bf2ce89, restored in f424eca84) |
| `x_neighbors_kfeat_schi2_weights` | _expansion_neighbors.py:2069 random_weights_ (older pickle _schi2_z) |
| `x_trees_dart_idn` | whole body of experiment switch IDN_DART_DEVICE (bindings/_mojolearn_x_trees.mojo:239-240) |
