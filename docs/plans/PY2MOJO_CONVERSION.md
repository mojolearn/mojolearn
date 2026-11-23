# Numeric Python tests into the Mojo TestSuite checks: map, recipe, batches and the clean pass

Lane py2mojo-pilot, 2026-11-22. Andrew (2026-11-22): the numeric and kernel tests move from Python to Mojo now; the
Python tests STAY as the fallback and are removed in ONE clean pass at the end, never piece by piece. So every batch
below ADDS Mojo checks and deletes no Python test. Rule: AGENTS.md "Move native correctness tests to Mojo"; recipe of
the TestSuite files: `docs/plans/TESTSUITE_CONVERSION.md`.

## The map

`docs/plans/PY2MOJO_MAP.tsv`: every test function of the 164 NATIVE and MIXED files of the inventory
(`~/mojolearn-evidence/briefs-20261122/inv-shrink/2_tests_classified.tsv`), 780 rows (a script file with no `test_*`
is one `<script>` row). Columns: python_file, test, class, mojo_test_or_suite, status, batch, reason.

- **a, 44**: already covered by a named Mojo check (path::test, and the unit of its case table where it is one).
- **b, 146**: numeric or kernel behaviour no Mojo check asserts: it needs a Mojo case; the column names the suite
  (`NEW:` when no existing file of the family fits). 9 of them are converted by this lane (status
  `converted (py2mojo-pilot)`), 5 belong to the paused Apple `fast` programme (batch P).
- **c, 590**: stays Python: buffer admission, exceptions, returned objects and dtypes, pickling, the scikit-learn
  protocol, callback refusal, framework adapters, loaders and binary selection, threads and lifetimes of Python
  objects, public-method compositions, verification tooling (202 rows of 18 tooling files). 6 rows only assert source
  text or a manifest: they are deletion candidates of the Mojo/Python test-removal pass, not ports.

How it was made: an AST pass (`~/mojolearn-evidence/briefs-20261122/py2mojo-work/classify.py`, files across the cores)
gave a first class from the assertions and the boundary patterns of each function and the Mojo files that name the
Python file; then every row was read and searched against the Mojo checks (`checks/native_tests/*.mojo`,
`checks/*.mojo`, `*/checks/*.mojo`) in five parallel review slices (`review_slice*.tsv` in that directory). Most of
the first pass's "a" rows were wrong: the Mojo ports of a file cover only the functions the earlier migration took,
and their docstrings say which ones stay in Python. Weakest calls (re-check when the batch reaches them): pagerank
and connected components are "a" against `x_neighbors/checks/graph_check.mojo`, which no TestSuite entry runs yet;
`metrics/checks/regression_errors_check.mojo` covers the overflow row but has a `main()` only; the
`test_estimator_score_p2` and stacking rows are c as public-method compositions.

## The pilot (two families, built and run on cpc1)

| family | Python tests (class b) | Mojo check | column |
|---|---|---|---|
| prep | `test_x_prep_scalers.py` `test_robust_scaler`, `test_maxabs_scaler`; `test_x_prep_normalizer.py` `test_normalizer`; `test_x_prep_binarizer.py` `test_binarizer` | `checks/native_tests/prep_scalers_sklearn.mojo` (4 tests, 18 table rows) | cpu |
| metrics | `test_fowlkes_mallows_score.py` `test_empty_is_zero_like_sklearn`, `test_matches_sklearn_bitwise_on_small_cases`, `test_matches_sklearn_bitwise_on_a_random_pair`; `test_metrics_undefined_score.py` both tests | `checks/native_tests/metrics_fowlkes_mallows.mojo` (9 rows), `checks/native_tests/metrics_r2_epilogue.mojo` (7 rows) | cpu gpu |

- The prep family had no TestSuite file for these estimators and the metrics family none for fowlkes_mallows or the
  r2 epilogue, so the three files are new; each names its Python origin in its docstring.
- Inputs and references: `tools/py2mojo_native_fixtures.py` writes the original tests' NumPy draws and the
  scikit-learn values (binary64 words) to `checks/native_tests/fixtures/prep_sanity/` and `metrics_fm/`, tracked as
  `prep_sanity.tar` and `metrics_fm.tar` (`tools/fixture_packs.py`); the command and the numpy / scikit-learn
  versions are in its docstring. `checks/native_tests/py2mojo_fixture_words.mojo` reads binary64 words.
- cpc1, Mojo of the box's pixi env, host column (`-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU=1`, the
  x86-64-v3 target): the three files compiled in 51 s wall (`tools/ops/native_testsuite.sh build`, 3 files at once)
  and passed: 6 tests, 0 failed. They fail when they should: a binary64 reference word changed in
  `fm-rand-want.f64` and a scale word in `robust-3-scale.f64` made `test_fowlkes_mallows` and `test_robust_scaler`
  fail with the value named. The device column of the two metrics files is compile and run owed on the L40S.
- What the Mojo checks call: the programs and binding bodies the product runs on that column (the Prep member's arena
  layouts and x_prep stages through `x_prep.host.program.run_program_host`; `host_fowlkes_mallows` and the device
  `fowlkes_mallows_score_host`; the `_Reg` r2 program through `metrics_plan_execute`). Doubt: the r2 check repeats the
  door's stage order (`bindings/metrics_reg_door.mojo` `_r2_options`) because that composition lives in the door; a
  typed `r2_stages(mut r, force_finite)` in `bindings/metrics_reg_run.mojo` called by both would remove the copy.
  The fowlkes oracle maps labels onto their sorted union as the door's `_cluster_codes` does.

## The recipe (one class-b test)

1. Read the Python test; find the native entry its public call reaches on the host column (the member, door or RUN
   module: `core/*_member.mojo`, `bindings/*_run.mojo`, `x_*/...`). Call that, never a re-implementation.
2. Inputs: literals when the test writes literals; NumPy draws are stored as words (they cannot be regenerated in
   Mojo). External oracles (scikit-learn, scipy, statsforecast, LightGBM, shap) become stored reference words. Add a
   set function to a generator of the batch's own (`tools/py2mojo_native_fixtures_b<N>.py`, importing `_put` from
   `tools/py2mojo_native_fixtures.py`), write a new case directory `checks/native_tests/fixtures/<family>_<topic>/`,
   pack it (`python3 tools/fixture_packs.py pack checks/native_tests/fixtures`) and commit only the new `.tar`.
   Record the generating command and the library versions in the generator's docstring.
3. Same inputs, expected values, tolerances and assertion sense as the Python test (assert_allclose is
   `|a - e| <= atol + rtol |e|`, NaN only against NaN). A parametrized test is a case table (`check_cases`, one row
   per parameter); independent rows run across the host cores.
4. The file: line 2 `# native-check: <columns>`, a docstring naming the Python file and test(s) it replaces and what
   stays in Python; `TestSuite` main as in `TESTSUITE_CONVERSION.md`. A check with a device column uses
   `CPU_COLUMN_BUILD` / `check_cases_serial` (`metrics_r2_epilogue.mojo`) or tests by name.
5. Build and run on the CPU box: `IN_FILTER=<names> NC_DEFINES=... NC_TARGET=... bash tools/ops/native_testsuite.sh
   build cpu <bin> <out>` then `run` (box script: `~/mojolearn-evidence/briefs-20261122/py2mojo-work/box_build.sh`).
   Break one reference word and see the check fail, then restore it (`fixture_packs.py verify`).
6. Update the row of `PY2MOJO_MAP.tsv`: status `converted (<lane>)`, mojo_test_or_suite = path::test and unit.
   Delete no Python test.

A class-b test that turns out to be covered after all becomes class a with the Mojo test named; a numeric test that
can only be checked through a public Python method (one public method against another) becomes c with that reason.

## The batches (disjoint Mojo files; the shared files stay untouched)

Each batch owns its target Mojo files and its own generator and fixture cases; no two batches write the same file.
Shared and not edited by any batch: `checks/native_tests/testsuite_cases.mojo`, `testsuite_serial.mojo`, the runner
(`tools/ops/native_testsuite.sh`, `native_checks_steps.sh`, `native_checks_report.py`), `tsa_metrics_fixture_words.mojo`,
`py2mojo_fixture_words.mojo` (add a reader there only through the orchestrator). Batch 1 owns
`prep_scalers_sklearn.mojo` for further scaler cases. Batch P waits for Andrew to reopen Apple `fast`.

| batch | b rows | Python files (rows) | Mojo targets (under checks/native_tests/ unless a path) |
|---|---|---|---|
| 1 | 17 | `minmax_scaler_smoke.py` (1), `standard_scaler_smoke.py` (1), `test_cpu_training_e2.py` (1), `test_x_prep_imputer.py` (2), `test_x_prep_iterative.py` (2), `test_x_prep_polynomial.py` (1), `test_x_prep_power.py` (1), `test_x_prep_quantile.py` (1), `test_x_prep_scaler_options.py` (5), `test_x_prep_spline.py` (2) | `NEW:prep_iterative_imputer.mojo`, `NEW:prep_polynomial.mojo`, `NEW:prep_power.mojo`, `NEW:prep_quantile.mojo`, `NEW:prep_scaler_options.mojo`, `NEW:prep_scalers.mojo`, `NEW:prep_simple_imputer.mojo`, `NEW:prep_spline.mojo`, `NEW:prep_standard_scaler.mojo` |
| 2 | 34 | `test_x_prep_discriminant.py` (4), `test_x_prep_encoders.py` (4), `test_x_prep_kbins.py` (2), `test_x_prep_labels.py` (3), `test_x_prep_naive_bayes.py` (6), `test_x_prep_parity.py` (6), `test_x_prep_selection.py` (6), `test_x_prep_target_encoder.py` (3) | `NEW:prep_discriminant.mojo`, `NEW:prep_encoders.mojo`, `NEW:prep_feature_selection.mojo`, `NEW:prep_kbins.mojo`, `NEW:prep_labels.mojo`, `NEW:prep_naive_bayes.mojo`, `NEW:prep_target_encoder.mojo`, `prep_label_binarizer_inverse_window.mojo` |
| 3 | 22 | `test_x_trees_sanity.py` (22) | `NEW:trees_shap.mojo`, `NEW:trees_sklearn_quality.mojo`, `trees_entry_glue.mojo` |
| 4 | 11 | `gbdt_binary_prediction_smoke.py` (1), `gbdt_feature_fraction_binding.py` (1), `gbdt_querywise_pointwise_check.py` (1), `test_class_weight_shared_families.py` (1), `test_cpu_training_forest_variants.py` (1), `test_meta_members_native.py` (5), `test_rf_class_weight.py` (1) | `NEW:gbdt_binary_probabilities.mojo`, `NEW:metrics_roc_auc_ovr.mojo`, `NEW:trees_meta_vote.mojo`, `NEW:xtrees_shap_additive.mojo`, `class_weight_plan.mojo`, `ensemble_forest_check.mojo`, `gbdt_public_fits.mojo`, `trees_entry_dart.mojo` |
| 5 | 11 | `binary_ranking_smoke.py` (1), `log_loss_smoke.py` (1), `test_ne2_search_workers.py` (2), `test_py_bugs.py` (4), `test_regression_metrics.py` (1), `test_resample_bca.py` (1), `test_weighted_score.py` (1) | `NEW:metrics_log_loss.mojo`, `NEW:msel_device_rows_folds.mojo`, `metrics_selection_host_p2.mojo`, `metrics_weighted_score.mojo`, `msel_split_plan.mojo`, `resample/checks/resample_check.mojo`, `resample_cv_row_order.mojo` |
| 6 | 17 | `test_hdbscan_host_lifetime.py` (1), `test_linalg_decompositions.py` (8), `test_linear_svc_native_boundary.py` (1), `test_pca_exact_native_owner.py` (1), `test_peer1_provider_boundary.py` (1), `test_svr_native_transport_boundary.py` (1), `test_x_linear_sanity.py` (1), `test_x_neighbors_sanity.py` (3) | `NEW:neighbors_knn_classify.mojo`, `NEW:neighbors_svgp.mojo`, `NEW:x_linear_sanity.mojo`, `NEW:x_neighbors_graph_check.mojo`, `hdbscan_native.mojo`, `linalg_numerical.mojo`, `linear_svc_native.mojo`, `ocsvm_native.mojo`, `pca_native.mojo`, `svr_native.mojo` |
| 7 | 12 | `test_byte_lm_step_routes.py` (7), `test_cpu_training_embedding_ivf.py` (3), `test_parallel_neural_native.py` (1), `test_samba_attention_wiring.py` (1) | `NEW:byte_lm_host_device_words.mojo`, `ann_ivf_surface.mojo`, `byte_lm_host_column.mojo`, `byte_lm_trainer_session.mojo`, `embedding_surface.mojo`, `sequence_samba_surface.mojo` |
| 8 | 8 | `test_x_cnn_gnn.py` (4), `test_x_sequence_ets.py` (1), `test_x_sequence_garch.py` (1), `test_x_sequence_mlp.py` (1), `test_x_sequence_prophet.py` (1) | `NEW:xcnn_graph_numeric.mojo`, `NEW:xseq_mlp_fit.mojo`, `prophet_state.mojo`, `sequence_garch.mojo`, `tsa_sequence_ets.mojo` |
| P | 5 | `test_kde_fused_fast.py` (3), `test_linalg_fast_bf16.py` (2) | `NEW:kde_fused_fast.mojo`, `bf16_linalg.mojo` |
| pilot | 9 | `test_fowlkes_mallows_score.py` (3), `test_metrics_undefined_score.py` (2), `test_x_prep_binarizer.py` (1), `test_x_prep_normalizer.py` (1), `test_x_prep_scalers.py` (2) | `metrics_fowlkes_mallows.mojo`, `metrics_r2_epilogue.mojo`, `prep_scalers_sklearn.mojo` |

Batch 2 is the largest (34 rows, 8 files, one x_prep family with seven estimators); split it in two by target file if
one lane cannot finish it. Batch 3 needs scikit-learn and shap reference values for 22 tree tests.

## The clean pass (once every batch is green)

When every class-b row of the map is `converted` and its Mojo check is green on the CPU box (and on the L40S for a
`gpu` column), ONE change deletes: the 40 files below whole (every test function in them is class a or b), and the
122 class-a/b functions of the files that stay; it keeps the class-c functions and every file of the second list.
Each deletion names, in the commit, the Mojo test the map gives for it. The batch P rows (Apple `fast`) keep their
Python tests until that programme is reopened and ported, so their two files are on the "stay" list.

Deletable (every test function class a or b, none paused):

- `checks/binary_ranking_smoke.py` (b 1)
- `checks/classification_metrics_binding.py` (a 1)
- `checks/forest_inference_public.py` (a 1)
- `checks/gbdt_binary_prediction_smoke.py` (b 1)
- `checks/gbdt_feature_fraction_binding.py` (b 1)
- `checks/gbdt_querywise_pointwise_check.py` (b 1)
- `checks/log_loss_smoke.py` (b 1)
- `checks/min_child_hessian_binding.py` (a 1)
- `checks/min_split_gain_binding.py` (a 1)
- `checks/minmax_scaler_smoke.py` (b 1)
- `checks/regression_errors_binding.py` (a 1)
- `checks/rf_class_weight_smoke.py` (a 1)
- `checks/standard_scaler_smoke.py` (b 1)
- `python/mojolearn/tests/test_cpu_training_embedding_ivf.py` (b 3)
- `python/mojolearn/tests/test_cpu_training_forest_variants.py` (b 1)
- `python/mojolearn/tests/test_cpu_training_gbdt_ordered.py` (a 1)
- `python/mojolearn/tests/test_hdbscan_host_lifetime.py` (b 1)
- `python/mojolearn/tests/test_metrics_undefined_score.py` (b 2)
- `python/mojolearn/tests/test_native_cv_row_order.py` (a 1)
- `python/mojolearn/tests/test_resample_bca.py` (b 1)
- `python/mojolearn/tests/test_x_linear_sanity.py` (b 1)
- `python/mojolearn/tests/test_x_prep_binarizer.py` (b 1)
- `python/mojolearn/tests/test_x_prep_discriminant.py` (b 4)
- `python/mojolearn/tests/test_x_prep_encoders.py` (b 4)
- `python/mojolearn/tests/test_x_prep_imputer.py` (b 2)
- `python/mojolearn/tests/test_x_prep_iterative.py` (b 2)
- `python/mojolearn/tests/test_x_prep_kbins.py` (b 2)
- `python/mojolearn/tests/test_x_prep_labels.py` (b 3)
- `python/mojolearn/tests/test_x_prep_naive_bayes.py` (b 6)
- `python/mojolearn/tests/test_x_prep_normalizer.py` (b 1)
- `python/mojolearn/tests/test_x_prep_polynomial.py` (b 1)
- `python/mojolearn/tests/test_x_prep_power.py` (b 1)
- `python/mojolearn/tests/test_x_prep_quantile.py` (b 1)
- `python/mojolearn/tests/test_x_prep_scalers.py` (b 2)
- `python/mojolearn/tests/test_x_prep_selection.py` (b 6)
- `python/mojolearn/tests/test_x_prep_spline.py` (b 2)
- `python/mojolearn/tests/test_x_prep_target_encoder.py` (b 3)
- `python/mojolearn/tests/test_x_sequence_ets.py` (b 1)
- `python/mojolearn/tests/test_x_sequence_garch.py` (b 1)
- `python/mojolearn/tests/test_x_sequence_prophet.py` (b 1)

Stay (at least one class-c or paused test); their class-a/b functions are deleted from them in the same pass:

- `checks/classification_forest_scoring.py`
- `checks/forest_export_public.py`
- `checks/gbdt_adapters_smoke.py`
- `checks/gbdt_feature_fraction_ab.py`
- `checks/model_selection_smoke.py`
- `checks/native_tests/fixtures/mamba/gen_corpus.py`
- `python/mojolearn/tests/_legacy_host_density_constructors.py`
- `python/mojolearn/tests/_package_evidence.py`
- `python/mojolearn/tests/test_arena_ranges.py`
- `python/mojolearn/tests/test_arima_parameter_lifetime.py`
- `python/mojolearn/tests/test_autograd_public.py`
- `python/mojolearn/tests/test_autograd_usability.py`
- `python/mojolearn/tests/test_byte_lm_host_ragged_lifetime.py`
- `python/mojolearn/tests/test_byte_lm_step_routes.py` (delete 9 function(s))
- `python/mojolearn/tests/test_byte_lm_surface.py`
- `python/mojolearn/tests/test_causal_lease_two_threads.py`
- `python/mojolearn/tests/test_class_weight_shared_families.py` (delete 1 function(s))
- `python/mojolearn/tests/test_classical_host_native_owner.py`
- `python/mojolearn/tests/test_classical_pandas_native_owner.py`
- `python/mojolearn/tests/test_classical_public_output_native_owner.py`
- `python/mojolearn/tests/test_cpu_training_e2.py` (delete 4 function(s))
- `python/mojolearn/tests/test_cpu_training_gp.py`
- `python/mojolearn/tests/test_cross_vendor.py` (delete 3 function(s))
- `python/mojolearn/tests/test_cross_vendor_native.py`
- `python/mojolearn/tests/test_cross_vendor_worker_native.py`
- `python/mojolearn/tests/test_dart_buffer_lifetime.py`
- `python/mojolearn/tests/test_decomp_resident_lifecycle.py`
- `python/mojolearn/tests/test_default_scorer_process.py`
- `python/mojolearn/tests/test_estimator_score_p2.py`
- `python/mojolearn/tests/test_fowlkes_mallows_score.py` (delete 3 function(s))
- `python/mojolearn/tests/test_generation_engine_admission.py`
- `python/mojolearn/tests/test_generation_route_controls.py`
- `python/mojolearn/tests/test_gp_svm_resident_lifecycle.py`
- `python/mojolearn/tests/test_hdbscan_native_boundary.py`
- `python/mojolearn/tests/test_host_density_hash_native.py`
- `python/mojolearn/tests/test_host_radius_bindings.py`
- `python/mojolearn/tests/test_iforest_output_boundary.py`
- `python/mojolearn/tests/test_kde_fused_fast.py` (delete 4 function(s))
- `python/mojolearn/tests/test_krr_options.py` (delete 1 function(s))
- `python/mojolearn/tests/test_label_binarizer_inverse_window.py`
- `python/mojolearn/tests/test_linalg_decompositions.py` (delete 9 function(s))
- `python/mojolearn/tests/test_linalg_fast_bf16.py` (delete 3 function(s))
- `python/mojolearn/tests/test_linear_archive_generic_native.py`
- `python/mojolearn/tests/test_linear_svc_native_boundary.py` (delete 1 function(s))
- `python/mojolearn/tests/test_logistic_native_predict.py`
- `python/mojolearn/tests/test_meta_members_native.py` (delete 5 function(s))
- `python/mojolearn/tests/test_metrics_device_score.py`
- `python/mojolearn/tests/test_native_array_view_lifetime.py`
- `python/mojolearn/tests/test_native_cv_gbdt_nb.py`
- `python/mojolearn/tests/test_native_cv_outputs.py`
- `python/mojolearn/tests/test_native_cv_scorers.py`
- `python/mojolearn/tests/test_native_decomp_buffer_owners.py`
- `python/mojolearn/tests/test_native_metric_scorer_helpers.py`
- `python/mojolearn/tests/test_native_rfe_cold_context.py`
- `python/mojolearn/tests/test_native_sgd_weight_lifetime.py`
- `python/mojolearn/tests/test_native_steady_state.py`
- `python/mojolearn/tests/test_native_stratified_connection.py`
- `python/mojolearn/tests/test_native_transient_buffer_lifetimes.py`
- `python/mojolearn/tests/test_ne2_g_decomp_entries.py`
- `python/mojolearn/tests/test_ne2_search_workers.py` (delete 3 function(s))
- `python/mojolearn/tests/test_ocsvm_native_transport_boundary.py`
- `python/mojolearn/tests/test_ops_graph.py`
- `python/mojolearn/tests/test_optics_native_boundary.py`
- `python/mojolearn/tests/test_parallel_cv_installed.py`
- `python/mojolearn/tests/test_parallel_cv_source.py`
- `python/mojolearn/tests/test_parallel_neural_native.py` (delete 2 function(s))
- `python/mojolearn/tests/test_pca_exact_native_owner.py` (delete 2 function(s))
- `python/mojolearn/tests/test_peer1_provider_boundary.py` (delete 1 function(s))
- `python/mojolearn/tests/test_portable_kernel_models.py`
- `python/mojolearn/tests/test_ptx_fallback.py`
- `python/mojolearn/tests/test_public_cnn_verifier_methods.py` (delete 1 function(s))
- `python/mojolearn/tests/test_public_model_selection_verifier_gaps.py`
- `python/mojolearn/tests/test_py_bugs.py` (delete 4 function(s))
- `python/mojolearn/tests/test_regression_metrics.py` (delete 2 function(s))
- `python/mojolearn/tests/test_resident_sampling_policy.py`
- `python/mojolearn/tests/test_rf_class_weight.py` (delete 3 function(s))
- `python/mojolearn/tests/test_runtime_classical_partition.py`
- `python/mojolearn/tests/test_runtime_report.py`
- `python/mojolearn/tests/test_samba_attention_wiring.py` (delete 1 function(s))
- `python/mojolearn/tests/test_samba_checkpoint_native.py`
- `python/mojolearn/tests/test_skewed_chi2_native_capability.py`
- `python/mojolearn/tests/test_standard_archive_native.py`
- `python/mojolearn/tests/test_svr_native_transport_boundary.py` (delete 1 function(s))
- `python/mojolearn/tests/test_tokenizer_fingerprint.py`
- `python/mojolearn/tests/test_torch_program_module_admission.py`
- `python/mojolearn/tests/test_torch_state_dict.py`
- `python/mojolearn/tests/test_trees_repeat.py`
- `python/mojolearn/tests/test_umap_transform.py` (delete 3 function(s))
- `python/mojolearn/tests/test_verification_coverage.py`
- `python/mojolearn/tests/test_verify_causal_lm.py`
- `python/mojolearn/tests/test_verify_compare_challenge.py`
- `python/mojolearn/tests/test_verify_compare_commitment.py`
- `python/mojolearn/tests/test_verify_compare_profiles.py`
- `python/mojolearn/tests/test_verify_distributed.py`
- `python/mojolearn/tests/test_verify_model.py`
- `python/mojolearn/tests/test_verify_public_api.py`
- `python/mojolearn/tests/test_weighted_score.py` (delete 1 function(s))
- `python/mojolearn/tests/test_x_ann_repeat.py`
- `python/mojolearn/tests/test_x_cnn_batchnorm.py`
- `python/mojolearn/tests/test_x_cnn_blocks.py`
- `python/mojolearn/tests/test_x_cnn_conv.py`
- `python/mojolearn/tests/test_x_cnn_gnn.py` (delete 4 function(s))
- `python/mojolearn/tests/test_x_cnn_pool.py`
- `python/mojolearn/tests/test_x_metrics_repeat.py`
- `python/mojolearn/tests/test_x_metrics_sanity.py`
- `python/mojolearn/tests/test_x_neighbors_repeat.py`
- `python/mojolearn/tests/test_x_neighbors_sanity.py` (delete 17 function(s))
- `python/mojolearn/tests/test_x_prep_parity.py` (delete 6 function(s))
- `python/mojolearn/tests/test_x_prep_scaler_options.py` (delete 5 function(s))
- `python/mojolearn/tests/test_x_prep_twice.py`
- `python/mojolearn/tests/test_x_sequence_mlp.py` (delete 1 function(s))
- `python/mojolearn/tests/test_x_sequence_rnn.py`
- `python/mojolearn/tests/test_x_sequence_theta.py`
- `python/mojolearn/tests/test_x_trees_sanity.py` (delete 22 function(s))
- `tests/api/test_digest.py`
- `tests/api/test_gbdt_category_inputs.py`
- `tests/api/test_host_gbdt_category_map.py`
- `tests/proving/_arm_cases.py`
- `tests/proving/test_proving_categories.py`
- `tests/proving/test_proving_digest.py`
- `tests/proving/test_proving_threads.py`
- `tests/test_autograd_smoke.py`
- `tests/tools/test_six_lane_matrix_io.py`
- `tests/tools/test_six_lane_review_history.py`

## The clean pass, first round (lane py-delete, 2026-11-23)

The rule for deleting a row: its Mojo twin is a TestSuite `test_*` on the integration head, it passed in the newest
green native run of an integration first-parent commit (cpc2-0016 at d19cf96968: 387 of 387 checks, per-test PASS
lines read from its logs), a read of both tests found that it asserts at least what the Python test asserts (same
inputs or a superset, same tolerance or tighter, no boundary assertion lost), and no GPU run is owed: the twin has no
unrun device column and the Python test is not a GPU-box test. Of the 150 rows marked covered or converted, that
removed 26 test functions, 7 of them as whole files (`test_x_prep_binarizer`, `_discriminant`, `_imputer`, `_kbins`,
`_normalizer`, `_power`, `_quantile`). The other 124 keep their Python test, and the status column says why:
partial (36), GPU run owed (67), a twin that is a `main()` check the runner does not run (8), or another run owed
(13). The lists above are the plan made before the batches ran. The map's status column is the record now.
