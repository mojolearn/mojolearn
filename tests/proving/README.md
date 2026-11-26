# Proving suite: the installed-wheel RUN OWED items of the 2026-11-21 plan wave

Every item below was written by a lane that ran nothing; this suite proves it on a built wheel at the Python
boundary. Results of the lanes: `~/mojolearn-evidence/briefs-20261121/<lane>-RESULT.md`. Small data, seconds per test.
Run it from a repo checkout with the wheel installed in the interpreter (the source `python/` tree is never imported
as the runtime; `mojolearn.tests` modules are mounted over the installed runtime as `tools/run_python_tests.py
--installed` does).

    python -m pytest tests/proving -x -q -n auto           # CPU (device="cpu"); every gpu test skips cleanly
    python -m pytest tests/proving -x -q -m gpu -n auto    # GPU-marked tests only (needs a GPU install)

On a GPU install the first command runs the CPU route and the GPU route together.

`-n auto` (pytest-xdist) spreads the tests over every core; without xdist drop it (`python -m pytest tests/proving -x -q`).
The 22 cases of `_arm_cases.py` run in subprocesses split across every core (`MOJOLEARN_PROVING_WORKERS` caps the
shards). Their two `_OFF` build arms (`MOJOLEARN_MSEL_MEMBER_UNITS_OFF`, `MOJOLEARN_MEMBER_SET_OFF`) were removed on
2026-11-23 as plumbing (docs/TOMBSTONES.md).

## Items -> tests -> source

| Item | Test (tests/proving/...) | RESULT file |
|---|---|---|
| cross_validate(KernelDensity / Ridge / LogisticRegression, return_estimator=True): kept estimators equal hand fits after the fold buffers are freed; `classes_` type | `test_proving_lifetime.py::test_cross_validate_kept_estimators_outlive_the_fold_buffers` | fx-borrow, ma-core (round 3) |
| BaggingClassifier(DecisionTreeClassifier()) after the input is freed (k-NN members are refused by name) | `test_proving_lifetime.py::test_bagging_predicts_the_same_after_the_input_is_freed` | fx-borrow |
| GP search refit, predict after `del X` | `test_proving_lifetime.py::test_gaussian_process_search_refit_predicts_after_del_x` | fx-borrow |
| IsolationForest held predict after adopt | `test_proving_lifetime.py::test_isolation_forest_held_predict_after_the_input_is_freed` | fx-borrow, ma-pub-trees |
| msel refit of RandomForestClassifier has `classes_` | `test_proving_lifetime.py::test_search_refit_of_a_forest_classifier_has_classes` | sweep-b |
| bagging over a non-tree member (GaussianNB, k-NN, Ridge, ...) is refused by name, as documented; the member-ABI route is held work | `test_proving_lifetime.py::test_bagging_refuses_a_member_that_is_not_a_tree_in_words` | fx-audit (session 2) |
| two threads fit + predict: gbdt, forest, tree, ET, DART, linear, cd, sgd, xlinear, logistic, qn_regressor, linear_svm | `test_proving_threads.py::test_two_threads_fit_and_predict_like_one` | fx-gil |
| four threads predict ONE fitted model (own-state path) | `test_proving_threads.py::test_two_threads_predict_one_fitted_model` | fx-gil (round 4) |
| two threads save ONE fitted RF / ET | `test_proving_threads.py::test_two_threads_save_one_fitted_forest` | fx-gil (round 4) |
| cross_val_score over members from two threads; n_jobs=-1 == n_jobs=1 | `test_proving_threads.py::test_two_threads_cross_val_score_like_one` | fx-gil (rounds 2, 4) |
| Bagging over RF / ET / GBDT / DART / Ridge from two threads | `test_proving_threads.py::test_two_threads_bagging_*` | fx-gil (rounds 2, 3) |
| LinearRegression fit from two threads on the GPU (kept TSQR state) | `test_proving_threads.py::test_two_threads_fit_linear_regression_on_the_gpu` (gpu) | fx-gil (round 4) |
| CausalLM lease: a second caller gets a private session, never a refusal | `test_proving_causal_lease.py` (reuses `mojolearn.tests.test_causal_lease_two_threads`; gpu) | causal-lease, causal-doors |
| digest: same fit = same digest, changed data, load restores it, CPU == GPU (linear), private state, ... | `test_proving_digest.py` (reuses every test of `tests/api/test_digest.py`) | digest (rounds 1-3) |
| GPU digest == CPU digest for every case family | `test_proving_digest.py::test_proving_gpu_digest_equals_cpu_digest_for_every_case` (gpu) | digest (round 2 item 7) |
| predict / transform leave the digest | `test_proving_digest.py::test_proving_predict_and_transform_leave_the_digest` | digest (round 3 item 8b) |
| load restores the digest of the public-rule loadable classes | `test_proving_digest.py::test_proving_loaded_model_has_the_fitted_digest` | digest (round 3 item 8a) |
| HostForest (rf, et) / HostGBDT digest == the saving model's | `test_proving_digest.py::test_proving_host_twin_has_the_saving_models_digest` | fx-audit (session 3) |
| `__init__(device="cpu")` on a fitted GPU model, then predict / refit | `test_proving_digest.py::test_proving_reinit_to_cpu_then_predict_runs_and_refit_keeps_the_digest` (gpu) | fx-audit (session 3) |
| resident(): cross_val_score(GaussianNB()) uploads X once, equals outside, no pin left | `test_proving_resident.py::test_gaussian_nb_cross_val_score_in_resident_equals_outside_and_uploads_once` (gpu) | ma-core (round 6 items 2) |
| resident() with a failing unit and error_score=0.0 | `test_proving_resident.py::test_failing_unit_in_resident_gives_error_score_and_releases_the_pin` (gpu) | ma-core (round 6 item 3) |
| nested resident() shares the outer uploads | `test_proving_resident.py::test_nested_resident_scope_shares_the_outer_uploads` (gpu) | arena |
| cross_val_score / cross_validate / GridSearchCV of Ridge, LinearRegression (n_jobs 1 and -1, KFold no shuffle, TimeSeriesSplit) | `test_proving_model_selection.py` cases `cvs_ridge_default`, `cvs_linear_regression_n_jobs`, `cv_ridge_kfold_and_time_series` | ma-core (rounds 1-2) |
| named scorers (neg_mean_squared_error; accuracy, f1_macro, neg_log_loss with train scores; f1 on labels {1, 2}) | cases `cvs_ridge_neg_mean_squared_error`, `cv_logistic_named_scorers`, `cvs_logistic_f1_labels_1_2`, `cvs_logistic_neg_log_loss` | ma-core (rounds 4, 6) |
| cross_val_predict Ridge / LogisticRegression / predict_proba / SVC decision_function (2 and 4 classes), dtype included | cases `cvp_ridge`, `cvp_logistic`, `cvp_svc_decision_function` | ma-core (rounds 4, 6) |
| error_score=0.5 with a failing candidate | case `cvs_error_score` (+ the failing candidate's splits 0.5 asserted) | ma-core (round 4) |
| GridSearchCV best_estimator_ predict, `classes_` type, get_params | case `grid_ridge_and_logistic` | ma-core (round 4) |
| fallback families (PCA, KMeans, LogisticRegression, k-NN, NearestCentroid) | case `cvs_member_fallback_families` | ma-core (rounds 2, 5) |
| MLP / PLS / CCA cross_val_score, GridSearchCV(MLPRegressor) | case `cvs_neural_and_pls` | ma-core (round 3) |
| LSTMRegressor cross_val_score | case `cvs_lstm_regressor` | ma-core (round 5) |
| Bagging (max_features < 1, bootstrap_features, oob_score, sample_weight, rare class) | case `bagging_over_trees` | ma-callers (rounds 1, 3, 4) |
| Voting soft / hard (n_jobs None, -1, 1, 2), Stacking proba / decision, StackingRegressor predict / transform, folds > cores, MultiOutput, OvR, Calibrated | case `voting_stacking_multioutput_ovr` | ma-callers (rounds 1-3) |
| members whose classes are not 0..k-1 | case `prefit_labels_not_zero_based` | ma-callers (round 3) |
| RFE over LogisticRegression (binary, multiclass), tree, forest; `estimator_.classes_` | case `rfe_members` | ma-callers (rounds 2, 4) |
| Kernel / Permutation explainers over Ridge, LinearRegression, forest | case `explainers_over_members` | ma-callers (round 2) |
| NaN in X and in y, fit and predict: the estimator's own refusal words, every meta-estimator, cross_val_score, cross_val_predict, GridSearchCV refit | case `nan_refusals` | ma-callers (rounds 3, 4) |
| a cv fold missing a class: Stacking and Calibrated admit it (as scikit-learn) | case `missing_class_folds` | ma-callers (round 3) |
| n_jobs=0 -> ValueError | `test_proving_model_selection.py::test_n_jobs_zero_is_refused` | ma-callers (round 2) |
| a failing candidate raises the estimator's own words | `test_proving_model_selection.py::test_failing_candidate_raises_the_estimators_own_words` | ma-core (rounds 1-2) |
| unregistered member list == ['CNNClassifier', 'DBSCAN'] | `test_proving_model_selection.py::test_every_family_is_registered_for_the_member_route` | ma-core (rounds 5-6) |
| GPU `import mojolearn` loads no `_mojolearn_*_host` image; device="cpu" loads it on demand | `test_proving_device_route.py::test_gpu_import_loads_no_cpu_host_image`, `::test_cpu_device_fit_loads_its_host_image_on_demand` (gpu) | fx-host-import, fx-audit (session 3) |
| device="cpu" StandardScaler / PCA / NMF / KernelPCA / Ridge / LinearRegression / Lasso / LogisticRegression / QNRegressor / RandomForest / MLP, and after pickle in a fresh process | `test_proving_device_route.py::test_cpu_device_model_runs_after_pickle_in_a_fresh_process` | fx-host-import, fx-audit (session 3) |
| device="cpu" LogisticRegression / SVC / RandomForestClassifier / k-NN / LinearSVC / LSTMClassifier fit + predict + score, int / str / sparse labels | `test_proving_device_route.py::test_cpu_device_classifier_labels_round_trip`, `::test_cpu_device_lstm_classifier_fit_predict_score` | label-steps, fx-audit (session 2) |
| device assignment (`model.device = ...`, blocks, pickles) | `test_proving_device_route.py` (reuses every test of `tests/api/test_device_assignment.py`) | fx-device-attr |
| kernel SVM (SVC binary, SVR) save / load round trip | `test_proving_device_route.py::test_kernel_svm_save_load_round_trip` | fx-audit (session 2) |
| PLS repeated predicts | `test_proving_device_route.py::test_pls_repeated_predicts_are_equal` | fx-audit (session 2) |
| GradientBoosting categorical inputs: object, U / S, pandas 2 / 3, Arrow string / large / chunked / sliced / dictionary, refusals | `test_proving_categories.py` (reuses the categorical-input tests of `mojolearn.tests.test_gbdt_boundary`) | cat-ctr (round 11), arrow-strings |
| numpy.str_ / bytes / numpy.int64 / numpy.float32 cells; str and bytes of one value = one category; every column categorical S / U | `test_proving_categories.py::test_proving_*` | cat-ctr (round 11 cells) |
| HostGBDT category map | `test_proving_categories.py` (reuses the HostGBDT category-map tests of `mojolearn.tests.test_gbdt_boundary`) | fx-hostgbdt |

## Existing installed-wheel test files the RESULT files name (run as they are, not copied)

    python tests/proving/run_installed.py <files>      # each file on the installed wheel, files across every core

- api-boost: `tests/api/test_boosting_parameter_table.py tests/api/test_boosting_parameter_names.py`
- api-sklearn: `python/mojolearn/tests/test_sklearn_compat.py python/mojolearn/tests/test_native_field_constructors.py`;
  `MOJOLEARN_RUN_SKLEARN_CHECKS=1 bash tools/sklearn_checks_per_estimator.sh <py> <site-packages>/mojolearn/tests <out>`
- api-torch, fx-leftovers, causal-lease: `tests/api/test_torch_conventions.py python/mojolearn/tests/test_parallel_causal_lm.py
  python/mojolearn/tests/test_x_cnn_gnn.py python/mojolearn/tests/test_family_device_boundary.py
  python/mojolearn/tests/test_device_rows_metadata_native.py python/mojolearn/tests/test_resident_lifecycle_native.py`
- causal-lease (GPU, `MOJOLEARN_GENERATION_DEVICE=gpu`): `python/mojolearn/tests/test_resident_generation_contract.py
  python/mojolearn/tests/test_resident_sampling_policy.py python/mojolearn/tests/test_generation_route_controls.py
  python/mojolearn/tests/test_resident_profile_contract.py`
- arena: `python/mojolearn/tests/test_device_cache_native_state.py`; `python3 tools/test_arena_rows_connection.py`;
  `python3 tools/test_qualify_native_runtime_installed.py`
- byte-state: `python/mojolearn/tests/test_byte_lm_session.py python/mojolearn/tests/test_byte_lm_native_state_restore.py
  python/mojolearn/tests/test_byte_lm_host_trainer.py`; `python tools/byte_lm_session_check.py`
- pools-frame: `python/mojolearn/tests/test_native_process_pool.py test_native_parallel_cv_wire.py
  test_parallel_model_selection.py test_parallel_public_drivers.py test_parallel_neural_native.py
  test_parallel_neural_admission.py test_parallel_trainers_coverage.py test_byte_checkpoint_native.py
  test_byte_lm_checkpoint_binary.py test_models_coverage.py test_native_parallel_public_boundary.py
  test_cpu_parallel_rbf_sampler.py test_numpy_free_features.py test_distributed_ivf.py
  test_parallel_api_regression.py` (all under python/mojolearn/tests; test_verify_distributed.py moved to tools/multi_gpu/tests/ on 2026-11-26; `test_native_parallel_cv_wire.py:115-118` is
  STALE, owed a rewrite: pools-frame 13.7); `python tools/pool_spawn_cost_probe.py`
- fx-hostgbdt: `python/mojolearn/tests/test_host_tree_lifecycle_native.py python/mojolearn/tests/test_gbdt_host_ctr.py`
- ctr-default: `python/mojolearn/tests/test_gbdt_catboost_defaults.py`
- ma-pub-linear: `python/mojolearn/tests/test_linear_svc_native_boundary.py python/mojolearn/tests/test_linear_svr_native_fit_boundary.py`
- ma-pub-neighbors: `python/mojolearn/tests/test_kpca_native_boundary.py`
- ma-pub-trees: `python/mojolearn/tests/test_host_iforest_native.py` and the IsolationForest save / load / pickle tests
  (`grep -l IsolationForest python/mojolearn/tests`)
- digest, causal-doors, fx-audit: `python tools/gen_digest_fields.py` then commit `docs/api/DIGEST_FIELDS.md`;
  `python tools/test_generation_metadata_boundary.py`; `python -m pytest -q tools/test_causal_model_native.py`
- verify (every lane that touched what verify imports): `bash tools/ops/wheel_verify_smoke.sh`

## Native-only and box items (NOT ported: they are Mojo checks, box jobs or tool tests, not the Python boundary)

- Mojo native suites: `checks/native_tests/member_abi.mojo` (member_abi_suite run_checks: every member family's
  fit / predict / export / import vs its door, sweep-a/b, ma-pub-*), `checks/native_tests/arena_state_suite.mojo`,
  `checks/native_tests/trees_entry_gbdt_suite.mojo`, `checks/native_tests/trees_entry_forest_suite.mojo`,
  `checks/native_tests/iforest_forest_export.mojo`, `checks/native_tests/causal_owner.mojo`,
  `checks/native_tests/fitted_digest_suite.mojo`, `checks/forest_inference_model.mojo`,
  `checks/device_context_concurrent_repro.mojo`, `tests/boundary/forest_export_protocol.mojo`, native checks
  `transport_frame`, `msel_helpers_entry`, `prep_entry_table`: `pixi run mojo run -I . <file>` on the CPU box.
- GBDT host vs device (byte-for-byte model text): `pixi run check-native-gpu gbdt_tree_ctr_host_check`,
  `ordered_ctr_device_check`, `gbdt_ordered_ctr_host_check`, `gbdt_tree_ctr_check`; host checks
  `pixi run mojo run -I . gbdt/checks/<name>_host_check.mojo` for every family (gbdt-host-par rounds 1-4),
  `gbdt/checks/category_map_check_main.mojo`, `combination_map_check_main.mojo`, `tree_ctr_check_main.mojo`,
  `checks/model_io_check.mojo`, `checks/ctr_apply_check.mojo`, `core/host_parallel_check.mojo`,
  `training/checks/byte_lm_optimizer_bits_check.mojo`; the nested-parallel reproducer of gbdt-host-par round 2.
- Feature proof job (digest lines per family, GPU then CPU replay):
  `tools/nvidia_central.sh submit ab --cap 30 --results box-out/proof-<tag> tools/ops/feature_proof_job.sh`, then
  `FP_REPLAY=<receipts> tools/ops/feature_proof_job.sh` on a CPU box.
- Busy cores of total during a `device="cpu"` GBDT fit and its 5-fold CV (gbdt-host-par): read on the CPU box.
- Identity column of a new verify lane (`gbdt-categorical-tree-ctr`, cat-ctr): `LANES=gbdt-categorical-tree-ctr bash
  tools/record_identity_column.sh <vendor> <outdir>` at the release's one verification.
- A/B rows (timing, NVIDIA L40S): member vs door
  inside `resident()` (ma-pub-transform rounds 4-10), resident read rows (causal-lease).
- Tool tests: `python3 tools/tests/test_check_tree_additions.py`, `python3 tools/test_box_persist.py`,
  `python3 -m pytest tools/test_native_checks_ci.py -k keeps_results tools/test_release_gate_jobs.py tools/test_record_index.py
  tools/test_verification_repository.py tools/test_byte_lm_callsites.py -q`, `python3 tools/record_index.py add
  --unlisted-under bench/results/identity_break && python3 tools/record_index.py check`, `python tools/test_cpu_central.py`.
