# Tests to Mojo, lane tests-to-mojo-classical (2026-11-15)

The 58 files of the `L-classical` rows of `TESTS_TO_MOJO_2026-11-15.md`, read function by function. Nothing was
compiled or run (lane rule). Per-function verdicts with evidence:
`~/mojolearn-evidence/briefs-20261115/tests-to-mojo-classical/verdict_{A,B,C,D,E}.tsv`.

Verdicts: KEEP-BOUNDARY (only Python can test it: admission, exceptions, returned objects, hooks, one-call audits,
pickling, archives, outside oracles); KEEP-NOPORT (native numerics whose inputs come from NumPy's random stream or
that need Python-side state; listed below); COVERED (an existing check asserts the same, assert by assert: none
found); SOURCE-TEXT (asserts on source, manifests, private names or layout: deleted, as the retired source guards
of AGENTS.md are); PORT (moved to a new Mojo check and deleted).

Totals: 58 files; 0 deleted, 20 shrunk, 38 kept unchanged; 1330 Python lines removed, 87
added (docstrings and one public shape test). Four new checks, registered in `checks/native_tests/registry.mojo`
(`tools/ops/native_checks_report.py plan` now lists 150 cpu and 148 gpu checks).

| file | lines | result | removed | added | function verdicts |
|---|---:|---|---:|---:|---|
| `test_c3_c5_no_copies.py` | 125 | kept | 0 | 0 | KEEP-BOUNDARY 6 |
| `test_cholesky_archive_native.py` | 111 | kept | 0 | 0 | KEEP-BOUNDARY 3 |
| `test_classical_archive_native_owner.py` | 332 | kept | 0 | 0 | KEEP-BOUNDARY 11 |
| `test_classical_archive_reader_native.py` | 837 | kept | 0 | 0 | KEEP-BOUNDARY 26 |
| `test_classical_infer_one_call.py` | 143 | kept | 0 | 0 | KEEP-BOUNDARY 6 |
| `test_cpu_kernel_variants.py` | 103 | moved to kernel_methods_poly_degree; shrunk | 42 | 3 | KEEP-BOUNDARY 2, PORT 1, SOURCE-TEXT 1 |
| `test_cpu_training_d.py` | 314 | shrunk (source-text deleted) | 206 | 7 | KEEP-NOPORT 1, SOURCE-TEXT 9 |
| `test_cpu_training_e.py` | 208 | shrunk (source-text deleted) | 113 | 8 | KEEP-NOPORT 1, SOURCE-TEXT 6 |
| `test_cpu_training_e2.py` | 412 | shrunk (source-text deleted) | 219 | 9 | KEEP-NOPORT 5, SOURCE-TEXT 12 |
| `test_cpu_training_e3.py` | 152 | shrunk (source-text deleted) | 104 | 8 | KEEP-NOPORT 1, SOURCE-TEXT 5 |
| `test_cpu_training_misc.py` | 402 | shrunk (source-text deleted) | 205 | 10 | KEEP-BOUNDARY 4, KEEP-NOPORT 2, SOURCE-TEXT 9 |
| `test_cpu_training_par_classical.py` | 346 | shrunk (source-text deleted) | 107 | 9 | KEEP-BOUNDARY 5, KEEP-NOPORT 1, SOURCE-TEXT 1 |
| `test_estimators_host_borrowed_buffers.py` | 58 | kept | 0 | 0 | KEEP-BOUNDARY 2 |
| `test_gp_normalize_y.py` | 135 | shrunk (source-text deleted) | 14 | 1 | KEEP-BOUNDARY 5, SOURCE-TEXT 1 |
| `test_gp_return_cov.py` | 53 | kept | 0 | 0 | KEEP-BOUNDARY 3 |
| `test_gpc_surface.py` | 236 | shrunk (source-text deleted) | 48 | 4 | KEEP-BOUNDARY 4, SOURCE-TEXT 2 |
| `test_host_archive_native.py` | 553 | shrunk (source-text deleted) | 12 | 0 | KEEP-BOUNDARY 23, SOURCE-TEXT 1 |
| `test_host_density_hash_native.py` | 278 | kept | 0 | 0 | KEEP-BOUNDARY 10 |
| `test_host_density_save_native.py` | 375 | kept | 0 | 0 | KEEP-BOUNDARY 16 |
| `test_host_kde_native.py` | 884 | shrunk (source-text deleted) | 12 | 0 | KEEP-BOUNDARY 42, SOURCE-TEXT 1 |
| `test_host_logistic_native.py` | 463 | shrunk (source-text deleted) | 12 | 0 | KEEP-BOUNDARY 26, SOURCE-TEXT 1 |
| `test_host_model_linear_kernel.py` | 101 | kept | 0 | 0 | KEEP-BOUNDARY 2 |
| `test_host_model_svm.py` | 95 | kept | 0 | 0 | KEEP-BOUNDARY 3 |
| `test_host_svc_native.py` | 579 | shrunk (source-text deleted) | 12 | 0 | KEEP-BOUNDARY 32, SOURCE-TEXT 1 |
| `test_host_svr_native.py` | 421 | shrunk (source-text deleted) | 11 | 0 | KEEP-BOUNDARY 25, SOURCE-TEXT 1 |
| `test_kde_fused_fast.py` | 156 | kept | 0 | 0 | KEEP-BOUNDARY 1, KEEP-NOPORT 3 |
| `test_km_kernels.py` | 71 | kept | 0 | 0 | KEEP-BOUNDARY 3 |
| `test_krr_options.py` | 57 | kept | 0 | 0 | KEEP-BOUNDARY 2 |
| `test_linear_archive_generic_native.py` | 88 | kept | 0 | 0 | KEEP-BOUNDARY 1 |
| `test_linear_svc_native_boundary.py` | 127 | kept | 0 | 0 | KEEP-BOUNDARY 8 |
| `test_logistic_native_predict.py` | 47 | kept | 0 | 0 | KEEP-BOUNDARY 2 |
| `test_logistic_query_width_native.py` | 258 | kept | 0 | 0 | KEEP-BOUNDARY 11 |
| `test_minmax_archive_generic_native.py` | 113 | kept | 0 | 0 | KEEP-BOUNDARY 2 |
| `test_minmax_scaler_native_lifecycle.py` | 295 | kept | 0 | 0 | KEEP-BOUNDARY 11, KEEP-NOPORT 4 |
| `test_multiclass_ova_surface.py` | 64 | kept | 0 | 0 | KEEP-BOUNDARY 3 |
| `test_native_codec.py` | 287 | kept | 0 | 0 | KEEP-BOUNDARY 13 |
| `test_native_decomp_buffer_owners.py` | 80 | kept | 0 | 0 | KEEP-BOUNDARY 1, KEEP-NOPORT 1 |
| `test_native_owner_qualification.py` | 91 | kept | 0 | 0 | KEEP-BOUNDARY 3 |
| `test_native_prep_iterimp_door.py` | 651 | shrunk (source-text deleted) | 26 | 1 | KEEP-BOUNDARY 1, KEEP-NOPORT 2, SOURCE-TEXT 1 |
| `test_native_prep_no_callbacks.py` | 397 | shrunk (source-text deleted) | 65 | 2 | KEEP-BOUNDARY 13, SOURCE-TEXT 5 |
| `test_native_sgd_weight_lifetime.py` | 62 | kept | 0 | 0 | KEEP-NOPORT 1 |
| `test_native_transient_buffer_lifetimes.py` | 213 | kept | 0 | 0 | KEEP-BOUNDARY 4, KEEP-NOPORT 4 |
| `test_native_transport.py` | 447 | kept | 0 | 0 | KEEP-BOUNDARY 15 |
| `test_ne2_g_decomp_entries.py` | 128 | kept | 0 | 0 | KEEP-BOUNDARY 2, KEEP-NOPORT 6 |
| `test_numpy_free_core.py` | 510 | moved to core_all_finite; shrunk | 16 | 6 | KEEP-BOUNDARY 9, PORT 1 |
| `test_numpy_free_late_connect.py` | 93 | kept | 0 | 0 | KEEP-BOUNDARY 2 |
| `test_ocsvm_native_transport_boundary.py` | 290 | kept | 0 | 0 | KEEP-BOUNDARY 14 |
| `test_pca_exact_native_owner.py` | 385 | kept | 0 | 0 | KEEP-BOUNDARY 9, KEEP-NOPORT 2 |
| `test_peer1_provider_boundary.py` | 484 | shrunk (source-text deleted) | 23 | 0 | KEEP-BOUNDARY 15, SOURCE-TEXT 1 |
| `test_public_verifier_gap_methods.py` | 69 | moved to sequence_theta_variants, training_residual_dropout_mask; shrunk | 28 | 11 | KEEP-BOUNDARY 3, KEEP-NOPORT 1, PORT 2 |
| `test_py_bugs.py` | 231 | shrunk (source-text deleted) | 55 | 8 | KEEP-BOUNDARY 6, SOURCE-TEXT 1 |
| `test_scaler_input_snapshot_provider.py` | 101 | kept | 0 | 0 | KEEP-BOUNDARY 3 |
| `test_skewed_chi2_native_capability.py` | 25 | kept | 0 | 0 | KEEP-BOUNDARY 1 |
| `test_standard_archive_native.py` | 122 | kept | 0 | 0 | KEEP-BOUNDARY 2 |
| `test_standard_scaler_native_fit.py` | 137 | kept | 0 | 0 | KEEP-BOUNDARY 7, KEEP-NOPORT 1 |
| `test_svc_multiclass.py` | 183 | kept | 0 | 0 | KEEP-BOUNDARY 8, KEEP-NOPORT 3 |
| `test_svc_probability.py` | 83 | kept | 0 | 0 | KEEP-BOUNDARY 4 |
| `test_x_prep_twice.py` | 93 | kept | 0 | 0 | KEEP-NOPORT 1 |

## New checks

| check | moved from | calls |
|---|---|---|
| `core_all_finite` | `test_numpy_free_core.py::test_all_finite_host_path` (numeric verdicts; the empty shortcut and the int TypeError stay as `test_all_finite_python_boundary`) | `core/host_helpers_impl.mojo` `all_finite_f32/f64` (cpu), `LabelsDeviceKernels.finite` (gpu) |
| `kernel_methods_poly_degree` | `test_cpu_kernel_variants.py::test_polynomial_degree_affects_host_fit_and_inference` | `kernel_ridge_fit_host_run` / `kernel_ridge_predict_host_run` (cpu), `kernel_ridge_fit_host` / `kernel_ridge_predict_host` (gpu) |
| `sequence_theta_variants` | `test_public_verifier_gap_methods.py::test_theta_variant_batch_forecasts_preserve_single_series` (the public 2-D/1-D shapes stay as `test_theta_variant_public_batch_and_single_shapes`) | `sequence/ops_impl.mojo` `theta_impl` (HostExec / DeviceExec) |
| `training_residual_dropout_mask` | `test_public_verifier_gap_methods.py::test_residual_dropout_forward_and_backward_obey_same_mask` | `residual_dropout_host` (cpu), `residual_dropout_device` (gpu) |

## Kept, not ported (KEEP-NOPORT)

| file | function | reason |
|---|---|---|
| `test_minmax_scaler_native_lifecycle.py` | `test_fit_transform_inverse_and_checkpoint_are_native` | inputs from np.random.default_rng(911); also returned ndarray type and save/load through B |
| `test_minmax_scaler_native_lifecycle.py` | `test_partial_fit_preserves_extrema_and_counts` | inputs from np.random.default_rng(911); compares public fit vs partial_fit |
| `test_minmax_scaler_native_lifecycle.py` | `test_partial_fit_preserves_previously_unseen_column_nan` | inputs from np.random.default_rng(911) |
| `test_minmax_scaler_native_lifecycle.py` | `test_unseen_column_mask_survives_finite_transform_input` | inputs from np.random.default_rng(911) |
| `test_standard_scaler_native_fit.py` | `test_disabled_direct_entry_stays_native_and_preserves_bits` | inputs from np.random.default_rng(102); route chosen by MOJOLEARN_SCALER_DIRECT read in th |
| `test_cpu_training_d.py` | `test_estimators_fit_on_the_host_when_built` | np.random.default_rng(0) inputs; public-API twice-same-bytes + per-process MOJOLEARN_CPU_T |
| `test_cpu_training_e.py` | `test_host_fits_run_and_repeat_on_a_cpu_only_install` | np.random.default_rng(7) inputs; also refusal message through the Python-callable host ent |
| `test_cpu_training_e2.py` | `test_spectral_runs_on_the_host_when_built` | np.random.default_rng(2) inputs; public-API repeat/range checks |
| `test_cpu_training_e2.py` | `test_scalers_run_on_the_host_when_built` | np.random.default_rng(3) inputs; public-API repeat/range checks |
| `test_cpu_training_e2.py` | `test_logistic_runs_on_the_host_when_built` | np.random.default_rng(4) inputs; public-API repeat/range checks |
| `test_cpu_training_e2.py` | `test_metrics_run_on_the_host_when_built` | np.random.default_rng(1) inputs; public-API repeat/range checks |
| `test_cpu_training_e2.py` | `test_kmeans_fit_runs_on_the_host_when_built` | np.random.default_rng(0) inputs; public-API repeat/range checks |
| `test_cpu_training_e3.py` | `test_random_forests_fit_on_the_host_when_built` | np.random.default_rng(0) inputs; public-API repeat/range |
| `test_cpu_training_misc.py` | `test_bootstrap_runs_on_the_host_when_built` | np.random.default_rng(2) inputs; public resample API |
| `test_cpu_training_misc.py` | `test_neural_primitives_run_on_the_host_when_built` | rng weights vs NumPy gather; clip refusal by name (boundary) |
| `test_cpu_training_par_classical.py` | `test_the_gpc_class_fit_worker_follows_the_public_fit_guard` | rng(11) inputs; public plain fit vs native process-pool driver (worker processes); decomp_ |
| `test_kde_fused_fast.py` | `test_fused_equals_staged_within_float32_fold_tolerance` | FAST-only arm (Apple fast programme paused; native checks build identical); staged arm sel |
| `test_kde_fused_fast.py` | `test_all_out_of_range_row_scores_float_min_not_minus_inf` | FAST-only arm (paused programme; checks build identical); staged arm selected via MOJOLEAR |
| `test_kde_fused_fast.py` | `test_query_score_does_not_depend_on_its_batch` | FAST-only arm (paused programme); inputs from Python random.Random gauss stream |
| `test_public_verifier_gap_methods.py` | `test_johnson_lindenstrauss_dimension_matches_stated_bound` | no typed RUN entry for the bound: composition and int conversion live in the PACK (binding |
| `test_svc_multiclass.py` | `test_each_pair_is_the_binary_solver_on_its_rows` | RNG-bound input (np.random.default_rng(5) normal); public SVC refits per pair |
| `test_svc_multiclass.py` | `test_gamma_scale_is_resolved_on_the_whole_x` | RNG-bound input; compares private _gamma with Python oracle _scale_gamma_oracle |
| `test_svc_multiclass.py` | `test_precomputed_is_the_linear_kernel_bit_for_bit` | RNG-bound integer input (default_rng(9).integers); whole public SVC/SVR fit through two ke |
| `test_native_prep_iterimp_door.py` | `test_door_matches_former_route[9 cases]` | RNG-bound input (np.random.default_rng in _data); oracle is the former Python route (_Form |
| `test_native_prep_iterimp_door.py` | `test_door_matches_former_route_mojolearn_estimator[2]` | RNG-bound input; wrapped Ridge estimator through public API; Python oracle |
| `test_x_prep_twice.py` | `test_x_prep_run_many_times_twice_in_one_process[gpu,cpu]` | RNG-bound input (default_rng(7)); child-process hang/timeout of 20 public estimators on th |
| `test_native_sgd_weight_lifetime.py` | `test_unit_class_weights_preserve_weighted_sgd_fit[2,3]` | RNG-bound X (default_rng(53)); purpose is lifetime of the Python-allocated class-weight Ar |
| `test_native_transient_buffer_lifetimes.py` | `test_voting_sum_keeps_native_fsum_input` | RNG-bound input; Python-allocator scratch lifetime |
| `test_native_transient_buffer_lifetimes.py` | `test_calibration_keeps_temporary_class_column_indices` | RNG-bound input; Python-allocator scratch lifetime |
| `test_native_transient_buffer_lifetimes.py` | `test_forest_and_adaboost_keep_encoded_labels_through_fit` | RNG-bound input; Python-allocator scratch lifetime |
| `test_native_transient_buffer_lifetimes.py` | `test_bagging_oob_keeps_sum_and_count_owners_until_native_r2_finishes` | RNG-bound input; Python-allocator owner lifetime; oracle built from public estimators_ obj |
| `test_native_decomp_buffer_owners.py` | `test_manifold_svd_keeps_resident_sort_scratch[lle,umap_pca]` | RNG-bound input (default_rng(51)); repeated public fit in one process (pool reuse); decomp |
| `test_ne2_g_decomp_entries.py` | `test_dictionary_transform_gram_kept_and_refit_seen` | RNG-bound input (_data: default_rng standard_normal); public-boundary caches (set_params,  |
| `test_ne2_g_decomp_entries.py` | `test_sparse_pca_transform_gram_kept_and_ridge_alpha_seen` | RNG-bound input (_data: default_rng standard_normal); public-boundary caches (set_params,  |
| `test_ne2_g_decomp_entries.py` | `test_pca_query_record_sees_replaced_state_and_refit` | RNG-bound input (_data: default_rng standard_normal); public-boundary caches (set_params,  |
| `test_ne2_g_decomp_entries.py` | `test_pca_inverse_after_transform_record` | RNG-bound input (_data: default_rng standard_normal); public-boundary caches (set_params,  |
| `test_ne2_g_decomp_entries.py` | `test_repeated_queries_after_once_per_table_check[3]` | RNG-bound input (_data: default_rng standard_normal); public-boundary caches (set_params,  |
| `test_ne2_g_decomp_entries.py` | `test_elliptic_envelope_score_and_predict` | RNG-bound input (_data: default_rng standard_normal); public-boundary caches (set_params,  |
| `test_pca_exact_native_owner.py` | `test_complete_exact_owner_matches_existing_bits_and_independent_subspace[18]` | RNG-bound fixture (default_rng(9021)); NumPy float64 SVD outside oracle; audit_native_call |
| `test_pca_exact_native_owner.py` | `test_randomized_complete_owner_preserves_seed_power_and_consumed_outputs[6]` | RNG-bound fixture; Python seed values (2**70+3, -1); audit_native_call |

## What would let the KEEP-NOPORT groups move (coordinator decision 3)

None moved: no group's assertion is a small native relation with a typed RUN entry to call. Per group:

- `test_minmax_scaler_native_lifecycle.py` (4) and `test_standard_scaler_native_fit.py` (1): 43 x 5 input, so a
  Mojo-side generator would do (the relations hold for any data with the constant and near-constant columns), but
  partial_fit's statistic merge and the unseen-column NaN mask run in `bindings/minmax_native_door.mojo` on
  PythonObject fields; they move once a typed `minmax_partial_fit_run` entry exists. The StandardScaler one picks its
  route by `MOJOLEARN_SCALER_DIRECT` at run time and stays until that switch is decided.
- `test_cpu_training_*` (11): public-API repeat/refusal smokes of a CPU-only install over whole estimators; they
  test the install, not a kernel, so they stay Python (the kernels are in the family checks).
- `test_cpu_training_par_classical.py` (1), `test_x_prep_twice.py` (1): worker processes and child-process hang
  timeouts: boundary, stay.
- `test_kde_fused_fast.py` (3): `fast`-only arms (paused programme; the checks build `identical`): move with the
  programme when it reopens.
- `test_public_verifier_gap_methods.py` JL bound (1): needs a typed `jl_min_dim` RUN entry.
- `test_svc_multiclass.py` (3): 240 x 5 and 120 x 4 generated inputs and public multiclass/SVR refits; a Mojo
  generator could replace the data, but OvO pairing and the precomputed/linear SVR arm have no typed entry in
  `svm/estimator.mojo` (binary `svc_fit_host` only), so a check would cover only the binary third and the Python
  test would stay: not moved.
- `test_native_prep_iterimp_door.py` (2): compared with the former Python route (an oracle helper); move when the
  oracle's expected outputs are stored as fixture words by a committed generator.
- `test_native_sgd_weight_lifetime.py`, `test_native_transient_buffer_lifetimes.py` (5),
  `test_native_decomp_buffer_owners.py` (1): Python-allocated owner lifetimes and pool reuse across public calls:
  boundary, stay.
- `test_ne2_g_decomp_entries.py` (6): public caches invalidated by `set_params`/refit: boundary, stay.
- `test_pca_exact_native_owner.py` (2): NumPy float64 SVD as the outside oracle and Python big-int seeds: stay.

## Native entries missing

- MinMaxScaler partial_fit merge: no typed RUN entry (`bindings/minmax_native_door.mojo` merges on PythonObjects).
- `johnson_lindenstrauss_min_dim`: no typed RUN entry; the bound is composed inside `bindings/jl_native_pack.mojo`
  `jl_min_dim`, so its test stays Python.

## Open points

- The frozen-digest "inherited fit" tests of `test_host_logistic_native.py`, `test_host_svc_native.py` and
  `test_host_svr_native.py` (digests of 32011bb32 outputs) are replaced by same-build relations: fitted state and one
  native call per method over every kernel, precomputed linear Gram equals the linear kernel bit for bit (SVC, SVR),
  nested score equals this build's predict accuracy (logistic).
- Most NATIVE rows of the inventory are leftovers of earlier migrations: their numerics already moved and what
  remains is boundary.
- The twelve `archive_*` checks (cholesky, codec, linear, minmax, parser, path, reader, repr, standard, text_cells,
  unicode_names, writer) were outside the runner; they are now registered (host column), their per-check pixi tasks
  are gone, and `native-checks.yml` / `tools/ops/native_checks_gpu_job.sh` run `tools/generated_sources.py prepare`
  before the one build, since eight of them import the gitignored generated archive Unicode table.
- `tools/release_public_runtime_profile.json` lists five deleted `test_native_prep_no_callbacks.py` nodes; it is the
  frozen record of a past Apple run (it checks executed blob hashes), so it was left unchanged.
