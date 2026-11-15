# Lane tests-delete-source-text, 2026-11-15

Scope: the 46 rows of `lanes.tsv` (`~/mojolearn-evidence/briefs-20261115/tests-to-mojo/`) whose lane is
`L1-source-text`, from `TESTS_TO_MOJO_2026-11-15.md`. Base `origin/preinteg/wave2-20261113` at `7dcd11184c`.
Nothing was compiled or run. Every function was read; the classifier's per-function class was a starting
point, not the decision.

Rule applied: a test is deleted when it asserts on source text, file layout, a manifest restated as a
literal, or a private name. A deleted assert that stood for a behaviour nothing else tests is replaced by a
test of that behaviour through the public call or the built module. Runtime tests in the same files stay.

Totals: 2,844 lines removed, 179 added (Python). 9 files deleted whole, 30 edited, 7 kept unchanged.
No Mojo check added: no deleted test held a kernel or numeric assert that a Mojo check would take (the
numerics of these files had already moved; their checks are named per row). No native entry is missing.
No product code changed. Nothing verify imports was touched.

## Behaviour tests added (replacing deleted source checks)

| file | test | replaces |
|---|---|---|
| `test_native_prep_labels_door.py` | `test_each_public_method_is_one_native_operation` (`audit_native_call(...).assert_native()` per public method of the three classes) | the `ast` shape check of `_expansion_prep.py` |
| `test_torch_compile_backend.py` | `test_torchdynamo_is_not_registered_by_packaging` now reads the installed `torch_dynamo_backends` entry points | a `pyproject.toml` text search |
| `test_weighted_score.py` | `test_gradient_boosting_adapters_score_takes_sample_weight` (`score(X, y, sample_weight)` equals the weighted metric of `predict`) | text search of `_gbdt_adapters.py` for the weight argument (forests: `test_forest_protocol.py`) |
| `test_tokenizer_manifest.py` | `test_installed_package_ships_no_vocabulary` (the installed distribution's file list) | `.gitignore` / tree-layout checks |
| `test_host_surface.py` | `test_built_host_bindings_load_and_export_the_manifest` (each built host binding loads through the read-back and has every manifest export; inference-only bindings export no `*_fit`) | 15 checks of binding sources, build shims, PyInit text, workflow and packaging text |

## Per file

| file | decision | lines removed | reason / where the behaviour is tested |
|---|---|---:|---|
| test_admitted_call_doors.py | kept 2, deleted 5 | 79 | door-module text; Ridge and GP runtime answers stay |
| test_autograd_save_resume.py | kept | 0 | classifier false positive (`read_bytes` of a checkpoint); every test is runtime |
| test_byte_checkpoint_native.py | kept | 0 | false positive; archive bytes, refusals and errno are runtime with `audit_native_call` |
| test_causal_lm_cli.py | kept | 0 | false positive; CLI routing of `verify-causal-lm` (verify tooling, not this lane's) |
| test_cholesky_cpu_inference.py | deleted 1 | 25 | binding-text check; CPU-route and solve tests stay |
| test_chunked_lm_head_door.py | kept 1, deleted 3 | 94 | registration/manifest text; refusals stay; numerics `training_chunked_lm_head_door.mojo` |
| test_cpu_training_arima.py | kept 1, deleted 5 | 111 | oracle spelling and manifest; sabotage-build probe stays; numerics `tsa_training_arima.mojo` |
| test_cpu_training_e3_gbdt.py | deleted file | 181 | all source text; runtime moved to `gbdt_public_fits.mojo` (2026-11-14) |
| test_cpu_training_e3_gbdt_rmse.py | kept 1, deleted 5 | 91 | oracle spelling; host RMSE fit check stays |
| test_cpu_training_gbdt_losses.py | deleted file | 156 | all source text; runtime in `gbdt_public_fits.mojo` |
| test_cpu_training_gp.py | kept 1, deleted 5 | 136 | oracle spelling; thread-count bytes check stays |
| test_cpu_training_mamba.py | kept 3, deleted 4 | 116 | registration text; generator `--check` and two runtime tests stay |
| test_cpu_training_samba.py | kept 1, deleted 4 | 125 | registration and generated-file text; seeded step repeat stays |
| test_cpu_training_transformer.py | deleted file | 126 | all source text; runtime in `transformer_host_block.mojo` |
| test_cpu_training_umap.py | kept 1, deleted 5 | 129 | kernel spelling; host fit repeat stays |
| test_expose_d_manifest.py | kept 1, deleted 6 | 137 | packaging lists, build scripts, docs text, private `_backend` tables; public exports test stays |
| test_fowlkes_mallows_score.py | deleted 1 | 14 | binding text; bitwise-vs-scikit-learn tests stay |
| test_gbdt_mode_serialization.py | kept | 0 | false positive (`read_bytes` of an archive that must not be overwritten) |
| test_generation_native.py | deleted 1 | 17 | `ast` one-call shape; `audit_native_call` tests in the file cover it |
| test_holtwinters_initialization.py | kept 2, deleted 3 | 68 | binding/kernel text and a private code table; save/load and surface tests stay; numerics `tsa_holtwinters_initialization.mojo` |
| test_host_density_constructors_native.py | deleted 1 | 14 | `ast` one-call shape; covered by the file's 18 audits |
| test_host_iforest_fit_native.py | deleted 1 | 8 | same, 31 audits |
| test_host_iforest_native.py | deleted 1 | 12 | same, 22 audits |
| test_host_inference_native.py | deleted 1 | 19 | same, 29 audits |
| test_host_kde_fit_native.py | deleted 1 | 12 | same, 26 audits |
| test_host_kernel_inference_native.py | deleted 1 | 16 | same, 32 audits |
| test_host_model_native_router.py | deleted 1 | 11 | same, 10 audits |
| test_host_projection_native.py | deleted 1 | 12 | same, 31 audits |
| test_host_surface.py | kept 23, deleted 15, added 1 | 276 | binding/shim/workflow/packaging text deleted; the 23 kept test the identity harness, gates, records, fixture floors and lane accounting against the manifest (verification tooling; they guard that verify can fail) |
| test_host_tree_lifecycle_native.py | deleted 1 | 13 | `ast` one-call shape; 18 audits |
| test_lda_fit_block_estep.py | deleted file | 62 | kernel launch-shape text; a timing property, judged by the measurement, not a unit test |
| test_native_operation_codes_table.py | deleted file | 36 | parses a Mojo table; a missing code fails the public call that uses it |
| test_native_parallel_drivers_wire.py | deleted file | 257 | `ast`/text contracts; numerics `test_parallel_public_drivers.py` |
| test_native_parallel_kernel_methods_wire.py | deleted file | 136 | same |
| test_native_path_admission.py | kept | 0 | false positive (checkpoint bytes equal across path types) |
| test_native_prep_labels_door.py | deleted 3, added 1 | 56 | `ast` shape and text; replaced by a runtime audit; oracle tests stay |
| test_parallel_public_drivers.py | deleted 1 | 11 | `docs-site/api_exclude.txt` content |
| test_portable_kernel_models.py | kept | 0 | integrity of `verify_reference` bundled models: verify fixtures, protected |
| test_samba_checkpoint_native.py | kept | 0 | false positive (failed save keeps the existing file) |
| test_svc_poly.py | deleted 4 | 50 | binding/oracle text and a manifest lane; refusal, repeat and save/load stay |
| test_tokenizer_manifest.py | kept 3, deleted 7, added 1 | 122 | manifest restatement, docs, shim, workflow and source text; see added test |
| test_torch_compile_backend.py | changed 1 | 3 | see added test |
| test_weighted_score.py | deleted 3, added 1 | 34 | see added test |
| test_x_decomp_native_api_cpu.py | deleted 1 | 9 | binding text; CPU-only route tests stay |
| tests/test_array_metadata_dtype_source.py | deleted file | 44 | Mojo source text; complex refusal and read-only inputs are boundary-tested in 25 and 54 files |
| tests/test_prep_support_provider_source.py | deleted file | 26 | `exec` of a method body; `test_prep_support_provider_boundary.py` tests `get_support` |

## Rerun of the classifier

`classify.py` + `final.py` over the lane branch (`~/mojolearn-evidence/briefs-20261115/tests-delete-source-text/rerun/`):
SOURCE-TEXT files 27 -> 6. The six are not source-text tests: `test_autograd_save_resume.py`,
`test_byte_checkpoint_native.py`, `test_causal_lm_cli.py` (the classifier counts `read_bytes` of a checkpoint
or output file as reading source), `test_portable_kernel_models.py` (verify fixtures), `test_host_surface.py`
(verification tooling against the manifest; its name matches the `manifest` rule), and
`test_tokenizer_manifest.py` (name rule; its remaining tests are public behaviour). The orchestrator may
reclassify them; this lane does not delete behaviour tests to satisfy a name pattern.

Stale rows left for their owners: `tools/ops/wheel_verify_smoke_plan/tests.tsv` still lists the deleted files
(a timing/memory lookup keyed by unit; a missing unit is ignored), and a few native-check and plan comments
name the deleted Python files. Both belong to verify or history and were not edited.
