# Python tests to Mojo: inventory and plan, 2026-11-15

AGENTS.md "Move native correctness tests to Mojo (Andrew, 2026-11-10)": kernel, numerical and native
state/lifetime tests belong in Mojo; Python keeps the real API boundary (buffer admission, exceptions, returned
objects, refusal of callbacks, framework interop), external oracles and tools tests. This is a static inventory of
`origin/preinteg/wave2-20261113` at `9685746f95`: nothing was compiled and no test was run. Branch
`lane/tests-to-mojo-inventory`.

Evidence and scripts: `~/mojolearn-evidence/briefs-20261115/tests-to-mojo/`. `classify.py` parses every file with
`ast` in a process pool sized to `os.cpu_count()` (10 workers here) and scores each test function; `final.py`
applies the rules below; `lanes.py` assigns lanes; `gen_doc.py` writes this file. Per-file rows with every
score: `lanes.tsv`; per-function scores: `functions.tsv`.

## Counts

706 files, 127,681 lines (python/mojolearn/tests 698 files, tests/ 8; 43 of them are helpers, conftest or `__init__`, classed with their users). 4,290 test functions.

| class | files | lines | meaning |
|---|---:|---:|---|
| BOUNDARY | 441 | 80,425 | only Python can test it: keep |
| NATIVE | 159 | 22,335 | kernel / numeric / native state or lifetime through Python: move to a Mojo check |
| SOURCE-TEXT | 27 | 4,920 | asserts on source text, layout, counts or private names: delete |
| ORACLE | 53 | 13,114 | compares against an external library or a Python reference: keep (testing tool) |
| TOOLING | 4 | 253 | tests tools/, bench or release scripts: keep |
| VERIFY | 22 | 6,634 | tests `python -m mojolearn verify` itself: protected, unchanged |

Inside the BOUNDARY files, 7,246 more lines are native-numeric test functions (files marked `split` below); inside kept files, 1,094 lines are source-text functions. 122 files are mixed (no class holds 60 percent of their test lines) and are marked `low` in `lanes.tsv`: the owning lane re-reads them function by function.

Already covered: 37 NATIVE files name the Mojo check that took their numerics (`checks/native_tests/...` in the file). They are deletion candidates once an assert-by-assert diff shows the Mojo check holds every value; a file name match is not proof, so the table says "after diff".

Smoke breakage: 176 files failed in at least one of the two smokes of 2026-11-15 (`SMOKE_TRIAGE_2026-11-15_tests.tsv`, `SMOKE_TRIAGE_2_2026-11-15.md`) as stale or product. By class: BOUNDARY 116, NATIVE 49, SOURCE-TEXT 10, ORACLE 1. Stale in either smoke: BOUNDARY 73, NATIVE 35, SOURCE-TEXT 10. Of the 69 smoke-2 rows, 26 name a Python oracle helper as the cause (`_mamba_block_reference` 16, `_typed_input` 5, `_legacy_neighbors_*` 3, others 2): the helpers import product internals, and an internal rename breaks every user at once. Most other stale BOUNDARY rows patched or read a private facade (`_HostBinding`, `_arena_io`, entry-table keys); the fix for those is to test through the public call, which each lane does for the files it owns.

## Rules used

Per test function (`classify.py` scores, `final.py:fclass`): SOURCE-TEXT when it reads `.mojo`/`.py` text, `inspect.getsource`, `ast.parse` or repo paths at least as often as it checks numbers; ORACLE when it calls a name imported from scikit-learn, SciPy, torch, statsforecast, arch (and similar) or `numpy.linalg`/`fft` and checks numbers; BOUNDARY when `pytest.raises` dominates or it inspects buffers, hooks, refcounts, pickling, params, dtypes, warnings or child processes; NATIVE when numeric asserts (allclose, array_equal, digests, tolerances) dominate. Per file: the class with the most test lines, then these overrides: VERIFY for `test_verify*`/`test_verification*` and files driving verify's own modules; TOOLING for `tests/tools/` and files importing `tools/`/`bench/` scripts; torch/pandas/ONNX/state_dict files are BOUNDARY (framework interop); files importing a real library oracle are ORACLE; `*_repeat`/`*_twice` are NATIVE (native state across calls); a file named for admission/refusal/protocol/import stays BOUNDARY while its boundary share is 20 percent or more.

Two judgments the orchestrator may overrule:

- A hand-written Python reference (`_legacy_*`, `_mamba_block_reference`, NumPy arithmetic in the test) is not an external library. Tests that compare a native result to one are NATIVE: the expected values become fixture bytes in the Mojo check. The reference helpers themselves are ORACLE while a Python user remains.
- 220 test files import `mojolearn._verify_native_runtime` (`audit_native_call`) to prove an operation is one native call with no Python frames. That is a boundary property (no calls back into Python), so they are classed by content, not VERIFY; they do couple product tests to a verify module, which the verify lanes must keep working.

## Proposed lanes (at most 8, disjoint file sets)

Ordered by lines removed plus smoke breaks. Each lane writes Mojo checks (registered in `checks/native_tests/registry.mojo`, run by `native-checks.yml`) and deletes the Python functions it moved; it does not compile. The file list of each lane is the `lane` column of `lanes.tsv`. Every lane brief carries the bits reminder: moved checks compare at the tolerance the Python test used, or by digest where it compared bytes; a digest is the current version's, never an older one's.

| # | lane | files | lines in files | lines removed | smoke breaks (files) | scope | decided by |
|---|---|---:|---:|---:|---:|---|---|
| 1 | `tests-to-mojo-classical` | 58 | 14,154 | 8,148 | 26 (21) | linear, logistic, SVM, GP, KDE/density, Cholesky, scalers, archives, PCA/decomposition, metrics, prep encoders/labels; plus the CPU-training lane files of those families. Joins the existing linear_*, logistic_*, svc_*, gp_*, kde_*, archive_*, pca_*, prep_* checks; new checks only where none exists. | `native-checks.yml` green on the new checks and the next verify smoke listing none of the moved files |
| 2 | `tests-delete-source-text` | 46 | 10,106 | 5,089 | 25 (21) | delete the SOURCE-TEXT files and the source-text test functions inside kept files (manifest, docstring, 'getenv' word, file-layout and private-name asserts). Where a deleted assert stood for a behaviour, name the boundary or Mojo test that covers it or add one public call test. | no SOURCE-TEXT row left in a rerun of `final.py` |
| 3 | `tests-to-mojo-neural` | 35 | 5,777 | 4,569 | 14 (10) | autograd, optimizers, training primitives, ops.Session/Program, Sequential, weight import, generator, low-bit linalg. Joins training_*, ops_*, optimizer_*, core_ops_*, core_program_*, lowbit_weights, int8/bf16 checks. | `native-checks.yml` green on the new checks and the next verify smoke listing none of the moved files |
| 4 | `tests-to-mojo-parallel` | 32 | 5,052 | 3,995 | 9 (7) | parallel drivers, cross-vendor coordinator/worker, distributed and partitioned programs, model selection/CV, device cache. Joins training_distributed_program, core_partitioned_program, resample_*, msel_helpers_entry, parallel_archive_entry. | `native-checks.yml` green on the new checks and the next verify smoke listing none of the moved files |
| 5 | `tests-to-mojo-spatial` | 32 | 4,151 | 3,896 | 8 (8) | neighbors/ANN/IVF/embedding, clustering, spectral, mixture, time series (ARIMA, Holt-Winters, Prophet, ETS/GARCH/Theta), CNN. Joins neighbors_*, ann_*, kmeans_*, spectral_*, sequence_*, tsa_*, xcnn_*/cnn_* checks. | `native-checks.yml` green on the new checks and the next verify smoke listing none of the moved files |
| 6 | `tests-to-mojo-models` | 17 | 3,404 | 2,677 | 15 (12) | Samba, Mamba, Transformer, byte-LM, causal LM, decode/generation, tokenizer. Joins samba_entry_table, mamba*_*, transformer_*, byte_lm_*, causal_lm_*. | `native-checks.yml` green on the new checks and the next verify smoke listing none of the moved files |
| 7 | `tests-oracle-public-only` | 25 | 10,120 | 0 | 26 rows | the Python oracle helpers (`_legacy_*`, `_mamba_block_reference`, `_typed_input`, `_prep_prog_oracle`, ...) import only the public API, NumPy and the standard library, never a product-internal module or name. Removes 0 test lines but caused 26 of the 69 smoke-2 failures (one deleted shim broke 16 files). Each oracle whose users all move to Mojo is deleted with them. | a scan of the oracle helpers showing no `mojolearn._*` import and the next smoke green on their users |
| 8 | `tests-to-mojo-trees` | 15 | 1,936 | 1,652 | 5 (4) | forests, GBDT, DART, isolation forest, feature-freq. Joins trees_entry_*, gbdt_*, host_entry_forest, iforest_forest_export, oob_exact_sum. | `native-checks.yml` green on the new checks and the next verify smoke listing none of the moved files |

Not a lane here: the verify modules (below) are owned by the eight lanes of `VERIFY_COVERAGE_AUDIT_2026-11-15.md` section 5; this plan touches none of them.

Removed by this lane: nothing. It is an inventory with no code change; the lanes above name what they remove.

## How to move a test

Written by `tests-to-mojo-classical`, the first move lane (its result: `TESTS_TO_MOJO_tests-to-mojo-classical.md`;
its four checks are the examples: `core_all_finite`, `kernel_methods_poly_degree`, `sequence_theta_variants`,
`training_residual_dropout_mask`).

1. Read every function of the file; the inventory's class is a heuristic. Most NATIVE rows are leftovers of earlier
   migrations (the docstring says "numerics run in checks/..."): what remains is usually boundary. Open the named
   check and compare assert by assert (same inputs, same tolerance) before calling anything COVERED; a check that is
   not imported in `registry.mojo` never runs in CI (the `archive_*` checks are not) and is no evidence.
2. Delete outright what asserts on source text, manifests (`host_surface`, `NOT_IMPLEMENTED.tsv`), `ast` of a module,
   or file layout: those are the retired source guards. Keep boundary functions (admission, `pytest.raises`,
   returned types and shapes, hook refusal, `audit_native_call`, pickling and archives, outside-library oracles).
3. Port only what a check can state with the same inputs: literals in the test, or arithmetic regenerated in Mojo.
   Inputs drawn from `np.random.default_rng` / `random.Random` stay Python (KEEP-NOPORT) unless the bytes are stored
   as a fixture by a committed generator. When a public test asserted both a numeric relation and a public shape or
   admission, move the relation and keep a small Python test for the shape.
4. Call the entry the binding's PACK calls, not the binding: grep the door (`bindings/*_pack.mojo`,
   `*_native_door.mojo`) for the run function and copy its argument order (the residual-dropout PACK passes
   `yp, yp, xp, rp` forward and `xp, rp, yp, yp` backward; the theta door passes seven words and three settings).
   The cpu column and the gpu column may call different entries (`_buffer.all_finite` is `host_helpers_impl` on a
   CPU install and `LabelsDeviceKernels.finite` on the GPU base binding).
5. Shape: one file `checks/native_tests/<family>_<topic>.mojo` with `_run_cpu(mut report)` (independent cases via
   `report.cases(n)` and `report.parallel(task, n)`; the task captures `{imm cases}` and Int addresses, rebuilt
   with `core.hostptr.f32_ptr` inside the worker, since a `List.unsafe_ptr()` origin does not match a
   `MutUntrackedOrigin` parameter), `_run_gpu(mut report)` (cases in order on the shared device context, tagged
   `# driver-loop:`), `run_cases` with `comptime if CPU_COLUMN_BUILD`, and `register() -> Check("<name>", ...)`.
   Register with one import and one append in `registry.mojo`; `python3 tools/ops/native_checks_report.py plan
   --column cpu` (source only, no compile) must then list the name. No `main()`, no pixi task.
6. Traps: Python's `Array.from_list(..., "<f4")` rounds literals, so state float32 words exactly (3.4e38 is
   `0x7F7FC99E`, 1e-45 is `0x00000001`); NumPy's float32 `sin` may differ from Mojo's by an ulp, so port relations
   (batch equals single) rather than values that depend on it; write `a * b + c` comparisons with
   `checks.numerics.identical_mul` so LLVM cannot fuse them; entries that refuse outside `identical` assume the
   check columns build in identical mode (`comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL`); pass every
   argument a sibling check passes explicitly when unsure of a default (`kernel_ridge_fit_host` `sw=`).
7. A deleted function takes its now-unused imports and helpers with it; grep `python/` and `tools/` for importers of
   the file's helpers (`test_host_archive_native.py` `_archive`/`_state` have 20 importers) and for its node ids.
   Frozen evidence records (`tools/release_public_runtime_profile.json`) are history: leave them.

## The verify modules (42 files, 21,863 lines)

Protected (AGENTS.md "The verify command is protected"); nothing here changes them. Role: `controller` must stay Python (it drives the installed wheel through the public API, the path users take, and judges the result); `fixture` is a public-API endpoint fixture; `data` is generated or reference data; `internal` reaches into product internals at the sites counted by the verify coverage audit (`coupling_product.tsv`, 75 sites in wheel modules; the other 85 are in `tools/identity_break.py` and `tools/identity_lanes/`). The replacement for each coupling kind is in the audit's section 4 table (one `mojolearn.runtime_report()` for vendor/route/binding identity, public `Session`/`ImportedProgram` calls, fixtures moved under verify ownership).

| module | lines | internal sites | role | what it is |
|---|---:|---:|---|---|
| `_verification_api_reference.py` | 1,625 | 0 | data | Generated endpoint reference; DATA records the actual validated vendors. |
| `_verification_api_targets.py` | 276 | 0 | data | Endpoint obligations from the 2026-10-27 gap audit; not execution evidence. |
| `_verification_catalog.py` | 1,386 | 0 | data | The algorithm inventory, mapped explicitly to verification lanes. |
| `_verification_coverage.py` | 262 | 0 | controller | Installed-package verification scope; inspection never executes a lane. |
| `_verification_ctr_models.py` | 39 | 0 | data | Checked-in GPU-trained CTR fixtures used only for CPU inference verification. |
| `_verification_endpoint_data.py` | 1,152 | 0 | data | Generated by tools/verification_endpoints.py --write; source mapping only. |
| `_verification_evidence_data.py` | 3,375 | 0 | data | Generated by tools/verification_evidence.py --write. Historical evidence only. |
| `_verification_light_reference.py` | 3 | 0 | data | Generated by tools/light_verifier_reference.py; eight fixtures only. |
| `_verification_profiles.py` | 50 | 0 | controller | Routine verification includes classical fits; neural training is opt-in. |
| `_verification_sources.py` | 34 | 0 | controller | Source identity for verifier inventories, in source trees and installed wheels. |
| `_verify.py` | 1,686 | 7 | controller + internal | `python -m mojolearn verify` -- turn the identity claim into a check. |
| `_verify_all.py` | 4,495 | 15 | controller + internal | `python -m mojolearn verify --all`: check mojolearn's identity claims on |
| `_verify_api.py` | 78 | 0 | controller | Bounded public endpoint verification. Test control, never product execution. |
| `_verify_api_availability.py` | 32 | 0 | fixture | Explicit availability contracts for over-broad installed-method census rows. |
| `_verify_api_estimators.py` | 202 | 1 | fixture + internal | Eight-row endpoint fixtures; no timings, accuracy targets or datasets. |
| `_verify_api_faults.py` | 82 | 0 | fixture | Test-only state/configuration faults through real public native operations. |
| `_verify_api_metadata.py` | 200 | 0 | fixture | Tiny public constructor/parameter/attribute contracts, no fitting. |
| `_verify_api_misc.py` | 110 | 1 | fixture + internal | Small forecasting, tokenizer and serialized-estimator endpoint fixtures. |
| `_verify_api_models.py` | 78 | 4 | fixture + internal | One-layer, two-token loaded-model and continuation fixtures; no downloads. |
| `_verify_api_neural.py` | 183 | 1 | fixture + internal | Bounded neural API fixtures: two examples/tokens, at most one update. |
| `_verify_api_optimizers.py` | 58 | 0 | fixture | One-update optimizer state, ownership, transfer and zeroing contracts. |
| `_verify_api_programs.py` | 58 | 2 | fixture + internal | Imported-program inherited doors on a single scalar captured graph. |
| `_verify_api_runner.py` | 215 | 1 | controller + internal | Shared endpoint registry and exact comparison for the ordinary verifier. |
| `_verify_api_training.py` | 156 | 0 | fixture | One-step portable training/state fixtures, with one logical shard. |
| `_verify_api_utilities.py` | 131 | 0 | fixture | Small array, scalar, random-stream and configuration endpoint checks. |
| `_verify_causal_lm.py` | 221 | 3 | fixture + internal | Whole loaded-model capture. Numerical captures are not reference admission. |
| `_verify_distributed.py` | 382 | 8 | fixture + internal | Bounded installed/source GPU capture for forecasting, GPC and IVF. |
| `_verify_distributed_program.py` | 155 | 2 | fixture + internal | Installed-package verification of native captured-program distribution. |
| `_verify_generation.py` | 190 | 5 | fixture + internal | Installed-package checks of native causal generation and its output bytes. |
| `_verify_graph.py` | 1,841 | 7 | fixture + internal | Test-only installed-package qualification for captured graph execution. |
| `_verify_light.py` | 288 | 1 | fixture + internal | Tiny bitwise fixtures. No performance clocks or model-quality campaign. |
| `_verify_light_command.py` | 91 | 0 | controller | Command controller for tiny identity checks (testing only). |
| `_verify_light_faults.py` | 59 | 0 | fixture | Test-only public-operation faults, never imported by product execution. |
| `_verify_model.py` | 328 | 7 | controller + internal | ``python -m mojolearn verify-model``: a model's verification report. |
| `_verify_native_runtime.py` | 248 | 1 | controller + internal | Verification-only profiling across a whole public operation. |
| `_verify_parallel_cv.py` | 285 | 7 | fixture + internal | One/two-GPU CV numerical and placement captures from source or a wheel. |
| `_verify_partitioned_program.py` | 180 | 2 | fixture + internal | Installed test controller for physical model/tensor/pipeline execution. |
| `_verify_public.py` | 21 | 0 | controller | Public endpoint checks and fault controls in the ordinary verifier. |
| `_verify_reference.py` | 1,035 | 0 | controller | The reference hash table `python -m mojolearn verify --all` compares against. |
| `_verify_resources.py` | 60 | 0 | controller | Conservative CLI thread budgets, applied before native imports in a child. |
| `_verify_small.py` | 108 | 0 | controller | Pinned small training pilot. Captures are not admitted reference evidence. |
| `_verify_worker.py` | 405 | 3 | controller + internal | Bounded workers for the installed identity verifier. |

Most internal sites are in the controllers (`_verify_all.py` 15, `_verify.py` 7, `_verify_model.py` 7) and the capture modules (`_verify_distributed.py` 8, `_verify_parallel_cv.py` 7, `_verify_graph.py` 7, `_verify_generation.py` 5). What they read is "what am I running" (vendor, mode, binding path, route) and private session handles: a public runtime report and the public `Session`/`Program` calls replace both, which is the audit's `verify-runtime-report` lane. The `_verify_api_*` fixtures already go through the public API (0 to 4 sites each). The fault arm `_native_parallel_shard('faults_take')` in `_verify_distributed.py` stays: it is the fault injector.

## Per-file table

`target`: for NATIVE, the existing check the file names (`mojo_ref`), else the nearest existing check by name, else a new check file; `split` on BOUNDARY rows gives the native lines to move. `smoke`: s1/s2 = failed in the first/second smoke of 2026-11-15 as stale or product. `lane` as above (blank = no work).

| file | class | target | lines | lane | smoke |
|---|---|---|---:|---|---|
| `__init__.py` | BOUNDARY | keep | 22 |  |  |
| `_allocator_diagnostics.py` | BOUNDARY | keep | 27 |  |  |
| `_bpe_trainer.py` | BOUNDARY | keep | 538 |  |  |
| `_build_arms.py` | BOUNDARY | keep | 62 |  |  |
| `_cast_in.py` | BOUNDARY | keep | 38 | tests-oracle-public-only |  |
| `_connected_exports.py` | BOUNDARY | keep | 22 |  |  |
| `_connection_refusal.py` | BOUNDARY | keep | 25 |  |  |
| `_expose_d_harness.py` | BOUNDARY | keep | 80 |  |  |
| `_former_prep_labels.py` | BOUNDARY | keep | 789 | tests-oracle-public-only |  |
| `_fresh_process.py` | BOUNDARY | keep | 204 |  |  |
| `_onnx_fixture.py` | BOUNDARY | keep | 352 |  |  |
| `_package_evidence.py` | BOUNDARY | keep | 86 |  |  |
| `_parallel_pool.py` | BOUNDARY | keep | 54 |  |  |
| `_samba_checkpoint.py` | BOUNDARY | keep | 187 |  |  |
| `_suite_env.py` | BOUNDARY | keep | 84 |  |  |
| `_weight_import.py` | BOUNDARY | keep | 147 |  |  |
| `conftest.py` | BOUNDARY | keep | 99 |  |  |
| `test_affinity_preference_native_boundary.py` | BOUNDARY | keep | 170 |  | s1:product |
| `test_agglo_native_boundary.py` | BOUNDARY | keep | 214 |  | s1:stale s2:stale |
| `test_archive_unicode_native.py` | BOUNDARY | keep | 164 |  | s1:product |
| `test_arena_ranges.py` | BOUNDARY | keep | 144 |  |  |
| `test_arima_surface.py` | BOUNDARY | keep | 1027 |  |  |
| `test_autograd_model.py` | BOUNDARY | keep; split: 73 l native -> training_autograd_model | 275 | tests-to-mojo-neural |  |
| `test_backend_owned_types.py` | BOUNDARY | keep | 43 |  |  |
| `test_backend_tier_layout.py` | BOUNDARY | keep | 20 |  |  |
| `test_binding_mode_guard.py` | BOUNDARY | keep | 85 |  |  |
| `test_boost_estimator_refusal.py` | BOUNDARY | keep | 42 |  |  |
| `test_byte_lm_config_native_provider.py` | BOUNDARY | keep | 43 |  |  |
| `test_byte_lm_host.py` | BOUNDARY | keep | 243 |  |  |
| `test_byte_lm_host_context_contract.py` | BOUNDARY | keep | 69 |  |  |
| `test_byte_lm_host_trainer.py` | BOUNDARY | keep | 278 |  |  |
| `test_byte_lm_logits_pooled_output_admission.py` | BOUNDARY | keep | 74 |  |  |
| `test_byte_lm_native_state_restore.py` | BOUNDARY | keep | 187 |  | s1:product |
| `test_byte_lm_pooled_logits_native.py` | BOUNDARY | keep | 66 |  |  |
| `test_byte_lm_runtime_shape.py` | BOUNDARY | keep | 181 |  | s1:stale |
| `test_byte_lm_session.py` | BOUNDARY | keep; split: 70 l native -> byte_lm_trainer_session | 247 | tests-to-mojo-models |  |
| `test_byte_lm_step_routes.py` | BOUNDARY | keep | 259 |  |  |
| `test_byte_lm_train_steps.py` | BOUNDARY | keep | 64 |  |  |
| `test_byte_lm_trainer_logits.py` | BOUNDARY | keep | 136 |  |  |
| `test_cache_get_native.py` | BOUNDARY | keep | 580 |  | s1:product |
| `test_cache_key_native.py` | BOUNDARY | keep | 69 |  |  |
| `test_cache_lifecycle_native.py` | BOUNDARY | keep | 183 |  |  |
| `test_cache_stats_native.py` | BOUNDARY | keep; split: 80 l native -> new check | 320 | tests-to-mojo-parallel |  |
| `test_cache_supports_native.py` | BOUNDARY | keep | 216 |  | s2:stale |
| `test_cache_upload_native.py` | BOUNDARY | keep; split: 96 l native -> new check | 417 | tests-to-mojo-parallel | s1:product |
| `test_causal_binding_connection.py` | BOUNDARY | keep | 26 |  |  |
| `test_causal_model_native_loading.py` | BOUNDARY | keep | 169 |  | s1:stale s2:stale |
| `test_cholesky_cpu_inference.py` | BOUNDARY | keep; drop 12 l source-text | 178 | tests-delete-source-text | s1:product |
| `test_cholesky_surface.py` | BOUNDARY | keep | 113 |  |  |
| `test_class_weight_native_plan_boundary.py` | BOUNDARY | keep | 121 |  | s1:product |
| `test_class_weight_shared_families.py` | BOUNDARY | keep | 112 |  | s1:product |
| `test_classical_archive_native_owner.py` | BOUNDARY | keep; split: 77 l native -> archive_path | 332 | tests-to-mojo-classical |  |
| `test_classical_archive_reader_native.py` | BOUNDARY | keep; split: 95 l native -> archive_reader | 837 | tests-to-mojo-classical |  |
| `test_classical_archive_writer_native.py` | BOUNDARY | keep | 250 |  |  |
| `test_classical_exceptions_native_owner.py` | BOUNDARY | keep | 144 |  |  |
| `test_classical_family_frontdoor.py` | BOUNDARY | keep | 124 |  |  |
| `test_classical_metadata_native_owner.py` | BOUNDARY | keep | 140 |  |  |
| `test_classical_native_owner.py` | BOUNDARY | keep | 129 |  |  |
| `test_classical_numpy_free.py` | BOUNDARY | keep | 406 |  | s2:stale |
| `test_classical_ols_native_owner.py` | BOUNDARY | keep | 181 |  |  |
| `test_classical_pandas_native_owner.py` | BOUNDARY | keep | 242 |  |  |
| `test_classical_pip_smoke_sep22.py` | BOUNDARY | keep | 221 |  | s1:product s2:product |
| `test_classical_public_output_native_owner.py` | BOUNDARY | keep | 68 |  |  |
| `test_classical_tags_native_owner.py` | BOUNDARY | keep | 40 |  |  |
| `test_classical_warning_native_owner.py` | BOUNDARY | keep | 212 |  | s1:stale |
| `test_classification_metrics.py` | BOUNDARY | keep | 224 |  |  |
| `test_cluster_array_buffer.py` | BOUNDARY | keep | 60 |  |  |
| `test_cluster_input_refusals.py` | BOUNDARY | keep | 102 |  |  |
| `test_cluster_native_boundary.py` | BOUNDARY | keep | 126 |  |  |
| `test_cluster_resident_native.py` | BOUNDARY | keep | 132 |  |  |
| `test_cpu_inference_boundary.py` | BOUNDARY | keep | 37 |  |  |
| `test_cpu_training_d.py` | BOUNDARY | keep; drop 21 l source-text | 314 | tests-to-mojo-classical |  |
| `test_cpu_training_gbdt_ordered.py` | BOUNDARY | keep; drop 24 l source-text | 246 | tests-to-mojo-trees | s1:stale |
| `test_cross_vendor.py` | BOUNDARY | keep | 233 |  |  |
| `test_cross_vendor_admission_native.py` | BOUNDARY | keep | 133 |  |  |
| `test_dbscan_native_boundary.py` | BOUNDARY | keep | 210 |  | s2:stale |
| `test_device_array_refusal.py` | BOUNDARY | keep | 157 |  |  |
| `test_device_rows_materialize_native.py` | BOUNDARY | keep | 328 |  |  |
| `test_device_rows_metadata_native.py` | BOUNDARY | keep | 253 |  |  |
| `test_distributed_ivf.py` | BOUNDARY | keep | 105 |  |  |
| `test_embedding_surface.py` | BOUNDARY | keep | 94 |  |  |
| `test_estimator_score_coverage_p1.py` | BOUNDARY | keep | 126 |  |  |
| `test_estimator_score_p2.py` | BOUNDARY | keep | 144 |  | s1:product |
| `test_extratrees_gpu_only.py` | BOUNDARY | keep | 46 |  |  |
| `test_flat_bytes_without_buffer_protocol.py` | BOUNDARY | keep | 39 |  |  |
| `test_forest_export_protocol.py` | BOUNDARY | keep | 47 |  |  |
| `test_forest_fit_mode.py` | BOUNDARY | keep | 104 |  |  |
| `test_forest_protocol.py` | BOUNDARY | keep | 177 |  |  |
| `test_fowlkes_mallows_score.py` | BOUNDARY | keep; drop 8 l source-text | 98 | tests-delete-source-text |  |
| `test_framework_metadata_native_provider.py` | BOUNDARY | keep | 31 |  |  |
| `test_framework_version_metadata.py` | BOUNDARY | keep | 69 |  |  |
| `test_full_reset_numpy_free_import.py` | BOUNDARY | keep | 131 |  |  |
| `test_gbdt_catboost_defaults.py` | BOUNDARY | keep | 295 |  |  |
| `test_gbdt_feature_fraction.py` | BOUNDARY | keep | 83 |  |  |
| `test_gbdt_group_id.py` | BOUNDARY | keep | 118 |  |  |
| `test_gbdt_host_ctr.py` | BOUNDARY | keep | 151 |  |  |
| `test_gbdt_input_safety.py` | BOUNDARY | keep | 76 |  |  |
| `test_gbdt_mode_serialization.py` | BOUNDARY | keep; drop 8 l source-text | 108 | tests-delete-source-text |  |
| `test_gbdt_multirmse.py` | BOUNDARY | keep | 114 |  |  |
| `test_gbdt_query_rmse.py` | BOUNDARY | keep | 84 |  |  |
| `test_gbdt_release_parity.py` | BOUNDARY | keep | 23 |  |  |
| `test_gbdt_search_option_guards.py` | BOUNDARY | keep | 74 |  |  |
| `test_generation_engine_admission.py` | BOUNDARY | keep | 54 |  | s2:stale |
| `test_generation_metadata_snapshot.py` | BOUNDARY | keep | 134 |  | s2:stale |
| `test_generation_native.py` | BOUNDARY | keep; drop 15 l source-text | 281 | tests-delete-source-text | s2:stale |
| `test_generation_public_claim.py` | BOUNDARY | keep | 33 |  | s2:stale |
| `test_generation_route_controls.py` | BOUNDARY | keep | 187 |  |  |
| `test_generator_initializer_native.py` | BOUNDARY | keep | 114 |  | s2:stale |
| `test_gmm_sample.py` | BOUNDARY | keep | 74 |  |  |
| `test_gp_mixed_label_identity.py` | BOUNDARY | keep | 61 |  | s1:product s2:product |
| `test_gp_optimizer.py` | BOUNDARY | keep | 199 |  |  |
| `test_gp_resident_freeze_boundary.py` | BOUNDARY | keep | 56 |  |  |
| `test_gp_sample_y.py` | BOUNDARY | keep | 129 |  |  |
| `test_gp_score_native_boundary.py` | BOUNDARY | keep | 52 |  |  |
| `test_gp_surface.py` | BOUNDARY | keep | 289 |  |  |
| `test_gpu_witness.py` | BOUNDARY | keep | 109 |  |  |
| `test_graph_metadata_native.py` | BOUNDARY | keep | 69 |  |  |
| `test_hdbscan_host_lifetime.py` | BOUNDARY | keep | 49 |  |  |
| `test_hdbscan_native_boundary.py` | BOUNDARY | keep | 115 |  |  |
| `test_hdbscan_surface.py` | BOUNDARY | keep | 160 |  |  |
| `test_hierarchy_attribute_connection.py` | BOUNDARY | keep | 73 |  |  |
| `test_hierarchy_sparse_boundary.py` | BOUNDARY | keep | 103 |  | s1:stale |
| `test_higher_rank_native.py` | BOUNDARY | keep | 46 |  |  |
| `test_host_archive_native.py` | BOUNDARY | keep; split: 67 l native -> archive_path; drop 9 l source-text | 553 | tests-to-mojo-classical | s1:stale |
| `test_host_archive_predict_native.py` | BOUNDARY | keep | 136 |  |  |
| `test_host_attribute_lifetime.py` | BOUNDARY | keep | 191 |  | s1:product s2:stale |
| `test_host_density_constructors_native.py` | BOUNDARY | keep; drop 9 l source-text | 302 | tests-delete-source-text |  |
| `test_host_density_save_native.py` | BOUNDARY | keep; split: 73 l native -> training_autograd_save_resume | 375 | tests-to-mojo-classical | s1:product |
| `test_host_embedding_finalizer.py` | BOUNDARY | keep | 59 |  |  |
| `test_host_iforest_fit_native.py` | BOUNDARY | keep; drop 7 l source-text | 462 | tests-delete-source-text | s1:product s2:stale |
| `test_host_iforest_native.py` | BOUNDARY | keep; drop 8 l source-text | 394 | tests-delete-source-text |  |
| `test_host_inference_native.py` | BOUNDARY | keep; drop 15 l source-text | 371 | tests-delete-source-text | s1:stale |
| `test_host_kde_fit_native.py` | BOUNDARY | keep; drop 8 l source-text | 605 | tests-delete-source-text | s2:stale |
| `test_host_kde_native.py` | BOUNDARY | keep; split: 106 l native -> kde_state; drop 8 l source-text | 884 | tests-to-mojo-classical | s1:product |
| `test_host_kernel_inference_native.py` | BOUNDARY | keep; drop 12 l source-text | 507 | tests-delete-source-text | s2:stale |
| `test_host_logistic_native.py` | BOUNDARY | keep; split: 76 l native -> logistic_config; drop 8 l source-text | 463 | tests-to-mojo-classical | s1:stale s2:stale |
| `test_host_model_kmeans.py` | BOUNDARY | keep; split: 89 l native -> kmeans_native | 344 | tests-to-mojo-spatial |  |
| `test_host_model_native_router.py` | BOUNDARY | keep; drop 9 l source-text | 210 | tests-delete-source-text | s1:stale |
| `test_host_neighbor_finalizer_layout.py` | BOUNDARY | keep | 105 |  | s1:product |
| `test_host_path_resolution.py` | BOUNDARY | keep | 216 |  |  |
| `test_host_projection_native.py` | BOUNDARY | keep; drop 10 l source-text | 408 | tests-delete-source-text | s2:stale |
| `test_host_scaler_training_native.py` | BOUNDARY | keep | 95 |  |  |
| `test_host_svc_native.py` | BOUNDARY | keep; split: 66 l native -> svc_probability; drop 8 l source-text | 579 | tests-to-mojo-classical |  |
| `test_host_svr_native.py` | BOUNDARY | keep; split: 85 l native -> svr_native; drop 7 l source-text | 421 | tests-to-mojo-classical |  |
| `test_host_tree_lifecycle_native.py` | BOUNDARY | keep; drop 9 l source-text | 199 | tests-delete-source-text |  |
| `test_hotpath_native.py` | BOUNDARY | keep | 701 |  | s1:stale |
| `test_iforest_depth_native.py` | BOUNDARY | keep | 217 |  | s1:product |
| `test_iforest_output_boundary.py` | BOUNDARY | keep | 31 |  |  |
| `test_iforest_save_forest.py` | BOUNDARY | keep | 135 |  | s1:stale s2:product |
| `test_import_lower.py` | BOUNDARY | keep | 252 |  |  |
| `test_import_lower_native.py` | BOUNDARY | keep | 289 |  | s1:product |
| `test_import_onnx.py` | BOUNDARY | keep | 204 |  |  |
| `test_import_onnx_native.py` | BOUNDARY | keep; drop 28 l source-text | 281 | tests-to-mojo-neural |  |
| `test_import_torch_module.py` | BOUNDARY | keep | 31 |  |  |
| `test_imported_run_native.py` | BOUNDARY | keep | 106 |  | s1:stale |
| `test_ipca_native_boundary.py` | BOUNDARY | keep | 210 |  |  |
| `test_ivf_surface.py` | BOUNDARY | keep | 88 |  |  |
| `test_ivfpq_cpu_gil.py` | BOUNDARY | keep | 34 |  |  |
| `test_jl_native_boundary.py` | BOUNDARY | keep | 93 |  | s1:product |
| `test_kde_native_boundary.py` | BOUNDARY | keep | 196 |  | s2:product |
| `test_kernel_methods_surface.py` | BOUNDARY | keep | 148 |  |  |
| `test_kmeans_metric_surface.py` | BOUNDARY | keep | 108 |  |  |
| `test_kmeans_predict.py` | BOUNDARY | keep | 81 |  |  |
| `test_kmeans_transform.py` | BOUNDARY | keep | 83 |  |  |
| `test_kmeans_variants_native_boundary.py` | BOUNDARY | keep | 104 |  |  |
| `test_knn_metrics.py` | BOUNDARY | keep | 76 |  | s1:product |
| `test_kpca_native_boundary.py` | BOUNDARY | keep | 277 |  | s1:stale |
| `test_label_binarizer_inverse_window.py` | BOUNDARY | keep | 24 |  |  |
| `test_label_score_key_boundary.py` | BOUNDARY | keep | 85 |  |  |
| `test_labels_legacy_admission.py` | BOUNDARY | keep | 82 |  |  |
| `test_labels_native.py` | BOUNDARY | keep | 162 |  | s1:product |
| `test_labels_native_entry.py` | BOUNDARY | keep | 109 |  |  |
| `test_labels_provider_version.py` | BOUNDARY | keep | 70 |  |  |
| `test_late_numpy_native_metadata.py` | BOUNDARY | keep | 164 |  | s2:stale |
| `test_lazy_family_import.py` | BOUNDARY | keep | 292 |  | s2:product |
| `test_linalg_identity.py` | BOUNDARY | keep | 1320 |  |  |
| `test_linalg_int15.py` | BOUNDARY | keep | 57 |  |  |
| `test_linear_entry_table_boundary.py` | BOUNDARY | keep | 196 |  | s1:stale |
| `test_linear_serialization_reads.py` | BOUNDARY | keep | 29 |  | s2:stale |
| `test_linear_svc_score_native_boundary.py` | BOUNDARY | keep | 92 |  |  |
| `test_linear_svr_native_fit_boundary.py` | BOUNDARY | keep | 81 |  | s1:stale s2:stale |
| `test_list_input_float32.py` | BOUNDARY | keep | 45 |  |  |
| `test_lm_corpus_native_boundary.py` | BOUNDARY | keep | 265 |  |  |
| `test_lof_native_state.py` | BOUNDARY | keep | 147 |  | s1:stale |
| `test_logistic_multiclass_surface.py` | BOUNDARY | keep | 130 |  |  |
| `test_logistic_scalar_native_boundary.py` | BOUNDARY | keep | 157 |  | s1:product |
| `test_mamba23_decode_sessions.py` | BOUNDARY | keep | 20 |  |  |
| `test_mamba_boundary_admission.py` | BOUNDARY | keep | 182 |  | s2:stale |
| `test_mamba_surface.py` | BOUNDARY | keep | 733 |  |  |
| `test_manifold_installed_contract.py` | BOUNDARY | keep | 123 |  |  |
| `test_manifold_native_lifecycle.py` | BOUNDARY | keep | 90 |  |  |
| `test_maxabs_scaler_native_lifecycle.py` | BOUNDARY | keep | 188 |  |  |
| `test_maxabs_scaler_native_transport.py` | BOUNDARY | keep | 199 |  |  |
| `test_measurement_startup.py` | BOUNDARY | keep | 42 |  |  |
| `test_meta_native.py` | BOUNDARY | keep | 130 |  |  |
| `test_metrics_public_coverage_p1.py` | BOUNDARY | keep | 147 |  |  |
| `test_metrics_selection_host_p2.py` | BOUNDARY | keep | 148 |  | s1:stale |
| `test_min_child_hessian.py` | BOUNDARY | keep | 84 |  |  |
| `test_min_split_gain.py` | BOUNDARY | keep | 38 |  |  |
| `test_minmax_scaler_native_lifecycle.py` | BOUNDARY | keep; split: 86 l native -> archive_minmax | 295 | tests-to-mojo-classical |  |
| `test_mixture_surface.py` | BOUNDARY | keep | 179 |  |  |
| `test_mode_readback.py` | BOUNDARY | keep | 63 |  |  |
| `test_model_exception_ownership.py` | BOUNDARY | keep | 50 |  |  |
| `test_model_native_lifetime.py` | BOUNDARY | keep | 40 |  |  |
| `test_model_selection_numpy_free.py` | BOUNDARY | keep | 265 |  |  |
| `test_models_coverage.py` | BOUNDARY | keep | 155 |  | s2:stale |
| `test_models_loader.py` | BOUNDARY | keep; split: 114 l native -> byte_lm_models_loader_f16 | 664 | tests-to-mojo-models | s1:stale |
| `test_native_array_address_context.py` | BOUNDARY | keep | 98 |  |  |
| `test_native_array_operations.py` | BOUNDARY | keep | 200 |  | s1:stale |
| `test_native_array_packing_boundary.py` | BOUNDARY | keep | 113 |  |  |
| `test_native_array_storage.py` | BOUNDARY | keep | 280 |  |  |
| `test_native_array_view_lifetime.py` | BOUNDARY | keep | 112 |  |  |
| `test_native_caller_frame.py` | BOUNDARY | keep | 180 |  | s1:stale |
| `test_native_classical_public.py` | BOUNDARY | keep | 254 |  |  |
| `test_native_codec.py` | BOUNDARY | keep; split: 67 l native -> archive_codec | 287 | tests-to-mojo-classical |  |
| `test_native_convert.py` | BOUNDARY | keep | 182 |  |  |
| `test_native_cv_estimator_dispatch.py` | BOUNDARY | keep | 65 |  |  |
| `test_native_cv_lda.py` | BOUNDARY | keep; split: 63 l native -> resample_cv_row_order | 224 | tests-to-mojo-parallel |  |
| `test_native_cv_logistic.py` | BOUNDARY | keep | 152 |  | s1:product s2:stale |
| `test_native_cv_module_connections.py` | BOUNDARY | keep | 84 |  |  |
| `test_native_cv_scorers.py` | BOUNDARY | keep | 181 |  |  |
| `test_native_declared_params_setter.py` | BOUNDARY | keep | 284 |  |  |
| `test_native_decomp_availability.py` | BOUNDARY | keep | 65 |  |  |
| `test_native_decomp_params.py` | BOUNDARY | keep | 147 |  |  |
| `test_native_dense_coo.py` | BOUNDARY | keep | 26 |  |  |
| `test_native_field_constructors.py` | BOUNDARY | keep | 276 |  |  |
| `test_native_forest_auxiliary.py` | BOUNDARY | keep | 218 |  | s1:stale s2:product |
| `test_native_full_reset_params.py` | BOUNDARY | keep | 340 |  |  |
| `test_native_helpers.py` | BOUNDARY | keep | 114 |  |  |
| `test_native_idna.py` | BOUNDARY | keep | 78 |  | s1:product |
| `test_native_import_artifact_boundary.py` | BOUNDARY | keep | 85 |  |  |
| `test_native_instance_lookup.py` | BOUNDARY | keep | 86 |  |  |
| `test_native_label_code_packing.py` | BOUNDARY | keep | 79 |  |  |
| `test_native_linear_constructors.py` | BOUNDARY | keep | 216 |  |  |
| `test_native_loaded_module_hooks.py` | BOUNDARY | keep | 86 |  |  |
| `test_native_meta_metadata_snapshot.py` | BOUNDARY | keep | 136 |  | s1:product |
| `test_native_meta_public_boundary.py` | BOUNDARY | keep | 82 |  |  |
| `test_native_metric_scorer_helpers.py` | BOUNDARY | keep | 138 |  | s1:product |
| `test_native_mode_metadata.py` | BOUNDARY | keep | 143 |  |  |
| `test_native_model_records.py` | BOUNDARY | keep | 220 |  |  |
| `test_native_model_selection_inputs.py` | BOUNDARY | keep | 312 |  |  |
| `test_native_nonzero.py` | BOUNDARY | keep | 94 |  |  |
| `test_native_owner_builtin_identity.py` | BOUNDARY | keep | 55 |  |  |
| `test_native_parallel_cv_wire.py` | BOUNDARY | keep | 212 |  | s1:product s2:stale |
| `test_native_parallel_public_boundary.py` | BOUNDARY | keep | 97 |  |  |
| `test_native_params_metadata.py` | BOUNDARY | keep | 249 |  |  |
| `test_native_params_setter.py` | BOUNDARY | keep | 198 |  |  |
| `test_native_path_admission.py` | BOUNDARY | keep; drop 8 l source-text | 57 | tests-delete-source-text |  |
| `test_native_prep_attributes.py` | BOUNDARY | keep | 168 |  |  |
| `test_native_prep_composition_lifecycle.py` | BOUNDARY | keep | 112 |  |  |
| `test_native_prep_finite_connection.py` | BOUNDARY | keep | 44 |  |  |
| `test_native_prep_labels_door.py` | BOUNDARY | keep; drop 5 l source-text | 195 | tests-delete-source-text | s1:stale |
| `test_native_prep_no_callbacks.py` | BOUNDARY | keep; split: 86 l native -> prep_entry_table; drop 20 l source-text | 397 | tests-to-mojo-classical | s1:product |
| `test_native_prep_params.py` | BOUNDARY | keep | 120 |  |  |
| `test_native_prep_scores_door.py` | BOUNDARY | keep | 192 |  |  |
| `test_native_prep_select_door.py` | BOUNDARY | keep | 436 |  | s2:stale |
| `test_native_process_pool.py` | BOUNDARY | keep | 300 |  | s1:product |
| `test_native_public_attributes.py` | BOUNDARY | keep | 306 |  |  |
| `test_native_registered_owner_replacement.py` | BOUNDARY | keep | 137 |  | s1:stale s2:product |
| `test_native_rfe_cold_context.py` | BOUNDARY | keep | 59 |  |  |
| `test_native_scalar_decode_public.py` | BOUNDARY | keep | 120 |  | s1:product |
| `test_native_search_dispatch.py` | BOUNDARY | keep | 85 |  |  |
| `test_native_smallfam_public.py` | BOUNDARY | keep | 160 |  |  |
| `test_native_split_descriptor_admission.py` | BOUNDARY | keep | 54 |  |  |
| `test_native_splitter_params.py` | BOUNDARY | keep | 141 |  |  |
| `test_native_steady_state.py` | BOUNDARY | keep | 164 |  |  |
| `test_native_transport.py` | BOUNDARY | keep; split: 124 l native -> new check | 447 | tests-to-mojo-classical |  |
| `test_native_tree_lifecycle.py` | BOUNDARY | keep; split: 64 l native -> new check | 181 | tests-to-mojo-trees |  |
| `test_ne2_rounds_pointer_forms.py` | BOUNDARY | keep | 134 |  |  |
| `test_ne2_search_workers.py` | BOUNDARY | keep | 198 |  |  |
| `test_ne3_blocks_embedding_entries.py` | BOUNDARY | keep | 66 |  |  |
| `test_ne3_fused_entries.py` | BOUNDARY | keep | 58 |  |  |
| `test_nearest_centroid_native_state.py` | BOUNDARY | keep | 106 |  | s1:stale |
| `test_neighbors_kde_native.py` | BOUNDARY | keep | 202 |  | s1:stale |
| `test_neighbors_knn_native.py` | BOUNDARY | keep | 299 |  | s1:product |
| `test_neighbors_loc_native.py` | BOUNDARY | keep | 232 |  | s1:stale s2:stale |
| `test_neighbors_semi_native.py` | BOUNDARY | keep | 175 |  | s2:stale |
| `test_neighbors_sparse_input.py` | BOUNDARY | keep | 55 |  | s1:product |
| `test_neighbors_weighted_largek.py` | BOUNDARY | keep | 49 |  |  |
| `test_neural_block_admission.py` | BOUNDARY | keep | 111 |  | s2:stale |
| `test_neural_coverage.py` | BOUNDARY | keep | 213 |  | s1:stale s2:stale |
| `test_neural_dtype_diagnostics.py` | BOUNDARY | keep | 93 |  |  |
| `test_neural_metadata_snapshot.py` | BOUNDARY | keep | 119 |  |  |
| `test_neural_numpy_free_merge.py` | BOUNDARY | keep | 95 |  | s1:product |
| `test_neural_surface_consistency.py` | BOUNDARY | keep | 33 |  |  |
| `test_nmf_native_boundary.py` | BOUNDARY | keep | 304 |  |  |
| `test_not_fitted_native_lifetime.py` | BOUNDARY | keep | 62 |  |  |
| `test_numeric_profile.py` | BOUNDARY | keep | 688 |  | s1:product s2:stale |
| `test_numpy_free_core.py` | BOUNDARY | keep; split: 83 l native -> core_convert | 510 | tests-to-mojo-classical | s1:product s2:stale |
| `test_numpy_free_cv_contract.py` | BOUNDARY | keep | 44 |  |  |
| `test_numpy_free_features.py` | BOUNDARY | keep | 125 |  |  |
| `test_numpy_text_scalar_admission.py` | BOUNDARY | keep | 56 |  |  |
| `test_ocsvm_native_transport_boundary.py` | BOUNDARY | keep; split: 85 l native -> ocsvm_native | 290 | tests-to-mojo-classical | s1:stale s2:stale |
| `test_onnx_reader_native_provider.py` | BOUNDARY | keep | 60 |  |  |
| `test_ops_higher.py` | BOUNDARY | keep | 55 |  |  |
| `test_ops_import_kinds.py` | BOUNDARY | keep | 66 |  |  |
| `test_ops_optimizer.py` | BOUNDARY | keep | 154 |  |  |
| `test_ops_program.py` | BOUNDARY | keep; split: 82 l native -> core_ops_program | 308 | tests-to-mojo-neural |  |
| `test_ops_rankn.py` | BOUNDARY | keep | 290 |  |  |
| `test_ops_session_coverage.py` | BOUNDARY | keep | 218 |  |  |
| `test_ops_transformer_program.py` | BOUNDARY | keep | 143 |  |  |
| `test_optim_maximize_seam.py` | BOUNDARY | keep | 23 |  |  |
| `test_optimizer_native_array_ownership.py` | BOUNDARY | keep | 180 |  | s1:product s2:stale |
| `test_optimizer_native_config_order.py` | BOUNDARY | keep | 54 |  | s1:product s2:stale |
| `test_optimizer_native_constructor_precedence.py` | BOUNDARY | keep | 67 |  |  |
| `test_optimizer_native_container_admission.py` | BOUNDARY | keep | 175 |  | s1:product |
| `test_optimizer_native_lifecycle.py` | BOUNDARY | keep | 104 |  |  |
| `test_optimizer_native_mapping_order.py` | BOUNDARY | keep | 31 |  |  |
| `test_optimizer_native_step_errors.py` | BOUNDARY | keep | 49 |  |  |
| `test_optimizer_public_native.py` | BOUNDARY | keep | 283 |  | s1:product |
| `test_ordered_rmse_surface.py` | BOUNDARY | keep | 73 |  |  |
| `test_parallel_api_regression.py` | BOUNDARY | keep | 181 |  |  |
| `test_parallel_byte_native.py` | BOUNDARY | keep | 180 |  |  |
| `test_parallel_causal_lm.py` | BOUNDARY | keep; split: 67 l native -> causal_lm_entry; drop 58 l source-text | 460 | tests-to-mojo-parallel | s1:product s2:stale |
| `test_parallel_finite_scan.py` | BOUNDARY | keep | 79 |  |  |
| `test_parallel_forest_first_call.py` | BOUNDARY | keep | 38 |  |  |
| `test_parallel_gaussian_process.py` | BOUNDARY | keep | 69 |  |  |
| `test_parallel_neural_admission.py` | BOUNDARY | keep | 168 |  | s1:stale |
| `test_parallel_public_drivers.py` | BOUNDARY | keep; drop 8 l source-text | 487 | tests-delete-source-text |  |
| `test_pca_full_surface.py` | BOUNDARY | keep | 47 |  |  |
| `test_pca_native_boundary.py` | BOUNDARY | keep | 169 |  | s1:product |
| `test_pca_whitening_surface.py` | BOUNDARY | keep | 58 |  |  |
| `test_pcs_native_boundary.py` | BOUNDARY | keep | 137 |  |  |
| `test_peer1_provider_boundary.py` | BOUNDARY | keep; split: 161 l native -> new check; drop 21 l source-text | 484 | tests-to-mojo-classical | s1:product |
| `test_perf_classical_0855_boundary.py` | BOUNDARY | keep | 86 |  | s1:stale s2:product |
| `test_pls_native_boundary.py` | BOUNDARY | keep | 209 |  | s1:product |
| `test_portable_math_fast.py` | BOUNDARY | keep | 146 |  | s1:stale |
| `test_prep_endpoints_p2.py` | BOUNDARY | keep | 382 |  |  |
| `test_prep_public_coverage_p1.py` | BOUNDARY | keep | 253 |  |  |
| `test_prep_support_provider_boundary.py` | BOUNDARY | keep | 135 |  |  |
| `test_program_decode.py` | BOUNDARY | keep | 110 |  |  |
| `test_program_loop.py` | BOUNDARY | keep | 59 |  |  |
| `test_program_native_admission.py` | BOUNDARY | keep | 78 |  |  |
| `test_prophet_native_admission.py` | BOUNDARY | keep | 76 |  |  |
| `test_provider_boundary_audit_controls.py` | BOUNDARY | keep | 250 |  |  |
| `test_ptx_fallback.py` | BOUNDARY | keep | 370 |  |  |
| `test_public_aliases.py` | BOUNDARY | keep | 173 |  |  |
| `test_public_call_no_mode_wrappers.py` | BOUNDARY | keep | 184 |  | s2:stale |
| `test_qn_objectives_surface.py` | BOUNDARY | keep | 161 |  |  |
| `test_radius_native_settings.py` | BOUNDARY | keep | 36 |  |  |
| `test_ragged_lengths.py` | BOUNDARY | keep | 183 |  |  |
| `test_refuse_ignored_knobs.py` | BOUNDARY | keep | 147 |  |  |
| `test_regression_metrics.py` | BOUNDARY | keep | 224 |  |  |
| `test_resample_bca.py` | BOUNDARY | keep | 75 |  |  |
| `test_resample_options.py` | BOUNDARY | keep | 126 |  |  |
| `test_resample_surface.py` | BOUNDARY | keep | 116 |  |  |
| `test_resident_foreign_materialize_native.py` | BOUNDARY | keep | 149 |  |  |
| `test_resident_generation_contract.py` | BOUNDARY | keep | 159 |  | s2:stale |
| `test_resident_lifecycle_native.py` | BOUNDARY | keep | 227 |  |  |
| `test_resident_profile_contract.py` | BOUNDARY | keep | 68 |  | s2:stale |
| `test_rf_fit_notices.py` | BOUNDARY | keep | 87 |  |  |
| `test_rf_leaf_budget.py` | BOUNDARY | keep | 711 |  |  |
| `test_robust_scaler_native_lifecycle.py` | BOUNDARY | keep | 260 |  |  |
| `test_robust_scaler_native_transport.py` | BOUNDARY | keep | 171 |  |  |
| `test_runtime_env_restored.py` | BOUNDARY | keep | 70 |  |  |
| `test_runtime_manifold_partition.py` | BOUNDARY | keep | 71 |  | s1:product |
| `test_runtime_model_selection_partition.py` | BOUNDARY | keep | 57 |  |  |
| `test_runtime_sequential_partition.py` | BOUNDARY | keep | 66 |  |  |
| `test_safetensors_f16_py310.py` | BOUNDARY | keep | 58 |  |  |
| `test_safetensors_mmap_exporter_without_corpus.py` | BOUNDARY | keep | 46 |  |  |
| `test_safetensors_native_ndarray.py` | BOUNDARY | keep | 72 |  |  |
| `test_samba_checkpoint_native.py` | BOUNDARY | keep; drop 12 l source-text | 168 | tests-delete-source-text | s1:stale s2:product |
| `test_samba_config_native.py` | BOUNDARY | keep | 81 |  |  |
| `test_samba_construct_native.py` | BOUNDARY | keep | 145 |  | s1:stale |
| `test_samba_gradients_native.py` | BOUNDARY | keep | 127 |  | s1:stale s2:product |
| `test_samba_native_admission.py` | BOUNDARY | keep | 283 |  |  |
| `test_samba_state_native.py` | BOUNDARY | keep | 230 |  | s1:stale s2:stale |
| `test_samba_surface.py` | BOUNDARY | keep | 376 |  |  |
| `test_samba_target_count_contract.py` | BOUNDARY | keep | 75 |  |  |
| `test_scaler_complex_warning_native.py` | BOUNDARY | keep | 61 |  |  |
| `test_scaler_rows_snapshot_native.py` | BOUNDARY | keep | 116 |  |  |
| `test_scaler_sklearn_tags_native.py` | BOUNDARY | keep | 134 |  |  |
| `test_sequence_array_native_candidate.py` | BOUNDARY | keep | 353 |  |  |
| `test_sequence_array_native_guards.py` | BOUNDARY | keep | 109 |  |  |
| `test_sequence_dtype_alias_native.py` | BOUNDARY | keep | 195 |  |  |
| `test_sequence_layers_p2.py` | BOUNDARY | keep | 109 |  |  |
| `test_sequence_object_native_status.py` | BOUNDARY | keep | 116 |  |  |
| `test_sequence_scalar_cast_status.py` | BOUNDARY | keep | 51 |  |  |
| `test_sequence_shutdown_native.py` | BOUNDARY | keep | 156 |  |  |
| `test_sequence_string_native_candidate.py` | BOUNDARY | keep | 386 |  |  |
| `test_sequence_temporal_cast_contract.py` | BOUNDARY | keep | 93 |  |  |
| `test_sequence_text_print_contract.py` | BOUNDARY | keep | 167 |  |  |
| `test_sequential_native_admission.py` | BOUNDARY | keep | 221 |  |  |
| `test_sequential_native_buffers.py` | BOUNDARY | keep | 256 |  |  |
| `test_sequential_native_execution.py` | BOUNDARY | keep | 209 |  |  |
| `test_sequential_native_provider.py` | BOUNDARY | keep | 48 |  |  |
| `test_serialize_native_codec.py` | BOUNDARY | keep | 60 |  |  |
| `test_session_native_provider.py` | BOUNDARY | keep | 111 |  |  |
| `test_sk_validate_native_admission.py` | BOUNDARY | keep | 237 |  |  |
| `test_sklearn_compat.py` | BOUNDARY | keep | 642 |  |  |
| `test_small_mlp_surface.py` | BOUNDARY | keep; split: 69 l native -> training_small_mlp; drop 19 l source-text | 410 | tests-to-mojo-neural | s1:stale s2:product |
| `test_small_mlp_tensor_errors.py` | BOUNDARY | keep | 54 |  |  |
| `test_spectral_embedding.py` | BOUNDARY | keep | 118 |  |  |
| `test_spectral_native_boundary.py` | BOUNDARY | keep | 144 |  |  |
| `test_spectral_no_dependencies.py` | BOUNDARY | keep | 42 |  |  |
| `test_spectral_predict.py` | BOUNDARY | keep | 179 |  |  |
| `test_standard_scaler_native_lifecycle.py` | BOUNDARY | keep | 228 |  |  |
| `test_svm_metrics_context_native_boundary.py` | BOUNDARY | keep | 103 |  |  |
| `test_svr_native_transport_boundary.py` | BOUNDARY | keep | 109 |  |  |
| `test_svr_surface.py` | BOUNDARY | keep | 871 |  |  |
| `test_tokenizer_native_loading.py` | BOUNDARY | keep | 82 |  |  |
| `test_tokenizer_public_admission.py` | BOUNDARY | keep | 193 |  |  |
| `test_tokenizer_surface.py` | BOUNDARY | keep | 419 |  |  |
| `test_torch_compile_backend.py` | BOUNDARY | keep; drop 3 l source-text | 82 | tests-delete-source-text |  |
| `test_torch_native_modes.py` | BOUNDARY | keep | 163 |  |  |
| `test_torch_program_module_admission.py` | BOUNDARY | keep | 100 |  |  |
| `test_torch_state_dict.py` | BOUNDARY | keep | 200 |  |  |
| `test_training_cross_entropy_public_native.py` | BOUNDARY | keep | 418 |  | s1:product |
| `test_training_device_native_allocator.py` | BOUNDARY | keep | 55 |  |  |
| `test_training_device_native_finalizer.py` | BOUNDARY | keep | 171 |  | s1:product |
| `test_training_device_native_rollback.py` | BOUNDARY | keep | 100 |  |  |
| `test_training_embedding_public_native.py` | BOUNDARY | keep | 292 |  | s1:product |
| `test_training_linear_public_native.py` | BOUNDARY | keep | 270 |  | s1:stale |
| `test_training_metadata_public_native.py` | BOUNDARY | keep | 166 |  |  |
| `test_training_primitives_surface.py` | BOUNDARY | keep | 103 |  |  |
| `test_training_rms_public_native.py` | BOUNDARY | keep | 291 |  | s1:product s2:stale |
| `test_training_surface.py` | BOUNDARY | keep | 941 |  |  |
| `test_transductive_predict.py` | BOUNDARY | keep | 169 |  |  |
| `test_transformer_hd128.py` | BOUNDARY | keep | 96 |  | s2:stale |
| `test_transformer_options.py` | BOUNDARY | keep | 507 |  |  |
| `test_transformer_public_boundary.py` | BOUNDARY | keep | 260 |  | s1:stale |
| `test_transformer_state_refusals.py` | BOUNDARY | keep | 53 |  |  |
| `test_transformer_surface.py` | BOUNDARY | keep | 895 |  |  |
| `test_trees_nonfinite_refusals.py` | BOUNDARY | keep | 147 |  |  |
| `test_tsvd_native_boundary.py` | BOUNDARY | keep | 347 |  |  |
| `test_umap_native_constructor.py` | BOUNDARY | keep | 140 |  | s1:stale |
| `test_umap_options.py` | BOUNDARY | keep | 48 |  |  |
| `test_umap_surface.py` | BOUNDARY | keep | 149 |  | s1:product |
| `test_umap_transform.py` | BOUNDARY | keep | 107 |  | s1:product |
| `test_user_objects_native.py` | BOUNDARY | keep | 63 |  |  |
| `test_weight_import.py` | BOUNDARY | keep; split: 89 l native -> class_weight_plan | 294 | tests-to-mojo-neural |  |
| `test_x_decomp_native_api_cpu.py` | BOUNDARY | keep; drop 3 l source-text | 74 | tests-delete-source-text | s1:product |
| `test_x_linear_sanity.py` | BOUNDARY | keep | 539 |  |  |
| `test_x_metrics_sanity.py` | BOUNDARY | keep | 225 |  |  |
| `test_x_sequence_autoarima.py` | BOUNDARY | keep | 32 |  |  |
| `test_x_sequence_norm.py` | BOUNDARY | keep | 18 |  |  |
| `test_x_sequence_optim.py` | BOUNDARY | keep | 61 |  |  |
| `test_x_sequence_rnn.py` | BOUNDARY | keep | 36 |  |  |
| `test_x_sequence_sched.py` | BOUNDARY | keep | 37 |  | s1:stale |
| `test_xseq_native_provider.py` | BOUNDARY | keep | 161 |  |  |
| `tests/test_native_family_raw_attr.py` | BOUNDARY | keep | 45 |  |  |
| `test_arima_exog.py` | NATIVE | join arima_state | 322 | tests-to-mojo-spatial | s1:stale |
| `test_arima_parameter_lifetime.py` | NATIVE | join arima_state | 75 | tests-to-mojo-spatial |  |
| `test_autoarima_search_ownership.py` | NATIVE | new check checks/native_tests/autoarima_search.mojo | 58 | tests-to-mojo-parallel |  |
| `test_autograd_public.py` | NATIVE | join training_autograd_model | 324 | tests-to-mojo-neural |  |
| `test_autograd_usability.py` | NATIVE | existing check training_autograd_usability (named in file): delete Python numerics after assert-by-assert diff | 41 | tests-to-mojo-neural |  |
| `test_byte_lm_checkpoint_binary.py` | NATIVE | join byte_lm_entry | 236 | tests-to-mojo-models |  |
| `test_byte_lm_host_native_provider.py` | NATIVE | join byte_lm_entry | 213 | tests-to-mojo-models | s1:product s2:stale |
| `test_byte_lm_host_ragged_lifetime.py` | NATIVE | join byte_lm_entry | 29 | tests-to-mojo-models |  |
| `test_byte_lm_surface.py` | NATIVE | existing check byte_lm_trainer_session (named in file): delete Python numerics after assert-by-assert diff | 327 | tests-to-mojo-models | s1:stale |
| `test_c3_c5_no_copies.py` | NATIVE | new check checks/native_tests/c3_c5_copies.mojo | 125 | tests-to-mojo-classical | s1:product s2:stale |
| `test_cache_release_native.py` | NATIVE | new check checks/native_tests/cache_release.mojo | 221 | tests-to-mojo-parallel |  |
| `test_centroid_fit_native_boundary.py` | NATIVE | join centroid_fit_native | 92 | tests-to-mojo-spatial |  |
| `test_checkpoint_native_loading.py` | NATIVE | new check checks/native_tests/checkpoint_loading.mojo | 64 | tests-to-mojo-neural |  |
| `test_cholesky_archive_native.py` | NATIVE | join archive_cholesky | 111 | tests-to-mojo-classical |  |
| `test_classical_infer_one_call.py` | NATIVE | join decomp_classical_infer_one_call | 143 | tests-to-mojo-classical |  |
| `test_classical_partition_native.py` | NATIVE | join decomp_classical_infer_one_call | 60 | tests-to-mojo-parallel |  |
| `test_cnn_max_gradient_ownership.py` | NATIVE | join cnn_conv | 35 | tests-to-mojo-spatial |  |
| `test_cpu_kernel_variants.py` | NATIVE | join kmeans_kernel_cases | 103 | tests-to-mojo-classical |  |
| `test_cpu_parallel_rbf_sampler.py` | NATIVE | join spectral_rbf | 74 | tests-to-mojo-parallel |  |
| `test_cpu_training_e.py` | NATIVE | new check checks/native_tests/.mojo | 208 | tests-to-mojo-classical |  |
| `test_cpu_training_e2.py` | NATIVE | new check checks/native_tests/.mojo | 412 | tests-to-mojo-classical | s1:stale |
| `test_cpu_training_e3.py` | NATIVE | new check checks/native_tests/.mojo | 152 | tests-to-mojo-classical | s1:stale |
| `test_cpu_training_e3_gbdt_trees.py` | NATIVE | existing check gbdt_public_fits (named in file): delete Python numerics after assert-by-assert diff | 161 | tests-to-mojo-trees | s1:stale |
| `test_cpu_training_embedding_ivf.py` | NATIVE | join ann_ivf_surface | 389 | tests-to-mojo-spatial | s1:stale |
| `test_cpu_training_forest_variants.py` | NATIVE | join host_entry_forest | 187 | tests-to-mojo-trees |  |
| `test_cpu_training_misc.py` | NATIVE | new check checks/native_tests/.mojo | 402 | tests-to-mojo-classical |  |
| `test_cpu_training_par_classical.py` | NATIVE | join decomp_classical_infer_one_call | 346 | tests-to-mojo-classical | s2:stale |
| `test_cross_vendor_coordinator_native.py` | NATIVE | join resample_cross_vendor_fold | 175 | tests-to-mojo-parallel | s1:product |
| `test_cross_vendor_native.py` | NATIVE | join resample_cross_vendor_fold | 56 | tests-to-mojo-parallel |  |
| `test_cross_vendor_worker_native.py` | NATIVE | join resample_cross_vendor_fold | 204 | tests-to-mojo-parallel |  |
| `test_dart_buffer_lifetime.py` | NATIVE | join trees_entry_dart | 17 | tests-to-mojo-trees | s1:product |
| `test_decode_n_tokens.py` | NATIVE | existing check mamba_decode_sessions (named in file): delete Python numerics after assert-by-assert diff | 188 | tests-to-mojo-models | s1:stale |
| `test_decode_sessions_cpu_route.py` | NATIVE | existing check mamba_decode_sessions (named in file): delete Python numerics after assert-by-assert diff | 244 | tests-to-mojo-models |  |
| `test_distributed_program.py` | NATIVE | existing check training_distributed_program (named in file): delete Python numerics after assert-by-assert diff | 158 | tests-to-mojo-parallel |  |
| `test_distributed_program_boundaries.py` | NATIVE | existing check training_distributed_program (named in file): delete Python numerics after assert-by-assert diff | 63 | tests-to-mojo-parallel |  |
| `test_estimators_host_borrowed_buffers.py` | NATIVE | existing check tsvd_native (named in file): delete Python numerics after assert-by-assert diff | 58 | tests-to-mojo-classical |  |
| `test_feature_freq_experimental_surface.py` | NATIVE | new check checks/native_tests/feature_freq_experimental.mojo | 86 | tests-to-mojo-trees |  |
| `test_forest_inference_engine.py` | NATIVE | join host_entry_forest | 191 | tests-to-mojo-trees | s1:product s2:stale |
| `test_forest_list_label_admission_cost.py` | NATIVE | join gp_label_owner | 39 | tests-to-mojo-trees |  |
| `test_gbdt_host_modes.py` | NATIVE | join gbdt_ova_link | 193 | tests-to-mojo-trees |  |
| `test_gbdt_pair_logit.py` | NATIVE | join gbdt_ova_link | 99 | tests-to-mojo-trees |  |
| `test_gbdt_tree_metadata.py` | NATIVE | join gbdt_ova_link | 52 | tests-to-mojo-trees |  |
| `test_gbdt_yeti_rank.py` | NATIVE | join core_vjp_rank | 89 | tests-to-mojo-trees |  |
| `test_generator_draw_native.py` | NATIVE | new check checks/native_tests/generator_draw.mojo | 230 | tests-to-mojo-neural |  |
| `test_generator_state_native.py` | NATIVE | new check checks/native_tests/generator.mojo | 45 | tests-to-mojo-neural |  |
| `test_gp_normalize_y.py` | NATIVE | join gp_owner | 135 | tests-to-mojo-classical | s1:product |
| `test_gp_return_cov.py` | NATIVE | join gp_owner | 53 | tests-to-mojo-classical | s1:product |
| `test_gpc_surface.py` | NATIVE | join gpc_score | 236 | tests-to-mojo-classical |  |
| `test_holtwinters_inference.py` | NATIVE | existing check decomp_holtwinters_inference (named in file): delete Python numerics after assert-by-assert diff | 153 | tests-to-mojo-spatial | s1:stale |
| `test_host_density_hash_native.py` | NATIVE | new check checks/native_tests/density_hash.mojo | 278 | tests-to-mojo-classical |  |
| `test_host_model_linear_kernel.py` | NATIVE | join linear_owner | 101 | tests-to-mojo-classical |  |
| `test_host_model_svm.py` | NATIVE | new check checks/native_tests/svm.mojo | 95 | tests-to-mojo-classical |  |
| `test_host_radius_bindings.py` | NATIVE | new check checks/native_tests/radius_bindings.mojo | 43 | tests-to-mojo-spatial | s1:product |
| `test_host_tree_native.py` | NATIVE | new check checks/native_tests/tree.mojo | 182 | tests-to-mojo-trees |  |
| `test_ivf_embedding_cpu_inference.py` | NATIVE | join ann_ivf_surface | 288 | tests-to-mojo-spatial | s1:stale |
| `test_kde_fused_fast.py` | NATIVE | join kde_state | 156 | tests-to-mojo-classical |  |
| `test_km_kernels.py` | NATIVE | existing check kernel_methods_cells (named in file): delete Python numerics after assert-by-assert diff | 71 | tests-to-mojo-classical |  |
| `test_kmeans_native_boundary.py` | NATIVE | join kmeans_native | 258 | tests-to-mojo-spatial |  |
| `test_krr_options.py` | NATIVE | existing check kernel_methods_krr_options (named in file): delete Python numerics after assert-by-assert diff | 57 | tests-to-mojo-classical |  |
| `test_layernorm_public_native.py` | NATIVE | join core_ops_layernorm_view | 36 | tests-to-mojo-neural |  |
| `test_linalg_fast_bf16.py` | NATIVE | join bf16_linalg | 173 | tests-to-mojo-neural |  |
| `test_linalg_lowbit.py` | NATIVE | existing check bf16_conversion (named in file): delete Python numerics after assert-by-assert diff | 138 | tests-to-mojo-neural |  |
| `test_linear_archive_generic_native.py` | NATIVE | join archive_linear | 88 | tests-to-mojo-classical | s1:stale |
| `test_linear_svc_native_boundary.py` | NATIVE | join linear_svc_native | 127 | tests-to-mojo-classical |  |
| `test_logistic_native_predict.py` | NATIVE | join logistic_config | 47 | tests-to-mojo-classical |  |
| `test_logistic_query_width_native.py` | NATIVE | join logistic_query_width | 258 | tests-to-mojo-classical | s1:product s2:stale |
| `test_lowbit_weights.py` | NATIVE | join lowbit_weights | 190 | tests-to-mojo-neural |  |
| `test_mamba3_constructor_native.py` | NATIVE | join mamba3_prefix_step | 35 | tests-to-mojo-models | s1:stale |
| `test_mamba_native_public.py` | NATIVE | existing check mamba3_prefix_step (named in file): delete Python numerics after assert-by-assert diff | 221 | tests-to-mojo-models | s1:stale s2:stale |
| `test_meta_members_native.py` | NATIVE | new check checks/native_tests/meta_members.mojo | 256 | tests-to-mojo-parallel | s1:product |
| `test_minmax_archive_generic_native.py` | NATIVE | existing check archive_minmax (named in file): delete Python numerics after assert-by-assert diff | 113 | tests-to-mojo-classical | s1:stale |
| `test_mixture_reload_state.py` | NATIVE | join mixture_native | 32 | tests-to-mojo-spatial |  |
| `test_model_config_native.py` | NATIVE | join logistic_config | 65 | tests-to-mojo-neural | s1:stale |
| `test_multiclass_ova_surface.py` | NATIVE | existing check gbdt_ova_link (named in file): delete Python numerics after assert-by-assert diff | 64 | tests-to-mojo-classical |  |
| `test_native_cv_gbdt_nb.py` | NATIVE | join gbdt_ova_link | 209 | tests-to-mojo-parallel |  |
| `test_native_cv_outputs.py` | NATIVE | join resample_cv_row_order | 56 | tests-to-mojo-parallel |  |
| `test_native_cv_row_order.py` | NATIVE | existing check resample_cv_row_order (named in file): delete Python numerics after assert-by-assert diff | 80 | tests-to-mojo-parallel |  |
| `test_native_decomp_buffer_owners.py` | NATIVE | join decomp_entry_table | 80 | tests-to-mojo-classical | s1:product |
| `test_native_owner_qualification.py` | NATIVE | new check checks/native_tests/qualification.mojo | 91 | tests-to-mojo-classical |  |
| `test_native_prep_iterimp_door.py` | NATIVE | join prep_entry_table | 651 | tests-to-mojo-classical | s1:stale |
| `test_native_sgd_weight_lifetime.py` | NATIVE | join class_weight_plan | 62 | tests-to-mojo-classical |  |
| `test_native_stratified_connection.py` | NATIVE | join core_stratified_group_fp64 | 66 | tests-to-mojo-parallel | s1:stale |
| `test_native_transient_buffer_lifetimes.py` | NATIVE | new check checks/native_tests/transient_buffer_lifetimes.mojo | 213 | tests-to-mojo-classical |  |
| `test_ne2_g_decomp_entries.py` | NATIVE | existing check decomp_ne2_g_decomp_entries (named in file): delete Python numerics after assert-by-assert diff | 128 | tests-to-mojo-classical |  |
| `test_neighbors_ann_native.py` | NATIVE | join neighbors_ann_run | 202 | tests-to-mojo-spatial | s2:stale |
| `test_neighbors_graph_native.py` | NATIVE | join neighbors_graph_run | 206 | tests-to-mojo-spatial | s2:stale |
| `test_neural_backward_native.py` | NATIVE | join neural_entry | 111 | tests-to-mojo-neural | s1:stale s2:stale |
| `test_neural_block_native.py` | NATIVE | join neural_entry | 70 | tests-to-mojo-neural | s1:stale s2:stale |
| `test_neural_inference.py` | NATIVE | existing check training_neural_inference (named in file): delete Python numerics after assert-by-assert diff | 378 | tests-to-mojo-neural | s1:stale s2:stale |
| `test_neural_public_coverage_p1.py` | NATIVE | existing check sequence_optimizer (named in file): delete Python numerics after assert-by-assert diff | 331 | tests-to-mojo-neural |  |
| `test_neural_repeat.py` | NATIVE | join neural_entry | 220 | tests-to-mojo-neural |  |
| `test_neural_repeat_calls.py` | NATIVE | join neural_entry | 84 | tests-to-mojo-neural | s1:stale |
| `test_numpy_free_late_connect.py` | NATIVE | new check checks/native_tests/late_connect.mojo | 93 | tests-to-mojo-classical |  |
| `test_ops_graph.py` | NATIVE | join rankn_graph | 19 | tests-to-mojo-neural |  |
| `test_ops_session.py` | NATIVE | existing check core_ops_session (named in file): delete Python numerics after assert-by-assert diff | 83 | tests-to-mojo-neural |  |
| `test_optics_native_boundary.py` | NATIVE | join optics_native | 55 | tests-to-mojo-spatial |  |
| `test_optimizer_native_moment_assignment.py` | NATIVE | join ops_optimizer | 132 | tests-to-mojo-neural | s1:product |
| `test_optimizer_native_owner_copy.py` | NATIVE | join ops_optimizer | 63 | tests-to-mojo-neural | s1:product |
| `test_optimizer_native_scalar_constructors.py` | NATIVE | join ops_optimizer | 118 | tests-to-mojo-neural | s1:product |
| `test_parallel_arima_serialization.py` | NATIVE | join arima_state | 39 | tests-to-mojo-parallel |  |
| `test_parallel_byte_borrowed_inputs.py` | NATIVE | join byte_lm_parallel_byte_native | 181 | tests-to-mojo-parallel |  |
| `test_parallel_forecasting.py` | NATIVE | join parallel_archive_entry | 59 | tests-to-mojo-parallel |  |
| `test_parallel_mlp_native.py` | NATIVE | existing check training_small_mlp (named in file): delete Python numerics after assert-by-assert diff | 76 | tests-to-mojo-parallel |  |
| `test_parallel_model_selection.py` | NATIVE | join parallel_archive_entry | 235 | tests-to-mojo-parallel |  |
| `test_parallel_neural_native.py` | NATIVE | existing check training_small_mlp_cases (named in file): delete Python numerics after assert-by-assert diff | 325 | tests-to-mojo-parallel | s1:stale s2:product |
| `test_parallel_pool_numpy_parity.py` | NATIVE | join cnn_pool | 65 | tests-to-mojo-parallel |  |
| `test_parallel_shard_payloads.py` | NATIVE | join parallel_archive_entry | 124 | tests-to-mojo-parallel | s1:product |
| `test_parallel_trainers_coverage.py` | NATIVE | join parallel_archive_entry | 224 | tests-to-mojo-parallel |  |
| `test_parallel_worker_raw_results.py` | NATIVE | join parallel_archive_entry | 91 | tests-to-mojo-parallel |  |
| `test_partitioned_program.py` | NATIVE | existing check core_partitioned_program (named in file): delete Python numerics after assert-by-assert diff | 128 | tests-to-mojo-parallel |  |
| `test_pca_exact_native_owner.py` | NATIVE | join pca_native | 385 | tests-to-mojo-classical | s2:product |
| `test_program_constructor_native.py` | NATIVE | join core_ops_program | 106 | tests-to-mojo-neural |  |
| `test_prophet_native_ownership.py` | NATIVE | join prophet_state | 42 | tests-to-mojo-spatial |  |
| `test_public_cnn_verifier_methods.py` | NATIVE | join cnn_conv | 53 | tests-to-mojo-spatial |  |
| `test_public_model_selection_verifier_gaps.py` | NATIVE | join core_verifier_gap_methods | 129 | tests-to-mojo-parallel |  |
| `test_public_verifier_gap_methods.py` | NATIVE | existing check core_verifier_gap_methods (named in file): delete Python numerics after assert-by-assert diff | 69 | tests-to-mojo-classical |  |
| `test_py_bugs.py` | NATIVE | existing check portable_math_suite (named in file): delete Python numerics after assert-by-assert diff | 231 | tests-to-mojo-classical |  |
| `test_resident_sampling_policy.py` | NATIVE | join spectral_resident | 150 | tests-to-mojo-models | s2:stale |
| `test_rf_class_weight.py` | NATIVE | join class_weight_plan | 60 | tests-to-mojo-trees |  |
| `test_runtime_classical_partition.py` | NATIVE | join decomp_classical_infer_one_call | 99 | tests-to-mojo-parallel |  |
| `test_samba_attention_wiring.py` | NATIVE | join samba_entry_table | 94 | tests-to-mojo-models | s1:stale |
| `test_samba_cpu_inference_native.py` | NATIVE | join samba_entry_table | 150 | tests-to-mojo-models | s1:stale |
| `test_samba_decode_session.py` | NATIVE | join ops_session | 94 | tests-to-mojo-models | s1:product |
| `test_samba_inference_native.py` | NATIVE | existing check mamba3_prefill_session (named in file): delete Python numerics after assert-by-assert diff | 269 | tests-to-mojo-models | s1:stale |
| `test_samba_step_native.py` | NATIVE | join samba_entry_table | 209 | tests-to-mojo-models | s1:stale s2:stale |
| `test_scaler_input_snapshot_provider.py` | NATIVE | join prep_robust_scaler_native_lifecycle | 101 | tests-to-mojo-classical |  |
| `test_sequence_optimizer_public_native.py` | NATIVE | join sequence_optimizer | 192 | tests-to-mojo-neural |  |
| `test_sequential_native_checkpoint.py` | NATIVE | join training_sequential_common | 258 | tests-to-mojo-neural |  |
| `test_sequential_native_surface.py` | NATIVE | join training_sequential_common | 119 | tests-to-mojo-neural |  |
| `test_skewed_chi2_native_capability.py` | NATIVE | join neighbors_nearest_centroid_native_capability | 25 | tests-to-mojo-classical |  |
| `test_standard_archive_native.py` | NATIVE | join archive_standard | 122 | tests-to-mojo-classical | s1:stale |
| `test_standard_scaler_native_fit.py` | NATIVE | join archive_standard | 137 | tests-to-mojo-classical |  |
| `test_svc_multiclass.py` | NATIVE | join svc_probability | 183 | tests-to-mojo-classical |  |
| `test_svc_probability.py` | NATIVE | existing check svc_probability (named in file): delete Python numerics after assert-by-assert diff | 83 | tests-to-mojo-classical |  |
| `test_tokenizer_fingerprint.py` | NATIVE | new check checks/native_tests/tokenizer_fingerprint.mojo | 34 | tests-to-mojo-models |  |
| `test_training_array_metadata_contract.py` | NATIVE | join native_array_layout_check | 73 | tests-to-mojo-neural |  |
| `test_training_scalar_buffer_native.py` | NATIVE | new check checks/native_tests/scalar_buffer.mojo | 293 | tests-to-mojo-neural | s1:product |
| `test_trees_repeat.py` | NATIVE | join trees_door_run | 153 | tests-to-mojo-trees |  |
| `test_tsa_resident_state.py` | NATIVE | existing check theta_state (named in file): delete Python numerics after assert-by-assert diff | 147 | tests-to-mojo-neural |  |
| `test_vjp_rank_native.py` | NATIVE | existing check core_vjp_rank (named in file): delete Python numerics after assert-by-assert diff | 29 | tests-to-mojo-neural |  |
| `test_x_ann_repeat.py` | NATIVE | join ann_ops | 97 | tests-to-mojo-spatial |  |
| `test_x_cluster_twice.py` | NATIVE | join cluster_kmeans_query | 99 | tests-to-mojo-spatial |  |
| `test_x_cnn_batchnorm.py` | NATIVE | existing check cnn_batchnorm (named in file): delete Python numerics after assert-by-assert diff | 88 | tests-to-mojo-spatial |  |
| `test_x_cnn_blocks.py` | NATIVE | existing check cnn_blocks (named in file): delete Python numerics after assert-by-assert diff | 62 | tests-to-mojo-spatial |  |
| `test_x_cnn_conv.py` | NATIVE | existing check cnn_conv (named in file): delete Python numerics after assert-by-assert diff | 172 | tests-to-mojo-spatial |  |
| `test_x_cnn_gnn.py` | NATIVE | join cnn_conv | 252 | tests-to-mojo-spatial |  |
| `test_x_cnn_pool.py` | NATIVE | existing check cnn_pool (named in file): delete Python numerics after assert-by-assert diff | 120 | tests-to-mojo-parallel |  |
| `test_x_cnn_repeat.py` | NATIVE | join cnn_conv | 157 | tests-to-mojo-spatial |  |
| `test_x_cnn_trainer.py` | NATIVE | existing check cnn_trainer (named in file): delete Python numerics after assert-by-assert diff | 21 | tests-to-mojo-spatial |  |
| `test_x_decomp_kernel_table.py` | NATIVE | existing check mani_decomp_repeat (named in file): delete Python numerics after assert-by-assert diff | 65 | tests-to-mojo-spatial |  |
| `test_x_linear_repeat.py` | NATIVE | join linear_owner | 128 | tests-to-mojo-spatial |  |
| `test_x_metrics_repeat.py` | NATIVE | join neighbors_knn_metrics | 94 | tests-to-mojo-spatial | s1:stale |
| `test_x_neighbors_repeat.py` | NATIVE | join neighbors_ops | 116 | tests-to-mojo-spatial |  |
| `test_x_prep_twice.py` | NATIVE | join prep_entry_table | 93 | tests-to-mojo-classical |  |
| `test_x_sequence_garch.py` | NATIVE | existing check sequence_garch (named in file): delete Python numerics after assert-by-assert diff | 31 | tests-to-mojo-spatial |  |
| `test_x_sequence_prophet.py` | NATIVE | join sequence_moe | 32 | tests-to-mojo-spatial |  |
| `test_x_sequence_repeat.py` | NATIVE | join sequence_moe | 120 | tests-to-mojo-spatial |  |
| `test_x_sequence_theta.py` | NATIVE | existing check sequence_theta (named in file): delete Python numerics after assert-by-assert diff | 28 | tests-to-mojo-spatial |  |
| `tests/test_autograd_smoke.py` | NATIVE | join training_autograd_model | 77 | tests-to-mojo-neural |  |
| `test_admitted_call_doors.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 130 | tests-delete-source-text | s1:stale |
| `test_autograd_save_resume.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 219 | tests-delete-source-text |  |
| `test_byte_checkpoint_native.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 154 | tests-delete-source-text |  |
| `test_causal_lm_cli.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 51 | tests-delete-source-text |  |
| `test_chunked_lm_head_door.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 160 | tests-delete-source-text |  |
| `test_cpu_training_arima.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 146 | tests-delete-source-text | s1:stale |
| `test_cpu_training_e3_gbdt.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 182 | tests-delete-source-text | s1:stale |
| `test_cpu_training_e3_gbdt_rmse.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 156 | tests-delete-source-text | s1:stale |
| `test_cpu_training_gbdt_losses.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 157 | tests-delete-source-text | s1:product s2:stale |
| `test_cpu_training_gp.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 237 | tests-delete-source-text | s1:stale |
| `test_cpu_training_mamba.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 208 | tests-delete-source-text |  |
| `test_cpu_training_samba.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 181 | tests-delete-source-text |  |
| `test_cpu_training_transformer.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 127 | tests-delete-source-text |  |
| `test_cpu_training_umap.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 180 | tests-delete-source-text | s1:stale |
| `test_expose_d_manifest.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 181 | tests-delete-source-text |  |
| `test_holtwinters_initialization.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 138 | tests-delete-source-text |  |
| `test_host_surface.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 1322 | tests-delete-source-text |  |
| `test_lda_fit_block_estep.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 63 | tests-delete-source-text |  |
| `test_native_operation_codes_table.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 37 | tests-delete-source-text |  |
| `test_native_parallel_drivers_wire.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 258 | tests-delete-source-text | s1:stale s2:stale |
| `test_native_parallel_kernel_methods_wire.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 137 | tests-delete-source-text | s1:stale |
| `test_portable_kernel_models.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 35 | tests-delete-source-text |  |
| `test_svc_poly.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 125 | tests-delete-source-text |  |
| `test_tokenizer_manifest.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 197 | tests-delete-source-text | s1:stale |
| `test_weighted_score.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 67 | tests-delete-source-text |  |
| `tests/test_array_metadata_dtype_source.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 45 | tests-delete-source-text |  |
| `tests/test_prep_support_provider_source.py` | SOURCE-TEXT | delete; behaviour belongs in its boundary/native test | 27 | tests-delete-source-text |  |
| `_array_reference.py` | ORACLE | keep (external oracle) | 807 | tests-oracle-public-only |  |
| `_fold_reference.py` | ORACLE | keep (external oracle) | 138 |  |  |
| `_legacy_density_archive.py` | ORACLE | keep (external oracle) | 204 | tests-oracle-public-only |  |
| `_legacy_forest_auxiliary.py` | ORACLE | keep (external oracle) | 43 | tests-oracle-public-only |  |
| `_legacy_host_density_constructors.py` | ORACLE | keep (external oracle) | 25 | tests-oracle-public-only |  |
| `_legacy_host_iforest_fit.py` | ORACLE | keep (external oracle) | 201 | tests-oracle-public-only |  |
| `_legacy_host_inference.py` | ORACLE | keep (external oracle) | 484 | tests-oracle-public-only |  |
| `_legacy_host_trees.py` | ORACLE | keep (external oracle) | 773 | tests-oracle-public-only |  |
| `_legacy_neighbors_ann.py` | ORACLE | keep (external oracle) | 354 | tests-oracle-public-only |  |
| `_legacy_neighbors_graph.py` | ORACLE | keep (external oracle) | 461 | tests-oracle-public-only |  |
| `_legacy_neighbors_kde.py` | ORACLE | keep (external oracle) | 226 | tests-oracle-public-only |  |
| `_legacy_neighbors_knn.py` | ORACLE | keep (external oracle) | 484 | tests-oracle-public-only |  |
| `_legacy_neighbors_loc.py` | ORACLE | keep (external oracle) | 409 | tests-oracle-public-only |  |
| `_legacy_neighbors_semi.py` | ORACLE | keep (external oracle) | 323 | tests-oracle-public-only |  |
| `_legacy_sample_weight.py` | ORACLE | keep (external oracle) | 85 | tests-oracle-public-only |  |
| `_lower_oracle.py` | ORACLE | keep (external oracle) | 1222 | tests-oracle-public-only |  |
| `_mamba_block_reference.py` | ORACLE | keep (external oracle) | 446 | tests-oracle-public-only |  |
| `_onnx_proto_oracle.py` | ORACLE | keep (external oracle) | 321 | tests-oracle-public-only |  |
| `_onnx_reader_oracle.py` | ORACLE | keep (external oracle) | 343 | tests-oracle-public-only |  |
| `_prep_prog_oracle.py` | ORACLE | keep (external oracle) | 729 | tests-oracle-public-only |  |
| `_resolve_oracle.py` | ORACLE | keep (external oracle) | 714 | tests-oracle-public-only |  |
| `_scale_gamma.py` | ORACLE | keep (external oracle) | 63 |  |  |
| `_scale_gamma_oracle.py` | ORACLE | keep (external oracle) | 41 |  |  |
| `_svm_reference.py` | ORACLE | keep (external oracle) | 331 | tests-oracle-public-only |  |
| `_transformer_block_reference.py` | ORACLE | keep (external oracle) | 88 | tests-oracle-public-only |  |
| `_typed_input.py` | ORACLE | keep (external oracle) | 220 | tests-oracle-public-only |  |
| `test_classical_host_native_owner.py` | ORACLE | keep (external oracle); native part 45 l -> Mojo | 146 |  |  |
| `test_linalg_decompositions.py` | ORACLE | keep (external oracle) | 313 |  |  |
| `test_mamba1_backward_surface.py` | ORACLE | keep (external oracle) | 139 |  |  |
| `test_mamba23_backward_surface.py` | ORACLE | keep (external oracle); native part 52 l -> Mojo | 224 |  |  |
| `test_x_neighbors_sanity.py` | ORACLE | keep (external oracle) | 374 |  |  |
| `test_x_prep_binarizer.py` | ORACLE | keep (external oracle) | 25 |  |  |
| `test_x_prep_discriminant.py` | ORACLE | keep (external oracle) | 131 |  |  |
| `test_x_prep_encoders.py` | ORACLE | keep (external oracle) | 149 |  |  |
| `test_x_prep_imputer.py` | ORACLE | keep (external oracle) | 46 |  |  |
| `test_x_prep_iterative.py` | ORACLE | keep (external oracle) | 90 |  |  |
| `test_x_prep_kbins.py` | ORACLE | keep (external oracle) | 58 |  |  |
| `test_x_prep_labels.py` | ORACLE | keep (external oracle) | 59 |  |  |
| `test_x_prep_naive_bayes.py` | ORACLE | keep (external oracle) | 120 |  |  |
| `test_x_prep_normalizer.py` | ORACLE | keep (external oracle) | 26 |  |  |
| `test_x_prep_parity.py` | ORACLE | keep (external oracle) | 265 |  |  |
| `test_x_prep_polynomial.py` | ORACLE | keep (external oracle) | 31 |  |  |
| `test_x_prep_power.py` | ORACLE | keep (external oracle) | 50 |  |  |
| `test_x_prep_quantile.py` | ORACLE | keep (external oracle) | 48 |  |  |
| `test_x_prep_scaler_options.py` | ORACLE | keep (external oracle) | 233 |  |  |
| `test_x_prep_scalers.py` | ORACLE | keep (external oracle) | 50 |  |  |
| `test_x_prep_selection.py` | ORACLE | keep (external oracle) | 190 |  |  |
| `test_x_prep_spline.py` | ORACLE | keep (external oracle) | 73 |  |  |
| `test_x_prep_target_encoder.py` | ORACLE | keep (external oracle) | 80 |  |  |
| `test_x_sequence_ets.py` | ORACLE | keep (external oracle) | 35 |  |  |
| `test_x_sequence_mlp.py` | ORACLE | keep (external oracle) | 105 |  |  |
| `test_x_sequence_var.py` | ORACLE | keep (external oracle) | 32 |  |  |
| `test_x_trees_sanity.py` | ORACLE | keep (external oracle); native part 70 l -> Mojo | 487 |  | s1:product |
| `tests/test_forest_experiment_board.py` | TOOLING | keep (tools test) | 18 |  |  |
| `tests/tools/test_six_lane_matrix_io.py` | TOOLING | keep (tools test) | 94 |  |  |
| `tests/tools/test_six_lane_qualification.py` | TOOLING | keep (tools test) | 71 |  |  |
| `tests/tools/test_six_lane_review_history.py` | TOOLING | keep (tools test) | 70 |  |  |
| `test_distributed_cli.py` | VERIFY | protected: unchanged | 48 |  |  |
| `test_host_only_verifier.py` | VERIFY | protected: unchanged | 70 |  |  |
| `test_verification_coverage.py` | VERIFY | protected: unchanged | 490 |  |  |
| `test_verification_profiles.py` | VERIFY | protected: unchanged | 56 |  |  |
| `test_verify_all.py` | VERIFY | protected: unchanged | 1687 |  |  |
| `test_verify_causal_lm.py` | VERIFY | protected: unchanged | 101 |  |  |
| `test_verify_compare_challenge.py` | VERIFY | protected: unchanged | 794 |  |  |
| `test_verify_compare_commitment.py` | VERIFY | protected: unchanged | 767 |  |  |
| `test_verify_compare_profiles.py` | VERIFY | protected: unchanged | 130 |  |  |
| `test_verify_cross_check.py` | VERIFY | protected: unchanged | 133 |  |  |
| `test_verify_ctr_model_roles.py` | VERIFY | protected: unchanged | 98 |  |  |
| `test_verify_distributed.py` | VERIFY | protected: unchanged | 124 |  |  |
| `test_verify_environment_json.py` | VERIFY | protected: unchanged | 20 |  |  |
| `test_verify_generation.py` | VERIFY | protected: unchanged | 46 |  |  |
| `test_verify_model.py` | VERIFY | protected: unchanged | 310 |  |  |
| `test_verify_pending.py` | VERIFY | protected: unchanged | 107 |  |  |
| `test_verify_public_api.py` | VERIFY | protected: unchanged | 239 |  |  |
| `test_verify_reference_admit.py` | VERIFY | protected: unchanged | 661 |  |  |
| `test_verify_resources.py` | VERIFY | protected: unchanged | 67 |  |  |
| `test_verify_small.py` | VERIFY | protected: unchanged | 140 |  |  |
| `test_verify_worker.py` | VERIFY | protected: unchanged | 523 |  |  |
| `test_verify_worktree_commit.py` | VERIFY | protected: unchanged | 23 |  |  |

