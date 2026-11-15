# Verify smoke triage, 2026-11-15 (lane smoke-triage)

Baseline: the first between-release verify smoke, on the CPU-only Linux wheel of `91d5fdea4e` (Python 3.12, 4 cores),
`~/mojolearn-evidence/briefs-20261115/verify-smoke-91d5/`. Groupings and scripts:
`~/mojolearn-evidence/briefs-20261115/smoke-triage/` (`tests_by_cause.tsv`, `tests_by_file.tsv`, `classify.py`).
Every failing test file with its class and cause id: [SMOKE_TRIAGE_2026-11-15_tests.tsv](SMOKE_TRIAGE_2026-11-15_tests.tsv).
Fixes are on `lane/smoke-triage`, based on `preinteg/wave2-20261113` at `f3baf4814d`; every cause below was checked
to still exist at that head before it was fixed. Nothing was compiled or run: the fixes are owed the next smoke.

Classes: ENVIRONMENT (needs GPU / torch / MAX / onnx / an in-tree build: the smoke must mark it), TEST-STALE (the test
checks something deliberately removed or renamed), PRODUCT (the wheel or verify's own tooling is wrong).

## Counts

| pool | failing | ENVIRONMENT | TEST-STALE | PRODUCT |
|---|---|---|---|---|
| imports | 3 of 76 | 3 (adapters.max, adapters.torch, linalg_fast) | 0 | 0 |
| self-test | 1 | 0 | 0 | 1 (C09) |
| lanes | 28 of 510 | 0 | 0 | 28 |
| tests | 180 of 629 | 32 | 77 | 71 |

The test-file classes of C15-C17 (58 files with one cause each) are a judgement from the first error line, not a
confirmed root cause; the table marks them so.

## Root causes

| id | units | class | cause | fix |
|---|---|---|---|---|
| C01 | 20 test files (487 + 138 audit cases) | PRODUCT | `bindings/native_scalar_decode_unpack.mojo:68` (N197, `92eb1919aa`, 2026-11-07): the Py_buffer words and scalar bytes moved from `List` to `Array[Int, 10]` / `Array[UInt8, 32]`; every exact NumPy scalar of every width then failed the len/itemsize check ("numeric scalar decoding requires one primitive buffer element"). Every other Py_buffer caller and the `py_cpython_table` contract use a `List`. | FIXED `6f4f919bd2`: both back to native `List`s (two small allocations per admitted scalar; callers loop over parameters or dict entries, never rows). |
| C02 | lanes knn x6, radius x4, x-neighbors-metrics; 8 test files | PRODUCT | `python/mojolearn/_native_neighbor_queries.py:32` read `neighbors_native_key` only from an exact module; on a CPU-only install the route binding is the `_HostBinding` facade, so the captured route had no key and `knn_ops_key` (`bindings/neighbors_knn_native_door.mojo:333`) refused. | FIXED `7230d15c6f`: read the key through the facade. |
| C03 | lanes umap, x-decomp-umap-options; 4 test files | PRODUCT | `python/mojolearn/_native_spectral_umap_bridge.py` `_capture` never copied `umap_native_key` (`36454f5244`), so `_umap_check_mode` (`bindings/mani_umap_native_door.mojo:90`) refused on every column. | FIXED `7230d15c6f`: the captured route carries the key. |
| C04 | lanes gp-normalize-y, gp-sample-y-normalize; 5 test files | PRODUCT | `_native_scaler_bridge.py:20` handed the CPU `_HostBinding` facade as `native_scaler_provider`/`native_program_provider`; `neural_descriptor_entry` (`providers/python/neural_descriptor_pyapi_door.mojo:262`) takes only an exact module. | FIXED `ddae073c50`: `_backend.exact_host_module` (one import-time resolver; the UMAP bridge's private copy is removed). The remaining facade cases (import_lower programs, DBSCAN/metric-scorer tests) are lane `cpu-facade-audit` below. |
| C05 | 13 test files | TEST-STALE | `0e718fef73` deleted `causal_lm._route`, `_block_classes`, `_float_weight`, `_argmax_last`; the shared poison helper and the loader test still patched/called them. | FIXED `dcaf92d7cf`, `cf5c7d03df`: the helper asserts they are gone; the loader test uses a test-side greedy oracle; `test_greedy_argmax_ties_go_to_the_lowest_index` DELETED (it tested the deleted Python helper). |
| C06 | 17 test files (incl. test_verify_all) | ENVIRONMENT | children spawned with the SOURCE tree on PYTHONPATH (`tests/_fresh_process.py:31` and ten hand-rolled copies), so under the installed-wheel bootstrap they imported a binary-less source runtime ("no identical binary exists"). Three of the 17 (`_ROWS_CALL`, `_ROWS_CONTEXT`, codec not connected) are child failures behind the same assertion and stay owed to the next smoke. | FIXED `86ec9b87a2`: `PACKAGE_PARENT` is the runtime this process imported; the ten copies use it. |
| C07 | lane lowbit-conversions; test_neighbors_loc_native | PRODUCT (verify tooling) + TEST-STALE | `3db84bb031` deleted `_portable_math.ldexp`/`isclose`; verify's harness (`tools/identity_break.py:3998`) and a test oracle still used them. | FIXED `f590f54f60`: exact powers of two from Python ints (checked equal to ldexp for every exponent used); stdlib isclose in the oracle. |
| C08 | 2 test files + 2 in C15 | ENVIRONMENT | `torch`/`onnx` not installed in the smoke venv. | OWED (lane `smoke-env-marks`): `pytest.importorskip` in those files. |
| C09 | self-test (rc 245); lanes trees-dart-clf, trees-dart-reg, trees-shap-tree (rc -11); 5 test files (ExtraTrees fit, meta-estimator score/predict, Stacking init) | PRODUCT | native segfaults in tree fit/predict on the CPU column; the self-test dies in `public_checks` after `ols`, inside the estimator loop (no Python frame names the estimator). | OPEN: lane `cpu-native-crashes`. |
| C10 | 27 test files | TEST-STALE | source-text assertions over moved or renamed code (`in '# SPDX...'`, import lists, file paths), plus private helpers deleted by the Python-at-the-boundary work. | OWED: lane `stale-tests`. |
| C11 | 7 test files (optimizer owners) | PRODUCT | optimizer native state reads garbage or NaN parameters (`assert [-640548240] == [0]`, "NaN in input.param at flat index 0"). Possibly C01 (scalar constructors) or the same fixed-`Array`-storage pattern as C01. | OPEN: recheck after C01; then lane `cpu-native-crashes`. |
| C12 | 7 test files + 2 C04 tests | ENVIRONMENT | refusal tests monkeypatch the CPU `_HostBinding` facade, which the product no longer reads per call (it connected the exact host module at import), so the expected TypeError never comes. | OWED: lane `cpu-facade-audit` (patch the exact host module on a CPU install). |
| C13 | test_native_process_pool (+ ::serial), test_parallel_causal_lm | PRODUCT | `transport: peer closed the connection` / `operation timed out`. `bindings/native_posix_transport_unpack.mojo` keeps its request/hint/result words in `Array[...]` storage, the same pattern as C01. | OPEN: lane `cpu-native-crashes`. |
| C14 | 4 test files | TEST-STALE | archive tests assign into or expect NumPy from results that are now `mojolearn.Array`; a linear entry-table test calls a private classical operation that was removed. | OWED: lane `stale-tests`. |
| C15 | 6 test files | ENVIRONMENT | GPU-only modules (INTERNALERROR "no supported GPU" at collection, PTX fallback, measurement start-up asserting a GPU vendor) and torch/onnx. | OWED: lane `smoke-env-marks`. |
| C16 | 32 test files | TEST-STALE (unconfirmed) | imports of test helpers or private names that are gone (`_STATE`, `independent_gradient`, `label_kind`, `arm_linear`, `_block`, `_fit_forest`), the byte-LM host merged into `_mojolearn_native_orchestration_host`, and changed refusal wording ("Regex pattern did not match"). | OWED: lane `stale-tests`. |
| C17 | 20 test files | PRODUCT (unconfirmed) | one cause each: archive unicode error text, affinity preference scalar admission, byte-LM state restore, cross-validation schema "pending migration", forest `format` field, DART "invalid shape", GP mixed labels, and others (see the TSV). | OPEN: recheck against the next smoke after C01-C06; lane `product-residue`. |

Lanes not named above: cross-val-folds (PRODUCT, verify tooling): `f4d37e5eec` deleted
`model_selection._default_fold_arrays`, which `tools/identity_break.py:6922` (verify's harness) calls. mlp, mlp-bf16w,
mlp-int8w, neural-trainer-unfused (rc -6): `Fatal Python error: PyImport_GetModuleDict: interpreter has no modules
dictionary` during interpreter finalization, after the lane wrote its JSON: a native finalizer imports during
shutdown. transformer-decode-load-state, mamba3-decode-session (the CPU binding lacks the decode-step entries),
samba-decode-session (load_state steps differ from per-call steps), trees-oob-cv-link (`rf_regressor_fit_dart: invalid
shape`): PRODUCT, lane `cpu-native-crashes`.

Imports: the brief's "libamdhip64.so not found" for `linalg_fast` is the box-probe explanation appended to the
refusal. The real refusal is `NoCpuEntry: no CPU implementation of _mojolearn_linalg.fast_bf16_available`: the
module is the Apple `fast` BF16 face, and `fast` is Apple only. ENVIRONMENT. FIXED `f3167bb440`:
`tools/ops/wheel_verify_smoke.py` marks these three `env` only when the log carries their exact refusal. Any other
failure of the same module is still a failure.

## Next lanes (one paragraph each)

- **cpu-native-crashes** (compile-capable; needs a CPU box with gdb): reproduce C09 with `python -X faulthandler -m
  mojolearn._identity_break --lanes trees-dart-clf --fixtures base --require-backend cpu` and `verify --self-test`
  under gdb on the CPU wheel. Fix the faulting native site. Then C13: check every `Array[...]`-storage pointer handed
  to C (`bindings/native_posix_transport_unpack.mojo`, `class_weight_boundary_unpack.mojo:23`,
  `gp_mapping_boundary_unpack.mojo:27`, `providers/python/hierarchy_connectivity_pyapi_door.mojo:50`,
  `logistic_*_boundary_unpack.mojo` overflow words) against the C01 finding. Then the mlp finalization abort (find
  the finalizer that imports at shutdown; release without importing once `Py_IsFinalizing`), the decode-session
  entries and the DART shape refusal. Decided by: the lanes pool green on the CPU wheel.
- **verify-folds**: restore a verify-owned default-folds route for `cross-val-folds` (protected): expose the product's
  native fold assignment (`fold_ids` / `select_fold_i64`, still exported by `_mojolearn` and `_mojolearn_core_host`)
  to `tools/identity_break.py` through a small verify module, with the fold-order sabotage control intact. Do not
  bring back the deleted Python product helpers. Decided by: lane `cross-val-folds` OK, and its sabotage arm moving
  the hashes.
- **cpu-facade-audit**: every `_native_*_bridge.py` that hands `_backend.binding(...)` to native provider admission
  resolves it with `_backend.exact_host_module` (import_lower programs, metric scorer, KDE). The C12 refusal tests
  patch the module the product reads. Decided by: no "requires an exact compiled module" and no C12 "DID NOT RAISE"
  in the next smoke.
- **stale-tests**: C10, C14, C16. Each test either checks the product behaviour that replaced the removed code or is
  deleted with a line naming the commit that removed its subject. Decided by: those files green or deleted.
- **smoke-env-marks**: C08/C15, `pytest.importorskip("torch")` / `("onnx")`; GPU-only files skip by the existing GPU
  marker at collection instead of erroring. Decided by: zero INTERNALERROR files in the next smoke.
- **product-residue**: C17 and C11, read against the next smoke once C01-C06 are in. Many may share C01.
