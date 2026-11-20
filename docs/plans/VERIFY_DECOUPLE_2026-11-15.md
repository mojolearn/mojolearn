# Verify decoupling, 2026-11-15 (lane verify-runtime-report)

Source: `docs/plans/VERIFY_COVERAGE_AUDIT_2026-11-15.md` section 4 and section 5 lanes 1 and 8. The shipped verify
modules (`python/mojolearn/_verif*.py`) read product internals (`_backend`, `_identity`, `_families`, `._b`, `._id`,
`._extension`, ...). They now read one public report instead, so a product change that moves an internal does not
break verify.

## API

`mojolearn.runtime_report()` returns a new plain `dict` on every call. It is built by ONE native call
(`backend_public_native_call("runtime_report", (), context)` in `bindings/backend_public_native_door.mojo`, the
compiled entry `_native_backend_public.py` already binds from the core host unit `_mojolearn_tokenizer_host`). No
new binding and no new loader. It never loads a binding, never calls a Python loader or hook, and is never on a
product call path: it is its own call, for verify and diagnostics. It reads the selection the package made at
import, and the loaded modules as they are at the moment of the call.

| field | type | meaning |
|---|---|---|
| `version` | `str` | `mojolearn.__version__`. |
| `vendor` | `str` | `'cuda'`, `'hip'`, `'metal'` or `'cpu'`; the same value and the same refusals as `mojolearn.vendor()`. |
| `vendor_how` | `str` or `None` | How the vendor was decided (the words `python -m mojolearn verify` prints beside it). |
| `numeric_mode` | `str` | The loaded tier, `'identical'` or `'fast'`; the same value and checks as `mojolearn.numeric_mode()` (raises when the loaded gbdt binary was compiled for the other tier). |
| `default_mode` | `str` | The process default a call without `numeric_mode=` uses (`mojolearn.set_numeric_mode`). |
| `code_mode` | `str` or `None` | What the loaded gbdt binary of the selected tier says it was compiled as (`'identical'`, `'fast'`, `'unknown'` for an unreadable code), or `None` when no gbdt binary is loaded or it has no read-back. Verify compares it with `numeric_mode`. |
| `code_kind` | `str` | Which code the loaded GPU set carries: `'ptx'` exactly when the PTX set (`cuda_ptx/sm_80`) was selected at import, else `'native'` (a native GPU set, or the CPU bindings). The identity harness labels the CUDA PTX column from it. Never confused with `code_mode`, which is the compiled numeric mode. |
| `cpu_only` | `bool` | True on a CPU-only install (no GPU binary set loaded). |
| `gpu_arch` | `str` or `None` | `mojolearn.gpu_arch()`. |
| `gpu_arch_how` | `str` or `None` | `mojolearn.gpu_arch_how()`. |
| `host_dir` | `str` | The directory the CPU (host) bindings load from (`mojolearn/host/`, or `MOJOLEARN_HOST_DIR`). |
| `host_families_built` | `tuple[str, ...]` | Sorted basenames (`_mojolearn_<family>_host`) of every host binding present in `host_dir`: its own `.so` files plus every member its merged-file index names whose file exists. |
| `host_paths` | `dict[str, str]` | Every `host_families_built` basename -> the file that binding loads or would load (the same resolution as a host row's `expected_path`), whether or not it is loaded. Added 2026-11-16 (lane verify-product-entries) so the CPU column's host record hashes each family's file without loading it. |
| `bindings` | `dict[str, dict]` | Every native binding module of this package that is loaded, plus every member of a loaded tier that is not loaded yet. Key: the module name (`mojolearn._mojolearn_gbdt` for the selected binding, `mojolearn._sets.<mode>._mojolearn_<x>` for a tier member, `mojolearn._host._mojolearn_<x>_host` for a host binding). Value: `{'path': str or None, 'expected_path': str or None (host bindings, keys `mojolearn._host.*`: the file the process loads or would load for that binding, the value `_backend.door_host_binary_path(<basename>)` returns: the `MOJOLEARN_<X>_HOST_BINARY` override, else the own `<basename>.so` in `host_dir`, else the merged file its index names; None for every other row), 'loaded': bool, 'identity': str or None (sha256 hex of the file's bytes, computed natively; cached per path, mtime and size, so repeated reports hash a file once), 'vendor': str or None (the binding's own compiled read-back `<suffix>_vendor()`), 'compiled_mode': 'identical', 'fast', 'unknown' or None (its `<suffix>_numeric_mode()` read-back), 'sabotage': bool or None (its `<suffix>_sabotage()` read-back, host bindings), 'column', 'detected_column': str or None (its `<suffix>_column()` and `<suffix>_detected_column()` read-backs, host bindings; added 2026-11-16, lane verify-product-entries), 'mode': 'identical', 'fast' or None (the tier, for tier members)}`. Read-backs are None when the binding is not loaded or exports none. A loaded binding module whose file cannot be resolved (no `__file__`, no `__spec__.origin`) is listed with `path` None, never skipped. Merged host members are named from the merged index the package read when it resolved a host binding. |
| `fallbacks` | `tuple[str, ...]` | The names in the frozen authorized Python fallback list (`_runtime_fallbacks.FALLBACKS`); empty today. |
| `harness_source` | `dict` | `{'path': str or None, 'how': str}`: the identity harness this install runs, from `MOJOLEARN_IDENTITY_BREAK` (how `'MOJOLEARN_IDENTITY_BREAK'`), the wheel copy `mojolearn/_identity_break.py` (`'wheel copy'`), or the checkout's `tools/identity_break.py` (`'checkout'`); `path` None and how `'missing'` when none exists. |
| `source_commit` | `dict` | `{'commit': str or None, 'how': str}`: `MOJOLEARN_COMMIT` (how `'MOJOLEARN_COMMIT'`), else the wheel's `identity_columns/COMMIT` witness (`'wheel COMMIT witness'`, first token, 7 to 40 lowercase hex), else commit None with how `'git rev-parse HEAD of the checkout (the harness reads it)'`. |
| `checkout_root` | `str` or `None` | The repository root when the package sits in a checkout (the nearest of six parents holding `tools/identity_break.py`), else None. |
| `package_dir` | `str` | The directory of the imported `mojolearn` package. |

Rules for readers:
- A binding verify qualifies (path, sha256, vendor) is read from `bindings` AFTER the operation that loads it has run.
  `runtime_report()` never loads one; a binding that is neither loaded nor a tier member is absent.
  `_verify_native_runtime.loaded_binding(name, mode)` is the one verify-side lookup (tier key first, then the
  selected binding; the row's compiled read-back must say `mode`).
- The report is read-only. Changing a value in the returned dict changes nothing in the product.
- A field is added here first, then in the native door, then used.

Context connected once at import (`_native_backend_public._report_context`): the live `sys.modules`, `posix.listdir`,
`posix.getcwd`, `posix.stat`, the merged-index cache, the frozen fallback names, the package directory and its six
parents, and the identity cache. No product call reads any of it.

## What changed (lane verify-runtime-report)

Coupling sites in the wheel verify modules (`coupling_scan.py` + `coupling_refine.py` over this branch,
`~/mojolearn-evidence/briefs-20261115/verify-runtime-report/scan/`): 75 before, 37 after, of which 3 are
verify-to-verify (`_verify._git_commit`, `_verify_model._install` twice) and 34 are kept below.

Removed (replaced by `runtime_report()` or a public name):
- `_backend` (vendor, numeric_mode, default_mode, vendor_how, `_CODE_MODE`, binding + read_vendor,
  load_host_module + host_module_path) in `_verify`, `_verify_all`, `_verify_api_runner`, `_verify_api_utilities`,
  `_verify_api_models`, `_verify_causal_lm`, `_verify_distributed`, `_verify_generation`, `_verify_graph`,
  `_verify_light`, `_verify_parallel_cv`, `_verify_worker`, and the two program verifiers.
- `_identity` (harness_path, commit_witness, `_checkout_root`) in `_verify_all`; `_identity.py` itself now reads the
  report through `_verify.harness_path` / `_verify.commit_witness` (its own resolution code is deleted: one mechanism).
- `_forest_host.host_model` / `binary_path` / `host._binding`: `mojolearn.host_model` and the report's
  `mojolearn._host._mojolearn_forest_host` row (`_verify_all._require_forest_host_binding`).
- `_cpu_reference.reference_training` (a no-op since 2026-09-20): the three `with` blocks are unwrapped.
- `_buffer`, `_bufcheck`, `_array.Array` in `_verify`: the fixture is built with `array.array` and
  `mojolearn.Array.from_buffer`; little-endian bytes from `Array.tobytes()` (byte-swapped on big-endian hosts).
- `m._extension().mamba3_owned_weights_enabled`: the public `install_owned_weights()` refusal ('owned') decides it.
- `model._blocks[0]._extension()` and `tokenizer._binding()`: the report rows of the block binding and of
  `mojolearn._host._mojolearn_tokenizer_host` (one unaudited forward loads the block binding first).
- `_optional_numpy.require_numpy`, `_x_sequence_model.Sequential`, `s._b` in `_verify_model`.
- Audit lane 1 (verify-dedupe-opened): the duplicate `native`/`opened` controllers of `_verify_distributed_program.py`
  and `_verify_partitioned_program.py` are deleted; both use `_verify_native_runtime.audited` and `audited_context`.
  The `_backend.binding(...)` + `hasattr("ops_distributed_*"/"ops_partitioned_*")` probes are deleted (the public
  `distribute`/`partition` call refuses by name when an entry is missing), as is the `hasattr("ops_program_*")` probe
  in `_verify_graph.run_checks`.

Kept, with the reason (34 sites):
- `_families.load_all()` (2: `_verify.py`, `_verify_worker.py`): verify and its workers run on the complete loaded set
  so every unit is vendor- and mode-checked and hashed. There is no public "load everything" call, and adding one to
  the product for verify would be verify steering the product.
- Fixture modules shared with the product and with about 25 tests and tools (15: `_causal_lm_fixtures` in
  `_verify_api_models`, `_verify_causal_lm`, `_verify_generation`; `_tokenizer_synthetic` in `_verify_api_misc`;
  `_gpu_witness` in `_verify_distributed`; `_parallel_cv_witness` in `_verify_parallel_cv`). `_causal_lm_fixtures`
  is imported by `_portable_math.py` and `_tokenizer_synthetic` by `tokenizer.py`, so they are not test-only. The
  move is audit lane 6 (verify-fixture-home), a rename across files outside this lane.
- `_native_parallel_shard('faults_take')` (3, `_verify_distributed.py`): the transport fault arm itself (audit §4).
  It is the named transport fault hook of the distributed verifier (`transport_fault`), beside the comparison-boundary
  hook that audit lane 5 adds.
- `_backend.binding("_mojolearn_x_sequence")` (1, `_verify_model._binding`): verify-model's own native entries
  (`verify_digest`, `verify_fill`, `verify_file_sha256`, `verify_model_info`) live in that binding. A public API for
  them would put verify entries in the product surface.
- `model._forward(X, OUT_RAW)` (1, `_verify_model`): the raw model output. `predict` returns labels for
  classification models, which would weaken the saved-model report.
- `s._transformer_linear`, `s._transformer_softmax` (2, `_verify_graph`): the transformer block's pinned linear and
  softmax. No public op has the same fold; the graph check compares against the block.
- `held[0]._id`, `._b.ops_shape`, `loaded_session._id`, `._b.ops_shape` (4, `_verify_graph.native_session_boundary`):
  a stale-session-handle refusal (a native lifetime fault arm). The public API cannot hold a stale handle.
- `importers._program.ImportedProgram` and `captured._id` (2, `_verify_api_programs`): the endpoint under test is the
  class (`mojolearn.importers._program.ImportedProgram` in the API inventory); the public constructor
  (`import_model`) needs an exported artifact verify does not ship.
- `parallel_model_selection._capture` and `model_selection._NATIVE_CONTEXT[...]['artifact_paths']` (2,
  `_verify_parallel_cv`): the testing entry that returns completed fold models, and the exact native owner artifacts
  that capture connected. No public call returns either.
- `_runtime_fallbacks.approved_metadata_callbacks` (1, `_verify_native_runtime`): the frozen fallback list's own
  owner; it returns callback code identities, which a native report cannot carry. Its docstring names verify as its
  reader.
- `m._estimator_type` (1, `_verify_api_estimators`): the scikit-learn estimator-kind attribute, read only to pick
  labels or targets for the fixture.

Not counted by the scanner, kept: `_verify.binding_artifacts` reads `vars(sys.modules["mojolearn._backend"])` for the
placeholder types and hashes loaded modules itself. Its record format (`path_source`, conflicting `__file__` and
origin detection, size and mtime) is tested by `tools/test_binding_artifact_provenance.py`; moving it onto the
report rows is a follow-up, not done here.

How verify was exercised afterwards: not run (lanes do not run tests or verify); decided by the smoke
(`tools/ops/wheel_verify_smoke.sh`: `verify --self-test` and 510 of 510 lanes).
</content>
</invoke>
<invoke name="Bash">
<parameter name="command">bash -c 'cd ~/mojolearn-wt/verify-runtime-report && git add docs/plans/VERIFY_DECOUPLE_2026-11-15.md && git commit -q -m "VERIFY_DECOUPLE_2026-11-15: the runtime_report() API (fields, types, reader rules) the verify lanes code against

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && MOJOLEARN_WIDE_REBUILD_OK=1 git push -q origin HEAD:lane/verify-runtime-report > ~/mojolearn-evidence/briefs-20261115/vrr-push1.log 2>&1; echo rc=$?; tail -3 ~/mojolearn-evidence/briefs-20261115/vrr-push1.log; git log -1 --format=%h'