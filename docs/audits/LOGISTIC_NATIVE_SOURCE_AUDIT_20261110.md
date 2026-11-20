# Logistic native dispatch source audit — 2026-11-10

Compared `71c1b26f53` with its parent `7b8db97ebd`, independently of the
producer handoff. Two defects were found and repaired below. This is source
review: no lane compilation, runtime check, timing or identity run occurred.
It establishes neither a performance win nor the absence of runtime defects.

## Concrete defects repaired

1. `bindings/host_logistic_native.mojo:256`: deleting the optional probability
   fallback also deleted the final `x`, `weights` and `scores` lifetime uses
   (parent lines 273–275). Their raw addresses do not retain converted owners
   across `logistic_run_query`/`logistic_run_log`. Restored the three final
   owner uses after all numerical work, before label publication. No allocation,
   copy, kernel, synchronization or result arithmetic was added.
2. `glm/logistic_fold_native.mojo:16`: the new fold path used
   `core.hostptr.f32_ptr`, which refuses only zero, instead of the old
   `bindings/msel_device.mojo:261` positivity/alignment guard. Negative or
   unaligned y/coefficient/info/weight addresses could reach the new native
   fold entry. `core/logistic_native_api.mojo:106` now owns the same native
   guard; both fold fit and query use it in their original order and retain
   the original refusal text. Existing legacy wrapper admission remains.

## Numerical and ownership trace

| Reviewed contract | Source evidence and result |
| --- | --- |
| Dense fit arguments | Parent device wrapper `bindings/_mojolearn_estimators.mojo:1175` and host wrapper `bindings/_mojolearn_estimators_host.mojo` map to `QnFitRequest` and the calls at `glm/logistic_native_device.mojo:37` / `glm/host/logistic_native_host.mojo:37`. X, y, coefficient/info outputs, rows/features/classes, l1/l2, both tolerances, max/line-search iterations, L-BFGS memory, intercept, normalized penalty, optional weights, loss, SVR epsilon and finite status keep their meaning and kernel argument order. Zero weight address represents absent weights; legacy wrappers still refuse an explicitly present null weight pointer before constructing the request. |
| Device probability path | Parent device lines 1347–1352 and new device lines 56–66 allocate the same device score buffer, call `qn_decision_function_device`, then `qn_link_from_device`. Scores stay on the device; logarithm remains in that link. No intermediate host scores or replacement sigmoid/softmax was introduced. |
| Prediction kernels | Device decision/binary/multiclass still call `qn_decision_function_host`, `qn_predict_binary_host` and `qn_predict_multiclass_host`. `QN_DEV_ARGMAX` still controls the multiclass capability. Host query lines 51–85 retain the same decision/binary/link kernels and `host_predict_task_count`. No tuned kernel or threshold changed. |
| Host storage and passes | Host proba retains the same zero-filled `List[Float32]` score scratch and the same decision, sigmoid/softmax and optional log passes. Explicit host `predict_log_proba` still allocates a separate output and calls rows-log after proba, as its parent did. No new tensor-sized copy/pass was found in the dispatch move. The scratch allocation moved inside the GIL-released call. |
| Resident folds | `glm/logistic_fold_native.mojo:16` and `:30` use the unchanged `msel_fold_entry_context`, `qn_fit_fold` and `qn_infer_fold`. Store IDs/base-row count, decision=0/codes=1/proba=2, class/target count, logarithm and finite/range word ordering match the parent wrappers. There is no host gather or extra upload in the moved fold path. |
| Numeric mode and context | Device context keys remain `MojolearnEstimatorsContextIdenticalV3` / `MojolearnEstimatorsContextFastV3`. Native mode metadata reads the same `GLOBAL_NUMERIC_MODE`; fold context selection remains in unchanged `core/msel_device_x.mojo:104`. Existing public identical-only resident-fold selection remains at `bindings/logistic_native.mojo:450`. |
| Provider and buffer lifetime | `bindings/logistic_descriptor_boundary.mojo:60` through `:134` copy the native entry and retain the provider through `finally` around GIL-released execution; fold status words are retained too. Fit retains X, encoded y, weights, coefficient/info and finite-status owners at `bindings/logistic_native.mojo:336`. Query retains `_QueryX`, outputs and fitted state through the borrowed fields snapshot. The explicit host owner regression is repaired above. |
| Raw ABI admission | `core/logistic_native_api.mojo:115` checks operation-specific metadata counts, result alignment and nonempty metadata pointer alignment before dereference. `glm/logistic_fold_native.mojo:49` checks fold spans plus status alignment; query preserves positive/aligned destination refusal. Dense payload capacity is still the admitted caller's responsibility, as before: these address ABIs are not a general memory-safety validator. Existing host dimension, finite and kernel refusal checks remain in their native callees. |

The complete diff changes wrappers and adds typed dispatch modules; the existing
kernel implementations are untouched. Eleven traced kernel/support files were
also exported and hash-compared byte-for-byte, including `glm/estimator.mojo`,
`glm/qn_fold.mojo`, both `core/estimators*_impl.mojo` files,
`core/classical_host_predict.mojo`, `core/py2mojo_rows.mojo`,
`core/msel_device_x.mojo` and native descriptor/provider support. These hashes
identify reviewed source, not numerical reference results.

## Cost and remaining Python work

- New per-call work is bounded request metadata: native address/int/float Lists,
  the existing descriptor output/error owner, connection-cache lookup and an
  indirect native call. The old route constructed Python parameter lists and
  called compiled numerical wrappers through Python. The new cost has not been
  timed; do not infer a speedup from removing those Python calls.
- No PythonObjects/callbacks were found in the four new numerical RUN modules.
  Provider connection and image pins remain in the explicit boundary file.
- This is **not the complete Logistic Python-object migration**.
  `bindings/logistic_native.mojo:169` still computes penalty strengths on Python
  scalar objects; line 319 multiplies Python tolerance by 0.01. Lines 217 and
  629 retain compiled-function dispatch through Python for weight multiplication
  and the explicit host score route. These existed in the parent.
- The coefficient publication loops at `bindings/logistic_native.mojo:353` and
  the class-code copy at `bindings/host_logistic_native.mojo:231` are loops over
  model/tensor values, not fixed ABI metadata. They also predate this change;
  their remaining migration/parallelism work is not excused by their comments.

## Coverage, removals and integration obligations

- Added 35 independent, parallel native admission fixtures in
  `checks/native_tests/logistic_contracts_suite.mojo`: actual owned float32
  pointer access, seven legacy pointer refusals, all eleven v1 operation span
  shapes and sixteen malformed metadata/result/span refusals. An importable
  `run_checks` and thin standalone entry use the existing host pool.
- Retained all Python API/protocol/layout/owner tests, including
  `test_host_logistic_native.py`, `test_logistic_native_predict.py`,
  `test_logistic_multiclass_surface.py` and `test_native_cv_logistic.py`.
  These still need execution against the rebuilt bindings; no pass is claimed.
- Removed by the reviewed migration: Python numerical parameter lists/calls
  for the moved QN operations, optional split decision/link fallback, obsolete
  rows capability probing and Python mode-verdict cache. Legacy exported QN
  wrappers remain because other consumers still use them. The repair removes
  the weaker fold pointer helper usage; no kernel or supported feature removed.
- A/B define: none. No intended arithmetic or bit change. No speed measurement
  or identity run queued by this lane.
- Root build scope: identical `core` (`_mojolearn`, fold entry), identical and
  host `estimators`, host `runtime_classical` and `runtime_trees`; resolve the
  shared `core/logistic_native_api.mojo` closure through the normal controller.
- Native command for root: `bash tools/compile_slot.sh pixi run
  check-native-logistic-contracts`. Product compile, native execution, installed
  boundary tests and eventual NVIDIA/AMD end-to-end performance evidence remain
  separate obligations.

Complete retained source evidence:
`~/mojolearn-evidence/python-mojo-migration-20261110/logistic-audit/`
contains `migration.diff`, `boundary.diff`, `exporters.diff`, `connections.diff`,
before/after snapshots and `source-manifest.json`. No log filtering was used to
turn a partial build or pending test into a pass.
