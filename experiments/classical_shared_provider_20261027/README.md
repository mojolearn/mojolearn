# Shared-provider SVC continuation (2026-10-27)

This bounded continuation targets repeated binary-machine calls inside SVC's
one-vs-one and probability-fold work. The solver's iterations were already native
in `svm.estimator`; the surrounding Mojo binding repeatedly read `_hp_*` settings
and dispatched that compiled solver/predictor through CPython.

Stable kernel/degree/C/tolerance/limits/gamma/cache settings are now captured once
per public fit or prediction operation in `SvcOperationSettings`, then passed
through pairs/folds. Prediction query dimensions are captured outside the pair
loop. Each later public call admits fresh model fields. No global metadata cache
or new guarantee for concurrent mutation is introduced.

`MOJOLEARN_CLASSICAL_NATIVE_DISPATCH=1` selects the existing common
`NativeDescriptorCall` / `NeuralDescriptorProvider` machinery with eight named
operations: binary fit0, binary predict1, pair/fold/gather/weight/epilogue2,
Platt calibration3, splitmix shuffle4, compile-time mode5, SVR fit6 and SVR predict7. Provider admission occurs once per
public operation, holds the exact module/capsule, and is not repeated per pair or
fold. No per-estimator provider class or public per-op adapter is added. The
checked identical GPU module supplies its own table. Candidate selection refuses
an unavailable table instead of silently claiming a converted route. Ordinary
CPU/other-mode routes remain supported with the candidate off (the default).

The table invokes the same `svc_fit_host_borrowed` and `svc_predict_host` numerical
owners with the same scalars, bounds, precomputed-kernel behavior, support layout
and output copies. Positive-shape, pointer and support-count refusals are retained.
Module, array and output owners remain live through calls. The common List-owned
result/error transport is reused. A plain private diagnostic tuple records
operation attempt counts; no capsule/provider is stored in model checkpoint state.

This is **not** a complete SVC runtime migration. Pair/fold construction,
class/weight preparation and output publication still use Python-owned containers.
Pair/fold/gather/weight/epilogue, Platt and shuffle dispatch now use the same
provider captured once by the full public operation. Gamma, class reductions,
one-vs-one layout and coefficient paths still use checked compiled-entry CPython
calls and require further work. The shared
provider itself is entered once per binary compute call to handle ownership and
GIL transitions; this tranche does not claim a Python-object-free outer pair loop.
SVR solver/predict dispatch now uses the same provider and original numerical
owners. Its surrounding input/output paths remain partial; OneClassSVM execution
remains pending. Separate source tranches connect
linear SVM and KMeans frontdoors and kernel methods through the common provider.
The whole-library scope remains recorded in
`experiments/library_provider_completion_20261027/inventory.json`.

## Validation and integration

Lane validation is source-only; no compilation or measurements were run. The
public probe is authored for root's rebuilt local Mac runtime. It compares public
fit outputs, decisions, predictions and probability outputs bit-for-bit with
candidate off/on for binary linear, multiclass RBF/poly and precomputed kernels,
including sample weights, refit after C mutation, operation counters, invalid
feature refusal and recovery. Controller NumPy preparation is test-only.

Compile changed SVM and tokenizer closures. Keep retained full compiler/binary/
source provenance and the public probe report. Other kernels, label types, class
weights, all refusal paths, CPU candidates and other targets stay pending. These
bounded checks are not performance evidence; no default promotion or speedup claim
follows.

The current common provider helper is inherited from the integration branch;
never replace it with an older lane snapshot.

The probe now poisons legacy binary/epilogue/calibration/shuffle/mode entries on
candidate calls, proving the corresponding operations do not redispatch through
those Python-facing bindings when executed. It checks each public call admits
mode metadata once. Source-only validation is not a passing runtime probe.

The same public probe includes bounded linear/RBF SVR with sample weights,
bitwise fitted-state/prediction comparison, mode and fit/predict counters, and
legacy SVR entry poisoning. No kernel arithmetic or solver choices changed.
