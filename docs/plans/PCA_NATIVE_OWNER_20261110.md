# PCA native execution and cached state

Base: frozen integration `e53128fe7f9206abe748dbe32052a81ed864ef3c`.
The lane's base merge `290a6ae27e` matches that source tree exactly. Selectively
integrate the additive native API commit `e8958ae5e2` and its boundary follow-up;
the base merge is not another product change.

The public fit, fit-transform, transform and inverse-transform boundaries now
pass `PCAFitRequest` and `PCAQueryRequest` to the fixed native descriptor. Native
result structs carry kept rank, noise variance and finite/range status. The
PCA owner's metadata readback uses the same native provider. Parameter lists,
Python numerical result tuples and callable dispatch between compiled functions
are removed from these operations.

The estimator descriptor retains the estimator image and its existing context
slot. Resident folds have a separate descriptor retained from the identical
base image, which owns the model-selection store. Fold IDs therefore continue
to reach `msel_fold_entry_context`; they are never sent to an estimator context.
Each execution copies its native driver and retains its provider pin before
releasing the GIL. Input/output and fitted-array owners stay alive through the
native call and any existing prefix copy.

| Route | Existing numerical implementation retained |
| --- | --- |
| Exact/dense fit and fit-transform | `decomposition/impl/pca_owner.mojo` |
| Randomized fit and projection | Same `pca_randomized_owned` and randomized driver |
| Plain and whitened query | `decomposition/estimator.mojo` |
| Resident-fold fit/transform | `decomposition/pca_fold.mojo` and the base-image store |
| Host fit | `core/estimators_host_impl.mojo` |
| Host query | Existing host predictors, whitening checks and host prediction workers |

No kernel, algorithm, experiment default, TSQR/exact/randomized route or transfer
count was changed. Optional projection stays a zero address when absent. The
existing exact/MLE prefix copies, host inverse fitted-state copies, device
finite tests, fold gathers and synchronizations are retained. Legacy exported
entries remain available to their direct callers; the PCA owner no longer uses
them for compiled-function dispatch.

The per-model query cache now holds a registered `PCAQueryCache` Mojo owner.
Decoded dimensions, whitening, provider connection, mode and generation are
typed native values. Admission keys are native identity words with separate
Python reference pins at the boundary. The existing generation shortcut and
identity fallback remain; there is no repeated full admission on a valid hit.
Mutable decoded scalar arrays remain ineligible for caching. Every query still
captures fitted-array shape/dtype and performs the existing whitening finite
and sign scans, so rebinding or in-place modification stays observable.

The cache remains in the existing weak model registry, outside public model
state and serialization. `_mojolearn_runtime_decomp_host` is its sole type
registrar, using guarded lookup/add/attach before exporting the allocating PCA
door. Cache records never strongly retain their owning model.

Removed mechanisms: two import-time collections of per-function registration
tuples; per-call compiled vendor/mode getter calls; the query cache's Python
overlay dictionary and seven-element state tuple; Python argument/result lists
between compiled entries; the now-unused `pca_whiten_entries` helper. Python
remains in exact scalar/buffer admission, storage/image pins, public output
packing and import-time package connection. The existing scalar parser's
warnings, errors and mutable-scalar snapshots are unchanged.

Affected roots for the integrator:

- Identical `core`: `_mojolearn` (resident-fold descriptor).
- Identical `estimators`: `_mojolearn_estimators` (device descriptor).
- Host `estimators`: `_mojolearn_estimators_host` (host descriptor).
- Host `runtime_decomp`: `_mojolearn_runtime_decomp_host` (public boundary and cache registrar).

Native correctness fixtures and migration of direct numerical Python tests are
owned by the native-test lane and will be bundled by the orchestrator. The new
`python/mojolearn/tests/test_pca_native_boundary.py` contains 10 definitions / 24
expanded cases: poisoned legacy callables, native-factory callback refusal,
mutable fitted arrays and scalar settings, copy/weak lifetime, whitening
rechecks and refusal ordering. They are Python boundary seams; numerical/state fixtures belong in
Mojo. Existing public PCA coverage is retained.

Static evidence:
`~/mojolearn-evidence/python-mojo-migration-20261110/pca-native-owner/source-audit.json`.
The native source closure covers 198 repository files with zero Python or
boundary imports. Seven directly reached numerical implementation files are
byte-identical to the frozen base. Both edited Python files parse; fixture
counts come from the AST. These are source findings only. No lane compilation,
test execution, timing or identity run occurred. Compilation and installed
boundary execution remain owed to the orchestrator; no speed or bits claim is
made and no A/B define was added.
