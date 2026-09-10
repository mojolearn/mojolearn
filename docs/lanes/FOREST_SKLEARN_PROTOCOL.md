# RF/ET sklearn protocol pilot

The four public RandomForest/ExtraTrees classifier/regressor estimators now
provide `get_params`, `set_params`, classifier/regressor tags, fitted-state
checks and `score`. This is a bounded interoperability pilot; it does not
claim complete sklearn estimator conformance or a cross-vendor pipeline.

The shared constructor registry retains the original parameter objects and
includes `numeric_mode` in the visible signature and cloning parameters.
Constructor parameters are public attributes. Native configuration is derived
separately and refreshed before fit, so parameter updates reach the learner.
`set_params` applies constructor-level checks to a replacement before changing
the estimator; successful updates clear fitted state. Constructor errors leave
the old estimator intact. Bounds checked only at native fit remain deferred.
A new fit also discards old fitted state before processing training inputs,
so a failed refit cannot predict using stale nodes with new class labels.
There are no nested estimator-valued forest parameters; nested Pipeline names
are routed by sklearn to the forest's explicit parameter registry. Subclasses
that introduce their own constructor must register its parameters with the
protocol; unregistered custom constructors refuse cloning/refitting rather than
silently dropping parameters.

Classifier `score` computes accuracy using the selected mode's existing GPU
metric. String or large-integer class labels are compared without narrowing;
exact integer equality indicators are passed to the GPU count kernel.
Regressor `score` uses the existing GPU R² metric in the Float32 target domain
used by tree fitting. Scores accept one-dimensional targets; sample weights
remain explicitly unsupported by this pilot. Existing metric edge-case
restrictions remain, including its finite-input and minimum-length contract.

sklearn is optional for constructing, fitting, predicting and scoring forests.
Tags require sklearn's public Tags API (introduced in 1.6); interoperability
checks in this slice qualify sklearn 1.8.0. Imports are lazy. CPU unit checks
mock the native fit/score boundary while exercising actual sklearn clone,
Pipeline and serial GridSearchCV/refit. They also check raw parameter objects,
mode preservation, invalid/atomic updates, fitted state and score limitations.

Existing `save`/`load` archives retain their inference-only format. They do not
store constructor parameters or numeric mode. Loaded archives predict, but
`get_params`/clone/refit explicitly refuse to invent training parameters.
Scoring such an archive follows its existing process-default mode unless the
caller explicitly sets `numeric_mode`. Python pickle preserves the new raw
parameter registry and mode; it is distinct from the legacy forest archive.
An archive/schema migration is separate work.

Validation commands (a Python environment with sklearn, pytest and NumPy):

```sh
PYTHONPATH=python python -m pytest -q python/mojolearn/tests/test_forest_protocol.py
tools/with_build_lock.sh env PYTHONPATH=python python checks/forest_protocol_binding.py
```

The public check uses rebuilt/existing native tree bindings, checks available
native tree mode metadata and exact artifact paths, requires metric numeric
mode readback, and runs all four estimators in all three modes through serial
GridSearchCV. Its sklearn StandardScaler is an interoperability fixture, not
an IDENTICAL preprocessing qualification. Multiworker GPU search, weighted
scoring, metadata routing and broader estimator conformance are not qualified.

Measured on Apple M4: all 12 public estimator/mode search cases passed
(60 GPU fits including refits), alongside 232 combined Python regression tests.
[Evidence and limitations](../../bench/results/regression_errors_2026-09-10/RESULTS.md).
