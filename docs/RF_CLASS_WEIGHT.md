# RandomForestClassifier class weights

`class_weight=None`, a dict keyed by observed labels, and `"balanced"` are
supported. Omitted dict labels receive weight 1; unknown labels are rejected.
`"balanced_subsample"`, arbitrary sample weights, and OOB output remain unsupported.
Balanced weights are `n_rows / (n_classes * class_count)` for the whole fit.
Weights must be finite, nonnegative, representable as Float32 without positive
values rounding to zero, and have positive total. All-unit weights use the
original unweighted binding, preserving its sampling and model behavior.

This is cuML-style weighted sampling, **not sklearn-equivalent weighting**:
with `bootstrap=True`, weights determine row sampling probability and are not
applied again to impurity. With `bootstrap=False`, zero-weight rows are excluded
and weights enter the native histogram objectives. The existing GPU tree engine
is reused. Its weighted sampler currently computes the cumulative weights,
random draws, and binary searches on the host; no separate CPU learner was added.
Its arithmetic/order and pinned RNG geometry do not promise identical models to
cuML. Weighted performance is unmeasured.

The reference is cuML v26.08.00 (`265b9da6`),
`python/cuml/cuml/common/classification.py:47–100`,
`python/cuml/cuml/ensemble/randomforestclassifier.py:279–286`, and
`cpp/src/randomforest/randomforest.cuh` RowSampler. cuML's RF requests Float64
weights; this bounded interface uses the existing Mojo Float32 weight engine.
Unknown dict labels are always rejected here, stricter than cuML's conditional
missing-label check. Host label bookkeeping computes the balanced class counts;
GPU histograms and the existing forest builder perform training.

The new `rf_classifier_fit_weighted(X_addr, y_addr, params, criterion,
weights_addr)` accepts a borrowed contiguous Float32 row-weight vector of length
`params[0]`. It returns the existing forest tuple. The original four-argument
`rf_classifier_fit` remains unchanged. Python keeps the weights alive through the
blocking call; native code validates/copies them and uploads device weights for
non-bootstrap objectives. No new numeric-mode defaults are introduced.

Host tests: `python/mojolearn/tests/test_rf_class_weight.py` checks validation and
mocked ABI dispatch, not GPU correctness. The future NVIDIA gate is:

```sh
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
  python checks/rf_class_weight_smoke.py
```

It requires CUDA and native mode readback, checks both bootstrap paths, repeated
models/predictions, exact None/unit behavior, weighted model changes, and finite
quality metrics. GPU qualification requires running this script on a freshly
built binding; local compile checks alone do not provide it.

Local validation on 2026-09-10: 19 host tests passed, and the IDENTICAL
Mojo shared-library compilation passed. No GPU execution was performed for
this implementation commit.
