# GBDT sklearn adapters: bounded implementation and remaining work

Separate `mojolearn.GradientBoostingClassifier` and
`GradientBoostingRegressor` adapters cover binary Logloss and scalar RMSE
respectively. SymmetricTree, Depthwise and Lossguide use the existing GPU
learner in FAST, DETERMINISTIC and IDENTICAL, subject to the learner's
objective/score/option restrictions. The legacy `GradientBoosting.predict`
raw-score contract remains unchanged. Focused local build, GPU and serial sklearn pipeline checks pass;
no broad sklearn or cross-device pipeline qualification is claimed.

With optional sklearn installed:

```python
import numpy as np
from sklearn.pipeline import Pipeline
from mojolearn import StandardScaler, GradientBoostingClassifier, metrics

X = np.array([[0], [1], [2], [3]], dtype=np.float32)
y = np.array(["no", "no", "yes", "yes"])
pipeline = Pipeline([
    ("scale", StandardScaler(numeric_mode="identical")),
    ("model", GradientBoostingClassifier(
        n_estimators=4, max_depth=2, numeric_mode="identical")),
]).fit(X, y)
model = pipeline.named_steps["model"]
labels = pipeline.predict(X)
loss = metrics.log_loss(
    y, pipeline.predict_proba(X), labels=model.classes_,
    numeric_mode="identical",
)
```

## Public contract

These are numeric-feature adapters. They expose an explicit subset of the
base learner's constructor parameters: tree count/depth/rate and regularization,
binning, leaf estimation, bootstrap/sampling, overfitting detection,
score/search controls, class weights and growth constraints. They do not
accept arbitrary base-learner options. Their fixed losses are not constructor
parameters; multiclass, other losses, categorical feature APIs and experimental
GBDT subclasses remain outside this adapter contract. The regressor refuses
class weights. Classifier class-weight order follows its sorted classes.

The classifier requires exactly two nonempty, one-dimensional training
classes of one type: strings or integers, including bools. Host encoding
preserves original labels, including large integers. sklearn's own scorers
may reject object-dtype labels needed for arbitrary-size Python integers;
native label preservation does not imply universal sklearn label compatibility.
`classes_` is sorted;
its second entry is positive. `predict` returns original labels,
`decision_function` returns raw Float32 margins, and `predict_proba` returns
Float32 columns in `classes_` order. `score` uses GPU accuracy; an unknown
score-time label of the same type counts as incorrect. Evaluation labels,
in contrast, must belong to the training vocabulary.

The regressor requires finite, nonempty, one-dimensional Float32 targets
for fit, evaluation and scoring. Its predictions are raw Float32 RMSE
predictions, and `score` uses the GPU Float32 R² metric. Both adapters forward
training sample weights through the existing learner's supported path;
weighted scoring is explicitly refused.

`eval_set` accepts one `(X_eval, y_eval)` tuple or a list containing one such
pair. Evaluation features must already have the same preprocessing as training
features. sklearn Pipeline does not automatically transform an `eval_set`
passed to its final estimator; these adapters do not provide that routing or
fit a scaler on validation data.

## Parameters, fitted state and persistence

Constructor objects are retained for `get_params` and cloning; a validated
internal learner is built for fit. `set_params` rejects unknown parameters,
validates replacement configuration through construction and clears fitted
state on success. Learner checks that depend on training data remain fit-time
checks. A new failed fit cannot expose the previous fitted model.

The mode resolves at fit, is checked against the native readback and is stored
as `numeric_mode_`. The internal learner retains it for prediction and scoring
even if the process default changes. Learned metadata includes `model_`,
`n_features_in_`, loss curves, best iteration and early-stop state; classifier
metadata additionally includes `classes_` and `n_classes_`. Fitted detection
requires an internal non-None model, not merely an attribute named `model_`.

The adapters provide bounded classifier/regressor tags, fitted hooks and
get/set parameter support. Python pickle retains constructor parameters,
label vocabulary, fitted mode and learner state. Adapter `save`/`load` are
explicitly refused: legacy inference-only GBDT archives do not contain the
adapter vocabulary and constructor configuration. Custom subclass cloning,
metadata routing and arbitrary sklearn meta-estimators are not qualified by
this bounded protocol.

## Separate GPU Float32 binary prediction path

`gbdt/binary_prediction.mojo` computes the mode-aware sigmoid p of each
finite Float32 raw margin and returns `[1 - p, p]`. There is no probability
clipping; saturation to zero or one is allowed. GPU log loss owns its own
epsilon clipping. IDENTICAL uses the portable sigmoid with operand/result
FTZ. The legacy learner's host Float64 probability methods are unchanged.

Class selection uses strict raw margin > 0, matching the default binary
border in the pinned CatBoost reference. Sign/magnitude bits implement the
comparison: either signed zero selects class 0, while a positive subnormal
margin selects class 1. Rounded probabilities can consequently both equal
0.5 while `predict` selects class 1. Classification is not reconstructed from
an argmax of rounded probabilities. Native headers identify these policies
as `BINARY-PRED-1` and `BINARY-PRED-2`.

Postprocessing accepts at most Int32.max finite margins. It uploads host
margins and returns host arrays; the GPU link does not make prediction a fully
resident GPU pipeline. Neither that implementation choice nor its mode name
qualifies final results across devices.

## Provenance and remaining qualification

The inspected [CatBoost pin 54a8143a Python source](https://github.com/catboost/catboost/blob/54a8143a/catboost/python-package/catboost/core.py)
provides the public meanings: classifier prediction defaults to class labels,
classifier score is accuracy and regressor score is R². Its
[binary evaluation source](https://github.com/catboost/catboost/blob/54a8143a/catboost/libs/model/eval_processing.h)
uses sigmoid probability conversion and a raw-score class border. These are
behavior references; the new Float32 GPU postprocessing is an independent
implementation, not a claim of CatBoost numerical parity or a CPU backend.

Local builds and focused GPU/sklearn smoke pass; see the [evidence ledger](../../bench/results/gbdt_adapters_2026-09-10/RESULTS.md), including unresolved CoreAnalytics diagnostics. Broader intermediate-value,
model/prediction/probability, serialization and cross-device pipeline
qualification remains separate. Multiclass softmax and independent one-vs-all
sigmoids need distinct contracts before further adapters are exposed.
Scaling remains optional for trees; composition with GPU scalers establishes
no tree-training speed gain. No remote work or cross-device performance claim
is part of this bounded implementation.
