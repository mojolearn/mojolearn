# GBDT sklearn adapters: implementation plan

Status: source audit and queued work, 2026-09-10. No adapter is implemented
by this document. The existing `GradientBoosting.predict` returns raw scores;
preserve that contract and introduce separate classifier/regressor adapters.

## Source and current boundaries

CatBoost pin `54a8143a`, `catboost/python-package/catboost/core.py`:
`CatBoostClassifier.predict` at 5587 defaults to Class; classifier `score`
at 5872 uses accuracy; regressor `score` at 6306 uses R². Mirror these
public meanings with our existing GPU learner and metrics, not their CPU
prediction backend. sklearn constructor objects must survive cloning without
normalization, as already handled for RF/ET in `_forest_protocol.py`.

The local source audit found these integration requirements:

| Existing surface | Required adapter behavior |
| --- | --- |
| `ensemble.py::GradientBoosting` constructor normalizes some options and creates `model_=None` | Preserve raw constructor parameters separately; build a validated internal learner at fit. Test fitted state using a non-None model, not attribute existence. |
| `predict` returns raw scalar or multiclass margins | Classifier `predict` maps predicted class codes to `classes_`; expose raw scores separately. Regressor prediction must define supported loss/link semantics explicitly. |
| Labels are currently converted to Float32; multiclass expects dense codes | Encode single-label classes on the host before GPU fitting, preserve original labels, and encode evaluation labels with the fitted training vocabulary. Refuse unknown evaluation labels. |
| `_mode.py::NumericModeMixin` resolves the live default on each call | Resolve and read back the mode at adapter fit, retain it for prediction/scoring/pickle, and preserve the original constructor value for clone. |
| `save/load` restores inference state without training constructor parameters | Do not claim old archives can clone/refit. Specify adapter archive metadata and label vocabulary before adding save/load; Python pickle can retain ordinary fitted state. |
| Binary `predict_proba` returns Float64; FAST/DETERMINISTIC use NumPy exp, IDENTICAL calls a host Mojo loop in `gbdt_sigmoid_binding` | Existing behavior is not a GPU probability-transform path. Add a separate GPU Float32 probability path for the bounded pipeline without silently changing legacy probability bits. |
| Public GPU `log_loss` accepts Float32 probabilities | Define Float32 probability output and clipping/normalization at the adapter boundary. An implicit dtype mismatch must not surface only inside a scorer. |
| `MultiClassOneVsAll` probabilities are independent sigmoids | Do not silently renormalize or advertise them as the same multiclass probability contract as softmax. Start with binary Logloss, then qualify softmax separately. |
| sklearn Pipeline does not automatically transform a raw `eval_set` passed to its final estimator | Document the restriction; do not leak validation data into scaler fitting or claim automatic evaluation-set routing. |

## Bounded first slice

1. Add explicit classifier/regressor types with clone/get_params/set_params,
   optional sklearn tags and fitted hooks. Reuse the forest protocol's raw
   parameter design, not its forest-specific fitted attributes. Start with
   RMSE regression and binary Logloss classification, all three growth policies
   and numeric modes already supported by those losses.
2. Preserve training labels and fitted mode. Validate replacement configuration
   before clearing state on `set_params`; clear stale fitted state when a new
   fit fails. Keep experimental GBDT subclasses outside the adapter contract.
3. Implement GPU probability/class selection needed by the classifier and use
   existing mode-aware GPU accuracy/R² for `score`. Retain the current refusal
   of weighted scoring until the metrics support it, even though training can
   accept weights. Label encoding/decoding remains host preparation.
4. Exercise a small serial sklearn Pipeline/GridSearchCV with the GPU scalers,
   fits and scoring. Compare raw learner predictions before/after wrapping,
   clone identity, class mapping, failed refits, fitted mode after a process
   default change, and pickle output bits. No broad sklearn parity claim.
5. Qualify intermediate statistics, transformed input, model and predictions,
   probabilities and scores across Metal/CUDA/HIP only when idle devices are
   authorized. Until then call it an implemented local pipeline, not a
   cross-vendor qualified end-to-end IDENTICAL pipeline.

Scaling is optional for decision trees. This integration makes preprocessing
available and composable; it does not establish a tree-training speed gain.
No CPU training backend or remote work is part of this plan.
