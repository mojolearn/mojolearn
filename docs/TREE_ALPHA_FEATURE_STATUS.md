# Tree alpha surface and remaining CatBoost coverage

This is a source audit, not a PyPI publication receipt or a new numerical
qualification. All test/build/measurement/model execution belongs to root on
authorized remote hardware, with at most three CPU cores. No subagent runs
qualification. Availability in a particular wheel requires its actual native
exports and retained installed-artifact evidence.

## Public entry points already exported

All currently implemented GBDT training entry points in
[the native binding](../bindings/_mojolearn_gbdt.mojo) already have public
Python routes in [ensemble.py](../python/mojolearn/ensemble.py), exported from
[mojolearn](../python/mojolearn/__init__.py). There is no additional unexposed
native ranking trainer to turn on with an import.

| Public API | Implemented scope | Explicit boundary |
| --- | --- | --- |
| `GradientBoosting` | Symmetric trees; also supported Depthwise/Lossguide combinations; weights, supported bootstrap/scorers, leaf estimation, heldout stopping, categorical codes, prediction and model save/load | Constructor validation restricts combinations; this is not full CatBoost API compatibility |
| `OrderedRMSE` | Numeric symmetric ordered RMSE, explicit permutation, weighted prefix cursors, zero bias, Newton-1 leaves, depth 1–8 | No ranking, categorical CTRs, general ordered objectives, bootstrap or early stopping |
| `ExperimentalTwoLevelFeatureFreq` | One RMSE tree, depth two, dense categorical source combinations and numeric columns, weighted scores/leaves | Experimental bounded combination route; not arbitrary-depth tensor CTR boosting |
| `get_tree_leaf_counts()`, `get_leaf_values()` | Model metadata from retained tree serialization | Metadata access does not qualify training arithmetic |

The standard `GradientBoosting` loss set is `RMSE`, `Logloss`, `CrossEntropy`,
`MultiClass`, `MultiClassOneVsAll`, `Quantile`, `MAE`, `LogLinQuantile`,
`MAPE`, `Poisson`, `Lq`, `Expectile`, `Tweedie`, and `Huber`. Parameterized losses
require the corresponding explicit loss parameter. Multiclass stores different
approximation dimensions for softmax and one-vs-all; use the documented
`predict_proba()`/`predict_classes()` routes rather than interpreting raw
approximations as probabilities.

## Ordered boosting is not ranking

`OrderedRMSE` estimates leaves from permutation prefixes to separate each
fold's leaf estimation from its evaluation tail. It still optimizes numeric
RMSE. It does not implement learning-to-rank objectives, relevance groups,
query weights, pairs, pair weights, or ranking metrics. Names such as
`PairLogit`, `YetiRank`, `QueryRMSE`, `QuerySoftMax` and `LambdaMart` are not
accepted objectives. Their appearance in source comments/options is not an
executable trainer. Do not introduce aliases that silently map them to RMSE.

## Categorical scope

[train.mojo](../gbdt/train.mojo) accepts explicitly declared dense numeric
category codes. Supported simple CTR processing builds `Borders` and
`FeatureFreq` columns and corresponding model tables; explicit one-hot
columns use equality splits. The wrapper exposes `cat_features`,
`one_hot_features`, `permutation_count` and
`ctr_estimation_permutation_id`. This is not a raw-string category hashing
API or unrestricted CatBoost CTR configuration.

The native source documents a material ordering difference: it does not add
CatBoost's preliminary dataset shuffle, so caller row order can affect
ordered categorical statistics. The experimental depth-two combination
entry does not establish general `max_ctr_complexity` support. Exposing
configuration knobs without an implemented consumer would be misleading.

## What is actually bitwise qualified

The [ordered implementation record](../gbdt/ORDERED_RMSE.md) identifies native
AMD MI325X/NVIDIA RTX 4090 qualification at `6dd44ac5`, with 130 ordered
records matching. Its [installed lane record](../bench/results/resume/2026-09-06-installed-gap-closure/installed-lane-comparison.json)
documents the `eb835021` Python fit/predict/save/load fixture across AMD and
NVIDIA, including the IDENTICAL model and 72 prediction cells. These are
named source/artifact/fixture claims, not qualification of all present source
or all wheel features. The later persisted-mode wrapper change needs the
remaining refreshed columns noted in the ordered record.

Selecting `numeric_mode="identical"` selects a compiled arithmetic profile;
it does not prove an installed wheel's identity on an untested workload.
Cross-vendor identity within mojolearn is also different from bitwise equality
to CatBoost. No universal CatBoost bit-parity claim is established here.

## Remaining implementation and alpha release queue

- [ ] Ranking: implement group/pair input contracts, a real ranking target,
  gradients and leaf estimation, appropriate sampling/scoring, prediction and
  retained metric checks before adding a public objective name.
- [ ] Ordered: extend beyond the bounded numeric RMSE profile only with
  explicit prefix-state semantics and independent leakage/correctness controls.
- [ ] Categorical: qualify broader combinations, depth/iteration composition,
  row ordering and unseen-category prediction before generalizing the
  experimental two-level route. Add raw-string encoding only with a specified
  stable hashing/category contract and serializable prediction mapping.
- [ ] Audit requested CatBoost conveniences separately: arbitrary categorical
  configuration, ranking/group metadata, custom objectives, multi-target
  objectives, feature-attribution/SHAP APIs and model interchange are not
  implied by symmetric-tree fit/predict support.
- [ ] Alpha packaging: retain existing public exports, require the actual
  GBDT extension symbols in each intended wheel, and verify public
  fit/predict/save/load and compiled mode/vendor witnesses. An alpha version
  label does not waive missing binaries or turn unsupported features into
  implemented ones. Root handles publication and qualification separately.

No production arithmetic or release gate was changed by this audit.
