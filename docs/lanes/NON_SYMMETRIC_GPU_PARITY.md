# GPU non-symmetric growth scope and minimum gain, 2026-09-09

This work implements GPU training controls. It does not add a CPU learner
or enable objectives solely to match CatBoost CPU. See
[TREE_GROWTH_SCOPE.md](TREE_GROWTH_SCOPE.md) for the persistent task scope.

Update 2026-09-10: [minimum child Hessian](GBDT_MIN_CHILD_HESSIAN.md) adds
optional candidate eligibility for RMSE/Logloss/CrossEntropy with Newton
scoring. Other objective/score combinations remain outside that option.

## Reference boundary

CatBoost's current GPU Depthwise/Lossguide registry has eleven objectives:
Poisson, MAPE, MAE, Quantile, LogLinQuantile, RMSE, Logloss, CrossEntropy,
Expectile, Tweedie and Huber. MojoLearn already exposes that same set.
Lq, MultiClass and MultiClassOneVsAll are absent from that GPU registry;
calling those omissions a CatBoost GPU parity gap would be incorrect.
The corresponding MojoLearn restrictions remain in place.

Sources: [CatBoost GPU non-symmetric registry](https://github.com/catboost/catboost/blob/master/catboost/cuda/train_lib/pointwise_non_symmetric.cpp),
[multiclass registry](https://github.com/catboost/catboost/blob/master/catboost/cuda/train_lib/multiclass.cpp).
The local upstream checkout is pinned at
`54a8143a5904ea1cfe98fe0d84c31d48cf13369b`; the current online registry was
also inspected. Availability is not a claim that all parameter combinations
produce identical models to CatBoost.

## Bounded growth-control matrix

| Capability | MojoLearn GPU status |
|---|---|
| Depthwise / Lossguide growth | Existing, public |
| Maximum depth / Lossguide maximum leaves | Existing, public |
| Minimum leaf row count | Existing, public; preserves CatBoost's terminal-size semantics |
| Selected-score improvement threshold | Added by this change: optional `min_split_gain` |
| Histogram reuse and stable row partitioning | Existing; IDENTICAL partition-statistics work is a separate lane |
| CatBoost GPU's eleven non-symmetric objectives | Existing, public |
| Coded categorical one-hot / simple CTR candidates | Existing; not general dynamic categorical combinations |
| NaN Min / Max / Forbidden | Existing; not a learned missing direction at each split |
| Four split scores and L2 regularization | Existing; score choice determines minimum-gain units |
| Interaction constraints / monotonic constraints | Still absent from public GPU growth |
| Minimum child Hessian | Optional `min_child_hessian` for the three audited scalar losses with Newton scoring; not a general child-weight control |
| Per-tree / per-node column sampling | Still absent from the main GBDT API |
| L1 leaf regularization / maximum leaf step | Still absent as general controls |
| CatBoost fixed binary splits / full feature-weight tuning | Still absent from the main public estimator |

LightGBM's [parameter reference](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
and XGBoost's [parameter reference](https://xgboost.readthedocs.io/en/stable/parameter.html)
identify useful growth controls (`min_gain_to_split`, `gamma`, child weight,
column sampling and constraints). Device/backend support must be checked
per feature; a parameter's presence in a general reference is not a claim
that every backend implements it. This change adds the minimum-improvement
control using MojoLearn's existing GPU scores, not their exact score formula.

## `min_split_gain` contract

```python
GradientBoosting(
    grow_policy="Lossguide", score_function="L2", min_split_gain=2.0,
    numeric_mode="identical",
)
```

- Public default `None` (native `-1`) disables the extra guard and preserves
  existing CatBoost-style selection, including Lossguide's ability to keep
  splitting without positive gain.
- An enabled value must be finite and nonnegative. It is supported only for
  Depthwise and Lossguide. Zero requires positive improvement; equality with
  the threshold rejects the split.
- Units are improvement in the selected split score. The GPU kernel computes
  its after-minus-before score and applies feature weights. The stored host
  record negates that value, so acceptance is
  `Float64(-best_split.gain) > min_split_gain`.
- The score may depend on the configured objective, score family, weight,
  regularization and score noise. It is not a universally normalized loss
  decrease, nor numerically interchangeable with XGBoost gamma or LightGBM
  `min_gain_to_split`.
- Filtering happens after the existing policy chooses its candidates. It
  leaves candidate ranking and ties intact. A trace adds `split.accepted`
  to distinguish accepted splits from Lossguide's earlier selected candidate.
- The binding accepts an optional final Float64 after the counted class
  weights. Default calls retain the original parameter length/layout. Older
  native extensions reject the longer enabled request and need rebuilding.

Code: `kernel/compute_scores.mojo` computes gain, the non-symmetric helper
restores the stored sign and filters selected leaves, and
`structure_searcher_options.mojo` validates the native option. Public and
prepared native APIs forward the same option; there is no CPU fallback.

## Checks

`checks/min_split_gain_check.mojo` drives actual GPU training on 256 rows
with four equal groups whose targets are 0, 0.25, 4 and 4.25. Under L2,
unit weights and zero regularization, the root improves score by exactly
1024; each child improves by exactly 2. Both growth policies must produce:

| Threshold | Leaves |
|---:|---:|
| disabled, 0, 1.999 | 4 |
| 2, 1023.999 | 2 |
| 1024 | 1 |

The check also requires exact group-mean predictions, bitwise model
round trips, and unchanged default model bytes. This exercises child
filtering and strict equality, which a single huge-threshold test cannot.

`python/mojolearn/tests/test_min_split_gain.py` covers invalid values,
symmetric-policy refusal and the counted class-weight ABI tail.
`checks/min_split_gain_binding.py --mode <mode>` exercises installed
extensions, reads back the compiled mode, repeats the boundary tests,
checks saved-model predictions and proves class weights remain effective
when the optional threshold is present. Run artifacts are under
`bench/results/min_split_gain_2026-09-09/`.

Validation completed on Apple M4 with Mojo 1.0.0 (`ed45d567`): native and
installed public-extension checks pass in FAST, DETERMINISTIC and IDENTICAL.
The selected Python regressions pass 76 tests plus 13 subtests. See
[validation results](../../bench/results/min_split_gain_2026-09-09/RESULTS.md).
Other GPU vendors have not been qualified by this local run.
