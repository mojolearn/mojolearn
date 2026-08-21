# What of CatBoost this is, and what it is not

Audited 2026-08-21. **The question this file answers is "am I going to hit a
wall", and the honest answer has three parts, because "CatBoost" means three
different-sized things.**

    1. their GPU symmetric pointwise learner    ~complete
    2. their GPU learner                        missing the DEFAULT arm
    3. CatBoost the product                     a slice

The previous version of this file was written before the loss breadth, the
CTR path, multiclass and early stopping landed, and listed all four as
missing. Those sentences are deleted rather than annotated (`PORTING_RULES.md`
17). What follows is measured against their source at `54a8143a`.

## 1. Complete: their GPU pointwise + multiclass targets

Every objective their GPU pointwise target ships
(`pointwise_target_impl.h:259-299`), plus both multiclass ones:

    RMSE  Logloss  CrossEntropy  Quantile  MAE  LogLinQuantile  MAPE
    Poisson  Lq  Expectile  Tweedie  Huber  MultiClass  MultiClassOneVsAll

with the leaf estimator each one selects, because the objective picks it and
not the caller: `SetLeavesEstimationDefault` (`catboost_options.cpp:273-360`)
is ported, so Newton at one iteration for RMSE, ten for Logloss, twenty for
Tweedie, Gradient for LogLinQuantile and low-`q` Lq, and the EXACT weighted
quantile estimator for MAE, MAPE and Quantile.

Around it: `border_count` and the GreedyLogSum grid, one-hot features, simple
CTRs (three `Borders` priors plus `FeatureFreq`, their GPU `simple_ctr`) with
apply-time tables, `permutation_count` with per-permutation cursors, sample
and class weights, Bayesian / Bernoulli / Poisson bootstrap, Cosine /
NewtonCosine / L2 / NewtonL2 score functions, eval sets, the overfitting
detector (`IncToDec`, `Iter`), `use_best_model`, and model text that round
trips bit-for-bit.

`MultiClassOneVsAll` trains and is gated in Mojo but is NOT reachable from
Python: its kernels do not fit in the CPython extension (PORTING.md 70).

## 2. The gap inside their GPU learner, and it is the default

**`boosting_type=Ordered` is CatBoost's shipped GPU default** for every
non-multiclass loss (`catboost_options.cpp:803-807`), and it is not here.
It is not a switch we failed to wire: it lives in a different boosting class
over a different data layout (`dynamic_boosting.h` + the feature-parallel
learner), and the learner this repository ported refuses it in their own
source -- "But no ordered boosting",
`train_template_pointwise_greedy_subsets_searcher.h:14`, `:34`. **This port
is their `boosting_type=Plain` arm.** Priced at ~4,000 lines plus a second
histogram kernel family in PORTING.md 88.

**Feature combinations (tree CTRs) are not ported.** Their
`max_ctr_complexity` defaults to 4; ours is pinned at 1 and `check()` refuses
anything larger rather than computing simple CTRs under a name that promises
combinations. With ordered boosting, this is the other half of what makes
CatBoost CatBoost.

Missing values were on this list until 2026-08-21 and are not any more:
`nan_mode` is ported (`gbdt/data/quantization.mojo`), defaults to their
`Min`, resolves PER FEATURE the way `ComputeNanMode` resolves it, spends one
of the column's borders on the sentinel as `CalcQuantization` does, travels
in the model text, and refuses a NaN on a column the learn pool had none in.
**The knob is not on the Python surface yet** -- the default reaches it
through `train`, so NaNs are handled from Python, but `nan_mode="Max"` is
not selectable there until the extension is rebuilt.

Also absent from their GPU learner: `Depthwise` / `Lossguide` / `Region` grow
policies and non-symmetric trees, snapshotting, multi-GPU, and custom
objectives.

## 3. CatBoost the product

Whole subsystems, none started:

* **Ranking and pairwise** -- YetiRank, PairLogit, QueryRMSE, QuerySoftMax,
  QueryCrossEntropy, LambdaMart. This port carries no `group_id` at all, so
  the data model is missing before the losses are.
* **Multi-target and uncertainty** -- MultiRMSE, MultiLogloss,
  MultiCrossEntropy, RMSEWithUncertainty.
* **Survival** -- Cox, SurvivalAft.
* **Text and embedding features**, and the online estimators behind them.
* **Interpretation** -- feature importances of every kind, SHAP values,
  interaction strength. Nothing.
* **Workflow** -- cross-validation, `staged_predict`, grid/randomized search,
  `ignored_features`, monotone constraints, feature penalties,
  `eval_metric` selection and the metric zoo (most of their 80
  `ELossFunction` entries are METRICS, not objectives).
* **Export** -- ONNX, CoreML, C++/Python model export. The model crosses as
  this repository's own text format.
* **CPU training.** There is none. This is a GPU learner.

## 4. Pinned, where the pin IS their GPU behaviour

Worth separating from the list above, because these look like gaps and are
not. Verified from their source:

* `rsm` -- appears in all of `catboost/cuda` only under
  pairwise oblivious trees. Their plain GPU learner has no feature sampling.
* `random_strength` -- a NO-OP on their GPU symmetric path: the noise draw is
  identical in `score` and `scoreBefore` and cancels in the gain the argmax
  compares (`score_calcers.cuh:160-168`, `compute_scores.cu:131-142`).
* `min_data_in_leaf` -- CatBoost itself ignores it under SymmetricTree
  (`greedy_search_helper.cpp:691-694`).
* `max_leaves` -- for a symmetric tree this IS `2^depth`; their own check
  refuses anything else (`catboost_options.cpp:993`).
* `MVS` bootstrap -- asserted away on their GPU searcher, so it is not a
  reachable arm to port.
* `boost_from_average`, `model_shrink_rate`, `model_size_reg` -- refused
  here; each would be a different model rather than a knob, and
  `check()` says so by name.

Every refusal above raises with its reason and its source line rather than
being silently ignored, which is the difference between a pin and a lie.
