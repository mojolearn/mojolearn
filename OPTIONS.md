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
and class weights, Bayesian / Bernoulli / Poisson bootstrap, the Cosine, L2,
NewtonCosine and NewtonL2 score functions, eval sets, the overfitting
detector (`IncToDec`, `Iter`), `use_best_model`, and model text that round
trips bit-for-bit.

**`NewtonCosine` and `NewtonL2` are claimed again as of 2026-08-21**: they
dispatch onto the Cosine and L2 calcers, which is CatBoost's own structure
-- the Newton spellings differ only in which derivative the caller put in
the stat planes -- and that choice, `secondDerAsWeights`, is now ported
(PORTING.md 144) and gated per cell and per model by
`original/second_der_weights_check.mojo`. `SolarL2`, `LOOL2` and `SatL2`
still fall through the greedy searcher's dispatch to Cosine and stay
refused by name from Python rather than accepted and silently substituted.

`MultiClassOneVsAll` trains and is gated in Mojo but is NOT reachable from
Python: `predict_proba` would have to route through the elementwise sigmoid
over `n_classes` independent approxes rather than MultiClass's softmax, and
nothing gates that yet. (The reason recorded here until 2026-08-21 -- that its
kernels did not fit in the extension -- was the basename theory PORTING.md 70
has retracted.)

## 2. The gap inside their GPU learner, and it is the default

**`boosting_type=Ordered` is CatBoost's GPU default only on SMALL pools**,
and the flat claim that used to stand here was wrong. `catboost_options.cpp:806`
does set `Ordered` for every non-multiclass GPU loss -- and then
`UpdateBoostingTypeOption` (`private/libs/options/defaults_helper.h:33-42`)
flips it straight back:

    if (boostingTypeOption.NotSet() &&
        (learnSampleCount >= 50000 || IterationCount < 500) && ...)
        boostingTypeOption = EBoostingType::Plain;

called from `SetDataDependentDefaults` (`options_helper.cpp:416`), which
`train_model.cpp:1128` runs BEFORE the CPU/GPU trainer is chosen, so it
applies to their GPU arm. **At every size this repository benchmarks --
800k synthetic, covtype's 464k -- CatBoost's own default IS Plain, which
is our arm.** Ordered is their default only for a pool under 50,000 rows
run for at least 500 iterations.

That does not make the port complete: `Ordered` is still unreachable here,
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
The knob is on the Python surface (`GradientBoosting(nan_mode=...)`,
`NAN_MODES` in `python/mojolearn/ensemble.py`).

`Depthwise` and `Lossguide` grow policies and their non-symmetric trees were
on this list until 2026-08-23 and are not any more: `grow_policy`,
`max_leaves` and `min_data_in_leaf` are on `train()` and on
`GradientBoosting` (DEVIATION 259; `DEPTHWISE.md`, `LOSSGUIDE.md`), with
CatBoost's own refusals beside them -- no non-symmetric trainer for `Lq` or
`MultiClass` (`pointwise_non_symmetric.cpp:7-29`), `max_leaves` pinned to
`1 << depth` off Lossguide (`catboost_options.cpp:993-1001`), the pointwise
searcher oblivious-only. Still absent from this port of their GPU learner:
the `Region` grow policy, snapshotting, multi-GPU, and custom objectives.

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
* `random_strength` -- a NO-OP on their GREEDY symmetric searcher, and only
  there: `beforeSplitCalcer` is copied from the calcer after `NextFeature`
  (`compute_scores.cu:85`), so both carry the same `GlobalSeed + FeatureId`,
  both draw the same normal, and `gain = score - scoreBefore` (`:134`)
  cancels it. It is NOT a no-op on their doc-parallel searcher, which is the
  one CatBoost uses for single-target symmetric trees: there the gain is
  `noisyScore - scoreBeforeSplit` against an unnoised host scalar carried
  from the previous level (`kernel/pointwise_scores.cu:396-402`), so the
  draw survives into the argmin. Both arms are ported and both behave that
  way here; see `pixi run check-random-strength`.
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
