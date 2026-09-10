# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gradient-boosted trees on the GPU, mirroring CatBoost: its three growth
policies (oblivious SymmetricTree, Depthwise, Lossguide), its losses.

**THE DEFAULTS ARE CatBoost's, NOT scikit-learn's**, and several of them
change results rather than just speed:

    learning_rate   CatBoost 0.03      sklearn 0.1
    n_estimators    CatBoost 1000      sklearn 100   (OURS IS 100, see below)
    max_depth       CatBoost 6         sklearn 3
    l2_leaf_reg     CatBoost 3.0       sklearn (none)
    border_count    CatBoost 128       sklearn (none)

**TWO OF OURS ARE NOT CatBoost's, and the table above used to hide it.**
`n_estimators` is 100 here and 1000 in CatBoost
(`boosting_options.cpp:13`). `learning_rate` is 0.03 here, which is
CatBoost's CONSTRUCTOR value (`boosting_options.cpp:10`) but not the
value a CatBoost user gets: with `learning_rate` unset CatBoost fits it
from the pool, `exp(A*log(n) + B)` scaled by the iteration count
(`libs/train_lib/options_helper.cpp:252-288`), which at 800k rows and
1000 iterations is about 0.097 and at 100 iterations about 0.38. That
retune is not ported. So these defaults equal a CatBoost user who
passed `learning_rate=0.03, iterations=100` explicitly -- they are not
"CatBoost's defaults", and any comparison run should pin both arms.

The DEFAULT tree shape is OBLIVIOUS (symmetric): every node at a level takes
the same split, which is CatBoost's default structure and not scikit-learn's.
A comparison against `GradientBoostingRegressor` at matched hyperparameters
is still comparing two different tree families. `grow_policy='Depthwise'`
and `'Lossguide'` grow CatBoost's NON-SYMMETRIC trees (on the surface since
2026-08-23, DEVIATION 259), and a Depthwise tree IS the level-wise binary
tree scikit-learn grows -- same shape, CatBoost's score and estimator.

X IS ROW-MAJOR HERE AND COLUMN-MAJOR INSIDE. `fit` materializes Fortran
order ONCE, straight from the caller's buffer (`_arrays.as_f32_colmajor`,
DEVIATION 1887). On a 800,000 x 100 C-order matrix that is a 320 MB copy,
and it is reported rather than hidden -- pass `X` already float32 and in
Fortran order and it is a zero-copy borrow, a promise this docstring made
before the code kept it (the old path copied F-order input TWICE, once to
C order and once back).

THE LOSS PICKS THE LEAF ESTIMATOR, and that is CatBoost's decision, not this
wrapper's. `leaf_estimation_method=None` (the default) means "let the loss
decide", which reproduces `SetLeavesEstimationDefault`
(`catboost_options.cpp:273-360`): Newton at one iteration for RMSE, Newton at
ten for Logloss, Newton at TWENTY for Tweedie, Gradient for LogLinQuantile and
for Lq below q=2, and the EXACT weighted-quantile estimator for MAE, MAPE and
Quantile. Overriding it means overriding CatBoost.

PREDICTIONS ARE RAW SCORES FOR EVERY LOSS, exactly as CatBoost's `predict`
without a `prediction_type` is. A `Logloss` model's `predict` returns the
logit; a `Poisson` model's returns the log-rate. This is deliberate: the
number this library benchmarks is the number it returns. `predict_proba`
applies the link where one exists, and `predict_classes` takes its argmax --
separate methods rather than a `predict` that means different things for
different losses.

MULTICLASS IS MULTI-OUTPUT AND ITS LAST CLASS IS NOT STORED. A softmax over
`k` free approxes is over-parameterized -- adding a constant to all of them
changes nothing -- so CatBoost pins the last class's approx at zero and
carries `k - 1`. Therefore:

    MultiClass          predict          (n, n_classes - 1)  free approxes
                        predict_proba    (n, n_classes)      softmax
                        predict_classes  (n,)                class codes

`MultiClassOneVsAll` stores `n_classes` independent approxes. Its `predict`
returns `(n, n_classes)` raw scores and `predict_proba` applies CatBoost's
`MultiProbability` elementwise sigmoid, without renormalizing the columns.
Both multiclass losses use the symmetric GPU trainer registered in
`cuda/train_lib/multiclass.cpp:5-7` at the pinned upstream `54a8143a`.

Dropping MultiClass's last probability column and renormalising the rest
gives a different and wrong answer: the pinned class is a real class whose
approx happens to be zero. Labels are dense codes `0..k-1` for both, and the
class count is derived from them, as their `TClassificationTargetHelper`
derives it.
"""

import numpy as np

# GBDT HAS ITS OWN EXTENSION, built by `bindings/build_gbdt.sh`, for the
# reason `_mojolearn_estimators` has one: an independently changing binding
# should not be a merge point. Every parameter added to `GbdtFitParams` used
# to have to be unpacked in two files that could silently disagree about the
# order of a flat list, which is a wrong answer rather than a failure.
#
# It was COMMISSIONED for a different reason -- a supposed per-module cap on
# ahead-of-time Metal compilation, keyed on the entry file's basename -- and
# that reason turned out not to exist. See archive/reference/PORTING.md 70: the kernels were
# being lost to `MACOSX_DEPLOYMENT_TARGET` in the environment plus a compiler
# cache that does not key on it, and the basename never mattered.
from . import _backend, _mojolearn_gbdt, _serialize
from ._mode import NumericModeMixin
from ._arrays import _addr, _addr_ro, as_f32_colmajor

#: The npz model-file format tag `save` writes and `load` requires.
_MODEL_FORMAT = "mojolearn-gbdt-1"

#: CatBoost's `ELossFunction` spellings that this port trains. The list is
#: the reachable set of their GPU pointwise target
#: (`pointwise_target_impl.h:259-299`), minus the ones whose leaf estimator
#: or kernel family is not ported.
LOSSES = (
    "MultiClass",
    "MultiClassOneVsAll",
    "RMSE",
    "Logloss",
    "CrossEntropy",
    "Quantile",
    "MAE",
    "LogLinQuantile",
    "MAPE",
    "Poisson",
    "Lq",
    "Expectile",
    "Tweedie",
    "Huber",
)

#: Their GPU target keeps numClasses - 1 planes for MultiClass and
#: numClasses for OneVsAll (`multiclass_targets.h:129-134`, 54a8143a).
MULTI_OUTPUT_LOSSES = ("MultiClass", "MultiClassOneVsAll")

#: `gbdt_predict_multi`'s transform, mirroring their `EPredictionType`
#: (`libs/model/eval_processing.h:186-226`).
_PREDICT_RAW = 0
_PREDICT_SOFTMAX = 1   # their `Probability`,      MultiClass
_PREDICT_SIGMOID = 2   # their `MultiProbability`, MultiClassOneVsAll

#: Losses whose parameter CatBoost makes MANDATORY. Passing the loss without
#: it raises here rather than in Mojo, so the message names the Python
#: keyword the caller has to add.
_REQUIRED_PARAM = {
    "Lq": ("q", "loss_q"),
    "Huber": ("delta", "loss_delta"),
    "Tweedie": ("variance_power", "loss_variance_power"),
    "Expectile": ("alpha", "loss_alpha"),
}

#: `EBootstrapType` spellings reachable from their GPU oblivious searcher.
#: MVS is absent because their own searcher asserts it away
#: (`weak_objective_impl.h:30`).
BOOTSTRAP_TYPES = ("Bayesian", "Bernoulli", "Poisson", "No")

#: `ELeavesEstimation`, for the override. `None` means let the loss decide.
LEAF_ESTIMATION_GRADIENT = 0
LEAF_ESTIMATION_NEWTON = 1
LEAF_ESTIMATION_EXACT = 2
LEAF_ESTIMATION_SIMPLE = 3

_LEAF_ESTIMATION_NAMES = {
    "Gradient": LEAF_ESTIMATION_GRADIENT,
    "Newton": LEAF_ESTIMATION_NEWTON,
    "Exact": LEAF_ESTIMATION_EXACT,
    "Simple": LEAF_ESTIMATION_SIMPLE,
}

#: `EScoreFunction` (`enums.h`), CatBoost's own codes. Cosine is their
#: shipped GPU default.
SCORE_FUNCTION_SOLAR_L2 = 0
SCORE_FUNCTION_COSINE = 1
SCORE_FUNCTION_NEWTON_L2 = 2
SCORE_FUNCTION_NEWTON_COSINE = 3
SCORE_FUNCTION_LOO_L2 = 4
SCORE_FUNCTION_SAT_L2 = 5
SCORE_FUNCTION_L2 = 6

#: The four score functions this port ACTUALLY COMPUTES. Cosine and L2
#: have a real calcer in the split kernel -- `TCosineScoreCalcer`
#: (`score_calcers.cuh:152-167`) and `TL2ScoreCalcer` (`:40-69`) -- and
#: the Newton spellings pair onto those SAME calcers, differing only in
#: which derivative the der launch puts in the histogram's weight plane:
#: `secondDerAsWeights = IsSecondOrderScoreFunction(scoreFunction)`
#: (`greedy_search_helper.cpp:286-296`) makes plane 0 `weight * der2`
#: instead of the raw weight (`pointwise_target_impl.h:193-201`), which is
#: CatBoost's own structure (`compute_scores.cu:201-219`). That flag IS
#: ported and gated per cell and per model by
#: `checks/second_der_weights_check.mojo`; for RMSE alone the Newton
#: spellings coincide with their pair bit for bit, because
#: `TRmseTarget::Der2` returns 1.0.
SCORE_FUNCTIONS = ("Cosine", "L2", "NewtonCosine", "NewtonL2")

_SCORE_FUNCTION_NAMES = {
    "Cosine": SCORE_FUNCTION_COSINE,
    "L2": SCORE_FUNCTION_L2,
    "NewtonCosine": SCORE_FUNCTION_NEWTON_COSINE,
    "NewtonL2": SCORE_FUNCTION_NEWTON_L2,
}

#: THE OTHER THREE ARE REFUSED BY NAME, AND THIS IS THE WHOLE REASON THE
#: OPTION COULD NOT SIMPLY BE FORWARDED.
#:
#: `score_function` was hard-coded to Cosine here for exactly as long as it
#: took someone to look at what the other values do on the way down. They do
#: not fail. They produce a DIFFERENT MODEL THAN THE ONE ASKED FOR, silently:
#:
#: SolarL2, LOOL2, SatL2 -- the greedy subsets searcher, which is the arm
#: `train` runs, dispatches on `score_function` with an `if L2 or NewtonL2
#: ... else Cosine` (`greedy_search_helper.mojo:3134-3162`). There is no arm
#: for these three, so they land in the `else` and fit a COSINE model. The
#: split kernel's own `comptime assert` would have caught it
#: (`compute_scores.mojo:205-212`) and never sees them, because the host
#: chose the Cosine instantiation before the kernel was reached. Their
#: calcers DO exist on the pointwise searcher
#: (`pointwise_scores.mojo:1569-1616`), which is why "it is implemented"
#: and "it is honored" are different sentences here.
#:
#: NewtonCosine and NewtonL2 used to sit in this dict for a different
#: reason -- `secondDerAsWeights` was unported, so each silently fit its
#: non-Newton twin -- and left it when that flag landed, gated by
#: `checks/second_der_weights_check.mojo`.
_UNPORTED_SCORE_FUNCTIONS = {
    "SolarL2": (
        "the greedy searcher this port runs has no SolarL2 arm and falls "
        "through to Cosine (greedy_search_helper.mojo:3134-3162), so the "
        "fit would silently be a Cosine fit. Its calcer exists only on the "
        "pointwise searcher (pointwise_scores.mojo:1569)"
    ),
    "LOOL2": (
        "the greedy searcher this port runs has no LOOL2 arm and falls "
        "through to Cosine (greedy_search_helper.mojo:3134-3162), so the "
        "fit would silently be a Cosine fit"
    ),
    "SatL2": (
        "the greedy searcher this port runs has no SatL2 arm and falls "
        "through to Cosine (greedy_search_helper.mojo:3134-3162), so the "
        "fit would silently be a Cosine fit"
    ),
}

#: their `EGrowPolicy` spellings (`oblivious_tree_options.cpp:23`), in
#: their enum order -- the binding carries the ORDINAL. `Region`, their
#: fourth, is absent: no lane ports it (`greedy_search_helper.cpp:325-350`).
#: SymmetricTree is `TObliviousTreeModel`; Depthwise and Lossguide are
#: `TNonSymmetricTree`, grown by `TGreedySubsetsSearcher<TNonSymmetricTree>`
#: (`structure_searcher_template.h:66`) and applied by their
#: `TAddModelDocParallel<TNonSymmetricTree>`. DEVIATION 259.
GROW_POLICIES = ("SymmetricTree", "Depthwise", "Lossguide")

_GROW_POLICY_CODES = {
    "SymmetricTree": 0,
    "Depthwise": 1,
    "Lossguide": 2,
}

#: THE LOSSES CatBoost's GPU REGISTERS A NON-SYMMETRIC TRAINER FOR --
#: `cuda/train_lib/pointwise_non_symmetric.cpp:7-29`, one
#: `TGpuTrainer<TPointwiseTargetsImpl, TNonSymmetricTree>` registration
#: per (loss, policy) pair, eleven losses x {Lossguide, Depthwise}. Any
#: other pair fails their `TGpuTrainerFactory::Has` with "Error:
#: optimization scheme is not supported for GPU learning
#: Loss=...;OptimizationScheme=..." (`train.cpp:279-280`). Two of this
#: surface's losses are NOT on their list and are refused here with that
#: message: `Lq` (unregistered) and `MultiClass` (`multiclass.cpp:5-14`
#: registers the multiclass targets at the default SymmetricTree policy
#: only).
_NON_SYMMETRIC_LOSSES = frozenset((
    "Poisson", "MAPE", "MAE", "Quantile", "LogLinQuantile", "RMSE",
    "Logloss", "CrossEntropy", "Expectile", "Tweedie", "Huber",
))

#: their `ENanMode` spellings (`data_processing_options.cpp:26`). 'Forbidden'
#: RAISES on a NaN rather than binning it, which is CatBoost's behavior and
#: not a validation nicety of ours.
NAN_MODES = ("Min", "Max", "Forbidden")

_UNSET = -1.0

#: their `EOverfittingDetectorType` spellings. `None` here means the
#: option was not given, and that is NOT the same as `od_type="None"`:
#: unset lets `TOverfittingDetectorOptions::Load`
#: (`overfitting_detector_options.cpp:24-32`) pick the type from whichever
#: of `od_pvalue` / `od_wait` was given, while "None" turns the detector
#: off outright. Wilcoxon is theirs and is not ported.
OD_TYPES = ("None", "IncToDec", "Iter")


def _tri(v):
    """A CatBoost `TOption` tri-state: `None` is `NotSet()`.

    Used for `use_best_model`, whose default is DATA-DEPENDENT on their
    side (`options_helper.cpp:106-108` turns it on when there is a test
    set with a non-constant target), so `False` and "unset" have to stay
    distinguishable across the boundary.
    """
    if v is None:
        return -1
    return 1 if v else 0


class GradientBoosting(NumericModeMixin):
    """Gradient-boosted trees, mirroring CatBoost's GPU learner: its three
    growth policies (`grow_policy`), its losses, its leaf estimators.

    Parameters
    ----------
    loss : str, default 'RMSE'
        One of `LOSSES`, CatBoost's own spellings. Four of them require a
        parameter: `Lq` needs `loss_q`, `Huber` needs `loss_delta`,
        `Tweedie` needs `loss_variance_power`, `Expectile` needs
        `loss_alpha`.
    n_estimators : int, default 100
        CatBoost's `iterations`.
    max_depth : int, default 6
        CatBoost's `depth`. Under `grow_policy='SymmetricTree'` the tree
        is oblivious, so this is exactly `2 ** max_depth` leaves; under
        Depthwise it is a bound on the level count and the leaf count is
        ragged; under Lossguide it bounds any one leaf's depth and
        `max_leaves` is what stops the tree.
    grow_policy : {'SymmetricTree', 'Depthwise', 'Lossguide'}, \
            default 'SymmetricTree'
        CatBoost's `grow_policy` (`oblivious_tree_options.cpp:23`).
        SymmetricTree is the oblivious tree; Depthwise splits every
        improving leaf of a level on ITS OWN best split; Lossguide splits
        ONE leaf per step, the one with the best score, until
        `max_leaves`. Both non-symmetric policies are grown by the same
        `TGreedySubsetsSearcher<TNonSymmetricTree>` CatBoost's GPU grows
        them with (`pointwise_non_symmetric.cpp`), then take their
        estimation arm -- bins off the model, the leaf estimator the loss
        picks, `AddBinModelValues` -- and the saved model carries the
        non-symmetric shape. **Refused by name, each where CatBoost
        refuses it**: a loss with no non-symmetric GPU trainer (`Lq`,
        `MultiClass` -- `pointwise_non_symmetric.cpp:7-29` lists the
        eleven that have one; `train.cpp:279` is the error), and
        `use_pointwise_searcher=True` (that is the doc-parallel OBLIVIOUS
        searcher). Their `Region` policy is not ported and is refused by
        name. DEVIATION 259.
    feature_fraction : float, default 1.0
        Per-tree numeric feature subsampling in (0, 1]. A deterministic
        subset of nonconstant features is chosen for each tree and shared
        by its split searches. The default preserves existing model bits
        and ABI layout. Values below one refuse categorical/one-hot/CTR paths.
    min_child_hessian : float or None, default None
        Minimum weighted Hessian sum in each candidate child, before split
        selection. Requires Depthwise/Lossguide, NewtonL2/NewtonCosine and
        RMSE/Logloss/CrossEntropy. Includes training and bootstrap weights;
        equality is allowed. None preserves existing growth.
    min_split_gain : float or None, default None
        Optional strict lower bound on a split's improvement in the selected
        score function, for Depthwise and Lossguide only. None preserves
        CatBoost's existing growth behavior, including Lossguide's ability
        to split without positive gain. Zero requires positive improvement;
        equality is rejected. Units depend on score_function, feature weights
        and score noise; this is not numerically interchangeable with XGBoost
        gamma or LightGBM min_gain_to_split. Enabled values must be finite
        and nonnegative.
    max_leaves : int, optional
        CatBoost's `max_leaves`, the Lossguide leaf budget. Their default
        is 31 (`oblivious_tree_options.cpp:24`) and their cap is 65536
        (`:130-133`). **Read under Lossguide only**: for every other policy
        CatBoost pins it to `2 ** max_depth` and refuses a different value
        with "max_leaves option works only with lossguide tree growing"
        (`catboost_options.cpp:993-1001`), and so does this.
    min_data_in_leaf : int, default 1
        CatBoost's `min_data_in_leaf` (`oblivious_tree_options.cpp:25`), a
        leaf with `size <= min_data_in_leaf` is terminal
        (`greedy_search_helper.cpp:693`, their `<=`). **Live under
        Depthwise and Lossguide only**: their `IsTerminalLeaf` guards the
        size test with `Policy != SymmetricTree` (`:685`) and DISCARDS the
        value on oblivious trees; this refuses any value but 1 there
        rather than accepting what it would drop.
    learning_rate : float, default 0.03
        CatBoost's default (`boosting_options.cpp:10`), not scikit-learn's
        0.1.
    l2_leaf_reg : float, default 3.0
    border_count : int, default 128
        Quantization bins per numeric feature.
    random_state : int, default 0
    loss_alpha : float, optional
        Quantile level for `Quantile` and `LogLinQuantile` (default 0.5),
        and MANDATORY for `Expectile`. Ignored by `MAE`, whose alpha
        CatBoost fixes at 0.5 (`pointwise_target_impl.h:272-275`).
    loss_q, loss_delta, loss_variance_power : float, optional
        `Lq`'s q, `Huber`'s delta, `Tweedie`'s variance_power.
    loss_border : float, optional
        `Logloss`'s target threshold, default 0.5.
    leaf_estimation_method : {'Newton','Gradient','Exact','Simple'}, optional
        None (default) means the LOSS decides, per CatBoost. 'Newton' is
        refused for Quantile, MAE, LogLinQuantile, MAPE and Lq with q < 2,
        with CatBoost's own message (`catboost_options.cpp:588-601`;
        their second derivative is zero there).
    leaf_estimation_iterations : int, optional
        None (default) means the loss decides.
    bootstrap_type : {'Bayesian','Bernoulli','Poisson','No'}, optional
        None means no row sampling. 'Bernoulli' is the familiar `subsample`
        knob; 'Bayesian' is CatBoost's GPU default and uses
        `bagging_temperature` instead.
    bagging_temperature : float, default 1.0
        Bayesian only. CatBoost refuses `subsample` beside it and so does
        this.
    subsample : float, optional
        Bernoulli and Poisson only. Default 0.66
        (`bootstrap_options.h:15`).
    cat_features : sequence of int, optional
        Column indices holding DENSE CATEGORY CODES 0..k-1. CatBoost's own
        dispatch decides what happens to each
        (`binarizations_manager.cpp:106-115`): one-hot when the cardinality
        is small enough, target statistics (CTRs) otherwise.

    od_type : {'None', 'IncToDec', 'Iter'}, optional
        The overfitting detector. LEAVING IT UNSET IS NOT THE SAME AS
        'None': their `Load` (`overfitting_detector_options.cpp:24-32`)
        picks the type from whichever of `od_pvalue` / `od_wait` was
        given -- a wait alone means 'Iter', a p-value alone means
        'IncToDec', neither means 'None'. Requires `eval_set`; stopping on
        the learn loss would stop on a curve that falls by construction,
        so it raises instead.
    od_pvalue : float, optional
        `stop_pvalue`, IncToDec's threshold. Their default is 0, which
        makes the detector INACTIVE (`IsActive()` is `Threshold > 0`).
    od_wait : int, optional
        `wait_iterations`, default 20. Iterations without a new best
        before stopping.
    use_best_model : bool, optional
        Truncate the returned ensemble to the best held-out iteration,
        their `ShrinkToBestIteration`. UNSET IS NOT FALSE: with an
        `eval_set` whose target is not constant it defaults to True, which
        is their own data-dependent default (`options_helper.cpp:106-108`).
        True without an `eval_set` raises -- they warn and carry on, and a
        warning on a returned model is invisible from here.
    best_model_min_trees : int, default 1
        The shrink may not cut below this many trees
        (`output_file_options.cpp:77`,
        `boosting_progress_tracker.cpp:162`).

    score_function : {'Cosine', 'L2', 'NewtonCosine', 'NewtonL2'}, optional
        The split score. None (default) is CatBoost's OWN default FOR THE
        POLICY: Cosine for SymmetricTree and Depthwise
        (`oblivious_tree_options.cpp:20`), and NewtonL2 for Lossguide,
        which their GPU option resolver sets when the option is unset
        (`catboost_options.cpp:980-991`; L2 there for MultiClass, which
        Lossguide refuses anyway). Cosine is what the shipped symmetric
        oracle certifies, and `TCosineScoreCalcer` is the
        only one of their five calcers that carries the `random_strength`
        noise term (`score_calcers.cuh:152-167`); `TL2ScoreCalcer` (`:40-69`)
        has none. The Newton spellings run the SAME calcers with
        `weight * der2` in the histogram's weight plane instead of the raw
        weight -- their `secondDerAsWeights`
        (`greedy_search_helper.cpp:286-296`) -- so for RMSE, whose Der2 is
        1.0, each is bit-identical to its twin. CatBoost's other three
        spellings -- SolarL2, LOOL2, SatL2 -- ARE REFUSED BY NAME rather
        than accepted, because on this port each silently fits a different
        model than the one asked for. See `_UNPORTED_SCORE_FUNCTIONS` for
        which, and why.
    nan_mode : {'Min', 'Max', 'Forbidden'}, default 'Min'
        Where a NaN sorts against the borders
        (`data_processing_options.cpp:26`). 'Min' puts it below every
        border, 'Max' above; 'Forbidden' RAISES on a NaN instead of binning
        it, which is CatBoost's behavior.
    random_strength : float, default 0.0
        CatBoost's `random_strength` (`oblivious_tree_options.cpp:17`).
        **THIS DEFAULT IS NOT CatBoost's, which is 1.0**, and the difference
        is deliberate rather than an oversight: on the greedy searcher this
        wrapper runs by default, CatBoost's own noise cancels in the gain
        (`compute_scores.cu:84-134`), so a non-zero value there changes only
        float rounding. It is a live knob on `use_pointwise_searcher=True`,
        where the noise is drawn before the bootstrap
        (`oblivious_tree_doc_parallel_structure_searcher.cpp:200-218`).
        Refused above 0.0 with `score_function='L2'` or `'NewtonL2'`,
        because the L2 calcer both run has no noise term and CatBoost
        itself would discard it.
    use_pointwise_searcher : bool, default False
        Grow with `TDocParallelObliviousTreeSearcher`, CatBoost's
        single-target symmetric learner, instead of the greedy subsets
        searcher. Both are theirs and both are reachable on their GPU. The
        pointwise arm returns the STRUCTURE ONLY, so it always runs the leaf
        estimator where the greedy arm can reuse the leaf it grew.
    border_build_max_samples : int, default 200000
        How many rows the BORDER SEARCH subsamples
        (`data_processing_options.cpp:37`). It does not subsample training;
        every row is still fitted. 0 means use every row for the borders
        too.
    class_weights : sequence of float, optional
        One weight per class slot, MULTIPLIED into the row weight rather
        than substituted for it (`target/data_providers.cpp:168`), so it
        composes with `sample_weight`. The length must match the label set
        or the fit raises. Refused with `loss='RMSE'`, which CatBoost also
        refuses.
    permutation_count : int, optional
        CatBoost's `permutation_count` (`boosting_options.cpp:16`). None
        (default) lets `UpdateGpuSpecificDefaults` resolve it. **ONLY THE
        CATEGORICAL PATH READS IT** -- with no `cat_features` it is inert,
        and it is refused there rather than accepted and ignored.
    ctr_estimation_permutation_id : int, optional
        Which permutation estimates the CTRs
        (`doc_parallel_boosting.h:101-103`). None (default) means
        `permutation_count - 1`, their estimation permutation. **ONLY THE
        CATEGORICAL PATH READS IT**, and it is refused otherwise.

    Attributes
    ----------
    model_ : str
        The fitted ensemble, as the text format `check-model-io` gates
        bit-for-bit. It is a plain string: save it, ship it, hand it to
        another process. Floats in it carry their hex bits beside the
        decimal, because `String(Float32)` on this toolchain is one ULP
        wrong for 0.46% of values and a decimal-only round trip would
        change predictions.
    loss_curve_ : ndarray
        The training loss after each iteration. It is CatBoost's
        `functionValue` negated and divided by the row count, so it FALLS.
    test_loss_curve_ : ndarray or None
        The HELD-OUT loss after each iteration, `None` without an
        `eval_set`. This is the curve the detector reads and the one worth
        plotting: `loss_curve_` falls almost by construction.
    best_iteration_ : int
        The index of the lowest `test_loss_curve_` entry, or of the lowest
        learn loss with no eval set. It is the ERROR tracker's best
        (`error_tracker.h:73-75`), so with `best_model_min_trees` above it
        the fitted model holds MORE trees than `best_iteration_ + 1` and
        both numbers are right.
    stopped_early_ : bool
        Whether the detector fired before `n_estimators` was reached.
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_gbdt"

    def __init__(
        self,
        loss="RMSE",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        border_count=128,
        random_state=0,
        loss_alpha=None,
        loss_q=None,
        loss_delta=None,
        loss_variance_power=None,
        loss_border=None,
        leaf_estimation_method=None,
        leaf_estimation_iterations=None,
        bootstrap_type=None,
        bagging_temperature=1.0,
        subsample=None,
        cat_features=None,
        one_hot_features=None,
        od_type=None,
        od_pvalue=None,
        od_wait=None,
        use_best_model=None,
        best_model_min_trees=1,
        score_function=None,
        nan_mode="Min",
        random_strength=0.0,
        use_pointwise_searcher=False,
        boost_from_average=None,
        border_build_max_samples=200000,
        class_weights=None,
        permutation_count=None,
        ctr_estimation_permutation_id=None,
        grow_policy="SymmetricTree",
        max_leaves=None,
        min_data_in_leaf=1,
        min_split_gain=None,
        min_child_hessian=None,
        feature_fraction=1.0,
    ):
        if loss not in LOSSES:
            raise ValueError(
                f"mojolearn: loss must be one of {LOSSES}, got {loss!r}"
            )
        if loss in _REQUIRED_PARAM:
            cb_name, py_name = _REQUIRED_PARAM[loss]
            if locals()[py_name] is None:
                raise ValueError(
                    f"mojolearn: {loss} requires {py_name}= "
                    f"(CatBoost's {cb_name!r}, which it makes mandatory)"
                )
        if bootstrap_type is not None and bootstrap_type not in BOOTSTRAP_TYPES:
            raise ValueError(
                f"mojolearn: bootstrap_type must be one of "
                f"{BOOTSTRAP_TYPES}, got {bootstrap_type!r}"
            )
        if bootstrap_type == "Bayesian" and subsample is not None:
            # CatBoost's own validator (`catboost_options.cpp:795`)
            raise ValueError(
                "mojolearn: bootstrap_type='Bayesian' does not support "
                "subsample; it takes bagging_temperature"
            )
        if leaf_estimation_method is not None:
            if leaf_estimation_method not in _LEAF_ESTIMATION_NAMES:
                raise ValueError(
                    f"mojolearn: leaf_estimation_method must be one of "
                    f"{tuple(_LEAF_ESTIMATION_NAMES)}, got "
                    f"{leaf_estimation_method!r}"
                )

        # ---- the grow policy, and what CatBoost refuses beside it ----
        # (DEVIATION 259; every refusal cites the line of theirs it mirrors)
        if grow_policy == "Region":
            raise NotImplementedError(
                "mojolearn: grow_policy='Region' is EGrowPolicy::Region, "
                "which no lane ports (greedy_search_helper.cpp:325-350); "
                f"reachable values are {GROW_POLICIES}"
            )
        if grow_policy not in _GROW_POLICY_CODES:
            raise ValueError(
                f"mojolearn: grow_policy must be one of {GROW_POLICIES}, "
                f"got {grow_policy!r}"
            )
        valid_fraction_type = (
            not isinstance(feature_fraction, (bool, np.bool_))
            and isinstance(feature_fraction, (int, float, np.integer, np.floating))
        )
        try:
            parsed_fraction = float(feature_fraction) if valid_fraction_type else float("nan")
        except (ValueError, TypeError, OverflowError):
            parsed_fraction = float("nan")
        if not np.isfinite(parsed_fraction) or not 0.0 < parsed_fraction <= 1.0:
            raise ValueError("mojolearn: feature_fraction must be finite and in (0, 1]")
        feature_fraction = parsed_fraction
        if feature_fraction < 1.0 and any(
            values is not None and np.asarray(values).size > 0
            for values in (cat_features, one_hot_features)
        ):
            raise NotImplementedError(
                "mojolearn: feature_fraction < 1 requires numeric features; "
                "categorical, one-hot and CTR features are unsupported"
            )
        if min_split_gain is not None:
            valid_type = (not isinstance(min_split_gain, (bool, np.bool_))
                          and isinstance(min_split_gain, (int, float, np.integer, np.floating)))
            try:
                parsed_gain = float(min_split_gain) if valid_type else float("nan")
            except (ValueError, TypeError, OverflowError):
                parsed_gain = float("nan")
            if not np.isfinite(parsed_gain) or parsed_gain < 0:
                raise ValueError("mojolearn: min_split_gain must be finite and nonnegative or None")
            if grow_policy == "SymmetricTree":
                raise ValueError("mojolearn: min_split_gain is only supported for Depthwise and Lossguide")
            min_split_gain = parsed_gain
        non_symmetric = grow_policy != "SymmetricTree"
        if non_symmetric and loss not in _NON_SYMMETRIC_LOSSES:
            # their `TGpuTrainerFactory::Has` failing: no
            # (loss, Depthwise/Lossguide) registration for this loss
            raise NotImplementedError(
                "mojolearn: Error: optimization scheme is not supported for "
                f"GPU learning Loss={loss};OptimizationScheme={grow_policy} "
                "-- CatBoost's GPU registers a non-symmetric trainer for "
                "exactly eleven losses (pointwise_non_symmetric.cpp:7-29; "
                f"train.cpp:279 is the refusal): {sorted(_NON_SYMMETRIC_LOSSES)}"
            )
        if non_symmetric and use_pointwise_searcher:
            raise ValueError(
                "mojolearn: use_pointwise_searcher=True is "
                "TDocParallelObliviousTreeSearcher, an OBLIVIOUS searcher; "
                f"grow_policy={grow_policy!r} is grown by "
                "TGreedySubsetsSearcher<TNonSymmetricTree> only "
                "(pointwise_non_symmetric.cpp:5-29)"
            )
        if max_leaves is not None:
            if int(max_leaves) < 2:
                raise ValueError(
                    f"mojolearn: max_leaves must be at least 2, got "
                    f"{max_leaves}"
                )
            if grow_policy != "Lossguide" and int(max_leaves) != (
                    1 << int(max_depth)):
                # `CB_ENSURE(MaxLeaves == maxLeaves, "max_leaves option
                # works only with lossguide tree growing")`
                # (`catboost_options.cpp:993-1001`)
                raise ValueError(
                    "mojolearn: max_leaves option works only with lossguide "
                    f"tree growing (catboost_options.cpp:998); under "
                    f"grow_policy={grow_policy!r} CatBoost pins it to "
                    f"2 ** max_depth == {1 << int(max_depth)}, got "
                    f"{max_leaves}"
                )
            if grow_policy == "Lossguide" and int(max_leaves) > 65536:
                # `oblivious_tree_options.cpp:130-133`
                raise ValueError(
                    "mojolearn: Maximum leaves count for Lossguide grow "
                    f"policy is 65536, got {max_leaves}"
                )
        if int(min_data_in_leaf) < 0:
            raise ValueError(
                f"mojolearn: min_data_in_leaf must be >= 0, got "
                f"{min_data_in_leaf}"
            )
        if not non_symmetric and int(min_data_in_leaf) != 1:
            # CatBoost ACCEPTS AND DISCARDS it here (`IsTerminalLeaf` tests
            # the size only when `Policy != SymmetricTree`,
            # greedy_search_helper.cpp:685); this port refuses what it
            # would drop
            raise ValueError(
                f"mojolearn: min_data_in_leaf={min_data_in_leaf} does "
                "nothing under grow_policy='SymmetricTree' -- CatBoost's "
                "IsTerminalLeaf tests the leaf size only for Depthwise and "
                "Lossguide (greedy_search_helper.cpp:685) and discards the "
                "value on oblivious trees. Use a non-symmetric grow_policy "
                "or leave it at 1."
            )
        # THE SCORE FUNCTION'S DEFAULT DEPENDS ON THE POLICY, as theirs
        # does: `SetNotSpecifiedOptionsToDefaults` leaves the constructed
        # Cosine (`oblivious_tree_options.cpp:20`) for SymmetricTree and
        # Depthwise and sets NewtonL2 for Lossguide on GPU
        # (`catboost_options.cpp:980-991`; L2 for MultiClass / OneVsAll /
        # RMSEWithUncertainty, none of which reaches Lossguide here)
        if score_function is None:
            score_function = "NewtonL2" if grow_policy == "Lossguide" else "Cosine"

        # AN OPTION ACCEPTED AND IGNORED IS WORSE THAN ONE ABSENT. Each
        # refusal below names the value the caller passed and the file:line
        # that makes it a lie, because "not supported" without a reason is
        # indistinguishable from "not implemented yet" and gets retried.
        if score_function in _UNPORTED_SCORE_FUNCTIONS:
            raise NotImplementedError(
                f"mojolearn: score_function={score_function!r} is not "
                f"honored here -- {_UNPORTED_SCORE_FUNCTIONS[score_function]}"
                f". Reachable values are {SCORE_FUNCTIONS}."
            )
        if score_function not in _SCORE_FUNCTION_NAMES:
            raise ValueError(
                f"mojolearn: score_function must be one of "
                f"{SCORE_FUNCTIONS}, got {score_function!r}"
            )
        if min_child_hessian is not None:
            valid_type = (not isinstance(min_child_hessian, (bool, np.bool_))
                          and isinstance(min_child_hessian, (int, float, np.integer, np.floating)))
            try:
                parsed_hessian = float(min_child_hessian) if valid_type else float("nan")
            except (ValueError, TypeError, OverflowError):
                parsed_hessian = float("nan")
            if not np.isfinite(parsed_hessian) or not 0 <= parsed_hessian <= float(np.finfo(np.float32).max):
                raise ValueError("mojolearn: min_child_hessian must be finite nonnegative and <= Float32.MAX_FINITE or None")
            if grow_policy == "SymmetricTree":
                raise ValueError("mojolearn: min_child_hessian requires Depthwise or Lossguide")
            if score_function not in ("NewtonL2", "NewtonCosine"):
                raise ValueError("mojolearn: min_child_hessian requires NewtonL2 or NewtonCosine")
            if loss not in ("RMSE", "Logloss", "CrossEntropy"):
                raise ValueError("mojolearn: min_child_hessian supports RMSE, Logloss and CrossEntropy only")
            min_child_hessian = parsed_hessian
        if nan_mode not in NAN_MODES:
            raise ValueError(
                f"mojolearn: nan_mode must be one of {NAN_MODES}, got "
                f"{nan_mode!r}"
            )
        if random_strength < 0.0:
            raise ValueError(
                f"mojolearn: random_strength must be >= 0, got "
                f"{random_strength}"
            )
        if random_strength != 0.0 and score_function in ("L2", "NewtonL2"):
            # CatBoost accepts this pair and discards the value: only
            # `TCosineScoreCalcer` has a noise term
            # (`score_calcers.cuh:159-167`), `TL2ScoreCalcer` (`:40-69`)
            # has none, and NewtonL2 runs that same L2 calcer. Copying
            # that silence would leave a knob that reads as live and is
            # not, so this port refuses where CatBoost does not. The same
            # refusal is in `CatBoostOptions.validate`.
            raise ValueError(
                f"mojolearn: random_strength={random_strength} does nothing "
                f"under score_function={score_function!r} -- only "
                "TCosineScoreCalcer carries the noise term "
                "(score_calcers.cuh:159-167). Use score_function='Cosine' "
                "or 'NewtonCosine', or random_strength=0.0. (Under "
                "grow_policy='Lossguide' an unset score_function resolves "
                "to NewtonL2, CatBoost's own GPU default there.)"
            )
        if border_build_max_samples < 0:
            raise ValueError(
                f"mojolearn: border_build_max_samples must be >= 0 (0 means "
                f"every row), got {border_build_max_samples}"
            )
        if class_weights is not None:
            if len(class_weights) == 0:
                raise ValueError(
                    "mojolearn: class_weights is empty; pass None for none"
                )
            if any(not np.isfinite(float(w)) or float(w) < 0 for w in class_weights):
                raise ValueError(
                    "mojolearn: class_weights must have finite nonnegative entries"
                )
            if loss == "RMSE":
                # `train` raises on this pair too; caught here so the
                # message names the Python keyword.
                raise ValueError(
                    "mojolearn: class_weights is not accepted with "
                    "loss='RMSE', which has no class structure to weight"
                )
        # THE TWO CTR PERMUTATION KNOBS ARE INERT WITHOUT CATEGORICALS.
        # Nothing outside the CTR path reads either one, so accepting them
        # on an all-numeric fit would be accepting an option that cannot do
        # anything. `cat_features` is checked rather than `one_hot_features`
        # because a one-hot column never grows CTRs either.
        for _name, _val in (
            ("permutation_count", permutation_count),
            ("ctr_estimation_permutation_id", ctr_estimation_permutation_id),
        ):
            if _val is None:
                continue
            if int(_val) < 0:
                raise ValueError(
                    f"mojolearn: {_name} must be >= 0 when given, got {_val}"
                )
            if not cat_features:
                raise ValueError(
                    f"mojolearn: {_name} is read only by the categorical "
                    "path (doc_parallel_dataset_builder.cpp:190-262); with "
                    "no cat_features it would be accepted and ignored. Pass "
                    "cat_features= or leave it unset."
                )

        self.loss = loss
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.border_count = border_count
        self.random_state = random_state
        self.loss_alpha = loss_alpha
        self.loss_q = loss_q
        self.loss_delta = loss_delta
        self.loss_variance_power = loss_variance_power
        self.loss_border = loss_border
        self.leaf_estimation_method = leaf_estimation_method
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.bootstrap_type = bootstrap_type
        self.bagging_temperature = bagging_temperature
        self.subsample = subsample
        self.cat_features = cat_features
        self.one_hot_features = one_hot_features
        self.od_type = od_type
        self.od_pvalue = od_pvalue
        self.od_wait = od_wait
        self.use_best_model = use_best_model
        self.best_model_min_trees = int(best_model_min_trees)
        self.score_function = score_function
        self.nan_mode = nan_mode
        self.random_strength = float(random_strength)
        self.use_pointwise_searcher = bool(use_pointwise_searcher)
        # their tri-state (`AdjustBoostFromAverageDefaultValue`): None is
        # unset and resolves inside `train` -- auto-True for RMSE (their
        # rule; NOT for Logloss, which is not on their list), False
        # otherwise; True is explicit and refused by name for losses
        # whose CalcOptimumConstApprox arm is not ported.
        self.boost_from_average = boost_from_average
        self.border_build_max_samples = int(border_build_max_samples)
        self.class_weights = (
            None if class_weights is None
            else [float(w) for w in class_weights]
        )
        self.permutation_count = permutation_count
        self.ctr_estimation_permutation_id = ctr_estimation_permutation_id
        self.grow_policy = grow_policy
        self.max_leaves = None if max_leaves is None else int(max_leaves)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.min_split_gain = None if min_split_gain is None else float(min_split_gain)
        self.min_child_hessian = min_child_hessian
        self.feature_fraction = feature_fraction

        self.model_ = None
        self.loss_curve_ = None
        self.test_loss_curve_ = None
        self.best_iteration_ = None
        self.stopped_early_ = None
        self.n_features_in_ = None
        #: 1 for every single-output loss; `n_classes - 1` for MultiClass,
        #: because the last class's approx is pinned at zero and is not
        #: stored. Read from the fitted model, never assumed.
        self.approx_dim_ = None
        self.n_classes_ = None

    # -- the parameter list, in the ONE order both sides name --------------
    #
    # `bindings/_mojolearn_gbdt.mojo:gbdt_fit_binding` writes this same order in
    # the same words. A silent reordering here is a wrong answer, not a
    # failure, which is why it is spelled out in both places.
    #
    # SLOTS 0..34 ARE FIXED AND SLOT 34 IS A COUNT: everything after it is
    # the class-weight tail, and the binding checks the length against it
    # rather than trusting it. Optional ordered tails AFTER the counted
    # weights are min_split_gain, min_child_hessian, then feature_fraction.
    # Trailing defaults are omitted, preserving all existing caller layouts.
    def _params(self, n_rows, n_features, n_flags, n_weights=0,
                n_eval_rows=0):
        def f(v):
            return _UNSET if v is None else float(v)

        cw = self.class_weights or ()
        method = self.leaf_estimation_method
        method_code = (
            -1 if method is None else _LEAF_ESTIMATION_NAMES[method]
        )
        iters = self.leaf_estimation_iterations
        tail = []
        fraction = getattr(self, "feature_fraction", 1.0)
        if fraction != 1.0:
            tail = [(-1.0 if self.min_split_gain is None else float(self.min_split_gain)),
                    (-1.0 if self.min_child_hessian is None else float(self.min_child_hessian)),
                    float(fraction)]
        elif self.min_child_hessian is not None:
            tail = [(-1.0 if self.min_split_gain is None else float(self.min_split_gain)),
                    float(self.min_child_hessian)]
        elif self.min_split_gain is not None:
            tail = [float(self.min_split_gain)]
        return [
            n_rows,                                     # 0
            n_features,                                 # 1
            int(n_weights),                             # 2  n_weights
            n_flags,                                    # 3
            int(self.border_count),                     # 4
            int(self.n_estimators),                     # 5
            int(self.max_depth),                        # 6
            float(self.learning_rate),                  # 7
            float(self.l2_leaf_reg),                    # 8
            int(self.random_state),                     # 9
            _SCORE_FUNCTION_NAMES[self.score_function],  # 10
            f(self.loss_alpha),                         # 11
            f(self.loss_q),                             # 12
            f(self.loss_delta),                         # 13
            f(self.loss_variance_power),                # 14
            f(self.loss_border),                        # 15
            -1 if iters is None else int(iters),        # 16
            method_code,                                # 17
            float(self.bagging_temperature),            # 18
            f(self.subsample),                          # 19
            int(n_eval_rows),                           # 20
            f(self.od_pvalue),                          # 21
            -1 if self.od_wait is None
            else int(self.od_wait),                     # 22
            _tri(self.use_best_model),                  # 23
            int(self.best_model_min_trees),             # 24
            float(self.random_strength),                # 25
            1 if self.use_pointwise_searcher else 0,    # 26
            int(self.border_build_max_samples),         # 27
            -1 if self.permutation_count is None
            else int(self.permutation_count),           # 28
            -1 if self.ctr_estimation_permutation_id is None
            else int(self.ctr_estimation_permutation_id),  # 29
            _tri(self.boost_from_average),              # 30
            _GROW_POLICY_CODES[self.grow_policy],       # 31
            -1 if self.max_leaves is None
            else int(self.max_leaves),                  # 32
            int(self.min_data_in_leaf),                 # 33
            len(cw),                                    # 34  n_class_weights
            # ---- and then `n_class_weights` MORE, the weights themselves.
            # They ride in this list rather than at a seventh buffer address
            # because `gbdt_fit` already takes eight arguments and
            # `PythonModuleBuilder.def_function` stops inferring a signature
            # at around nine. A Python float reaches Mojo's `Float64(py=)`
            # exactly; the same number written into a string and parsed back
            # would not, and a class weight is the caller's number, not ours
            # to round.
            *cw,
            # Optional ABI tail preserves the old layout for default fits.
            *tail,
        ]

    def _flags(self, n_features):
        """One packed word per feature: bit 0 categorical, bit 1 one-hot."""
        cat = self.cat_features
        one_hot = self.one_hot_features
        if not cat and not one_hot:
            return None
        flags = np.zeros(n_features, dtype=np.uint32)
        for i in cat or ():
            if not 0 <= i < n_features:
                raise ValueError(
                    f"mojolearn: cat_features index {i} out of range for "
                    f"{n_features} features"
                )
            flags[i] |= 1
        for i in one_hot or ():
            if not 0 <= i < n_features:
                raise ValueError(
                    f"mojolearn: one_hot_features index {i} out of range "
                    f"for {n_features} features"
                )
            flags[i] |= 2
        return flags

    def _eval_arrays(self, eval_set, n_features):
        """`(X_eval, y_eval)` -> column-major float32 plus a row count.

        Returns `(None, None, 0)` when there is no eval set, which is the
        "unread" contract the binding documents for the two addresses.
        """
        if eval_set is None:
            return None, None, 0
        if isinstance(eval_set, list):
            if len(eval_set) != 1:
                raise ValueError(
                    "mojolearn: eval_set takes ONE (X, y) pair; the "
                    "boosting carries a single test cursor. CatBoost "
                    f"accepts a list, this takes {len(eval_set)} as an "
                    "error rather than scoring the first and dropping "
                    "the rest"
                )
            eval_set = eval_set[0]
        try:
            Xe, ye = eval_set
        except (TypeError, ValueError):
            raise ValueError(
                "mojolearn: eval_set must be (X_eval, y_eval)"
            ) from None

        # one column-major materialization at most (zero for float32
        # F-order input); see DEVIATION 1887 in `_arrays.as_f32_colmajor`
        Xea, Ecol, _ = as_f32_colmajor(Xe, "eval_set X")
        n_eval_rows, n_eval_features = Xea.shape
        if n_eval_features != n_features:
            raise ValueError(
                f"mojolearn: eval_set X has {n_eval_features} features "
                f"for a fit on {n_features}"
            )
        if n_eval_rows == 0:
            raise ValueError("mojolearn: eval_set X has no rows")
        yea = np.ascontiguousarray(
            np.asarray(ye).ravel(), dtype=np.float32
        )
        if yea.shape[0] != n_eval_rows:
            raise ValueError(
                f"mojolearn: eval_set y has {yea.shape[0]} values for "
                f"{n_eval_rows} rows"
            )
        return Ecol, yea, n_eval_rows

    def fit(self, X, y, sample_weight=None, eval_set=None):
        """Fit the ensemble. `X` is (n_samples, n_features), `y` is 1-D.

        `sample_weight` is a per-row weight, `None` meaning all ones. It
        MULTIPLIES with `class_weights` where both are given, which is
        their own combination at pool build
        (`target/data_providers.cpp:168`:
        `rawWeights[i] * rawGroupWeights[i] * classWeights[...]`). Their
        group-weight factor is absent because this port carries no
        `group_id`.

        `eval_set` is `(X_eval, y_eval)`, or a one-element list holding
        that pair. **CatBoost takes a LIST of eval sets and this takes
        one**, because the boosting carries a single test cursor; more
        than one raises rather than silently scoring the first.

        Passing it changes the DEFAULT of `use_best_model` to True, which
        is CatBoost's own data-dependent default
        (`options_helper.cpp:106-108`) and means `model_` holds the trees
        up to the best held-out iteration rather than all of them. Pass
        `use_best_model=False` to keep every tree.

        After the call: `test_loss_curve_` is the held-out loss per
        iteration, `best_iteration_` its argmin, and `stopped_early_` says
        whether the detector fired before `n_estimators`.
        """
        # COLUMN-MAJOR is what the quantizer walks, so X is materialized
        # STRAIGHT to Fortran order: one copy at most, zero when the
        # caller already passes float32 F-order. The old `as_f32_c` +
        # `ascontiguousarray(Xa.T)` pair cost up to two full copies of X
        # inside the timed fit. DEVIATION 1887, declared in
        # `_arrays.as_f32_colmajor`.
        Xa, Xcol, _ = as_f32_colmajor(X, "X")
        n_rows, n_features = Xa.shape

        if n_rows == 0 or n_features == 0:
            raise ValueError("mojolearn: fit requires at least one row and one feature")

        ya = np.ascontiguousarray(np.asarray(y).ravel(), dtype=np.float32)
        if ya.shape[0] != n_rows:
            raise ValueError(
                f"mojolearn: y has {ya.shape[0]} values for {n_rows} rows"
            )

        # `nan_mode='Forbidden'` MEANS "THERE ARE NO NaNs", AND IT HAS TO BE
        # CHECKED HERE OR IT MEANS NOTHING.
        #
        # CatBoost raises on this pair. Filtering the NaNs during border
        # search would otherwise silently route them into an ordinary bin.
        # The native quantizer enforces the same upstream CB_ENSURE for
        # direct Mojo callers; this guard avoids entering the GPU binding.
        if self.nan_mode == "Forbidden" and not np.isfinite(Xa).all():
            if np.isnan(Xa).any():
                raise ValueError(
                    "mojolearn: nan_mode='Forbidden' but X contains NaN. "
                    "CatBoost refuses this pair; this port would otherwise "
                    "bin the NaNs silently with no NaN bin. Use "
                    "nan_mode='Min' or 'Max', or clean the column."
                )

        if self.loss in MULTI_OUTPUT_LOSSES:
            # Validate before native class-weight indexing. Upstream checks
            # the class index before multiplication (54a8143a,
            # private/libs/target/data_providers.cpp:162-168).
            if (not ya.size or not np.isfinite(ya).all()
                    or (ya < 0).any() or (ya != np.floor(ya)).any()):
                raise ValueError(
                    f"mojolearn: {self.loss} labels must be finite "
                    "nonnegative integer class codes"
                )
            classes = np.unique(ya)
            if classes.size < 2:
                raise ValueError(
                    f"mojolearn: {self.loss} needs at least two classes"
                )
            if classes[0] != 0 or classes[-1] != classes.size - 1:
                raise ValueError(
                    f"mojolearn: {self.loss} labels must be dense class "
                    "codes 0..k-1; encode labels before fitting"
                )
            if self.class_weights is not None and len(self.class_weights) != classes.size:
                raise ValueError(
                    f"mojolearn: class_weights needs {classes.size} entries "
                    f"for {self.loss}"
                )

        flags = self._flags(n_features)
        n_flags = 0 if flags is None else n_features
        flags_holder = flags if flags is not None else np.zeros(1, np.uint32)

        if sample_weight is None:
            wa = Xcol[:1]  # unread while params[2] == 0
            n_weights = 0
        else:
            wa = np.ascontiguousarray(
                np.asarray(sample_weight).ravel(), dtype=np.float32
            )
            if wa.shape[0] != n_rows:
                raise ValueError(
                    f"mojolearn: sample_weight has {wa.shape[0]} entries "
                    f"for {n_rows} rows"
                )
            if not np.isfinite(wa).all() or (wa < 0).any():
                raise ValueError(
                    "mojolearn: sample_weight must have finite nonnegative entries"
                )
            if not (wa > 0).any():
                raise ValueError("mojolearn: sample_weight must have positive total weight")
            n_weights = n_rows

        Ecol, ea, n_eval_rows = self._eval_arrays(eval_set, n_features)
        if n_eval_rows and self.loss in MULTI_OUTPUT_LOSSES:
            if (not np.isfinite(ea).all() or (ea < 0).any()
                    or (ea != np.floor(ea)).any() or (ea >= classes.size).any()):
                raise ValueError(
                    "mojolearn: eval_set labels must be integer class codes "
                    "within the training class range"
                )

        params = self._params(
            n_rows, n_features, n_flags, n_weights, n_eval_rows
        )
        if self.od_type is not None and self.od_type not in OD_TYPES:
            raise ValueError(
                f"mojolearn: od_type must be one of {OD_TYPES}, got "
                f"{self.od_type!r}"
            )
        strs = [
            self.loss,
            self.bootstrap_type or "",
            self.od_type or "",
            self.nan_mode,
        ]

        # THE EVAL ADDRESSES ARE UNREAD WHEN params[20] IS 0, and the
        # learn buffer stands in so nothing has to allocate a throwaway --
        # the same stand-in the weights use.
        eval_x_holder = Ecol if Ecol is not None else Xcol
        eval_y_holder = ea if ea is not None else ya

        out = self._bind("_mojolearn_gbdt").gbdt_fit(
            _addr_ro(Xcol),
            _addr_ro(ya),
            _addr_ro(wa),
            _addr_ro(flags_holder),
            _addr_ro(eval_x_holder),
            _addr_ro(eval_y_holder),
            params,
            strs,
        )
        self.model_ = out[0]
        # the model's bias (CatBoost's `get_scale_and_bias()[1]`), parsed
        # from the text's BITS half so it round-trips exactly. 0.0 on
        # every fit without boost_from_average, exactly as theirs is.
        self.bias_ = 0.0
        for _line in str(out[0]).split("\n"):
            if _line.startswith("bias "):
                _bits = int(_line.split()[1].split("/")[1], 16)
                self.bias_ = float(
                    np.uint64(_bits).view(np.float64)
                )
                break
        self.best_iteration_ = int(out[1])
        self.stopped_early_ = bool(out[2])
        self.loss_curve_ = np.asarray(out[3], dtype=np.float64)
        self.test_loss_curve_ = (
            np.asarray(out[4], dtype=np.float64) if n_eval_rows else None
        )
        self.n_features_in_ = n_features
        self.approx_dim_ = self._bind("_mojolearn_gbdt").gbdt_model_dim(self.model_)
        # MULTICLASS DROPS A CLASS AND ONEVSALL DOES NOT. `dim` is
        # `n_classes - 1` for the first (the last class's approx is pinned
        # at zero and not stored) and `n_classes` for the second, whose
        # classes are independent (`multiclass_targets.h:129-134`).
        self.n_classes_ = (
            self.approx_dim_ + 1
            if self.loss == "MultiClass"
            else self.approx_dim_ if self.loss == "MultiClassOneVsAll" else None
        )
        return self

    def _check_fitted(self, X):
        if self.model_ is None:
            raise RuntimeError("mojolearn: predict() before fit()")
        # one column-major materialization at most (zero for float32
        # F-order input); see DEVIATION 1887 in `_arrays.as_f32_colmajor`.
        # `Xcol.base` keeps the backing buffer alive across the Mojo call.
        Xa, Xcol, _ = as_f32_colmajor(X, "X")
        n_rows, n_features = Xa.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"mojolearn: model was fitted on {self.n_features_in_} "
                f"features, got {n_features}"
            )
        return Xcol, n_rows

    def predict(self, X):
        """RAW SCORES, not probabilities. See the module docstring.

        For a single-output loss the result is `(n_samples,)`. For
        `MultiClass` it is `(n_samples, n_classes - 1)` -- the free
        approxes, with the LAST class's pinned at zero and therefore not
        returned. `predict_proba` is what turns those into `n_classes`
        columns; dropping the pinned class and renormalising the rest
        would give a different answer. `MultiClassOneVsAll` returns
        `(n_samples, n_classes)` with one raw score per independent head.
        """
        Xcol, n_rows = self._check_fitted(X)

        if self.approx_dim_ > 1:
            out = np.empty(n_rows * self.approx_dim_, dtype=np.float32)
            width = self._bind("_mojolearn_gbdt").gbdt_predict_multi(
                self.model_, _addr_ro(Xcol), _addr(out),
                [n_rows, _PREDICT_RAW],
            )
            if width != self.approx_dim_:
                raise RuntimeError(
                    f"mojolearn: predict wrote width {width}, expected "
                    f"{self.approx_dim_}"
                )
            return out.reshape(n_rows, width)

        out = np.empty(n_rows, dtype=np.float32)
        wrote = self._bind("_mojolearn_gbdt").gbdt_predict(
            self.model_, _addr_ro(Xcol), _addr(out), [n_rows]
        )
        if wrote != n_rows:
            raise RuntimeError(
                f"mojolearn: predict wrote {wrote} of {n_rows} rows"
            )
        return out

    def predict_proba(self, X):
        """Class probabilities, `(n_samples, n_classes)`.

        Defined for `Logloss` and `CrossEntropy` (the sigmoid),
        `MultiClass` (the softmax), and `MultiClassOneVsAll` (independent
        sigmoids). It
        refuses every other loss rather than returning a number that looks
        like a probability. CatBoost's `prediction_type='Probability'` is
        the same transform over the same raw scores.

        THE TWO MULTI-OUTPUT LOSSES GET DIFFERENT TRANSFORMS, and it is
        not a style choice. `MultiClass` is a softmax -- one shared
        denominator, columns summing to 1, their `Probability`
        (`eval_processing.h:214-221`). `MultiClassOneVsAll` is an
        ELEMENTWISE sigmoid -- their `MultiProbability` (`:222-226`) --
        and **its columns do NOT sum to 1**, because it fitted
        `n_classes` independent "is this row class k" problems and
        renormalising would assert an exclusivity it never learned.

        THE MULTICLASS SOFTMAX IS TAKEN OVER ALL `n_classes`, INCLUDING THE
        PINNED ONE. The model stores `n_classes - 1` free approxes and the
        last class's is zero by construction; the device applies the
        transform, so the max-subtraction matches the one in their
        `MultiLogitValAndFirstDerImpl` rather than being re-derived here.
        The returned columns are in class-code order, `0 .. n_classes - 1`.
        """
        if self.loss in MULTI_OUTPUT_LOSSES:
            Xcol, n_rows = self._check_fitted(X)
            # 54a8143a libs/model/eval_processing.h:214-226:
            # MultiProbability applies CalcSigmoid elementwise.
            mode = (_PREDICT_SIGMOID if self.loss == "MultiClassOneVsAll"
                    else _PREDICT_SOFTMAX)
            out = np.empty(n_rows * self.n_classes_, dtype=np.float32)
            width = self._bind("_mojolearn_gbdt").gbdt_predict_multi(
                self.model_, _addr_ro(Xcol), _addr(out), [n_rows, mode]
            )
            if width != self.n_classes_:
                raise RuntimeError(
                    f"mojolearn: predict_proba wrote width {width}, "
                    f"expected {self.n_classes_}"
                )
            return out.reshape(n_rows, width)

        if self.loss not in ("Logloss", "CrossEntropy"):
            raise ValueError(
                f"mojolearn: predict_proba is defined for Logloss, "
                f"CrossEntropy, MultiClass and MultiClassOneVsAll; this model was fitted with "
                f"{self.loss!r}. Use predict() and apply the link "
                f"yourself."
            )
        raw = np.ascontiguousarray(self.predict(X).astype(np.float64))
        if self._bind("_mojolearn_gbdt").gbdt_numeric_mode() == 1:
            # DEVIATION 258: under NUMERIC_IDENTICAL the sigmoid runs
            # through the portable double exp so the probability bits are
            # the same on every host; numpy's exp carries the host libm's
            # last bit (E2 round 1: gbdt_logloss DIVERGENT@proba with
            # identical cards and identical raw margins)
            p1 = np.empty_like(raw)
            self._bind("_mojolearn_gbdt").gbdt_sigmoid(_addr_ro(raw), _addr(p1), raw.shape[0])
        else:
            p1 = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack((1.0 - p1, p1))

    def predict_classes(self, X):
        """The argmax of `predict_proba`, as dense class codes.

        Named `predict_classes` rather than overloading `predict`, because
        `predict` returns RAW SCORES for every loss in this library and
        making one loss return labels instead would be the kind of silent
        contract change that is worse than an extra method.
        """
        if self.loss not in MULTI_OUTPUT_LOSSES + (
            "Logloss", "CrossEntropy",
        ):
            raise ValueError(
                f"mojolearn: predict_classes needs a classification loss; "
                f"this model was fitted with {self.loss!r}."
            )
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def _tree_metadata(self):
        """Read host model metadata without selecting a GPU extension.

        This is archive inspection, not a second model evaluator. Float
        values come exclusively from the authoritative IEEE bits.
        """
        if self.model_ is None:
            raise RuntimeError("mojolearn: tree inspection before fit()")
        counts, dimensions, leaves = [], [], []
        declared_trees = None
        for line in str(self.model_).splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "trees":
                declared_trees = int(fields[1])
            elif fields[0] in ("tree", "ntree"):
                if int(fields[1]) != len(counts):
                    raise ValueError("mojolearn: invalid tree order in model metadata")
                size, dim = int(fields[3]), int(fields[5])
                if size < 0 or dim < 1 or (fields[0] == "tree" and size > 31):
                    raise ValueError("mojolearn: invalid tree dimensions in model metadata")
                counts.append((1 << size) if fields[0] == "tree" else size + 1)
                dimensions.append(dim)
                leaves.append([])
            elif fields[0] == "leaf":
                tree, index = int(fields[1]), int(fields[2])
                if tree < 0 or tree >= len(leaves) or index != len(leaves[tree]):
                    raise ValueError("mojolearn: invalid leaf order in model metadata")
                bits = int(fields[3].split("/")[1], 16)
                if not 0 <= bits <= 0xffffffff:
                    raise ValueError("mojolearn: invalid leaf bits in model metadata")
                leaves[tree].append(bits)
        if declared_trees != len(counts) or any(
            len(values) != count * dim
            for values, count, dim in zip(leaves, counts, dimensions)
        ):
            raise ValueError("mojolearn: incomplete tree metadata")
        return counts, leaves

    def get_tree_leaf_counts(self):
        """Return leaf counts per tree as uint32, as in CatBoost.

        Reads fitted or loaded model metadata; does not launch GPU work.
        """
        counts, _ = self._tree_metadata()
        return np.asarray(counts, dtype=np.uint32)

    def get_leaf_values(self):
        """Return tree-major, leaf-major stored values as float64.

        Multiclass stores each leaf's dimensions consecutively. MultiClass
        contains k-1 free dimensions; OneVsAll contains k dimensions.
        Conversion from stored float32 bits to float64 is exact, including
        signed zero. No decimal model text is used to recover the values.
        """
        _, leaves = self._tree_metadata()
        bits = np.asarray([value for tree in leaves for value in tree], dtype=np.uint32)
        return bits.view(np.float32).astype(np.float64)

    def save(self, path):
        """Write the fitted ensemble to `path` as an npz.

        The payload is `model_`, the text format `check-model-io` already
        gates bit-for-bit. Every float in that text carries its hex bit
        pattern beside the decimal and the loader reads the BITS half, so
        the text is a bit-exact carrier by construction. It travels here
        as raw utf-8 bytes; nothing in this method formats a number. The
        model text is self contained for prediction. The quantization
        borders live in it as one `feature ... borders ...` table per
        feature, each border with its hex bits, and every split references
        a border by index, which is why `gbdt_predict` takes only the text
        and raw feature rows.

        The effective numeric mode is saved and restored independently of a
        future process default. Legacy files without this field keep the
        loader process default; select their desired tier explicitly.

        The curve attributes (`loss_curve_`, `test_loss_curve_`) are not
        reconstructed by `load`; a loaded model carries them as None. The
        model text does record per-iteration learn losses on its `loss`
        lines, so nothing is destroyed, but `load` populates only what
        prediction reads.
        """
        if self.model_ is None:
            raise RuntimeError("mojolearn: save() before fit()")
        # Persist the tier prediction would use now, including an inherited
        # process default. Saving None would let a different process silently
        # turn an IDENTICAL model into FAST arithmetic after loading.
        mode = getattr(self, "numeric_mode", None) or _backend.default_mode()
        if not isinstance(mode, str) or mode.strip().lower() not in (
            "fast", "deterministic", "identical"
        ):
            raise ValueError(f"mojolearn: cannot save invalid numeric_mode {mode!r}")
        mode = mode.strip().lower()
        arrays = {
            "format": np.asarray(_MODEL_FORMAT),
            "numeric_mode": np.asarray(mode),
            "estimator": np.asarray(type(self).__name__),
            "loss": np.asarray(self.loss),
            "model": np.frombuffer(
                str(self.model_).encode("utf-8"), dtype=np.uint8
            ),
            "meta": np.asarray(
                [
                    self.n_features_in_,
                    self.approx_dim_,
                    -1 if self.n_classes_ is None else self.n_classes_,
                    -1 if self.best_iteration_ is None
                    else self.best_iteration_,
                    1 if self.stopped_early_ else 0,
                ],
                dtype=np.int64,
            ),
            # raw float64 bytes, already parsed from the text's BITS half
            "bias": np.asarray(self.bias_, dtype=np.float64),
        }
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load an ensemble saved by `save`. The result predicts; it does
        not refit, and it does not carry the training configuration or
        the loss curves. `approx_dim_` is re-derived from the model text
        exactly as `fit` derives it, and a file whose stored value
        disagrees is refused as corrupt."""
        arrays = _serialize.read_npz(path, _MODEL_FORMAT)
        saved_as = _serialize.scalar_str(arrays, "estimator")
        if saved_as != cls.__name__:
            raise ValueError(
                f"mojolearn: {path!r} was saved by {saved_as}, not "
                f"{cls.__name__}"
            )
        obj = cls.__new__(cls)
        # Restore before even gbdt_model_dim binds an extension. Old format-1
        # archives omitted this optional field and retain their historical
        # process-default behavior; malformed new metadata must never fall
        # back to that legacy interpretation.
        obj.numeric_mode = None
        if "numeric_mode" in arrays:
            stored_mode = arrays["numeric_mode"]
            if stored_mode.dtype.kind != "U" or stored_mode.size != 1:
                raise ValueError("mojolearn: saved numeric_mode must be one string")
            mode = str(stored_mode.reshape(-1)[0])
            if mode not in ("fast", "deterministic", "identical"):
                raise ValueError(f"mojolearn: invalid saved numeric_mode {mode!r}")
            obj.numeric_mode = mode
        obj.loss = _serialize.scalar_str(arrays, "loss")
        obj.model_ = bytes(
            _serialize.exact(arrays, "model", np.uint8)
        ).decode("utf-8")
        meta = _serialize.exact(arrays, "meta", np.int64)
        obj.n_features_in_ = int(meta[0])
        obj.approx_dim_ = int(obj._bind("_mojolearn_gbdt").gbdt_model_dim(obj.model_))
        if obj.approx_dim_ != int(meta[1]):
            raise ValueError(
                f"mojolearn: {path!r} stores approx_dim {int(meta[1])} but "
                f"its model text holds {obj.approx_dim_}; the file is "
                "corrupt"
            )
        obj.n_classes_ = None if meta[2] < 0 else int(meta[2])
        obj.best_iteration_ = None if meta[3] < 0 else int(meta[3])
        obj.stopped_early_ = bool(meta[4])
        obj.bias_ = float(
            _serialize.exact(arrays, "bias", np.float64).reshape(-1)[0]
        )
        obj.loss_curve_ = None
        obj.test_loss_curve_ = None
        return obj


class ExperimentalTwoLevelFeatureFreq(GradientBoosting):
    """Experimental one-tree, depth-two FeatureFreq combination estimator.

    Source columns must hold dense categorical codes; non-source columns are
    ordinary numeric split candidates. It is separate from
    :class:`GradientBoosting`; selecting it cannot change standard fit
    behavior or imply support for deeper/general CatBoost combinations.
    ``fit(..., sample_weight=...)`` accepts finite non-negative row weights;
    structure scores and leaf values both use the same weights.
    """

    def __init__(
        self, sources, learning_rate=0.03, l2_leaf_reg=3.0, random_state=0
    ):
        super().__init__(
            loss="RMSE", n_estimators=1, max_depth=2,
            learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
            random_state=random_state,
        )
        try:
            parsed = tuple(int(i) for i in sources)
        except (TypeError, ValueError):
            raise ValueError("mojolearn: sources must be integer indices") from None
        if len(parsed) < 2 or len(set(parsed)) != len(parsed):
            raise ValueError("mojolearn: sources need at least two unique indices")
        self.sources = parsed

    def fit(self, X, y, sample_weight=None):
        Xa, Xcol, _ = as_f32_colmajor(X, "X")
        n_rows, n_features = Xa.shape
        if n_rows == 0 or n_features < 2:
            raise ValueError("mojolearn: experimental FeatureFreq needs rows and columns")
        if any(i < 0 or i >= n_features for i in self.sources):
            raise ValueError(
                f"mojolearn: sources {self.sources!r} outside {n_features} features"
            )
        if not np.isfinite(Xa).all():
            raise ValueError(
                "mojolearn: experimental FeatureFreq X must be finite"
            )
        for f in self.sources:
            column = Xa[:, f]
            if (column < 0).any() or not np.equal(
                column, np.floor(column)
            ).all():
                raise ValueError(
                    "mojolearn: experimental FeatureFreq source columns must "
                    "contain non-negative integer category codes"
                )
            values = np.unique(column)
            if values.size < 2 or not np.array_equal(
                values, np.arange(values.size, dtype=values.dtype)
            ):
                raise ValueError(
                    "mojolearn: every experimental FeatureFreq source must "
                    f"be densely coded 0..k-1; column {f} has {values!r}"
                )
        for f in set(range(n_features)).difference(self.sources):
            if np.unique(Xa[:, f]).size < 2:
                raise ValueError(
                    "mojolearn: experimental FeatureFreq numeric columns "
                    f"must vary; column {f} is constant"
                )
        ya = np.ascontiguousarray(np.asarray(y).ravel(), dtype=np.float32)
        if ya.shape[0] != n_rows or not np.isfinite(ya).all():
            raise ValueError("mojolearn: y must be finite with one value per row")
        if sample_weight is None:
            weights = ya[:1]
            n_weights = 0
        else:
            weights = np.ascontiguousarray(
                np.asarray(sample_weight).ravel(), dtype=np.float32
            )
            if weights.shape[0] != n_rows:
                raise ValueError(
                    "mojolearn: sample_weight must have one value per row"
                )
            if not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError(
                    "mojolearn: sample_weight must be finite and non-negative"
                )
            if not float(weights.sum(dtype=np.float64)) > 0.0:
                raise ValueError("mojolearn: sample_weight must have positive sum")
            n_weights = n_rows
        source_array = np.ascontiguousarray(self.sources, dtype=np.uint32)
        self.model_ = self._bind(
            "_mojolearn_gbdt"
        ).gbdt_fit_two_level_feature_freq(
            _addr_ro(Xcol), _addr_ro(ya), _addr_ro(weights),
            _addr_ro(source_array),
            [
                n_rows, n_features, n_weights, source_array.size,
                float(self.learning_rate), float(self.l2_leaf_reg),
                int(self.random_state),
            ],
        )
        self.n_features_in_ = n_features
        self.approx_dim_ = 1
        self.n_classes_ = None
        self.bias_ = 0.0
        self.best_iteration_ = 0
        self.stopped_early_ = False
        self.loss_curve_ = None
        self.test_loss_curve_ = None
        return self


class OrderedRMSE(GradientBoosting):
    """Numeric ordered RMSE with one explicit row permutation.

    Fits symmetric trees using independent prefix approximation cursors and
    one full-data cursor for exported leaves. Supports finite numeric X/y,
    nonnegative sample weights, zero initial bias, Newton-1 leaves, and
    depths 1..8. Categorical CTRs, other objectives, bootstrap and early
    stopping are absent from this narrow API and rejected as unknown kwargs.
    This is not the general CatBoost ``boosting_type="Ordered"`` interface.

    Pass ``numeric_mode="identical"`` to select pinned arithmetic. Native
    AMD/NVIDIA fixtures are certified; the Python binding needs independent
    installed-artifact qualification. ``loss_curve_`` and
    ``best_iteration_`` are None because this path does not track losses.
    """

    def __init__(
        self, n_estimators=100, max_depth=6, learning_rate=0.03,
        l2_leaf_reg=3.0, border_count=128,
    ):
        self._ordered_options(n_estimators, max_depth, border_count,
                              learning_rate, l2_leaf_reg)
        super().__init__(
            loss="RMSE", n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
            border_count=border_count, boost_from_average=False,
            nan_mode="Forbidden",
        )

    @staticmethod
    def _ordered_options(n_estimators, max_depth, border_count,
                         learning_rate, l2_leaf_reg):
        for name, value, lower, upper in (
            ("n_estimators", n_estimators, 1, 2**31 - 1),
            ("max_depth", max_depth, 1, 8),
            ("border_count", border_count, 1, 255),
        ):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or not lower <= value <= upper):
                raise ValueError(f"mojolearn: {name} must be an integer in {lower}..{upper}")
        for name, value, positive in (
            ("learning_rate", learning_rate, True),
            ("l2_leaf_reg", l2_leaf_reg, False),
        ):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)
                    or abs(value) > np.finfo(np.float32).max
                    or value < 0 or (positive and np.float32(value) <= 0)):
                raise ValueError(f"mojolearn: {name} must be finite float32 and "
                                 + ("positive" if positive else "non-negative"))

    def fit(self, X, y, *, permutation, sample_weight=None):
        """Fit using a bijection of original row ids in ``permutation``.

        X, y and weights remain in original row order; only permutation
        defines ordered prefix membership. No host random shuffle is hidden.
        """
        self._ordered_options(self.n_estimators, self.max_depth,
                              self.border_count, self.learning_rate,
                              self.l2_leaf_reg)
        Xa, Xcol, _ = as_f32_colmajor(X, "X")
        n_rows, n_features = Xa.shape
        if n_rows < 4 or n_rows > np.iinfo(np.uint32).max or n_features < 1:
            raise ValueError("mojolearn: ordered RMSE requires >=4 rows and >=1 feature")
        if not np.isfinite(Xa).all():
            raise ValueError("mojolearn: ordered RMSE X must be finite")
        ya = np.asarray(y)
        if ya.ndim != 1 or ya.size != n_rows:
            raise ValueError("mojolearn: y must be one-dimensional with one value per row")
        ya = np.ascontiguousarray(ya, dtype=np.float32)
        if not np.isfinite(ya).all():
            raise ValueError("mojolearn: y must be finite")
        order = np.asarray(permutation)
        if (order.ndim != 1 or order.size != n_rows
                or order.dtype.kind not in "iu"
                or (order < 0).any() or (order >= n_rows).any()
                or np.unique(order).size != n_rows):
            raise ValueError("mojolearn: permutation must be an integer bijection of row ids")
        order = np.ascontiguousarray(order, dtype=np.uint32)
        if sample_weight is None:
            weights = ya[:1]
            n_weights = 0
        else:
            weights = np.asarray(sample_weight)
            if weights.ndim != 1 or weights.size != n_rows:
                raise ValueError("mojolearn: sample_weight must have one value per row")
            weights = np.ascontiguousarray(weights, dtype=np.float32)
            if not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("mojolearn: sample_weight must be finite and non-negative")
            if not weights.sum(dtype=np.float64) > 0:
                raise ValueError("mojolearn: sample_weight must have positive sum")
            n_weights = n_rows
        binding = self._bind("_mojolearn_gbdt")
        if not hasattr(binding, "gbdt_fit_ordered_rmse"):
            raise RuntimeError(
                "mojolearn: this GBDT binary predates OrderedRMSE; "
                "install or build a matching native extension"
            )
        model = binding.gbdt_fit_ordered_rmse(
            _addr_ro(Xcol), _addr_ro(ya), _addr_ro(weights), _addr_ro(order),
            [n_rows, n_features, n_weights, order.size,
             int(self.n_estimators), int(self.max_depth), int(self.border_count),
             float(self.learning_rate), float(self.l2_leaf_reg)],
        )
        self.model_ = model
        self.n_features_in_ = n_features
        self.approx_dim_ = 1
        self.n_classes_ = None
        self.bias_ = 0.0
        self.best_iteration_ = None
        self.stopped_early_ = False
        self.loss_curve_ = None
        self.test_loss_curve_ = None
        return self


# Separate sklearn prediction contracts; legacy GradientBoosting stays unchanged.
from ._gbdt_adapters import GradientBoostingClassifier, GradientBoostingRegressor
