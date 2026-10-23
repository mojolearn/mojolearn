# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Gradient-boosted trees on the GPU, with CatBoost as the reference: its three growth
policies (oblivious SymmetricTree, Depthwise, Lossguide), its losses.

**UNDER grow_policy='SymmetricTree' (THE DEFAULT) EVERY DEFAULT IS CatBoost's
GPU LEARNER'S** (`task_type='GPU'`, catboost 1.2.10 at the pinned reference
54a8143a), not scikit-learn's, and several of them change results rather
than just speed (lane/catboost-parity, 2026-09-19; before that date two of
them were not, see CHANGELOG):

    n_estimators    1000                       boosting_options.cpp:13
    learning_rate   auto from the pool, as     options_helper.cpp:252-288
                    theirs (0.03 otherwise)    boosting_options.cpp:10
    max_depth       6                          oblivious_tree_options.cpp:12
    l2_leaf_reg     3.0 (0 for YetiRank)       catboost_options.cpp:34-37
    border_count    128 on GPU (254 on CPU)    data_processing_options.cpp:16
    feature_border_type  GreedyLogSum          data_processing_options.cpp:15
    random_strength 1.0                        oblivious_tree_options.cpp:17
    bootstrap_type  Bayesian, temperature 1    bootstrap_options.h:16-18
    boosting_type   Ordered below 50,000 rows  catboost_options.cpp:802-807,
                    at >= 500 iterations,      defaults_helper.h:33-42
                    else Plain

The learning rate is fitted from the pool exactly when CatBoost fits it:
`learning_rate`, `l2_leaf_reg`, `leaf_estimation_method` and
`leaf_estimation_iterations` all unset and the loss one of RMSE, Logloss or
MultiClass; it is `exp(A*log(n) + B)` scaled by the iteration count, with
the GPU coefficients, rounded to six decimals and capped at 0.5
(`learning_rate_` holds the value used). A comparison run should still pin
both arms explicitly. Depthwise and Lossguide keep the library's earlier
defaults (100 iterations, 0.03, no noise, no bootstrap, Plain); see the
parameter list.

The DEFAULT tree shape is OBLIVIOUS (symmetric): every node at a level takes
the same split, which is CatBoost's default structure and not scikit-learn's.
A comparison against `GradientBoostingRegressor` at matched hyperparameters
is still comparing two different tree families. `grow_policy='Depthwise'`
and `'Lossguide'` grow CatBoost's NON-SYMMETRIC trees (on the surface since
2026-08-23, DEVIATION 259), and a Depthwise tree IS the level-wise binary
tree scikit-learn grows -- same shape, CatBoost's score and estimator.

X IS ROW-MAJOR HERE AND COLUMN-MAJOR INSIDE. `fit` materializes Fortran
order ONCE, straight from the caller's buffer (`_buffer.as_f32_colmajor`,
DEVIATION 1887). On a 800,000 x 100 C-order matrix that is a 320 MB copy,
and it is reported rather than hidden -- pass `X` already float32 and in
Fortran order and it is a zero-copy borrow, a promise this docstring made
before the code kept it (the old path copied F-order input TWICE, once to
C order and once back).

INPUTS AND OUTPUTS ARE NOT NumPy (DEVIATION 2330). `X`, `y`,
`sample_weight` and the eval pair are anything the buffer protocol
exposes -- an ndarray, an `array.array`, a `mojolearn.Array` -- or a
nested list; `_buffer.as_f32_colmajor` / `as_f32_c` do the one
conversion. Every array this module RETURNS (`predict`, `predict_proba`,
`predict_classes`, `loss_curve_`, `test_loss_curve_`,
`get_tree_leaf_counts`, `get_leaf_values`) is a `mojolearn.Array`, not an
ndarray: `numpy.asarray(result)` is a zero-copy view through
`__array_interface__` for a caller who has NumPy. Nothing in this module
imports NumPy.

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
`cuda/train_lib/multiclass.cpp:5-7` at the pinned reference `54a8143a`.

Dropping MultiClass's last probability column and renormalising the rest
gives a different and wrong answer: the pinned class is a real class whose
approx happens to be zero. Labels are dense codes `0..k-1` for both, and the
class count is derived from them, as their `TClassificationTargetHelper`
derives it.
"""

# The public names: the boosting estimators and their option tables defined here, and
# scikit-learn's ensemble names resolved lazily from the package (_sklearn_modules.LAYOUT).
__all__ = [
    'GradientBoosting', 'GradientBoostingClassifier', 'GradientBoostingRegressor', 'OrderedRMSE',
    'ExperimentalTwoLevelFeatureFreq', 'catboost_auto_learning_rate', 'LOSSES',
    'MULTI_OUTPUT_LOSSES', 'MULTI_REGRESSION_LOSSES', 'BORDER_TYPES', 'BOOTSTRAP_TYPES',
    'BOOSTING_TYPES', 'LEAF_ESTIMATION_GRADIENT', 'LEAF_ESTIMATION_NEWTON', 'LEAF_ESTIMATION_EXACT',
    'LEAF_ESTIMATION_SIMPLE', 'SCORE_FUNCTION_SOLAR_L2', 'SCORE_FUNCTION_COSINE',
    'SCORE_FUNCTION_NEWTON_L2', 'SCORE_FUNCTION_NEWTON_COSINE', 'SCORE_FUNCTION_LOO_L2',
    'SCORE_FUNCTION_SAT_L2', 'SCORE_FUNCTION_L2', 'SCORE_FUNCTIONS', 'GROW_POLICIES', 'NAN_MODES',
    'OD_TYPES', 'AdaBoostClassifier', 'AdaBoostRegressor', 'BaggingClassifier', 'BaggingRegressor',
    'ExtraTreesClassifier', 'ExtraTreesRegressor', 'IsolationForest', 'RandomForestClassifier',
    'RandomForestRegressor', 'RandomTreesEmbedding', 'StackingClassifier', 'StackingRegressor',
    'VotingClassifier', 'VotingRegressor',
]

import numbers
from . import _portable_math as math
import struct

# GBDT HAS ITS OWN EXTENSION, built by `bindings/build_gbdt.sh`, for the
# reason `_mojolearn_estimators` has one: an independently changing binding
# should not be a merge point. Every parameter added to `GbdtFitParams` used
# to have to be unpacked in two files that could silently disagree about the
# order of a flat list, which is a wrong answer rather than a failure.
#
# It was COMMISSIONED for a different reason -- a supposed per-module cap on
# ahead-of-time Metal compilation, keyed on the entry file's basename -- and
# that reason turned out not to exist: the kernels were
# being lost to `MACOSX_DEPLOYMENT_TARGET` in the environment plus a compiler
# cache that does not key on it, and the basename never mattered.
from . import _backend, _mojolearn_gbdt, _native_archive_bridge, _native_trees_bridge
from ._mode import NumericModeMixin
from ._labels import is_bool

#: The npz model-file format tag `save` writes and `load` requires.
_MODEL_FORMAT = "mojolearn-gbdt-1"

# The compiled saved-model owner, connected once at import: `save` is one
# native call (the schema registered after `GradientBoosting`).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()
# The compiled tree-ensemble owner: `predict`, `predict_proba`,
# `predict_classes`, the tree inspection reads and the OrderedRMSE and
# FeatureFreq fits are one native call each (bindings/trees_public_gbdt.mojo).
_TREES, _TREES_CONTEXT = _native_trees_bridge.connect()


#: CatBoost's `ELossFunction` spellings that this implementation trains. The list is
#: the reachable set of their GPU pointwise target
#: (`pointwise_target_impl.h:259-299`), minus the ones whose leaf estimator
#: or kernel family is not implemented.
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
    "QueryRMSE",
    "PairLogit",
    "YetiRank",
    "MultiRMSE",
)

#: Their GPU target keeps numClasses - 1 planes for MultiClass and
#: numClasses for OneVsAll (`multiclass_targets.h:129-134`, 54a8143a).
MULTI_OUTPUT_LOSSES = ("MultiClass", "MultiClassOneVsAll")

#: MULTI-TARGET REGRESSION (lane/algos-trees, 2026-09-27): their
#: `TMultiClassificationTargets` with `NumClasses = GetTargetDimension()`
#: (`multiclass_targets.h:155-156`), `y` of shape `(n_samples, n_targets)`
#: with `n_targets >= 2`, `predict` RAW `(n_samples, n_targets)`, no
#: `predict_proba`. Not in `MULTI_OUTPUT_LOSSES`, whose members read class
#: codes. Carried on the Plain SymmetricTree greedy fit with numeric
#: features and no eval set; the rest is refused by name at `fit`.
#: DEVIATION 5951: CatBoost's unset `boost_from_average` is True for
#: MultiRMSE (`options_helper.cpp:367`) with a per-dimension start this
#: model's one-value bias does not carry, so unset resolves False here and
#: True is refused.
MULTI_REGRESSION_LOSSES = ("MultiRMSE",)

#: `gbdt_predict_multi`'s transform, following their `EPredictionType`
#: (`libs/model/eval_processing.h:186-226`).
_PREDICT_RAW = 0
_PREDICT_SOFTMAX = 1   # their `Probability`,      MultiClass
_PREDICT_SIGMOID = 2   # their `MultiProbability`, MultiClassOneVsAll
#: `gbdt_resident_predict` only: the Logloss / CrossEntropy `predict_proba`
#: pair `[1 - p, p]` as float64 (DEVIATION 2980, the resident door's spelling
#: of `gbdt_sigmoid_pair`)
_PREDICT_SIGMOID_PAIR = 3
_PREDICT_CLASSES = 4
_PREDICT_CLASSES_PINNED = 5
_PREDICT_CLASSES_OVA = 6

#: DEVIATION 2980: the device-resident parsed model is THE door of
#: `GradientBoosting.predict` and `predict_proba` wherever the loaded binding
#: exports `gbdt_resident_prepare`. The env switch that took the per-call
#: parse instead is gone (cpu-gpu-cleanup t-forest); the per-call parse
#: (`gbdt_predict`, `gbdt_predict_multi`) serves only a binary without the
#: resident entry points.

#: Losses whose parameter CatBoost makes MANDATORY. Passing the loss without
#: it raises here rather than in Mojo, so the message names the Python
#: keyword the caller has to add.
_REQUIRED_PARAM = {
    "Lq": ("q", "loss_q"),
    "Huber": ("delta", "loss_delta"),
    "Tweedie": ("variance_power", "loss_variance_power"),
    "Expectile": ("alpha", "loss_alpha"),
}

#: Their `EBorderSelectionType` spellings (`enums.h`), the seven
#: `feature_border_type`s `MakeBinarizer` dispatches
#: (`library/cpp/grid_creator/binarization.cpp:114-134`). GreedyLogSum is
#: their default for float features (`data_processing_options.cpp:15`).
BORDER_TYPES = ("GreedyLogSum", "Median", "Uniform", "UniformAndQuantiles",
                "MaxLogSum", "MinEntropy", "GreedyMinEntropy")

#: `EBootstrapType` spellings reachable from their GPU oblivious searcher.
#: MVS is absent because their own searcher asserts it away
#: (`weak_objective_impl.h:30`).
BOOTSTRAP_TYPES = ("Bayesian", "Bernoulli", "Poisson", "No")

# ---------------------------------------------------------------------------
# CATBOOST'S GPU DEFAULTS FOR THE OBLIVIOUS (SymmetricTree) LEARNER
# (lane/catboost-parity, 2026-09-19). Every line cites the pinned reference
# 54a8143a (catboost 1.2.10). Depthwise and Lossguide keep this library's
# older defaults, which are listed beside each one; see the class docstring.
# ---------------------------------------------------------------------------

#: `IterationCount("iterations", 1000)` (`boosting_options.cpp:13`).
_CATBOOST_ITERATIONS = 1000
#: `LearningRate("learning_rate", 0.03)` (`boosting_options.cpp:10`): the
#: value when the auto-selection below does not apply.
_CATBOOST_LEARNING_RATE = 0.03
#: `RandomStrength("random_strength", 1.0)` (`oblivious_tree_options.cpp:17`).
_CATBOOST_RANDOM_STRENGTH = 1.0
#: `BootstrapType("type", EBootstrapType::Bayesian)` (`bootstrap_options.h:18`).
#: The MVS default of `SetNotSpecifiedOptionsToDefaults` is CPU only
#: (`catboost_options.cpp:782-787`, `TaskType == ETaskType::CPU`).
_CATBOOST_BOOTSTRAP = "Bayesian"
#: The older defaults Depthwise and Lossguide keep.
_LEGACY_ITERATIONS = 100
_LEGACY_LEARNING_RATE = 0.03
_LEGACY_RANDOM_STRENGTH = 0.0

#: `EBoostingType` spellings (`boosting_options.cpp:16`).
BOOSTING_TYPES = ("Plain", "Ordered")

#: `UpdateBoostingTypeOption` (`defaults_helper.h:33-42`): an unset boosting
#: type becomes Plain at or above this many learn rows, or below
#: `_ORDERED_MIN_ITERATIONS` iterations.
_ORDERED_MAX_ROWS = 50000
_ORDERED_MIN_ITERATIONS = 500

#: The greedy searcher's ranking targets form grouped gradients first, then
#: multiply both statistic planes by row bootstrap weights, as CatBoost's
#: querywise_targets_impl.h::StochasticDer and weak_objective_impl.h do.
_QUERYWISE_LOSSES = ("QueryRMSE", "PairLogit", "YetiRank")

#: `TAutoLRParamsGuesser` (`libs/train_lib/options_helper.cpp:176-243`), the
#: GPU rows (`:221-243`), keyed (target type, use_best_model,
#: boost_from_average) -> (DatasetSizeCoeff A, DatasetSizeConst B,
#: IterCountCoeff C, IterCountConst D). `GetTargetType` (`:179-192`) maps
#: Logloss (and MultiLogloss / MultiCrossEntropy, absent here) to Logloss,
#: MultiClass to MultiClass, RMSE to RMSE and every other loss to Unknown,
#: which has no row and keeps 0.03.
_GPU_AUTO_LEARNING_RATE = {
    ("Logloss", True, True): (0.04, -3.226, -0.488, 0.758),
    ("Logloss", False, True): (0.427, -7.316, -0.907, 2.354),
    ("Logloss", True, False): (-0.085, -2.055, -0.414, 0.427),
    ("Logloss", False, False): (-0.055, -3.01, -0.896, 2.366),
    ("MultiClass", True, False): (0.101, -2.95, -0.437, 1.136),
    ("MultiClass", False, False): (0.204, -4.144, -0.833, 2.889),
    ("RMSE", True, True): (0.108, -3.525, -0.285, 0.058),
    ("RMSE", False, True): (0.131, -4.114, -0.597, 1.693),
    ("RMSE", True, False): (0.051, -3.001, -0.449, 0.859),
    ("RMSE", False, False): (0.047, -3.034, -0.591, 1.554),
}

#: The CPU rows of the same table (`:196-219`). Not used to fit anything:
#: it is here so the formula can be checked against the learning rate a
#: CatBoost CPU install reports from `get_all_params()`, the one arm of
#: their auto-selection this Mac can run (tests/test_gbdt_catboost_defaults.py).
_CPU_AUTO_LEARNING_RATE = {
    ("Logloss", True, True): (0.246, -5.127, -0.451, 0.978),
    ("Logloss", False, True): (0.408, -7.299, -0.928, 2.701),
    ("Logloss", True, False): (0.247, -5.158, -0.435, 0.934),
    ("Logloss", False, False): (0.427, -7.525, -0.917, 2.63),
    ("MultiClass", True, False): (0.02, -2.364, -0.382, 0.924),
    ("MultiClass", False, False): (0.051, -2.889, -0.845, 2.928),
    ("RMSE", True, True): (0.157, -4.062, -0.61, 1.557),
    ("RMSE", False, True): (0.158, -4.287, -0.813, 2.571),
    ("RMSE", True, False): (0.189, -4.383, -0.623, 1.439),
    ("RMSE", False, False): (0.178, -4.473, -0.76, 2.133),
}

#: `AdjustBoostFromAverageDefaultValue`'s list (`options_helper.cpp:353-374`):
#: an unset `boost_from_average` becomes True for these on a single host
#: with no baseline. Only the auto learning-rate key reads this in Python;
#: the native trainer resolves the fit's own value (`gbdt/train.mojo`) by the
#: same list. MAE, Quantile and MAPE resolved False there until 2026-09-19,
#: when their constant (`CalcSampleQuantile`, `gbdt/metrics/sample_quantile.mojo`)
#: landed (lane/catboost-parity).
_BOOST_FROM_AVERAGE_LOSSES = ("RMSE", "MAE", "Quantile", "MAPE")


#: THE NEGATIVE CONTROL of the SymmetricTree defaults (the
#: gbdt-catboost-defaults lane, lane/catboost-parity): with
#: `MOJOLEARN_CATBOOST_DEFAULTS_SABOTAGE=1` AND `MOJOLEARN_HOST_ALLOW_SABOTAGE=1`
#: the auto learning rate reads CatBoost's CPU coefficient rows instead of
#: the GPU learner's. Every resolved default stays a plausible CatBoost
#: default, so nothing but the hash can tell. The lane's sabotage arm must
#: read DIVERGENT on its train cell (its 50-tree fit reads the rate below
#: the 0.5 cap: 0.475 on the GPU rows, 0.216 on the CPU rows); its probed
#: 20-tree fit is capped at 0.5 under both tables, so infer and model stay. No build script, workflow or gate sets it
#: (the `MOJOLEARN_FOLD_ORDER_SABOTAGE` rule, `model_selection.py`).


def catboost_auto_learning_rate(loss, n_rows, iterations, use_best_model,
                                boost_from_average, table=None):
    """The learning rate ``GradientBoosting`` picks when ``learning_rate`` is unset.

    It is CatBoost's automatic learning rate for its GPU learner:
    ``exp(A * log(n_rows) + B)``, scaled by the iteration count, capped at 0.5
    and rounded to six decimals, with coefficients for the ``'RMSE'``,
    ``'Logloss'`` and ``'MultiClass'`` losses.

    Parameters
    ----------
    loss : str
        Loss name, for example ``'Logloss'``.
    n_rows : int
        Number of training rows.
    iterations : int
        Number of boosting iterations (``n_estimators``).
    use_best_model : bool
        Whether the fit keeps the best iteration on an evaluation set.
    boost_from_average : bool
        Whether the fit starts from the target average.
    table : dict, optional
        Coefficient table to use instead of the GPU learner's.

    Returns
    -------
    float or None
        The learning rate, or ``None`` when CatBoost has no automatic rate for
        this loss and setting (``GradientBoosting`` then uses 0.03).

    Example
    -------
    >>> from mojolearn import ensemble
    >>> ensemble.catboost_auto_learning_rate('Logloss', 10000, 1000, False, False)
    0.029701
    """
    return _TREES(None, "trees_gbdt_auto_lr", (loss, n_rows, iterations, use_best_model, boost_from_average, table),
                  _TREES_CONTEXT)


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

#: The four score functions this implementation ACTUALLY COMPUTES. Cosine and L2
#: have a real calcer in the split kernel -- `TCosineScoreCalcer`
#: (`score_calcers.cuh:152-167`) and `TL2ScoreCalcer` (`:40-69`) -- and
#: the Newton spellings pair onto those SAME calcers, differing only in
#: which derivative the der launch puts in the histogram's weight plane:
#: `secondDerAsWeights = IsSecondOrderScoreFunction(scoreFunction)`
#: (`greedy_search_helper.cpp:286-296`) makes plane 0 `weight * der2`
#: instead of the raw weight (`pointwise_target_impl.h:193-201`), which is
#: CatBoost's own structure (`compute_scores.cu:201-219`). That flag IS
#: implemented and gated per cell and per model by
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
#: reason -- `secondDerAsWeights` was unimplemented, so each silently fit its
#: non-Newton twin -- and left it when that flag landed, gated by
#: `checks/second_der_weights_check.mojo`.
_UNSUPPORTED_SCORE_FUNCTIONS = {
    "SolarL2": (
        "the greedy searcher this implementation runs has no SolarL2 arm and falls "
        "through to Cosine (greedy_search_helper.mojo:3134-3162), so the "
        "fit would silently be a Cosine fit. Its calcer exists only on the "
        "pointwise searcher (pointwise_scores.mojo:1569)"
    ),
    "LOOL2": (
        "the greedy searcher this implementation runs has no LOOL2 arm and falls "
        "through to Cosine (greedy_search_helper.mojo:3134-3162), so the "
        "fit would silently be a Cosine fit"
    ),
    "SatL2": (
        "the greedy searcher this implementation runs has no SatL2 arm and falls "
        "through to Cosine (greedy_search_helper.mojo:3134-3162), so the "
        "fit would silently be a Cosine fit"
    ),
}

#: their `EGrowPolicy` spellings (`oblivious_tree_options.cpp:23`), in
#: their enum order -- the binding carries the ORDINAL. `Region`, their
#: fourth, is absent: no lane implements it (`greedy_search_helper.cpp:325-350`).
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


#: their `EOverfittingDetectorType` spellings. `None` here means the
#: option was not given, and that is NOT the same as `od_type="None"`:
#: unset lets `TOverfittingDetectorOptions::Load`
#: (`overfitting_detector_options.cpp:24-32`) pick the type from whichever
#: of `od_pvalue` / `od_wait` was given, while "None" turns the detector
#: off outright. Wilcoxon is theirs and is not implemented.
OD_TYPES = ("None", "IncToDec", "Iter")


# DEVIATION 2330: the small exact replacements for what NumPy used to do
# on the host side of this module. None of them touches a result bit.


def _f32(value):
    """`float(np.float32(value))`: one round-to-nearest-even to float32
    (an underflowing value becomes 0.0, exactly as the cast does)."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


#: `np.finfo(np.float32).max`, as the literal.
_F32_MAX = 3.4028234663852886e+38
#: `np.iinfo(np.uint32).max`, as the literal.
_U32_MAX = 4294967295


# Maintainer notes moved out of the public docstring (CatBoost 1.2.10 source references):
# - n_estimators None: 1000 under SymmetricTree (boosting_options.cpp:13); 100 under Depthwise/Lossguide,
#   this library's earlier default, which those policies keep.
# - grow_policy: oblivious_tree_options.cpp:23. Depthwise/Lossguide are grown by the same
#   TGreedySubsetsSearcher<TNonSymmetricTree> CatBoost's GPU uses (pointwise_non_symmetric.cpp), then take
#   its estimation arm (bins off the model, the loss's leaf estimator, AddBinModelValues). Refusals follow
#   CatBoost: losses with no non-symmetric GPU trainer (Lq, MultiClass; pointwise_non_symmetric.cpp:7-29
#   lists the eleven that have one; train.cpp:279 is the error) and use_pointwise_searcher=True (the
#   doc-parallel oblivious searcher). Region is not implemented. DEVIATION 259.
# - feature_fraction default 1.0 preserves existing model bits and ABI layout.
# - max_leaves: default 31 (oblivious_tree_options.cpp:24), cap 65536 (:130-133); CatBoost pins it to
#   2**max_depth for other policies and refuses another value (catboost_options.cpp:993-1001).
# - min_data_in_leaf: oblivious_tree_options.cpp:25; terminal test is size <= value
#   (greedy_search_helper.cpp:693); IsTerminalLeaf guards it with Policy != SymmetricTree (:685) and
#   discards it on oblivious trees, so any value but 1 is refused there.
# - learning_rate None: UpdateLearningRate (libs/train_lib/options_helper.cpp:269-288), rate =
#   Round(min(exp(A*log(n_rows) + B) * exp(C*log(n_estimators) + D) / exp(C*log(1000) + D), 0.5), 6)
#   with GPU coefficients keyed on loss, resolved use_best_model and boost_from_average (:221-243,
#   :252-262); otherwise 0.03 (boosting_options.cpp:10).
# - l2_leaf_reg None: catboost_options.cpp:34-37, :166-172; explicit value disables auto rate
#   (options_helper.cpp:278).
# - border_count: GPU default (data_processing_options.cpp:16); 254 on their CPU.
# - feature_border_type: data_processing_options.cpp:15; MakeBinarizer dispatch
#   (library/cpp/grid_creator/binarization.cpp:114-134). Border selection runs on the host here as in
#   CatBoost, the same host function on every vendor and on the CPU host path; all seven types reproduce
#   CatBoost 1.2.10 borders bit for bit on 294 cases (checks/border_types_check.mojo). A value below
#   2**-126 is binned as zero (the flush GreedyLogSum's border build already applies).
# - loss_alpha: MAE fixes alpha at 0.5 (pointwise_target_impl.h:272-275).
# - boost_from_average None: AdjustBoostFromAverageDefaultValue (options_helper.cpp:353-374); accepted
#   losses catboost_options.cpp:705-709. MAE/Quantile/MAPE constant is CalcSampleQuantile with the 1e-6
#   delta adjust, host code shared by device fit and CPU host path; reproduces CatBoost 1.2.10 CPU
#   get_scale_and_bias()[1] by bits on 40 cases. Quantile level enters as float32 widened to double
#   (theirs parses a double), so a level a float does not hold exactly (0.3) can move a quantile that
#   sits exactly on the boundary.
# - leaf_estimation_method Newton refusal: catboost_options.cpp:588-601 (second derivative zero).
# - leaf_estimation_iterations: UpdateLeavesEstimationIterations (options_helper.cpp:290-307),
#   IsSmallIterationCount (catboost_options.h:88-90).
# - bootstrap_type None under SymmetricTree: Bayesian, GPU default (bootstrap_options.h:18; the MVS
#   default of catboost_options.cpp:782-787 is CPU only). bagging_temperature bootstrap_options.h:16;
#   subsample default 0.66 bootstrap_options.h:15.
# - cat_features dispatch: binarizations_manager.cpp:106-115. CatBoost shuffles the learn pool at load
#   when there are categorical features and no time column (preprocess.cpp:161-199), CPU-side upstream of
#   catboost/cuda; this implementation does not, so CTRs read caller row order
#   (gbdt/data/permutation.mojo).
# - od_type: overfitting_detector_options.cpp:24-32; IncToDec inactive when stop_pvalue is 0
#   (IsActive() is Threshold > 0).
# - use_best_model default: options_helper.cpp:106-108 (ShrinkToBestIteration). CatBoost warns without
#   an eval set; we raise because a warning on a returned model is invisible.
# - best_model_min_trees: output_file_options.cpp:77, boosting_progress_tracker.cpp:162.
# - score_function None: Cosine for SymmetricTree/Depthwise (oblivious_tree_options.cpp:20), NewtonL2
#   for Lossguide (catboost_options.cpp:980-991). TCosineScoreCalcer is the only calcer with the
#   random_strength noise term (score_calcers.cuh:152-167); TL2ScoreCalcer (:40-69) has none. Newton
#   spellings run the same calcers with weight*der2 in the weight plane (secondDerAsWeights,
#   greedy_search_helper.cpp:286-296), so for RMSE each is bit-identical to its twin. SolarL2, LOOL2,
#   SatL2: see _UNSUPPORTED_SCORE_FUNCTIONS.
# - nan_mode: data_processing_options.cpp:26.
# - random_strength: oblivious_tree_options.cpp:17. On the greedy searcher CatBoost's noise largely
#   cancels in the gain (compute_scores.cu:84-134); live on the pointwise searcher, where noise is drawn
#   before the bootstrap (oblivious_tree_doc_parallel_structure_searcher.cpp:200-218).
# - border_build_max_samples: data_processing_options.cpp:37.
# - class_weights: multiplied into row weight (target/data_providers.cpp:168).
# - boosting_type: boosting_options.cpp:16; Ordered default catboost_options.cpp:802-807,
#   UpdateBoostingTypeOption (defaults_helper.h:33-42); GPU Plain-only cases :949-978. Ordered is their
#   GPU TDynamicBoosting (gbdt/methods/ordered_boosting.mojo carries the account). The CPU host path
#   restates Ordered at unit weights on numeric columns and agrees with Apple Metal bit for bit (the
#   gbdt-ordered* identity lanes). parallel_ensemble.fit_boosting partitions its fold histograms across
#   GPUs as it does OrderedRMSE's.
# - fold_len_multiplier: boosting_options.cpp:11. fold_permutation_block: boosting_options.cpp:12; GPU
#   64 (cuda/train_lib/train.cpp:115-118), halved while block*128 > rows (dynamic_boosting.h:115-128).
# - permutation_count: boosting_options.cpp:14 (UpdateGpuSpecificDefaults).
# - ctr_estimation_permutation_id: doc_parallel_boosting.h:101-103.
# - model_: the text format check-model-io gates bit-for-bit; floats carry hex bits beside the decimal
#   because String(Float32) on this toolchain is one ULP wrong for 0.46% of values.
# - loss_curve_: CatBoost's functionValue negated and divided by the row count.
# - best_iteration_: the error tracker's best (error_tracker.h:73-75).
class GradientBoosting(NumericModeMixin):
    """Gradient-boosted decision trees with CatBoost's GPU learner as the reference.

    Fits an ensemble of trees with CatBoost's growth policies (oblivious
    ``SymmetricTree``, ``Depthwise``, ``Lossguide``), losses and leaf estimators.
    ``X`` is ``(n_samples, n_features)`` and is converted to float32.

    Under the default ``grow_policy='SymmetricTree'`` every default follows
    CatBoost's GPU learner (``task_type='GPU'``), not scikit-learn, and several
    of these change results: 1000 iterations, an automatically chosen learning
    rate, depth 6, ``l2_leaf_reg=3``, 128 borders, ``random_strength=1``,
    Bayesian bootstrap, and Ordered boosting on smaller pools. ``predict``
    returns raw scores for every loss; use ``predict_proba`` or
    ``predict_classes`` for classification output.

    Supports the ``numeric_mode`` parameter (``'identical'``
    or ``'fast'`` on Apple GPUs).

    Parameters
    ----------
    loss : str, default 'RMSE'
        One of ``LOSSES``, in CatBoost's spellings: ``MultiClass``,
        ``MultiClassOneVsAll``, ``RMSE``, ``Logloss``, ``CrossEntropy``,
        ``Quantile``, ``MAE``, ``LogLinQuantile``, ``MAPE``, ``Poisson``, ``Lq``,
        ``Expectile``, ``Tweedie``, ``Huber``, ``QueryRMSE``, ``PairLogit``,
        ``YetiRank``, ``MultiRMSE``. Four require a parameter: ``Lq`` needs
        ``loss_q``, ``Huber`` needs ``loss_delta``, ``Tweedie`` needs
        ``loss_variance_power`` and ``Expectile`` needs ``loss_alpha``; a
        missing one raises ``ValueError``. ``MultiRMSE`` is multi-target
        regression: ``y`` is ``(n_samples, n_targets)`` with ``n_targets >= 2``
        and ``predict`` returns ``(n_samples, n_targets)`` (Plain SymmetricTree,
        numeric features, no eval set, ``boost_from_average`` unset or False).
    n_estimators : int or None, default None
        CatBoost's ``iterations``. None is 1000 under SymmetricTree (CatBoost's
        default) and 100 under Depthwise and Lossguide.
    max_depth : int, default 6
        CatBoost's ``depth``. Under SymmetricTree the tree is oblivious, so it
        has exactly ``2 ** max_depth`` leaves; under Depthwise it bounds the
        number of levels; under Lossguide it bounds any one leaf's depth and
        ``max_leaves`` is what stops the tree.
    learning_rate : float or None, default None
        None under SymmetricTree is CatBoost's GPU auto-selection: when
        ``l2_leaf_reg``, ``leaf_estimation_method`` and
        ``leaf_estimation_iterations`` are also unset and the loss is RMSE,
        Logloss or MultiClass, the rate is computed from the row count and
        ``n_estimators`` with CatBoost's GPU coefficients (capped at 0.5,
        rounded to six decimals); otherwise 0.03. Under Depthwise and
        Lossguide None is 0.03. The value used is stored in ``learning_rate_``.
    l2_leaf_reg : float or None, default None
        None takes the loss's CatBoost default: 3.0, and 0 for YetiRank. An
        explicit value is used as given and, as in CatBoost, turns the
        learning-rate auto-selection off.
    border_count : int, default 128
        Quantization bins per numeric feature (CatBoost's GPU default; their
        CPU default is 254).
    random_state : int, default 0
        Random seed.
    loss_alpha : float, optional
        Quantile level for ``Quantile`` and ``LogLinQuantile`` (default 0.5),
        and required for ``Expectile``. Ignored by ``MAE``, whose alpha is
        fixed at 0.5.
    loss_q, loss_delta, loss_variance_power : float, optional
        ``Lq``'s q, ``Huber``'s delta, ``Tweedie``'s variance power.
    loss_border : float, optional
        ``Logloss``'s target threshold, default 0.5.
    leaf_estimation_method : {'Newton', 'Gradient', 'Exact', 'Simple'}, optional
        None lets the loss decide, as in CatBoost. ``'Newton'`` is refused for
        Quantile, MAE, LogLinQuantile, MAPE and Lq with q < 2, whose second
        derivative is zero.
    leaf_estimation_iterations : int, optional
        None lets the loss decide; under SymmetricTree, fewer than 200
        iterations on fewer than 20 features then sets it to 1, as CatBoost
        does.
    bootstrap_type : {'Bayesian', 'Bernoulli', 'Poisson', 'No'}, optional
        None under SymmetricTree is ``'Bayesian'`` (CatBoost's GPU default);
        under Depthwise and Lossguide it means no row sampling. ``'Bernoulli'``
        and ``'Poisson'`` take ``subsample``; ``'Bayesian'`` takes
        ``bagging_temperature``. Ranking losses apply row weights after
        computing their grouped gradients.
    bagging_temperature : float, default 1.0
        Read by ``bootstrap_type='Bayesian'`` only; a non-default value with
        any other bootstrap raises ``ValueError``.
    subsample : float, optional
        Read by Bernoulli and Poisson only (default 0.66). Passing it with
        Bayesian (including the default) or no bootstrap raises ``ValueError``.
    cat_features : sequence of int, optional
        Column indices holding dense category codes ``0..k-1``. As in
        CatBoost, each column is one-hot encoded when its cardinality is small
        and otherwise uses target statistics (CTRs). CTRs are computed over
        the rows in the order you pass them; CatBoost shuffles the pool first
        and this does not. Rows sorted by target give a worse estimator, so
        shuffle before fitting and record the order you used. Without
        ``cat_features`` row order does not matter.
    one_hot_features : sequence of int, optional
        Column indices to one-hot encode. An index out of range raises
        ``ValueError``.
    od_type : {'None', 'IncToDec', 'Iter'}, optional
        The overfitting detector. Unset is not the same as ``'None'``: as in
        CatBoost, the type follows from what was given (``od_wait`` alone
        means ``'Iter'``, ``od_pvalue`` alone means ``'IncToDec'``, neither
        means ``'None'``). Requires ``eval_set``; otherwise it raises.
    od_pvalue : float, optional
        IncToDec's threshold. The default 0 leaves the detector inactive.
    od_wait : int, optional
        Iterations without a new best before stopping, default 20.
    use_best_model : bool, optional
        Truncate the fitted ensemble to the best held-out iteration. Unset is
        not False: with an ``eval_set`` whose target is not constant it
        defaults to True, as in CatBoost. True without an ``eval_set`` raises.
    best_model_min_trees : int, default 1
        The truncation never keeps fewer than this many trees.
    score_function : {'Cosine', 'L2', 'NewtonCosine', 'NewtonL2'}, optional
        The split score. None is CatBoost's default for the policy: Cosine for
        SymmetricTree and Depthwise, NewtonL2 for Lossguide. Only Cosine
        scores carry ``random_strength`` noise. For RMSE each Newton spelling
        gives the same model as its plain twin. CatBoost's SolarL2, LOOL2 and
        SatL2 are refused by name.
    nan_mode : {'Min', 'Max', 'Forbidden'}, default 'Min'
        Where a NaN sorts against the borders: below every border, above, or
        ``'Forbidden'`` to raise on a NaN, as CatBoost does.
    random_strength : float or None, default None
        CatBoost's score noise. None is 1.0 under SymmetricTree and 0.0 under
        Depthwise and Lossguide. An explicit value above 0 is refused with
        ``score_function='L2'`` or ``'NewtonL2'``, which have no noise term;
        unset resolves to 0 there.
    use_pointwise_searcher : bool, default False
        Grow with CatBoost's doc-parallel oblivious searcher instead of the
        greedy subsets searcher. It returns tree structure only, so it always
        runs the leaf estimator.
    boost_from_average : bool or None, default None
        Start every row at the loss's optimal constant, stored as ``bias_``.
        None is CatBoost's default: True for RMSE, MAE, Quantile and MAPE,
        False otherwise. True is accepted for those four and for Logloss and
        CrossEntropy.
    border_build_max_samples : int, default 200000
        Rows subsampled for the border search only; every row is still
        fitted. 0 uses every row for the borders too.
    class_weights : sequence of float, optional
        One weight per class, multiplied into each row's weight (so it
        composes with ``sample_weight``). The length must match the label set
        or the fit raises. Refused with ``loss='RMSE'``, as in CatBoost.
    permutation_count : int, optional
        CatBoost's ``permutation_count``. None resolves in ``fit``: 4 for an
        Ordered fit and for CTR categoricals, 1 for a Plain fit without CTRs.
        Only the categorical path and Ordered boosting read it; a Plain fit
        with no ``cat_features`` refuses it.
    ctr_estimation_permutation_id : int, optional
        Which permutation estimates the CTRs. None means
        ``permutation_count - 1``. Only the categorical path reads it; it is
        refused otherwise.
    grow_policy : {'SymmetricTree', 'Depthwise', 'Lossguide'}, default 'SymmetricTree'
        ``SymmetricTree`` grows oblivious trees; ``Depthwise`` splits every
        improving leaf of a level on its own best split; ``Lossguide`` splits
        one leaf per step, the best-scoring one, until ``max_leaves``. The
        non-symmetric policies refuse, as CatBoost does, losses without a
        non-symmetric GPU trainer (``Lq``, ``MultiClass``) and
        ``use_pointwise_searcher=True``. ``'Region'`` raises
        ``NotImplementedError``.
    max_leaves : int, optional
        The Lossguide leaf budget (CatBoost's default 31, cap 65536). Read
        under Lossguide only; for other policies any value other than
        ``2 ** max_depth`` is refused, as in CatBoost.
    min_data_in_leaf : int, default 1
        A leaf with at most this many rows is terminal. Used under Depthwise
        and Lossguide only; under SymmetricTree any value but 1 is refused.
    min_split_gain : float or None, default None
        Strict lower bound on a split's improvement in the selected score
        function, Depthwise and Lossguide only. None keeps CatBoost's growth,
        including Lossguide's ability to split without positive gain; 0
        requires positive improvement. Units depend on ``score_function``, so
        it is not interchangeable with XGBoost's ``gamma`` or LightGBM's
        ``min_gain_to_split``. Must be finite and nonnegative.
    min_child_hessian : float or None, default None
        Minimum weighted Hessian sum in each candidate child (training and
        bootstrap weights included; equality allowed). Requires Depthwise or
        Lossguide, NewtonL2 or NewtonCosine scores, and RMSE, Logloss or
        CrossEntropy.
    feature_fraction : float, default 1.0
        Per-tree numeric feature subsampling in (0, 1]: a deterministic subset
        of nonconstant features is chosen per tree. Values below 1 require
        numeric features only (categorical and one-hot columns raise
        ``NotImplementedError``).
    feature_border_type : str, default 'GreedyLogSum'
        How numeric borders are chosen, one of ``BORDER_TYPES``: GreedyLogSum
        and GreedyMinEntropy (greedy splitting), MaxLogSum and MinEntropy
        (exact dynamic program), Median (quantiles), Uniform (equal width) and
        UniformAndQuantiles (half of each). Each matches CatBoost's borders.
        CTR columns keep their own grids.
    boosting_type : {'Plain', 'Ordered'} or None, default None
        None under SymmetricTree is CatBoost's data-dependent GPU default:
        Ordered when the pool has fewer than 50,000 rows and there are at
        least 500 iterations, otherwise Plain; always Plain for the multiclass
        losses, the L2 scores, ``use_pointwise_searcher`` and
        ``feature_fraction < 1``. None under Depthwise and Lossguide is Plain.
        The value used is stored in ``boosting_type_``.

        Ordered follows CatBoost's GPU ordered boosting: ``permutation_count``
        permutations, growing folds with leaves estimated on each fold's
        prefix, and the exported model estimated on the last permutation. It
        supports every pointwise loss, Cosine and NewtonCosine scores, every
        bootstrap, ``random_strength``, every border type and NaN mode,
        sample and class weights, one-hot categorical columns,
        ``boost_from_average`` and eval sets. On a CPU-only install, sample
        weights, class weights and one-hot columns are refused under Ordered.
        Refused, as in CatBoost: Depthwise and Lossguide, the multiclass
        losses, L2 and NewtonL2 scores, the Exact leaf estimator. Not
        implemented, refused by name: CTR categorical columns and the ranking
        losses. Its random streams are this library's, so an Ordered model
        matches CatBoost in behavior, not in bits.
    fold_len_multiplier : float, default 2.0
        Ordered only: the fold growth factor, greater than 1. Refused by a
        Plain fit.
    fold_permutation_block : int or None, default None
        Ordered only. None or 0 means 64 from 50,000 rows up (halved while
        ``block * 128 > rows``) and 1 below 50,000 rows.

    Attributes
    ----------
    model_ : str
        The fitted ensemble as text. It is a plain string you can store or
        pass to another process; floats carry their exact bits, so a round
        trip does not change predictions.
    loss_curve_ : ndarray
        The training loss after each iteration.
    test_loss_curve_ : ndarray or None
        The held-out loss after each iteration, or None without an
        ``eval_set``. This is the curve the overfitting detector reads.
    learning_rate_ : float
        The learning rate the fit used.
    boosting_type_ : str
        ``'Plain'`` or ``'Ordered'``, the boosting type the fit used.
    best_iteration_ : int
        Index of the lowest ``test_loss_curve_`` entry, or of the lowest
        training loss without an eval set. With ``best_model_min_trees`` above
        it, the model holds more than ``best_iteration_ + 1`` trees.
    stopped_early_ : bool
        Whether the detector stopped before ``n_estimators``.
    bias_ : float
        The starting constant (0.0 unless ``boost_from_average``).
    n_features_in_ : int
        Number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = (X[:, 0] + X[:, 1] > 1).astype(np.float32)
    >>> model = mojolearn.GradientBoosting(loss="Logloss", n_estimators=100).fit(X, y)
    >>> proba = model.predict_proba(X)
    """

    #: This family's binding, for `NumericModeMixin._bind`.
    _BINDING = "_mojolearn_gbdt"

    def _sklearn_instance_target_tags(self):
        # loss='MultiRMSE' fits a 2-D y (n_samples, n_targets): scikit-learn's
        # multi_output tag, so a column y is not raveled before the fit
        # refuses it by name (n_targets >= 2)
        return {"multi_output": getattr(self, "loss", None) in MULTI_REGRESSION_LOSSES}

    def __init__(
        self,
        loss="RMSE",
        n_estimators=None,
        max_depth=6,
        learning_rate=None,
        l2_leaf_reg=None,
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
        random_strength=None,
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
        feature_border_type="GreedyLogSum",
        boosting_type=None,
        fold_len_multiplier=2.0,
        fold_permutation_block=None,
    ):
        # The constructor's checks and its stored state are the native
        # owner's (bindings/trees_public_gbdt_fit.mojo `gbdt_initialize`),
        # shared with the scikit-learn adapters' learner.
        _TREES(self, "trees_gbdt_init", (
            loss, n_estimators, max_depth, learning_rate, l2_leaf_reg, border_count, random_state, loss_alpha,
            loss_q, loss_delta, loss_variance_power, loss_border, leaf_estimation_method,
            leaf_estimation_iterations, bootstrap_type, bagging_temperature, subsample, cat_features,
            one_hot_features, od_type, od_pvalue, od_wait, use_best_model, best_model_min_trees,
            score_function, nan_mode, random_strength, use_pointwise_searcher, boost_from_average,
            border_build_max_samples, class_weights, permutation_count, ctr_estimation_permutation_id,
            grow_policy, max_leaves, min_data_in_leaf, min_split_gain, min_child_hessian, feature_fraction,
            feature_border_type, boosting_type, fold_len_multiplier, fold_permutation_block,
        ), _TREES_CONTEXT)

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
    def _resolved_boosting_type(self, n_rows):
        """The boosting type a fit on `n_rows` rows uses (`boosting_type_`).

        Explicit is used as given. Unset under SymmetricTree is CatBoost's
        GPU chain: `SetNotSpecifiedOptionsToDefaults` defaults it to Ordered
        (`catboost_options.cpp:802-807`) except for the multiclass losses,
        which their GPU trains Plain only (`:949-967`), and except for score
        functions with no ordered kernel (`:972-978`); then
        `UpdateBoostingTypeOption` (`defaults_helper.h:33-42`) makes it Plain
        at 50,000 learn rows or more, or below 500 iterations. Two options of
        this library that CatBoost's GPU does not have resolve it to Plain,
        because they name Plain arms: `use_pointwise_searcher` (the
        doc-parallel searcher) and `feature_fraction < 1`. Unset under
        Depthwise and Lossguide is Plain, as theirs (`:757-759`)."""
        return _TREES(self, "trees_gbdt_boosting_type", (n_rows,), _TREES_CONTEXT)

    # group_id keys follow get_id_object_bytes_string_representation; consecutive-group refusal is
    # libs/data/objects.cpp:60-87. A QueryRMSE fit without groups mirrors TWithoutQueriesGrouping
    # (gpu_data/doc_parallel_dataset.h:26-38). YetiRank derivative draws come from a stream of random_state
    # kept apart from the searcher's, so its trees cannot match the reference GPU bit for bit. sample_weight
    # x class_weights follows target/data_providers.cpp:168 (rawWeights * rawGroupWeights * classWeights);
    # there are no group weights here. use_best_model default: options_helper.cpp:106-108.
    def fit(self, X, y, sample_weight=None, eval_set=None, group_id=None,
            subgroup_id=None, pairs=None, pairs_weight=None):
        """Fit the ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, converted to float32.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Targets; 2-D only for ``loss='MultiRMSE'``.
        sample_weight : array-like of shape (n_samples,), optional
            Per-row weights, None meaning all ones. Multiplied with
            ``class_weights`` where both are given.
        eval_set : tuple (X_eval, y_eval) or list of one such tuple, optional
            Held-out data for ``test_loss_curve_``, the overfitting detector and
            ``use_best_model``. Unlike CatBoost, only one eval set is accepted;
            more than one raises. Passing it makes ``use_best_model`` default to
            True, so ``model_`` keeps the trees up to the best held-out iteration;
            pass ``use_best_model=False`` to keep every tree.
        group_id : sequence, optional
            One query id per row (string or integer; integers compare by their
            decimal spelling, so 7 and "7" are one group; floats are refused).
            Rows of a group must be consecutive. Read by ``QueryRMSE``,
            ``PairLogit`` and ``YetiRank`` (SymmetricTree, no categorical
            features, no eval set); every other loss refuses it. A QueryRMSE fit
            with no groups, or one row per group, predicts zero.
        subgroup_id : sequence, optional
            Not read by any loss; refused by name.
        pairs : sequence of [winner, loser], optional
            ``PairLogit`` only: integer row-index pairs, each inside one group.
            Requires ``group_id``. Without it, pairs are generated from the groups
            and ``y`` (every two rows of a query with different grades, the higher
            grade winning, weighted by the query's first row weight).
        pairs_weight : sequence of float, optional
            One weight per pair, default 1.0.

        Returns
        -------
        self
            The fitted estimator.

        Notes
        -----
        ``YetiRank`` uses CatBoost's defaults (10 permutations, decay 0.85,
        ``l2_leaf_reg`` 0 unless given, Newton leaves at one iteration, which it
        refuses to change), refuses a query over 1023 rows, and leaves
        ``loss_curve_`` at zero.
        """
        return _TREES(self, "trees_gbdt_fit", (X, y, sample_weight, eval_set, group_id, subgroup_id, pairs,
                                               pairs_weight), _TREES_CONTEXT)

    def _resolved_learning_rate(self, n_rows, n_eval_rows=0, eval_target_constant=False):
        """The learning rate a fit on `n_rows` rows uses (`learning_rate_`),
        resolved by the native fit owner."""
        return _TREES(self, "trees_gbdt_learning_rate", (n_rows, n_eval_rows, eval_target_constant), _TREES_CONTEXT)

    def _params(self, n_rows, n_features, n_flags, n_weights=0, n_eval_rows=0, eval_target_constant=False):
        """The parameter list the native fit hands `gbdt_fit`, packed by the
        native fit owner (inspection of the packing)."""
        return _TREES(self, "trees_gbdt_params",
                      (n_rows, n_features, n_flags, n_weights, n_eval_rows, eval_target_constant), _TREES_CONTEXT)

    # -- the device-resident parsed model (DEVIATION 2980) ------------------
    #
    # `gbdt/resident_model.mojo`: the first `predict` or `predict_proba`
    # after a fit or a load parses the text once and uploads the packed
    # ensemble once; every later call searches through an integer handle
    # held here as `_resident = (binding, text, sha256, handle)`. The rules
    # are `neighbors.py`'s for the k-NN index (DEVIATION 2921): released on
    # refit, released when the instance is collected, never carried through
    # pickle or deepcopy, and a handle is meaningful only for the binding
    # (tier, vendor) that minted it.

    def __getstate__(self):
        """A pickle or a deepcopy carries no device handle: the integer is
        meaningful only in the process and registry that minted it, and an
        unpickled instance holding it could release ANOTHER model's live
        copy. The copy prepares its own at its first call."""
        state = self.__dict__.copy()
        state.pop("_resident", None)
        return state

    def predict(self, X):
        """Return raw scores (not probabilities) as a float32 ``mojolearn.Array``.

        For a single-output loss the result has shape ``(n_samples,)``. For
        ``MultiClass`` it is ``(n_samples, n_classes - 1)``: the last class's
        score is fixed at zero and not returned, so use ``predict_proba`` for
        probabilities. ``MultiClassOneVsAll`` returns ``(n_samples, n_classes)``,
        one raw score per class. ``MultiRMSE`` returns ``(n_samples, n_targets)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        mojolearn.Array
        """
        return _TREES(self, "trees_gbdt_predict", (X,), _TREES_CONTEXT)

    # MultiClass softmax is their Probability (eval_processing.h:214-221); OneVsAll elementwise sigmoid is
    # MultiProbability (:222-226). The device applies the transform, so the max-subtraction matches
    # MultiLogitValAndFirstDerImpl rather than being re-derived here.
    def predict_proba(self, X):
        """Return class probabilities of shape ``(n_samples, n_classes)``.

        Defined for ``Logloss`` and ``CrossEntropy`` (sigmoid), ``MultiClass``
        (softmax over all classes, columns sum to 1) and ``MultiClassOneVsAll``
        (independent sigmoids per class; columns do not sum to 1, because each
        class was fitted as a separate problem). Every other loss raises. This
        is the same transform as CatBoost's ``prediction_type='Probability'``.
        Columns are in class-code order. The result is a float32
        ``mojolearn.Array`` for the two multiclass losses and float64 for
        Logloss and CrossEntropy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        mojolearn.Array
        """
        return _TREES(self, "trees_gbdt_predict_proba", (X,), _TREES_CONTEXT)

    def predict_classes(self, X):
        """Return the predicted class codes (argmax of ``predict_proba``).

        A separate method because ``predict`` returns raw scores for every loss.
        Ties go to the first class. Raises ``ValueError`` for a model fitted with
        a non-classification loss.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        mojolearn.Array
            int64 dense class codes of shape ``(n_samples,)``.
        """
        return _TREES(self, "trees_gbdt_predict_classes", (X,), _TREES_CONTEXT)

    def get_tree_leaf_counts(self):
        """Return the number of leaves in each tree, as in CatBoost.

        Returns
        -------
        mojolearn.Array
            uint32 array with one entry per tree.
        """
        return _TREES(self, "trees_gbdt_leaf_counts", (), _TREES_CONTEXT)

    def get_leaf_values(self):
        """Return the stored leaf values, tree-major then leaf-major.

        For multiclass models each leaf's dimensions are consecutive:
        ``MultiClass`` stores ``n_classes - 1`` values per leaf and
        ``MultiClassOneVsAll`` stores ``n_classes``. Values are the stored
        float32 values converted exactly to float64.

        Returns
        -------
        mojolearn.Array
            float64 array.
        """
        return _TREES(self, "trees_gbdt_leaf_values", (), _TREES_CONTEXT)

    # The payload is model_, the text format check-model-io gates bit-for-bit; the loader reads the hex-bits
    # half of each float, and nothing here formats a number. The model text carries the quantization borders
    # (one `feature ... borders ...` table per feature), which is why gbdt_predict takes only the text and
    # raw rows. The text also records per-iteration learn losses on its `loss` lines.
    def save(self, path):
        """Write the fitted model to ``path`` as an ``.npz`` file.

        The saved model predicts bit for bit like the original. The numeric mode
        the model used is saved with it. ``loss_curve_`` and ``test_loss_curve_``
        are not restored by ``load``.

        Parameters
        ----------
        path : str or path-like
            Destination file.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by ``save``.

        The loaded model predicts; it does not carry the training configuration
        or the loss curves. A file whose stored metadata disagrees with its model
        is refused as corrupt.

        Parameters
        ----------
        path : str or path-like

        Returns
        -------
        GradientBoosting
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)

# `save` and `load` are one native call each: bindings/archive_schema_native.mojo
# writes this schema, the members, dtypes and refusals of the 0.8.55 Python
# `save` (the model text as UTF-8 `<u1` bytes, the numeric mode `predict` would
# use now, the int64 meta and the float64 bias), and
# bindings/host_archive_native.mojo `host_archive_public_gbdt` builds the loaded
# model, reading the output dimension through the GBDT binding connected here.
_native_archive_bridge.register(GradientBoosting, _MODEL_FORMAT, (
    ("need", "model_", "AttributeError", "'{cls}' object has no attribute 'model_'"),
    ("need_set", "model_", "RuntimeError", "mojolearn: save() before fit()"),
    ("format", "format", _MODEL_FORMAT),
    ("mode", "numeric_mode", "mojolearn: "),
    ("estimator", "estimator"),
    ("array", "loss", "loss"),
    ("utf8", "model", "model_"),
    ("i8", "meta", (("int", "n_features_in_"), ("int", "approx_dim_"),
                    ("none", "n_classes_", -1), ("none", "best_iteration_", -1),
                    ("bool", "stopped_early_"))),
    ("f8", "bias", (("float", "bias_"),)),
), host_archive_gbdt_bindings=_native_trees_bridge.bindings("_mojolearn_gbdt"))

# The fixed option tables `fit` reads natively (bindings/trees_public_gbdt_fit.mojo).
_native_trees_bridge.register(GradientBoosting, "gbdt", trees_gbdt_tables=dict(
    multi_output_losses=MULTI_OUTPUT_LOSSES, multi_regression_losses=MULTI_REGRESSION_LOSSES,
    querywise_losses=_QUERYWISE_LOSSES, od_types=OD_TYPES, border_types=BORDER_TYPES,
    leaf_estimation_names=_LEAF_ESTIMATION_NAMES, score_function_names=_SCORE_FUNCTION_NAMES,
    grow_policy_codes=_GROW_POLICY_CODES, gpu_auto_learning_rate=_GPU_AUTO_LEARNING_RATE,
    cpu_auto_learning_rate=_CPU_AUTO_LEARNING_RATE, boost_from_average_losses=_BOOST_FROM_AVERAGE_LOSSES,
    legacy_learning_rate=_LEGACY_LEARNING_RATE, catboost_learning_rate=_CATBOOST_LEARNING_RATE,
    catboost_random_strength=_CATBOOST_RANDOM_STRENGTH, legacy_random_strength=_LEGACY_RANDOM_STRENGTH,
    ordered_max_rows=_ORDERED_MAX_ROWS, ordered_min_iterations=_ORDERED_MIN_ITERATIONS,
    init_names=('loss', 'n_estimators', 'max_depth', 'learning_rate', 'l2_leaf_reg', 'border_count', 'random_state', 'loss_alpha', 'loss_q', 'loss_delta', 'loss_variance_power', 'loss_border', 'leaf_estimation_method', 'leaf_estimation_iterations', 'bootstrap_type', 'bagging_temperature', 'subsample', 'cat_features', 'one_hot_features', 'od_type', 'od_pvalue', 'od_wait', 'use_best_model', 'best_model_min_trees', 'score_function', 'nan_mode', 'random_strength', 'use_pointwise_searcher', 'boost_from_average', 'border_build_max_samples', 'class_weights', 'permutation_count', 'ctr_estimation_permutation_id', 'grow_policy', 'max_leaves', 'min_data_in_leaf', 'min_split_gain', 'min_child_hessian', 'feature_fraction', 'feature_border_type', 'boosting_type', 'fold_len_multiplier', 'fold_permutation_block'),
    losses=LOSSES, required_param=_REQUIRED_PARAM, bootstrap_types=BOOTSTRAP_TYPES,
    grow_policies=GROW_POLICIES, non_symmetric_losses=_NON_SYMMETRIC_LOSSES,
    unsupported_score_functions=_UNSUPPORTED_SCORE_FUNCTIONS, score_functions=SCORE_FUNCTIONS,
    nan_modes=NAN_MODES, boosting_types=BOOSTING_TYPES, catboost_iterations=_CATBOOST_ITERATIONS,
    legacy_iterations=_LEGACY_ITERATIONS, catboost_bootstrap=_CATBOOST_BOOTSTRAP,
))


# Former docstring: Experimental one-tree, depth-two FeatureFreq combination estimator. Source columns
# must hold dense categorical codes; non-source columns are ordinary numeric split candidates. Separate
# from GradientBoosting; selecting it cannot change standard fit behavior or imply support for
# deeper/general CatBoost combinations. fit(sample_weight=) accepts finite non-negative row weights;
# structure scores and leaf values both use the same weights.
class ExperimentalTwoLevelFeatureFreq(GradientBoosting):
    """One-tree, depth-two RMSE model over a combination of categorical source columns.

    Parameters
    ----------
    sources : sequence of int
        At least two distinct column indices holding dense category codes
        ``0..k-1``.
    learning_rate : float, default 0.03
    l2_leaf_reg : float, default 3.0
    random_state : int, default 0
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
            parsed = tuple(int(i) for i in sources)  # glue: parses the sources argument tuple
        except (TypeError, ValueError):
            raise ValueError("mojolearn: sources must be integer indices") from None
        if len(parsed) < 2 or len(set(parsed)) != len(parsed):
            raise ValueError("mojolearn: sources need at least two unique indices")
        self.sources = parsed

    def fit(self, X, y, sample_weight=None):
        """Fit the tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Finite values; source columns hold dense codes ``0..k-1`` and the
            other columns must not be constant.
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), optional
            Finite, non-negative row weights with a positive sum.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_gbdt_feature_freq_fit", (X, y, sample_weight), _TREES_CONTEXT)

# Former docstring: Numeric ordered RMSE with one explicit row permutation. Fits symmetric trees using
# independent prefix approximation cursors and one full-data cursor for exported leaves. Supports finite
# numeric X/y, nonnegative sample weights, zero initial bias, Newton-1 leaves, and depths 1..8.
# Categorical CTRs, other objectives, bootstrap and early stopping are absent and rejected as unknown
# kwargs. Not the general boosting_type="Ordered" interface; that is
# GradientBoosting(boosting_type="Ordered") (lane/catboost-parity). numeric_mode="identical" selects
# pinned arithmetic. Native AMD/NVIDIA fixtures are certified; the Python binding needs independent
# installed-artifact qualification. loss_curve_ and best_iteration_ are None.
class OrderedRMSE(GradientBoosting):
    """Ordered-boosting RMSE regressor with an explicit row permutation.

    Parameters
    ----------
    n_estimators : int, default 100
    max_depth : int, default 6
        1 to 8.
    learning_rate : float, default 0.03
    l2_leaf_reg : float, default 3.0
    border_count : int, default 128
        1 to 255.
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
        for name, value, lower, upper in (  # glue: four integer hyperparameter checks
            ("n_estimators", n_estimators, 1, 2**31 - 1),
            ("max_depth", max_depth, 1, 8),
            ("border_count", border_count, 1, 255),
        ):
            if (is_bool(value)
                    or not isinstance(value, numbers.Integral)
                    or not lower <= value <= upper):
                raise ValueError(f"mojolearn: {name} must be an integer in {lower}..{upper}")
        for name, value, positive in (  # glue: two real hyperparameter checks
            ("learning_rate", learning_rate, True),
            ("l2_leaf_reg", l2_leaf_reg, False),
        ):
            # DEVIATION 2336: `numbers.Real` is every Python and NumPy
            # int and float; the float32 bound is the literal
            # `np.finfo(np.float32).max`; `_f32` is the cast that sent
            # 1e-100 to 0.0 and refused it as non-positive.
            if (is_bool(value)
                    or not isinstance(value, numbers.Real)
                    or not math.isfinite(value)
                    or abs(value) > _F32_MAX
                    or value < 0 or (positive and _f32(value) <= 0)):
                raise ValueError(f"mojolearn: {name} must be finite float32 and "
                                 + ("positive" if positive else "non-negative"))

    def fit(self, X, y, *, permutation, sample_weight=None):
        """Fit with the ordering given by ``permutation``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Finite values, at least 4 rows.
        y : array-like of shape (n_samples,)
        permutation : sequence of int
            A bijection of row ids ``0..n_samples-1``. ``X``, ``y`` and weights
            stay in their original order; only the permutation sets the order.
        sample_weight : array-like of shape (n_samples,), optional
            Finite, non-negative weights with a positive sum.

        Returns
        -------
        self
        """
        return _TREES(self, "trees_gbdt_ordered_fit", (X, y, permutation, sample_weight), _TREES_CONTEXT)

# Separate sklearn prediction contracts; legacy GradientBoosting stays unchanged.
from ._gbdt_adapters import GradientBoostingClassifier, GradientBoostingRegressor


# scikit-learn keeps more estimators in this module than it defines; they resolve from
# the package at first use (`_sklearn_modules`), so `from mojolearn.ensemble import ...`
# takes the same names as scikit-learn's module.
from ._sklearn_modules import LAYOUT as _SKLEARN_LAYOUT, lazy_exports as _lazy_exports  # noqa: E402
__getattr__ = _lazy_exports("ensemble")
