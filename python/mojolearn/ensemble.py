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

import hashlib
import itertools
import numbers
from . import _portable_math as math
import os
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
from . import _backend, _mojolearn_gbdt, _serialize
from ._mode import NumericModeMixin
from ._array import Array
from ._buffer import (
    as_f32_forest_layout,
    addr, addr_ro, all_finite, as_f32_c, as_f32_colmajor, as_i64_c,
    empty, frombytes, zeros, _native,
)
from ._labels import argmax_rows, finite_integer_codes, flat_view, is_bool

#: The npz model-file format tag `save` writes and `load` requires.
_MODEL_FORMAT = "mojolearn-gbdt-1"


def _model_text_sha256(text):
    """The key of the device-resident copy (DEVIATION 2980): the sha256 of
    the model text's utf-8 bytes, the bytes `save` writes."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

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

def _group_id_key(value, index):
    """The bytes CatBoost hashes for one group id
    (`_catboost.pyx:2171-2196`, `get_id_object_bytes_string_representation`):
    a string or bytes object as itself, an integer as its decimal spelling.
    A float, a bool or anything else is refused in their words."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return str(int(value)).encode("ascii")
    raise ValueError(
        f"mojolearn: group_id[{index}] object ({value!r}) is unsuitable "
        "(should be string or integral type)"
    )


def _group_sizes(group_id, n_rows):
    """`group_id` as the pool's run lengths, in row order.

    Their grouping is built from runs of equal consecutive ids and a
    repeated id is refused (`libs/data/objects.cpp:60-87`: the run starts
    are collected, sorted, and `adjacent_find` raises "group Ids are not
    consecutive"). The length check is their Pool's
    (`core.py:1071-1076`)."""
    ids = group_id.tolist() if hasattr(group_id, "tolist") else group_id
    try:
        ids = list(ids)
    except TypeError:
        raise ValueError(
            f"mojolearn: Invalid group_id type={type(group_id)}: must be "
            "array like."
        ) from None
    if len(ids) != n_rows:
        raise ValueError(
            f"mojolearn: Length of group_id={len(ids)} and length of "
            f"data={n_rows} are different."
        )
    if n_rows > 0xFFFFFFFF:
        raise ValueError("mojolearn: group_id needs at most 2**32 - 1 rows")
    sizes = _group_sizes_by_runs(ids)
    if sizes is None:
        sizes = _group_sizes_rowwise(ids)
    return sizes


def _group_sizes_rowwise(ids):
    """The definition: one key per row, in row order, and every refusal."""
    sizes = []
    seen = set()
    last = None
    for i, value in enumerate(ids):  # cpu-route: group_id label objects (int/str/bytes) checked for contiguous runs [py-data-loop]
        key = _group_id_key(value, i)
        if i > 0 and key == last:
            sizes[-1] += 1
            continue
        if key in seen:
            raise ValueError(
                "mojolearn: group Ids are not consecutive: the rows of group "
                f"{value!r} are split into more than one run (row {i}); "
                "CatBoost requires every group's rows to be contiguous"
            )
        seen.add(key)
        sizes.append(1)
        last = key
    return sizes


def _group_sizes_by_runs(ids):
    """`_group_sizes_rowwise`'s answer with one key per RUN of equal ids, or
    None when the row walk must decide.

    The row walk spent 1,278 ms on Istella-S's 2,043,304 ids
    (bench/results/istella_ranking_2026-09-15), a fixed cost of every
    ranking fit. `itertools.groupby` finds the runs of `==` ids in C. Equal
    ids of ONE type (int, str or bytes, exactly) spell the same key, so a
    run is keyed once, by its first id, and adjacent runs with the same key
    (7 then "7") merge as the row walk merges them. Everything else returns
    None: a run mixing types (5 and 5.0, or 1 and True, compare equal), an
    id of any other type, an id whose `==` raises, and a repeated group,
    so the row walk owns every refusal, its wording and its row index.
    """
    sizes = []
    seen = set()
    last = None
    try:
        for value, run in itertools.groupby(ids):  # cpu-route: group_id label objects (int/str/bytes) checked for contiguous runs [py-data-loop]
            kind = type(value)
            if kind is not int and kind is not str and kind is not bytes:
                return None
            members = list(run)
            if set(map(type, members)) != {kind}:
                return None
            key = _group_id_key(value, 0)
            if key == last:
                sizes[-1] += len(members)
                continue
            if key in seen:
                return None
            seen.add(key)
            sizes.append(len(members))
            last = key
    except Exception:
        return None
    return sizes


def _pairs_arrays(pairs, pairs_weight, n_rows):
    """`pairs` flattened winner then loser, and one float weight per pair.

    Their Pool reads each pair as `pair[0]`, `pair[1]` and requires integer
    indices (`core.py:946-955`, `_catboost.pyx:4157-4175`), a weight per pair
    when `pairs_weight` is given, 1.0 otherwise. Rows outside the pool, a row
    paired with itself, a non-finite or negative weight and an empty pair list
    are refused here; that both rows share a group is checked in the binding."""
    rows = pairs.tolist() if hasattr(pairs, "tolist") else list(pairs)
    if len(rows) == 0:
        raise ValueError("mojolearn: pairs is empty")
    flat = []
    for i, pair in enumerate(rows):  # cpu-route: user pairs list of Python index pairs validated and flattened [py-data-loop]
        if len(pair) != 2:
            raise ValueError(f"mojolearn: Length of pairs[{i}] isn't equal to 2.")
        for j, index in enumerate(pair):  # cpu-route: user pairs list of Python index pairs validated and flattened [py-data-loop]
            if not isinstance(index, numbers.Integral) or isinstance(index, bool):
                raise ValueError(
                    f"mojolearn: Invalid pairs[{i}][{j}] = '{index}' value "
                    f"type={type(index)}: must be an integer."
                )
            if not 0 <= int(index) < n_rows:
                raise ValueError(
                    f"mojolearn: pairs[{i}][{j}] = {int(index)} is outside the "
                    f"{n_rows} rows"
                )
        if int(pair[0]) == int(pair[1]):
            raise ValueError(f"mojolearn: pairs[{i}] pairs row {int(pair[0])} with itself")
        flat.extend((int(pair[0]), int(pair[1])))
    if pairs_weight is None:
        weights = [1.0] * len(rows)
    else:
        weights = pairs_weight.tolist() if hasattr(pairs_weight, "tolist") else list(pairs_weight)
        if len(weights) != len(rows):
            raise ValueError(
                f"mojolearn: len(pairs_weight) = {len(weights)} is not equal to "
                f"len(pairs) = {len(rows)} "
            )
        for i, w in enumerate(weights):  # cpu-route: user pairs list of Python index pairs validated and flattened [py-data-loop]
            if not isinstance(w, numbers.Real) or not math.isfinite(float(w)) or float(w) < 0:
                raise ValueError(
                    f"mojolearn: pairs_weight[{i}] must be a finite nonnegative number"
                )
        weights = [float(w) for w in weights]  # cpu-route: user pairs list of Python index pairs validated and flattened [py-data-loop]
    return flat, weights


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
_CATBOOST_DEFAULTS_SABOTAGE = "MOJOLEARN_CATBOOST_DEFAULTS_SABOTAGE"


def _catboost_defaults_sabotaged():
    return (os.environ.get(_CATBOOST_DEFAULTS_SABOTAGE) == "1"
            and os.environ.get("MOJOLEARN_HOST_ALLOW_SABOTAGE") == "1")


def _c_round(number, precision):
    """Their `Round` (`options_helper.cpp:15-18`): `round(number * 10^p) /
    10^p` with C `round`, which rounds halves AWAY from zero (Python's
    `round` rounds them to even)."""
    multiplier = math.powi(10.0, precision)  # exact for 0 <= precision <= 22; no platform pow
    scaled = number * multiplier
    rounded = math.floor(abs(scaled) + 0.5)
    return math.copysign(rounded, scaled) / multiplier


def catboost_auto_learning_rate(loss, n_rows, iterations, use_best_model,
                                boost_from_average, table=None):
    """`TAutoLRParamsGuesser::GetLearningRate` (`options_helper.cpp:252-262`)
    for the GPU learner, or None where their `NeedToUpdate` (`:245-249`) has
    no row for the key.

        customIterationConstant  = exp(C * log(iterations) + D)
        defaultIterationConstant = exp(C * log(1000) + D)
        defaultLearningRate      = exp(A * log(n_rows) + B)
        learning_rate = Round(min(default * custom / defaultIter, 0.5), 6)

    stored in their `TOption<float>` (`boosting_options.h:26`), so the fit
    receives it rounded to float32. `exp` and `log` are this package's
    portable binary64 routines rather than the host libm theirs call, so a
    value within one ulp of a sixth-decimal rounding boundary could round the
    other way; no such key was found on the checked grid.
    """
    target = {"Logloss": "Logloss", "MultiClass": "MultiClass",
              "RMSE": "RMSE"}.get(loss)
    if target is None:
        return None
    coeffs = (table or _GPU_AUTO_LEARNING_RATE).get(
        (target, bool(use_best_model), bool(boost_from_average)))
    if coeffs is None:
        return None
    a, b, c, d = coeffs
    custom = math.exp(c * math.log(float(iterations)) + d)
    default_iter = math.exp(c * math.log(1000.0) + d)
    default_rate = math.exp(a * math.log(float(n_rows)) + b)
    return _c_round(min(default_rate * custom / default_iter, 0.5), 6)

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

_UNSET = -1.0

#: their `EOverfittingDetectorType` spellings. `None` here means the
#: option was not given, and that is NOT the same as `od_type="None"`:
#: unset lets `TOverfittingDetectorOptions::Load`
#: (`overfitting_detector_options.cpp:24-32`) pick the type from whichever
#: of `od_pvalue` / `od_wait` was given, while "None" turns the detector
#: off outright. Wilcoxon is theirs and is not implemented.
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


def _validate_search_options(random_strength, use_pointwise_searcher,
                             grow_policy, score_function):
    """Shared constructor/fit guards; these conflicts apply in every mode.

    Returns the strength the fit uses. None is UNSET and resolves to
    CatBoost's 1.0 (`oblivious_tree_options.cpp:17`) under SymmetricTree
    with a noise-bearing score function, and to 0.0 under L2 / NewtonL2
    (whose calcer has no noise term, so 0.0 is the model CatBoost fits) and
    under Depthwise and Lossguide (the library's earlier default, kept)."""
    if use_pointwise_searcher and grow_policy != "SymmetricTree":
        raise ValueError(
            "mojolearn: use_pointwise_searcher=True is an OBLIVIOUS searcher; "
            f"grow_policy={grow_policy!r} requires the greedy subsets searcher"
        )
    if random_strength is None:
        if grow_policy == "SymmetricTree" and score_function not in ("L2", "NewtonL2"):
            return _CATBOOST_RANDOM_STRENGTH
        return _LEGACY_RANDOM_STRENGTH
    valid_type = (not is_bool(random_strength)
                  and isinstance(random_strength, numbers.Real))
    try:
        strength = float(random_strength) if valid_type else float("nan")
    except (TypeError, ValueError, OverflowError):
        strength = float("nan")
    if (not math.isfinite(strength)
            or not 0 <= strength <= 3.4028234663852886e+38):
        raise ValueError(
            "mojolearn: random_strength must be finite, nonnegative and "
            "<= Float32.MAX_FINITE"
        )
    if strength != 0.0 and score_function in ("L2", "NewtonL2"):
        raise ValueError(
            f"mojolearn: random_strength={strength} does nothing under "
            f"score_function={score_function!r}; use Cosine or NewtonCosine, "
            "or random_strength=0.0"
        )
    return strength
# DEVIATION 2330: the small exact replacements for what NumPy used to do
# on the host side of this module. None of them touches a result bit.


def _f32_bits_to_float(bits):
    """The float32 whose IEEE bit pattern is `bits`, widened to float64
    exactly (signed zero included): `np.uint32(bits).view(np.float32)`."""
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _f64_bits_to_float(bits):
    """`np.uint64(bits).view(np.float64)`."""
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _f32(value):
    """`float(np.float32(value))`: one round-to-nearest-even to float32
    (an underflowing value becomes 0.0, exactly as the cast does)."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


#: `np.finfo(np.float32).max`, as the literal.
_F32_MAX = 3.4028234663852886e+38
#: `np.iinfo(np.uint32).max`, as the literal.
_U32_MAX = 4294967295


def _has_nan(arr):
    """Whether a float32 `Array` holds a NaN (an inf is NOT one).

    Reached ONLY after `all_finite` said the matrix is not finite, so the
    common path never runs it; on that rare path it is a C-driven
    `any(map(isnan, view))` over the storage, O(size) without a Python
    loop body. A native NaN-vs-inf helper would retire it."""
    return any(map(math.isnan, flat_view(arr, "f")))


def _f64_list(values):
    """A binding's list of floats as a float64 `Array`."""
    return Array.from_list(values, "<f8")


def _one_string(member):
    """The single str a loaded npz member holds, or None when the member
    is not exactly one string (a number, a bytes value, a list of two, an
    empty `<U` array)."""
    if isinstance(member, str):
        return member
    if isinstance(member, (list, tuple)):
        if len(member) == 1 and isinstance(member[0], str):
            return member[0]
        return None
    dtype = str(getattr(member, "dtype", ""))
    if dtype.lstrip("<>|=")[:1] != "U" or getattr(member, "size", None) != 1:
        return None
    values = member.tolist() if hasattr(member, "tolist") else list(member)
    while isinstance(values, (list, tuple)):
        if len(values) != 1:
            return None
        values = values[0]
    return values if isinstance(values, str) else None


def _check_ordered_options(boosting_type, loss, grow_policy, score_function,
                           leaf_estimation_method, use_pointwise_searcher,
                           feature_fraction):
    """What an EXPLICIT boosting_type='Ordered' refuses, by name, where
    CatBoost refuses it (their words) or where this implementation does not
    restate their arm (said so). An unset boosting type never reaches an
    Ordered it cannot run: `_resolved_boosting_type` resolves those to the
    same refusals at fit time, so the answer does not depend on how the
    Ordered fit was asked for."""
    if boosting_type != "Ordered":
        return
    if grow_policy != "SymmetricTree":
        raise ValueError(
            "mojolearn: Ordered boosting is not supported for nonsymmetric "
            "trees. (catboost_options.cpp:757-759)"
        )
    if loss in MULTI_OUTPUT_LOSSES:
        raise ValueError(
            f"mojolearn: On GPU loss {loss} can't be used with ordered "
            "boosting (catboost_options.cpp:949-967)"
        )
    if loss in MULTI_REGRESSION_LOSSES:
        raise NotImplementedError(
            f"mojolearn: boosting_type='Ordered' with loss={loss!r} is not "
            "carried: the Ordered arm is one-dimensional; use "
            "boosting_type='Plain'"
        )
    if score_function not in ("Cosine", "NewtonCosine"):
        raise ValueError(
            f"mojolearn: Score function {score_function} can't be used with "
            "ordered boosting (catboost_options.cpp:972-978)"
        )
    if leaf_estimation_method == "Exact":
        raise ValueError(
            "mojolearn: Exact leaf estimation method don't work with ordered "
            "boosting on GPU (catboost_options.cpp:346-350)"
        )
    if loss in _QUERYWISE_LOSSES:
        raise NotImplementedError(
            f"mojolearn: boosting_type='Ordered' with loss={loss!r} is not "
            "implemented: their folds follow the query grouping "
            "(dynamic_boosting.h:189-223); use boosting_type='Plain'"
        )
    if use_pointwise_searcher:
        raise ValueError(
            "mojolearn: use_pointwise_searcher selects the doc-parallel Plain "
            "searcher; an Ordered fit runs the feature-parallel fold searcher"
        )
    if feature_fraction < 1.0:
        raise NotImplementedError(
            "mojolearn: boosting_type='Ordered' with feature_fraction < 1 is "
            "not implemented"
        )


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
        # ---- CatBoost's GPU defaults for the oblivious learner, resolved
        # where they do not depend on the pool (lane/catboost-parity). The
        # pool-dependent ones -- the learning rate, the small-iteration
        # leaf count -- resolve in `fit`, as theirs do in
        # `SetDataDependentDefaults` (`options_helper.cpp:403-435`).
        symmetric_policy = grow_policy == "SymmetricTree"
        bootstrap_defaulted = bootstrap_type is None and symmetric_policy
        if n_estimators is None:
            n_estimators = (_CATBOOST_ITERATIONS if symmetric_policy
                            else _LEGACY_ITERATIONS)
        if bootstrap_type is None and symmetric_policy:
            bootstrap_type = _CATBOOST_BOOTSTRAP
        if bootstrap_type is not None and bootstrap_type not in BOOTSTRAP_TYPES:
            raise ValueError(
                f"mojolearn: bootstrap_type must be one of "
                f"{BOOTSTRAP_TYPES}, got {bootstrap_type!r}"
            )
        if bootstrap_type == "Bayesian" and subsample is not None:
            # CatBoost's own validators: `catboost_options.cpp:795` for the
            # default ("default bootstrap type (bayesian) doesn't support
            # 'subsample' option") and `bootstrap_options.cpp:15-18` for an
            # explicit Bayesian
            if bootstrap_defaulted:
                raise ValueError(
                    "mojolearn: the default bootstrap_type='Bayesian' "
                    "(CatBoost's GPU default, bootstrap_options.h:18) does "
                    "not support subsample; pass bootstrap_type='Bernoulli' "
                    "or 'Poisson' with it"
                )
            raise ValueError(
                "mojolearn: bootstrap_type='Bayesian' does not support "
                "subsample; it takes bagging_temperature"
            )
        # A knob no sampler reads is refused, not ignored (the claim-surface
        # census, 2026-09-14): with no bootstrap_type the trainer sets
        # boot_kind = -1 and never reads either (gbdt/train.mojo::_bootstrap),
        # Bernoulli and Poisson read subsample only, Bayesian reads
        # bagging_temperature only. Accepting the other one silently would
        # hand a caller a model that ignores what they asked for.
        if bagging_temperature != 1.0 and bootstrap_type != "Bayesian":
            raise ValueError(
                "mojolearn: bagging_temperature is read only by "
                "bootstrap_type='Bayesian'; with bootstrap_type="
                f"{bootstrap_type!r} it would be ignored, so it is refused"
            )
        if subsample is not None and bootstrap_type not in ("Bernoulli", "Poisson"):
            raise ValueError(
                "mojolearn: subsample is read only by bootstrap_type="
                "'Bernoulli' or 'Poisson'; with bootstrap_type="
                f"{bootstrap_type!r} it would be ignored, so it is refused"
            )
        if leaf_estimation_method is not None:
            if leaf_estimation_method not in _LEAF_ESTIMATION_NAMES:
                raise ValueError(
                    f"mojolearn: leaf_estimation_method must be one of "
                    f"{tuple(_LEAF_ESTIMATION_NAMES)}, got "
                    f"{leaf_estimation_method!r}"
                )

        # ---- the grow policy, and what CatBoost refuses beside it ----
        # (DEVIATION 259; every refusal cites the line of theirs it follows)
        if grow_policy == "Region":
            raise NotImplementedError(
                "mojolearn: grow_policy='Region' is EGrowPolicy::Region, "
                "which no lane implements (greedy_search_helper.cpp:325-350); "
                f"reachable values are {GROW_POLICIES}"
            )
        if grow_policy not in _GROW_POLICY_CODES:
            raise ValueError(
                f"mojolearn: grow_policy must be one of {GROW_POLICIES}, "
                f"got {grow_policy!r}"
            )
        valid_fraction_type = (
            not is_bool(feature_fraction)
            and isinstance(feature_fraction, numbers.Real)
        )
        try:
            parsed_fraction = float(feature_fraction) if valid_fraction_type else float("nan")
        except (ValueError, TypeError, OverflowError):
            parsed_fraction = float("nan")
        if not math.isfinite(parsed_fraction) or not 0.0 < parsed_fraction <= 1.0:
            raise ValueError("mojolearn: feature_fraction must be finite and in (0, 1]")
        feature_fraction = parsed_fraction
        if feature_fraction < 1.0 and any(
            values is not None and len(values) > 0
            for values in (cat_features, one_hot_features)  # glue: the two feature-list arguments
        ):
            raise NotImplementedError(
                "mojolearn: feature_fraction < 1 requires numeric features; "
                "categorical, one-hot and CTR features are unsupported"
            )
        if min_split_gain is not None:
            valid_type = (not is_bool(min_split_gain)
                          and isinstance(min_split_gain, numbers.Real))
            try:
                parsed_gain = float(min_split_gain) if valid_type else float("nan")
            except (ValueError, TypeError, OverflowError):
                parsed_gain = float("nan")
            if not math.isfinite(parsed_gain) or parsed_gain < 0:
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
            # greedy_search_helper.cpp:685); this implementation refuses what it
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
        if score_function in _UNSUPPORTED_SCORE_FUNCTIONS:
            raise NotImplementedError(
                f"mojolearn: score_function={score_function!r} is not "
                f"honored here -- {_UNSUPPORTED_SCORE_FUNCTIONS[score_function]}"
                f". Reachable values are {SCORE_FUNCTIONS}."
            )
        if score_function not in _SCORE_FUNCTION_NAMES:
            raise ValueError(
                f"mojolearn: score_function must be one of "
                f"{SCORE_FUNCTIONS}, got {score_function!r}"
            )
        if min_child_hessian is not None:
            valid_type = (not is_bool(min_child_hessian)
                          and isinstance(min_child_hessian, numbers.Real))
            try:
                parsed_hessian = float(min_child_hessian) if valid_type else float("nan")
            except (ValueError, TypeError, OverflowError):
                parsed_hessian = float("nan")
            if not math.isfinite(parsed_hessian) or not 0 <= parsed_hessian <= 3.4028234663852886e+38:
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
        if feature_border_type not in BORDER_TYPES:
            raise ValueError(
                f"mojolearn: feature_border_type must be one of "
                f"{BORDER_TYPES}, got {feature_border_type!r}"
            )
        # validated here, KEPT UNSET: None resolves at `_params` against the
        # score function and policy the fit actually runs
        checked_strength = _validate_search_options(
            random_strength, use_pointwise_searcher, grow_policy, score_function
        )
        random_strength = None if random_strength is None else checked_strength
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
            if any(not math.isfinite(float(w)) or float(w) < 0 for w in class_weights):  # glue: validates the class_weights argument
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
        # ---- boosting_type (lane/catboost-parity), and what CatBoost
        # refuses beside Ordered, in their words where they have them ----
        if boosting_type is not None and boosting_type not in BOOSTING_TYPES:
            raise ValueError(
                f"mojolearn: boosting_type must be one of {BOOSTING_TYPES} or "
                f"None, got {boosting_type!r}"
            )
        _check_ordered_options(
            boosting_type, loss, grow_policy, score_function,
            leaf_estimation_method, use_pointwise_searcher, feature_fraction,
        )
        if (is_bool(fold_len_multiplier)
                or not isinstance(fold_len_multiplier, numbers.Real)
                or not math.isfinite(float(fold_len_multiplier))
                or not float(fold_len_multiplier) > 1.0):
            # `boosting_options.cpp:64`
            raise ValueError(
                "mojolearn: fold len multiplier should be greater than 1, got "
                f"{fold_len_multiplier!r}"
            )
        if fold_permutation_block is not None and (
                is_bool(fold_permutation_block)
                or not isinstance(fold_permutation_block, numbers.Integral)
                or not 0 <= int(fold_permutation_block) <= 256):
            raise ValueError(
                "mojolearn: fold_permutation_block must be an integer in "
                f"0..256 or None, got {fold_permutation_block!r}"
            )
        if boosting_type == "Plain" and (
                float(fold_len_multiplier) != 2.0
                or fold_permutation_block is not None):
            raise ValueError(
                "mojolearn: fold_len_multiplier and fold_permutation_block are "
                "read only by Ordered boosting (dynamic_boosting.h:115-223); "
                "with boosting_type='Plain' they would be accepted and ignored"
            )
        # THE CTR PERMUTATION KNOBS ARE INERT WITHOUT CATEGORICALS -- except
        # that `permutation_count` is ALSO the number of Ordered boosting's
        # permutations (`dynamic_boosting.h:137-141`), so it stands wherever
        # the fit may be Ordered (an unset boosting type resolves in `fit`,
        # and a Plain resolution with no cat_features refuses it there).
        # `cat_features` is checked rather than `one_hot_features` because a
        # one-hot column never grows CTRs either.
        for _name, _val in (  # glue: two permutation argument checks
            ("permutation_count", permutation_count),
            ("ctr_estimation_permutation_id", ctr_estimation_permutation_id),
        ):
            if _val is None:
                continue
            if int(_val) < 0:
                raise ValueError(
                    f"mojolearn: {_name} must be >= 0 when given, got {_val}"
                )
            may_be_ordered = (_name == "permutation_count"
                              and boosting_type != "Plain"
                              and grow_policy == "SymmetricTree")
            if not cat_features and not may_be_ordered:
                raise ValueError(
                    f"mojolearn: {_name} is read only by the categorical "
                    "path (doc_parallel_dataset_builder.cpp:190-262)"
                    + (" and by Ordered boosting" if _name == "permutation_count" else "")
                    + "; with no cat_features it would be accepted and "
                    "ignored. Pass cat_features= or leave it unset."
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
        self.random_strength = (None if random_strength is None
                                else float(random_strength))
        self.use_pointwise_searcher = bool(use_pointwise_searcher)
        # their tri-state (`AdjustBoostFromAverageDefaultValue`): None is
        # unset and resolves inside `train` -- auto-True for RMSE (their
        # rule; NOT for Logloss, which is not on their list), False
        # otherwise; True is explicit and refused by name for losses
        # whose CalcOptimumConstApprox arm is not implemented.
        self.boost_from_average = boost_from_average
        self.border_build_max_samples = int(border_build_max_samples)
        self.class_weights = (
            None if class_weights is None
            else [float(w) for w in class_weights]  # glue: stores the class_weights argument
        )
        self.permutation_count = permutation_count
        self.ctr_estimation_permutation_id = ctr_estimation_permutation_id
        self.grow_policy = grow_policy
        self.max_leaves = None if max_leaves is None else int(max_leaves)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.min_split_gain = None if min_split_gain is None else float(min_split_gain)
        self.min_child_hessian = min_child_hessian
        self.feature_fraction = feature_fraction
        self.feature_border_type = feature_border_type
        self.boosting_type = boosting_type
        self.fold_len_multiplier = float(fold_len_multiplier)
        self.fold_permutation_block = (None if fold_permutation_block is None
                                       else int(fold_permutation_block))

        self.model_ = None
        self.loss_curve_ = None
        self.test_loss_curve_ = None
        self.best_iteration_ = None
        self.stopped_early_ = None
        #: the learning rate the last fit used (CatBoost's auto-selection
        #: resolved against that fit's pool), None before a fit
        self.learning_rate_ = None
        #: the boosting type the last fit used ('Plain' or 'Ordered')
        self.boosting_type_ = None
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
    def _resolved_learning_rate(self, n_rows, n_eval_rows=0,
                                eval_target_constant=False):
        """The learning rate a fit on `n_rows` rows uses (`learning_rate_`).

        An explicit value is used as given. Unset under Depthwise and
        Lossguide it is 0.03. Unset under SymmetricTree it is CatBoost's
        `UpdateLearningRate` (`options_helper.cpp:269-288`): auto-selected
        from the GPU table only while `l2_leaf_reg`, `leaf_estimation_method`
        and `leaf_estimation_iterations` are unset too (`:273-278`), keyed on
        `use_best_model` as `UpdateUseBestModel` resolves it (`:100-113`,
        True with an eval set whose target is not constant) and on
        `boost_from_average` as `AdjustBoostFromAverageDefaultValue`
        resolves it (`:353-374`); 0.03 where their table has no row."""
        if self.learning_rate is not None:
            return float(self.learning_rate)
        if self.grow_policy != "SymmetricTree":
            return _LEGACY_LEARNING_RATE
        if (self.l2_leaf_reg is not None
                or self.leaf_estimation_method is not None
                or self.leaf_estimation_iterations is not None):
            return _CATBOOST_LEARNING_RATE
        use_best = self.use_best_model
        if use_best is None:
            use_best = n_eval_rows > 0 and not eval_target_constant
        bfa = self.boost_from_average
        if bfa is None:
            bfa = self.loss in _BOOST_FROM_AVERAGE_LOSSES
        table = None
        if _catboost_defaults_sabotaged():
            # the negative control: their CPU learner's coefficient rows
            table = _CPU_AUTO_LEARNING_RATE
        rate = catboost_auto_learning_rate(
            self.loss, n_rows, int(self.n_estimators), use_best, bfa, table)
        return _CATBOOST_LEARNING_RATE if rate is None else rate

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
        if self.boosting_type is not None:
            return self.boosting_type
        if (self.grow_policy != "SymmetricTree"
                or self.loss in MULTI_OUTPUT_LOSSES
                # their chain would reach Ordered for MultiRMSE (it is not
                # on `IsGpuPlainDocParallelOnlyMode`'s list), whose arm is
                # one-dimensional here and refused by name
                or self.loss in MULTI_REGRESSION_LOSSES
                or self.score_function not in ("Cosine", "NewtonCosine")
                or self.use_pointwise_searcher
                or getattr(self, "feature_fraction", 1.0) < 1.0):
            return "Plain"
        if (n_rows >= _ORDERED_MAX_ROWS
                or int(self.n_estimators) < _ORDERED_MIN_ITERATIONS):
            return "Plain"
        return "Ordered"

    def _resolved_leaf_iterations(self, n_features):
        """`UpdateLeavesEstimationIterations` (`options_helper.cpp:290-307`)
        under SymmetricTree: an unset count becomes 1 when there are fewer
        than 200 iterations (`IsSmallIterationCount`,
        `catboost_options.h:88-90`) and fewer than 20 features. It runs
        AFTER the learning-rate auto-selection in their
        `SetDataDependentDefaults` (`:416-429`), so this 1 does not count
        as a set option there, and it does not here either. None otherwise
        (the loss decides natively)."""
        iters = self.leaf_estimation_iterations
        if (iters is None and self.grow_policy == "SymmetricTree"
                and int(self.n_estimators) < 200 and n_features < 20):
            return 1
        return iters

    def _params(self, n_rows, n_features, n_flags, n_weights=0,
                n_eval_rows=0, eval_target_constant=False):
        strength = _validate_search_options(
            self.random_strength, self.use_pointwise_searcher,
            self.grow_policy, self.score_function,
        )
        def f(v):
            return _UNSET if v is None else float(v)

        cw = self.class_weights or ()
        method = self.leaf_estimation_method
        method_code = (
            -1 if method is None else _LEAF_ESTIMATION_NAMES[method]
        )
        iters = self._resolved_leaf_iterations(n_features)
        rate = self._resolved_learning_rate(
            n_rows, n_eval_rows, eval_target_constant)
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
            float(rate),                                # 7  resolved
            float((0.0 if self.loss == "YetiRank" else 3.0)
                  if self.l2_leaf_reg is None
                  else self.l2_leaf_reg),               # 8, None -> the loss's default
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
            strength,                                   # 25
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
        # DEVIATION 2330: built as a Python list (O(features)) and packed
        # once; `Array` has no item assignment.
        flags = [0] * n_features
        for i in cat or ():  # glue: validates cat_features index arguments
            if not 0 <= i < n_features:
                raise ValueError(
                    f"mojolearn: cat_features index {i} out of range for "
                    f"{n_features} features"
                )
            flags[i] |= 1
        for i in one_hot or ():  # glue: validates one_hot_features index arguments
            if not 0 <= i < n_features:
                raise ValueError(
                    f"mojolearn: one_hot_features index {i} out of range "
                    f"for {n_features} features"
                )
            flags[i] |= 2
        return Array.from_list(flags, "<u4")

    def _eval_arrays(self, eval_set, n_features):
        """`(X_eval, y_eval)` -> column-major float32 `Array`, float32
        `Array`, row count.

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
        # F-order input); see DEVIATION 1887 in `_buffer.as_f32_colmajor`
        Xea, _ = as_f32_colmajor(Xe, name="eval_set X")
        n_eval_rows, n_eval_features = Xea.shape
        if n_eval_features != n_features:
            raise ValueError(
                f"mojolearn: eval_set X has {n_eval_features} features "
                f"for a fit on {n_features}"
            )
        if n_eval_rows == 0:
            raise ValueError("mojolearn: eval_set X has no rows")
        yea, _ = as_f32_c(ye, ndim=1, name="eval_set y")
        if yea.shape[0] != n_eval_rows:
            raise ValueError(
                f"mojolearn: eval_set y has {yea.shape[0]} values for "
                f"{n_eval_rows} rows"
            )
        if not all_finite(yea):
            # a multiclass eval target is class codes, and its refusal names
            # them as the class-code check after this one does
            what = ("eval_set y (the eval_set labels)"
                    if getattr(self, "loss", None) in MULTI_OUTPUT_LOSSES else "eval_set y")
            raise ValueError(f"mojolearn: {what} must be finite (no NaN or infinity)")
        return Xea, yea, n_eval_rows

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
        # COLUMN-MAJOR is what the quantizer walks, so X is materialized
        # STRAIGHT to Fortran order: one copy at most, zero when the
        # caller already passes float32 F-order (DEVIATION 1887, declared
        # in `_buffer.as_f32_colmajor`). DEVIATION 2330: `Xa` is an
        # F-order float32 `Array` whose STORAGE is the flat column-major
        # buffer the binding reads, so its address is what crosses; every
        # host-side check below is a native helper or a C-driven scan over
        # `flat_view` of that storage, never a per-element Python loop
        # except where the contract permits one (label codes).
        # DEVIATION 2980: a refit drops the device copy of the previous
        # model HERE, not at the next call. The copy is also keyed on the
        # text's sha256 (`_resident_handle`), so a subclass fit that
        # replaces `model_` without passing here is served a fresh copy.
        self._release_resident()
        # lane/apple-fast-sym-feat: a FAST Apple binding built with the
        # device quantizer (`gbdt_fit_row_major_available`) takes a C-order
        # float32 array as it is through `gbdt_fit_rowmajor` and transposes
        # it on the device; every other binding, and every fit with
        # categorical or one-hot features, stages the column-major copy
        # exactly as before.
        fit_binding = self._bind("_mojolearn_gbdt")
        x_row_major = False
        if (not self.cat_features and not self.one_hot_features
                and _fit_row_major_available(fit_binding)):
            Xa, x_row_major = as_f32_forest_layout(X, name="X")
        else:
            Xa, _ = as_f32_colmajor(X, name="X")
        n_rows, n_features = Xa.shape

        if n_rows == 0 or n_features == 0:
            raise ValueError("mojolearn: fit requires at least one row and one feature")

        target_dim = 1
        if self.loss in MULTI_REGRESSION_LOSSES:
            # (n_rows, n_targets) staged COLUMN-major: the flat storage is
            # then `y[dim * n_rows + row]`, the dim-major planes the native
            # fit reads (their `targets + idx + dim * targetAlignSize`)
            ya, _ = as_f32_colmajor(y, name="y")
            target_dim = int(ya.shape[1])
            if target_dim < 2:
                raise ValueError(
                    f"mojolearn: {self.loss} needs y of shape (n_samples, "
                    f"n_targets) with n_targets >= 2, got {target_dim} "
                    "(CatBoost: \"Only one class found, can't learn "
                    "multiclass objective\", multiclass_targets.h:167)"
                )
            if eval_set is not None:
                raise NotImplementedError(
                    f"mojolearn: loss={self.loss!r} with eval_set is not "
                    "carried: the held-out arm's target is one value per row"
                )
            if group_id is not None or pairs is not None:
                raise ValueError(
                    f"mojolearn: loss={self.loss!r} does not read group_id "
                    "or pairs"
                )
            if self.class_weights is not None:
                raise ValueError(
                    f"mojolearn: class_weights do not apply to loss={self.loss!r}"
                )
            if self.cat_features or self.one_hot_features:
                raise NotImplementedError(
                    f"mojolearn: loss={self.loss!r} with cat_features or "
                    "one_hot_features is not carried: the CTR target "
                    "binarization reads one target per row"
                )
            if self.boost_from_average:
                raise NotImplementedError(
                    f"mojolearn: boost_from_average=True with loss="
                    f"{self.loss!r} is not carried: its StartingPoint is "
                    "per-dimension (optimal_const_for_loss.h:230-239) and "
                    "this model's bias is one value (DEVIATION 5951)"
                )
        else:
            ya, _ = as_f32_c(y, ndim=1, name="y")
        if ya.shape[0] != n_rows:
            raise ValueError(
                f"mojolearn: y has {ya.shape[0]} values for {n_rows} rows"
            )
        # A NaN label reached the searcher and surfaced as a bare Exception
        # ("All splits have infinite score"); CatBoost refuses NaN targets
        # at pool construction. Refused here, with inf, before the binding.
        if not all_finite(ya):
            what = ("y (the class labels)"
                    if getattr(self, "loss", None) in MULTI_OUTPUT_LOSSES else "y")
            raise ValueError(f"mojolearn: {what} must be finite (no NaN or infinity)")

        # `nan_mode='Forbidden'` MEANS "THERE ARE NO NaNs", AND IT HAS TO BE
        # CHECKED HERE OR IT MEANS NOTHING.
        #
        # CatBoost raises on this pair. Filtering the NaNs during border
        # search would otherwise silently route them into an ordinary bin.
        # The native quantizer enforces the reference's CB_ENSURE for
        # direct Mojo callers; this guard avoids entering the GPU binding.
        # An inf is not a NaN and passes, as it did (`_has_nan`).
        if self.nan_mode == "Forbidden" and not all_finite(Xa):
            if _has_nan(Xa):
                raise ValueError(
                    "mojolearn: nan_mode='Forbidden' but X contains NaN. "
                    "CatBoost refuses this pair; this implementation would otherwise "
                    "bin the NaNs silently with no NaN bin. Use "
                    "nan_mode='Min' or 'Max', or clean the column."
                )

        n_classes = 0
        if self.loss in MULTI_OUTPUT_LOSSES:
            # Validate before native class-weight indexing. Upstream checks
            # the class index before multiplication (54a8143a,
            # private/libs/target/data_providers.cpp:162-168).
            # `finite_integer_codes` is the label-encoding scan the contract
            # permits: native finiteness, then `set()` over the storage.
            codes = finite_integer_codes(ya)
            if codes is None:
                raise ValueError(
                    f"mojolearn: {self.loss} labels must be finite "
                    "nonnegative integer class codes"
                )
            n_classes = len(codes)
            if n_classes < 2:
                raise ValueError(
                    f"mojolearn: {self.loss} needs at least two classes"
                )
            if codes != list(range(n_classes)):
                raise ValueError(
                    f"mojolearn: {self.loss} labels must be dense class "
                    "codes 0..k-1; encode labels before fitting"
                )
            if self.class_weights is not None and len(self.class_weights) != n_classes:
                raise ValueError(
                    f"mojolearn: class_weights needs {n_classes} entries "
                    f"for {self.loss}"
                )

        flags = self._flags(n_features)
        n_flags = 0 if flags is None else n_features
        flags_holder = flags if flags is not None else zeros((1,), "<u4")

        if sample_weight is None:
            wa = Xa  # unread while params[2] == 0; any live buffer stands in
            n_weights = 0
        else:
            wa, _ = as_f32_c(sample_weight, ndim=1, name="sample_weight")
            if wa.shape[0] != n_rows:
                raise ValueError(
                    f"mojolearn: sample_weight has {wa.shape[0]} entries "
                    f"for {n_rows} rows"
                )
            # DEVIATION 2331: the sign and positivity scans are builtin
            # `min`/`max` over the storage view -- C-driven, O(rows), no
            # Python loop body, but a host scan all the same; a native
            # min/max helper would retire them.
            if not all_finite(wa) or wa.min() < 0:
                raise ValueError(
                    "mojolearn: sample_weight must have finite nonnegative entries"
                )
            if not wa.max() > 0:
                raise ValueError("mojolearn: sample_weight must have positive total weight")
            n_weights = n_rows

        Xea, ea, n_eval_rows = self._eval_arrays(eval_set, n_features)
        if n_eval_rows and self.loss in MULTI_OUTPUT_LOSSES:
            eval_codes = finite_integer_codes(ea)
            if eval_codes is None or eval_codes[-1] >= n_classes:
                raise ValueError(
                    "mojolearn: eval_set labels must be integer class codes "
                    "within the training class range"
                )

        # their `IsConstTarget(testDataMetaInfo)` (`options_helper.cpp:
        # 316-318`), which `UpdateUseBestModel` and so the learning-rate key
        # read; `min`/`max` over the storage view (DEVIATION 2331's scan)
        eval_constant = False
        if n_eval_rows:
            eval_constant = ea.min() == ea.max()
        params = self._params(
            n_rows, n_features, n_flags, n_weights, n_eval_rows,
            eval_target_constant=eval_constant,
        )
        #: the learning rate this fit used, after CatBoost's auto-selection
        self.learning_rate_ = float(params[7])
        if target_dim != 1:
            # THE TARGET-DIMENSION TAIL: the three optional float tails,
            # filled with their disabled values when the fit left them out,
            # then the dimension, 35 + n_class_weights + 4 values in all
            # (`bindings/_mojolearn_gbdt.mojo`'s docstring)
            fixed = 35 + int(params[34])
            tail = list(params[fixed:])
            tail += [-1.0, -1.0, 1.0][len(tail):]
            params = params[:fixed] + tail + [target_dim]
        # THE GROUP TAIL. `subgroup_id` and `pairs` are refused before
        # anything crosses; `group_id` becomes run lengths and rides after
        # the three optional float slots, which are filled with their
        # disabled values when the fit left them out.
        if subgroup_id is not None:
            raise NotImplementedError(
                "mojolearn: subgroup_id is read by no loss this implementation "
                "trains; the ranking losses that read CatBoost's subgroup ids "
                "are not implemented"
            )
        if pairs_weight is not None and pairs is None:
            raise ValueError("mojolearn: pairs_weight needs pairs")
        if pairs is not None:
            if self.loss != "PairLogit":
                raise NotImplementedError(
                    "mojolearn: pairs are read by loss='PairLogit' only; "
                    f"loss={self.loss!r} does not use them"
                )
            if group_id is None:
                raise NotImplementedError(
                    "mojolearn: pairs without group_id are not implemented: "
                    "the CatBoost reference then regroups and reorders the "
                    "whole pool by the pairs' connected components; pass "
                    "group_id"
                )
        group_holder = None
        pairs_holder = None
        pairs_weight_holder = None
        if group_id is not None:
            sizes = _group_sizes(group_id, n_rows)
            group_holder = Array.from_list(sizes, "<u4")
            fixed = 35 + int(params[34])
            tail = list(params[fixed:])
            tail += [-1.0, -1.0, 1.0][len(tail):]
            params = params[:fixed] + tail + [
                addr_ro(group_holder, name="group_id"), len(sizes),
            ]
            if self.loss == "PairLogit":
                # THE PAIRS TAIL: -1 generates the pairs inside the binding
                # and leaves both addresses unread; the group buffer stands in
                if pairs is None:
                    params += [addr_ro(group_holder, name="group_id"), -1,
                               addr_ro(group_holder, name="group_id")]
                else:
                    flat, weights = _pairs_arrays(pairs, pairs_weight, n_rows)
                    pairs_holder = Array.from_list(flat, "<u4")
                    pairs_weight_holder = Array.from_list(weights, "<f4")
                    params += [addr_ro(pairs_holder, name="pairs"), len(weights),
                               addr_ro(pairs_weight_holder, name="pairs_weight")]
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
        # the optional fifth string: sent only when it is not the default,
        # so a default fit keeps the four-string call every binding reads
        border_type = getattr(self, "feature_border_type", "GreedyLogSum")
        if border_type not in BORDER_TYPES:
            raise ValueError(
                f"mojolearn: feature_border_type must be one of "
                f"{BORDER_TYPES}, got {border_type!r}"
            )
        # THE BOOSTING TYPE, resolved against this pool (lane/catboost-
        # parity). An Ordered fit sends the three-string Ordered tail after
        # the border type: `boosting_type`, `fold_len_multiplier` as the
        # decimal of its float64 bits, `fold_permutation_block` (0 unset).
        boosting = self._resolved_boosting_type(n_rows)
        _check_ordered_options(
            boosting, self.loss, self.grow_policy, self.score_function,
            self.leaf_estimation_method, self.use_pointwise_searcher,
            getattr(self, "feature_fraction", 1.0),
        )
        fold_len = float(getattr(self, "fold_len_multiplier", 2.0))
        fold_block = getattr(self, "fold_permutation_block", None)
        if boosting == "Plain":
            if fold_len != 2.0 or fold_block is not None:
                raise ValueError(
                    "mojolearn: fold_len_multiplier and fold_permutation_block "
                    "are read only by Ordered boosting, and this fit resolved "
                    "to Plain; they would be accepted and ignored"
                )
            if self.permutation_count is not None and not self.cat_features:
                raise ValueError(
                    "mojolearn: permutation_count is read only by the "
                    "categorical path and by Ordered boosting, and this fit "
                    "resolved to Plain with no cat_features; it would be "
                    "accepted and ignored"
                )
        if border_type != "GreedyLogSum" or boosting == "Ordered":
            strs.append(border_type)
        if boosting == "Ordered":
            strs += [
                "Ordered",
                str(struct.unpack("<Q", struct.pack("<d", fold_len))[0]),
                str(0 if fold_block is None else int(fold_block)),
            ]
        #: the boosting type this fit used
        self.boosting_type_ = boosting

        # THE EVAL ADDRESSES ARE UNREAD WHEN params[20] IS 0, and the
        # learn buffer stands in so nothing has to allocate a throwaway --
        # the same stand-in the weights use.
        eval_x_holder = Xea if Xea is not None else Xa
        eval_y_holder = ea if ea is not None else ya

        # Every Array whose address crosses is a local of this frame until
        # the call returns: the borrow contract at the top of `_buffer`.
        fit_args = (
            addr_ro(Xa, name="X"),
            addr_ro(ya, name="y"),
            addr_ro(wa, name="sample_weight"),
            addr_ro(flags_holder, name="flags"),
            addr_ro(eval_x_holder, name="eval_set X"),
            addr_ro(eval_y_holder, name="eval_set y"),
            params,
            strs,
        )
        fit_fn = (fit_binding.gbdt_fit_rowmajor if x_row_major
                  else fit_binding.gbdt_fit)
        out = fit_fn(*fit_args)
        self.model_ = out[0]
        # the model's bias (CatBoost's `get_scale_and_bias()[1]`), parsed
        # from the text's BITS half so it round-trips exactly. 0.0 on
        # every fit without boost_from_average, exactly as theirs is.
        self.bias_ = 0.0
        _text = str(out[0])
        _at = 0 if _text.startswith("bias ") else _text.find("\nbias ") + 1
        if _at > 0 or _text.startswith("bias "):
            _end = _text.find("\n", _at)
            _line = _text[_at:] if _end < 0 else _text[_at:_end]
            _bits = int(_line.split()[1].split("/")[1], 16)
            self.bias_ = _f64_bits_to_float(_bits)
        self.best_iteration_ = int(out[1])
        self.stopped_early_ = bool(out[2])
        self.loss_curve_ = _f64_list(out[3])
        self.test_loss_curve_ = _f64_list(out[4]) if n_eval_rows else None
        self.n_features_in_ = n_features
        dim_binding = self._bind("_mojolearn_gbdt")
        if _has_ctr_records(self.model_) and _binding_vendor(dim_binding) == "cpu":
            # a CPU fit with CTR categoricals (gbdt/host/gbdt_oracle_ctr.mojo):
            # the host binding's parser refuses CTR records by name, so the
            # dim comes from the CPU binding's CTR reader (_gbdt_host)
            self.approx_dim_ = int(dim_binding.gbdt_ctr_model_dim(self.model_))
        else:
            self.approx_dim_ = dim_binding.gbdt_model_dim(self.model_)
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

    def _resident_handle(self, binding):
        """The handle of the device-resident copy of `model_`, prepared on
        the first call and reused while the text is the same bytes; None
        where the loaded binding has no `gbdt_resident_prepare` (a binary
        built before DEVIATION 2980, a CPU-only install's proxy, which
        refuses an absent name with ImportError), in which case the call
        takes the per-call parse.

        The fast check is object identity on the text (`str` is
        immutable, and the entry keeps a reference so the id cannot be
        reused). A different object is hashed and compared to the sha256
        the entry was prepared from: the same text keeps the handle, a
        new text releases it and prepares again."""
        try:
            prepare = binding.gbdt_resident_prepare
        except (ImportError, AttributeError):
            return None
        text = self.model_
        cached = getattr(self, "_resident", None)
        if cached is not None:
            c_binding, c_text, c_sha, c_handle = cached
            if c_binding is binding and c_text is text:
                return c_handle
            sha = _model_text_sha256(text)
            if c_binding is binding and c_sha == sha:
                self._resident = (binding, text, sha, c_handle)
                return c_handle
            self._release_resident()
        else:
            sha = _model_text_sha256(text)
        handle = int(prepare(text))
        self._resident = (binding, text, sha, handle)
        return handle

    def _release_resident(self):
        """Drop the device copy, if one is held. Quiet on a binding that
        cannot be reached any more (interpreter shutdown) and on a handle
        already released."""
        cached = getattr(self, "_resident", None)
        if cached is None:
            return
        self._resident = None
        try:
            cached[0].gbdt_resident_release(cached[3])
        except Exception:  # noqa: BLE001
            pass

    def __del__(self):
        try:
            self._release_resident()
        except Exception:  # noqa: BLE001
            pass

    def __getstate__(self):
        """A pickle or a deepcopy carries no device handle: the integer is
        meaningful only in the process and registry that minted it, and an
        unpickled instance holding it could release ANOTHER model's live
        copy. The copy prepares its own at its first call."""
        state = self.__dict__.copy()
        state.pop("_resident", None)
        return state

    def _check_fitted_layout(self, X):
        """`_check_fitted` for the resident door (DEVIATION 2980):
        `(array, n_rows, row_major)`. A 2-D float32 C-order input is a
        zero-copy borrow with `row_major` True and the native staging pass
        transposes it (DEVIATION 2637's shape for the forests); every other
        input takes the column-major materialization exactly as before."""
        if self.model_ is None:
            raise RuntimeError("mojolearn: predict() before fit()")
        Xa, row_major = as_f32_forest_layout(X, name="X")
        n_rows, n_features = Xa.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"mojolearn: model was fitted on {self.n_features_in_} "
                f"features, got {n_features}"
            )
        return Xa, n_rows, row_major

    def _check_fitted(self, X):
        if self.model_ is None:
            raise RuntimeError("mojolearn: predict() before fit()")
        # one column-major materialization at most (zero for float32
        # F-order input); see DEVIATION 1887 in `_buffer.as_f32_colmajor`.
        # The returned `Array` owns the storage the binding reads; the
        # caller keeps it in a local across the Mojo call.
        Xa, _ = as_f32_colmajor(X, name="X")
        n_rows, n_features = Xa.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"mojolearn: model was fitted on {self.n_features_in_} "
                f"features, got {n_features}"
            )
        return Xa, n_rows

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
        if self.model_ is None:
            raise RuntimeError("mojolearn: predict() before fit()")
        binding = self._bind("_mojolearn_gbdt")

        # A CPU-trained model whose text carries CTR tables or a tensor CTR
        # registry: the gbdt host binding's walk refuses those records by
        # name, and the CPU restatement of their apply is `HostGBDT`
        # (`forest_host_gbdt_expand_ctr`, the same expand_raw_columns /
        # expand_tensor_ctr_columns the device applies), the reader a CPU
        # column already uses for a GPU-saved CTR model.
        if _has_ctr_records(self.model_) and _binding_vendor(binding) == "cpu":
            return binding.gbdt_ctr_predict(
                X, loss=self.loss, text=self.model_,
                n_features_in=self.n_features_in_, approx_dim=self.approx_dim_,
                n_classes=self.n_classes_, estimator=type(self).__name__,
            )

        # DEVIATION 2980: the parsed, packed and uploaded model stays on
        # the device between calls; the per-call parse below is the door
        # for a binary without the entry point
        handle = self._resident_handle(binding)
        if handle is not None:
            Xa, n_rows, row_major = self._check_fitted_layout(X)
            out = empty((n_rows * self.approx_dim_,), "<f4")
            width = binding.gbdt_resident_predict(
                handle, addr_ro(Xa, name="X"),
                addr(out, name="predict output"),
                [n_rows, _PREDICT_RAW, 1 if row_major else 0],
            )
            if width != self.approx_dim_:
                raise RuntimeError(
                    f"mojolearn: predict wrote width {width}, expected "
                    f"{self.approx_dim_}"
                )
            if self.approx_dim_ == 1:
                return out
            return out.reshape((n_rows, width))

        Xa, n_rows = self._check_fitted(X)
        if self.approx_dim_ > 1:
            out = empty((n_rows * self.approx_dim_,), "<f4")
            width = binding.gbdt_predict_multi(
                self.model_, addr_ro(Xa, name="X"),
                addr(out, name="predict output"),
                [n_rows, _PREDICT_RAW],
            )
            if width != self.approx_dim_:
                raise RuntimeError(
                    f"mojolearn: predict wrote width {width}, expected "
                    f"{self.approx_dim_}"
                )
            return out.reshape((n_rows, width))

        out = empty((n_rows,), "<f4")
        wrote = binding.gbdt_predict(
            self.model_, addr_ro(Xa, name="X"),
            addr(out, name="predict output"), [n_rows]
        )
        if wrote != n_rows:
            raise RuntimeError(
                f"mojolearn: predict wrote {wrote} of {n_rows} rows"
            )
        return out

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
        if self.loss in MULTI_OUTPUT_LOSSES:
            if self.model_ is None:
                raise RuntimeError("mojolearn: predict_proba() before fit()")
            # 54a8143a libs/model/eval_processing.h:214-226:
            # MultiProbability applies CalcSigmoid elementwise.
            mode = (_PREDICT_SIGMOID if self.loss == "MultiClassOneVsAll"
                    else _PREDICT_SOFTMAX)
            binding = self._bind("_mojolearn_gbdt")
            handle = self._resident_handle(binding)  # DEVIATION 2980
            if handle is not None:
                Xa, n_rows, row_major = self._check_fitted_layout(X)
                out = empty((n_rows * self.n_classes_,), "<f4")
                width = binding.gbdt_resident_predict(
                    handle, addr_ro(Xa, name="X"),
                    addr(out, name="predict_proba output"),
                    [n_rows, mode, 1 if row_major else 0],
                )
            else:
                Xa, n_rows = self._check_fitted(X)
                out = empty((n_rows * self.n_classes_,), "<f4")
                width = binding.gbdt_predict_multi(
                    self.model_, addr_ro(Xa, name="X"),
                    addr(out, name="predict_proba output"), [n_rows, mode]
                )
            if width != self.n_classes_:
                raise RuntimeError(
                    f"mojolearn: predict_proba wrote width {width}, "
                    f"expected {self.n_classes_}"
                )
            return out.reshape((n_rows, width))

        if self.loss not in ("Logloss", "CrossEntropy"):
            raise ValueError(
                f"mojolearn: predict_proba is defined for Logloss, "
                f"CrossEntropy, MultiClass and MultiClassOneVsAll; this model was fitted with "
                f"{self.loss!r}. Use predict() and apply the link "
                f"yourself."
            )
        if self.model_ is None:
            raise RuntimeError("mojolearn: predict_proba() before fit()")
        binding = self._bind("_mojolearn_gbdt")
        handle = self._resident_handle(binding)
        if handle is not None:
            Xa, n_rows, row_major = self._check_fitted_layout(X)
            # DEVIATION 2980: both columns from the resident call, written
            # by the binding as float64 `[1 - p, p]` with `p` exactly
            # `gbdt_sigmoid`'s value over the exact widening of the raw
            # float32 (`gbdt_sigmoid_pair`'s statements, DEVIATION 2902);
            # the float32 raw array and its `astype` never exist
            out = empty((n_rows, 2), "<f8")
            width = binding.gbdt_resident_predict(
                handle, addr_ro(Xa, name="X"), addr(out, name="proba"),
                [n_rows, _PREDICT_SIGMOID_PAIR, 1 if row_major else 0],
            )
            if int(width) != 2:
                raise RuntimeError(
                    f"mojolearn: predict_proba wrote width {width}, expected 2"
                )
            return out
        raw = self.predict(X).astype("<f8")  # exact widening
        n_rows = raw.shape[0]
        # DEVIATION 258 made the IDENTICAL tier take the portable double
        # exp of `gbdt_sigmoid` so the probability bits are the same on
        # every host, while fast kept `1 / (1 +
        # np.exp(-raw))` with the host libm's last bit.
        # DEVIATION 2332: EVERY TIER now routes through `gbdt_sigmoid`.
        # IDENTICAL's bits do not move (same call as before). The FAST
        # `predict_proba` bits DO move, from numpy's exp to
        # the binding's portable exp, by at most the last-bit difference
        # between two exps; FAST has no bitwise card on proba, and
        # FAST is never asked a bitwise question. `1 - p` is the same one
        # float64 subtraction as before, so column 0 keeps its bits.
        # a CPU-only install's binding proxy raises ImportError, by name, for
        # an entry point its host family does not export
        try:
            pair = getattr(binding, "gbdt_sigmoid_pair", None)
        except (ImportError, AttributeError):
            pair = None
        if pair is not None:
            # DEVIATION 2902 (lane/infer-speed-trees, 2026-09-17): both
            # columns from the binding in one pass. `p` is `gbdt_sigmoid`'s
            # value and `1 - p` is the same one IEEE double subtraction the
            # comprehension below computed per row; the bits are unchanged,
            # only the O(rows) Python loop is gone.
            out = empty((n_rows, 2), "<f8")
            wrote = pair(addr_ro(raw, name="raw"), addr(out, name="proba"), n_rows)
            if int(wrote) != n_rows:
                raise RuntimeError(
                    f"mojolearn: gbdt_sigmoid_pair wrote {wrote} of {n_rows} rows"
                )
            return out
        # Lane pyglue-sweep (Oct 3): the `[1 - p, p]` Python comprehension
        # for a binary without `gbdt_sigmoid_pair` is gone; such a binary
        # refuses by name (every GPU and host gbdt binding exports it).
        raise RuntimeError(
            "mojolearn: this gbdt binding predates gbdt_sigmoid_pair; rebuild it"
        )

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
        if self.loss not in MULTI_OUTPUT_LOSSES + (
            "Logloss", "CrossEntropy",
        ):
            raise ValueError(
                f"mojolearn: predict_classes needs a classification loss; "
                f"this model was fitted with {self.loss!r}."
            )
        if self.numeric_mode_used() == "fast":
            binding = self._bind("_mojolearn_gbdt")
            if self.loss in ("Logloss", "CrossEntropy"):
                direct = getattr(binding, "gbdt_resident_binary_classes", None)
                if callable(direct):
                    handle = self._resident_handle(binding)
                    if handle is not None:
                        Xa, n_rows, row_major = self._check_fitted_layout(X)
                        out = empty((n_rows,), "<i8")
                        wrote = direct(
                            handle, addr_ro(Xa, name="X"),
                            addr(out, name="class codes"),
                            [n_rows, 1 if row_major else 0],
                        )
                        if wrote != n_rows:
                            raise RuntimeError(
                                f"mojolearn: predict_classes wrote {wrote} of {n_rows} rows"
                            )
                        return out
            # At wider multiclass widths the already-parallel probability pass
            # plus native argmax is faster than fusing the exact transform here.
            elif self.n_classes_ <= 3 and self.grow_policy == "SymmetricTree":
                if self.model_ is None:
                    raise RuntimeError("mojolearn: predict_proba() before fit()")
                handle = self._resident_handle(binding)
                if handle is not None:
                    Xa, n_rows, row_major = self._check_fitted_layout(X)
                    out = empty((n_rows,), "<i8")
                    mode = (_PREDICT_CLASSES_PINNED
                            if self.loss == "MultiClass"
                            else _PREDICT_CLASSES_OVA)
                    width = binding.gbdt_resident_predict(
                        handle, addr_ro(Xa, name="X"),
                        addr(out, name="predict_classes output"),
                        [n_rows, mode, 1 if row_major else 0],
                    )
                    if int(width) != 1:
                        raise RuntimeError(
                            f"mojolearn: predict_classes wrote width {width}, expected 1"
                        )
                    return out
        return argmax_rows(self.predict_proba(X))

    def _tree_metadata(self):
        """Read host model metadata without selecting a GPU extension.

        This is archive inspection, not a second model evaluator. Float
        values come exclusively from the authoritative IEEE bits.
        """
        if self.model_ is None:
            raise RuntimeError("mojolearn: tree inspection before fit()")
        counts, dimensions, leaves = [], [], []
        declared_trees = None
        for line in str(self.model_).splitlines():  # cpu-route: model text metadata parsed for inspection [py-data-loop]
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
            for values, count, dim in zip(leaves, counts, dimensions)  # glue: leaf count check over the trees
        ):
            raise ValueError("mojolearn: incomplete tree metadata")
        return counts, leaves

    def get_tree_leaf_counts(self):
        """Return the number of leaves in each tree, as in CatBoost.

        Returns
        -------
        mojolearn.Array
            uint32 array with one entry per tree.
        """
        counts, _ = self._tree_metadata()
        return Array.from_list([int(c) for c in counts], "<u4")  # glue: leaf counts per tree as words

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
        _, leaves = self._tree_metadata()
        values = [_f32_bits_to_float(bits) for tree in leaves for bits in tree]  # cpu-route: model text metadata parsed for inspection [py-data-loop]
        return Array.from_list(values, "<f8")

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
        if self.model_ is None:
            raise RuntimeError("mojolearn: save() before fit()")
        # Persist the tier prediction would use now, including an inherited
        # process default. Saving None would let a different process silently
        # turn an IDENTICAL model into FAST arithmetic after loading.
        mode = getattr(self, "numeric_mode", None) or _backend.default_mode()
        if not isinstance(mode, str) or mode.strip().lower() not in (
            "fast", "identical"
        ):
            raise ValueError(f"mojolearn: cannot save invalid numeric_mode {mode!r}")
        mode = mode.strip().lower()
        # DEVIATION 2334: the members are `Array`s and plain str (which
        # `_serialize.write_npz` encodes as the `<U` scalar member
        # `np.asarray(str)` gave). `bias` is a one-element float64 member,
        # the shape `write_npz` gives a 0-d value anyway (the 0.6.x writer
        # promoted 0-d to `(1,)` through `np.ascontiguousarray`), so the
        # file bytes are those of a 0.6.x save of the same model; the
        # loader on both sides reads the first element.
        model_bytes = str(self.model_).encode("utf-8")
        arrays = {
            "format": _MODEL_FORMAT,
            "numeric_mode": mode,
            "estimator": type(self).__name__,
            "loss": self.loss,
            "model": frombytes(model_bytes, "<u1", (len(model_bytes),)),
            "meta": Array.from_list(
                [
                    int(self.n_features_in_),
                    int(self.approx_dim_),
                    -1 if self.n_classes_ is None else int(self.n_classes_),
                    -1 if self.best_iteration_ is None
                    else int(self.best_iteration_),
                    1 if self.stopped_early_ else 0,
                ],
                "<i8",
            ),
            # raw float64 bytes, already parsed from the text's BITS half
            "bias": Array.from_list([float(self.bias_)], "<f8"),
        }
        return _serialize.write_npz(path, arrays)

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
            mode = _one_string(arrays["numeric_mode"])
            if mode is None:
                raise ValueError("mojolearn: saved numeric_mode must be one string")
            if mode not in ("fast", "identical"):
                raise ValueError(f"mojolearn: invalid saved numeric_mode {mode!r}")
            obj.numeric_mode = mode
        obj.loss = _serialize.scalar_str(arrays, "loss")
        obj.model_ = _serialize.exact(arrays, "model", "<u1").tobytes().decode("utf-8")
        meta = _serialize.exact(arrays, "meta", "<i8")
        obj.n_features_in_ = int(meta[0])
        binding = obj._bind("_mojolearn_gbdt")
        if _has_ctr_records(obj.model_) and _binding_vendor(binding) == "cpu":
            # the host binding's parser refuses CTR records by name; the
            # dim comes from the CPU binding's CTR reader (_gbdt_host)
            obj.approx_dim_ = int(binding.gbdt_ctr_model_dim(obj.model_))
        else:
            obj.approx_dim_ = int(binding.gbdt_model_dim(obj.model_))
        if obj.approx_dim_ != int(meta[1]):
            raise ValueError(
                f"mojolearn: {path!r} stores approx_dim {int(meta[1])} but "
                f"its model text holds {obj.approx_dim_}; the file is "
                "corrupt"
            )
        obj.n_classes_ = None if int(meta[2]) < 0 else int(meta[2])
        obj.best_iteration_ = None if int(meta[3]) < 0 else int(meta[3])
        obj.stopped_early_ = bool(int(meta[4]))
        # the first element, whether the member is 0-d (a 0.6.x file) or
        # one-element (this version's `save`)
        bias = _serialize.exact(arrays, "bias", "<f8").tolist()
        while isinstance(bias, list):
            bias = bias[0]
        obj.bias_ = float(bias)
        obj.loss_curve_ = None
        obj.test_loss_curve_ = None
        return obj


def _has_ctr_records(text):
    """Does a model text carry a CTR table or a tensor CTR registry (the
    records `gbdt_host_predict`'s parser refuses by name)?"""
    t = str(text)
    return any(("\n" + k + " ") in t or t.startswith(k + " ")
               for k in ("ctr_columns", "ctr_table", "tensor_ctr_registry"))  # glue: three fixed record names


def _binding_vendor(binding):
    """The binding's own vendor answer ("cpu" for the host binding), or
    None for a binary without the entry point."""
    try:
        return str(binding.gbdt_vendor())
    except (AttributeError, ImportError):
        return None


def _fit_row_major_available(binding):
    """lane/apple-fast-sym-feat: whether the binding serves
    `gbdt_fit_rowmajor` (a FAST Apple build with the device quantizer).
    False for every binary without the probe or whose probe says 0."""
    try:
        return bool(int(binding.gbdt_fit_row_major_available()))
    except (AttributeError, ImportError, TypeError, ValueError):
        return False


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
        Xa, _ = as_f32_colmajor(X, name="X")
        n_rows, n_features = Xa.shape
        if n_rows == 0 or n_features < 2:
            raise ValueError("mojolearn: experimental FeatureFreq needs rows and columns")
        if any(i < 0 or i >= n_features for i in self.sources):  # glue: validates the sources argument indices
            raise ValueError(
                f"mojolearn: sources {self.sources!r} outside {n_features} features"
            )
        if not all_finite(Xa):
            raise ValueError(
                "mojolearn: experimental FeatureFreq X must be finite"
            )
        # DEVIATION 2335 retired (lane pyglue-sweep, Oct 3): each column of
        # the column-major storage is checked by the base binding's helpers
        # at its own address (`reduce_stat` min / max / integral,
        # `cast_elements` + `bincount_i64` for the dense 0..k-1 coding)
        # instead of `set()` / `sorted()` over the column in Python.
        stat = _native("reduce_stat")
        base = addr_ro(Xa, name="X")
        for f in self.sources:  # glue: dispatches the named source columns
            col = base + 4 * f * n_rows
            hi = float(stat(col, 0, n_rows, 1))
            if (not int(stat(col, 0, n_rows, 4)) or float(stat(col, 0, n_rows, 0)) < 0):
                raise ValueError(
                    "mojolearn: experimental FeatureFreq source columns must "
                    "contain non-negative integer category codes"
                )
            k = int(hi) + 1
            dense = 2 <= k <= n_rows
            if dense:
                codes = empty((n_rows,), "<i8")
                _native("cast_elements")(col, 0, addr(codes, name="codes"), 3, n_rows)
                counts = empty((k,), "<i8")
                _native("bincount_i64")(addr_ro(codes, name="codes"), 3, n_rows, k,
                                        addr(counts, name="counts"), 0)
                dense = counts.min() > 0
            if not dense:
                raise ValueError(
                    "mojolearn: every experimental FeatureFreq source must "
                    f"be densely coded 0..k-1; column {f} has codes up to {int(hi)} "
                    "with at least one missing (or fewer than two codes)"
                )
        source_set = set(self.sources)
        for f in range(n_features):  # glue: constant-column check, one native stat per feature
            if f in source_set:
                continue
            col = base + 4 * f * n_rows
            if float(stat(col, 0, n_rows, 0)) == float(stat(col, 0, n_rows, 1)):
                raise ValueError(
                    "mojolearn: experimental FeatureFreq numeric columns "
                    f"must vary; column {f} is constant"
                )
        ya, _ = as_f32_c(y, ndim=1, name="y")
        if ya.shape[0] != n_rows or not all_finite(ya):
            raise ValueError("mojolearn: y must be finite with one value per row")
        if sample_weight is None:
            weights = ya  # unread while params[2] == 0
            n_weights = 0
        else:
            weights, _ = as_f32_c(sample_weight, ndim=1, name="sample_weight")
            if weights.shape[0] != n_rows:
                raise ValueError(
                    "mojolearn: sample_weight must have one value per row"
                )
            if not all_finite(weights) or weights.min() < 0:
                raise ValueError(
                    "mojolearn: sample_weight must be finite and non-negative"
                )
            if not weights.max() > 0.0:
                raise ValueError("mojolearn: sample_weight must have positive sum")
            n_weights = n_rows
        source_array = Array.from_list([int(i) for i in self.sources], "<u4")  # glue: packs the sources argument tuple
        self.model_ = self._bind(
            "_mojolearn_gbdt"
        ).gbdt_fit_two_level_feature_freq(
            addr_ro(Xa, name="X"), addr_ro(ya, name="y"),
            addr_ro(weights, name="sample_weight"),
            addr_ro(source_array, name="sources"),
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
        self._ordered_options(self.n_estimators, self.max_depth,
                              self.border_count, self.learning_rate,
                              self.l2_leaf_reg)
        Xa, _ = as_f32_colmajor(X, name="X")
        n_rows, n_features = Xa.shape
        if n_rows < 4 or n_rows > _U32_MAX or n_features < 1:
            raise ValueError("mojolearn: ordered RMSE requires >=4 rows and >=1 feature")
        if not all_finite(Xa):
            raise ValueError("mojolearn: ordered RMSE X must be finite")
        try:
            ya, _ = as_f32_c(y, ndim=1, name="y")
        except (TypeError, ValueError):
            raise ValueError("mojolearn: y must be one-dimensional with one value per row") from None
        if ya.shape[0] != n_rows:
            raise ValueError("mojolearn: y must be one-dimensional with one value per row")
        if not all_finite(ya):
            raise ValueError("mojolearn: y must be finite")
        # DEVIATION 2337: the permutation is validated as int64 -- a list
        # is refused if any entry is not an integer (floats and bools were
        # refused by dtype kind before), a buffer by its dtype -- then
        # bounds and bijectivity are `min`/`max`/`set` over the storage
        # view (C-driven, O(rows)), and a uint32 copy is what crosses,
        # the dtype the binding reads and the old code passed.
        bijection_error = ValueError(
            "mojolearn: permutation must be an integer bijection of row ids"
        )
        if isinstance(permutation, (list, tuple)):
            if any(is_bool(v) or not isinstance(v, numbers.Integral)
                   for v in permutation):  # cpu-route: user permutation Python list type check [py-data-loop]
                raise bijection_error
        else:
            dtype = getattr(permutation, "dtype", None)
            if dtype is not None:
                kind = getattr(dtype, "kind", None) or str(dtype).lstrip("<>|=")[:1]
                if kind not in ("i", "u"):
                    raise bijection_error
        try:
            order64, _ = as_i64_c(permutation, ndim=1, name="permutation")
        except (TypeError, ValueError, OverflowError):
            raise bijection_error from None
        # n distinct ids in [0, n) are a bijection: the base binding's
        # `check_indices_i64` (range, then a bit set) answers both in one
        # pass (lane pyglue-sweep, Oct 3: was min/max/set over the rows)
        if order64.shape[0] != n_rows or n_rows < 1 or int(_native("check_indices_i64")(
                addr_ro(order64, name="permutation"), n_rows, n_rows)) != 0:
            raise bijection_error
        order = order64.astype("<u4")
        if sample_weight is None:
            weights = ya  # unread while params[2] == 0
            n_weights = 0
        else:
            try:
                weights, _ = as_f32_c(sample_weight, ndim=1, name="sample_weight")
            except (TypeError, ValueError):
                raise ValueError("mojolearn: sample_weight must have one value per row") from None
            if weights.shape[0] != n_rows:
                raise ValueError("mojolearn: sample_weight must have one value per row")
            if not all_finite(weights) or weights.min() < 0:
                raise ValueError("mojolearn: sample_weight must be finite and non-negative")
            if not weights.max() > 0:
                raise ValueError("mojolearn: sample_weight must have positive sum")
            n_weights = n_rows
        binding = self._bind("_mojolearn_gbdt")
        if not hasattr(binding, "gbdt_fit_ordered_rmse"):
            raise RuntimeError(
                "mojolearn: this GBDT binary predates OrderedRMSE; "
                "install or build a matching native extension"
            )
        model = binding.gbdt_fit_ordered_rmse(
            addr_ro(Xa, name="X"), addr_ro(ya, name="y"),
            addr_ro(weights, name="sample_weight"),
            addr_ro(order, name="permutation"),
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
