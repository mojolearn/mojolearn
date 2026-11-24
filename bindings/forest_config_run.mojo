# SPDX-License-Identifier: Apache-2.0
"""The forest configuration as plain values, and the fit request built from
it (RUN of bindings/trees_public_forest_door.mojo; no Python import).

`ForestConfig` is what a forest fit reads of the estimator's derived
configuration (the `_cfg` dictionary and, for Extra Trees, `_criterion_code`),
read ONCE per fit by the door's UNPACK (`forest_config`). It holds the fit's
parameter slots in the image's layout (the 16/17 RF slots of
`_RandomForestBase._fit_params`, the 22 ET slots of `extratrees._fit_params`)
as the two views and the flag byte the request carries, with the slots that
depend on the data left empty, and the few values the door's decisions read
(the criterion, `oob_score`, whether the classes are weighted). `forest_request`
fills the data slots (rows, columns, classes, the RF feature fraction, a
DecisionTreeRegressor's image bin count, the ET bootstrap row count) and
returns the typed `ForestFitArgs`.

A parameter conversion that Python refused during UNPACK is kept as its text
and raised here, at the place the fit reached it before (after the labels,
the weights and the finiteness check), so a fit with two faults reports the
same one.

Each function is a per-fit scalar decision over about twenty parameter
slots; there is no independent work to split across cores.
"""
from std.math import sqrt, isinf, isnan
from checks.numerics import portable_log2_64
from core.entry_table import EntryRef
from core.forest_calls import xf_dt_default_bins
from core.forest_fit_request import ForestFitArgs
from xtrees.trees_forest_run import ForestFit

comptime FOREST_FEATURES_ALL = 0
comptime FOREST_FEATURES_SQRT = 1
comptime FOREST_FEATURES_LOG2 = 2
comptime FOREST_FEATURES_COUNT = 3
comptime FOREST_FEATURES_FRACTION = 4
comptime FOREST_FEATURES_REFUSED = 5

#: the data slots of both layouts
comptime FOREST_SLOT_ROWS = 0
comptime FOREST_SLOT_COLS = 1
comptime FOREST_SLOT_CLASSES = 2
#: RF layout
comptime FOREST_RF_SLOT_FRACTION = 6
comptime FOREST_RF_SLOT_BINS = 7
comptime FOREST_RF_SLOT_BOOTSTRAP = 11
comptime FOREST_RF_SLOT_MAX_SAMPLES = 12
#: ET layout
comptime FOREST_ET_SLOT_CLASS_WEIGHTED = 16
comptime FOREST_ET_SLOT_BOOTSTRAP_ROWS = 18

#: what the regression target check reads of y
comptime FOREST_TARGET_ANY = 0
comptime FOREST_TARGET_POISSON = 1
comptime FOREST_TARGET_POSITIVE = 2


struct ForestFeatures(Movable):
    """RandomForest's `max_features` as given: all features, 'sqrt', 'log2',
    a count, a fraction, or a form refused in words (`refusal`)."""
    var form: Int
    var count: Int
    var fraction: Float64
    var refusal: String

    def __init__(out self):
        self.form = FOREST_FEATURES_ALL
        self.count = 0
        self.fraction = 1.0
        self.refusal = String("")


struct ForestConfig(Movable):
    """One forest fit's configuration (see the module docstring)."""
    var et: Bool
    var ints: List[Int]
    var floats: List[Float64]
    var bad: List[UInt8]
    #: the XF_W_CRITERION word: the RF criterion code, -1 for the ET fits
    var criterion: Int
    #: the criterion code the regression target check reads, with the codes
    #: of the deviance criteria (-2 where the door did not need to read one)
    var target_code: Int
    var poisson: Int
    var gamma: Int
    var inverse_gaussian: Int
    var oob_score: Bool
    var class_weighted: Bool
    var features: ForestFeatures
    #: a DecisionTreeRegressor left at n_bins=None: the image's bin count
    var bins_image: Bool
    #: ET `max_samples` as a float: the bootstrap rows are this fraction of
    #: the fit's rows (0.0: the slot already holds the count)
    var rows_fraction: Float64
    #: the first refused parameter conversion, and the first one the fit
    #: reached after the bootstrap row count
    var refusal: String
    var refusal_late: String

    def __init__(out self, et: Bool, var ints: List[Int], var floats: List[Float64], var bad: List[UInt8]):
        self.et = et
        self.ints = ints^
        self.floats = floats^
        self.bad = bad^
        self.criterion = -1
        self.target_code = -1
        self.poisson = -2
        self.gamma = -2
        self.inverse_gaussian = -2
        self.oob_score = False
        self.class_weighted = False
        self.features = ForestFeatures()
        self.bins_image = False
        self.rows_fraction = 0.0
        self.refusal = String("")
        self.refusal_late = String("")


def forest_features_fraction(features: ForestFeatures, n_features: Int) raises -> Float64:
    """`randomforest._max_features_fraction`: the fraction the RF image
    takes. 'sqrt' uses the correctly rounded square root and 'log2' the
    pinned `portable_log2_64` the builder uses."""
    if features.form == FOREST_FEATURES_ALL:
        return 1.0
    if features.form == FOREST_FEATURES_SQRT:
        return sqrt(Float64(n_features)) / Float64(n_features)
    if features.form == FOREST_FEATURES_LOG2:
        return portable_log2_64(Float64(max(2, n_features))) / Float64(n_features)
    if features.form == FOREST_FEATURES_COUNT:
        if not (1 <= features.count and features.count <= n_features):
            raise Error("max_features=" + String(features.count) + " is outside [1, " + String(n_features) + "]")
        return Float64(features.count) / Float64(n_features)
    if features.form == FOREST_FEATURES_FRACTION:
        return features.fraction
    raise Error(features.refusal)


def _slot_int(mut ints: List[Int], mut floats: List[Float64], mut bad: List[UInt8], k: Int, value: Int):
    ints[k] = value
    floats[k] = Float64(value)
    bad[k] = UInt8(0)


def _slot_float(mut ints: List[Int], mut floats: List[Float64], mut bad: List[UInt8], k: Int, value: Float64):
    # Both readings succeed: the integer view is the truncated value where it
    # has one.
    floats[k] = value
    ints[k] = 0
    if not isnan(value) and value > -9.0e18 and value < 9.0e18:
        ints[k] = Int(value)
    bad[k] = UInt8(0)


def _bootstrap_rows(fraction: Float64, rows: Int) raises -> Int:
    """`extratrees._n_samples_bootstrap` of a float `max_samples`:
    max(int(fraction * rows), 1), with Python's refusals of a product that
    has no integer."""
    var product = fraction * Float64(rows)
    if isnan(product):
        raise Error("ValueError: cannot convert float NaN to integer")
    if isinf(product):
        raise Error("OverflowError: cannot convert float infinity to integer")
    if product >= 9.0e18:
        raise Error("OverflowError: Python int too large to convert to C long")
    return max(Int(product), 1)


def forest_request(config: ForestConfig, key: EntryRef, rows: Int, cols: Int, classes: Int, x_addr: Int, y_addr: Int,
                   fit: ForestFit, unweighted: Bool) raises -> ForestFitArgs:
    """The typed request of one fit: the configuration's slots with the data
    slots filled, the X and y (label codes or targets) addresses and the
    trailing words of `fit` (row weights, the tree-ID start, the auxiliary
    outputs, the largest weight). `unweighted` clears the ET class-weight
    slot (a class-weighted ET fit carries its class weights as row weights).
    `key` is read only for a DecisionTreeRegressor left at n_bins=None
    (XF_DT_DEFAULT_BINS)."""
    var ints = config.ints.copy()
    var floats = config.floats.copy()
    var bad = config.bad.copy()
    if config.et:
        if config.refusal != "":
            raise Error(config.refusal)
        if config.rows_fraction != 0.0:
            _slot_int(ints, floats, bad, FOREST_ET_SLOT_BOOTSTRAP_ROWS, _bootstrap_rows(config.rows_fraction, rows))
        if config.refusal_late != "":
            raise Error(config.refusal_late)
        if unweighted:
            _slot_int(ints, floats, bad, FOREST_ET_SLOT_CLASS_WEIGHTED, 0)
    else:
        _slot_float(ints, floats, bad, FOREST_RF_SLOT_FRACTION, forest_features_fraction(config.features, cols))
        if config.refusal != "":
            raise Error(config.refusal)
        if config.bins_image:
            # XF_DT_DEFAULT_BINS: the image's bin count (128 in one without its own)
            _slot_int(ints, floats, bad, FOREST_RF_SLOT_BINS, xf_dt_default_bins(key))
    _slot_int(ints, floats, bad, FOREST_SLOT_ROWS, rows)
    _slot_int(ints, floats, bad, FOREST_SLOT_COLS, cols)
    _slot_int(ints, floats, bad, FOREST_SLOT_CLASSES, classes)
    var args = ForestFitArgs(x_addr, y_addr, ints^, floats^, bad^, config.criterion)
    args.weights = fit.weights
    args.tree_start = fit.tree_start
    args.oob = fit.oob
    args.score = fit.score
    args.importance = fit.importance
    args.weight_max = fit.weight_max
    args.loop = fit.loop
    return args^


def forest_target_need(config: ForestConfig) -> Int:
    """What a regressor's criterion requires of y: nothing, Poisson's
    (y >= 0 with a positive sum) or a positive y (gamma, inverse Gaussian)."""
    if config.target_code == config.poisson:
        return FOREST_TARGET_POISSON
    if config.target_code == config.gamma or config.target_code == config.inverse_gaussian:
        return FOREST_TARGET_POSITIVE
    return FOREST_TARGET_ANY


def forest_target_check(config: ForestConfig, finite: Bool, lowest: Float64, highest: Float64) raises:
    """The deviance criteria's target refusals (RF and ET wording) over the
    statistics the door read of y: every value finite, the smallest and
    (Poisson) the largest value."""
    var need = forest_target_need(config)
    if need == FOREST_TARGET_POISSON:
        if not finite or lowest < 0.0 or not highest > 0.0:
            if not config.et:
                raise Error("criterion='poisson' requires y >= 0 with a positive"
                            + " sum: PoissonGain returns -max() for a non-positive"
                            + " label sum (objectives.cuh:251-253), which would fit"
                            + " a stump silently")
            raise Error("criterion='poisson' requires y >= 0 with a positive sum: the"
                        + " Poisson gain is -max() for a non-positive label sum"
                        + " (objectives.cuh:251-253), which would fit a stump silently")
    elif need == FOREST_TARGET_POSITIVE:
        if not finite or lowest <= 0.0:
            var name = String("'gamma'") if config.target_code == config.gamma else String("'inverse_gaussian'")
            if not config.et:
                raise Error("criterion=" + name + " requires y > 0: its"
                            + " gain returns -max() for a non-positive label sum"
                            + " (objectives.cuh:279-281, :306-308), which would fit"
                            + " a stump silently")
            raise Error("criterion=" + name + " requires y > 0: its gain is"
                        + " -max() for a non-positive label sum, which would fit a stump"
                        + " silently")
