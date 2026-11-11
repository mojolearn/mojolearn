# SPDX-License-Identifier: Apache-2.0
"""RUN of the GradientBoosting fit door (bindings/trees_public_gbdt_fit.mojo):
the parts of `GradientBoosting.fit` and its option checks that read only
typed values. The door's UNPACK reads the estimator's Python fields and the
connected option tables into `GbdtSearchFacts` / plain scalars; these
functions decide on those values with the same order and wording the door
had. No Python import: `tools/python_free_dirs.sh` covers this directory.

Lane doorsplit-trees (2026-11-11): moved word for word from the door;
`gbdt_auto_rate` is the arithmetic half of `gbdt_auto_learning_rate` (the
coefficient lookup stays in UNPACK). Each function is a scalar decision on
one fit's options; there is no independent work to split across cores.
"""
from std.math import isfinite, floor, copysign
from std.memory import bitcast
from checks.numerics import portable_exp64, portable_log64, pinned_mul_f64

comptime GBDT_F32_MAX = 3.4028234663852886e+38


def gbdt_nan() -> Float64:
    return bitcast[DType.float64](UInt64(0x7FF8000000000000))


def gbdt_c_round(number: Float64, precision: Int) -> Float64:
    """`ensemble._c_round`: C `round` (halves away from zero) at
    `precision` decimals."""
    var multiplier = Float64(1.0)
    for _ in range(precision):  # small-loop(precision): six decimal places
        multiplier = multiplier * 10.0
    var scaled = pinned_mul_f64(number, multiplier)
    var rounded = floor(abs(scaled) + 0.5)
    return copysign(rounded, scaled) / multiplier


def gbdt_auto_rate(a: Float64, b: Float64, c: Float64, d: Float64, n_rows: Float64,
                   iterations: Float64) raises -> Float64:
    """CatBoost's GPU automatic learning rate from one table row's four
    coefficients (`ensemble.catboost_auto_learning_rate` after its lookup).
    `exp` and `log` are the package's portable binary64 routines."""
    if iterations <= 0.0 or n_rows <= 0.0:
        raise Error("math domain error")
    var custom = portable_exp64(pinned_mul_f64(c, portable_log64(iterations)) + d)
    var default_iter = portable_exp64(pinned_mul_f64(c, portable_log64(1000.0)) + d)
    var default_rate = portable_exp64(pinned_mul_f64(a, portable_log64(n_rows)) + b)
    if not isfinite(custom) or not isfinite(default_iter) or not isfinite(default_rate):
        raise Error("OverflowError: math range error")
    var rate = pinned_mul_f64(default_rate, custom) / default_iter
    return gbdt_c_round(min(rate, 0.5), 6)


@fieldwise_init
struct GbdtOrderedFacts(Copyable, Movable):
    """What `ensemble._check_ordered_options` reads, as values: the option
    tables' membership answers and the texts its refusals print (Python
    `str` and `repr` of the loss and score, taken by UNPACK)."""
    var grow_symmetric: Bool
    var loss_multi_output: Bool
    var loss_multi_regression: Bool
    var loss_querywise: Bool
    var score_cosine: Bool          # Cosine or NewtonCosine
    var leaf_exact: Bool
    var pointwise: Bool
    var fraction_below_one: Bool
    var loss_str: String
    var loss_repr: String
    var score_str: String


def gbdt_check_ordered_options(boosting: String, facts: GbdtOrderedFacts) raises:
    """`ensemble._check_ordered_options` on typed facts."""
    if boosting != "Ordered":
        return
    if not facts.grow_symmetric:
        raise Error("mojolearn: Ordered boosting is not supported for nonsymmetric "
                    + "trees. (catboost_options.cpp:757-759)")
    if facts.loss_multi_output:
        raise Error("mojolearn: On GPU loss " + facts.loss_str + " can't be used with ordered "
                    + "boosting (catboost_options.cpp:949-967)")
    if facts.loss_multi_regression:
        raise Error("NotImplementedError: mojolearn: boosting_type='Ordered' with loss=" + facts.loss_repr + " is not "
                    + "carried: the Ordered arm is one-dimensional; use "
                    + "boosting_type='Plain'")
    if not facts.score_cosine:
        raise Error("mojolearn: Score function " + facts.score_str + " can't be used with "
                    + "ordered boosting (catboost_options.cpp:972-978)")
    if facts.leaf_exact:
        raise Error("mojolearn: Exact leaf estimation method don't work with ordered "
                    + "boosting on GPU (catboost_options.cpp:346-350)")
    if facts.loss_querywise:
        raise Error("NotImplementedError: mojolearn: boosting_type='Ordered' with loss=" + facts.loss_repr + " is not "
                    + "implemented: their folds follow the query grouping "
                    + "(dynamic_boosting.h:189-223); use boosting_type='Plain'")
    if facts.pointwise:
        raise Error("mojolearn: use_pointwise_searcher selects the doc-parallel Plain "
                    + "searcher; an Ordered fit runs the feature-parallel fold searcher")
    if facts.fraction_below_one:
        raise Error("NotImplementedError: mojolearn: boosting_type='Ordered' with feature_fraction < 1 is "
                    + "not implemented")


@fieldwise_init
struct GbdtSearchFacts(Copyable, Movable):
    """What `ensemble._validate_search_options` reads, as values. `strength`
    is `float(random_strength)` of a real non-bool, NaN for anything else or
    a refused conversion; `strength_str` is Python's `str` of that float
    (the refusal prints it)."""
    var pointwise: Bool
    var grow_symmetric: Bool
    var grow_repr: String
    var score_l2: Bool              # L2 or NewtonL2
    var score_repr: String
    var strength_unset: Bool        # random_strength is None
    var strength: Float64
    var strength_str: String
    var catboost_strength: Float64  # the tables' defaults
    var legacy_strength: Float64


def gbdt_validate_search_options(facts: GbdtSearchFacts) raises -> Float64:
    """`ensemble._validate_search_options`: the strength the fit uses."""
    if facts.pointwise and not facts.grow_symmetric:
        raise Error("mojolearn: use_pointwise_searcher=True is an OBLIVIOUS searcher; "
                    + "grow_policy=" + facts.grow_repr + " requires the greedy subsets searcher")
    if facts.strength_unset:
        if facts.grow_symmetric and not facts.score_l2:
            return facts.catboost_strength
        return facts.legacy_strength
    var strength = facts.strength
    if not isfinite(strength) or not (strength >= 0 and strength <= GBDT_F32_MAX):
        raise Error("mojolearn: random_strength must be finite, nonnegative and "
                    + "<= Float32.MAX_FINITE")
    if strength != 0.0 and facts.score_l2:
        raise Error("mojolearn: random_strength=" + facts.strength_str + " does nothing under "
                    + "score_function=" + facts.score_repr + "; use Cosine or NewtonCosine, "
                    + "or random_strength=0.0")
    return strength


def gbdt_id_code(typestr: String) -> Int:
    """The id vector's element kind as one word, chosen once per vector."""
    if typestr == "<i8":
        return 0
    if typestr == "<i4":
        return 1
    if typestr == "<u4":
        return 2
    if typestr == "<i2":
        return 3
    if typestr == "<u2":
        return 4
    if typestr == "|i1" or typestr == "<i1":
        return 5
    return 6


def gbdt_id_typestr_supported(t: String) -> Bool:
    """The integer id kinds the one-pass group scan reads."""
    return (t == "<i8" or t == "<i4" or t == "<u4" or t == "<i2" or t == "<u2" or t == "|i1" or t == "<i1"
            or t == "|u1" or t == "<u1")


def gbdt_check_group_rows(n: Int, n_rows: Int) raises:
    """`_group_sizes`' length and row-count refusals."""
    if n != n_rows:
        raise Error("mojolearn: Length of group_id=" + String(n) + " and length of "
                    + "data=" + String(n_rows) + " are different.")
    if n_rows > 0xFFFFFFFF:
        raise Error("mojolearn: group_id needs at most 2**32 - 1 rows")


def gbdt_bias_of(text: String) raises -> Float64:
    """The `bias ...` line's float64 bits (0.0 without one)."""
    var at = -1
    if text.startswith("bias "):
        at = 0
    else:
        var found = text.find("\nbias ")
        if found >= 0:
            at = found + 1
    if at < 0:
        return 0.0
    var end = text.find("\n", at)
    var line = String(text[byte=at:text.byte_length()]) if end < 0 else String(text[byte=at:end])
    var words = line.split()
    if len(words) < 2:
        raise Error("IndexError: list index out of range")
    var second = String(words[1])
    var slash = second.find("/")
    if slash < 0:
        raise Error("IndexError: list index out of range")
    var rest = String(second[byte=slash + 1:second.byte_length()])
    var next = rest.find("/")
    var hex = rest if next < 0 else String(rest[byte=0:next])
    var bits = UInt64(0)
    var digits = hex.as_bytes()
    for i in range(len(digits)):  # small-loop(hex): at most sixteen hex digits of one word
        var c = Int(digits[i])
        var d = -1
        if c >= 48 and c <= 57:
            d = c - 48
        elif c >= 97 and c <= 102:
            d = c - 87
        elif c >= 65 and c <= 70:
            d = c - 55
        if d < 0:
            raise Error("invalid literal for int() with base 16: '" + hex + "'")
        bits = (bits << 4) | UInt64(d)
    return bitcast[DType.float64](bits)
