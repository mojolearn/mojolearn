# SPDX-License-Identifier: Apache-2.0
"""RUN of the GradientBoosting fit door (bindings/trees_public_gbdt_fit_door.mojo):
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
from std.os import getenv
from std.memory import bitcast
from checks.numerics import portable_exp64, portable_log64, pinned_mul_f64
from core.host_parallel import host_parallelize
from core.host_finite_scan import host_all_finite_f32

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


def _sabotaged() -> Bool:
    return getenv("MOJOLEARN_CATBOOST_DEFAULTS_SABOTAGE") == "1" and getenv("MOJOLEARN_HOST_ALLOW_SABOTAGE") == "1"


def _pair_end_range(v: Int, i: Int, j: Int, n_rows: Int) raises:
    if not (0 <= v and v < n_rows):
        raise Error("mojolearn: pairs[" + String(i) + "][" + String(j) + "] = " + String(v) + " is outside the "
                    + String(n_rows) + " rows")


# ------------------------------------------------------------ the two-level FeatureFreq checks (lane mf-meta5)
# ONE BODY of `ExperimentalTwoLevelFeatureFreq.fit`'s data checks: the fit door
# (bindings/trees_public_gbdt_door.mojo `feature_freq_unpack`) and the member's fit (core/gbdt_member.mojo
# kind 3, `_fit_feature_freq`) call these, in the door's order and words. Every column is an independent
# unit of the scan (host workers); the refusal is then the first one in the door's order.
comptime _FF_OK = 0
comptime _FF_NONFINITE = 1
comptime _FF_CODES = 2
comptime _FF_SPARSE = 3
comptime _FF_CONSTANT = 4


def _ff_columns_text(values: List[Int]) -> String:
    """`[a, b, ...]`, the refusal's spelling of the source columns."""
    var text = String("[")
    for i in range(len(values)):  # small-loop(the named source columns of one refusal message)
        if i > 0:
            text += ", "
        text += String(values[i])
    return text + "]"


def gbdt_feature_freq_check(xcm: Int, n: Int, d: Int, sources: List[Int]) raises:
    """The FeatureFreq fit's X checks over the column-major float32 X at `xcm` (n x d) and the source
    columns: rows and columns, the sources in range, X finite, every source a column of non-negative
    integer codes densely coded 0..k-1 (k >= 2, k <= n), every other column varying."""
    if n == 0 or d < 2:
        raise Error("mojolearn: experimental FeatureFreq needs rows and columns")
    for i in range(len(sources)):  # small-loop(the named source columns)
        if sources[i] < 0 or sources[i] >= d:
            raise Error("mojolearn: sources " + _ff_columns_text(sources) + " outside " + String(d) + " features")
    var is_source = List[Int](length=d, fill=0)
    for i in range(len(sources)):  # small-loop(the named source columns)
        is_source[sources[i]] = 1
    var status = List[Int](length=d, fill=_FF_OK)
    var highs = List[Int](length=d, fill=0)
    var source_at = Int(is_source.unsafe_ptr())
    var status_at = Int(status.unsafe_ptr())
    var highs_at = Int(highs.unsafe_ptr())

    def _column(f: Int) {imm xcm, imm n, imm source_at, imm status_at, imm highs_at}:
        var col = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=xcm + 4 * f * n)
        var state = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=status_at)
        var high = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=highs_at)
        var lo = Float64(col[0])
        var hi = Float64(col[0])
        var integral = True
        for i in range(n):  # native data loop (one column's rows)
            var v = Float64(col[i])
            if not isfinite(v):
                state[f] = _FF_NONFINITE
                return
            lo = min(lo, v)
            hi = max(hi, v)
            if v != floor(v):
                integral = False
        if MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=source_at)[f] == 0:
            if lo == hi:
                state[f] = _FF_CONSTANT
            return
        if not integral or lo < 0.0:
            state[f] = _FF_CODES
            return
        high[f] = Int(hi)
        var k = Int(hi) + 1
        if k < 2 or k > n:
            state[f] = _FF_SPARSE
            return
        var seen = List[Bool](length=k, fill=False)
        for i in range(n):  # native data loop (one source column's codes)
            seen[Int(col[i])] = True
        for c in range(k):  # native data loop (one source column's code presence)
            if not seen[c]:
                state[f] = _FF_SPARSE
                return

    host_parallelize(_column, d)
    _ = len(is_source)  # read through the scan above
    for f in range(d):  # driver-loop: over the feature columns (the door's first refusal: X finite)
        if status[f] == _FF_NONFINITE:
            raise Error("mojolearn: experimental FeatureFreq X must be finite")
    for i in range(len(sources)):  # small-loop(the named source columns, in the door's order)
        var f = sources[i]
        if status[f] == _FF_CODES:
            raise Error("mojolearn: experimental FeatureFreq source columns must "
                        + "contain non-negative integer category codes")
        if status[f] == _FF_SPARSE:
            raise Error("mojolearn: every experimental FeatureFreq source must "
                        + "be densely coded 0..k-1; column " + String(f) + " has codes up to " + String(highs[f])
                        + " with at least one missing (or fewer than two codes)")
    for f in range(d):  # driver-loop: over the feature columns (the numeric ones vary)
        if status[f] == _FF_CONSTANT:
            raise Error("mojolearn: experimental FeatureFreq numeric columns "
                        + "must vary; column " + String(f) + " is constant")


#: The OrderedRMSE fit's permutation refusal (the door's words).
comptime GBDT_ORDERED_BIJECTION = "mojolearn: permutation must be an integer bijection of row ids"


def _finite_words(x: Int, count: Int) -> Bool:
    """The `count` float32 words at `x` are finite: the one host scan (SIMD tasks across cores)."""
    if count <= 0:
        return True
    return host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=x), count)


def gbdt_ordered_rows_check(x: Int, n: Int, d: Int) raises:
    """THE ONE BODY of the OrderedRMSE fit's X checks (lane mf-hostent5; the door's `ordered_rmse_unpack` and
    the Gbdt member's OrderedRMSE kind both run it, after their admission of X as float32 words): at least 4
    rows (at most 2^32 - 1, the uint32 row ids) and one feature, then every value finite (one parallel scan;
    the layout of X does not matter to it)."""
    if n < 4 or n > 4294967295 or d < 1:
        raise Error("mojolearn: ordered RMSE requires >=4 rows and >=1 feature")
    if not _finite_words(x, n * d):
        raise Error("mojolearn: ordered RMSE X must be finite")


def gbdt_ordered_target_check(y: Int, n: Int) raises:
    """THE ONE BODY of the OrderedRMSE fit's target check (door and member): the `n` float32 targets finite."""
    if not _finite_words(y, n):
        raise Error("mojolearn: y must be finite")


def gbdt_ordered_order(ids: Int, m: Int, n: Int) raises -> List[UInt32]:
    """THE ONE BODY of the OrderedRMSE fit's permutation check and cast (door and member): the `m` int64 row
    ids at `ids` must be a bijection of 0 .. n-1 (`m == n`, each in range, none repeated; the door's
    `base_check_indices_i64` test), returned as the uint32 order the fit reads (the door's `trees_cast`
    <i8 -> <u4 of the same values). Refused in the door's words (`GBDT_ORDERED_BIJECTION`). SERIAL BY NATURE:
    the repeat test reads the marks of every earlier id (one pass over n words, as the helper's)."""
    if m != n or n < 1:
        raise Error(GBDT_ORDERED_BIJECTION)
    var src = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=ids)
    var seen = List[Bool](length=n, fill=False)
    var order = List[UInt32](length=n, fill=UInt32(0))
    for i in range(n):  # native data loop (the permutation, one pass: the repeat marks)
        var v = Int(src[i])
        if v < 0 or v >= n or seen[v]:
            raise Error(GBDT_ORDERED_BIJECTION)
        seen[v] = True
        order[i] = UInt32(v)
    return order^


def gbdt_weights_check(w: Int, n: Int) raises:
    """The OrderedRMSE / FeatureFreq fits' sample_weight checks (the door's `_checked_weights` after its
    admission): finite and non-negative, with a positive sum."""
    var p = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=w)
    var lowest = 0.0
    var highest = 0.0
    for i in range(n):  # native data loop (the weights, one pass)
        var v = Float64(p[i])
        if not isfinite(v):
            raise Error("mojolearn: sample_weight must be finite and non-negative")
        if i == 0 or v < lowest:
            lowest = v
        if i == 0 or v > highest:
            highest = v
    if n > 0 and lowest < 0.0:
        raise Error("mojolearn: sample_weight must be finite and non-negative")
    if not highest > 0.0:
        raise Error("mojolearn: sample_weight must have positive sum")
