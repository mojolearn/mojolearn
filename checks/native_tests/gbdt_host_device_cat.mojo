# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""The host GradientBoosting fit against the device fit, model text byte for
byte, on the symmetric configurations the host column trains since lane
pv-gbdt-cpu-cat (2db07105a5): numeric Logloss and RMSE, RMSE with CTR and
one-hot columns, each bootstrap, and random_strength above zero.

    pixi run check-native-gpu gbdt_host_device_cat

Lane pv-verify-gbdt-cat (2026-11-22). The host loop (`gbdt/host/
gbdt_oracle.mojo::gbdt_host_boost`) took `fma(leaf, rate, cursor)` where the
device takes `cursor + round(leaf * rate)` (fb11), and the only native
host-against-device comparison of that loop (`gbdt/checks/
tree_ctr_host_check.mojo`) fitted at learning rate 0.5, where `leaf * 0.5`
is exact and the two forms are the same bits, so it passed with the defect
live. Every fit here runs at `CAT_RATE` 0.1, not a power of two.

Both fits go through the request `GradientBoosting.fit` builds
(`gbdt_public_fits.fit`): the host image (`gbdt_fit_native_run`, the CPU
binding's fit) and the device image (`gbdt.mojolearn_gbdt_run`, the GPU
binding's fit). A difference is reported as the first differing line.

`test_rank_host_text_is_device_text` (lane verify-gbdt-rank, 2026-11-22)
does the same for the three ranking losses (QueryRMSE, PairLogit, YetiRank):
the greedy searcher deterministic, under the Bayesian bootstrap with score
noise and under Bernoulli; the pointwise searcher at Cosine and NewtonL2;
PairLogit on explicit weighted pairs on both searchers. Uneven query groups,
every fit at `RANK_RATE` 0.1. The ranking host fits
(`gbdt/host/gbdt_oracle_losses.mojo`, `gbdt_oracle_pointwise_rank.mojo`)
restate the device's `cursor + round(leaf * rate)` and its grouped
derivatives; a fused or reordered update there moves the model text.

Parallel plan: the host fits are independent and run across the host cores
first (one unit each); the device fits then run one at a time, because each
owns the one GPU for its whole graph (as `gbdt_public_fits`'s device column).
Device column only: the CPU build has no device arm.

`test_gaps_host_text_is_device_text` (lane gbdt-gaps, 2026-11-23): sample
weights on RMSE, Poisson, Tweedie, Depthwise, Lossguide, YetiRank and
generated-pair PairLogit, and eval sets (the Iter detector and
use_best_model) on the symmetric RMSE fit and on Depthwise and Lossguide,
which the CPU column refused until that lane.

`test_rest_host_text_is_device_text` (lane gbdt-cpu-rest, 2026-11-23): the
rest of what the device trained and the CPU column refused until that lane:
sample weights with QueryRMSE (greedy and pointwise searchers), YetiRank on
the pointwise searcher, Exact leaves (MAE, MAPE), MultiClass (times class
weights), one-hot and CTR columns; X carrying NaN on RMSE (symmetric and
Depthwise); Cosine under Lossguide; Quantile on Depthwise (weighted) and MAE
Exact on Lossguide with an eval set; feature_fraction 0.5 on the symmetric
RMSE and Logloss loop and on Depthwise RMSE; the pointwise searcher without
sample weights or an eval set. Every fit at `REST_RATE` 0.1.
"""
from std.math import nan
from std.memory import UnsafePointer

from checks.native_tests.gbdt_public_fits import (
    GbdtInputs,
    GbdtOpts,
    _DeviceFit,
    _HostFit,
    fit,
)
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases

comptime CAT_ROWS = 2000
comptime CAT_TREES = 12
#: NOT a power of two (see the module docstring)
comptime CAT_RATE = 0.1
comptime CAT_UNITS = 11


def _unit01(i: Int, salt: Int) -> Float32:
    """A value in [0, 1) from an integer hash of (i, salt): the same bits on
    every host."""
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt + 1) * UInt64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> 31)) * UInt64(0x94D049BB133111EB)
    h = h ^ (h >> 29)
    return Float32(Int(h >> 40)) / Float32(16777216.0)


def _inputs(rmse: Bool) -> GbdtInputs:
    """Column-major X: column 0 holds 40 categories (a CTR column), column 1
    two (a one-hot column), columns 2 and 3 are numeric. The target depends
    on all four."""
    var n = CAT_ROWS
    var x = List[Float32](capacity=4 * n)
    for r in range(n):  # driver-loop: over the fixture's rows (fixture construction, once per unit)
        x.append(Float32(Int(_unit01(r, 0) * Float32(40.0))))
    for r in range(n):  # driver-loop: over the fixture's rows
        x.append(Float32(1.0) if _unit01(r, 1) > Float32(0.5) else Float32(0.0))
    for r in range(n):  # driver-loop: over the fixture's rows
        x.append(_unit01(r, 2) * Float32(4.0) - Float32(2.0))
    for r in range(n):  # driver-loop: over the fixture's rows
        x.append(_unit01(r, 3) * Float32(4.0) - Float32(2.0))
    var y = List[Float32](capacity=n)
    for r in range(n):  # driver-loop: over the fixture's rows
        var v = (Float32(0.05) * x[r] + Float32(0.7) * x[n + r] + x[2 * n + r]
                 - Float32(0.5) * x[3 * n + r] + Float32(0.3) * (_unit01(r, 4) - Float32(0.5)))
        if rmse:
            y.append(v)
        else:
            y.append(Float32(1.0) if v > Float32(1.0) else Float32(0.0))
    return GbdtInputs(x^, n, 4, y^)


def _opts(unit: Int) raises -> GbdtOpts:
    """The configuration of `unit`; every bootstrap and noise option explicit."""
    var rmse = _is_rmse(unit)
    var o = GbdtOpts(String("RMSE") if rmse else String("Logloss"), CAT_TREES, 6)
    o.learning_rate = CAT_RATE
    o.l2_leaf_reg = 3.0
    o.leaf_iterations = 1
    o.random_state = 7
    o.bootstrap = "No"
    o.random_strength = 0.0
    if unit == 0:  # numeric Logloss, deterministic
        pass
    elif unit == 1:  # numeric RMSE, deterministic
        pass
    elif unit == 2:  # numeric Logloss at the stochastic defaults
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 1.0
        o.random_strength = 1.0
    elif unit == 3:  # RMSE + CTR at the stochastic defaults
        o.cat_features = [0]
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 1.0
        o.random_strength = 1.0
    elif unit == 4:  # RMSE + CTR, Bernoulli
        o.cat_features = [0]
        o.bootstrap = "Bernoulli"
        o.subsample = 0.7
    elif unit == 5:  # RMSE + CTR, Poisson
        o.cat_features = [0]
        o.bootstrap = "Poisson"
        o.subsample = 0.66
    elif unit == 6:  # Logloss + CTR, Bayesian at temperature 2 with noise
        o.cat_features = [0]
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 2.0
        o.random_strength = 1.0
    elif unit == 7:  # Logloss + one-hot at the stochastic defaults
        o.cat_features = [1]
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 1.0
        o.random_strength = 1.0
    elif unit == 8:  # RMSE + one-hot, Bernoulli
        o.cat_features = [1]
        o.bootstrap = "Bernoulli"
        o.subsample = 0.7
    elif unit == 9:  # RMSE + CTR + one-hot, deterministic, boost_from_average off
        o.cat_features = [0, 1]
        o.boost_from_average = 0
    elif unit == 10:  # Logloss + CTR, score noise alone at strength 2
        o.cat_features = [0]
        o.random_strength = 2.0
    else:
        raise Error("gbdt_host_device_cat: no unit " + String(unit))
    return o^


def _is_rmse(unit: Int) -> Bool:
    return unit == 1 or unit == 3 or unit == 4 or unit == 5 or unit == 8 or unit == 9


def _first_difference(device: String, host: String) -> String:
    var a = device.as_bytes()
    var b = host.as_bytes()
    var n = len(a) if len(a) < len(b) else len(b)
    var at = n
    for i in range(n):  # driver-loop: over the two texts' bytes (a failure report)
        if a[i] != b[i]:
            at = i
            break
    var lo = at
    while lo > 0 and a[lo - 1] != UInt8(10):
        lo -= 1
    var hi_a = lo
    while hi_a < len(a) and a[hi_a] != UInt8(10) and hi_a - lo < 200:
        hi_a += 1
    var hi_b = lo
    while hi_b < len(b) and b[hi_b] != UInt8(10) and hi_b - lo < 200:
        hi_b += 1
    var dl = String(StringSlice(unsafe_from_utf8=a[lo:hi_a]))
    var hl = String(StringSlice(unsafe_from_utf8=b[lo:hi_b]))
    return ("host " + String(len(b)) + " bytes, device " + String(len(a)) + "; first difference at byte "
            + String(at) + "\n  device: " + dl + "\n  host:   " + hl)


def test_host_text_is_device_text() raises:
    """Every host fit across the cores first, then one device graph at a time against its text."""
    var host_texts = List[String](length=CAT_UNITS, fill=String(""))
    var texts_addr = Int(host_texts.unsafe_ptr())

    def host_fit(unit: Int) raises {imm texts_addr}:
        var p = UnsafePointer[String, MutUntrackedOrigin](unsafe_from_address=texts_addr)
        var outcome = fit[_HostFit](_opts(unit), _inputs(_is_rmse(unit)))
        p[unit] = outcome.text.copy()

    var host_error = String("")
    try:
        check_cases("gbdt_host_device_cat host", CAT_UNITS, host_fit)
    except error:
        host_error = String(error)
    var failed = String("")
    var count = 0
    for unit in range(CAT_UNITS):  # one device graph at a time (see the docstring)
        if host_texts[unit].byte_length() == 0:
            continue  # the host fit failed and is recorded
        try:
            var outcome = fit[_DeviceFit](_opts(unit), _inputs(_is_rmse(unit)))
            if outcome.text != host_texts[unit]:
                raise Error(
                    "gbdt_host_device_cat unit " + String(unit) + ": the host model text is not the device's ("
                    + _first_difference(outcome.text, host_texts[unit]) + ")"
                )
        except error:
            count += 1
            failed += "\n      gbdt_host_device_cat device row " + String(unit) + ": " + String(error)
    _ = host_texts^
    if count > 0 or host_error.byte_length() > 0:
        raise Error(host_error + ("\n" if host_error.byte_length() > 0 and count > 0 else "")
                    + ("gbdt_host_device_cat device: " + String(count) + " of " + String(CAT_UNITS) + " cases failed" + failed
                       if count > 0 else ""))


comptime RANK_ROWS = 1200
comptime RANK_TREES = 10
#: NOT a power of two (see the module docstring)
comptime RANK_RATE = 0.1
comptime RANK_UNITS = 17
#: the query sizes, cycled in row order (tools/identity_break.py `RANK_QUERY_SIZES`)
comptime RANK_SIZE_CYCLE = 10


def _rank_size(q: Int) -> Int:
    var cycle: List[Int] = [1, 2, 7, 3, 16, 1, 5, 40, 2, 9]
    return cycle[q % RANK_SIZE_CYCLE]


def _rank_inputs(unit: Int) -> GbdtInputs:
    """Column-major X of four numeric columns, graded relevance 0..4 from a
    hashed target, uneven query groups; the explicit-pair units add every two
    rows of a query with different grades whose index sum is even (higher
    grade wins) with hashed weights in [0.5, 2)."""
    var n = RANK_ROWS
    var x = List[Float32](capacity=4 * n)
    for j in range(4):  # driver-loop: over the fixture's columns (fixture construction, once per unit)
        for r in range(n):  # driver-loop: over the fixture's rows
            x.append(_unit01(r, 10 + j) * Float32(4.0) - Float32(2.0))
    var y = List[Float32](capacity=n)
    for r in range(n):  # driver-loop: over the fixture's rows
        var v = (x[r] - Float32(0.5) * x[n + r] + Float32(0.25) * x[2 * n + r]
                 + Float32(0.8) * (_unit01(r, 20) - Float32(0.5)))
        var grade = Float32(0.0)
        if v > Float32(-1.0):
            grade = Float32(1.0)
        if v > Float32(0.0):
            grade = Float32(2.0)
        if v > Float32(0.8):
            grade = Float32(3.0)
        if v > Float32(1.6):
            grade = Float32(4.0)
        y.append(grade)
    var sizes = List[UInt32]()
    var at = 0
    var q = 0
    while at < n:  # driver-loop: over the fixture's queries
        var size = min(_rank_size(q), n - at)
        sizes.append(UInt32(size))
        at += size
        q += 1
    var inp = GbdtInputs(x^, n, 4, y^)
    if _rank_explicit(unit):
        var begin = 0
        for g in range(len(sizes)):  # driver-loop: over the fixture's queries
            var end = begin + Int(sizes[g])
            for a in range(begin, end):  # driver-loop: over one query's rows
                for b in range(a + 1, end):  # driver-loop: over one query's rows
                    if inp.y[a] != inp.y[b] and (a + b) % 2 == 0:
                        var a_wins = inp.y[a] > inp.y[b]
                        inp.pairs.append(UInt32(a if a_wins else b))
                        inp.pairs.append(UInt32(b if a_wins else a))
                        inp.pair_weights.append(Float32(0.5) + Float32(1.5) * _unit01(len(inp.pair_weights), 30))
            begin = end
    inp.group_sizes = sizes^
    return inp^


def _rank_explicit(unit: Int) -> Bool:
    return unit >= 15


def _rank_opts(unit: Int) raises -> GbdtOpts:
    """Units 0-14: loss `unit % 3` (QueryRMSE, PairLogit, YetiRank) under
    arm `unit // 3`: greedy deterministic, greedy Bayesian (temperature 1)
    with score noise 1, greedy Bernoulli 0.7, pointwise Cosine, pointwise
    NewtonL2. Units 15-16: PairLogit on explicit weighted pairs, greedy and
    pointwise Cosine. Every option explicit."""
    var loss = String("PairLogit")
    var arm = 0
    if unit < 15:
        var k = unit % 3
        loss = String("QueryRMSE") if k == 0 else (String("PairLogit") if k == 1 else String("YetiRank"))
        arm = unit // 3
    elif unit == 16:
        arm = 3
    elif unit != 15:
        raise Error("gbdt_host_device_cat rank: no unit " + String(unit))
    var o = GbdtOpts(loss, RANK_TREES, 4)
    o.learning_rate = RANK_RATE
    o.l2_leaf_reg = 3.0
    o.leaf_iterations = 1
    o.random_state = 7
    o.bootstrap = "No"
    o.random_strength = 0.0
    if arm == 1:
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 1.0
        o.random_strength = 1.0
    elif arm == 2:
        o.bootstrap = "Bernoulli"
        o.subsample = 0.7
    elif arm == 3:
        o.pointwise = True
        o.score_function = "Cosine"
    elif arm == 4:
        o.pointwise = True
        o.score_function = "NewtonL2"
    return o^


def test_rank_host_text_is_device_text() raises:
    """The ranking units: every host fit across the cores first, then one device graph at a time."""
    var host_texts = List[String](length=RANK_UNITS, fill=String(""))
    var texts_addr = Int(host_texts.unsafe_ptr())

    def host_fit(unit: Int) raises {imm texts_addr}:
        var p = UnsafePointer[String, MutUntrackedOrigin](unsafe_from_address=texts_addr)
        var outcome = fit[_HostFit](_rank_opts(unit), _rank_inputs(unit))
        p[unit] = outcome.text.copy()

    var host_error = String("")
    try:
        check_cases("gbdt_host_device_cat rank host", RANK_UNITS, host_fit)
    except error:
        host_error = String(error)
    var failed = String("")
    var count = 0
    for unit in range(RANK_UNITS):  # one device graph at a time (see the docstring)
        if host_texts[unit].byte_length() == 0:
            continue  # the host fit failed and is recorded
        try:
            var outcome = fit[_DeviceFit](_rank_opts(unit), _rank_inputs(unit))
            if outcome.text != host_texts[unit]:
                raise Error(
                    "gbdt_host_device_cat rank unit " + String(unit) + ": the host model text is not the device's ("
                    + _first_difference(outcome.text, host_texts[unit]) + ")"
                )
        except error:
            count += 1
            failed += "\n      gbdt_host_device_cat rank device row " + String(unit) + ": " + String(error)
    _ = host_texts^
    if count > 0 or host_error.byte_length() > 0:
        raise Error(host_error + ("\n" if host_error.byte_length() > 0 and count > 0 else "")
                    + ("gbdt_host_device_cat rank device: " + String(count) + " of " + String(RANK_UNITS)
                       + " cases failed" + failed if count > 0 else ""))


comptime GAPS_ROWS = 1500
comptime GAPS_TREES = 10
#: NOT a power of two (see the module docstring)
comptime GAPS_RATE = 0.1
comptime GAPS_UNITS = 14


def _gaps_uses_eval(unit: Int) -> Bool:
    return unit >= 11


def _gaps_inputs(unit: Int) -> GbdtInputs:
    """Units 0-8 and 11-13: four numeric columns, a regression target
    (positive for Poisson and Tweedie, 0/1 for the Logloss units); units
    9-10 the ranking fixture. Every unit but the eval units carries hashed
    sample weights in [0.25, 4); the eval units carry a held-out slice of
    the same rule with hashed target noise (every fifth label flipped for
    Logloss)."""
    if unit == 9 or unit == 10:
        var r = _rank_inputs(0)
        for i in range(r.n):  # driver-loop: over the fixture's rows (fixture construction)
            r.weights.append(Float32(0.25) + Float32(3.75) * _unit01(i, 60))
        return r^
    var logloss = _gaps_logloss(unit)
    var positive = unit == 3 or unit == 4
    var n = GAPS_ROWS
    var x = List[Float32](capacity=4 * n)
    for j in range(4):  # driver-loop: over the fixture's columns (fixture construction, once per unit)
        for r in range(n):  # driver-loop: over the fixture's rows
            x.append(_unit01(r, 40 + j) * Float32(4.0) - Float32(2.0))
    var y = List[Float32](capacity=n)
    for r in range(n):  # driver-loop: over the fixture's rows
        var v = (x[r] - Float32(0.5) * x[n + r] + Float32(0.25) * x[2 * n + r] * x[3 * n + r]
                 + Float32(0.4) * (_unit01(r, 50) - Float32(0.5)))
        if logloss:
            y.append(Float32(1.0) if v > Float32(0.0) else Float32(0.0))
        elif positive:
            y.append(abs(v) + Float32(1.0))
        else:
            y.append(v)
    var inp = GbdtInputs(x^, n, 4, y^)
    if _gaps_uses_eval(unit):
        var ne = n // 2
        for j in range(4):  # driver-loop: over the held-out columns
            for r in range(ne):  # driver-loop: over the held-out rows
                inp.eval_x.append(_unit01(r, 70 + j) * Float32(4.0) - Float32(2.0))
        for r in range(ne):  # driver-loop: over the held-out rows
            var v = (inp.eval_x[r] - Float32(0.5) * inp.eval_x[ne + r]
                     + Float32(0.25) * inp.eval_x[2 * ne + r] * inp.eval_x[3 * ne + r])
            if logloss:
                var c = Float32(1.0) if v > Float32(0.0) else Float32(0.0)
                inp.eval_y.append(Float32(1.0) - c if r % 5 == 0 else c)
            else:
                inp.eval_y.append(v + Float32(6.0) * (_unit01(r, 80) - Float32(0.5)))
    else:
        for r in range(n):  # driver-loop: over the fixture's rows
            inp.weights.append(Float32(0.25) + Float32(3.75) * _unit01(r, 60))
    return inp^


def _gaps_logloss(unit: Int) -> Bool:
    return unit == 6 or unit == 8 or unit == 12


def _gaps_opts(unit: Int) raises -> GbdtOpts:
    """Lane gbdt-gaps (2026-11-23): what the CPU column refused until then.
    Sample weights: 0 RMSE deterministic, 1 RMSE Bayesian + noise +
    boost_from_average, 2 RMSE Bernoulli, 3 Poisson, 4 Tweedie (1.5), 5
    Depthwise RMSE, 6 Depthwise Logloss Newton x10 under Bernoulli, 7 Lossguide RMSE
    NewtonL2 Bernoulli, 8 Lossguide Logloss NewtonCosine with Gradient
    leaves x2 under the Bayesian bootstrap, 9 YetiRank, 10 PairLogit on
    generated pairs. Eval sets with the Iter detector and use_best_model at rate 1.0:
    11 symmetric RMSE, 12 Depthwise Logloss, 13 Lossguide RMSE."""
    var loss = String("RMSE")
    if unit == 3:
        loss = String("Poisson")
    elif unit == 4:
        loss = String("Tweedie")
    elif _gaps_logloss(unit):
        loss = String("Logloss")
    elif unit == 9:
        loss = String("YetiRank")
    elif unit == 10:
        loss = String("PairLogit")
    elif unit < 0 or unit >= GAPS_UNITS:
        raise Error("gbdt_host_device_cat gaps: no unit " + String(unit))
    var o = GbdtOpts(loss, GAPS_TREES, 4)
    o.learning_rate = GAPS_RATE
    o.l2_leaf_reg = 3.0
    o.leaf_method = "Newton"
    o.leaf_iterations = 1
    o.random_state = 7
    o.bootstrap = "No"
    o.random_strength = 0.0
    if loss == String("RMSE"):
        o.boost_from_average = 0
    if unit == 1:
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 1.0
        o.random_strength = 1.0
        o.boost_from_average = 1
    elif unit == 2:
        o.bootstrap = "Bernoulli"
        o.subsample = 0.7
    elif unit == 4:
        o.loss_variance_power = 1.5
    elif unit == 5:
        o.grow_policy = "Depthwise"
        o.score_function = "Cosine"
        o.boost_from_average = 1
    elif unit == 6:
        o.grow_policy = "Depthwise"
        o.score_function = "Cosine"
        o.leaf_iterations = 10
        o.bootstrap = "Bernoulli"
        o.subsample = 0.7
    elif unit == 7:
        o.grow_policy = "Lossguide"
        o.score_function = "NewtonL2"
        o.max_leaves = 16
        o.min_data_in_leaf = 4
        o.bootstrap = "Bernoulli"
        o.subsample = 0.7
    elif unit == 8:
        o.grow_policy = "Lossguide"
        o.score_function = "NewtonCosine"
        o.max_leaves = 16
        o.min_data_in_leaf = 4
        o.leaf_method = "Gradient"
        o.leaf_iterations = 2
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 1.0
    if _gaps_uses_eval(unit):
        o.n_estimators = 30
        o.max_depth = 6
        o.learning_rate = 1.0
        o.od_type = "Iter"
        o.od_wait = 2
        o.use_best_model = 1
        o.best_model_min_trees = 3
        if unit == 12:
            o.grow_policy = "Depthwise"
            o.score_function = "Cosine"
        elif unit == 13:
            o.grow_policy = "Lossguide"
            o.score_function = "NewtonL2"
            o.max_leaves = 16
            o.min_data_in_leaf = 4
    return o^


def test_gaps_host_text_is_device_text() raises:
    """The weighted and held-out units (lane gbdt-gaps): every host fit across the cores first, then one device
    graph at a time."""
    var host_texts = List[String](length=GAPS_UNITS, fill=String(""))
    var texts_addr = Int(host_texts.unsafe_ptr())

    def host_fit(unit: Int) raises {imm texts_addr}:
        var p = UnsafePointer[String, MutUntrackedOrigin](unsafe_from_address=texts_addr)
        var outcome = fit[_HostFit](_gaps_opts(unit), _gaps_inputs(unit))
        p[unit] = outcome.text.copy()

    var host_error = String("")
    try:
        check_cases("gbdt_host_device_cat gaps host", GAPS_UNITS, host_fit)
    except error:
        host_error = String(error)
    var failed = String("")
    var count = 0
    for unit in range(GAPS_UNITS):  # one device graph at a time (see the docstring)
        if host_texts[unit].byte_length() == 0:
            continue  # the host fit failed and is recorded
        try:
            var outcome = fit[_DeviceFit](_gaps_opts(unit), _gaps_inputs(unit))
            if outcome.text != host_texts[unit]:
                raise Error(
                    "gbdt_host_device_cat gaps unit " + String(unit) + ": the host model text is not the device's ("
                    + _first_difference(outcome.text, host_texts[unit]) + ")"
                )
        except error:
            count += 1
            failed += "\n      gbdt_host_device_cat gaps device row " + String(unit) + ": " + String(error)
    _ = host_texts^
    if count > 0 or host_error.byte_length() > 0:
        raise Error(host_error + ("\n" if host_error.byte_length() > 0 and count > 0 else "")
                    + ("gbdt_host_device_cat gaps device: " + String(count) + " of " + String(GAPS_UNITS)
                       + " cases failed" + failed if count > 0 else ""))


comptime REST_ROWS = 1500
comptime REST_TREES = 10
#: NOT a power of two (see the module docstring)
comptime REST_RATE = 0.1
comptime REST_UNITS = 18


def _rest_rank(unit: Int) -> Bool:
    return unit <= 2


def _rest_cat(unit: Int) -> Bool:
    return unit == 7 or unit == 8


def _rest_weighted(unit: Int) -> Bool:
    return unit <= 8 or unit == 12 or unit == 16


def _rest_loss(unit: Int) raises -> String:
    if unit == 0 or unit == 2:
        return String("QueryRMSE")
    if unit == 1:
        return String("YetiRank")
    if unit == 3 or unit == 13:
        return String("MAE")
    if unit == 4:
        return String("MAPE")
    if unit == 5:
        return String("MultiClass")
    if unit == 7 or unit == 11 or unit == 16 or unit == 17:
        return String("Logloss")
    if unit == 12:
        return String("Quantile")
    if unit < 0 or unit >= REST_UNITS:
        raise Error("gbdt_host_device_cat rest: no unit " + String(unit))
    return String("RMSE")


def _rest_inputs(unit: Int) raises -> GbdtInputs:
    """Units 0-2 the ranking fixture; 7-8 the categorical fixture of
    `_inputs` (column 0 a CTR column of 40 categories, column 1 a one-hot
    column); every other unit four numeric columns with a target of its
    loss (0/1 for Logloss, three classes for MultiClass, positive for
    MAPE). Units 9-10 put a NaN in every seventh row of column 1; unit 13
    carries a held-out slice. The weighted units carry hashed sample
    weights in [0.25, 4)."""
    var inp: GbdtInputs
    if _rest_rank(unit):
        inp = _rank_inputs(0)
    elif _rest_cat(unit):
        inp = _inputs(unit == 8)
    else:
        var loss = _rest_loss(unit)
        var n = REST_ROWS
        var x = List[Float32](capacity=4 * n)
        for j in range(4):  # driver-loop: over the fixture's columns (fixture construction, once per unit)
            for r in range(n):  # driver-loop: over the fixture's rows
                x.append(_unit01(r, 90 + j) * Float32(4.0) - Float32(2.0))
        var y = List[Float32](capacity=n)
        for r in range(n):  # driver-loop: over the fixture's rows
            var v = (x[r] - Float32(0.5) * x[n + r] + Float32(0.25) * x[2 * n + r] * x[3 * n + r]
                     + Float32(0.4) * (_unit01(r, 95) - Float32(0.5)))
            if loss == String("Logloss"):
                y.append(Float32(1.0) if v > Float32(0.0) else Float32(0.0))
            elif loss == String("MultiClass"):
                y.append(Float32(0.0) if v < Float32(-0.5) else (Float32(1.0) if v < Float32(0.5) else Float32(2.0)))
            elif loss == String("MAPE"):
                y.append(abs(v) + Float32(1.0))
            else:
                y.append(v)
        if unit == 9 or unit == 10:
            for r in range(0, n, 7):  # driver-loop: over every seventh row (fixture construction)
                x[n + r] = nan[DType.float32]()
        inp = GbdtInputs(x^, n, 4, y^)
        if unit == 13:
            var ne = n // 2
            for j in range(4):  # driver-loop: over the held-out columns
                for r in range(ne):  # driver-loop: over the held-out rows
                    inp.eval_x.append(_unit01(r, 100 + j) * Float32(4.0) - Float32(2.0))
            for r in range(ne):  # driver-loop: over the held-out rows
                inp.eval_y.append(inp.eval_x[r] - Float32(0.5) * inp.eval_x[ne + r]
                                  + Float32(3.0) * (_unit01(r, 105) - Float32(0.5)))
    if _rest_weighted(unit):
        for r in range(inp.n):  # driver-loop: over the fixture's rows
            inp.weights.append(Float32(0.25) + Float32(3.75) * _unit01(r, 110))
    return inp^


def _rest_opts(unit: Int) raises -> GbdtOpts:
    """Lane gbdt-cpu-rest (2026-11-23), weighted: 0 QueryRMSE greedy, 1
    YetiRank pointwise Cosine, 2 QueryRMSE pointwise NewtonL2, 3 MAE Exact,
    4 MAPE Exact, 5 MultiClass with class weights, 6 RMSE Bayesian + noise,
    7 Logloss one-hot + CTR, 8 RMSE CTR + one-hot Bernoulli; NaN: 9 RMSE
    symmetric (nan_mode Min), 10 Depthwise RMSE (Max); 11 Lossguide Logloss
    Cosine; 12 Depthwise Quantile (weighted, boost_from_average unset); 13
    Lossguide MAE Exact with an eval set (Iter, use_best_model); feature
    fraction 0.5: 14 symmetric RMSE, 15 Depthwise RMSE, 16 symmetric Logloss
    (weighted, Bernoulli); 17 the pointwise searcher, Logloss L2 Bayesian,
    no sample weight and no eval set."""
    var loss = _rest_loss(unit)
    var o = GbdtOpts(loss, REST_TREES, 4)
    o.learning_rate = REST_RATE
    o.l2_leaf_reg = 3.0
    o.random_state = 7
    o.bootstrap = "No"
    o.random_strength = 0.0
    if loss == String("RMSE"):
        o.boost_from_average = 0
    if unit == 1 or unit == 2:
        o.pointwise = True
        o.score_function = "Cosine" if unit == 1 else "NewtonL2"
    elif unit == 3 or unit == 4:
        o.leaf_method = "Exact"
    elif unit == 5:
        o.class_weights = [1.0, 2.0, 0.5]
    elif unit == 6:
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 1.0
        o.random_strength = 1.0
    elif unit == 7:
        o.cat_features = [0, 1]
    elif unit == 8:
        o.cat_features = [0, 1]
        o.bootstrap = "Bernoulli"
        o.subsample = 0.7
    elif unit == 9:
        o.nan_mode = "Min"
    elif unit == 10:
        o.nan_mode = "Max"
        o.grow_policy = "Depthwise"
        o.score_function = "Cosine"
    elif unit == 11:
        o.grow_policy = "Lossguide"
        o.score_function = "Cosine"
        o.max_leaves = 16
        o.min_data_in_leaf = 4
    elif unit == 12:
        o.grow_policy = "Depthwise"
        o.score_function = "Cosine"
    elif unit == 13:
        o.grow_policy = "Lossguide"
        o.score_function = "NewtonL2"
        o.max_leaves = 16
        o.min_data_in_leaf = 4
        o.leaf_method = "Exact"
        o.n_estimators = 30
        o.learning_rate = 1.0
        o.od_type = "Iter"
        o.od_wait = 2
        o.use_best_model = 1
        o.best_model_min_trees = 3
    elif unit == 14 or unit == 15 or unit == 16:
        o.feature_fraction = 0.5
        if unit == 15:
            o.grow_policy = "Depthwise"
            o.score_function = "Cosine"
        if unit == 16:
            o.bootstrap = "Bernoulli"
            o.subsample = 0.7
    elif unit == 17:
        o.pointwise = True
        o.score_function = "L2"
        o.bootstrap = "Bayesian"
        o.bagging_temperature = 0.5
        o.boost_from_average = 1
    return o^


def test_rest_host_text_is_device_text() raises:
    """The units of lane gbdt-cpu-rest: every host fit across the cores
    first, then one device graph at a time."""
    var host_texts = List[String](length=REST_UNITS, fill=String(""))
    var texts_addr = Int(host_texts.unsafe_ptr())

    def host_fit(unit: Int) raises {imm texts_addr}:
        var p = UnsafePointer[String, MutUntrackedOrigin](unsafe_from_address=texts_addr)
        var outcome = fit[_HostFit](_rest_opts(unit), _rest_inputs(unit))
        p[unit] = outcome.text.copy()

    var host_error = String("")
    try:
        check_cases("gbdt_host_device_cat rest host", REST_UNITS, host_fit)
    except error:
        host_error = String(error)
    var failed = String("")
    var count = 0
    for unit in range(REST_UNITS):  # one device graph at a time (see the docstring)
        if host_texts[unit].byte_length() == 0:
            continue  # the host fit failed and is recorded
        try:
            var outcome = fit[_DeviceFit](_rest_opts(unit), _rest_inputs(unit))
            if outcome.text != host_texts[unit]:
                raise Error(
                    "gbdt_host_device_cat rest unit " + String(unit) + ": the host model text is not the device's ("
                    + _first_difference(outcome.text, host_texts[unit]) + ")"
                )
        except error:
            count += 1
            failed += "\n      gbdt_host_device_cat rest device row " + String(unit) + ": " + String(error)
    _ = host_texts^
    if count > 0 or host_error.byte_length() > 0:
        raise Error(host_error + ("\n" if host_error.byte_length() > 0 and count > 0 else "")
                    + ("gbdt_host_device_cat rest device: " + String(count) + " of " + String(REST_UNITS)
                       + " cases failed" + failed if count > 0 else ""))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
