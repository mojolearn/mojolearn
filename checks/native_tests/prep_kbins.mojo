# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""KBinsDiscretizer programs against scikit-learn's values (lane py2mojo-B, 2026-11-22; ported from
python/mojolearn/tests/test_x_prep_kbins.py `test_strategies` and `test_onehot`, and test_x_prep_parity.py
`test_kbins_quantile_methods` and `test_kbins_sample_weight`, whose references scikit-learn computed on the float32
X with the float32 weights; with weights the door's `_check_weights` program counts the positive weights first).

The inputs are the original tests' NumPy draws (500 x 4 float32: a tie column, a constant column) and the references
are the scikit-learn 1.9.1 calls they made on the float64 copies, stored as words in `fixtures/prep_kb/` by
`tools/py2mojo_native_fixtures_b2.py`. The programs are the typed builders the door builds through
(bindings/prep_encode_kbins_run.mojo `kbins_fit_stage`, `kbins_transform_stage`, `kbins_inverse_stage`) with the
requests bindings/prep_encode_native_door.mojo `_kbins_fit`, `_kbins_transform` and `_kbins_inverse` make (no
subsample at 500 rows, no weights, the strategy words of `_kbins_strategy`; bin_edges_[j] the first `_ne[j]` edge
words of column j, n_bins_ one less), run on the host column by `prep_enc_host.enc_host_run` with the identical
image's blocked column statistics and without them.

    test_strategies  n_bins 5, ordinal: uniform, quantile (averaged_inverted_cdf, 1.9's default), quantile linear,
                     kmeans (not the tie column, as the original): bin_edges_ rtol = atol = 1e-4, n_bins_ exact,
                     transform more than 99.5% equal, inverse_transform of the reference's codes rtol = atol = 1e-4
    test_onehot      n_bins 4, onehot-dense, uniform: the same shape and more than 99.5% equal, inverse_transform
                     of the reference's block NaN where it is NaN and rtol = atol = 1e-5

Stays in Python: the encode / strategy admission, the finite-input refusal and the array shapes (the public
boundary). Every case is an independent unit; the case tables run across host cores.
"""
from std.math import isnan
from std.memory import bitcast
from bindings.prep_encode_prog_run import EncProgram
from bindings.prep_encode_kbins_run import (
    KBinsFit, KBinsApply, KBinsInverse, kbins_fit_stage, kbins_transform_stage, kbins_inverse_stage, KB_DRAW_NONE,
    enc_weights_stage,
)
from checks.native_tests.prep_enc_host import PREP_KB_DIR, enc_host_run
from checks.native_tests.prep_cls_host import all_close, arena_words
from checks.native_tests.py2mojo_fixture_words import fixture_f64
from x_prep.blocked import IDN_STATS_BLOCKED
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require, fixture_f32, fixture_i32

comptime KB_N = 500
comptime KB_D = 4


def to_f32(values: List[Float64]) -> List[Float32]:
    var out = List[Float32](length=len(values), fill=0)
    for i in range(len(values)):  # small-loop(fixture: reference codes as the float32 words the door admits)
        out[i] = Float32(values[i])
    return out^


@fieldwise_init
struct KBinsModel(Movable):
    var edges: List[Float32]   # `_edges`, d x stride
    var counts: List[Float32]  # `_ne`
    var stride: Int
    var n_bins: List[Int]


def positive_weights(w: List[Float32], n: Int, blocked: Bool) raises -> Int:
    """`_check_weights`' program: the count and minimum of the weights and their count of positive entries."""
    var prog = EncProgram(False)
    var plan = enc_weights_stage(prog, 0, n, blocked and IDN_STATS_BLOCKED)
    var owners = List[List[Float32]]()
    owners.append(w.copy())
    var arena = enc_host_run(prog, owners)
    require(Int(arena[plan.stats]) == n and arena[plan.stats + 3] >= 0, "sample_weight must be nonnegative")
    var m = Int(bitcast[DType.int32](arena[plan.positives]))
    require(m > 0, "sample_weight is all zero")
    return m


def kbins_fit(x: List[Float32], n: Int, d: Int, strategy: Int, bins: Int, blocked: Bool,
              w: List[Float32]) raises -> KBinsModel:
    var prog = EncProgram(False)
    var nb = List[Int](length=d, fill=bins)
    var weighted = len(w) > 0
    var positive = positive_weights(w, n, blocked) if weighted else 0
    var plan = kbins_fit_stage(prog, KBinsFit(n, d, 0, strategy, nb^, bins, KB_DRAW_NONE, 0, 0, 0, weighted,
        1 if weighted else -1, positive, blocked and IDN_STATS_BLOCKED))
    var owners = List[List[Float32]]()
    owners.append(x.copy())
    if weighted:
        owners.append(w.copy())
    var arena = enc_host_run(prog, owners)
    var counts = arena_words(arena, plan.counts, d)
    var n_bins = List[Int](length=d, fill=0)
    for j in range(d):  # small-loop(oracle: the per-column edge counts)
        n_bins[j] = Int(counts[j]) - 1
    return KBinsModel(arena_words(arena, plan.edges, d * (bins + 1)), counts^, bins + 1, n_bins^)


def kbins_transform(m: KBinsModel, x: List[Float32], n: Int, d: Int, onehot: Bool) raises -> List[Float32]:
    var prog = EncProgram(False)
    var cols = d
    if onehot:
        cols = 0
        for j in range(d):  # small-loop(params: the fitted bin counts)
            cols += m.n_bins[j]
    var out = kbins_transform_stage(prog, KBinsApply(n, d, 0, 1, len(m.edges), 2, len(m.counts), m.stride, onehot,
        m.n_bins.copy()), False)
    var owners = List[List[Float32]]()
    owners.append(x.copy())
    owners.append(m.edges.copy())
    owners.append(m.counts.copy())
    var arena = enc_host_run(prog, owners)
    return arena_words(arena, out, n * cols)


def kbins_inverse(m: KBinsModel, z: List[Float32], n: Int, cols: Int, ordinal: Bool) raises -> List[Float32]:
    var prog = EncProgram(False)
    var d = len(m.n_bins)
    var plan = kbins_inverse_stage(prog, KBinsInverse(n, cols, 0, ordinal, m.n_bins.copy(), 1, len(m.edges),
        m.stride), False)
    var owners = List[List[Float32]]()
    owners.append(z.copy())
    owners.append(m.edges.copy())
    var arena = enc_host_run(prog, owners)
    require(arena[plan.bad_count] == 0, "inverse_transform: " + String(arena[plan.bad_count]) + " rows name no bin")
    return arena_words(arena, plan.centers, n * d)


def check_edges(m: KBinsModel, tag: String, d: Int, cols: List[Int], tol: Float64, label: String) raises:
    var want_bins = fixture_i32(PREP_KB_DIR + tag + "-nbins.i32", d)
    var total = 0
    for j in range(d):  # small-loop(oracle: the reference's edge counts)
        total += Int(want_bins[j]) + 1
    var want = fixture_f64(PREP_KB_DIR + tag + "-edges.f64", total)
    var at = 0
    var c = 0
    for j in range(d):  # small-loop(oracle: four columns)
        var count = Int(want_bins[j]) + 1
        if c < len(cols) and cols[c] == j:
            c += 1
            require(m.n_bins[j] == Int(want_bins[j]), label + ": n_bins_[" + String(j) + "] " + String(m.n_bins[j])
                + " expected " + String(want_bins[j]))
            for e in range(count):  # small-loop(oracle: one column's edges)
                all_close(m.edges[j * m.stride + e], want[at + e], tol, tol,
                    label + ": bin_edges_[" + String(j) + "][" + String(e) + "]")
        at += count


# ---- test_strategies: uniform, quantile, quantile linear, kmeans x the two routes ----
def strategy_unit(unit: Int) raises:
    var k = unit // 2
    var blocked = unit % 2 == 0
    var n = KB_N
    var d = KB_D
    var tag = "s" + String(k)
    var strategies: List[Int] = [0, 1, 2, 3]   # `_kbins_strategy`: uniform, averaged_inverted_cdf, linear, kmeans
    var label = "kbins " + tag + " blocked " + String(blocked)
    var cols: List[Int] = [0, 1, 2, 3]
    if k == 3:
        cols = [0, 2, 3]
    var x = fixture_f32(PREP_KB_DIR + "x0.f32", n * d)
    var m = kbins_fit(x, n, d, strategies[k], 5, blocked, List[Float32]())
    check_edges(m, tag, d, cols, 1e-4, label)
    var got = kbins_transform(m, x, n, d, False)
    var z = fixture_f64(PREP_KB_DIR + tag + "-transform.f64", n * d)
    var same = 0
    for i in range(n):  # small-loop(oracle: the codes of the compared columns)
        for c in range(len(cols)):  # small-loop(oracle: at most four columns)
            if Float64(got[i * d + cols[c]]) == z[i * d + cols[c]]:
                same += 1
    var share = Float64(same) / Float64(n * len(cols))
    require(share > 0.995, label + ": transform agrees with scikit-learn on " + String(share) + " of the codes")
    var a = kbins_inverse(m, to_f32(z), n, d, True)
    var b = fixture_f64(PREP_KB_DIR + tag + "-inverse.f64", n * d)
    for i in range(n):  # small-loop(oracle: the bin centres of the compared columns)
        for c in range(len(cols)):  # small-loop(oracle: at most four columns)
            var at = i * d + cols[c]
            all_close(a[at], b[at], 1e-4, 1e-4, label + ": inverse_transform row " + String(i) + " column "
                + String(cols[c]))


def test_strategies() raises:
    check_cases("kbins_strategies", 8, strategy_unit)


# ---- test_onehot: n_bins 4, onehot-dense, uniform x the two routes ----
def onehot_unit(unit: Int) raises:
    var blocked = unit == 0
    var n = KB_N
    var d = KB_D
    var label = "kbins onehot blocked " + String(blocked)
    var x = fixture_f32(PREP_KB_DIR + "x1.f32", n * d)
    var m = kbins_fit(x, n, d, 0, 4, blocked, List[Float32]())
    var shape = fixture_i32(PREP_KB_DIR + "oh-shape.i32", 2)
    var cols = Int(shape[1])
    var total = 0
    for j in range(d):  # small-loop(oracle: the fitted bin counts)
        total += m.n_bins[j]
    require(Int(shape[0]) == n and total == cols, label + ": " + String(total) + " columns, scikit-learn "
        + String(cols))
    var got = kbins_transform(m, x, n, d, True)
    var want = fixture_f64(PREP_KB_DIR + "oh-transform.f64", n * cols)
    var same = 0
    for i in range(n * cols):  # small-loop(oracle: the indicator words)
        if Float64(got[i]) == want[i]:
            same += 1
    var share = Float64(same) / Float64(n * cols)
    require(share > 0.995, label + ": transform agrees with scikit-learn on " + String(share) + " of the words")
    var a = kbins_inverse(m, to_f32(want), n, cols, False)
    var b = fixture_f64(PREP_KB_DIR + "oh-inverse.f64", n * d)
    for i in range(n * d):  # small-loop(oracle: the bin centres; all_close holds NaN only against NaN)
        all_close(a[i], b[i], 1e-5, 1e-5, label + ": inverse_transform word " + String(i))


def test_onehot() raises:
    check_cases("kbins_onehot", 2, onehot_unit)


# ---- test_x_prep_parity.py::test_kbins_quantile_methods: 299 x 3 quarter-integers, the nine quantile methods x n_bins
# 3, 5, 7, x the two routes: every column's bin_edges_ rtol = atol = 1e-6 ----
def quantile_method_unit(unit: Int) raises:
    var blocked = unit % 2 == 0
    var row = unit // 2
    var i = row // 3
    var nbs: List[Int] = [3, 5, 7]
    var nb = nbs[row % 3]
    # `_kbins_strategy`: inverted_cdf 4, closest_observation 5, interpolated_inverted_cdf 6, hazen 7, weibull 8,
    # median_unbiased 9, normal_unbiased 10, linear 2, averaged_inverted_cdf 1
    var words: List[Int] = [4, 5, 6, 7, 8, 9, 10, 2, 1]
    var n = 299
    var d = 3
    var tag = "qm" + String(i) + "-" + String(nb)
    var x = fixture_f32(PREP_KB_DIR + "qm-x.f32", n * d)
    var m = kbins_fit(x, n, d, words[i], nb, blocked, List[Float32]())
    var cols: List[Int] = [0, 1, 2]
    check_edges(m, tag, d, cols, 1e-6, "kbins " + tag + " blocked " + String(blocked))


def test_kbins_quantile_methods() raises:
    check_cases("kbins_quantile_methods", 54, quantile_method_unit)


# ---- test_x_prep_parity.py::test_kbins_sample_weight: 301 x 3, three float32 weight vectors (integers with zeros,
# uniform, ones) x quantile (averaged_inverted_cdf), quantile inverted_cdf, uniform, kmeans x n_bins 3, 5, x the two
# routes: bin_edges_ rtol = atol = 1e-6 (kmeans 1e-4). The refusal of the other quantile methods with weights stays
# Python (admission). ----
def sample_weight_unit(unit: Int) raises:
    var blocked = unit % 2 == 0
    var row = unit // 2
    var s = row // 8
    var j = (row % 8) // 2
    var nb = 3 if row % 2 == 0 else 5
    var words: List[Int] = [1, 4, 0, 3]
    var n = 301
    var d = 3
    var tag = "sw" + String(s) + "-" + String(j) + "-" + String(nb)
    var x = fixture_f32(PREP_KB_DIR + "sw-x.f32", n * d)
    var w = fixture_f32(PREP_KB_DIR + "sw" + String(s) + "-w.f32", n)
    var m = kbins_fit(x, n, d, words[j], nb, blocked, w)
    var cols: List[Int] = [0, 1, 2]
    check_edges(m, tag, d, cols, 1e-4 if j == 3 else 1e-6, "kbins " + tag + " blocked " + String(blocked))


def test_kbins_sample_weight() raises:
    check_cases("kbins_sample_weight", 48, sample_weight_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
