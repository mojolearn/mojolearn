# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The round's additions of sub-lane session-ops-extra (2026-11-25): autograd/session_ops_extra.mojo, its bodies
autograd/session_ops_extra_ops.mojo and its rules autograd/session_ops_extra_rules.mojo (through
LayerRules[ImportRules], the rules a session uses once it recorded one of these ops).

One test, test_session_ops_extra, runs six independent units across host cores (each on its own HostExec):
  - unit 0, sequence_mask of lengths 1, 0, 4, 2.5 over 4 positions: the words t < lengths[b] (1.0 / 0.0).
  - unit 1, topk of a (2, 6) block with exact ties and a -0.0 / 0.0 pair, largest and smallest, k = 3: the indices
    against the stated rank order (ties to the lower position), the values (the gather) against the selected words,
    and the gradient of sum(w * values) against w placed at the selected positions; a (3, 37) block (a padded line
    of 64) against a float64 sort with the same tie rule.
  - unit 2, gather along axis 0 of a (4, 3) x by a (5, 3) index with repeats: the values against x's words, the
    gradient of sum(w * y) against a float64 sum of w over every gathered copy, and central differences.
  - unit 3, logsumexp along axis 1 of a (4, 5) block (one row of 1000s, one of only -inf): against a float64
    oracle (1000 + log 5; -inf), the gradient against float64 softmax * w (0 on the -inf row) and central
    differences; along axis 0 against the oracle.
  - unit 4, the soft-target cross-entropy (mean) of (4, 3) logits against a distribution: the loss against
    float64 -mean(sum(t log_softmax(z))), the logits' gradient against float64 (softmax * sum(t) - t) / rows and
    central differences.
  - unit 5, the refusals in words (topk k out of range, gather index out of range by position and value, gather
    index off x's shape, sequence_mask max_length 0, a target distribution of another shape).
  - one `SESSION_OPS_EXTRA <unit> digest` line over each unit's words.

Columns: the CPU column runs on HostExec. The device column runs each unit on DeviceExec AND HostExec and requires
the same words.
"""
from std.math import exp, log
from std.memory import bitcast
from std.testing import TestSuite
from core.host_parallel import host_parallelize
from autograd.tape import Tape, Tensor
from autograd.ops import AXIS_ALL
from autograd.rankn import ShapeN
from autograd.rankn_api import rn_as_shape
from autograd.import_api import ImportRules
from autograd.session_layer_rules import LayerRules
from autograd.losses import REDUCTION_MEAN
from autograd.session_ops_extra import (
    sox_sequence_mask, sox_topk_indices, sox_gather, sox_logsumexp, sox_soft_cross_entropy,
)
from sequence.ops import FP
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD

comptime _UNITS = 6


def _fail(message: String) raises:
    print("FAIL SESSION_OPS_EXTRA", message)
    raise Error(message)


def _require(ok: Bool, message: String) raises:
    if not ok:
        _fail(message)


def _digest(words: List[Float32]) -> UInt64:
    var d = UInt64(1469598103934665603)
    for i in range(len(words)):  # small-loop(check digest): fixture-sized, no product work
        d = (d ^ UInt64(bitcast[DType.uint32](words[i]))) * UInt64(1099511628211)
    return d


def _same_words(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": word counts differ " + String(len(a)) + " vs " + String(len(b)))
    for i in range(len(a)):  # small-loop(check comparison): fixture-sized
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _close(name: String, got: Float32, want: Float64, tol: Float64) raises:
    if not (abs(Float64(got) - want) <= tol * (1.0 + abs(want))):
        _fail(name + ": " + String(got) + ", want " + String(want))


def _lcg(mut state: UInt64) -> Float64:
    state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Float64(Int((state >> 33) % UInt64(2000)) - 1000) * 0.001


def _values(n: Int, seed: UInt64, scale: Float64) -> List[Float32]:
    var state = seed
    var out = List[Float32]()
    for _ in range(n):  # small-loop(fixture)
        out.append(Float32(_lcg(state) * scale))
    return out^


def _ids_fp[E: Exec](mut tape: Tape, mut ex: E, ids: List[Int], rows: Int, cols: Int) raises -> FP:
    var w = List[Float32]()
    for i in range(len(ids)):  # small-loop(fixture)
        w.append(bitcast[DType.float32](Int32(ids[i])))
    var t = tape.tensor(ex, w, rows, cols, False)
    return tape.value_of(ex, t)


def _int_words(words: List[Float32]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(words)):  # small-loop(fixture)
        out.append(Int(bitcast[DType.int32](words[i])))
    return out^


def _weighted_backward[E: Exec](mut tape: Tape, mut ex: E, y: Tensor, w: List[Float32]) raises:
    """sum(w * y) over y's matrix view, then the backward through LayerRules[ImportRules]."""
    var r = tape.rows(y)
    var c = tape.cols(y)
    var ym = rn_as_shape(tape, ex, y, ShapeN.matrix(r, c))
    var wt = tape.tensor(ex, w, r, c, False)
    var loss = tape.sum(ex, tape.multiply(ex, ym, wt), AXIS_ALL)
    tape.backward_scalar_with_rules(ex, loss, LayerRules[ImportRules](ImportRules()))


# ---------------------------------------------------------------- unit 0: sequence_mask
def run_mask[E: Exec](mut ex: E) raises -> List[Float32]:
    var tape = Tape()
    var lens: List[Float32] = [1.0, 0.0, 4.0, 2.5]
    var lt = tape.tensor(ex, lens, 4, 1, False)
    var m = tape.values(ex, sox_sequence_mask(tape, ex, lt, 4))
    tape.release(ex)
    for b in range(4):  # small-loop(fixture)
        for t in range(4):  # small-loop(fixture)
            var want = Float32(1.0) if Float32(t) < lens[b] else Float32(0.0)
            _require(m[b * 4 + t] == want, "sequence_mask (" + String(b) + ", " + String(t) + ") is " + String(m[b * 4 + t]))
    return m^


# ---------------------------------------------------------------- unit 1: topk (and its gather)
def _rank_order(row: List[Float64], largest: Bool) -> List[Int]:
    """Positions of `row` in the top-k order: by value (descending for largest), ties to the lower position."""
    var order = List[Int]()
    for j in range(len(row)):  # small-loop(fixture)
        order.append(j)
    for a in range(len(order)):  # small-loop(fixture: insertion sort of a short row)
        var b = a
        while b > 0:
            var p = order[b - 1]
            var q = order[b]
            var before = (row[q] > row[p]) if largest else (row[q] < row[p])
            if not before:
                break
            order[b - 1] = q
            order[b] = p
            b -= 1
    return order^


def _topk_case[E: Exec](
    mut ex: E, x: List[Float32], rows: Int, cols: Int, k: Int, largest: Bool, mut words: List[Float32],
) raises:
    var tape = Tape()
    var xt = tape.tensor(ex, x, rows, cols, True)
    var it = sox_topk_indices(tape, ex, xt, k, 1, largest)
    var iw = tape.values(ex, it)
    var vt = sox_gather(tape, ex, xt, 1, tape.value_of(ex, it), tape.shape_of(it))
    var vw = tape.values(ex, vt)
    var w = _values(rows * k, UInt64(5 + k), 1.0)
    _weighted_backward(tape, ex, vt, w)
    var g = tape.gradients(ex, xt)
    tape.release(ex)
    var idx = _int_words(iw)
    var want_g = List[Float32](length=rows * cols, fill=Float32(0.0))
    for r in range(rows):  # small-loop(fixture)
        var row = List[Float64]()
        for c in range(cols):  # small-loop(fixture)
            row.append(Float64(x[r * cols + c]))
        var order = _rank_order(row, largest)
        for q in range(k):  # small-loop(fixture)
            var at = r * k + q
            _require(idx[at] == order[q], "topk row " + String(r) + " rank " + String(q) + ": position "
                     + String(idx[at]) + ", want " + String(order[q]))
            _require(bitcast[DType.uint32](vw[at]) == bitcast[DType.uint32](x[r * cols + order[q]]),
                     "topk value at row " + String(r) + " rank " + String(q))
            want_g[r * cols + order[q]] = w[at]
    _same_words("topk gradient (w at the selected positions)", g, want_g)
    words.extend(iw^)
    words.extend(vw^)
    words.extend(g^)


def run_topk[E: Exec](mut ex: E) raises -> List[Float32]:
    var words = List[Float32]()
    var tied: List[Float32] = [1.0, 5.0, 3.0, 5.0, -0.0, 0.0, 0.0, -0.0, 7.0, -1.0, 7.0, 2.0]
    _topk_case(ex, tied, 2, 6, 3, True, words)
    _topk_case(ex, tied, 2, 6, 3, False, words)
    # a padded line (37 -> 64) with repeated values: rounded to a coarse grid so ties are common
    var wide = _values(3 * 37, UInt64(91), 4.0)
    for i in range(len(wide)):  # small-loop(fixture)
        wide[i] = Float32(Int(wide[i] * 2)) * Float32(0.5)
    _topk_case(ex, wide, 3, 37, 9, True, words)
    _topk_case(ex, wide, 3, 37, 37, False, words)
    return words^


# ---------------------------------------------------------------- unit 2: gather with repeats
def run_gather[E: Exec](mut ex: E) raises -> List[Float32]:
    var x = _values(12, UInt64(17), 2.0)
    var ids: List[Int] = [3, 0, 3, 0, 0, 1, 3, 2, 3, 1, 0, 0, 3, 3, 3]
    var w = _values(15, UInt64(29), 1.0)
    var tape = Tape()
    var xt = tape.tensor(ex, x, 4, 3, True)
    var ip = _ids_fp(tape, ex, ids, 5, 3)
    var yt = sox_gather(tape, ex, xt, 0, ip, ShapeN.matrix(5, 3))
    var y = tape.values(ex, yt)
    _weighted_backward(tape, ex, yt, w)
    var g = tape.gradients(ex, xt)
    tape.release(ex)
    var g64 = List[Float64](length=12, fill=0.0)
    for e in range(15):  # small-loop(fixture)
        var c = e % 3
        var src = ids[e] * 3 + c
        _require(bitcast[DType.uint32](y[e]) == bitcast[DType.uint32](x[src]), "gather value " + String(e))
        g64[src] += Float64(w[e])
    for i in range(12):  # small-loop(fixture)
        _close("gather gradient " + String(i) + " (float64 sum of every copy)", g[i], g64[i], 1.0e-6)
        # central difference of the float64 objective sum(w * x[ids]) (linear: exact up to rounding)
        var h = 1.0e-3
        var plus = 0.0
        var minus = 0.0
        for e in range(15):  # small-loop(fixture)
            var src = ids[e] * 3 + e % 3
            var v = Float64(x[src])
            plus += Float64(w[e]) * (v + (h if src == i else 0.0))
            minus += Float64(w[e]) * (v - (h if src == i else 0.0))
        _close("gather gradient " + String(i) + " (central difference)", g[i], (plus - minus) / (2.0 * h), 1.0e-4)
    var words = y^
    words.extend(g^)
    return words^


# ---------------------------------------------------------------- unit 3: logsumexp
def _lse64(row: List[Float64]) -> Float64:
    var m = row[0]
    for j in range(1, len(row)):  # small-loop(fixture)
        m = max(m, row[j])
    if m < -1.0e308:
        return m  # a row of only -inf
    var s = 0.0
    for j in range(len(row)):  # small-loop(fixture)
        s += exp(row[j] - m)
    return m + log(s)


def run_lse[E: Exec](mut ex: E) raises -> List[Float32]:
    var x = _values(20, UInt64(43), 3.0)
    var ninf = bitcast[DType.float32](UInt32(0xFF800000))
    for j in range(5):  # small-loop(fixture)
        x[5 + j] = Float32(1000.0)
        x[15 + j] = ninf
    var w = _values(4, UInt64(47), 1.0)
    var tape = Tape()
    var xt = tape.tensor(ex, x, 4, 5, True)
    var lt = sox_logsumexp(tape, ex, xt, 1, False)
    var l = tape.values(ex, lt)
    _weighted_backward(tape, ex, lt, w)
    var g = tape.gradients(ex, xt)
    var l0 = tape.values(ex, sox_logsumexp(tape, ex, xt, 0, True))
    tape.release(ex)
    _require(len(l) == 4 and len(l0) == 5, "logsumexp shapes")
    for r in range(4):  # small-loop(fixture)
        var row = List[Float64]()
        for j in range(5):  # small-loop(fixture)
            row.append(Float64(x[r * 5 + j]))
        var want = _lse64(row)
        if r == 3:
            _require(l[r] == ninf, "logsumexp of a row of only -inf is " + String(l[r]) + ", want -inf")
            for j in range(5):  # small-loop(fixture)
                _require(g[r * 5 + j] == Float32(0.0), "the -inf row passes a zero gradient")
            continue
        _close("logsumexp row " + String(r), l[r], want, 1.0e-6)
        for j in range(5):  # small-loop(fixture)
            var g64 = Float64(w[r]) * exp(row[j] - want)
            _close("logsumexp gradient (" + String(r) + ", " + String(j) + ")", g[r * 5 + j], g64, 1.0e-5)
            if r != 1:
                var h = 1.0e-4
                row[j] += h
                var up = _lse64(row)
                row[j] -= 2.0 * h
                var down = _lse64(row)
                row[j] += h
                _close("logsumexp central difference (" + String(r) + ", " + String(j) + ")", g[r * 5 + j],
                       Float64(w[r]) * (up - down) / (2.0 * h), 1.0e-4)
    for c in range(5):  # small-loop(fixture)
        var col = List[Float64]()
        for r in range(4):  # small-loop(fixture)
            col.append(Float64(x[r * 5 + c]))
        _close("logsumexp axis 0 column " + String(c), l0[c], _lse64(col), 1.0e-6)
    var words = l^
    words.extend(g^)
    words.extend(l0^)
    return words^


# ---------------------------------------------------------------- unit 4: soft-target cross-entropy
def _soft_ce64(z: List[Float64], t: List[Float32], rows: Int, cols: Int) -> Float64:
    var total = 0.0
    for r in range(rows):  # small-loop(fixture)
        var row = List[Float64]()
        for c in range(cols):  # small-loop(fixture)
            row.append(z[r * cols + c])
        var lse = _lse64(row)
        for c in range(cols):  # small-loop(fixture)
            total -= Float64(t[r * cols + c]) * (row[c] - lse)
    return total / Float64(rows)


def run_soft_ce[E: Exec](mut ex: E) raises -> List[Float32]:
    var z = _values(12, UInt64(53), 2.0)
    var t: List[Float32] = [0.8, 0.1, 0.1, 0.0, 1.0, 0.0, 0.2, 0.3, 0.5, 0.1, 0.1, 0.8]
    var tape = Tape()
    var zt = tape.tensor(ex, z, 4, 3, True)
    var tt = tape.tensor(ex, t, 4, 3, False)
    var loss = sox_soft_cross_entropy(tape, ex, zt, tt, REDUCTION_MEAN, None)
    var lv = tape.values(ex, loss)
    tape.backward_scalar_with_rules(ex, loss, LayerRules[ImportRules](ImportRules()))
    var g = tape.gradients(ex, zt)
    tape.release(ex)
    var z64 = List[Float64]()
    for i in range(12):  # small-loop(fixture)
        z64.append(Float64(z[i]))
    _close("soft cross-entropy loss", lv[0], _soft_ce64(z64, t, 4, 3), 1.0e-6)
    for r in range(4):  # small-loop(fixture)
        var row = List[Float64]()
        var ts = 0.0
        for c in range(3):  # small-loop(fixture)
            row.append(z64[r * 3 + c])
            ts += Float64(t[r * 3 + c])
        var lse = _lse64(row)
        for c in range(3):  # small-loop(fixture)
            var i = r * 3 + c
            _close("soft cross-entropy gradient " + String(i), g[i],
                   (exp(row[c] - lse) * ts - Float64(t[i])) / 4.0, 1.0e-5)
            var h = 1.0e-4
            z64[i] += h
            var up = _soft_ce64(z64, t, 4, 3)
            z64[i] -= 2.0 * h
            var down = _soft_ce64(z64, t, 4, 3)
            z64[i] += h
            _close("soft cross-entropy central difference " + String(i), g[i], (up - down) / (2.0 * h), 1.0e-4)
    var words = lv^
    words.extend(g^)
    return words^


# ---------------------------------------------------------------- unit 5: refusals
def _refused(what: String, err: String, want: String) raises:
    if want not in err:
        _fail(what + ": the refusal reads '" + err + "', want '" + want + "'")


def run_refusals[E: Exec](mut ex: E) raises -> List[Float32]:
    var tape = Tape()
    var x = tape.tensor(ex, _values(6, UInt64(3), 1.0), 2, 3, False)
    try:
        _ = sox_topk_indices(tape, ex, x, 4, 1, True)
        _fail("topk k 4 of 3 was not refused")
    except e:
        _refused("topk k", String(e), "topk: k must be in [1, 3]")
    var bad: List[Int] = [0, 1, 5, 2, 0, 1]
    try:
        _ = sox_gather(tape, ex, x, 1, _ids_fp(tape, ex, bad, 2, 3), ShapeN.matrix(2, 3))
        _fail("a gather index out of range was not refused")
    except e:
        _refused("gather range", String(e), "gather: index 5 at position 2 is outside [0, 3)")
    var ok: List[Int] = [0, 1, 2]
    try:
        _ = sox_gather(tape, ex, x, 1, _ids_fp(tape, ex, ok, 3, 1), ShapeN.matrix(3, 1))
        _fail("a gather index off x's shape was not refused")
    except e:
        _refused("gather shape", String(e), "gather: index must have x's size on every dimension except axis")
    try:
        _ = sox_sequence_mask(tape, ex, x, 0)
        _fail("sequence_mask max_length 0 was not refused")
    except e:
        _refused("sequence_mask", String(e), "sequence_mask: max_length must be an integer in [1, 2^24)")
    var t = tape.tensor(ex, _values(4, UInt64(9), 1.0), 2, 2, False)
    try:
        _ = sox_soft_cross_entropy(tape, ex, x, t, REDUCTION_MEAN, None)
        _fail("a target distribution of another shape was not refused")
    except e:
        _refused("soft target", String(e), "a target distribution must have the logits' shape")
    tape.release(ex)
    return List[Float32](length=1, fill=Float32(1.0))


def _run_unit[E: Exec](mut ex: E, unit: Int) raises -> List[Float32]:
    var words: List[Float32]
    if unit == 0:
        words = run_mask(ex)
    elif unit == 1:
        words = run_topk(ex)
    elif unit == 2:
        words = run_gather(ex)
    elif unit == 3:
        words = run_lse(ex)
    elif unit == 4:
        words = run_soft_ce(ex)
    else:
        words = run_refusals(ex)
    print("SESSION_OPS_EXTRA unit", unit, "digest", _digest(words))
    return words^


def test_session_ops_extra() raises:
    # Independent units: on the host column each runs on its own HostExec across host cores; the device column
    # then runs them in order on the one device and requires the host words.
    var failed = List[Int](length=_UNITS, fill=0)
    var host_words = List[List[Float32]](length=_UNITS, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = _run_unit(host, unit)
        except error:
            print("FAIL SESSION_OPS_EXTRA unit", unit, error)
            fp[unit] = 1
    host_parallelize(task, _UNITS)
    for u in range(_UNITS):  # small-loop(_UNITS: failure flags)
        if failed[u] != 0:
            _fail("unit " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for unit in range(_UNITS):  # driver-loop: over the units on the one device
            var dev = DeviceExec()
            var dw = _run_unit(dev, unit)
            _same_words("unit " + String(unit) + " host vs device", host_words[unit], dw)
    _ = failed^
    _ = host_words^
    print("PASS SESSION_OPS_EXTRA units", _UNITS)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
