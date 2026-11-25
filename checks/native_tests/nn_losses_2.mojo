# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The log_softmax layer and the ctc loss of lane nn-losses-2 (2026-11-23), and nll / kl_div fitted through
Sequential (autograd/nn_log_softmax.mojo, autograd/nn_ctc.mojo, autograd/nn_ctc_ops.mojo).

One test, test_nn_losses_2, runs twelve independent units across host cores (each on its own HostExec):
  - units 0-2, log_softmax: the layer (nn_log_softmax_forward, the Sequential layer's recording) on (3, 4, 5) samples
    along dim -1 (the row op) and dim 1 (permute, row op, permute back): values against a float64 oracle; the input
    gradient of sum(w * y) against the oracle's closed form w - softmax * sum(w), and that closed form against
    central finite differences of the oracle, element by element.
  - units 3-7, the ctc tape cases: ctc_loss on log-probabilities (4 samples, T 5, C 3; repeated labels, an empty target, a
    shorter input; blank 0 and blank 2; mean and sum; zero_infinity on an impossible sample, and +inf without it):
    the value and the gradient in every log-probability against a float64 oracle that ENUMERATES every alignment
    (C^T paths a sample, collapsed and compared with the target), and the oracle's gradient against central finite
    differences of its own -log p.
  - units 8-10, Sequential fits through the public fit entry (model_fit_native):
      nll on a final log_softmax against cross_entropy on the same logits (same seed, same init): not the same
      bits, by construction, and the stated difference (below); both reach accuracy >= 0.9;
      kl_div (batchmean) on a final log_softmax: the loss falls under a quarter of its first value;
      ctc (25 epochs): a GRU (24) -> linear(4) -> log_softmax model learns to transcribe variable-length synthetic frame
      sequences (1 to 3 labels of 3 symbols, repeats included, inputs of 4 to 10 of the 10 padded steps through
      `lengths`): greedy decoding reaches a label accuracy (1 - edit distance / labels) >= 0.9 on its 96
      samples with finite loss, parameters and predictions; the admission refusals in words.
  - unit 11, the ctc gradient under a large common offset (lane stream-fits, 2026-11-25): log-probabilities and the
      same words 2^14 lower give the same gradient words (the normalized form), the oracle's gradient, and the moved
      loss = the loss + 2^14 x each row's input length, reduced.
  - one `NN_LOSSES_2 <case> digest` line over the value and gradient words of each case.

nll on log_softmax against cross_entropy (the stated difference): the row loss is the same word (cross_entropy's
row is log(sum exp(z - m)) - (z_y - m), nll's is -((z_y - m) - log(sum exp(z - m))), one subtraction negated, exact)
but the two differ (a) in the batch fold (cross_entropy folds the B row losses, nll's loss_reduced folds the B * C
terms, zeros included, in 64-term leaves: another grouping, so the last bits of the mean can differ) and (b) in the
backward: cross_entropy writes (exp(z - m) / sum - onehot) / B in one launch; nll passes -onehot / B through
log_softmax's backward, g - exp(log p) * sum(g), so its probabilities are exp(log p), not exp(z - m) / sum. The two
probabilities agree to a few ulps, so the fits agree to rounding (checked: curves within 1e-5 relative, parameters
within 1e-4) and are not bit-identical. Keeping cross_entropy's fused node for nll would be a second copy of the
softmax's math; the layer composition is the product's one body each.

Columns: the CPU column runs on HostExec. The device column runs each case on DeviceExec AND HostExec and requires
the same words.
Sabotage: `-D MOJOLEARN_NN_LOSSES_SABOTAGE_CTC_REPEAT` lets the ctc recursion skip the blank between two equal labels
(the enumeration oracle must FAIL on the repeated-label samples); `-D MOJOLEARN_NN_LOSSES_SABOTAGE_LOG_SOFTMAX` makes
the layer record the softmax instead of its log (the log_softmax oracle must FAIL).
"""
from std.math import exp, log, isinf
from std.memory import bitcast
from std.testing import TestSuite
from std.time import perf_counter_ns
from core.host_parallel import host_parallelize
from autograd.tape import Tape, Tensor
from autograd.rankn_api import rn_shape
from autograd.losses import REDUCTION_MEAN, REDUCTION_SUM, REDUCTION_BATCHMEAN
from autograd.model_blocks import ModelRules
from autograd.nn_log_softmax import nn_log_softmax_forward
from autograd.model_blocks import LAYER_LOG_SOFTMAX
from autograd.nn_ctc import ctc_loss, ctc_blank_word, NnCtcRules
from autograd.model import (
    LOSS_NLL, LOSS_KL_DIV, LOSS_CROSS_ENTROPY, LOSS_CTC, LOSS_WORD_REDUCTION, TARGET_DENSE, TARGET_CODES,
    OUT_RAW, FIT_STATE_WORDS, parse_model,
)
from autograd.model_execution import model_fit_native, model_float_words
from core.cast_in import CastIn
from sequence.ops import FP
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.training_sequential_common import SeqModel, L_LINEAR, codes_as_words
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD

comptime L_RECURRENT = 41
comptime CELL_GRU = 2


def _fail(message: String) raises:
    print("FAIL NN_LOSSES_2", message)
    raise Error(message)


def _require(ok: Bool, message: String) raises:
    if not ok:
        _fail(message)


def _digest(words: List[Float32]) -> UInt64:
    var d = UInt64(1469598103934665603)
    for i in range(len(words)):  # small-loop(check digest): fixture-sized, no product work
        d = (d ^ UInt64(bitcast[DType.uint32](words[i]))) * UInt64(1099511628211)
    return d


def _close(name: String, actual: List[Float32], expected: List[Float64], rtol: Float64 = 3.0e-4,
           atol: Float64 = 3.0e-5) raises:
    if len(actual) != len(expected):
        _fail(name + ": wrong length " + String(len(actual)) + " vs " + String(len(expected)))
    for i in range(len(actual)):  # small-loop(check comparison): fixture-sized
        var e = expected[i]
        if not (abs(Float64(actual[i]) - e) <= atol + rtol * abs(e)):
            _fail(name + ": index " + String(i) + " is " + String(actual[i]) + ", the float64 oracle " + String(e))


def _same_words(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": host and device word counts differ")
    for i in range(len(a)):  # small-loop(check comparison): fixture-sized
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": host and device words differ at " + String(i) + " (" + String(a[i]) + " vs "
                  + String(b[i]) + ")")


def _finite(words: List[Float32], what: String) raises:
    for i in range(len(words)):  # small-loop(check): fixture-sized
        var v = words[i]
        if not (v == v) or abs(v) > Float32(3.0e38):
            _fail(what + ": word " + String(i) + " is not finite")


def _lcg(mut state: UInt64) -> Float64:
    """A uniform value in [-1, 1) of the fixture generator."""
    state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
    return Float64(Int((state >> 33) % UInt64(2000)) - 1000) * 0.001


# ---------------------------------------------------------------- log_softmax: float64 oracle
def _ls64(x: List[Float64], B: Int, d0: Int, d1: Int, axis: Int) -> List[Float64]:
    """log_softmax of x (B, d0, d1) along sample axis `axis` (0: d0, 1: d1)."""
    var out = List[Float64](length=len(x), fill=0.0)
    var n = d0 if axis == 0 else d1
    var m_ = d1 if axis == 0 else d0
    for b in range(B):  # small-loop(fixture)
        for o in range(m_):  # small-loop(fixture)
            var mx = -1.0e300
            for k in range(n):  # small-loop(fixture)
                var i = b * d0 * d1 + (k * d1 + o if axis == 0 else o * d1 + k)
                mx = max(mx, x[i])
            var s = 0.0
            for k in range(n):  # small-loop(fixture)
                var i = b * d0 * d1 + (k * d1 + o if axis == 0 else o * d1 + k)
                s += exp(x[i] - mx)
            for k in range(n):  # small-loop(fixture)
                var i = b * d0 * d1 + (k * d1 + o if axis == 0 else o * d1 + k)
                out[i] = x[i] - mx - log(s)
    return out^


def _ls_grad64(x: List[Float64], w: List[Float64], B: Int, d0: Int, d1: Int, axis: Int) -> List[Float64]:
    """d sum(w * log_softmax(x)) / dx = w - softmax(x) * sum(w) along the axis."""
    var y = _ls64(x, B, d0, d1, axis)
    var g = List[Float64](length=len(x), fill=0.0)
    var n = d0 if axis == 0 else d1
    var m_ = d1 if axis == 0 else d0
    for b in range(B):  # small-loop(fixture)
        for o in range(m_):  # small-loop(fixture)
            var sw = 0.0
            for k in range(n):  # small-loop(fixture)
                sw += w[b * d0 * d1 + (k * d1 + o if axis == 0 else o * d1 + k)]
            for k in range(n):  # small-loop(fixture)
                var i = b * d0 * d1 + (k * d1 + o if axis == 0 else o * d1 + k)
                g[i] = w[i] - exp(y[i]) * sw
    return g^


def _ls_obj64(x: List[Float64], w: List[Float64], B: Int, d0: Int, d1: Int, axis: Int) -> Float64:
    var y = _ls64(x, B, d0, d1, axis)
    var s = 0.0
    for i in range(len(y)):  # small-loop(fixture)
        s += w[i] * y[i]
    return s


def run_log_softmax_case[E: Exec](mut ex: E, dim: Int) raises -> List[Float32]:
    var B = 3
    var d0 = 4
    var d1 = 5
    var state = UInt64(77 + dim)
    var x32 = List[Float32]()
    var w32 = List[Float32]()
    for _ in range(B * d0 * d1):  # small-loop(fixture)
        x32.append(Float32(3.0 * _lcg(state)))
        w32.append(Float32(_lcg(state)))
    var x = List[Float64]()
    var w = List[Float64]()
    for i in range(len(x32)):  # small-loop(fixture)
        x.append(Float64(x32[i]))
        w.append(Float64(w32[i]))
    var axis = 1 if dim == -1 or dim == 2 else 0
    var dims: List[Int] = [B, d0, d1]
    var sd: List[Int] = [d0, d1]
    var tape = Tape()
    var xt = tape.tensor(ex, x32, B * d0, d1, True)
    var wt = tape.tensor(ex, w32, B * d0, d1, False)
    var xs = tape.view_nd(ex, xt, rn_shape(dims, "x"))
    var y = nn_log_softmax_forward(tape, ex, dim, rn_shape(sd, "sample"), xs)
    var yv = tape.values(ex, y)
    var prod = tape.multiply(ex, tape.view_nd(ex, y, rn_shape(dims, "y")), tape.view_nd(ex, wt, rn_shape(dims, "w")))
    var loss = tape.sum(ex, prod)
    tape.backward_scalar_with_rules(ex, loss, ModelRules())
    var g = tape.gradients(ex, xt)
    var name = "log_softmax dim " + String(dim)
    _close(name + " value", yv, _ls64(x, B, d0, d1, axis))
    var g64 = _ls_grad64(x, w, B, d0, d1, axis)
    _close(name + " input gradient", g, g64)
    # the oracle's closed form against central differences of the oracle
    var h = 1.0e-6
    for i in range(len(x)):  # small-loop(check): fixture-sized
        var xp = x.copy()
        var xm = x.copy()
        xp[i] += h
        xm[i] -= h
        var fd = (_ls_obj64(xp, w, B, d0, d1, axis) - _ls_obj64(xm, w, B, d0, d1, axis)) / (2.0 * h)
        _require(abs(fd - g64[i]) <= 1.0e-5 + 1.0e-5 * abs(fd), name + ": oracle gradient " + String(g64[i])
                 + " vs central difference " + String(fd) + " at " + String(i))
    tape.release(ex)
    var words = yv.copy()
    for i in range(len(g)):  # small-loop(fixture)
        words.append(g[i])
    print("NN_LOSSES_2 log_softmax_dim" + String(dim), "digest", _digest(words))
    return words^


# ---------------------------------------------------------------- ctc: the enumeration oracle
@fieldwise_init
struct CtcRow(Copyable, Movable):
    var tin: Int
    var labels: List[Int]


def _collapse(path: List[Int], blank: Int) -> List[Int]:
    var out = List[Int]()
    var prev = -1
    for k in range(len(path)):  # small-loop(fixture)
        var c = path[k]
        if c != prev and c != blank:
            out.append(c)
        prev = c
    return out^


def _ctc64(lp: List[Float64], base: Int, C: Int, row: CtcRow, blank: Int) -> Tuple[Float64, List[Float64]]:
    """(p, the sum of the valid paths' probabilities through (t, c)) of one sample, every path enumerated."""
    var T = row.tin
    var through = List[Float64](length=T * C, fill=0.0)
    var p = 0.0
    var total = 1
    for _ in range(T):  # small-loop(fixture)
        total *= C
    for idx in range(total):  # small-loop(fixture: C^T paths, at most 243)
        var path = List[Int]()
        var v = idx
        for _ in range(T):  # small-loop(fixture)
            path.append(v % C)
            v //= C
        var col = _collapse(path, blank)
        if len(col) != len(row.labels):
            continue
        var ok = True
        for k in range(len(col)):  # small-loop(fixture)
            if col[k] != row.labels[k]:
                ok = False
        if not ok:
            continue
        var s = 0.0
        for t in range(T):  # small-loop(fixture)
            s += lp[base + t * C + path[t]]
        var pr = exp(s)
        p += pr
        for t in range(T):  # small-loop(fixture)
            through[t * C + path[t]] += pr
    return (p, through^)


def _ctc_total64(lp: List[Float64], rows: List[CtcRow], T: Int, C: Int, blank: Int, reduction: Int) -> Float64:
    var B = len(rows)
    var total = 0.0
    for b in range(B):  # small-loop(fixture)
        var r = _ctc64(lp, b * T * C, C, rows[b], blank)
        var nll = -log(r[0]) if r[0] > 0.0 else 1.0e300
        var L = len(rows[b].labels)
        total += nll if reduction == REDUCTION_SUM else nll / Float64(L if L > 0 else 1)
    return total if reduction == REDUCTION_SUM else total / Float64(B)


@fieldwise_init
struct CtcCase(Copyable, Movable):
    var name: String
    var rows: List[CtcRow]
    var blank: Int
    var reduction: Int
    var zero_infinity: Bool
    var impossible: Int  # the row no alignment produces (-1: none)


def _ctc_cases() -> List[CtcCase]:
    var base = List[CtcRow]()
    base.append(CtcRow(5, [1, 2]))
    base.append(CtcRow(4, [1, 1]))      # a repeat: the blank between them is required
    base.append(CtcRow(5, [2, 2, 1]))   # a repeat and a third label: 4 of the 5 steps needed
    base.append(CtcRow(3, List[Int]()))  # the empty target: every step blank
    var other = List[CtcRow]()           # blank 2: labels 0 and 1
    other.append(CtcRow(5, [0, 1]))
    other.append(CtcRow(4, [0, 0]))
    other.append(CtcRow(5, [1]))
    other.append(CtcRow(2, [0]))
    var bad = List[CtcRow]()             # sample 1 cannot be aligned: [1, 1] needs 3 steps, it has 2
    bad.append(CtcRow(5, [1, 2]))
    bad.append(CtcRow(2, [1, 1]))
    bad.append(CtcRow(5, [2]))
    bad.append(CtcRow(4, [1, 2, 1]))
    var out = List[CtcCase]()
    out.append(CtcCase("blank0/mean", base.copy(), 0, REDUCTION_MEAN, False, -1))
    out.append(CtcCase("blank0/sum", base.copy(), 0, REDUCTION_SUM, False, -1))
    out.append(CtcCase("blank2/mean", other.copy(), 2, REDUCTION_MEAN, False, -1))
    out.append(CtcCase("impossible/zero_infinity/mean", bad.copy(), 0, REDUCTION_MEAN, True, 1))
    out.append(CtcCase("impossible/inf/sum", bad.copy(), 0, REDUCTION_SUM, False, 1))
    return out^


comptime _CT = 5
comptime _CC = 3
comptime _CS = 3


def run_ctc_case[E: Exec](mut ex: E, c: CtcCase) raises -> List[Float32]:
    var B = len(c.rows)
    var T = _CT
    var C = _CC
    var state = UInt64(20261123) + UInt64(c.blank * 7 + c.reduction)
    # log-probabilities: a float64 log_softmax of random logits, rounded once to float32
    var lp32 = List[Float32]()
    for _ in range(B * T):  # small-loop(fixture)
        var z = List[Float64]()
        for _ in range(C):  # small-loop(fixture)
            z.append(2.0 * _lcg(state))
        var mx = max(z[0], max(z[1], z[2]))
        var s = 0.0
        for k in range(C):  # small-loop(fixture)
            s += exp(z[k] - mx)
        for k in range(C):  # small-loop(fixture)
            lp32.append(Float32(z[k] - mx - log(s)))
    var lp = List[Float64]()
    for i in range(len(lp32)):  # small-loop(fixture)
        lp.append(Float64(lp32[i]))
    var TW = 2 + _CS
    var rows = List[Float32]()
    for b in range(B):  # small-loop(fixture)
        rows.append(Float32(c.rows[b].tin))
        rows.append(Float32(len(c.rows[b].labels)))
        for k in range(_CS):  # small-loop(fixture)
            rows.append(Float32(c.rows[b].labels[k]) if k < len(c.rows[b].labels) else Float32(-9))
    var tape = Tape()
    var lt = tape.tensor(ex, lp32, B * T, C, True)
    var tt = tape.tensor(ex, rows, B, TW, False)
    var loss = ctc_loss(tape, ex, lt, tt, T, C, c.blank, c.reduction, c.zero_infinity)
    tape.backward_scalar_with_rules(ex, loss, NnCtcRules())
    var value = tape.values(ex, loss)[0]
    var g = tape.gradients(ex, lt)
    tape.release(ex)
    # the oracle: each sample's p and its paths' mass through (t, c); the reduced loss and its gradient
    var g64 = List[Float64](length=B * T * C, fill=0.0)
    var total = 0.0
    for b in range(B):  # small-loop(fixture)
        var r = _ctc64(lp, b * T * C, C, c.rows[b], c.blank)
        var L = len(c.rows[b].labels)
        var factor = 1.0 if c.reduction == REDUCTION_SUM else 1.0 / Float64(L if L > 0 else 1) / Float64(B)
        if r[0] == 0.0:
            _require(b == c.impossible, c.name + ": sample " + String(b) + " has no alignment in the oracle")
            continue  # loss +inf (or 0 under zero_infinity), gradient 0
        total += -log(r[0]) * factor
        for t in range(c.rows[b].tin):  # small-loop(fixture)
            for k in range(C):  # small-loop(fixture)
                g64[b * T * C + t * C + k] = -r[1][t * C + k] / r[0] * factor
    if c.impossible >= 0 and not c.zero_infinity:
        _require(isinf(value) and value > 0, c.name + ": the loss " + String(value) + " is not +inf")
    else:
        var ev: List[Float64] = [total]
        var av: List[Float32] = [value]
        _close(c.name + " value", av, ev)
    _close(c.name + " gradient", g, g64)
    # the oracle's gradient against central differences of its own reduced loss (possible samples only)
    if c.impossible < 0:
        var h = 1.0e-6
        for i in range(len(lp)):  # small-loop(check): fixture-sized
            var lpp = lp.copy()
            var lpm = lp.copy()
            lpp[i] += h
            lpm[i] -= h
            var fd = (_ctc_total64(lpp, c.rows, T, C, c.blank, c.reduction)
                      - _ctc_total64(lpm, c.rows, T, C, c.blank, c.reduction)) / (2.0 * h)
            _require(abs(fd - g64[i]) <= 1.0e-5 + 1.0e-4 * abs(fd), c.name + ": oracle gradient " + String(g64[i])
                     + " vs central difference " + String(fd) + " at " + String(i))
    var words = g.copy()
    words.append(value)
    print("NN_LOSSES_2 ctc_" + c.name, "value", value, "digest", _digest(words))
    return words^


def _ctc_tape[E: Exec](
    mut ex: E, lp32: List[Float32], rows: List[Float32], B: Int, T: Int, C: Int, blank: Int, reduction: Int,
) raises -> Tuple[Float32, List[Float32]]:
    """(the reduced loss, its gradient in the log-probabilities) of one tape ctc_loss call."""
    var tape = Tape()
    var lt = tape.tensor(ex, lp32, B * T, C, True)
    var tt = tape.tensor(ex, rows, B, len(rows) // B, False)
    var loss = ctc_loss(tape, ex, lt, tt, T, C, blank, reduction, False)
    tape.backward_scalar_with_rules(ex, loss, NnCtcRules())
    var value = tape.values(ex, loss)[0]
    var g = tape.gradients(ex, lt)
    tape.release(ex)
    return (value, g^)


comptime _CTC_OFFSET = 16384.0


def run_ctc_offset_case[E: Exec](mut ex: E) raises -> List[Float32]:
    """Unit 11 (lane stream-fits, 2026-11-25): the gradient under a large common offset. Log-probabilities on a
    2^-8 grid (|lp| < 8) and the same words minus 2^14 (exact: 22 significant bits): every step's shift is exact
    on both, so the normalized scans see the same words and the gradient must be the SAME WORDS (the raw form of
    before 2026-11-25, exp(alpha + beta - lp + nll) with nll near 5 * 2^14, moves them by ulp(nll), about 1e-2
    relative). The unshifted gradient against the enumeration oracle; the shifted loss against the unshifted one
    plus the offset times each row's input length (scaled as the reduction scales it)."""
    var rows_c = _ctc_cases()[0].rows.copy()
    var B = len(rows_c)
    var T = _CT
    var C = _CC
    var state = UInt64(20261125)
    var lp32 = List[Float32]()
    var shifted = List[Float32]()
    for _ in range(B * T):  # small-loop(fixture)
        var z = List[Float64]()
        for _ in range(C):  # small-loop(fixture)
            z.append(2.0 * _lcg(state))
        var mx = max(z[0], max(z[1], z[2]))
        var s = 0.0
        for k in range(C):  # small-loop(fixture)
            s += exp(z[k] - mx)
        for k in range(C):  # small-loop(fixture)
            var q = Float64(Int((z[k] - mx - log(s)) * 256.0)) / 256.0  # on the 2^-8 grid, |q| < 8
            lp32.append(Float32(q))
            shifted.append(Float32(q - _CTC_OFFSET))
    var rows = List[Float32]()
    for b in range(B):  # small-loop(fixture)
        rows.append(Float32(rows_c[b].tin))
        rows.append(Float32(len(rows_c[b].labels)))
        for k in range(_CS):  # small-loop(fixture)
            rows.append(Float32(rows_c[b].labels[k]) if k < len(rows_c[b].labels) else Float32(-9))
    var plain = _ctc_tape(ex, lp32, rows, B, T, C, 0, REDUCTION_MEAN)
    var moved = _ctc_tape(ex, shifted, rows, B, T, C, 0, REDUCTION_MEAN)
    _same_words("ctc offset gradient (2^14 below vs unmoved)", plain[1], moved[1])
    var lp = List[Float64]()
    for i in range(len(lp32)):  # small-loop(fixture)
        lp.append(Float64(lp32[i]))
    var g64 = List[Float64](length=B * T * C, fill=0.0)
    var total = 0.0
    var extra = 0.0
    for b in range(B):  # small-loop(fixture)
        var r = _ctc64(lp, b * T * C, C, rows_c[b], 0)
        var L = len(rows_c[b].labels)
        var factor = 1.0 / Float64(L if L > 0 else 1) / Float64(B)
        total += -log(r[0]) * factor
        extra += _CTC_OFFSET * Float64(rows_c[b].tin) * factor
        for t in range(rows_c[b].tin):  # small-loop(fixture)
            for k in range(C):  # small-loop(fixture)
                g64[b * T * C + t * C + k] = -r[1][t * C + k] / r[0] * factor
    _close("ctc offset gradient", plain[1], g64)
    var pv: List[Float32] = [plain[0]]
    var mv: List[Float32] = [moved[0]]
    var pe: List[Float64] = [total]
    var me: List[Float64] = [total + extra]
    _close("ctc offset value (unmoved)", pv, pe)
    _close("ctc offset value (2^14 below)", mv, me, 1.0e-6, 1.0e-2)
    var words = plain[1].copy()
    words.append(plain[0])
    words.append(moved[0])
    print("NN_LOSSES_2 ctc_offset value", moved[0], "digest", _digest(words))
    return words^


# ---------------------------------------------------------------- Sequential fits
comptime _N = 96
comptime _D = 4
comptime _K = 3


def _class_data() -> Tuple[List[Float32], List[Int], List[Float32]]:
    """X (96, 4) in [-1, 1), the labels argmax(X T) and the teacher's probabilities softmax(X T)."""
    var state = UInt64(4242)
    var x = List[Float32]()
    for _ in range(_N * _D):  # small-loop(fixture)
        x.append(Float32(_lcg(state)))
    var teacher: List[Float64] = [2.0, -1.0, 0.5, -1.5, 2.0, 0.0, 1.0, 1.0, -2.0, 0.5, -0.5, 1.5]
    var labels = List[Int]()
    var probs = List[Float32]()
    for i in range(_N):  # small-loop(fixture)
        var z = List[Float64]()
        for k in range(_K):  # small-loop(fixture)
            var s = 0.0
            for d in range(_D):  # small-loop(fixture)
                s += Float64(x[i * _D + d]) * teacher[d * _K + k]
            z.append(s)
        var best = 0
        for k in range(1, _K):  # small-loop(fixture)
            if z[k] > z[best]:
                best = k
        labels.append(best)
        var mx = max(z[0], max(z[1], z[2]))
        var s = 0.0
        for k in range(_K):  # small-loop(fixture)
            s += exp(z[k] - mx)
        for k in range(_K):  # small-loop(fixture)
            probs.append(Float32(exp(z[k] - mx) / s))
    return (x^, labels^, probs^)


def _classifier(loss: Int, log_softmax: Bool) raises -> SeqModel:
    var m = SeqModel()
    m.sample = [_D]
    m.layer(L_LINEAR, _K, 1)
    if log_softmax:
        m.layer(LAYER_LOG_SOFTMAX, -1)
    m.compile(loss, "sgd", 0.5)
    return m^


def _accuracy(scores: List[Float32], labels: List[Int]) -> Float64:
    var right = 0
    for i in range(len(labels)):  # small-loop(check): fixture-sized
        var best = 0
        for k in range(1, _K):  # small-loop(check)
            if scores[i * _K + k] > scores[i * _K + best]:
                best = k
        if best == labels[i]:
            right += 1
    return Float64(right) / Float64(len(labels))


def run_class_fits[E: Exec](mut ex: E) raises -> List[Float32]:
    """nll on log_softmax vs cross_entropy on logits, and kl_div batchmean on log_softmax."""
    var data = _class_data()
    var x = data[0].copy()
    var labels = data[1].copy()
    var probs = data[2].copy()
    var codes = codes_as_words(labels)
    var a = _classifier(LOSS_NLL, True)
    _ = a.fit(ex, x, _N, _D, codes, 1, TARGET_CODES, 30, 16, 0, 3)
    var b = _classifier(LOSS_CROSS_ENTROPY, False)
    _ = b.fit(ex, x, _N, _D, codes, 1, TARGET_CODES, 30, 16, 0, 3)
    var pa = a.predict(ex, x, _N, _D, OUT_RAW)
    var pb = b.predict(ex, x, _N, _D, OUT_RAW)
    _finite(a.losses(), "nll fit loss curve")
    _finite(a.params, "nll fit parameters")
    _finite(pa, "nll fit log-probabilities")
    # the layer through Sequential's predict: log_softmax of the fitted logits, against float64
    var want = List[Float64]()
    for i in range(_N):  # small-loop(check)
        var z = List[Float64]()
        for k in range(_K):  # small-loop(check)
            var s = Float64(a.params[_D * _K + k])
            for d in range(_D):  # small-loop(check)
                s += Float64(x[i * _D + d]) * Float64(a.params[d * _K + k])
            z.append(s)
        var mx = max(z[0], max(z[1], z[2]))
        var s = 0.0
        for k in range(_K):  # small-loop(check)
            s += exp(z[k] - mx)
        for k in range(_K):  # small-loop(check)
            want.append(z[k] - mx - log(s))
    _close("Sequential log_softmax predictions", pa, want, 1.0e-4, 1.0e-5)
    var acc_a = _accuracy(pa, labels)
    var acc_b = _accuracy(pb, labels)
    _require(acc_a >= 0.9 and acc_b >= 0.9, "nll / cross_entropy accuracy " + String(acc_a) + " / " + String(acc_b)
             + " under 0.9")
    var ca = a.losses()
    var cb = b.losses()
    var curve_rel = 0.0
    for e in range(len(ca)):  # small-loop(check)
        curve_rel = max(curve_rel, abs(Float64(ca[e]) - Float64(cb[e])) / abs(Float64(cb[e])))
    var pdiff = 0.0
    var same = 0
    for k in range(len(a.params)):  # small-loop(check)
        pdiff = max(pdiff, abs(Float64(a.params[k]) - Float64(b.params[k])))
        if bitcast[DType.uint32](a.params[k]) == bitcast[DType.uint32](b.params[k]):
            same += 1
    # the stated difference (module docstring): rounding only, not bits
    _require(curve_rel < 1.0e-5 and pdiff < 1.0e-4, "nll on log_softmax vs cross_entropy: curves differ by "
             + String(curve_rel) + " relative, parameters by " + String(pdiff))
    print("NN_LOSSES_2 nll_vs_cross_entropy accuracy", acc_a, acc_b, "first loss words equal",
          bitcast[DType.uint32](ca[0]) == bitcast[DType.uint32](cb[0]), "curve max rel diff", curve_rel,
          "param max abs diff", pdiff, "equal param words", same, "of", len(a.params))
    var kl = _classifier(LOSS_KL_DIV + LOSS_WORD_REDUCTION * REDUCTION_BATCHMEAN, True)
    _ = kl.fit(ex, x, _N, _D, probs, _K, TARGET_DENSE, 40, 16, 0, 3)
    var ck = kl.losses()
    _finite(ck, "kl_div fit loss curve")
    _finite(kl.params, "kl_div fit parameters")
    _require(ck[len(ck) - 1] < Float32(0.25) * ck[0], "kl_div fit: the loss " + String(ck[len(ck) - 1])
             + " is not under a quarter of " + String(ck[0]))
    print("NN_LOSSES_2 Sequential kl_div loss", ck[0], "->", ck[len(ck) - 1])
    var words = a.params.copy()
    for k in range(len(b.params)):  # small-loop(fixture)
        words.append(b.params[k])
    for k in range(len(kl.params)):  # small-loop(fixture)
        words.append(kl.params[k])
    print("NN_LOSSES_2 class_fits digest", _digest(words))
    return words^


# ---------------------------------------------------------------- ctc fit
comptime _FT = 10   # padded steps
comptime _FF = 4    # features: one-hot of silence (0) and the three symbols
comptime _FS = 3    # labels per sample at most
comptime _FC = 4    # classes: blank 0 and the three symbols


def _ctc_data() -> Tuple[List[Float32], List[Float32], List[Float32], List[Float32], List[List[Int]]]:
    """(X (96, 10 * 4), labels (96, 3), target lengths, input lengths, the label lists). Sample i has L in 1..3
    labels of {1, 2, 3}; its frames are silence, then for each label two frames of the label's symbol and one of
    silence: 1 + 3 L of the 10 steps, the rest zero padding (past `lengths`). Each feature carries noise in
    [-0.1, 0.1)."""
    var state = UInt64(31337)
    var x = List[Float32]()
    var y = List[Float32]()
    var tl = List[Float32]()
    var il = List[Float32]()
    var seqs = List[List[Int]]()
    for i in range(_N):  # small-loop(fixture)
        var L = 1 + i % 3
        var labs = List[Int]()
        for _ in range(L):  # small-loop(fixture)
            labs.append(1 + Int((_lcg(state) + 1.0) * 1.5) % 3)
        var frames = List[Int]()
        frames.append(0)
        for k in range(L):  # small-loop(fixture)
            frames.append(labs[k])
            frames.append(labs[k])
            frames.append(0)
        for t in range(_FT):  # small-loop(fixture)
            for f in range(_FF):  # small-loop(fixture)
                if t < len(frames):
                    var v = 0.1 * _lcg(state)
                    if frames[t] == f:
                        v += 1.0
                    x.append(Float32(v))
                else:
                    x.append(Float32(0))
        for k in range(_FS):  # small-loop(fixture)
            y.append(Float32(labs[k]) if k < L else Float32(0))
        tl.append(Float32(L))
        il.append(Float32(len(frames)))
        seqs.append(labs^)
    return (x^, y^, tl^, il^, seqs^)


def _ctc_model(blank_word: Float32) raises -> SeqModel:
    var m = SeqModel()
    m.sample = [_FT, _FF]
    m.layer(L_RECURRENT, 24, CELL_GRU)
    m.layer(L_LINEAR, _FC, 1)
    m.layer(LAYER_LOG_SOFTMAX, -1)
    m.compile(LOSS_CTC, "adam", 0.02)
    m.huber_delta = Float64(blank_word)
    return m^


def _fit_ctc[E: Exec](
    mut ex: E, mut m: SeqModel, x: List[Float32], y: List[Float32], S: Int, tl: List[Float32], il: List[Float32],
    epochs: Int, with_target_lengths: Bool = True,
) raises:
    """`fit(X, y, lengths=il, target_lengths=tl)` through the fit entry: the stateful words plus the ninth (lengths)
    and tenth (target lengths) addresses."""
    var D = _FT * _FF
    var prog = m.program(D)
    var table = m.table()
    var spec = parse_model(prog, model_float_words(table))
    m.params = List[Float32](length=spec.n_params, fill=Float32(0))
    m.m = List[Float32](length=spec.n_params, fill=Float32(0))
    m.v = List[Float32](length=spec.n_params, fill=Float32(0))
    m.state = List[Int64](length=FIT_STATE_WORDS, fill=Int64(0))
    m.curve = List[Float32](length=epochs, fill=Float32(0))
    var ip: List[Int] = [_N, m.loss, TARGET_DENSE, m.opt, m.flags, m.sched, m.step_size, 16, epochs, 1, 5, 0,
                         m.patience, 1, 0]
    var fp: List[Float64] = [m.lr, m.f1, m.f2, m.eps, m.weight_decay, 0.0, m.tol, m.gamma, m.huber_delta]
    for i in range(len(table)):  # driver-loop: over layers
        fp.append(table[i])
    var addrs: List[Int] = [
        0, 0, Int(m.params.unsafe_ptr()), Int(m.curve.unsafe_ptr()), Int(m.m.unsafe_ptr()), Int(m.v.unsafe_ptr()),
        Int(m.state.unsafe_ptr()), 0, Int(il.unsafe_ptr()),
    ]
    if with_target_lengths:
        addrs.append(Int(tl.unsafe_ptr()))
    var sources: List[CastIn] = [CastIn.f32_c(Int(x.unsafe_ptr()), _N, D), CastIn.f32_c(Int(y.unsafe_ptr()), _N, S)]
    m.n_iter = model_fit_native(ex, addrs, sources, prog, ip, fp)
    ex.sync()
    m.fitted = True
    _ = x
    _ = y
    _ = tl
    _ = il


def _edit(a: List[Int], b: List[Int]) -> Int:
    var prev = List[Int]()
    for j in range(len(b) + 1):  # small-loop(check)
        prev.append(j)
    for i in range(1, len(a) + 1):  # small-loop(check)
        var cur = List[Int]()
        cur.append(i)
        for j in range(1, len(b) + 1):  # small-loop(check)
            var cost = 0 if a[i - 1] == b[j - 1] else 1
            cur.append(min(min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost))
        prev = cur^
    return prev[len(b)]


def run_ctc_refusals[E: Exec](mut ex: E) raises -> List[Float32]:
    """The ctc fit's admission refusals, in words, before anything trains."""
    var data = _ctc_data()
    var x = data[0].copy()
    var y = data[1].copy()
    var tl = data[2].copy()
    var il = data[3].copy()
    # refusals in words: a label equal to the blank, a target length past y's width, no target lengths
    var y_bad = y.copy()
    y_bad[0] = Float32(0)
    var r = _ctc_model(ctc_blank_word(0, False))
    try:
        _fit_ctc(ex, r, x, y_bad, _FS, tl, il, 1)
        _fail("a ctc label equal to the blank was accepted")
    except e:
        _refused("blank label", String(e), "must not hold the blank")
    var tl_bad = tl.copy()
    tl_bad[3] = Float32(4)
    try:
        _fit_ctc(ex, r, x, y, _FS, tl_bad, il, 1)
        _fail("a target length past the targets' width was accepted")
    except e:
        _refused("target length", String(e), "target_lengths must be whole numbers")
    try:
        _fit_ctc(ex, r, x, y, _FS, tl, il, 1, False)
        _fail("a ctc fit without target lengths was accepted")
    except e:
        _refused("no target lengths", String(e), "needs target_lengths")
    print("NN_LOSSES_2 ctc refusals 3")
    var words: List[Float32] = [Float32(3)]
    return words^


def _refused(what: String, err: String, want: String) raises:
    if want not in err:
        _fail("ctc refusal (" + what + "): '" + err + "' does not say '" + want + "'")


def run_ctc_fit[E: Exec](mut ex: E) raises -> List[Float32]:
    var data = _ctc_data()
    var x = data[0].copy()
    var y = data[1].copy()
    var tl = data[2].copy()
    var il = data[3].copy()
    var labs = data[4].copy()
    var m = _ctc_model(ctc_blank_word(0, False))
    _fit_ctc(ex, m, x, y, _FS, tl, il, 25)
    var c = m.losses()
    _finite(c, "ctc fit loss curve")
    _finite(m.params, "ctc fit parameters")
    m.predict_lengths = il.copy()
    var lp = m.predict(ex, x, _N, _FT * _FF, OUT_RAW)
    _finite(lp, "ctc fit log-probabilities")
    var edits = 0
    var count = 0
    for i in range(_N):  # small-loop(check): greedy decoding of the fixture
        var path = List[Int]()
        for t in range(Int(il[i])):  # small-loop(check)
            var best = 0
            for k in range(1, _FC):  # small-loop(check)
                if lp[i * _FT * _FC + t * _FC + k] > lp[i * _FT * _FC + t * _FC + best]:
                    best = k
            path.append(best)
        edits += _edit(_collapse(path, 0), labs[i])
        count += len(labs[i])
    var accuracy = 1.0 - Float64(edits) / Float64(count)
    # stated accuracy: at least 0.9 of the labels (1 - edit distance / labels), with finite loss
    _require(accuracy >= 0.9, "ctc fit: label accuracy " + String(accuracy) + " under 0.9 (loss " + String(c[0])
             + " -> " + String(c[len(c) - 1]) + ")")
    print("NN_LOSSES_2 ctc fit label accuracy", accuracy, "loss", c[0], "->", c[len(c) - 1])
    var words = m.params.copy()
    for i in range(len(lp)):  # small-loop(fixture)
        words.append(lp[i])
    print("NN_LOSSES_2 ctc_fit digest", _digest(words))
    return words^


# ---------------------------------------------------------------- tests
comptime _UNITS = 12


def _run_unit[E: Exec](mut ex: E, unit: Int) raises -> List[Float32]:
    """Units: 0-2 log_softmax (dim -1, 1, 2), 3-7 the ctc tape cases, 8 the class fits, 9 the ctc fit, 10 the ctc
    refusals, 11 the ctc gradient under a large common offset."""
    var t0 = perf_counter_ns()
    var words: List[Float32]
    var dims: List[Int] = [-1, 1, 2]
    if unit < 3:
        words = run_log_softmax_case(ex, dims[unit])
    elif unit < 8:
        words = run_ctc_case(ex, _ctc_cases()[unit - 3])
    elif unit == 8:
        words = run_class_fits(ex)
    elif unit == 9:
        words = run_ctc_fit(ex)
    elif unit == 10:
        words = run_ctc_refusals(ex)
    else:
        words = run_ctc_offset_case(ex)
    print("NN_LOSSES_2 unit", unit, "seconds", Float64(perf_counter_ns() - t0) * 1.0e-9)
    return words^


def test_nn_losses_2() raises:
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
            print("FAIL NN_LOSSES_2 unit", unit, error)
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
    print("PASS NN_LOSSES_2 units", _UNITS)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
