# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Sequence lengths of the recurrent models (lane seq-blocks-2, 2026-11-22).

A row of length L inside a batch padded to T steps runs as a packed sequence:
its steps L .. T - 1 never enter its state (sequence/ops.mojo op_cell_fwd copies
h and c forward there), the loss and the gradient never see them (op_cell_bwd
passes dh and dc through with zero gate gradients; op_gather_seq stores +0.0 for
their inputs), and the head reads the state at the row's own last valid step.

Proofs, for the GRU and the LSTM cell (two layers, hidden 8, three features, two
outputs, T = 12):
  (a) each row of a padded batch against the same row run ALONE, unpadded (T' = L,
      no lengths): the head output and every parameter gradient (one forward and
      backward on the tape, the trainer's AutogradRnnStep, the same seed dy) are
      the same words when the padded row runs as its own batch of one (the
      padded steps add only +0.0 products at the end of each weight-gradient
      chain: T B <= 512 rows, one block, sequence/recurrent.mojo WGRAD_MIN_BLOCK);
      in the padded batch of six, every row's output is the alone row's words and
      the batch gradient is the float64 sum of the six alone gradients within
      1e-5 of the sum of the terms' magnitudes (the batch folds the six rows' terms in one chain, the alone
      runs fold each row's terms by itself: a different order of the same terms);
  (b) the fourth baseline: a GRU and an LSTM classifier fitted with lengths on
      variable-length synthetic tracks (class = the sign of the drift of channel
      0 over the row's valid steps; lengths 4 .. 16 of T = 16; the padded steps
      hold large noise) reach held-out accuracy >= 0.9 and their loss falls;
  (c) changing the VALUES in the padded steps (normal noise against +-inf, NaN
      and 1e30) changes no word: outputs and gradients of one step, and a whole
      fit's parameters, losses and predictions.
Columns: the CPU column runs everything on HostExec, units across host cores. The
device column runs every unit on DeviceExec AND on HostExec and requires the same
words: the host-column-versus-device bytes check.
"""
from std.math import sqrt
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args, OP_SEQ_IOTA, OPT_ADAM, CELL_GRU, CELL_LSTM
from sequence.ops_impl import rnn_fit_impl, rnn_predict_impl
from sequence.recurrent import Net, Work, gather_rows, gather_seq
from autograd.rules_recurrent import AutogradRnnStep
from checks.native_tests.sequence_port_common import normal, u01
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize


comptime _T = 12
comptime _D = 3
comptime _H = 8
comptime _L = 2
comptime _O = 2
comptime _ROWS = 6


def _fail(message: String) raises:
    print("FAIL SEQ_LENGTHS", message)
    raise Error(message)


def _fp(values: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(values.unsafe_ptr()))


def _mix(mut digest: UInt64, values: List[Float32]):
    for i in range(len(values)):  # small-loop(len(values): check digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ (" + String(len(a)) + " vs " + String(len(b)) + ")")
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _cell(which: Int) -> Int:
    return CELL_GRU if which == 0 else CELL_LSTM


def _cell_name(which: Int) -> String:
    return "gru" if which == 0 else "lstm"


def _row_lengths() -> List[Int]:
    var lens: List[Int] = [12, 7, 3, 1, 9, 5]
    return lens^


def _params(net: Net, salt: Int) -> List[Float32]:
    var n = net.n_params()
    var p = List[Float32](length=n, fill=0.0)
    var bound = 1.0 / sqrt(Float64(net.H))
    for i in range(n):  # small-loop(n: check parameters)
        p[i] = Float32((2.0 * u01(salt, i) - 1.0) * bound)
    return p^


def _pad_value(kind: Int, i: Int) -> Float32:
    """kind 0: normal noise times 1e3; kind 1: +inf, -inf, NaN and 1e30 in turn."""
    if kind == 0:
        return Float32(1000.0 * normal(0x5E61, i))
    var r = i % 4
    if r == 0:
        return bitcast[DType.float32](UInt32(0x7F800000))
    if r == 1:
        return bitcast[DType.float32](UInt32(0xFF800000))
    if r == 2:
        return bitcast[DType.float32](UInt32(0x7FC00000))
    return Float32(1e30)


def _row_value(row: Int, s: Int, d: Int) -> Float32:
    return Float32(normal(0x5E60, (row * _T + s) * _D + d))


def _batch(rows: List[Int], lens: List[Int], T: Int, pad_kind: Int) -> List[Float32]:
    """Batch-first (B, T, D): row r's first lens[r] steps are its values, the rest padding."""
    var B = len(rows)
    var x = List[Float32](length=B * T * _D, fill=0.0)
    for b in range(B):  # small-loop(B: check data)
        for s in range(T):  # small-loop(T: check data)
            for d in range(_D):  # small-loop(D: check data)
                var at = (b * T + s) * _D + d
                x[at] = _row_value(rows[b], s, d) if s < lens[b] else _pad_value(pad_kind, at)
    return x^


def _seed(rows: List[Int]) -> List[Float32]:
    var dy = List[Float32](length=len(rows) * _O, fill=0.0)
    for b in range(len(rows)):  # small-loop(B: check data)
        for o in range(_O):  # small-loop(O: check data)
            dy[b * _O + o] = Float32(normal(0x5E62, rows[b] * _O + o))
    return dy^


def step_words[E: Exec](
    mut ex: E, net: Net, x: List[Float32], B: Int, T: Int, lens: List[Int], dy: List[Float32],
    params: List[Float32],
) raises -> List[Float32]:
    """One forward and backward of the trainer's step (AutogradRnnStep) with the seed dy:
    the head output (B x O) then every parameter gradient. `lens` empty: no lengths."""
    var np_ = net.n_params()
    var has = len(lens) > 0
    var lw = List[Float32](length=B, fill=0.0)
    for b in range(len(lens)):  # small-loop(B: check lengths)
        lw[b] = Float32(lens[b])
    var dX = ex.alloc(B * T * net.D)
    ex.upload(dX, _fp(x), B * T * net.D)
    var idx = ex.alloc(B)
    var ia = Args()
    ia.p0 = idx
    ex.launch[OP_SEQ_IOTA](ia, B)
    var dL = ex.alloc(B)
    ex.upload(dL, _fp(lw), B)
    var P = ex.alloc(np_)
    ex.upload(P, _fp(params), np_)
    var Gr = ex.alloc(np_)
    var w = Work(ex, net, T, B, True, has)
    if has:
        gather_seq(ex, dX, idx, w.xb, T, B, net.D, 0, Optional[FP](dL))
        gather_rows(ex, dL, idx, w.lens, B, 1, 0)
    else:
        gather_seq(ex, dX, idx, w.xb, T, B, net.D, 0)
    var step = AutogradRnnStep()
    step.forward_head(ex, net, P, Gr, w.xb, T, B, w)
    ex.upload(w.dy, _fp(dy), B * net.O)
    step.backward(ex, net, P, Gr, w.xb, T, B, w)
    ex.sync()
    var out = List[Float32](length=B * net.O + np_, fill=0.0)
    ex.download(_fp(out), w.yhat, B * net.O)
    ex.download(FP(unsafe_from_address=Int(out.unsafe_ptr()) + 4 * B * net.O), Gr, np_)
    _ = lw^
    _ = step^
    return out^


def run_alone_case[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    """Proof (a) for one cell; returns every word it compared (the device column's check)."""
    var name = _cell_name(which)
    var net = Net(_cell(which), _D, _H, _L, _O)
    var params = _params(net, 0x5E63 + which)
    var np_ = net.n_params()
    var lens = _row_lengths()
    var all_rows = List[Int]()
    for r in range(_ROWS):  # small-loop(rows)
        all_rows.append(r)
    var words = List[Float32]()
    var grad_sum = List[Float64](length=np_, fill=0.0)
    var grad_abs = List[Float64](length=np_, fill=0.0)
    var alone_out = List[Float32]()
    for r in range(_ROWS):  # small-loop(rows: one alone and one padded run each)
        var one: List[Int] = [r]
        var len_one: List[Int] = [lens[r]]
        var dy = _seed(one)
        var alone = step_words(ex, net, _batch(one, len_one, lens[r], 0), 1, lens[r], List[Int](), dy, params)
        var padded = step_words(ex, net, _batch(one, len_one, _T, 0), 1, _T, len_one, dy, params)
        _same(name + " row " + String(r) + " (length " + String(lens[r]) + ") padded alone vs unpadded", alone, padded)
        for o in range(_O):  # small-loop(O)
            alone_out.append(alone[o])
        for i in range(np_):  # small-loop(np: gradient sum)
            grad_sum[i] += Float64(alone[_O + i])
            grad_abs[i] += abs(Float64(alone[_O + i]))
        words.extend(alone^)
    var dy_all = _seed(all_rows)
    var batch = step_words(ex, net, _batch(all_rows, lens, _T, 0), _ROWS, _T, lens, dy_all, params)
    for r in range(_ROWS):  # small-loop(rows: outputs)
        for o in range(_O):  # small-loop(O)
            var got = batch[r * _O + o]
            var want = alone_out[r * _O + o]
            if bitcast[DType.uint32](got) != bitcast[DType.uint32](want):
                _fail(name + " batch row " + String(r) + " output " + String(o) + ": " + String(got) + " vs alone "
                      + String(want))
    var worst = 0.0
    for i in range(np_):  # small-loop(np: gradient comparison)
        var got = Float64(batch[_ROWS * _O + i])
        # the float32 rounding of a sum is bounded by its terms' magnitudes, not by the sum
        var d = abs(got - grad_sum[i]) / (1e-30 + grad_abs[i])
        if d > worst:
            worst = d
    if worst > 1e-5:
        _fail(name + " batch gradient off the sum of the alone gradients: " + String(worst) + " of the terms' sum")
    print("SEQ_LENGTHS", name, "alone: 6 rows bit-equal padded vs unpadded (output + " + String(np_)
          + " gradients); batch outputs bit-equal; batch gradient vs float64 sum of alone: max |diff| / sum|terms|", worst)
    words.extend(batch^)
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("SEQ_LENGTHS", name, "alone digest", digest)
    return words^


def run_padding_case[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    """Proof (c) for one cell: the padded steps' values change no word."""
    var name = _cell_name(which)
    var cell = _cell(which)
    var net = Net(cell, _D, _H, _L, _O)
    var params = _params(net, 0x5E64 + which)
    var lens = _row_lengths()
    var rows = List[Int]()
    for r in range(_ROWS):  # small-loop(rows)
        rows.append(r)
    var dy = _seed(rows)
    var a = step_words(ex, net, _batch(rows, lens, _T, 0), _ROWS, _T, lens, dy, params)
    var b = step_words(ex, net, _batch(rows, lens, _T, 1), _ROWS, _T, lens, dy, params)
    _same(name + " one step, noise vs inf/NaN padding", a, b)
    var words = a^
    # a whole fit: 4 epochs, batch 4 (a short last batch), shuffled; then the predictions
    var lw = List[Float32](length=_ROWS, fill=0.0)
    var y = List[Float32](length=_ROWS, fill=0.0)
    for r in range(_ROWS):  # small-loop(rows)
        lw[r] = Float32(lens[r])
        y[r] = Float32(r % 2)
    var fits = List[List[Float32]]()
    for kind in range(2):  # small-loop(two padding kinds)
        var x = _batch(rows, lens, _T, kind)
        var p = params.copy()
        var epochs = 4
        var batch = 4
        var steps = epochs * ((_ROWS + batch - 1) // batch)
        var losses = List[Float32](length=steps, fill=0.0)
        var rates = List[Float32](length=steps, fill=Float32(1e-2))
        var ints: List[Int] = [cell, _D, _H, _L, _O, 1, _ROWS, _T, epochs, batch, OPT_ADAM, 0, 1, 77, 0]
        var floats: List[Float64] = [0.9, 0.999, 1e-8, 0.0, 0.0, 0.0]
        _ = rnn_fit_impl(ex, [Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(p.unsafe_ptr()), Int(losses.unsafe_ptr()),
                              Int(rates.unsafe_ptr()), Int(lw.unsafe_ptr())], ints, floats)
        var out = List[Float32](length=_ROWS * _O, fill=0.0)
        _ = rnn_predict_impl(ex, [Int(x.unsafe_ptr()), Int(p.unsafe_ptr()), Int(out.unsafe_ptr()),
                                  Int(out.unsafe_ptr())],
                             [cell, _D, _H, _L, _O, 1, _ROWS, _T, 4, 0, Int(lw.unsafe_ptr())])
        var fw = p^
        fw.extend(losses^)
        fw.extend(out^)
        fits.append(fw^)
        _ = x^
        _ = rates^
    _same(name + " fit + predict, noise vs inf/NaN padding", fits[0], fits[1])
    words.extend(fits[0].copy())
    print("SEQ_LENGTHS", name, "padding: one step and a whole fit + predict unchanged by the padded values")
    _ = lw^
    _ = y^
    return words^


# ---------------------------------------------------------------- baseline (b)
comptime _BT = 16
comptime _BD = 2
comptime _TRAIN = 256
comptime _TEST = 128


def _tracks(n: Int, salt: Int) -> Tuple[List[Float32], List[Float32], List[Float32]]:
    """(X, y, lengths): lengths 4 .. 16; channel 0 drifts up (class 1) or down (class 0) over the
    valid steps with noise, channel 1 is noise; padded steps hold noise of scale 5."""
    var x = List[Float32](length=n * _BT * _BD, fill=0.0)
    var y = List[Float32](length=n, fill=0.0)
    var lens = List[Float32](length=n, fill=0.0)
    for i in range(n):  # small-loop(n: check data)
        var cls = 1 if u01(salt, i, 7) < 0.5 else 0
        var length = 4 + Int(u01(salt, i, 8) * 13.0)
        if length > _BT:
            length = _BT
        y[i] = Float32(cls)
        lens[i] = Float32(length)
        var drift = 0.25 if cls == 1 else -0.25
        var level = 0.0
        for s in range(_BT):  # small-loop(T: check data)
            var at = (i * _BT + s) * _BD
            if s < length:
                level += drift + 0.3 * normal(salt + 1, i * _BT + s)
                x[at] = Float32(level)
                x[at + 1] = Float32(normal(salt + 2, i * _BT + s))
            else:
                x[at] = Float32(5.0 * normal(salt + 3, i * _BT + s))
                x[at + 1] = Float32(5.0 * normal(salt + 4, i * _BT + s))
    return (x^, y^, lens^)


def run_track_baseline[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    var name = _cell_name(which)
    var cell = _cell(which)
    var H = 16
    var net = Net(cell, _BD, H, 1, 2)
    var p = _params(net, 0x5E65 + which)
    var train = _tracks(_TRAIN, 0x5E70)
    var test = _tracks(_TEST, 0x5E80)
    var epochs = 12
    var batch = 32
    var steps = epochs * ((_TRAIN + batch - 1) // batch)
    var losses = List[Float32](length=steps, fill=0.0)
    var rates = List[Float32](length=steps, fill=Float32(2e-2))
    var ints: List[Int] = [cell, _BD, H, 1, 2, 1, _TRAIN, _BT, epochs, batch, OPT_ADAM, 0, 1, 4242, 0]
    var floats: List[Float64] = [0.9, 0.999, 1e-8, 0.0, 0.0, 0.0]
    _ = rnn_fit_impl(ex, [Int(train[0].unsafe_ptr()), Int(train[1].unsafe_ptr()), Int(p.unsafe_ptr()),
                          Int(losses.unsafe_ptr()), Int(rates.unsafe_ptr()), Int(train[2].unsafe_ptr())], ints, floats)
    var per = (_TRAIN + batch - 1) // batch
    var first = 0.0
    var last = 0.0
    for k in range(per):  # small-loop(steps of one epoch)
        first += Float64(losses[k])
        last += Float64(losses[steps - per + k])
    first /= Float64(per)
    last /= Float64(per)
    if not (last < first):
        _fail(name + " tracks: the loss did not fall (" + String(first) + " -> " + String(last) + ")")
    var out = List[Float32](length=_TEST * 2, fill=0.0)
    _ = rnn_predict_impl(ex, [Int(test[0].unsafe_ptr()), Int(p.unsafe_ptr()), Int(out.unsafe_ptr()),
                              Int(out.unsafe_ptr())],
                         [cell, _BD, H, 1, 2, 1, _TEST, _BT, 4096, 0, Int(test[2].unsafe_ptr())])
    var right = 0
    for i in range(_TEST):  # small-loop(check accuracy)
        var pred = 1 if out[2 * i + 1] > out[2 * i] else 0
        if pred == Int(test[1][i]):
            right += 1
    var acc = Float64(right) / Float64(_TEST)
    print("SEQ_LENGTHS baseline", name, "tracks: loss (first epoch mean)", first, "-> (last)", last,
          "held-out accuracy", acc)
    if acc < 0.9:
        _fail(name + " tracks: held-out accuracy " + String(acc) + " is below 0.9")
    var words = p^
    words.extend(out^)
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("SEQ_LENGTHS baseline", name, "digest", digest)
    _ = train^
    _ = test^
    _ = losses^
    _ = rates^
    return words^


def _run_unit[E: Exec](mut ex: E, unit: Int) raises -> List[Float32]:
    # units: 0-1 proof (a), 2-3 proof (c), 4-5 baseline (b); gru then lstm
    var which = unit % 2
    if unit < 2:
        return run_alone_case(ex, which)
    if unit < 4:
        return run_padding_case(ex, which)
    return run_track_baseline(ex, which)


# ---------------------------------------------------------------- tests
def test_sequence_lengths() raises:
    # The six units are independent: on the host column each runs on its own HostExec across host
    # cores; the device column then runs them in order on the one device and requires the host words.
    comptime U = 6
    var failed = List[Int](length=U, fill=0)
    var host_words = List[List[Float32]](length=U, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = _run_unit(host, unit)
        except error:
            print("FAIL SEQ_LENGTHS unit", unit, error)
            fp[unit] = 1
    host_parallelize(task, U)
    for u in range(U):  # small-loop(U: failure flags)
        if failed[u] != 0:
            _fail("unit " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for unit in range(U):  # driver-loop: over the units on the one device
            var dev = DeviceExec()
            var dw = _run_unit(dev, unit)
            _same("unit " + String(unit) + " host vs device", host_words[unit], dw)
    _ = failed^
    _ = host_words^
    print("PASS SEQ_LENGTHS 6")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
