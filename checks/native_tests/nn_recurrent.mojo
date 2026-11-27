# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The recurrent layer's options (lane nn-recurrent, 2026-11-23): reverse, final state, stacked, bidirectional.

Units (each independent: the CPU column runs them across host cores on HostExec; the device column runs each on
DeviceExec and requires the host words):
  0-5  the layer (autograd/model_blocks.mojo block_recurrent; autograd/nn_recurrent.mojo) for the tanh RNN, GRU
       and LSTM cells, B 3, T 5, F 2, H 3, with lengths 5 3 1 and without: (a) reverse=True, every step's state;
       (b) reverse=True, final state; (c) the forward direction, final state. Each forward against a float64
       oracle written here from PyTorch's cell equations (the reverse direction from each row's own last valid
       step), and every gradient (the input, weight_ih, weight_hh, bias_ih, bias_hh) against central differences
       of the oracle's loss sum(seed * y), every element probed (tolerance 2e-4 relative);
  6-7  GRU and LSTM: a two-layer bidirectional module (the description the pack writes for
       `gru(4, num_layers=2, bidirectional=True, output="final")`), each of six rows (lengths 9 6 3 1 7 4 of T 9)
       padded with lengths and fitted alone for one SGD step gives the parameter words of the row unpadded
       (T = its length, no lengths), and the padded batch's predictions are, row by row, the unpadded rows'
       words: both directions start at each row's own last valid step;
  8    the baseline: a two-layer bidirectional GRU (hidden 12, dropout 0.1 between the layers, the final states
       into a linear head) on variable-length tracks (lengths 4 .. 16 of T 16, noise of scale 5 in the pads)
       reaches held-out accuracy 0.9 with a falling loss and finite parameters and predictions;
  9    refusals: reverse or final outside {0, 1};
  10   projected LSTM feedback and its projection gradient against a scalar float64 oracle.
"""
from std.math import tanh, exp, isfinite
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, CELL_GRU, CELL_LSTM, CELL_RNN_TANH, gates_of
from autograd.tape import Tape, Tensor
from autograd.rankn_api import rn_shape
from autograd.projected_lstm import projected_lstm
from autograd.model_blocks import ModelRules, block_recurrent
from autograd.model import LOSS_CROSS_ENTROPY, LOSS_MSE, OUT_ARGMAX, OUT_RAW, TARGET_CODES, TARGET_DENSE
from checks.native_tests.sequence_port_common import normal, u01
from checks.native_tests.training_sequential_common import SeqModel, codes_as_words, L_DROPOUT, L_FLATTEN, L_LINEAR
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize

comptime L_BRANCH = 19
comptime L_CONCAT = 20
comptime L_RECURRENT = 41
comptime UNUSED = -2


def _fail(message: String) raises:
    print("FAIL NN_RECURRENT", message)
    raise Error(message)


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ (" + String(len(a)) + " vs " + String(len(b)) + ")")
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _close(name: String, got: List[Float32], want: List[Float64], tol: Float64) raises:
    if len(got) != len(want):
        _fail(name + ": " + String(len(got)) + " values, " + String(len(want)) + " expected")
    for i in range(len(got)):  # small-loop(len(got): check comparison)
        if abs(Float64(got[i]) - want[i]) > tol * (1.0 + abs(want[i])):
            _fail(name + ": index " + String(i) + " got " + String(got[i]) + " want " + String(want[i]))


def _finite(name: String, v: List[Float32]) raises:
    for i in range(len(v)):  # small-loop(len(v): check values)
        if not isfinite(v[i]):
            _fail(name + ": a value is not finite at " + String(i))


def _mix(mut digest: UInt64, values: List[Float32]):
    for i in range(len(values)):  # small-loop(len(values): check digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)


def _upload[E: Exec](mut ex: E, values: List[Float32]) raises -> FP:
    var p = ex.alloc(len(values))
    ex.upload(p, FP(unsafe_from_address=Int(values.unsafe_ptr())), len(values))
    return p


# ---------------------------------------------------------------- the layer against a float64 oracle
comptime _B = 3
comptime _T = 5
comptime _F = 2
comptime _H = 3
comptime _FD = Float64(1e-4)


def _cell_of(which: Int) -> Int:
    if which == 0:
        return CELL_RNN_TANH
    if which == 1:
        return CELL_GRU
    return CELL_LSTM


def _cell_name(which: Int) -> String:
    if which == 0:
        return "rnn"
    if which == 1:
        return "gru"
    return "lstm"


def _sig(v: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-v))


def _step(which: Int, vals: List[List[Float64]], x: List[Float64], h: List[Float64], c: List[Float64]
          ) -> Tuple[List[Float64], List[Float64]]:
    """One step of PyTorch's cell in float64: (h', c'). vals: x (unused here), weight_ih (G H, F),
    weight_hh (G H, H), bias_ih, bias_hh; gate order rnn [h], gru [r z n], lstm [i f g o]."""
    var G = 1 if which == 0 else (3 if which == 1 else 4)
    var gi = List[Float64](length=G * _H, fill=0)
    var gh = List[Float64](length=G * _H, fill=0)
    for r in range(G * _H):  # small-loop(oracle)
        var a = vals[3][r]
        for k in range(_F):  # small-loop(oracle)
            a += vals[1][r * _F + k] * x[k]
        gi[r] = a
        var bsum = vals[4][r]
        for k in range(_H):  # small-loop(oracle)
            bsum += vals[2][r * _H + k] * h[k]
        gh[r] = bsum
    var hn = List[Float64](length=_H, fill=0)
    var cn = c.copy()
    for j in range(_H):  # small-loop(oracle)
        if which == 0:
            hn[j] = tanh(gi[j] + gh[j])
        elif which == 1:
            var rg = _sig(gi[j] + gh[j])
            var zg = _sig(gi[_H + j] + gh[_H + j])
            var ng = tanh(gi[2 * _H + j] + rg * gh[2 * _H + j])
            hn[j] = (1.0 - zg) * ng + zg * h[j]
        else:
            var ig = _sig(gi[j] + gh[j])
            var fg = _sig(gi[_H + j] + gh[_H + j])
            var gg = tanh(gi[2 * _H + j] + gh[2 * _H + j])
            var og = _sig(gi[3 * _H + j] + gh[3 * _H + j])
            cn[j] = fg * c[j] + ig * gg
            hn[j] = og * tanh(cn[j])
    return (hn^, cn^)


def _oracle(which: Int, vals: List[List[Float64]], lens: List[Int], reverse: Bool, final: Bool) -> List[Float64]:
    """(B, T, H), or (B, 1, H) when final, in float64: each row over its own valid steps (forward 0 .. len - 1,
    reverse len - 1 .. 0); past its length the forward output repeats its final state and the reverse output
    is 0 (PyTorch's padding)."""
    var out = List[Float64]()
    for b in range(_B):  # small-loop(oracle)
        var n = lens[b]
        var h = List[Float64](length=_H, fill=0)
        var c = List[Float64](length=_H, fill=0)
        var states = List[List[Float64]](length=_T, fill=List[Float64]())
        for s in range(n):  # small-loop(oracle)
            var t = n - 1 - s if reverse else s
            var x = List[Float64](length=_F, fill=0)
            for k in range(_F):  # small-loop(oracle)
                x[k] = vals[0][(b * _T + t) * _F + k]
            var nx = _step(which, vals, x, h, c)
            h = nx[0].copy()
            c = nx[1].copy()
            states[t] = h.copy()
        if final:
            out.extend(h.copy())
            continue
        for t in range(_T):  # small-loop(oracle)
            if t < n:
                out.extend(states[t].copy())
            elif reverse:
                out.extend(List[Float64](length=_H, fill=0))
            else:
                out.extend(h.copy())
    return out^


def _counts(which: Int) -> List[Int]:
    var G = 1 if which == 0 else (3 if which == 1 else 4)
    var out: List[Int] = [_B * _T * _F, G * _H * _F, G * _H * _H, G * _H, G * _H]
    return out^


def _values(which: Int) -> List[List[Float32]]:
    var counts = _counts(which)
    var out = List[List[Float32]]()
    for a in range(len(counts)):  # small-loop(5 inputs)
        var v = List[Float32](length=counts[a], fill=0)
        for i in range(counts[a]):  # small-loop(check data)
            v[i] = Float32(0.5 * normal(0x7E00 + 16 * which + a, i))
        out.append(v^)
    return out^


def _wide(vals: List[List[Float32]]) -> List[List[Float64]]:
    var out = List[List[Float64]]()
    for a in range(len(vals)):  # small-loop(5 inputs)
        var v = List[Float64](length=len(vals[a]), fill=0)
        for i in range(len(vals[a])):  # small-loop(check data)
            v[i] = Float64(vals[a][i])
        out.append(v^)
    return out^


def _seed(n: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(check data)
        v[i] = Float32(normal(0x7E80, i))
    return v^


def _oracle_loss(which: Int, vals: List[List[Float64]], lens: List[Int], reverse: Bool, final: Bool,
                 seed: List[Float32]) -> Float64:
    var y = _oracle(which, vals, lens, reverse, final)
    var s = Float64(0)
    for i in range(len(y)):  # small-loop(check loss)
        s += Float64(seed[i]) * y[i]
    return s


def run_layer_case[E: Exec](mut ex: E, which: Int, masked: Bool) raises -> List[Float32]:
    var name = _cell_name(which) + (" lengths" if masked else "")
    var lens: List[Int] = [_T, _T, _T]
    if masked:
        lens = [5, 3, 1]
    var lw: List[Float32] = [Float32(lens[0]), Float32(lens[1]), Float32(lens[2])]
    var lp = _upload(ex, lw)
    var vals = _values(which)
    var wide = _wide(vals)
    var cell = _cell_of(which)
    var gh = gates_of(cell) * _H
    var words = List[Float32]()
    var forms: List[Int] = [0, 1, 2]  # 0 reverse sequence, 1 reverse final, 2 forward final
    for form in forms:  # small-loop(3 forms)
        var reverse = form != 2
        var final = form != 0
        var what = name + (" reverse" if reverse else " forward") + (" final" if final else " sequence")
        var tape = Tape()
        var xd: List[Int] = [_B, _T, _F]
        var x = tape.tensor_nd(ex, vals[0], rn_shape(xd, "x"), True)
        var w_ih = tape.tensor(ex, vals[1], gh, _F, True)
        var w_hh = tape.tensor(ex, vals[2], gh, _H, True)
        var b_ih = tape.tensor(ex, vals[3], 1, gh, True)
        var b_hh = tape.tensor(ex, vals[4], 1, gh, True)
        var lengths: Optional[FP] = None
        if masked:
            lengths = Optional[FP](lp)
        var y = block_recurrent(tape, ex, x, w_ih, w_hh, b_ih, b_hh, _B, _T, _F, _H, cell, lengths, reverse, final)
        tape.materialize(ex, y)
        var yv = tape.values(ex, y)
        _close(what + " forward", yv, _oracle(which, wide, lens, reverse, final), Float64(1e-5))
        var seed = _seed(len(yv))
        var st = tape.tensor(ex, seed, tape.rows(y), tape.cols(y))
        tape.backward_with_rules(ex, y, st, ModelRules())
        words.extend(yv.copy())
        var ins: List[Tensor] = [x, w_ih, w_hh, b_ih, b_hh]
        for a in range(len(ins)):  # small-loop(5 inputs)
            var g = tape.gradients(ex, ins[a])
            var fd = List[Float64](length=len(g), fill=0)
            for i in range(len(g)):  # small-loop(len(g): every element probed)
                var up = wide.copy()
                up[a][i] = wide[a][i] + _FD
                var down = wide.copy()
                down[a][i] = wide[a][i] - _FD
                fd[i] = (_oracle_loss(which, up, lens, reverse, final, seed)
                         - _oracle_loss(which, down, lens, reverse, final, seed)) / (2.0 * _FD)
            _close(what + " gradient " + String(a), g, fd, Float64(2e-4))
            words.extend(g^)
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_RECURRENT layer", name, "reverse / final: forward vs float64, every gradient vs central differences; digest",
          digest)
    _ = lw^
    return words^


# ---------------------------------------------------------------- the bidirectional stack, padded vs alone
comptime _ST = 9
comptime _SF = 2
comptime _SH = 4
comptime _O = 2


def _stack(mut m: SeqModel, cell_code: Int, hidden: Int, layers: Int, dropout: Float64):
    """The layers the pack writes for (gru|lstm)(hidden, num_layers=layers, bidirectional=True,
    dropout=dropout, output='final') on an empty model, then flatten and linear(_O)."""
    var entry = -1
    for l in range(layers):  # small-loop(layers)
        var top = l == layers - 1
        var fin = 1 if top else 0
        var fwd = len(m.codes)
        m.layer(L_RECURRENT, hidden, cell_code, 0, fin)
        m.layer(L_BRANCH, entry)
        var rev = len(m.codes)
        m.layer(L_RECURRENT, hidden, cell_code, 1, fin)
        m.layer(L_CONCAT, fwd, rev, UNUSED, UNUSED, UNUSED, UNUSED, 0.0 if top else 1.0)
        if not top and dropout > 0.0:
            m.layer(L_DROPOUT, 0, 0, 0, 0, 0, 0, dropout)
        entry = len(m.codes) - 1
    m.layer(L_FLATTEN)
    m.layer(L_LINEAR, _O, 1)


def _model(cell_code: Int, T: Int) -> SeqModel:
    var m = SeqModel()
    m.sample = [T, _SF]
    _stack(m, cell_code, _SH, 2, 0.0)
    return m^


def _row_lengths() -> List[Int]:
    var lens: List[Int] = [9, 6, 3, 1, 7, 4]
    return lens^


def _rows(rows: List[Int], lens: List[Int], T: Int) -> List[Float32]:
    """(B, T, F): row r's first lens[r] steps are its values, the rest noise of scale 1e3."""
    var x = List[Float32](length=len(rows) * T * _SF, fill=0)
    for b in range(len(rows)):  # small-loop(B: check data)
        for s in range(T):  # small-loop(T: check data)
            for d in range(_SF):  # small-loop(F: check data)
                var at = (b * T + s) * _SF + d
                x[at] = Float32(normal(0x7E90, (rows[b] * _ST + s) * _SF + d)) if s < lens[b] else Float32(
                    1000.0 * normal(0x7E91, at))
    return x^


def _targets(rows: List[Int]) -> List[Float32]:
    var y = List[Float32](length=len(rows) * _O, fill=0)
    for b in range(len(rows)):  # small-loop(B: check data)
        for o in range(_O):  # small-loop(O: check data)
            y[b * _O + o] = Float32(normal(0x7E92, rows[b] * _O + o))
    return y^


def _as_words(lens: List[Int]) -> List[Float32]:
    var out = List[Float32](length=len(lens), fill=0)
    for i in range(len(lens)):  # small-loop(check data)
        out[i] = Float32(lens[i])
    return out^


def run_alone_case[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    var cell_code = 2 if which == 0 else 1
    var name = "gru" if which == 0 else "lstm"
    var lens = _row_lengths()
    var B = len(lens)
    var words = List[Float32]()
    var fitted = List[List[Float32]]()
    for r in range(B):  # driver-loop: over the six rows (one fit each)
        var one: List[Int] = [r]
        var ln: List[Int] = [lens[r]]
        var y = _targets(one)
        var padded = _model(cell_code, _ST)
        padded.compile(LOSS_MSE, "sgd", 0.1)
        padded.lengths = _as_words(ln)
        _ = padded.fit(ex, _rows(one, ln, _ST), 1, _ST * _SF, y, _O, TARGET_DENSE, 1, 1, 0, 7)
        var alone = _model(cell_code, lens[r])
        alone.compile(LOSS_MSE, "sgd", 0.1)
        _ = alone.fit(ex, _rows(one, ln, lens[r]), 1, lens[r] * _SF, y, _O, TARGET_DENSE, 1, 1, 0, 7)
        _same(name + " bidirectional row " + String(r) + " (length " + String(lens[r])
              + "): one SGD step, padded with lengths vs alone", padded.params, alone.params)
        fitted.append(padded.params.copy())
        words.extend(padded.params.copy())
    var all_rows = List[Int]()
    for r in range(B):  # small-loop(B)
        all_rows.append(r)
    var batch = _model(cell_code, _ST)
    batch.params = fitted[0].copy()
    batch.predict_lengths = _as_words(lens)
    var pb = batch.predict(ex, _rows(all_rows, lens, _ST), B, _ST * _SF, OUT_RAW)
    for r in range(B):  # driver-loop: over the six rows
        var one: List[Int] = [r]
        var ln: List[Int] = [lens[r]]
        var alone = _model(cell_code, lens[r])
        alone.params = fitted[0].copy()
        var pa = alone.predict(ex, _rows(one, ln, lens[r]), 1, lens[r] * _SF, OUT_RAW)
        var got = List[Float32]()
        for o in range(_O):  # small-loop(O)
            got.append(pb[r * _O + o])
        _same(name + " bidirectional batch row " + String(r) + " prediction vs the row alone", got, pa)
    words.extend(pb^)
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_RECURRENT", name, "two-layer bidirectional, final state: 6 rows, one SGD step padded == unpadded "
          + "(every parameter word); padded batch predictions == each row alone; digest", digest)
    return words^


# ---------------------------------------------------------------- baseline: bidirectional GRU on tracks
comptime _BT = 16
comptime _BD = 2
comptime _TRAIN = 256
comptime _TEST = 128


def _tracks(n: Int, salt: Int) -> Tuple[List[Float32], List[Int], List[Float32]]:
    """(X, classes, lengths): lengths 4 .. 16; channel 0 drifts up (class 1) or down (class 0) over the
    valid steps with noise, channel 1 is noise; padded steps hold noise of scale 5."""
    var x = List[Float32](length=n * _BT * _BD, fill=0.0)
    var y = List[Int](length=n, fill=0)
    var lens = List[Float32](length=n, fill=0.0)
    for i in range(n):  # small-loop(n: check data)
        var cls = 1 if u01(salt, i, 7) < 0.5 else 0
        var length = 4 + Int(u01(salt, i, 8) * 13.0)
        if length > _BT:
            length = _BT
        y[i] = cls
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


def run_track_baseline[E: Exec](mut ex: E) raises -> List[Float32]:
    var train = _tracks(_TRAIN, 0x7EA0)
    var test = _tracks(_TEST, 0x7EB0)
    var m = SeqModel()
    m.sample = [_BT, _BD]
    _stack(m, 2, 12, 2, 0.1)
    m.compile(LOSS_CROSS_ENTROPY, "adam", 0.02)
    m.lengths = train[2].copy()
    _ = m.fit(ex, train[0], _TRAIN, _BT * _BD, codes_as_words(train[1]), 1, TARGET_CODES, 12, 32, 1, 4242)
    var curve = m.losses()
    _finite("bigru tracks loss curve", curve)
    if not (curve[len(curve) - 1] < curve[0]):
        _fail("bigru tracks: the loss did not fall (" + String(curve[0]) + " -> " + String(curve[len(curve) - 1]) + ")")
    _finite("bigru tracks parameters", m.params)
    m.predict_lengths = test[2].copy()
    var raw = m.predict(ex, test[0], _TEST, _BT * _BD, OUT_RAW)
    _finite("bigru tracks predictions", raw)
    var pred = m.predict(ex, test[0], _TEST, _BT * _BD, OUT_ARGMAX)
    var right = 0
    for i in range(_TEST):  # small-loop(check accuracy)
        if Int(pred[i]) == test[1][i]:
            right += 1
    var acc = Float64(right) / Float64(_TEST)
    print("NN_RECURRENT baseline bidirectional 2-layer gru (hidden 12, dropout 0.1) tracks: loss", curve[0], "->",
          curve[len(curve) - 1], "held-out accuracy", acc)
    if acc < 0.9:
        _fail("bigru tracks: held-out accuracy " + String(acc) + " is below 0.9")
    var words = m.params.copy()
    words.extend(raw^)
    var digest = UInt64(1469598103934665603)
    _mix(digest, words)
    print("NN_RECURRENT baseline digest", digest)
    return words^


def run_refusals[E: Exec](mut ex: E) raises -> List[Float32]:
    var bad: List[Int] = [2, -1]
    for k in range(len(bad)):  # small-loop(2 values)
        for slot in range(2):  # small-loop(reverse, final)
            var m = SeqModel()
            m.sample = [4, 2]
            if slot == 0:
                m.layer(L_RECURRENT, 3, 2, bad[k], 0)
            else:
                m.layer(L_RECURRENT, 3, 2, 0, bad[k])
            m.layer(L_FLATTEN)
            m.layer(L_LINEAR, 1, 1)
            m.compile(LOSS_MSE, "sgd", 0.1)
            var x = List[Float32](length=8, fill=0.5)
            var y: List[Float32] = [Float32(1)]
            var refused = False
            try:
                _ = m.fit(ex, x, 1, 8, y, 1, TARGET_DENSE, 1, 1, 0, 1)
            except e:
                refused = String(e).find("reverse and final must be 0 or 1") >= 0
            if not refused:
                _fail("refusal: reverse / final " + String(bad[k]) + " was not refused in words")
    print("NN_RECURRENT refusals: reverse and final outside {0, 1}")
    var words: List[Float32] = [Float32(1)]
    return words^


def _projected_oracle(w: Float64) -> List[Float64]:
    # Three gates are sigmoid(0) = 1/2. Only the first cell has a candidate.
    var h = Float64(0.25)
    var c = Float64(0.5)
    var words = List[Float64]()
    for _ in range(2):  # small-loop(2 recurrent steps)
        c = 0.5 * c + 0.5 * tanh(0.25 + h)
        h = w * 0.5 * tanh(c)
        words.append(h)
    words.append(c)
    return words^


def run_projected_feedback[E: Exec](mut ex: E) raises -> List[Float32]:
    var tape = Tape()
    var x = tape.tensor_nd(ex, [0, 0], rn_shape([2, 1, 1], "x"))
    var wi = tape.tensor(ex, List[Float32](length=8, fill=0), 8, 1)
    var wh = tape.tensor(ex, [0, 0, 0, 0, 1, 0, 0, 0], 8, 1)
    var bi = tape.tensor(ex, [0, 0, 0, 0, 0.25, 0, 0, 0], 1, 8)
    var bh = tape.tensor(ex, List[Float32](length=8, fill=0), 1, 8)
    var hr = tape.tensor(ex, [2, -1], 1, 2, True)
    var h0 = tape.tensor(ex, [0.25], 1, 1)
    var c0 = tape.tensor(ex, [0.5, 0], 1, 2)
    var out = projected_lstm(tape, ex, x, wi, wh, bi, bh, hr, False, h0, c0)
    var hs = tape.values(ex, out[0])
    var cs = tape.values(ex, out[2])
    var want = _projected_oracle(2)
    _close("projected LSTM feedback", hs, [want[0], want[1]], Float64(1e-5))
    _close("projected LSTM cell", cs, [want[2], 0], Float64(1e-5))
    var seed = tape.tensor(ex, [1, 1], 2, 1)
    tape.backward_with_rules(ex, out[0], seed, ModelRules())
    var up = _projected_oracle(2 + _FD)
    var down = _projected_oracle(2 - _FD)
    var dw = (up[0] + up[1] - down[0] - down[1]) / (2 * _FD)
    var grads = tape.gradients(ex, hr)
    _close("projected LSTM feedback gradient", grads, [dw, 0], Float64(2e-4))
    hs.extend(cs^)
    hs.extend(grads^)
    return hs^


def _run_unit[E: Exec](mut ex: E, unit: Int) raises -> List[Float32]:
    # units: 0-5 the layer (rnn, gru, lstm; without and with lengths), 6-7 padded vs alone (gru, lstm),
    # 8 the baseline, 9 refusals, 10 projected feedback
    if unit < 6:
        return run_layer_case(ex, unit // 2, unit % 2 == 1)
    if unit < 8:
        return run_alone_case(ex, unit - 6)
    if unit == 8:
        return run_track_baseline(ex)
    if unit == 9:
        return run_refusals(ex)
    return run_projected_feedback(ex)


# ---------------------------------------------------------------- tests
def test_nn_recurrent() raises:
    comptime U = 11
    var failed = List[Int](length=U, fill=0)
    var host_words = List[List[Float32]](length=U, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = _run_unit(host, unit)
        except error:
            print("FAIL NN_RECURRENT unit", unit, error)
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
    print("PASS NN_RECURRENT 11")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
