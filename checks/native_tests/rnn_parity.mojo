# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The recurrent estimators' options (lane rnn-parity, Andrew 2026-11-25; checks by lane rnn-parity-D).

The Mojo entries are called directly: sequence/ops_impl.mojo rnn_fit_impl / rnn_predict_impl (their plain forms
and their options forms: 28 integer, 9 float and 16 address words for the fit, 20 integer words for predict),
the trainer's step (autograd/rules_recurrent.mojo AutogradRnnStep forward_options / weighted_backward, with
sequence/recurrent.mojo Work and RnnFitSetup) and autograd/epoch_monitor.mojo EpochMonitor.

Units (each independent: the CPU column runs them across host cores on HostExec; the device column runs each on
DeviceExec and requires the host words):
  0-1  (a) the no-cost promise, GRU and LSTM (two layers, lengths, Adam, batches of 4 of 10 rows shuffled, two
       epochs): the options form with every option off is the plain form's words; the options LOOP (forced by
       the resume words of a fresh optimizer state, what partial_fit's first call sends) is the plain loop's
       words, parameters and losses, and returns the step count; without shuffling, two one-epoch calls carrying
       the optimizer state are one two-epoch call, word for word;
  2-3  (b) GRU and LSTM, two bidirectional layers, lengths 6 4 1 5 3 of T 6: the final state of every layer and
       direction (PyTorch's h_n / c_n: forward at each row's last valid step, reverse at time 0), the head on
       the top layer's two final states and the valid steps of the top layer's output, from zero states and from
       a given h_0 / c_0, against a brute-force float64 oracle of both directions written here from PyTorch's
       cell equations (tolerance 1e-4 relative);
  4-5  (c) the weighted per-step loss (output='sequence', lengths, a sample weight per step; the classifier also
       class weights): the loss AutogradRnnStep.weighted_backward records equals a direct float64 loop over the
       head's rows, sum w l / sum w with w = sample weight x class weight, 0 at a padded step; and its gradient
       equals the plain backward seeded with that loop's d loss / d head output (LSTM regression, GRU
       classification);
  6-7  (d) h_0: GRU and LSTM, two layers, T 9: the run of steps 4 .. 8 from h_0 / c_0 = the state the run of steps
       0 .. 3 returns gives the whole run's words for those steps (every step's head and the top layer's output)
       and its final state;
  8-9  (e) streaming: rnn_predict_options over T = 6 steps (output='sequence', softmax head) equals six calls of
       one step carrying (h, c), every step's probabilities and the final states, word for word (GRU one layer,
       LSTM two layers);
  10   (g) clip_grad_norm: one SGD step (lr 1, so the update is the clipped gradient) with max_norm a quarter of
       the gradient's norm moves the parameters by a vector of norm <= max_norm, parallel to the unclipped step;
       with max_norm four times the norm the step is the plain step's words (coefficient exactly 1);
  15   (f) EpochMonitor's decisions (host scalars; no device words): the improvement, best epoch, stall and stop
       of scikit-learn's rule over a fixed loss sequence, patience 0 never stops, a loss that is not finite is
       refused.
  11-14 (h) streaming at every input width (lane rnn-step-bits, Andrew 2026-11-25): an LSTM (hidden 32, one
       layer, output='sequence') over 4 sequences of 64 steps of 8, 64, 65 and 300 inputs: 64 calls of one step
       carrying (h, c) give the one 64-step call's words, every step's probabilities and the final states. The
       shape is one the split-K rule (sequence/gemm_split.mojo gemm_split_shape) separates: the whole call's input
       projection 256 x 128 is not a split shape, a one-step call's 4 x 128 is (above 64 inputs); the unit fails if
       the rule stops separating them, so it keeps testing what it claims. The recurrent projections take the split
       from K alone at every row count (autograd/rules_recurrent.mojo rnn_linear_width_split), so 65 and 300 agree
       too; under
       MOJOLEARN_RNN_STEP_BITS_SABOTAGE_HOST (or _DEVICE) the host (device) column must fail them.
The streaming unit (e) and the h_0 unit (d) keep small widths (inputs 3 or 4, hidden 5 or 6); the widths that
used to separate the orders are unit (h)'s.
"""
from std.math import tanh, exp, log, sqrt
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args, OP_SEQ_IOTA, OPT_ADAM, OPT_SGD, CELL_GRU, CELL_LSTM, CELL_RNN_TANH, gates_of
from sequence.ops_impl import rnn_fit_impl, rnn_predict_impl
from sequence.recurrent import Net, Work, RnnOptions, RnnFitSetup, TASK_MSE, TASK_CE, gather_rows, gather_seq
from autograd.rules_recurrent import AutogradRnnStep
from sequence.gemm_split import gemm_split_shape
from autograd.epoch_monitor import EpochMonitor
from checks.native_tests.sequence_port_common import normal, u01
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD
from core.host_parallel import host_parallelize


def _fail(message: String) raises:
    print("FAIL RNN_PARITY", message)
    raise Error(message)


def _fp(values: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(values.unsafe_ptr()))


def _addr(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr()) if len(values) > 0 else 0


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ (" + String(len(a)) + " vs " + String(len(b)) + ")")
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _close(name: String, got: Float32, want: Float64, tol: Float64) raises:
    if not (abs(Float64(got) - want) <= tol * (1.0 + abs(want))):
        _fail(name + ": " + String(got) + " vs float64 " + String(want))


def _slice(values: List[Float32], start: Int, count: Int) -> List[Float32]:
    var out = List[Float32](length=count, fill=0.0)
    for i in range(count):  # small-loop(count: check copy)
        out[i] = values[start + i]
    return out^


def _cell_of(which: Int) -> Int:
    return CELL_GRU if which == 0 else CELL_LSTM


def _cell_name(which: Int) -> String:
    return "gru" if which == 0 else "lstm"


def _values(n: Int, salt: Int, scale: Float64) -> List[Float32]:
    var v = List[Float32](length=n, fill=0.0)
    for i in range(n):  # small-loop(n: check data)
        v[i] = Float32(scale * normal(salt, i))
    return v^


def _params(n: Int, salt: Int, bound: Float64) -> List[Float32]:
    var p = List[Float32](length=n, fill=0.0)
    for i in range(n):  # small-loop(n: check parameters)
        p[i] = Float32((2.0 * u01(salt, i) - 1.0) * bound)
    return p^


def _lengths(values: List[Int]) -> List[Float32]:
    var out = List[Float32](length=len(values), fill=0.0)
    for i in range(len(values)):  # small-loop(rows: check lengths)
        out[i] = Float32(values[i])
    return out^


def _steps(x: List[Float32], N: Int, T: Int, D: Int, s0: Int, count: Int) -> List[Float32]:
    """Steps s0 .. s0 + count - 1 of every row of x [N, T, D], as [N, count, D]."""
    var out = List[Float32](length=N * count * D, fill=0.0)
    for b in range(N):  # small-loop(N: check data)
        for s in range(count):  # small-loop(count: check data)
            for k in range(D):  # small-loop(D: check data)
                out[(b * count + s) * D + k] = x[(b * T + s0 + s) * D + k]
    return out^


# ---------------------------------------------------------------- the entries, plain and options forms
def _fit[E: Exec](
    mut ex: E, shape: List[Int], N: Int, T: Int, epochs: Int, bs: Int, shuffle: Int, kind: Int,
    x: List[Float32], y: List[Float32], p0: List[Float32], lens: List[Float32], rate: Float32,
    opt_ints: List[Int], opt_floats: List[Float64], opt_addrs: List[Int],
) raises -> List[Float32]:
    """rnn_fit_impl; shape = [cell, D, H, L, O, task]. `opt_ints` empty: the plain form (15 integer, 6 float
    words, 6 addresses); else the options form (13 option integers, 3 option floats, a zero loop-words address,
    then the 9 option addresses). Returns the trained parameters followed by the per-step losses."""
    var per = (N + bs - 1) // bs
    var steps = epochs * per
    var p = p0.copy()
    var losses = List[Float32](length=steps, fill=0.0)
    var rates = List[Float32](length=steps, fill=rate)
    var ints: List[Int] = [shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], N, T, epochs, bs, kind, 0,
                           shuffle, 4141, 0]
    var floats = List[Float64](length=6, fill=0.0)
    if kind == OPT_ADAM:
        floats[0] = 0.9
        floats[1] = 0.999
        floats[2] = 1e-8
    var addrs: List[Int] = [Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(p.unsafe_ptr()),
                            Int(losses.unsafe_ptr()), Int(rates.unsafe_ptr()), _addr(lens)]
    if len(opt_ints) > 0:
        ints.extend(opt_ints.copy())
        floats.extend(opt_floats.copy())
        addrs.append(0)
        addrs.extend(opt_addrs.copy())
    _ = rnn_fit_impl(ex, addrs, ints, floats)
    _ = rates^
    var out = p^
    out.extend(losses^)
    return out^


def _predict[E: Exec](
    mut ex: E, shape: List[Int], dirs: Int, seq_out: Int, N: Int, T: Int, x: List[Float32], p: List[Float32],
    lens: List[Float32], h0: List[Float32], c0: List[Float32],
) raises -> List[List[Float32]]:
    """rnn_predict_impl's options form with want_seq and want_state: [out ([N, O] or [N, T, O]), seq
    ([N, T, dirs H]), h_n, c_n ([L dirs, N, H] each; c_n zeros but for the LSTM)]. h0 / c0 empty: absent."""
    var O = shape[4]
    var H = shape[2]
    var nh = shape[3] * dirs * N * H
    var out = List[Float32](length=N * (T if seq_out == 1 else 1) * O, fill=0.0)
    var seq = List[Float32](length=N * T * dirs * H, fill=0.0)
    var sh = List[Float32](length=nh, fill=0.0)
    var sc = List[Float32](length=nh, fill=0.0)
    var ints: List[Int] = [shape[0], shape[1], H, shape[3], O, shape[5], N, T, 4096, 1, _addr(lens), dirs, seq_out,
                           1 if len(h0) > 0 else 0, 1 if len(c0) > 0 else 0, 1, _addr(h0), _addr(c0),
                           Int(sh.unsafe_ptr()), Int(sc.unsafe_ptr())]
    _ = rnn_predict_impl(ex, [Int(x.unsafe_ptr()), Int(p.unsafe_ptr()), Int(out.unsafe_ptr()),
                              Int(seq.unsafe_ptr())], ints)
    var res = List[List[Float32]]()
    res.append(out^)
    res.append(seq^)
    res.append(sh^)
    res.append(sc^)
    return res^


def _no_ints() -> List[Int]:
    return List[Int]()


def _no_floats() -> List[Float64]:
    return List[Float64]()


def _zero_addrs() -> List[Int]:
    return List[Int](length=9, fill=0)


def _empty() -> List[Float32]:
    return List[Float32]()


# ---------------------------------------------------------------- (a) the no-cost promise
def run_forms_case[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    var name = _cell_name(which)
    var cell = _cell_of(which)
    var N = 10
    var T = 5
    var D = 3
    var H = 6
    var O = 2
    var shape: List[Int] = [cell, D, H, 2, O, TASK_MSE]
    var np_ = Net(cell, D, H, 2, O).n_params()
    var x = _values(N * T * D, 0x7A00 + which, 1.0)
    var y = _values(N * O, 0x7A10 + which, 1.0)
    var lens = _lengths([5, 3, 1, 4, 2, 5, 5, 1, 3, 4])
    var p0 = _params(np_, 0x7A20 + which, 0.4)
    var plain = _fit(ex, shape, N, T, 2, 4, 1, OPT_ADAM, x, y, p0, lens, Float32(1e-2), _no_ints(), _no_floats(),
                     _no_ints())
    var off_ints: List[Int] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var off_floats: List[Float64] = [0.0, 0.0, 0.0]
    var off = _fit(ex, shape, N, T, 2, 4, 1, OPT_ADAM, x, y, p0, lens, Float32(1e-2), off_ints, off_floats,
                   _zero_addrs())
    _same(name + ": options form, every option off, vs plain form", plain, off)
    # the options loop: the resume words of a fresh state (zeros, beta1^0 = beta2^0 = mu = 1, t0 = 0)
    var state = List[Float32](length=3 * np_ + 4, fill=0.0)
    state[3 * np_] = 1.0
    state[3 * np_ + 1] = 1.0
    state[3 * np_ + 2] = 1.0
    var resume_ints: List[Int] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    var resume_addrs = _zero_addrs()
    resume_addrs[8] = Int(state.unsafe_ptr())
    var looped = _fit(ex, shape, N, T, 2, 4, 1, OPT_ADAM, x, y, p0, lens, Float32(1e-2), resume_ints, off_floats,
                      resume_addrs)
    _same(name + ": options loop (fresh resume state) vs plain loop", plain, looped)
    if state[3 * np_ + 3] != Float32(6.0):
        _fail(name + ": the resume state's step count is " + String(state[3 * np_ + 3]) + ", not 6")
    # without shuffling: two one-epoch calls carrying the state equal one two-epoch call
    var whole = _fit(ex, shape, N, T, 2, 4, 0, OPT_ADAM, x, y, p0, lens, Float32(1e-2), _no_ints(), _no_floats(),
                     _no_ints())
    var carried = List[Float32](length=3 * np_ + 4, fill=0.0)
    carried[3 * np_] = 1.0
    carried[3 * np_ + 1] = 1.0
    carried[3 * np_ + 2] = 1.0
    var carry_addrs = _zero_addrs()
    carry_addrs[8] = Int(carried.unsafe_ptr())
    var first = _fit(ex, shape, N, T, 1, 4, 0, OPT_ADAM, x, y, p0, lens, Float32(1e-2), resume_ints, off_floats,
                     carry_addrs)
    var p1 = _slice(first, 0, np_)
    var second = _fit(ex, shape, N, T, 1, 4, 0, OPT_ADAM, x, y, p1, lens, Float32(1e-2), resume_ints, off_floats,
                      carry_addrs)
    var joined = _slice(second, 0, np_)
    joined.extend(_slice(first, np_, 3))
    joined.extend(_slice(second, np_, 3))
    _same(name + ": two resumed one-epoch calls vs one two-epoch call", whole, joined)
    print("RNN_PARITY", name, "(a) options form off, options loop and resumed epochs: the plain words")
    _ = state^
    _ = carried^
    var words = plain^
    words.extend(whole^)
    return words^


# ---------------------------------------------------------------- the float64 oracle
def _sig(v: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-v))


def _cell64(cell: Int, P: List[Float64], wih: Int, whh: Int, bih: Int, bhh: Int, din: Int, H: Int,
            x: List[Float64], h: List[Float64], c: List[Float64]) -> Tuple[List[Float64], List[Float64]]:
    """One step of PyTorch's cell in float64 from the flat layout's offsets: (h', c'). Gate orders: rnn [h],
    gru [r z n] (n = tanh(W_in x + b_in + r (W_hn h + b_hn))), lstm [i f g o]."""
    var G = gates_of(cell)
    var gi = List[Float64](length=G * H, fill=0.0)
    var gh = List[Float64](length=G * H, fill=0.0)
    for r in range(G * H):  # small-loop(oracle)
        var a = P[bih + r]
        for k in range(din):  # small-loop(oracle)
            a += P[wih + r * din + k] * x[k]
        gi[r] = a
        var s = P[bhh + r]
        for k in range(H):  # small-loop(oracle)
            s += P[whh + r * H + k] * h[k]
        gh[r] = s
    var hn = List[Float64](length=H, fill=0.0)
    var cn = c.copy()
    for j in range(H):  # small-loop(oracle)
        if cell == CELL_RNN_TANH:
            hn[j] = tanh(gi[j] + gh[j])
        elif cell == CELL_GRU:
            var rg = _sig(gi[j] + gh[j])
            var zg = _sig(gi[H + j] + gh[H + j])
            var ng = tanh(gi[2 * H + j] + rg * gh[2 * H + j])
            hn[j] = (1.0 - zg) * ng + zg * h[j]
        else:
            var ig = _sig(gi[j] + gh[j])
            var fg = _sig(gi[H + j] + gh[H + j])
            var gg = tanh(gi[2 * H + j] + gh[2 * H + j])
            var og = _sig(gi[3 * H + j] + gh[3 * H + j])
            cn[j] = fg * c[j] + ig * gg
            hn[j] = og * tanh(cn[j])
    return (hn^, cn^)


def _wide(values: List[Float32]) -> List[Float64]:
    var out = List[Float64](length=len(values), fill=0.0)
    for i in range(len(values)):  # small-loop(check data)
        out[i] = Float64(values[i])
    return out^


def _oracle_net(net: Net, P: List[Float64], x: List[Float32], N: Int, T: Int, lens: List[Int],
                h0: List[Float32], c0: List[Float32]) -> List[List[Float64]]:
    """[the top layer's output [N, T, dirs H] at each row's valid steps (0 elsewhere), h_n, c_n
    ([L dirs, N, H])], each row over its own valid steps; the reverse direction from the row's last valid step
    down to 0; every layer-direction from h0 / c0 (empty: zeros)."""
    var H = net.H
    var DH = net.dirs * H
    var slots = net.slots()
    var top = List[Float64](length=N * T * DH, fill=0.0)
    var hn = List[Float64](length=slots * N * H, fill=0.0)
    var cn = List[Float64](length=slots * N * H, fill=0.0)
    for b in range(N):  # small-loop(oracle rows)
        var n = lens[b]
        var din = net.D
        var inp = List[Float64](length=n * din, fill=0.0)
        for s in range(n):  # small-loop(oracle)
            for k in range(din):  # small-loop(oracle)
                inp[s * din + k] = Float64(x[(b * T + s) * net.D + k])
        for l in range(net.L):  # small-loop(oracle layers)
            var outl = List[Float64](length=n * DH, fill=0.0)
            for d in range(net.dirs):  # small-loop(oracle directions)
                var k = l * net.dirs + d
                var h = List[Float64](length=H, fill=0.0)
                var c = List[Float64](length=H, fill=0.0)
                for j in range(H):  # small-loop(oracle)
                    if len(h0) > 0:
                        h[j] = Float64(h0[(k * N + b) * H + j])
                    if len(c0) > 0:
                        c[j] = Float64(c0[(k * N + b) * H + j])
                for s in range(n):  # small-loop(oracle steps)
                    var t = n - 1 - s if d == 1 else s
                    var xt = List[Float64](length=din, fill=0.0)
                    for q in range(din):  # small-loop(oracle)
                        xt[q] = inp[t * din + q]
                    var r = _cell64(net.cell, P, net.w_ih(l, d), net.w_hh(l, d), net.b_ih(l, d), net.b_hh(l, d), din,
                                    H, xt, h, c)
                    h = r[0].copy()
                    c = r[1].copy()
                    for j in range(H):  # small-loop(oracle)
                        outl[t * DH + d * H + j] = h[j]
                for j in range(H):  # small-loop(oracle)
                    hn[(k * N + b) * H + j] = h[j]
                    cn[(k * N + b) * H + j] = c[j]
            inp = outl^
            din = DH
        for t in range(n):  # small-loop(oracle)
            for j in range(DH):  # small-loop(oracle)
                top[(b * T + t) * DH + j] = inp[t * DH + j]
    var res = List[List[Float64]]()
    res.append(top^)
    res.append(hn^)
    res.append(cn^)
    return res^


# ---------------------------------------------------------------- (b) bidirectional against the oracle
def run_bidi_case[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    var name = _cell_name(which) + " bidirectional"
    var cell = _cell_of(which)
    var N = 5
    var T = 6
    var D = 3
    var H = 4
    var L = 2
    var O = 2
    var net = Net(cell, D, H, L, O, 2)
    var np_ = net.n_params()
    var shape: List[Int] = [cell, D, H, L, O, TASK_MSE]
    var lens_i: List[Int] = [6, 4, 1, 5, 3]
    var lens = _lengths(lens_i)
    var x = _values(N * T * D, 0x7B00 + which, 1.0)
    var p = _params(np_, 0x7B10 + which, 0.5)
    var P = _wide(p)
    var nh = net.slots() * N * H
    var words = List[Float32]()
    for form in range(2):  # small-loop(2 forms: zero state, given h_0 / c_0)
        var h0 = _values(nh, 0x7B20 + which, 0.5) if form == 1 else _empty()
        var c0 = _values(nh, 0x7B30 + which, 0.5) if form == 1 and cell == CELL_LSTM else _empty()
        var what = name + (" from h_0 / c_0" if form == 1 else " from zeros")
        var got = _predict(ex, shape, 2, 0, N, T, x, p, lens, h0, c0)
        var want = _oracle_net(net, P, x, N, T, lens_i, h0, c0)
        for i in range(nh):  # small-loop(check h_n, c_n)
            _close(what + " h_n[" + String(i) + "]", got[2][i], want[1][i], 1e-4)
            if cell == CELL_LSTM:
                _close(what + " c_n[" + String(i) + "]", got[3][i], want[2][i], 1e-4)
        var top_f = (L - 1) * 2
        for b in range(N):  # small-loop(check head)
            for o in range(O):  # small-loop(check head)
                var v = P[net.head() + net.O * net.head_in() + o]
                for j in range(H):  # small-loop(check head)
                    v += P[net.head() + o * 2 * H + j] * want[1][(top_f * N + b) * H + j]
                    v += P[net.head() + o * 2 * H + H + j] * want[1][((top_f + 1) * N + b) * H + j]
                _close(what + " head[" + String(b) + ", " + String(o) + "]", got[0][b * O + o], v, 1e-4)
            for t in range(lens_i[b]):  # small-loop(check output)
                for j in range(2 * H):  # small-loop(check output)
                    var at = (b * T + t) * 2 * H + j
                    _close(what + " output[" + String(at) + "]", got[1][at], want[0][at], 1e-4)
        for r in range(4):  # small-loop(4 results)
            words.extend(got[r].copy())
        _ = h0^
        _ = c0^
    print("RNN_PARITY", name, "(b) h_n, c_n, head and valid outputs of both directions within 1e-4 of float64")
    _ = lens^
    return words^


# ---------------------------------------------------------------- (c) the weighted per-step loss
def run_weighted_case[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    """which 0: LSTM regression (O 2, mean squared error per row); 1: GRU classification (O 3, class weights)."""
    var task = TASK_MSE if which == 0 else TASK_CE
    var cell = CELL_LSTM if which == 0 else CELL_GRU
    var name = (String("lstm regression") if which == 0 else String("gru classification")) + " weighted steps"
    var N = 6
    var T = 7
    var D = 3
    var H = 5
    var O = 2 if which == 0 else 3
    var ycols = O if task == TASK_MSE else 1
    var net = Net(cell, D, H, 2, O)
    var np_ = net.n_params()
    var R = N * T
    var lens_i: List[Int] = [7, 4, 1, 6, 2, 5]
    var lens = _lengths(lens_i)
    var x = _values(N * T * D, 0x7C00 + which, 1.0)
    var p = _params(np_, 0x7C10 + which, 0.4)
    var y = List[Float32](length=N * T * ycols, fill=0.0)
    var sw = List[Float32](length=N * T, fill=0.0)
    for i in range(N * T):  # small-loop(check data)
        if task == TASK_MSE:
            for o in range(O):  # small-loop(check data)
                y[i * O + o] = Float32(normal(0x7C20, i * O + o))
            sw[i] = Float32(0.5 + 1.5 * u01(0x7C30, i))
        else:
            var code = Int(u01(0x7C40, i) * 3.0)
            if code > 2:
                code = 2
            y[i] = Float32(code)
            sw[i] = Float32(30.0) if code == 1 else Float32(1.0)
    var cw: List[Float32] = [1.0, 2.0, 0.5]
    var opts = RnnOptions()
    opts.seq_out = True
    opts.has_sw = True
    opts.sw_steps = True
    opts.sw = _fp(sw)
    if task == TASK_CE:
        opts.has_cw = True
        opts.cw = _fp(cw)
    var dX = ex.alloc(N * T * D)
    ex.upload(dX, _fp(x), N * T * D)
    var idx = ex.alloc(N)
    var ia = Args()
    ia.p0 = idx
    ex.launch[OP_SEQ_IOTA](ia, N)
    var dL = ex.alloc(N)
    ex.upload(dL, _fp(lens), N)
    var dY = ex.alloc(N * T * ycols)
    ex.upload(dY, _fp(y), N * T * ycols)
    var P = ex.alloc(np_)
    ex.upload(P, _fp(p), np_)
    var w = Work(ex, net, T, N, True, True, opts)
    var setup = RnnFitSetup(ex, net, task, opts, N, T, np_, P, dY, True, w, T, ycols, N)
    gather_seq(ex, dX, idx, w.xb, T, N, D, 0, Optional[FP](dL))
    gather_rows(ex, dL, idx, w.lens, N, 1, 0)
    # the options loop's step (sequence/recurrent.mojo _rnn_options_step): forward, targets, weights, loss
    var Gr = ex.alloc(np_)
    var step = AutogradRnnStep()
    step.forward_options(ex, net, opts, P, Gr, w.xb, T, N, w, UInt64(0), 0)
    gather_seq(ex, dY, idx, w.yb, T, N, ycols, 0)
    gather_seq(ex, setup.wt, idx, w.wrow, T, N, 1, 0, Optional[FP](dL))
    var dloss = ex.alloc(1)
    step.weighted_backward(ex, net, task, R, w, dloss)
    ex.sync()
    var yhat = List[Float32](length=R * O, fill=0.0)
    var grad = List[Float32](length=np_, fill=0.0)
    var loss = List[Float32](length=1, fill=0.0)
    ex.download(_fp(yhat), w.yhat, R * O)
    ex.download(_fp(grad), Gr, np_)
    ex.download(_fp(loss), dloss, 1)
    _ = x^  # the uploads above read these host words (y and sw are read below)
    _ = p^
    _ = lens^
    # the direct loop over the head's time-major rows r = t N + b
    var num = 0.0
    var den = 0.0
    var dl = List[Float64](length=R * O, fill=0.0)
    var wr = List[Float64](length=R, fill=0.0)
    for t in range(T):  # small-loop(T: check loop)
        for b in range(N):  # small-loop(N: check loop)
            var r = t * N + b
            var i = b * T + t
            if t >= lens_i[b]:
                continue
            var wt = Float64(sw[i])
            var l = 0.0
            if task == TASK_MSE:
                for o in range(O):  # small-loop(O: check loop)
                    var e = Float64(yhat[r * O + o]) - Float64(y[i * O + o])
                    l += e * e / Float64(O)
                    dl[r * O + o] = 2.0 * e / Float64(O)
            else:
                var code = Int(y[i])
                wt *= Float64(cw[code])
                var m = Float64(yhat[r * O])
                for o in range(O):  # small-loop(O: check loop)
                    m = max(m, Float64(yhat[r * O + o]))
                var z = 0.0
                for o in range(O):  # small-loop(O: check loop)
                    z += exp(Float64(yhat[r * O + o]) - m)
                l = m + log(z) - Float64(yhat[r * O + code])
                for o in range(O):  # small-loop(O: check loop)
                    dl[r * O + o] = exp(Float64(yhat[r * O + o]) - m) / z - (1.0 if o == code else 0.0)
            wr[r] = wt
            num += wt * l
            den += wt
    _close(name + " loss vs the direct loop", loss[0], num / den, 1e-5)
    # the gradient: the plain backward seeded with the loop's d loss / d head output
    var seed = List[Float32](length=R * O, fill=0.0)
    for r in range(R):  # small-loop(R: check seed)
        for o in range(O):  # small-loop(O: check seed)
            seed[r * O + o] = Float32(wr[r] / den * dl[r * O + o])
    var Gr2 = ex.alloc(np_)
    var step2 = AutogradRnnStep()
    step2.forward_options(ex, net, opts, P, Gr2, w.xb, T, N, w, UInt64(0), 0)
    ex.upload(w.dy, _fp(seed), R * O)
    step2.backward(ex, net, P, Gr2, w.xb, T, R, w)
    ex.sync()
    var grad2 = List[Float32](length=np_, fill=0.0)
    ex.download(_fp(grad2), Gr2, np_)
    var scale = 0.0
    for i in range(np_):  # small-loop(np: check gradient)
        scale = max(scale, abs(Float64(grad2[i])))
    for i in range(np_):  # small-loop(np: check gradient)
        if abs(Float64(grad[i]) - Float64(grad2[i])) > 1e-4 * scale + 1e-9:
            _fail(name + " gradient[" + String(i) + "]: weighted " + String(grad[i]) + " vs seeded loop "
                  + String(grad2[i]))
    print("RNN_PARITY", name, "(c) loss", loss[0], "vs loop", num / den, "; gradient within 1e-4 of the seeded loop")
    _ = setup^
    _ = step^
    _ = step2^
    _ = sw^
    _ = cw^
    _ = seed^
    var words = yhat^
    words.extend(grad^)
    words.extend(grad2^)
    words.extend(loss^)
    return words^


# ---------------------------------------------------------------- (d) h_0: the tail of a longer run
def run_tail_case[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    var name = _cell_name(which) + " h_0 tail"
    var cell = _cell_of(which)
    var N = 5
    var T = 9
    var t1 = 4
    var D = 3
    var H = 6
    var O = 2
    var shape: List[Int] = [cell, D, H, 2, O, TASK_MSE]
    var np_ = Net(cell, D, H, 2, O).n_params()
    var x = _values(N * T * D, 0x7D00 + which, 1.0)
    var p = _params(np_, 0x7D10 + which, 0.4)
    var whole = _predict(ex, shape, 1, 1, N, T, x, p, _empty(), _empty(), _empty())
    var head = _predict(ex, shape, 1, 1, N, t1, _steps(x, N, T, D, 0, t1), p, _empty(), _empty(), _empty())
    var c_in = head[3].copy() if cell == CELL_LSTM else _empty()
    var tail = _predict(ex, shape, 1, 1, N, T - t1, _steps(x, N, T, D, t1, T - t1), p, _empty(), head[2], c_in)
    _same(name + ": steps 0 .. 3 alone vs the whole run", _steps(whole[0], N, T, O, 0, t1), head[0])
    _same(name + ": steps 4 .. 8 from h_0 vs the whole run (head)", _steps(whole[0], N, T, O, t1, T - t1), tail[0])
    _same(name + ": steps 4 .. 8 from h_0 vs the whole run (output)", _steps(whole[1], N, T, H, t1, T - t1), tail[1])
    _same(name + ": final h from h_0 vs the whole run", whole[2], tail[2])
    if cell == CELL_LSTM:
        _same(name + ": final c from c_0 vs the whole run", whole[3], tail[3])
    print("RNN_PARITY", name, "(d) the run from the state at step 4 is the whole run's words")
    _ = c_in^
    var words = List[Float32]()
    for r in range(4):  # small-loop(4 results)
        words.extend(whole[r].copy())
    return words^


# ---------------------------------------------------------------- (e) streaming one step at a time
def run_stream_case[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    var name = _cell_name(which) + " streaming"
    var cell = _cell_of(which)
    var N = 4
    var T = 6
    var D = 4
    var H = 5
    var L = 1 if which == 0 else 2
    var O = 3
    var shape: List[Int] = [cell, D, H, L, O, TASK_CE]
    var np_ = Net(cell, D, H, L, O).n_params()
    var x = _values(N * T * D, 0x7E00 + which, 1.0)
    var p = _params(np_, 0x7E10 + which, 0.4)
    var whole = _predict(ex, shape, 1, 1, N, T, x, p, _empty(), _empty(), _empty())
    var probs = List[Float32](length=N * T * O, fill=0.0)
    var h = _empty()
    var c = _empty()
    for s in range(T):  # driver-loop: over time steps (the streaming form under test)
        var one = _predict(ex, shape, 1, 1, N, 1, _steps(x, N, T, D, s, 1), p, _empty(), h, c)
        for b in range(N):  # small-loop(N: check assembly)
            for o in range(O):  # small-loop(O: check assembly)
                probs[(b * T + s) * O + o] = one[0][b * O + o]
        h = one[2].copy()
        if cell == CELL_LSTM:
            c = one[3].copy()
    _same(name + ": six one-step calls vs one six-step call (probabilities)", whole[0], probs)
    _same(name + ": streamed h_n vs whole", whole[2], h)
    if cell == CELL_LSTM:
        _same(name + ": streamed c_n vs whole", whole[3], c)
    print("RNN_PARITY", name, "(e) one step at a time is the whole sequence's words")
    var words = whole[0].copy()
    words.extend(whole[2].copy())
    words.extend(whole[3].copy())
    return words^


# ---------------------------------------------------------------- (h) streaming at every input width
def run_width_case[E: Exec](mut ex: E, which: Int) raises -> List[Float32]:
    var widths: List[Int] = [8, 64, 65, 300]
    var D = widths[which]
    var name = "lstm streaming, " + String(D) + " inputs"
    var cell = CELL_LSTM
    var N = 4
    var T = 64
    var H = 32
    var L = 1
    var O = 3
    var GH = gates_of(cell) * H
    # the shape must separate the orders the old rule took (it keys on the rows above 64 inputs)
    if D > 64 and (gemm_split_shape(N * T, GH, D) or not gemm_split_shape(N, GH, D)):
        _fail(name + ": the shape no longer separates a one-step call's split rule from the whole call's")
    var shape: List[Int] = [cell, D, H, L, O, TASK_CE]
    var np_ = Net(cell, D, H, L, O).n_params()
    var x = _values(N * T * D, 0x7F00 + which, 1.0)
    var p = _params(np_, 0x7F10 + which, 0.1)
    var whole = _predict(ex, shape, 1, 1, N, T, x, p, _empty(), _empty(), _empty())
    var probs = List[Float32](length=N * T * O, fill=0.0)
    var h = _empty()
    var c = _empty()
    for s in range(T):  # driver-loop: over time steps (the streaming form under test)
        var one = _predict(ex, shape, 1, 1, N, 1, _steps(x, N, T, D, s, 1), p, _empty(), h, c)
        for b in range(N):  # small-loop(N: check assembly)
            for o in range(O):  # small-loop(O: check assembly)
                probs[(b * T + s) * O + o] = one[0][b * O + o]
        h = one[2].copy()
        c = one[3].copy()
    _same(name + ": 64 one-step calls vs one 64-step call (probabilities)", whole[0], probs)
    _same(name + ": streamed h_n vs whole", whole[2], h)
    _same(name + ": streamed c_n vs whole", whole[3], c)
    print("RNN_PARITY", name, "(h) one step at a time is the whole sequence's words")
    var words = whole[0].copy()
    words.extend(whole[2].copy())
    words.extend(whole[3].copy())
    return words^


# ---------------------------------------------------------------- (g) the gradient clip
def run_clip_case[E: Exec](mut ex: E) raises -> List[Float32]:
    var name = String("gru clip_grad_norm")
    var N = 8
    var T = 4
    var D = 3
    var H = 5
    var O = 2
    var shape: List[Int] = [CELL_GRU, D, H, 1, O, TASK_MSE]
    var np_ = Net(CELL_GRU, D, H, 1, O).n_params()
    var x = _values(N * T * D, 0x7F00, 1.0)
    var y = _values(N * O, 0x7F10, 2.0)
    var p0 = _params(np_, 0x7F20, 0.4)
    var raw = _fit(ex, shape, N, T, 1, N, 0, OPT_SGD, x, y, p0, _empty(), Float32(1.0), _no_ints(), _no_floats(),
                   _no_ints())
    var n_raw = 0.0
    for i in range(np_):  # small-loop(np: check norm)
        var d = Float64(p0[i]) - Float64(raw[i])
        n_raw += d * d
    n_raw = sqrt(n_raw)
    var clip_ints: List[Int] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    var tight: List[Float64] = [0.0, 0.25 * n_raw, 0.0]
    var clipped = _fit(ex, shape, N, T, 1, N, 0, OPT_SGD, x, y, p0, _empty(), Float32(1.0), clip_ints, tight,
                       _zero_addrs())
    var max_norm = Float64(Float32(0.25 * n_raw))
    var n_c = 0.0
    var worst = 0.0
    for i in range(np_):  # small-loop(np: check clipped step)
        var dc = Float64(p0[i]) - Float64(clipped[i])
        var dr = Float64(p0[i]) - Float64(raw[i])
        n_c += dc * dc
        worst = max(worst, abs(dc - dr * max_norm / n_raw))
    n_c = sqrt(n_c)
    if not (n_c <= max_norm * (1.0 + 1e-4) + 1e-6):
        _fail(name + ": the clipped step's norm " + String(n_c) + " is above max_norm " + String(max_norm))
    if not (n_c >= max_norm * (1.0 - 1e-3)):
        _fail(name + ": the clip did not bite: norm " + String(n_c) + " for max_norm " + String(max_norm))
    if worst > 1e-3 * max_norm + 1e-6:
        _fail(name + ": the clipped step is not the scaled plain step (max |diff| " + String(worst) + ")")
    if clipped[np_] != raw[np_]:
        _fail(name + ": the clip changed the loss of the step")
    var loose: List[Float64] = [0.0, 4.0 * n_raw, 0.0]
    var unclipped = _fit(ex, shape, N, T, 1, N, 0, OPT_SGD, x, y, p0, _empty(), Float32(1.0), clip_ints, loose,
                         _zero_addrs())
    _same(name + ": max_norm above the norm vs no clip", raw, unclipped)
    print("RNN_PARITY", name, "(g) step norm", n_c, "<= max_norm", max_norm, "(raw", n_raw, "); loose clip = no clip")
    var words = raw^
    words.extend(clipped^)
    return words^


# ---------------------------------------------------------------- (f) the epoch monitor
def run_monitor_case() raises -> List[Float32]:
    var m = EpochMonitor(1e-4, 2)
    var losses: List[Float64] = [1.0, 0.8, 0.79995, 0.9, 0.7, 0.75, 0.76]
    var improved: List[Bool] = [True, True, True, False, True, False, False]
    var stalls: List[Int] = [0, 0, 1, 2, 0, 1, 2]
    var stops: List[Bool] = [False, False, False, True, False, False, True]
    var bests: List[Int] = [1, 2, 3, 3, 5, 5, 5]
    for e in range(len(losses)):  # small-loop(7 epochs)
        var got = m.observe(losses[e], e + 1)
        if got != improved[e] or m.stall != stalls[e] or m.stop() != stops[e] or m.best_epoch != bests[e]:
            _fail("EpochMonitor epoch " + String(e + 1) + ": improved " + String(got) + " stall " + String(m.stall)
                  + " stop " + String(m.stop()) + " best epoch " + String(m.best_epoch))
    # patience 0 never stops; tol 0: an equal loss is neither an improvement nor a stall (scikit-learn's rule)
    var never = EpochMonitor(0.0, 0)
    for e in range(5):  # small-loop(5 epochs)
        _ = never.observe(1.0 + Float64(e), e + 1)
        if never.stop():
            _fail("EpochMonitor patience 0 stopped at epoch " + String(e + 1))
    if never.best_epoch != 1:
        _fail("EpochMonitor patience 0: best epoch " + String(never.best_epoch))
    var flat = EpochMonitor(0.0, 1)
    _ = flat.observe(1.0, 1)
    if flat.observe(1.0, 2) or flat.stall != 0 or flat.stop():
        _fail("EpochMonitor tol 0: an equal loss counted")
    var bad: List[UInt64] = [UInt64(0x7FF8000000000000), UInt64(0x7FF0000000000000), UInt64(0xFFF0000000000000)]
    for i in range(len(bad)):  # small-loop(3 values)
        var refused = False
        try:
            _ = m.observe(bitcast[DType.float64](bad[i]), 8)
        except error:
            _ = error
            refused = True
        if not refused:
            _fail("EpochMonitor took a loss that is not finite (" + String(i) + ")")
    print("RNN_PARITY (f) EpochMonitor: improvements, stalls, stops and best epochs as scikit-learn's rule")
    return List[Float32]()


def _run_unit[E: Exec](mut ex: E, unit: Int) raises -> List[Float32]:
    if unit < 2:
        return run_forms_case(ex, unit)
    if unit < 4:
        return run_bidi_case(ex, unit - 2)
    if unit < 6:
        return run_weighted_case(ex, unit - 4)
    if unit < 8:
        return run_tail_case(ex, unit - 6)
    if unit < 10:
        return run_stream_case(ex, unit - 8)
    if unit == 10:
        return run_clip_case(ex)
    if unit < 15:
        return run_width_case(ex, unit - 11)
    return run_monitor_case()


# ---------------------------------------------------------------- tests
def test_rnn_parity() raises:
    # The sixteen units are independent: on the host column each runs on its own HostExec across host cores;
    # the device column then runs them in order on the one device and requires the host words.
    comptime U = 16
    var failed = List[Int](length=U, fill=0)
    var host_words = List[List[Float32]](length=U, fill=List[Float32]())
    var fp = failed.unsafe_ptr()
    var wp = host_words.unsafe_ptr()
    def task(unit: Int) {imm fp, imm wp}:
        try:
            var host = HostExec()
            wp[unit] = _run_unit(host, unit)
        except error:
            print("FAIL RNN_PARITY unit", unit, error)
            fp[unit] = 1
    host_parallelize(task, U)
    for u in range(U):  # small-loop(U: failure flags)
        if failed[u] != 0:
            _fail("unit " + String(u) + " failed on the host column")
    comptime if not CPU_COLUMN_BUILD:
        for unit in range(U - 1):  # driver-loop: over the units on the one device (the monitor is host scalars)
            var dev = DeviceExec()
            var dw = _run_unit(dev, unit)
            _same("unit " + String(unit) + " host vs device", host_words[unit], dw)
    _ = failed^
    _ = host_words^
    print("PASS RNN_PARITY 16")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
