# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The composed tagger (lane nn-blocks-composed, 2026-11-25): the nn-blocks pieces in one model, the native twin of
the verify lane ops-composed-cases, case tagger (python/mojolearn/verify_reference/composed-tagger.json).

The model: three sequences of lengths 1, 3 and 4 (T = 4), token ids 0..4 with the padding id 5 past each length;
embedding (6 x 3, padding_idx 5) concatenated with two float features per step; a captured cell
h' = tanh(dropout(x, 0.25) W + h U + b) (hidden 3) run by `scan_session` with the lengths and counter = the
training step; a head (3 -> 4 logits: tags 0..2 and the padding label 3, which no real position has); the
cross-entropy weighted by the mask t < length (OPS_WEIGHTED_LOSS, mean); AdamW with three parameter groups (the
embedding at lr 0.01 with decay 0.01, the cell at 0.02, the head at 0.05 without decay; one global clip 1.0).

Units (each independent, own sessions; the CPU column runs them across host cores on HostExec, the device column
runs them one after another on DeviceExec, which shares one device context):
  0  train two steps; at each step the padded steps contribute nothing: the feature gradient past each length and
     the padding row of the embedding gradient are exactly zero, and the loss equals the float64 mean of the
     cross-entropy over the real positions only (from the step's logits; 1e-5). Then save (the parameters, every
     parameter's optimizer state words and the group epoch), run step 3 uninterrupted, and in a fresh session
     load the save and run step 3 (counter 2): parameters and optimizer state the same words;
  1  the scan's key: the tagger's cell over fixed inputs with counter 0 twice gives the same words, and counter 1
     (the next minibatch) other words: consecutive minibatches draw different dropout masks.
Sabotage arms coverage-autograd-nn_composed_{scan_key,group_state}_{gpu,cpu} (autograd/nn_composed.mojo) fail unit 1
and unit 0 on their own column.
"""
from std.math import exp, log
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import OPT_ADAMW
from core.cast_in import CastIn, CAST_KIND_SIGNED
from core.hostptr import f32_ptr
from autograd.rankn import ShapeN
from autograd.ops_api import (
    OpsSession, OPS_MATMUL, OPS_ADD, OPS_TANH, OPS_LINEAR, OPS_DROPOUT, OPS_EMBEDDING, OPS_RESHAPE_N, OPS_PERMUTE,
    OPS_CONCAT, OPS_COMPARE, OPS_CROSS_ENTROPY,
)
from autograd.per_step_ops import OPS_WEIGHTED_LOSS
from autograd.losses import REDUCTION_MEAN
from autograd.program import OpsProgram
from autograd.optimizer_requests import (
    OpsOptimizerRequest, OpsOptimizerExtras, OpsOptimizerRestoreRequest, OpsOptimizerStateValue,
)
from autograd.nn_scan_frontend import scan_session
from checks.native_tests.ops_higher_common import tensor, values, gradient
from checks.native_tests.sequence_port_common import normal
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require

comptime NN_COMPOSED_UNITS = 2

comptime _B = 3
comptime _T = 4
comptime _E = 3
comptime _F = 2
comptime _H = 3
comptime _C = 4
comptime _V = 6
comptime _PAD_ID = 5
comptime _PAD_TAG = 3
comptime _N_PARAMS = 6


def _fail(message: String) raises:
    print("FAIL NN_COMPOSED", message)
    raise Error(message)


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ")
    for i in range(len(a)):  # small-loop(check comparison)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _draw(n: Int, salt: Int, scale: Float64) -> List[Float32]:
    var out = List[Float32](length=n, fill=0)
    for i in range(n):  # small-loop(check data)
        out[i] = Float32(scale * normal(salt, i))
    return out^


def _lengths() -> List[Int]:
    return [1, 3, 4]


def _ids() -> List[Int32]:
    """(B, T) token ids, the padding id 5 past each length."""
    return [2, 5, 5, 5, 0, 4, 1, 5, 3, 1, 2, 0]


def _codes_bt() -> List[Float32]:
    """(B, T) tag codes, the padding label 3 past each length."""
    return [1, 3, 3, 3, 0, 2, 1, 3, 2, 2, 0, 1]


def _codes_tm() -> List[Float32]:
    """The codes time-major, one per logits row (row t * B + b)."""
    var bt = _codes_bt()
    var out = List[Float32](length=_T * _B, fill=0)
    for t in range(_T):  # small-loop(check data)
        for b in range(_B):  # small-loop(check data)
            out[t * _B + b] = bt[b * _T + t]
    return out^


def _dims(k: Int) -> List[Int]:
    """Parameter k's shape: table, W, U, b, head W, head b."""
    if k == 0:
        return [_V, _E]
    if k == 1:
        return [_E + _F, _H]
    if k == 2:
        return [_H, _H]
    if k == 3:
        return [1, _H]
    if k == 4:
        return [_H, _C]
    return [1, _C]


def _count(k: Int) -> Int:
    var d = _dims(k)
    return d[0] * d[1]


def _init(k: Int) -> List[Float32]:
    if k == 5:
        return List[Float32](length=_C, fill=0)
    return _draw(_count(k), 0xC0310 + k, 0.1 if k == 3 else 0.5)


def _op[E: Exec & Movable & Deinitable](mut s: OpsSession[E], code: Int, hs: List[Int], ints: List[Int]) raises -> Int:
    return s.op(code, hs, ints, List[Float32]())


def _words[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=0)
    s.download(h, f32_ptr(Int(out.unsafe_ptr())))
    return out^


def _bind(cell: OpsProgram, captured: List[Int], ours: List[Int]) raises -> List[Int]:
    """The handles of `ours` for the cell's parameters, matched through the capture session's handles."""
    var bound = List[Int]()
    for j in range(len(cell.params)):  # small-loop(the cell's parameters)
        var found = -1
        for i in range(len(captured)):  # small-loop(the capture session's weights)
            if captured[i] == cell.params[j]:
                found = i
        require(found >= 0, "a cell parameter is not one of the captured weights")
        bound.append(ours[found])
    return bound^


def _cell[E: Exec & Movable & Deinitable](mut c: OpsSession[E], w: List[Int]) raises -> Int:
    """h' = tanh(dropout(x, 0.25) W + h U + b) on inference session c (w = [W, U, b]); returns the program id."""
    var h = tensor(c, List[Float32](length=_B * _H, fill=0), [_B, _H])
    var x = tensor(c, List[Float32](length=_B * (_E + _F), fill=0), [_B, _E + _F])
    var dh: List[Int] = [x]
    var di: List[Int] = [13, 0, 0]
    var df: List[Float32] = [Float32(0.25)]
    var d = c.op(OPS_DROPOUT, dh, di, df)
    var a = _op(c, OPS_ADD, [_op(c, OPS_MATMUL, [d, w[0]], []), _op(c, OPS_MATMUL, [h, w[1]], [])], [])
    var h2 = _op(c, OPS_TANH, [_op(c, OPS_ADD, [a, w[2]], [])], [])
    var ins: List[Int] = [h, x]
    var outs: List[Int] = [h2, h2]
    return c.capture(ins, outs)


def _forward[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], cell: OpsProgram, p: List[Int], bound: List[Int], feats: Int, counter: Int,
) raises -> List[Int]:
    """The tagger on session s; returns [loss, logits]."""
    var ids = _ids()
    var id_dims: List[Int] = [_B, _T]
    var src = CastIn(Int(ids.unsafe_ptr()), CAST_KIND_SIGNED, 4, _B, _T, _T * 4, 4)
    var idh = s.upload_ids(src, ShapeN.of(id_dims))
    _ = ids^  # keep the id words alive past the upload
    var emb = _op(s, OPS_EMBEDDING, [p[0], idh], [_PAD_ID])
    var emb3 = _op(s, OPS_RESHAPE_N, [emb], [0, _B, _T, _E])
    var xcat = _op(s, OPS_CONCAT, [emb3, feats], [2])
    var xt = _op(s, OPS_PERMUTE, [xcat], [0, 1, 0, 2])
    var lens_v: List[Float32] = [1.0, 3.0, 4.0]
    var lens = tensor(s, lens_v, [_B, 1])
    var h0 = tensor(s, List[Float32](length=_B * _H, fill=0), [_B, _H])
    var states: List[Int] = [h0]
    var got = scan_session(s, cell, xt, states, lens, bound, False, counter)
    var flat = _op(s, OPS_RESHAPE_N, [got[0]], [0, _T * _B, _H])
    var logits = _op(s, OPS_LINEAR, [flat, p[4], p[5]], [0])
    var codes = tensor(s, _codes_tm(), [_T * _B, 1])
    var steps_v: List[Float32] = [0.0, 1.0, 2.0, 3.0]
    var steps = tensor(s, steps_v, [_T, 1])
    var lrow = _op(s, OPS_RESHAPE_N, [lens], [0, 1, _B])
    var mask = _op(s, OPS_RESHAPE_N, [_op(s, OPS_COMPARE, [steps, lrow], [2])], [0, _T * _B, 1])
    var lh: List[Int] = [logits, codes, mask]
    var li: List[Int] = [OPS_CROSS_ENTROPY, REDUCTION_MEAN]
    var lf: List[Float32] = [Float32(1.0)]
    var loss = s.op(OPS_WEIGHTED_LOSS, lh, li, lf)
    var r: List[Int] = [loss, logits]
    return r^


def _request(lr: Float32, decay: Float32) -> OpsOptimizerRequest:
    return OpsOptimizerRequest(OPT_ADAMW, False, 0, 1, lr, Float32(0), Float32(0.9), Float32(0.999), Float32(1e-8),
                               decay, Float32(0.1), Float32(0))


def _extras() -> OpsOptimizerExtras:
    return OpsOptimizerExtras(Float32(0.99), False, Float32(0), Float32(0), Float32(1.0), Float32(0))


def _optimizer[E: Exec & Movable & Deinitable](mut s: OpsSession[E], p: List[Int]) raises -> Int:
    """AdamW: the embedding, the cell and the head in three groups."""
    var reqs: List[OpsOptimizerRequest] = [_request(Float32(0.01), Float32(0.01)),
                                           _request(Float32(0.02), Float32(0.001)),
                                           _request(Float32(0.05), Float32(0))]
    var exts: List[OpsOptimizerExtras] = [_extras(), _extras(), _extras()]
    var groups = List[List[Int]]()
    groups.append([p[0]])
    groups.append([p[1], p[2], p[3]])
    groups.append([p[4], p[5]])
    return s.param_group_optimizer(_request(Float32(0.01), Float32(0.001)), _extras(), reqs, exts, groups)


def _check_mask(loss: List[Float32], logits: List[Float32], gfeats: List[Float32], gtable: List[Float32],
                step: Int) raises:
    """Padded steps contribute nothing (see the module docstring, unit 0)."""
    var lens = _lengths()
    var codes = _codes_bt()
    for b in range(_B):  # small-loop(check sequences)
        for t in range(lens[b], _T):  # small-loop(check padded steps)
            for f in range(_F):  # small-loop(check features)
                if gfeats[(b * _T + t) * _F + f] != 0:
                    _fail("step " + String(step) + ": sequence " + String(b) + " has a nonzero feature gradient "
                          + "at padded step " + String(t))
    for e in range(_E):  # small-loop(check the padding row)
        if gtable[_PAD_ID * _E + e] != 0:
            _fail("step " + String(step) + ": the padding id's embedding row has a nonzero gradient")
    var total = Float64(0)
    var n = 0
    for t in range(_T):  # small-loop(oracle steps)
        for b in range(_B):  # small-loop(oracle sequences)
            if t < lens[b]:
                var row = (t * _B + b) * _C
                var top = Float64(logits[row])
                for k in range(1, _C):  # small-loop(oracle classes)
                    top = max(top, Float64(logits[row + k]))
                var z = Float64(0)
                for k in range(_C):  # small-loop(oracle classes)
                    z += exp(Float64(logits[row + k]) - top)
                total += top + log(z) - Float64(logits[row + Int(codes[b * _T + t])])
                n += 1
    var want = total / Float64(n)
    var got = Float64(loss[0])
    if abs(got - want) > 1e-5 * max(1.0, abs(want)):
        _fail("step " + String(step) + ": the masked loss is " + String(got) + "; the mean over the real positions is "
              + String(want) + " (the padding label " + String(_PAD_TAG) + " entered the loss?)")


def _train_step[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], cell: OpsProgram, bound: List[Int], p: List[Int], o: Int, counter: Int, check: Bool,
) raises:
    var feats = tensor(s, _draw(_B * _T * _F, 0xC0320, 1.0), [_B, _T, _F], True)
    var out = _forward(s, cell, p, bound, feats, counter)
    var loss = values(s, out[0], [1, 1])
    var logits = values(s, out[1], [_T * _B, _C])
    s.backward(out[0], False)
    if check:
        _check_mask(loss, logits, gradient(s, feats, [_B, _T, _F]), gradient(s, p[0], [_V, _E]), counter)
    s.step(o, p)
    s.reset()


def _state_words[E: Exec & Movable & Deinitable](mut s: OpsSession[E], o: Int, p: List[Int]) raises -> List[Float32]:
    """Every parameter's AdamW state as words: step, beta1^t, beta2^t, then exp_avg and exp_avg_sq."""
    var out = List[Float32]()
    for k in range(_N_PARAMS):  # small-loop(the six parameters)
        var st = s.optimizer_state_value(o, p[k])
        out.append(Float32(st.step))
        out.append(st.beta1_power)
        out.append(st.beta2_power)
        out.extend(_words(s, st.first, _count(k)))
        out.extend(_words(s, st.second, _count(k)))
    return out^


# ---------------------------------------------------------------- unit 0: train, save, resume
def run_resume_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E], mut s: OpsSession[E], mut t: OpsSession[E]) raises:
    var cw = List[Int]()
    for k in range(1, 4):  # small-loop(the cell's three weights)
        cw.append(tensor(c, _init(k), _dims(k)))
    var cell = c.program(_cell(c, cw))
    var p = List[Int]()
    for k in range(_N_PARAMS):  # small-loop(the six parameters)
        p.append(tensor(s, _init(k), _dims(k), True))
    var sb = _bind(cell, cw, [p[1], p[2], p[3]])
    var o = _optimizer(s, p)
    for step in range(2):  # small-loop(two training steps)
        _train_step(s, cell, sb, p, o, step, True)
    # save: the parameters, every parameter's state, the group epoch and the step counter (the scan's key)
    var saved_p = List[List[Float32]]()
    var saved_st = List[OpsOptimizerStateValue]()
    var saved_m1 = List[List[Float32]]()
    var saved_m2 = List[List[Float32]]()
    for k in range(_N_PARAMS):  # small-loop(the six parameters)
        saved_p.append(_words(s, p[k], _count(k)))
        var st = s.optimizer_state_value(o, p[k])
        saved_st.append(st)
        saved_m1.append(_words(s, st.first, _count(k)))
        saved_m2.append(_words(s, st.second, _count(k)))
    var saved_epoch = s.pg_epoch(o)
    var saved_counter = 2
    s.reset()
    # step 3 uninterrupted
    _train_step(s, cell, sb, p, o, saved_counter, False)
    var straight = List[Float32]()
    for k in range(_N_PARAMS):  # small-loop(the six parameters)
        straight.extend(_words(s, p[k], _count(k)))
    var straight_state = _state_words(s, o, p)
    # step 3 in a fresh session from the save
    var q = List[Int]()
    for k in range(_N_PARAMS):  # small-loop(the six parameters)
        q.append(tensor(t, saved_p[k], _dims(k), True))
    var tb = _bind(cell, cw, [q[1], q[2], q[3]])
    var to = _optimizer(t, q)
    for k in range(_N_PARAMS):  # small-loop(the six parameters)
        var st = saved_st[k]
        var h1 = tensor(t, saved_m1[k], _dims(k))
        var h2 = tensor(t, saved_m2[k], _dims(k))
        t.load_optimizer_state(to, q[k], OpsOptimizerRestoreRequest(st.step, h1, h2, 0, st.beta1_power,
                                                                    st.beta2_power, st.mu_product))
    t.pg_set_epoch(to, saved_epoch)
    _train_step(t, cell, tb, q, to, saved_counter, False)
    var resumed = List[Float32]()
    for k in range(_N_PARAMS):  # small-loop(the six parameters)
        resumed.extend(_words(t, q[k], _count(k)))
    _same("step 3 parameters, uninterrupted vs resumed", straight, resumed)
    _same("step 3 optimizer state, uninterrupted vs resumed", straight_state, _state_words(t, to, q))


# ---------------------------------------------------------------- unit 1: the scan's key
def run_key_unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E]) raises:
    var cw = List[Int]()
    for k in range(1, 4):  # small-loop(the cell's three weights)
        cw.append(tensor(c, _init(k), _dims(k)))
    var cell = c.program(_cell(c, cw))
    var xs = tensor(c, _draw(_T * _B * (_E + _F), 0xC0330, 1.0), [_T, _B, _E + _F])
    var h0 = tensor(c, List[Float32](length=_B * _H, fill=0), [_B, _H])
    var states: List[Int] = [h0]
    var first = values(c, scan_session(c, cell, xs, states, 0, cell.params, False, 0)[0], [_T, _B, _H])
    var again = values(c, scan_session(c, cell, xs, states, 0, cell.params, False, 0)[0], [_T, _B, _H])
    var other = values(c, scan_session(c, cell, xs, states, 0, cell.params, False, 1)[0], [_T, _B, _H])
    _same("the tagger cell scanned twice with counter 0", first, again)
    var moved = False
    for i in range(len(first)):  # small-loop(check values)
        if bitcast[DType.uint32](first[i]) != bitcast[DType.uint32](other[i]):
            moved = True
    if not moved:
        _fail("counter 1 (the next minibatch) drew the dropout masks of counter 0")


# ---------------------------------------------------------------- the columns
def _unit[E: Exec & Movable & Deinitable](mut c: OpsSession[E], mut s: OpsSession[E], mut t: OpsSession[E],
                                         unit: Int) raises:
    if unit == 0:
        run_resume_unit(c, s, t)
    else:
        run_key_unit(c)


def test_nn_composed() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_unit(unit: Int) raises:
            var c = OpsSession[HostExec](HostExec(), False)
            var s = OpsSession[HostExec](HostExec(), True)
            var t = OpsSession[HostExec](HostExec(), True)
            try:
                _unit(c, s, t, unit)
            except error:
                c.close()
                s.close()
                t.close()
                raise error
            c.close()
            s.close()
            t.close()
        check_cases("nn_composed", NN_COMPOSED_UNITS, host_unit)
    else:
        def device_unit(unit: Int) raises:
            var c = OpsSession[DeviceExec](DeviceExec(), False)
            var s = OpsSession[DeviceExec](DeviceExec(), True)
            var t = OpsSession[DeviceExec](DeviceExec(), True)
            try:
                _unit(c, s, t, unit)
            except error:
                c.close()
                s.close()
                t.close()
                raise error
            c.close()
            s.close()
            t.close()
        check_cases_serial("nn_composed", NN_COMPOSED_UNITS, device_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
