# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""User-defined backward rules (sub-lane nn-blocks-cbw, autograd/nn_cbw.mojo):
the deciding cases of Andrew's block 5 contract, natively, on the host column
and the device column.

The forward and backward programs are captured from an inference session
(MyTanh: forward x -> tanh(x), saving y; backward (y, g) -> g * (1 - y * y)),
bound into a recording session with `cbw_define`, and called with `cbw_call`.

  0 tanh: loss = 3 * sum(my_tanh(w)) against the built-in tanh's
    3 * sum(tanh(w)) in a second session: the gradient bit for bit, and within
    float32 rounding of the float64 3 * (1 - tanh(w)^2).
  1 two paths, seed 3: loss = 3 * (sum(my_tanh(x)) + sum(x * x)), the custom
    node recorded BEFORE the product, so the tape adds its gradient after the
    product's (a rule that overwrote the input's gradient, the DROP_PATH arm,
    loses 2x; one that assumed the incoming gradient 1, the SEED_ONE arm, loses
    the factor 3): the gradient against the float64 3 * (1 - tanh(x)^2) + 6 x,
    and against a central finite difference of the session's own loss.
  2 words round trip: the operation decoded from its node words (what a saved
    program holds) is bound again with the same key; a second session trains
    with the decoded operation and gives the same gradient bits.
  4 scratch storage (fable-review-neural, 2026-11-25 review): the rule runs its
    backward program on the session's own `cbw_tape`, never on `prog_tape`
    (Program.fit and run_program hold that one swapped; a rule that borrowed
    it allocated on a tape the fit dropped, storage kept until close). Two
    backwards of the same graph with a reset between: the rule's tape holds
    storage after the first, the SAME number of buffers after the second
    (reused, not grown), `prog_tape` holds none, and the gradient bits repeat.
  3 refusals: a backward with the wrong input count, a gradient of the wrong
    shape, a saved index past the forward's outputs, a forward output neither
    returned nor saved; and a recorded (create_graph) backward through the rule.
"""
from std.math import tanh as tanh64
from autograd.ops_api import OpsSession
from autograd.nn_cbw import OPS_CUSTOM_VJP, cbw_from_words
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from std.testing import TestSuite
from checks.native_tests.ops_higher_common import tensor, op1, op2, total, scale, values, gradient, same_words
from checks.native_tests.core_ops_common import normal
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, check_cases_serial, require


comptime NN_CBW_CASES = 5

comptime _SUB = 3
comptime _MUL = 4
comptime _TANH = 8
comptime _FULL = 30
comptime _ADD = 2

comptime _ROWS = 4
comptime _COLS = 3


def _dims() -> List[Int]:
    var d: List[Int] = [_ROWS, _COLS]
    return d^


def _full[E: Exec & Movable & Deinitable](mut s: OpsSession[E], value: Float32) raises -> Int:
    var ints: List[Int] = [1, 1]
    var fs: List[Float32] = [value]
    return s.op(_FULL, List[Int](), ints, fs)


def _my_tanh[E: Exec & Movable & Deinitable](mut cap: OpsSession[E], mut s: OpsSession[E]) raises -> Int:
    """Capture MyTanh's two programs in `cap` and bind them in `s`."""
    var zeros = List[Float32](length=_ROWS * _COLS, fill=0)
    var x = tensor(cap, zeros, _dims())
    var y = op1(cap, _TANH, x)
    var fin: List[Int] = [x]
    var fout: List[Int] = [y]
    var fwd = cap.capture(fin, fout)
    var ys = tensor(cap, zeros, _dims())
    var g = tensor(cap, zeros, _dims())
    var one = _full(cap, 1)
    var dx = op2(cap, _MUL, g, op2(cap, _SUB, one, op2(cap, _MUL, ys, ys)))
    var bin: List[Int] = [ys, g]
    var bout: List[Int] = [dx]
    var bwd = cap.capture(bin, bout)
    var saved: List[Int] = [0]
    return s.cbw_define(cap.program(fwd), cap.program(bwd), saved, 1)


def _call[E: Exec & Movable & Deinitable](mut s: OpsSession[E], op_id: Int, x: Int) raises -> Int:
    var hs: List[Int] = [x]
    var outs = s.cbw_call(op_id, hs)
    require(len(outs) == 1, "MyTanh returns one output")
    return outs[0]


def _tanh_case[E: Exec & Movable & Deinitable](mut cap: OpsSession[E], mut s: OpsSession[E], mut r: OpsSession[E]) raises:
    var w0 = normal(_ROWS * _COLS, 71)
    var op_id = _my_tanh(cap, s)
    var w = tensor(s, w0, _dims(), True)
    s.backward(scale(s, total(s, _call(s, op_id, w)), 3), False)
    var got = gradient(s, w, _dims())
    var wr = tensor(r, w0, _dims(), True)
    r.backward(scale(r, total(r, op1(r, _TANH, wr)), 3), False)
    var builtin = gradient(r, wr, _dims())
    same_words(got, builtin, "MyTanh's gradient against the built-in tanh's")
    for i in range(len(got)):
        var t = tanh64(Float64(w0[i]))
        var want = 3.0 * (1.0 - t * t)
        require(abs(Float64(got[i]) - want) <= 1e-5 + 1e-5 * abs(want),
                "MyTanh's gradient at " + String(i) + ": " + String(got[i]) + ", float64 " + String(want))


def _two_path_loss[E: Exec & Movable & Deinitable](mut s: OpsSession[E], op_id: Int, x: Int) raises -> Int:
    # the custom node first: the tape reaches it after the product
    var t = _call(s, op_id, x)
    var z = op2(s, _MUL, x, x)
    return scale(s, op2(s, _ADD, total(s, t), total(s, z)), 3)


def _two_path_case[E: Exec & Movable & Deinitable](mut cap: OpsSession[E], mut s: OpsSession[E], mut fd: OpsSession[E]) raises:
    var x0 = normal(_ROWS * _COLS, 73)
    var op_id = _my_tanh(cap, s)
    var x = tensor(s, x0, _dims(), True)
    s.backward(_two_path_loss(s, op_id, x), False)
    var got = gradient(s, x, _dims())
    # the float64 reference: 3 (1 - tanh(x)^2) + 6 x
    for i in range(len(got)):
        var t = tanh64(Float64(x0[i]))
        var want = 3.0 * (1.0 - t * t) + 6.0 * Float64(x0[i])
        require(abs(Float64(got[i]) - want) <= 1e-5 + 1e-5 * abs(want),
                "two paths, seed 3: the gradient at " + String(i) + " is " + String(got[i])
                + ", the analytic gradient " + String(want) + " (a rule dropped a path or assumed 1)")
    # a central finite difference of the session's own loss (inference: the
    # forward program runs, no rule), float32 values, step 1e-2
    var fd_op = _my_tanh(cap, fd)
    var dims1: List[Int] = [1, 1]
    for i in range(len(x0)):
        var plus = x0.copy()
        var minus = x0.copy()
        plus[i] += Float32(1e-2)
        minus[i] -= Float32(1e-2)
        var hp = tensor(fd, plus, _dims())
        var lp = values(fd, _two_path_loss(fd, fd_op, hp), dims1)
        var hm = tensor(fd, minus, _dims())
        var lm = values(fd, _two_path_loss(fd, fd_op, hm), dims1)
        fd.reset()
        var est = (Float64(lp[0]) - Float64(lm[0])) / (Float64(plus[i]) - Float64(minus[i]))
        require(abs(Float64(got[i]) - est) <= 2e-2 + 2e-2 * abs(est),
                "two paths: the gradient at " + String(i) + " is " + String(got[i])
                + ", the finite difference " + String(est))


def _words_case[E: Exec & Movable & Deinitable](mut cap: OpsSession[E], mut s: OpsSession[E], mut r: OpsSession[E]) raises:
    var x0 = normal(_ROWS * _COLS, 79)
    var op_id = _my_tanh(cap, s)
    var words = s.cbw.ops[op_id - 1].words.copy()
    var floats = s.cbw.ops[op_id - 1].floats.copy()
    var decoded = cbw_from_words(words, floats)
    require(decoded.key == s.cbw.ops[op_id - 1].key, "the decoded operation keeps its key")
    require(len(decoded.words) == len(words), "the decoded operation keeps its words")
    for i in range(len(words)):
        require(decoded.words[i] == words[i], "the decoded operation's word " + String(i))
    var x = tensor(s, x0, _dims(), True)
    s.backward(_two_path_loss(s, op_id, x), False)
    var first = gradient(s, x, _dims())
    # the second session holds only the words (a loaded program's node)
    var xr = tensor(r, x0, _dims(), True)
    var hs: List[Int] = [xr]
    var t = r.op(OPS_CUSTOM_VJP, hs, words, floats)
    var z = op2(r, _MUL, xr, xr)
    r.backward(scale(r, op2(r, _ADD, total(r, t), total(r, z)), 3), False)
    same_words(gradient(r, xr, _dims()), first, "the operation decoded from its words")


def _refused[E: Exec & Movable & Deinitable](
    mut cap: OpsSession[E], mut s: OpsSession[E], fwd: Int, bwd: Int, saved: List[Int], n_out: Int, words: String,
) raises:
    var refused = False
    try:
        _ = s.cbw_define(cap.program(fwd), cap.program(bwd), saved, n_out)
    except e:
        refused = String(e).find(words) >= 0
        require(refused, "the refusal names the mismatch: wanted '" + words + "', got '" + String(e) + "'")
    require(refused, "a signature mismatch must be refused: " + words)


def _refusal_case[E: Exec & Movable & Deinitable](mut cap: OpsSession[E], mut s: OpsSession[E]) raises:
    var zeros = List[Float32](length=_ROWS * _COLS, fill=0)
    var x = tensor(cap, zeros, _dims())
    var y = op1(cap, _TANH, x)
    var fin: List[Int] = [x]
    var fout: List[Int] = [y]
    var fwd = cap.capture(fin, fout)
    var two: List[Int] = [y, x]
    var fwd2 = cap.capture(fin, two)
    var g = tensor(cap, zeros, _dims())
    var bin1: List[Int] = [g]
    var bout1: List[Int] = [op1(cap, _TANH, g)]
    var bwd1 = cap.capture(bin1, bout1)  # one input: no saved tensor
    var small = List[Float32](length=_COLS, fill=0)
    var dsmall: List[Int] = [1, _COLS]
    var gs = tensor(cap, small, dsmall)
    var ys = tensor(cap, zeros, _dims())
    var bin2: List[Int] = [ys, gs]
    var bout2: List[Int] = [gs]
    var bwd2 = cap.capture(bin2, bout2)  # the gradient input of the wrong shape
    var saved: List[Int] = [0]
    var none = List[Int]()
    var past: List[Int] = [2]
    _refused(cap, s, fwd, bwd1, saved, 1, "the backward program takes 1 inputs; it must take 2")
    _refused(cap, s, fwd, bwd2, saved, 1, "backward input 1 (the gradient of output 0) has shape")
    _refused(cap, s, fwd, bwd1, past, 1, "saved entry 0 names forward output 2, which does not exist")
    _refused(cap, s, fwd2, bwd1, none, 1, "forward output 1 is neither returned nor saved")
    # first order only
    var op_id = _my_tanh(cap, s)
    var w = tensor(s, zeros, _dims(), True)
    var loss = total(s, _call(s, op_id, w))
    var refused = False
    try:
        s.backward_graph(loss)
    except e:
        refused = String(e).find("first order only") >= 0
    require(refused, "create_graph through a custom rule must be refused by name")


def _scratch_case[E: Exec & Movable & Deinitable](mut cap: OpsSession[E], mut s: OpsSession[E]) raises:
    var w0 = normal(_ROWS * _COLS, 73)
    var op_id = _my_tanh(cap, s)
    var w = tensor(s, w0, _dims(), True)
    s.backward(scale(s, total(s, _call(s, op_id, w)), 3), False)
    var first = gradient(s, w, _dims())
    var kept = len(s.cbw_tape._buffers)
    require(kept > 0, "the custom rule's backward program allocates on the session's cbw_tape")
    require(len(s.prog_tape._buffers) == 0, "the custom rule must not allocate on prog_tape")
    s.reset()
    s.backward(scale(s, total(s, _call(s, op_id, w)), 3), False)
    same_words(gradient(s, w, _dims()), first, "the same graph after a reset gives the same gradient")
    require(len(s.cbw_tape._buffers) == kept,
            "the rule's scratch storage is reused: " + String(kept) + " buffers after one backward, "
            + String(len(s.cbw_tape._buffers)) + " after two")
    require(len(s.prog_tape._buffers) == 0, "the custom rule must not allocate on prog_tape (second backward)")


def run_nn_cbw_case[E: Exec & Movable & Deinitable](
    mut cap: OpsSession[E], mut s: OpsSession[E], mut r: OpsSession[E], case_id: Int,
) raises:
    if case_id == 0:
        _tanh_case(cap, s, r)
    elif case_id == 1:
        _two_path_case(cap, s, r)
    elif case_id == 2:
        _words_case(cap, s, r)
    elif case_id == 3:
        _refusal_case(cap, s)
    elif case_id == 4:
        _scratch_case(cap, s)
    else:
        raise Error("unknown nn_cbw case")


def nn_cbw_third_records(case_id: Int) -> Bool:
    """The third session records, except for the finite difference (case 1)."""
    return case_id != 1


# The host column: each case on its own sessions, the cases across host cores. The device column: the sessions share
# sequence_ctx() and its pool, so the cases run one after another (as core_ops_program's device column).
def test_nn_cbw() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_case(case_id: Int) raises:
            var cap = OpsSession[HostExec](HostExec(), False)
            var s = OpsSession[HostExec](HostExec(), True)
            var r = OpsSession[HostExec](HostExec(), nn_cbw_third_records(case_id))
            try:
                run_nn_cbw_case(cap, s, r, case_id)
            except error:
                cap.close()
                s.close()
                r.close()
                raise error
            cap.close()
            s.close()
            r.close()
        check_cases("nn_cbw", NN_CBW_CASES, host_case)
    else:
        def device_case(case_id: Int) raises:
            var cap = OpsSession[DeviceExec](DeviceExec(), False)
            var s = OpsSession[DeviceExec](DeviceExec(), True)
            var r = OpsSession[DeviceExec](DeviceExec(), nn_cbw_third_records(case_id))
            try:
                run_nn_cbw_case(cap, s, r, case_id)
            except error:
                cap.close()
                s.close()
                r.close()
                raise error
            cap.close()
            s.close()
            r.close()
        check_cases_serial("nn_cbw", NN_CBW_CASES, device_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
