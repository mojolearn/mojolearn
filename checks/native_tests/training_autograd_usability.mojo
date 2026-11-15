# SPDX-License-Identifier: Apache-2.0
"""Validation, best-state restore and captured-program training (moved from
python/mojolearn/tests/test_autograd_usability.py, lane tmig-training-2,
2026-11-15), through the entries the public doors call: the Sequential fit
words of training_sequential_common.mojo (model_fit_native with validation
data) and autograd/program_entries.mojo (program_execute_prepare /
program_execute_run, program_fit_parameters / program_fit_native) with the
doors' words.

Data: x (43, 3) standard normals (core_ops_common.normal; the original drew
NumPy's default_rng(41), and no assertion depended on the stream),
y = 1.3 x0 - 0.7 x1; the regression model is linear(1), mse, adam lr 0.1.

  0 best-state restore: references fit 1..6 epochs (batch 16, no shuffle,
    seed 3) and their validation MSE on (x[:17], -y[:17]); the 6-epoch
    validated fit with restore_best_weights runs 6 epochs, keeps n_iter =
    best epoch, its validation curve equals the references' (rtol 2e-5, atol
    2e-6), its parameters equal the best reference's words, and one resumed
    epoch equals a fresh fit of best + 1 epochs word for word.
  1 validation early stopping: lr 1e-6, n_iter_no_change 2, tol 10, 20
    epochs validated on (x, -y): 3 epochs run, 3 validation losses.
  2 linear(1, bias=False), lr 0.01, n_iter_no_change 2, tol 0: one epoch on
    y * 100 validated on zeros gives the validation curve [0]; 5 resumed
    epochs without validation reach n_iter 6 (training loss is never
    compared with the validation loss).
  3 dynamic replay: tanh(x @ w) captured at (2, 4, 3), run at (5, 7, 3),
    (1, 2, 3), (2, 4, 3) against float64 (rtol 2e-5, atol 2e-6); a (2, 4, 5)
    replay is refused and the next (2, 4, 3) replay still works.
  4 program fit equals eager: five recorded Adam (lr 0.02) steps of
    mse(x @ w, y) with scheduler steps; a captured x @ w fitted 2 then 3
    full-batch epochs without shuffle gives the same w word for word, and
    run(x[:9]) is x[:9] @ w (atol 2e-6).
  5 custom loss: the program outputs mse(2 (x @ w), y) itself (loss
    'output'), SGD lr 0.03 on w only, 12 epochs of 16 (seed 4): the last loss
    is under a tenth of the first and the frozen scale stays 2.
  6 failure recovery: a fit on x[:, :2] is refused and w is unchanged;
    2 epochs (batch 32, seed 0) then lower the loss and run gives 43 rows.

The validation label vocabulary (class codes of the door's label sort) stays
in the Python test.
"""
from std.math import tanh
from std.memory import bitcast
from autograd.ops_api import (
    OpsSession, ops_session, ops_session_open, ops_session_close, OPS_HOST, OPS_DEVICE, OPS_MATMUL, OPS_MUL,
    OPS_MSE,
)
from autograd.optimizer_requests import OpsOptimizerRequest
from autograd.program_entries import (
    program_execute_prepare, program_execute_run, program_fit_parameters, program_fit_native,
)
from autograd.program_io import ProgramInput
from autograd.rankn import ShapeN
from autograd.model import LOSS_MSE, TARGET_DENSE, OUT_RAW
from core.cast_in import CastIn
from sequence.exec_trait import Exec
from sequence.ops import FP
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.native_tests.ops_higher import tensor, op2, linear, loss, values
from checks.native_tests.core_ops_common import normal, dims2, dims3
from checks.native_tests.training_sequential_common import SeqModel, L_LINEAR
from checks.native_tests.training_check_common import same_words
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require


comptime USABILITY_CASES = 7
comptime _N = 43
comptime _D = 3
comptime _ACT_TANH = 3


def _data() -> Tuple[List[Float32], List[Float32]]:
    var x = normal(_N * _D, 41)
    var y = List[Float32](length=_N, fill=0)
    for i in range(_N):  # small-loop(fixture targets): fixture-sized, no product work
        y[i] = x[_D * i] * Float32(1.3) - x[_D * i + 1] * Float32(0.7)
    return (x^, y^)


def _regression() raises -> SeqModel:
    var m = SeqModel()
    m.linear(1)
    m.compile(LOSS_MSE, "adam", 0.1)
    return m^


def _sgd(lr: Float32) -> OpsOptimizerRequest:
    return OpsOptimizerRequest(0, False, 0, 1, lr, Float32(0.0), Float32(0.9), Float32(0.999), Float32(1e-8),
                               Float32(0.0), Float32(0.1), Float32(0.0))


def _adam(lr: Float32) -> OpsOptimizerRequest:
    return OpsOptimizerRequest(1, False, 0, 1, lr, Float32(0.0), Float32(0.9), Float32(0.999), Float32(1e-8),
                               Float32(0.0), Float32(0.1), Float32(0.0))


def _first_rows(values: List[Float32], rows: Int, cols: Int) -> List[Float32]:
    var out = List[Float32](length=rows * cols, fill=0)
    for i in range(rows * cols):  # small-loop(fixture slice)
        out[i] = values[i]
    return out^


def _negated(values: List[Float32]) -> List[Float32]:
    var out = List[Float32](length=len(values), fill=0)
    for i in range(len(values)):  # small-loop(fixture targets)
        out[i] = -values[i]
    return out^


def run_sequential_case[E: Exec](mut ex: E, unit: Int) raises:
    var d = _data()
    var x = d[0].copy()
    var y = d[1].copy()
    if unit == 0:
        var xv = _first_rows(x, 17, _D)
        var yv = _negated(_first_rows(y, 17, 1))
        var losses = List[Float64]()
        var params = List[List[Float32]]()
        for count in range(1, 7):  # small-loop(6: reference fits)
            var m = _regression()
            _ = m.fit(ex, x, _N, _D, y, 1, TARGET_DENSE, count, 16, 0, 3)
            var p = m.predict(ex, xv, 17, _D, OUT_RAW)
            var s = Float64(0)
            for i in range(17):
                var e = Float64(p[i] - yv[i])
                s += e * e
            losses.append(s / 17.0)
            params.append(m.params.copy())
        var best = 1
        for k in range(1, 6):
            if losses[k] < losses[best - 1]:
                best = k + 1
        var m = _regression()
        _ = m.fit_validated(ex, x, _N, _D, y, 1, TARGET_DENSE, 6, xv, 17, yv, True, 16, 0, 3)
        require(Int(m.validation_info[0]) == 6, "six epochs ran")
        require(m.n_iter == best and Int(m.validation_info[1]) == best, "n_iter and best_epoch are the best epoch")
        for k in range(6):
            var a = Float64(m.validation_curve[k])
            require(abs(a - losses[k]) <= 2e-6 + 2e-5 * abs(losses[k]), "validation loss " + String(k + 1))
        require(same_words(m.params, params[best - 1]), "the best epoch's parameters are restored")
        _ = m.fit(ex, x, _N, _D, y, 1, TARGET_DENSE, 1, -1, -1, 0, True)
        var expected = _regression()
        _ = expected.fit(ex, x, _N, _D, y, 1, TARGET_DENSE, best + 1, 16, 0, 3)
        require(same_words(m.params, expected.params), "resuming from the restored state equals best + 1 epochs")
        _ = xv^
        _ = yv^
    elif unit == 1:
        var m = SeqModel()
        m.linear(1)
        m.compile(LOSS_MSE, "adam", 1e-6, tol=10.0, patience=2)
        var yn = _negated(y)
        _ = m.fit_validated(ex, x, _N, _D, y, 1, TARGET_DENSE, 20, x, _N, yn, False)
        require(Int(m.validation_info[0]) == 3, "validation early stopping after 3 epochs")
        _ = yn^
    else:
        require(unit == 2, "unknown sequential usability case")
        var m = SeqModel()
        m.layer(L_LINEAR, 1, 0)
        m.compile(LOSS_MSE, "adam", 0.01, tol=0.0, patience=2)
        var big = List[Float32](length=_N, fill=0)
        for i in range(_N):
            big[i] = y[i] * 100
        var zx = List[Float32](length=5 * _D, fill=0)
        var zy = List[Float32](length=5, fill=0)
        _ = m.fit_validated(ex, x, _N, _D, big, 1, TARGET_DENSE, 1, zx, 5, zy, False)
        require(Int(m.validation_info[0]) == 1 and m.validation_curve[0] == 0, "the validation curve is [0]")
        _ = m.fit(ex, x, _N, _D, big, 1, TARGET_DENSE, 5, -1, -1, 0, True)
        require(m.n_iter == 6, "five resumed epochs reach n_iter 6")
        _ = big^
        _ = zx^
        _ = zy^
    _ = x^
    _ = y^


def _input(values: List[Float32], dims: List[Int]) raises -> ProgramInput:
    var cols = dims[len(dims) - 1]
    return ProgramInput(CastIn.f32_c(Int(values.unsafe_ptr()), len(values) // cols, cols), ShapeN.of(dims),
                        List[Float64](), List[Int64]())


def _run[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, values: List[Float32], dims: List[Int]
) raises -> List[Float32]:
    var plan = program_execute_prepare[E, KIND](sid, pid, sid, False, False)
    var inputs = List[ProgramInput]()
    inputs.append(_input(values, dims))
    var result = program_execute_run[E, KIND](sid, plan, inputs^, False)
    _ = values
    return result.outputs[0].data.copy()


def _program_fit[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, opt: Int, var owners: List[ProgramInput], var targets: List[ProgramInput], kind: Int,
    params: List[Int], epochs: Int, batch: Int, shuffle: Bool, seed: Int,
) raises -> List[Float32]:
    var curve = List[Float32](length=epochs, fill=Float32(-1))
    var ran = program_fit_native[E, KIND](sid, pid, owners^, targets^, kind, Float32(1.0), opt, params, epochs, batch,
                                          shuffle, UInt64(seed), FP(unsafe_from_address=Int(curve.unsafe_ptr())), 0, 0)
    var out = List[Float32]()
    for i in range(ran[2]):  # small-loop(epochs run): the written curve
        out.append(curve[i])
    # the fit wrote through curve's address: keep it alive past the call
    _ = curve^
    return out^


def run_program_case[E: Exec & Movable & Deinitable, KIND: Int](sid: Int, unit: Int) raises:
    var s = ops_session[E, KIND](sid)
    var d = _data()
    var x = d[0].copy()
    var y = d[1].copy()
    if unit == 3:
        var w = normal(6, 19)
        var xh = tensor(s[], List[Float32](length=24, fill=1.0), dims3(2, 4, 3))
        var wh = tensor(s[], w, dims2(3, 2))
        var ins: List[Int] = [xh]
        var outs: List[Int] = [linear(s[], xh, wh, 0, _ACT_TANH)]
        var pid = s[].capture(ins, outs)
        for k in range(3):  # small-loop(3: replay shapes)
            var b = 5 if k == 0 else (1 if k == 1 else 2)
            var t = 7 if k == 0 else (2 if k == 1 else 4)
            var a = normal(b * t * 3, 190 + k)
            var got = _run[E, KIND](sid, pid, a, dims3(b, t, 3))
            require(len(got) == b * t * 2, "the replay's output shape")
            for r in range(b * t):
                for c in range(2):
                    var z = Float64(0)
                    for j in range(3):
                        z += Float64(a[3 * r + j]) * Float64(w[2 * j + c])
                    var want = tanh(z)
                    require(abs(Float64(got[2 * r + c]) - want) <= 2e-6 + 2e-5 * abs(want), "dynamic replay value")
            _ = a^
        var refused = False
        try:
            _ = _run[E, KIND](sid, pid, List[Float32](length=40, fill=1.0), dims3(2, 4, 5))
        except:
            refused = True
        require(refused, "a (2, 4, 5) replay is refused")
        var ones = List[Float32](length=24, fill=1.0)
        var got = _run[E, KIND](sid, pid, ones, dims3(2, 4, 3))
        for c in range(2):
            var z = Float64(0)
            for j in range(3):
                z += Float64(w[2 * j + c])
            for r in range(8):
                require(abs(Float64(got[2 * r + c]) - tanh(z)) <= 2e-6, "the replay after a refusal")
        _ = w^
    elif unit == 4:
        var w0: List[Float32] = [0.2, -0.1, 0.3]
        var expected = List[Float32]()
        var rec = _open_recording[KIND]()
        try:
            var r = ops_session[E, KIND](rec)
            var w = tensor(r[], w0, dims2(3, 1), True)
            var opt = r[].optimizer(_adam(0.02))
            for _ in range(5):  # dependent training/optimizer recurrence
                var a = tensor(r[], x, dims2(_N, _D))
                var b = tensor(r[], y, dims2(_N, 1))
                r[].backward(loss(r[], OPS_MSE, op2(r[], OPS_MATMUL, a, w), b), False)
                var ws: List[Int] = [w]
                r[].step(opt, ws)
                r[].scheduler_step(opt)
                r[].reset()
            expected = values(r[], w, dims2(3, 1))
        except error:
            ops_session_close[E, KIND](rec)
            raise error
        ops_session_close[E, KIND](rec)
        var a = tensor(s[], x, dims2(_N, _D))
        var w = tensor(s[], w0, dims2(3, 1))
        var ins: List[Int] = [a]
        var outs: List[Int] = [op2(s[], OPS_MATMUL, a, w)]
        var pid = s[].capture(ins, outs)
        var opt = s[].optimizer(_adam(0.02))
        var params = program_fit_parameters[E, KIND](sid, pid, 0, False, List[Int]())
        for epochs in range(2, 4):  # small-loop(2: the two fit calls, 2 then 3 epochs)
            var owners = List[ProgramInput]()
            owners.append(_input(x, dims2(_N, _D)))
            var targets = List[ProgramInput]()
            targets.append(_input(y, dims2(_N, 1)))
            _ = _program_fit[E, KIND](sid, pid, opt, owners^, targets^, 1, params, epochs, _N, False, 0)
        require(same_words(values(s[], w, dims2(3, 1)), expected), "program fit equals the eager steps")
        var head = _first_rows(x, 9, _D)
        var got = _run[E, KIND](sid, pid, head, dims2(9, _D))
        for i in range(9):
            var z = Float64(0)
            for j in range(3):
                z += Float64(head[3 * i + j]) * Float64(expected[j])
            require(abs(Float64(got[i]) - z) <= 2e-6, "run after the fit")
        _ = head^
    elif unit == 5:
        var a = tensor(s[], x, dims2(_N, _D))
        var target = tensor(s[], y, dims2(_N, 1))
        var w = tensor(s[], List[Float32](length=3, fill=0.0), dims2(3, 1))
        var two: List[Float32] = [2.0]
        var scale = tensor(s[], two, dims2(1, 1))
        var pred = op2(s[], OPS_MUL, op2(s[], OPS_MATMUL, a, w), scale)
        var ins: List[Int] = [a, target]
        var outs: List[Int] = [loss(s[], OPS_MSE, pred, target)]
        var pid = s[].capture(ins, outs)
        var opt = s[].optimizer(_sgd(0.03))
        var selected: List[Int] = [w]
        var params = program_fit_parameters[E, KIND](sid, pid, 0, True, selected)
        var owners = List[ProgramInput]()
        owners.append(_input(x, dims2(_N, _D)))
        owners.append(_input(y, dims2(_N, 1)))
        var curve = _program_fit[E, KIND](sid, pid, opt, owners^, List[ProgramInput](), 0, params, 12, 16, True, 4)
        require(len(curve) == 12, "12 epochs")
        require(curve[11] < curve[0] * Float32(0.1), "the custom loss fell below a tenth")
        require(same_words(values(s[], scale, dims2(1, 1)), two), "the frozen scale stays 2")
    else:
        require(unit == 6, "unknown program usability case")
        var a = tensor(s[], x, dims2(_N, _D))
        var initial = List[Float32](length=3, fill=0.0)
        var w = tensor(s[], initial, dims2(3, 1))
        var ins: List[Int] = [a]
        var outs: List[Int] = [op2(s[], OPS_MATMUL, a, w)]
        var pid = s[].capture(ins, outs)
        var opt = s[].optimizer(_sgd(0.01))
        var params = program_fit_parameters[E, KIND](sid, pid, 0, False, List[Int]())
        var narrow = List[Float32](length=_N * 2, fill=0)
        for i in range(_N):
            narrow[2 * i] = x[3 * i]
            narrow[2 * i + 1] = x[3 * i + 1]
        var refused = False
        try:
            var owners = List[ProgramInput]()
            owners.append(_input(narrow, dims2(_N, 2)))
            var targets = List[ProgramInput]()
            targets.append(_input(y, dims2(_N, 1)))
            _ = _program_fit[E, KIND](sid, pid, opt, owners^, targets^, 1, params, 1, 32, True, 0)
        except:
            refused = True
        require(refused, "a fit on the wrong width is refused")
        require(same_words(values(s[], w, dims2(3, 1)), initial), "a refused fit leaves w unchanged")
        var owners = List[ProgramInput]()
        owners.append(_input(x, dims2(_N, _D)))
        var targets = List[ProgramInput]()
        targets.append(_input(y, dims2(_N, 1)))
        var curve = _program_fit[E, KIND](sid, pid, opt, owners^, targets^, 1, params, 2, 32, True, 0)
        require(len(curve) == 2 and curve[1] < curve[0], "the loss falls after the recovery")
        require(len(_run[E, KIND](sid, pid, x, dims2(_N, _D))) == _N, "run gives one output per row")
        _ = narrow^
    _ = x^
    _ = y^


def _open_recording[KIND: Int]() raises -> Int:
    """A second, recording session (`ops.Session(record=True)`) on a fresh
    executor of the column's kind."""
    comptime if KIND == OPS_HOST:
        return ops_session_open[HostExec, OPS_HOST](HostExec(), True)
    else:
        return ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), True)


# ---- The host column ----
# The three Sequential cases run across host cores on private executors; the
# four program cases then run in order (process-wide session table).
def _run_cpu(mut report: Report) raises:
    var seq = report.cases(3)
    def task(unit: Int) {imm seq}:
        try:
            var ex = HostExec()
            run_sequential_case(ex, unit)
        except error:
            seq.fail(unit, error)
    report.parallel(task, 3)
    var prog = report.cases(4)
    for unit in range(4):  # driver-loop: over program cases (unlocked session table)
        try:
            var sid = ops_session_open[HostExec, OPS_HOST](HostExec(), False)
            try:
                run_program_case[HostExec, OPS_HOST](sid, unit + 3)
            except error:
                ops_session_close[HostExec, OPS_HOST](sid)
                raise error
            ops_session_close[HostExec, OPS_HOST](sid)
        except error:
            prog.fail(unit, error)


# ---- The device column ----
# Every case runs in order on the one device.
def _run_gpu(mut report: Report) raises:
    var ex = DeviceExec()
    var seq = report.cases(3)
    for unit in range(3):  # driver-loop: over cases on the one device
        try:
            run_sequential_case(ex, unit)
        except error:
            seq.fail(unit, error)
    var prog = report.cases(4)
    for unit in range(4):  # driver-loop: over program cases on the one device
        try:
            var sid = ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), False)
            try:
                run_program_case[DeviceExec, OPS_DEVICE](sid, unit + 3)
            except error:
                ops_session_close[DeviceExec, OPS_DEVICE](sid)
                raise error
            ops_session_close[DeviceExec, OPS_DEVICE](sid)
        except error:
            prog.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("training_autograd_usability", cpu=True, gpu=True, serial="", run=run_cases)
