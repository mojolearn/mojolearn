# SPDX-License-Identifier: Apache-2.0
"""Save, load and resume for Sequential and the operation session (moved
from python/mojolearn/tests/test_autograd_save_resume.py, lane
tmig-training-2, 2026-11-15), through the entries the doors call:
model_fit_native, autograd/model_checkpoint.mojo model_save_checkpoint /
model_checkpoint_info / model_load_checkpoint (training_sequential_common.mojo)
and ops_save_state / ops_load_state on a recording session.

Data: X (120, 5) standard normals (core_ops_common.normal; the original drew
NumPy's default_rng(0), and no assertion depended on the stream),
yr = 2 x0 - x1 + 0.5 x2, y3 = [x0 > 0] + [x1 > 0.5].

  0-3 straight 4 epochs (seed 7) equals 2 epochs, a checkpoint, a load and 2
      resumed epochs, and equals 2 + 2 resumed in memory, word for word
      (parameters, both optimizer buffers, the loss curve, n_iter), with the
      same predictions: adam with a step schedule (1, 0.7), batch 32, on yr;
      Nesterov momentum lr 0.05 cross_entropy, batch 25, on y3; adamw lr 0.01
      weight decay 0.01 with layer norm, huber, exponential schedule 0.9,
      batch 40, on yr; sgd lr 0.05 without shuffle, batch 30, on yr.
  4   the same state gives the same checkpoint bytes (two fits of 2 epochs,
      seed 3), and a loaded checkpoint saves to the same bytes.
  5   early stopping (adam lr 0.05, tol 10, n_iter_no_change 2): 1 epoch,
      a checkpoint and 5 resumed epochs stop where 6 straight epochs stop,
      with the same parameters.
  6   an export (the Nesterov classifier, 3 epochs, seed 0) loads to the same
      argmax and softmax predictions word for word and holds no training state.
  7   session parameters and counters: w (4, 3) and b (1, 3) with seed
      2^40 + 5, epoch 3, step 17 round-trip exactly and are trainable after
      loading.
  8   session optimizer: six Adam (lr 0.01) steps of mse(x @ w, y) equal three
      steps, a save with the optimizer state, a load into a new session and
      three more steps, word for word.

The damaged and foreign file refusals and the cross-column subprocess
resume stay in the Python test (the columns are this check's two columns).
"""
from std.ffi import external_call
from std.os import getenv, remove
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from autograd.ops_api import (
    OpsSession, ops_session, ops_session_open, ops_session_close, OPS_HOST, OPS_DEVICE, OPS_MATMUL, OPS_MSE,
)
from autograd.optimizer_requests import OpsOptimizerRequest
from autograd.model_checkpoint import ops_save_state, ops_load_state, model_checkpoint_info
from autograd.model import (
    LOSS_MSE, LOSS_CROSS_ENTROPY, LOSS_HUBER, TARGET_DENSE, TARGET_CODES, OUT_RAW, OUT_SOFTMAX, OUT_ARGMAX,
)
from checks.native_tests.core_ops_common import normal, dims2
from checks.native_tests.ops_higher_common import tensor, op2, linear, loss, total, values, gradient
from checks.native_tests.training_sequential_common import (
    SeqModel, codes_as_words, L_RELU, L_TANH, L_SIGMOID, L_LAYER_NORM,
)
from checks.native_tests.training_check_common import same_words
from checks.native_tests.runner import CPU_COLUMN_BUILD, Check, Column, Report, require, read_fixture_bytes


comptime SEQ_CASES = 7
comptime SESSION_CASES = 2
comptime _N = 120
comptime _D = 5


def _path(column: String, name: String) -> String:
    var base = getenv("TMPDIR", "/tmp")
    return (base + "/mojolearn-check-save-resume-" + String(external_call["getpid", Int32]()) + "-" + column + "-"
            + name + ".ckpt")


def _drop(path: String):
    try:
        remove(path)
    except:
        pass


def _targets() -> Tuple[List[Float32], List[Float32], List[Float32]]:
    var x = normal(_N * _D, 0)
    var yr = List[Float32](length=_N, fill=0)
    var c3 = List[Int](length=_N, fill=0)
    for i in range(_N):  # small-loop(fixture targets): fixture-sized, no product work
        var r = _D * i
        yr[i] = x[r] * 2 - x[r + 1] + Float32(0.5) * x[r + 2]
        c3[i] = (1 if x[r] > 0 else 0) + (1 if x[r + 1] > Float32(0.5) else 0)
    return (x^, yr^, codes_as_words(c3))


def _case_model(which: Int) raises -> SeqModel:
    var m = SeqModel()
    if which == 0:
        m.linear(8)
        m.act(L_TANH)
        m.linear(1)
        m.compile(LOSS_MSE, "adam", 0.02, sched=1, step_size=1, gamma=0.7)
    elif which == 1 or which == 6:
        m.linear(8)
        m.act(L_RELU)
        m.linear(3)
        m.compile(LOSS_CROSS_ENTROPY, "momentum", 0.05, nesterov=True)
    elif which == 2:
        m.linear(6)
        m.layer(L_LAYER_NORM, f=1e-5)
        m.act(L_RELU)
        m.linear(1)
        m.compile(LOSS_HUBER, "adamw", 0.01, weight_decay=0.01, sched=2, gamma=0.9)
    elif which == 3:
        m.linear(4)
        m.act(L_SIGMOID)
        m.linear(1)
        m.compile(LOSS_MSE, "sgd", 0.05)
    else:
        m.linear(4)
        m.act(L_RELU)
        m.linear(1)
        m.compile(LOSS_MSE, "adam", 0.05, tol=10.0, patience=2)
    return m^


def _batch(which: Int) -> Int:
    if which == 0:
        return 32
    if which == 1 or which == 6:
        return 25
    if which == 2:
        return 40
    return 30


def _same_words(a: SeqModel, b: SeqModel, what: String) raises:
    require(same_words(a.params, b.params), what + ": parameters differ")
    require(same_words(a.m, b.m), what + ": first optimizer buffer differs")
    require(same_words(a.v, b.v), what + ": second optimizer buffer differs")
    require(same_words(a.losses(), b.losses()), what + ": loss curves differ")
    require(a.n_iter == b.n_iter, what + ": n_iter differs")


def _same_bytes(a: String, b: String, what: String) raises:
    var x = read_fixture_bytes(a)
    var y = read_fixture_bytes(b)
    require(len(x) == len(y), what + ": file sizes differ")
    for i in range(len(x)):  # small-loop(check comparison): file-sized, no product work
        require(x[i] == y[i], what + ": byte " + String(i) + " differs")


def run_sequential_case[E: Exec](mut ex: E, column: String, unit: Int) raises:
    var d = _targets()
    var x = d[0].copy()
    var yr = d[1].copy()
    var y3 = d[2].copy()
    if unit <= 3:
        var classify = unit == 1
        var target = TARGET_CODES if classify else TARGET_DENSE
        var y = y3.copy() if classify else yr.copy()
        var shuffle = 0 if unit == 3 else -1
        var straight = _case_model(unit)
        _ = straight.fit(ex, x, _N, _D, y, 1, target, 4, _batch(unit), shuffle, 7)
        var first = _case_model(unit)
        _ = first.fit(ex, x, _N, _D, y, 1, target, 2, _batch(unit), shuffle, 7)
        var path = _path(column, "which" + String(unit))
        var again = _case_model(unit)
        try:
            _ = first.save(path, _D, False)
            again.load(path, _D)
        except error:
            _drop(path)
            raise error
        _drop(path)
        _ = again.fit(ex, x, _N, _D, y, 1, target, 2, -1, -1, 0, True)
        _same_words(again, straight, "save, load and continue")
        var inmem = _case_model(unit)
        _ = inmem.fit(ex, x, _N, _D, y, 1, target, 2, _batch(unit), shuffle, 7)
        _ = inmem.fit(ex, x, _N, _D, y, 1, target, 2, -1, -1, 0, True)
        _same_words(inmem, straight, "the in-memory continuation")
        var transform = OUT_ARGMAX if classify else OUT_RAW
        require(same_words(again.predict(ex, x, _N, _D, transform), straight.predict(ex, x, _N, _D, transform)),
                "the resumed model predicts as the straight one")
        _ = y^
    elif unit == 4:
        var paths = List[String]()
        for i in range(2):  # small-loop(2: two fits of the same state)
            var m = _case_model(0)
            _ = m.fit(ex, x, _N, _D, yr, 1, TARGET_DENSE, 2, 32, -1, 3)
            paths.append(_path(column, "run" + String(i)))
            _ = m.save(paths[i], _D, False)
        var back = _path(column, "back")
        try:
            _same_bytes(paths[0], paths[1], "the same state")
            var loaded = _case_model(0)
            loaded.load(paths[0], _D)
            _ = loaded.save(back, _D, False)
            _same_bytes(back, paths[0], "load then save")
        except error:
            _drop(paths[0])
            _drop(paths[1])
            _drop(back)
            raise error
        _drop(paths[0])
        _drop(paths[1])
        _drop(back)
    elif unit == 5:
        var straight = _case_model(5)
        _ = straight.fit(ex, x, _N, _D, yr, 1, TARGET_DENSE, 6, -1, -1, 1)
        var part = _case_model(5)
        _ = part.fit(ex, x, _N, _D, yr, 1, TARGET_DENSE, 1, -1, -1, 1)
        var path = _path(column, "es")
        var rest = _case_model(5)
        try:
            _ = part.save(path, _D, False)
            rest.load(path, _D)
        except error:
            _drop(path)
            raise error
        _drop(path)
        _ = rest.fit(ex, x, _N, _D, yr, 1, TARGET_DENSE, 5, -1, -1, 0, True)
        require(rest.n_iter == straight.n_iter, "the resumed early stop ends where the straight one does")
        require(same_words(rest.params, straight.params), "the resumed early stop's parameters")
    else:
        require(unit == 6, "unknown save/resume which")
        var c = _case_model(6)
        _ = c.fit(ex, x, _N, _D, y3, 1, TARGET_CODES, 3, 25, -1, 0)
        var path = _path(column, "clf")
        var served = _case_model(6)
        try:
            _ = c.save(path, _D, True)
            require(model_checkpoint_info(path).is_export, "the export is marked as an export")
            served.load(path, _D)
        except error:
            _drop(path)
            raise error
        _drop(path)
        require(not served.fitted, "an export holds no training state")
        require(same_words(served.predict(ex, x, _N, _D, OUT_ARGMAX), c.predict(ex, x, _N, _D, OUT_ARGMAX)),
                "the export predicts the same classes")
        require(same_words(served.predict(ex, x, _N, _D, OUT_SOFTMAX), c.predict(ex, x, _N, _D, OUT_SOFTMAX)),
                "the export predicts the same probabilities")
    _ = x^
    _ = yr^
    _ = y3^


def _adam(lr: Float32) -> OpsOptimizerRequest:
    return OpsOptimizerRequest(1, False, 0, 1, lr, Float32(0.0), Float32(0.9), Float32(0.999), Float32(1e-8),
                               Float32(0.0), Float32(0.1), Float32(0.0))


def _train[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], w: Int, opt: Int, xv: List[Float32], yv: List[Float32], steps: Int
) raises:
    var xh = tensor(s, xv, dims2(32, 4))
    var yh = tensor(s, yv, dims2(32, 2))
    for _ in range(steps):  # dependent training/optimizer recurrence
        s.reset()
        s.backward(loss(s, OPS_MSE, op2(s, OPS_MATMUL, xh, w), yh), False)
        var ws: List[Int] = [w]
        s.step(opt, ws)


def _open[KIND: Int]() raises -> Int:
    """A recording session (`ops.Session(record=True)`) of the column's kind."""
    comptime if KIND == OPS_HOST:
        return ops_session_open[HostExec, OPS_HOST](HostExec(), True)
    else:
        return ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), True)


def run_session_case[E: Exec & Movable & Deinitable, KIND: Int](column: String, unit: Int) raises:
    if unit == 0:
        var w0 = normal(12, 70)
        var b0 = normal(3, 71)
        var path = _path(column, "session")
        var seed = (1 << 40) + 5
        var sid = _open[KIND]()
        try:
            var s = ops_session[E, KIND](sid)
            var w = tensor(s[], w0, dims2(4, 3), True)
            var b = tensor(s[], b0, dims2(1, 3), True)
            var hs: List[Int] = [w, b]
            var counters: List[Int] = [seed & 0xFFFFFFFF, seed >> 32, 3, 17]
            _ = ops_save_state[E, KIND](sid, path, hs, 0, counters)
        except error:
            ops_session_close[E, KIND](sid)
            _drop(path)
            raise error
        ops_session_close[E, KIND](sid)
        var lid = _open[KIND]()
        try:
            var st = ops_load_state[E, KIND](lid, path, True, 0)
            _drop(path)
            var s = ops_session[E, KIND](lid)
            require(st.counters[0] == (seed & 0xFFFFFFFF) and st.counters[1] == (seed >> 32), "the seed round-trips")
            require(st.counters[2] == 3 and st.counters[3] == 17, "epoch and step round-trip")
            require(len(st.handles) == 2, "two tensors")
            require(same_words(values(s[], st.handles[0], dims2(4, 3)), w0), "w round-trips")
            require(same_words(values(s[], st.handles[1], dims2(1, 3)), b0), "b round-trips")
            var xh = tensor(s[], normal(20, 72), dims2(5, 4))
            s[].backward(total(s[], linear(s[], xh, st.handles[0], st.handles[1], 0)), False)
            require(len(gradient(s[], st.handles[0], dims2(4, 3))) == 12, "trainable after loading")
        except error:
            ops_session_close[E, KIND](lid)
            _drop(path)
            raise error
        ops_session_close[E, KIND](lid)
        return
    require(unit == 1, "unknown session which")
    var xv = normal(128, 80)
    var yv = normal(64, 81)
    var w0 = normal(8, 82)
    var straight = List[Float32]()
    var a = _open[KIND]()
    try:
        var s = ops_session[E, KIND](a)
        var w = tensor(s[], w0, dims2(4, 2), True)
        var opt = s[].optimizer(_adam(0.01))
        _train(s[], w, opt, xv, yv, 6)
        straight = values(s[], w, dims2(4, 2))
    except error:
        ops_session_close[E, KIND](a)
        raise error
    ops_session_close[E, KIND](a)
    var path = _path(column, "opt")
    var b = _open[KIND]()
    try:
        var s = ops_session[E, KIND](b)
        var w = tensor(s[], w0, dims2(4, 2), True)
        var opt = s[].optimizer(_adam(0.01))
        _train(s[], w, opt, xv, yv, 3)
        var hs: List[Int] = [w]
        var counters: List[Int] = [0, 0, 0, 3]
        _ = ops_save_state[E, KIND](b, path, hs, opt, counters)
    except error:
        ops_session_close[E, KIND](b)
        _drop(path)
        raise error
    ops_session_close[E, KIND](b)
    var c = _open[KIND]()
    try:
        var s = ops_session[E, KIND](c)
        var opt = s[].optimizer(_adam(0.01))
        var st = ops_load_state[E, KIND](c, path, True, opt)
        _drop(path)
        _train(s[], st.handles[0], opt, xv, yv, 3)
        require(same_words(values(s[], st.handles[0], dims2(4, 2)), straight),
                "3 + save + load + 3 Adam steps equal 6 steps")
    except error:
        ops_session_close[E, KIND](c)
        _drop(path)
        raise error
    ops_session_close[E, KIND](c)


# ---- The host column ----
# The seven Sequential cases run across host cores (private executors and
# files); the two session cases then run in order (the unlocked process-wide
# session table, autograd/ops_api.mojo _OPS_SESSIONS).
def _run_cpu(mut report: Report) raises:
    var seq = report.cases(SEQ_CASES)
    def task(unit: Int) {imm seq}:
        try:
            var ex = HostExec()
            run_sequential_case(ex, "cpu", unit)
        except error:
            seq.fail(unit, error)
    report.parallel(task, SEQ_CASES)
    var sess = report.cases(SESSION_CASES)
    for unit in range(SESSION_CASES):  # driver-loop: over session cases (unlocked session table)
        try:
            run_session_case[HostExec, OPS_HOST]("cpu", unit)
        except error:
            sess.fail(unit, error)


# ---- The device column ----
# Every which runs in order on the one device.
def _run_gpu(mut report: Report) raises:
    var ex = DeviceExec()
    var seq = report.cases(SEQ_CASES)
    for unit in range(SEQ_CASES):  # driver-loop: over cases on the one device
        try:
            run_sequential_case(ex, "gpu", unit)
        except error:
            seq.fail(unit, error)
    var sess = report.cases(SESSION_CASES)
    for unit in range(SESSION_CASES):  # driver-loop: over session cases on the one device
        try:
            run_session_case[DeviceExec, OPS_DEVICE]("gpu", unit)
        except error:
            sess.fail(unit, error)


def run_cases(column: Column, mut report: Report):
    try:
        comptime if CPU_COLUMN_BUILD:
            _run_cpu(report)
        else:
            _run_gpu(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("training_autograd_save_resume", cpu=True, gpu=True, serial="", run=run_cases)
