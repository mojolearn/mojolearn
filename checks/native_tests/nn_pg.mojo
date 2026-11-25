# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""Optimizer parameter groups of `mojolearn.ops.Session` (lane nn-blocks-pg, 2026-11-25; autograd/nn_pg.mojo,
OpsSession.param_group_optimizer / _pg_step).

Two 2x2 parameters p (group 0) and q (group 1), loss = sum(A p + B q) with fixed A, B, so the gradients are the
column sums of A and of B (every step the same gradient).

Units (each independent, own sessions; the CPU column runs them across host cores on HostExec, the device column
runs each on DeviceExec):
  0-7   per kind (sgd, adam, adamw, rmsprop, adagrad, lion, adamax, nadam): one grouped optimizer (group 0 lr 0.01,
        weight decay 0.1; group 1 lr 0.05, no decay, its own momentum / beta1) stepped three times gives, word
        for word, the parameters and the state of two separately configured single-group optimizers;
  8     the global clip: SGD (no momentum), group lrs 0.5 and 0.25, clip_grad_norm 0.1: one step moves each
        parameter by -lr_g coef g with coef = 0.1 / (||(g_p, g_q)|| + 1e-6), the norm over BOTH groups in float64
        (tolerance 1e-6 relative); a clip per group would move them differently (checked: the per-group
        coefficients differ from the global one by more than the tolerance at these values);
  9     the schedule scales every group: step schedule gamma 0.5, step size 1: learning_rates after one
        scheduler_step is [0.005, 0.025] (exact words: 0.01 * 0.5, 0.05 * 0.5);
  10    resume: grouped adam stepped twice; its per-parameter state read back and loaded (through the grouped
        optimizer, routed to each parameter's group) into a second session at the same weights, then one step in
        each: the same words;
  11    refusals: a parameter in two groups, an empty group, a parameter listed twice in one group, a step naming a
        parameter outside every group, a grouped optimizer where one configuration is trained
        (optimizer_settings), a group whose kind differs from the optimizer's.
"""
from std.math import sqrt
from std.memory import bitcast
from std.testing import TestSuite
from autograd.ops_api import OpsSession, OPS_MATMUL, OPS_ADD, OPS_SUM
from autograd.optimizer_requests import (
    OpsOptimizerRequest, OpsOptimizerExtras, OpsOptimizerRestoreRequest, OpsOptimizerStateValue,
)
from core.cast_in import CastIn
from core.hostptr import f32_ptr
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import OPT_SGD, OPT_ADAM, OPT_ADAMW, OPT_RMSPROP, OPT_ADAGRAD, OPT_LION, OPT_ADAMAX, OPT_NADAM
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, check_cases_serial, require


comptime PG_FIXTURES = 12


def _p0() -> List[Float32]:
    return [1, -2, 0.5, -0.25]


def _q0() -> List[Float32]:
    return [0.75, 0.125, -1.5, 2]


def _a() -> List[Float32]:
    return [0.5, -1, 2, 0.25]


def _b() -> List[Float32]:
    return [-0.75, 1.5, 0.5, 1]


def _kind(fixture: Int) -> Int:
    var kinds: List[Int] = [OPT_SGD, OPT_ADAM, OPT_ADAMW, OPT_RMSPROP, OPT_ADAGRAD, OPT_LION, OPT_ADAMAX, OPT_NADAM]
    return kinds[fixture]


def _upload[E: Exec & Movable & Deinitable](mut s: OpsSession[E], values: List[Float32], grad: Bool = False) raises -> Int:
    var h = s.upload(CastIn.f32_c(Int(values.unsafe_ptr()), 2, 2), grad)
    _ = values
    return h


def _download[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int) raises -> List[Float32]:
    var values = List[Float32](length=4, fill=0)
    s.download(h, f32_ptr(Int(values.unsafe_ptr())))
    return values^


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    require(len(a) == len(b), name + ": lengths differ")
    for i in range(len(a)):  # small-loop(len(a): check comparison)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]),
                name + ": words differ at " + String(i) + " (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _backward[E: Exec & Movable & Deinitable](mut s: OpsSession[E], p: Int, q: Int, a: Int, b: Int) raises:
    """loss = sum(A p + B q): d/dp = A^T 1, d/dq = B^T 1."""
    var ap = s.run_op(OPS_MATMUL, [a, p], [], [])
    var bq = s.run_op(OPS_MATMUL, [b, q], [], [])
    s.backward(s.run_op(OPS_SUM, [s.run_op(OPS_ADD, [ap, bq], [], [])], [-1], []), False)


def _request(kind: Int, group: Int, schedule: Int = 0) -> OpsOptimizerRequest:
    """Group 0: lr 0.01, weight decay 0.1, momentum 0.9, beta1 0.9; group 1: lr 0.05, no decay, momentum 0.5
    (Nesterov for sgd), beta1 0.8. Group -1: the optimizer-wide defaults (group 0's)."""
    var g1 = group == 1
    return OpsOptimizerRequest(kind, g1 and kind == OPT_SGD, schedule, 1,
        Float32(0.05) if g1 else Float32(0.01),
        (Float32(0.5) if g1 else Float32(0.9)) if (kind == OPT_SGD or kind == OPT_RMSPROP) else Float32(0),
        Float32(0.8) if g1 else Float32(0.9), Float32(0.999), Float32(1e-8),
        Float32(0) if g1 else Float32(0.1), Float32(0.5), Float32(0))


def _extras(clip: Float32 = Float32(0)) -> OpsOptimizerExtras:
    return OpsOptimizerExtras(Float32(0.99), False, Float32(0), Float32(0), clip, Float32(0))


def _grouped[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], kind: Int, p: Int, q: Int, clip: Float32 = Float32(0), schedule: Int = 0,
) raises -> Int:
    var reqs: List[OpsOptimizerRequest] = [_request(kind, 0, schedule), _request(kind, 1, schedule)]
    var exts: List[OpsOptimizerExtras] = [_extras(clip), _extras(clip)]
    var groups = List[List[Int]]()
    groups.append([p])
    groups.append([q])
    return s.param_group_optimizer(_request(kind, -1, schedule), _extras(clip), reqs, exts, groups)


def _state_words[E: Exec & Movable & Deinitable](mut s: OpsSession[E], st: OpsOptimizerStateValue) raises -> List[Float32]:
    """step, the three running scalars and every state buffer the kind keeps, as words."""
    var out: List[Float32] = [Float32(st.step), st.beta1_power, st.beta2_power, st.mu_product]
    var hs: List[Int] = [st.first, st.second, st.third]
    for k in range(3):  # small-loop(3: state buffers)
        if hs[k] != 0:
            out.extend(_download(s, hs[k]))
    return out^


def run_groups_equal_separate[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut t: OpsSession[E], kind: Int) raises:
    var p = _upload(s, _p0(), True)
    var q = _upload(s, _q0(), True)
    var a = _upload(s, _a())
    var b = _upload(s, _b())
    var o = _grouped(s, kind, p, q)
    var tp = _upload(t, _p0(), True)
    var tq = _upload(t, _q0(), True)
    var ta = _upload(t, _a())
    var tb = _upload(t, _b())
    var op = t.optimizer_with(_request(kind, 0), _extras())
    var oq = t.optimizer_with(_request(kind, 1), _extras())
    for _ in range(3):  # small-loop(3: steps)
        _backward(s, p, q, a, b)
        s.step(o, [p, q])
        s.reset()
        _backward(t, tp, tq, ta, tb)
        t.step(op, [tp])
        t.step(oq, [tq])
        t.reset()
    _same("group 0 parameter", _download(s, p), _download(t, tp))
    _same("group 1 parameter", _download(s, q), _download(t, tq))
    _same("group 0 state", _state_words(s, s.optimizer_state_value(o, p)), _state_words(t, t.optimizer_state_value(op, tp)))
    _same("group 1 state", _state_words(s, s.optimizer_state_value(o, q)), _state_words(t, t.optimizer_state_value(oq, tq)))


def run_global_clip[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var p = _upload(s, _p0(), True)
    var q = _upload(s, _q0(), True)
    var a = _upload(s, _a())
    var b = _upload(s, _b())
    var max_norm = Float32(0.1)
    var base = OpsOptimizerRequest(OPT_SGD, False, 0, 1, 0.5, 0, 0.9, 0.999, 1e-8, 0, 0.5, 0)
    var r1 = base
    r1.lr = Float32(0.25)
    var reqs: List[OpsOptimizerRequest] = [base, r1]
    var exts: List[OpsOptimizerExtras] = [_extras(max_norm), _extras(max_norm)]
    var groups = List[List[Int]]()
    groups.append([p])
    groups.append([q])
    var o = s.param_group_optimizer(base, _extras(max_norm), reqs, exts, groups)
    _backward(s, p, q, a, b)
    s.step(o, [p, q])
    # float64 reference: d/dp[i, j] = sum_k A[k, i] (A^T 1), the same for q with B
    var av = _a()
    var bv = _b()
    var gp = List[Float64](length=4, fill=0.0)
    var gq = List[Float64](length=4, fill=0.0)
    for i in range(2):  # small-loop(2: rows)
        for j in range(2):  # small-loop(2: columns)
            gp[2 * i + j] = Float64(av[i]) + Float64(av[2 + i])
            gq[2 * i + j] = Float64(bv[i]) + Float64(bv[2 + i])
    var total = 0.0
    var sp = 0.0
    var sq = 0.0
    for k in range(4):  # small-loop(4: elements)
        sp += gp[k] * gp[k]
        sq += gq[k] * gq[k]
    total = sp + sq
    var coef = min(1.0, Float64(max_norm) / (sqrt(total) + 1e-6))
    var coef_p = min(1.0, Float64(max_norm) / (sqrt(sp) + 1e-6))
    require(abs(coef_p - coef) > 1e-3 * coef, "the fixture does not separate a global clip from a per-group clip")
    var p1 = _download(s, p)
    var q1 = _download(s, q)
    var p0 = _p0()
    var q0 = _q0()
    for k in range(4):  # small-loop(4: elements)
        var want_p = Float64(p0[k]) - 0.5 * coef * gp[k]
        var want_q = Float64(q0[k]) - 0.25 * coef * gq[k]
        require(abs(Float64(p1[k]) - want_p) <= 1e-6 * (1.0 + abs(want_p)),
                "group 0 moved by another coefficient than the global clip's at " + String(k))
        require(abs(Float64(q1[k]) - want_q) <= 1e-6 * (1.0 + abs(want_q)),
                "group 1 moved by another coefficient than the global clip's at " + String(k))


def run_schedule[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var p = _upload(s, _p0(), True)
    var q = _upload(s, _q0(), True)
    var o = _grouped(s, OPT_ADAM, p, q, Float32(0), 1)
    var before = s.pg_learning_rates(o)
    require(before[0] == Float32(0.01) and before[1] == Float32(0.05), "group rates before the schedule")
    s.scheduler_step(o)
    var after = s.pg_learning_rates(o)
    require(after[0] == Float32(0.01) * Float32(0.5) and after[1] == Float32(0.05) * Float32(0.5),
            "the schedule scales every group: " + String(after[0]) + ", " + String(after[1]))


def run_resume[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut t: OpsSession[E]) raises:
    var p = _upload(s, _p0(), True)
    var q = _upload(s, _q0(), True)
    var a = _upload(s, _a())
    var b = _upload(s, _b())
    var o = _grouped(s, OPT_ADAM, p, q)
    for _ in range(2):  # small-loop(2: steps)
        _backward(s, p, q, a, b)
        s.step(o, [p, q])
        s.reset()
    var sp = s.optimizer_state_value(o, p)
    var sq = s.optimizer_state_value(o, q)
    var tp = _upload(t, _download(s, p), True)
    var tq = _upload(t, _download(s, q), True)
    var ta = _upload(t, _a())
    var tb = _upload(t, _b())
    var to = _grouped(t, OPT_ADAM, tp, tq)
    var hp1 = _upload(t, _download(s, sp.first))
    var hp2 = _upload(t, _download(s, sp.second))
    var hq1 = _upload(t, _download(s, sq.first))
    var hq2 = _upload(t, _download(s, sq.second))
    t.load_optimizer_state(to, tp, OpsOptimizerRestoreRequest(sp.step, hp1, hp2, 0, sp.beta1_power, sp.beta2_power, sp.mu_product))
    t.load_optimizer_state(to, tq, OpsOptimizerRestoreRequest(sq.step, hq1, hq2, 0, sq.beta1_power, sq.beta2_power, sq.mu_product))
    t.pg_set_epoch(to, s.pg_epoch(o))
    s.reset()
    _backward(s, p, q, a, b)
    s.step(o, [p, q])
    _backward(t, tp, tq, ta, tb)
    t.step(to, [tp, tq])
    _same("resumed group 0", _download(s, p), _download(t, tp))
    _same("resumed group 1", _download(s, q), _download(t, tq))


def _refused[E: Exec & Movable & Deinitable](mut s: OpsSession[E], which: Int, p: Int, q: Int) raises -> String:
    try:
        var groups = List[List[Int]]()
        if which == 0:
            groups.append([p])
            groups.append([p, q])
        elif which == 1:
            groups.append([p])
            groups.append(List[Int]())
        else:
            groups.append([p, p])
            groups.append([q])
        var reqs: List[OpsOptimizerRequest] = [_request(OPT_ADAM, 0), _request(OPT_ADAM, 1)]
        var exts: List[OpsOptimizerExtras] = [_extras(), _extras()]
        _ = s.param_group_optimizer(_request(OPT_ADAM, -1), _extras(), reqs, exts, groups)
    except error:
        return String(error)
    return String("")


def run_refusals[E: Exec & Movable & Deinitable](mut s: OpsSession[E]) raises:
    var p = _upload(s, _p0(), True)
    var q = _upload(s, _q0(), True)
    var r = _upload(s, _a(), True)
    var count = len(s.opts)
    require("also in param_groups[0]" in _refused(s, 0, p, q), "a parameter in two groups was accepted")
    require("param_groups[1] has no parameters" in _refused(s, 1, p, q), "an empty group was accepted")
    require("lists parameter 1 twice" in _refused(s, 2, p, q), "a parameter listed twice was accepted")
    require(len(s.opts) == count, "a refused construction added optimizers")
    var o = _grouped(s, OPT_ADAM, p, q)
    var words = String("")
    try:
        s.step(o, [p, r])
    except error:
        words = String(error)
    require("in no parameter group" in words, "a step naming a parameter outside every group was accepted")
    words = String("")
    try:
        _ = s.optimizer_settings(o)
    except error:
        words = String(error)
    require("has parameter groups" in words, "a grouped optimizer was accepted where one configuration is trained")
    words = String("")
    var groups = List[List[Int]]()
    groups.append([p])
    groups.append([q])
    try:
        var reqs: List[OpsOptimizerRequest] = [_request(OPT_ADAM, 0), _request(OPT_ADAMW, 1)]
        var exts: List[OpsOptimizerExtras] = [_extras(), _extras()]
        _ = s.param_group_optimizer(_request(OPT_ADAM, -1), _extras(), reqs, exts, groups)
    except error:
        words = String(error)
    require("optimizer-wide setting" in words, "a group of another kind was accepted")


def run_fixture[E: Exec & Movable & Deinitable](mut s: OpsSession[E], mut t: OpsSession[E], fixture: Int) raises:
    if fixture < 8:
        run_groups_equal_separate(s, t, _kind(fixture))
    elif fixture == 8:
        run_global_clip(s)
    elif fixture == 9:
        run_schedule(s)
    elif fixture == 10:
        run_resume(s, t)
    else:
        run_refusals(s)


# ---- The host and device columns ----
def test_nn_pg() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_fixture(fixture_id: Int) raises:
            var s = OpsSession[HostExec](HostExec(), True)
            var t = OpsSession[HostExec](HostExec(), True)
            try:
                run_fixture(s, t, fixture_id)
            except error:
                t.close()
                s.close()
                raise error
            t.close()
            s.close()
        check_cases("nn_pg", PG_FIXTURES, host_fixture)
    else:
        def device_fixture(fixture_id: Int) raises:
            var s = OpsSession[DeviceExec](DeviceExec(), True)
            var t = OpsSession[DeviceExec](DeviceExec(), True)
            try:
                run_fixture(s, t, fixture_id)
            except error:
                t.close()
                s.close()
                raise error
            t.close()
            s.close()
        check_cases_serial("nn_pg", PG_FIXTURES, device_fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
