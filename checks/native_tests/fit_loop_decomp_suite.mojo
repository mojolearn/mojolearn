# SPDX-License-Identifier: Apache-2.0
"""The fit loop clock of the decomposition iterative fits (lane ab-loop-mode /
alm-iter, 2026-11-24): NMF (multiplicative update and coordinate descent),
FastICA (parallel and deflation), FactorAnalysis's EM loop and
AlternatingLeastSquares's sweeps, on tiny fixtures with the iteration cap K
and early stopping off. Each solver must report a recorded loop of exactly K
iterations (equal to the count it returns, the fitted `n_iter_`) and a
non-negative time, on the host column and, in a device build, on the device.

Suite module (no `main`): checks/native_tests/fit_loop_decomp.mojo runs
`run_checks`."""
from std.math import inf
from checks.native_tests.testsuite_cases import CPU_COLUMN_BUILD, require
from core.fit_loop_clock import FitLoop
from x_decomp.cells import F32Ptr
from x_decomp.host import HostExec
from x_decomp.nmf import NmfArgs, nmf_solve_run
from x_decomp.nmf_dev import nmf_solve_dev_run
from x_decomp.ica import ica_solve_run
from x_decomp.ica_dev import ica_solve_dev_run
from x_decomp.fa_em import FaArgs, fa_em_main_run
from x_decomp.fa_em_dev import fa_em_main_dev_run
from x_decomp.als import AlsArgs, als_fit_run
from x_decomp.als_dev import als_fit_dev_run

#: the iteration cap every fixture runs to
comptime K = 4


def _p(values: List[Float32]) -> F32Ptr:
    """The address of a List for a solver call; the caller keeps it alive."""
    return F32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def _loop(loop: FitLoop, returned: Int, what: String) raises:
    require(loop.recorded, what + ": the fit loop was not recorded")
    require(returned == K, what + ": ran " + String(returned) + " iterations, the cap is " + String(K))
    require(loop.iterations == returned, what + ": fit_loop_ iterations " + String(loop.iterations)
            + " != n_iter_ " + String(returned))
    require(loop.loop_ns >= 0 and loop.milliseconds() >= 0.0, what + ": negative loop time")


def _column(device: Bool) -> String:
    return String(" device") if device else String(" host")


def _source() -> List[Float32]:
    # 6 x 3, non-negative, full rank
    var values: List[Float32] = [1, 2, 1, 3, 1, 2, 1, 1, 4, 2, 3, 1, 1, 2, 2, 4, 1, 1]
    return values^


def _nmf(solver: Int, device: Bool) raises:
    var x = _source()
    var w = List[Float32](length=12, fill=Float32(0.5))
    var h = List[Float32](length=6, fill=Float32(0.5))
    # tol 0 turns the multiplicative update's check off; coordinate descent
    # stops at viol / v_init <= tol, so it takes a negative tol
    var tol = 0.0 if solver == 1 else -1.0
    var args = NmfArgs(2.0, True, K, tol, 0.0, 0.0, 0.0, 0.0, False, 7)
    var loop = FitLoop.empty()
    var it = -1
    if device:
        # the host column's build never elaborates a device fit
        comptime if not CPU_COLUMN_BUILD:
            it = nmf_solve_dev_run(_p(x), _p(w), _p(h), 6, 3, 2, solver, args, loop)
    else:
        it = nmf_solve_run[HostExec](_p(x), _p(w), _p(h), 6, 3, 2, solver, args, loop)
    _ = x^
    _ = w^
    _ = h^
    _loop(loop, it, "NMF " + (String("mu") if solver == 1 else String("cd")) + _column(device))


def _ica(algorithm: Int, device: Bool) raises:
    # X1: 2 whitened rows of 6 samples; W: the 2 x 2 start
    var x1: List[Float32] = [1.0, -0.5, 0.25, -1.0, 0.5, -0.25, 0.3, 0.9, -1.1, 0.2, -0.4, 0.1]
    var w: List[Float32] = [0.8, 0.6, -0.6, 0.8]
    var loop = FitLoop.empty()
    var it = -1
    # tol 0: the limit is never below it, every iteration runs
    if device:
        # the host column's build never elaborates a device fit
        comptime if not CPU_COLUMN_BUILD:
            it = ica_solve_dev_run(_p(x1), _p(w), 2, 2, 6, algorithm, 0, K, 1.0, 0.0, loop)
    else:
        it = ica_solve_run[HostExec](_p(x1), _p(w), 2, 2, 6, algorithm, 0, K, 1.0, 0.0, loop)
    _ = x1^
    _ = w^
    _loop(loop, it, "FastICA " + (String("parallel") if algorithm == 0 else String("deflation"))
          + _column(device))


def _fa(device: Bool) raises:
    var a = _source()
    var psi = List[Float32](length=3, fill=Float32(1.0))
    var w = List[Float32](length=3, fill=Float32(0.0))
    var ll = List[Float64](length=K, fill=0.0)
    var pl = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(ll.unsafe_ptr()))
    # tol -inf: the log-likelihood gain is never below it
    var args = FaArgs(6, 3, 1, K, -inf[DType.float64](), 0.0)
    var loop = FitLoop.empty()
    var it = -1
    if device:
        # the host column's build never elaborates a device fit
        comptime if not CPU_COLUMN_BUILD:
            it = fa_em_main_dev_run(_p(a), _p(psi), _p(w), pl, 6, args, loop)
    else:
        it = fa_em_main_run[HostExec](_p(a), _p(psi), _p(w), pl, 6, args, loop)
    _ = a^
    _ = psi^
    _ = w^
    _ = ll^
    _loop(loop, it, "FactorAnalysis" + _column(device))


def _als(use_cg: Bool, device: Bool) raises:
    # 6 users x 3 items of confidences, 2 factors
    var c = _source()
    var x = List[Float32](length=12, fill=Float32(0.01))
    var y = List[Float32](length=6, fill=Float32(0.02))
    var losses = List[Float64](length=K, fill=0.0)
    var pl = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(losses.unsafe_ptr()))
    var args = AlsArgs(K, use_cg, 3, False, 0.01)
    var loop = FitLoop.empty()
    if device:
        # the host column's build never elaborates a device fit
        comptime if not CPU_COLUMN_BUILD:
            _ = als_fit_dev_run(_p(c), _p(x), _p(y), pl, 6, 3, 2, args, loop)
    else:
        _ = als_fit_run[HostExec](_p(c), _p(x), _p(y), pl, 6, 3, 2, args, loop)
    _ = c^
    _ = x^
    _ = y^
    _ = losses^
    # ALS has no n_iter_: every one of the `iterations` sweeps runs
    _loop(loop, loop.iterations, "AlternatingLeastSquares " + (String("cg") if use_cg else String("exact"))
          + _column(device))


def run_checks() raises:
    for device_pass in range(2):  # small-loop(two columns: host, then the device in a device build)
        var device = device_pass == 1
        comptime if CPU_COLUMN_BUILD:
            if device:
                continue
        _nmf(1, device)
        _nmf(0, device)
        _ica(0, device)
        _ica(1, device)
        _fa(device)
        _als(False, device)
        _als(True, device)
