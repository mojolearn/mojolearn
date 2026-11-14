# SPDX-License-Identifier: Apache-2.0
"""Program.fit's stopping criterion on the device, moved from
test_program_fit_stop.py (2026-11-14), through the typed entries the public
door calls (autograd/program_entries.mojo program_fit_parameters and
program_fit_native, with the door's words: loss 'mse' = 1, Huber delta 1.0,
shuffle on, seed 3, batch 8, condition and otherwise 0).

x (40, 3) seeded, y = x @ [1, -2, 0.5]; w = zeros (3, 1); the program
captures x -> x @ w; Adam lr 0.05 (Session.optimizer defaults otherwise).
40 rows in batches of 8 are 5 steps an epoch, so several epochs are enqueued
between two reads of the loop's status.

  0  tol 1e3, n_iter_no_change 2, 12 epochs: stops after epoch 3 (curve of
     3), and the loss curve, the weights and the weights after one more
     epoch (the optimizer continues from the stopping epoch's state) equal
     a 3-epoch fit without tol word for word.
  1  4 epochs without tol and 4 epochs with tol -1e30, n_iter_no_change 1:
     both run every epoch.
"""
from autograd.ops_api import OpsSession, ops_session, ops_session_open, ops_session_close
from autograd.optimizer_requests import OpsOptimizerRequest
from autograd.program_entries import program_fit_parameters, program_fit_native
from autograd.program_io import ProgramInput
from autograd.rankn import ShapeN
from core.cast_in import CastIn
from sequence.exec_trait import Exec
from sequence.ops import FP
from checks.native_tests.ops_higher_cases import require, tensor, op2, same_words
from checks.native_tests.core_ops_common import normal, dims2

comptime CORE_PROGRAM_FIT_STOP_FIXTURES = 2  # check_tol_stop, check_every_epoch

comptime _MATMUL = 1
comptime _LOSS_MSE = 1


struct FitRun(Movable):
    var ran: Int
    var curve: List[Float32]
    var after: List[Float32]
    var continued: List[Float32]

    def __init__(out self, ran: Int, var curve: List[Float32], var after: List[Float32], var continued: List[Float32]):
        self.ran = ran
        self.curve = curve^
        self.after = after^
        self.continued = continued^


def _data() -> Tuple[List[Float32], List[Float32]]:
    var x = normal(120, 60)
    var y = List[Float32](length=40, fill=0)
    for i in range(40):
        y[i] = x[3 * i] * Float32(1.0) + x[3 * i + 1] * Float32(-2.0) + x[3 * i + 2] * Float32(0.5)
    return (x^, y^)


def _owner(values: List[Float32], rows: Int, cols: Int) raises -> ProgramInput:
    return ProgramInput(CastIn.f32_c(Int(values.unsafe_ptr()), rows, cols), ShapeN.of(dims2(rows, cols)),
                        List[Float64](), List[Int64]())


def _fit_once[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, opt: Int, x: List[Float32], y: List[Float32], epochs: Int,
    tol: Float32, n_iter_no_change: Int, mut curve: List[Float32],
) raises -> Int:
    var params = program_fit_parameters[E, KIND](sid, pid, 0, False, List[Int]())
    var owners = List[ProgramInput]()
    owners.append(_owner(x, 40, 3))
    var targets = List[ProgramInput]()
    targets.append(_owner(y, 40, 1))
    var branches = program_fit_native[E, KIND](
        sid, pid, owners^, targets^, _LOSS_MSE, Float32(1.0), opt, params, epochs, 8, True, UInt64(3),
        FP(unsafe_from_address=Int(curve.unsafe_ptr())), 0, 0, tol, n_iter_no_change,
    )
    _ = x
    _ = y
    return branches[2]


def fit_run[E: Exec & Movable & Deinitable, KIND: Int](
    var ex: E, epochs: Int, tol: Float32, n_iter_no_change: Int
) raises -> FitRun:
    """The former `_fit`: one session, a fit of `epochs`, then one more epoch."""
    var data = _data()
    var x = data[0].copy()
    var y = data[1].copy()
    var sid = ops_session_open[E, KIND](ex^, False)
    try:
        var s = ops_session[E, KIND](sid)
        var a = tensor(s[], x, dims2(40, 3))
        var w = tensor(s[], List[Float32](length=3, fill=0), dims2(3, 1))
        var ins: List[Int] = [a]
        var outs: List[Int] = [op2(s[], _MATMUL, a, w)]
        var pid = s[].capture(ins, outs)
        var opt = s[].optimizer(OpsOptimizerRequest(1, False, 0, 1, Float32(0.05), Float32(0.0), Float32(0.9),
                                                    Float32(0.999), Float32(1e-8), Float32(0.0), Float32(0.1), Float32(0.0)))
        var curve = List[Float32](length=epochs, fill=Float32(-1.0))
        var ran = _fit_once[E, KIND](sid, pid, opt, x, y, epochs, tol, n_iter_no_change, curve)
        var ran_curve = List[Float32](length=ran, fill=0)
        for k in range(ran):
            ran_curve[k] = curve[k]
        var after = List[Float32](length=3, fill=0)
        s[].download(w, FP(unsafe_from_address=Int(after.unsafe_ptr())))
        var one = List[Float32](length=1, fill=Float32(-1.0))
        _ = _fit_once[E, KIND](sid, pid, opt, x, y, 1, Float32(0.0), 0, one)
        var continued = List[Float32](length=3, fill=0)
        s[].download(w, FP(unsafe_from_address=Int(continued.unsafe_ptr())))
        ops_session_close[E, KIND](sid)
        return FitRun(ran, ran_curve^, after^, continued^)
    except error:
        ops_session_close[E, KIND](sid)
        raise error


def check_tol_stop(stopped: FitRun, ref_: FitRun) raises:
    """Fixture 0: `stopped` = fit_run(12 epochs, tol 1e3, n_iter_no_change 2),
    `ref_` = fit_run(3 epochs, no tol)."""
    require(stopped.ran == 3 and len(stopped.curve) == 3, "tol stops after epoch 3, ran " + String(stopped.ran))
    require(ref_.ran == 3, "a 3-epoch fit runs 3 epochs")
    same_words(stopped.curve, ref_.curve, "the loss curve of the stopped fit")
    same_words(stopped.after, ref_.after, "the weights at the stop")
    same_words(stopped.continued, ref_.continued, "the weights after one more epoch")


def check_every_epoch(plain: FitRun, never: FitRun) raises:
    """Fixture 1: `plain` = fit_run(4, no tol), `never` = fit_run(4, tol -1e30, 1)."""
    require(plain.ran == 4 and len(plain.curve) == 4, "no tol runs every epoch")
    require(never.ran == 4 and len(never.curve) == 4, "tol -1e30 runs every epoch")
