# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ADDRESS CONTRACT of the sequence lane's bindings, written once over
`Exec`: `bindings/_mojolearn_x_sequence.mojo` calls these with a
`DeviceExec`, `bindings/_mojolearn_x_sequence_host.mojo` with a `HostExec`.
Every buffer is a caller-owned C-contiguous array handed over by address;
nothing is retained after the call."""
from std.python import PythonObject

from std.os import getenv as _getenv_seq
from sequence.exec_trait import Exec
from sequence.ops import FP
from sequence.ops_impl import (
    ETS_FAST_STALL_ITERS,
    ETS_FAST_STALL_REL,
    GARCH_FAST_STALL_ITERS,
    GARCH_FAST_STALL_REL,
    PROPHET_FAST_MIN_N,
    adafactor_core,
    adafactor_step_impl,
    croston_forecast_impl,
    croston_impl,
    ets_impl,
    garch_impl,
    lamb_bias,
    lamb_core,
    lamb_step_impl,
    lamb_table,
    layer_norm_impl,
    mlp_fit_impl,
    mlp_predict_impl,
    moe_backward_impl,
    moe_backward_run_impl,
    moe_forward_core,
    moe_forward_impl,
    moe_forward_run_impl,
    opt_slots,
    opt_step_impl,
    prophet_fit_impl,
    prophet_predict_impl,
    rnn_fit_impl,
    rnn_predict_impl,
    stl_impl,
    theta_impl,
    var_fit_impl,
    var_forecast_impl,
)


def fptr(addr: PythonObject, what: String) raises -> FP:
    var a = Int(py=addr)
    if a == 0:
        raise Error("sequence: null float32 buffer for " + what)
    return FP(unsafe_from_address=a)


def iptr(addr: PythonObject, what: String) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    var a = Int(py=addr)
    if a == 0:
        raise Error("sequence: null int32 buffer for " + what)
    return MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=a)


def ival(p: PythonObject, i: Int) raises -> Int:
    return Int(py=p[i])


def fval(p: PythonObject, i: Int) raises -> Float32:
    return Float32(Float64(py=p[i]))


def py_ints(p: PythonObject) raises -> List[Int]:
    """A Python list of integers (addresses, handles, parameters) as typed
    Ints for a kernel entry; a conversion only, no data."""
    var out = List[Int]()
    for i in range(len(p)):  # small-loop(p: entry arguments): argument conversion, no data
        out.append(Int(py=p[i]))
    return out^


def py_f64s(p: PythonObject) raises -> List[Float64]:
    """A Python list of floats as typed Float64s for a kernel entry; a
    conversion only, no data (an entry's Float32 is Float32 of this value,
    `fval`'s rounding)."""
    var out = List[Float64]()
    for i in range(len(p)):  # small-loop(p: entry arguments): argument conversion, no data
        out.append(Float64(py=p[i]))
    return out^


def rnn_fit_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `rnn_fit_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(rnn_fit_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))


def rnn_predict_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `rnn_predict_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(rnn_predict_impl(ex, py_ints(addrs), py_ints(ip)))


def opt_step_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `opt_step_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(opt_step_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))


def stl_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `stl_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(stl_impl(ex, py_ints(addrs), py_ints(ip)))


def var_fit_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `var_fit_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(var_fit_impl(ex, py_ints(addrs), py_ints(ip)))


def var_forecast_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `var_forecast_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(var_forecast_impl(ex, py_ints(addrs), py_ints(ip)))


def mlp_fit_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `mlp_fit_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(mlp_fit_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))


def mlp_predict_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `mlp_predict_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(mlp_predict_impl(ex, py_ints(addrs), py_ints(ip)))


def lamb_step_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `lamb_step_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(lamb_step_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))


def layer_norm_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `layer_norm_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(layer_norm_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))


def theta_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `theta_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(theta_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))


def croston_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `croston_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(croston_impl(ex, py_ints(addrs), py_ints(ip)))


def croston_forecast_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `croston_forecast_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(croston_forecast_impl(ex, py_ints(addrs), py_ints(ip)))


def ets_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `ets_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(ets_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))


def garch_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `garch_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(garch_impl(ex, py_ints(addrs), py_ints(ip)))


def prophet_fit_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """The Python door of `prophet_fit_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(prophet_fit_impl(ex, py_ints(addrs), py_ints(ip), py_f64s(fp)))


def prophet_predict_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `prophet_predict_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(prophet_predict_impl(ex, py_ints(addrs), py_ints(ip)))


def moe_forward_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `moe_forward_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(moe_forward_impl(ex, py_ints(addrs), py_ints(ip)))


def moe_backward_py[E: Exec](mut ex: E, addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """The Python door of `moe_backward_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(moe_backward_impl(ex, py_ints(addrs), py_ints(ip)))


def moe_forward_check(addrs: PythonObject, ip: PythonObject, n_ip: Int) raises -> Tuple[Int, Int, Int, Int, Int]:
    """`moe_forward`'s argument checks: (T, D, F, E, k)."""
    if len(addrs) != 8 or len(ip) != n_ip:
        raise Error("moe_forward: requires 8 addresses and " + String(n_ip) + " integer parameters")
    var T = ival(ip, 0)
    var D = ival(ip, 1)
    var F = ival(ip, 2)
    var En = ival(ip, 3)
    var k = ival(ip, 4)
    if T < 1 or D < 1 or F < 1 or En < 1 or k < 1 or k > En:
        raise Error("moe_forward: T, D, F, E >= 1 and 1 <= k <= E")
    return (T, D, F, En, k)


def moe_forward_run[E: Exec](
    mut ex: E, addrs: PythonObject, T: Int, D: Int, F: Int, En: Int, k: Int, renorm: Int, Wg: FP, Gu: FP, Dn: FP,
) raises -> PythonObject:
    """The Python door of `moe_forward_run_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(moe_forward_run_impl(ex, py_ints(addrs), T, D, F, En, k, renorm, Wg, Gu, Dn))


def moe_backward_run[E: Exec](
    mut ex: E, addrs: PythonObject, T: Int, D: Int, F: Int, En: Int, k: Int, renorm: Int, Wg: FP, Gu: FP, Dn: FP,
) raises -> PythonObject:
    """The Python door of `moe_backward_run_impl` (sequence/ops_impl.mojo)."""
    return PythonObject(moe_backward_run_impl(ex, py_ints(addrs), T, D, F, En, k, renorm, Wg, Gu, Dn))
