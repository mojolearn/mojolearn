# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU binding of the sequence expansion lane (algorithm
expansion lane 6). Every entry is `sequence/pyapi.mojo`'s, run on a `DeviceExec`; the
CPU host binding `bindings/_mojolearn_x_sequence_host.mojo` exports the same
names over a `HostExec`, the same element bodies (`sequence/ops.mojo`)."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.pyapi import ival, _getenv_seq, opt_step_py, rnn_fit_py, rnn_n_params_py, rnn_predict_py, stl_py, var_fit_py, var_forecast_py, mlp_fit_py, mlp_predict_py, adafactor_step_py, lamb_step_py, layer_norm_py, theta_py, croston_py, ets_py, garch_py, prophet_fit_py, prophet_predict_py, moe_forward_py
from sequence.pyapi import moe_forward_check, moe_forward_run, fptr
from sequence.moe_weights import moe_weights_put, moe_weights_ptrs, moe_weights_free


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


def rnn_fit_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return rnn_fit_py(ex, addrs, ip, fp)


def rnn_predict_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return rnn_predict_py(ex, addrs, ip)


def rnn_n_params_binding(ip: PythonObject) raises -> PythonObject:
    return rnn_n_params_py(ip)


def optimizer_step_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return opt_step_py(ex, addrs, ip, fp)


def stl_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return stl_py(ex, addrs, ip)


def var_fit_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return var_fit_py(ex, addrs, ip)


def var_forecast_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return var_forecast_py(ex, addrs, ip)


def mlp_fit_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return mlp_fit_py(ex, addrs, ip, fp)


def mlp_predict_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return mlp_predict_py(ex, addrs, ip)


def adafactor_step_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return adafactor_step_py(ex, addrs, ip, fp)


def lamb_step_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return lamb_step_py(ex, addrs, ip, fp)


def layer_norm_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return layer_norm_py(ex, addrs, ip, fp)


def theta_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return theta_py(ex, addrs, ip, fp)


def croston_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return croston_py(ex, addrs, ip)


def ets_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return ets_py(ex, addrs, ip, fp)


def garch_host_max() -> Int:
    """Batches of at most this many series run GARCH on a HostExec inside the
    GPU binding (lane neural-pass14): each series is one serial Nelder-Mead
    MLE (`sequence/garch.mojo::op_garch`), which one GPU thread runs far
    slower than one CPU core (the board's 64 x 1,440 race: 2 to 9 s on the
    GPUs against 60 ms for arch on the CPU), while the host executor runs
    the series over host tasks. The same element body, the host column's own
    statements, which the identity gates hold to the device column.
    MOJOLEARN_SEQ_GARCH_HOST_MAX overrides it (0: always the device)."""
    var v = String(_getenv_seq("MOJOLEARN_SEQ_GARCH_HOST_MAX", "4096"))
    try:
        return Int(v)
    except:
        return 4096


def garch_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    if len(ip) >= 1 and ival(ip, 0) <= garch_host_max():
        var hx = HostExec()
        return garch_py(hx, addrs, ip)
    var ex = DeviceExec()
    return garch_py(ex, addrs, ip)


def prophet_host_max() -> Int:
    """Batches of at most this many series run the Prophet fit on a HostExec
    inside the GPU binding (lane neural-pass23), as garch does above: each
    series is one serial optimizer (`sequence/prophet.mojo::op_prophet_fit`),
    which one GPU thread runs far slower than one CPU core (the board's 64 x
    1,440 race: 17 to 37 s on the GPUs against 0.25 to 0.9 s for prophet on
    the CPU), while the host executor runs the series over host tasks. The
    same element body, the host column's own statements, which the identity
    gates hold to the device column. MOJOLEARN_SEQ_PROPHET_HOST_MAX
    overrides it (0: always the device)."""
    var v = String(_getenv_seq("MOJOLEARN_SEQ_PROPHET_HOST_MAX", "4096"))
    try:
        return Int(v)
    except:
        return 4096


def prophet_fit_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    if len(ip) >= 1 and ival(ip, 0) <= prophet_host_max():
        var hx = HostExec()
        return prophet_fit_py(hx, addrs, ip, fp)
    var ex = DeviceExec()
    return prophet_fit_py(ex, addrs, ip, fp)


def prophet_predict_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return prophet_predict_py(ex, addrs, ip)


def moe_forward_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """ip = [T, D, F, E, k, renormalise] uploads the weights at addrs[1..3];
    a seventh entry > 0 is a `moe_weights_put` handle whose device copies
    are read instead (addrs[1..3] are then unread). Lane
    gap-neural-overhead2: the same launches on the same words."""
    if len(ip) == 7 and ival(ip, 6) > 0:
        var t = moe_forward_check(addrs, ip, 7)
        var w = moe_weights_ptrs(ival(ip, 6))
        var ex = DeviceExec()
        return moe_forward_run(ex, addrs, t[0], t[1], t[2], t[3], t[4], ival(ip, 5), w[0], w[1], w[2])
    var ex = DeviceExec()
    return moe_forward_py(ex, addrs, ip)


def moe_weights_put_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    """addrs = [router (E, D), gate_up (E, 2F, D), down (E, D, F)]; ip = [E, D, F]."""
    if len(addrs) != 3 or len(ip) != 3:
        raise Error("moe_weights_put: requires 3 addresses and [E, D, F]")
    return PythonObject(moe_weights_put(
        fptr(addrs[0], "router"), fptr(addrs[1], "gate_up_proj"), fptr(addrs[2], "down_proj"),
        ival(ip, 0), ival(ip, 1), ival(ip, 2),
    ))


def moe_weights_free_binding(h: PythonObject) raises -> PythonObject:
    moe_weights_free(Int(py=h))
    return PythonObject(0)


@export
def PyInit__mojolearn_x_sequence() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_sequence")
        m.def_function[numeric_mode_binding]("x_sequence_numeric_mode")
        m.def_function[vendor_binding]("x_sequence_vendor")
        m.def_function[rnn_fit_binding]("rnn_fit")
        m.def_function[rnn_predict_binding]("rnn_predict")
        m.def_function[rnn_n_params_binding]("rnn_n_params")
        m.def_function[optimizer_step_binding]("optimizer_step")
        m.def_function[stl_binding]("stl")
        m.def_function[var_fit_binding]("var_fit")
        m.def_function[var_forecast_binding]("var_forecast")
        m.def_function[mlp_fit_binding]("mlp_fit")
        m.def_function[mlp_predict_binding]("mlp_predict")
        m.def_function[adafactor_step_binding]("adafactor_step")
        m.def_function[lamb_step_binding]("lamb_step")
        m.def_function[layer_norm_binding]("layer_norm")
        m.def_function[theta_binding]("theta")
        m.def_function[croston_binding]("croston")
        m.def_function[ets_binding]("ets")
        m.def_function[garch_binding]("garch")
        m.def_function[prophet_fit_binding]("prophet_fit")
        m.def_function[prophet_predict_binding]("prophet_predict")
        m.def_function[moe_forward_binding]("moe_forward")
        m.def_function[moe_weights_put_binding]("moe_weights_put")
        m.def_function[moe_weights_free_binding]("moe_weights_free")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_sequence: ", e))
