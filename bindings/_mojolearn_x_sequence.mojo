# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""GPU binding of the sequence expansion lane (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md,
Lane 6). Every entry is `sequence/pyapi.mojo`'s, run on a `DeviceExec`; the
CPU host binding `bindings/_mojolearn_x_sequence_host.mojo` exports the same
names over a `HostExec`, the same element bodies (`sequence/ops.mojo`)."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from sequence.exec_device import DeviceExec
from sequence.pyapi import opt_step_py, rnn_fit_py, rnn_n_params_py, rnn_predict_py, stl_py, var_fit_py, var_forecast_py, mlp_fit_py, mlp_predict_py, adafactor_step_py, lamb_step_py, layer_norm_py, theta_py, croston_py, ets_py, garch_py, prophet_fit_py, prophet_predict_py, moe_forward_py


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


def garch_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return garch_py(ex, addrs, ip)


def prophet_fit_binding(addrs: PythonObject, ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return prophet_fit_py(ex, addrs, ip, fp)


def prophet_predict_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return prophet_predict_py(ex, addrs, ip)


def moe_forward_binding(addrs: PythonObject, ip: PythonObject) raises -> PythonObject:
    var ex = DeviceExec()
    return moe_forward_py(ex, addrs, ip)


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
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_sequence: ", e))
