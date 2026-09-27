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
from sequence.pyapi import opt_step_py, rnn_fit_py, rnn_n_params_py, rnn_predict_py


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
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_sequence: ", e))
