# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_prep`, carrying the proof dummy
(lane/algos-prep, 2026-09-27; the dummy is removed before merge). HOST ONLY;
the GPU binding's export names and address contract."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from bindings.hostptr import copy_f32, f32_ptr, read_f32
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_prep.host.dummy_oracle import X_PREP_HOST_SABOTAGE, host_l1_mean_fit, host_scale


def l1_mean_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    if n <= 0 or d <= 0:
        raise Error("x_prep: positive dimensions required")
    var x = read_f32(Int(py=x_addr), n * d)
    var output = f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var result = host_l1_mean_fit(x, n, d)
        copy_f32(result.unsafe_ptr(), output, d)
    return PythonObject(d)


def scale_binding(x_addr: PythonObject, s_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    if n <= 0 or d <= 0:
        raise Error("x_prep: positive dimensions required")
    var x = read_f32(Int(py=x_addr), n * d)
    var s = read_f32(Int(py=s_addr), d)
    var output = f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var result = host_scale(x, s, n, d)
        copy_f32(result.unsafe_ptr(), output, n * d)
    return PythonObject(n * d)


def x_prep_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_prep_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_prep_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_prep host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_prep_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_PREP_HOST_SABOTAGE)


def x_prep_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def x_prep_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


@export
def PyInit__mojolearn_x_prep_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_prep_host")
        m.def_function[x_prep_host_numeric_mode_binding]("x_prep_host_numeric_mode")
        m.def_function[x_prep_host_vendor_binding]("x_prep_host_vendor")
        m.def_function[x_prep_host_column_binding]("x_prep_host_column")
        m.def_function[x_prep_host_sabotage_binding]("x_prep_host_sabotage")
        m.def_function[l1_mean_binding]("x_prep_l1_mean")
        m.def_function[scale_binding]("x_prep_scale")
        m.def_function[x_prep_numeric_mode_binding]("x_prep_numeric_mode")
        m.def_function[x_prep_vendor_binding]("x_prep_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_prep_host: ", e))
