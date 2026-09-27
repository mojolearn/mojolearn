# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""THE PREP LANE'S GPU BINDING, carrying the proof dummy (lane/algos-prep,
2026-09-27; the dummy is removed before merge)."""
from bindings.hostptr import f32_ptr, read_f32, copy_f32
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_prep.dummy_device import l1_mean_fit_device, scale_device


def l1_mean_binding(x_addr: PythonObject, out_addr: PythonObject, params: PythonObject) raises -> PythonObject:
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    if n <= 0 or d <= 0:
        raise Error("x_prep: positive dimensions required")
    var x = read_f32(Int(py=x_addr), n * d)
    var output = f32_ptr(Int(py=out_addr))
    with GILReleased(Python()):
        var result = l1_mean_fit_device(x, n, d)
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
        var result = scale_device(x, s, n, d)
        copy_f32(result.unsafe_ptr(), output, n * d)
    return PythonObject(n * d)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_x_prep() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_prep")
        m.def_function[l1_mean_binding]("x_prep_l1_mean")
        m.def_function[scale_binding]("x_prep_scale")
        m.def_function[numeric_mode_binding]("x_prep_numeric_mode")
        m.def_function[vendor_binding]("x_prep_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_prep: ", e))
