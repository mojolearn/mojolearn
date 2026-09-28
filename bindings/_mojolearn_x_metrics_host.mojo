# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for `_mojolearn_x_metrics`. HOST ONLY: the same units as the
device, run in a loop on the caller's arena (x_metrics/host/program.mojo), with
the GPU binding's export names and address contract."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_metrics.common import X_METRICS_HOST_SABOTAGE, IP
from x_metrics.host.program import run_program_host


def run_binding(arena_addr: PythonObject, arena_len: PythonObject, prog_addr: PythonObject,
                stages: PythonObject) raises -> PythonObject:
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    if fa == 0 or qa == 0 or n < 0 or s < 0:
        raise Error("x_metrics: invalid program buffers")
    with GILReleased(Python()):
        run_program_host(fa, n, qa, s)
    return PythonObject(s)


def run_out_binding(arena_addr: PythonObject, arena_len: PythonObject, prog_addr: PythonObject,
                    stages: PythonObject, outs_addr: PythonObject, nouts: PythonObject) raises -> PythonObject:
    """The device binding's `x_metrics_run_out` (lane metrics-apple2): the
    host runs in the caller's arena, so every word is already there and the
    output ranges only need checking."""
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    var oa = Int(py=outs_addr)
    var no = Int(py=nouts)
    if fa == 0 or qa == 0 or oa == 0 or n < 0 or s < 0 or no < 0:
        raise Error("x_metrics: invalid program buffers")
    for k in range(no):
        var lo = Int(IP(unsafe_from_address=oa).unsafe_load(2 * k))
        var hi = Int(IP(unsafe_from_address=oa).unsafe_load(2 * k + 1))
        if lo < 0 or hi < lo or hi > n:
            raise Error("x_metrics: output range outside the arena")
    with GILReleased(Python()):
        run_program_host(fa, n, qa, s)
    return PythonObject(s)


def x_metrics_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def x_metrics_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_metrics_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, "x_metrics host: pass -D MOJOLEARN_COLUMN_CPU"
    return PythonObject(column_name(TARGET_COLUMN))


def x_metrics_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(X_METRICS_HOST_SABOTAGE)


def x_metrics_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def x_metrics_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


@export
def PyInit__mojolearn_x_metrics_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_metrics_host")
        m.def_function[x_metrics_host_numeric_mode_binding]("x_metrics_host_numeric_mode")
        m.def_function[x_metrics_host_vendor_binding]("x_metrics_host_vendor")
        m.def_function[x_metrics_host_column_binding]("x_metrics_host_column")
        m.def_function[x_metrics_host_sabotage_binding]("x_metrics_host_sabotage")
        m.def_function[run_binding]("x_metrics_run")
        m.def_function[run_out_binding]("x_metrics_run_out")
        m.def_function[x_metrics_numeric_mode_binding]("x_metrics_numeric_mode")
        m.def_function[x_metrics_vendor_binding]("x_metrics_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_metrics_host: ", e))
