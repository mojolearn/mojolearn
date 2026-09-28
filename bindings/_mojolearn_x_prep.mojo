# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PREP LANE'S GPU BINDING (preprocessing additions, naive Bayes and
discriminant analysis). One entry runs a program of units on the device
(x_prep/common.mojo); the host binding runs the same units on the CPU."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.vendor import COMPILED_VENDOR
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_prep.device import run_program_device


def run_binding(arena_addr: PythonObject, arena_len: PythonObject, prog_addr: PythonObject,
                stages: PythonObject) raises -> PythonObject:
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    if fa == 0 or qa == 0 or n < 0 or s < 0:
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_device(fa, n, qa, s)
    return PythonObject(s)


def run_scratch_binding(arena_addr: PythonObject, arena_len: PythonObject, scratch_len: PythonObject,
                        prog_addr: PythonObject, stages: PythonObject) raises -> PythonObject:
    """x_prep_run with scratch_len device-only words after the arena (lane prep-apple2)."""
    var fa = Int(py=arena_addr)
    var n = Int(py=arena_len)
    var sc = Int(py=scratch_len)
    var qa = Int(py=prog_addr)
    var s = Int(py=stages)
    if fa == 0 or qa == 0 or n < 0 or sc < 0 or s < 0:
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_device(fa, n, qa, s, sc)
    return PythonObject(s)


def run_out_binding(arena_addr: PythonObject, prog_addr: PythonObject, out_addr: PythonObject,
                    sizes: PythonObject) raises -> PythonObject:
    """x_prep_run_scratch plus one OUTPUT region after the scratch, zeroed on
    the device and copied back into the host buffer at out_addr (lane
    prep-apple2). sizes = (arena_len, scratch_len, out_len, stages)."""
    var fa = Int(py=arena_addr)
    var qa = Int(py=prog_addr)
    var oa = Int(py=out_addr)
    var n = Int(py=sizes[0])
    var sc = Int(py=sizes[1])
    var on = Int(py=sizes[2])
    var s = Int(py=sizes[3])
    if fa == 0 or qa == 0 or n < 0 or sc < 0 or on < 0 or s < 0 or (on > 0 and oa == 0):
        raise Error("x_prep: invalid program buffers")
    with GILReleased(Python()):
        run_program_device(fa, n, qa, s, sc, oa, on)
    return PythonObject(s)


def numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def vendor_binding() raises -> PythonObject:
    return PythonObject(String(COMPILED_VENDOR))


@export
def PyInit__mojolearn_x_prep() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_prep")
        m.def_function[run_binding]("x_prep_run")
        m.def_function[run_scratch_binding]("x_prep_run_scratch")
        m.def_function[run_out_binding]("x_prep_run_out")
        m.def_function[numeric_mode_binding]("x_prep_numeric_mode")
        m.def_function[vendor_binding]("x_prep_vendor")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_prep: ", e))
