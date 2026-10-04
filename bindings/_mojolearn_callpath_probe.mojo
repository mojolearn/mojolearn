# SPDX-License-Identifier: Apache-2.0
"""Test-only C1..C4 binding. No estimator dispatch and no timing."""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from checks.numerics import GLOBAL_NUMERIC_MODE
from std.sys.info import has_apple_gpu_accelerator
from experiments.apple_callpath.resident_slot import CALLPATH_ENABLED
from bench.apple_callpath_quality import main as fixture


def mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def vendor_binding() raises -> PythonObject:
    comptime if has_apple_gpu_accelerator():
        return PythonObject("apple")
    return PythonObject("other")


def reach_binding() raises -> PythonObject:
    comptime if CALLPATH_ENABLED:
        return PythonObject(1)
    return PythonObject(0)


def quality_binding() raises -> PythonObject:
    comptime if CALLPATH_ENABLED:
        with GILReleased(Python()):
            fixture()
        return PythonObject(1)
    return PythonObject(0)


def PyInit__mojolearn_callpath_probe() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_callpath_probe")
        m.def_function[mode_binding]("numeric_mode")
        m.def_function[vendor_binding]("vendor")
        m.def_function[reach_binding]("reach")
        m.def_function[quality_binding]("run_quality")
        return m.finalize()
    except e:
        abort(String("failed to create callpath probe: ", e))
