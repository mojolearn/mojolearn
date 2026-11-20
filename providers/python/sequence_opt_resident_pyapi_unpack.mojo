# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entries of the resident optimizer state
(`sequence/opt_resident.mojo`): PythonObject arguments to typed handles,
address lists and parameter lists, and the typed results back. Nothing
else; every check and every launch is the kernel file's."""
from std.python import Python, PythonObject

from sequence.opt_resident import (
    adafactor_resident_open,
    adafactor_resident_step,
    adafactor_resident_step_dev,
    lamb_resident_open,
    lamb_resident_step,
    lamb_resident_step_dev,
    opt_resident_close,
    opt_resident_move,
    opt_resident_open,
    opt_resident_step,
    opt_resident_step_dev,
)
from providers.python.sequence_pyapi_door import py_f64s, py_ints


from providers.python.sequence_opt_resident_pyapi_pack import (
    _py_list,
)


def opt_resident_open_py(ip: PythonObject, fp: PythonObject) raises -> PythonObject:
    """ip = [n, kind, flags], fp = [lr, f1, f2, eps, weight_decay, f7].
    Returns [handle, used slot mask]."""
    return _py_list(opt_resident_open(py_ints(ip), py_f64s(fp)))


def lamb_resident_open_py(ip: PythonObject) raises -> PythonObject:
    """ip = [n_tensors, off_0, ..., off_n]. Returns [handle, 3]."""
    return _py_list(lamb_resident_open(py_ints(ip)))


def adafactor_resident_open_py(ip: PythonObject) raises -> PythonObject:
    """ip = [R, C]. Returns [handle, used mask]."""
    return _py_list(adafactor_resident_open(py_ints(ip)))
