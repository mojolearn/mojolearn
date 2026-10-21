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
from sequence.pyapi import py_f64s, py_ints


def _py_list(xs: List[Int]) raises -> PythonObject:
    """A Python list of the ints. `PythonObject([a, b])` does not build a
    list of two ints (the caller got a bare int back and its `h, used =
    ...` unpack raised TypeError)."""
    var res = Python.list()
    for i in range(len(xs)):  # small-loop(xs: an open's [handle, mask(, zeroed)]): result conversion, no data
        res.append(PythonObject(xs[i]))
    return res


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


def opt_resident_close_py(handle: PythonObject) raises -> PythonObject:
    """Free a handle's slots. Returns the handle."""
    return PythonObject(opt_resident_close(Int(py=handle)))


def opt_resident_move_py(handle: PythonObject, slot: PythonObject, addr: PythonObject,
                         up: PythonObject) raises -> PythonObject:
    """Slot `slot` to (up = 0) or from (up = 1) the host array at `addr`.
    Returns 1, or 0 for a slot the handle does not keep."""
    return PythonObject(opt_resident_move(Int(py=handle), Int(py=slot), Int(py=addr), Int(py=up) != 0))


def opt_resident_step_py(handle: PythonObject, addrs: PythonObject, ip: PythonObject,
                         fp: PythonObject) raises -> PythonObject:
    """`opt_resident_step`'s arguments as Python lists. Returns n."""
    return PythonObject(opt_resident_step(Int(py=handle), py_ints(addrs), py_ints(ip), py_f64s(fp)))


def lamb_resident_step_py(handle: PythonObject, addrs: PythonObject, ip: PythonObject,
                          fp: PythonObject) raises -> PythonObject:
    """`lamb_resident_step`'s arguments as Python lists. Returns n."""
    return PythonObject(lamb_resident_step(Int(py=handle), py_ints(addrs), py_ints(ip), py_f64s(fp)))


def adafactor_resident_step_py(handle: PythonObject, addrs: PythonObject, ip: PythonObject,
                               fp: PythonObject) raises -> PythonObject:
    """`adafactor_resident_step`'s arguments as Python lists. Returns n."""
    return PythonObject(adafactor_resident_step(Int(py=handle), py_ints(addrs), py_ints(ip), py_f64s(fp)))


def opt_resident_step_dev_py(handle: PythonObject, handles: PythonObject, ip: PythonObject,
                             fp: PythonObject) raises -> PythonObject:
    """`opt_resident_step_dev`'s arguments as Python lists. Returns n."""
    return PythonObject(opt_resident_step_dev(Int(py=handle), py_ints(handles), py_ints(ip), py_f64s(fp)))


def lamb_resident_step_dev_py(handle: PythonObject, handles: PythonObject, ip: PythonObject,
                              fp: PythonObject) raises -> PythonObject:
    """`lamb_resident_step_dev`'s arguments as Python lists. Returns n."""
    return PythonObject(lamb_resident_step_dev(Int(py=handle), py_ints(handles), py_ints(ip), py_f64s(fp)))


def adafactor_resident_step_dev_py(handle: PythonObject, handles: PythonObject, ip: PythonObject,
                                   fp: PythonObject) raises -> PythonObject:
    """`adafactor_resident_step_dev`'s arguments as Python lists. Returns n."""
    return PythonObject(adafactor_resident_step_dev(Int(py=handle), py_ints(handles), py_ints(ip), py_f64s(fp)))
