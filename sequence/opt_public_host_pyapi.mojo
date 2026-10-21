# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python doors of the public sequence optimizers on the CPU host
binding (sequence/opt_public.mojo on a `HostExec`): the same entry names
and arguments as sequence/opt_public_pyapi.mojo, nothing resident.
PythonObject arguments to typed addresses and words, one kernel entry, the
count back. Nothing else here."""
from std.python import PythonObject

from sequence.exec import HostExec
from sequence.opt_public import (
    seq_adafactor_step_on,
    seq_lamb_step_on,
    seq_optimizer_close_on,
    seq_optimizer_export_on,
    seq_optimizer_import_on,
    seq_optimizer_init,
    seq_optimizer_state_on,
    seq_optimizer_step_on,
)
from sequence.pyapi import py_f64s, py_ints


def seq_optimizer_init_host_py(words: PythonObject, ip: PythonObject) raises -> PythonObject:
    return PythonObject(seq_optimizer_init(Int(py=words), py_ints(ip)))


def seq_optimizer_step_host_py(words: PythonObject, handles: PythonObject, addrs: PythonObject, ip: PythonObject,
                               fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return PythonObject(seq_optimizer_step_on(ex, Int(py=words), py_ints(addrs), py_ints(ip), py_f64s(fp)))


def seq_lamb_step_host_py(words: PythonObject, handles: PythonObject, addrs: PythonObject, ip: PythonObject,
                          fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return PythonObject(seq_lamb_step_on(ex, Int(py=words), py_ints(addrs), py_ints(ip), py_f64s(fp)))


def seq_adafactor_step_host_py(words: PythonObject, handles: PythonObject, addrs: PythonObject, ip: PythonObject,
                               fp: PythonObject) raises -> PythonObject:
    var ex = HostExec()
    return PythonObject(seq_adafactor_step_on(ex, Int(py=words), py_ints(addrs), py_ints(ip), py_f64s(fp)))


def seq_optimizer_state_host_py(words: PythonObject, handles: PythonObject, addrs: PythonObject) raises -> PythonObject:
    return PythonObject(seq_optimizer_state_on(Int(py=words)))


def seq_optimizer_export_host_py(words: PythonObject, handles: PythonObject, addrs: PythonObject,
                                 ip: PythonObject) raises -> PythonObject:
    return PythonObject(seq_optimizer_export_on(Int(py=words), py_ints(addrs), py_ints(ip)))


def seq_optimizer_import_host_py(words: PythonObject, handles: PythonObject, addrs: PythonObject,
                                 ip: PythonObject) raises -> PythonObject:
    return PythonObject(seq_optimizer_import_on(Int(py=words), py_ints(addrs), py_ints(ip)))


def seq_optimizer_close_host_py(words: PythonObject, handles: PythonObject, nh: PythonObject) raises -> PythonObject:
    return PythonObject(seq_optimizer_close_on(Int(py=words)))
