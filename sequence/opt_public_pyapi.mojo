# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python doors of the public sequence optimizers on the device binding
(sequence/opt_public.mojo, sequence/opt_public_device.mojo): PythonObject
arguments to typed addresses and words, one kernel entry, the count back.
The kernel writes its results into the optimizer's int64 state words; this
file assigns nothing on a Python object. Nothing else here."""
from std.python import PythonObject

from sequence.opt_public import seq_optimizer_init
from sequence.opt_public_device import (
    seq_adafactor_step_device,
    seq_lamb_step_device,
    seq_optimizer_close_device,
    seq_optimizer_export_device,
    seq_optimizer_import_device,
    seq_optimizer_state_device,
    seq_optimizer_step_device,
)
from sequence.pyapi import py_f64s, py_ints


def seq_optimizer_init_py(words: PythonObject, ip: PythonObject) raises -> PythonObject:
    return PythonObject(seq_optimizer_init(Int(py=words), py_ints(ip)))


def seq_optimizer_step_py(words: PythonObject, handles: PythonObject, addrs: PythonObject, ip: PythonObject,
                          fp: PythonObject) raises -> PythonObject:
    return PythonObject(seq_optimizer_step_device(Int(py=words), Int(py=handles), py_ints(addrs), py_ints(ip),
                                                  py_f64s(fp)))


def seq_lamb_step_py(words: PythonObject, handles: PythonObject, addrs: PythonObject, ip: PythonObject,
                     fp: PythonObject) raises -> PythonObject:
    return PythonObject(seq_lamb_step_device(Int(py=words), Int(py=handles), py_ints(addrs), py_ints(ip),
                                             py_f64s(fp)))


def seq_adafactor_step_py(words: PythonObject, handles: PythonObject, addrs: PythonObject, ip: PythonObject,
                          fp: PythonObject) raises -> PythonObject:
    return PythonObject(seq_adafactor_step_device(Int(py=words), Int(py=handles), py_ints(addrs), py_ints(ip),
                                                  py_f64s(fp)))


def seq_optimizer_state_py(words: PythonObject, handles: PythonObject, addrs: PythonObject) raises -> PythonObject:
    return PythonObject(seq_optimizer_state_device(Int(py=words), Int(py=handles), py_ints(addrs)))


def seq_optimizer_export_py(words: PythonObject, handles: PythonObject, addrs: PythonObject,
                            ip: PythonObject) raises -> PythonObject:
    return PythonObject(seq_optimizer_export_device(Int(py=words), Int(py=handles), py_ints(addrs), py_ints(ip)))


def seq_optimizer_import_py(words: PythonObject, handles: PythonObject, addrs: PythonObject,
                            ip: PythonObject) raises -> PythonObject:
    return PythonObject(seq_optimizer_import_device(Int(py=words), Int(py=handles), py_ints(addrs), py_ints(ip)))


def seq_optimizer_close_py(words: PythonObject, handles: PythonObject, nh: PythonObject) raises -> PythonObject:
    return PythonObject(seq_optimizer_close_device(Int(py=words), Int(py=handles), Int(py=nh)))
