# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entry of LayerNorm on resident tensors
(`sequence/layernorm_dev.mojo`): PythonObject arguments to typed lists and
the typed result back. Nothing else."""
from std.python import PythonObject

from sequence.layernorm_dev import layer_norm_dev
from providers.python.sequence_pyapi import py_f64s, py_ints


def layer_norm_dev_py(handles: PythonObject, addrs: PythonObject, ip: PythonObject,
                      fp: PythonObject) raises -> PythonObject:
    """`layer_norm_dev`'s arguments as Python lists. Returns M * D."""
    return PythonObject(layer_norm_dev(py_ints(handles), py_ints(addrs), py_ints(ip), py_f64s(fp)))
