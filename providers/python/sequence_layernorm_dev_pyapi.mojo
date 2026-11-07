# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entry of LayerNorm on resident tensors
(`sequence/layernorm_dev.mojo`): PythonObject arguments to typed lists and
the typed result back. Nothing else."""
from std.python import PythonObject

from sequence.layernorm_dev import layer_norm_dev
from providers.python.sequence_pyapi import py_f64s, py_ints
from providers.python.sequence_seq_tensor_pyapi import (
    seq_tensor_alloc_slots, seq_tensor_free_made, seq_tensor_handles_list,
)


def layer_norm_dev_py(handles: PythonObject, addrs: PythonObject, ip: PythonObject,
                      fp: PythonObject) raises -> PythonObject:
    """`layer_norm_dev`'s arguments as Python lists. Returns M * D."""
    return PythonObject(layer_norm_dev(py_ints(handles), py_ints(addrs), py_ints(ip), py_f64s(fp)))


def layer_norm_dev_alloc_py(handles: PythonObject, addrs: PythonObject, ip: PythonObject,
                            fp: PythonObject, sizes: PythonObject) raises -> PythonObject:
    """`layer_norm_dev_py` that first allocates its output tensors (lane
    ne3-c-blocks-embedding, row N162): a handle slot whose `sizes` word is
    positive is a new resident tensor of that many floats (y, and the kept
    mean and rstd). Returns the handle list the layer ran on. The launches
    and the values are `layer_norm_dev`'s: no bit moves. On a refusal the
    tensors made here are released."""
    var hs = py_ints(handles)
    var made = seq_tensor_alloc_slots(hs, py_ints(sizes))
    try:
        _ = layer_norm_dev(hs, py_ints(addrs), py_ints(ip), py_f64s(fp))
    except error:
        seq_tensor_free_made(made)
        raise error^
    return seq_tensor_handles_list(hs)
