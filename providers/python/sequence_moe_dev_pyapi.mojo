# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entry of the MoE forward on resident tensors
(`sequence/moe_dev.mojo`): PythonObject arguments to typed lists and the
typed result back. Nothing else."""
from std.python import PythonObject

from sequence.moe_dev import moe_forward_dev
from providers.python.sequence_pyapi import py_ints
from providers.python.sequence_seq_tensor_pyapi import (
    seq_tensor_alloc_slots, seq_tensor_free_made, seq_tensor_handles_list,
)


def moe_forward_dev_py(handles: PythonObject, ip: PythonObject) raises -> PythonObject:
    """`moe_forward_dev`'s arguments as Python lists. Returns T * D."""
    return PythonObject(moe_forward_dev(py_ints(handles), py_ints(ip)))


def moe_forward_dev_alloc_py(handles: PythonObject, ip: PythonObject, sizes: PythonObject) raises -> PythonObject:
    """`moe_forward_dev_py` that first allocates its four outputs (y,
    logits, sel, w) in one native step (lane ne3-c-blocks-embedding, row
    N163): a handle slot whose `sizes` word is positive is a new resident
    tensor of that many floats. Returns the handle list the forward ran on.
    The launches and the values are `moe_forward_dev`'s: no bit moves. On a
    refusal the tensors made here are released."""
    var hs = py_ints(handles)
    var made = seq_tensor_alloc_slots(hs, py_ints(sizes))
    try:
        _ = moe_forward_dev(hs, py_ints(ip))
    except error:
        seq_tensor_free_made(made)
        raise error^
    return seq_tensor_handles_list(hs)
