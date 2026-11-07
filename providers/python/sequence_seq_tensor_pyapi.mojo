# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entries of the resident sequence tensors
(`sequence/seq_tensor.mojo`): PythonObject arguments to typed handles,
addresses and counts, and the typed result back. Nothing else."""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call

from sequence.seq_tensor import (
    seq_tensor_alloc,
    seq_tensor_copy,
    seq_tensor_download,
    seq_tensor_free,
    seq_tensor_upload,
)


def seq_tensor_alloc_py(n_obj: PythonObject, zero_obj: PythonObject) raises -> PythonObject:
    """A new resident tensor of n >= 1 floats; zero = 1 fills it with +0.0
    on the device. Returns the handle (>= 1)."""
    return PythonObject(seq_tensor_alloc(Int(py=n_obj), Int(py=zero_obj) != 0))


def seq_tensor_free_py(handle: PythonObject) raises -> PythonObject:
    """Release a handle. Returns it."""
    return PythonObject(seq_tensor_free(Int(py=handle)))


def seq_tensor_upload_py(handle: PythonObject, addr: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    """The caller's n floats at `addr` into the handle's first n cells."""
    return PythonObject(seq_tensor_upload(Int(py=handle), Int(py=addr), Int(py=n_obj)))


def seq_tensor_download_py(handle: PythonObject, addr: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    """The handle's first n cells into the caller's array at `addr`."""
    return PythonObject(seq_tensor_download(Int(py=handle), Int(py=addr), Int(py=n_obj)))


def seq_tensor_copy_py(dst: PythonObject, src: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    """Device to device: src's first n cells into dst's."""
    return PythonObject(seq_tensor_copy(Int(py=dst), Int(py=src), Int(py=n_obj)))


def seq_tensor_alloc_slots(mut handles: List[Int], sizes: List[Int]) raises -> List[Int]:
    """Allocate the output slots of one device layer call (lane
    ne3-c-blocks-embedding, rows N162 / N163): every slot `i` with
    `sizes[i] > 0` gets a new resident tensor of that many floats (contents
    unspecified, as `seq_tensor_alloc(n, 0)`), written into `handles[i]`.
    Returns the handles made, for `seq_tensor_free_made` when the layer call
    fails. One native step for every output instead of one Python crossing
    per output."""
    if len(sizes) != len(handles):
        raise Error("seq_tensor_alloc_slots: one size per handle slot is required")
    var made = List[Int]()
    try:
        for i in range(len(handles)):  # small-loop(output slots of one device layer call, at most six)
            if sizes[i] > 0:
                handles[i] = seq_tensor_alloc(sizes[i], False)
                made.append(handles[i])
    except error:
        seq_tensor_free_made(made)
        raise error^
    return made^


def seq_tensor_free_made(made: List[Int]):
    """Release the handles `seq_tensor_alloc_slots` made (a failed call)."""
    for i in range(len(made)):  # small-loop(handles made by one device layer call, at most six)
        try:
            _ = seq_tensor_free(made[i])
        except:
            pass


def seq_tensor_handles_list(handles: List[Int]) raises -> PythonObject:
    """The handles as a Python list of ints (the fused entries' result)."""
    var out = Python.list()
    for i in range(len(handles)):  # small-loop(handle slots of one device layer call, at most six)
        _ = py_call(out.append, PythonObject(handles[i]))
    return out
