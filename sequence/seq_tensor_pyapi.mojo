# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python entries of the resident sequence tensors
(`sequence/seq_tensor.mojo`): PythonObject arguments to typed handles,
addresses and counts, and the typed result back. Nothing else."""
from std.python import PythonObject

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
