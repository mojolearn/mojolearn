# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the public optimizers on the CPU binding
(`_mojolearn_training_host`; lane neural-optimizers, 2026-10-21; lane
optimizers-thin-glue): the GPU door's entries (bindings/optimizer_public_pyapi.mojo)
over training/host/optimizer_public_host.mojo, through the shared door
bindings/optimizer_public_door.mojo (`HostOptKernels`)."""
from std.python import PythonObject
from std.python._cpython import PyObjectPtr

from bindings.optimizer_public_door import (
    entry_accumulate, entry_clip, entry_io, entry_load_state, entry_state, entry_state_set,
    entry_step,
)
from training.host.optimizer_public_host import (
    host_close, host_download, host_open, host_upload,
)



def optimizer_door_step_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_step(opt, grads, max_norm, op, tokens) on the host."""
    return entry_step(py_self, packed)


def optimizer_door_io_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_io(opt, op, tensors, name, tensor) on the host."""
    return entry_io(py_self, packed)


def optimizer_door_state_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_state(opt, which) on the host."""
    return entry_state(py_self, packed)


def optimizer_door_state_set_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_state_set(opt, which, value) on the host."""
    return entry_state_set(py_self, packed)


def optimizer_door_load_state_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_load_state(opt, t, ea, c1, es, c2, bi, c3) on the host."""
    return entry_load_state(py_self, packed)


def optimizer_door_clip_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_clip(grads, max_norm, norm_type, error_if_nonfinite, kwargs) on the host."""
    return entry_clip(py_self, packed)


def optimizer_door_accumulate_host_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_accumulate(microbatch_grads, tokens) on the host."""
    return entry_accumulate(py_self, packed)
