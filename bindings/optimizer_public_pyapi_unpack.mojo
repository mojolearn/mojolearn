# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the public optimizers on the GPU binding
(`_mojolearn_training`; lane neural-optimizers, 2026-10-21; lane
optimizers-thin-glue): the kernels behind bindings/optimizer_public_door.mojo
(`GpuOptKernels`, each one typed entry of training/optimizer_public.mojo with
the GIL released) and the C entries of the public methods. Each public
method of `SGD` / `Adam` / `AdamW`, `clip_grad_norm_` and `accumulate_grads`
is one of these entries: the door reads the optimizer object and the
caller's tensors, calls ONE kernel entry here and assigns the result."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr

from core.neural_context import neural_ctx
from training.optimizer_public import (
    public_accumulate, public_clip, public_io, public_step, resident_close,
    resident_download, resident_get, resident_open, resident_put, resident_step, resident_step_io,
    resident_upload,
)
from training.optimizer_public_common import F32P, I32P
from training.optimizer_kernels_gpu import _NEURAL_CTX
from bindings.optimizer_public_door import (
    entry_accumulate, entry_clip, entry_io, entry_load_state, entry_state, entry_state_set,
    entry_step,
)


def optimizer_door_step_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_step(opt, grads, max_norm, op, tokens)."""
    return entry_step(py_self, packed)


def optimizer_door_io_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_io(opt, op, tensors, name, tensor)."""
    return entry_io(py_self, packed)


def optimizer_door_state_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_state(opt, which)."""
    return entry_state(py_self, packed)


def optimizer_door_state_set_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_state_set(opt, which, value)."""
    return entry_state_set(py_self, packed)


def optimizer_door_load_state_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_load_state(opt, t, ea, c1, es, c2, bi, c3)."""
    return entry_load_state(py_self, packed)


def optimizer_door_clip_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_clip(grads, max_norm, norm_type, error_if_nonfinite, kwargs)."""
    return entry_clip(py_self, packed)


def optimizer_door_accumulate_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_accumulate(microbatch_grads, tokens)."""
    return entry_accumulate(py_self, packed)


# ---------------------------------------------------------------------------
# The state pool's entries (Samba's native step and the optimizer finalizer
# call them): the training binding's `optimizer_resident_*`, unchanged
# words, now each one call into training/optimizer_public.mojo.


def _step_params(params: PythonObject, name: String) raises -> Tuple[List[Int], List[Float32], Bool]:
    """`optimizer_step`'s params list (12 or 13 words) as (ints, floats,
    maximize): ints [n_tensors, kind, t, nesterov], floats [lr, beta1,
    beta2, eps, weight_decay, momentum, dampening, max_norm]."""
    if len(params) != 12 and len(params) != 13:
        raise Error(name + ": params must contain 12 or 13 values, got " + String(len(params)))
    var ints = List[Int]()
    for i in range(4):  # driver-loop: over argument words (the four integer words)
        ints.append(Int(py=params[i]))
    var fl = List[Float32]()
    for i in range(4, 12):  # driver-loop: over argument words (the eight float words)
        fl.append(Float32(Float64(py=params[i])))
    return (ints^, fl^, len(params) == 13 and Int(py=params[12]) != 0)
