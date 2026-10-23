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

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx
from training.optimizer_public import (
    public_accumulate, public_clip, public_io, public_step, resident_close, resident_copy_dev,
    resident_download, resident_get, resident_open, resident_put, resident_step, resident_step_io,
    resident_upload,
)
from training.optimizer_public_common import F32P, I32P, PublicStep, PublicStepOut
from bindings.optimizer_public_door import (
    OptKernels, entry_accumulate, entry_clip, entry_io, entry_load_state, entry_state, entry_state_set,
    entry_step,
)

comptime _NEURAL_CTX = "MojoNeuralTrainingContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralTrainingContextFast"


struct GpuOptKernels(OptKernels):
    """The GPU binding's kernel entries, each with the GIL released."""

    @staticmethod
    def step(req: PublicStep) raises -> PublicStepOut:
        var out = PublicStepOut(0, -1, False, False, Float32(0.0), Float32(0.0), Float32(0.0), -1, False)
        with GILReleased(Python()):
            var ctx = neural_ctx[_NEURAL_CTX]()
            out = public_step(ctx, req)
        return out

    @staticmethod
    def io(req: PublicStep, op_code: Int, tensor: Int) raises -> Tuple[Int, Int]:
        var r = (-1, 0)
        with GILReleased(Python()):
            var ctx = neural_ctx[_NEURAL_CTX]()
            r = public_io(ctx, req, op_code, tensor)
        return r

    @staticmethod
    def clip(addrs: List[Int], nbytes: List[Int], max_norm: Float32) raises -> Tuple[Float32, Float32]:
        var r = (Float32(0.0), Float32(0.0))
        with GILReleased(Python()):
            var ctx = neural_ctx[_NEURAL_CTX]()
            r = public_clip(ctx, addrs, nbytes, max_norm)
        return r

    @staticmethod
    def accumulate(
        in_addrs: List[Int], nbytes: List[Int], out_addrs: List[Int], a_count: Int, tokens: Int
    ) raises -> Int:
        var n = 0
        with GILReleased(Python()):
            var ctx = neural_ctx[_NEURAL_CTX]()
            n = public_accumulate(ctx, in_addrs, nbytes, out_addrs, a_count, tokens)
        return n

    @staticmethod
    def download(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        var r = 0
        with GILReleased(Python()):
            var ctx = neural_ctx[_NEURAL_CTX]()
            r = resident_download(ctx, handle, m_addr, v_addr, n)
        return r

    @staticmethod
    def upload(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
        var r = 0
        with GILReleased(Python()):
            var ctx = neural_ctx[_NEURAL_CTX]()
            r = resident_upload(ctx, handle, m_addr, v_addr, n)
        return r


def optimizer_door_step_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_step(opt, grads, max_norm, op, tokens)."""
    return entry_step[GpuOptKernels](py_self, packed)


def optimizer_door_io_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_io(opt, op, tensors, name, tensor)."""
    return entry_io[GpuOptKernels](py_self, packed)


def optimizer_door_state_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_state(opt, which)."""
    return entry_state[GpuOptKernels](py_self, packed)


def optimizer_door_state_set_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_state_set(opt, which, value)."""
    return entry_state_set[GpuOptKernels](py_self, packed)


def optimizer_door_load_state_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_load_state(opt, t, ea, c1, es, c2, bi, c3)."""
    return entry_load_state[GpuOptKernels](py_self, packed)


def optimizer_door_clip_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_clip(grads, max_norm, norm_type, error_if_nonfinite, kwargs)."""
    return entry_clip[GpuOptKernels](py_self, packed)


def optimizer_door_accumulate_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_door_accumulate(microbatch_grads, tokens)."""
    return entry_accumulate[GpuOptKernels](py_self, packed)


# ---------------------------------------------------------------------------
# The state pool's entries (Samba's native step and the optimizer finalizer
# call them): the training binding's `optimizer_resident_*`, unchanged
# words, now each one call into training/optimizer_public.mojo.


def optimizer_resident_open_binding(n_total: PythonObject) raises -> PythonObject:
    """Allocate a resident `(m, v)` pair of `n_total` floats each, zero
    filled. Returns the handle."""
    var n = Int(py=n_total)
    var h = -1
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        h = resident_open(ctx, n)
    return PythonObject(h)


def optimizer_resident_close_binding(handle: PythonObject) raises -> PythonObject:
    """Free a handle's slot (after the queue drains). Returns the handle."""
    var h = Int(py=handle)
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        h = resident_close(ctx, h)
    return PythonObject(h)


def optimizer_resident_download_binding(
    handle: PythonObject, m_addr: PythonObject, v_addr: PythonObject, n_total: PythonObject,
) raises -> PythonObject:
    """The slot's moments to host memory (`n_total` floats each; an address of 0 leaves that
    moment). Returns `n_total`."""
    var h = Int(py=handle)
    var n = Int(py=n_total)
    var ma = Int(py=m_addr)
    var va = Int(py=v_addr)
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        n = resident_download(ctx, h, ma, va, n)
    return PythonObject(n)


def optimizer_resident_upload_binding(
    handle: PythonObject, m_addr: PythonObject, v_addr: PythonObject, n_total: PythonObject,
) raises -> PythonObject:
    """Host memory (`n_total` floats each) into the slot's moments. Returns `n_total`."""
    var h = Int(py=handle)
    var n = Int(py=n_total)
    var ma = Int(py=m_addr)
    var va = Int(py=v_addr)
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        n = resident_upload(ctx, h, ma, va, n)
    return PythonObject(n)


def maximize_dev_available_binding() raises -> PythonObject:
    """Present when `Optimizer.step` runs maximize through the resident step
    with its transfers named (the sign flip on the device)."""
    return PythonObject(1)


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


def optimizer_resident_step_binding(
    param_addr: PythonObject,
    grad_addr: PythonObject,
    offsets_addr: PythonObject,
    init_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
    handle: PythonObject,
) raises -> PythonObject:
    """`optimizer_step` with `m` and `v` taken from the handle's resident
    pair: the same `params` list (its 12 or 13 slots, word for word).
    Returns `N`."""
    var w = _step_params(params, "optimizer_resident_step")
    var i = w[0].copy()
    var f = w[1].copy()
    var h = Int(py=handle)
    var pp = F32P(unsafe_from_address=Int(py=param_addr))
    var gp = F32P(unsafe_from_address=Int(py=grad_addr))
    var op = I32P(unsafe_from_address=Int(py=offsets_addr))
    var ip = I32P(unsafe_from_address=Int(py=init_addr))
    var fp = F32P(unsafe_from_address=Int(py=info_addr))
    var n_total = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        n_total = resident_step(
            ctx, h, pp, gp, op, ip, fp, i[0], i[1], i[2], i[3], f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], w[2],
        )
    return PythonObject(n_total)


def optimizer_resident_step_io_binding(
    param_addr: PythonObject,
    grad_addr: PythonObject,
    offsets_addr: PythonObject,
    init_addr: PythonObject,
    info_addr: PythonObject,
    params: PythonObject,
    handle_io: PythonObject,
) raises -> PythonObject:
    """`optimizer_resident_step` moving only the transfers `io` names
    (handle_io = [handle, io]; io bits: 1 parameters up, 2 gradient up, 4
    parameters down, 8 clipped gradient down). Returns `N`."""
    var w = _step_params(params, "optimizer_resident_step_io")
    var i = w[0].copy()
    var f = w[1].copy()
    if len(handle_io) != 2:
        raise Error("optimizer_resident_step_io: the last argument is [handle, io]")
    var h = Int(py=handle_io[0])
    var io = Int(py=handle_io[1])
    var pp = F32P(unsafe_from_address=Int(py=param_addr))
    var gp = F32P(unsafe_from_address=Int(py=grad_addr))
    var op = I32P(unsafe_from_address=Int(py=offsets_addr))
    var ip = I32P(unsafe_from_address=Int(py=init_addr))
    var fp = F32P(unsafe_from_address=Int(py=info_addr))
    var n_total = 0
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        n_total = resident_step_io(
            ctx, h, io, pp, gp, op, ip, fp, i[0], i[1], i[2], i[3], f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7],
            w[2],
        )
    return PythonObject(n_total)


def optimizer_resident_copy_dev_binding(handle: PythonObject, params: PythonObject) raises -> PythonObject:
    """A device-to-device copy between an optimizer handle's parameter or
    gradient buffer and a device array. params = [which (0 parameters, 1
    gradient), tensor handle, offset (floats, into the optimizer's flat
    registry), n, direction (0: tensor -> optimizer, 1: optimizer ->
    tensor)]. Returns `n`."""
    if len(params) != 5:
        raise Error("optimizer_resident_copy_dev: params must contain 5 values, got " + String(len(params)))
    var h = Int(py=handle)
    var which = Int(py=params[0])
    var th = Int(py=params[1])
    var off = Int(py=params[2])
    var n = Int(py=params[3])
    var direction = Int(py=params[4])
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        n = resident_copy_dev(ctx, h, which, th, off, n, direction)
    return PythonObject(n)
