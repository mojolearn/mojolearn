# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the public optimizers on the GPU binding
(`_mojolearn_training`; lane neural-optimizers, 2026-10-21). Each entry
unpacks its argument words, calls ONE typed kernel entry of
training/optimizer_public.mojo and writes the typed result into the
caller's result buffer (or returns one integer). Nothing is assigned on a
Python object here: the Python glue assigns what the result buffer holds
(docs/MODULAR_ASKS.md, PythonObject ownership). A kernel error that names
its class (`ValueError: ...`) raises that class.

The argument words, in this exact order (mirrored in
python/mojolearn/_training_impl.py `_step_words`):

  step words   0 J, 1 t, 2 kind, 3 nesterov, 4 maximize, 5 schedule kind
               (-1 none), 6-8 schedule a, b, c, 9 N, 10 offsets address,
               11 buf_initialized address, 12 gradient mode, 13 microbatch
               count, 14 tokens (-1 none), 15 handle (-1 none), 16 resident
               (-1 None, 0 False, 1 True), 17 pristine, 18 exp_avg address
               (0 none), 19 exp_avg_sq address (0 none), 20 parameters on the
               device, 21 gradient on the device, 22 step_device
  step floats  0 lr, 1 beta1, 2 beta2, 3 eps, 4 weight_decay, 5 momentum,
               6 dampening, 7 max_norm (<= 0: no clip), 8-11 schedule f0-f3
  io words     0 operation, 1 tensor, 2 J, 3 N, 4 offsets address, 5 handle,
               6 resident, 7 pristine, 8 exp_avg address, 9 exp_avg_sq address
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.neural_context import neural_ctx
from training.optimizer_public import (
    public_accumulate, public_clip, public_io, public_step, resident_close, resident_copy_dev,
    resident_download, resident_get, resident_open, resident_put, resident_step, resident_step_io,
    resident_upload,
)
from training.optimizer_public_common import F32P, F64P, I32P, PublicStep, zero_tensors
from bindings.optimizer_public_words import int_words, optimizer_python_error, step_request

comptime _NEURAL_CTX = "MojoNeuralTrainingContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralTrainingContextFast"


def optimizer_public_step_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_public_step(words, floats, p_addrs, p_bytes, g_addrs,
    g_bytes, out_addr): one step of a public optimizer; the 10 float64
    result words go to `out_addr` (training/optimizer_public_common.mojo
    `PublicStepOut.write`). Returns 0."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 7:
            raise Error("TypeError: optimizer_public_step expects seven arguments")
        var req = step_request(args[0], args[1], args[2], args[3], args[4], args[5])
        var out = F64P(unsafe_from_address=Int(py=args[6]))
        with GILReleased(Python()):
            var ctx = neural_ctx[_NEURAL_CTX]()
            public_step(ctx, req).write(out)
        return PythonObject(0).steal_data()
    except error:
        return optimizer_python_error(error)


def optimizer_public_io_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_public_io(words, addrs, nbytes, out_addr): the device
    transfers of a public optimizer (training/optimizer_public.mojo
    `public_io`); out_addr gets [handle, result] as float64. Returns 0."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 4:
            raise Error("TypeError: optimizer_public_io expects four arguments")
        var w = int_words(args[0])
        if len(w) != 10:
            raise Error("TypeError: optimizer_public_io takes 10 words")
        var req = PublicStep()
        req.j = w[2]
        req.n_total = w[3]
        req.offsets = w[4]
        req.handle = w[5]
        req.resident_wanted = w[6]
        req.pristine = w[7] != 0
        req.m_addr = w[8]
        req.v_addr = w[9]
        req.g_addr = int_words(args[1])
        req.g_bytes = int_words(args[2])
        var out = F64P(unsafe_from_address=Int(py=args[3]))
        with GILReleased(Python()):
            var ctx = neural_ctx[_NEURAL_CTX]()
            var r = public_io(ctx, req, w[0], w[1])
            out[0] = Float64(r[0])
            out[1] = Float64(r[1])
        return PythonObject(0).steal_data()
    except error:
        return optimizer_python_error(error)


def optimizer_public_clip_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_public_clip(addrs, nbytes, max_norm, out_addr):
    `clip_grad_norm_` over the tensors, in place; out_addr gets [total
    norm, coefficient] as float64. Returns 0."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 4:
            raise Error("TypeError: optimizer_public_clip expects four arguments")
        var addrs = int_words(args[0])
        var nbytes = int_words(args[1])
        var max_norm = Float32(Float64(py=args[2]))
        var out = F64P(unsafe_from_address=Int(py=args[3]))
        with GILReleased(Python()):
            var ctx = neural_ctx[_NEURAL_CTX]()
            var r = public_clip(ctx, addrs, nbytes, max_norm)
            out[0] = Float64(r[0])
            out[1] = Float64(r[1])
        return PythonObject(0).steal_data()
    except error:
        return optimizer_python_error(error)


def optimizer_public_accumulate_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_public_accumulate(in_addrs, nbytes, out_addrs, a, tokens):
    the clause 9.2 tree of A microbatches into the result tensors. Returns
    N."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 5:
            raise Error("TypeError: optimizer_public_accumulate expects five arguments")
        var in_addrs = int_words(args[0])
        var nbytes = int_words(args[1])
        var out_addrs = int_words(args[2])
        var a = Int(py=args[3])
        var tokens = Int(py=args[4])
        var n = 0
        with GILReleased(Python()):
            var ctx = neural_ctx[_NEURAL_CTX]()
            n = public_accumulate(ctx, in_addrs, nbytes, out_addrs, a, tokens)
        return PythonObject(n).steal_data()
    except error:
        return optimizer_python_error(error)


def optimizer_public_zero_c(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_public_zero(addrs, nbytes): `zero_grad`, +0.0 into every
    tensor. Returns the tensor count."""
    try:
        var args = PythonObject(from_borrowed=packed)
        if len(args) != 2:
            raise Error("TypeError: optimizer_public_zero expects two arguments")
        var addrs = int_words(args[0])
        var nbytes = int_words(args[1])
        with GILReleased(Python()):
            zero_tensors(addrs, nbytes)
        return PythonObject(len(addrs)).steal_data()
    except error:
        return optimizer_python_error(error)


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
    for i in range(4):  # small-loop(argument words: the four integer words)
        ints.append(Int(py=params[i]))
    var fl = List[Float32]()
    for i in range(4, 12):  # small-loop(argument words: the eight float words)
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


def optimizer_resident_put_binding(
    handle: PythonObject, which: PythonObject, addr: PythonObject, n_total: PythonObject,
) raises -> PythonObject:
    """Host `n_total` floats into the handle's parameter (which = 0) or
    gradient (which = 1) device buffer. Returns `n_total`."""
    var h = Int(py=handle)
    var w = Int(py=which)
    var n = Int(py=n_total)
    var src = F32P(unsafe_from_address=Int(py=addr))
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        n = resident_put(ctx, h, w, src, n)
    return PythonObject(n)


def optimizer_resident_get_binding(
    handle: PythonObject, which: PythonObject, addr: PythonObject, n_total: PythonObject,
) raises -> PythonObject:
    """The handle's parameter (which = 0) or gradient (which = 1) device
    buffer into host memory (`n_total` floats). Returns `n_total`."""
    var h = Int(py=handle)
    var w = Int(py=which)
    var n = Int(py=n_total)
    var dst = F32P(unsafe_from_address=Int(py=addr))
    with GILReleased(Python()):
        var ctx = neural_ctx[_NEURAL_CTX]()
        n = resident_get(ctx, h, w, dst, n)
    return PythonObject(n)


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
