# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The public optimizers' GPU half (lane neural-optimizers, 2026-10-21): the
optimizer state pool, and the one native call behind each public method of
`mojolearn.SGD` / `Adam` / `AdamW`, `clip_grad_norm_` and
`accumulate_grads`. No Python here; bindings/optimizer_public_pyapi.mojo is
the door.

THE STATE POOL. Moved from the shell bindings/_mojolearn_training.mojo
(`_OptPool`, `optimizer_resident_*`), the same buffers under the same
`_Global` name, with one addition: a slot may hold the moments ON THE HOST
(`on_host`), for an optimizer whose state stays off the device
(`resident=False`, `MOJOLEARN_OPTIMIZER_RESIDENT=0`, several optimizer
devices). Such a slot runs the per-call step (`optimizer_step`'s route) on
its own host pair; before, the Python object owned that pair. Either way the
optimizer's state is allocated here, on the first step, and the Python
`exp_avg` / `exp_avg_sq` are mirrors read and written through
download / upload.

THE STEP. `public_step` is what `step`, `step_device` and
`step_accumulated` were in Python: the learning rate (a built-in schedule's
value at t + 1), the parameter and gradient traversal (training/
optimizer_public_common.mojo `gather` / `scatter`), the microbatch tree
(`parallel_accumulate_host`), the choice of route and the update, which is
the existing entry for that route called with the same arguments it always
had: the resident step, the resident step with its transfers named (the
device-parameter step and the device sign flip), or the per-call step. The
same kernels on the same buffers in the same order: the same bits."""
from std.ffi import _Global
from std.memory import memcpy
from std.os import getenv

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.accumulate_multi_gpu import parallel_accumulate_host
from training.clip_multi_gpu import parallel_clip_grad_norm_host
from training.dev_tensors import IDN_TRAIN_DEV_TENSORS, train_dev_alloc, train_dev_copy, train_dev_view, train_dev_free
from training.estimator import (
    IDN_MAXIMIZE_DEV, IDN_OPT_PARAMS_RESIDENT, OPT_IO_ALL,
    identical_clip_grad_norm_addrs_host,
    identical_optimizer_step_resident_host,
    identical_optimizer_step_resident_io,
    maximize_negate_device, opt_download_staged, opt_pool_buffers,
)
from training.optimizer_multi_gpu import parallel_optimizer_step_host
from training.optimizer_public_common import (
    F32P, I32P, Flat, GRADS_DEVICE, GRADS_HOST, GRADS_RESIDENT, OptHyper, PublicStep, PublicStepOut,
    gather, registry_check, resident_wanted_by_env, scatter, sizes_check, stack_microbatches, step_lr,
)


def negate_through_device(ctx: DeviceContext, src: F32P, dst: F32P, n: Int) raises:
    """dst[0:n] = `maximize_negate(src[0:n])` ON THE DEVICE: one bulk upload,
    one sign-flip launch (`maximize_negate_device`), one bulk download. The
    flip is exact, so the bits are the host loop's on every column. `src`
    and `dst` may be the same buffer. (Moved from the training binding.)"""
    if n <= 0:
        return
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_buf=d, src_ptr=src)
    maximize_negate_device(ctx, d, n)
    ctx.enqueue_copy(dst_ptr=dst, src_buf=d)
    ctx.synchronize()
    _ = d^


# ===========================================================================
# THE STATE POOL (lane/neural-pass4, 2026-09-30; host slots lane
# neural-optimizers). A handle is an index into the pool; a closed handle's
# slot is reused by the next open.
# ===========================================================================


struct OptPool(Defaultable, Movable):
    var m: List[DeviceBuffer[DType.float32]]
    var v: List[DeviceBuffer[DType.float32]]
    #: the handle's device copies of the parameters and the gradients (lane
    #: neural-pass26): the resident step uploads into these and downloads
    #: the parameters from them
    var p: List[DeviceBuffer[DType.float32]]
    var g: List[DeviceBuffer[DType.float32]]
    #: pinned host staging for the two uploads
    var sp: List[HostBuffer[DType.float32]]
    var sg: List[HostBuffer[DType.float32]]
    #: floats held by each slot; 0 marks a free slot
    var held: List[Int]
    #: a host slot's moments (the per-call step's state); empty on a device slot
    var hm: List[List[Float32]]
    var hv: List[List[Float32]]
    var on_host: List[Bool]

    def __init__(out self):
        self.m = List[DeviceBuffer[DType.float32]]()
        self.v = List[DeviceBuffer[DType.float32]]()
        self.p = List[DeviceBuffer[DType.float32]]()
        self.g = List[DeviceBuffer[DType.float32]]()
        self.sp = List[HostBuffer[DType.float32]]()
        self.sg = List[HostBuffer[DType.float32]]()
        self.held = List[Int]()
        self.hm = List[List[Float32]]()
        self.hv = List[List[Float32]]()
        self.on_host = List[Bool]()


comptime _OPT_POOL_NAME = "MojoNeuralTrainingOptPoolIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoNeuralTrainingOptPoolFast"
comptime OPT_POOL = _Global[StorageType=OptPool, name=_OPT_POOL_NAME, init_fn=OptPool.__init__]


def opt_pool_handle(h: Int, want: Int) raises -> Int:
    """The open slot `h` (holding `want` floats when want > 0)."""
    var pool = OPT_POOL.get_or_create_ptr()
    if h < 0 or h >= len(pool[].held) or pool[].held[h] == 0:
        raise Error("optimizer_resident: handle " + String(h) + " is not open")
    if want > 0 and pool[].held[h] != want:
        raise Error(
            "optimizer_resident: handle " + String(h) + " holds "
            + String(pool[].held[h]) + " floats, the call names " + String(want)
        )
    return h


def _device_slot(h: Int) raises -> Int:
    var pool = OPT_POOL.get_or_create_ptr()
    if pool[].on_host[h]:
        raise Error("optimizer_resident: handle " + String(h) + " keeps its state on the host")
    return h


def resident_open(ctx: DeviceContext, n: Int, on_host: Bool = False) raises -> Int:
    """Allocate a slot of `n` floats: a device `(m, v)` pair, zero filled,
    or (`on_host`) a host pair, zero filled. Returns the handle."""
    if n < 1:
        raise Error("optimizer_resident_open: n_total must be >= 1, got " + String(n))
    var pool = OPT_POOL.get_or_create_ptr()
    var nd = 1 if on_host else n
    var m = ctx.enqueue_create_buffer[DType.float32](nd)
    var v = ctx.enqueue_create_buffer[DType.float32](nd)
    m.enqueue_fill(Float32(0.0))
    v.enqueue_fill(Float32(0.0))
    # The fills are ordered before every later use on the one in-order
    # context, and every host read (the download) waits.
    var n_pool = n if (opt_pool_buffers() and not on_host) else 1
    var pbuf = ctx.enqueue_create_buffer[DType.float32](n_pool)
    var gbuf = ctx.enqueue_create_buffer[DType.float32](n_pool)
    var n_stage = n if (opt_download_staged() and not on_host) else 1
    var spin = ctx.enqueue_create_host_buffer[DType.float32](n_stage)
    var sgin = ctx.enqueue_create_host_buffer[DType.float32](n_stage)
    var hm = List[Float32]()
    var hv = List[Float32]()
    if on_host:
        hm = List[Float32](length=n, fill=Float32(0.0))
        hv = List[Float32](length=n, fill=Float32(0.0))
    var h = -1
    for j in range(len(pool[].held)):  # driver-loop: over optimizer handle slots (a free one)
        if pool[].held[j] == 0 and h < 0:
            h = j
    if h < 0:
        pool[].m.append(m^)
        pool[].v.append(v^)
        pool[].p.append(pbuf^)
        pool[].g.append(gbuf^)
        pool[].sp.append(spin^)
        pool[].sg.append(sgin^)
        pool[].held.append(n)
        pool[].hm.append(hm^)
        pool[].hv.append(hv^)
        pool[].on_host.append(on_host)
        h = len(pool[].held) - 1
    else:
        pool[].m[h] = m^
        pool[].v[h] = v^
        pool[].p[h] = pbuf^
        pool[].g[h] = gbuf^
        pool[].sp[h] = spin^
        pool[].sg[h] = sgin^
        pool[].held[h] = n
        pool[].hm[h] = hm^
        pool[].hv[h] = hv^
        pool[].on_host[h] = on_host
    return h


def resident_close(ctx: DeviceContext, handle: Int) raises -> Int:
    """Free a slot (after the queue drains). Returns the handle."""
    var h = opt_pool_handle(handle, 0)
    ctx.synchronize()
    var pool = OPT_POOL.get_or_create_ptr()
    pool[].m[h] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].v[h] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].p[h] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].g[h] = ctx.enqueue_create_buffer[DType.float32](1)
    pool[].sp[h] = ctx.enqueue_create_host_buffer[DType.float32](1)
    pool[].sg[h] = ctx.enqueue_create_host_buffer[DType.float32](1)
    pool[].hm[h] = List[Float32]()
    pool[].hv[h] = List[Float32]()
    pool[].on_host[h] = False
    pool[].held[h] = 0
    return h


def resident_download(ctx: DeviceContext, handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
    """The slot's moments to host memory (`n` floats each; an address of 0
    skips that moment). Returns `n`."""
    var h = opt_pool_handle(handle, n)
    var pool = OPT_POOL.get_or_create_ptr()
    if pool[].on_host[h]:
        if m_addr != 0:
            memcpy(dest=F32P(unsafe_from_address=m_addr), src=pool[].hm[h].unsafe_ptr(), count=n)
        if v_addr != 0:
            memcpy(dest=F32P(unsafe_from_address=v_addr), src=pool[].hv[h].unsafe_ptr(), count=n)
        return n
    if m_addr != 0:
        ctx.enqueue_copy(dst_ptr=F32P(unsafe_from_address=m_addr), src_buf=pool[].m[h])
    if v_addr != 0:
        ctx.enqueue_copy(dst_ptr=F32P(unsafe_from_address=v_addr), src_buf=pool[].v[h])
    ctx.synchronize()
    return n


def resident_upload(ctx: DeviceContext, handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
    """Host memory (`n` floats each) into the slot's moments (an address of
    0 leaves that moment). Returns `n`."""
    var h = opt_pool_handle(handle, n)
    var pool = OPT_POOL.get_or_create_ptr()
    if pool[].on_host[h]:
        if m_addr != 0:
            memcpy(dest=pool[].hm[h].unsafe_ptr(), src=F32P(unsafe_from_address=m_addr), count=n)
        if v_addr != 0:
            memcpy(dest=pool[].hv[h].unsafe_ptr(), src=F32P(unsafe_from_address=v_addr), count=n)
        return n
    if m_addr != 0:
        ctx.enqueue_copy(dst_buf=pool[].m[h], src_ptr=F32P(unsafe_from_address=m_addr))
    if v_addr != 0:
        ctx.enqueue_copy(dst_buf=pool[].v[h], src_ptr=F32P(unsafe_from_address=v_addr))
    ctx.synchronize()
    return n


def per_call_step(
    ctx: DeviceContext, pp: F32P, gp: F32P, mp: F32P, vp: F32P, op: I32P, ip: I32P, fp: F32P,
    n_tensors: Int, kind: Int, t: Int, nesterov: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32, momentum: Float32, dampening: Float32, max_norm: Float32,
    maximize: Bool,
) raises -> Int:
    """`optimizer_step`: the per-call step on host moments (moved from the
    training binding, word for word)."""
    var n_total = 0
    if maximize:
        # The step reads a negated COPY, so the caller's gradient is never
        # negated, not even for the length of the call; a clipped
        # gradient is written back through the same sign flip.
        if n_tensors < 1 or op[n_tensors] < Int32(0):
            raise Error("optimizer_step: maximize needs a registry with offsets[J] >= 0")
        var n_flat = Int(op[n_tensors])
        var neg = List[Float32](length=max(n_flat, 1), fill=Float32(0.0))
        var np_ = neg.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        negate_through_device(ctx, gp, np_, n_flat)
        n_total = parallel_optimizer_step_host(
            ctx, pp, np_, mp, vp, op, ip, fp, n_tensors, kind, t, nesterov,
            lr, beta1, beta2, eps, weight_decay, momentum, dampening,
            max_norm,
        )
        if max_norm > Float32(0.0):
            negate_through_device(ctx, np_, gp, n_flat)
        _ = neg^  # `np_` is untracked: keep the copy alive past its last use
    else:
        n_total = parallel_optimizer_step_host(
            ctx, pp, gp, mp, vp, op, ip, fp, n_tensors, kind, t, nesterov,
            lr, beta1, beta2, eps, weight_decay, momentum, dampening,
            max_norm,
        )
    return n_total


def resident_step(
    ctx: DeviceContext, handle: Int, pp: F32P, gp: F32P, op: I32P, ip: I32P, fp: F32P,
    n_tensors: Int, kind: Int, t: Int, nesterov: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32, momentum: Float32, dampening: Float32, max_norm: Float32,
    maximize: Bool,
) raises -> Int:
    """`optimizer_resident_step`: the step with the moments in the device
    slot (moved from the training binding, word for word)."""
    var h = _device_slot(opt_pool_handle(handle, 0))
    var n_total = 0
    var pool = OPT_POOL.get_or_create_ptr()
    if maximize:
        if n_tensors < 1 or op[n_tensors] < Int32(0):
            raise Error("optimizer_resident_step: maximize needs a registry with offsets[J] >= 0")
        var n_flat = Int(op[n_tensors])
        var neg = List[Float32](length=max(n_flat, 1), fill=Float32(0.0))
        var np_ = neg.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        negate_through_device(ctx, gp, np_, n_flat)
        # the parameter/gradient device buffers: the handle's pooled pair on
        # Apple, a fresh pair per step elsewhere (lane neural-pass34)
        var pooled = opt_pool_buffers()
        var pb = ctx.enqueue_create_buffer[DType.float32](1 if pooled else pool[].held[h])
        var gb = ctx.enqueue_create_buffer[DType.float32](1 if pooled else pool[].held[h])
        if pooled:
            n_total = identical_optimizer_step_resident_host(
                ctx, pp, np_, pool[].m[h], pool[].v[h], pool[].p[h], pool[].g[h], pool[].sp[h], pool[].sg[h], op, ip, fp, n_tensors,
                kind, t, nesterov, lr, beta1, beta2, eps, weight_decay,
                momentum, dampening, max_norm,
            )
        else:
            n_total = identical_optimizer_step_resident_host(
                ctx, pp, np_, pool[].m[h], pool[].v[h], pb, gb, pool[].sp[h], pool[].sg[h], op, ip, fp, n_tensors,
                kind, t, nesterov, lr, beta1, beta2, eps, weight_decay,
                momentum, dampening, max_norm,
            )
        if max_norm > Float32(0.0):
            negate_through_device(ctx, np_, gp, n_flat)
        _ = neg^  # `np_` is untracked: keep the copy alive past its last use
    else:
        # the parameter/gradient device buffers: the handle's pooled pair on
        # Apple, a fresh pair per step elsewhere (lane neural-pass34)
        var pooled = opt_pool_buffers()
        var pb = ctx.enqueue_create_buffer[DType.float32](1 if pooled else pool[].held[h])
        var gb = ctx.enqueue_create_buffer[DType.float32](1 if pooled else pool[].held[h])
        if pooled:
            n_total = identical_optimizer_step_resident_host(
                ctx, pp, gp, pool[].m[h], pool[].v[h], pool[].p[h], pool[].g[h], pool[].sp[h], pool[].sg[h], op, ip, fp, n_tensors,
                kind, t, nesterov, lr, beta1, beta2, eps, weight_decay,
                momentum, dampening, max_norm,
            )
        else:
            n_total = identical_optimizer_step_resident_host(
                ctx, pp, gp, pool[].m[h], pool[].v[h], pb, gb, pool[].sp[h], pool[].sg[h], op, ip, fp, n_tensors,
                kind, t, nesterov, lr, beta1, beta2, eps, weight_decay,
                momentum, dampening, max_norm,
            )
    return n_total


def opt_dev_pair(ctx: DeviceContext, h: Int) raises:
    """The slot's parameter and gradient device buffers at full size (off
    Apple they are one float until first needed here)."""
    var pool = OPT_POOL.get_or_create_ptr()
    var n = pool[].held[h]
    if len(pool[].p[h]) < n:
        pool[].p[h] = ctx.enqueue_create_buffer[DType.float32](n)
    if len(pool[].g[h]) < n:
        pool[].g[h] = ctx.enqueue_create_buffer[DType.float32](n)


def resident_put(ctx: DeviceContext, handle: Int, which: Int, src: F32P, n: Int) raises -> Int:
    """Host `n` floats into the slot's parameter (which = 0) or gradient
    (which = 1) device buffer. Returns `n`."""
    var h = _device_slot(opt_pool_handle(handle, n))
    if which != 0 and which != 1:
        raise Error("optimizer_resident_put: which is 0 (parameters) or 1 (gradient)")
    opt_dev_pair(ctx, h)
    var pool = OPT_POOL.get_or_create_ptr()
    if which == 0:
        ctx.enqueue_copy(dst_buf=pool[].p[h], src_ptr=src)
    else:
        ctx.enqueue_copy(dst_buf=pool[].g[h], src_ptr=src)
    ctx.synchronize()
    return n


def resident_get(ctx: DeviceContext, handle: Int, which: Int, dst: F32P, n: Int) raises -> Int:
    """The slot's parameter (which = 0) or gradient (which = 1) device buffer
    into host memory (`n` floats). Returns `n`."""
    var h = _device_slot(opt_pool_handle(handle, n))
    if which != 0 and which != 1:
        raise Error("optimizer_resident_get: which is 0 (parameters) or 1 (gradient)")
    opt_dev_pair(ctx, h)
    var pool = OPT_POOL.get_or_create_ptr()
    if which == 0:
        ctx.enqueue_copy(dst_ptr=dst, src_buf=pool[].p[h])
    else:
        ctx.enqueue_copy(dst_ptr=dst, src_buf=pool[].g[h])
    ctx.synchronize()
    return n


def resident_step_io(
    ctx: DeviceContext, handle: Int, io: Int, pp: F32P, gp: F32P, op: I32P, ip: I32P, fp: F32P,
    n_tensors: Int, kind: Int, t: Int, nesterov: Int, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, weight_decay: Float32, momentum: Float32, dampening: Float32, max_norm: Float32,
    maximize: Bool,
) raises -> Int:
    """`optimizer_resident_step_io`: the resident step moving only the
    transfers `io` names (1 parameters up, 2 gradient up, 4 parameters
    down, 8 clipped gradient down); maximize negates on the device. (Moved
    from the training binding.)"""
    var h = _device_slot(opt_pool_handle(handle, 0))
    if io < 0 or io > OPT_IO_ALL:
        raise Error("optimizer_resident_step_io: io is a mask of 1, 2, 4 and 8")
    opt_dev_pair(ctx, h)
    var pool = OPT_POOL.get_or_create_ptr()
    return identical_optimizer_step_resident_io(
        ctx, pp, gp, pool[].m[h], pool[].v[h], pool[].p[h], pool[].g[h], pool[].sp[h], pool[].sg[h], op, ip,
        fp, n_tensors, kind, t, nesterov, lr, beta1, beta2, eps, weight_decay, momentum, dampening,
        max_norm, io, maximize,
    )


def resident_copy_dev(ctx: DeviceContext, handle: Int, which: Int, th: Int, off: Int, n: Int, direction: Int) raises -> Int:
    """A device-to-device copy between the slot's parameter (which = 0) or
    gradient (1) buffer at float offset `off` and device array `th`
    (direction 0: array -> optimizer, 1: optimizer -> array). Returns `n`."""
    var h = _device_slot(opt_pool_handle(handle, 0))
    if (which != 0 and which != 1) or (direction != 0 and direction != 1):
        raise Error("optimizer_resident_copy_dev: which and direction are 0 or 1")
    opt_dev_pair(ctx, h)
    var pool = OPT_POOL.get_or_create_ptr()
    var tv = train_dev_view(ctx, th, n, "the optimizer copy")
    if which == 0:
        if direction == 0:
            train_dev_copy(ctx, pool[].p[h], off, tv, 0, n)
        else:
            train_dev_copy(ctx, tv, 0, pool[].p[h], off, n)
    else:
        if direction == 0:
            train_dev_copy(ctx, pool[].g[h], off, tv, 0, n)
        else:
            train_dev_copy(ctx, tv, 0, pool[].g[h], off, n)
    _ = tv^
    return n


# ===========================================================================
# THE PUBLIC METHODS
# ===========================================================================


def _open_slot(ctx: DeviceContext, req: PublicStep) raises -> Int:
    """The optimizer's slot, opened on its first use: on the device unless
    the optimizer was built with `resident=False` or the environment keeps
    the state off the device (`resident_wanted_by_env`). A state the caller
    has touched (a `load_state_dict`, an assignment, a read) is uploaded;
    an untouched one is the slot's own zeros."""
    if req.handle >= 0:
        return opt_pool_handle(req.handle, 0)
    var device = req.resident_wanted != 0 and resident_wanted_by_env()
    var h = resident_open(ctx, req.n_total, not device)
    if not req.pristine and req.m_addr != 0 and req.v_addr != 0:
        _ = resident_upload(ctx, h, req.m_addr, req.v_addr, req.n_total)
    return h


def _on_device(h: Int) raises -> Bool:
    var pool = OPT_POOL.get_or_create_ptr()
    return not pool[].on_host[h]


def public_step(ctx: DeviceContext, req: PublicStep) raises -> PublicStepOut:
    """`step`, `step_device` and `step_accumulated` of the public optimizers
    in one call (see the file docstring)."""
    var hy = req.hyper
    var lr64 = step_lr(req)
    var t = req.t + 1
    var lr = Float32(lr64)
    var beta1 = Float32(hy.beta1)
    var beta2 = Float32(hy.beta2)
    var eps = Float32(hy.eps)
    var wd = Float32(hy.weight_decay)
    var mom = Float32(hy.momentum)
    var damp = Float32(hy.dampening)
    var max_norm = Float32(hy.max_norm)
    var clip = hy.max_norm > 0.0
    var op = I32P(unsafe_from_address=req.offsets)
    var ip = I32P(unsafe_from_address=req.init)
    var info = List[Float32](length=3, fill=Float32(0.0))
    var fp = info.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var h: Int
    var packed = -1
    var g_dev = req.g_dev

    if req.p_dev:
        # the device-parameter step (lane fam2-neural): the parameters live
        # in the slot; the gradient is uploaded, copied on the device, or
        # already there
        comptime if not IDN_OPT_PARAMS_RESIDENT:
            raise Error("RuntimeError: mojolearn optimizer: this build has no device-resident parameter entries")
        if req.resident_wanted == 0 or req.handle < 0:
            raise Error("RuntimeError: mojolearn optimizer: device-resident parameters need a "
                        "device-resident optimizer (call params_to_device() first)")
        h = _open_slot(ctx, req)
        if not _on_device(h):
            raise Error("RuntimeError: mojolearn optimizer: this optimizer keeps its state on the host; "
                        "device-resident parameters need a device-resident optimizer")
        var io = 0
        var fg = Flat()
        var g_host = req.g_mode == GRADS_HOST
        if req.a_count > 1:
            # step_accumulated on device parameters: the clause 9.2 tree on
            # the host-given microbatches, then the combined gradient goes up
            # (it is not written back: the caller's microbatches stay as given)
            var stacked = stack_microbatches(req)
            fg.own = List[Float32](unsafe_uninit_length=req.n_total)
            fg.packed = True
            _ = parallel_accumulate_host(
                ctx, fg.ptr(), stacked.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), req.n_total,
                req.a_count, req.tokens,
            )
            _ = stacked^
            io = 10 if clip else 2
        elif req.g_mode == GRADS_DEVICE:
            comptime if not IDN_TRAIN_DEV_TENSORS:
                raise Error("RuntimeError: mojolearn optimizer: this build has no training device arrays")
            for j in range(req.j):  # driver-loop: over parameter tensors (one device copy each)
                var n_j = Int(op[j + 1]) - Int(op[j])
                if req.g_bytes[j] != n_j:
                    raise Error("ValueError: mojolearn optimizer step_device: grads[" + String(j)
                                + "] must be a device array of " + String(n_j) + " floats")
                _ = resident_copy_dev(ctx, h, 1, req.g_addr[j], Int(op[j]), n_j, 0)
        elif req.g_mode == GRADS_HOST:
            sizes_check(req.g_bytes, req.j, req.n_total)
            fg = gather(req.g_addr, req.g_bytes, 0, req.j, req.n_total)
            io = 10 if clip else 2
        elif not req.g_dev:
            raise Error("RuntimeError: mojolearn optimizer step_device: no gradient on the device "
                        "(pass grads or call grads_to_device)")
        var gp = fg.ptr() if (g_host or req.a_count > 1) else fp
        _ = resident_step_io(
            ctx, h, io, fp, gp, op, ip, fp, req.j, hy.kind, t, hy.nesterov, lr, beta1, beta2, eps, wd,
            mom, damp, max_norm if clip else Float32(0.0), hy.maximize,
        )
        if clip and g_host and req.a_count == 1:
            scatter(fg, req.g_addr, req.g_bytes, 0, req.j)
        _ = fg^
        g_dev = True
    else:
        registry_check(req)
        var fp_ = gather(req.p_addr, req.p_bytes, 0, req.j, req.n_total)
        var fg = Flat()
        var g_packed: Bool
        if req.a_count > 1:
            # step_accumulated: the clause 9.2 balanced tree in ascending
            # microbatch index, then the step on the combined gradient
            var stacked = stack_microbatches(req)
            fg.own = List[Float32](unsafe_uninit_length=req.n_total)
            fg.packed = True
            _ = parallel_accumulate_host(
                ctx, fg.ptr(), stacked.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), req.n_total,
                req.a_count, req.tokens,
            )
            _ = stacked^
            g_packed = req.j > 1
        else:
            fg = gather(req.g_addr, req.g_bytes, 0, req.j, req.n_total)
            g_packed = fg.packed
        packed = 1 if (fp_.packed or g_packed) else 0
        # the state's slot, opened on the first step; an optimizer built
        # with `resident=False` keeps its state in its own host arrays
        # (exp_avg / exp_avg_sq), as it always did, and has no slot
        h = -1
        if req.resident_wanted != 0:
            h = _open_slot(ctx, req)
        var pp = fp_.ptr()
        var gp = fg.ptr()
        var mn = max_norm if clip else Float32(0.0)
        if h >= 0 and _on_device(h):
            var dev_flip = False
            comptime if IDN_OPT_PARAMS_RESIDENT and IDN_MAXIMIZE_DEV:
                dev_flip = hy.maximize
            if dev_flip:
                # lane fam2-neural: maximize negates on the device, all four
                # transfers
                _ = resident_step_io(
                    ctx, h, OPT_IO_ALL, pp, gp, op, ip, fp, req.j, hy.kind, t, hy.nesterov, lr, beta1, beta2,
                    eps, wd, mom, damp, mn, True,
                )
            else:
                _ = resident_step(
                    ctx, h, pp, gp, op, ip, fp, req.j, hy.kind, t, hy.nesterov, lr, beta1, beta2, eps, wd,
                    mom, damp, mn, hy.maximize,
                )
        else:
            var mp: F32P
            var vp: F32P
            if h >= 0:
                var pool = OPT_POOL.get_or_create_ptr()
                mp = pool[].hm[h].unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
                vp = pool[].hv[h].unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            else:
                if req.m_addr == 0 or req.v_addr == 0:
                    raise Error("RuntimeError: mojolearn optimizer: the host state arrays are missing")
                mp = F32P(unsafe_from_address=req.m_addr)
                vp = F32P(unsafe_from_address=req.v_addr)
            _ = per_call_step(
                ctx, pp, gp, mp, vp, op, ip, fp, req.j, hy.kind, t, hy.nesterov, lr, beta1, beta2, eps, wd,
                mom, damp, mn, hy.maximize,
            )
        scatter(fp_, req.p_addr, req.p_bytes, 0, req.j)
        if clip and req.a_count == 1:
            scatter(fg, req.g_addr, req.g_bytes, 0, req.j)
        _ = fp_^
        _ = fg^
    var ran = info[0] != Float32(0.0)
    var out = PublicStepOut(
        t, h, h >= 0 and _on_device(h), ran, info[1] if ran else Float32(0.0), info[2] if ran else Float32(0.0),
        lr, packed, g_dev,
    )
    _ = info^
    return out


#: `public_io` operations
comptime IO_QUERY = 0          # open a device slot if possible; report whether there is one
comptime IO_PARAMS_UP = 1
comptime IO_PARAMS_DOWN = 2
comptime IO_GRADS_UP = 3
comptime IO_GRADS_DOWN = 4
comptime IO_PARAM_DEVICE = 5   # tensor j of the slot's parameters into a new device array


def public_io(
    ctx: DeviceContext, req: PublicStep, op_code: Int, tensor: Int,
) raises -> Tuple[Int, Int]:
    """`params_resident_available`, `params_to_device`, `params_to_host`,
    `grads_to_device`, `grads_to_host` and `param_device`: one call each.
    The tensors are `req.g_addr` / `req.g_bytes` (parameters or gradients,
    the registry's tensors in order). Returns (handle, result): result is 1
    / 0 for IO_QUERY, the device array handle for IO_PARAM_DEVICE, else
    the floats moved."""
    var h = -1
    var available = IDN_OPT_PARAMS_RESIDENT
    if available:
        if req.handle >= 0:
            h = opt_pool_handle(req.handle, 0)
            available = _on_device(h)
        elif req.resident_wanted != 0 and resident_wanted_by_env():
            h = _open_slot(ctx, req)
        else:
            available = False
    if op_code == IO_QUERY:
        return (h, 1 if available else 0)
    if not available:
        raise Error(
            "RuntimeError: mojolearn optimizer: this build has no device-resident parameter "
            "entries (IDENTICAL GPU builds only; MOJOLEARN_IDN_OPT_PARAMS_RESIDENT_OFF or a "
            "resident-less optimizer turns them off)"
        )
    if op_code == IO_PARAM_DEVICE:
        comptime if not IDN_TRAIN_DEV_TENSORS:
            raise Error("RuntimeError: mojolearn optimizer: this build has no training device arrays")
        var op = I32P(unsafe_from_address=req.offsets)
        var off = Int(op[tensor])
        var n_j = Int(op[tensor + 1]) - off
        var dh = train_dev_alloc(ctx, n_j)
        try:
            _ = resident_copy_dev(ctx, h, 0, dh, off, n_j, 1)
        except error:
            try:
                train_dev_free(ctx, dh)
            except:
                pass
            raise error^
        return (h, dh)
    var which = 0 if (op_code == IO_PARAMS_UP or op_code == IO_PARAMS_DOWN) else 1
    sizes_check(req.g_bytes, req.j, req.n_total)
    var f = gather(req.g_addr, req.g_bytes, 0, req.j, req.n_total)
    if op_code == IO_PARAMS_UP or op_code == IO_GRADS_UP:
        _ = resident_put(ctx, h, which, f.ptr(), req.n_total)
    else:
        _ = resident_get(ctx, h, which, f.ptr(), req.n_total)
        scatter(f, req.g_addr, req.g_bytes, 0, req.j)
    _ = f^
    return (h, req.n_total)


def public_clip(ctx: DeviceContext, addrs: List[Int], nbytes: List[Int], max_norm: Float32) raises -> Tuple[Float32, Float32]:
    """`clip_grad_norm_`: the two-level global L2 norm of the tensors and the
    in-place scale (optimizer contract section 3). Several tensors go to the
    device straight from their own memory (`identical_clip_grad_norm_addrs`)
    unless MOJOLEARN_CLIP_PACKED=1 or several optimizer devices ask for the
    packed buffer (the A/B; the same device buffer, offsets and clip, so
    the same bits). Returns (pre-clip total norm, clamped coefficient)."""
    var j = len(addrs)
    var offsets = List[Int32](length=j + 1, fill=Int32(0))
    var op = offsets.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var n_total = 0
    for k in range(j):  # driver-loop: over gradient tensors
        n_total += nbytes[k] // 4
        op[k + 1] = Int32(n_total)
    var info = List[Float32](length=2, fill=Float32(0.0))
    var fp = info.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var multi = (j > 1 and String(getenv("MOJOLEARN_CLIP_PACKED")) != "1"
                 and String(getenv("MOJOLEARN_OPTIMIZER_DEVICE_COUNT", "1")) == "1")
    if multi:
        _ = identical_clip_grad_norm_addrs_host(ctx, addrs, op, fp, j, max_norm)
    else:
        var f = gather(addrs, nbytes, 0, j, n_total)
        _ = parallel_clip_grad_norm_host(ctx, f.ptr(), op, fp, j, max_norm)
        scatter(f, addrs, nbytes, 0, j)
        _ = f^
    var out = (info[0], info[1])
    _ = offsets^
    _ = info^
    return out


def public_accumulate(
    ctx: DeviceContext, in_addrs: List[Int], nbytes: List[Int], out_addrs: List[Int], a_count: Int, tokens: Int,
) raises -> Int:
    """`accumulate_grads`: the A microbatches (in_addrs microbatch-major, J
    tensors each, `nbytes` per tensor) combined by the clause 9.2 balanced
    tree into the J result tensors. Returns N."""
    var req = PublicStep()
    req.j = len(out_addrs)
    req.a_count = a_count
    req.g_addr = in_addrs.copy()
    for i in range(a_count):  # driver-loop: over microbatches
        for k in range(req.j):  # driver-loop: over gradient tensors
            req.g_bytes.append(nbytes[k])
    for k in range(req.j):  # driver-loop: over gradient tensors
        req.n_total += nbytes[k] // 4
    var stacked = stack_microbatches(req)
    var f = Flat()
    f.own = List[Float32](unsafe_uninit_length=max(req.n_total, 1))
    f.packed = True
    _ = parallel_accumulate_host(
        ctx, f.ptr(), stacked.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), req.n_total, a_count, tokens,
    )
    _ = stacked^
    scatter(f, out_addrs, nbytes, 0, req.j)
    _ = f^
    return req.n_total
