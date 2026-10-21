# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The public optimizers' CPU half (lane neural-optimizers, 2026-10-21): the
one native call behind each public method of `mojolearn.SGD` / `Adam` /
`AdamW`, `clip_grad_norm_` and `accumulate_grads` on a CPU-only install,
the GPU half's contract (training/optimizer_public.mojo) on the host
oracles of core/mojolearn_training_host_impl.mojo. No Python here;
bindings/optimizer_public_host_pyapi.mojo is the door.

The optimizer's state lives in a host slot here, allocated on the first
step; there is no device, so there is no resident state and no
device-parameter step (refused by name)."""
from std.ffi import _Global
from std.memory import memcpy

from core.mojolearn_training_host_impl import clip_grad_norm_host_into, optimizer_step_host_into
from training.host.samba_ops_oracle import host_samba_accumulate
from training.optimizer_public_common import (
    F32P, I32P, Flat, GRADS_HOST, PublicStep, PublicStepOut, gather, registry_check, scatter,
    stack_microbatches, step_lr,
)


struct HostOptPool(Defaultable, Movable):
    var m: List[List[Float32]]
    var v: List[List[Float32]]
    #: floats held by each slot; 0 marks a free slot
    var held: List[Int]

    def __init__(out self):
        self.m = List[List[Float32]]()
        self.v = List[List[Float32]]()
        self.held = List[Int]()


comptime HOST_OPT_POOL = _Global[StorageType=HostOptPool, name="MojoNeuralTrainingOptPoolHost", init_fn=HostOptPool.__init__]


def host_slot(h: Int, want: Int) raises -> Int:
    var pool = HOST_OPT_POOL.get_or_create_ptr()
    if h < 0 or h >= len(pool[].held) or pool[].held[h] == 0:
        raise Error("optimizer_resident: handle " + String(h) + " is not open")
    if want > 0 and pool[].held[h] != want:
        raise Error(
            "optimizer_resident: handle " + String(h) + " holds "
            + String(pool[].held[h]) + " floats, the call names " + String(want)
        )
    return h


def host_open(n: Int) raises -> Int:
    if n < 1:
        raise Error("optimizer_resident_open: n_total must be >= 1, got " + String(n))
    var pool = HOST_OPT_POOL.get_or_create_ptr()
    var h = -1
    for j in range(len(pool[].held)):  # driver-loop: over optimizer handle slots (a free one)
        if pool[].held[j] == 0 and h < 0:
            h = j
    if h < 0:
        pool[].m.append(List[Float32](length=n, fill=Float32(0.0)))
        pool[].v.append(List[Float32](length=n, fill=Float32(0.0)))
        pool[].held.append(n)
        return len(pool[].held) - 1
    pool[].m[h] = List[Float32](length=n, fill=Float32(0.0))
    pool[].v[h] = List[Float32](length=n, fill=Float32(0.0))
    pool[].held[h] = n
    return h


def host_close(handle: Int) raises -> Int:
    var h = host_slot(handle, 0)
    var pool = HOST_OPT_POOL.get_or_create_ptr()
    pool[].m[h] = List[Float32]()
    pool[].v[h] = List[Float32]()
    pool[].held[h] = 0
    return h


def host_download(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
    """The slot's moments into host memory (an address of 0 skips that moment)."""
    var h = host_slot(handle, n)
    var pool = HOST_OPT_POOL.get_or_create_ptr()
    if m_addr != 0:
        memcpy(dest=F32P(unsafe_from_address=m_addr), src=pool[].m[h].unsafe_ptr(), count=n)
    if v_addr != 0:
        memcpy(dest=F32P(unsafe_from_address=v_addr), src=pool[].v[h].unsafe_ptr(), count=n)
    return n


def host_upload(handle: Int, m_addr: Int, v_addr: Int, n: Int) raises -> Int:
    """Host memory into the slot's moments (an address of 0 leaves that moment)."""
    var h = host_slot(handle, n)
    var pool = HOST_OPT_POOL.get_or_create_ptr()
    if m_addr != 0:
        memcpy(dest=pool[].m[h].unsafe_ptr(), src=F32P(unsafe_from_address=m_addr), count=n)
    if v_addr != 0:
        memcpy(dest=pool[].v[h].unsafe_ptr(), src=F32P(unsafe_from_address=v_addr), count=n)
    return n


def public_step_host(req: PublicStep) raises -> PublicStepOut:
    """`step` and `step_accumulated` on the host (`step_device` and device
    parameters are refused: a CPU install has no device)."""
    if req.p_dev or req.device_op or req.g_mode != GRADS_HOST:
        raise Error("RuntimeError: mojolearn optimizer: this build has no device-resident parameter entries "
                    "(a CPU-only install)")
    var hy = req.hyper
    var lr = Float32(step_lr(req))
    var t = req.t + 1
    var clip = hy.max_norm > 0.0
    var op = I32P(unsafe_from_address=req.offsets)
    var ip = I32P(unsafe_from_address=req.init)
    var info = List[Float32](length=3, fill=Float32(0.0))
    var fp = info.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    registry_check(req)
    var fpar = gather(req.p_addr, req.p_bytes, 0, req.j, req.n_total)
    var fg = Flat()
    var g_packed: Bool
    if req.a_count > 1:
        var stacked = stack_microbatches(req)
        var combined = host_samba_accumulate(stacked, req.n_total, req.a_count, req.tokens)
        fg.own = combined^
        fg.packed = True
        g_packed = req.j > 1
    else:
        fg = gather(req.g_addr, req.g_bytes, 0, req.j, req.n_total)
        g_packed = fg.packed
    # the state's slot, opened on the first step; an optimizer built with
    # `resident=False` keeps its state in its own host arrays and has none
    var h = -1
    var mp: F32P
    var vp: F32P
    if req.resident_wanted != 0:
        if req.handle >= 0:
            h = host_slot(req.handle, 0)
        else:
            h = host_open(req.n_total)
            if not req.pristine and req.m_addr != 0 and req.v_addr != 0:
                _ = host_upload(h, req.m_addr, req.v_addr, req.n_total)
        var pool = HOST_OPT_POOL.get_or_create_ptr()
        mp = pool[].m[h].unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        vp = pool[].v[h].unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    else:
        if req.m_addr == 0 or req.v_addr == 0:
            raise Error("RuntimeError: mojolearn optimizer: the host state arrays are missing")
        mp = F32P(unsafe_from_address=req.m_addr)
        vp = F32P(unsafe_from_address=req.v_addr)
    _ = optimizer_step_host_into(
        fpar.ptr(), fg.ptr(), mp, vp, op, ip, fp, req.j, hy.kind, t, hy.nesterov, lr, Float32(hy.beta1),
        Float32(hy.beta2), Float32(hy.eps), Float32(hy.weight_decay), Float32(hy.momentum), Float32(hy.dampening),
        Float32(hy.max_norm) if clip else Float32(0.0), hy.maximize,
    )
    scatter(fpar, req.p_addr, req.p_bytes, 0, req.j)
    if clip and req.a_count == 1:
        scatter(fg, req.g_addr, req.g_bytes, 0, req.j)
    var packed = 1 if (fpar.packed or g_packed) else 0
    _ = fpar^
    _ = fg^
    var ran = info[0] != Float32(0.0)
    var out = PublicStepOut(
        t, h, False, ran, info[1] if ran else Float32(0.0), info[2] if ran else Float32(0.0), lr, packed, False,
    )
    _ = info^
    return out


def public_clip_host(addrs: List[Int], nbytes: List[Int], max_norm: Float32) raises -> Tuple[Float32, Float32]:
    """`clip_grad_norm_` on the host: the packed registry through the host
    clip, scaled in place. Returns (pre-clip total norm, coefficient)."""
    var j = len(addrs)
    var offsets = List[Int32](length=j + 1, fill=Int32(0))
    var op = offsets.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var n_total = 0
    for k in range(j):  # driver-loop: over gradient tensors
        n_total += nbytes[k] // 4
        op[k + 1] = Int32(n_total)
    var info = List[Float32](length=2, fill=Float32(0.0))
    var fp = info.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var f = gather(addrs, nbytes, 0, j, n_total)
    _ = clip_grad_norm_host_into(f.ptr(), op, fp, j, max_norm)
    scatter(f, addrs, nbytes, 0, j)
    _ = f^
    var out = (info[0], info[1])
    _ = offsets^
    _ = info^
    return out


def public_accumulate_host(
    in_addrs: List[Int], nbytes: List[Int], out_addrs: List[Int], a_count: Int, tokens: Int,
) raises -> Int:
    """`accumulate_grads` on the host (the clause 9.2 tree,
    `host_samba_accumulate`). Returns N."""
    var req = PublicStep()
    req.j = len(out_addrs)
    req.a_count = a_count
    req.g_addr = in_addrs.copy()
    for i in range(a_count):  # driver-loop: over microbatches
        for k in range(req.j):  # driver-loop: over gradient tensors
            req.g_bytes.append(nbytes[k])
    for k in range(req.j):  # driver-loop: over gradient tensors
        req.n_total += nbytes[k] // 4
    if req.n_total < 1 or a_count < 1:
        raise Error("mojolearn samba ops: accumulate n and accumulation_steps must be at least 1")
    var stacked = stack_microbatches(req)
    var f = Flat()
    f.own = host_samba_accumulate(stacked, req.n_total, a_count, tokens)
    f.packed = True
    scatter(f, out_addrs, nbytes, 0, req.j)
    _ = f^
    return req.n_total
