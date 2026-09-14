# SPDX-License-Identifier: Apache-2.0
"""Host-staged optimizer shards with the original whole-registry GPU clip.

Only raw copies and ownership happen here. The existing optimizer entry runs
on each disjoint parameter range with clipping already completed. SGD keeps
each intersected tensor's original momentum flag. Caller buffers publish only
after all workers finish; no cross-device floating-point sum is introduced.
"""
from std.os import getenv
from max.gpu.host import DeviceContext
from max.algorithm import sync_parallelize
from bindings.hostptr import copy_f32
from core.step_phase import STEP_PHASE_TIMERS
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.clip_multi_gpu import parallel_clip_grad_norm_host
from training.estimator import (
    identical_optimizer_step_host, identical_clip_grad_norm_host,
    _offsets_from_ptr, _refuse_hyperparameters,
)
from training.checks.optimizer_oracle import OptimizerConfig, OPT_SGD, OPT_ADAM, OPT_ADAMW


@fieldwise_init
struct OptimizerHostShard(Movable):
    var first: Int
    var count: Int
    var offsets: List[Int32]
    var flags: List[Int32]
    var global_ids: List[Int]


def parallel_optimizer_step_host(
    ctx: DeviceContext,
    param_ptr: MutPointer[Float32, MutUntrackedOrigin],
    grad_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m_ptr: MutPointer[Float32, MutUntrackedOrigin],
    v_ptr: MutPointer[Float32, MutUntrackedOrigin],
    offsets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    init_ptr: MutPointer[Int32, MutUntrackedOrigin],
    info_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n_tensors: Int,
    kind: Int,
    t: Int,
    nesterov: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    momentum: Float32,
    dampening: Float32,
    max_norm: Float32,
) raises -> Int:
    var devices = Int(getenv("MOJOLEARN_OPTIMIZER_DEVICE_COUNT", "1"))
    if devices == 1:
        return identical_optimizer_step_host(ctx, param_ptr, grad_ptr, m_ptr,
            v_ptr, offsets_ptr, init_ptr, info_ptr, n_tensors, kind, t,
            nesterov, lr, beta1, beta2, eps, weight_decay, momentum,
            dampening, max_norm)
    if devices < 1 or devices > 64 or GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel optimizer requires IDENTICAL and 1..64 devices")
    comptime if STEP_PHASE_TIMERS:
        raise Error("parallel optimizer cannot use process-global phase counters")
    if kind != OPT_SGD and kind != OPT_ADAM and kind != OPT_ADAMW:
        raise Error("parallel optimizer: kind must be SGD, Adam or AdamW")
    if t < 1:
        raise Error("parallel optimizer: t is ONE-BASED")
    var offsets = _offsets_from_ptr(offsets_ptr, n_tensors)
    var n = offsets[n_tensors]
    var cfg = OptimizerConfig(kind, lr, beta1, beta2, eps, weight_decay,
        momentum, dampening, nesterov != 0, max_norm)
    _refuse_hyperparameters(cfg)
    devices = min(devices,n)
    # Full host staging is the transaction boundary. No worker allocates a
    # full model's parameter/moment buffers on its GPU.
    var p = List[Float32](length=n, fill=Float32(0))
    var g = List[Float32](length=n, fill=Float32(0))
    var m = List[Float32](length=n, fill=Float32(0))
    var v = List[Float32](length=n, fill=Float32(0))
    copy_f32(param_ptr,p.unsafe_ptr(),n)
    copy_f32(grad_ptr,g.unsafe_ptr(),n)
    copy_f32(m_ptr,m.unsafe_ptr(),n)
    copy_f32(v_ptr,v.unsafe_ptr(),n)
    var info = List[Float32](length=3,fill=Float32(0))
    if max_norm > Float32(0):
        # Whole tensors have owners; the original global norm uses their
        # canonical sumsq vector. Clipping storage is freed before updates.
        _ = parallel_clip_grad_norm_host(ctx,
            rebind[MutPointer[Float32, MutUntrackedOrigin]](g.unsafe_ptr()),
            offsets_ptr,
            rebind[MutPointer[Float32, MutUntrackedOrigin]](info.unsafe_ptr()+1),
            n_tensors,max_norm)
        info[0] = Float32(1)
    var contexts = List[DeviceContext]()
    var shards = List[OptimizerHostShard]()
    for rank in range(devices):
        contexts.append(DeviceContext(device_id=rank))
        var first = n*rank//devices
        var end = n*(rank+1)//devices
        var local_offsets: List[Int32] = [Int32(0)]
        var flags = List[Int32]()
        var ids = List[Int]()
        for j in range(n_tensors):
            var lo = max(first,offsets[j])
            var hi = min(end,offsets[j+1])
            if lo < hi:
                local_offsets.append(Int32(hi-first))
                flags.append(init_ptr[j])
                ids.append(j)
        shards.append(OptimizerHostShard(first,end-first,local_offsets^,flags^,ids^))
    var failures = List[Int](length=devices,fill=0)
    var cp = rebind[MutPointer[DeviceContext, MutUntrackedOrigin]](contexts.unsafe_ptr())
    var sp = rebind[MutPointer[OptimizerHostShard, MutUntrackedOrigin]](shards.unsafe_ptr())
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](failures.unsafe_ptr())
    var pp = rebind[MutPointer[Float32, MutUntrackedOrigin]](p.unsafe_ptr())
    var gp = rebind[MutPointer[Float32, MutUntrackedOrigin]](g.unsafe_ptr())
    var mp = rebind[MutPointer[Float32, MutUntrackedOrigin]](m.unsafe_ptr())
    var vp = rebind[MutPointer[Float32, MutUntrackedOrigin]](v.unsafe_ptr())
    def _update(rank: Int) {imm cp, imm sp, imm fp, imm pp, imm gp, imm mp, imm vp,
        imm kind, imm t, imm nesterov, imm lr, imm beta1, imm beta2,
        imm eps, imm weight_decay, imm momentum, imm dampening}:
        try:
            var first = sp[rank].first
            var unused = List[Float32](length=3,fill=Float32(0))
            _ = identical_optimizer_step_host(cp[rank],pp+first,gp+first,mp+first,vp+first,
                rebind[MutPointer[Int32, MutUntrackedOrigin]](sp[rank].offsets.unsafe_ptr()),
                rebind[MutPointer[Int32, MutUntrackedOrigin]](sp[rank].flags.unsafe_ptr()),
                rebind[MutPointer[Float32, MutUntrackedOrigin]](unused.unsafe_ptr()),
                len(sp[rank].flags),kind,t,nesterov,lr,beta1,beta2,eps,
                weight_decay,momentum,dampening,Float32(0))
            _ = unused^
        except:
            fp[rank] = 1
    sync_parallelize(_update,devices)
    for rank in range(devices):
        if failures[rank] != 0:
            raise Error("parallel optimizer: shard " + String(rank) + " refused; caller state unchanged")
    # Tensor boundaries can cross a device split. Both chunks start with the
    # same momentum flag and must finish with the same flag.
    var flags_out = List[Int32](length=n_tensors,fill=Int32(-1))
    for rank in range(devices):
        for local in range(len(shards[rank].flags)):
            var j = shards[rank].global_ids[local]
            var flag = shards[rank].flags[local]
            if flags_out[j] != -1 and flags_out[j] != flag:
                raise Error("parallel optimizer: inconsistent tensor momentum flag")
            flags_out[j] = flag
    # Untracked task pointers do not retain their owners. Keep contexts alive
    # through every join and all completed device-to-host copies.
    _ = contexts^
    copy_f32(p.unsafe_ptr(),param_ptr,n)
    copy_f32(m.unsafe_ptr(),m_ptr,n)
    copy_f32(v.unsafe_ptr(),v_ptr,n)
    if max_norm > Float32(0):
        copy_f32(g.unsafe_ptr(),grad_ptr,n)
    for j in range(n_tensors):
        init_ptr[j] = flags_out[j]
    copy_f32(info.unsafe_ptr(),info_ptr,3)
    return n
