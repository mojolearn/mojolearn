# SPDX-License-Identifier: Apache-2.0
"""Partition gradient columns; retain the original microbatch tree per cell.

Host copies transpose the storage ownership only. Every owner calls the
original accumulation entry with all microbatches in their original order.
The caller receives output only after every owner succeeds.
"""
from std.os import getenv
from std.sys import is_defined
from max.gpu.host import DeviceContext
from max.algorithm import sync_parallelize
from bindings.hostptr import copy_f32
from core.step_phase import STEP_PHASE_TIMERS
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from training.samba_ops import samba_accumulate_host, samba_validate_accumulation, _refuse_nonfinite


comptime ACCUMULATE_POOL_FAULT = is_defined["MOJOLEARN_ACCUMULATE_POOL_FAULT"]()


def accumulate_pool_fault_available() -> Int:
    return 1 if ACCUMULATE_POOL_FAULT else 0


def parallel_accumulate_host(
    ctx: DeviceContext,
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    parts_ptr: MutPointer[Float32, MutUntrackedOrigin],
    n: Int, a: Int, t_tokens: Int,
) raises -> Int:
    var devices = Int(getenv("MOJOLEARN_OPTIMIZER_DEVICE_COUNT", "1"))
    if devices == 1:
        return samba_accumulate_host(ctx,out_ptr,parts_ptr,n,a,t_tokens)
    if devices < 1 or devices > 64 or GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("parallel accumulation requires IDENTICAL and 1..64 devices")
    comptime if STEP_PHASE_TIMERS:
        raise Error("parallel accumulation cannot use process-global phase counters")
    samba_validate_accumulation(n,a,t_tokens)
    if n > 2147483647 or a > 9223372036854775807 // n:
        raise Error("parallel accumulation: shape exceeds native index bounds")
    _refuse_nonfinite("accumulate parts",parts_ptr,n*a)
    if a == 1:
        return samba_accumulate_host(ctx,out_ptr,parts_ptr,n,a,t_tokens)
    devices = min(devices,n)
    var result = List[Float32](length=n,fill=Float32(0))
    var contexts = List[DeviceContext]()
    for rank in range(devices):
        contexts.append(DeviceContext(device_id=rank))
    var failures = List[Int](length=devices,fill=0)
    var cp = rebind[MutPointer[DeviceContext, MutUntrackedOrigin]](contexts.unsafe_ptr())
    var rp = rebind[MutPointer[Float32, MutUntrackedOrigin]](result.unsafe_ptr())
    var fp = rebind[MutPointer[Int, MutUntrackedOrigin]](failures.unsafe_ptr())
    def _reduce(rank: Int) {imm cp, imm rp, imm fp, imm parts_ptr, imm n, imm a, imm t_tokens, imm devices}:
        try:
            var first = n*rank//devices
            var count = n*(rank+1)//devices-first
            var parts = List[Float32](length=a*count,fill=Float32(0))
            for microbatch in range(a):
                copy_f32(parts_ptr+microbatch*n+first,parts.unsafe_ptr()+microbatch*count,count)
            _ = samba_accumulate_host(cp[rank],rp+first,
                rebind[MutPointer[Float32, MutUntrackedOrigin]](parts.unsafe_ptr()),count,a,t_tokens)
            # Raw task pointers do not keep their owners alive.
            _ = parts^
            comptime if ACCUMULATE_POOL_FAULT:
                if Int(getenv("MOJOLEARN_ACCUMULATE_FAIL_RANK", "-1")) == rank:
                    raise Error("injected post-compute accumulation refusal")
        except:
            fp[rank] = 1
    sync_parallelize(_reduce,devices)
    for rank in range(devices):
        if failures[rank] != 0:
            raise Error("parallel accumulation: shard "+String(rank)+" refused; output unchanged")
    _ = contexts^
    copy_f32(result.unsafe_ptr(),out_ptr,n)
    return n
