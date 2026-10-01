# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A process pool for the small scratch buffers a host read needs (lane
neural-pass48, 2026-10-01): the pinned host mirror of a flag or a partials
slice, and the partials themselves.

The attention's corner flag read, its regime scan and the per-layer refuse
scans each allocated a pinned host buffer (and a partials buffer) per call
and freed it after the one read: on the MI325X a pinned allocation is slow
(`attn.bwd_corner_flag` read 0.47 ms a layer there against 0.025 on the
L40S, for a 4 B read), about seven such allocations a layer, fifty a train
step. The pool keeps the freed buffers by capacity and hands them back on
the next take; the words copied are the same, so no bit moves.
`MOJOLEARN_SCRATCH_POOL=0` restores the per-call allocations. Single
threaded by design, as the sequence executor's pool is: one session steps
at a time in a process.
"""
from std.ffi import _Global
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

#: free entries kept per kind; a give past this frees the buffer instead
comptime SCRATCH_POOL_KEEP = 64


struct _ScratchPool(Defaultable, Movable):
    var hf: List[HostBuffer[DType.float32]]
    var hf_n: List[Int]
    var hi: List[HostBuffer[DType.int32]]
    var hi_n: List[Int]
    var df: List[DeviceBuffer[DType.float32]]
    var df_n: List[Int]
    var di: List[DeviceBuffer[DType.int32]]
    var di_n: List[Int]

    def __init__(out self):
        self.hf = List[HostBuffer[DType.float32]]()
        self.hf_n = List[Int]()
        self.hi = List[HostBuffer[DType.int32]]()
        self.hi_n = List[Int]()
        self.df = List[DeviceBuffer[DType.float32]]()
        self.df_n = List[Int]()
        self.di = List[DeviceBuffer[DType.int32]]()
        self.di_n = List[Int]()


comptime _SCRATCH_NAME = "MojoScratchPoolIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoScratchPoolFast"
comptime SCRATCH_POOL = _Global[StorageType=_ScratchPool, name=_SCRATCH_NAME, init_fn=_ScratchPool.__init__]


def scratch_pool_on() -> Bool:
    return String(getenv("MOJOLEARN_SCRATCH_POOL")) != "0"


def _pick(caps: List[Int], n: Int) -> Int:
    """The smallest free entry that holds `n`, or -1."""
    var best = -1
    for i in range(len(caps)):
        if caps[i] >= n and (best < 0 or caps[i] < caps[best]):
            best = i
    return best


def take_host_f32(ctx: DeviceContext, n: Int) raises -> HostBuffer[DType.float32]:
    var want = n if n > 0 else 1
    if scratch_pool_on():
        var pool = SCRATCH_POOL.get_or_create_ptr()
        var i = _pick(pool[].hf_n, want)
        if i >= 0:
            _ = pool[].hf_n.pop(i)
            return pool[].hf.pop(i)
    return ctx.enqueue_create_host_buffer[DType.float32](want)


def give_host_f32(var buf: HostBuffer[DType.float32]) raises:
    if scratch_pool_on():
        var pool = SCRATCH_POOL.get_or_create_ptr()
        if len(pool[].hf) < SCRATCH_POOL_KEEP:
            pool[].hf_n.append(len(buf))
            pool[].hf.append(buf^)
            return
    _ = buf^


def take_host_i32(ctx: DeviceContext, n: Int) raises -> HostBuffer[DType.int32]:
    var want = n if n > 0 else 1
    if scratch_pool_on():
        var pool = SCRATCH_POOL.get_or_create_ptr()
        var i = _pick(pool[].hi_n, want)
        if i >= 0:
            _ = pool[].hi_n.pop(i)
            return pool[].hi.pop(i)
    return ctx.enqueue_create_host_buffer[DType.int32](want)


def give_host_i32(var buf: HostBuffer[DType.int32]) raises:
    if scratch_pool_on():
        var pool = SCRATCH_POOL.get_or_create_ptr()
        if len(pool[].hi) < SCRATCH_POOL_KEEP:
            pool[].hi_n.append(len(buf))
            pool[].hi.append(buf^)
            return
    _ = buf^


def take_dev_f32(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    var want = n if n > 0 else 1
    if scratch_pool_on():
        var pool = SCRATCH_POOL.get_or_create_ptr()
        var i = _pick(pool[].df_n, want)
        if i >= 0:
            _ = pool[].df_n.pop(i)
            return pool[].df.pop(i)
    return ctx.enqueue_create_buffer[DType.float32](want)


def give_dev_f32(var buf: DeviceBuffer[DType.float32]) raises:
    if scratch_pool_on():
        var pool = SCRATCH_POOL.get_or_create_ptr()
        if len(pool[].df) < SCRATCH_POOL_KEEP:
            pool[].df_n.append(len(buf))
            pool[].df.append(buf^)
            return
    _ = buf^


def take_dev_i32(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.int32]:
    var want = n if n > 0 else 1
    if scratch_pool_on():
        var pool = SCRATCH_POOL.get_or_create_ptr()
        var i = _pick(pool[].di_n, want)
        if i >= 0:
            _ = pool[].di_n.pop(i)
            return pool[].di.pop(i)
    return ctx.enqueue_create_buffer[DType.int32](want)


def give_dev_i32(var buf: DeviceBuffer[DType.int32]) raises:
    if scratch_pool_on():
        var pool = SCRATCH_POOL.get_or_create_ptr()
        if len(pool[].di) < SCRATCH_POOL_KEEP:
            pool[].di_n.append(len(buf))
            pool[].di.append(buf^)
            return
    _ = buf^
