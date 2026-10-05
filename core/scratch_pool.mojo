# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A process pool for the small scratch buffers a host read needs: the
pinned host mirror of a flag or a partials slice, and the partials
themselves. Ported by hand from lane/neural-pass48 (4d164cce7, 2026-10-01)
onto current source by lane fix-r1-rescue (2026-10-04).

The attention's corner flag read, its regime scan and the per-layer refuse
scans each allocated a pinned host buffer (and a partials buffer) per call
and freed it after the one read: on the MI325X a pinned allocation is slow
(`attn.bwd_corner_flag` read 0.47 ms a layer there against 0.025 on the
L40S, for a 4 B read), about seven such allocations a layer. The pool keeps
the freed buffers by capacity and hands them back on the next take; the
words copied are the same, so no bit moves.

DEFAULT OFF CANDIDATE (never timed on current source): `-D
MOJOLEARN_IDN_SCRATCH_POOL` turns it on; `-D MOJOLEARN_IDN_SCRATCH_POOL_OFF`
or `-D MOJOLEARN_IDN_ALL_OFF` keeps it off. Off, every take allocates and
every give frees, exactly the per-call form.

Changes against the old branch: (1) the process environment switch
`MOJOLEARN_SCRATCH_POOL=0` became a compile define; (2) every entry carries
its context key (`core/ctx_key.mojo`), so a buffer made on one
`DeviceContext` is never handed to another (two-GPU drivers run two contexts
in one process); (3) a pooled buffer can be LARGER than asked: callers copy
through a sub-buffer of the asked length, never the whole buffer; (4) the
step-phase allocation counters count only real allocations.

Single threaded by design, as the sequence executor's pool is: one session
steps at a time in a process. A buffer is given back only after the wait
that ends its last use.
"""
from std.ffi import _Global
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.ctx_key import ctx_cache_key
from core.step_phase import step_count_device_alloc, step_count_host_alloc

comptime IDN_SCRATCH_POOL = is_defined["MOJOLEARN_IDN_SCRATCH_POOL"]() and not (
    is_defined["MOJOLEARN_IDN_SCRATCH_POOL_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: free entries kept per kind; a give past this frees the buffer instead
comptime SCRATCH_POOL_KEEP = 64


struct _ScratchPool(Defaultable, Movable):
    var hf: List[HostBuffer[DType.float32]]
    var hf_n: List[Int]
    var hf_k: List[Int]
    var hi: List[HostBuffer[DType.int32]]
    var hi_n: List[Int]
    var hi_k: List[Int]
    var df: List[DeviceBuffer[DType.float32]]
    var df_n: List[Int]
    var df_k: List[Int]
    var di: List[DeviceBuffer[DType.int32]]
    var di_n: List[Int]
    var di_k: List[Int]

    def __init__(out self):
        self.hf = List[HostBuffer[DType.float32]]()
        self.hf_n = List[Int]()
        self.hf_k = List[Int]()
        self.hi = List[HostBuffer[DType.int32]]()
        self.hi_n = List[Int]()
        self.hi_k = List[Int]()
        self.df = List[DeviceBuffer[DType.float32]]()
        self.df_n = List[Int]()
        self.df_k = List[Int]()
        self.di = List[DeviceBuffer[DType.int32]]()
        self.di_n = List[Int]()
        self.di_k = List[Int]()


comptime _SCRATCH_NAME = "MojoScratchPoolIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoScratchPoolFast"
comptime SCRATCH_POOL = _Global[StorageType=_ScratchPool, name=_SCRATCH_NAME, init_fn=_ScratchPool.__init__]


def _pick(caps: List[Int], keys: List[Int], key: Int, n: Int) -> Int:
    """The smallest free entry of context `key` that holds `n`, or -1."""
    var best = -1
    for i in range(len(caps)):
        if keys[i] == key and caps[i] >= n and (best < 0 or caps[i] < caps[best]):
            best = i
    return best


def take_host_f32(ctx: DeviceContext, n: Int) raises -> HostBuffer[DType.float32]:
    """A pinned host buffer of at least `n` (>= 1) floats."""
    var want = n if n > 0 else 1
    comptime if IDN_SCRATCH_POOL:
        var key = ctx_cache_key(ctx)
        var pool = SCRATCH_POOL.get_or_create_ptr()
        var i = _pick(pool[].hf_n, pool[].hf_k, key, want)
        if i >= 0:
            _ = pool[].hf_n.pop(i)
            _ = pool[].hf_k.pop(i)
            return pool[].hf.pop(i)
    step_count_host_alloc()
    return ctx.enqueue_create_host_buffer[DType.float32](want)


def give_host_f32(ctx: DeviceContext, var buf: HostBuffer[DType.float32]) raises:
    comptime if IDN_SCRATCH_POOL:
        var pool = SCRATCH_POOL.get_or_create_ptr()
        if len(pool[].hf) < SCRATCH_POOL_KEEP:
            pool[].hf_n.append(len(buf))
            pool[].hf_k.append(ctx_cache_key(ctx))
            pool[].hf.append(buf^)
            return
    _ = buf^


def take_host_i32(ctx: DeviceContext, n: Int) raises -> HostBuffer[DType.int32]:
    """A pinned host buffer of at least `n` (>= 1) int32 words."""
    var want = n if n > 0 else 1
    comptime if IDN_SCRATCH_POOL:
        var key = ctx_cache_key(ctx)
        var pool = SCRATCH_POOL.get_or_create_ptr()
        var i = _pick(pool[].hi_n, pool[].hi_k, key, want)
        if i >= 0:
            _ = pool[].hi_n.pop(i)
            _ = pool[].hi_k.pop(i)
            return pool[].hi.pop(i)
    step_count_host_alloc()
    return ctx.enqueue_create_host_buffer[DType.int32](want)


def give_host_i32(ctx: DeviceContext, var buf: HostBuffer[DType.int32]) raises:
    comptime if IDN_SCRATCH_POOL:
        var pool = SCRATCH_POOL.get_or_create_ptr()
        if len(pool[].hi) < SCRATCH_POOL_KEEP:
            pool[].hi_n.append(len(buf))
            pool[].hi_k.append(ctx_cache_key(ctx))
            pool[].hi.append(buf^)
            return
    _ = buf^


def take_dev_f32(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.float32]:
    """A device buffer of at least `n` (>= 1) floats."""
    var want = n if n > 0 else 1
    comptime if IDN_SCRATCH_POOL:
        var key = ctx_cache_key(ctx)
        var pool = SCRATCH_POOL.get_or_create_ptr()
        var i = _pick(pool[].df_n, pool[].df_k, key, want)
        if i >= 0:
            _ = pool[].df_n.pop(i)
            _ = pool[].df_k.pop(i)
            return pool[].df.pop(i)
    step_count_device_alloc()
    return ctx.enqueue_create_buffer[DType.float32](want)


def give_dev_f32(ctx: DeviceContext, var buf: DeviceBuffer[DType.float32]) raises:
    comptime if IDN_SCRATCH_POOL:
        var pool = SCRATCH_POOL.get_or_create_ptr()
        if len(pool[].df) < SCRATCH_POOL_KEEP:
            pool[].df_n.append(len(buf))
            pool[].df_k.append(ctx_cache_key(ctx))
            pool[].df.append(buf^)
            return
    _ = buf^


def take_dev_i32(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.int32]:
    """A device buffer of at least `n` (>= 1) int32 words."""
    var want = n if n > 0 else 1
    comptime if IDN_SCRATCH_POOL:
        var key = ctx_cache_key(ctx)
        var pool = SCRATCH_POOL.get_or_create_ptr()
        var i = _pick(pool[].di_n, pool[].di_k, key, want)
        if i >= 0:
            _ = pool[].di_n.pop(i)
            _ = pool[].di_k.pop(i)
            return pool[].di.pop(i)
    step_count_device_alloc()
    return ctx.enqueue_create_buffer[DType.int32](want)


def give_dev_i32(ctx: DeviceContext, var buf: DeviceBuffer[DType.int32]) raises:
    comptime if IDN_SCRATCH_POOL:
        var pool = SCRATCH_POOL.get_or_create_ptr()
        if len(pool[].di) < SCRATCH_POOL_KEEP:
            pool[].di_n.append(len(buf))
            pool[].di_k.append(ctx_cache_key(ctx))
            pool[].di.append(buf^)
            return
    _ = buf^
