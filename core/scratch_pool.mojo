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

The pool itself (`MOJOLEARN_IDN_SCRATCH_POOL`, never timed) is deleted:
every take allocates and every give frees, exactly the per-call form; the
take/give names stay so their callers keep one call site.

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
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.step_phase import step_count_device_alloc, step_count_host_alloc

# TOMBSTONE: MOJOLEARN_IDN_SCRATCH_POOL_OFF (old path, no A/B recorded beside the default) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at 2ec9cce908.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_SCRATCH_POOL_OFF.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (old path, no A/B recorded beside the default) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at 7d3bf6d8e8.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-04-base.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_IDN_SCRATCH_POOL (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-04-base; code recoverable at 25fca5d3e3.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_SCRATCH_POOL.patch; record in docs/TOMBSTONES.md.
# The name stays (core/device_scan.mojo imports it) as the folded constant.
comptime IDN_SCRATCH_POOL = False


def take_host_f32(ctx: DeviceContext, n: Int) raises -> HostBuffer[DType.float32]:
    """A pinned host buffer of at least `n` (>= 1) floats."""
    var want = n if n > 0 else 1
    step_count_host_alloc()
    return ctx.enqueue_create_host_buffer[DType.float32](want)


def give_host_f32(ctx: DeviceContext, var buf: HostBuffer[DType.float32]) raises:
    _ = buf^


def take_host_i32(ctx: DeviceContext, n: Int) raises -> HostBuffer[DType.int32]:
    """A pinned host buffer of at least `n` (>= 1) int32 words."""
    var want = n if n > 0 else 1
    step_count_host_alloc()
    return ctx.enqueue_create_host_buffer[DType.int32](want)


def give_host_i32(ctx: DeviceContext, var buf: HostBuffer[DType.int32]) raises:
    _ = buf^


def take_dev_i32(ctx: DeviceContext, n: Int) raises -> DeviceBuffer[DType.int32]:
    """A device buffer of at least `n` (>= 1) int32 words."""
    var want = n if n > 0 else 1
    step_count_device_alloc()
    return ctx.enqueue_create_buffer[DType.int32](want)


def give_dev_i32(ctx: DeviceContext, var buf: DeviceBuffer[DType.int32]) raises:
    _ = buf^
