# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The metrics lane's device runner: the arena goes up once, every stage of the
program is one launch of one thread per unit on the same stream (so stage s
sees every write of stage s-1), and the arena comes back once."""
from std.gpu import block_idx, block_dim, thread_idx
from std.ffi import _Global
from std.os import getenv
from std.time import perf_counter_ns
from std.memory import bitcast
from max.gpu.host import DeviceContext, DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_metrics.common import FP, IP, STAGE_INTS
from x_metrics.units import N_OPS, run_unit
from x_metrics.plan import Plan, plan_program, is_user_op, is_host_op, HOST_RD, HOST_WR, OP_SORT_MERGE
from x_metrics.par import sort_merge_path_unit, merge_path_chunks

comptime BLOCK = 128


struct _MetricsContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext for every x_metrics call (the
    x_cnn pattern: a context per call exhausts Metal's per-process command
    queues, and a cross-validation loop scores thousands of times). One slot
    per numeric tier, so a FAST and an IDENTICAL .so never share it."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXMetricsContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXMetricsContextFast"
comptime X_METRICS_CONTEXT = _Global[StorageType=_MetricsContext, name=_CTX_NAME, init_fn=_MetricsContext.__init__]


def metrics_ctx() raises -> DeviceContext:
    var slot = X_METRICS_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


def metrics_kernel[OP: Int](f: FP, q: IP, total: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(total):
        run_unit[OP](t, f, q)


def metrics_merge_path_kernel(f: FP, q: IP, total: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(total):
        sort_merge_path_unit(t, f, q)


def run_program_device(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int) raises:
    run_program_device_ptr(
        FP(unsafe_from_address=arena_addr), arena_len, IP(unsafe_from_address=prog_addr), stages
    )


def run_program_device_out(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int,
                           outs_addr: Int, nouts: Int) raises:
    """`run_program_device` that brings back only the caller's OUTPUT
    ranges (lane metrics-apple2): `nouts` Int32 quads [lo, hi, CNT, mult]
    at `outs_addr`, inside the arena. CNT < 0: [lo, hi) comes back; CNT >=
    0: the arena word CNT (an Int32 the program wrote) bounds it to
    [lo, lo + min(hi - lo, mult * CNT)), so a curve's unused tail stays on
    the device. The Apple GPU's device-to-host copy is its slowest link
    (2 to 3.4 GB/s), and the inputs and the order slots a caller never
    reads need not come back. Every other arena word keeps what the caller
    put there."""
    var o = IP(unsafe_from_address=outs_addr)
    for k in range(nouts):
        var lo = Int(o.unsafe_load(4 * k))
        var hi = Int(o.unsafe_load(4 * k + 1))
        var cn = Int(o.unsafe_load(4 * k + 2))
        if lo < 0 or hi < lo or hi > arena_len or cn >= arena_len or Int(o.unsafe_load(4 * k + 3)) < 0:
            raise Error("x_metrics: output range outside the arena")
    run_program_device_ptr(
        FP(unsafe_from_address=arena_addr), arena_len, IP(unsafe_from_address=prog_addr), stages,
        False, outs_addr, nouts,
    )


def run_program_device_ptr(host_f: FP, arena_len: Int, host_q: IP, stages: Int, legacy: Bool = False,
                           outs_addr: Int = 0, nouts: Int = -1) raises:
    """The PLANNED program (x_metrics/plan.mojo, the host runner's plan):
    the arena goes up once into a device buffer of arena + scratch, every
    planned stage is one launch on one stream, the caller's arena comes back
    once (only the `nouts` ranges at `outs_addr` when nouts >= 0). `legacy` runs
    the caller's stages unplanned (the seam gate)."""
    for s in range(stages):
        var op = Int(host_q.unsafe_load(s * STAGE_INTS))
        if not is_user_op(op):
            raise Error(String("x_metrics: unknown op ", op))
    var pl = Plan(arena_len)
    if legacy:
        for s in range(stages):
            pl.copy_stage(host_q, s)
    else:
        pl = plan_program(host_q, stages, arena_len)
    var nst = pl.stages
    var keep = List[List[Float32]]()
    var prof = getenv("MOJOLEARN_XMETRICS_PROFILE") != ""
    var t_setup = perf_counter_ns()
    var ctx = metrics_ctx()
    var df = ctx.enqueue_create_buffer[DType.float32](pl.size if pl.size > 0 else 1)
    var dq = ctx.enqueue_create_buffer[DType.int32](nst * STAGE_INTS if nst > 0 else 1)
    if prof:
        ctx.synchronize()
        print("XMPROF alloc us", (perf_counter_ns() - t_setup) // 1000, "floats", pl.size, "arena", arena_len)
    if arena_len > 0:
        ctx.enqueue_copy(dst_buf=df.create_sub_buffer[DType.float32](0, arena_len), src_ptr=host_f)
    if nst > 0:
        ctx.enqueue_copy(dst_buf=dq, src_ptr=pl.rows.unsafe_ptr())
    var t_last = perf_counter_ns()
    if prof:
        ctx.synchronize()
        t_last = perf_counter_ns()
        print("XMPROF setup plan+alloc+upload us", (t_last - t_setup) // 1000, "floats", pl.size)
    for s in range(nst):
        var op = Int(pl.rows[s * STAGE_INTS])
        var total = Int(pl.rows[s * STAGE_INTS + 1])
        if prof and s > 0:
            ctx.synchronize()
            var now = perf_counter_ns()
            print("XMPROF stage", s - 1, "op", Int(pl.rows[(s - 1) * STAGE_INTS]), "us", (now - t_last) // 1000)
            t_last = now
        if total <= 0:
            continue
        if is_host_op(op):
            _host_stage(ctx, df, pl, s, op, total, keep)
            continue
        var qp = dq.unsafe_ptr() + (s * STAGE_INTS + 2)
        if op == OP_SORT_MERGE and not legacy:
            # the device's merge schedule: one thread per MERGE_CHUNK outputs
            var chunks = merge_path_chunks(total, Int(pl.rows[s * STAGE_INTS + 2]))
            ctx.enqueue_function[metrics_merge_path_kernel](
                df.unsafe_ptr(), qp, Int32(chunks),
                grid_dim=(chunks + BLOCK - 1) // BLOCK, block_dim=BLOCK,
            )
            continue
        comptime for k in range(N_OPS):
            if op == k:
                ctx.enqueue_function[metrics_kernel[k]](
                    df.unsafe_ptr(), qp, Int32(total),
                    grid_dim=(total + BLOCK - 1) // BLOCK, block_dim=BLOCK,
                )
    if prof and nst > 0:
        ctx.synchronize()
        print("XMPROF stage", nst - 1, "op", Int(pl.rows[(nst - 1) * STAGE_INTS]), "us", (perf_counter_ns() - t_last) // 1000)
        t_last = perf_counter_ns()
    if nouts >= 0:
        var outs = IP(unsafe_from_address=outs_addr)
        var bounded = False
        for k in range(nouts):
            var cn = Int(outs.unsafe_load(4 * k + 2))
            if cn >= 0:
                bounded = True
                ctx.enqueue_copy(dst_ptr=host_f + cn, src_buf=df.create_sub_buffer[DType.float32](cn, 1))
        if bounded:
            ctx.synchronize()
        for k in range(nouts):
            var lo = Int(outs.unsafe_load(4 * k))
            var hi = Int(outs.unsafe_load(4 * k + 1))
            var cn = Int(outs.unsafe_load(4 * k + 2))
            if cn >= 0:
                var c = Int(bitcast[DType.int32](host_f.unsafe_load(cn))) * Int(outs.unsafe_load(4 * k + 3))
                hi = lo + max(0, min(hi - lo, c))
            if hi > lo:
                ctx.enqueue_copy(dst_ptr=host_f + lo, src_buf=df.create_sub_buffer[DType.float32](lo, hi - lo))
    elif arena_len > 0:
        ctx.enqueue_copy(dst_ptr=host_f, src_buf=df.create_sub_buffer[DType.float32](0, arena_len))
    ctx.synchronize()
    if prof:
        print("XMPROF download us", (perf_counter_ns() - t_last) // 1000)
    _ = len(keep)
    _ = len(pl.rows)
    _ = dq^
    _ = df^
    _ = ctx^


def _host_stage(
    ctx: DeviceContext, df: DeviceBuffer[DType.float32], pl: Plan, s: Int, op: Int, total: Int,
    mut keep: List[List[Float32]],
) raises:
    """A HOST stage (x_metrics/plan.mojo): its read slots come down, its
    units run in ascending t on the host (the host runner's loop, the same
    unit), its write slots go back up, all in stream order."""
    var row = s * STAGE_INTS + 2
    var rlo = Int(pl.rows[row + HOST_RD])
    var rhi = Int(pl.rows[row + HOST_RD + 1])
    var wlo = Int(pl.rows[row + HOST_WR])
    var whi = Int(pl.rows[row + HOST_WR + 1])
    var lo = min(rlo, wlo)
    var hi = max(rhi, whi)
    var hb = List[Float32](length=hi - lo, fill=Float32(0))
    var hp = hb.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    if rhi > rlo:
        ctx.enqueue_copy(dst_ptr=hp + (rlo - lo), src_buf=df.create_sub_buffer[DType.float32](rlo, rhi - rlo))
    ctx.synchronize()
    var hf = FP(unsafe_from_address=Int(hp) - 4 * lo)
    var hq = IP(unsafe_from_address=Int(pl.rows.unsafe_ptr()) + 4 * row)
    comptime for k in range(N_OPS):
        if op == k:
            for t in range(total):
                run_unit[k](t, hf, hq)
    if whi > wlo:
        ctx.enqueue_copy(dst_buf=df.create_sub_buffer[DType.float32](wlo, whi - wlo), src_ptr=hp + (wlo - lo))
    keep.append(hb^)
