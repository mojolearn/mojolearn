# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The metrics lane's device runner: the arena goes up once, every stage of the
program is one launch of one thread per unit on the same stream (so stage s
sees every write of stage s-1), and the arena comes back once."""
from std.gpu import block_idx, block_dim, thread_idx
from std.ffi import _Global
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_metrics.common import FP, IP, STAGE_INTS
from x_metrics.units import N_OPS, run_unit

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


def run_program_device(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int) raises:
    run_program_device_ptr(
        FP(unsafe_from_address=arena_addr), arena_len, IP(unsafe_from_address=prog_addr), stages
    )


def run_program_device_ptr(host_f: FP, arena_len: Int, host_q: IP, stages: Int) raises:
    for s in range(stages):
        var op = Int(host_q.unsafe_load(s * STAGE_INTS))
        if op < 0 or op >= N_OPS:
            raise Error(String("x_metrics: unknown op ", op))
    var ctx = metrics_ctx()
    var df = ctx.enqueue_create_buffer[DType.float32](arena_len if arena_len > 0 else 1)
    var dq = ctx.enqueue_create_buffer[DType.int32](stages * STAGE_INTS if stages > 0 else 1)
    if arena_len > 0:
        ctx.enqueue_copy(dst_buf=df, src_ptr=host_f)
    if stages > 0:
        ctx.enqueue_copy(dst_buf=dq, src_ptr=host_q)
    for s in range(stages):
        var op = Int(host_q.unsafe_load(s * STAGE_INTS))
        var total = Int(host_q.unsafe_load(s * STAGE_INTS + 1))
        if total <= 0:
            continue
        var qp = dq.unsafe_ptr() + (s * STAGE_INTS + 2)
        comptime for k in range(N_OPS):
            if op == k:
                ctx.enqueue_function[metrics_kernel[k]](
                    df.unsafe_ptr(), qp, Int32(total),
                    grid_dim=(total + BLOCK - 1) // BLOCK, block_dim=BLOCK,
                )
    if arena_len > 0:
        ctx.enqueue_copy(dst_ptr=host_f, src_buf=df)
    ctx.synchronize()
    _ = dq^
    _ = df^
    _ = ctx^
