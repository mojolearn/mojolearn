# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep lane's device runner: the arena goes up once, every stage of the
program is one launch of one thread per unit on the same stream (so stage s
sees every write of stage s-1), and the arena comes back once."""
from std.gpu import block_idx, block_dim, thread_idx
from std.ffi import _Global
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, STAGE_INTS
from x_prep.units import N_OPS, run_unit
from x_prep.dsort import sort_cols_device, sort_scratch_words

#: op 0 (`sort_cols`) runs as the device sort of x_prep/dsort.mojo, not as
#: one heapsort thread per column: the same words (a sort under a total
#: order has one answer), at every thread of the GPU.
comptime OP_SORT_COLS = 0

comptime BLOCK = 128


struct _PrepContext(Defaultable, Movable):
    """ONE process-lifetime DeviceContext for every `x_prep_run` (the x_cnn
    `_Global` pattern; CURRENT DIRECTIVES, 2026-09-27: a context per call hung
    the SECOND call of x_cluster / x_neighbors on an RTX 4090, and on Metal a
    context per call exhausts the per-process command queues). The slot keeps
    a reference for the life of the process, so every call's buffers die
    inside it. One slot per numeric tier, so a FAST and an IDENTICAL .so in
    one process never share it. Context lifetime moves no bit."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXPrepContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXPrepContextFast"
comptime X_PREP_CONTEXT = _Global[StorageType=_PrepContext, name=_CTX_NAME, init_fn=_PrepContext.__init__]


def x_prep_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_PREP_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


def prep_kernel[OP: Int](f: FP, q: IP, total: Int32):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(total):
        run_unit[OP](t, f, q)


def run_program_device(arena_addr: Int, arena_len: Int, prog_addr: Int, stages: Int, upload: Int = -1) raises:
    run_program_device_ptr(
        FP(unsafe_from_address=arena_addr), arena_len, IP(unsafe_from_address=prog_addr), stages, upload
    )


def run_program_device_ptr(host_f: FP, arena_len: Int, host_q: IP, stages: Int, upload: Int = -1) raises:
    """`upload` (lane prep-apple): when 0 <= upload < arena_len, only the
    arena's first `upload` words are copied up and the rest is zeroed on the
    device: the caller promises they are zero on the host (the program's
    outputs and scratch, laid out after its last input). A dense output (a
    one-hot block of 1M x 570) is then never copied up, only back."""
    for s in range(stages):
        var op = Int(host_q.unsafe_load(s * STAGE_INTS))
        if op < 0 or op >= N_OPS:
            raise Error(String("x_prep: unknown op ", op))
    var scratch = 1
    for s in range(stages):
        if Int(host_q.unsafe_load(s * STAGE_INTS)) == OP_SORT_COLS:
            var sq = host_q + (s * STAGE_INTS + 2)
            scratch = max(scratch, sort_scratch_words(Int(sq[1]), Int(host_q.unsafe_load(s * STAGE_INTS + 1))))
    var ctx = x_prep_ctx()
    var df = ctx.enqueue_create_buffer[DType.float32](arena_len if arena_len > 0 else 1)
    var dw = ctx.enqueue_create_buffer[DType.uint32](scratch)
    var dq = ctx.enqueue_create_buffer[DType.int32](stages * STAGE_INTS if stages > 0 else 1)
    var up = arena_len if upload < 0 or upload > arena_len else upload
    if arena_len > 0:
        if up < arena_len:
            ctx.enqueue_memset(df, Float32(0))
            if up > 0:
                var head = df.create_sub_buffer[DType.float32](0, up)
                ctx.enqueue_copy(dst_buf=head, src_ptr=host_f)
        else:
            ctx.enqueue_copy(dst_buf=df, src_ptr=host_f)
    if stages > 0:
        ctx.enqueue_copy(dst_buf=dq, src_ptr=host_q)
    for s in range(stages):
        var op = Int(host_q.unsafe_load(s * STAGE_INTS))
        var total = Int(host_q.unsafe_load(s * STAGE_INTS + 1))
        if total <= 0:
            continue
        var qp = dq.unsafe_ptr() + (s * STAGE_INTS + 2)
        if op == OP_SORT_COLS:
            var hq = host_q + (s * STAGE_INTS + 2)
            sort_cols_device(ctx, df, dw, total, Int(hq[0]), Int(hq[1]), Int(hq[2]),
                             Int(hq[3]), Int(hq[4]))
            continue
        comptime for k in range(N_OPS):
            if op == k:
                ctx.enqueue_function[prep_kernel[k]](
                    df.unsafe_ptr(), qp, Int32(total),
                    grid_dim=(total + BLOCK - 1) // BLOCK, block_dim=BLOCK,
                )
    if arena_len > 0:
        ctx.enqueue_copy(dst_ptr=host_f, src_buf=df)
    ctx.synchronize()
    _ = dw^
    _ = dq^
    _ = df^
    _ = ctx^
