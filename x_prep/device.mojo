# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The prep lane's device runner: the arena goes up once, every stage of the
program is one launch of one thread per unit on the same stream (so stage s
sees every write of stage s-1), and the arena comes back once."""
from std.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceContext
from x_prep.common import FP, IP, STAGE_INTS
from x_prep.units import N_OPS, run_unit

comptime BLOCK = 128


def prep_kernel[OP: Int](f: FP, q: IP, total: Int32):
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
            raise Error(String("x_prep: unknown op ", op))
    var ctx = DeviceContext()
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
                ctx.enqueue_function[prep_kernel[k]](
                    df.unsafe_ptr(), qp, Int32(total),
                    grid_dim=(total + BLOCK - 1) // BLOCK, block_dim=BLOCK,
                )
    if arena_len > 0:
        ctx.enqueue_copy(dst_ptr=host_f, src_buf=df)
    ctx.synchronize()
    _ = dq^
    _ = df^
    _ = ctx^
