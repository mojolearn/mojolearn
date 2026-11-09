# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Compile root of the executor's linear-algebra step ops (lane
# n7-device-linalg-ops, docs/plans/EXECUTOR_N7_OPS.md) until the family lanes
# N8-N10 consume them: every op and host twin instantiated once, on a tiny
# problem, so `mojo build checks/executor_n7_ops_compile.mojo` type-checks
# every kernel for the target. Running it enqueues 4 x 4 problems and prints
# the status words; it is not a verifier.
from max.gpu.host import DeviceContext

from cholesky.chol_ops import (
    chol_op_begin,
    chol_op_cho_solve,
    chol_op_factor,
    chol_op_info_joins_loop,
    chol_op_logdet,
    chol_op_logdet_scratch_floats,
)
from core.compact_ops import CompactScratch, compact_op_f32, compact_op_i32, compact_op_indices
from core.compact_ops_host import compact_indices_host
from core.device_zero import enqueue_fill
from core.sort_ops import SortScratch, sort_op_f32, sort_op_gather_rows_f32
from core.sort_ops_host import sort_op_f32_host
from core.step_gate import step_gate_open_word
from autograd.program_loop_device import LoopRegisters
from x_decomp.eigh_ops import eigh_op_dense, eigh_op_status_joins_loop, eigh_op_tridiag
from x_decomp.eigh_ops_host import eigh_op_dense_host, eigh_op_tridiag_host


def main() raises:
    var ctx = DeviceContext()
    var n = 4
    var gate = step_gate_open_word(ctx)
    var regs = LoopRegisters(ctx)
    regs.reset(ctx)
    # Cholesky
    var a = ctx.enqueue_create_buffer[DType.float32](n * n)
    enqueue_fill(ctx, a, Float32(0.0))
    var b = ctx.enqueue_create_buffer[DType.float32](n)
    enqueue_fill(ctx, b, Float32(1.0))
    var info = ctx.enqueue_create_buffer[DType.int32](1)
    var ld_ws = ctx.enqueue_create_buffer[DType.float32](chol_op_logdet_scratch_floats(n))
    var ld = ctx.enqueue_create_buffer[DType.float32](1)
    chol_op_begin(ctx, gate, info)
    chol_op_factor(ctx, a, info, n)
    chol_op_info_joins_loop(ctx, regs.words, info, 1)
    chol_op_cho_solve(ctx, a, b, info, n, 1)
    chol_op_logdet(ctx, a, info, ld_ws, ld, n)
    # eigh
    var m = ctx.enqueue_create_buffer[DType.float32](n * n)
    enqueue_fill(ctx, m, Float32(1.0))
    var w = ctx.enqueue_create_buffer[DType.float32](n)
    var v = ctx.enqueue_create_buffer[DType.float32](n * n)
    var st = ctx.enqueue_create_buffer[DType.int32](1)
    eigh_op_dense(ctx, gate, m, w, v, st, 1, n)
    eigh_op_status_joins_loop(ctx, regs.words, st, 1)
    var d = ctx.enqueue_create_buffer[DType.float32](n)
    enqueue_fill(ctx, d, Float32(2.0))
    var e = ctx.enqueue_create_buffer[DType.float32](n)
    enqueue_fill(ctx, e, Float32(-1.0))
    eigh_op_tridiag(ctx, gate, d, e, w, v, st, n)
    # sort
    var order = ctx.enqueue_create_buffer[DType.int32](n)
    var sorted = ctx.enqueue_create_buffer[DType.float32](n)
    var ss = SortScratch(ctx, n)
    sort_op_f32(ctx, gate, w, order, n, True, ss)
    sort_op_gather_rows_f32(ctx, gate, w, order, sorted, n, 1)
    # compaction
    var flags = ctx.enqueue_create_buffer[DType.uint8](n)
    enqueue_fill(ctx, flags, UInt8(1))
    var cs = CompactScratch(ctx, n)
    var idx = ctx.enqueue_create_buffer[DType.int32](n)
    var cnt = ctx.enqueue_create_buffer[DType.int32](1)
    var outf = ctx.enqueue_create_buffer[DType.float32](n)
    compact_op_indices(ctx, gate, flags, n, cs, idx, cnt)
    compact_op_i32(ctx, gate, order, flags, n, cs, idx, cnt)
    compact_op_f32(ctx, gate, w, flags, n, cs, outf, cnt)
    regs.read(ctx)
    print("N7_OPS_COMPILE failure_word=" + String(regs.failure()))
    # host twins
    var hm = List[Float32](length=n * n, fill=Float32(0.0))
    for i in range(n):
        hm[i * n + i] = Float32(i + 1)
    var hw = List[Float32](length=n, fill=Float32(0.0))
    var hv = List[Float32](length=n * n, fill=Float32(0.0))
    var hp = hm.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var hwp = hw.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var hvp = hv.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var bad = eigh_op_dense_host(hp, hwp, hvp, 1, n)
    var hd = List[Float32](length=n, fill=Float32(2.0))
    var he = List[Float32](length=n, fill=Float32(-1.0))
    bad += eigh_op_tridiag_host(
        hd.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), he.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), hwp, hvp, n
    )
    var ho = List[Int32](length=n, fill=Int32(0))
    var hop = ho.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    sort_op_f32_host(hwp, hop, n, True)
    var hf = List[UInt8](length=n, fill=UInt8(1))
    var k = compact_indices_host(hf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), n, hop)
    print("N7_OPS_COMPILE host_bad=" + String(bad) + " host_count=" + String(k))
    ctx.synchronize()
