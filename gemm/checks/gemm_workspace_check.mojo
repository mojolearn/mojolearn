# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Retained workspace: queued growth/reuse, all orientations, oracle bits.

Build with MOJOLEARN_NUMERIC_IDENTICAL=1 and MOJOLEARN_STEP_PHASE_TIMERS=1.
The counter assertions run without enabling synchronized phase timing.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from core.step_phase import step_counts_now
from core.device_scan_check import upload
from gemm.checks.gemm_identical import GemmWorkspace, identical_gemm_workspace_max_floats
from gemm.checks.gemm_oracle import gemm_oracle, OP_NN, OP_NT, OP_TN


def values(n: Int, salt: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(Float32((i * salt) % 31 - 15) * Float32(0.03125))
    return out^


def check(ctx: DeviceContext, mut got: DeviceBuffer[DType.float32],
          a: List[Float32], b: List[Float32], op: Int,
          m: Int, n: Int, k: Int) raises:
    var want = gemm_oracle(a, b, op, m, n, k)
    var host = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=got)
    ctx.synchronize()
    for i in range(m * n):
        if bitcast[DType.uint32](host.unsafe_ptr().unsafe_load(i)) != bitcast[DType.uint32](want[i]):
            raise Error("workspace GEMM differs from oracle at " + String(i))
    _ = host^


def main() raises:
    comptime assert is_defined["MOJOLEARN_STEP_PHASE_TIMERS"](), "enable counters"
    comptime assert is_defined["MOJOLEARN_NUMERIC_IDENTICAL"](), "IDENTICAL required"
    var ctx = DeviceContext()
    var small = identical_gemm_workspace_max_floats(16, 16, 513)
    var large = identical_gemm_workspace_max_floats(32, 32, 1025)
    if small <= 1 or large <= small:
        raise Error("fixture must exercise two growing split workspaces")
    var ops: List[Int] = [OP_NN, OP_NT, OP_TN]
    for oi in range(len(ops)):
        var op = ops[oi]
        var ah = values(16 * 513, 7)
        var bh = values(16 * 513, 11)
        var alh = values(32 * 1025, 13)
        var blh = values(32 * 1025, 17)
        var a = upload(ctx, ah)
        var b = upload(ctx, bh)
        var al = upload(ctx, alh)
        var bl = upload(ctx, blh)
        var c = ctx.enqueue_create_buffer[DType.float32](16 * 16)
        var cl = ctx.enqueue_create_buffer[DType.float32](32 * 32)
        var again = ctx.enqueue_create_buffer[DType.float32](16 * 16)
        var workspace = GemmWorkspace(ctx)
        # No caller wait between calls: growth must protect the old scratch;
        # reuse of the larger buffer must not overwrite pending partials.
        workspace.run[False](ctx, c, a, b, 16, 16, 513, op)
        workspace.run[False](ctx, cl, al, bl, 32, 32, 1025, op)
        var before = step_counts_now()
        workspace.run[False](ctx, again, a, b, 16, 16, 513, op)
        var after = step_counts_now()
        if after.device_allocs != before.device_allocs or after.syncs != before.syncs:
            raise Error("reused workspace must not allocate or wait")
        ctx.synchronize()
        check(ctx, c, ah, bh, op, 16, 16, 513)
        check(ctx, cl, alh, blh, op, 32, 32, 1025)
        check(ctx, again, ah, bh, op, 16, 16, 513)
        _ = workspace^
        _ = a^
        _ = b^
        _ = al^
        _ = bl^
        print("PASS workspace growth and queued reuse, op", op)
    print("PASS: 9 GEMMs, 4608 cells bitwise equal to host oracle")
