# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What IDENTICAL costs on the three matrix products PCA, tSVD and OLS run.

Driven by `tools/price_linalg_identity.sh`, which alternates the two modes
round by round. Emits the same `PRICE <mode> <arm> <ms>` lines
`bench/identity_price_main.mojo` emits, so the shell driver's median table
is shared and the two files' numbers are directly comparable.

WHY THIS IS A SECOND FILE AND NOT THREE MORE ARMS IN THAT ONE. It is a hot
file owned by the k-NN lane and under edit while this lane was written
(DEVIATION 509), and cross-lane edits to hot files are how two sessions
collide. THE DEBT IS NAMED, NOT HIDDEN: the two mains share a report format
and a fixture idiom and should be merged into one driver with an arm filter
when either lane next touches the other's file. Nothing here is a second
opinion about anything -- there is no numeric code in this file at all, only
launches and a clock.

THE THREE ARMS, and each is a real call site rather than a microbenchmark:

  gram.32x32x1M     `gemm_tn`, the Gram shape `A^T A` at the shipped
                    PCA/OLS aspect: 32 features, a million rows. Under FAST
                    on the Apple column this takes the split-K kernel at
                    240 chunks; under IDENTICAL the same kernel at the
                    pinned 128 (DEVIATION 520). So this arm prices the PIN,
                    not a change of kernel, which is the cleanest reading
                    available: same code, two partitions.
  nt.4096x64x64     `gemm_nt`, the N-T product. Under FAST this is MAX's
                    tuned matmul; under IDENTICAL it is
                    `pinned_gemm_nt_kernel`, one thread per output cell
                    (DEVIATION 526). So this arm prices the REPLACEMENT of
                    a closed vendor library, and it is the expensive one by
                    construction -- IDENTITY_PATHS row 24 measured 2.85x
                    for the same trade on the k-NN distance step.
  gemv.128x128      `gemv_n`, OLS's step 6. Tiny, and priced anyway,
                    because "it is small so it cannot matter" is an
                    argument and this file exists to replace arguments with
                    seconds.

READ THE RATIO WITH A WIDE BAND. Two modes are two BINARIES, so this cannot
interleave inside one process; and the M4's governor drifts up to 1.7x over
twenty minutes, which is why the driver alternates rounds rather than
running a block of each.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.time import perf_counter_ns

from core.gemm import gemm_nt, gemm_tn, gemv_n
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name


comptime REPEATS = 3


def _mode() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. A local
    two-way IDENTICAL-or-FAST answers "FAST" for a DETERMINISTIC
    build, mislabelling every line the driver prints.
    """
    return numeric_mode_name()


def _mix(i: Int, salt: Int) -> UInt64:
    var z = (
        UInt64(i + 1) * 0x9E3779B97F4A7C15
        + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _val(i: Int, salt: Int) -> Float32:
    return Float32(Int(_mix(i, salt) % 2000001) - 1000000) * Float32(1.0e-6)


def _fill(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32],
          n: Int, salt: Int) raises:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr().unsafe_store(i, _val(i, salt))
    ctx.enqueue_copy(dst_buf=buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()


def _report(name: String, ms: Float64):
    print("PRICE", _mode(), name, ms)


def _gram_arm(ctx: DeviceContext) raises:
    """`gemm_tn` at the shipped PCA/OLS Gram aspect."""
    var m = 32
    var k = 1_000_000
    var x = ctx.enqueue_create_buffer[DType.float32](k * m)
    var z = ctx.enqueue_create_buffer[DType.float32](m * m)
    var xt = ctx.enqueue_create_buffer[DType.float32](k * m)
    var xt2 = ctx.enqueue_create_buffer[DType.float32](k * m)
    _fill(ctx, x, k * m, 11)

    # One untimed call: the first launch of a kernel pays its compile and
    # its cache misses, and a price that includes those is a price for a
    # thing nobody runs twice.
    gemm_tn(ctx, z, x, xt, xt2, m, m, k)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(REPEATS):
        gemm_tn(ctx, z, x, xt, xt2, m, m, k)
    ctx.synchronize()
    _report(
        String("gram.32x32x1M"),
        Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(REPEATS),
    )
    _ = x
    _ = z
    _ = xt
    _ = xt2


def _nt_arm(ctx: DeviceContext) raises:
    """`gemm_nt`: the tuned vendor matmul against the pinned cell kernel."""
    var m = 4096
    var n = 64
    var k = 64
    var x = ctx.enqueue_create_buffer[DType.float32](m * k)
    var y = ctx.enqueue_create_buffer[DType.float32](n * k)
    var z = ctx.enqueue_create_buffer[DType.float32](m * n)
    _fill(ctx, x, m * k, 21)
    _fill(ctx, y, n * k, 22)

    gemm_nt(ctx, z, x, y, m, n, k)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(REPEATS):
        gemm_nt(ctx, z, x, y, m, n, k)
    ctx.synchronize()
    _report(
        String("nt.4096x64x64"),
        Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(REPEATS),
    )
    _ = x
    _ = y
    _ = z


def _gemv_arm(ctx: DeviceContext) raises:
    """`gemv_n`: OLS's step 6 at its shipped size."""
    var m = 128
    var k = 128
    var x = ctx.enqueue_create_buffer[DType.float32](m * k)
    var y = ctx.enqueue_create_buffer[DType.float32](k)
    var z = ctx.enqueue_create_buffer[DType.float32](m)
    _fill(ctx, x, m * k, 31)
    _fill(ctx, y, k, 32)

    gemv_n(ctx, z, x, y, m, k)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(REPEATS):
        gemv_n(ctx, z, x, y, m, k)
    ctx.synchronize()
    _report(
        String("gemv.128x128"),
        Float64(perf_counter_ns() - t0) / 1.0e6 / Float64(REPEATS),
    )
    _ = x
    _ = y
    _ = z


def main() raises:
    with DeviceContext() as ctx:
        _gram_arm(ctx)
        _nt_arm(ctx)
        _gemv_arm(ctx)
