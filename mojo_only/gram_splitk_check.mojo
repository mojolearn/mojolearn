"""The split-K Gram path, per cell against a Float64 host oracle.

BOTH SIDES OF THE DISPATCH, BY NAME. `PORTING_RULES.md 8`: a switch is
exercised on both sides by a named check per side, with the switch set
explicitly inside the check. So `check_gram_splitk_oracle` calls
`gemm_tn_splitk` DIRECTLY, `check_gram_vendor_arm` calls
`gemm_tn_via_transpose` DIRECTLY at one shared shape, and
`check_gram_dispatch` asserts what the predicate decides and then runs the
real `gemm_tn` wrapper once per arm, printing which arm it took.

THE FIXTURE IS HASHED AND SCATTERED (splitmix64), the output is poisoned
before every call, and every cell is compared -- the uniform-fixture lesson
(`PORTING_RULES.md 7`): a check whose expected value repeats verifies a
total and nothing about placement, and a poison that survives is a cell
that was never written, which is worse than a wrong one.

THE SHAPES ARE THE HAZARDS, not round numbers:

- m = n = 1 with odd k: the degenerate Gram every one-feature fit reaches.
- k = 33 and k = 239: SMALLER than the 240-chunk grid, so most chunks must
  write all-zero partials and the reduce must not double-count them.
- k = 241 and k = 100003 (prime): k not a multiple of the chunk size, so
  the last live chunk is short.
- m = 33: does not divide the 256-thread block, so the guarded tail cells
  are live.
- m = 64 and m = 128: the CELLS = 16 and CELLS = 64 instantiations, which
  otherwise only wide fits would reach.
- m = 3 and m = 12: the scalar and vector sides of the staging copy's
  width split (`gram_splitk_stage_vectorized`, kernel-internal, keyed on
  `m % GRAM_STAGE_W`), at a k whose chunks are short and never
  tile-aligned. `check_gram_dispatch` asserts the predicate itself.

THE CENTERED-FUSED ARM IS HELD TO `!=`, NOT TO A TOLERANCE.
`check_gram_centered_fused` runs the exact shipped center pipeline
(`column_mean_kernel` then `shift_columns_kernel(-1)` then
`gemm_tn_splitk`) against `gram_centered_splitk` on the SAME device mu and
the same hashed X with a deliberately nonzero column mean, and compares
every cell BITWISE -- the fused tile load performs the identical fp32
subtraction the center pass stores, so any difference at all is a bug. It
also asserts X is bit-identical after the fused call (the arm's whole
point is that X is never written) and that the poison in z died (reach).

TOLERANCE: mag-relative 1e-5 + 1e-6 absolute, the same budget as the
`gemm_tn` vendor-table row and for the same reason -- the operands are one
matrix, the diagonal is a same-sign sum, and the budget covers fp32
accumulation-order spread (measured 4.09e-6 relative at 32x32x10007 on two
independent routes), not cancellation. BITWISE symmetry is asserted
separately with `!=` and no tolerance, exactly as
`check_covariance_is_symmetric` does, because the split-K kernel promises
it by construction.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.gemm import gemm_tn, gemm_tn_via_transpose
from core.column_stats import (
    STATS_TPB,
    column_mean_kernel,
    shift_columns_kernel,
)
from core.gram_splitk import (
    gemm_tn_splitk,
    gram_centered_splitk,
    gram_centered_splitk_into,
    gram_splitk_applies,
    gram_splitk_cells_for,
    gram_splitk_chunk_count,
    gram_splitk_reg_tiled,
    gram_splitk_stage_vectorized,
)


comptime ARM_SPLITK = 0
comptime ARM_TRANSPOSE = 1
comptime ARM_WRAPPER = 2


def _mix(i: Int, salt: Int) -> UInt64:
    """splitmix64. Adjacent indices land nowhere near each other."""
    var z = (
        UInt64(i + 1) * 0x9E3779B97F4A7C15
        + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _val_f32(i: Int, salt: Int) -> Float32:
    """A signed value in [-1, 1) for the arithmetic checks."""
    return Float32(Int(_mix(i, salt) % 2000001) - 1000000) * Float32(1.0e-6)


def _gram_one(ctx: DeviceContext, m: Int, k: Int, arm: Int) raises -> String:
    """One shape through one arm. Empty string on success.

    X is `k x m` row-major hashed, z is poisoned, and for the transpose and
    wrapper arms the alias scratch buffers are pre-poisoned too, so a
    transpose that does not run cannot pass on that arm.
    """
    var salt = m * 1000003 + k
    var x = ctx.enqueue_create_buffer[DType.float32](k * m)
    var z = ctx.enqueue_create_buffer[DType.float32](m * m)
    var xt = ctx.enqueue_create_buffer[DType.float32](k * m)
    var xt2 = ctx.enqueue_create_buffer[DType.float32](k * m)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](k * m)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](m * m)
    var hpois = ctx.enqueue_create_host_buffer[DType.float32](k * m)
    ctx.synchronize()
    for i in range(k * m):
        hx.unsafe_ptr().unsafe_store(i, _val_f32(i, salt))
        hpois.unsafe_ptr().unsafe_store(i, Float32(-123456.0))
    for i in range(m * m):
        hz.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=z, src_ptr=hz.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=xt, src_ptr=hpois.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=xt2, src_ptr=hpois.unsafe_ptr())
    ctx.synchronize()

    if arm == ARM_SPLITK:
        gemm_tn_splitk(ctx, z, x, m, k)
    elif arm == ARM_TRANSPOSE:
        gemm_tn_via_transpose(ctx, z, x, xt, xt2, m, m, k)
    else:
        gemm_tn(ctx, z, x, xt, xt2, m, m, k)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hz.unsafe_ptr(), src_buf=z)
    ctx.synchronize()

    for i in range(m):
        for j in range(m):
            var acc = Float64(0.0)
            var mag = Float64(0.0)
            for t in range(k):
                var av = Float64(hx.unsafe_ptr().unsafe_load(t * m + i))
                var bv = Float64(hx.unsafe_ptr().unsafe_load(t * m + j))
                acc += av * bv
                mag += abs(av * bv)
            var got = Float64(hz.unsafe_ptr().unsafe_load(i * m + j))
            if got == Float64(-987654.0):
                return (
                    "cell ("
                    + String(i)
                    + ", "
                    + String(j)
                    + ") was NEVER WRITTEN (poison survived)"
                )
            if abs(got - acc) > mag * 1.0e-5 + 1.0e-6:
                return (
                    "cell ("
                    + String(i)
                    + ", "
                    + String(j)
                    + ") device "
                    + String(got)
                    + " host "
                    + String(acc)
                )

    # Bitwise symmetry, no tolerance -- the same tripwire as
    # check_covariance_is_symmetric, at every shape this file runs.
    for i in range(m):
        for j in range(i + 1, m):
            var a = hz.unsafe_ptr().unsafe_load(i * m + j)
            var b = hz.unsafe_ptr().unsafe_load(j * m + i)
            if a != b:
                return (
                    "cells ("
                    + String(i)
                    + ", "
                    + String(j)
                    + ") and mirror are not bitwise equal: "
                    + String(a)
                    + " vs "
                    + String(b)
                )
    return String("")


def check_gram_splitk_oracle() raises:
    """`gemm_tn_splitk` DIRECTLY (the split-K side of the switch, by name),
    per cell against Float64, poison + bitwise symmetry, at the hazard
    shapes in the module docstring."""
    var ctx = DeviceContext()
    var n_chunks = gram_splitk_chunk_count()
    var ms: List[Int] = [1, 3, 8, 8, 8, 12, 33, 32, 64, 128]
    var ks: List[Int] = [
        7, 1021, 33, n_chunks - 1, n_chunks + 1, 1021, 257, 100003, 4001, 1025
    ]
    for s in range(len(ms)):
        var err = _gram_one(ctx, ms[s], ks[s], ARM_SPLITK)
        if err != "":
            raise Error(
                "check_gram_splitk_oracle FAILED at m="
                + String(ms[s])
                + " k="
                + String(ks[s])
                + ": "
                + err
            )
    print(
        "check_gram_splitk_oracle OK: split-K arm matches the Float64 oracle"
        " per cell and is bitwise symmetric at 10 shapes (m 1..128 covering"
        " all three CELLS widths and both staging-copy arms; k odd, prime,"
        " below/above the "
        + String(n_chunks)
        + "-chunk grid, and never a chunk multiple)"
    )


def _centered_one(
    ctx: DeviceContext, m: Int, k: Int, use_into: Bool
) raises -> String:
    """Center-then-split-K vs fused-centered-split-K, all cells bitwise.

    The fill is hashed AND given a per-column offset (`col * 0.25`) so every
    column mean is decidedly nonzero: a fused arm that silently read RAW X
    would differ in every cell, and a mu producer that returned zeros would
    too. z2 is poisoned, so a fused arm that never ran cannot pass either.
    """
    var salt = m * 2000003 + k
    var x = ctx.enqueue_create_buffer[DType.float32](k * m)
    var xc = ctx.enqueue_create_buffer[DType.float32](k * m)
    var mu = ctx.enqueue_create_buffer[DType.float32](m)
    var z1 = ctx.enqueue_create_buffer[DType.float32](m * m)
    var z2 = ctx.enqueue_create_buffer[DType.float32](m * m)
    var scratch = ctx.enqueue_create_buffer[DType.float32](k * m)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](k * m)
    var hz1 = ctx.enqueue_create_host_buffer[DType.float32](m * m)
    var hz2 = ctx.enqueue_create_host_buffer[DType.float32](m * m)
    var hx_after = ctx.enqueue_create_host_buffer[DType.float32](k * m)
    ctx.synchronize()
    for i in range(k * m):
        hx.unsafe_ptr().unsafe_store(
            i, _val_f32(i, salt) + Float32(i % m) * Float32(0.25)
        )
    for i in range(m * m):
        hz1.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=xc, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=z1, src_ptr=hz1.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=z2, src_ptr=hz1.unsafe_ptr())
    ctx.synchronize()

    # mu from the REAL producer, on the UNCENTERED data -- exactly what
    # compute_covariance feeds the fused arm.
    ctx.enqueue_function[column_mean_kernel](
        mu.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(k),
        Int32(m),
        grid_dim=(m, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    # Arm 1: the shipped center pass, then the plain split-K Gram.
    ctx.enqueue_function[shift_columns_kernel](
        xc.unsafe_ptr(),
        mu.unsafe_ptr(),
        Int32(k),
        Int32(m),
        Float32(-1.0),
        grid_dim=((k * m + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    gemm_tn_splitk(ctx, z1, xc, m, k)
    # Arm 2: the fused centered read on RAW x + the same device mu.
    if use_into:
        gram_centered_splitk_into(ctx, z2, x, mu, scratch, m, k)
    else:
        gram_centered_splitk(ctx, z2, x, mu, m, k)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hz1.unsafe_ptr(), src_buf=z1)
    ctx.enqueue_copy(dst_ptr=hz2.unsafe_ptr(), src_buf=z2)
    ctx.enqueue_copy(dst_ptr=hx_after.unsafe_ptr(), src_buf=x)
    ctx.synchronize()

    for i in range(m * m):
        var b = hz2.unsafe_ptr().unsafe_load(i)
        if b == Float32(-987654.0):
            return (
                "fused cell " + String(i)
                + " was NEVER WRITTEN (poison survived)"
            )
        var a = hz1.unsafe_ptr().unsafe_load(i)
        if a != b:
            return (
                "cell " + String(i) + " center-then-gemm " + String(a)
                + " fused " + String(b) + " NOT bitwise equal"
            )
    for i in range(k * m):
        if (
            hx_after.unsafe_ptr().unsafe_load(i)
            != hx.unsafe_ptr().unsafe_load(i)
        ):
            return (
                "x[" + String(i) + "] was MODIFIED by the fused arm, which"
                " promises a read-only X"
            )
    return String("")


def check_gram_centered_fused() raises:
    """The fused-epilogue arm (RAFT's `stable=false` @todo,
    `raft/stats/detail/cov.cuh:67-69`; DEVIATION 42) is BIT-IDENTICAL to
    the shipped center-then-split-K pipeline, and X survives untouched.

    Shapes: the checks' 4-wide PCA shape; the bench-family 32-wide shape at
    prime k through BOTH workspace paths (allocating and scratch-reusing);
    and m = 33, which does not divide the 256-thread block, so the
    non-hoisted accumulation column runs centered too.
    """
    var ms: List[Int] = [4, 32, 32, 33]
    var ks: List[Int] = [8192, 100003, 100003, 257]
    var intos: List[Bool] = [False, False, True, False]
    var ctx = DeviceContext()
    for i in range(len(ms)):
        var err = _centered_one(ctx, ms[i], ks[i], intos[i])
        if err != "":
            raise Error(
                "check_gram_centered_fused FAILED at m=" + String(ms[i])
                + " k=" + String(ks[i])
                + (" (into)" if intos[i] else "") + ": " + err
            )
    print(
        "check_gram_centered_fused OK: fused centered read is bitwise equal"
        " to center-then-split-K at every cell (m=4/32/33, k=8192/100003/257,"
        " both workspace paths), and x is bit-identical afterwards"
    )


def check_gram_vendor_arm() raises:
    """`gemm_tn_via_transpose` DIRECTLY (the vendor side of the switch, by
    name), at one shape `check_gram_splitk_oracle` also runs, so the two
    arms are judged against the same oracle at the same numbers."""
    var ctx = DeviceContext()
    var err = _gram_one(ctx, 33, 257, ARM_TRANSPOSE)
    if err != "":
        raise Error(
            "check_gram_vendor_arm FAILED at m=33 k=257: " + err
        )
    print(
        "check_gram_vendor_arm OK: transpose+matmul arm matches the Float64"
        " oracle per cell and is bitwise symmetric at 33x33x257"
    )


def check_gram_dispatch() raises:
    """The predicate decides what it was derived to decide, and the REAL
    `gemm_tn` wrapper is run once per arm, printing which arm it took."""
    # Tile-starved outputs (fewer vendor tiles than resident-block slots)
    # go split-K; everything else stays on the vendor matmul.
    if not gram_splitk_applies(32, 32, 4000000):
        raise Error("dispatch: 32x32x4M must take split-K and does not")
    if not gram_splitk_applies(1, 1, 7):
        raise Error("dispatch: 1x1x7 must take split-K and does not")
    if not gram_splitk_applies(128, 128, 1025):
        raise Error("dispatch: 128x128 must take split-K and does not")
    if gram_splitk_applies(129, 129, 1025):
        raise Error(
            "dispatch: 129 columns exceeds the staging tile and must fall"
            " back, and does not"
        )
    if gram_splitk_applies(768, 768, 257):
        raise Error(
            "dispatch: a 768x768 output has 144 vendor tiles >= the block"
            " slots and must stay on the vendor matmul, and does not"
        )
    if gram_splitk_applies(8, 4, 100):
        raise Error("dispatch: m != n is not a Gram and must fall back")

    # The staging copy's width split, same discipline: ONE predicate
    # (`gram_splitk_stage_vectorized`, the same symbol the kernel body
    # branches on) asserted here, and BOTH its arms held to the per-cell
    # oracle by `check_gram_splitk_oracle`'s m = 12 (vector) and
    # m = 3 / 33 / 1 (scalar) shapes. Reach of the vector arm was also
    # proven destructively once: +1.0 planted in the vector load failed
    # the oracle at every m % 4 == 0 shape and passed the scalar shapes
    # (LANE_splitk-interior_2026-08-20.md).
    if not gram_splitk_stage_vectorized(32):
        raise Error("stage arms: m=32 (the bench width) must vectorize")
    if not gram_splitk_stage_vectorized(4):
        raise Error("stage arms: m=4 (the PCA checks' width) must vectorize")
    if not gram_splitk_stage_vectorized(128):
        raise Error("stage arms: m=128 (the widest fit) must vectorize")
    if gram_splitk_stage_vectorized(33):
        raise Error("stage arms: m=33 must take the scalar arm")
    if gram_splitk_stage_vectorized(1):
        raise Error("stage arms: m=1 must take the scalar arm")

    # The accumulation loop's ownership split, same discipline a third
    # time: ONE predicate (`gram_splitk_reg_tiled`, the symbol the kernel
    # body branches on) asserted here, with `gram_splitk_cells_for` (the
    # function the launchers call) pinning which CELLS instantiation each
    # width reaches, and BOTH ownership arms held to the per-cell oracle by
    # `check_gram_splitk_oracle`'s m = 8/12/32/64/128 (register-tile) and
    # m = 1/3/33 (strided) shapes. Reach of the register-tile arm was also
    # proven destructively once: +1.0 planted in its inner FMA failed the
    # oracle at every tiled shape and left the strided shapes' bits
    # untouched, and the mirrored sabotage of the strided arm inverted
    # that split (LANE_gram-tile_2026-08-20.md).
    if gram_splitk_cells_for(32) != 4 or gram_splitk_cells_for(33) != 16:
        raise Error("reg tile: the CELLS width dispatch moved (narrow)")
    if gram_splitk_cells_for(64) != 16 or gram_splitk_cells_for(128) != 64:
        raise Error("reg tile: the CELLS width dispatch moved (wide)")
    if not gram_splitk_reg_tiled[4](32):
        raise Error("reg tile: m=32 (the bench width) must take the 2x2 arm")
    if not gram_splitk_reg_tiled[16](64):
        raise Error("reg tile: m=64 must take the 4x4 arm")
    if not gram_splitk_reg_tiled[64](128):
        raise Error("reg tile: m=128 must take the 8x8 arm")
    if gram_splitk_reg_tiled[4](1) or gram_splitk_reg_tiled[4](3):
        raise Error("reg tile: m=1/3 must keep the strided arm")
    if gram_splitk_reg_tiled[16](33):
        raise Error("reg tile: m=33 must keep the strided arm")

    var ctx = DeviceContext()
    # One wrapper run per arm, same oracle machinery.
    var arm_a = String("split-K") if gram_splitk_applies(
        32, 32, 100003
    ) else String("transpose+matmul")
    var err = _gram_one(ctx, 32, 100003, ARM_WRAPPER)
    if err != "":
        raise Error(
            "check_gram_dispatch FAILED through the wrapper (arm "
            + arm_a
            + ") at 32x32x100003: "
            + err
        )
    var arm_b = String("split-K") if gram_splitk_applies(
        768, 768, 257
    ) else String("transpose+matmul")
    var err2 = _gram_one(ctx, 768, 257, ARM_WRAPPER)
    if err2 != "":
        raise Error(
            "check_gram_dispatch FAILED through the wrapper (arm "
            + arm_b
            + ") at 768x768x257: "
            + err2
        )
    print(
        "check_gram_dispatch OK: predicate routes 32x32x4M/1x1x7/128x128 to"
        " split-K and 129x129/768x768/m!=n to the fallback; staging copy"
        " vectorizes m=32/4/128 and falls back scalar at m=33/1; register"
        " tile owns m=32/64/128 (2x2/4x4/8x8) and declines m=1/3/33 to the"
        " strided arm; wrapper"
        " verified per cell on arm '"
        + arm_a
        + "' at 32x32x100003 and arm '"
        + arm_b
        + "' at 768x768x257"
    )
