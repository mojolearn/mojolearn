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
from core.gram_splitk import (
    gemm_tn_splitk,
    gram_splitk_applies,
    gram_splitk_chunk_count,
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
    var ms: List[Int] = [1, 8, 8, 8, 33, 32, 64, 128]
    var ks: List[Int] = [7, 33, n_chunks - 1, n_chunks + 1, 257, 100003, 4001, 1025]
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
        " per cell and is bitwise symmetric at 8 shapes (m 1..128 covering"
        " all three CELLS widths; k odd, prime, below/above the "
        + String(n_chunks)
        + "-chunk grid, and never a chunk multiple)"
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
        " split-K and 129x129/768x768/m!=n to the fallback; wrapper verified"
        " per cell on arm '"
        + arm_a
        + "' at 32x32x100003 and arm '"
        + arm_b
        + "' at 768x768x257"
    )
