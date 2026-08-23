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

THE VENDOR ARM'S TOLERANCE IS PER COLUMN, AND THAT IS DEVIATION 540
(2026-08-23). On the H100 leg of E2 this file's `check_gram_vendor_arm`
FAILED under FAST at 33x33x257, cell (0,0) device 88.9696044921875 against
host 88.96752322962544 -- 2.4e-5 of the magnitude, 2.4x the fp32 budget --
while `check_gram_splitk_oracle` passed every cell at ten shapes in the
same process. The split-K kernel is ours and is fp32; the vendor arm is
MAX 26.5.0's `linalg.matmul`, and on NVIDIA that is a TF32 tensor-core
product BY DEFAULT with NO compilable opt-out before Blackwell
(`mojo_only/kernel_matrix.mojo::column_vendor_fp32_matmul_is_tf32`, with
the MAX source lines). So (a) forcing fp32 is NOT AVAILABLE on the H100,
and (b) is what ships: on a lossy column the vendor arm is held to
`VENDOR_TF32_PRODUCT_REL_BOUND` (1e-3 of the magnitude) plus the fp32
budget, the tight budget stays on every exact column, and the printed line
names the precision class it judged against. (c) IDENTICAL is unaffected:
`gemm_tn_via_transpose` reaches `gemm_nt`, which under IDENTICAL is
`pinned_gemm_nt_kernel` and never `linalg.matmul` (DEVIATION 526), so the
"vendor arm" check under IDENTICAL judges a pinned fp32 kernel and keeps
the fp32 budget on every column. The bound is read from the kernel matrix
rather than written here because `cluster/mojo_only/kmeans_check.mojo`'s
unfused assignment arm is the SAME product (`gemm_nt`) and failed on the
same leg for the same reason (DEVIATION 529); two checks, one row.

AND THE FP32 BUDGET ITSELF WAS APPLE-SHAPED. `check_gram_dispatch` runs
the wrapper at 32x32x100003, which on the Apple column is split-K and on
every other column is the vendor matmul; the `-D MOJOLEARN_COLUMN_AMD`
build on the M4 forces that route and Apple's own fp32 `gemm_kernel_apple_8x8`
then lands 1.3e-5 of the magnitude at cell (0,0) -- over the flat 1e-5,
with no tensor core involved, because a serial fp32 fold at k=1e5 is
simply wider than one calibrated at k=10007. So a CLOSED product (order
unknown) now earns `VENDOR_SERIAL_FOLD_REL_PER_SQRT_K x sqrt(k)` on top of
the flat budget on EVERY column (3.8e-6 at k=257, 7.5e-5 at k=100003),
and the split-K kernel, whose order is ours and two-level, keeps the flat
budget at every k. A real MI300X under FAST would have failed at exactly
that line; the next AMD leg is what says so.

Bitwise symmetry on the vendor arm is a REPORT on a lossy column: a
closed tensor-core kernel promises nothing about it, and the next NVIDIA
leg is what says whether cuBLAS's TF32 product is symmetric. On an exact
column it stays an assertion, which is the measured Apple M4 behaviour.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from std.math import sqrt

from core.gemm import gemm_tn, gemm_tn_via_transpose
from core.column_stats import (
    STATS_TPB,
    column_mean_kernel,
    shift_columns_kernel,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from mojo_only.kernel_matrix import (
    TARGET_COLUMN,
    VENDOR_TF32_PRODUCT_REL_BOUND,
    column_name,
    vendor_fp32_matmul_is_lossy,
    vendor_fp32_matmul_precision_name,
)
from mojo_only.hardware_matrix import gram_splitk_is_target_arm
from core.gram_splitk import (
    GRAM_SPLITK_RESOLVED_COLUMN,
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

#: The fp32 budget every arm is held to on every column: mag-relative, the
#: `gemm_tn` vendor-table row's number (module docstring, TOLERANCE).
comptime GRAM_FP32_REL_BUDGET = Float64(1.0e-5)
comptime GRAM_ABS_BUDGET = Float64(1.0e-6)


def _arm_is_vendor_product(m: Int, k: Int, arm: Int) raises -> Bool:
    """Whether this call reaches `linalg.matmul` under FAST: the transpose
    arm always does, the wrapper does iff the predicate declines split-K.
    Under IDENTICAL nothing does (DEVIATIONS 521/526), so this is False."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return False
    if arm == ARM_TRANSPOSE:
        return True
    if arm == ARM_WRAPPER:
        return not gram_splitk_applies(m, m, k)
    return False


#: The fp32 allowance a CLOSED product earns for its unknown summation
#: order, per sqrt(k): `4 x 2^-24`. A serial fp32 fold's rounding error
#: random-walks like `u x sqrt(k)` of the magnitude (u = 2^-24); the factor
#: four puts the budget several standard deviations out. MEASURED, NOT
#: DESIGNED: the `-D MOJOLEARN_COLUMN_AMD` build on the M4 routes the
#: 32x32x100003 wrapper call to the vendor arm (the AMD column declines
#: split-K everywhere), and Apple's own `gemm_kernel_apple_8x8` then lands
#: cell (0,0) at 33290.25390625 against host 33290.69632720772 -- 1.3e-5 of
#: the magnitude, OVER the flat 1e-5 budget, with fp32 arithmetic and no
#: tensor core in sight. The flat budget was calibrated at k=10007 (4.09e-6
#: measured) and had simply never met a serial fp32 fold at k=1e5; a real
#: MI300X under FAST would have met one at exactly this line. At k=257 the
#: allowance is 3.8e-6, at 10007 2.4e-5, at 100003 7.5e-5. The split-K
#: kernel (ours, two-level, known order) keeps the flat budget: it passed
#: 1e-5 at k=100003 and k=4M.
comptime VENDOR_SERIAL_FOLD_REL_PER_SQRT_K = Float64(4.0) / Float64(16777216.0)


def _rel_budget_for(
    ctx: DeviceContext, m: Int, k: Int, arm: Int
) raises -> Float64:
    """DEVIATION 540: the fp32 budget; on a vendor product widened by the
    serial-fold allowance for its unknown order, and by the TF32 bound
    only when this build's column (plus the device generation, for Apple)
    runs that product lossy."""
    var budget = GRAM_FP32_REL_BUDGET
    if _arm_is_vendor_product(m, k, arm):
        budget += VENDOR_SERIAL_FOLD_REL_PER_SQRT_K * sqrt(Float64(k))
        if vendor_fp32_matmul_is_lossy(TARGET_COLUMN, ctx.compute_capability()):
            budget += VENDOR_TF32_PRODUCT_REL_BOUND
    return budget


def _precision_label(
    ctx: DeviceContext, m: Int, k: Int, arm: Int
) raises -> String:
    """What the check prints: the arm's precision class on this build."""
    if _arm_is_vendor_product(m, k, arm):
        return (
            "vendor matmul, "
            + vendor_fp32_matmul_precision_name(
                TARGET_COLUMN, ctx.compute_capability()
            )
            + " on column "
            + column_name(TARGET_COLUMN)
        )
    return String("fp32 (pinned or split-K kernel)")


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

    var rel_budget = _rel_budget_for(ctx, m, k, arm)
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
            if abs(got - acc) > mag * rel_budget + GRAM_ABS_BUDGET:
                return (
                    "cell ("
                    + String(i)
                    + ", "
                    + String(j)
                    + ") device "
                    + String(got)
                    + " host "
                    + String(acc)
                    + " (|diff| "
                    + String(abs(got - acc))
                    + " > mag "
                    + String(mag)
                    + " x "
                    + String(rel_budget)
                    + " + "
                    + String(GRAM_ABS_BUDGET)
                    + "; arm judged as "
                    + _precision_label(ctx, m, k, arm)
                    + ")"
                )

    # Bitwise symmetry, no tolerance -- the same tripwire as
    # check_covariance_is_symmetric, at every shape this file runs. On a
    # LOSSY vendor product it is a report (module docstring, DEVIATION 540).
    var asym = 0
    var asym_first = String("")
    for i in range(m):
        for j in range(i + 1, m):
            var a = hz.unsafe_ptr().unsafe_load(i * m + j)
            var b = hz.unsafe_ptr().unsafe_load(j * m + i)
            if a != b:
                asym += 1
                if asym_first == "":
                    asym_first = (
                        "cells ("
                        + String(i)
                        + ", "
                        + String(j)
                        + ") and mirror are not bitwise equal: "
                        + String(a)
                        + " vs "
                        + String(b)
                    )
    if asym != 0:
        if _arm_is_vendor_product(m, k, arm) and vendor_fp32_matmul_is_lossy(
            TARGET_COLUMN, ctx.compute_capability()
        ):
            print(
                "  REPORT: the lossy vendor product at "
                + String(m)
                + "x"
                + String(m)
                + "x"
                + String(k)
                + " is not bitwise symmetric in "
                + String(asym)
                + " cell pairs ("
                + asym_first
                + "); a closed tensor-core kernel promises no symmetry and"
                " this is the measurement of whether it has it"
            )
        else:
            return asym_first
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
    arms are judged against the same oracle at the same numbers.

    THE BUDGET IS THE COLUMN'S (DEVIATION 540, module docstring): fp32 on
    an exact column, fp32 + the TF32 bound where MAX's matmul is a
    tensor-core product, and the line printed says which. Under IDENTICAL
    this arm is the pinned kernel and the fp32 budget applies everywhere.
    """
    var ctx = DeviceContext()
    var err = _gram_one(ctx, 33, 257, ARM_TRANSPOSE)
    if err != "":
        raise Error(
            "check_gram_vendor_arm FAILED at m=33 k=257: " + err
        )
    var budget = _rel_budget_for(ctx, 33, 257, ARM_TRANSPOSE)
    var lossy = _arm_is_vendor_product(
        33, 257, ARM_TRANSPOSE
    ) and vendor_fp32_matmul_is_lossy(TARGET_COLUMN, ctx.compute_capability())
    print(
        "check_gram_vendor_arm OK: transpose+matmul arm matches the Float64"
        " oracle per cell at 33x33x257 within mag x "
        + String(budget)
        + " ("
        + _precision_label(ctx, 33, 257, ARM_TRANSPOSE)
        + "; "
        + (
            "the TF32 bound, DEVIATION 540 -- FAST products on this column"
            " are TF32-accuracy"
            if lossy
            else (
                "the fp32 budget plus the closed product's serial-fold"
                " allowance"
                if _arm_is_vendor_product(33, 257, ARM_TRANSPOSE)
                else "the fp32 budget; IDENTICAL, pinned kernel"
            )
        )
        + ")"
        + (
            "; bitwise symmetry reported, not asserted"
            if lossy
            else "; bitwise symmetry asserted"
        )
    )


def _round_to_tf32(x: Float32) -> Float32:
    """Host model of the tensor core's operand conversion: round the fp32
    mantissa to 10 explicit bits, nearest-even (`cvt.rna.tf32.f32` rounds
    ties away; the difference is below what this check measures)."""
    var bits = bitcast[DType.uint32](x)
    var lsb = (bits >> 13) & 1
    bits = (bits + 0xFFF + lsb) & ~UInt32(0x1FFF)
    return bitcast[DType.float32](bits)


def check_gram_tf32_bound_separates() raises:
    """THE GATE DEVIATION 540 STANDS ON, and it runs on EVERY column,
    Apple included, because it needs no tensor core: it MAKES a TF32-class
    product out of the fp32 split-K kernel by rounding the operands to ten
    mantissa bits on the host before upload, judges it against the
    UNROUNDED Float64 oracle, and asserts two things at the failing H100
    shape (33x33x257):

    1. the fp32 budget (`GRAM_FP32_REL_BUDGET`, what every column was held
       to before this deviation) REJECTS it -- so the budget the H100 leg
       failed on has teeth against exactly this error class, on this box;
    2. the TF32 bound (`GRAM_FP32_REL_BUDGET + VENDOR_TF32_PRODUCT_REL_BOUND`)
       ADMITS it -- so the loosened NVIDIA tolerance is not a blanket.

    If either half fails, the two bounds do not separate a TF32 product
    from an fp32 one and the per-column tolerance is decoration. The
    split-K kernel is used for the product (not the vendor arm) so that
    the arithmetic between upload and readback is known fp32 on every
    column and the ONLY lossy step is the one planted.
    """
    var ctx = DeviceContext()
    var m = 33
    var k = 257
    var salt = m * 1000003 + k + 77
    var x = ctx.enqueue_create_buffer[DType.float32](k * m)
    var z = ctx.enqueue_create_buffer[DType.float32](m * m)
    var hx_exact = ctx.enqueue_create_host_buffer[DType.float32](k * m)
    var hx_tf32 = ctx.enqueue_create_host_buffer[DType.float32](k * m)
    var hz = ctx.enqueue_create_host_buffer[DType.float32](m * m)
    ctx.synchronize()
    var moved = 0
    for i in range(k * m):
        var v = _val_f32(i, salt)
        hx_exact.unsafe_ptr().unsafe_store(i, v)
        var t = _round_to_tf32(v)
        if t != v:
            moved += 1
        hx_tf32.unsafe_ptr().unsafe_store(i, t)
    if moved < (k * m) // 2:
        raise Error(
            "check_gram_tf32_bound_separates: the TF32 rounding moved only "
            + String(moved)
            + " of "
            + String(k * m)
            + " operands; the fixture is not exercising the truncation"
        )
    for i in range(m * m):
        hz.unsafe_ptr().unsafe_store(i, Float32(-987654.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx_tf32.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=z, src_ptr=hz.unsafe_ptr())
    ctx.synchronize()
    gemm_tn_splitk(ctx, z, x, m, k)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hz.unsafe_ptr(), src_buf=z)
    ctx.synchronize()

    var worst = Float64(0.0)
    var worst_cell = String("")
    for i in range(m):
        for j in range(m):
            var acc = Float64(0.0)
            var mag = Float64(0.0)
            for t in range(k):
                var av = Float64(hx_exact.unsafe_ptr().unsafe_load(t * m + i))
                var bv = Float64(hx_exact.unsafe_ptr().unsafe_load(t * m + j))
                acc += av * bv
                mag += abs(av * bv)
            var got = Float64(hz.unsafe_ptr().unsafe_load(i * m + j))
            if got == Float64(-987654.0):
                raise Error(
                    "check_gram_tf32_bound_separates: cell was never written"
                )
            var rel = (abs(got - acc) - GRAM_ABS_BUDGET) / mag
            if rel > worst:
                worst = rel
                worst_cell = "(" + String(i) + ", " + String(j) + ")"
    # The fp32 budget a VENDOR product at this k is held to on an exact
    # column (flat budget plus the serial-fold allowance), and the TF32
    # bound on a lossy one: the two the per-column split actually uses.
    var fp32_vendor_budget = (
        GRAM_FP32_REL_BUDGET
        + VENDOR_SERIAL_FOLD_REL_PER_SQRT_K * sqrt(Float64(k))
    )
    var tf32_bound = fp32_vendor_budget + VENDOR_TF32_PRODUCT_REL_BOUND
    if worst <= fp32_vendor_budget:
        raise Error(
            "check_gram_tf32_bound_separates: a TF32-rounded product passed"
            " the fp32 budget (worst mag-relative "
            + String(worst)
            + " at cell "
            + worst_cell
            + " <= "
            + String(fp32_vendor_budget)
            + "), so the budget the H100 leg failed on cannot see the"
            " error class it failed on and DEVIATION 540's split is"
            " untested"
        )
    if worst > tf32_bound:
        raise Error(
            "check_gram_tf32_bound_separates: a TF32-rounded product"
            " EXCEEDS the TF32 bound (worst mag-relative "
            + String(worst)
            + " at cell "
            + worst_cell
            + " > "
            + String(tf32_bound)
            + "); VENDOR_TF32_PRODUCT_REL_BOUND is too tight for what it"
            " names"
        )
    print(
        "check_gram_tf32_bound_separates OK: operands rounded to 10"
        " mantissa bits ("
        + String(moved)
        + " of "
        + String(k * m)
        + " moved) through the fp32 split-K kernel at 33x33x257 land"
        " worst mag-relative "
        + String(worst)
        + " at cell "
        + worst_cell
        + " from the exact oracle: REJECTED by the fp32 budget "
        + String(fp32_vendor_budget)
        + ", ADMITTED by the TF32 bound "
        + String(tf32_bound)
        + " -- the two per-column tolerances of DEVIATION 540 separate a"
        " TF32-class product from an fp32 one on this box (the H100 leg"
        " measured 2.4e-5 on the real vendor arm at this shape)"
    )


def check_gram_dispatch() raises:
    """The predicate decides what it was derived to decide, and the REAL
    `gemm_tn` wrapper is run once per arm, printing which arm it took.

    THE PREDICATE'S ANSWERS ARE THE RESOLVED COLUMN'S, NOT APPLE'S
    (DEVIATION 540's second finding, 2026-08-23). The first six assertions
    below used to be unconditional, and they are the Apple column's
    answers: on NVIDIA and AMD under FAST `gram_splitk_is_target_arm` is
    False and `gram_splitk_applies` answers False AT EVERY SHAPE (the
    vendor matmul owns the Gram there, `core/gram_splitk.mojo`), so this
    check would have been the NEXT failure on the H100 leg, one line after
    the vendor-arm tolerance it died on. Now the split-K expectations are
    asserted where the resolved column takes that arm (Apple under FAST,
    every column under IDENTICAL) and the vendor-arm expectations -- the
    predicate declines every shape -- where it does not, and the printed
    line names which set was asserted.
    """
    comptime splitk_is_arm = gram_splitk_is_target_arm[
        GRAM_SPLITK_RESOLVED_COLUMN
    ]()
    comptime if splitk_is_arm:
        # Tile-starved outputs (fewer vendor tiles than resident-block
        # slots) go split-K; everything else stays on the vendor matmul.
        if not gram_splitk_applies(32, 32, 4000000):
            raise Error("dispatch: 32x32x4M must take split-K and does not")
        if not gram_splitk_applies(1, 1, 7):
            raise Error("dispatch: 1x1x7 must take split-K and does not")
        if not gram_splitk_applies(128, 128, 1025):
            raise Error("dispatch: 128x128 must take split-K and does not")
        if gram_splitk_applies(129, 129, 1025):
            raise Error(
                "dispatch: 129 columns exceeds the staging tile and must"
                " fall back, and does not"
            )
        if gram_splitk_applies(768, 768, 257):
            raise Error(
                "dispatch: a 768x768 output exceeds the staging tile and"
                " must not take split-K, and does. Under FAST it also has"
                " 144 vendor tiles >= the block slots, so the starvation"
                " test declines it as well; under IDENTICAL the starvation"
                " test is not consulted (DEVIATION 521) and the capacity"
                " bound is what declines it."
            )
        if gram_splitk_applies(8, 4, 100):
            raise Error("dispatch: m != n is not a Gram and must fall back")
    else:
        # A non-Apple column under FAST: the vendor matmul is the arm at
        # EVERY shape, including the tile-starved ones, and the predicate
        # must say so consistently -- a True anywhere here would launch a
        # kernel this column has declared is not its arm.
        if (
            gram_splitk_applies(32, 32, 4000000)
            or gram_splitk_applies(1, 1, 7)
            or gram_splitk_applies(128, 128, 1025)
            or gram_splitk_applies(129, 129, 1025)
            or gram_splitk_applies(768, 768, 257)
            or gram_splitk_applies(8, 4, 100)
        ):
            raise Error(
                "dispatch: column "
                + column_name(GRAM_SPLITK_RESOLVED_COLUMN)
                + " declares the vendor matmul as its Gram arm"
                " (gram_splitk_is_target_arm False) and the predicate"
                " answered True at some shape anyway"
            )

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
    # THE FALLBACK ARM IS A REFUSAL UNDER IDENTICAL, NOT AN ARM
    # (DEVIATION 521, IDENTITY_PATHS row 27). `gemm_tn`'s other arm is
    # `linalg.matmul`, a closed vendor library whose k-split is a per-vendor
    # summation order, so an identity build must not run it: it raises by
    # name instead. Running the wrapper here and asserting per-cell
    # correctness is therefore a FAST-arm assertion, and under IDENTICAL the
    # assertion is that it refuses -- which is a stronger statement about
    # the same call, not a skipped check.
    var arm_b = String("split-K") if gram_splitk_applies(
        768, 768, 257
    ) else String("transpose+matmul (FAST) / REFUSED (IDENTICAL)")
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var refused = False
        try:
            _ = _gram_one(ctx, 768, 257, ARM_WRAPPER)
        except e:
            if String(e).find("IDENTITY_PATHS row 27") >= 0:
                refused = True
            else:
                raise Error(
                    "check_gram_dispatch: 768x768x257 raised under"
                    " IDENTICAL but not with the row-27 refusal: "
                    + String(e)
                )
        if not refused:
            raise Error(
                "check_gram_dispatch: 768x768x257 COMPLETED under"
                " IDENTICAL. It is past the split-K kernel's capacity, so"
                " it ran on linalg.matmul and returned a Gram product this"
                " mode promises is vendor-independent and is not."
            )
    else:
        var err2 = _gram_one(ctx, 768, 257, ARM_WRAPPER)
        if err2 != "":
            raise Error(
                "check_gram_dispatch FAILED through the wrapper (arm "
                + arm_b
                + ") at 768x768x257: "
                + err2
            )
    var routing = String(
        "predicate routes 32x32x4M/1x1x7/128x128 to split-K and"
        " 129x129/768x768/m!=n to the fallback"
    ) if splitk_is_arm else String(
        "predicate declines split-K at every shape (column "
        + column_name(GRAM_SPLITK_RESOLVED_COLUMN)
        + "'s Gram arm is the vendor matmul under FAST)"
    )
    print(
        "check_gram_dispatch OK: "
        + routing
        + "; staging copy"
        " vectorizes m=32/4/128 and falls back scalar at m=33/1; register"
        " tile owns m=32/64/128 (2x2/4x4/8x8) and declines m=1/3/33 to the"
        " strided arm; wrapper"
        " verified per cell on arm '"
        + arm_a
        + "' at 32x32x100003 ("
        + _precision_label(ctx, 32, 100003, ARM_WRAPPER)
        + ") and arm '"
        + arm_b
        + "' at 768x768x257 ("
        + _precision_label(ctx, 768, 257, ARM_WRAPPER)
        + ")"
    )


def main() raises:
    # STANDALONE DRIVER. The four calls `decomposition/pca_main.mojo`
    # makes, in its order. Each builds its own `DeviceContext`.
    check_gram_splitk_oracle()
    check_gram_vendor_arm()
    check_gram_tf32_bound_separates()
    check_gram_dispatch()
    check_gram_centered_fused()
