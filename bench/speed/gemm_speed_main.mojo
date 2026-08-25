"""The FAST path's GEMM arm, over the whole shape table, in FSPEED lines.

    MOJOLEARN_SPEED_ROUNDS=10 pixi run mojo run -I . bench/speed/gemm_speed_main.mojo

WHAT THIS MEASURES AND WHY IT IS NOT bench/gemm_price_main.mojo
===============================================================
`bench/gemm_price_main.mojo` prices the IDENTITY CONTRACT: pinned against
unpinned against strict, inside one binary, to answer "what does the pin
cost". This driver answers a different question and is deliberately much
smaller: what does an ordinary mojolearn user's GEMM cost on an NVIDIA box,
against cuBLAS, when nothing is pinned at all.

**UNDER FAST, `core/gemm.mojo` CALLS MAX's `linalg.matmul`.** That is not a
disclaimer, it is the finding this arm exists to record. The arm a user gets
is Modular's tuned kernel, so this driver's number is a measurement of MAX's
matmul reached through our call, and the honest way to report it is with
that sentence attached. Whether `linalg.matmul` itself dispatches to cuBLAS
is not something this repository knows; `tools/speed_gemm_arm.py` settles the
comparison by timing cuBLAS directly through torch on the same shapes in the
same run.

**AND THE COMPARISON HAS A PRECISION TRAP IN IT.** On an H100 this FAST arm
measured 200 TFLOP/s at `llama8b.mlp_up.t512`, against an FP32 non-tensor
peak of 67 TFLOP/s. Three times over, which is arithmetically impossible in
strict FP32. cuBLAS through torch on the same box measured 44.4 TFLOP/s with
`allow_tf32=False` and 207.5 with it on. MAX matches the TF32 column. So the
fair opponent for THIS arm is `cublas-tf32`, and reading it against
`cublas-fp32` would flatter us by about five times for a precision cut we
also took. Both columns are measured so the reader can see it.

THE SHAPE TABLE IS IMPORTED, NEVER RESPELLED
=============================================
`bench/gemm_shapes.mojo` is the single source of truth for the twenty shapes,
every one of which carries the `file:line` or the published model
configuration it came from. `tools/speed_gemm_arm.py` PARSES that same file
for the cuBLAS arm rather than keeping a second copy, so both sides of the
comparison are provably on the same table.

WHAT THE POISON IS FOR, AND IT IS NOT A CORRECTNESS CHECK
==========================================================
`C` is filled with a poison value before every timed call and the result is
read back afterwards. A kernel that launches, returns immediately and writes
nothing produces a beautiful millisecond, and a timing harness with no such
check cannot tell that from a fast kernel. The read-back happens OUTSIDE the
timed region. Same value and same role as `bench/gemm_price_main.mojo`'s
`DEVICE_POISON`.

THE MODE IS READ FROM THE COMPILE, NOT FROM THE ENVIRONMENT
============================================================
`_mode_name()` reads the comptime constant. This whole run is supposed to be
the FAST path, and a driver that reported FAST because it was invoked without
a flag, while the binary was built IDENTICAL, is the exact failure the
witness exists to prevent. Three mislabelled measurements were caught by that
witness on 2026-08-23.

THE NN ROWS ARE REFUSED BY NAME
================================
`core/gemm.mojo` exports `gemm_nt` and `gemm_tn` and has no NN entry, so the
NN rows of the table cannot be run through the user-facing surface at all.
That is reported as `FSPEED-REFUSED` per row rather than quietly skipped, and
rather than smuggled in through `identical_gemm`, which is a different arm
with a different contract and would put a pinned number in a FAST column.
"""

from std.memory import bitcast
from std.os import getenv
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext

from bench.gemm_shapes import (
    GEMM_SHAPE_COUNT,
    gemm_shape_k,
    gemm_shape_m,
    gemm_shape_n,
    gemm_shape_name,
    gemm_shape_op,
)
from bench.gemm_shapes import OP_NN as TBL_OP_NN
from bench.gemm_shapes import OP_NT as TBL_OP_NT
from bench.gemm_shapes import OP_TN as TBL_OP_TN
from core.gemm import gemm_nt, gemm_tn
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime DEVICE_POISON = Float32(-987654.0)


def _mode_name() -> String:
    """The mode this binary COMPILED in, from the comptime constant."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _bits(v: Float32) -> UInt32:
    return bitcast[DType.uint32](v)


def _exact(i: Int, salt: Int) -> Float32:
    """A FULL-MANTISSA fixture value, assembled from bits.

    Not a scaled integer. `bench/results/SPEED_LANE_2026-08-25.md` records
    what the scaled-integer generator did to the identity half of that
    experiment: every product was exactly representable, so sixteen of twenty
    rows came back bit-identical between two arms that are supposed to
    differ, and the bit half of the run was vacuous while the timing was
    unaffected. Timing is all this driver reports, so the generator could not
    make it wrong here, but the same fixture is used so that a later reader
    comparing this table with that one is comparing the same inputs.
    """
    var z = (
        UInt64(i + 1) * 0x9E3779B97F4A7C15 + UInt64(salt + 1) * 0xBF58476D1CE4E5B9
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    # Exponent pinned into [2^-4, 2^-3) so a long k cannot overflow, mantissa
    # taken whole from the mixer.
    var mant = UInt32(Int(z & 0x7FFFFF))
    var word = UInt32(0x3D800000) | mant
    return bitcast[DType.float32](word)


def _fill(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    n: Int,
    salt: Int,
) raises:
    """Upload `n` fixture floats through a host buffer.

    Through a host buffer rather than a `List` so a `k` in the millions does
    not materialize a second copy on the heap.
    """
    if n < 1:
        return
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr().unsafe_store(i, _exact(i, salt))
    ctx.enqueue_copy(dst_buf=buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h


def _poison(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], mn: Int
) raises:
    if mn < 1:
        return
    var h = ctx.enqueue_create_host_buffer[DType.float32](mn)
    ctx.synchronize()
    for i in range(mn):
        h.unsafe_ptr().unsafe_store(i, DEVICE_POISON)
    ctx.enqueue_copy(dst_buf=buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h


def _digest(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    mn: Int,
    tag: String,
) raises -> UInt64:
    """Read `C` back OUTSIDE the timed region, refuse a surviving poison, and
    return an FNV-1a64 digest of its bits.

    The digest is the `[[reached-but-inert]]` check for a timing harness. It
    is also what makes this a determinism report: under FAST the same call on
    the same inputs may return different bits round to round, and that is
    printed rather than failed, because it is the direct evidence for what
    the IDENTICAL mode buys.
    """
    var h = ctx.enqueue_create_host_buffer[DType.float32](mn if mn > 0 else 1)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var d = UInt64(0xCBF29CE484222325)
    for i in range(mn):
        var v = h.unsafe_ptr().unsafe_load(i)
        if _bits(v) == _bits(DEVICE_POISON):
            raise Error(
                "POISON SURVIVED at cell "
                + String(i)
                + " of "
                + tag
                + ": the kernel never wrote it, so the milliseconds beside"
                " it are a timing of a launch that produced no answer."
            )
        d = (d ^ UInt64(Int(_bits(v)))) * UInt64(0x100000001B3)
    _ = h
    return d


def _nibble(n: Int) -> String:
    """One hex digit. Spelled as a branch rather than as `"0123..."[n]`
    because Mojo strings are UTF-8 and refuse direct positional indexing:
    `s[i]` is a compile error naming three different things it could mean."""
    if n < 10:
        return String(n)
    if n == 10:
        return String("a")
    if n == 11:
        return String("b")
    if n == 12:
        return String("c")
    if n == 13:
        return String("d")
    if n == 14:
        return String("e")
    return String("f")


def _hex16(v: UInt64) -> String:
    var s = String("")
    for i in range(16):
        s += _nibble(Int((v >> UInt64(60 - 4 * i)) & UInt64(0xF)))
    return s


def _env_int(name: String, dflt: Int) -> Int:
    var s = String(getenv(name))
    # `len(String)` is a compile error in this language: UTF-8 makes a single
    # length ambiguous. Bytes is the right question for "did the operator set
    # this variable".
    if s.byte_length() == 0:
        return dflt
    try:
        return Int(s)
    except:
        return dflt


def main() raises:
    var rounds = _env_int("MOJOLEARN_SPEED_ROUNDS", 10)
    var smoke = String(getenv("MOJOLEARN_SPEED_SIZE")) == "smoke"
    # The cap exists so one enormous row cannot eat a rented hour. It is
    # ANNOUNCED per skipped row, never silent: a table that dropped its
    # largest shapes without saying so reads as full coverage of the table.
    var max_macs = Float64(_env_int("MOJOLEARN_SPEED_MAX_GMACS", 0)) * 1.0e9

    var ctx = DeviceContext()
    print(
        "FSPEED-HEADER family=gemm lane=gemm arm=ours mode="
        + _mode_name()
        + " device="
        + String(ctx.name())
        + " rounds="
        + String(rounds)
        + " size="
        + (String("smoke") if smoke else String("shipped"))
    )
    print(
        "FSPEED-NOTE lane=gemm arm=ours under FAST core/gemm.mojo calls MAX"
        " linalg.matmul, so this arm is Modular's tuned kernel reached"
        " through our surface. Its fair opponent is cublas-tf32, not"
        " cublas-fp32. See this file's docstring."
    )

    for i in range(GEMM_SHAPE_COUNT):
        var name = gemm_shape_name(i)
        var op = gemm_shape_op(i)
        var m = gemm_shape_m(i)
        var n = gemm_shape_n(i)
        var k = gemm_shape_k(i)
        if smoke and (Float64(m) * Float64(n) * Float64(k) > 1.0e8):
            print(
                "FSPEED-NOTE lane=gemm arm=ours shape="
                + name
                + " SKIPPED under size=smoke"
            )
            continue
        if max_macs > 0.0 and Float64(m) * Float64(n) * Float64(k) > max_macs:
            print(
                "FSPEED-NOTE lane=gemm arm=ours shape="
                + name
                + " SKIPPED above MOJOLEARN_SPEED_MAX_GMACS"
            )
            continue
        if op == TBL_OP_NN:
            print(
                "FSPEED-REFUSED lane=gemm arm=ours reason="
                + name
                + " is an NN row and core/gemm.mojo exports no NN entry;"
                " routing it through identical_gemm would put a pinned"
                " number in a FAST column"
            )
            continue

        # THE TWO ORIENTATIONS DO NOT TAKE THE SAME OPERANDS, and that is a
        # property of our surface rather than of the table.
        #
        #   NT  `gemm_nt(ctx, z, x, y, m, n, k)`      x[m x k] . y[n x k]^T
        #       two independent operands, the general case.
        #   TN  `gemm_tn(ctx, z, x, xt, xt2, m, n, k)`  x[k x m]^T . x[k x n]
        #       **ONE operand used twice.** It is the GRAM entry, not a
        #       general transposed GEMM, and it takes two scratch buffers of
        #       `k * m` floats because it reaches MAX's matmul by
        #       transposing X on the device first. Every TN row in the table
        #       is a Gram shape with m == n, so this covers all four of them;
        #       a TN row with m != n would have no entry here and would be
        #       refused, which is checked below rather than assumed.
        #
        # The cuBLAS arm builds two independent operands for a TN row. The
        # FLOP count is identical either way, so the comparison stands, and
        # this note is here so nobody reads the two files side by side and
        # concludes one of them has the wrong fixture.
        if op == TBL_OP_TN and m != n:
            print(
                "FSPEED-REFUSED lane=gemm arm=ours reason="
                + name
                + " is a TN row with m != n and core/gemm.mojo's TN entry is"
                " the Gram case x^T . x, which cannot express it"
            )
            continue
        var na = m * k if op == TBL_OP_NT else k * m
        var nb = n * k if op == TBL_OP_NT else k * m
        var nscratch = 1 if op == TBL_OP_NT else k * m
        var mn = m * n
        var da = ctx.enqueue_create_buffer[DType.float32](na)
        var db = ctx.enqueue_create_buffer[DType.float32](nb)
        var dx2 = ctx.enqueue_create_buffer[DType.float32](nscratch)
        var dc = ctx.enqueue_create_buffer[DType.float32](mn)
        ctx.synchronize()
        _fill(ctx, da, na, 11 + i)
        if op == TBL_OP_NT:
            _fill(ctx, db, nb, 977 + i)

        # THE WARM-UP IS TIMED AND PRINTED AND NEVER AVERAGED IN. On a cold
        # context the first call pays for kernel selection; a reader who
        # cannot see that number cannot tell it from a cost.
        _poison(ctx, dc, mn)
        var t0 = perf_counter_ns()
        if op == TBL_OP_NT:
            gemm_nt(ctx, dc, da, db, m, n, k)
        else:
            gemm_tn(ctx, dc, da, db, dx2, m, n, k)
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var warm = _digest(ctx, dc, mn, name + " warmup")
        print(
            "FSPEED-WARMUP lane=gemm arm=ours shape="
            + name
            + " ms="
            + String(Float64(t1 - t0) / 1.0e6)
        )

        var first = warm
        var moved = False
        for r in range(1, rounds + 1):
            _poison(ctx, dc, mn)
            var s0 = perf_counter_ns()
            if op == TBL_OP_NT:
                gemm_nt(ctx, dc, da, db, m, n, k)
            else:
                gemm_tn(ctx, dc, da, db, dx2, m, n, k)
            ctx.synchronize()
            var s1 = perf_counter_ns()
            var d = _digest(ctx, dc, mn, name + " round " + String(r))
            if r == 1:
                first = d
            elif d != first and not moved:
                moved = True
                print(
                    "FSPEED-NOTE lane=gemm arm=ours hash moved across rounds:"
                    + " "
                    + _hex16(first)
                    + " "
                    + _hex16(d)
                )
            print(
                "FSPEED lane=gemm arm=ours shape="
                + name
                + " round="
                + String(r)
                + " ms="
                + String(Float64(s1 - s0) / 1.0e6)
                + " hash="
                + _hex16(d)
            )
        # The buffers are used past the last synchronize on purpose: a
        # DeviceBuffer is freed at its LAST USE in this language, and a
        # buffer freed before the copy that reads it is the trap this
        # repository has already been bitten by.
        _ = da
        _ = db
        _ = dx2
        _ = dc
