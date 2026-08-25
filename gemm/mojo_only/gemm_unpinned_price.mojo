"""THE SINGLE-VARIABLE PRICE: the pinned kernel against the unpinned one.

    tools/with_identical_mode.sh pixi run mojo run -I . \\
        -D MOJOLEARN_GEMM_UNPINNED_ARM=1 \\
        gemm/mojo_only/gemm_unpinned_price.mojo

WHAT THIS ANSWERS AND WHY NOTHING ELSE IN THE REPOSITORY DOES
=============================================================
`bench/gemm_price_main.mojo` DEVIATION 1092 says it plainly: no arm in that
file isolates the cost of `mojolearn.identical.gemm.fp32.v1`.

    device vs pinned    is tiling against no tiling. `pinned_gemm_nt_kernel`
                        is FLAT -- one thread per output cell, serial whole-k
                        loop, no shared memory -- and our identical kernel
                        BEATS it by 1.6x to 1.8x at every t512 row.
    device vs vendor    is our kernel engineering plus the pin, against
                        Modular's kernel engineering, both moving at once.
    device vs cuBLAS    is the same against a mature vendor library, and on
                        an H100 with a PRECISION difference on top (MAX and
                        cuBLAS both reach TF32 there, 10 mantissa bits).

This file times `identical_gemm_into` against `unpinned_gemm_into`, which is
the SAME plan selection, the SAME tile constants, the SAME grid, the SAME
staging loop and the SAME leaf schedule, with ONLY the fold-order pin removed.
`gemm/UNPINNED_CONTROL.md` carries the clause-by-clause table of what was
dropped and the confounds that survive.

ONE BINARY, ALTERNATING CALL BY CALL. This is the whole reason this file
exists rather than a second mode of the price shell. `GLOBAL_NUMERIC_MODE` is
comptime, so IDENTICAL-against-FAST is two BINARIES and the shell can only
alternate them ROUND by round -- which on a box whose governor drifts 1.7x in
twenty minutes leaves a band, not a number (DEVIATION 1094 measured a 1.8x
spread between two single-round H100 runs an hour apart). Pinned and unpinned
are two FUNCTIONS in one binary, so they alternate inside one timed loop and
the drift cancels at the arm level instead of being averaged over.

THREE ARMS, NOT TWO
===================
    pinned      `identical_gemm_into`, the profile.
    unpinned    `unpinned_gemm_into`, NACC = 4 independent accumulator lanes.
    strict      `unpinned_gemm_into_one_acc`, NACC = 1.

The third is not optional. `unpinned` drops the seams AND the
no-sub-partition clause at once, so on its own it cannot say which of the two
was paid for. `strict` holds the dependency chain exactly as long as the
pinned kernel's and removes only the flush, the contraction pin and the fold.
UNPINNED_CONTROL.md's prediction 2 says pinned-to-strict and strict-to-unpinned
will be within 10% of each other on a GPU, and names that as the prediction
most likely to be wrong.

THE PREDICTION IS ON RECORD BEFORE THE RUN (UNPINNED_CONTROL.md):
    1.10x to 1.45x at the tiled t512 rows
    1.00x to 1.06x at the bandwidth-bound t1 rows
    1.3x to 1.8x at gram.32x32x1M
    the fold itself under 2% anywhere
A prediction that turns out wrong is the most useful thing this file can
produce, which is why it is written down here as well as there.

THE BIT CHECK IS A GATE ON THE EXPERIMENT, NOT ON THE KERNELS
=============================================================
Every shape compares the two arms' output digests. **They are EXPECTED to
differ**, and a shape where they do NOT differ is reported as INERT rather
than as agreement, because an unpinned arm that produces the pinned bits at
that shape did not remove anything there and its timing is not a price. The
`P == 1` rows are the known-inert set: with one partition there is no tree to
remove, so UNPINNED_CONTROL.md predicts bit equality there. Those rows are the
control for the fold's own contribution.

PLAN EQUALITY IS ASSERTED, NOT ASSUMED. If the two arms ever choose different
plans at a shape, the run is not the experiment and this file REFUSES that row
rather than printing a ratio between two different kernels.

NEVER COMPILED, NEVER EXECUTED at the time of writing. DEVIATIONS 1137-1145.
"""

from std.math import sqrt
from std.memory import bitcast
from std.os import getenv
from std.sys.compile import is_defined
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
from gemm.mojo_only.gemm_identical import (
    choose_gemm_plan,
    gemm_plan_name,
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.mojo_only.gemm_oracle import (
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_count,
    contract_leaf_size,
)
from gemm.mojo_only.gemm_unpinned import (
    unpinned_gemm_banner,
    unpinned_gemm_into,
    unpinned_gemm_into_one_acc,
    unpinned_gemm_plan_name,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


#: Timed calls per arm per shape, after an untimed warm-up. Three arms
#: alternate inside the loop, so a round is `REPEATS` interleaved triples.
comptime REPEATS = 3

#: `m` and `n` cap so `m * n * k` stays under this. Default is the full
#: llama8b set except `lm_head.t512`, which is the same cap the vendor arm
#: uses so the two files' rows line up.
comptime DEFAULT_MAC_BUDGET = 50_000_000_000

#: Written into `C` before every arm, and read back after. A kernel that
#: launches without writing its output cannot post the best time.
comptime DEVICE_POISON = Float32(-1.0e37)


def _mode_name() -> String:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _oracle_op(tbl: Int) -> Int:
    """`bench/gemm_shapes.mojo` numbers the orientations `(NT, TN, NN) =
    (0, 1, 2)`; `gemm_oracle.mojo` numbers them `(NN, NT, TN) = (0, 1, 2)`.

    THEY ARE A FULL PERMUTATION OF EACH OTHER AND PASSING ONE RAW INTO THE
    OTHER IS SILENT. Every one of the twenty rows would address a DIFFERENT
    operand layout, produce an in-bounds plausible product, and turn in a
    perfectly good millisecond. `bench/gemm_price_main.mojo` carries a gate
    named `check_op_encodings_are_not_interchangeable` for exactly this, and
    this file converts by NAME rather than by arithmetic so the mapping is
    readable rather than derived.
    """
    if tbl == TBL_OP_NT:
        return OP_NT
    if tbl == TBL_OP_TN:
        return OP_TN
    return OP_NN


def _a_elems(op: Int, m: Int, n: Int, k: Int) -> Int:
    """LEFT operand floats under the oracle's addressing. TN reads `A` as
    `k x m`; NN and NT read it as `m x k`."""
    if op == OP_TN:
        return k * m
    return m * k


def _b_elems(op: Int, m: Int, n: Int, k: Int) -> Int:
    """RIGHT operand floats. NT reads `B` as `n x k`; NN and TN read it as
    `k x n`."""
    if op == OP_NT:
        return n * k
    return k * n


def _mix(i: Int, salt: Int) -> UInt64:
    """splitmix64, Steele/Lea/Flood's finalizer, verbatim.

    **`+` HERE IS A REAL ADD AND MUST STAY ONE.** DEVIATION 1146. This
    function was first written with `&+`, which in Mojo computes `x & k` with
    NO COMPILE ERROR (`[[mojo-amp-plus-is-bitwise-and]]`), and the identical
    mistake had already produced wrong hashes in this repository once.
    Caught here by reading `transformer/mojo_only/transformer_fixture.mojo
    ::fixture_splitmix64`, which carries the same warning, rather than by any
    check -- a fixture generator that is merely a DIFFERENT deterministic
    function still fills both arms with the same values, so nothing
    downstream in this file could have noticed. Plain `+` on UInt64 wraps,
    which is what splitmix64 wants. Do not "fix" one into a `&+`."""
    var z = UInt64(i) + UInt64(salt) + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def _exact(i: Int, salt: Int) -> Float32:
    """A small exactly-representable value, ASSEMBLED FROM BITS.

    Never parsed from a decimal string, because `String(float)` does not
    round-trip in this toolchain. The value is a signed integer in
    `[-128, 127]` scaled by `2^-4`, which is exact in FP32 and keeps products
    and their sums well inside the normal range at every `k` here, so the
    arms are separated by their ORDER and not by an overflow.
    """
    var h = _mix(i, salt)
    var v = Int(h & UInt64(0xFF)) - 128
    return Float32(v) * Float32(0.0625)


def _dev_fill(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int, salt: Int
) raises:
    """Upload `n` bit-assembled fixture floats through a host buffer, so a
    `k` in the millions does not materialize a second copy on the heap."""
    if n < 1:
        return
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr().unsafe_store(i, _exact(i, salt))
    ctx.enqueue_copy(dst_buf=buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h


def _dev_poison(
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


def _dev_digest(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], mn: Int
) raises -> Tuple[UInt64, Int]:
    """FNV-1a64 over the output's BITS, plus the count of cells still holding
    the poison.

    The second half is the load-bearing one. `[[reached-but-inert]]`: a
    kernel that never wrote its output would otherwise turn in the best time
    in the table and a digest that is merely 'different'.
    """
    var h = ctx.enqueue_create_host_buffer[DType.float32](mn if mn > 0 else 1)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    var acc = UInt64(0xCBF29CE484222325)
    var left = 0
    for i in range(mn):
        var v = h.unsafe_ptr().unsafe_load(i)
        if _bits(v) == _bits(DEVICE_POISON):
            left += 1
        var b = UInt64(_bits(v))
        for s in range(4):
            acc = (acc ^ ((b >> UInt64(8 * s)) & UInt64(0xFF))) & UInt64(
                0xFFFFFFFFFFFFFFFF
            )
            acc = (acc * UInt64(0x100000001B3)) & UInt64(0xFFFFFFFFFFFFFFFF)
    _ = h
    return (acc, left)


def _capped(m: Int, n: Int, k: Int, budget: Int) -> Tuple[Int, Int]:
    """Shrink `m` and `n` so `m * n * k` fits the budget. `k` IS NEVER
    TOUCHED, because contract 6.1 forbids the leaf rule to read `m` or `n`,
    so `L`, `P`, the ragged tail and the tree are the full shape's on both
    arms even where the output is smaller."""
    var mm = m
    var nn = n
    while mm > 1 and mm * nn * k > budget:
        mm = (mm + 1) // 2
    while nn > 1 and mm * nn * k > budget:
        nn = (nn + 1) // 2
    return (mm, nn)


def main() raises:
    comptime if not is_defined["MOJOLEARN_GEMM_UNPINNED_ARM"]():
        raise Error(
            "gemm_unpinned_price: build with -D MOJOLEARN_GEMM_UNPINNED_ARM=1."
            " The unpinned arm is a MEASUREMENT INSTRUMENT and is refused"
            " unless asked for by name."
        )

    print("== gemm/mojo_only/gemm_unpinned_price.mojo [" + _mode_name() + "] ==")
    print("   " + unpinned_gemm_banner(4))
    print(
        "   THE ONE-VARIABLE PRICE. Same plan, same tiles, same grid, same"
        " staging, same leaf schedule; ONLY the fold-order pin removed."
    )
    print(
        "   Three arms in ONE binary alternating CALL BY CALL, so the"
        " governor drift cancels at the arm level rather than being averaged"
        " over rounds."
    )
    print(
        "   PREDICTION ON RECORD (gemm/UNPINNED_CONTROL.md): 1.10x-1.45x at"
        " the tiled t512 rows, 1.00x-1.06x at the t1 rows, 1.3x-1.8x at"
        " gram.32x32x1M, and the fold under 2% anywhere."
    )
    if _mode_name() != "IDENTICAL":
        print(
            "   !! THIS BUILD IS FAST. The pinned arm is not the profile here"
            " and the ratio below is not the profile's price."
        )

    var budget = DEFAULT_MAC_BUDGET
    var bs = String(getenv("MOJOLEARN_UNPINNED_MAC_BUDGET"))
    if bs != "":
        budget = Int(atol(bs))

    print()
    print(
        "   name                            op     m x n            k      "
        " P   pinned ms  unpin ms  strict ms   unpin/pin  strict/pin  bits"
    )

    var refused = 0
    var inert = 0
    var plan_mismatch = 0
    for i in range(GEMM_SHAPE_COUNT):
        var name = gemm_shape_name(i)
        var op = _oracle_op(gemm_shape_op(i))
        var k = gemm_shape_k(i)
        var caps = _capped(gemm_shape_m(i), gemm_shape_n(i), k, budget)
        var m = caps[0]
        var n = caps[1]
        var mn = m * n

        # PLAN EQUALITY IS THE EXPERIMENT'S OWN PRECONDITION. If the two arms
        # ever pick different plans the ratio is between two kernels, not
        # between two fold orders, and the row is refused rather than
        # explained.
        var pin_plan = gemm_plan_name(choose_gemm_plan(m, n, k))
        var unp_plan = unpinned_gemm_plan_name(m, n, k)
        if pin_plan != unp_plan:
            print(
                "   REFUSED " + name + ": plans differ, pinned '" + pin_plan
                + "' vs unpinned '" + unp_plan + "'. Not the experiment."
            )
            plan_mismatch += 1
            refused += 1
            continue

        var ctx = DeviceContext()
        var na = _a_elems(op, m, n, k)
        var nb = _b_elems(op, m, n, k)
        var nws = identical_gemm_workspace_max_floats(m, n, k)
        var da = ctx.enqueue_create_buffer[DType.float32](na if na > 0 else 1)
        var db = ctx.enqueue_create_buffer[DType.float32](nb if nb > 0 else 1)
        var dc = ctx.enqueue_create_buffer[DType.float32](mn if mn > 0 else 1)
        var dw = ctx.enqueue_create_buffer[DType.float32](nws if nws > 0 else 1)
        ctx.synchronize()
        _dev_fill(ctx, da, na, 11 + i)
        _dev_fill(ctx, db, nb, 22 + i)

        # Untimed warm-up on each arm, each poisoned first and read back.
        _dev_poison(ctx, dc, mn)
        identical_gemm_into(ctx, dc, da, db, dw, m, n, k, op)
        ctx.synchronize()
        var dpin = _dev_digest(ctx, dc, mn)

        _dev_poison(ctx, dc, mn)
        unpinned_gemm_into(ctx, dc, da, db, dw, m, n, k, op)
        ctx.synchronize()
        var dunp = _dev_digest(ctx, dc, mn)

        _dev_poison(ctx, dc, mn)
        unpinned_gemm_into_one_acc(ctx, dc, da, db, dw, m, n, k, op)
        ctx.synchronize()
        var dstr = _dev_digest(ctx, dc, mn)

        if dpin[1] != 0 or dunp[1] != 0 or dstr[1] != 0:
            print(
                "   REFUSED " + name + ": cells left POISON -- pinned "
                + String(dpin[1]) + ", unpinned " + String(dunp[1])
                + ", strict " + String(dstr[1])
                + ". An arm that did not write its output cannot be timed."
            )
            refused += 1
            _ = da^
            _ = db^
            _ = dc^
            _ = dw^
            continue

        var ns_pin = 0
        var ns_unp = 0
        var ns_str = 0
        for _ in range(REPEATS):
            var t0 = perf_counter_ns()
            identical_gemm_into(ctx, dc, da, db, dw, m, n, k, op)
            ctx.synchronize()
            ns_pin += perf_counter_ns() - t0

            var t1 = perf_counter_ns()
            unpinned_gemm_into(ctx, dc, da, db, dw, m, n, k, op)
            ctx.synchronize()
            ns_unp += perf_counter_ns() - t1

            var t2 = perf_counter_ns()
            unpinned_gemm_into_one_acc(ctx, dc, da, db, dw, m, n, k, op)
            ctx.synchronize()
            ns_str += perf_counter_ns() - t2

        var ms_pin = Float64(ns_pin) / 1.0e6 / Float64(REPEATS)
        var ms_unp = Float64(ns_unp) / 1.0e6 / Float64(REPEATS)
        var ms_str = Float64(ns_str) / 1.0e6 / Float64(REPEATS)

        var L = contract_leaf_size(k)
        var P = contract_leaf_count(k)

        # A shape where the two arms agree bit for bit is one where the
        # unpinned arm removed NOTHING, so its time is not a price. Reported
        # as INERT, never as agreement. `P == 1` is the KNOWN-inert set --
        # with one partition there is no tree to remove -- and those rows are
        # the control for the fold's own contribution.
        var bitword = String("differ")
        if dpin[0] == dunp[0]:
            if P <= 1:
                bitword = String("INERT(P=1,expected)")
            else:
                bitword = String("INERT(UNEXPECTED)")
            inert += 1

        var opname = String("NN")
        if op == OP_NT:
            opname = String("NT")
        elif op == OP_TN:
            opname = String("TN")

        # `String.ljust` does not exist in this toolchain; pad by hand.
        var padded = name
        while len(padded.as_bytes()) < 32:
            padded += " "
        print(
            "   " + padded + opname + "  " + String(m) + "x" + String(n)
            + "  k=" + String(k) + "  P=" + String(P) + "   "
            + String(ms_pin) + "  " + String(ms_unp) + "  " + String(ms_str)
            + "   " + String(ms_pin / ms_unp) + "x  "
            + String(ms_pin / ms_str) + "x  " + bitword
        )
        _ = da^
        _ = db^
        _ = dc^
        _ = dw^

    print()
    print(
        "   ran " + String(GEMM_SHAPE_COUNT - refused) + " shapes, refused "
        + String(refused) + " (" + String(plan_mismatch)
        + " for a plan mismatch), " + String(inert)
        + " bitwise INERT. An INERT row removed nothing and its ratio is NOT"
        " a price; an INERT(UNEXPECTED) row at P > 1 is a FINDING about the"
        " unpinned arm and not about the pin."
    )
    print(
        "   unpin/pin above 1.00 is the PINNED arm costing more, which is the"
        " direction the pin is expected to push. strict/pin isolates the"
        " seams with the dependency chain held as long as the pinned kernel's;"
        " the gap between the two columns is contract 7.1's"
        " no-sub-partition clause."
    )
    print(
        "   SCOPE: this box, this build, these twenty shapes, "
        + String(REPEATS)
        + " timed calls per arm per shape after one untimed warm-up."
        " gemm/UNPINNED_CONTROL.md lists seven confounds this file does not"
        " remove, C1 (the pinned fold's register cost acting through"
        " occupancy) being the one timing alone cannot separate."
    )
