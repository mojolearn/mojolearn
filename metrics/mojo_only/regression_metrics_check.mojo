"""Group B gates: r2_score and kl_divergence, the float reductions over n.

    pixi run mojo run -I . metrics/mojo_only/regression_metrics_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . metrics/mojo_only/regression_metrics_check.mojo

DEVIATION 653. The device sum is `virtual_block_sum` (`pinned_sum.mojo`):
a PINNED_SUM_W slab tree per chunk, the chunk totals folded ascending on
the host. The oracle is `host_tree_sum` -- the same additions in the same
order, Float32, through the same `ftz` -- applied to the per-term values
the check maps on the host through the same helpers. IDENTICAL: the device
equals the oracle BIT FOR BIT and the bits do not move across launches.
FAST: the same lines are REPORTS and the Float64 reference (sklearn's
spelling) is the assertion, to 1e-5 relative.

The checks, in order:

    check_virtual_sum_equals_pinned_block_sum
                                   at block_size == PINNED_SUM_W the slab
                                   tree IS core/pinned_reduce.mojo's
                                   pinned_block_sum, bit for bit, on 256
                                   hashed values (IDENTICAL asserts)
    check_sum_order_is_visible     the fixture separates the pinned tree
                                   from (i) the same tree over chunk
                                   boundaries shifted by 1 / 7 / 64 values
                                   and (ii) a serial ascending fold, in the
                                   last bits, so the bitwise gates have
                                   teeth (a ROTATION inside a chunk is
                                   invisible to a halving tree, see
                                   pinned_sum.mojo)
    check_r2_matches_oracle        r2 on 4099 hashed rows with 7 planted
                                   SUBNORMAL squares (FTZ reached) vs the
                                   host model, bitwise (IDENTICAL) and vs
                                   Float64 sklearn r2 (both, 1e-5)
    check_r2_launch_invariant      THE HEADLINE: the Float32 BYTES of r2
                                   do not move across block 64 / 256, grid
                                   1-D / 2-D (gx = 3), and buffers padded
                                   by 0 / 37 cells poisoned with NaN
    check_kl_matches_oracle        KL on 4099-term hashed pdfs with zeros
                                   planted every 97th p (the `p == 0`
                                   branch) vs the host model, bitwise, and
                                   vs Float64 scipy spelling (1e-5); plus
                                   `q == 0, p > 0` -> +inf
    check_kl_launch_invariant      as r2's

SABOTAGES PERFORMED (2026-08-23), each reverted; outputs in the README:
    (a) the device chunk start shifted by one value (`i = chunk * W + 1 +
        tid + r * block_size`, what a partition that is not a function of
        n alone would fold): under IDENTICAL check_r2_matches_oracle
        fails bitwise (FAST: the report line moves)
    (b) `ftz` dropped from the ORACLE's sse term (the host keeps the
        planted subnormal squares the device flushes): under IDENTICAL
        check_r2_ftz_seam_is_visible fails bitwise (sse 0x00000000 vs a
        ~5e-35 sum) -- the seam is REACHED AND VISIBLE there, and a
        denormal-honoring vendor without the pin would diverge exactly
        here; on check_r2_matches_oracle the same sabotage moves NOTHING
        (1e-42 into a 1e6 sum), which is why the second fixture exists
    (c) `ftz` dropped from the DEVICE kernel's sse term: no bit moves on
        Apple (the hardware flushes; the pin is inert on this column, as
        numerics.mojo says) -- recorded as the expected null

ROW 39 AUDIT (2026-08-23) additions:
    check_r2_undefined_cases       DEVIATION 657: constant y -> 1.0 / 0.0,
                                   overflow y -> canonical NaN 0x7fc00000
                                   (both modes; oracle through the same
                                   epilogue)
    check_kl_subnormal_p_and_nan   DEVIATION 658: a subnormal p is finite
                                   and bitwise (IDENTICAL), a negative q
                                   is the canonical NaN (both modes)
    sabotages: (n) the operand flush dropped from kld_op -> `BITWISE
    MISMATCH kl(subnormal p) device 0x3f7b6548 oracle -inf (0xff800000)`;
    (o) the ssto == 0 arm dropped -> `r2(constant y, perfect) must be 1.0,
    got 0x7fc00000`; (p) canonicalize_nan dropped -> NO CHANGE on Apple
    (the ARM host's NaN is already 0x7fc00000; an x86 host's is
    0xffc00000, where it fails)
"""

from std.gpu import thread_idx
from std.math import log, sqrt
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from core.pinned_reduce import pinned_block_sum
from metrics.mojo_only.device_io import download_f32, upload_f32
from metrics.mojo_only.fixtures import (
    bits32,
    bits64,
    hashed_floats,
    hashed_pdf,
    plant_subnormal_squares,
)
from metrics.mojo_only.pinned_sum import (
    PINNED_SUM_W,
    canonicalize_nan,
    host_tree_sum,
    sabotage_shifted_host_tree_sum,
    virtual_block_sum,
)
from metrics.ported.metrics.kl_divergence import kl_divergence
from metrics.ported.metrics.r2_score import r2_score_py
from metrics.ported.stats.detail.kl_divergence import (
    kl_divergence_launch,
    kld_op,
)
from metrics.ported.stats.detail.scores import (
    r2_epilogue,
    r2_score_launch,
    r2_score_parts,
)
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_log,
)


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime N = 4099


def _mode_name() -> String:
    comptime if IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


def _nan32() -> Float32:
    return bitcast[DType.float32](UInt32(0x7FC00000))


# ---------------------------------------------------------------- oracles


def _oracle_r2(
    y: List[Float32], y_hat: List[Float32], n: Int
) -> Tuple[Float32, Float32, Float32, Float32]:
    """The host model of scores.mojo::r2_score_parts: three `host_tree_sum`s
    over terms mapped through the same helpers, `mean = sum * (1/n)`.
    Returns `(y_bar, sse, ssto, r2)`."""
    var ys = List[Float32]()
    for i in range(n):
        ys.append(y[i])
    var y_sum = host_tree_sum(ys, n)
    var ratio = ftz(Float32(1.0) / Float32(n))
    var y_bar = ftz(y_sum * ratio)
    var se = List[Float32]()
    var st = List[Float32]()
    for i in range(n):
        var d1 = ftz(y[i] - y_hat[i])
        var d2 = ftz(y[i] - y_bar)
        se.append(ftz(d1 * d1))
        st.append(ftz(d2 * d2))
    var sse = host_tree_sum(se, n)
    var ssto = host_tree_sum(st, n)
    # DEVIATION 657: the same epilogue as the device path (the oracle's
    # job is the three SUMS; the epilogue is two compares and one division)
    return (y_bar, sse, ssto, r2_epilogue(sse, ssto))


def _ref_r2_f64(y: List[Float32], y_hat: List[Float32], n: Int) -> Float64:
    """sklearn `r2_score`: `1 - sum((y - yhat)^2) / sum((y - mean)^2)`,
    Float64, serial."""
    var m = 0.0
    for i in range(n):
        m += Float64(y[i])
    m /= Float64(n)
    var sse = 0.0
    var ssto = 0.0
    for i in range(n):
        var d1 = Float64(y[i]) - Float64(y_hat[i])
        var d2 = Float64(y[i]) - m
        sse += d1 * d1
        ssto += d2 * d2
    return 1.0 - sse / ssto


def _oracle_kl(p: List[Float32], q: List[Float32], n: Int) -> Float32:
    var terms = List[Float32]()
    for i in range(n):
        # The per-term map through the SAME device helper is not an
        # independent oracle of the map; the Float64 reference below is.
        # What this oracle is independent of is the FOLD, which is the
        # thing DEVIATION 653 pins.
        terms.append(kld_op(p[i], q[i]))
    # DEVIATION 658 (2): the same canonical NaN as the device path
    return canonicalize_nan(host_tree_sum(terms, n))


def _ref_kl_f64(p: List[Float32], q: List[Float32], n: Int) -> Float64:
    """scipy `entropy(pk, qk)` spelling without the normalization RAFT
    does not do: `sum(pk * log(pk / qk))`."""
    var acc = 0.0
    for i in range(n):
        var pi = Float64(p[i])
        if pi != 0.0:
            acc += pi * log(pi / Float64(q[i]))
    return acc


def _rel(a: Float64, b: Float64) -> Float64:
    var d = abs(a - b)
    var m = abs(a) if abs(a) > abs(b) else abs(b)
    if m == 0.0:
        return d
    return d / m


def _bits_line(tag: String, got: Float32, want: Float32) -> String:
    return (
        tag
        + " device "
        + String(got)
        + " ("
        + bits32(got)
        + ") oracle "
        + String(want)
        + " ("
        + bits32(want)
        + ")"
    )


def _assert_bits32(tag: String, got: Float32, want: Float32) raises:
    if bits32(got) == bits32(want):
        print("    bitwise OK  " + _bits_line(tag, got, want))
    elif IDENTICAL:
        raise Error("BITWISE MISMATCH " + _bits_line(tag, got, want))
    else:
        print("    bitwise REPORT (FAST, not asserted) " + _bits_line(tag, got, want))


# ----------------------------------------------------------------- checks


def both_sums_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    results: MutPointer[Float32, MutAnyOrigin],
):
    """One block of PINNED_SUM_W threads: `results[0] = virtual_block_sum`,
    `results[1] = pinned_block_sum`, same 256 values."""
    var tid = Int(thread_idx.x)
    var v = x.unsafe_load(tid)
    var vals = SIMD[DType.float32, 1](v)
    var a = virtual_block_sum[PINNED_SUM_W](vals)
    var b = pinned_block_sum[PINNED_SUM_W](v)
    if tid == 0:
        results.unsafe_store(0, a)
        results.unsafe_store(1, b)


def check_virtual_sum_equals_pinned_block_sum(ctx: DeviceContext) raises:
    print("check_virtual_sum_equals_pinned_block_sum [" + _mode_name() + "]")
    var xs = hashed_floats(PINNED_SUM_W, 77, -12, 12)
    var dx = upload_f32(ctx, xs)
    var out = ctx.enqueue_create_buffer[DType.float32](2)
    ctx.enqueue_function[both_sums_kernel](
        dx.unsafe_ptr(),
        out.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(PINNED_SUM_W, 1, 1),
    )
    var h = download_f32(ctx, out, 2)
    var model = host_tree_sum(xs, PINNED_SUM_W)
    print(
        "    virtual "
        + bits32(h[0])
        + " pinned_block_sum "
        + bits32(h[1])
        + " host model "
        + bits32(model)
    )
    comptime if IDENTICAL:
        if bits32(h[0]) != bits32(h[1]) or bits32(h[0]) != bits32(model):
            raise Error("virtual_block_sum != pinned_block_sum at W == block")
    else:
        print("    (FAST: block.sum vs block.sum; the model is the tree and differs, as it should)")
    print("  OK")


def check_sum_order_is_visible(ctx: DeviceContext) raises:
    """A ROTATION inside the chunk is invisible to this tree by
    construction (pinned_sum.mojo::sabotage_shifted_host_tree_sum says
    why), so what is tried is what a launch-dependent partition would do:
    shift the chunk boundaries by 1, 7, 64 values. Serial is the fold a
    naive port would write. A fixture that separated none of these would
    have no teeth."""
    print("check_sum_order_is_visible [" + _mode_name() + "]")
    var xs = hashed_floats(N, 5, -10, 10)
    var tree = host_tree_sum(xs, N)
    var serial = Float32(0.0)
    for i in range(N):
        serial = ftz(serial + ftz(xs[i]))
    var shifts = List[Int]()
    shifts.append(1)
    shifts.append(7)
    shifts.append(64)
    var differing = 0
    var line = String("    tree ") + bits32(tree)
    for r in range(len(shifts)):
        var shifted = sabotage_shifted_host_tree_sum(xs, N, shifts[r])
        line += " shift" + String(shifts[r]) + " " + bits32(shifted)
        if bits32(shifted) != bits32(tree):
            differing += 1
    print(line + " serial " + bits32(serial))
    if differing == 0:
        raise Error("fixture does not separate any shifted partition; no teeth")
    if bits32(tree) == bits32(serial):
        raise Error("fixture does not separate a serial fold; no teeth")
    print("    " + String(differing) + " of 3 shifts and the serial fold differ")
    print("  OK")


def _r2_fixture() -> Tuple[List[Float32], List[Float32]]:
    var y = hashed_floats(N, 1, -6, 6)
    var y_hat = List[Float32]()
    # y_hat = y + hashed residual of comparable magnitude, so r2 is well
    # inside (0, 1) and its own bits can move; a residual far below y
    # makes sse << ssto and r2 absorbs a last-bit move in sse.
    var res = hashed_floats(N, 2, -6, 5)
    for i in range(N):
        y_hat.append(y[i] + res[i])
    plant_subnormal_squares(y, y_hat, 7, 3)
    return (y^, y_hat^)


def check_r2_matches_oracle(ctx: DeviceContext) raises:
    print("check_r2_matches_oracle [" + _mode_name() + "]")
    var f = _r2_fixture()
    var y = f[0].copy()
    var y_hat = f[1].copy()
    # prove the FTZ seam is planted: at least one (y - y_hat)^2 is subnormal
    var subnormal = 0
    for i in range(N):
        var d = y[i] - y_hat[i]
        var sq = d * d
        if sq != Float32(0.0) and abs(sq) < Float32(1.1754943508222875e-38):
            subnormal += 1
    print("    planted subnormal squares: " + String(subnormal))
    if subnormal == 0:
        raise Error("fixture plants no subnormal square; the FTZ seam is unreached")
    var dy = upload_f32(ctx, y)
    var dyh = upload_f32(ctx, y_hat)
    var got = r2_score_parts[256](ctx, dy, dyh, N, 0)
    var want = _oracle_r2(y, y_hat, N)
    _assert_bits32("y_bar", got[0], want[0])
    _assert_bits32("sse", got[1], want[1])
    _assert_bits32("ssto", got[2], want[2])
    _assert_bits32("r2_score", got[3], want[3])
    var via_wrapper = r2_score_py(ctx, dy, dyh, N)
    if bits32(via_wrapper) != bits32(got[3]):
        raise Error("r2_score_py and r2_score_parts disagree")
    var reference = _ref_r2_f64(y, y_hat, N)
    var rel = _rel(Float64(got[3]), reference)
    print("    vs Float64 sklearn r2 " + String(reference) + ": rel " + String(rel))
    if rel > 1e-5:
        raise Error("r2 off the Float64 reference by " + String(rel))
    print("  OK")


def check_r2_ftz_seam_is_visible(ctx: DeviceContext) raises:
    """A planted subnormal square inside a sum of order 1e6 is REACHED and
    INVISIBLE: the first draft dropped `ftz` from the oracle's sse term
    and nothing moved, because 1e-42 added to 1e6 is 1e6 in Float32. This
    fixture makes EVERY square subnormal (y_hat = y + 2^-69 u), so sse is
    a sum of 4099 subnormals: +0.0 under the row-10 policy (Apple's
    hardware, and `ftz` on a denormal-honoring vendor), ~5e-35 without it.
    The sse PART is gated (r2 = 1 - 0/ssto = 1.0 absorbs it either way),
    and sabotage (b) fails here."""
    print("check_r2_ftz_seam_is_visible [" + _mode_name() + "]")
    var y = hashed_floats(N, 21, -3, 3)
    var y_hat = List[Float32]()
    for i in range(N):
        # y_hat == y exactly everywhere (sse term +0.0), except the 64
        # planted rows, which carry y = 2^-69 (1 + u), y_hat = 2^-69, so
        # their squares are subnormal and ssto (about the mean of a
        # normal-spread y) stays healthy.
        y_hat.append(y[i])
    plant_subnormal_squares(y, y_hat, 64, 9)
    var dy = upload_f32(ctx, y)
    var dyh = upload_f32(ctx, y_hat)
    var parts = r2_score_parts[256](ctx, dy, dyh, N, 0)
    var want = _oracle_r2(y, y_hat, N)
    # an unflushed host sum of the 64 subnormal squares, for the record
    var kept = Float32(0.0)
    for i in range(N):
        var d = y[i] - y_hat[i]
        kept += d * d
    print(
        "    sse device "
        + bits32(parts[1])
        + " oracle "
        + bits32(want[1])
        + " (unflushed host sum of the squares would be "
        + bits32(kept)
        + ")"
    )
    _assert_bits32("sse(all-subnormal)", parts[1], want[1])
    comptime if IDENTICAL:
        if bits32(parts[1]) != "0x00000000":
            raise Error("sse of subnormal squares must flush to +0.0 under IDENTICAL")
        if bits32(kept) == "0x00000000":
            raise Error("fixture's squares are not subnormal; the seam is unreached")
    _assert_bits32("r2(all-subnormal sse)", parts[3], want[3])
    print("  OK")


def _launch_r2_variants(
    ctx: DeviceContext, y: List[Float32], y_hat: List[Float32], pad: Int
) raises -> List[Float32]:
    """r2 at (block 256, 1-D), (block 64, 1-D), (block 256, 2-D gx=3),
    (block 64, 2-D gx=3), on buffers padded by `pad` NaN cells."""
    var yp = y.copy()
    var yhp = y_hat.copy()
    for _ in range(pad):
        yp.append(_nan32())
        yhp.append(_nan32())
    var dy = upload_f32(ctx, yp)
    var dyh = upload_f32(ctx, yhp)
    var out = List[Float32]()
    out.append(r2_score_launch[256](ctx, dy, dyh, N, 0))
    out.append(r2_score_launch[64](ctx, dy, dyh, N, 0))
    out.append(r2_score_launch[256](ctx, dy, dyh, N, 3))
    out.append(r2_score_launch[64](ctx, dy, dyh, N, 3))
    return out^


def check_r2_launch_invariant(ctx: DeviceContext) raises:
    print("check_r2_launch_invariant [" + _mode_name() + "]")
    var f = _r2_fixture()
    var a = _launch_r2_variants(ctx, f[0], f[1], 0)
    var b = _launch_r2_variants(ctx, f[0], f[1], 37)
    var names = List[String]()
    names.append("b256 g1d pad0")
    names.append("b64  g1d pad0")
    names.append("b256 g2d pad0")
    names.append("b64  g2d pad0")
    names.append("b256 g1d pad37")
    names.append("b64  g1d pad37")
    names.append("b256 g2d pad37")
    names.append("b64  g2d pad37")
    var all_ = List[Float32]()
    for i in range(4):
        all_.append(a[i])
    for i in range(4):
        all_.append(b[i])
    var moved = 0
    for i in range(8):
        print("    " + names[i] + " " + bits32(all_[i]))
        if bits32(all_[i]) != bits32(all_[0]):
            moved += 1
    comptime if IDENTICAL:
        if moved != 0:
            raise Error(
                "r2 bytes moved across launches: " + String(moved) + " of 8 differ"
            )
        print("    8 launches, one byte pattern")
    else:
        print("    FAST: " + String(moved) + " of 8 differ from the first (block.sum is a function of the block; a REPORT)")
    print("  OK")


def _kl_fixture() -> Tuple[List[Float32], List[Float32]]:
    var p = hashed_pdf(N, 8, 97)
    var q = hashed_pdf(N, 9, 0)
    return (p^, q^)


def check_kl_matches_oracle(ctx: DeviceContext) raises:
    print("check_kl_matches_oracle [" + _mode_name() + "]")
    var f = _kl_fixture()
    var p = f[0].copy()
    var q = f[1].copy()
    var zeros = 0
    for i in range(N):
        if p[i] == Float32(0.0):
            zeros += 1
    print("    planted p == 0 terms: " + String(zeros))
    if zeros == 0:
        raise Error("fixture plants no p == 0; the branch is unreached")
    var dp = upload_f32(ctx, p)
    var dq = upload_f32(ctx, q)
    var got = kl_divergence(ctx, dp, dq, N)
    var want = _oracle_kl(p, q, N)
    _assert_bits32("kl_divergence", got, want)
    var reference = _ref_kl_f64(p, q, N)
    var rel = _rel(Float64(got), reference)
    print("    vs Float64 scipy spelling " + String(reference) + ": rel " + String(rel))
    if rel > 1e-5:
        raise Error("KL off the Float64 reference by " + String(rel))
    # q == 0 with p > 0 -> +inf, as RAFT and scipy
    var q0 = q.copy()
    q0[5] = Float32(0.0)
    var dq0 = upload_f32(ctx, q0)
    var inf = kl_divergence(ctx, dp, dq0, N)
    print("    q[5] = 0: " + String(inf) + " (" + bits32(inf) + ")")
    if bits32(inf) != "0x7f800000":
        raise Error("KL with q == 0, p > 0 must be +inf")
    print("  OK")


def check_r2_undefined_cases(ctx: DeviceContext) raises:
    """DEVIATION 657's gate. n = 512 (a power of two, so `sum * (1/n)` is
    exact and `y_bar == y` for a constant y under ANY fold) and y = 2.0:
    ssto == 0 exactly in both modes. y_hat == y -> sse == 0 -> 1.0; y_hat
    = y + 1 -> sse = 512 -> 0.0 (cuML's surface / sklearn force_finite).
    Then an OVERFLOW-SCALE y: +-2^64 alternating (every partial sum of
    +-2^64 over 512 terms is an exact multiple of 2^64, so y_bar is exactly
    0 under ANY fold and no `inf - inf` forms in the mean), y_hat = -y, so
    every (y - y_hat)^2 = 2^130 and (y - y_bar)^2 = 2^128 is +inf, sse =
    ssto = +inf, and `inf / inf` is a NaN that leaves with the canonical
    payload 0x7fc00000. Asserted in BOTH modes (exact sums and IEEE infinities,
    no rounding involved); the oracle goes through the same epilogue."""
    print("check_r2_undefined_cases [" + _mode_name() + "]")
    var n = 512
    var y = List[Float32]()
    var same = List[Float32]()
    var off = List[Float32]()
    for _ in range(n):
        y.append(Float32(2.0))
        same.append(Float32(2.0))
        off.append(Float32(3.0))
    var dy = upload_f32(ctx, y)
    var dsame = upload_f32(ctx, same)
    var doff = upload_f32(ctx, off)
    var p1 = r2_score_parts[256](ctx, dy, dsame, n, 0)
    var p2 = r2_score_parts[256](ctx, dy, doff, n, 0)
    var o1 = _oracle_r2(y, same, n)
    var o2 = _oracle_r2(y, off, n)
    print(
        "    constant y, y_hat == y: y_bar " + bits32(p1[0]) + " sse " + bits32(p1[1])
        + " ssto " + bits32(p1[2]) + " r2 " + bits32(p1[3]) + " (oracle " + bits32(o1[3]) + ")"
    )
    print(
        "    constant y, y_hat != y: sse " + bits32(p2[1]) + " ssto " + bits32(p2[2])
        + " r2 " + bits32(p2[3]) + " (oracle " + bits32(o2[3]) + ")"
    )
    if bits32(p1[2]) != "0x00000000" or bits32(p2[2]) != "0x00000000":
        raise Error("ssto of a constant y must be exactly +0.0 on this fixture")
    if bits32(p1[3]) != "0x3f800000" or bits32(o1[3]) != "0x3f800000":
        raise Error("r2(constant y, perfect) must be 1.0 (force_finite), got " + bits32(p1[3]))
    if bits32(p2[3]) != "0x00000000" or bits32(o2[3]) != "0x00000000":
        raise Error("r2(constant y, imperfect) must be 0.0 (force_finite), got " + bits32(p2[3]))
    var via = r2_score_py(ctx, dy, doff, n)
    if bits32(via) != bits32(p2[3]):
        raise Error("r2_score_py and r2_score_parts disagree on the constant fixture")
    # overflow: sse = ssto = +inf -> inf / inf -> canonical NaN
    var big = List[Float32]()
    var neg = List[Float32]()
    for i in range(n):
        var v = Float32(18446744073709551616.0) if i % 2 == 0 else Float32(-18446744073709551616.0)
        big.append(v)
        neg.append(-v)
    var dbig = upload_f32(ctx, big)
    var dneg = upload_f32(ctx, neg)
    var p3 = r2_score_parts[256](ctx, dbig, dneg, n, 0)
    var o3 = _oracle_r2(big, neg, n)
    print(
        "    overflow y: sse " + bits32(p3[1]) + " ssto " + bits32(p3[2]) + " r2 "
        + bits32(p3[3]) + " (oracle " + bits32(o3[3]) + ")"
    )
    if bits32(p3[0]) != "0x00000000":
        raise Error("y_bar of the +-2^64 fixture must be exactly +0.0, got " + bits32(p3[0]))
    if bits32(p3[1]) != "0x7f800000" or bits32(p3[2]) != "0x7f800000":
        raise Error("overflow fixture must drive sse and ssto to +inf")
    if bits32(p3[3]) != "0x7fc00000" or bits32(o3[3]) != "0x7fc00000":
        raise Error("r2 = inf / inf must leave as the canonical NaN 0x7fc00000, got " + bits32(p3[3]))
    print("    1.0 / 0.0 / 0x7fc00000: the three undefined cases are defined, both modes")
    print("  OK")


def check_kl_subnormal_p_and_nan(ctx: DeviceContext) raises:
    """DEVIATION 658's gate. (1) p[3] is SUBNORMAL (1e-40) with q[3] normal:
    an FTZ device reads it as 0 and takes the `0` branch, a denormal-
    honoring device reads it as nonzero and `identical_log` flushes it to
    `-inf`; with the operand flush both take the `0` branch. The device sum
    is FINITE (both modes; FAST's stdlib log of 1e-40 is finite too) and
    bitwise the host model's under IDENTICAL. (2) q[7] = -1: `log(-1)` is
    NaN, the term and the sum are NaN, and the returned scalar is the
    canonical 0x7fc00000 in both modes (the canonicalization is a host
    compare; IEEE says log of a negative is NaN everywhere)."""
    print("check_kl_subnormal_p_and_nan [" + _mode_name() + "]")
    var f = _kl_fixture()
    var p = f[0].copy()
    var q = f[1].copy()
    p[3] = Float32(1.0e-40)
    if not (p[3] != Float32(0.0) and abs(p[3]) < Float32(1.1754943508222875e-38)):
        raise Error("p[3] is not subnormal on this host; the fixture is not planted")
    var dp = upload_f32(ctx, p)
    var dq = upload_f32(ctx, q)
    var got = kl_divergence(ctx, dp, dq, N)
    var want = _oracle_kl(p, q, N)
    print("    subnormal p[3]: device " + bits32(got) + " oracle " + bits32(want))
    # Leg 11 (144aa5b) on the H100: under FAST this device keeps the
    # subnormal (no FTZ) and its stdlib log returns -inf for p = 1e-40, so
    # the sum is -inf -- a vendor-shaped FAST answer. The docstring's "FAST's
    # stdlib log of 1e-40 is finite too" was Apple's flush, not a rule.
    # IDENTICAL asserts (the operand is flushed on load, DEVIATION 658);
    # FAST records.
    if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if got != got or bits32(got) == "0xff800000" or bits32(got) == "0x7f800000":
            raise Error("a subnormal p must contribute 0, not -inf or NaN: " + bits32(got))
        _assert_bits32("kl(subnormal p)", got, want)
    else:
        print("    RECORDED [FAST]: kl(subnormal p) device " + bits32(got) + " (vendor log of a kept subnormal; not judged)")
    var qn = q.copy()
    qn[7] = Float32(-1.0)
    var dqn = upload_f32(ctx, qn)
    var nan = kl_divergence(ctx, dp, dqn, N)
    var nan_o = _oracle_kl(p, qn, N)
    print("    q[7] = -1: device " + bits32(nan) + " oracle " + bits32(nan_o))
    if bits32(nan) != "0x7fc00000" or bits32(nan_o) != "0x7fc00000":
        raise Error("KL's NaN must leave as the canonical 0x7fc00000, got " + bits32(nan))
    print("  OK")


def check_kl_launch_invariant(ctx: DeviceContext) raises:
    print("check_kl_launch_invariant [" + _mode_name() + "]")
    var f = _kl_fixture()
    var outs = List[Float32]()
    var names = List[String]()
    for pad_i in range(2):
        var pad = 0 if pad_i == 0 else 37
        var pp = f[0].copy()
        var qq = f[1].copy()
        for _ in range(pad):
            pp.append(_nan32())
            qq.append(_nan32())
        var dp = upload_f32(ctx, pp)
        var dq = upload_f32(ctx, qq)
        outs.append(kl_divergence_launch[256](ctx, dp, dq, N, 0))
        names.append("b256 g1d pad" + String(pad))
        outs.append(kl_divergence_launch[64](ctx, dp, dq, N, 0))
        names.append("b64  g1d pad" + String(pad))
        outs.append(kl_divergence_launch[256](ctx, dp, dq, N, 3))
        names.append("b256 g2d pad" + String(pad))
        outs.append(kl_divergence_launch[64](ctx, dp, dq, N, 3))
        names.append("b64  g2d pad" + String(pad))
    var moved = 0
    for i in range(len(outs)):
        print("    " + names[i] + " " + bits32(outs[i]))
        if bits32(outs[i]) != bits32(outs[0]):
            moved += 1
    comptime if IDENTICAL:
        if moved != 0:
            raise Error("KL bytes moved across launches: " + String(moved) + " of 8")
        print("    8 launches, one byte pattern")
    else:
        print("    FAST: " + String(moved) + " of 8 differ from the first (a REPORT)")
    print("  OK")


def main() raises:
    print(
        "== metrics/mojo_only/regression_metrics_check.mojo ["
        + _mode_name()
        + "] =="
    )
    var ctx = DeviceContext()
    check_virtual_sum_equals_pinned_block_sum(ctx)
    check_sum_order_is_visible(ctx)
    check_r2_matches_oracle(ctx)
    check_r2_ftz_seam_is_visible(ctx)
    check_r2_launch_invariant(ctx)
    check_r2_undefined_cases(ctx)
    check_kl_matches_oracle(ctx)
    check_kl_subnormal_p_and_nan(ctx)
    check_kl_launch_invariant(ctx)
    print("ALL GROUP B CHECKS PASSED [" + _mode_name() + "]")
