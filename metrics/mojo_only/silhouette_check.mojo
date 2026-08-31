# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Group C gates: silhouette_score / silhouette_samples (the batched path).

    pixi run mojo run -I . metrics/mojo_only/silhouette_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . metrics/mojo_only/silhouette_check.mojo

DEVIATION 654. The oracle (`_oracle_silhouette`) is the host model of the
row kernel: for every row i and cluster c, the terms `d(i,j)/denom_c` over
j ascending (0 where `y[j] != c` or `j == i`), folded by `host_tree_sum`
(the same slab tree), then the ascending min over c and `sil_op`; the
distances through `host_l2sqrt_unexpanded` (the same helpers); the mean
through `host_tree_sum` over the scores. IDENTICAL: every per-sample
score and the mean equal the device BIT FOR BIT, and the bytes do not move
across launches. FAST: REPORTs; the Float64 sklearn spelling is the
assertion (1e-4 relative on the mean, 1e-4 absolute per sample).

The checks, in order:

    check_silhouette_matches_oracle     1031 hashed points, 5 dims, 6
                                        skewed clusters (one or more may
                                        be EMPTY slots: the counts == 0
                                        arm): per-sample bitwise + mean
                                        bitwise (IDENTICAL); vs Float64
                                        sklearn silhouette_samples /
                                        silhouette_score (both modes)
    check_silhouette_singleton_and_empty
                                        a planted singleton cluster scores
                                        +0.0 exactly; an EMPTY label slot
                                        (n_labels = 8, label 6 unused) is
                                        skipped by the min; bitwise vs
                                        oracle
    check_silhouette_planted_ties       the 3-point fixture (0,0),(1,0) |
                                        (0,1): row 0 has a == b == 1
                                        EXACTLY (SilOp's tie branch ->
                                        +0.0, bits 0x00000000, never
                                        -0.0), row 2 is a singleton (+0.0),
                                        row 1 is (sqrt2 - 1)/sqrt2 bitwise
                                        vs the host formula; plus a
                                        two-way tie in the min over
                                        clusters (two clusters at the same
                                        mean distance: the value is the
                                        same whichever index wins)
                                        the a == b == 0 case (all points
                                        identical) in BOTH cluster orders
                                        -> 0x00000000, both modes (ROW 39:
                                        -0.0 is unreachable as an operand,
                                        so the tie IS the fixture)
    check_silhouette_inf_distances      DEVIATION 656: an overflow-scale
                                        finite X drives a = +inf; the NaN
                                        quotient records +0.0, the b = inf
                                        arm is shown unreachable (b stays
                                        FLT_MAX -> exactly 1.0), ordinary
                                        rows stay ordinary; both modes
    (every fixture)                     negative-zero scores counted: 0
    check_silhouette_refusals           n_labels < 2, n_labels > n_rows-1,
                                        metric != L2SqrtUnexpanded, chunk
                                        < 1 RAISE by name; chunk 1 vs
                                        40000 vs 7 give the same bytes
                                        (chunk is scheduling)
    check_silhouette_launch_invariant   THE HEADLINE: all n per-sample
                                        BYTES and the mean do not move
                                        across block 64 / 256, grid 1-D /
                                        2-D (gx = 5), pad 0 / 37 NaN rows

SABOTAGES PERFORMED (2026-08-23), each reverted; outputs in the README:
    (g) the row kernel's j partition shifted by one with wrap (`j = (j0 +
        1) % n_rows`, the same multiset in different chunks): IDENTICAL
        check_silhouette_matches_oracle fails bitwise
    (h) `identical_sqrt` replaced by `std.math.sqrt` in the device
        distance: NO bit moves on Apple (its sqrt is correctly rounded;
        DEVIATION 258 measured the NVIDIA one approximate) -- recorded as
        the expected null; the pin's value is the NVIDIA column
    (i) `ftz` dropped from the device distance's stored diff: null on
        Apple (hardware flush; no subnormal on this fixture anyway)
    (k) DEVIATION 656's NaN guard removed from sil_op: check_silhouette_
        inf_distances fails (`row 0 ... must record +0.0, got 0x7fc00000`)
    (l) the min over clusters flipped from `<` to `<=`: NO CHANGE on any
        vendor, by proof (no candidate pair differs only in zero sign)
"""

from std.math import sqrt
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.mojo_only.device_io import download_f32, upload_f32, upload_i32
from metrics.mojo_only.fixtures import bits32, bits64, hashed_points, u01
from metrics.mojo_only.pinned_distance import host_l2sqrt_unexpanded
from metrics.mojo_only.pinned_sum import host_tree_sum
from metrics.ported.metrics.silhouette_score_batched_float import (
    silhouette_score,
)
from metrics.ported.stats.detail.batched.silhouette_score import (
    DISTANCE_L2_SQRT_UNEXPANDED,
    silhouette_score_launch,
)
from metrics.ported.stats.detail.silhouette_score import sil_op
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, numeric_mode_name


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime N = 1031
comptime D = 5
comptime K = 6


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


def _nan32() -> Float32:
    return bitcast[DType.float32](UInt32(0x7FC00000))


def _f32_max() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F7FFFFF))


# ---------------------------------------------------------------- oracles


def _oracle_silhouette(
    x: List[Float32], y: List[Int32], n: Int, d: Int, n_labels: Int
) -> Tuple[List[Float32], Float32]:
    """Host model of the row kernel (DEVIATION 654): returns (scores, mean)."""
    var counts = List[Int]()
    for _ in range(n_labels):
        counts.append(0)
    for i in range(n):
        counts[Int(y[i])] += 1
    var scores = List[Float32]()
    for i in range(n):
        var rc = Int(y[i])
        var singleton = counts[rc] == 1
        var a = Float32(0.0)
        var b = List[Float32]()
        for c in range(n_labels):
            if c == rc or counts[c] == 0:
                b.append(Float32(0.0) if singleton else _f32_max())
            else:
                b.append(Float32(0.0))
        if not singleton:
            # distances of row i to every j, once
            var dist = List[Float32]()
            for j in range(n):
                dist.append(
                    host_l2sqrt_unexpanded(x, i, j, d) if j != i else Float32(0.0)
                )
            for c in range(n_labels):
                var denom = Float32(counts[c] - 1) if c == rc else Float32(counts[c])
                var terms = List[Float32]()
                for j in range(n):
                    if j != i and Int(y[j]) == c:
                        terms.append(ftz(dist[j] / denom))
                    else:
                        terms.append(Float32(0.0))
                var s = host_tree_sum(terms, n)
                if c == rc:
                    a = ftz(a + ftz(s))
                else:
                    b[c] = ftz(b[c] + ftz(s))
        var bmin = _f32_max()
        for c in range(n_labels):
            if b[c] < bmin:
                bmin = b[c]
        scores.append(sil_op(a, bmin))
    var mean = ftz(host_tree_sum(scores, n) / Float32(n))
    return (scores^, mean)


def _ref_silhouette_f64(
    x: List[Float32], y: List[Int32], n: Int, d: Int, n_labels: Int
) -> Tuple[List[Float64], Float64]:
    """sklearn `silhouette_samples` in Float64: a = mean intra (sum then
    divide), b = min over other clusters of the mean distance, s = (b - a)
    / max(a, b) with nan -> 0, singleton -> 0; `silhouette_score` = mean."""
    var counts = List[Int]()
    for _ in range(n_labels):
        counts.append(0)
    for i in range(n):
        counts[Int(y[i])] += 1
    var out = List[Float64]()
    var total = 0.0
    for i in range(n):
        var rc = Int(y[i])
        if counts[rc] == 1:
            out.append(0.0)
            continue
        var sums = List[Float64]()
        for _ in range(n_labels):
            sums.append(0.0)
        for j in range(n):
            if j == i:
                continue
            var acc = 0.0
            for f in range(d):
                var diff = Float64(x[i * d + f]) - Float64(x[j * d + f])
                acc += diff * diff
            sums[Int(y[j])] += sqrt(acc)
        var a = sums[rc] / Float64(counts[rc] - 1)
        var b = 1.7976931348623157e308
        for c in range(n_labels):
            if c != rc and counts[c] > 0:
                var m = sums[c] / Float64(counts[c])
                if m < b:
                    b = m
        var mx = a if a > b else b
        var s = 0.0
        if mx != 0.0:
            s = (b - a) / mx
        out.append(s)
        total += s
    return (out^, total / Float64(n))


def _rel(a: Float64, b: Float64) -> Float64:
    var dd = abs(a - b)
    var m = abs(a) if abs(a) > abs(b) else abs(b)
    if m == 0.0:
        return dd
    return dd / m


def _count_neg_zero(v: List[Float32], n: Int) -> Int:
    """IDENTITY_PATHS row 39: a `-0.0` score is unreachable on any legal X
    (silhouette_score.mojo's header proves it); every fixture counts them
    and asserts none, in BOTH modes (the proof does not depend on the fold
    shape: no sum of nonnegative terms seeded +0.0 is -0.0 under any tree,
    and the quotient never underflows)."""
    var c = 0
    for i in range(n):
        if bits32(v[i]) == "0x80000000":
            c += 1
    return c


def _compare_samples(
    tag: String,
    got: List[Float32],
    want: List[Float32],
    n: Int,
    assert_bits: Bool,
) raises -> Int:
    var bad = 0
    for i in range(n):
        if bits32(got[i]) != bits32(want[i]):
            if bad < 4:
                print(
                    "    "
                    + tag
                    + " row "
                    + String(i)
                    + " device "
                    + bits32(got[i])
                    + " oracle "
                    + bits32(want[i])
                )
            bad += 1
    if bad == 0:
        print("    " + tag + ": " + String(n) + " per-sample scores bitwise")
    elif assert_bits:
        raise Error(tag + ": " + String(bad) + " of " + String(n) + " scores differ")
    else:
        print("    " + tag + ": REPORT (FAST, not asserted) " + String(bad) + " of " + String(n) + " differ")
    return bad


def _run(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Int32],
    n: Int,
    d: Int,
    n_labels: Int,
) raises -> Tuple[List[Float32], Float32]:
    var dx = upload_f32(ctx, x)
    var dy = upload_i32(ctx, y)
    var ds = ctx.enqueue_create_buffer[DType.float32](n)
    var mean = silhouette_score(ctx, dx, n, d, dy, n_labels, ds)
    var s = download_f32(ctx, ds, n)
    return (s^, mean)


# ----------------------------------------------------------------- checks


def check_silhouette_matches_oracle(ctx: DeviceContext) raises:
    print("check_silhouette_matches_oracle [" + _mode_name() + "]")
    var f = hashed_points(N, D, K, 3)
    var x = f[0].copy()
    var y = f[1].copy()
    var counts = List[Int]()
    for _ in range(K):
        counts.append(0)
    for i in range(N):
        counts[Int(y[i])] += 1
    var line = String("    cluster counts:")
    for c in range(K):
        line += " " + String(counts[c])
    print(line)
    var got = _run(ctx, x, y, N, D, K)
    var want = _oracle_silhouette(x, y, N, D, K)
    _ = _compare_samples("samples", got[0], want[0], N, IDENTICAL)
    var mline = (
        "mean device "
        + String(got[1])
        + " ("
        + bits32(got[1])
        + ") oracle "
        + String(want[1])
        + " ("
        + bits32(want[1])
        + ")"
    )
    if bits32(got[1]) == bits32(want[1]):
        print("    bitwise OK  " + mline)
    elif IDENTICAL:
        raise Error("BITWISE MISMATCH " + mline)
    else:
        print("    bitwise REPORT (FAST, not asserted) " + mline)
    var reference = _ref_silhouette_f64(x, y, N, D, K)
    var worst = 0.0
    for i in range(N):
        var dd = abs(Float64(got[0][i]) - reference[0][i])
        if dd > worst:
            worst = dd
    var rel = _rel(Float64(got[1]), reference[1])
    print(
        "    vs Float64 sklearn: mean "
        + String(reference[1])
        + " rel "
        + String(rel)
        + "; worst per-sample abs "
        + String(worst)
    )
    if rel > 1e-4 or worst > 1e-4:
        raise Error("silhouette off the Float64 reference")
    var nz = _count_neg_zero(got[0], N)
    print("    negative-zero scores: " + String(nz) + " (row 39: must be 0)")
    if nz != 0:
        raise Error("a -0.0 silhouette score was recorded; row 39 says unreachable")
    print("  OK")


def check_silhouette_singleton_and_empty(ctx: DeviceContext) raises:
    print("check_silhouette_singleton_and_empty [" + _mode_name() + "]")
    var f = hashed_points(N, D, K, 7)
    var x = f[0].copy()
    var y = f[1].copy()
    # label 7 is a singleton at row 41; label 6 is never used (empty slot)
    var n_labels = 8
    y[41] = Int32(7)
    var got = _run(ctx, x, y, N, D, n_labels)
    var want = _oracle_silhouette(x, y, N, D, n_labels)
    _ = _compare_samples("samples(singleton+empty)", got[0], want[0], N, IDENTICAL)
    print("    singleton row 41 score " + bits32(got[0][41]))
    if bits32(got[0][41]) != "0x00000000":
        raise Error("a singleton cluster must score +0.0 exactly")
    var reference = _ref_silhouette_f64(x, y, N, D, n_labels)
    var rel = _rel(Float64(got[1]), reference[1])
    print("    vs Float64 sklearn mean rel " + String(rel))
    if rel > 1e-4:
        raise Error("silhouette off the Float64 reference")
    print("  OK")


def check_silhouette_planted_ties(ctx: DeviceContext) raises:
    print("check_silhouette_planted_ties [" + _mode_name() + "]")
    # (0,0) and (1,0) in cluster 0; (0,1) in cluster 1
    var x = List[Float32]()
    x.append(0.0)
    x.append(0.0)
    x.append(1.0)
    x.append(0.0)
    x.append(0.0)
    x.append(1.0)
    var y = List[Int32]()
    y.append(0)
    y.append(0)
    y.append(1)
    var got = _run(ctx, x, y, 3, 2, 2)
    var want = _oracle_silhouette(x, y, 3, 2, 2)
    _ = _compare_samples("samples(3-point tie)", got[0], want[0], 3, IDENTICAL)
    print(
        "    row0 (a == b == 1, tie) "
        + bits32(got[0][0])
        + " row1 "
        + bits32(got[0][1])
        + " row2 (singleton) "
        + bits32(got[0][2])
    )
    if bits32(got[0][0]) != "0x00000000":
        raise Error("tie a == b must return +0.0 (not -0.0, not NaN)")
    if bits32(got[0][2]) != "0x00000000":
        raise Error("singleton must return +0.0")
    # row 1: a = 1, b = sqrt(2): (b - a) / b through the same helpers
    var b1 = host_l2sqrt_unexpanded(x, 1, 2, 2)
    var s1 = sil_op(Float32(1.0), b1)
    if bits32(got[0][1]) != bits32(s1):
        comptime if IDENTICAL:
            raise Error("row 1 " + bits32(got[0][1]) + " != host " + bits32(s1))
        else:
            print("    row1 REPORT (FAST): host formula " + bits32(s1))
    # two-way tie in the min over clusters: (0,0),(2,0) cluster 0; (1,1)
    # cluster 1; (1,-1) cluster 2 -> for row 0: b_1 = b_2 = sqrt(2)
    var x2 = List[Float32]()
    var pts = List[Float32]()
    pts.append(0.0)
    pts.append(0.0)
    pts.append(2.0)
    pts.append(0.0)
    pts.append(1.0)
    pts.append(1.0)
    pts.append(1.0)
    pts.append(-1.0)
    for i in range(8):
        x2.append(pts[i])
    var y2 = List[Int32]()
    y2.append(0)
    y2.append(0)
    y2.append(1)
    y2.append(2)
    var g2 = _run(ctx, x2, y2, 4, 2, 3)
    var w2 = _oracle_silhouette(x2, y2, 4, 2, 3)
    _ = _compare_samples("samples(min tie)", g2[0], w2[0], 4, IDENTICAL)
    print("    row0 with b_1 == b_2 == sqrt2: " + bits32(g2[0][0]) + " (a = 2 > b: (b-a)/a)")
    # ROW 39: the `a == b == 0` case (ALL points identical, two clusters),
    # in BOTH cluster orders. sklearn's spelling is `(b - a) / max(a, b)` =
    # `0 / 0` = NaN -> nan_to_num -> 0; RAFT's (ours) takes the tie branch
    # BEFORE any division, so no NaN is computed and no `max(+0, -0)` is
    # asked: the recorded bits are `0x00000000` on every vendor, not -0.0,
    # not a NaN payload. `-0.0` as an operand is unreachable (a and b are
    # +0.0-seeded sums of nonnegative terms), which is why the fixture is
    # the tie and not a planted negative zero. Asserted in BOTH modes: a
    # branch on equal inputs, not a rounding.
    var x3 = List[Float32]()
    for _ in range(6):
        x3.append(Float32(0.75))
        x3.append(Float32(-2.5))
    var y3a = List[Int32]()
    var y3b = List[Int32]()
    for i in range(6):
        y3a.append(Int32(0) if i < 3 else Int32(1))
        y3b.append(Int32(1) if i < 3 else Int32(0))
    var g3a = _run(ctx, x3, y3a, 6, 2, 2)
    var w3a = _oracle_silhouette(x3, y3a, 6, 2, 2)
    var g3b = _run(ctx, x3, y3b, 6, 2, 2)
    var w3b = _oracle_silhouette(x3, y3b, 6, 2, 2)
    _ = _compare_samples("samples(all identical, labels 0|1)", g3a[0], w3a[0], 6, True)
    _ = _compare_samples("samples(all identical, labels 1|0)", g3b[0], w3b[0], 6, True)
    var bad3 = 0
    for i in range(6):
        if bits32(g3a[0][i]) != "0x00000000" or bits32(g3b[0][i]) != "0x00000000":
            bad3 += 1
    print(
        "    all-identical points (a == b == 0) both orders: "
        + bits32(g3a[0][0])
        + " / "
        + bits32(g3b[0][0])
        + ", mean "
        + bits32(g3a[1])
        + " / "
        + bits32(g3b[1])
    )
    if bad3 != 0 or bits32(g3a[1]) != "0x00000000" or bits32(g3b[1]) != "0x00000000":
        raise Error("a == b == 0 must record +0.0 (0x00000000) in both cluster orders")
    if _count_neg_zero(got[0], 3) + _count_neg_zero(g2[0], 4) != 0:
        raise Error("a -0.0 score was recorded on a tie fixture")
    print("  OK")


def check_silhouette_inf_distances(ctx: DeviceContext) raises:
    """DEVIATION 656's gate. An OVERFLOW-SCALE but finite X, 9 rows, 4
    clusters:
      c0 = {row 0 (0, 0), row 1 (3e38, 0)}: `(3e38 - 0)^2` is +inf, so
           d(0,1) = +inf and a(0) = a(1) = +inf;
      c1 = rows 2-4 near the origin; c3 = rows 7-8 near (10, 10);
      c2 = {row 5 (-3e38, 1e18), row 6 (-3e38, -1e18)}: d(5,6) = 2e18 is
           FINITE, every other distance from them is +inf.
    Row 0: a = inf, b = min(c1 finite, c2 inf, c3 finite) finite -> `(b -
    a) / a = -inf / inf` = NaN -> +0.0 (THE NaN ARM, finite b). Row 1: a =
    inf and every other cluster's mean is +inf, which never wins the min
    seeded FLT_MAX, so b = FLT_MAX -> the same arm with b = FLT_MAX -> +0.0.
    Rows 5, 6: a = 2e18, every other cluster's mean is +inf -> b = FLT_MAX
    -> `(FLT_MAX - a) / FLT_MAX` = exactly 1.0 (0x3f800000): the `b = inf`
    arm is UNREACHABLE (b <= FLT_MAX always), and the score is finite and
    defined (RAFT's). Rows 2-4 and 7-8: a and b finite -> ORDINARY scores
    (reach: the fixture is not all zeros). Asserted in BOTH modes (inf
    arithmetic, an exact division and the guard's compare are IEEE on every
    vendor); bitwise vs the oracle under IDENTICAL."""
    print("check_silhouette_inf_distances [" + _mode_name() + "]")
    var x = List[Float32]()
    var y = List[Int32]()
    x.append(0.0); x.append(0.0); y.append(0)          # row 0
    x.append(3.0e38); x.append(0.0); y.append(0)       # row 1
    x.append(1.0); x.append(0.5); y.append(1)          # row 2
    x.append(-1.0); x.append(0.25); y.append(1)        # row 3
    x.append(0.5); x.append(-1.5); y.append(1)         # row 4
    x.append(-3.0e38); x.append(1.0e18); y.append(2)   # row 5
    x.append(-3.0e38); x.append(-1.0e18); y.append(2)  # row 6
    x.append(10.0); x.append(10.5); y.append(3)        # row 7
    x.append(9.5); x.append(10.0); y.append(3)         # row 8
    var n = 9
    var got = _run(ctx, x, y, n, 2, 4)
    var want = _oracle_silhouette(x, y, n, 2, 4)
    _ = _compare_samples("samples(inf distances)", got[0], want[0], n, IDENTICAL)
    var line = String("    scores:")
    for i in range(n):
        line += " " + bits32(got[0][i])
    print(line + "  mean " + bits32(got[1]))
    # the NaN arm (a = inf): row 0 (b finite) and row 1 (b = FLT_MAX)
    for r in range(2):
        if bits32(got[0][r]) != "0x00000000":
            raise Error(
                "row " + String(r) + " (the NaN arm of SilOp) must record +0.0, got "
                + bits32(got[0][r])
            )
    # b = FLT_MAX with a finite a: exactly 1.0 (the `b = inf` arm is unreachable)
    for r in range(5, 7):
        if bits32(got[0][r]) != "0x3f800000":
            raise Error(
                "row " + String(r) + " (a finite, b = FLT_MAX) must be exactly 1.0, got "
                + bits32(got[0][r])
            )
    # REACH of both arms through the oracle's own distances
    var d01 = host_l2sqrt_unexpanded(x, 0, 1, 2)
    var d50 = host_l2sqrt_unexpanded(x, 5, 0, 2)
    var d52 = host_l2sqrt_unexpanded(x, 5, 2, 2)
    var d56 = host_l2sqrt_unexpanded(x, 5, 6, 2)
    print(
        "    d(0,1) " + bits32(d01) + " d(5,0) " + bits32(d50) + " d(5,2) " + bits32(d52)
        + " d(5,6) " + bits32(d56)
    )
    if bits32(d01) != "0x7f800000" or bits32(d50) != "0x7f800000" or bits32(d52) != "0x7f800000":
        raise Error("fixture does not reach the inf distances; a NaN arm is unreached")
    if bits32(d56) == "0x7f800000":
        raise Error("d(5,6) must be finite (rows 5, 6 are the finite-a, b = FLT_MAX rows)")
    # the ordinary rows: finite, nonzero, in (-1, 1)
    var ord_rows = List[Int]()
    ord_rows.append(2); ord_rows.append(3); ord_rows.append(4); ord_rows.append(7); ord_rows.append(8)
    for q in range(5):
        var v = got[0][ord_rows[q]]
        if not (v > Float32(-1.0) and v < Float32(1.0)) or v == Float32(0.0):
            raise Error("row " + String(ord_rows[q]) + " must be an ordinary finite score, got " + bits32(v))
    if got[1] != got[1]:
        raise Error("the mean must not be NaN")
    if _count_neg_zero(got[0], n) != 0:
        raise Error("a -0.0 score was recorded on the inf fixture")
    print("    NaN arm rows 0 (b finite), 1 (b = FLT_MAX) -> +0.0; rows 5,6 (b = FLT_MAX) = 1.0; rows 2-4, 7-8 ordinary")
    print("  OK")


def check_silhouette_refusals(ctx: DeviceContext) raises:
    print("check_silhouette_refusals [" + _mode_name() + "]")
    var f = hashed_points(64, D, 3, 9)
    var dx = upload_f32(ctx, f[0])
    var dy = upload_i32(ctx, f[1])
    var ds = ctx.enqueue_create_buffer[DType.float32](64)
    var refused = 0
    try:
        _ = silhouette_score(ctx, dx, 64, D, dy, 1, ds)
    except e:
        print("    n_labels=1: " + String(e))
        refused += 1
    try:
        _ = silhouette_score(ctx, dx, 64, D, dy, 64, ds)
    except e:
        print("    n_labels=64=n_rows: " + String(e))
        refused += 1
    try:
        _ = silhouette_score(ctx, dx, 64, D, dy, 3, ds, 40000, 1)
    except e:
        print("    metric=1 (L2SqrtExpanded): " + String(e))
        refused += 1
    try:
        _ = silhouette_score(ctx, dx, 64, D, dy, 3, ds, 0)
    except e:
        print("    chunk=0: " + String(e))
        refused += 1
    if refused != 4:
        raise Error("expected 4 refusals by name, got " + String(refused))
    # chunk is scheduling: three values, one byte pattern
    var m1 = silhouette_score(ctx, dx, 64, D, dy, 3, ds, 1)
    var s1 = download_f32(ctx, ds, 64)
    var m2 = silhouette_score(ctx, dx, 64, D, dy, 3, ds, 40000)
    var s2 = download_f32(ctx, ds, 64)
    var m3 = silhouette_score(ctx, dx, 64, D, dy, 3, ds, 7)
    var s3 = download_f32(ctx, ds, 64)
    var moved = 0
    for i in range(64):
        if bits32(s1[i]) != bits32(s2[i]) or bits32(s1[i]) != bits32(s3[i]):
            moved += 1
    if moved != 0 or bits32(m1) != bits32(m2) or bits32(m1) != bits32(m3):
        raise Error("chunk moved bytes; it must be scheduling only")
    print("    chunk 1 / 40000 / 7: one byte pattern, mean " + bits32(m1))
    print("  OK")


def _launch_variants(
    ctx: DeviceContext,
    x: List[Float32],
    y: List[Int32],
    pad_rows: Int,
) raises -> Tuple[List[List[Float32]], List[Float32]]:
    var xp = x.copy()
    var yp = y.copy()
    for _ in range(pad_rows):
        for _ in range(D):
            xp.append(_nan32())
        yp.append(Int32(0))
    var dx = upload_f32(ctx, xp)
    var dy = upload_i32(ctx, yp)
    var ds = ctx.enqueue_create_buffer[DType.float32](N + pad_rows)
    ctx.enqueue_memset(ds, _nan32())
    var samples = List[List[Float32]]()
    var means = List[Float32]()
    means.append(silhouette_score_launch[256](ctx, dx, N, D, dy, K, ds, 40000, DISTANCE_L2_SQRT_UNEXPANDED, 0))
    samples.append(download_f32(ctx, ds, N))
    ctx.enqueue_memset(ds, _nan32())
    means.append(silhouette_score_launch[64](ctx, dx, N, D, dy, K, ds, 40000, DISTANCE_L2_SQRT_UNEXPANDED, 0))
    samples.append(download_f32(ctx, ds, N))
    ctx.enqueue_memset(ds, _nan32())
    means.append(silhouette_score_launch[256](ctx, dx, N, D, dy, K, ds, 40000, DISTANCE_L2_SQRT_UNEXPANDED, 5))
    samples.append(download_f32(ctx, ds, N))
    ctx.enqueue_memset(ds, _nan32())
    means.append(silhouette_score_launch[64](ctx, dx, N, D, dy, K, ds, 40000, DISTANCE_L2_SQRT_UNEXPANDED, 5))
    samples.append(download_f32(ctx, ds, N))
    return (samples^, means^)


def check_silhouette_launch_invariant(ctx: DeviceContext) raises:
    print("check_silhouette_launch_invariant [" + _mode_name() + "]")
    var f = hashed_points(N, D, K, 3)
    var a = _launch_variants(ctx, f[0], f[1], 0)
    var b = _launch_variants(ctx, f[0], f[1], 37)
    var names = List[String]()
    names.append("b256 g1d pad0")
    names.append("b64  g1d pad0")
    names.append("b256 g2d pad0")
    names.append("b64  g2d pad0")
    names.append("b256 g1d pad37")
    names.append("b64  g1d pad37")
    names.append("b256 g2d pad37")
    names.append("b64  g2d pad37")
    var moved_launches = 0
    for v in range(8):
        var samples = a[0][v].copy() if v < 4 else b[0][v - 4].copy()
        var mean = a[1][v] if v < 4 else b[1][v - 4]
        var cells_moved = 0
        for i in range(N):
            if bits32(samples[i]) != bits32(a[0][0][i]):
                cells_moved += 1
        var mean_moved = bits32(mean) != bits32(a[1][0])
        print(
            "    "
            + names[v]
            + " mean "
            + bits32(mean)
            + " cells moved "
            + String(cells_moved)
        )
        if cells_moved != 0 or mean_moved:
            moved_launches += 1
    comptime if IDENTICAL:
        if moved_launches != 0:
            raise Error(
                "silhouette bytes moved across launches: "
                + String(moved_launches)
                + " of 8"
            )
        print("    8 launches, " + String(N) + " cells + mean, one byte pattern")
    else:
        print("    FAST: " + String(moved_launches) + " of 8 launches differ (a REPORT)")
    print("  OK")


def main() raises:
    print("== metrics/mojo_only/silhouette_check.mojo [" + _mode_name() + "] ==")
    var ctx = DeviceContext()
    check_silhouette_matches_oracle(ctx)
    check_silhouette_singleton_and_empty(ctx)
    check_silhouette_planted_ties(ctx)
    check_silhouette_inf_distances(ctx)
    check_silhouette_refusals(ctx)
    check_silhouette_launch_invariant(ctx)
    print("ALL GROUP C CHECKS PASSED [" + _mode_name() + "]")
