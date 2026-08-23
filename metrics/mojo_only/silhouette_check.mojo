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
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime N = 1031
comptime D = 5
comptime K = 6


def _mode_name() -> String:
    comptime if IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")


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
    check_silhouette_refusals(ctx)
    check_silhouette_launch_invariant(ctx)
    print("ALL GROUP C CHECKS PASSED [" + _mode_name() + "]")
