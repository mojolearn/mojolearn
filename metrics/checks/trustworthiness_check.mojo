# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Group D gates: trustworthiness.

    pixi run mojo run -I . metrics/checks/trustworthiness_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . metrics/checks/trustworthiness_check.mojo

DEVIATION 655. The metric is an INTEGER rank sum over two neighbor
structures plus one Float64 closed form. The gates:

    check_trust_rank_sum_exact        523 hashed points (m = 6), embedded
                                      into d = 2 with a hashed perturbation:
                                      the device rank sum EQUALS a host count
                                      (Float32 distances through the same
                                      helpers, the same tie-break) given the
                                      same embedded neighbors; the embedded
                                      neighbor SETS from knn_search equal a
                                      Float64 host k-NN on every row
                                      (reported; asserted 0 rows differ);
                                      t vs sklearn's Float64 formula from
                                      the host structures BITWISE
    check_trust_perfect_and_scrambled perfect embedding (X_embedded == X)
                                      scores 1.0 exactly (rank sum 0); a
                                      hashed row permutation as the
                                      embedding scores well below 1
                                      (REACH: the sum moves)
    check_trust_planted_duplicates    rows 10 and 11 identical in X and in
                                      X_embedded (exact distance ties): the
                                      stable-sort tie-break (lower index
                                      first) is what the count reproduces;
                                      host count vs device EXACT; the
                                      flipped tie-break (`j <= e`) gives a
                                      DIFFERENT sum here (teeth)
    check_trust_refusals              n_neighbors 0, n_neighbors = n,
                                      2 * n_neighbors >= n (cuML .pyx:114),
                                      n_neighbors + 1 > TRUST_MAX_K,
                                      batchSize 0 RAISE by name
    check_trust_launch_invariant      block 64 / 256, grid 1-D / 2-D: the
                                      same integer (trivially, and run)

ROW 39 AUDIT (2026-08-23): the device-vs-host and neighbor-set claims
are vendor-shaped under FAST (the embedded k-NN is a vendor matmul path,
the rank distances the vendor's sqrt) and go through `_gate`: asserted
under IDENTICAL, `RECORDED [FAST]` under FAST. Refusals now 5 (2 *
n_neighbors >= n, cuML trustworthiness.pyx:114).

SABOTAGES PERFORMED (2026-08-23), each reverted; outputs in the README:
    (j) the device tie-break flipped to `j <= ei`: check_trust_rank_sum_
        exact fails EXACTLY (293696 vs 291291: every neighbor now counts
        itself), and the planted-duplicate fixture separates the DISTINCT-
        row tie too (host 189301 vs flipped 191658)
"""

from std.math import sqrt
from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace

from metrics.checks.fixtures import bits64, hashed_points, u01, splitmix
from metrics.checks.pinned_distance import host_l2sqrt_unexpanded
from metrics.impl.metrics.trustworthiness import trustworthiness_score
from metrics.impl.stats.detail.trustworthiness_score import (
    TRUST_MAX_K,
    trustworthiness_rank_sum,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime N = 523
comptime M = 6
comptime DE = 2
comptime K = 5


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


def _fixture(salt: Int) -> Tuple[List[Float32], List[Float32]]:
    var f = hashed_points(N, M, 4, salt)
    var x = f[0].copy()
    var emb = List[Float32]()
    for i in range(N):
        for q in range(DE):
            # the first two coordinates plus a hashed perturbation
            emb.append(x[i * M + q] + Float32((u01(i, q, salt + 9) - 0.5) * 0.8))
    return (x^, emb^)


def _host_rank_sum(
    x: List[Float32],
    emb_ind: List[UInt32],
    n: Int,
    m: Int,
    k1: Int,
    flip_tie: Bool,
) -> Int64:
    """The host count of DEVIATION 655, Float32 distances through the same
    helper, `j < e` tie-break (or `j <= e` for the teeth test)."""
    var total = Int64(0)
    for i in range(n):
        var dist = List[Float32]()
        for j in range(n):
            dist.append(host_l2sqrt_unexpanded(x, i, j, m))
        for q in range(k1):
            var e = Int(emb_ind[i * k1 + q])
            var de = dist[e]
            var r = 0
            for j in range(n):
                var dj = dist[j]
                if dj < de or (dj == de and ((j <= e) if flip_tie else (j < e))):
                    r += 1
            var tmp = r - k1 + 1
            if tmp > 0:
                total += Int64(tmp)
    return total


def _host_knn_f64(emb: List[Float32], n: Int, d: Int, k1: Int) -> List[Int]:
    """Float64 exact k+1 nearest (including self), ties by index."""
    var out = List[Int]()
    for i in range(n):
        var dist = List[Float64]()
        for j in range(n):
            var acc = 0.0
            for q in range(d):
                var diff = Float64(emb[i * d + q]) - Float64(emb[j * d + q])
                acc += diff * diff
            dist.append(acc)
        var taken = List[Bool]()
        for _ in range(n):
            taken.append(False)
        for _ in range(k1):
            var best = -1
            var bd = 1.7976931348623157e308
            for j in range(n):
                if not taken[j] and dist[j] < bd:
                    bd = dist[j]
                    best = j
            taken[best] = True
            out.append(best)
    return out^


def _sets_differ(a: List[UInt32], b: List[Int], n: Int, k1: Int) -> Int:
    var rows_differ = 0
    for i in range(n):
        var miss = 0
        for q in range(k1):
            var v = Int(a[i * k1 + q])
            var found = False
            for p in range(k1):
                if b[i * k1 + p] == v:
                    found = True
            if not found:
                miss += 1
        if miss > 0:
            rows_differ += 1
    return rows_differ


def _gate(ok: Bool, what: String) raises:
    """FACT 3 of the row-39 audit: the device-vs-host comparisons below
    ride on (a) neighbors' `knn_search`, a vendor matmul path under FAST
    (expanded distances), and (b) the rank compare on distances whose
    FAST spelling is the vendor's `sqrt` (approximate on NVIDIA, DEVIATION
    258). Both are vendor-shaped under FAST: asserted under IDENTICAL,
    RECORDED under FAST. Integer-only and host-only claims (refusals, the
    flipped tie-break's teeth, coarse ranges) stay asserted in both."""
    if ok:
        return
    comptime if IDENTICAL:
        raise Error(what)
    else:
        print("    RECORDED [FAST] " + what + " (vendor-shaped under FAST; not asserted)")


def _closed_form(t: Int64, n: Int, k: Int) -> Float64:
    var nn = Float64(n)
    var kk = Float64(k)
    return 1.0 - ((2.0 / ((nn * kk) * ((2.0 * nn) - (3.0 * kk) - 1.0))) * Float64(t))


def check_trust_rank_sum_exact(ctx: DeviceContext) raises:
    var trace = IdentityTrace.disabled()
    print("check_trust_rank_sum_exact [" + _mode_name() + "]")
    var f = _fixture(3)
    var x = f[0].copy()
    var emb = f[1].copy()
    var r = trustworthiness_rank_sum(ctx, trace, x, emb, N, M, DE, K, False, 0)
    var host = _host_rank_sum(x, r[1], N, M, K + 1, False)
    print("    rank sum device " + String(r[0]) + " host " + String(host))
    _gate(r[0] == host, "rank sum differs")
    var hk = _host_knn_f64(emb, N, DE, K + 1)
    var rows = _sets_differ(r[1], hk, N, K + 1)
    print("    embedded neighbor sets vs Float64 host k-NN: " + String(rows) + " rows differ")
    _gate(rows == 0, "knn_search neighbor sets differ from the Float64 host k-NN")
    var t = trustworthiness_score(ctx, x, emb, N, M, DE, K)
    var want = _closed_form(host, N, K)
    print("    t " + String(t) + " (" + bits64(t) + ") sklearn formula " + bits64(want))
    _gate(bits64(t) == bits64(want), "t differs from the closed form of the host sum")
    if not (t > 0.5 and t < 1.0):
        raise Error("t out of the expected range for this embedding")
    print("  OK")


def check_trust_perfect_and_scrambled(ctx: DeviceContext) raises:
    var trace = IdentityTrace.disabled()
    print("check_trust_perfect_and_scrambled [" + _mode_name() + "]")
    var f = _fixture(5)
    var x = f[0].copy()
    var same = x.copy()
    var r = trustworthiness_rank_sum(ctx, trace, x, same, N, M, M, K, False, 0)
    var t1 = trustworthiness_score(ctx, x, same, N, M, M, K)
    print("    perfect embedding: rank sum " + String(r[0]) + " t " + String(t1))
    # exact 1.0 needs the embedded k-NN (expanded L2, vendor matmul under
    # FAST) and the unexpanded rank distances to agree on every neighbor:
    # IDENTICAL asserts, FAST records (FACT 3)
    _gate(r[0] == Int64(0) and t1 == 1.0, "a perfect embedding must score exactly 1.0")
    # scrambled: a hashed permutation of the rows as the embedding
    var perm = List[Int]()
    for i in range(N):
        perm.append(i)
    for i in range(N - 1, 0, -1):
        var j = Int(splitmix(i, 3, 77) % UInt64(i + 1))
        var tmp = perm[i]
        perm[i] = perm[j]
        perm[j] = tmp
    var scr = List[Float32]()
    for i in range(N):
        for q in range(M):
            scr.append(x[perm[i] * M + q])
    var t2 = trustworthiness_score(ctx, x, scr, N, M, M, K)
    print("    scrambled embedding: t " + String(t2))
    if not (t2 < 0.8):
        raise Error("a scrambled embedding must score well below 1")
    print("  OK")


def check_trust_planted_duplicates(ctx: DeviceContext) raises:
    var trace = IdentityTrace.disabled()
    print("check_trust_planted_duplicates [" + _mode_name() + "]")
    var f = _fixture(11)
    var x = f[0].copy()
    var emb = f[1].copy()
    for q in range(M):
        x[11 * M + q] = x[10 * M + q]
    for q in range(DE):
        emb[11 * DE + q] = emb[10 * DE + q]
    var r = trustworthiness_rank_sum(ctx, trace, x, emb, N, M, DE, K, False, 0)
    var host = _host_rank_sum(x, r[1], N, M, K + 1, False)
    var flipped = _host_rank_sum(x, r[1], N, M, K + 1, True)
    print(
        "    rank sum device "
        + String(r[0])
        + " host(j < e) "
        + String(host)
        + " host(j <= e) "
        + String(flipped)
    )
    _gate(r[0] == host, "rank sum differs on the duplicate fixture")
    if flipped == host:
        raise Error("the flipped tie-break does not move the sum; no teeth")
    print("  OK")


def check_trust_refusals(ctx: DeviceContext) raises:
    print("check_trust_refusals [" + _mode_name() + "]")
    var f = _fixture(13)
    var x = f[0].copy()
    var emb = f[1].copy()
    var refused = 0
    try:
        _ = trustworthiness_score(ctx, x, emb, N, M, DE, 0)
    except e:
        print("    n_neighbors=0: " + String(e))
        refused += 1
    try:
        _ = trustworthiness_score(ctx, x, emb, N, M, DE, N)
    except e:
        print("    n_neighbors=n: " + String(e))
        refused += 1
    # cuML trustworthiness.pyx:114: 2 * n_neighbors >= n_samples is refused
    # (the closed form's denominator would reach 0 at 2n = 3k + 1; row 39)
    try:
        _ = trustworthiness_score(ctx, x, emb, N, M, DE, (N + 1) // 2)
    except e:
        print("    2 * n_neighbors >= n: " + String(e))
        refused += 1
    try:
        _ = trustworthiness_score(ctx, x, emb, N, M, DE, TRUST_MAX_K)
    except e:
        print("    n_neighbors=TRUST_MAX_K: " + String(e))
        refused += 1
    try:
        _ = trustworthiness_score(ctx, x, emb, N, M, DE, K, 0)
    except e:
        print("    batchSize=0: " + String(e))
        refused += 1
    if refused != 5:
        raise Error("expected 5 refusals, got " + String(refused))
    print("  OK")


def check_trust_launch_invariant(ctx: DeviceContext) raises:
    var trace = IdentityTrace.disabled()
    print("check_trust_launch_invariant [" + _mode_name() + "]")
    var f = _fixture(3)
    var x = f[0].copy()
    var emb = f[1].copy()
    var a = trustworthiness_rank_sum(ctx, trace, x, emb, N, M, DE, K, False, 0)
    var b = trustworthiness_rank_sum(ctx, trace, x, emb, N, M, DE, K, True, 0)
    var c = trustworthiness_rank_sum(ctx, trace, x, emb, N, M, DE, K, False, 7)
    var d = trustworthiness_rank_sum(ctx, trace, x, emb, N, M, DE, K, True, 7)
    print(
        "    b256 g1d "
        + String(a[0])
        + " b64 g1d "
        + String(b[0])
        + " b256 g2d "
        + String(c[0])
        + " b64 g2d "
        + String(d[0])
    )
    # the rank kernel itself is one thread per (row, neighbor) with no fold,
    # but each call re-runs the embedded k-NN (vendor matmul under FAST):
    # IDENTICAL asserts, FAST records (FACT 3)
    _gate(a[0] == b[0] and a[0] == c[0] and a[0] == d[0], "rank sum moved across launches")
    print("  OK")


def main() raises:
    print("== metrics/checks/trustworthiness_check.mojo [" + _mode_name() + "] ==")
    var ctx = DeviceContext()
    check_trust_rank_sum_exact(ctx)
    check_trust_perfect_and_scrambled(ctx)
    check_trust_planted_duplicates(ctx)
    check_trust_refusals(ctx)
    check_trust_launch_invariant(ctx)
    print("ALL GROUP D CHECKS PASSED [" + _mode_name() + "]")
