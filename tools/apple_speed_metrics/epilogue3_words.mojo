# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# lane metrics-apple3: the host epilogues the lane changed, word for word
# against their sequential definitions, at 1, 2, 3 and 8 host tasks
# (MOJOLEARN_CPU_THREADS): (1) row_sum_range == the largest and smallest
# `fsum` of each row; (2) expected_mi == fsum of every cell's terms in cell
# order; (3) encode_small_i64 == the ascending distinct labels and each
# row's rank; (4) first_rows_i32 and scatter_rows.
from std.memory import bitcast
from std.os import setenv
from x_metrics.epilogue import (
    fsum, row_sum_range, row_sum_range_tasks, expected_mi, expected_mi_tasks, _emi_cell, encode_small_i64,
    first_rows_i32, scatter_rows,
)


def u(i: Int, s: Int) -> Float64:
    var z = UInt64(i) * 0x9E3779B97F4A7C15 + UInt64(s) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) / Float64(1 << 53)


def bits(x: Float64) -> UInt64:
    return bitcast[DType.uint64](x)


def threads(t: Int) raises:
    _ = setenv("MOJOLEARN_CPU_THREADS", String(t), True)


def scores(n: Int, k: Int, kind: Int, seed: Int) -> List[Float32]:
    """kind 0: uniform rows scaled to sum near 1; 1: magnitudes spread over
    about 2^-90 .. 1 (the binary64 row sum is not exact); 2: eighths (every
    sum exact); 3: signed values with zeros and -0.0."""
    var out = List[Float32](capacity=n * k)
    for r in range(n):
        for c in range(k):
            var v = u(r * k + c, seed)
            if kind == 0:
                out.append(Float32(v / Float64(k) * 2.0))
            elif kind == 1:
                var e = Int(u(r * k + c, seed + 1) * 90.0)
                var x = v
                for _ in range(e):
                    x = x * 0.5
                out.append(Float32(x))
            elif kind == 2:
                out.append(Float32(Float64(Int(v * 8.0)) / 8.0))
            else:
                var w = u(r * k + c, seed + 2)
                if w < 0.2:
                    out.append(Float32(0.0))
                elif w < 0.3:
                    out.append(-Float32(0.0))
                elif w < 0.65:
                    out.append(Float32(v))
                else:
                    out.append(-Float32(v * 3.0))
    return out^


def row_sums(n: Int, k: Int, kind: Int) raises -> Int:
    var s = scores(n, k, kind, 100 + kind)
    var row = List[Float64](length=k, fill=0.0)
    var hi: Float64 = 0.0
    var lo: Float64 = 0.0
    var inexact = 0
    for r in range(n):
        var naive: Float64 = 0.0
        for c in range(k):
            row[c] = Float64(s[r * k + c])
            naive = naive + row[c]
        var v = fsum(row)
        if bits(v) != bits(naive):
            inexact += 1
        if r == 0 or v > hi:
            hi = v
        if r == 0 or v < lo:
            lo = v
    var bad = 0
    var ts: List[Int] = [1, 2, 3, 8]
    for t in ts:
        threads(t)
        var got = List[Float64](length=2, fill=-1.0)
        row_sum_range_tasks(Int(s.unsafe_ptr()), n, k, Int(got.unsafe_ptr()))
        var old = List[Float64](length=2, fill=-1.0)
        row_sum_range(Int(s.unsafe_ptr()), n, k, Int(old.unsafe_ptr()))
        if bits(old[0]) != bits(hi) or bits(old[1]) != bits(lo):
            bad += 1
        if bits(got[0]) != bits(hi) or bits(got[1]) != bits(lo):
            bad += 1
    _ = len(s)
    print("ROWSUM n", n, "k", k, "kind", kind, "rows whose running sum is not fsum", inexact, "arms differ", bad)
    return bad


def counts_of(n: Int, m: Int, seed: Int) -> List[Int64]:
    """m positive class counts that add up to n."""
    var w = List[Float64](capacity=m)
    var total: Float64 = 0.0
    for i in range(m):
        var v = 0.05 + u(i, seed)
        w.append(v)
        total = total + v
    var out = List[Int64](capacity=m)
    var left = n
    for i in range(m):
        var c = Int(Float64(n) * w[i] / total)
        if c < 1:
            c = 1
        if i == m - 1 or c > left - (m - 1 - i):
            c = left - (m - 1 - i)
        out.append(Int64(c))
        left -= c
    return out^


def emi(n: Int, na: Int, nb: Int, seed: Int) raises -> Int:
    var a = counts_of(n, na, seed)
    var b = counts_of(n, nb, seed + 1)
    var terms = List[Float64]()
    for i in range(na):
        for j in range(nb):
            _emi_cell(Int(a[i]), Int(b[j]), n, terms)
    var want = fsum(terms)
    var bad = 0
    var ts: List[Int] = [1, 2, 3, 8]
    for t in ts:
        threads(t)
        var got = expected_mi_tasks(Int(a.unsafe_ptr()), na, Int(b.unsafe_ptr()), nb, n)
        if bits(got) != bits(want):
            bad += 1
        var old = expected_mi(Int(a.unsafe_ptr()), na, Int(b.unsafe_ptr()), nb, n)
        if bits(old) != bits(want):
            bad += 1
    _ = len(a)
    _ = len(b)
    print("EMI n", n, "na", na, "nb", nb, "terms", len(terms), "arms differ", bad)
    if len(terms) < 1:
        raise Error("VACUOUS: no expected MI term")
    return bad


def encode(n: Int, kind: Int) raises -> Int:
    """kind 0: {0, 1}; 1: five labels from -2; 2: sparse labels inside a
    span of 60000; 3: a span too wide (-3); 4: more classes than the
    caller's bound (-1)."""
    var src = List[Int64](capacity=n)
    for r in range(n):
        var v = u(r, 300 + kind)
        if kind == 0:
            src.append(Int64(1 if v < 0.3 else 0))
        elif kind == 1:
            src.append(Int64(Int(v * 5.0) - 2))
        elif kind == 2:
            src.append(Int64(Int(v * 40.0) * 1499 - 7))
        elif kind == 3:
            src.append(Int64(Int(v * 3.0) * 70000))
        else:
            src.append(Int64(Int(v * 300.0)))
    var cap = 64
    # the definition: ascending distinct labels by insertion, each row's rank
    var classes = List[Int64]()
    var over = False
    for r in range(n):
        var v = src[r]
        var at = 0
        while at < len(classes) and classes[at] < v:
            at += 1
        if at < len(classes) and classes[at] == v:
            continue
        classes.append(v)
        var j = len(classes) - 1
        while j > at:
            classes[j] = classes[j - 1]
            j -= 1
        classes[at] = v
    if len(classes) > cap:
        over = True
    var wide = Int(classes[len(classes) - 1]) - Int(classes[0]) >= 65536
    var bad = 0
    var ts: List[Int] = [1, 2, 3, 8]
    for t in ts:
        threads(t)
        var cls = List[Int64](length=cap, fill=Int64(-99))
        var codes = List[Int32](length=n, fill=Int32(-99))
        var k = encode_small_i64(Int(src.unsafe_ptr()), n, Int(cls.unsafe_ptr()), cap, Int(codes.unsafe_ptr()))
        if wide:
            if k != -3:
                bad += 1
            continue
        if over:
            if k != -1:
                bad += 1
            continue
        if k != len(classes):
            bad += 1
            continue
        for c in range(k):
            if cls[c] != classes[c]:
                bad += 1
        for r in range(n):
            var c = Int(codes[r])
            if c < 0 or c >= k or classes[c] != src[r]:
                bad += 1
    _ = len(src)
    print("ENCODE n", n, "kind", kind, "classes", len(classes), "wide", wide, "over", over, "arms differ", bad)
    return bad


def rows_and_scatter(n: Int, k: Int) raises -> Int:
    var bad = 0
    var codes = List[Int32](capacity=n)
    for r in range(n):
        codes.append(Int32(Int(u(r, 400) * Float64(k))))
    var want = List[Int64](length=k, fill=Int64(-1))
    for r in range(n):
        var c = Int(codes[r])
        if Int(want[c]) < 0:
            want[c] = Int64(r)
    var got = List[Int64](length=k, fill=Int64(-7))
    first_rows_i32(Int(codes.unsafe_ptr()), n, k, Int(got.unsafe_ptr()))
    for c in range(k):
        if got[c] != want[c]:
            bad += 1
    # a permutation of the rows: dst row idx[j] = src row j, 12 bytes a row
    var idx = List[Int64](capacity=n)
    for j in range(n):
        idx.append(Int64((j * 7919 + 13) % n))
    var seen = List[Int](length=n, fill=0)
    for j in range(n):
        seen[Int(idx[j])] += 1
    var perm = True
    for j in range(n):
        if seen[j] != 1:
            perm = False
    var srcw = List[Float32](capacity=3 * n)
    for j in range(3 * n):
        srcw.append(Float32(u(j, 401)))
    var dst = List[Float32](length=3 * n, fill=Float32(-5.0))
    scatter_rows(Int(srcw.unsafe_ptr()), Int(dst.unsafe_ptr()), Int(idx.unsafe_ptr()), n, n, 12)
    if perm:
        for j in range(n):
            for c in range(3):
                if bitcast[DType.uint32](dst[3 * Int(idx[j]) + c]) != bitcast[DType.uint32](srcw[3 * j + c]):
                    bad += 1
    idx[n // 2] = Int64(n)
    var refused = False
    try:
        scatter_rows(Int(srcw.unsafe_ptr()), Int(dst.unsafe_ptr()), Int(idx.unsafe_ptr()), n, n, 12)
    except:
        refused = True
    if not refused:
        bad += 1
    _ = len(codes)
    _ = len(idx)
    _ = len(srcw)
    print("ROWS n", n, "k", k, "permutation", perm, "out of range refused", refused, "differ", bad)
    if not perm:
        raise Error("VACUOUS: the scatter index is not a permutation")
    return bad


def main() raises:
    var bad = 0
    var ns: List[Int] = [1, 7, 1000, 200000]
    var ks: List[Int] = [2, 5, 10]
    for n in ns:
        for k in ks:
            for kind in range(4):
                bad += row_sums(n, k, kind)
    var sizes: List[Int] = [5000, 100003, 1000000]
    for n in sizes:
        bad += emi(n, 2, 2, 11)
        bad += emi(n, 5, 5, 12)
        bad += emi(n, 3, 20, 13)
        bad += emi(n, 40, 7, 14)
    var es: List[Int] = [1, 300, 200000]
    for n in es:
        for kind in range(5):
            bad += encode(n, kind)
    bad += rows_and_scatter(997, 5)
    bad += rows_and_scatter(200003, 40)
    threads(0)
    if bad:
        raise Error(String("FAIL epilogue3 words differ: ", bad))
    print("PASS epilogue3: row sums, expected MI, label codes, first rows and the row scatter equal their sequential definitions at 1, 2, 3 and 8 host tasks")
