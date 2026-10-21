# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed kernels of `bindings/array_helpers.mojo` (the NumPy-free layer's
layout helpers and exact sums).

The binding keeps the Python door (argument unpacking, GILReleased, the
result object); the loops live here with no Python import, word for word as
they were in the binding (lane thin-core, 2026-10-21: a pure move, no bit
change).
"""
from std.builtin.sort import sort
from std.memory import bitcast
from std.math import inf, isinf, isnan, sqrt
from checks.numerics import pinned_mul_f64
from glm.impl.lm_finish import _exact_sum

comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]
comptime _U32P = MutPointer[UInt32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


def _strided_copy[dt: DType](
    src: Int, dst: Int, dims: MutPointer[Int64, MutUntrackedOrigin], nd: Int, total: Int
):
    """Walk the index space of `shape` in C order; element (i_0..i_k) moves
    from src[src_base + sum i_j * s_j] to dst[dst_base + sum i_j * d_j]."""
    var sp = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=src)
    var dp = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=dst)
    var sbase = Int(dims[3 * nd])
    var dbase = Int(dims[3 * nd + 1])
    var inner = Int(dims[nd - 1])
    var sstep = Int(dims[2 * nd - 1])
    var dstep = Int(dims[3 * nd - 1])
    var idx = List[Int](length=nd, fill=0)
    var so = sbase
    var do_ = dbase
    var done = 0
    while done < total:
        for j in range(inner):
            dp.unsafe_store(do_ + j * dstep, sp.unsafe_load(so + j * sstep))
        done += inner
        # advance the outer mixed-radix counter (row-major)
        var k = nd - 2
        while k >= 0:
            idx[k] += 1
            so += Int(dims[nd + k])
            do_ += Int(dims[2 * nd + k])
            if idx[k] < Int(dims[k]):
                break
            so -= idx[k] * Int(dims[nd + k])
            do_ -= idx[k] * Int(dims[2 * nd + k])
            idx[k] = 0
            k -= 1


def strided_total(dp: _I64P, nd: Int) raises -> Int:
    """The element count of the shape in `dp[0:nd]`; raises on a negative
    dimension."""
    var total = 1
    for k in range(nd):
        var s = Int(dp[k])
        if s < 0:
            raise Error("strided_copy_bytes: negative dimension")
        total *= s
    return total


def strided_copy_bytes(src: Int, dst: Int, dp: _I64P, nd: Int, isz: Int, total: Int):
    """`_strided_copy` at the element width `isz` (1, 2, 4 or 8 bytes)."""
    if isz == 1:
        _strided_copy[DType.uint8](src, dst, dp, nd, total)
    elif isz == 2:
        _strided_copy[DType.uint16](src, dst, dp, nd, total)
    elif isz == 4:
        _strided_copy[DType.uint32](src, dst, dp, nd, total)
    else:
        _strided_copy[DType.uint64](src, dst, dp, nd, total)


def check_lengths_i64(lp: _I64P, n: Int, hi: Int64) -> Int:
    """The first i whose length is outside [1, hi], or -1."""
    for i in range(n):
        var v = lp.unsafe_load(i)
        if v < 1 or v > hi:
            return i
    return -1


def ragged_rows_bytes(dp: _U8P, sp: _U8P, lp: _I64P, n: Int, row: Int, pos: Int, m: Int):
    """The per-row ragged passes of `ragged_rows_bytes_binding` (modes 0, 1, 2)."""
    if m == 0:
        for i in range(n):
            var keep = Int(lp.unsafe_load(i)) * pos
            var base = i * row
            for k in range(keep):
                dp.unsafe_store(base + k, sp.unsafe_load(base + k))
            for k in range(keep, row):
                dp.unsafe_store(base + k, 0)
    elif m == 1:
        for i in range(n):
            var keep = Int(lp.unsafe_load(i)) * pos
            var base = i * row
            for k in range(keep, row):
                dp.unsafe_store(base + k, 0)
    else:
        for i in range(n):
            var at = i * pos + (Int(lp.unsafe_load(i)) - 1) * row
            for k in range(row):
                dp.unsafe_store(i * row + k, sp.unsafe_load(at + k))


def nsum_f64(p: _F64P, count: Int) -> Float64:
    """`_portable_math.nsum` of `count` >= 1 float64 values."""
    var total = 0.0 + p.unsafe_load(0)
    var c = Float64(0.0)
    for i in range(1, count):
        var x = p.unsafe_load(i)
        var t = total + x
        if abs(total) >= abs(x):
            c += (total - t) + x
        else:
            c += (x - t) + total
        total = t
    if c != 0.0 and c - c == 0.0:
        total += c
    return total


def row_means_f64(sp: _F64P, op: _F64P, nr: Int, nc: Int) -> Int:
    """`fsum(row) / nc` per row; returns the number of +inf/-inf rows."""
    var bad = 0
    for r in range(nr):  # small-loop(nr: search candidates): one mean per candidate row
        var terms = List[Float64](capacity=nc)
        var nan_v = Float64(0.0)
        var any_nan = False
        var pinf = False
        var ninf = False
        for c in range(nc):  # small-loop(nc: cross-validation splits): one score per split
            var v = sp.unsafe_load(r * nc + c)
            if isnan(v):
                any_nan = True
                nan_v = v
            elif isinf(v):
                if v > 0:
                    pinf = True
                else:
                    ninf = True
            else:
                terms.append(v)
        if pinf and ninf:
            bad += 1
            continue
        var total: Float64
        if any_nan:
            total = nan_v
        elif pinf:
            total = inf[DType.float64]()
        elif ninf:
            total = -inf[DType.float64]()
        else:
            total = _exact_sum(terms)
        op.unsafe_store(r, total / Float64(nc))
    return bad


comptime _FSUM_DOWN = Float64(1.0) / Float64(18446744073709551616.0)  # 2^-64, exact
comptime _FSUM_UP = Float64(18446744073709551616.0)  # 2^64, exact


def _exact_sum_wide(terms: List[Float64]) -> Float64:
    """`_exact_sum` without Shewchuk's intermediate overflow: a sum whose
    partials pass the binary64 range (1e308 + 1e308 - 1e308) is redone with
    every term scaled by 2^-64, which is exact for each term kept there (its
    scaled value round-trips), and the rounded sum scaled back. Terms that
    would lose bits under the scaling (below about 2^-1010) are added
    unscaled to the scaled-back sum and its exact remainder. Returns +-inf
    when the exact sum itself is outside the binary64 range (fsum's
    OverflowError). Off the overflow path this is `_exact_sum`, the same
    bits."""
    var s = _exact_sum(terms)
    if not isinf(s) and not isnan(s):
        return s
    var big = List[Float64](capacity=len(terms))
    var small = List[Float64]()
    for i in range(len(terms)):  # small-loop(terms: terms of one fsum): every caller sums a k-sized list
        var w = terms[i] * _FSUM_DOWN
        if w * _FSUM_UP == terms[i]:
            big.append(w)
        else:
            small.append(terms[i])
    var b = _exact_sum(big)
    var up = b * _FSUM_UP
    if isinf(up) or len(small) == 0:
        return up
    # the scaled sum's remainder B - b, then everything at the unscaled scale
    var rest = big.copy()
    rest.append(-b)
    var tail = List[Float64](capacity=len(small) + 2)
    tail.append(up)
    tail.append(_exact_sum(rest) * _FSUM_UP)
    for i in range(len(small)):  # small-loop(small: terms below the scaling range): a subset of one k-sized fsum
        tail.append(small[i])
    return _exact_sum(tail)


def _fsum_terms(vals: List[Float64], count: Int, mut bad: Bool) -> Float64:
    """`_portable_math.fsum` of count float64 terms: the last NaN if any,
    else one kind of infinity, else Shewchuk's exact partials rounded once
    (`_exact_sum`, a zero sum +0.0); bad = +inf with -inf (fsum's
    ValueError)."""
    var terms = List[Float64](capacity=count)
    var nan_v = Float64(0.0)
    var any_nan = False
    var pinf = False
    var ninf = False
    for i in range(count):  # small-loop(count: terms of one fsum): every caller sums a k-sized list (estimator weights, class priors, one candidate's split scores)
        var v = vals[i]
        if isnan(v):
            any_nan = True
            nan_v = v
        elif isinf(v):
            if v > 0:
                pinf = True
            else:
                ninf = True
        else:
            terms.append(v)
    bad = pinf and ninf
    if any_nan:
        return nan_v
    if pinf:
        return inf[DType.float64]()
    if ninf:
        return -inf[DType.float64]()
    return _exact_sum_wide(terms)


def fsum_f64(sp: _F64P, op: _F64P, count: Int) -> Int:
    """out[0] = the exact sum of `count` >= 1 float64 terms; returns the
    status of `fsum_f64_binding` (1 for +inf with -inf or an overflow)."""
    var vals = List[Float64](capacity=count)
    for i in range(count):  # small-loop(count: terms of one fsum): every caller sums a k-sized list (estimator weights, class priors)
        vals.append(sp.unsafe_load(i))
    var bad = False
    var t = _fsum_terms(vals, count, bad)
    if bad:
        return 1
    if isinf(t):
        # an infinite result from finite terms is the exact sum outside the
        # binary64 range: fsum's OverflowError (the caller raises it)
        var any_inf = False
        for i in range(count):  # small-loop(count: terms of one fsum): every caller sums a k-sized list
            if isinf(vals[i]):
                any_inf = True
        if not any_inf:
            # flagged as status 1 with +inf in out[0] (a -inf + inf leaves
            # out[0] at the caller's 0.0): the caller raises OverflowError
            op.unsafe_store(0, t)
            return 1
    op.unsafe_store(0, t)
    return 0



def row_stds_f64(sp: _F64P, mp: _F64P, op: _F64P, nr: Int, nc: Int):
    """The population standard deviation of each row about `mp[row]`."""
    var sq = List[Float64](length=nc, fill=0.0)
    for r in range(nr):  # small-loop(nr: search candidates): one deviation per candidate row
        var m = mp.unsafe_load(r)
        for c in range(nc):  # small-loop(nc: cross-validation splits): one squared deviation per split
            var d = sp.unsafe_load(r * nc + c) - m
            sq[c] = pinned_mul_f64(d, d)
        var bad = False
        var t = _fsum_terms(sq, nc, bad)
        op.unsafe_store(r, sqrt(t / Float64(nc)))


def class_ratio_f64(cp: _I64P, op: _F64P, kk: Int, nn: Int, m: Int) -> Int:
    """Per-class ratios (mode 0 balanced, mode 1 prior); returns the number
    of zero counts in mode 0."""
    var zeros = 0
    for c in range(kk):
        var cnt = Int(cp.unsafe_load(c))
        if m == 0:
            if cnt == 0:
                zeros += 1
                op.unsafe_store(c, 0.0)
            else:
                op.unsafe_store(c, Float64(nn) / Float64(kk * cnt))
        else:
            op.unsafe_store(c, Float64(cnt) / Float64(nn))
    return zeros


def shard_topk_merge_f32(
    tp: _I64P, od: _U32P, oi: _U32P, s_count: Int, nq: Int, kk: Int
) -> Int:
    """The per-query shard top-k merge of `shard_topk_merge_f32_host_binding`.
    Returns 0, 1 for a local id outside its shard, 2 for fewer than k
    candidates."""
    var rc = 0
    var keys = List[UInt64]()
    var cb = List[UInt32](length=kk, fill=0)
    var ci = List[UInt32](length=kk, fill=0)
    for row in range(nq):
        keys.clear()
        for r in range(s_count):
            var dp = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(tp[5 * r]))
            var ip = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(tp[5 * r + 1]))
            var first = Int(tp[5 * r + 2])
            var end = Int(tp[5 * r + 3])
            var width = Int(tp[5 * r + 4])
            for j in range(width):
                var local = Int(ip.unsafe_load(row * width + j))
                if local < 0 or local >= end - first:
                    rc = 1
                    break
                var bits = dp.unsafe_load(row * width + j)
                var tw = bits ^ (UInt32(0xFFFFFFFF) if (bits & UInt32(0x80000000)) != 0 else UInt32(0x80000000))
                keys.append((UInt64(tw) << 32) | UInt64(first + local))
            if rc != 0:
                break
        if rc != 0:
            break
        if len(keys) < kk:
            rc = 2
            break
        sort(keys)
        for a in range(kk):
            var tw = UInt32(keys[a] >> 32)
            cb[a] = tw ^ (UInt32(0x80000000) if (tw & UInt32(0x80000000)) != 0 else UInt32(0xFFFFFFFF))
            ci[a] = UInt32(keys[a] & UInt64(0xFFFFFFFF))
        # estimator.mojo's host insertion sort on the float distance
        for a in range(1, kk):
            var tb = cb[a]
            var tix = ci[a]
            var dv = bitcast[DType.float32](tb)
            var b = a - 1
            while b >= 0:
                var dprev = bitcast[DType.float32](cb[b])
                if dprev < dv or (dprev == dv and ci[b] <= tix):
                    break
                cb[b + 1] = cb[b]
                ci[b + 1] = ci[b]
                b -= 1
            cb[b + 1] = tb
            ci[b + 1] = tix
        for j in range(kk):
            od.unsafe_store(row * kk + j, cb[j])
            oi.unsafe_store(row * kk + j, ci[j])
    return rc
