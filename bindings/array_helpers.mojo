# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Layout helpers of the NumPy-free layer (lane pyglue-sweep, Oct 3 2026:
Python = glue only). Each replaces a Python loop over rows or columns that
moved bytes between host buffers: `_array._reorder` (C <-> F), the strided
`Array.__getitem__` copy, `parallel_forecasting`'s column gather and
scatter, `_ragged`'s per-row padding passes, and `_portable_math.nsum`.

Byte moves only (plus `nsum_f64`, the same Neumaier fold in the same order
as the Python routine it replaces), so no result bit depends on them. The
same file is compiled into the base binding (`_mojolearn`) and the core
host binding (`_mojolearn_core_host`), so a CPU-only install runs the same
code.
"""
from std.builtin.sort import sort
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.math import inf, isinf, isnan, sqrt
from checks.numerics import pinned_mul_f64
from glm.impl.lm_finish import _exact_sum


@always_inline
def _addr_ptr[dt: DType](addr: Int) raises -> MutPointer[Scalar[dt], MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=addr)


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


def strided_copy_bytes_binding(
    src_addr: PythonObject, dst_addr: PythonObject, dims_addr: PythonObject,
    ndim: PythonObject, itemsize: PythonObject,
) raises -> PythonObject:
    """Copy an n-d strided selection between two flat host buffers.

    `dims` is int64[3 * ndim + 2]: the shape, the source strides, the
    destination strides (all in ELEMENTS, a stride may be negative), then
    the source and destination base offsets. Elements are `itemsize` bytes
    (1, 2, 4 or 8) and are moved as bits. Returns the element count."""
    var nd = Int(py=ndim)
    var isz = Int(py=itemsize)
    if nd < 1:
        raise Error("strided_copy_bytes: ndim must be positive")
    var dp = _addr_ptr[DType.int64](Int(py=dims_addr))
    var total = 1
    for k in range(nd):
        var s = Int(dp[k])
        if s < 0:
            raise Error("strided_copy_bytes: negative dimension")
        total *= s
    if total == 0:
        return PythonObject(0)
    var src = Int(py=src_addr)
    var dst = Int(py=dst_addr)
    if src == 0 or dst == 0:
        raise Error("strided_copy_bytes: null buffer address")
    if isz != 1 and isz != 2 and isz != 4 and isz != 8:
        raise Error("strided_copy_bytes: itemsize must be 1, 2, 4 or 8")
    with GILReleased(Python()):
        if isz == 1:
            _strided_copy[DType.uint8](src, dst, dp, nd, total)
        elif isz == 2:
            _strided_copy[DType.uint16](src, dst, dp, nd, total)
        elif isz == 4:
            _strided_copy[DType.uint32](src, dst, dp, nd, total)
        else:
            _strided_copy[DType.uint64](src, dst, dp, nd, total)
    return PythonObject(total)


def check_lengths_i64_binding(
    lengths_addr: PythonObject, b: PythonObject, length: PythonObject
) raises -> PythonObject:
    """The first i whose length is outside [1, length], or -1."""
    var n = Int(py=b)
    var hi = Int64(Int(py=length))
    if n <= 0:
        return PythonObject(-1)
    var lp = _addr_ptr[DType.int64](Int(py=lengths_addr))
    for i in range(n):
        var v = lp.unsafe_load(i)
        if v < 1 or v > hi:
            return PythonObject(i)
    return PythonObject(-1)


def ragged_rows_bytes_binding(
    src_addr: PythonObject, dst_addr: PythonObject, lengths_addr: PythonObject,
    b: PythonObject, row_bytes: PythonObject, pos_bytes: PythonObject, mode: PythonObject,
) raises -> PythonObject:
    """Per-row ragged passes over a (B, L, ...) block of B rows of
    `row_bytes`, `pos_bytes` per position, int64 lengths:
      mode 0: dst row i = src row i's first lengths[i] positions, then zero
              bytes (dst must be zeroed or is fully written here);
      mode 1: zero dst row i from position lengths[i] on (src unused);
      mode 2: dst row i (row_bytes = one position, V values) = src position
              lengths[i] - 1 of row i, where src rows are pos_bytes long."""
    var n = Int(py=b)
    var row = Int(py=row_bytes)
    var pos = Int(py=pos_bytes)
    var m = Int(py=mode)
    if n <= 0 or row <= 0:
        return PythonObject(0)
    var lp = _addr_ptr[DType.int64](Int(py=lengths_addr))
    var dp = _addr_ptr[DType.uint8](Int(py=dst_addr))
    var sp = _addr_ptr[DType.uint8](Int(py=src_addr))
    with GILReleased(Python()):
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
    return PythonObject(0)


def nsum_f64_binding(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """`_portable_math.nsum`: CPython 3.12+'s float `sum`, `0 + x0`, then
    Neumaier's compensated sum in order, the compensation added once at the
    end when it is nonzero and finite. 0.0 for an empty input."""
    var count = Int(py=n)
    if count <= 0:
        return PythonObject(0.0)
    var p = _addr_ptr[DType.float64](Int(py=addr))
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
    return PythonObject(total)


def row_means_f64_binding(
    src_addr: PythonObject, rows: PythonObject, cols: PythonObject, out_addr: PythonObject,
) raises -> PythonObject:
    """The mean of each row of a C-order float64 (rows, cols) table:
    `_portable_math.fsum(row) / cols` (lane py-runtime-b: the search
    results' mean test and train scores, one row per candidate and one
    column per split; they were Python `fsum` over lists). The sum is
    CPython's math.fsum (Shewchuk's exact partials, rounded once; a zero sum
    is +0.0; glm/impl/lm_finish.mojo `_exact_sum`) with `fsum`'s special
    values: a row holding +inf and -inf writes nothing and is counted, a
    NaN gives the row's last NaN, one kind of infinity gives it. Returns the
    number of +inf/-inf rows (the caller raises fsum's ValueError)."""
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    if nr <= 0 or nc <= 0:
        return PythonObject(0)
    var sp = _addr_ptr[DType.float64](Int(py=src_addr))
    var op = _addr_ptr[DType.float64](Int(py=out_addr))
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
    return PythonObject(bad)


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
    for i in range(len(small)):  # small-loop(small: the few terms below the scaling's exact range)
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


def fsum_f64_binding(src_addr: PythonObject, n: PythonObject, out_addr: PythonObject) raises -> PythonObject:
    """`_portable_math.fsum` (lane py-runtime round 3: it was `math.fsum`
    plus a Python big-integer fold): out[0] = the exact sum of n float64
    terms (see `_fsum_terms`). Returns 1 for +inf with -inf (the caller
    raises fsum's ValueError), 2 when the exact sum of finite terms is
    outside the binary64 range (fsum's OverflowError), else 0."""
    var count = Int(py=n)
    var op = _addr_ptr[DType.float64](Int(py=out_addr))
    if count <= 0:
        op.unsafe_store(0, Float64(0.0))
        return PythonObject(0)
    var sp = _addr_ptr[DType.float64](Int(py=src_addr))
    var vals = List[Float64](capacity=count)
    for i in range(count):  # small-loop(count: terms of one fsum): every caller sums a k-sized list (estimator weights, class priors)
        vals.append(sp.unsafe_load(i))
    var bad = False
    var t = _fsum_terms(vals, count, bad)
    if bad:
        return PythonObject(1)
    if isinf(t):
        # an infinite result from finite terms is the exact sum outside the
        # binary64 range: fsum's OverflowError (the caller raises it)
        var any_inf = False
        for i in range(count):  # small-loop(count: terms of one fsum)
            if isinf(vals[i]):
                any_inf = True
        if not any_inf:
            return PythonObject(2)
    op.unsafe_store(0, t)
    return PythonObject(0)


def row_stds_f64_binding(
    src_addr: PythonObject, rows: PythonObject, cols: PythonObject, means_addr: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """The population standard deviation of each row of a C-order float64
    (rows, cols) score table about its mean (lane py-runtime round 3; it was
    Python's sqrt(fsum((v - m) * (v - m)) / cols) per search candidate): the
    same binary64 subtraction and product per score, the exact fsum, one
    division and the correctly rounded sqrt."""
    var nr = Int(py=rows)
    var nc = Int(py=cols)
    if nr <= 0 or nc <= 0:
        return PythonObject(0)
    var sp = _addr_ptr[DType.float64](Int(py=src_addr))
    var mp = _addr_ptr[DType.float64](Int(py=means_addr))
    var op = _addr_ptr[DType.float64](Int(py=out_addr))
    var sq = List[Float64](length=nc, fill=0.0)
    for r in range(nr):  # small-loop(nr: search candidates): one deviation per candidate row
        var m = mp.unsafe_load(r)
        for c in range(nc):  # small-loop(nc: cross-validation splits): one squared deviation per split
            var d = sp.unsafe_load(r * nc + c) - m
            sq[c] = pinned_mul_f64(d, d)
        var bad = False
        var t = _fsum_terms(sq, nc, bad)
        op.unsafe_store(r, sqrt(t / Float64(nc)))
    return PythonObject(0)

def class_ratio_f64_host_binding(
    counts_addr: PythonObject, k: PythonObject, n: PythonObject, mode: PythonObject, out_addr: PythonObject,
) raises -> PythonObject:
    """Per-class ratios of the k int64 class counts (lane py-runtime-b: the
    Python list arithmetic of the class-weight and prior glue). mode 0:
    scikit-learn's 'balanced' weight n / (k * count_c); mode 1: the
    empirical prior count_c / n. Both operands are exact integers (below
    2^53) and the quotient is one IEEE binary64 division, so the words are
    Python's `int / int`. Returns the number of zero counts in mode 0 (the
    caller raises Python's ZeroDivisionError; those cells are left 0)."""
    var kk = Int(py=k)
    var nn = Int(py=n)
    var m = Int(py=mode)
    if kk <= 0:
        return PythonObject(0)
    if m != 0 and m != 1:
        raise Error("class_ratio_f64: mode is 0 (balanced) or 1 (prior)")
    if m == 1 and nn == 0:
        raise Error("class_ratio_f64: the prior of zero rows")
    var cp = _addr_ptr[DType.int64](Int(py=counts_addr))
    var op = _addr_ptr[DType.float64](Int(py=out_addr))
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
    return PythonObject(zeros)


def shard_topk_merge_f32_host_binding(
    table_addr: PythonObject, n_shards: PythonObject, n_queries: PythonObject, k: PythonObject,
    out_dist_addr: PythonObject, out_idx_addr: PythonObject,
) raises -> PythonObject:
    """`parallel_neighbors_reference._merge` (lane pyglue-sweep: out of the
    Python per-query loop): per query row, every reference shard's
    candidates as composite keys (neighbors/checks/select_radix_identical
    `composite_key`: twiddled float32 distance bits << 32 | global index),
    the k smallest keys, then estimator.mojo's host insertion sort on the
    float distance (ties by index). `table` is int64[5 * n_shards]: per
    shard the float32 distance address, the int64 local-id address, the
    first and end global rows and the width (columns per query row).
    Writes the winning float bits and uint32 global indices. Returns 0, 1
    for a local id outside its shard, 2 for fewer than k candidates."""
    var s_count = Int(py=n_shards)
    var nq = Int(py=n_queries)
    var kk = Int(py=k)
    if s_count < 1 or nq < 0 or kk < 1:
        raise Error("shard_topk_merge_f32: bad dimensions")
    if nq == 0:
        return PythonObject(0)
    var tp = _addr_ptr[DType.int64](Int(py=table_addr))
    var od = _addr_ptr[DType.uint32](Int(py=out_dist_addr))
    var oi = _addr_ptr[DType.uint32](Int(py=out_idx_addr))
    var rc = 0
    with GILReleased(Python()):
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
    return PythonObject(rc)
