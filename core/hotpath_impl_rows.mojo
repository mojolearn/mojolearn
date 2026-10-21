# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed kernels of the row, fold and draw helpers of
`bindings/hotpath_helpers.mojo` (fold_pair_f32 onward).

The Python door stays in `bindings/hotpath_helpers.mojo`: it unpacks the
PythonObject arguments, calls one function here (with the GIL released
where it was released before) and returns the result as a Python int.
Every statement here is the binding's, word for word, in the same order
(lane thin-core, 2026-10-21: a pure move, no bit change). Each function
takes buffer addresses as Int and returns 0 (or the count / status the
binding returned).
"""
from std.math import fma, sqrt
from std.memory import bitcast
from sequence.schedule import fill_epoch_order, splitmix64
from checks.numerics import portable_cosf, portable_log64
from metrics.checks.pinned_sum import canonicalize_nan
from core.hotpath_impl import HOTPATH_SABOTAGE, HP_F32, HP_F64, HP_I32, HP_I64, _ptr


@always_inline
def _ftz_bits(v: Float32) -> Float32:
    """A subnormal becomes a zero of the same sign (checks/numerics `ftz`)."""
    var b = bitcast[DType.uint32](v)
    if (b & UInt32(0x7F800000)) == 0 and (b & UInt32(0x007FFFFF)) != 0:
        return bitcast[DType.float32](b & UInt32(0x80000000))
    return v


def fold_pair_f32(dst_addr: Int, src_addr: Int, count: Int) raises -> Int:
    """cross_vendor's coordinator fold: dst[i] = ftz(ftz(dst[i]) +
    ftz(src[i])), one float32 rounding per add, elementwise."""
    if count < 0:
        raise Error("fold_pair_f32: n must be non-negative")
    if count == 0:
        return 0
    var dp = _ptr[DType.float32](dst_addr)
    var sp = _ptr[DType.float32](src_addr)
    for i in range(count):
        # NaN -> the canonical word, as the device kernel (lane/review-fixes)
        var s = canonicalize_nan(_ftz_bits(_ftz_bits(dp.unsafe_load(i)) + _ftz_bits(sp.unsafe_load(i))))
        comptime if HOTPATH_SABOTAGE:
            s = -s
        dp.unsafe_store(i, s)
    return 0


def threshold_labels_i64(
    src_addr: Int, c: Int, count: Int, t: Float64, st: Bool, lo: Int64, hi: Int64, dst_addr: Int,
) raises -> Int:
    """dst[i] = hi if src[i] > t (st) / >= t (not st), else lo; int64 out
    from a float32 (code 0) or float64 (code 1) score. NaN takes `lo`."""
    if count < 0:
        raise Error("threshold_labels_i64: n must be non-negative")
    if count == 0:
        return 0
    var dp = _ptr[DType.int64](dst_addr)
    if c != HP_F32 and c != HP_F64:
        raise Error("threshold_labels_i64: float32 or float64 scores only")
    var fp = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=src_addr)
    var gp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=src_addr)
    for i in range(count):
        var v = Float64(fp.unsafe_load(i)) if c == HP_F32 else gp.unsafe_load(i)
        var up = v > t if st else v >= t
        comptime if HOTPATH_SABOTAGE:
            up = not up
        dp.unsafe_store(i, hi if up else lo)
    return 0


def scale_shift_ftz_f32(
    src_addr: Int, count: Int, fa: Float32, fb: Float32, o: Int, dst_addr: Int,
) raises -> Int:
    """Elementwise float32, each step one rounding then `ftz`: op 0 dst =
    ftz(ftz(a * v) + b); op 1 dst = ftz(sqrt(ftz(v * a))); op 2 dst =
    ftz(v * a). src and dst may be the same buffer."""
    if count < 0 or o < 0 or o > 2:
        raise Error("scale_shift_ftz_f32: bad n or op")
    if count == 0:
        return 0
    var sp = _ptr[DType.float32](src_addr)
    var dp = _ptr[DType.float32](dst_addr)
    for i in range(count):
        var v = sp.unsafe_load(i)
        var r: Float32
        if o == 0:
            r = _ftz_bits(_ftz_bits(fa * v) + fb)
        elif o == 1:
            r = _ftz_bits(sqrt(_ftz_bits(v * fa)))
        else:
            r = _ftz_bits(v * fa)
        comptime if HOTPATH_SABOTAGE:
            r = -r
        dp.unsafe_store(i, r)
    return 0


def bincount_i64(
    src_addr: Int, c: Int, count: Int, kk: Int, counts_addr: Int, accumulate: Bool,
) raises -> Int:
    """counts[v] += 1 for each of the n int32 (code 2) or int64 (code 3)
    values at `src` (counts zeroed first unless `accumulate`); a value
    outside [0, k) raises before any count is written."""
    if count < 0 or kk < 1 or (c != HP_I32 and c != HP_I64):
        raise Error("bincount_i64: needs n >= 0, k >= 1 and int32 or int64 values")
    var cp = _ptr[DType.int64](counts_addr)
    var bad = False
    if not accumulate:
        for j in range(kk):
            cp.unsafe_store(j, 0)
    if count > 0:
        var ip = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=src_addr)
        var lp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=src_addr)
        for i in range(count):
            var v = Int(ip.unsafe_load(i)) if c == HP_I32 else Int(lp.unsafe_load(i))
            if v < 0 or v >= kk:
                bad = True
                break
        if not bad:
            for i in range(count):
                var v = Int(ip.unsafe_load(i)) if c == HP_I32 else Int(lp.unsafe_load(i))
                comptime if HOTPATH_SABOTAGE:
                    v = (v + 1) % kk
                cp.unsafe_store(v, cp.unsafe_load(v) + 1)
    if bad:
        raise Error("bincount_i64: value out of range")
    return 0


def bincount2_i32(
    a_addr: Int, b_addr: Int, count: Int, ka: Int, kb: Int, counts_addr: Int,
) raises -> Int:
    """int64 counts[v * ka + u] over the n rows of two int32 code columns
    (u = a[i], v = b[i]): the host column's twin of the GPU base binding's
    `bincount2_i32` (core/hotpath_device.mojo device_bincount2_i32,
    StratifiedGroupKFold's group x class table). The counts are integers, so
    the table is the device's bit for bit. A code outside [0, ka) x [0, kb)
    raises before any count is written (the device leaves the table
    untouched then too)."""
    if count < 0 or ka < 1 or kb < 1:
        raise Error("bincount2_i32: 0 <= n, 1 <= ka, kb")
    var ap = _ptr[DType.int32](a_addr)
    var bp = _ptr[DType.int32](b_addr)
    var cp = _ptr[DType.int64](counts_addr)
    for i in range(count):
        var u = Int(ap.unsafe_load(i))
        var v = Int(bp.unsafe_load(i))
        if u < 0 or u >= ka or v < 0 or v >= kb:
            raise Error("bincount2_i32: a code is out of range")
    for j in range(ka * kb):
        cp.unsafe_store(j, 0)
    for i in range(count):
        var u = Int(ap.unsafe_load(i))
        comptime if HOTPATH_SABOTAGE:
            u = (u + 1) % ka
        var at = Int(bp.unsafe_load(i)) * ka + u
        cp.unsafe_store(at, cp.unsafe_load(at) + 1)
    return 0


def compact_notnan_f32(src_addr: Int, count: Int, st: Int, dst_addr: Int) raises -> Int:
    """dst = the non-NaN values of src[0], src[stride], ..., in order;
    returns their count."""
    if count < 0 or st < 1:
        raise Error("compact_notnan_f32: n >= 0, stride >= 1")
    if count == 0:
        return 0
    var sp = _ptr[DType.float32](src_addr)
    var dp = _ptr[DType.float32](dst_addr)
    var m = 0
    for i in range(count):
        var v = sp.unsafe_load(i * st)
        if v == v:
            dp.unsafe_store(m, v)
            m += 1
    return m


def gather_keep_neg_i32(
    table_addr: Int, nt: Int, src_addr: Int, count: Int, dst_addr: Int,
) raises -> Int:
    """dst[i] = table[src[i]] for src[i] >= 0, else src[i]; int32. A
    non-negative value outside the table raises before any write."""
    if count < 0 or nt < 0:
        raise Error("gather_keep_neg_i32: n and n_table must be non-negative")
    if count == 0:
        return 0
    var tp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=table_addr)
    var sp = _ptr[DType.int32](src_addr)
    var dp = _ptr[DType.int32](dst_addr)
    var bad = False
    for i in range(count):
        if Int(sp.unsafe_load(i)) >= nt:
            bad = True
            break
    if not bad:
        for i in range(count):
            var v = sp.unsafe_load(i)
            dp.unsafe_store(i, tp.unsafe_load(Int(v)) if v >= 0 else v)
    if bad:
        raise Error("gather_keep_neg_i32: index out of range")
    return 0


def dot_rows_f32(a_addr: Int, x_addr: Int, rows: Int, cols: Int, dst_addr: Int) raises -> Int:
    """dst[j] = float32(sum_i a[i] * x[i, j]), the sum in float64 in row
    order (one fixed fold): SVC's linear coef_."""
    if rows < 0 or cols < 0:
        raise Error("dot_rows_f32: m and d must be non-negative")
    if cols == 0:
        return 0
    var dp = _ptr[DType.float32](dst_addr)
    var acc = List[Float64](length=cols, fill=0.0)
    if rows > 0:
        var ap = _ptr[DType.float32](a_addr)
        var xp = _ptr[DType.float32](x_addr)
        for i in range(rows):
            var a = Float64(ap.unsafe_load(i))
            for j in range(cols):
                acc[j] = acc[j] + a * Float64(xp.unsafe_load(i * cols + j))
    for j in range(cols):
        dp.unsafe_store(j, Float32(acc[j]))
    return 0


def assign_fold_i64(idx_addr: Int, count: Int, rows: Int, f: Int32, folds_addr: Int) raises -> Int:
    """folds[idx[i]] = fold for the m int64 test indices of one split.
    Returns 0, 1 when an index is outside [0, n), 2 on an overlap."""
    if count < 0 or rows < 1:
        raise Error("assign_fold_i64: m >= 0, n >= 1")
    if count == 0:
        return 0
    var ip = _ptr[DType.int64](idx_addr)
    var fp = _ptr[DType.int32](folds_addr)
    var status = 0
    for i in range(count):
        var r = Int(ip.unsafe_load(i))
        if r < 0 or r >= rows:
            status = 1
            break
        if fp.unsafe_load(r) != -1:
            status = 2
            break
        fp.unsafe_store(r, f)
    return status


def count_fold_hits_i64(idx_addr: Int, count: Int, folds_addr: Int, rows: Int, f: Int32) raises -> Int:
    """How many of the m int64 indices sit in fold `f` of the folds array."""
    if count <= 0:
        return 0
    var ip = _ptr[DType.int64](idx_addr)
    var fp = _ptr[DType.int32](folds_addr)
    var hits = 0
    for i in range(count):
        var r = Int(ip.unsafe_load(i))
        if r >= 0 and r < rows and fp.unsafe_load(r) == f:
            hits += 1
    return hits


def mark_split_roles_f32(
    idx_addr: Int, count: Int, rows: Int, k: Int, r: Int, roles_addr: Int,
) raises -> Int:
    """roles[split * n + idx[i]] = role (1 training, 2 test). Returns 0, 1
    when an index is outside [0, n), 2 when a row already has a role."""
    if count < 0 or rows < 1 or k < 0 or (r != 1 and r != 2):
        raise Error("mark_split_roles_f32: m >= 0, n >= 1, split >= 0, role 1 or 2")
    if count == 0:
        return 0
    var ip = _ptr[DType.int64](idx_addr)
    var rp = _ptr[DType.float32](roles_addr)
    var base = k * rows
    var value = Float32(r)
    var status = 0
    for i in range(count):
        var row = Int(ip.unsafe_load(i))
        if row < 0 or row >= rows:
            status = 1
            break
        if rp.unsafe_load(base + row) != Float32(0):
            status = 2
            break
        rp.unsafe_store(base + row, value)
    return status


def split_table_i32(
    perm_addr: Int, mm: Int, te: Int, tr: Int, counts_addr: Int, table_addr: Int, sums_addr: Int,
) raises -> Int:
    """One GroupShuffleSplit draw as a per-group side table and the row
    sums of each side."""
    if mm < 1 or te < 0 or tr < 0 or te + tr > mm:
        raise Error("split_table_i32: bad sizes")
    var pp = _ptr[DType.int64](perm_addr)
    var cp = _ptr[DType.int64](counts_addr)
    var tp = _ptr[DType.int32](table_addr)
    var sp = _ptr[DType.int64](sums_addr)
    var c_tr = Int64(0)
    var c_te = Int64(0)
    var bad = False
    for g in range(mm):
        tp.unsafe_store(g, 2)
    for i in range(te + tr):
        var g = Int(pp.unsafe_load(i))
        if g < 0 or g >= mm:
            bad = True
            break
        if i < te:
            tp.unsafe_store(g, 1)
            c_te += cp.unsafe_load(g)
        else:
            tp.unsafe_store(g, 0)
            c_tr += cp.unsafe_load(g)
    if bad:
        raise Error("split_table_i32: group index out of range")
    sp.unsafe_store(0, c_tr)
    sp.unsafe_store(1, c_te)
    return 0


def scatter_rows_bytes_host(
    src_addr: Int, rows_addr: Int, count: Int, width: Int, dst_addr: Int, nd: Int,
) raises -> Int:
    """dst row rows[i] = src row i (row_bytes each); a row outside
    [0, dst_rows) raises before any write."""
    if count < 0 or width < 0 or nd < 0:
        raise Error("scatter_rows_bytes: dimensions must be non-negative")
    if count == 0 or width == 0:
        return 0
    var sp = _ptr[DType.uint8](src_addr)
    var dp = _ptr[DType.uint8](dst_addr)
    var rp = _ptr[DType.int64](rows_addr)
    var bad = False
    for i in range(count):
        var r = Int(rp.unsafe_load(i))
        if r < 0 or r >= nd:
            bad = True
            break
    if not bad:
        for i in range(count):
            var r = Int(rp.unsafe_load(i))
            for b in range(width):
                dp.unsafe_store(r * width + b, sp.unsafe_load(i * width + b))
    if bad:
        raise Error("scatter_rows_bytes: row index out of bounds")
    return 0


def uniform_init_f32(
    dst_addr: Int, count: Int, lo: Float64, hi: Float64, seed: UInt64, off: UInt64,
) raises -> Int:
    """dst[i] = float32(low + (high - low) * u_i), u_i the 53-bit uniform of
    splitmix64 at counter offset + i of the seed."""
    if count < 0:
        raise Error("uniform_init_f32: n must be non-negative")
    if count == 0:
        return 0
    var dp = _ptr[DType.float32](dst_addr)
    for i in range(count):
        var s = seed + (off + UInt64(i)) * UInt64(0x9E3779B97F4A7C15)
        var u = Float64(splitmix64(s) >> 11) * 1.1102230246251565e-16
        # lane fix-s1-shared: an explicit fma (Mojo's default fp-mode
        # contracts `lo + (hi - lo) * u` anyway); the device twin
        # (core/hotpath_device.mojo) is the same sf64_fma
        dp.unsafe_store(i, Float32(fma(hi - lo, u, lo)))
    return 0


def normal_init_f32(
    dst_addr: Int, count: Int, mu: Float64, sd: Float64, seed: UInt64, off: UInt64,
) raises -> Int:
    """dst[i] = float32(mean + std * z_i), Box-Muller over the splitmix64
    uniforms at counters 2c and 2c + 1, c = offset + i."""
    if count < 0:
        raise Error("normal_init_f32: n must be non-negative")
    if count == 0:
        return 0
    var dp = _ptr[DType.float32](dst_addr)
    for i in range(count):
        var c = (off + UInt64(i)) * UInt64(2)
        var s1 = seed + c * UInt64(0x9E3779B97F4A7C15)
        var s2 = seed + (c + UInt64(1)) * UInt64(0x9E3779B97F4A7C15)
        var u1 = 1.0 - Float64(splitmix64(s1) >> 11) * 1.1102230246251565e-16
        var u2 = Float64(splitmix64(s2) >> 11) * 1.1102230246251565e-16
        var r = sqrt(-2.0 * portable_log64(u1))
        var cz = portable_cosf(Float32(6.283185307179586 * u2))
        # lane fix-s1-shared: an explicit fma, as the device twin's
        # sf64_fma (core/hotpath_device.mojo::_normal_init_kernel)
        dp.unsafe_store(i, Float32(fma(sd, r * Float64(cz), mu)))
    return 0


def epoch_order_i32(dst_addr: Int, count: Int, shuffle: Bool, state_addr: Int) raises -> Int:
    """One epoch's row order (sequence/schedule.mojo `fill_epoch_order`)."""
    if count < 1:
        raise Error("epoch_order_i32: n must be positive")
    var dp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=dst_addr)
    var stp = _ptr[DType.uint64](state_addr)
    var s = stp.unsafe_load(0)
    fill_epoch_order(dp, count, shuffle, s)
    stp.unsafe_store(0, s)
    return 0


@always_inline
def _powi64(b: Float64, e: Int) -> Float64:
    """b ** e for e >= 0 by squaring, in a fixed order (the same bits on
    every column)."""
    var r = Float64(1)
    var x = b
    var k = e
    while k > 0:
        if (k & 1) != 0:
            r = r * x
        x = x * x
        k >>= 1
    return r


def adam_hyper_f64(
    dst_addr: Int, s0: Int, ns: Int, lr: Float64, b1: Float64, b2: Float64, eps: Float64,
    wd: Float64, dec: Float64,
) raises -> Int:
    """The Adam / AdamW hyper block of nsteps rows of 9 float64."""
    if s0 < 1 or ns < 0:
        raise Error("adam_hyper_f64: step0 >= 1, nsteps >= 0")
    var dp = _ptr[DType.float64](dst_addr)
    for t in range(ns):
        var step = s0 + t
        var bc1 = 1.0 - _powi64(b1, step)
        var bc2 = 1.0 - _powi64(b2, step)
        var row = t * 9
        dp.unsafe_store(row + 0, lr / bc1)
        dp.unsafe_store(row + 1, 1.0 - b1)
        dp.unsafe_store(row + 2, b2)
        dp.unsafe_store(row + 3, 1.0 - b2)
        dp.unsafe_store(row + 4, eps)
        dp.unsafe_store(row + 5, sqrt(bc2))
        dp.unsafe_store(row + 6, wd)
        dp.unsafe_store(row + 7, dec)
        dp.unsafe_store(row + 8, 1.0 - lr * wd)
    return 0


def mean_std_f32(src_addr: Int, count: Int, st: Int, dst_addr: Int) raises -> Int:
    """dst (float64, 2) = [mean, population std], two passes in float64 in
    row order."""
    if count < 1 or st < 1:
        raise Error("mean_std_f32: n >= 1, stride >= 1")
    var sp = _ptr[DType.float32](src_addr)
    var dp = _ptr[DType.float64](dst_addr)
    var s = Float64(0)
    for i in range(count):
        s += Float64(sp.unsafe_load(i * st))
    var mean = s / Float64(count)
    var q = Float64(0)
    for i in range(count):
        var dv = Float64(sp.unsafe_load(i * st)) - mean
        q += dv * dv
    dp.unsafe_store(0, mean)
    dp.unsafe_store(1, sqrt(q / Float64(count)))
    return 0


def first_seen_i32(codes_addr: Int, rows: Int, kk: Int, enc_addr: Int, counts_addr: Int) raises -> Int:
    """Renumber int32 class codes by first appearance; returns the number of
    classes present."""
    if rows < 0 or kk < 1:
        raise Error("first_seen_i32: n >= 0, k >= 1")
    var cp = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes_addr)
    var ep = _ptr[DType.int32](enc_addr)
    var np = _ptr[DType.int64](counts_addr)
    var rank = List[Int](length=kk, fill=-1)
    var m = 0
    var bad = False
    for j in range(kk):
        np.unsafe_store(j, 0)
    for i in range(rows):
        var c = Int(cp.unsafe_load(i))
        if c < 0 or c >= kk:
            bad = True
            break
        if rank[c] < 0:
            rank[c] = m
            m += 1
        ep.unsafe_store(i, Int32(rank[c]))
        np.unsafe_store(rank[c], np.unsafe_load(rank[c]) + 1)
    if bad:
        raise Error("first_seen_i32: code out of range")
    return m


@always_inline
def _floor_div_pos(a: Int, b: Int) -> Int:
    """floor(a / b) for b > 0 and any a (Python's `//`)."""
    if a >= 0:
        return a // b
    return -((-a + b - 1) // b)


def strat_alloc_i64(counts_addr: Int, kk: Int, K: Int, dst_addr: Int) raises -> Int:
    """StratifiedKFold's allocation table (sklearn _make_test_folds)."""
    if kk < 1 or K < 1:
        raise Error("strat_alloc_i64: k and n_folds must be >= 1")
    var cp = _ptr[DType.int64](counts_addr)
    var dp = _ptr[DType.int64](dst_addr)
    var s = 0
    for c in range(kk):
        var e = s + Int(cp.unsafe_load(c))
        for i in range(K):
            dp.unsafe_store(i * kk + c, Int64(_floor_div_pos(e - 1 - i, K) - _floor_div_pos(s - 1 - i, K)))
        s = e
    return 0


def _stable_order_f64(keys: List[Float64]) -> List[Int]:
    """The indices 0 .. len(keys) - 1 sorted ascending by key, ties in index
    order (a bottom-up merge sort: stable, O(m log m))."""
    var m = len(keys)
    var a = List[Int](capacity=m)
    for i in range(m):
        a.append(i)
    var b = List[Int](length=m, fill=0)
    var width = 1
    while width < m:
        var lo = 0
        while lo < m:
            var mid = min(lo + width, m)
            var hi = min(lo + 2 * width, m)
            var i = lo
            var j = mid
            var t = lo
            while i < mid and j < hi:
                if keys[a[j]] < keys[a[i]]:
                    b[t] = a[j]
                    j += 1
                else:
                    b[t] = a[i]
                    i += 1
                t += 1
            while i < mid:
                b[t] = a[i]
                i += 1
                t += 1
            while j < hi:
                b[t] = a[j]
                j += 1
                t += 1
            lo = hi
        var tmp = a.copy()
        a = b.copy()
        b = tmp^
        width *= 2
    return a^


def group_fold_assign_i32(
    counts_addr: Int, mm: Int, K: Int, pa: Int, dst_addr: Int, sizes_addr: Int,
) raises -> Int:
    """GroupKFold's fold of each of the m groups and each fold's row count
    (scikit-learn 1.9)."""
    if mm < 0 or K < 1:
        raise Error("group_fold_assign_i32: m >= 0 and n_folds >= 1")
    var cp = _ptr[DType.int64](counts_addr)
    var dp = _ptr[DType.int32](dst_addr)
    var sp = _ptr[DType.int64](sizes_addr)
    for f in range(K):
        sp.unsafe_store(f, Int64(0))
    if pa != 0:
        var pp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=pa)
        var start = 0
        for f in range(K):
            var size = mm // K + (1 if f < mm % K else 0)
            for j in range(start, start + size):
                var g = Int(pp.unsafe_load(j))
                if g < 0 or g >= mm:
                    raise Error("group_fold_assign_i32: a permuted group is out of range")
                dp.unsafe_store(g, Int32(f))
            start += size
    else:
        var keys = List[Float64](capacity=mm)
        for g in range(mm):
            keys.append(Float64(Int(cp.unsafe_load(g))))
        var asc = _stable_order_f64(keys)
        var load = List[Int](length=K, fill=0)
        for t in range(mm - 1, -1, -1):
            var g = asc[t]
            var best = 0
            for f in range(1, K):
                if load[f] < load[best]:
                    best = f
            load[best] += Int(cp.unsafe_load(g))
            dp.unsafe_store(g, Int32(best))
    for g in range(mm):
        var f = Int(dp.unsafe_load(g))
        sp.unsafe_store(f, sp.unsafe_load(f) + cp.unsafe_load(g))
    return 0


def strat_group_assign_i32(
    yenc_addr: Int, gidx_addr: Int, pa: Int, dst_addr: Int, sizes_addr: Int,
    nn: Int, kk: Int, mm: Int, K: Int,
) raises -> Int:
    """StratifiedGroupKFold's fold of each of the m groups (scikit-learn 1.9
    `_find_best_fold`). Returns 1 when the largest class has fewer rows than
    n_folds, else 0."""
    if nn < 0 or kk < 1 or mm < 1 or K < 1:
        raise Error("strat_group_assign_i32: bad sizes")
    var yp = _ptr[DType.int32](yenc_addr)
    var gp = _ptr[DType.int32](gidx_addr)
    var dp = _ptr[DType.int32](dst_addr)
    var sp = _ptr[DType.int64](sizes_addr)
    var counts = List[Int](length=kk, fill=0)
    var dist = List[Int](length=mm * kk, fill=0)
    for i in range(nn):
        var c = Int(yp.unsafe_load(i))
        var g = Int(gp.unsafe_load(i))
        if c < 0 or c >= kk or g < 0 or g >= mm:
            raise Error("strat_group_assign_i32: a code is out of range")
        counts[c] += 1
        dist[g * kk + c] += 1
    var most = 0
    for c in range(kk):
        most = max(most, counts[c])
    if most < K:
        return 1
    var order = List[Int](capacity=mm)
    if pa != 0:
        var pp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=pa)
        for j in range(mm):
            order.append(Int(pp.unsafe_load(j)))
    else:
        for j in range(mm):
            order.append(j)
    # -std per group in `order`'s positions, stably sorted ascending
    var keys = List[Float64](capacity=mm)
    for j in range(mm):
        var g = order[j]
        var tot = Float64(0)
        for c in range(kk):
            tot += Float64(dist[g * kk + c])
        var mu = tot / Float64(kk)
        var ss = Float64(0)
        for c in range(kk):
            var dv = Float64(dist[g * kk + c]) - mu
            ss += dv * dv
        keys.append(-sqrt(ss / Float64(kk)))
    var pos = _stable_order_f64(keys)
    var fold_dist = List[Int](length=K * kk, fill=0)
    var fold_n = List[Int](length=K, fill=0)
    var col = List[Float64](length=K, fill=0)
    for t in range(mm):
        var g = order[pos[t]]
        var best = -1
        var best_score = Float64(0)
        var best_n = 0
        for f in range(K):
            var score = Float64(0)
            for c in range(kk):
                var mu = Float64(0)
                for j in range(K):
                    var v = fold_dist[j * kk + c] + (dist[g * kk + c] if j == f else 0)
                    col[j] = Float64(v) / Float64(counts[c])
                    mu += col[j]
                mu = mu / Float64(K)
                var ss = Float64(0)
                for j in range(K):
                    var dv = col[j] - mu
                    ss += dv * dv
                score += sqrt(ss / Float64(K))
            score = score / Float64(kk)
            if best < 0 or score < best_score or (score == best_score and fold_n[f] < best_n):
                best = f
                best_score = score
                best_n = fold_n[f]
        for c in range(kk):
            fold_dist[best * kk + c] += dist[g * kk + c]
            fold_n[best] += dist[g * kk + c]
        dp.unsafe_store(g, Int32(best))
    for f in range(K):
        sp.unsafe_store(f, Int64(fold_n[f]))
    return 0


def strat_group_plan_i32(
    dist_addr: Int, pa: Int, dst_addr: Int, sizes_addr: Int, kk: Int, mm: Int, K: Int,
) raises -> Int:
    """`strat_group_assign_i32` from the group x class table (int64
    dist[g * k + c]): the same statements in the same order after its row
    loop. Returns 1 when the largest class has fewer rows than n_folds."""
    if kk < 1 or mm < 1 or K < 1:
        raise Error("strat_group_plan_i32: bad sizes")
    var tp = _ptr[DType.int64](dist_addr)
    var dp = _ptr[DType.int32](dst_addr)
    var sp = _ptr[DType.int64](sizes_addr)
    var counts = List[Int](length=kk, fill=0)
    var dist = List[Int](length=mm * kk, fill=0)
    for g in range(mm):
        for c in range(kk):
            var v = Int(tp.unsafe_load(g * kk + c))
            dist[g * kk + c] = v
            counts[c] += v
    var most = 0
    for c in range(kk):
        most = max(most, counts[c])
    if most < K:
        return 1
    var order = List[Int](capacity=mm)
    if pa != 0:
        var pp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=pa)
        for j in range(mm):
            order.append(Int(pp.unsafe_load(j)))
    else:
        for j in range(mm):
            order.append(j)
    # -std per group in `order`'s positions, stably sorted ascending
    var keys = List[Float64](capacity=mm)
    for j in range(mm):
        var g = order[j]
        var tot = Float64(0)
        for c in range(kk):
            tot += Float64(dist[g * kk + c])
        var mu = tot / Float64(kk)
        var ss = Float64(0)
        for c in range(kk):
            var dv = Float64(dist[g * kk + c]) - mu
            ss += dv * dv
        keys.append(-sqrt(ss / Float64(kk)))
    var pos = _stable_order_f64(keys)
    var fold_dist = List[Int](length=K * kk, fill=0)
    var fold_n = List[Int](length=K, fill=0)
    var col = List[Float64](length=K, fill=0)
    for t in range(mm):
        var g = order[pos[t]]
        var best = -1
        var best_score = Float64(0)
        var best_n = 0
        for f in range(K):
            var score = Float64(0)
            for c in range(kk):
                var mu = Float64(0)
                for j in range(K):
                    var v = fold_dist[j * kk + c] + (dist[g * kk + c] if j == f else 0)
                    col[j] = Float64(v) / Float64(counts[c])
                    mu += col[j]
                mu = mu / Float64(K)
                var ss = Float64(0)
                for j in range(K):
                    var dv = col[j] - mu
                    ss += dv * dv
                score += sqrt(ss / Float64(K))
            score = score / Float64(kk)
            if best < 0 or score < best_score or (score == best_score and fold_n[f] < best_n):
                best = f
                best_score = score
                best_n = fold_n[f]
        for c in range(kk):
            fold_dist[best * kk + c] += dist[g * kk + c]
            fold_n[best] += dist[g * kk + c]
        dp.unsafe_store(g, Int32(best))
    for f in range(K):
        sp.unsafe_store(f, Int64(fold_n[f]))
    return 0


@always_inline
def _sm64_next(mut state: UInt64) -> UInt64:
    """Python `_splitmix64`: state += golden, then the splitmix64 mix of it."""
    state = state + UInt64(0x9E3779B97F4A7C15)
    var z = state
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def draw_rows_without_replacement_i32(nn: Int, kk: Int, seed: UInt64, dst_addr: Int) raises -> Int:
    """k distinct rows of n, ascending, int32 at dst: a splitmix64 partial
    Fisher-Yates then the first k sorted."""
    if nn < 0 or kk < 0 or kk > nn:
        raise Error("draw_rows_without_replacement_i32: 0 <= k <= n")
    var state = seed
    var dp = _ptr[DType.int32](dst_addr)
    var perm = List[Int](capacity=nn)
    for i in range(nn):
        perm.append(i)
    for i in range(kk):
        var z = _sm64_next(state)
        var j = i + Int(z % UInt64(nn - i))
        var t = perm[i]
        perm[i] = perm[j]
        perm[j] = t
    var mark = List[Bool](length=nn, fill=False)
    for i in range(kk):
        mark[perm[i]] = True
    var at = 0
    for r in range(nn):
        if mark[r]:
            dp.unsafe_store(at, Int32(r))
            at += 1
    return 0


def weighted_draw_rows_i32(w_addr: Int, nn: Int, kk: Int, seed: UInt64, dst_addr: Int) raises -> Int:
    """k rows drawn with replacement in proportion to the float32 weights
    (bisect_right over the binary64 running sum)."""
    if nn < 1 or kk < 0:
        raise Error("weighted_draw_rows_i32: n >= 1, k >= 0")
    var state = seed
    var wp = _ptr[DType.float32](w_addr)
    var dp = _ptr[DType.int32](dst_addr)
    var cum = List[Float64](capacity=nn)
    var acc = Float64(0)
    for i in range(nn):
        acc += Float64(wp.unsafe_load(i))
        cum.append(acc)
    for t in range(kk):
        var z = _sm64_next(state)
        var target = Float64(z >> 11) * 1.1102230246251565e-16 * acc
        # bisect_right: the first index whose running sum exceeds target
        var lo = 0
        var hi = nn
        while lo < hi:
            var mid = (lo + hi) // 2
            if target < cum[mid]:
                hi = mid
            else:
                lo = mid + 1
        dp.unsafe_store(t, Int32(min(lo, nn - 1)))
    return 0


def weighted_pick_i32(m_addr: Int, n: Int, j: Int, kk: Int, state_addr: Int, dst_addr: Int) raises -> Int:
    """IterativeImputer's n_nearest_features draw for feature `j`; returns
    the number of picks written (ascending, int32)."""
    if n < 1 or j < 0 or j >= n or kk < 0:
        raise Error("weighted_pick_i32: bad sizes")
    var mp = _ptr[DType.float32](m_addr)
    var sp = _ptr[DType.uint64](state_addr)
    var dp = _ptr[DType.int32](dst_addr)
    var state = sp.unsafe_load(0)
    var chosen = List[Bool](length=n, fill=False)
    for _ in range(kk):
        var tot = Float64(0)
        var last = -1
        for a in range(n):
            var w = Float64(mp.unsafe_load(a * n + j))
            if not chosen[a] and w > 0:
                tot += w
                last = a
        if last < 0:
            raise Error("weighted_pick_i32: no feature left to draw")
        var u = Float64(_sm64_next(state) >> 11) * 1.1102230246251565e-16 * tot
        var pick = last
        var cum = Float64(0)
        for a in range(n):
            var w = Float64(mp.unsafe_load(a * n + j))
            if not chosen[a] and w > 0:
                cum += w
                if cum > u:
                    pick = a
                    break
        chosen[pick] = True
    sp.unsafe_store(0, state)
    var at = 0
    for a in range(n):
        if chosen[a]:
            dp.unsafe_store(at, Int32(a))
            at += 1
    return at


def ocsvm_alpha_init_f32(c_addr: Int, mm: Int, v: Float64, weighted: Bool, dst_addr: Int) raises -> Int:
    """libsvm's solve_one_class start: alpha_i = min(C_i, nu_l) while
    nu_l > 0, nu_l reduced by each (the rest 0)."""
    if mm < 0:
        raise Error("ocsvm_alpha_init_f32: m >= 0")
    var cp = _ptr[DType.float32](c_addr)
    var dp = _ptr[DType.float32](dst_addr)
    var nl = v * Float64(mm)
    if weighted:
        nl = Float64(0)
        for i in range(mm):
            nl += Float64(cp.unsafe_load(i)) * v
    for i in range(mm):
        var a = Float64(0)
        if nl > 0:
            a = min(Float64(cp.unsafe_load(i)), nl)
            nl -= a
        dp.unsafe_store(i, Float32(a))
    return 0


def strat_fold_assign_i32(
    enc_addr: Int, rows: Int, kk: Int, K: Int, alloc_addr: Int, pa: Int, counts_addr: Int, dst_addr: Int,
) raises -> Int:
    """StratifiedKFold's test fold of every row (sklearn _make_test_folds)."""
    if rows < 1 or kk < 1 or K < 1:
        raise Error("strat_fold_assign_i32: n, k, n_folds >= 1")
    var ep = _ptr[DType.int32](enc_addr)
    var ap = _ptr[DType.int64](alloc_addr)
    var cp = _ptr[DType.int64](counts_addr)
    var dp = _ptr[DType.int32](dst_addr)
    var off = List[Int](length=kk + 1, fill=0)
    for c in range(kk):
        off[c + 1] = off[c] + Int(cp.unsafe_load(c))
    if off[kk] != rows:
        raise Error("strat_fold_assign_i32: counts do not sum to n")
    var seq = List[Int32](length=rows, fill=0)
    for c in range(kk):
        var at = off[c]
        for f in range(K):
            for _ in range(Int(ap.unsafe_load(f * kk + c))):
                if at >= off[c + 1]:
                    raise Error("strat_fold_assign_i32: alloc exceeds the class count")
                seq[at] = Int32(f)
                at += 1
    var seen = List[Int](length=kk, fill=0)
    var pp = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=pa)
    for i in range(rows):
        var c = Int(ep.unsafe_load(i))
        if c < 0 or c >= kk:
            raise Error("strat_fold_assign_i32: class out of range")
        var j = seen[c]
        seen[c] = j + 1
        var src = j
        if pa != 0:
            src = Int(pp.unsafe_load(off[c] + j))
        var f = seq[off[c] + src]
        comptime if HOTPATH_SABOTAGE:
            f = Int32((Int(f) + 1) % K)
        dp.unsafe_store(i, f)
    return 0
