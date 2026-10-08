"""The CPU-safe half of `gbdt/grid_creator/gls_borders.mojo` (lane
rehearsal-suite-green, 2026-10-08): GreedyLogSum's constants and per-column
scalar functions, moved verbatim so the host oracle
(`gbdt/host/gbdt_oracle.mojo`) imports no `std.gpu` module. The device file
imports every name back.
"""
from std.memory import bitcast

from checks.numerics import ftz, pinned_mul_f32
from checks.soft_f64 import (
    sf64_add,
    sf64_from_int,
    sf64_log,
    sf64_lt,
    sf64_sub,
)

comptime GLS_EPS8 = UInt64(0x3E45798EE2308C3A)  # 1e-8
comptime GLS_NO_SCORE = UInt64(0xFFE1CCF385EBC8A0)  # -1.0e308
comptime GLS_NAN_KEY = UInt32(0xFFFFFFFF)
comptime GLS_BLOCK = 256


# ---- 1. the parallel subsample ------------------------------------------


@always_inline
def _mix64(x_in: UInt64) -> UInt64:
    var x = x_in
    x = (x ^ (x >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> 27)) * UInt64(0x94D049BB133111EB)
    return x ^ (x >> 31)


def border_sample_row(i: Int, n: Int, key: UInt64) -> Int:
    """Row `i` of the sample: a 4-round Feistel bijection on `[0, 2^(2h))`
    (`2^(2h) >= n`), cycle-walked until the value lands in `[0, n)`. Distinct
    `i` give distinct rows. `key` is `generate_seed_for_borders(seed)`."""
    var h = 1
    while (1 << (2 * h)) < n:
        h += 1
    var mask = (UInt64(1) << UInt64(h)) - 1
    var x = UInt64(i)
    while True:
        var l = x >> UInt64(h)
        var r = x & mask
        for rnd in range(4):
            var f = _mix64(r ^ (key + UInt64(rnd) * UInt64(0x9E3779B97F4A7C15))) & mask
            var t = l ^ f
            l = r
            r = t
        x = (l << UInt64(h)) | r
        if x < UInt64(n):
            return Int(x)


# ---- 2. the keys ----------------------------------------------------------


@always_inline
def border_key(v: Float32) -> UInt32:
    """The flushed value as the sortable twiddle; a NaN is `GLS_NAN_KEY`."""
    if v != v:
        return GLS_NAN_KEY
    var bits = bitcast[DType.uint32](ftz(v))
    if (bits & UInt32(0x80000000)) != UInt32(0):
        return ~bits
    return bits | UInt32(0x80000000)


@always_inline
def key_value(k: UInt32) -> Float32:
    if (k & UInt32(0x80000000)) != UInt32(0):
        return bitcast[DType.float32](k & UInt32(0x7FFFFFFF))
    return bitcast[DType.float32](~k)


# ---- 4. the search ----------------------------------------------------------


@always_inline
def gls_log_table_entry(w: Int) -> UInt64:
    """`log(w + 1e-8)` in soft binary64: `-Penalty<MaxSumLog>(w)`."""
    return sf64_log(sf64_add(sf64_from_int(w), GLS_EPS8))


@always_inline
def _score(
    logtab: MutPointer[UInt64, MutAnyOrigin], start: Int, end: Int, p: Int
) -> UInt64:
    """`CalcSplitScore`: `left + right - curr`."""
    if p == start or p == end:
        return GLS_NO_SCORE
    return sf64_sub(
        sf64_add(logtab[p - start], logtab[end - p]), logtab[end - start]
    )


@always_inline
def _best(
    keys: MutPointer[UInt32, MutAnyOrigin],
    logtab: MutPointer[UInt64, MutAnyOrigin],
    start: Int,
    end: Int,
) -> Tuple[Int, UInt64]:
    """`UpdateBestSplitProperties`: the first index holding the midpoint's
    value and the first past it, by binary search over the sorted keys
    (key order is value order, `-0` and `+0` apart)."""
    var mid = start + (end - start) // 2
    var mk = keys[mid]
    var lb = start
    var hi = mid
    while lb < hi:
        var m = lb + (hi - lb) // 2
        if keys[m] < mk:
            lb = m + 1
        else:
            hi = m
    var ub = mid
    var hi2 = end
    while ub < hi2:
        var m2 = ub + (hi2 - ub) // 2
        if keys[m2] <= mk:
            ub = m2 + 1
        else:
            hi2 = m2
    var sl = _score(logtab, start, end, lb)
    var sr = _score(logtab, start, end, ub)
    if not sf64_lt(sl, sr):
        return (lb, sl)
    return (ub, sr)


def gls_column(
    keys: MutPointer[UInt32, MutAnyOrigin],
    n: Int,
    max_borders: Int,
    logtab: MutPointer[UInt64, MutAnyOrigin],
    h_start: MutPointer[Int32, MutAnyOrigin],
    h_end: MutPointer[Int32, MutAnyOrigin],
    h_split: MutPointer[Int32, MutAnyOrigin],
    h_score: MutPointer[UInt64, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
) -> Int:
    """GreedyLogSum over `n` sorted NaN-free keys, at most `max_borders`
    borders into `dst` (ascending, distinct). The heap holds up to
    `max_borders + 2` bins in the four scratch planes."""
    if n == 0 or max_borders <= 0:
        return 0
    var size = 0
    # push the root
    var r0 = _best(keys, logtab, 0, n)
    h_start[0] = 0
    h_end[0] = Int32(n)
    h_split[0] = Int32(r0[0])
    h_score[0] = r0[1]
    size = 1
    while size <= max_borders:
        var ts = Int(h_start[0])
        var te = Int(h_end[0])
        var tp = Int(h_split[0])
        if ts == tp or te == tp:
            break  # `!CanSplit()`
        # pop: libc++ pop_heap, the last element re-seated from the root
        size -= 1
        var vs = h_start[size]
        var ve = h_end[size]
        var vp = h_split[size]
        var vsc = h_score[size]
        if size > 0:
            var hole = 0
            while True:
                var child = 2 * hole + 1
                if child >= size:
                    break
                if child + 1 < size and sf64_lt(h_score[child], h_score[child + 1]):
                    child += 1
                if sf64_lt(h_score[child], vsc):
                    break
                h_start[hole] = h_start[child]
                h_end[hole] = h_end[child]
                h_split[hole] = h_split[child]
                h_score[hole] = h_score[child]
                hole = child
            h_start[hole] = vs
            h_end[hole] = ve
            h_split[hole] = vp
            h_score[hole] = vsc
        # the two halves, left first (the push order is load bearing)
        for half in range(2):
            var s: Int
            var e: Int
            if half == 0:
                s = ts
                e = tp
            else:
                s = tp
                e = te
            var b = _best(keys, logtab, s, e)
            # push: libc++ push_heap, sift up past STRICTLY smaller parents
            var hole2 = size
            size += 1
            while hole2 > 0:
                var parent = (hole2 - 1) // 2
                if sf64_lt(h_score[parent], b[1]):
                    h_start[hole2] = h_start[parent]
                    h_end[hole2] = h_end[parent]
                    h_split[hole2] = h_split[parent]
                    h_score[hole2] = h_score[parent]
                    hole2 = parent
                else:
                    break
            h_start[hole2] = Int32(s)
            h_end[hole2] = Int32(e)
            h_split[hole2] = Int32(b[0])
            h_score[hole2] = b[1]
    # midpoints of every bin but the first, flushed, no contraction
    var count = 0
    for i in range(size):
        var s2 = Int(h_start[i])
        if s2 == 0:
            continue
        var below = ftz(pinned_mul_f32(Float32(0.5), key_value(keys[s2 - 1])))
        var above = ftz(pinned_mul_f32(Float32(0.5), key_value(keys[s2])))
        var b2 = ftz(below + above)
        # insertion by twiddled key, so the order never depends on a float
        # compare of -0 and +0
        var kb = border_key(b2)
        var j = count
        while j > 0 and border_key(dst[j - 1]) > kb:
            dst[j] = dst[j - 1]
            j -= 1
        dst[j] = b2
        count += 1
    # de-duplicate (float equality, as `best_split` did)
    var w = 0
    for i in range(count):
        if i == 0 or dst[i] != dst[w - 1]:
            dst[w] = dst[i]
            w += 1
    return w
