"""The GreedyLogSum float-column border build, one definition for the device
and the host column (lane hr2-gbdt-host, 2026-10-02).

WHAT MOVED. `train._quantize_training_columns` drew the border subsample on
the host (a serial `TRandom` Fisher-Yates / rejection draw), gathered every
float column through it on host threads, and ran `calc_quantization` per
column on host threads. This file is the whole build on the device:

  1. THE SUBSAMPLE, PARALLEL. Row `i < sn` of the sample is `perm(i)` for a
     keyed Feistel bijection on `[0, 2^bits)` cycle-walked into `[0, n)`
     (`border_sample_row`). A bijection draws WITHOUT REPETITION, the
     property their `SampleIndices` provides; every index is computed
     independently, so the draw is one GPU thread per sample row. (The set
     drawn differs from the old `TRandom` draw: same-bits-within-a-version
     is the contract, not old bits.)
  2. THE KEYS. `border_keys_kernel`: per (column, sample row) the value,
     flushed (`ftz`, DEVIATION 5900's flushed search), as the monotone
     sortable twiddle; a NaN becomes `0xFFFFFFFF`, which no value maps to,
     so NaNs sort last and are counted per column (their `filterNans`).
  3. THE SORT. `launch_segmented_radix_sort`, every column of the chunk at
     once (one segment per column).
  4. THE SEARCH. `gls_column`: CatBoost's GreedyLogSum (`binarization.mojo`
     `best_split`, libc++ heap semantics included) over the sorted keys,
     one GPU thread per column, its heap in global scratch. The split score
     `log(l + 1e-8) + log(r + 1e-8) - log(n + 1e-8)` reads a table of
     `log(w + 1e-8)` built by one thread per `w` in soft binary64
     (`checks/soft_f64.mojo`), so every vendor and the host column score
     the same words; the sums are soft binary64 too.
  5. The midpoints, flushed and with a pinned product (no contraction),
     sorted by the twiddled key and de-duplicated.

The host column (`gbdt/host/gbdt_oracle.mojo::gbdt_host_grid`) calls the
same `border_sample_row`, `border_key`, `gls_log_table_entry` and
`gls_column` on host memory, so the grids agree by construction.
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
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


# ---- the kernels -------------------------------------------------------------


def gls_log_table_kernel(logtab: MutPointer[UInt64, MutAnyOrigin], n: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < Int(n):
        logtab[i] = gls_log_table_entry(i)
        i += stride


def border_sample_kernel(
    idx: MutPointer[UInt32, MutAnyOrigin], sn: Int32, n: Int32, key: UInt64
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < Int(sn):
        idx[i] = UInt32(border_sample_row(i, Int(n), key))
        i += stride


def border_keys_kernel(
    cols: MutPointer[Float32, MutAnyOrigin],
    idx: MutPointer[UInt32, MutAnyOrigin],
    keys: MutPointer[UInt32, MutAnyOrigin],
    vals: MutPointer[UInt32, MutAnyOrigin],
    nan_in_sample: MutPointer[Int32, MutAnyOrigin],
    nan_in_column: MutPointer[Int32, MutAnyOrigin],
    n_cols: Int32,
    n_rows: Int32,
    sn: Int32,
    sampled: Int32,
):
    """Grid-stride over `n_cols * max(sn, n_rows)`: the sample's keys
    (`keys[c * sn + i]`, NaN counted per column) and, on the sampled path,
    the full column's NaN flag."""
    var nc = Int(n_cols)
    var n = Int(n_rows)
    var s = Int(sn)
    var span = s
    if sampled != 0 and n > s:
        span = n
    var total = nc * span
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < total:
        var c = i // span
        var j = i - c * span
        if j < s:
            var row = Int(idx[j]) if sampled != 0 else j
            var v = cols[c * n + row]
            var k = border_key(v)
            keys[c * s + j] = k
            vals[c * s + j] = UInt32(j)
            if k == GLS_NAN_KEY:
                _ = Atomic.fetch_add(nan_in_sample + c, Int32(1))
        if sampled != 0 and j < n:
            var v2 = cols[c * n + j]
            if v2 != v2:
                _ = Atomic.fetch_add(nan_in_column + c, Int32(1))
        i += stride


def gls_columns_kernel(
    keys: MutPointer[UInt32, MutAnyOrigin],
    sn: Int32,
    n_cols: Int32,
    valid: MutPointer[Int32, MutAnyOrigin],
    budget: MutPointer[Int32, MutAnyOrigin],
    logtab: MutPointer[UInt64, MutAnyOrigin],
    h_start: MutPointer[Int32, MutAnyOrigin],
    h_end: MutPointer[Int32, MutAnyOrigin],
    h_split: MutPointer[Int32, MutAnyOrigin],
    h_score: MutPointer[UInt64, MutAnyOrigin],
    heap_cap: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
    out_cap: Int32,
    counts: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per column of the chunk: `gls_column` over its sorted
    keys' first `valid[c]` entries with `budget[c]` borders."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= Int(n_cols):
        return
    var hc = Int(heap_cap)
    var oc = Int(out_cap)
    counts[c] = Int32(
        gls_column(
            keys + c * Int(sn), Int(valid[c]), Int(budget[c]), logtab,
            h_start + c * hc, h_end + c * hc, h_split + c * hc,
            h_score + c * hc, dst + c * oc,
        )
    )


#: `gls_budget_kernel`'s mode word for a NaN under `nan_mode=Forbidden`
comptime GLS_MODE_REFUSED = Int32(-1)


def gls_budget_kernel(
    nan_in_sample: MutPointer[Int32, MutAnyOrigin],
    nan_in_column: MutPointer[Int32, MutAnyOrigin],
    sn: Int32,
    sampled: Int32,
    n_cols: Int32,
    border_count: Int32,
    nan_mode_option: Int32,
    forbidden: Int32,
    valid: MutPointer[Int32, MutAnyOrigin],
    budget: MutPointer[Int32, MutAnyOrigin],
    mode: MutPointer[Int32, MutAnyOrigin],
):
    """Per column: `ComputeNanMode` over the column's NaN flag (the full
    column on the sampled path), the non-NaN border budget, and the count
    of non-NaN sorted keys."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= Int(n_cols):
        return
    var ns = nan_in_sample[c]
    var has_nan = ns > 0
    if sampled != 0:
        has_nan = nan_in_column[c] > 0
    var m = forbidden
    if has_nan:
        if nan_mode_option == forbidden:
            m = GLS_MODE_REFUSED
        else:
            m = nan_mode_option
    var b = border_count
    if m != forbidden:
        b -= 1
    valid[c] = sn - ns
    budget[c] = b
    mode[c] = m
