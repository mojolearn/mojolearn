"""The six non-default numeric border types on the device, one definition for
the device and the host column (cpu-gpu-cleanup w2-trees, 2026-10-02).

WHAT MOVED. `train._quantize_training_columns` built the Median, Uniform,
UniformAndQuantiles, MaxLogSum, MinEntropy and GreedyMinEntropy grids on
the host: a serial `TRandom` subsample, a host-thread gather (`_draw_task`)
and `select_borders` per column on host threads (`_dp_task`). They now ride
GreedyLogSum's device pipeline (`gls_borders.mojo`: the Feistel subsample,
the flushed twiddled keys, the segmented sort, `ComputeNanMode`) and differ
only in the per-column search, which runs here, one GPU thread per column
of a chunk, on the sorted keys:

  * Median / Uniform / UniformAndQuantiles: `simple_column`, their
    `GenerateMedianBorders`, `RegularBorder` and uniform values
    (`binarization.mojo` `_median_borders`, `_regular_border`,
    `_uniform_value`) over the sorted keys, every float32 product pinned and
    the uniform step through `portable_divf` (one correctly rounded
    division on every column);
  * GreedyMinEntropy: `gls_column` itself over a score table of
    `-(w * log(w + 1e-8))` (`TGreedyBinarizer<MinEntropy>` is
    `TGreedyBinarizer<MaxSumLog>` with the other penalty, the same heap);
  * MaxLogSum / MinEntropy: `exact_column`, `_exact_best_split`'s E_RLM2
    dynamic program statement for statement, every double operation in soft
    binary64 (`checks/soft_f64.mojo`; the Apple GPU has no float64) and the
    penalty read from a table (`border_table_entry`): the DP's penalty
    arguments are differences of integer count sums, so `Penalty(w)` for
    `w` in `[0, sample]` is every value it can ask for.

Every result then goes through `finish_borders` (`SetQuantization`: -0.0
written as +0.0, sorted, de-duplicated; the sabotage arm as
`select_borders`). The host column (`gbdt/host/gbdt_oracle.mojo`) calls the
same functions on host memory, so the grids agree by construction.

BITS. The subsample is GreedyLogSum's (the old `TRandom` draw and its NaN
seed into the sample are gone), so the six types' grids move once on the
sampled path; the searches themselves are the old host arithmetic
(`portable_log64` is `sf64_log` statement for statement, and IEEE
add/mul/div are what soft binary64 returns).
"""
from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from checks.numerics import ftz, pinned_mul_f32, portable_divf
from checks.soft_f64 import (
    sf64_add,
    sf64_from_int,
    sf64_gt,
    sf64_lt,
    sf64_mul,
    sf64_neg,
)
from gbdt.grid_creator.binarization import (
    BORDER_TYPE_GREEDY_LOG_SUM,
    BORDER_TYPE_GREEDY_MIN_ENTROPY,
    BORDER_TYPE_MAX_LOG_SUM,
    BORDER_TYPE_MEDIAN,
    BORDER_TYPE_MIN_ENTROPY,
    BORDER_TYPE_UNIFORM,
    BORDER_TYPE_UNIFORM_AND_QUANTILES,
    BORDER_TYPES_SABOTAGE,
)
from gbdt.grid_creator.gls_borders import (
    border_key,
    gls_column,
    gls_log_table_entry,
    key_value,
)

comptime BT_EPS12 = UInt64(0x3D719799812DEA11)  # 1e-12, their `Eps`


# ---- the tables ---------------------------------------------------------------


@always_inline
def border_table_entry(w: Int, border_type: Int) -> UInt64:
    """Entry `w` of the type's table, soft binary64:
    GreedyLogSum `log(w + 1e-8)` (the score `-Penalty<MaxSumLog>`),
    GreedyMinEntropy `-(w * log(w + 1e-8))` (the score `-Penalty<MinEntropy>`),
    MaxLogSum `-log(w + 1e-8)` (`Penalty<MaxSumLog>`),
    MinEntropy `w * log(w + 1e-8)` (`Penalty<MinEntropy>`, the product
    unfused as `_penalty_min_entropy`'s `@no_inline` keeps it). The other
    types read no table (0)."""
    var lg = gls_log_table_entry(w)
    if border_type == BORDER_TYPE_GREEDY_LOG_SUM:
        return lg
    if border_type == BORDER_TYPE_MAX_LOG_SUM:
        return sf64_neg(lg)
    if border_type == BORDER_TYPE_MIN_ENTROPY:
        return sf64_mul(sf64_from_int(w), lg)
    if border_type == BORDER_TYPE_GREEDY_MIN_ENTROPY:
        return sf64_neg(sf64_mul(sf64_from_int(w), lg))
    return UInt64(0)


def border_table_kernel(tab: MutPointer[UInt64, MutAnyOrigin], n: Int32, border_type: Int32):
    """One thread per entry, grid-stride."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < Int(n):
        tab[i] = border_table_entry(i, Int(border_type))
        i += stride


# ---- SetQuantization ------------------------------------------------------------


def finish_borders(dst: MutPointer[Float32, MutAnyOrigin], count: Int) -> Int:
    """`select_borders`' tail: -0.0 as +0.0, ascending (insertion by the
    twiddled key, which is float order once no -0.0 is left), duplicates
    merged, then the sabotage arm."""
    for i in range(count):
        if dst[i] == Float32(0.0):
            dst[i] = Float32(0.0)
    for i in range(1, count):
        var v = dst[i]
        var kv = border_key(v)
        var j = i
        while j > 0 and border_key(dst[j - 1]) > kv:
            dst[j] = dst[j - 1]
            j -= 1
        dst[j] = v
    var w = 0
    for i in range(count):
        if i == 0 or dst[i] != dst[w - 1]:
            dst[w] = dst[i]
            w += 1
    comptime if BORDER_TYPES_SABOTAGE:
        if w > 0:
            var mid = w // 2
            for i in range(mid, w - 1):
                dst[i] = dst[i + 1]
            w -= 1
    return w


# ---- Median, Uniform, UniformAndQuantiles ----------------------------------------


@always_inline
def _lower_bound(keys: MutPointer[UInt32, MutAnyOrigin], n: Int, value: Float32) -> Int:
    """`LowerBound` by float compare over the sorted keys (-0.0 and +0.0
    sit together, so `v < value` holds on a prefix)."""
    var lo = 0
    var hi = n
    while lo < hi:
        var m = lo + (hi - lo) // 2
        if key_value(keys[m]) < value:
            lo = m + 1
        else:
            hi = m
    return lo


def _regular_border(keys: MutPointer[UInt32, MutAnyOrigin], n: Int, border: Float32) -> Float32:
    """`binarization._regular_border` over the sorted keys, products pinned."""
    var lb = _lower_bound(keys, n, border)
    if lb == n:
        var back = key_value(keys[n - 1])
        return max(ftz(pinned_mul_f32(Float32(2.0), back)), ftz(back + Float32(1.0)))
    if lb == 0:
        var front = key_value(keys[0])
        return min(ftz(pinned_mul_f32(Float32(0.5), front)), ftz(pinned_mul_f32(Float32(2.0), front)))
    var a = key_value(keys[lb])
    var b = key_value(keys[lb - 1])
    var res = ftz(pinned_mul_f32(ftz(a + b), Float32(0.5)))
    if res == a:
        res = b
    return res


def _median_borders(
    keys: MutPointer[UInt32, MutAnyOrigin], n: Int, count: Int,
    dst: MutPointer[Float32, MutAnyOrigin], start: Int,
) -> Int:
    """`binarization._median_borders`; returns the new fill of `dst`."""
    var w = start
    if n == 0:
        return w
    var first = key_value(keys[0])
    if first == key_value(keys[n - 1]):
        return w
    for i in range(count):
        var i1 = (i + 1) * n // (count + 1)
        if i1 > n - 1:
            i1 = n - 1
        var val1 = key_value(keys[i1])
        if val1 != first:
            dst[w] = _regular_border(keys, n, val1)
            w += 1
    return w


@always_inline
def _uniform_value(min_value: Float32, max_value: Float32, i: Int, parts: Int) -> Float32:
    """`binarization._uniform_value`: float32, one rounding per operation,
    each flushed; the product pinned, the division `portable_divf`."""
    var span = ftz(max_value - min_value)
    var scaled = ftz(pinned_mul_f32(Float32(i + 1), span))
    var step = portable_divf(scaled, Float32(parts))
    return ftz(min_value + step)


def simple_column(
    keys: MutPointer[UInt32, MutAnyOrigin], n: Int, max_borders: Int, border_type: Int,
    dst: MutPointer[Float32, MutAnyOrigin],
) -> Int:
    """Median, Uniform or UniformAndQuantiles over `n` sorted NaN-free keys
    (`select_borders`' three arms), then `finish_borders`."""
    if n == 0 or max_borders <= 0:
        return 0
    var lo = key_value(keys[0])
    var hi = key_value(keys[n - 1])
    var w = 0
    if border_type == BORDER_TYPE_MEDIAN:
        w = _median_borders(keys, n, max_borders, dst, 0)
    elif border_type == BORDER_TYPE_UNIFORM:
        if lo == hi:
            return 0
        for i in range(max_borders):
            dst[w] = _uniform_value(lo, hi, i, max_borders + 1)
            w += 1
    else:
        if lo == hi:
            return 0
        var half = max_borders // 2
        w = _median_borders(keys, n, max_borders - half, dst, 0)
        for i in range(half):
            dst[w] = _regular_border(keys, n, _uniform_value(lo, hi, i, half + 1))
            w += 1
    return finish_borders(dst, w)


# ---- MaxLogSum, MinEntropy: the exact dynamic program ------------------------------


@always_inline
def _pen(
    tab: MutPointer[UInt64, MutAnyOrigin], sw: MutPointer[Int32, MutAnyOrigin], a: Int, b: Int
) -> UInt64:
    """`Penalty(sweights[a] - sweights[b])`: the counts are integers."""
    return tab[Int(sw[a]) - Int(sw[b])]


def exact_column(
    keys: MutPointer[UInt32, MutAnyOrigin],
    n: Int,
    max_borders: Int,
    tab: MutPointer[UInt64, MutAnyOrigin],
    f64s: MutPointer[UInt64, MutAnyOrigin],
    i32s: MutPointer[Int32, MutAnyOrigin],
    uniq: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
) -> Int:
    """`binarization._exact_best_split` (their `TExactBinarizer`, mode
    E_RLM2, `flush=True`) over `n` sorted NaN-free keys, in soft binary64.
    Scratch: `f64s` holds 4 * n words (current error, previous error, e1,
    e2), `i32s` holds 3 * n + max(0, max_borders - 1) * n + max_borders
    (sweights, bs1, bs2, best solutions, thresholds), `uniq` n floats."""
    if n == 0 or max_borders <= 0:
        return 0
    # their unique values plus per-value weights, as running counts
    var sw = i32s
    var wsize = 0
    for i in range(n):
        var v = key_value(keys[i])
        if i > 0 and v == key_value(keys[i - 1]):
            sw[wsize - 1] = sw[wsize - 1] + 1
        else:
            uniq[wsize] = v
            var run = Int32(1)
            if wsize > 0:
                run = sw[wsize - 1] + 1
            sw[wsize] = run
            wsize += 1
    var bins = max_borders + 1
    var thr = i32s + 3 * n + max(0, max_borders - 1) * n
    if wsize <= bins:
        for i in range(wsize - 1):
            thr[i] = Int32(i)
        for i in range(wsize - 1, bins - 1):
            thr[i] = Int32(wsize - 1)
        return _thresholds_to_borders(thr, bins - 1, uniq, wsize, dst)

    var dsize = wsize - bins + 1
    var cur = f64s
    var prev = f64s + n
    var e1 = f64s + 2 * n
    var e2 = f64s + 3 * n
    var bs1 = i32s + n
    var bs2 = i32s + 2 * n
    var best = i32s + 3 * n
    for i in range(dsize):
        cur[i] = tab[Int(sw[i])]
        bs1[i] = 0
        bs2[i] = 0
        e1[i] = UInt64(0)
        e2[i] = UInt64(0)

    for l in range(bins - 2):
        for i in range(dsize):
            prev[i] = cur[i]

        # their "First forward loop"
        var fi = 0
        for j in range(dsize):
            var best_error = sf64_add(prev[fi], _pen(tab, sw, l + j + 1, l + fi))
            fi += 1
            while fi <= j:
                var new_error = sf64_add(prev[fi], _pen(tab, sw, l + j + 1, l + fi))
                if sf64_gt(new_error, sf64_add(best_error, BT_EPS12)):
                    break
                best_error = new_error
                fi += 1
            fi -= 1
            bs1[j] = Int32(fi)
            e1[j] = best_error

        # their "First inverted loop"
        var vi = 0
        for j in range(dsize):
            if j > vi:
                vi = j
            var maxi = dsize - Int(bs1[dsize - j - 1]) - 1
            if vi + 1 >= maxi:
                bs2[dsize - j - 1] = bs1[dsize - j - 1]
                e2[dsize - j - 1] = e1[dsize - j - 1]
                vi = maxi
                continue
            var best_error = e1[dsize - j - 1]
            while vi + 1 < maxi:
                var new_error = sf64_add(
                    prev[dsize - vi - 1], _pen(tab, sw, l + dsize - j, l + dsize - vi - 1)
                )
                if sf64_lt(sf64_add(new_error, BT_EPS12), best_error):
                    best_error = new_error
                    break
                vi += 1
            if vi + 1 >= maxi:
                vi = maxi
            else:
                vi += 1
                while vi + 1 < maxi:
                    var new_error = sf64_add(
                        prev[dsize - vi - 1], _pen(tab, sw, l + dsize - j, l + dsize - vi - 1)
                    )
                    if sf64_gt(new_error, sf64_add(best_error, BT_EPS12)):
                        break
                    best_error = new_error
                    vi += 1
                vi -= 1
            bs2[dsize - j - 1] = Int32(dsize - vi - 1)
            e2[dsize - j - 1] = best_error

        # their reconciliation: rebuild until the two bounds meet
        for k in range(dsize):
            while Int(bs1[k]) + 1 < Int(bs2[k]):
                var maxj = dsize

                # "Forward loop"
                var ri = Int(bs1[k]) + 2
                var j = k
                while j < maxj:
                    if ri <= Int(bs1[j]):
                        maxj = j
                        break
                    var maxi = Int(bs2[j])
                    if ri + 1 >= maxi:
                        ri = maxi
                        bs1[j] = Int32(ri)
                        e1[j] = e2[j]
                        j += 1
                        continue
                    var best_error = e2[j]
                    while ri + 1 < maxi:
                        var new_error = sf64_add(prev[ri], _pen(tab, sw, l + j + 1, l + ri))
                        if sf64_lt(sf64_add(new_error, BT_EPS12), best_error):
                            best_error = new_error
                            break
                        ri += 1
                    if ri + 1 >= maxi:
                        ri = maxi
                    else:
                        ri += 1
                        while ri + 1 < maxi:
                            var new_error = sf64_add(prev[ri], _pen(tab, sw, l + j + 1, l + ri))
                            if sf64_gt(new_error, sf64_add(best_error, BT_EPS12)):
                                break
                            best_error = new_error
                            ri += 1
                        ri -= 1
                    bs1[j] = Int32(ri)
                    e1[j] = best_error
                    j += 1

                # "Inverted loop"
                var j1 = dsize - maxj
                var j2 = dsize - k
                var qi = dsize - Int(bs2[dsize - j1 - 1]) - 1 + 2
                var jj = j1
                while jj < j2:
                    var maxi = dsize - Int(bs1[dsize - jj - 1]) - 1
                    if qi + 1 >= maxi:
                        bs2[dsize - jj - 1] = bs1[dsize - jj - 1]
                        e2[dsize - jj - 1] = e1[dsize - jj - 1]
                        qi = maxi
                        jj += 1
                        continue
                    var best_error = e1[dsize - jj - 1]
                    while qi + 1 < maxi:
                        var new_error = sf64_add(
                            prev[dsize - qi - 1], _pen(tab, sw, l + dsize - jj, l + dsize - qi - 1)
                        )
                        if sf64_lt(sf64_add(new_error, BT_EPS12), best_error):
                            best_error = new_error
                            break
                        qi += 1
                    if qi + 1 >= maxi:
                        qi = maxi
                    else:
                        qi += 1
                        while qi + 1 < maxi:
                            var new_error = sf64_add(
                                prev[dsize - qi - 1], _pen(tab, sw, l + dsize - jj, l + dsize - qi - 1)
                            )
                            if sf64_gt(new_error, sf64_add(best_error, BT_EPS12)):
                                break
                            best_error = new_error
                            qi += 1
                        qi -= 1
                    bs2[dsize - jj - 1] = Int32(dsize - qi - 1)
                    e2[dsize - jj - 1] = best_error
                    jj += 1

            # "Everything is fine now!"
            best[l * dsize + k] = bs1[k]
            cur[k] = e1[k]

    # their "Last match": `<`, the FIRST index wins a tie
    var l_last = bins - 2
    var j_last = dsize - 1
    var best_index = 0
    var best_error = sf64_add(cur[0], _pen(tab, sw, l_last + j_last + 1, l_last))
    for i in range(1, j_last + 1):
        var new_error = sf64_add(cur[i], _pen(tab, sw, l_last + j_last + 1, l_last + i))
        if sf64_lt(new_error, best_error):
            best_error = new_error
            best_index = i

    thr[bins - 2] = Int32(best_index)
    var l_back = bins - 2
    while l_back > 0:
        best_index = Int(best[(l_back - 1) * dsize + best_index])
        thr[l_back - 1] = Int32(best_index)
        l_back -= 1
    # their "Adjust", undoing the `l` offset baked into dsize
    for i in range(bins - 1):
        thr[i] = thr[i] + Int32(i)
    return _thresholds_to_borders(thr, bins - 1, uniq, wsize, dst)


def _thresholds_to_borders(
    thr: MutPointer[Int32, MutAnyOrigin], n_thr: Int,
    uniq: MutPointer[Float32, MutAnyOrigin], wsize: Int,
    dst: MutPointer[Float32, MutAnyOrigin],
) -> Int:
    """`binarization._thresholds_to_borders` (flush): the flushed midpoint
    after each threshold but the last value's, then `finish_borders` (their
    set, sorted; the halving is a pinned exact product)."""
    var w = 0
    for k in range(n_thr):
        var t = Int(thr[k])
        if t + 1 == wsize:
            continue
        dst[w] = ftz(pinned_mul_f32(ftz(uniq[t] + uniq[t + 1]), Float32(0.5)))
        w += 1
    return finish_borders(dst, w)


# ---- the per-column dispatch -----------------------------------------------------


def border_type_column(
    border_type: Int,
    keys: MutPointer[UInt32, MutAnyOrigin],
    n: Int,
    max_borders: Int,
    tab: MutPointer[UInt64, MutAnyOrigin],
    h_start: MutPointer[Int32, MutAnyOrigin],
    h_end: MutPointer[Int32, MutAnyOrigin],
    h_split: MutPointer[Int32, MutAnyOrigin],
    h_score: MutPointer[UInt64, MutAnyOrigin],
    f64s: MutPointer[UInt64, MutAnyOrigin],
    i32s: MutPointer[Int32, MutAnyOrigin],
    uniq: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
) -> Int:
    """One column's non-GreedyLogSum grid (no NaN sentinel)."""
    if border_type == BORDER_TYPE_GREEDY_MIN_ENTROPY:
        var nb = gls_column(keys, n, max_borders, tab, h_start, h_end, h_split, h_score, dst)
        return finish_borders(dst, nb)
    if border_type == BORDER_TYPE_MAX_LOG_SUM or border_type == BORDER_TYPE_MIN_ENTROPY:
        return exact_column(keys, n, max_borders, tab, f64s, i32s, uniq, dst)
    return simple_column(keys, n, max_borders, border_type, dst)


@always_inline
def exact_i32_words(n: Int, border_count: Int) -> Int:
    """`exact_column`'s Int32 scratch per column for `n` keys and a budget of
    at most `border_count`."""
    return 3 * n + max(0, border_count - 1) * n + max(1, border_count)


def border_types_columns_kernel(
    border_type: Int32,
    keys: MutPointer[UInt32, MutAnyOrigin],
    sn: Int32,
    n_cols: Int32,
    valid: MutPointer[Int32, MutAnyOrigin],
    budget: MutPointer[Int32, MutAnyOrigin],
    tab: MutPointer[UInt64, MutAnyOrigin],
    h_start: MutPointer[Int32, MutAnyOrigin],
    h_end: MutPointer[Int32, MutAnyOrigin],
    h_split: MutPointer[Int32, MutAnyOrigin],
    h_score: MutPointer[UInt64, MutAnyOrigin],
    heap_cap: Int32,
    f64s: MutPointer[UInt64, MutAnyOrigin],
    f64_stride: Int64,
    i32s: MutPointer[Int32, MutAnyOrigin],
    i32_stride: Int64,
    uniq: MutPointer[Float32, MutAnyOrigin],
    uniq_stride: Int64,
    dst: MutPointer[Float32, MutAnyOrigin],
    out_cap: Int32,
    counts: MutPointer[Int32, MutAnyOrigin],
):
    """One thread per column of the chunk: `border_type_column` over its
    sorted keys' first `valid[c]` entries with `budget[c]` borders. A
    scratch plane the type does not read may be one word (stride 0)."""
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if c >= Int(n_cols):
        return
    var hc = Int(heap_cap)
    counts[c] = Int32(
        border_type_column(
            Int(border_type), keys + c * Int(sn), Int(valid[c]), Int(budget[c]), tab,
            h_start + c * hc, h_end + c * hc, h_split + c * hc, h_score + c * hc,
            f64s + c * Int(f64_stride), i32s + c * Int(i32_stride),
            uniq + c * Int(uniq_stride), dst + c * Int(out_cap),
        )
    )
