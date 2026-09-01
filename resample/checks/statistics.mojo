# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The statistic kernels, comptime-selected, each with its own pinned fold.

NO UPSTREAM. `scipy.stats.bootstrap` and `scipy.stats.permutation_test` take
the statistic as a CALLABLE and vectorise it over an axis, so there is no
upstream file holding "their mean kernel" to transcribe. What SciPy defines is
the SEMANTICS of each statistic, and the semantics are what is honoured here,
name for name (`resample/README.md` carries the mapping table). Their designs
are `numpy` reductions over the last axis: serial, CPU-shaped, and free to
choose any summation order they like because they ship one backend.

A CALLER COMPOSES, IT DOES NOT PASS A POINTER. Every statistic is a comptime
arm of one kernel, selected by `STAT_*` at the launch site (`launch_bootstrap_
statistic`). A function pointer would put the fold inside the caller's code
where nothing in this lane could pin it, and the pin is the product.

============ WHAT IS PINNED, AND WHERE IT COMES FROM ============

EVERY FOLD IN THIS FILE IS `metrics/checks/pinned_sum.mojo`'s TREE. Not a
copy of it, not a sibling of it: the same `virtual_block_sum` and the same
`PINNED_SUM_W = 256` slab, called from here. That file's whole reason for
existing is that `pinned_block_sum[block_size]` makes the fold a function of
the LAUNCH, where `virtual_block_sum` folds a fixed 256-wide tree that a
64-thread block and a 256-thread block fill identically. This lane needs
exactly that property, because `check_launch_invariance` is one of its gates
and a statistic whose fold shape moved with the block size could not pass it.

So: the statistic of replicate `r` is a pure function of

    (the drawn values in ascending position order, n, PINNED_SUM_W = 256)

and of nothing else. Chunk `c` holds positions `[c*256, (c+1)*256)`, positions
past `n` hold `+0.0`, each chunk folds as a halving tree, and thread 0 folds
the chunk totals ASCENDING through `ftz` -- `host_fold_partials`' sequence,
executed in the kernel because the whole statistic has to land in one device
scalar before the distribution can be sorted.

============ TWO-PASS, NEVER ONE-PASS ============

`std`, `pearson` and the centred sums are TWO-PASS: the mean first, then the
deviations. The one-pass alternatives (a raw sum of squares minus n*mean^2, or
a streaming Welford) are both rejected, and not for accuracy:

  * the raw-moment form is catastrophic cancellation at large offsets and
    would make the answer a function of where the data sits on the number
    line, which is a worse thing to publish than a slow kernel;
  * Welford's update is SEQUENTIAL -- element k's update reads the state
    element k-1 left -- so its parallel form is a pairwise merge whose tree
    shape is a launch property. That is the exact hazard `virtual_block_sum`
    exists to remove, reintroduced one level up.

The price is one extra pass over the drawn values per centred statistic, and
the draws are recomputed rather than stored (DEVIATION 1694).

============ ROW 39 (signed zero, NaN) IN THIS FILE ============

  * Every fold is seeded `+0.0` and every pad is `+0.0`, both inside
    `virtual_block_sum` and in the chunk chain. `x + (+0.0) == x` for every
    finite x, every infinity and every NaN EXCEPT `x = -0.0`, so a sample
    whose drawn values are all `-0.0` sums to `+0.0` here. That is the
    documented behaviour of the shared tree, it is the same on every vendor
    (row 39 answer (a): a seed separates PROFILES, not vendors), and
    `check_percentile_interval`'s signed-zero fixture plants it.
  * NO `max`/`min` OF FLOATS APPEARS ANYWHERE IN THIS FILE. The order
    statistics are an INDEX into a sorted array, not a selection, so row 39's
    answer (b) -- the one place a vendor may legitimately disagree -- has no
    site here at all.
  * A statistic that can be NaN (`pearson` at a degenerate sample) is passed
    through `canonicalize_nan` before it is stored, because a COMPUTED NaN
    carries the vendor's payload (Apple 0x7fc00000, NVIDIA 0x7fffffff, AMD
    0xffc00000) and `resample.theta` is a certified card stage. DEVIATION
    1696.

================= DEVIATION BLOCK =================

DEVIATION 1694. THE DRAWN VALUES ARE RECOMPUTED PER PASS, NOT MATERIALISED.
A two-pass statistic needs the same `n` drawn values twice. Storing them costs
`n_resamples * n` floats -- 40 GB at a million resamples of a thousand
observations -- and recomputing them costs one `PhiloxState.init` per value
per pass. The map is a pure function of `(key, r, i)` (DEVIATION 1690), so the
second pass provably sees the same values as the first; that is a property of
the design rather than a hope about a cache. PRICE: ~20 Philox rounds per
value per extra pass. That price has never been measured.

The ORDER statistics are the exception and they materialise, because a sort
has nothing to recompute from. See `RESAMPLE_MAX_SORT_CELLS`.

DEVIATION 1695. `pearson`'s DENOMINATOR IS `sqrt(sxx * syy)`, NOT
`sqrt(sxx) * sqrt(syy)`. Two roundings against one, so they are different
floats; numpy's `corrcoef` reaches the two-sqrt form through a normalisation
of each row, and `scipy.stats.pearsonr` reaches the one-sqrt form. The
one-sqrt form is pinned here because it is one rounding fewer, and it is
STATED because a reader comparing against `np.corrcoef` will see the last bit
differ and must be able to find out why. Overflow is possible where
`sxx * syy` exceeds float32 range and each factor does not; that input is
refused by name at the host entry (`resample_check` plants the boundary).

DEVIATION 1696. A DEGENERATE `pearson` RETURNS THE CANONICAL NaN, NOT THE
VENDOR'S. `sxx == 0` (a resample that drew one observation `n` times) makes
the correlation `0/0`. numpy returns NaN and warns. Ours returns
`metrics/checks/pinned_sum.mojo::CANONICAL_NAN_BITS` -- one payload, chosen
once -- because IDENTITY_PATHS row 39 FACT 2 measured three payloads for one
IEEE answer across the three vendors and `resample.theta` is a recorded stage.
This is reachable at small `n` with real probability (`n^(1-n)` per replicate,
1/1024 at n = 2), so it is not a corner.

DEVIATION 1697. `std` IS ddof = 1 AND `standard_error` IS ddof = 1, MATCHING
SciPy. `scipy.stats.bootstrap` computes its returned `standard_error` as
`xp.std(theta_hat_b, correction=1, axis=-1)` (`_resampling.py`), i.e. the
sample standard deviation of the bootstrap distribution with one degree of
freedom removed. `STAT_STD` uses the same convention so a caller who bootstraps
`np.std(..., ddof=1)` gets the statistic they named. `ddof = 0` is NOT offered
under a different name and is not silently available; add it as `STAT_STD_POP`
if it is ever wanted, and give it its own gate.

DEVIATION 1698. THE INTERPOLATION BETWEEN ORDER STATISTICS IS PINNED HERE AND
NOT INHERITED. `quantile_position` and `quantile_interpolate` spell out
Hyndman-Fan type 7 (numpy's and `scipy.stats.quantile`'s default `'linear'`
method) as

    h    = Float32(m - 1) * q          one multiply, correctly rounded
    lo   = floor(h)                    exact, h >= 0 and finite
    frac = h - Float32(lo)             one subtraction, correctly rounded
    v    = identical_mul_add(frac, ftz(a[lo+1] - a[lo]), a[lo])

rather than calling anything. THREE REASONS, in order of how much they cost if
ignored: (1) the last line is a multiply-add and therefore IDENTITY_PATHS row
9's contraction hazard, so it must be an `identical_mul_add` and not an
expression; (2) a library is free to change its default method between
versions, and a confidence interval that moves when a dependency is upgraded
is the same defect as one that moves when the machine changes; (3) `h` landing
exactly on an integer is a knife edge -- `frac` is then `+0.0` and the second
order statistic must contribute nothing -- and `check_percentile_interval`
gates BOTH sides of it by construction rather than by luck of the fixture.
=================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import floor
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from metrics.checks.pinned_sum import (
    PINNED_SUM_W,
    canonicalize_nan,
    chunk_count,
    host_fold_partials,
    linear_block_id,
    physical_block_count,
    virtual_block_sum,
)
from core.segmented_sort import float_to_sortable
from checks.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from resample.checks.index_map import (
    PERM_MAX_POOLED,
    draw_permutation_key,
    draw_row_index,
    draw_uniform_in,
    key_join,
    permutation_key_lt,
)


# ===========================================================================
# The statistic ids
# ===========================================================================

#: `np.mean(sample)` of column 0.
comptime STAT_MEAN = 0

#: `np.std(sample, ddof=1)` of column 0. DEVIATION 1697.
comptime STAT_STD = 1

#: `np.quantile(sample, q, method='linear')` of column 0, DEVIATION 1698.
#: `q = 0.5` IS the median and is not a separate arm.
comptime STAT_QUANTILE = 2

#: `scipy.stats.pearsonr(x, y).statistic` over columns 0 and 1 of the SAME
#: resampled rows (the pairing is preserved; SciPy calls that `paired=True`).
comptime STAT_PEARSON = 3

#: `np.mean(col0) - np.mean(col1)` over the same resampled rows, paired.
#: For `permutation_test` the same id means the BETWEEN-GROUP difference,
#: which is a different fold and a different kernel; see `perm_stat_kernel`.
comptime STAT_DIFF_MEANS = 4

#: `scipy.stats.trim_mean(sample, proportiontocut)` of column 0.
comptime STAT_TRIMMED_MEAN = 5

comptime STAT_COUNT = 6

#: The ceiling on `n_resamples * n` for the two sorting arms. 4.2 million
#: cells is ~17 MB of keys plus the sort's four scratch buffers of the same
#: count. Above it the run RAISES and names batching as the closure rather
#: than allocating a gigabyte behind the caller's back.
comptime RESAMPLE_MAX_SORT_CELLS = 1 << 22


def stat_from_name(name: String) raises -> Int:
    """The SciPy-facing name of each statistic. Anything else is refused BY
    NAME with the list, in scikit-learn's `Unknown metric` idiom."""
    if name == "mean":
        return STAT_MEAN
    if name == "std":
        return STAT_STD
    if name == "quantile" or name == "median":
        return STAT_QUANTILE
    if name == "pearson":
        return STAT_PEARSON
    if name == "diff_means":
        return STAT_DIFF_MEANS
    if name == "trimmed_mean":
        return STAT_TRIMMED_MEAN
    raise Error(
        "resample: unknown statistic '"
        + name
        + "'. The supplied statistics are mean, std, quantile (median is"
        " quantile at q=0.5), pearson, diff_means and trimmed_mean. A"
        " statistic this lane does not supply cannot be passed as a"
        " callable: every fold in resample/ is pinned at compile time"
        " (statistics.mojo's header), and a caller-supplied reduction would"
        " put the summation order back where nothing can pin it."
    )


def stat_needs_sort(stat: Int) -> Bool:
    """Whether the arm needs the replicate SORTED. Everything else folds in
    place and materialises nothing (DEVIATION 1694)."""
    return stat == STAT_QUANTILE or stat == STAT_TRIMMED_MEAN


def stat_columns_needed(stat: Int) -> Int:
    """How many columns of the sample the arm reads. Checked at the host
    entry against the caller's `n_features`, refused by name on a
    mismatch -- a `pearson` bootstrap of a one-column sample would
    otherwise read past the row and correlate the data with whatever
    followed it in memory."""
    if stat == STAT_PEARSON or stat == STAT_DIFF_MEANS:
        return 2
    return 1


def stat_name(stat: Int) -> String:
    if stat == STAT_MEAN:
        return String("mean")
    if stat == STAT_STD:
        return String("std")
    if stat == STAT_QUANTILE:
        return String("quantile")
    if stat == STAT_PEARSON:
        return String("pearson")
    if stat == STAT_DIFF_MEANS:
        return String("diff_means")
    if stat == STAT_TRIMMED_MEAN:
        return String("trimmed_mean")
    return String("?")


# ===========================================================================
# The two pinned building blocks, both `metrics/checks/pinned_sum.mojo`'s
# ===========================================================================


@always_inline
def _block_broadcast[tpb: Int](value: Float32) -> Float32:
    """Thread 0's `value` to every thread of the block, through one
    threadgroup float.

    Needed because `virtual_block_sum` returns a meaningful value on thread
    0 only, and a two-pass statistic has to hand the first pass's mean to
    every thread of the second. No warp primitive: a `warp.broadcast` would
    be a lane-width read on a numeric path and this repository does not put
    one there (IDENTITY_PATHS row 17's class). Two barriers, so the slab is
    safe to reuse at the next call site.
    """
    var slot = stack_allocation[
        1,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    barrier()
    if Int(thread_idx.x) == 0:
        slot[unsafe_offset=0] = value
    barrier()
    var got = slot[unsafe_offset=0]
    barrier()
    return got


@always_inline
def _chunked_sum[tpb: Int, mode: Int](
    x: MutPointer[Float32, MutAnyOrigin],
    key: UInt64,
    r: Int,
    n: Int,
    n_features: Int,
    n_rows: Int32,
    col_a: Int,
    col_b: Int,
    centre_a: Float32,
    centre_b: Float32,
) -> Float32:
    """One pinned sum over a replicate's `n` drawn positions.

    `mode` selects the per-position TERM and nothing else -- the fold is the
    same tree in every mode, which is the point:

        0  x[row, col_a]                              raw sum
        1  (x[row, col_a] - centre_a)^2               centred square
        2  (x[row, col_a] - centre_a) *
           (x[row, col_b] - centre_b)                 centred cross product

    EVERY THREAD REACHES `virtual_block_sum` on every chunk. Returning early
    from a block-wide primitive is how this construction breaks
    (`core/segmented_sort.mojo` records the same rule); the bound check
    supplies `+0.0` instead.

    Only thread 0's return is meaningful.
    """
    comptime lanes = PINNED_SUM_W // tpb
    comptime assert lanes * tpb == PINNED_SUM_W, (
        "resample: the statistic block width must divide PINNED_SUM_W"
    )
    var tid = Int(thread_idx.x)
    var chunks = chunk_count(n)
    var acc = Float32(0.0)
    for c in range(chunks):
        var vals = SIMD[DType.float32, lanes](0.0)
        comptime for lane in range(lanes):
            var i = c * PINNED_SUM_W + tid + lane * tpb
            if i < n:
                var row = Int(draw_row_index(key, r, i, n_rows))
                var va = ftz(x.unsafe_load(row * n_features + col_a))
                comptime if mode == 0:
                    vals[lane] = va
                comptime if mode == 1:
                    var da = ftz(va - centre_a)
                    vals[lane] = ftz(identical_mul(da, da))
                comptime if mode == 2:
                    var vb = ftz(x.unsafe_load(row * n_features + col_b))
                    var da2 = ftz(va - centre_a)
                    var db2 = ftz(vb - centre_b)
                    vals[lane] = ftz(identical_mul(da2, db2))
        var total = virtual_block_sum[tpb](vals)
        if tid == 0:
            acc = ftz(acc + total)
    return acc


@always_inline
def _mean_of_sum(total: Float32, n: Int) -> Float32:
    """`sum / n`, through `identical_div`.

    `identical_div` is DEVIATION 740's `portable_divf` under IDENTICAL --
    operands flushed to signed zero BY BITS, one hardware division, result
    flushed -- and a plain `/` under FAST. It is here rather than a plain
    `/` because a vendor may substitute a fast reciprocal (IDENTITY_PATHS
    row 49) and `n` is not always a power of two.
    """
    return ftz(identical_div(total, Float32(n)))


# ===========================================================================
# The order-statistic rule (DEVIATION 1698)
# ===========================================================================


@always_inline
def quantile_position(m: Int, q: Float32) -> Float32:
    """`h = (m - 1) * q`, Hyndman-Fan type 7. One multiply, correctly
    rounded, no contraction to pin because there is no addend."""
    return Float32(m - 1) * q


@always_inline
def quantile_lower_index(h: Float32, m: Int) -> Int:
    """`floor(h)`, clamped into `[0, m - 1]`.

    The clamp is not decoration: `q = 1.0` gives `h = m - 1` exactly and
    `lo + 1` would read one past the segment. It is an INTEGER clamp on an
    integer, so no row-39 zero-sign question arises.
    """
    var lo = Int(floor(h))
    if lo < 0:
        lo = 0
    if lo > m - 1:
        lo = m - 1
    return lo


@always_inline
def quantile_interpolate(a_lo: Float32, a_hi: Float32, frac: Float32) -> Float32:
    """`a_lo + frac * (a_hi - a_lo)`, spelled as ONE `identical_mul_add`.

    Written this way and not as `(1 - frac) * a_lo + frac * a_hi`: the
    two-term form is two multiplies and an add, three roundings against
    this form's two, and it does NOT return `a_lo` exactly at `frac = 0`
    for every `a_lo` (it returns `1.0 * a_lo + 0.0 * a_hi`, which is `a_lo`
    unless `a_hi` is an infinity or a NaN, where this form is also
    well-behaved because `0.0 * inf` never arises). At `frac = +0.0` this
    form is `fma(+0.0, d, a_lo) == a_lo` for every finite `d`, which is the
    knife-edge property `check_percentile_interval` asserts.
    """
    return ftz(identical_mul_add(frac, ftz(a_hi - a_lo), a_lo))


@always_inline
def quantile_of_sorted_host(
    a: List[Float32], base: Int, m: Int, q: Float32
) -> Float32:
    """The same three lines on the host, over `a[base : base + m]` already
    ascending. The host mirror the gates compare against, and the routine
    `intervals.mojo` calls for the two interval endpoints (four scalars: the
    host is where they belong)."""
    if m == 1:
        return a[base]
    var h = quantile_position(m, q)
    var lo = quantile_lower_index(h, m)
    var frac = ftz(h - Float32(lo))
    var hi = lo + 1
    if hi > m - 1:
        hi = m - 1
    return quantile_interpolate(a[base + lo], a[base + hi], frac)


@always_inline
def trim_count(m: Int, proportion: Float32) -> Int:
    """`int(m * proportiontocut)`, `scipy.stats.trim_mean`'s own truncation.

    Their line is `lowercut = int(proportiontocut * nobs)` with a C-style
    truncation towards zero, and `nobs` up to 2^24 keeps the product exact
    in float32 for the proportions anyone writes down. Transcribed rather
    than improved: a rounded cut would silently disagree with every
    published `trim_mean` number.
    """
    return Int(Float32(m) * proportion)


# ===========================================================================
# The stable sort the order statistics need, on the host
# ===========================================================================


def host_sort_stable(values: List[Float32], base: Int, m: Int) -> List[Float32]:
    """`values[base : base + m]` ascending under the TOTAL ORDER
    `(float_to_sortable(bits), original position)`.

    `float_to_sortable` is `core/segmented_sort.mojo`'s port of
    `cub::NumericTraits<float>::TwiddleIn`, imported rather than repeated --
    it is the function that makes a radix pass order floats correctly, and
    the one place `-0.0` and `+0.0` become DISTINCT KEYS (`0x7FFFFFFF` and
    `0x80000000`) even though they compare equal as floats.

    INSERTION SORT, and the choice is deliberate. This is an oracle over `m`
    values on the host: an `O(m^2)` sort that is OBVIOUSLY stable and
    obviously ordered by the stated key is worth more here than a fast one
    that has to be trusted. `m` is bounded by the fixtures.

    NaN: `float_to_sortable` of a NaN is a key like any other, so NaNs sort
    to one end deterministically rather than making the comparison
    non-transitive. That is the same behaviour the device radix sort has, by
    construction, and it is why neither side needs a NaN special case.
    """
    var keys = List[UInt32]()
    var out = List[Float32]()
    for i in range(m):
        var v = values[base + i]
        keys.append(float_to_sortable(bitcast[DType.uint32](v)))
        out.append(v)
    for i in range(1, m):
        var kv = keys[i]
        var vv = out[i]
        var j = i - 1
        while j >= 0 and keys[j] > kv:
            keys[j + 1] = keys[j]
            out[j + 1] = out[j]
            j -= 1
        keys[j + 1] = kv
        out[j + 1] = vv
    return out^


# ===========================================================================
# The bootstrap statistic kernel: one block per replicate, no materialisation
# ===========================================================================


def bootstrap_stat_kernel[stat: Int, tpb: Int](
    theta: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    r_first_in: Int32,
    n_replicates_in: Int32,
    n_in: Int32,
    n_rows_in: Int32,
    n_features_in: Int32,
):
    """`theta[r] = statistic(resample(x, r))` for the fold statistics.

    ONE BLOCK PER REPLICATE. `block_idx.x` is the replicate offset from
    `r_first`, so the grid is `n_replicates` blocks of `tpb` threads;
    `r_first` is the batch-invariance handle (DEVIATION 1690(b)) and nothing
    in the arithmetic reads `n_replicates` at all.

    `tpb` is SCHEDULING and free: it divides `PINNED_SUM_W`, and
    `virtual_block_sum` folds the same 256-wide tree whichever value it
    takes (`metrics/checks/pinned_sum.mojo`'s header). `PINNED_SUM_W`
    itself is NUMERIC and pinned.

    `STAT_QUANTILE` and `STAT_TRIMMED_MEAN` are NOT arms here; they need the
    replicate sorted and go through `materialize_resample_kernel` +
    `core/segmented_sort.mojo` + `order_stat_kernel`.
    """
    comptime assert stat != STAT_QUANTILE and stat != STAT_TRIMMED_MEAN, (
        "resample: the order statistics have their own kernel"
    )
    var rr = Int(block_idx.x)
    if rr >= Int(n_replicates_in):
        return
    var r = Int(r_first_in) + rr
    var tid = Int(thread_idx.x)
    var key = key_join(lo_bits, hi_bits)
    var n = Int(n_in)
    var d = Int(n_features_in)
    var value = Float32(0.0)

    comptime if stat == STAT_MEAN:
        var s = _chunked_sum[tpb, 0](
            x, key, r, n, d, n_rows_in, 0, 0, Float32(0.0), Float32(0.0)
        )
        value = _mean_of_sum(s, n)

    comptime if stat == STAT_DIFF_MEANS:
        var s0 = _chunked_sum[tpb, 0](
            x, key, r, n, d, n_rows_in, 0, 0, Float32(0.0), Float32(0.0)
        )
        var s1 = _chunked_sum[tpb, 0](
            x, key, r, n, d, n_rows_in, 1, 1, Float32(0.0), Float32(0.0)
        )
        value = ftz(_mean_of_sum(s0, n) - _mean_of_sum(s1, n))

    comptime if stat == STAT_STD:
        var st = _chunked_sum[tpb, 0](
            x, key, r, n, d, n_rows_in, 0, 0, Float32(0.0), Float32(0.0)
        )
        var m = _block_broadcast[tpb](_mean_of_sum(st, n))
        var ssd = _chunked_sum[tpb, 1](
            x, key, r, n, d, n_rows_in, 0, 0, m, Float32(0.0)
        )
        # ddof = 1 (DEVIATION 1697). `n == 1` is refused at the host entry,
        # so the divisor is never zero here.
        value = ftz(identical_sqrt(ftz(identical_div(ssd, Float32(n - 1)))))

    comptime if stat == STAT_PEARSON:
        var sx = _chunked_sum[tpb, 0](
            x, key, r, n, d, n_rows_in, 0, 0, Float32(0.0), Float32(0.0)
        )
        var sy = _chunked_sum[tpb, 0](
            x, key, r, n, d, n_rows_in, 1, 1, Float32(0.0), Float32(0.0)
        )
        var mx = _block_broadcast[tpb](_mean_of_sum(sx, n))
        var my = _block_broadcast[tpb](_mean_of_sum(sy, n))
        var sxy = _chunked_sum[tpb, 2](x, key, r, n, d, n_rows_in, 0, 1, mx, my)
        var sxx = _chunked_sum[tpb, 1](
            x, key, r, n, d, n_rows_in, 0, 0, mx, Float32(0.0)
        )
        var syy = _chunked_sum[tpb, 1](
            x, key, r, n, d, n_rows_in, 1, 1, my, Float32(0.0)
        )
        if tid == 0:
            # DEVIATION 1696: a degenerate resample is `0/0`. Refuse to let
            # the hardware answer, because the three vendors write three
            # different NaN payloads for it.
            if sxx == Float32(0.0) or syy == Float32(0.0):
                value = canonicalize_nan(Float32(0.0) / Float32(0.0))
            else:
                # DEVIATION 1695: ONE sqrt of the product, not two.
                value = ftz(
                    identical_div(
                        sxy, ftz(identical_sqrt(ftz(identical_mul(sxx, syy))))
                    )
                )

    if tid == 0:
        theta.unsafe_store(rr, canonicalize_nan(value))


def materialize_resample_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    r_first_in: Int32,
    n_replicates_in: Int32,
    n_in: Int32,
    n_rows_in: Int32,
    n_features_in: Int32,
    col_in: Int32,
):
    """Write `n_replicates x n` drawn values, one segment per replicate, for
    the two arms that need a sort. One thread per POSITION -- no fold, no
    shared memory, `tpb` entirely free.

    DEVIATION 1694's exception. The segments this writes are exactly the
    segments `core/segmented_sort.mojo::segmented_sort_keys_f32` was written
    for (`n_segments` runs of `seg_size` float32 keys, the offsets arithmetic
    rather than a lookup), which is why this lane needs no sort of its own.
    """
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    var total = Int(n_replicates_in) * n
    if tid >= total:
        return
    var r = Int(r_first_in) + tid // n
    var i = tid % n
    var row = Int(draw_row_index(key_join(lo_bits, hi_bits), r, i, n_rows_in))
    dst.unsafe_store(
        tid, ftz(x.unsafe_load(row * Int(n_features_in) + Int(col_in)))
    )


def order_stat_kernel[stat: Int, tpb: Int](
    theta: MutPointer[Float32, MutAnyOrigin],
    sorted_vals: MutPointer[Float32, MutAnyOrigin],
    n_replicates_in: Int32,
    n_in: Int32,
    q_or_prop: Float32,
):
    """`theta[r]` from replicate `r`'s SORTED segment.

    `STAT_QUANTILE`: three lines of DEVIATION 1698, thread 0 only -- an
    index into a sorted array is not a fold and does not want a block.

    `STAT_TRIMMED_MEAN`: `scipy.stats.trim_mean`'s cut, then the SAME pinned
    tree over the surviving `m - 2k` values. The chunking starts at the cut,
    so the fold is a pure function of `(the kept values, m - 2k, 256)` and
    the cut count is a pure function of `(m, proportion)`.
    """
    comptime assert stat == STAT_QUANTILE or stat == STAT_TRIMMED_MEAN, (
        "order_stat_kernel: only the sorting arms"
    )
    comptime lanes = PINNED_SUM_W // tpb
    var rr = Int(block_idx.x)
    if rr >= Int(n_replicates_in):
        return
    var tid = Int(thread_idx.x)
    var m = Int(n_in)
    var base = rr * m

    comptime if stat == STAT_QUANTILE:
        if tid == 0:
            var v: Float32
            if m == 1:
                v = sorted_vals.unsafe_load(base)
            else:
                var h = quantile_position(m, q_or_prop)
                var lo = quantile_lower_index(h, m)
                var frac = ftz(h - Float32(lo))
                var hi = lo + 1
                if hi > m - 1:
                    hi = m - 1
                v = quantile_interpolate(
                    sorted_vals.unsafe_load(base + lo),
                    sorted_vals.unsafe_load(base + hi),
                    frac,
                )
            theta.unsafe_store(rr, canonicalize_nan(v))

    comptime if stat == STAT_TRIMMED_MEAN:
        var k = trim_count(m, q_or_prop)
        var kept = m - 2 * k
        # `kept < 1` is refused at the host entry; the guard is here so a
        # future caller cannot reach a zero divisor through this kernel.
        if kept < 1:
            if tid == 0:
                theta.unsafe_store(rr, canonicalize_nan(Float32(0.0) / Float32(0.0)))
            return
        var chunks = chunk_count(kept)
        var acc = Float32(0.0)
        for c in range(chunks):
            var vals = SIMD[DType.float32, lanes](0.0)
            comptime for lane in range(lanes):
                var i = c * PINNED_SUM_W + tid + lane * tpb
                if i < kept:
                    vals[lane] = ftz(sorted_vals.unsafe_load(base + k + i))
            var total = virtual_block_sum[tpb](vals)
            if tid == 0:
                acc = ftz(acc + total)
        if tid == 0:
            theta.unsafe_store(rr, canonicalize_nan(_mean_of_sum(acc, kept)))


# ===========================================================================
# The permutation statistic kernel: rank by a total order, then fold
# ===========================================================================


def perm_stat_kernel[stat: Int, tpb: Int](
    null_dist: MutPointer[Float32, MutAnyOrigin],
    pooled: MutPointer[Float32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    r_first_in: Int32,
    n_replicates_in: Int32,
    n_pooled_in: Int32,
    n_x_in: Int32,
):
    """`null_dist[r] = statistic(x_r, y_r)` where `(x_r, y_r)` is the pooled
    sample split by replicate `r`'s permutation.

    ONE BLOCK PER REPLICATE, three phases and two barriers:

      1. every pooled position `j` gets its 64-bit key
         `draw_permutation_key(key, r, j)` -- a pure function of
         `(key, r, j)`, so no thread reads another's anything;
      2. every position's RANK is the count of positions below it in the
         total order `(key, j)` (`permutation_key_lt`). A rank is a pure
         function of the key vector, the count is over a strict order so no
         two positions claim one rank, and the result is a bijection;
      3. the statistic folds over the ORIGINAL pooled positions ascending,
         with `rank < n_x` as the membership mask and `+0.0` for the other
         group. Two folds, both `virtual_block_sum`'s tree.

    WHY A RANK AND NOT A SHUFFLE. `numpy.random.Generator.permutation` is
    Fisher-Yates: position `i`'s destination depends on every swap before
    it, so it cannot be evaluated at a position and it consumes a
    stream-position-dependent number of words. Ranking a keyed order is the
    positional spelling of the same object. DEVIATION 1690 covers it; the
    tie rule and the 64-bit key width are argued at
    `index_map.mojo::draw_permutation_key`.

    COST: `O(N^2)` compares per replicate, `8*N` bytes of threadgroup
    memory. `PERM_MAX_POOLED` bounds both and `validate_pooled` refuses
    above it by name with the closure.

    NO ATOMIC, NO WARP PRIMITIVE, NO FLOAT COMPARE ANYWHERE IN THE RANK. The
    keys are UInt64 and the tie-break is an Int compare, so the permutation
    is the same permutation on every vendor by construction rather than by
    measurement.
    """
    comptime assert (
        stat == STAT_MEAN or stat == STAT_STD or stat == STAT_DIFF_MEANS
    ), "perm_stat_kernel: mean, std and diff_means are the ported arms"
    comptime lanes = PINNED_SUM_W // tpb
    var rr = Int(block_idx.x)
    if rr >= Int(n_replicates_in):
        return
    var r = Int(r_first_in) + rr
    var tid = Int(thread_idx.x)
    var key = key_join(lo_bits, hi_bits)
    var n_pooled = Int(n_pooled_in)
    var n_x = Int(n_x_in)
    var n_y = n_pooled - n_x

    var keys = stack_allocation[
        PERM_MAX_POOLED,
        Scalar[DType.uint64],
        address_space = AddressSpace.SHARED,
    ]()
    var ranks = stack_allocation[
        PERM_MAX_POOLED,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var j = tid
    while j < n_pooled:
        keys[unsafe_offset=j] = draw_permutation_key(key, r, j)
        j += tpb
    barrier()

    var jj = tid
    while jj < n_pooled:
        var kj = keys[unsafe_offset=jj]
        var below = Int32(0)
        for l in range(n_pooled):
            if permutation_key_lt(keys[unsafe_offset=l], l, kj, jj):
                below += 1
        ranks[unsafe_offset=jj] = below
        jj += tpb
    barrier()

    # Phase 3. `_perm_group_sum` is spelled inline because the mask has to
    # be read from `ranks`, which is a threadgroup pointer this kernel owns.
    var chunks = chunk_count(n_pooled)
    var sum_x = Float32(0.0)
    var sum_y = Float32(0.0)
    for c in range(chunks):
        var vx = SIMD[DType.float32, lanes](0.0)
        var vy = SIMD[DType.float32, lanes](0.0)
        comptime for lane in range(lanes):
            var i = c * PINNED_SUM_W + tid + lane * tpb
            if i < n_pooled:
                var v = ftz(pooled.unsafe_load(i))
                if Int(ranks[unsafe_offset=i]) < n_x:
                    vx[lane] = v
                else:
                    vy[lane] = v
        var tx = virtual_block_sum[tpb](vx)
        var ty = virtual_block_sum[tpb](vy)
        if tid == 0:
            sum_x = ftz(sum_x + tx)
            sum_y = ftz(sum_y + ty)

    var value = Float32(0.0)
    comptime if stat == STAT_DIFF_MEANS:
        if tid == 0:
            value = ftz(_mean_of_sum(sum_x, n_x) - _mean_of_sum(sum_y, n_y))
    comptime if stat == STAT_MEAN:
        if tid == 0:
            value = _mean_of_sum(sum_x, n_x)
    comptime if stat == STAT_STD:
        var mx = _block_broadcast[tpb](_mean_of_sum(sum_x, n_x))
        var ssd = Float32(0.0)
        for c2 in range(chunks):
            var vs = SIMD[DType.float32, lanes](0.0)
            comptime for lane in range(lanes):
                var i2 = c2 * PINNED_SUM_W + tid + lane * tpb
                if i2 < n_pooled:
                    if Int(ranks[unsafe_offset=i2]) < n_x:
                        var dv = ftz(ftz(pooled.unsafe_load(i2)) - mx)
                        vs[lane] = ftz(identical_mul(dv, dv))
            var ts = virtual_block_sum[tpb](vs)
            if tid == 0:
                ssd = ftz(ssd + ts)
        if tid == 0:
            value = ftz(
                identical_sqrt(ftz(identical_div(ssd, Float32(n_x - 1))))
            )

    if tid == 0:
        null_dist.unsafe_store(rr, canonicalize_nan(value))


# ===========================================================================
# THE MONTE CARLO INTEGRAND, AND WHY IT IS IN THIS FILE
#
# `monte_carlo_integrate(f, lower, upper, n_samples, seed)` is
# `volume * mean(f(x_i))` over points drawn uniformly from a rectangle. The
# `mean` is a fold and the fold has to be pinned, so the integrand is a
# comptime arm of a kernel exactly as the statistics are, and it lives beside
# them rather than in a file of its own.
#
# NO SciPy COUNTERPART. `scipy.stats.monte_carlo_test` is a HYPOTHESIS TEST --
# it draws null samples from `rvs` and compares a statistic -- and is not an
# integrator; `scipy.integrate` has no Monte Carlo rule at all. So this entry
# point takes its name from what it does and NOT from a SciPy function, which
# is the README's rule: a parameter means what SciPy's parameter of that name
# means, or it is named differently.
#
# WHY IT IS THE LANE'S ARITHMETIC-INTENSITY ARGUMENT. Every other entry point
# reads a sample out of memory once per drawn index. This one reads nothing:
# the coordinate IS the draw, the draw is register-resident integer
# arithmetic (twenty Philox rounds per coordinate), and the only memory
# traffic in the whole run is `ceil(n_samples / 256)` partial floats written
# out. It is the case where the position map's price (DEVIATION 1690) buys
# the most and costs the least. THE INTEGRATOR HAS NEVER BEEN TIMED; the
# lane's one timing is the bootstrap's FAST NVIDIA row in
# `resample/README.md`.
#
# THE TWO-STAGE FOLD IS `metrics/checks/pinned_sum.mojo`'S, WHOLE. Chunk
# `c` is samples `[c*256, (c+1)*256)`, one `virtual_block_sum` tree per chunk,
# one partial float per chunk, and the host folds the partials ASCENDING with
# that file's own `host_fold_partials`. A physical block serves chunks
# `linear_block_id(), + physical_block_count(), ...`, so the GRID SHAPE and
# the BLOCK COUNT reach nothing: they decide which block computes a chunk and
# never which values share one.
# ===========================================================================

#: `f(x) = 1`. The control: the estimate is the box VOLUME exactly, for any
#: draws, because the fold of `n` ones is `n` exactly (integers below 2^24)
#: and `n/n` is 1 exactly. It DOES NOT DRAW -- deliberately, so that a
#: sabotage of the position map leaves it untouched and a sabotage of the
#: volume factor does not.
comptime MC_F_CONST = 0

#: `f(x) = x0 + x1`. Integral over the box is `V * (mid0 + mid1)`.
comptime MC_F_SUM = 1

#: `f(x) = x0 * x1`. Integral over the box is `V * mid0 * mid1`, the
#: coordinates being independent. The one integrand here whose closed form
#: needs that independence, so a correlated position map shows up in it and
#: in nothing else.
comptime MC_F_PRODUCT = 2

comptime MC_F_COUNT = 3

#: Every supplied integrand is two-dimensional. The map itself is not: it is
#: `(sample, dimension)` for any dimension count, and a third integrand over
#: three dimensions is one arm plus one closed form.
comptime MC_DIMS = 2


def mc_integrand_name(f_id: Int) -> String:
    if f_id == MC_F_CONST:
        return String("const")
    if f_id == MC_F_SUM:
        return String("sum")
    if f_id == MC_F_PRODUCT:
        return String("product")
    return String("?")


def mc_integrand_from_name(name: String) raises -> Int:
    if name == "const":
        return MC_F_CONST
    if name == "sum":
        return MC_F_SUM
    if name == "product":
        return MC_F_PRODUCT
    raise Error(
        "monte_carlo_integrate: unknown integrand '"
        + name
        + "'. The supplied integrands are const (f = 1), sum (f = x0 + x1)"
        " and product (f = x0 * x1), each with a closed form this lane gates"
        " against. A new integrand is one comptime arm of mc_integrand plus"
        " one line of mc_closed_form; it cannot be passed in as a callable,"
        " for statistics.mojo's reason -- the fold has to be pinned where"
        " this lane can see it."
    )


@always_inline
def mc_integrand[f_id: Int](
    key: UInt64,
    i: Int,
    lower: MutPointer[Float32, MutAnyOrigin],
    span: MutPointer[Float32, MutAnyOrigin],
) -> Float32:
    """`f` at the `i`-th point of the box.

    The point is not stored anywhere: dimension `d` is
    `draw_uniform_in(key, i, d, lower[d], span[d])`, position `(i, d)` of the
    ONE map (`index_map.mojo`). `monte_carlo_point_kernel` writes the same
    coordinates when the card wants them recorded, and
    `check_index_map_is_positional` holds the two to each other.
    """
    comptime if f_id == MC_F_CONST:
        return Float32(1.0)
    var x0 = draw_uniform_in(
        key, i, 0, lower.unsafe_load(0), span.unsafe_load(0)
    )
    var x1 = draw_uniform_in(
        key, i, 1, lower.unsafe_load(1), span.unsafe_load(1)
    )
    comptime if f_id == MC_F_SUM:
        return ftz(x0 + x1)
    return ftz(identical_mul(x0, x1))


def monte_carlo_chunk_kernel[f_id: Int, tpb: Int](
    partials: MutPointer[Float32, MutAnyOrigin],
    lower: MutPointer[Float32, MutAnyOrigin],
    span: MutPointer[Float32, MutAnyOrigin],
    lo_bits: Int32,
    hi_bits: Int32,
    i_first_in: Int32,
    n_samples_in: Int32,
    n_chunks_in: Int32,
):
    """One `PINNED_SUM_W` chunk of integrand values per iteration, its tree
    total into `partials[chunk]`.

    `i_first` is the batch-invariance handle again: sample 7 of a million is
    `i_first = 7, n_samples = 1`, and it must produce the same integrand
    value as sample 7 of the full run.

    EVERY THREAD REACHES `virtual_block_sum` on every iteration, and the loop
    bound is uniform across the block (`linear_block_id` and
    `physical_block_count` are block-uniform), so no thread can leave the
    primitive early.
    """
    comptime lanes = PINNED_SUM_W // tpb
    comptime assert lanes * tpb == PINNED_SUM_W, (
        "monte_carlo_chunk_kernel: tpb must divide PINNED_SUM_W"
    )
    var key = key_join(lo_bits, hi_bits)
    var tid = Int(thread_idx.x)
    var n_samples = Int(n_samples_in)
    var n_chunks = Int(n_chunks_in)
    var stride = physical_block_count()
    var chunk = linear_block_id()
    while chunk < n_chunks:
        var vals = SIMD[DType.float32, lanes](0.0)
        comptime for lane in range(lanes):
            var i = chunk * PINNED_SUM_W + tid + lane * tpb
            if i < n_samples:
                vals[lane] = mc_integrand[f_id](
                    key, Int(i_first_in) + i, lower, span
                )
        var total = virtual_block_sum[tpb](vals)
        if tid == 0:
            partials.unsafe_store(chunk, ftz(total))
        chunk += stride


def mc_finish_host(
    partials: List[Float32], n_chunks: Int, n_samples: Int, volume: Float32
) -> Float32:
    """`volume * (fold(partials) / n_samples)`, on the host, in that order.

    `host_fold_partials` is `metrics/checks/pinned_sum.mojo`'s ascending
    serial chain through `ftz` -- the same second stage every metric in this
    repository uses, not a second one.

    THE ORDER IS PINNED AND IT MATTERS: `V * (S / n)` and `(V * S) / n` are
    different floats. The mean-then-scale form is chosen because the mean is
    the quantity with a meaning (it is the integrand's average over the box,
    and it is what `resample.mc.mean` records), so a reader can check the
    scaling separately from the estimate.
    """
    var s = host_fold_partials(partials, n_chunks)
    var mean = _mean_of_sum(s, n_samples)
    return ftz(identical_mul(volume, mean))


def mc_box_volume(lower: List[Float32], upper: List[Float32]) -> Float32:
    """`prod(upper[d] - lower[d])`, ascending in `d`, through `ftz`. Two
    dimensions today; the loop is general so a third integrand needs no
    change here."""
    var v = Float32(1.0)
    for d in range(len(lower)):
        v = ftz(identical_mul(v, ftz(upper[d] - lower[d])))
    return v


def mc_closed_form[f_id: Int](
    lower: List[Float32], upper: List[Float32]
) -> Float32:
    """The exact integral, for the gate. Derived by hand, not by quadrature:

        const    V
        sum      V * (mid0 + mid1)
        product  V * mid0 * mid1          (the coordinates are independent)

    with `mid_d = (lower[d] + upper[d]) / 2`. On the unit square those are
    1, 1 and 0.25; on the `[-2, 2] x [0.5, 2.5]` box they are 8, 12 and 0.
    """
    var v = mc_box_volume(lower, upper)
    comptime if f_id == MC_F_CONST:
        return v
    var mid0 = ftz(ftz(lower[0] + upper[0]) * Float32(0.5))
    var mid1 = ftz(ftz(lower[1] + upper[1]) * Float32(0.5))
    comptime if f_id == MC_F_SUM:
        return ftz(identical_mul(v, ftz(mid0 + mid1)))
    return ftz(identical_mul(v, ftz(identical_mul(mid0, mid1))))
