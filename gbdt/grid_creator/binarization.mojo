# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Border selection: turning a raw float column into split points.

Reference: `library/cpp/grid_creator/binarization.cpp` (CatBoost `54a8143a`),
the `GreedyLogSum` path, which is CatBoost's DEFAULT
(`catboost/private/libs/options/data_processing_options.cpp:15`).

**This runs on the HOST in CatBoost and it runs on the host here.** Border
selection is a sort plus a priority queue over at most `border_count` bins;
there is nothing in it a GPU wants. Implementing their host code as host code is
the point, not a compromise.

Everything before this file assumed a dataset that was ALREADY binned. This
is what turns raw floats into the compressed index the histogram kernels
read, and without it nothing here can be pointed at a real column.

## The algorithm, from their source

A bin is a half-open range `[BinStart, BinEnd)` into the SORTED values. Its
score is the gain from splitting it at its own best point:

    Penalty<MaxSumLog>(w) = -log(w + 1e-8)                        (`:179`)

    CalcSplitScore(p) = -Penalty(p - BinStart)
                      + -Penalty(BinEnd - p)
                      - -Penalty(BinEnd - BinStart)               (`:1398`)

so maximising it maximises the sum of the logs of the bin sizes, which is
what `GreedyLogSum` names.

The candidate split is not searched exhaustively. `UpdateBestSplitProperties`
(`:1408`) takes the MIDPOINT of the range, reads the value there, and offers
exactly two candidates: the first index holding that value and the first
index past it. That is what keeps the whole thing near-linear, and it is why
a bin of identical values reports `CanSplit() == false` rather than looping.

`GreedySplit` (`:1500`) then pops the highest-scoring bin, splits it, pushes
both halves, and stops at `border_count` bins.

The border between two bins is the MIDPOINT of the values either side
(`:1367`), not one of the values.
"""


from std.sys.compile import is_defined

from checks.numerics import ftz, portable_log64

comptime LINEAR_BOUNDS_2635 = is_defined["MOJOLEARN_2635_LINEAR_BOUNDS"]()
"""DEVIATION 2635: `-D MOJOLEARN_2635_LINEAR_BOUNDS=1` restores the linear
LowerBound/UpperBound walks in `_update_best_split`; unset (the default) takes
the binary searches, which return the same indices. RUN OWED; see
`_update_best_split`."""


def _penalty_max_sum_log(weight: Float64) -> Float64:
    """`Penalty<EPenaltyType::MaxSumLog>` (`:179`). The `1e-8` is theirs and
    is what keeps a bin of size zero finite.

    DEVIATION 2263 applied here too (2026-09-22): `portable_log64`, not
    `std.math.log`. The heap compares split scores of DIFFERENT bins, and
    the `1e-8` makes mathematically equal scores differ by ~1e-10: bins
    3+6 of 9 and 4+4 of 8 both score log 2, and the epsilon separates them
    by 1.4e-10, while `std.math.log` errs by ~5e-11 per call (three calls
    per score). That noise flipped the heap order on duplicate-heavy
    columns: a pip-install smoke test found 4 of 120 random columns whose
    borders differed from CatBoost's (70 rows, border_count 27: a border at
    -0.515 where CatBoost has 0.860), and a correctly rounded log matched
    CatBoost on 399 of 399. `portable_log64` is within 2 ulp and gives the
    same bits on every host and device."""
    return -portable_log64(weight + 1e-8)


struct TFeatureBin(Copyable, ImplicitlyCopyable, Movable):
    """Their `TFeatureBin<MaxSumLog>` (`:1379`), over sorted values."""

    var bin_start: Int
    var bin_end: Int
    var best_split: Int
    var best_score: Float64

    def __init__(out self):
        self.bin_start = 0
        self.bin_end = 0
        self.best_split = 0
        self.best_score = 0.0

    def size(self) -> Int:
        return self.bin_end - self.bin_start

    def can_split(self) -> Bool:
        """Their `CanSplit()` (`:1353`)."""
        return self.bin_start != self.best_split and (
            self.bin_end != self.best_split
        )

    def is_first(self) -> Bool:
        return self.bin_start == 0


def _calc_split_score(bin_start: Int, bin_end: Int, split_pos: Int) -> Float64:
    """Their `CalcSplitScore` (`:1398`)."""
    if split_pos == bin_start or split_pos == bin_end:
        return -1.0e308
    var left = -_penalty_max_sum_log(Float64(split_pos - bin_start))
    var right = -_penalty_max_sum_log(Float64(bin_end - split_pos))
    var curr = -_penalty_max_sum_log(Float64(bin_end - bin_start))
    return left + right - curr


def _update_best_split(mut b: TFeatureBin, values: List[Float32]):
    """Their `UpdateBestSplitProperties` (`:1408`).

    Two candidates only, both derived from the value at the MIDPOINT: the
    first index holding it and the first index past it. Their `LowerBound`
    and `UpperBound`.

    DEVIATION 2635 (2026-09-11, gbdt-speed lane; RUN OWED, see below): both
    are BINARY searches now, as theirs are (`std::lower_bound` /
    `std::upper_bound`). The linear walks assumed "the ranges shrink fast",
    which is false on a column with a long run of one value (taxi's
    mostly-missing and low-cardinality columns): the bin holding the run is
    re-created and walked again at every split, up to rows x borders
    compares per column. `values` is sorted ascending and NaN-free, so
    `v < mid_value` and `v <= mid_value` are each true on a prefix of any
    range, and the binary search returns the same `lb` and `ub` the walks
    did; the scores, the heap order and the borders follow unchanged.
    `-D MOJOLEARN_2635_LINEAR_BOUNDS=1` restores the walks (the A/B arm).
    RUN OWED on a GPU box: `pixi run check-greedylogsum`,
    `pixi run check-binarization`, the gbdt identity checks, and the
    interleaved taxi and Istella-S timing against the walks.
    """
    var mid = b.bin_start + (b.bin_end - b.bin_start) // 2
    var mid_value = values[mid]

    var lb = b.bin_start
    var ub = mid
    comptime if LINEAR_BOUNDS_2635:
        while lb < mid and values[lb] < mid_value:
            lb += 1
        while ub < b.bin_end and values[ub] <= mid_value:
            ub += 1
    else:
        # first index in [bin_start, mid) with values[i] >= mid_value, else mid
        var hi = mid
        while lb < hi:
            var m = lb + (hi - lb) // 2
            if values[m] < mid_value:
                lb = m + 1
            else:
                hi = m
        # first index in [mid, bin_end) with values[i] > mid_value, else bin_end
        var hi2 = b.bin_end
        while ub < hi2:
            var m2 = ub + (hi2 - ub) // 2
            if values[m2] <= mid_value:
                ub = m2 + 1
            else:
                hi2 = m2

    var score_left = _calc_split_score(b.bin_start, b.bin_end, lb)
    var score_right = _calc_split_score(b.bin_start, b.bin_end, ub)
    if score_left >= score_right:
        b.best_split = lb
        b.best_score = score_left
    else:
        b.best_split = ub
        b.best_score = score_right


def _heap_push(mut a: List[TFeatureBin], var item: TFeatureBin):
    """libc++ `push_heap`: sift the new element up while the parent is
    STRICTLY smaller. On an equal score the new element stays below, which is
    half of the tie behavior `best_split` must reproduce exactly."""
    a.append(item^)
    var hole = len(a) - 1
    var v = a[hole].copy()
    while hole > 0:
        var parent = (hole - 1) // 2
        if a[parent].best_score < v.best_score:
            a[hole] = a[parent].copy()
            hole = parent
        else:
            break
    a[hole] = v^


def _heap_pop(mut a: List[TFeatureBin]):
    """libc++ `pop_heap` on a max-heap: the last element re-seats from the
    root, descending toward the LARGER child, choosing the LEFT child on an
    equal pair (`a[child] < a[child+1]` is false on a tie), and stopping only
    when the child is STRICTLY smaller than the re-seating value."""
    var n = len(a)
    if n <= 1:
        if n == 1:
            _ = a.pop()
        return
    var v = a.pop()
    n -= 1
    var hole = 0
    while True:
        var child = 2 * hole + 1
        if child >= n:
            break
        if child + 1 < n and a[child].best_score < a[child + 1].best_score:
            child += 1
        if a[child].best_score < v.best_score:
            break
        a[hole] = a[child].copy()
        hole = child
    a[hole] = v^


def best_split(
    var values: List[Float32], max_borders_count: Int
) raises -> List[Float32]:
    """Their `BestSplit` for `GreedyLogSum` (`binarization.h:23`).

    `values` is consumed and sorted, as theirs is. NaNs are dropped, which is
    their `filterNans`.

    The queue is `std::priority_queue` REPRODUCED, libc++ heap semantics
    included, in `_heap_push` / `_heap_pop`. This file used to scan a List
    for its maximum with a lowest-index tie-break, under a note claiming pop
    order only matters when scores tie and that theirs promises nothing on a
    tie either. **The first half of that note was the bug and the second
    half was an excuse.** The score is `log(n) - log(l) - log(r)` over
    INTEGER bin sizes, so equal-size bins tie EXACTLY, and they are the
    common case, not the rare one: near-halving makes whole tiers of
    same-size bins. A budget that lands on a complete tier (15 = 1+2+4+8)
    is order-invariant, which is why the border-parity check passed for
    months; budget 100 cuts mid-tier, and WHICH tied bins get split is
    decided by the heap. Measured on the oracle fixture at budget 100: the
    list scan got 1392 of 1600 borders wrong; a Python simulation with
    libc++ `push_heap`/`pop_heap` semantics matched CatBoost's saved
    borders on 16 of 16 features exactly. CatBoost wheels are built with
    clang and libc++, so libc++'s tie behavior is the one to reproduce:
    push sifts up on STRICTLY-smaller parents, pop re-seats the last
    element from the root taking the LEFT child of an equal pair. The push
    ORDER in the loop below (left, then the mutated right) is also load
    bearing for the same reason.
    """
    # their `filterNans`
    var clean = List[Float32]()
    for i in range(len(values)):
        if values[i] == values[i]:
            clean.append(values[i])
    if len(clean) == 0:
        return List[Float32]()

    _sort_ascending(clean)

    var root = TFeatureBin()
    root.bin_start = 0
    root.bin_end = len(clean)
    root.best_split = 0
    root.best_score = 0.0
    _update_best_split(root, clean)

    var bins = List[TFeatureBin]()
    _heap_push(bins, root)

    # their `GreedySplit` (`binarization.cpp:1500-1511`), verbatim: check the
    # TOP, pop it, split it, push LEFT then the mutated right.
    while len(bins) <= max_borders_count and bins[0].can_split():
        var top = bins[0].copy()
        _heap_pop(bins)

        # their `Split()`: the LEFT half becomes a new bin, `this` keeps the
        # right half and is rescored.
        var left = TFeatureBin()
        left.bin_start = top.bin_start
        left.bin_end = top.best_split
        _update_best_split(left, clean)

        top.bin_start = top.best_split
        _update_best_split(top, clean)

        _heap_push(bins, left)
        _heap_push(bins, top)

    # their border loop: every bin but the first contributes its LEFT edge,
    # placed midway between the last value below and the first value in it.
    var borders = List[Float32]()
    for i in range(len(bins)):
        if bins[i].is_first():
            continue
        var s = bins[i].bin_start
        borders.append(
            Float32(0.5) * clean[s - 1] + Float32(0.5) * clean[s]
        )
    _sort_ascending(borders)

    # their `THashSet<float>` drops duplicates; two adjacent bins can round to
    # the same border on a column with ties.
    var out = List[Float32]()
    for i in range(len(borders)):
        if i == 0 or borders[i] != borders[i - 1]:
            out.append(borders[i])
    return out^


def _sort_ascending(mut v: List[Float32]):
    """Their `Sort` is `std::sort`; ours is an LSD radix sort above a small
    cutoff and the stdlib prelude `sort` below it. Any ascending sort of
    NaN-free floats yields the same array bit-for-bit (the one ambiguity,
    +-0.0 placement, washes out: a border midway between the two zeros is
    +0.0 in either order), so borders are algorithm-invariant -- which is
    what makes the algorithm swap legal, and the recorded mse gates hold
    through it as they held through shell -> prelude. Measured on 100k
    random f32 (hostsort_bench, six interleaved reps): prelude 5.7-6.6 ms,
    radix 0.73-1.06 ms -- 8x -- and phase B of the border build runs this
    once per float column at the 100k subsample, so the swap is worth
    ~0.25 s at 500 features and ~1 s at 2000. The cutoff keeps tiny
    inputs (border lists, up to border_count entries) on the prelude
    sort, where counting-sort setup would dominate.

    The key transform is the same monotone float<->u32 twiddle the device
    sorter uses (`gpu_util/kernel/radix_sort.mojo`): negatives map to
    `~bits`, non-negatives to `bits | 0x80000000`, so unsigned key order
    is exactly ascending float order. NaN-free by contract here --
    `best_split` filters NaNs before sorting, as their `filterNans`
    does."""
    var n = len(v)
    if n < 2048:
        sort(v)
        return
    var keys = List[UInt32](capacity=n)
    keys.resize(n, UInt32(0))
    var tmp = List[UInt32](capacity=n)
    tmp.resize(n, UInt32(0))
    var vbits = v.unsafe_ptr().unsafe_bitcast[UInt32]()
    var kp = keys.unsafe_ptr()
    for i in range(n):
        var bits = vbits.unsafe_load(i)
        var k: UInt32
        if (bits & UInt32(0x80000000)) != UInt32(0):
            k = ~bits
        else:
            k = bits | UInt32(0x80000000)
        kp.unsafe_store(i, k)
    var src = rebind[MutPointer[UInt32, MutAnyOrigin]](keys.unsafe_ptr())
    var dst = rebind[MutPointer[UInt32, MutAnyOrigin]](tmp.unsafe_ptr())
    for p in range(4):
        var shift = UInt32(p * 8)
        var count = List[Int](capacity=256)
        count.resize(256, 0)
        var cp = count.unsafe_ptr()
        for i in range(n):
            var b = Int((src.unsafe_load(i) >> shift) & UInt32(0xFF))
            cp.unsafe_store(b, cp.unsafe_load(b) + 1)
        var run = 0
        for b in range(256):
            var c = cp.unsafe_load(b)
            cp.unsafe_store(b, run)
            run += c
        for i in range(n):
            var k = src.unsafe_load(i)
            var b = Int((k >> shift) & UInt32(0xFF))
            var pos = cp.unsafe_load(b)
            dst.unsafe_store(pos, k)
            cp.unsafe_store(b, pos + 1)
        var t = src
        src = dst
        dst = t
    # four passes: `src` points back at `keys`; untwiddle in place into v
    for i in range(n):
        var k = src.unsafe_load(i)
        var bits: UInt32
        if (k & UInt32(0x80000000)) != UInt32(0):
            bits = k ^ UInt32(0x80000000)
        else:
            bits = ~k
        vbits.unsafe_store(i, bits)


def binarize(value: Float32, borders: List[Float32]) -> Int:
    """Which bin a raw value falls in: the count of borders it exceeds.

    Their quantization convention, and the one
    `write_compressed_index_kernel` already assumes: a feature with `Folds`
    borders takes bins `0..Folds`, so this returns `len(borders)` at most.
    """
    var b = 0
    for i in range(len(borders)):
        if value > borders[i]:
            b += 1
        else:
            break
    return b


@no_inline
def _penalty_min_entropy(weight: Float64) -> Float64:
    """`Penalty<EPenaltyType::MinEntropy>` (`binarization.cpp:175`).

    The `1e-8` is theirs, and unlike `MaxSumLog`'s it is not only guarding
    zero: `w * log(w)` is already 0 at w=0 by limit, so the epsilon is a
    plain reproduction, not a fix.

    ## Why `@no_inline`, and it is NOT a performance hint

    **Mojo contracts this multiply-then-add into an FMA across the inlined
    call, and clang does not.** The DP evaluates `prev_error[i] +
    _penalty_min_entropy(...)`; inlined, Mojo emits `fma(w, log(w),
    prev_error[i])` keeping the product at full width, while C++ at clang's
    default `-ffp-contract=on` contracts only within ONE SOURCE EXPRESSION
    and `Penalty<type>(...)` is a separate call, so CatBoost adds the
    ROUNDED product. One ULP, and it lands exactly where it does damage: on
    a plateau the two arms of a symmetric pair stop being equal and the
    tie-break fires on the difference. `@no_inline` restores every
    symmetric pair bit-identically. Recorded as deviation 54.

    ## Why this does NOT call `std.math.log` (history: the libm fix)

    **`std.math.log` is not accurate enough to reproduce their tie-breaks,
    and this was measured, not assumed.** At w = 840 it returns a product
    of 5656.057589200282 where libm gives 5656.057589153382, an ABSOLUTE
    error of 4.7e-8. Their comparisons carry `Eps = 1e-12`
    (`binarization.cpp:219`), so our noise is four orders of magnitude
    larger than the tolerance the algorithm was written against.

    That does not merely perturb the answer, it changes which answer is
    reached. The DP has genuine PLATEAUS: with equal per-value weights,
    `sweights[j] - sweights[i]` depends only on `j - i`, so many cut sets
    have exactly equal cost and the tie-break alone decides. Exactly equal
    costs compare exactly equal in any deterministic arithmetic, so the
    tie-break fires consistently and CatBoost packs every wide bin at the
    end. With a noisy log the plateau stops being flat, the comparison
    fires on rounding instead, and the wide bins scatter. Measured on
    `bench/minentropy_oracle.txt` column 5 at budget 13: CatBoost cuts at
    1,3,5,7,9,11,13,15,17,19,22,25,28 and `std.math.log` gave
    1,3,5,8,10,13,16,18,20,22,24,26,28. Both are optimal, with the same
    multiset of bin sizes. Only one of them is CatBoost's.

    The recorded fix was the HOST LIBM's `log` through `external_call`,
    which reproduced CatBoost's borders exactly on `bench/minentropy_oracle.txt`
    because CatBoost itself computes them with the host's libm. That fix
    was host-only by nature -- no libm is reachable on a device -- and it
    made the identical path's bits depend on which C library the host
    links (IDENTITY_PATHS row 18's class: macOS and glibc disagree in the
    last bit).

    ## What it calls NOW (DEVIATION 2263, 2026-09-08)

    `portable_log64` from `checks/numerics.mojo`: the Cephes double log
    through fma and basic ops only, measured within 2 ulp of libm over
    2^18 hashed doubles (`check-portable-log64`), and -- the property that
    matters here -- the SAME BITS on every host and every device, so the
    plateau argument above holds everywhere at once: exactly equal costs
    still compare exactly equal, and the tie-break, not a libm, decides.
    Operand order and spelling are theirs, unchanged: `w * log(w + 1e-8)`.

    CONSEQUENCE, stated rather than hoped away: our bits now come from OUR
    log rather than the host's libm, while CatBoost's come from theirs.
    Where two DIFFERENT cut sets' costs are within a couple of ulp of each
    other (a near-tie, not an exact one -- exact ties are tie-broken the
    same way on both sides), the border selection can differ from
    CatBoost's by that ULP difference, and `check-minentropy` /
    `check-greedylogsum` / `ctr-target-oracle` compare against CatBoost's
    borders EXACTLY. If one of them moves after this change, that is the
    ULP consequence and not a regression of the tie-break: re-baseline the
    oracle card with the differing (column, budget) pair documented as a
    1-ULP near-tie, or accept a documented tolerance ONLY where the input
    is such a near-tie. Do not reach for libm again to make it green.

    `_penalty_max_sum_log` (GreedyLogSum) calls `portable_log64` as well.
    Its heap compares scores of bins with DIFFERENT integer sizes whose
    exact values differ by ~1e-10 (the `1e-8` epsilon's doing), inside
    `std.math.log`'s error; see its docstring for the measured divergence.
    """
    return weight * portable_log64(weight + 1e-8)  # DEVIATION 2263


def best_split_min_entropy(
    var values: List[Float32], max_borders_count: Int
) raises -> List[Float32]:
    """Their `MinEntropy` border selection for the CTR grids; see
    `_exact_best_split`, which carries the whole account. This entry is the
    CTR path's, unchanged in behavior since before the numeric border types
    arrived (lane/catboost-parity): no subnormal flush."""
    return _exact_best_split[PENALTY_MIN_ENTROPY](
        values^, max_borders_count, False
    )


def _exact_best_split[
    penalty: Int
](
    var values: List[Float32], max_borders_count: Int, flush: Bool
) raises -> List[Float32]:
    """Their `TExactBinarizer<penalty>` (`binarization.cpp:1151-1160`), the
    `MinEntropy` and `MaxLogSum` border selections.

    `flush` models a denormals-as-zero reader BY BITS (`ftz`) on the values
    as they enter and on the threshold midpoints, which is what the numeric
    border build's phase B worker does to a column (see
    `gbdt/host/gbdt_oracle.mojo::_calc_quantization_phase_b`); the CTR
    grids pass False, as they always have. The penalty is a comptime
    parameter so the MinEntropy instantiation is the code this function
    was before it took one.

    WHY THIS EXISTS SEPARATELY FROM `best_split`. `best_split` is
    `GreedyLogSum`, CatBoost's default for NUMERIC features. It is not the
    default for CTR values. `CreateDefaultCounter`
    (`catboost_options.cpp:392-415`) builds the GPU's FeatureFreq
    description with `TBinarizationOptions(MinEntropy, 15)` for simple CTRs
    and `Median` for tree CTRs, and `SetDefaultBinarizationsIfNeeded`
    (`:418-427`) sets it a second time for any description that left it
    unset. **The `Uniform, 15` at `cat_feature_options.cpp:169` is the
    generic `TCtrDescription` constructor default and the GPU path never
    reaches it for FeatureFreq.** Citing that line as the CTR binarizer is
    a real line on the wrong code path, which is how it survived review.

    `MinEntropy` dispatches to `TExactBinarizer<EPenaltyType::MinEntropy>`
    (`binarization.cpp:124`). This is their `BestSplit` in mode `E_RLM2`
    (`:678` picks the mode; `:450-585` is the body), over the sorted UNIQUE
    values weighted by their counts.

    ## Which mode, and what the mode did NOT explain

    This is their mode `E_RLM2` because that is the mode they run. It was
    first written as `E_Base` (`:241`), the plain O(wsize^2 * bins) form,
    and when the equal-weight fixture started failing the mode looked like
    the obvious culprit: `E_RLM2` is not an exact search, its scans advance
    while the error is within `Eps = 1e-12` and stop at the first strict
    improvement, so on a plateau it settles wherever its scan stops.

    **That was a good story and it was wrong. Switching E_Base to E_RLM2
    changed nothing: the same two cases failed with the same counts, 9 of
    13 and 16 of 23.** The cause was the LOG (see
    `_penalty_min_entropy`), and it was found by running the same dynamic
    program in Python against libm, where it reproduced CatBoost exactly.
    Both modes are correct here; the arithmetic under them was not. The
    mode stayed `E_RLM2` afterwards because faithfulness is the rule, not
    because it fixed anything.

    ## The shape of their scan

    `dsize = wsize - bins + 1`, so index `i` into the DP means split point
    `l + i` at level `l` (`:225`), and the thresholds get `+= l` at the end
    (`:661`) to undo it. Per level:

    * a FORWARD pass giving `bs1[j]`, the earliest good split for `j`,
    * an INVERTED pass giving `bs2[j]`, the same from the other end,
    * a reconciliation that rebuilds any `k` where the two disagree by more
      than one (`while bs1[k] + 1 < bs2[k]`) until they meet.

    **The tie-breaks are opposite between the levels and the last match.**
    The per-level scans keep advancing while `newError <= bestError + Eps`,
    so the LAST index of a plateau wins; the last match (`:637`) takes
    `newError < bestError`, so the FIRST wins. Ties are the common case
    here, not an edge: FeatureFreq values are `count / (n + 1)`, so
    categories sharing a count share a value exactly, and equal weights
    make `Penalty(sweights[j] - sweights[i])` depend only on `j - i`.
    `bench/minentropy_oracle.txt` carries a column family built to have
    exactly that structure, because the first family did not: a deliberate
    sabotage of the last-match tie-break on it did not move one border.
    """
    var clean = List[Float32]()
    for i in range(len(values)):
        if values[i] == values[i]:
            clean.append(ftz(values[i]) if flush else values[i])
    if len(clean) == 0:
        return List[Float32]()

    _sort_ascending(clean)

    # their unique values plus per-value weights (`:1029`)
    var uniques = List[Float32]()
    var weights = List[Float64]()
    for i in range(len(clean)):
        if i > 0 and clean[i] == clean[i - 1]:
            weights[len(weights) - 1] += 1.0
        else:
            uniques.append(clean[i])
            weights.append(1.0)

    var wsize = len(uniques)
    var bins = max_borders_count + 1
    if bins <= 1 or wsize == 0:
        return List[Float32]()

    var thresholds = List[Int]()
    for _ in range(bins - 1):
        thresholds.append(0)

    if wsize <= bins:
        # their short-circuit (`:208-216`): every value gets its own bin
        for i in range(wsize - 1):
            thresholds[i] = i
        for i in range(wsize - 1, bins - 1):
            thresholds[i] = wsize - 1
        return _thresholds_to_borders(thresholds, uniques, flush)

    var sweights = List[Float64]()
    var running = 0.0
    for i in range(wsize):
        running += weights[i]
        sweights.append(running)

    comptime EPS = 1e-12
    var dsize = wsize - bins + 1

    var best_solutions = List[Int]()
    for _ in range((bins - 2) * dsize):
        best_solutions.append(0)

    var current_error = List[Float64]()
    for i in range(dsize):
        current_error.append(_exact_penalty[penalty](sweights[i]))
    var prev_error: List[Float64]

    var bs1 = List[Int]()
    var bs2 = List[Int]()
    var e1 = List[Float64]()
    var e2 = List[Float64]()
    for _ in range(dsize):
        bs1.append(0)
        bs2.append(0)
        e1.append(0.0)
        e2.append(0.0)

    for l in range(bins - 2):
        prev_error = current_error.copy()

        # their "First forward loop" (`:451`)
        var fi = 0
        for j in range(dsize):
            var best_error = prev_error[fi] + _exact_penalty[penalty](
                sweights[l + j + 1] - sweights[l + fi]
            )
            fi += 1
            while fi <= j:
                var new_error = prev_error[fi] + _exact_penalty[penalty](
                    sweights[l + j + 1] - sweights[l + fi]
                )
                if new_error > best_error + EPS:
                    break
                best_error = new_error
                fi += 1
            fi -= 1
            bs1[j] = fi
            e1[j] = best_error

        # their "First inverted loop" (`:468`)
        var vi = 0
        for j in range(dsize):
            if j > vi:
                vi = j
            var maxi = dsize - bs1[dsize - j - 1] - 1
            if vi + 1 >= maxi:
                bs2[dsize - j - 1] = bs1[dsize - j - 1]
                e2[dsize - j - 1] = e1[dsize - j - 1]
                vi = maxi
                continue
            var best_error = e1[dsize - j - 1]
            while vi + 1 < maxi:
                var new_error = prev_error[
                    dsize - vi - 1
                ] + _exact_penalty[penalty](
                    sweights[l + dsize - j] - sweights[l + dsize - vi - 1]
                )
                if new_error + EPS < best_error:
                    best_error = new_error
                    break
                vi += 1
            if vi + 1 >= maxi:
                vi = maxi
            else:
                vi += 1
                while vi + 1 < maxi:
                    var new_error = prev_error[
                        dsize - vi - 1
                    ] + _exact_penalty[penalty](
                        sweights[l + dsize - j] - sweights[l + dsize - vi - 1]
                    )
                    if new_error > best_error + EPS:
                        break
                    best_error = new_error
                    vi += 1
                vi -= 1
            bs2[dsize - j - 1] = dsize - vi - 1
            e2[dsize - j - 1] = best_error

        # their reconciliation (`:503`): rebuild until the two bounds meet
        for k in range(dsize):
            while bs1[k] + 1 < bs2[k]:
                var maxj = dsize

                # "Forward loop" (`:508`)
                var ri = bs1[k] + 2
                var j = k
                while j < maxj:
                    if ri <= bs1[j]:
                        maxj = j
                        break
                    var maxi = bs2[j]
                    if ri + 1 >= maxi:
                        ri = maxi
                        bs1[j] = ri
                        e1[j] = e2[j]
                        j += 1
                        continue
                    var best_error = e2[j]
                    while ri + 1 < maxi:
                        var new_error = prev_error[
                            ri
                        ] + _exact_penalty[penalty](
                            sweights[l + j + 1] - sweights[l + ri]
                        )
                        if new_error + EPS < best_error:
                            best_error = new_error
                            break
                        ri += 1
                    if ri + 1 >= maxi:
                        ri = maxi
                    else:
                        ri += 1
                        while ri + 1 < maxi:
                            var new_error = prev_error[
                                ri
                            ] + _exact_penalty[penalty](
                                sweights[l + j + 1] - sweights[l + ri]
                            )
                            if new_error > best_error + EPS:
                                break
                            best_error = new_error
                            ri += 1
                        ri -= 1
                    bs1[j] = ri
                    e1[j] = best_error
                    j += 1

                # "Inverted loop" (`:551`)
                var j1 = dsize - maxj
                var j2 = dsize - k
                var qi = dsize - bs2[dsize - j1 - 1] - 1 + 2
                var jj = j1
                while jj < j2:
                    var maxi = dsize - bs1[dsize - jj - 1] - 1
                    if qi + 1 >= maxi:
                        bs2[dsize - jj - 1] = bs1[dsize - jj - 1]
                        e2[dsize - jj - 1] = e1[dsize - jj - 1]
                        qi = maxi
                        jj += 1
                        continue
                    var best_error = e1[dsize - jj - 1]
                    while qi + 1 < maxi:
                        var new_error = prev_error[
                            dsize - qi - 1
                        ] + _exact_penalty[penalty](
                            sweights[l + dsize - jj]
                            - sweights[l + dsize - qi - 1]
                        )
                        if new_error + EPS < best_error:
                            best_error = new_error
                            break
                        qi += 1
                    if qi + 1 >= maxi:
                        qi = maxi
                    else:
                        qi += 1
                        while qi + 1 < maxi:
                            var new_error = prev_error[
                                dsize - qi - 1
                            ] + _exact_penalty[penalty](
                                sweights[l + dsize - jj]
                                - sweights[l + dsize - qi - 1]
                            )
                            if new_error > best_error + EPS:
                                break
                            best_error = new_error
                            qi += 1
                        qi -= 1
                    bs2[dsize - jj - 1] = dsize - qi - 1
                    e2[dsize - jj - 1] = best_error
                    jj += 1

            # "Everything is fine now!" (`:583`)
            best_solutions[l * dsize + k] = bs1[k]
            current_error[k] = e1[k]

    # their "Last match" (`:629`), the non-E_Base arm
    prev_error = current_error.copy()
    var l_last = bins - 2
    var j_last = dsize - 1
    var best_index = 0
    var best_error = prev_error[0] + _exact_penalty[penalty](
        sweights[l_last + j_last + 1] - sweights[l_last]
    )
    for i in range(1, j_last + 1):
        var new_error = prev_error[i] + _exact_penalty[penalty](
            sweights[l_last + j_last + 1] - sweights[l_last + i]
        )
        # `<`: the FIRST index wins a tie here, opposite to the scans above
        if new_error < best_error:
            best_error = new_error
            best_index = i

    thresholds[bins - 2] = best_index
    var l_back = bins - 2
    while l_back > 0:
        best_index = best_solutions[(l_back - 1) * dsize + best_index]
        thresholds[l_back - 1] = best_index
        l_back -= 1

    # their "Adjust" (`:661`), undoing the `l` offset baked into dsize
    for i in range(len(thresholds)):
        thresholds[i] += i

    return _thresholds_to_borders(thresholds, uniques, flush)


def _thresholds_to_borders(
    thresholds: List[Int], uniques: List[Float32], flush: Bool = False
) raises -> List[Float32]:
    """Their threshold-to-border conversion (`binarization.cpp:679-694`).

    A threshold is the index AFTER which a border falls. `t + 1 ==
    size` is dropped (a border past the last value splits nothing), the
    border is the MIDPOINT of the pair, and the results go into a set, so
    repeated thresholds collapse to one border.
    """
    var out = List[Float32]()
    for k in range(len(thresholds)):
        var t = thresholds[k]
        if t + 1 == len(uniques):
            continue
        var b = (uniques[t] + uniques[t + 1]) / 2
        if flush:
            b = ftz(ftz(uniques[t] + uniques[t + 1]) / 2)
        var seen = False
        for m in range(len(out)):
            if out[m] == b:
                seen = True
                break
        if not seen:
            out.append(b)
    _sort_ascending(out)
    return out^


# ===========================================================================
# THE SEVEN NUMERIC BORDER TYPES (`feature_border_type`, lane/catboost-parity
# 2026-09-19). Their `MakeBinarizer` (`binarization.cpp:114-134`):
#
#     UniformAndQuantiles -> TMedianPlusUniformBinarizer      (:1224-1260)
#     GreedyLogSum        -> TGreedyBinarizer<MaxSumLog>      (:1677-1715)
#     GreedyMinEntropy    -> TGreedyBinarizer<MinEntropy>     (:1677-1715)
#     MaxLogSum           -> TExactBinarizer<MaxSumLog>       (:1151-1160)
#     MinEntropy          -> TExactBinarizer<MinEntropy>      (:1151-1160)
#     Median              -> TMedianBinarizer                 (:1201-1222)
#     Uniform             -> TUniformBinarizer                (:1262-1310)
#
# The dense float path reaches each with no default value and no initial
# borders, so those two arms of theirs are absent here, and every result
# goes through `SetQuantization` (`:888-951`): -0.0 becomes +0.0, the set is
# sorted. GreedyLogSum is `best_split` above, UNCHANGED: it is the default,
# and its bits are the ones every recorded lane holds.
#
# THE FLUSH. The six other types run with `flush=True` on BOTH the device
# fit's border build and the host oracle, applying `ftz` BY BITS to the
# values as they enter and to every float32 intermediate a border is made
# from. The device fit's phase B worker reads subnormals as signed zeros
# (the measurement in `gbdt/host/gbdt_oracle.mojo::_calc_quantization_phase_b`)
# and `best_split` relies on that thread mode; these types do not rely on
# it, so the border is the same whatever a runner's MXCSR or FPCR says.
# Against CatBoost this is the same deviation GreedyLogSum already carries:
# a column of subnormals is binned as zeros.
# ===========================================================================

comptime BORDER_TYPES_SABOTAGE = is_defined["MOJOLEARN_BORDER_TYPES_SABOTAGE"]()
"""THE NEGATIVE CONTROL of the six non-default types:
`-D MOJOLEARN_BORDER_TYPES_SABOTAGE=1` drops the MIDDLE border
`select_borders` returns (the one a split most often takes; the largest,
tried first, moved only the saved model's border table on the identity
lane), on every type it serves and never on GreedyLogSum
(`best_split` is not touched). `checks/border_types_check.mojo` must then
fail every non-default case that has a border and pass every GreedyLogSum
case, and the gbdt-border-types identity lane must move."""

comptime BORDER_TYPE_GREEDY_LOG_SUM = 0
comptime BORDER_TYPE_MEDIAN = 1
comptime BORDER_TYPE_UNIFORM = 2
comptime BORDER_TYPE_UNIFORM_AND_QUANTILES = 3
comptime BORDER_TYPE_MAX_LOG_SUM = 4
comptime BORDER_TYPE_MIN_ENTROPY = 5
comptime BORDER_TYPE_GREEDY_MIN_ENTROPY = 6

comptime PENALTY_MIN_ENTROPY = 0
comptime PENALTY_MAX_SUM_LOG = 1


def border_type_from_name(name: String) raises -> Int:
    """Their `EBorderSelectionType` spellings (`enums.h`). Empty is the
    default, GreedyLogSum (`data_processing_options.cpp:15`)."""
    if name == "" or name == "GreedyLogSum":
        return BORDER_TYPE_GREEDY_LOG_SUM
    if name == "Median":
        return BORDER_TYPE_MEDIAN
    if name == "Uniform":
        return BORDER_TYPE_UNIFORM
    if name == "UniformAndQuantiles":
        return BORDER_TYPE_UNIFORM_AND_QUANTILES
    if name == "MaxLogSum":
        return BORDER_TYPE_MAX_LOG_SUM
    if name == "MinEntropy":
        return BORDER_TYPE_MIN_ENTROPY
    if name == "GreedyMinEntropy":
        return BORDER_TYPE_GREEDY_MIN_ENTROPY
    raise Error(
        "unknown feature_border_type '" + name + "': GreedyLogSum, Median,"
        " Uniform, UniformAndQuantiles, MaxLogSum, MinEntropy, GreedyMinEntropy"
    )


def border_type_name(code: Int) -> String:
    if code == BORDER_TYPE_MEDIAN:
        return "Median"
    if code == BORDER_TYPE_UNIFORM:
        return "Uniform"
    if code == BORDER_TYPE_UNIFORM_AND_QUANTILES:
        return "UniformAndQuantiles"
    if code == BORDER_TYPE_MAX_LOG_SUM:
        return "MaxLogSum"
    if code == BORDER_TYPE_MIN_ENTROPY:
        return "MinEntropy"
    if code == BORDER_TYPE_GREEDY_MIN_ENTROPY:
        return "GreedyMinEntropy"
    return "GreedyLogSum"


@no_inline
def _penalty_max_sum_log_portable(weight: Float64) -> Float64:
    """`Penalty<EPenaltyType::MaxSumLog>` (`binarization.cpp:179`) through
    `portable_log64`, for the two NUMERIC types that reach it outside
    `best_split` (MaxLogSum's dynamic program). The dynamic program
    compares costs reached by DIFFERENT summation paths, which is the case
    `_penalty_min_entropy`'s docstring measured a host libm moving; the
    portable log gives every host the same bits (DEVIATION 2263's reason,
    applied to the second penalty)."""
    return -portable_log64(weight + 1e-8)


@always_inline
def _exact_penalty[penalty: Int](weight: Float64) -> Float64:
    comptime if penalty == PENALTY_MIN_ENTROPY:
        return _penalty_min_entropy(weight)
    else:
        return _penalty_max_sum_log_portable(weight)


# ---- TGreedyBinarizer<MinEntropy> ------------------------------------------


def _greedy_split_score[
    penalty: Int
](bin_start: Int, bin_end: Int, split_pos: Int) -> Float64:
    """`TFeatureBin<penalty>::CalcSplitScore` (`binarization.cpp:1398-1406`)
    over integer bin sizes. Their edge value is `-infinity`; this is
    `-1.0e308`, `_calc_split_score`'s stand-in, which orders the same
    against every finite score."""
    if split_pos == bin_start or split_pos == bin_end:
        return -1.0e308
    var left = -_exact_penalty[penalty](Float64(split_pos - bin_start))
    var right = -_exact_penalty[penalty](Float64(bin_end - split_pos))
    var curr = -_exact_penalty[penalty](Float64(bin_end - bin_start))
    return left + right - curr


def _greedy_update[penalty: Int](mut b: TFeatureBin, values: List[Float32]):
    """`UpdateBestSplitProperties` (`:1408-1423`): the lower and upper bound
    of the midpoint value, binary searches as theirs, the left candidate on
    a tie (`scoreLeft >= scoreRight`)."""
    var mid = b.bin_start + (b.bin_end - b.bin_start) // 2
    var mid_value = values[mid]
    var lb = b.bin_start
    var hi = mid
    while lb < hi:
        var m = lb + (hi - lb) // 2
        if values[m] < mid_value:
            lb = m + 1
        else:
            hi = m
    var ub = mid
    var hi2 = b.bin_end
    while ub < hi2:
        var m2 = ub + (hi2 - ub) // 2
        if values[m2] <= mid_value:
            ub = m2 + 1
        else:
            hi2 = m2
    var score_left = _greedy_split_score[penalty](b.bin_start, b.bin_end, lb)
    var score_right = _greedy_split_score[penalty](b.bin_start, b.bin_end, ub)
    if score_left >= score_right:
        b.best_split = lb
        b.best_score = score_left
    else:
        b.best_split = ub
        b.best_score = score_right


def _greedy_best_split[
    penalty: Int
](values: List[Float32], max_borders_count: Int) raises -> List[Float32]:
    """`TGreedyBinarizer<penalty>::BestSplit` with no default value
    (`:1698-1714`) over already-flushed values: sort, `GreedySplit`
    (`:1500-1520`, the libc++ heap `best_split` reproduces), then each bin
    but the first contributes its LEFT border, `0.5f * below + 0.5f *
    above` (`IFeatureBin::LeftBorder`, `:1360-1373`)."""
    var clean = values.copy()
    _sort_ascending(clean)
    var root = TFeatureBin()
    root.bin_start = 0
    root.bin_end = len(clean)
    root.best_split = 0
    root.best_score = 0.0
    _greedy_update[penalty](root, clean)
    var bins = List[TFeatureBin]()
    _heap_push(bins, root)
    while len(bins) <= max_borders_count and bins[0].can_split():
        var top = bins[0].copy()
        _heap_pop(bins)
        var left = TFeatureBin()
        left.bin_start = top.bin_start
        left.bin_end = top.best_split
        _greedy_update[penalty](left, clean)
        top.bin_start = top.best_split
        _greedy_update[penalty](top, clean)
        _heap_push(bins, left)
        _heap_push(bins, top)
    var out = List[Float32]()
    for i in range(len(bins)):
        if bins[i].is_first():
            continue
        var s = bins[i].bin_start
        var half_below = ftz(Float32(0.5) * clean[s - 1])
        var half_above = ftz(Float32(0.5) * clean[s])
        out.append(ftz(half_below + half_above))
    return out^


# ---- TMedianBinarizer, TUniformBinarizer, TMedianPlusUniformBinarizer -------


def _lower_bound(sorted: List[Float32], value: Float32) -> Int:
    """`LowerBound`: the first index holding a value `>= value`."""
    var lo = 0
    var hi = len(sorted)
    while lo < hi:
        var m = lo + (hi - lo) // 2
        if sorted[m] < value:
            lo = m + 1
        else:
            hi = m
    return lo


def _regular_border(border: Float32, sorted: List[Float32]) -> Float32:
    """`RegularBorder` (`binarization.cpp:698-733`) with no initial borders:
    the border BEFORE the first element holding `border`, the float32
    midpoint of it and its predecessor, or the predecessor itself on a
    wrong-side rounding. The two edge arms (a border past the last value, a
    border at or below the first) are theirs verbatim."""
    var lb = _lower_bound(sorted, border)
    var n = len(sorted)
    if lb == n:
        var back = sorted[n - 1]
        return max(ftz(Float32(2.0) * back), ftz(back + Float32(1.0)))
    if lb == 0:
        var front = sorted[0]
        return min(ftz(Float32(0.5) * front), ftz(Float32(2.0) * front))
    var res = ftz(ftz(sorted[lb] + sorted[lb - 1]) * Float32(0.5))
    if res == sorted[lb]:
        res = sorted[lb - 1]
    return res


def _median_borders(
    sorted: List[Float32], max_borders_count: Int, mut out: List[Float32]
):
    """`GenerateMedianBorders` (`binarization.cpp:1046-1063`): border i sits
    before the value at the `(i + 1) / (count + 1)` quantile index, integer
    arithmetic in 64 bits as theirs, skipped where that value is the
    minimum."""
    var total = len(sorted)
    if total == 0 or sorted[0] == sorted[total - 1]:
        return
    for i in range(max_borders_count):
        var i1 = (i + 1) * total // (max_borders_count + 1)
        if i1 > total - 1:
            i1 = total - 1
        var val1 = sorted[i1]
        if val1 != sorted[0]:
            out.append(_regular_border(val1, sorted))


def _uniform_value(
    min_value: Float32, max_value: Float32, i: Int, parts: Int
) -> Float32:
    """`minValue + (i + 1) * (maxValue - minValue) / (parts)`, which C++
    evaluates in FLOAT (`int * float` is float, `:1289`, `:1249`), one
    rounding per operation, each flushed as the worker would."""
    var span = ftz(max_value - min_value)
    var scaled = ftz(Float32(i + 1) * span)
    var step = ftz(scaled / Float32(parts))
    return ftz(min_value + step)


def select_borders(
    var values: List[Float32], max_borders_count: Int, border_type: Int
) raises -> List[Float32]:
    """Their `NSplitSelection::BestSplit` (`binarization.cpp:71-110`) for the
    six NON-default numeric border types, `flush` always on (see the block
    above). NaNs are dropped first (their `filterNans`); an empty column has
    no borders. The result is sorted ascending, duplicates merged and
    -0.0 written as +0.0 (`SetQuantization`, `:897-904`)."""
    if border_type == BORDER_TYPE_GREEDY_LOG_SUM:
        raise Error(
            "select_borders is the non-default arm; GreedyLogSum is best_split"
        )
    var clean = List[Float32]()
    for i in range(len(values)):
        if values[i] == values[i]:
            clean.append(ftz(values[i]))
    var raw = List[Float32]()
    if len(clean) == 0 or max_borders_count <= 0:
        return raw^
    if border_type == BORDER_TYPE_MIN_ENTROPY:
        raw = _exact_best_split[PENALTY_MIN_ENTROPY](
            clean^, max_borders_count, True
        )
    elif border_type == BORDER_TYPE_MAX_LOG_SUM:
        raw = _exact_best_split[PENALTY_MAX_SUM_LOG](
            clean^, max_borders_count, True
        )
    elif border_type == BORDER_TYPE_GREEDY_MIN_ENTROPY:
        raw = _greedy_best_split[PENALTY_MIN_ENTROPY](clean, max_borders_count)
    elif border_type == BORDER_TYPE_MEDIAN:
        _sort_ascending(clean)
        _median_borders(clean, max_borders_count, raw)
    elif border_type == BORDER_TYPE_UNIFORM:
        # `TUniformBinarizer::BestSplit` (`:1262-1310`): MinMaxElement, no
        # sort, `maxBordersCount` evenly spaced values strictly inside
        var lo = clean[0]
        var hi = clean[0]
        for i in range(1, len(clean)):
            if clean[i] < lo:
                lo = clean[i]
            if hi < clean[i]:
                hi = clean[i]
        if lo == hi:
            return List[Float32]()
        for i in range(max_borders_count):
            raw.append(_uniform_value(lo, hi, i, max_borders_count + 1))
    elif border_type == BORDER_TYPE_UNIFORM_AND_QUANTILES:
        # `TMedianPlusUniformBinarizer::BestSplit` (`:1224-1260`): the median
        # borders take `count - count / 2`, the uniform ones `count / 2`, each
        # uniform value then regularized onto the data like a median one
        _sort_ascending(clean)
        var n = len(clean)
        if clean[0] == clean[n - 1]:
            return List[Float32]()
        var half = max_borders_count // 2
        _median_borders(clean, max_borders_count - half, raw)
        for i in range(half):
            raw.append(
                _regular_border(
                    _uniform_value(clean[0], clean[n - 1], i, half + 1), clean
                )
            )
    else:
        raise Error("unknown border type code " + String(border_type))
    # `SetQuantization`: -0.0 -> +0.0, then the set, sorted
    for i in range(len(raw)):
        if raw[i] == Float32(0.0):
            raw[i] = Float32(0.0)
    _sort_ascending(raw)
    var out = List[Float32]()
    for i in range(len(raw)):
        if i == 0 or raw[i] != raw[i - 1]:
            out.append(raw[i])
    comptime if BORDER_TYPES_SABOTAGE:
        if len(out) > 0:
            _ = out.pop(len(out) // 2)
    return out^
