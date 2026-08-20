"""Border selection: turning a raw float column into split points.

PORT OF `library/cpp/grid_creator/binarization.cpp` at CatBoost `54a8143a`,
the `GreedyLogSum` path, which is CatBoost's DEFAULT
(`catboost/private/libs/options/data_processing_options.cpp:15`).

**This runs on the HOST in CatBoost and it runs on the host here.** Border
selection is a sort plus a priority queue over at most `border_count` bins;
there is nothing in it a GPU wants. Porting their host code as host code is
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

from std.math import log


def _penalty_max_sum_log(weight: Float64) -> Float64:
    """`Penalty<EPenaltyType::MaxSumLog>` (`:179`). The `1e-8` is theirs and
    is what keeps a bin of size zero finite."""
    return -log(weight + 1e-8)


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
    and `UpperBound`, done linearly here because the ranges shrink fast and a
    binary search would be the only clever thing in this file.
    """
    var mid = b.bin_start + (b.bin_end - b.bin_start) // 2
    var mid_value = values[mid]

    var lb = b.bin_start
    while lb < mid and values[lb] < mid_value:
        lb += 1

    var ub = mid
    while ub < b.bin_end and values[ub] <= mid_value:
        ub += 1

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
    """Plain insertion-free shell sort. Their `Sort` is `std::sort`; nothing
    downstream depends on which algorithm gets there."""
    var n = len(v)
    var gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            var tmp = v[i]
            var j = i
            while j >= gap and v[j - gap] > tmp:
                v[j] = v[j - gap]
                j -= gap
            v[j] = tmp
        gap //= 2


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
