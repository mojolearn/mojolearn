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


def best_split(
    var values: List[Float32], max_borders_count: Int
) raises -> List[Float32]:
    """Their `BestSplit` for `GreedyLogSum` (`binarization.h:23`).

    `values` is consumed and sorted, as theirs is. NaNs are dropped, which is
    their `filterNans`.

    ================= DEVIATION BLOCK =================
    Theirs uses `std::priority_queue<TBinType>`. Mojo 1.0 has no heap in the
    standard library, so the queue is a `List` scanned for its maximum each
    round. That is O(bins) per pop against their O(log bins), and the loop
    runs at most `max_borders_count` times, so at CatBoost's default of 254
    it is about 32,000 comparisons for a whole column. The sort above it is
    already O(n log n) over every row, so this is not the cost.

    The ORDER of pops is identical whenever scores are distinct. On an exact
    tie `std::priority_queue` gives no ordering guarantee either, so nothing
    is being promised here that theirs promises.
    ===================================================
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
    bins.append(root)

    # their `while (splits.size() <= maxBordersCount && splits.top().CanSplit())`
    while len(bins) <= max_borders_count:
        var top = 0
        for i in range(1, len(bins)):
            if bins[i].best_score > bins[top].best_score:
                top = i
        if not bins[top].can_split():
            break

        # their `Split()`: the LEFT half becomes a new bin, `this` keeps the
        # right half and is rescored.
        var parent = bins[top].copy()
        var left = TFeatureBin()
        left.bin_start = parent.bin_start
        left.bin_end = parent.best_split
        _update_best_split(left, clean)

        var right = TFeatureBin()
        right.bin_start = parent.best_split
        right.bin_end = parent.bin_end
        _update_best_split(right, clean)

        bins[top] = right
        bins.append(left)

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
