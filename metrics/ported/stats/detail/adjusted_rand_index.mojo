"""RAFT `cpp/include/raft/stats/detail/adjusted_rand_index.cuh` (ebf9268).

THEIRS (:113-192), `MathT = unsigned long long` (cuML passes it, `adjusted_
rand_index.cu`):

    if (size < 2) return 1.0;
    nUniqFirst/Second = countUnique(...)  (:74-101: minmax + histogram + count nonzero)
    lower = min(minFirst, minSecond); upper = max(maxFirst, maxSecond)
    if (nUniqFirst == nUniqSecond && (nUniqFirst == 1 || nUniqFirst == size)) return 1.0;
    C = contingencyMatrix(first, second, size, lower, upper)     int, k x k
    nChooseTwoSum = sum_ij nC2(C_ij)           a = row sums, b = column sums
    aCTwoSum = sum_i nC2(a_i); bCTwoSum = sum_j nC2(b_j)          (all MathT, exact)
    nChooseTwo    = double(size) * double(size - 1) / 2.0
    expectedIndex = double(aC2) * double(bC2) / nChooseTwo
    maxIndex      = (double(bC2) + double(aC2)) / 2.0
    index         = double(nChooseTwoSum)
    return (maxIndex - expectedIndex) ? (index - expectedIndex) / (maxIndex - expectedIndex) : 0

`nCTwo(in) = in % 2 ? ((in - 1) >> 1) * in : (in >> 1) * (in - 1)` (:43-49),
the integer `in * (in - 1) / 2` without the intermediate overflow.

sklearn `adjusted_rand_score`: `(n_classes == n_clusters == 1) or
(n_classes == n_clusters == 0) or (n_classes == n_clusters == n_samples)
-> 1.0`, then `(tp*tn - fn*fp) / ((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn))` from
the pair confusion matrix -- algebraically the same ratio as RAFT's
`(index - expected) / (max - expected)`, with the 0/0 case returning ... a
NaN warning in sklearn where RAFT returns 0 (:188-191). Ours mirrors RAFT.

EVERYTHING BUT THE LAST FIVE HOST OPS IS INTEGER and exact; those five are
Float64 multiplies, divisions and subtractions on the host, correctly
rounded everywhere (no transcendental), so this metric is identity-safe in
both modes with no IDENTICAL arm. The integer sums over the matrix
(`mapThenSumReduce<MathT, nCTwo>`, `reduce<MathT>`) are done on the host
from the read-back matrix (DEVIATION 650's split) in Int64, which holds
`nC2` of any count below 2^32.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from metrics.ported.stats.detail.contingency_matrix import (
    get_input_class_cardinality,
)
from metrics.ported.stats.detail.histogram import histogram
from metrics.ported.stats.detail.mutual_info_score import (
    col_sums,
    contingency_matrix_host,
    row_sums,
)


def n_c_two(v: Int64) -> Int64:
    """`nCTwo` (:43-49)."""
    if v % 2 != 0:
        return ((v - 1) >> 1) * v
    return (v >> 1) * (v - 1)


def count_unique(
    ctx: DeviceContext,
    mut arr: DeviceBuffer[DType.int32],
    size: Int,
) raises -> Tuple[Int, Int32, Int32]:
    """`countUnique` (:74-101): returns `(numUniques, minLabel, maxLabel)`.
    The histogram is the device's (integer, exact); the count of nonzero
    bins (`mapThenSumReduce(val != 0)`, an integer sum) is done on the host
    from the read-back bins (DEVIATION 650's split)."""
    var mm = get_input_class_cardinality(ctx, arr, size)
    var min_label = mm[0]
    var max_label = mm[1]
    var total_labels = Int(max_label - min_label + 1)
    var bins = ctx.enqueue_create_buffer[DType.int32](total_labels)
    histogram(ctx, bins, total_labels, arr, size, min_label)
    var h = ctx.enqueue_create_host_buffer[DType.int32](total_labels)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=bins)
    ctx.synchronize()
    var n_uniq = 0
    for i in range(total_labels):
        if h.unsafe_ptr().unsafe_load(i) != Int32(0):
            n_uniq += 1
    _ = h^
    _ = bins^
    return (n_uniq, min_label, max_label)


def adjusted_rand_index_from_contingency(
    c: List[Int32], k: Int, size: Int
) raises -> Float64:
    """Lines :161-191 from a host copy of the matrix. The untraced entry;
    `adjusted_rand_index_from_contingency_traced` is the implementation."""
    var off = IdentityTrace.disabled()
    return adjusted_rand_index_from_contingency_traced(off, c, k, size)


def adjusted_rand_index_from_contingency_traced(
    mut trace: IdentityTrace, c: List[Int32], k: Int, size: Int
) raises -> Float64:
    """The same epilogue, recording `metrics.ari.pair_sums` (i64, 3):
    `(nChooseTwoSum, aCTwoSum, bCTwoSum)`.

    WHY THESE THREE. ARI is `(index - expected) / (max - expected)`, A
    DIFFERENCE OF LARGE NUMBERS OVER A DIFFERENCE OF LARGE NUMBERS: with
    `expected` close to `max`, both differences are catastrophic
    cancellations and the quotient tells you nothing about which operand
    moved. These three integers are every input to those five host Float64
    ops, they are EXACT (Int64 sums of `nCTwo`), and therefore a difference
    in one of them is a DEFECT and not a rounding -- a distinction the
    single recorded `metrics.adjusted_rand_index` cannot make.

    `n_choose_two`, `expected_index`, `max_index` and `index` are NOT
    recorded: each is a correctly-rounded host op on these three recorded
    integers and on `size`, so they are derivable, and adding them would
    record the same information four more times. The `max - expected == 0`
    guard is likewise a compare on derivable values.

    Tags are fixed rather than caller-supplied because ARI is computed once
    per card; entropy and MI take a `tag_prefix` because they are not."""
    var n_choose_two_sum = Int64(0)
    for idx in range(k * k):
        n_choose_two_sum += n_c_two(Int64(c[idx]))
    var a = row_sums(c, k)
    var b = col_sums(c, k)
    var a_c_two_sum = Int64(0)
    var b_c_two_sum = Int64(0)
    for i in range(k):
        a_c_two_sum += n_c_two(a[i])
        b_c_two_sum += n_c_two(b[i])
    var pair_sums = List[Int64]()
    pair_sums.append(n_choose_two_sum)
    pair_sums.append(a_c_two_sum)
    pair_sums.append(b_c_two_sum)
    trace.record_host("metrics.ari.pair_sums", pair_sums.unsafe_ptr(), 3)
    _ = pair_sums^
    var n_choose_two = Float64(size) * Float64(size - 1) / 2.0
    var expected_index = (
        Float64(a_c_two_sum) * Float64(b_c_two_sum) / n_choose_two
    )
    var max_index = (Float64(b_c_two_sum) + Float64(a_c_two_sum)) / 2.0
    var index = Float64(n_choose_two_sum)
    if max_index - expected_index != 0.0:
        return (index - expected_index) / (max_index - expected_index)
    return 0.0


def compute_adjusted_rand_index(
    ctx: DeviceContext,
    mut first_cluster_array: DeviceBuffer[DType.int32],
    mut second_cluster_array: DeviceBuffer[DType.int32],
    size: Int,
) raises -> Float64:
    """`compute_adjusted_rand_index(first, second, size, stream)` (:113-192)."""
    var off = IdentityTrace.disabled()
    return compute_adjusted_rand_index_traced(
        ctx, off, first_cluster_array, second_cluster_array, size
    )


def compute_adjusted_rand_index_traced(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut first_cluster_array: DeviceBuffer[DType.int32],
    mut second_cluster_array: DeviceBuffer[DType.int32],
    size: Int,
) raises -> Float64:
    """ARI carrying a card: `metrics.ari.uniq`, `metrics.ari.contingency`,
    `metrics.ari.pair_sums`.

    `metrics.ari.uniq` (i32, 4) is `(nUniqFirst, nUniqSecond, lowerLabel,
    upperLabel)` -- ARI's OWN label range, derived from `countUnique` over
    BOTH arrays (:123-127), which need not be the `(lower, upper)` the
    caller passes to entropy and MI. It is recorded BEFORE the early
    return, so the DECISION at :130-132 (`nUniqFirst == nUniqSecond` and
    either 1 or `size` -> return 1.0) is on the card whether or not it
    fires. A decision that changes no number still needs a stage: that is
    the holtwinters `CRIT_ORDER` lesson (DEVIATION 665), where the stop
    criterion moves ZERO of 2800 cells and is visible only because it is
    recorded.

    `metrics.ari.contingency` (i32, `nClasses^2`) is the matrix THIS call
    consumed, at ARI's own label range. Without it the card carried one
    contingency matrix at the driver's range and silently assumed ARI used
    the same one.

    The `size < 2` arm (:119-122) returns RAFT's 1.0 with NO stage
    recorded. UNREACHABLE from the card driver, whose fixture has 2053
    rows; a run that took it would record three fewer stages and the differ
    would report a STRUCTURAL divergence, which is correct and readable."""
    if size < 2:
        return 1.0  # (:119-122)
    var u1 = count_unique(ctx, first_cluster_array, size)
    var u2 = count_unique(ctx, second_cluster_array, size)
    var n_uniq_first = u1[0]
    var n_uniq_second = u2[0]
    var lower_label_range = u1[1] if u1[1] < u2[1] else u2[1]
    var upper_label_range = u1[2] if u1[2] > u2[2] else u2[2]
    var n_classes = Int(upper_label_range - lower_label_range + 1)
    var uniq = List[Int32]()
    uniq.append(Int32(n_uniq_first))
    uniq.append(Int32(n_uniq_second))
    uniq.append(lower_label_range)
    uniq.append(upper_label_range)
    trace.record_host("metrics.ari.uniq", uniq.unsafe_ptr(), 4)
    _ = uniq^
    if n_uniq_first == n_uniq_second:
        if n_uniq_first == 1 or n_uniq_first == size:
            return 1.0  # (:130-132)
    var c = contingency_matrix_host(
        ctx,
        first_cluster_array,
        second_cluster_array,
        size,
        lower_label_range,
        upper_label_range,
    )
    trace.record_list_i32("metrics.ari.contingency", c)
    return adjusted_rand_index_from_contingency_traced(
        trace, c, n_classes, size
    )
