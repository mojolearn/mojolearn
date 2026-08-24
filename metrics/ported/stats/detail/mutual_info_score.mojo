"""RAFT `cpp/include/raft/stats/detail/mutual_info_score.cuh` (ebf9268).

THEIRS (:93-162):

    numUniqueClasses = upper - lower + 1
    contingencyMatrix(first, second, size, C, lower, upper)       int C[k][k]
    a = reduce<rowMajor=true, alongRows=true>(C)                  row sums   (int)
    b = reduce<rowMajor=true, alongRows=false>(C)                 column sums (int)
    mutual_info_kernel (:46-81), one thread per (i, j):
        if a[i]*b[j] != 0 && C[i][j] != 0:
            localMI += C[i][j] * (log(size * C[i][j]) - log(a[i] * b[j]))
        BlockReduce.Sum; atomicAdd(d_MI)                          double, arrival order
    return h_MI / size

sklearn (`mutual_info_score`): `contingency_nm * (log(contingency_nm) -
log(pi.take(nzx)) - log(pj.take(nzy))) + contingency_nm * log(total)`
summed, all over `total`; i.e. sum c/n * (log n + log c - log a - log b).
The same quantity; the spelling differs in how the logs are grouped.
Ours mirrors RAFT's grouping: `log(size * c) - log(a * b)`.

THE FLOAT EPILOGUE is DEVIATIONS 650 and 651 (entropy.mojo carries the
banners): the integer contingency matrix is the device product and is
exact; `a`, `b` and the double sum are done on the host, serially,
ascending over (i, j); IDENTICAL in Float32 through `identical_log` /
`identical_mul_add` / `ftz`, FAST in Float64 through the host `log`.

ONE INTEGER HAZARD OF THEIRS IS NOT PORTED: `a[i] * b[j]` (:61, :65) is an
`int` times an `int` -- it overflows once one class has more than 46,340
samples in both labelings (`a_i * b_j > 2^31`), which is a dataset of
fifty thousand rows. Ours forms the product in Int64 and converts. Below
the overflow the two are the same number; above it theirs is wrong.
Named here so it is not mistaken for a re-design (PORTING_RULES 0c: fix
their BUGS, numbered). It is part of DEVIATION 650's block.
"""

from std.math import log
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from metrics.ported.stats.detail.contingency_matrix import contingency_matrix
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_log,
    identical_mul_add,
)


def contingency_matrix_host(
    ctx: DeviceContext,
    mut first_cluster_array: DeviceBuffer[DType.int32],
    mut second_cluster_array: DeviceBuffer[DType.int32],
    size: Int,
    lower_label_range: Int32,
    upper_label_range: Int32,
) raises -> List[Int32]:
    """The device contingency matrix, read back as `k*k` row-major ints
    (DEVIATION 650)."""
    var k = Int(upper_label_range - lower_label_range + 1)
    var c = ctx.enqueue_create_buffer[DType.int32](k * k)
    contingency_matrix(
        ctx,
        first_cluster_array,
        second_cluster_array,
        size,
        c,
        lower_label_range,
        upper_label_range,
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](k * k)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=c)
    ctx.synchronize()
    var out = List[Int32]()
    for i in range(k * k):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = c^
    return out^


def row_sums(c: List[Int32], k: Int) -> List[Int64]:
    """`raft::linalg::reduce<true, true>` (:139-140): `a[i] = sum_j C[i][j]`.
    Integers, widened."""
    var a = List[Int64]()
    for i in range(k):
        var s = Int64(0)
        for j in range(k):
            s += Int64(c[i * k + j])
        a.append(s)
    return a^


def col_sums(c: List[Int32], k: Int) -> List[Int64]:
    """`raft::linalg::reduce<true, false>` (:143-144): `b[j] = sum_i C[i][j]`."""
    var b = List[Int64]()
    for j in range(k):
        var s = Int64(0)
        for i in range(k):
            s += Int64(c[i * k + j])
        b.append(s)
    return b^


def mutual_info_from_contingency(
    c: List[Int32], k: Int, size: Int
) raises -> Float64:
    """`mutual_info_kernel` (:46-81) then `h_MI / size` (:161), on the host
    in ascending (i, j) order (DEVIATIONS 650, 651). The untraced entry;
    `mutual_info_from_contingency_traced` is the implementation."""
    var off = IdentityTrace.disabled()
    return mutual_info_from_contingency_traced(off, c, k, size, String(""))


def mutual_info_from_contingency_traced(
    mut trace: IdentityTrace,
    c: List[Int32],
    k: Int,
    size: Int,
    tag_prefix: String,
) raises -> Float64:
    """The same epilogue, recording four stages:

        <tag_prefix>.row_sums    i64, k     `a[i] = sum_j C[i][j]`
        <tag_prefix>.col_sums    i64, k     `b[j] = sum_i C[i][j]`
        <tag_prefix>.terms       f64, k*k   `log(size*c) - log(a*b)` per cell
        <tag_prefix>.acc         f64, k*k   the running total after each cell

    WHY. MI was one recorded scalar folded from `k*k` log terms on the
    HOST, so a divergence anywhere in it -- a row sum, a log, one cell --
    arrived as "the MI bits differ" with no address. `a` and `b` are
    INTEGER and exact, so a difference there is a real defect and not a
    rounding, and separating them from the float epilogue is the first
    question to ask. `.terms` is the per-cell log difference BEFORE the
    fused accumulate consumes it (`identical_mul_add(fc, diff, acc)` never
    materializes the product, so `diff` is the last value the cell owns
    alone). `.acc` localizes an absorbed term to the cell after which the
    total stopped moving.

    Both float stages are Float64 in BOTH modes: IDENTICAL's arithmetic is
    Float32 and Float32 -> Float64 is EXACT AND INJECTIVE, so the widening
    loses no bit the hash could have seen, and one dtype per stage keeps an
    IDENTICAL card and a FAST card aligned stage for stage.

    A GUARDED-OUT CELL RECORDS `+0.0` IN `.terms` AND ITS UNCHANGED `.acc`.
    The guard `ab != 0 and cij != 0` is not given a stage of its own
    because it is a deterministic INTEGER function of stages the card
    already carries (`.row_sums`, `.col_sums`, and the contingency matrix
    recorded by `mutual_info_score_traced`): if those three agree the guard
    agrees. A decision derivable from recorded integers needs no stage; a
    decision derivable only from unrecorded floats does.

    The lists are built only when the trace is enabled, because
    `bench/lanes_price_main.mojo` reaches this function through
    `homogeneity_score` inside a TIMED window."""
    var a = row_sums(c, k)
    var b = col_sums(c, k)
    if trace.enabled:
        trace.record_host(tag_prefix + ".row_sums", a.unsafe_ptr(), k)
        trace.record_host(tag_prefix + ".col_sums", b.unsafe_ptr(), k)
    var terms = List[Float64]()
    var accs = List[Float64]()
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var acc = Float32(0.0)
        var fsize = Float32(size)
        for i in range(k):
            for j in range(k):
                var cij = c[i * k + j]
                var ab = a[i] * b[j]
                var seen = Float32(0.0)
                if ab != Int64(0) and cij != Int32(0):
                    var fc = Float32(cij)
                    var l1 = ftz(identical_log(ftz(fsize * fc)))
                    var l2 = ftz(identical_log(ftz(Float32(ab))))
                    var diff = ftz(l1 - l2)
                    seen = diff
                    acc = ftz(identical_mul_add(fc, diff, acc))
                if trace.enabled:
                    terms.append(Float64(seen))
                    accs.append(Float64(acc))
        _record_mi_trail(trace, tag_prefix, terms, accs)
        _ = terms^
        _ = accs^
        return Float64(ftz(acc / fsize))
    else:
        var acc = Float64(0.0)
        var dsize = Float64(size)
        for i in range(k):
            for j in range(k):
                var cij = c[i * k + j]
                var ab = a[i] * b[j]
                var seen = Float64(0.0)
                if ab != Int64(0) and cij != Int32(0):
                    var dc = Float64(cij)
                    var diff = log(dsize * dc) - log(Float64(ab))
                    seen = diff
                    acc += dc * diff
                if trace.enabled:
                    terms.append(seen)
                    accs.append(acc)
        _record_mi_trail(trace, tag_prefix, terms, accs)
        _ = terms^
        _ = accs^
        return acc / dsize


def _record_mi_trail(
    mut trace: IdentityTrace,
    tag_prefix: String,
    mut terms: List[Float64],
    mut accs: List[Float64],
) raises:
    """The two float trails, written once so the two mode arms cannot
    drift in what they record."""
    if not trace.enabled:
        return
    trace.record_host(tag_prefix + ".terms", terms.unsafe_ptr(), len(terms))
    trace.record_host(tag_prefix + ".acc", accs.unsafe_ptr(), len(accs))


def mutual_info_score(
    ctx: DeviceContext,
    mut first_cluster_array: DeviceBuffer[DType.int32],
    mut second_cluster_array: DeviceBuffer[DType.int32],
    size: Int,
    lower_label_range: Int32,
    upper_label_range: Int32,
) raises -> Float64:
    """`mutual_info_score(first, second, size, lower, upper, stream)`
    (:93-162)."""
    var off = IdentityTrace.disabled()
    return mutual_info_score_traced(
        ctx,
        off,
        first_cluster_array,
        second_cluster_array,
        size,
        lower_label_range,
        upper_label_range,
        String(""),
    )


def mutual_info_score_traced(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut first_cluster_array: DeviceBuffer[DType.int32],
    mut second_cluster_array: DeviceBuffer[DType.int32],
    size: Int,
    lower_label_range: Int32,
    upper_label_range: Int32,
    tag_prefix: String,
) raises -> Float64:
    """`mutual_info_score` carrying a card: `<tag_prefix>.contingency` plus
    the four stages of `mutual_info_from_contingency_traced`.

    THE MATRIX RECORDED HERE IS THE ONE THIS CALL CONSUMED, not a second
    one built beside it. The card driver also records `metrics.contingency`
    from its own `contingency_matrix_host` call, and those two stages
    agreeing is a genuine internal-consistency check of a device kernel run
    twice; the hazard CARD_GAPS.md names against hierarchy/linkage is a
    card that records a RE-RUN and calls it the buffer the metric used,
    which is what this signature exists to avoid.

    `tag_prefix` is the caller's because MI is computed over BOTH argument
    orders in one card: `MI(y_true, y_pred)` and `MI(y_pred, y_true)`, the
    numerators of homogeneity and of completeness. They are the same
    quantity mathematically and NOT necessarily the same bits -- the second
    folds the transposed matrix, so the host's serial ascending order visits
    the cells in a different sequence -- which is precisely why both are on
    the card."""
    # `h_MI / size` (:161) with size 0 is 0 / 0 in theirs; REFUSED by name
    # (a NaN must not reach the recorded scalar, IDENTITY_PATHS row 39).
    # `homogeneity_score` / `entropy` return 1.0 for size 0 before reaching
    # here, as theirs do.
    if size <= 0:
        raise Error(
            "mutual_info_score: size must be positive, got "
            + String(size)
            + " (0 / 0 is refused by name)"
        )
    var k = Int(upper_label_range - lower_label_range + 1)
    var c = contingency_matrix_host(
        ctx,
        first_cluster_array,
        second_cluster_array,
        size,
        lower_label_range,
        upper_label_range,
    )
    trace.record_list_i32(tag_prefix + ".contingency", c)
    return mutual_info_from_contingency_traced(trace, c, k, size, tag_prefix)
