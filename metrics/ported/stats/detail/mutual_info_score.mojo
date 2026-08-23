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
) -> Float64:
    """`mutual_info_kernel` (:46-81) then `h_MI / size` (:161), on the host
    in ascending (i, j) order (DEVIATIONS 650, 651)."""
    var a = row_sums(c, k)
    var b = col_sums(c, k)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var acc = Float32(0.0)
        var fsize = Float32(size)
        for i in range(k):
            for j in range(k):
                var cij = c[i * k + j]
                var ab = a[i] * b[j]
                if ab != Int64(0) and cij != Int32(0):
                    var fc = Float32(cij)
                    var l1 = ftz(identical_log(ftz(fsize * fc)))
                    var l2 = ftz(identical_log(ftz(Float32(ab))))
                    var diff = ftz(l1 - l2)
                    acc = ftz(identical_mul_add(fc, diff, acc))
        return Float64(ftz(acc / fsize))
    else:
        var acc = Float64(0.0)
        var dsize = Float64(size)
        for i in range(k):
            for j in range(k):
                var cij = c[i * k + j]
                var ab = a[i] * b[j]
                if ab != Int64(0) and cij != Int32(0):
                    var dc = Float64(cij)
                    acc += dc * (log(dsize * dc) - log(Float64(ab)))
        return acc / dsize


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
    return mutual_info_from_contingency(c, k, size)
