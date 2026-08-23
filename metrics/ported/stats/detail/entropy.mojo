"""RAFT `cpp/include/raft/stats/detail/entropy.cuh` (ebf9268).

THEIRS (:105-143):

    if (!size) return 1.0;
    numUniqueClasses = upper - lower + 1
    countLabels(clusterArray, prob, size, lower, upper)      CUB HistogramEven -> double counts
    divideScalar(prob, prob, (double)size)                   p_i = count_i / size
    mapThenSumReduce(d_entropy, entropyOp)                   sum over i of (p ? -p*log(p) : 0)
    return h_entropy

sklearn (`sklearn/metrics/cluster/_supervised.py::entropy`): `pi = counts /
sum(counts); -sum(pi * (log(pi) - log(pi_sum)))` with `pi_sum = sum(counts)`
-- the same quantity, natural log, with their `return 1.0` for an empty
label array too. NOTE sklearn's `log(pi) - log(pi_sum)` spelling where
RAFT divides first; both are `-sum p log p` and differ in the last bits.
Ours mirrors RAFT's spelling.

=========================================================================
DEVIATION 650 (metrics lane, 2026-08-23): THE LABEL METRICS' FLOAT
EPILOGUE RUNS ON THE HOST, IN A FIXED SERIAL ORDER, FROM THE INTEGER
COUNTS THE DEVICE PRODUCED.
=========================================================================
THEIRS: the counts are written as `double` by CUB, divided on the device,
and `mapThenSumReduce` folds `-p log p` per block and lands the block
partials in one double through `atomicAdd` (`raft/linalg/detail/
map_then_reduce.cuh:33-38`) -- an ARRIVAL order, different run to run.
OURS: the device produces the INTEGER histogram (`histogram.mojo`, exact
and order-free), the host reads `numUniqueClasses` ints back (instead of
one double) and performs the float ops serially, ascending over the
classes. The same split for `mutual_info_score` (its contingency matrix
comes back, `numUniqueClasses^2` ints). Why the host: Apple's GPU has no
Float64 (`mojolearn-hardware-limits`), so "their double on the device"
is not available on one of the three columns, and a serial fold on one
host thread is the one place the order is trivially the same everywhere.
The cost is a readback of `k` or `k^2` ints where theirs reads back one
scalar; the metrics are O(n) on the device either way.

=========================================================================
DEVIATION 651 (metrics lane, 2026-08-23): UNDER IDENTICAL THE LOG-BEARING
EPILOGUES ARE FLOAT32 THROUGH `identical_log`; UNDER FAST THEY ARE
FLOAT64 THROUGH THE HOST'S `log`, AS THEIRS ARE.
=========================================================================
A host `log` on Float64 is the host's libm -- macOS Accelerate and glibc
disagree in the last bit (the `portable_exp64` precedent, numerics.mojo),
and `mojo_only/numerics.mojo` has no portable Float64 log. IDENTICAL
therefore computes the per-class term with `identical_log` on Float32
(row 12: ONE arithmetic everywhere), accumulates through
`identical_mul_add` (row 9, so no codegen can contract `acc + (-p)*lp`
differently) with `ftz` at every stored seam (row 10), and widens the
Float32 total to the Float64 the signature returns. FAST is RAFT's
precision: Float64 with `std.math.log`, serial ascending. CONSEQUENCE:
IDENTICAL's value differs from FAST's in roughly the 7th significant
digit, by design, and is the same bits on every vendor AND every host; the
checks gate IDENTICAL bitwise against a host oracle written through the
same helpers and both modes against a Float64 reference to 1e-6
relative. The card (`metrics_main.mojo`) records the Float64 bits.
"""

from std.math import log
from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.ported.stats.detail.histogram import histogram
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_log,
    identical_mul_add,
)


def count_labels(
    ctx: DeviceContext,
    mut labels: DeviceBuffer[DType.int32],
    n_rows: Int,
    lower_label_range: Int32,
    upper_label_range: Int32,
) raises -> List[Int32]:
    """`countLabels` (:57-92): `cub::DeviceHistogram::HistogramEven` with
    unit bins over `[lower, upper]`, returned to the host as the integers
    CUB produced (DEVIATION 650)."""
    var n_unique = Int(upper_label_range - lower_label_range + 1)
    var bins = ctx.enqueue_create_buffer[DType.int32](n_unique)
    histogram(ctx, bins, n_unique, labels, n_rows, lower_label_range)
    var h = ctx.enqueue_create_host_buffer[DType.int32](n_unique)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=bins)
    ctx.synchronize()
    var out = List[Int32]()
    for i in range(n_unique):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = bins^
    return out^


def entropy_from_counts(counts: List[Int32], size: Int) -> Float64:
    """`divideScalar` then `mapThenSumReduce(entropyOp)` (:129-135), on the
    host (DEVIATIONS 650, 651). `entropyOp(p) = p ? -1 * p * log(p) : 0`
    (:35-43)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var acc = Float32(0.0)
        var fsize = Float32(size)
        for i in range(len(counts)):
            var p = ftz(Float32(counts[i]) / fsize)
            if p != Float32(0.0):
                var lp = ftz(identical_log(p))
                acc = ftz(identical_mul_add(-p, lp, acc))
        return Float64(acc)
    else:
        var acc = Float64(0.0)
        var dsize = Float64(size)
        for i in range(len(counts)):
            var p = Float64(counts[i]) / dsize
            if p != 0.0:
                acc += -1.0 * p * log(p)
        return acc


def entropy(
    ctx: DeviceContext,
    mut cluster_array: DeviceBuffer[DType.int32],
    size: Int,
    lower_label_range: Int32,
    upper_label_range: Int32,
) raises -> Float64:
    """`entropy(clusterArray, size, lower, upper, stream)` (:105-143)."""
    if size == 0:
        return 1.0  # `if (!size) return 1.0;` (:112); sklearn agrees
    var counts = count_labels(
        ctx, cluster_array, size, lower_label_range, upper_label_range
    )
    return entropy_from_counts(counts, size)
