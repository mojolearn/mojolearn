"""The per-pair distance ops cuML's `pairwise_distances` reaches for KDE.

PORT OF cuVS `cpp/src/distance/detail/distance_ops/{l2_unexp,l1,l_inf}.cuh`
at cuVS `94c2819` (the `core()` and `epilog()` of each op), plus the
one-thread-per-cell loop that stands in for their `pairwise_matrix` tile.
Partial. Do not improve.

WHICH METRICS, AND WHY THESE
-----------------------------
cuML's `KernelDensity.score_samples` (`python/cuml/cuml/neighbors/
kernel_density.py:332-340`, cuML `00094f7`) calls
`cuml.metrics.pairwise_distances(X_query, X_train, metric=self.metric)`,
whose dense table (`metrics/pairwise_distances.pyx:68-86`) sends

    "euclidean" / "l2"                  -> DistanceType.L2SqrtUnexpanded
    "sqeuclidean"                       -> DistanceType.L2Expanded
    "l1" / "cityblock" / "manhattan"    -> DistanceType.L1
    "chebyshev"                         -> DistanceType.Linf

and cuVS `distance-inl.cuh:286-302` dispatches each of those to the op of
the same name. **Their default `euclidean` is UNEXPANDED** -- `sum_k
(x_k - y_k)^2` straight from the coordinates, then `sqrt` -- NOT the
`||x||^2 + ||y||^2 - 2 x.y` identity the k-NN tile uses, so the k-NN lane's
`pinned_distance_tile_kernel` is the wrong arithmetic for it and is called
here only for `sqeuclidean`, which IS their expanded op (`distance.cuh`,
the `L2Expanded` tag). The four ops:

    l2_unexp.cuh:62-63   core: diff = x - y; acc += diff * diff
    l2_unexp.cuh:74-80   epilog: if (sqrt) acc = raft::sqrt(acc)
    l1.cuh:32            core: acc += raft::abs(x - y)
    l_inf.cuh:33-34      core: diff = raft::abs(x - y); acc = raft::max(acc, diff)

Every other metric in their table (cosine, canberra, minkowski, hellinger,
correlation, jensenshannon, hamming, kldivergence, russellrao,
nan_euclidean) is REFUSED BY NAME at `kde/ported/neighbors/
kernel_density.mojo::metric_from_name`; see `kde/UNPORTED.tsv`.

WHAT IS NOT PORTED: THEIR TILE
------------------------------
`pairwise_matrix_cuda` is a `Policy4x4` Contractions kernel (Kblk 32, a 4x4
register tile per thread, 256 threads, a 64x64 output tile) -- the same
policy `dbscan/ported/neighbors/epsilon_neighborhood.mojo` transcribes.
This file walks the feature axis in ONE thread per output cell, ascending,
which is the discipline `neighbors/mojo_only/pinned_distance_tile.mojo`
adopted for the k-NN identical arm and for the same reason: the summation
order is then a pure function of `n_features` and nothing else. The tile is
a later port; its only effect would be speed, and this lane does not
measure speed. Inside one thread the arithmetic IS their `core()` op in
their k order.

IDENTITY (the rows this kernel answers to)
------------------------------------------
- row 9: `acc += diff * diff` is `identical_mul_add(diff, diff, acc)`.
  L1 and Linf have no multiply-add; their adds and max are plain.
- row 10: every stored intermediate through `ftz`; the `sqrt` epilog is
  `identical_sqrt` (the stdlib sqrt is approximate on NVIDIA, DEVIATION
  258; Apple's native sqrt is correctly rounded so no Apple bit moves).
- Linf is a SELECTION, exactly commutative away from `-0.0`/`+0.0` and NaN.
  `abs()` makes every candidate non-negative with a clear sign bit, so no
  negative zero is ever compared, and finite inputs produce no NaN.
- Under FAST the helpers compile away and this is the plain loop: the
  stdlib's device sqrt, the codegen's contraction choice.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast

from mojo_only.numerics import ftz, identical_mul_add, identical_sqrt

#: Their `DistanceType` values as this lane spells them. The metric string
#: table lives in `kde/ported/neighbors/kernel_density.mojo`.
comptime DIST_L2_SQRT_UNEXPANDED = 0
comptime DIST_L2_EXPANDED = 1
comptime DIST_L1 = 2
comptime DIST_LINF = 3

#: SCHEDULING: one thread owns one cell; the block width moves no bit.
comptime PAIRWISE_ELEM_TPB = 256


def l2_unexp_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """`l2_unexp.cuh:62-63`. `diff = x - y; acc += diff * diff`."""
    var diff = ftz(x - y)
    return ftz(identical_mul_add(diff, diff, acc))


def l1_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """`l1.cuh:32`. `acc += abs(x - y)`."""
    return ftz(acc + abs(ftz(x - y)))


def linf_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """`l_inf.cuh:33-34`. `diff = abs(x - y); acc = max(acc, diff)`.

    Written as their `max`: the larger survives, and on a tie either one
    (they are the same bits -- both non-negative, so no zero-sign question).

    IDENTITY_PATHS row 39 (2026-08-23), proof that neither zero's sign nor
    a NaN can make this selection vendor-shaped: `acc` is seeded `+0.0` by
    the caller and only ever takes a `diff`; every `diff` is `abs(...)`,
    whose sign bit is clear, so `-0.0` is never a candidate and a `+0.0`
    tie returns identical bits whichever side wins; the compare is a
    strict `>` (first-seen wins), not a hardware `max`; inputs are refused
    non-finite before any launch (DEVIATION 604), and a finite `x - y`
    is finite or +-inf, whose `abs` is +inf, never NaN. The host oracle
    (`kde_oracle.mojo::oracle_distance`) spells the same strict `>`.
    """
    var diff = abs(ftz(x - y))
    if diff > acc:  # row 39: strict `>`, operands non-negative
        return diff
    return acc


def pairwise_unexpanded_kernel(
    dist: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    metric_in: Int32,
):
    """`dist[i][j] = op(x_i, y_j)` for the three unexpanded ops, one thread
    per cell, the feature axis walked ascending in that thread.

    `metric_in` is `DIST_L2_SQRT_UNEXPANDED`, `DIST_L1` or `DIST_LINF`
    (`DIST_L2_EXPANDED` never reaches this kernel; see
    `kde/ported/distance/distance.mojo`). A metric this kernel does not know
    writes NaN so the host's oracle gate cannot mistake it for a result.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= m * n:
        return
    var i = idx // n
    var j = idx % n
    var metric = Int(metric_in)

    var acc = Float32(0.0)
    if metric == DIST_L2_SQRT_UNEXPANDED:
        for f in range(k):
            acc = l2_unexp_core(
                acc,
                ftz(x.unsafe_load(i * k + f)),
                ftz(y.unsafe_load(j * k + f)),
            )
        # epilog, `l2_unexp.cuh:74-80`: the KDE call is `L2SqrtUnexpanded`,
        # so `sqrt` is always true here.
        acc = ftz(identical_sqrt(acc))
    elif metric == DIST_L1:
        for f in range(k):
            acc = l1_core(
                acc,
                ftz(x.unsafe_load(i * k + f)),
                ftz(y.unsafe_load(j * k + f)),
            )
    elif metric == DIST_LINF:
        for f in range(k):
            acc = linf_core(
                acc,
                ftz(x.unsafe_load(i * k + f)),
                ftz(y.unsafe_load(j * k + f)),
            )
    else:
        acc = bitcast[DType.float32](UInt32(0x7FC00000))  # quiet NaN
    dist.unsafe_store(idx, acc)
