# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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
    "cosine"                            -> DistanceType.CosineExpanded
    "minkowski"                         -> DistanceType.LpUnexpanded

(the last two added 2026-09-01; `pairwise_distances.pyx:70` and `:78`) and
cuVS `distance-inl.cuh:268-306` dispatches each of those to the op of
the same name. **Their default `euclidean` is UNEXPANDED** -- `sum_k
(x_k - y_k)^2` straight from the coordinates, then `sqrt` -- NOT the
`||x||^2 + ||y||^2 - 2 x.y` identity the k-NN tile uses, so the k-NN lane's
`pinned_distance_tile_kernel` is the wrong arithmetic for it and is called
here only for `sqeuclidean`, which IS their expanded op (`distance.cuh`,
the `L2Expanded` tag). The three ops THIS FILE'S KERNEL runs:

    l2_unexp.cuh:62-63   core: diff = x - y; acc += diff * diff
    l2_unexp.cuh:74-80   epilog: if (sqrt) acc = raft::sqrt(acc)
    l1.cuh:32            core: acc += raft::abs(x - y)
    l_inf.cuh:33-34      core: diff = raft::abs(x - y); acc = raft::max(acc, diff)

COSINE AND MINKOWSKI, ADDED 2026-09-01, DO NOT COME THROUGH THIS KERNEL.
They are `cosine_distance_op` (`cosine.cuh:43-96`, `use_norms = true`, an
inner product with a `1 - dot/(nx*ny)` epilogue) and `lp_unexp_distance_op`
(`lp_unexp.cuh:30-76`, `acc += pow(|x-y|, p)` then `pow(acc, 1/p)`), and
both live with every other op in `neighbors/impl/distance/detail/
distance_ops.mojo::metric_distance_kernel`. `kde/impl/distance/
distance.mojo` routes them there. Cosine also needs a norm this file never
computes: the TRUE L2 norm, not the squared one (`knn_brute_force.cuh:122`).

The rest of their table (canberra, hellinger, correlation, jensenshannon,
hamming, kldivergence, russellrao, nan_euclidean) is still REFUSED BY NAME
at `kde/impl/neighbors/kernel_density.mojo::metric_from_name`; see
`kde/NOT_IMPLEMENTED.tsv`.

WHAT IS NOT PORTED: THEIR TILE
------------------------------
`pairwise_matrix_cuda` is a `Policy4x4` Contractions kernel (Kblk 32, a 4x4
register tile per thread, 256 threads, a 64x64 output tile) -- the same
policy `dbscan/impl/neighbors/epsilon_neighborhood.mojo` transcribes.
This file walks the feature axis in ONE thread per output cell, ascending,
which is the discipline `neighbors/checks/pinned_distance_tile.mojo`
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

from checks.numerics import ftz, identical_sqrt

# ===========================================================================
# THE OPS AND THE ENUM MOVED, 2026-09-01. This lane no longer spells either.
#
# `l2_unexp_core`, `l1_core` and `linf_core` used to have their bodies HERE,
# and `neighbors/` was about to need the same three beside cosine and
# Minkowski. Two copies of one arithmetic have two chances to drift, which
# is the objection `checks/numerics.mojo::identical_mul` already records
# against its own four copies. The three moved to
# `neighbors/impl/distance/detail/distance_ops.mojo` -- cuVS's own layout,
# where ONE `distance_ops/` directory serves every consumer -- body for body
# and comment for comment. IDENTITY_PATHS rows 9, 10 and 39 still describe
# them; only the file changed.
#
# The four `DIST_*` values used to be a LOCAL INVENTION here (0, 1, 2, 3).
# They are now cuVS's actual enumerators (`cuvs/distance/distance.h:22-69`:
# L2Expanded = 0, CosineExpanded = 2, L1 = 3, L2SqrtUnexpanded = 5,
# Linf = 7, LpUnexpanded = 9). Every use in this lane was BY NAME -- the
# oracle, the nine gates, `metric_from_name`, `metric_name`, the trace
# header, which records the metric's NAME -- so no gate and no identity
# card moves. The names are re-exported below so no import in this lane
# changes either.
# ===========================================================================

from neighbors.impl.distance.detail.distance_ops import (
    DIST_COSINE_EXPANDED,
    DIST_L1,
    DIST_L2_EXPANDED,
    DIST_L2_SQRT_UNEXPANDED,
    DIST_LINF,
    DIST_LP_UNEXPANDED,
    l1_core,
    l2_unexp_core,
    linf_core,
)

#: SCHEDULING: one thread owns one cell; the block width moves no bit.
comptime PAIRWISE_ELEM_TPB = 256


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
    `kde/impl/distance/distance.mojo`). A metric this kernel does not know
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
