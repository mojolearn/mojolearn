"""Host-pointer surface for the hdbscan section.

The shape is `hierarchy/estimator.mojo`'s and `dbscan/estimator.mojo`'s:
host pointers in, device buffers owned here for exactly one call, results
read back, nothing retained. The ported entry is
`hdbscan/ported/hdbscan/runner.mojo::fit_hdbscan` (cuML `runner.h:152-234`)
and this file re-decides none of it.

**THIS IS NOT WIRED TO `bindings/` AND THERE IS NO PYTHON CLASS YET.**
Nothing under `bindings/` or `python/` calls it, which makes it an
`UNWIRED.md` row until the orchestrator wires it; the lane README's
HAND-OFF section says what a `mojolearn.HDBSCAN` would have to map.

THIS PATH DOES EMIT AN IDENTITY CARD, unlike `hierarchy`'s. `fit_hdbscan`
takes a live `IdentityTrace` and records every stage itself, so a fit
through this surface writes the SAME card `hdbscan_main.mojo` writes, from
the same call, with no re-run of any stage. That closes for this lane the
gap `hierarchy/estimator.mojo`'s header records as owed for its own.

X IS ROW-MAJOR. cuML's `HDBSCAN.fit` takes `order='C'` and every distance
step here reads the dense design row by row.

WHAT IS REFUSED, AND WHERE
  metric != L2SqrtExpanded (1)          `fit_hdbscan`, their RAFT_EXPECTS
  build_algo != BRUTE_FORCE_KNN         `fit_hdbscan` (NN_DESCENT, rung 2)
  cluster_selection_method not in {0,1} `fit_hdbscan`
  cluster_selection_epsilon != 0.0      `select_clusters` (rung 2)
  min_samples < 1, min_samples > n_rows `runner.mojo`
  min_cluster_size < 2 or > n_rows      `build_condensed_hierarchy`
  alpha <= 0 or non-finite              `build_mr_linkage`
  n_rows < 2, n_rows > 46340            `build_mr_linkage`
  a NaN or infinite anywhere            DEVIATION 1607
  `probabilities_`                      `hdbscan_probabilities_host` below
  the SPARSE mutual reachability graph  `reachability.mojo` (DEVIATION 1600)
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from hdbscan.ported.hdbscan.runner import (
    GRAPH_BUILD_BRUTE_FORCE_KNN,
    HDBSCANParams,
    fit_hdbscan,
)
from hdbscan.ported.hdbscan.detail.select import CLUSTER_SELECTION_EOM
from hierarchy.ported.cluster.detail.connectivities import (
    DISTANCE_L2_SQRT_EXPANDED,
)


def hdbscan_fit_host(
    ctx: DeviceContext,
    x_ptr: MutPointer[Float32, MutUntrackedOrigin],
    labels_ptr: MutPointer[Int32, MutUntrackedOrigin],
    core_dists_ptr: MutPointer[Float32, MutUntrackedOrigin],
    info_ptr: MutPointer[Int32, MutUntrackedOrigin],
    n_rows: Int,
    n_cols: Int,
    min_samples: Int = 5,
    min_cluster_size: Int = 5,
    max_cluster_size: Int = 0,
    alpha: Float32 = Float32(1.0),
    allow_single_cluster: Bool = False,
    cluster_selection_method: Int = CLUSTER_SELECTION_EOM,
    cluster_selection_epsilon: Float32 = Float32(0.0),
    metric: Int = DISTANCE_L2_SQRT_EXPANDED,
) raises -> Int:
    """Cluster host-resident ROW-MAJOR data. Returns the cluster count.

    `x_ptr` is `n_rows x n_cols` Float32, row-major.
    `labels_ptr` receives `n_rows` Int32 in `0 .. n_clusters-1`, or `-1`
    for NOISE. The numbering is cuML's: ascending condensed cluster id
    through `label_map` (`runner.h:226-233`), which is the same rule
    scikit-learn-contrib's `hdbscan` uses and is NOT an arbitrary choice
    this surface makes.
    `core_dists_ptr` receives `n_rows` Float32, the distance to the
    `min_samples`-th nearest neighbour EXCLUDING self (their
    `min_samples + 1` adjustment, `runner.h:68-80`).
    `info_ptr` receives FOUR Int32:
        [0] n_clusters                (the return value, read back)
        [1] n_outliers                (points labelled -1)
        [2] Boruvka round count       (an integer card stage)
        [3] n_condensed_clusters      (the condensed tree's cluster count,
                                       which is the length `stabilities`
                                       and `is_cluster` would have)
    """
    if n_rows < 2:
        raise Error(
            "hdbscan_fit_host needs n_rows >= 2, got " + String(n_rows)
        )
    if n_cols < 1:
        raise Error(
            "hdbscan_fit_host needs n_cols >= 1, got " + String(n_cols)
        )

    var x_host = List[Float32](capacity=n_rows * n_cols)
    for i in range(n_rows * n_cols):
        x_host.append(x_ptr.unsafe_load(i))

    var x = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)
    ctx.synchronize()

    var params = HDBSCANParams(
        min_samples,
        min_cluster_size,
        max_cluster_size,
        cluster_selection_epsilon,
        allow_single_cluster,
        alpha,
        cluster_selection_method,
        GRAPH_BUILD_BRUTE_FORCE_KNN,
    )
    var trace = IdentityTrace()
    var out = fit_hdbscan(
        ctx, trace, x_host, x, n_rows, n_cols, metric, params
    )

    for i in range(n_rows):
        labels_ptr.unsafe_store(i, out.labels[i])
        core_dists_ptr.unsafe_store(i, out.core_dists[i])
    info_ptr.unsafe_store(0, Int32(out.n_clusters))
    info_ptr.unsafe_store(1, Int32(out.n_outliers))
    info_ptr.unsafe_store(2, Int32(out.n_boruvka_rounds))
    info_ptr.unsafe_store(3, Int32(out.condensed.n_clusters))

    # [[mojo-buffer-freed-at-last-use]]: every buffer outlives the queue.
    _ = x^
    _ = x_host^
    return out.n_clusters


def hdbscan_probabilities_host(n_rows: Int) raises:
    """`Membership::get_probabilities` (`membership.cuh:39-98`) and
    `probabilities_` at the Python surface. NOT PORTED; raises by name.

    DEVIATION 1610. It is a CUB segmented MAX over the same condensed-tree
    CSR `compute_stabilities` already builds (`deaths[c]`), followed by
    `min(child_lambda, cluster_death) / cluster_death` per point
    (`kernels/membership.cuh:44-52`). The reason it is deferred is scope,
    not difficulty: the segmented max is DEVIATION 1604's fold with the
    comparison reversed, and closing it is a day's work inside this lane.
    Returning zeros or ones instead would be a NUMBER NOBODY COMPUTED
    sitting in a field a caller will plot.
    """
    raise Error(
        "hdbscan.probabilities: NOT PORTED (DEVIATION 1610), refused by"
        " name for " + String(n_rows) + " points. Their"
        " Membership::get_probabilities (membership.cuh:39-98) is a CUB"
        " segmented Max over the condensed tree's parent CSR plus a"
        " per-point ratio. To close this refusal, add a `deaths` fold"
        " beside DEVIATION 1604's `births` fold in"
        " hdbscan/ported/hdbscan/detail/stabilities.mojo (same segments,"
        " reversed comparison) and a per-edge epilogue"
    )
