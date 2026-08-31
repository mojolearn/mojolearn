# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HDBSCAN driver: one fit, one identity card.

    tools/with_build_lock.sh     pixi run mojo run -I . hdbscan/hdbscan_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . hdbscan/hdbscan_main.mojo

    MOJOLEARN_IDENTITY_TRACE=/tmp/hdbscan.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . hdbscan/hdbscan_main.mojo
    python3 tools/identity_trace_diff.py /tmp/hdbscan.apple.card /tmp/hdbscan.other.card

Runs `hdbscan/ported/hdbscan/runner.mojo::fit_hdbscan` (cuML's
`_fit_hdbscan`, dense mutual reachability by DEVIATION 1600,
L2SqrtExpanded, Excess of Mass) on the `blobs96` fixture and records every
stage through `core/identity_trace.mojo`. THE FIT ITSELF EMITS THE CARD:
no stage is recomputed beside the fit to produce it, so a card and an
answer cannot disagree about which run they describe.

THE STAGE LIST, IN THE ORDER A DIFFER SEES IT. The order is the pipeline's,
so two vendors diff at the FIRST stage that moved and everything after it
is downstream of that one:

    knn.index_norm          |  the k-NN's own stages, written by
    knn.query_norm          |  `neighbors/estimator.mojo` into THIS trace
    knn.out_dist            |  (one seq per file; the DEVIATION 518
    knn.out_idx             |  lesson). `sorted_*` is what the core
    knn.sorted_dist         |  distance is read from; `out_*` is pre-sort
    knn.sorted_idx          |  and localizes WHICH ARM produced a change
    hdbscan.x               the input bytes
    hdbscan.core_dists      the k-th order statistic per row
    hdbscan.mr.dists        the m x m mutual reachability matrix
    hdbscan.mst.rounds      Boruvka round count (INTEGER; a card that
                            first differs here disagreed about how much
                            work to do, not about a number)
    hdbscan.mst.edges       the sorted MST as (src, dst) pairs, Int32
    hdbscan.mst.weights     the sorted MST weights, Float32 bytes
    hdbscan.dendrogram.children / .deltas / .sizes
    hdbscan.condensed.parents / .children / .lambdas / .sizes
    hdbscan.stabilities     per condensed cluster, AFTER Excess of Mass
    hdbscan.selected        the selection, Int32 0/1
    hdbscan.raw_labels      condensed ids, before the remap
    hdbscan.labels          the final labels, -1 for noise
    hdbscan.stability_scores

WHY `hdbscan.mr.dists` IS THE "SORTED MUTUAL REACHABILITY EDGES". For the
DENSE graph the COO is `(i, j)` for every ordered pair, and cell `i*m + j`
of the matrix IS entry `i*m + j` of that COO -- already in ascending
`(row, col)` order, which is the order their `sorted_coo_to_csr` requires
and the order a sort would produce. There is no sort to record because
there is nothing to sort. The SPARSE arm's symmetrize-and-sort is
DEVIATION 1600's refused half.

Every line prints the mode this binary COMPILED in. No timing is measured
or printed here.
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from hdbscan.mojo_only.hdbscan_fixture import (
    HFIX_BLOBS,
    build_hfixture,
    hfixture_as_list,
    hfixture_d,
    hfixture_min_cluster_size,
    hfixture_min_samples,
    hfixture_n,
    hfixture_name,
)
from hdbscan.ported.hdbscan.runner import (
    GRAPH_BUILD_BRUTE_FORCE_KNN,
    HDBSCANParams,
    fit_hdbscan,
)
from hdbscan.ported.hdbscan.detail.select import CLUSTER_SELECTION_EOM
from hierarchy.ported.cluster.detail.connectivities import (
    DISTANCE_L2_SQRT_EXPANDED,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def main() raises:
    var ctx = DeviceContext()
    var trace = IdentityTrace()
    var fix = HFIX_BLOBS
    var m = hfixture_n(fix)
    var d = hfixture_d(fix)

    var hx = build_hfixture(ctx, fix)
    var x = ctx.enqueue_create_buffer[DType.float32](m * d)
    ctx.synchronize()
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()
    var x_host = hfixture_as_list(fix)

    var params = HDBSCANParams(
        hfixture_min_samples(fix),
        hfixture_min_cluster_size(fix),
        0,
        Float32(0.0),
        False,
        Float32(1.0),
        CLUSTER_SELECTION_EOM,
        GRAPH_BUILD_BRUTE_FORCE_KNN,
    )
    var out = fit_hdbscan(
        ctx, trace, x_host, x, m, d, DISTANCE_L2_SQRT_EXPANDED, params
    )

    print(
        "hdbscan_main mode=" + _mode_name() + " fixture=" + hfixture_name(fix)
        + " n_rows=" + String(m) + " n_cols=" + String(d)
        + " min_samples=" + String(params.min_samples)
        + " min_cluster_size=" + String(params.min_cluster_size)
        + " metric=L2SqrtExpanded graph=DENSE_MUTUAL_REACHABILITY"
        + " selection=EOM"
    )
    print(
        "n_clusters=" + String(out.n_clusters)
        + " n_outliers=" + String(out.n_outliers)
        + " n_condensed_clusters=" + String(out.condensed.n_clusters)
        + " n_condensed_edges=" + String(out.condensed.n_edges)
        + " boruvka_rounds=" + String(out.n_boruvka_rounds)
    )
    var counts = List[Int]()
    for _ in range(out.n_clusters):
        counts.append(0)
    for i in range(m):
        var l = Int(out.labels[i])
        if l >= 0 and l < out.n_clusters:
            counts[l] += 1
    var sizes = String("cluster sizes:")
    for kk in range(out.n_clusters):
        sizes += " " + String(counts[kk])
    sizes += "  noise: " + String(out.n_outliers)
    print(sizes)

    if trace.enabled:
        print("identity card written to " + trace.path)
    else:
        print("set MOJOLEARN_IDENTITY_TRACE=<path> to write the identity card")
    _ = hx^
    _ = x^
    _ = x_host^
