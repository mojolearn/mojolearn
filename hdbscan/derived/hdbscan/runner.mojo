# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""cuML's HDBSCAN runner: linkage, condense, extract, score, relabel.

PORT OF `cuml-v26.08.00/cpp/src/hdbscan/runner.h` (cuML `265b9da`):
`build_linkage` (`:54-150`) and `_fit_hdbscan` (`:152-234`).
Transliterated, their order. Do not improve.

WHAT THEIR `build_linkage` DOES THAT OURS DOES NOT, AND WHY. Their
`:66-113` fills a `mutual_reachability_params` and, inside it, an
`all_neighbors_params` carrying `overlap_factor`, `n_clusters` and either
`brute_force_params` or `nn_descent_params`. That whole block is the
26.08 all-neighbors graph builder, which exists to build the mutual
reachability graph on datasets larger than device memory by partitioning
the data into overlapping clusters (`hdbscan.hpp:159-176`). It reaches
`cuvs::neighbors::all_neighbors::build`, which the pinned cuVS checkout
(`94c2819`) does not have at all, and it is the SPARSE graph path
DEVIATION 1600 refuses. `hdbscan/NOT_IMPLEMENTED.tsv` has the row, and
`GRAPH_BUILD_ALGO::NN_DESCENT` is refused BY NAME rather than downgraded
to brute force.

WHAT SURVIVES OF IT, AND IT IS THE PART THAT CHANGES THE ANSWER: their
`min_samples + 1` adjustment and its clamp (`:68-80`),

    // (min_samples+1) is used to account for self-loops in the KNN graph
    // and be consistent with scikit-learn-contrib.
    if (min_samples + 1 > m) { warn; min_samples = m; }
    else                     { min_samples = min_samples + 1; }

which is why the core distance is the distance to the `min_samples`-th
neighbour EXCLUDING self. Kept exactly, warning included.

THE HOST/DEVICE SPLIT, stated once for the whole lane. cuML 26.08 already
put `condense` and the labelling on the host (`condense.cuh:92-212`,
`extract.cuh:88-167`) and keeps stabilities and the selection BFS on the
device. This lane keeps that split: `build_mr_linkage`'s distances, MST
and mutual reachability are device kernels; `compute_stabilities` and
`propagate_cluster_negation_kernel` are device kernels; condense, the
Excess-of-Mass loop (DEVIATION 1605) and the labelling are host. Nothing
of theirs that runs on the device was moved to the host except the two
places a DEVIATION BLOCK names.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from hdbscan.original.hdbscan_sabotage import HDB_SAB_NONE
from hdbscan.original.mutual_reachability_dense import MR_TPB
from hdbscan.derived.cluster.detail.single_linkage import build_mr_linkage
from hdbscan.derived.hdbscan.condensed_hierarchy import CondensedHierarchy
from hdbscan.derived.hdbscan.detail.condense import build_condensed_hierarchy
from hdbscan.derived.hdbscan.detail.extract import ExtractOutput, extract_clusters
from hdbscan.derived.hdbscan.detail.reachability import CORE_TPB
from hdbscan.derived.hdbscan.detail.select import (
    CLUSTER_SELECTION_EOM,
    CLUSTER_SELECTION_LEAF,
    SELECT_TPB,
)
from hdbscan.derived.hdbscan.detail.stabilities import (
    STAB_TPB,
    get_stability_scores,
    max_lambda_of,
)
from hierarchy.derived.cluster.detail.connectivities import (
    DISTANCE_L2_SQRT_EXPANDED,
)
from neighbors.original.pinned_distance_tile import PINNED_TILE_TPB


comptime GRAPH_BUILD_BRUTE_FORCE_KNN = 0
comptime GRAPH_BUILD_NN_DESCENT = 1
"""`hdbscan.hpp:127` `enum GRAPH_BUILD_ALGO { BRUTE_FORCE_KNN, NN_DESCENT }`.
`BRUTE_FORCE_KNN` is their default (`:198`) and the only value this lane
accepts; NN_DESCENT is refused by name in `fit_hdbscan`."""


@fieldwise_init
struct HDBSCANParams(Copyable, Movable):
    """`hdbscan.hpp:129-140` `RobustSingleLinkageParams` plus `:192-199`
    `HDBSCANParams`, minus `build_params` (the all-neighbors graph
    builder, refused; see this file's header). Their defaults are in
    `default_hdbscan_params`."""

    var min_samples: Int
    var min_cluster_size: Int
    var max_cluster_size: Int
    var cluster_selection_epsilon: Float32
    var allow_single_cluster: Bool
    var alpha: Float32
    var cluster_selection_method: Int
    var build_algo: Int


def default_hdbscan_params() -> HDBSCANParams:
    """`hdbscan.hpp:132-140` and `:197-198`, value for value:
    `min_samples = 5`, `min_cluster_size = 5`, `max_cluster_size = 0`,
    `cluster_selection_epsilon = 0.0`, `allow_single_cluster = false`,
    `alpha = 1.0`, `cluster_selection_method = EOM`,
    `build_algo = BRUTE_FORCE_KNN`."""
    return HDBSCANParams(
        5, 5, 0, Float32(0.0), False, Float32(1.0),
        CLUSTER_SELECTION_EOM, GRAPH_BUILD_BRUTE_FORCE_KNN,
    )


struct HDBSCANOutput(Movable):
    """`hdbscan.hpp:203-...`'s `hdbscan_output` plus
    `robust_single_linkage_output`, reduced to what rung 1 produces.
    `probabilities` is absent by DEVIATION 1610."""

    var n_clusters: Int
    var n_outliers: Int
    var n_boruvka_rounds: Int
    var labels: List[Int32]
    """`n_rows`, FINAL labels: `0 .. n_clusters-1` or `-1` for noise,
    after their `runner.h:226-233` remap through `label_map`."""
    var raw_labels: List[Int32]
    """`n_rows`, the CONDENSED cluster ids `do_labelling_on_host` returned,
    before the remap. Recorded so a card diff can separate a selection
    difference from a numbering one."""
    var core_dists: List[Float32]
    var stabilities: List[Float32]
    """`n_clusters`, `get_stability_scores`' normalized output."""
    var tree_stabilities: List[Float32]
    """`n_condensed_clusters`, AFTER Excess of Mass mutated it."""
    var is_cluster: List[Int32]
    var inverse_label_map: List[Int32]
    var condensed: CondensedHierarchy

    def __init__(
        out self,
        n_clusters: Int,
        n_outliers: Int,
        n_boruvka_rounds: Int,
        var labels: List[Int32],
        var raw_labels: List[Int32],
        var core_dists: List[Float32],
        var stabilities: List[Float32],
        var tree_stabilities: List[Float32],
        var is_cluster: List[Int32],
        var inverse_label_map: List[Int32],
        var condensed: CondensedHierarchy,
    ):
        self.n_clusters = n_clusters
        self.n_outliers = n_outliers
        self.n_boruvka_rounds = n_boruvka_rounds
        self.labels = labels^
        self.raw_labels = raw_labels^
        self.core_dists = core_dists^
        self.stabilities = stabilities^
        self.tree_stabilities = tree_stabilities^
        self.is_cluster = is_cluster^
        self.inverse_label_map = inverse_label_map^
        self.condensed = condensed^


def effective_min_samples(min_samples: Int, m: Int) raises -> Int:
    """`runner.h:68-80`, the `min_samples + 1` self-loop adjustment and
    its clamp. Returns the `k` the k-NN actually runs at.

    Their warning text, printed rather than logged because this tree has
    no logger: "min_samples (%d) must be less than the number of samples
    in X (%zu), setting min_samples to %zu".

    THE CLAMP IS THEIRS AND IT IS NOT THE SAME AS THEIR MESSAGE. `:77`
    sets `linkage_params.min_samples = m`, not `m - 1`, while the warning
    says it is setting it to `m - 1`. `m` is the right value -- a k-NN
    over `m` points can return at most `m` neighbours including self --
    and the message is what is wrong. Transcribed with THEIR value and
    THEIR text, and the discrepancy is named here rather than silently
    corrected in one direction or the other.
    """
    if min_samples < 1:
        raise Error(
            "hdbscan: min_samples=" + String(min_samples) + " refused by"
            " name; it must be at least 1. The core distance is the k-th"
            " order statistic of a row's distances and there is no 0-th"
        )
    if min_samples + 1 > m:
        print(
            "hdbscan warning: min_samples (" + String(min_samples) + ") must"
            " be less than the number of samples in X (" + String(m) + "),"
            " setting min_samples to " + String(m - 1)
            + " (runner.h:71-77; the clamp their code applies is m = "
            + String(m) + ", which is the value used here)"
        )
        return m
    return min_samples + 1


def fit_hdbscan(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut x_host: List[Float32],
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    metric: Int,
    params: HDBSCANParams,
    tile_tpb: Int = PINNED_TILE_TPB,
    mst_tpb: Int = 256,
    mr_tpb: Int = MR_TPB,
    core_tpb: Int = CORE_TPB,
    stab_tpb: Int = STAB_TPB,
    select_tpb: Int = SELECT_TPB,
    sabotage: Int32 = HDB_SAB_NONE,
) raises -> HDBSCANOutput:
    """`runner.h:152-234` `_fit_hdbscan`, with their `build_linkage`
    (`:54-150`) inlined at the point their `:170` calls it -- one Mojo
    function for their pair, recorded in `hdbscan/DERIVATION_MAP.tsv`.
    """
    # NOT THEIRS, and it is first on purpose. `build_mr_linkage` refuses
    # `n_rows < 2` too, but only after this function has sized `n_edges =
    # m - 1` buffers -- and a zero-length allocation raises with a message
    # about a buffer rather than about the caller's data.
    if m < 2:
        raise Error(
            "hdbscan.fit_hdbscan: n_rows=" + String(m) + " < 2 refused by"
            " name; a dendrogram over fewer than two points has no merge"
            " and their root index 2 * (n_rows - 1) is not a node"
        )
    if n < 1:
        raise Error(
            "hdbscan.fit_hdbscan: n_cols=" + String(n) + " < 1 refused by"
            " name"
        )
    # `:166-168` RAFT_EXPECTS(min_samples <= m, "min_samples must be at
    # most the number of samples in X"), their wording.
    if params.min_samples > m:
        raise Error(
            "hdbscan.fit_hdbscan: min_samples must be at most the number of"
            " samples in X (min_samples=" + String(params.min_samples)
            + ", n_rows=" + String(m) + ")"
        )
    if metric != DISTANCE_L2_SQRT_EXPANDED:
        raise Error(
            "hdbscan.fit_hdbscan: metric=" + String(metric)
            + " refused by name; Currently only L2 expanded distance is"
            " supported (their RAFT_EXPECTS, reachability.cuh:109). cuML's"
            " Python surface passes L2SqrtExpanded (1) for 'euclidean' and"
            " 'l2' and nothing else reaches this path"
        )
    if params.build_algo != GRAPH_BUILD_BRUTE_FORCE_KNN:
        raise Error(
            "hdbscan.fit_hdbscan: build_algo=" + String(params.build_algo)
            + " refused by name; only BRUTE_FORCE_KNN (0, their default,"
            " hdbscan.hpp:198) is ported. NN_DESCENT reaches"
            " cuvs::neighbors::all_neighbors::build, an approximate graph"
            " builder that is not in the pinned cuVS checkout and whose"
            " output is not a function of the input alone. To close this"
            " refusal, port nn_descent in the neighbors lane and pin its"
            " initialization"
        )
    if (
        params.cluster_selection_method != CLUSTER_SELECTION_EOM
        and params.cluster_selection_method != CLUSTER_SELECTION_LEAF
    ):
        raise Error(
            "hdbscan.fit_hdbscan: cluster_selection_method="
            + String(params.cluster_selection_method)
            + " refused by name; their enum has exactly two values, EOM=0"
            " and LEAF=1 (hdbscan.hpp:126)"
        )

    var k = effective_min_samples(params.min_samples, m)
    var n_edges = m - 1

    trace.header(
        "hdbscan/derived/hdbscan/runner.mojo n_rows=" + String(m) + " n_cols="
        + String(n) + " min_samples=" + String(params.min_samples)
        + " knn_k=" + String(k) + " min_cluster_size="
        + String(params.min_cluster_size) + " max_cluster_size="
        + String(params.max_cluster_size) + " alpha=" + String(params.alpha)
        + " allow_single_cluster=" + String(params.allow_single_cluster)
        + " cluster_selection_method="
        + String(params.cluster_selection_method)
        + " metric=L2SqrtExpanded graph=DENSE_MUTUAL_REACHABILITY"
        + " (DEVIATION 1600)"
    )
    trace.record_device[DType.float32](ctx, "hdbscan.x", x, m * n)

    var core_dists = ctx.enqueue_create_buffer[DType.float32](m)
    var mst_rows = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var mst_cols = ctx.enqueue_create_buffer[DType.int32](n_edges)
    var mst_weights = ctx.enqueue_create_buffer[DType.float32](n_edges)
    var children = ctx.enqueue_create_buffer[DType.int32](n_edges * 2)
    var deltas = ctx.enqueue_create_buffer[DType.float32](n_edges)
    var sizes = ctx.enqueue_create_buffer[DType.int32](n_edges)
    ctx.synchronize()

    # `:120-133` helpers::build_linkage(..., mutual_reachability_params)
    var rounds = build_mr_linkage(
        ctx, trace, x_host, x, m, n, k, params.alpha, metric,
        core_dists, mst_rows, mst_cols, mst_weights, children, deltas, sizes,
        tile_tpb, mst_tpb, mr_tpb, core_tpb, sabotage,
    )

    # `:172-181` Condense branches of tree according to min cluster size
    var tree = build_condensed_hierarchy(
        ctx, children, deltas, sizes, params.min_cluster_size, m, sabotage
    )
    trace.record_list_i32("hdbscan.condensed.parents", tree.parents)
    trace.record_list_i32("hdbscan.condensed.children", tree.children)
    trace.record_list_f32("hdbscan.condensed.lambdas", tree.lambdas)
    trace.record_list_i32("hdbscan.condensed.sizes", tree.sizes)

    # `:183-204` Extract labels from stability
    var ext = extract_clusters(
        ctx, tree, m, params.cluster_selection_method,
        params.allow_single_cluster, params.max_cluster_size,
        params.cluster_selection_epsilon, stab_tpb, select_tpb, sabotage,
    )
    trace.record_list_f32("hdbscan.stabilities", ext.tree_stabilities)
    trace.record_list_i32("hdbscan.selected", ext.is_cluster)
    trace.record_list_i32("hdbscan.raw_labels", ext.labels)

    # `:208-210` max_lambda = *thrust::max_element(lambdas)
    var max_lambda = max_lambda_of(tree)

    # `:212-219` get_stability_scores
    var scores = get_stability_scores(
        ext.labels, ext.tree_stabilities, tree.n_clusters, max_lambda, m,
        ext.label_map, ext.n_selected,
    )

    # `:221-233` Normalize labels so they are drawn from a monotonically
    # increasing set starting at 0 even in the presence of noise (-1).
    var labels = List[Int32](capacity=m)
    var n_outliers = 0
    for i in range(m):
        var l = ext.labels[i]
        if l != Int32(-1):
            labels.append(ext.label_map[Int(l)])
        else:
            labels.append(Int32(-1))
        if labels[i] == Int32(-1):
            n_outliers += 1
    trace.record_list_i32("hdbscan.labels", labels)
    trace.record_list_f32("hdbscan.stability_scores", scores)

    var h_core = _download_f32(ctx, core_dists, m)

    _ = core_dists^
    _ = mst_rows^
    _ = mst_cols^
    _ = mst_weights^
    _ = children^
    _ = deltas^
    _ = sizes^
    return HDBSCANOutput(
        ext.n_selected,
        n_outliers,
        rounds,
        labels^,
        ext.labels.copy(),
        h_core^,
        scores^,
        ext.tree_stabilities.copy(),
        ext.is_cluster.copy(),
        ext.inverse_label_map.copy(),
        tree^,
    )


def _download_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    var v = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=v)
    ctx.synchronize()
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = v^
    return out^
