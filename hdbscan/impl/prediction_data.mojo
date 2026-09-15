# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`PredictionData` and `generate_prediction_data`, the `prediction_data=True`
half of a fit.

Reference: `cuml-v26.08.00/cpp/src/hdbscan/prediction_data.cu` (`allocate`
`:34-46`, `build_index_into_children` `:60-79`, `generate_prediction_data`
`:92-239`) and the container at `cpp/include/cuml/cluster/hdbscan.hpp:369-435`.
Steps run in the reference order.

WHAT IT HOLDS (their getters, `hdbscan.hpp:401-408`)
  deaths               n_clusters, the largest child lambda under each
                       condensed cluster (`:112-145`)
  exemplar_idx         the points of a LEAF cluster that are still in it
                       at its death, grouped by label (`:158-211`)
  exemplar_label_offsets  n_selected + 1, the CSR over those groups
                       (`:221-226`)
  selected_clusters    n_selected, condensed node id of each final label
                       (`:228-234`)
  index_into_children  n_edges + 1, node id -> its edge in the condensed
                       tree (`:236-237`)
  core_dists           theirs keeps a POINTER to the fit's; the Python
                       estimator keeps `core_distances_` for the same use.

======================================================================
DEVIATION BLOCK -- DEVIATION 1612. PREDICTION DATA IS BUILT ON THE HOST.
======================================================================
WHAT THEIRS DOES. Thrust transforms and CUB segmented reductions over
the condensed tree's device vectors.

WHAT OURS DOES. The same passes as loops over host `List`s, because this
lane's condensed tree is already a host structure (DEVIATION 1611 and
`condensed_hierarchy.mojo`'s header: their own builder, `condense.cuh:92-212`,
is host code). No pass here does arithmetic. `deaths` is a MAX, which is
exact and order-free on values the fit has already refused to be NaN
(DEVIATION 1607); the rest are integer compares, copies and counts. So the
result is a pure function of the tree, the labels and the label map, and
the device and host bindings call this one function.

THE ONE ORDER CHOICE. Their `thrust::sort_by_key` of the exemplars by
original label (`:208-211`) is not specified stable. Ours sorts on the
packed key `(label << 32) | point index` (the `merge_sort_u64_with_index`
DEVIATION 1611 already uses), a total order that equals a stable sort of
the ascending `copy_if` output. The exemplar ORDER feeds nothing but a
minimum over each label's exemplars in `membership_vector`, which does not
depend on it; the array itself is what a caller could read.

THE UNWRITTEN SLOTS. Their `is_exemplar` and `index_into_children` are
`rmm::device_uvector`s, uninitialized, and `index_into_children[root]` is
never written (the root is never a child). Every leaf is a child exactly
once, so `is_exemplar` is fully written; the root slot of
`index_into_children` is filled with -1 here and no reader reaches it
(`kernels/predict.cuh:64-65` reads it only for a selected cluster above the
root, `kernels/soft_clustering.cuh:34-44` climbs only from a larger id).
======================================================================
"""

from hierarchy.impl.sparse.op.sort import merge_sort_u64_with_index


comptime PD_FLOAT32_LOWEST = Float32(-3.4028234663852886e38)
"""`std::numeric_limits<float>::lowest()`, CUB `DeviceSegmentedReduce::Max`'s
initial value."""


struct PredictionData(Copyable, Movable):
    """`hdbscan.hpp:369-435`, host lists."""

    var n_leaves: Int
    var n_edges: Int
    var n_clusters: Int
    var n_selected_clusters: Int
    var n_exemplars: Int
    var deaths: List[Float32]
    var exemplar_idx: List[Int32]
    var exemplar_label_offsets: List[Int32]
    var selected_clusters: List[Int32]
    var index_into_children: List[Int32]

    def __init__(
        out self,
        n_leaves: Int,
        n_edges: Int,
        n_clusters: Int,
        n_selected_clusters: Int,
        n_exemplars: Int,
        var deaths: List[Float32],
        var exemplar_idx: List[Int32],
        var exemplar_label_offsets: List[Int32],
        var selected_clusters: List[Int32],
        var index_into_children: List[Int32],
    ):
        self.n_leaves = n_leaves
        self.n_edges = n_edges
        self.n_clusters = n_clusters
        self.n_selected_clusters = n_selected_clusters
        self.n_exemplars = n_exemplars
        self.deaths = deaths^
        self.exemplar_idx = exemplar_idx^
        self.exemplar_label_offsets = exemplar_label_offsets^
        self.selected_clusters = selected_clusters^
        self.index_into_children = index_into_children^


def generate_prediction_data(
    parents: List[Int32],
    children: List[Int32],
    lambdas: List[Float32],
    sizes: List[Int32],
    n_edges: Int,
    n_leaves: Int,
    n_clusters: Int,
    labels: List[Int32],
    inverse_label_map: List[Int32],
    n_selected_clusters: Int,
) raises -> PredictionData:
    """`prediction_data.cu:92-239`. `labels` are the FINAL labels (`0 ..
    n_selected-1` or -1), `inverse_label_map` maps a final label to its
    condensed cluster id counted from 0 (their `inverse_label_map`)."""
    if n_edges < 1 or len(parents) < n_edges or len(children) < n_edges:
        raise Error(
            "hdbscan.generate_prediction_data: the condensed tree has "
            + String(n_edges) + " edges but the arrays hold fewer; refused"
            " by name"
        )
    if len(labels) < n_leaves or len(inverse_label_map) < n_selected_clusters:
        raise Error(
            "hdbscan.generate_prediction_data: labels or inverse_label_map is"
            " shorter than n_leaves / n_selected_clusters; refused by name"
        )
    # `:112-145` the death of each cluster: parent CSR + segmented Max.
    # Their segments are the parent-sorted edges; a max over each segment
    # is the same value visited in any order.
    var deaths = List[Float32](length=n_clusters, fill=PD_FLOAT32_LOWEST)
    for e in range(n_edges):
        var p = Int(parents[e]) - n_leaves
        if p < 0 or p >= n_clusters:
            raise Error(
                "hdbscan.generate_prediction_data: parent "
                + String(Int(parents[e])) + " at edge " + String(e)
                + " is outside the condensed cluster ids; refused by name"
            )
        if deaths[p] < lambdas[e]:
            deaths[p] = lambdas[e]

    # `:147-156` is_leaf_cluster: a cluster with a cluster child is not a leaf.
    var is_leaf_cluster = List[Int32](length=n_clusters, fill=Int32(1))
    for e in range(n_edges):
        if sizes[e] > Int32(1):
            is_leaf_cluster[Int(parents[e]) - n_leaves] = Int32(0)

    # `:162-179` exemplar_op
    var is_exemplar = List[Int32](length=n_leaves, fill=Int32(0))
    for e in range(n_edges):
        var c = Int(children[e])
        if c < n_leaves:
            var p = Int(parents[e]) - n_leaves
            var ex = (
                labels[c] != Int32(-1)
                and is_leaf_cluster[p] != Int32(0)
                and lambdas[e] == deaths[p]
            )
            is_exemplar[c] = Int32(1) if ex else Int32(0)

    # `:181-182` count_if
    var n_exemplars = 0
    for i in range(n_leaves):
        if is_exemplar[i] != Int32(0):
            n_exemplars += 1

    # `:184` allocate
    var exemplar_idx = List[Int32](capacity=n_exemplars)
    var exemplar_label_offsets = List[Int32](
        length=n_selected_clusters + 1, fill=Int32(0)
    )
    var selected_clusters = List[Int32](
        length=n_selected_clusters, fill=Int32(0)
    )
    var index_into_children = List[Int32](length=n_edges + 1, fill=Int32(-1))

    # `:186-191` copy_if over the counting iterator (ascending, stable)
    for i in range(n_leaves):
        if is_exemplar[i] != Int32(0):
            exemplar_idx.append(Int32(i))

    # `:194-206` exemplar_labels through the inverse label map
    var exemplar_labels = List[Int32](capacity=n_exemplars)
    for j in range(n_exemplars):
        var label = labels[Int(exemplar_idx[j])]
        if label != Int32(-1):
            exemplar_labels.append(inverse_label_map[Int(label)])
        else:
            exemplar_labels.append(Int32(-1))

    # `:208-211` sort_by_key(exemplar_labels, exemplar_idx). DEVIATION 1612.
    if n_exemplars > 0:
        var keys = List[UInt64](capacity=n_exemplars)
        var order = List[Int](capacity=n_exemplars)
        for j in range(n_exemplars):
            var lab = UInt64(Int(exemplar_labels[j])) & UInt64(0xFFFFFFFF)
            keys.append((lab << UInt64(32)) | UInt64(Int(exemplar_idx[j])))
            order.append(j)
        merge_sort_u64_with_index(keys, order)
        var s_idx = List[Int32](capacity=n_exemplars)
        var s_lab = List[Int32](capacity=n_exemplars)
        for j in range(n_exemplars):
            s_idx.append(exemplar_idx[order[j]])
            s_lab.append(exemplar_labels[order[j]])
        exemplar_idx = s_idx^
        exemplar_labels = s_lab^

    # `:214-219` converted (final) labels of the sorted exemplars
    # `:221-238`, only when there is an exemplar
    if n_exemplars > 0:
        # raft sorted_coo_to_csr (sparse/convert/detail/csr.cuh:78-90):
        # row counts over m = n_selected + 1 rows, then an exclusive scan.
        var counts = List[Int32](length=n_selected_clusters + 1, fill=Int32(0))
        for j in range(n_exemplars):
            var cl = Int(labels[Int(exemplar_idx[j])])
            if cl < 0 or cl > n_selected_clusters:
                raise Error(
                    "hdbscan.generate_prediction_data: exemplar label "
                    + String(cl) + " is outside [0, "
                    + String(n_selected_clusters) + "]; refused by name"
                )
            counts[cl] += Int32(1)
        var acc = Int32(0)
        for c in range(n_selected_clusters + 1):
            exemplar_label_offsets[c] = acc
            acc += counts[c]
        # `:228-234` selected_clusters[c] = exemplar_labels[offsets[c]] + n_leaves
        for c in range(n_selected_clusters):
            var off = Int(exemplar_label_offsets[c])
            if off >= n_exemplars:
                raise Error(
                    "hdbscan.generate_prediction_data: selected cluster "
                    + String(c) + " has no exemplar, and their transform at"
                    " prediction_data.cu:228-234 would read past"
                    " exemplar_labels; refused by name"
                )
            selected_clusters[c] = exemplar_labels[off] + Int32(n_leaves)
        # `:236-237` build_index_into_children (`:60-79`)
        for e in range(n_edges):
            var ch = Int(children[e])
            if ch < 0 or ch > n_edges:
                raise Error(
                    "hdbscan.generate_prediction_data: child "
                    + String(ch) + " is outside [0, n_edges]; refused by name"
                )
            index_into_children[ch] = Int32(e)
    return PredictionData(
        n_leaves, n_edges, n_clusters, n_selected_clusters, n_exemplars,
        deaths^, exemplar_idx^, exemplar_label_offsets^,
        selected_clusters^, index_into_children^,
    )


def prediction_neighborhood(min_samples: Int, n_rows: Int) raises -> Int:
    """`predict.cuh:161`, `neighborhood = (min_samples - 1) * 2`, with the
    two values theirs does not guard refused by name: below 2 the
    neighborhood is empty (their k-NN at k = 0), and above `n_rows` the
    brute-force k-NN has fewer points than neighbors."""
    if min_samples < 2:
        raise Error(
            "hdbscan.approximate_predict: min_samples=" + String(min_samples)
            + " refused by name; the prediction neighborhood (min_samples - 1)"
            " * 2 (predict.cuh:161) is empty below 2"
        )
    var k = (min_samples - 1) * 2
    if k > n_rows:
        raise Error(
            "hdbscan.approximate_predict: the prediction neighborhood "
            + String(k) + " = (min_samples - 1) * 2 exceeds the "
            + String(n_rows) + " training rows; refused by name"
        )
    return k


def refuse_nonfinite_queries(q: List[Float32], n_q: Int, d: Int) raises:
    """DEVIATION 1607 at the query boundary: a NaN or infinite query cell
    would carry the vendor's NaN payload into a distance. Refused by name."""
    from std.math import isfinite

    for i in range(n_q * d):
        if not isfinite(q[i]):
            raise Error(
                "hdbscan.approximate_predict: points_to_predict has a NaN or"
                " infinite value at row " + String(i // d) + ", column "
                + String(i % d) + "; refused by name (DEVIATION 1607)"
            )
