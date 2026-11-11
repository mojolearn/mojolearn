# SPDX-License-Identifier: Apache-2.0
"""AgglomerativeClustering door RUN: the checks on typed values and the typed
fit requests.

`bindings/cfb_agglo.mojo` is the UNPACK/PACK side: it reads the estimator's
fields, admits the arrays, allocates and publishes the results and pins the
public owners; `bindings/agglo_native_boundary.mojo` connects the typed
requests to the compiled modules. The door calls these checks at the point the
public method made them, so the refusal order and wording are unchanged. A
refusal whose wording quotes the caller's value takes that text from the
door. The requests, scratch and route plans are `core/agglo_native_api.mojo`
and `core/agglo_native_state.mojo`'s. Nothing here imports Python.
"""
from core.agglo_native_api import AggloSolverRequest, AggloClusterRequest, AggloZeroRowsRequest
from core.agglo_native_state import (
    AggloOptions, AggloTreePlan, AGGLO_PAIRWISE_MAX_ROWS, AGGLO_FULL_AUTO, AGGLO_FULL_TRUE, AGGLO_FULL_FALSE,
    agglo_connectivity_code, agglo_predicts,
)

comptime AGGLO_METRIC_UNKNOWN = -2
comptime AGGLO_METRIC_EUCLIDEAN = -1
comptime AGGLO_METRIC_COSINE = 4
comptime AGGLO_METRIC_PRECOMPUTED = 5
comptime AGGLO_LINKAGE_WARD = 0
#: The solver ABI uses cuML's sqrt-Euclidean code 1; the reference ABI uses
#: code 0.
comptime AGGLO_SOLVER_METRIC = 1
#: The x_cluster fit's Minkowski power (unused by its metrics, fixed by its ABI).
comptime AGGLO_CLUSTER_MINKOWSKI = 2.0


def agglo_linkage_refusal(linkage_repr: String) -> Error:
    return Error("linkage must be one of ['average', 'complete', 'single', 'ward'], got " + linkage_repr)


def agglo_check_connectivity(name: String) raises:
    """A named connectivity: 'pairwise' is served, 'knn' is refused by name."""
    if agglo_connectivity_code(name) < 0:
        raise Error("'connectivity' can only be one of {'knn', 'pairwise'}, "
                    + "a connectivity matrix or a callable")
    if name == "knn":
        raise Error("NotImplementedError: mojolearn AgglomerativeClustering: connectivity='knn' is "
                    + "REFUSED BY NAME. The Linkage::KNN_GRAPH specialization "
                    + "(connectivities.cuh:49), the cross-component fix-up its "
                    + "forest MST needs (connect_knn_graph, mst.cuh:67 and :131) "
                    + "and merge_msts are rung 2 and NOT IMPLEMENTED -- and their host "
                    + "overload picks a RANDOM vertex per component from "
                    + "std::mt19937(std::random_device()), which would have to be "
                    + "pinned first. Use connectivity='pairwise' (cuML's C++ "
                    + "default and scikit-learn's dense tree), or pass the k-NN "
                    + "graph itself as a connectivity matrix (scikit-learn's "
                    + "kneighbors_graph). See hierarchy/NOT_IMPLEMENTED.tsv")


def agglo_metric_refusal(metric_repr: String) -> Error:
    return Error("NotImplementedError: mojolearn AgglomerativeClustering: metric=" + metric_repr + " is "
                 + "refused by name; one of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'precomputed']. A callable "
                 + "metric would compute the dissimilarities outside every "
                 + "identity column. See x_cluster/NOT_IMPLEMENTED.tsv")


def agglo_ward_accepts(linkage: Int, metric: Int) -> Bool:
    return linkage != AGGLO_LINKAGE_WARD or metric == AGGLO_METRIC_EUCLIDEAN


def agglo_ward_refusal(metric_text: String) -> Error:
    return Error(metric_text + " was provided as metric. Ward can only work with "
                 + "euclidean distances (i.e. 'euclidean' and 'l2').")


def agglo_check_cluster_choice(has_clusters: Bool, has_threshold: Bool) raises:
    if has_clusters == has_threshold:
        raise Error("Exactly one of n_clusters and distance_threshold has to be "
                    + "set, and the other needs to be None.")


def agglo_check_threshold_tree(has_threshold: Bool, full_tree: Bool) raises:
    if has_threshold and not full_tree:
        raise Error("compute_full_tree must be True if distance_threshold is set.")


def agglo_full_tree_mode(is_auto: Bool, full_tree: Bool) -> Int:
    if is_auto:
        return AGGLO_FULL_AUTO
    return AGGLO_FULL_TRUE if full_tree else AGGLO_FULL_FALSE


def agglo_check_solver_rows(rows: Int, features: Int, fast_mst: Bool) raises:
    """The Boruvka solver route's row checks: scikit-learn's check_array
    minimum, then the dense connectivity matrix's `m * m` int bound (FAST on
    Apple builds the MST without it)."""
    if rows < 2:
        # scikit-learn's check_array wording (ensure_min_samples=2)
        raise Error("Found array with " + String(rows) + " sample(s) (shape=(" + String(rows) + ", "
                    + String(features) + ")) while a minimum of 2 is required.")
    if rows > AGGLO_PAIRWISE_MAX_ROWS and not fast_mst:
        raise Error("mojolearn AgglomerativeClustering: n_rows=" + String(rows) + " > "
                    + String(AGGLO_PAIRWISE_MAX_ROWS) + "; the dense connectivity matrix is "
                    + "m * m of cuVS's int and overflows past that "
                    + "(hierarchy/impl/cluster/detail/connectivities.mojo "
                    + "refuses it by name too)")


def agglo_check_solver_clusters(clusters: Int, rows: Int) raises:
    if clusters < 1 or clusters > rows:
        raise Error("Expected 1 <= n_clusters <= n_rows (" + String(rows) + "), got "
                    + "n_clusters=" + String(clusters))


def agglo_solver_request(source: Int, children: Int, labels: Int, info: Int, rows: Int, features: Int,
                         clusters: Int) -> AggloSolverRequest:
    return AggloSolverRequest(source, children, labels, info, rows, features, clusters, AGGLO_SOLVER_METRIC, False)


def agglo_check_tree_rows(rows: Int) raises:
    if rows < 2:
        raise Error("mojolearn AgglomerativeClustering: n_rows=" + String(rows) + " < 2; a tree "
                    + "needs at least two points")


def agglo_check_precomputed(metric: Int, rows: Int, features: Int) raises:
    if metric == AGGLO_METRIC_PRECOMPUTED and rows != features:
        raise Error("Distance matrix should be square, got matrix of shape (" + String(rows) + ", "
                    + String(features) + ")")


def agglo_checks_zero_rows(metric: Int) -> Bool:
    """Cosine affinity tests every row for zero on the device."""
    return metric == AGGLO_METRIC_COSINE


def agglo_check_zero_rows(any_zero: Bool) raises:
    """Run with `features == 0` (every row is zero) before the device test,
    and with its result after it."""
    if any_zero:
        raise Error("Cosine affinity cannot be used when X contains zero vectors")


def agglo_zero_rows_request(source: Int, flags: Int, rows: Int, features: Int) -> AggloZeroRowsRequest:
    return AggloZeroRowsRequest(source, flags, rows, features)


def agglo_check_requested_clusters(clusters: Int, rows: Int) raises:
    if clusters < 1 or clusters > rows:
        raise Error("Cannot extract more clusters than samples: " + String(clusters) + " clusters "
                    + "were given for a tree with " + String(rows) + " leaves.")


def agglo_cluster_request(source: Int, connectivity: Int, children: Int, labels: Int, distances: Int,
                          scalars: Int, source_count: Int, connectivity_count: Int, rows: Int, features: Int,
                          options: AggloOptions, edges: Int, plan: AggloTreePlan, edge_mode: Int,
                          threshold: Float64) -> AggloClusterRequest:
    return AggloClusterRequest(source, connectivity, children, labels, distances, scalars, source_count,
        connectivity_count, rows, features, options.linkage, options.metric, edges, plan.merges, edge_mode,
        plan.cut, plan.use_threshold, AGGLO_CLUSTER_MINKOWSKI, threshold)


def agglo_warns_components(constrained: Bool, components: Int) -> Bool:
    return constrained and components > 1


def agglo_publishes_distances(options: AggloOptions) -> Bool:
    return options.has_threshold or options.compute_distances


def agglo_fitted_clusters(options: AggloOptions, plan: AggloTreePlan, cut_clusters: Int, requested: Int) -> Int:
    """`n_clusters_`: the threshold cut's count on a full tree, else the
    requested count."""
    return cut_clusters if options.has_threshold and plan.full else requested


def agglo_x_predicts(options: AggloOptions) -> Bool:
    return options.prediction_data and agglo_predicts(options.linkage, options.metric)


def agglo_check_reference_width(reference_features: Int, query_features: Int) raises:
    if reference_features != query_features:
        raise Error("mojolearn AgglomerativeClustering.predict: fitted reference width differs from X")


def agglo_check_label_rank(ndim: Int) raises:
    if ndim != 1:
        raise Error("TypeError: hierarchy fitted labels must be one-dimensional")


def agglo_check_label_words(kind: Int, itemsize: Int, c_order: Bool, count: Int, rows: Int) raises:
    if kind != 1 or itemsize != 4 or not c_order or count != rows:
        raise Error("TypeError: hierarchy fitted labels require one native contiguous int32 value per reference row")
