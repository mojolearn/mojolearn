# SPDX-License-Identifier: Apache-2.0
"""AgglomerativeClustering door RUN (`cluster/cfb_agglo_run.mojo`) without
Python.

Each fixture is one typed check or request builder: accepted and refused
values, the refusal wording and the request fields. Independent fixtures run
across the host's cores.
"""
from core.host_parallel import host_parallelize
from core.agglo_native_state import (
    AggloOptions, AggloTreePlan, AGGLO_PAIRWISE_MAX_ROWS, AGGLO_FULL_AUTO, AGGLO_FULL_TRUE, AGGLO_FULL_FALSE,
)
from cluster.cfb_agglo_run import (
    AGGLO_METRIC_EUCLIDEAN, AGGLO_METRIC_COSINE, AGGLO_METRIC_PRECOMPUTED, AGGLO_LINKAGE_WARD, AGGLO_SOLVER_METRIC,
    agglo_linkage_refusal, agglo_check_connectivity, agglo_metric_refusal, agglo_ward_accepts, agglo_ward_refusal,
    agglo_check_cluster_choice, agglo_check_threshold_tree, agglo_full_tree_mode, agglo_check_solver_rows,
    agglo_check_solver_clusters, agglo_solver_request, agglo_check_tree_rows, agglo_check_precomputed,
    agglo_checks_zero_rows, agglo_check_zero_rows, agglo_zero_rows_request, agglo_check_requested_clusters,
    agglo_cluster_request, agglo_warns_components, agglo_publishes_distances, agglo_fitted_clusters,
    agglo_x_predicts, agglo_check_reference_width, agglo_check_label_rank, agglo_check_label_words,
)

comptime CFB_AGGLO_RUN_FIXTURES = 14


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def refused(message: String, prefix: String) -> Bool:
    return message.startswith(prefix)


def fixture(unit: Int) raises:
    if unit == 0:
        require(String(agglo_linkage_refusal("'median'"))
                == "linkage must be one of ['average', 'complete', 'single', 'ward'], got 'median'",
                "linkage refusal wording")
        require(refused(String(agglo_metric_refusal("'chebyshev'")),
                        "NotImplementedError: mojolearn AgglomerativeClustering: metric='chebyshev' is refused by name;"),
                "metric refusal wording")
    elif unit == 1:
        agglo_check_connectivity("pairwise")
        try:
            agglo_check_connectivity("dense")
            raise Error("accepted")
        except e:
            require(String(e) == "'connectivity' can only be one of {'knn', 'pairwise'}, a connectivity matrix or a callable",
                    "connectivity refusal wording")
        try:
            agglo_check_connectivity("knn")
            raise Error("accepted")
        except e:
            require(refused(String(e), "NotImplementedError: mojolearn AgglomerativeClustering: connectivity='knn' is REFUSED BY NAME."),
                    "knn refusal wording")
    elif unit == 2:
        require(agglo_ward_accepts(AGGLO_LINKAGE_WARD, AGGLO_METRIC_EUCLIDEAN), "ward with euclidean")
        require(agglo_ward_accepts(3, AGGLO_METRIC_COSINE), "single with cosine")
        require(not agglo_ward_accepts(AGGLO_LINKAGE_WARD, AGGLO_METRIC_COSINE), "ward with cosine")
        require(String(agglo_ward_refusal("cosine"))
                == "cosine was provided as metric. Ward can only work with euclidean distances (i.e. 'euclidean' and 'l2').",
                "ward refusal wording")
    elif unit == 3:
        agglo_check_cluster_choice(True, False)
        agglo_check_cluster_choice(False, True)
        for both in range(2):
            try:
                agglo_check_cluster_choice(both == 1, both == 1)
                raise Error("accepted")
            except e:
                require(String(e) == "Exactly one of n_clusters and distance_threshold has to be set, and the other needs to be None.",
                        "cluster choice refusal wording")
    elif unit == 4:
        agglo_check_threshold_tree(True, True)
        agglo_check_threshold_tree(False, False)
        try:
            agglo_check_threshold_tree(True, False)
            raise Error("accepted")
        except e:
            require(String(e) == "compute_full_tree must be True if distance_threshold is set.", "threshold tree refusal")
        require(agglo_full_tree_mode(True, True) == AGGLO_FULL_AUTO and agglo_full_tree_mode(False, True) == AGGLO_FULL_TRUE
                and agglo_full_tree_mode(False, False) == AGGLO_FULL_FALSE, "full tree modes")
    elif unit == 5:
        agglo_check_solver_rows(2, 3, False)
        agglo_check_solver_rows(AGGLO_PAIRWISE_MAX_ROWS + 1, 3, True)
        try:
            agglo_check_solver_rows(1, 3, False)
            raise Error("accepted")
        except e:
            require(String(e) == "Found array with 1 sample(s) (shape=(1, 3)) while a minimum of 2 is required.",
                    "solver minimum refusal wording")
        try:
            agglo_check_solver_rows(AGGLO_PAIRWISE_MAX_ROWS + 1, 3, False)
            raise Error("accepted")
        except e:
            require(refused(String(e), "mojolearn AgglomerativeClustering: n_rows=46341 > 46340; the dense connectivity"),
                    "solver dense bound refusal wording")
    elif unit == 6:
        agglo_check_solver_clusters(1, 4)
        agglo_check_solver_clusters(4, 4)
        try:
            agglo_check_solver_clusters(5, 4)
            raise Error("accepted")
        except e:
            require(String(e) == "Expected 1 <= n_clusters <= n_rows (4), got n_clusters=5", "solver clusters refusal")
        var r = agglo_solver_request(1, 2, 3, 4, 5, 6, 7)
        require(r.source == 1 and r.children == 2 and r.labels == 3 and r.info == 4 and r.rows == 5
                and r.features == 6 and r.clusters == 7, "solver request order")
        require(r.metric == AGGLO_SOLVER_METRIC and r.metric == 1 and not r.use_knn, "solver ABI metric code")
    elif unit == 7:
        agglo_check_tree_rows(2)
        try:
            agglo_check_tree_rows(1)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn AgglomerativeClustering: n_rows=1 < 2; a tree needs at least two points",
                    "tree rows refusal wording")
        agglo_check_precomputed(AGGLO_METRIC_PRECOMPUTED, 3, 3)
        agglo_check_precomputed(AGGLO_METRIC_EUCLIDEAN, 3, 2)
        try:
            agglo_check_precomputed(AGGLO_METRIC_PRECOMPUTED, 3, 2)
            raise Error("accepted")
        except e:
            require(String(e) == "Distance matrix should be square, got matrix of shape (3, 2)", "square refusal wording")
    elif unit == 8:
        require(agglo_checks_zero_rows(AGGLO_METRIC_COSINE) and not agglo_checks_zero_rows(AGGLO_METRIC_EUCLIDEAN),
                "cosine tests zero rows")
        agglo_check_zero_rows(False)
        try:
            agglo_check_zero_rows(True)
            raise Error("accepted")
        except e:
            require(String(e) == "Cosine affinity cannot be used when X contains zero vectors", "zero rows refusal")
        var z = agglo_zero_rows_request(1, 2, 3, 4)
        require(z.source == 1 and z.flags == 2 and z.rows == 3 and z.features == 4, "zero rows request fields")
    elif unit == 9:
        agglo_check_requested_clusters(3, 3)
        for k in range(2):
            try:
                agglo_check_requested_clusters(k * 4, 3)
                raise Error("accepted")
            except e:
                require(String(e) == "Cannot extract more clusters than samples: " + String(k * 4)
                        + " clusters were given for a tree with 3 leaves.", "requested clusters refusal")
    elif unit == 10:
        var options = AggloOptions(2, AGGLO_METRIC_COSINE, False, True, False, False, AGGLO_FULL_AUTO)
        var plan = AggloTreePlan(True, 9, 1, True)
        var r = agglo_cluster_request(1, 2, 3, 4, 5, 6, 7, 8, 10, 11, options, 12, plan, 13, 0.75)
        require(r.source == 1 and r.connectivity == 2 and r.children == 3 and r.labels == 4 and r.distances == 5
                and r.scalars == 6 and r.source_count == 7 and r.connectivity_count == 8, "cluster request addresses")
        require(r.rows == 10 and r.features == 11 and r.linkage == 2 and r.metric == AGGLO_METRIC_COSINE
                and r.edges == 12 and r.merges == 9 and r.edge_mode == 13 and r.cut == 1 and r.use_threshold
                and r.minkowski == 2.0 and r.threshold == 0.75, "cluster request fields")
    elif unit == 11:
        var threshold = AggloOptions(3, AGGLO_METRIC_EUCLIDEAN, False, True, False, True, AGGLO_FULL_TRUE)
        var counted = AggloOptions(3, AGGLO_METRIC_EUCLIDEAN, False, False, False, False, AGGLO_FULL_AUTO)
        var full = AggloTreePlan(True, 9, 1, True)
        var cut = AggloTreePlan(False, 7, 0, False)
        require(agglo_fitted_clusters(threshold, full, 4, 0) == 4, "a threshold cut counts its clusters")
        require(agglo_fitted_clusters(threshold, cut, 4, 2) == 2 and agglo_fitted_clusters(counted, full, 4, 3) == 3,
                "otherwise the requested count")
        require(agglo_publishes_distances(threshold) and not agglo_publishes_distances(counted), "distances_ published")
        require(agglo_x_predicts(threshold) and not agglo_x_predicts(counted), "single euclidean prediction data")
    elif unit == 12:
        require(agglo_warns_components(True, 2) and not agglo_warns_components(True, 1)
                and not agglo_warns_components(False, 3), "component warning")
        agglo_check_reference_width(3, 3)
        try:
            agglo_check_reference_width(3, 4)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn AgglomerativeClustering.predict: fitted reference width differs from X",
                    "reference width refusal")
    else:
        agglo_check_label_rank(1)
        agglo_check_label_words(1, 4, True, 5, 5)
        try:
            agglo_check_label_rank(2)
            raise Error("accepted")
        except e:
            require(String(e) == "TypeError: hierarchy fitted labels must be one-dimensional", "label rank refusal")
        for arm in range(4):
            try:
                if arm == 0:
                    agglo_check_label_words(2, 4, True, 5, 5)
                elif arm == 1:
                    agglo_check_label_words(1, 8, True, 5, 5)
                elif arm == 2:
                    agglo_check_label_words(1, 4, False, 5, 5)
                else:
                    agglo_check_label_words(1, 4, True, 4, 5)
                raise Error("accepted")
            except e:
                require(String(e) == "TypeError: hierarchy fitted labels require one native contiguous int32 value per reference row",
                        "label words refusal")


def run_checks() raises:
    var failures = List[Int](length=CFB_AGGLO_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL cfb_agglo_run fixture", unit, String(error))
    host_parallelize(task, CFB_AGGLO_RUN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "cfb_agglo_run failed " + String(failed) + " fixtures")
    print("PASS cfb_agglo_run:", CFB_AGGLO_RUN_FIXTURES, "typed check and request fixtures")
