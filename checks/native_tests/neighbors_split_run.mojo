# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The typed RUN requests of the NearestCentroid, LocalOutlierFactor and
RadiusNeighbors doors (`neighbors/nearest_centroid_run.mojo`,
`neighbors/lof_run.mojo`, `neighbors/radius_run.mojo`) without Python, and the
two radius operations of the neighbors table.

Each fixture is one typed step that needs no kernel image: request fields,
accepted and refused values with their wording, the priors step, the id
widening and the table's span checks. Independent fixtures run across the
host's cores. The kernels behind the requests are covered by the plan suites
and by the installed-wheel tests.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases
from core.neighbors_entry_table import (
    NB_OP_RADIUS_COUNT, NB_OP_RADIUS_FILL, NB_OPERATION_COUNT, nb_ops_check_spans,
    NB_OP_SEARCH, NB_OP_SEARCH_RESIDENT, NB_OP_RBC_SEARCH, NB_OP_CLASSIFY_NEIGHBORS, NB_OP_REGRESS_NEIGHBORS,
    NbFold,
)
from neighbors.neighbors_local_run import LOF_RESULT_LABELS, LOF_AUTO_OFFSET
from neighbors.nearest_centroid_run import (
    NC_PRIORS_EMPIRICAL, NC_PRIORS_UNIFORM, NC_PRIORS_GIVEN,
    NcFitRequest, NcFitOutcome, NcQueryRequest, nc_fit_priors_run, nc_proba_wide,
)
from neighbors.neighbors_knn_run import (
    KnnDistWords, KnnQueryRequest, KnnQueryOutcome, knn_check_fold_entry,
)
from neighbors.lof_run import LofFitRequest, LofFitOutcome, LofQueryRequest
from neighbors.radius_run import (
    RadiusRequest, RadiusCount, radius_widen_run,
)
from bindings.parallel_neighbor_worker_run import (
    neighbor_radius_admit, neighbor_sort_admit, neighbor_features_admit,
)

comptime NEIGHBORS_SPLIT_RUN_FIXTURES = 9


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _near(a: Float64, b: Float64) -> Bool:
    return abs(a - b) < 1e-12


def fixture(unit: Int) raises:
    if unit == 0:
        # radius: accepted values pass, refusals keep their wording
        neighbor_radius_admit(0.5)
        neighbor_sort_admit(True, True)
        neighbor_sort_admit(False, False)
        neighbor_features_admit(3, 3)
        try:
            neighbor_radius_admit(0.0)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: radius must be positive and finite", "radius refusal wording")
        try:
            neighbor_sort_admit(True, False)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: sort_results=True requires return_distance=True",
                    "sort refusal wording")
        try:
            neighbor_features_admit(2, 3)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: input feature count differs from fit", "feature refusal wording")
    elif unit == 1:
        # the two radius operations: their spans are accepted, any other refused
        require(NB_OP_RADIUS_COUNT == 7 and NB_OP_RADIUS_FILL == 8 and NB_OPERATION_COUNT == 14, "operation ids")
        var words = List[Int](length=8, fill=0)
        var reals = List[Float64](length=2, fill=0)
        var result = List[Int](length=1, fill=0)
        var aa = Int(words.unsafe_ptr())
        var ff = Int(reals.unsafe_ptr())
        var rr = Int(result.unsafe_ptr())
        nb_ops_check_spans(NB_OP_RADIUS_COUNT, aa, 4, aa, 4, ff, 2, rr)
        nb_ops_check_spans(NB_OP_RADIUS_FILL, aa, 5, aa, 6, ff, 2, rr)
        try:
            nb_ops_check_spans(NB_OP_RADIUS_COUNT, aa, 5, aa, 4, ff, 2, rr)
            raise Error("accepted")
        except e:
            require(String(e) == "neighbors ops: request spans differ from operation 7", "count span refusal")
        try:
            nb_ops_check_spans(NB_OP_RADIUS_FILL, aa, 5, aa, 6, ff, 1, rr)
            raise Error("accepted")
        except e:
            require(String(e) == "neighbors ops: request spans differ from operation 8", "fill span refusal")
        _ = words^
        _ = reals^
        _ = result^
    elif unit == 2:
        # NearestCentroid priors: uniform is written, given is normalized, empirical is left to the label step
        var prior = List[Float64](length=4, fill=7.0)
        var request = NcFitRequest(0, 0, 8, 2, 4, True, NC_PRIORS_UNIFORM, 0.0)
        request.prior = Int(prior.unsafe_ptr())
        nc_fit_priors_run(request)
        for i in range(4):
            require(prior[i] == 0.25, "uniform prior")
        var given = List[Float64](length=2, fill=1.0)
        given[1] = 3.0
        var second = NcFitRequest(0, 0, 8, 2, 2, True, NC_PRIORS_GIVEN, 0.0)
        second.prior = Int(given.unsafe_ptr())
        nc_fit_priors_run(second)
        require(_near(given[0], 0.25) and _near(given[1], 0.75), "given priors normalized")
        var kept = List[Float64](length=2, fill=5.0)
        var third = NcFitRequest(0, 0, 8, 2, 2, True, NC_PRIORS_EMPIRICAL, 0.0)
        third.prior = Int(kept.unsafe_ptr())
        nc_fit_priors_run(third)
        require(kept[0] == 5.0 and kept[1] == 5.0, "empirical priors untouched")
        _ = prior^
        _ = given^
        _ = kept^
    elif unit == 3:
        # NearestCentroid requests: fields and the probability width
        var fit = NcFitRequest(11, 12, 5, 3, 2, False, NC_PRIORS_GIVEN, 0.5)
        require(fit.x == 11 and fit.labels == 12 and fit.rows == 5 and fit.features == 3 and fit.classes == 2
                and not fit.euclidean and fit.priors == NC_PRIORS_GIVEN and fit.shrink == 0.5
                and fit.prior == 0 and fit.stats == 0 and fit.centroids == 0 and fit.deviations == 0, "fit request")
        var outcome = NcFitOutcome()
        require(outcome.status == 0 and not outcome.constant, "fit outcome default")
        var query = NcQueryRequest(1, 2, 3, 4, 6, 3, 2, True, True, True)
        require(query.query == 1 and query.centroids == 2 and query.std == 3 and query.prior == 4
                and query.rows == 6 and query.features == 3 and query.classes == 2, "query request")
        require(nc_proba_wide(query, False) and not nc_proba_wide(query, True), "wide probabilities")
        var narrow = NcQueryRequest(1, 2, 3, 4, 6, 3, 2, True, True, False)
        require(not nc_proba_wide(narrow, False), "float32 probabilities")
    elif unit == 4:
        # LocalOutlierFactor requests
        var fit = LofFitRequest(21, 9, 4, 3, False, True, True, 0.0)
        require(fit.x == 21 and fit.rows == 9 and fit.features == 4 and fit.neighbors == 3 and not fit.unfused
                and fit.old_items and fit.auto and fit.distances == 0 and fit.densities == 0 and fit.scores == 0,
                "fit request")
        var outcome = LofFitOutcome()
        require(outcome.status == 0 and outcome.offset == LOF_AUTO_OFFSET, "fit outcome default")
        var query = LofQueryRequest(1, 2, 3, 4, 5, 9, 4, 3, 9, True, False, LOF_RESULT_LABELS, 1.5)
        require(query.query == 1 and query.fit_x == 2 and query.fit_distances == 3 and query.fit_densities == 4
                and query.rows == 5 and query.reference_rows == 9 and query.features == 4 and query.neighbors == 3
                and query.fit_rows == 9 and query.unfused and not query.old_items
                and query.kind == LOF_RESULT_LABELS and query.shift == 1.5 and query.dst == 0, "query request")
    elif unit == 5:
        # RadiusNeighbors request and count
        var request = RadiusRequest(1, 2, 3, 10, 4, 6, 0, 0.5, 2.0, True)
        require(request.index == 1 and request.queries == 2 and request.row_starts == 3 and request.n_index == 10
                and request.n_queries == 4 and request.n_features == 6 and request.metric == 0
                and request.radius == 0.5 and request.metric_arg == 2.0 and request.device_status, "radius request")
        var count = RadiusCount()
        require(count.edges == 0 and count.status == 0, "radius count default")
    elif unit == 7:
        # the k-NN query and vote operations: ids and spans
        require(NB_OP_SEARCH == 9 and NB_OP_SEARCH_RESIDENT == 10 and NB_OP_RBC_SEARCH == 11
                and NB_OP_CLASSIFY_NEIGHBORS == 12 and NB_OP_REGRESS_NEIGHBORS == 13, "k-NN operation ids")
        var words = List[Int](length=8, fill=0)
        var reals = List[Float64](length=1, fill=0)
        var result = List[Int](length=1, fill=0)
        var aa = Int(words.unsafe_ptr())
        var ff = Int(reals.unsafe_ptr())
        var rr = Int(result.unsafe_ptr())
        nb_ops_check_spans(NB_OP_SEARCH, aa, 4, aa, 7, ff, 1, rr)
        nb_ops_check_spans(NB_OP_SEARCH_RESIDENT, aa, 5, aa, 8, ff, 1, rr)
        nb_ops_check_spans(NB_OP_RBC_SEARCH, aa, 4, aa, 5, ff, 1, rr)
        nb_ops_check_spans(NB_OP_CLASSIFY_NEIGHBORS, aa, 6, aa, 8, 0, 0, rr)
        nb_ops_check_spans(NB_OP_REGRESS_NEIGHBORS, aa, 4, aa, 5, 0, 0, rr)
        try:
            nb_ops_check_spans(NB_OP_SEARCH, aa, 4, aa, 6, ff, 1, rr)
            raise Error("accepted")
        except e:
            require(String(e) == "neighbors ops: request spans differ from operation 9", "search span refusal")
        try:
            nb_ops_check_spans(NB_OP_CLASSIFY_NEIGHBORS, aa, 6, aa, 5, 0, 0, rr)
            raise Error("accepted")
        except e:
            require(String(e) == "neighbors ops: classify neighbors needs at least 6 words", "vote span refusal")
        try:
            nb_ops_check_spans(NB_OP_REGRESS_NEIGHBORS, aa, 4, aa, 5, ff, 1, rr)
            raise Error("accepted")
        except e:
            require(String(e) == "neighbors ops: request spans differ from operation 13", "mean span refusal")
        _ = words^
        _ = reals^
        _ = result^
    elif unit == 8:
        # the k-NN query request: its words, no handle and no scan by default; a fold needs a resident index
        var request = KnnQueryRequest(1, 2, 10, 4, 6, 3, 0, KnnDistWords(9, 1.5, 1))
        require(request.index == 1 and request.queries == 2 and request.n_index == 10 and request.n_queries == 4
                and request.n_features == 6 and request.k == 3 and request.query_tile == 0 and request.metric == 9
                and request.metric_arg == 1.5 and request.weights == 1, "k-NN request words")
        require(not request.resident and request.handle == 0 and not request.scan_index
                and not request.scan_queries and not request.device_status and request.fold.has_fold == 0,
                "k-NN request defaults")
        knn_check_fold_entry(request)
        request.fold = NbFold(1, 0, 10, 1)
        try:
            knn_check_fold_entry(request)
            raise Error("accepted")
        except e:
            require(String(e) == "RuntimeError: a resident k-NN fold needs the binding's resident entries",
                    "fold refusal wording")
        request.resident = True
        knn_check_fold_entry(request)
        var outcome = KnnQueryOutcome()
        require(outcome.used == 0 and outcome.index_status == 0 and outcome.status == 0
                and outcome.range_status == 0 and not outcome.ran, "k-NN outcome default")
    else:
        # the public int64 ids are the int32 ids, widened
        var ids = List[Int32](length=3, fill=Int32(1))
        ids[1] = Int32(-2)
        ids[2] = Int32(2147483647)
        var wide = List[Int64](length=3, fill=Int64(0))
        radius_widen_run(Int(ids.unsafe_ptr()), Int(wide.unsafe_ptr()), 3)
        require(wide[0] == 1 and wide[1] == -2 and wide[2] == 2147483647, "widened ids")
        radius_widen_run(Int(ids.unsafe_ptr()), Int(wide.unsafe_ptr()), 0)
        _ = ids^
        _ = wide^


def test_fixtures() raises:
    check_cases("fixture", NEIGHBORS_SPLIT_RUN_FIXTURES, fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
