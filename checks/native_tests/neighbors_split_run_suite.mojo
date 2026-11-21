# SPDX-License-Identifier: Apache-2.0
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
from core.host_parallel import host_parallelize
from core.neighbors_entry_table import (
    NB_OP_RADIUS_COUNT, NB_OP_RADIUS_FILL, NB_OPERATION_COUNT, nb_ops_check_spans,
)
from neighbors.neighbors_local_run import LOF_RESULT_LABELS, LOF_AUTO_OFFSET
from neighbors.nearest_centroid_run import (
    NC_PRIORS_EMPIRICAL, NC_PRIORS_UNIFORM, NC_PRIORS_GIVEN,
    NcFitRequest, NcFitOutcome, NcQueryRequest, nc_fit_priors_run, nc_proba_wide,
)
from neighbors.lof_run import LofFitRequest, LofFitOutcome, LofQueryRequest
from neighbors.radius_run import (
    RadiusRequest, RadiusCount, radius_check, radius_check_options, radius_check_features, radius_widen_run,
)

comptime NEIGHBORS_SPLIT_RUN_FIXTURES = 7


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _near(a: Float64, b: Float64) -> Bool:
    return abs(a - b) < 1e-12


def fixture(unit: Int) raises:
    if unit == 0:
        # radius: accepted values pass, refusals keep their wording
        radius_check(0.5)
        radius_check_options(True, True)
        radius_check_options(False, False)
        radius_check_features(3, 3)
        try:
            radius_check(0.0)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: radius must be positive and finite", "radius refusal wording")
        try:
            radius_check_options(True, False)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: sort_results=True requires return_distance=True",
                    "sort refusal wording")
        try:
            radius_check_features(2, 3)
            raise Error("accepted")
        except e:
            require(String(e) == "ValueError: input feature count differs from fit", "feature refusal wording")
    elif unit == 1:
        # the two radius operations: their spans are accepted, any other refused
        require(NB_OP_RADIUS_COUNT == 7 and NB_OP_RADIUS_FILL == 8 and NB_OPERATION_COUNT == 9, "operation ids")
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


def run_checks() raises:
    var failures = List[Int](length=NEIGHBORS_SPLIT_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL neighbors_split_run fixture", unit, String(error))
    host_parallelize(task, NEIGHBORS_SPLIT_RUN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "neighbors_split_run failed " + String(failed) + " fixtures")
    print("PASS neighbors_split_run:", NEIGHBORS_SPLIT_RUN_FIXTURES, "typed request, check and buffer fixtures")
