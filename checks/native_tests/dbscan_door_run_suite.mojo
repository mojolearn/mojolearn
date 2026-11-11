# SPDX-License-Identifier: Apache-2.0
"""DBSCAN door RUN (`cluster/dbscan_door_run.mojo`) without Python.

Each fixture is one typed check or request builder: accepted and refused
values, the refusal wording, and the precomputed brute-arm override.
Independent fixtures run across the host's cores.
"""
from core.host_parallel import host_parallelize
from cluster.dbscan_door_run import (
    DBSCAN_METRIC_PRECOMPUTED, DBSCAN_EPS_NN_BRUTE_FORCE, dbscan_check_method, dbscan_check_prediction,
    dbscan_check_eps, dbscan_check_min_samples, dbscan_check_max_iterations, dbscan_check_budget,
    dbscan_check_x, dbscan_fit_request, dbscan_core_request,
)

comptime DBSCAN_DOOR_RUN_FIXTURES = 14


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def refused(message: String, prefix: String) -> Bool:
    return message.startswith(prefix)


def fixture(unit: Int) raises:
    if unit == 0:
        dbscan_check_eps(0.5)
        dbscan_check_min_samples(1)
        dbscan_check_max_iterations(0)
        dbscan_check_budget(0)
        dbscan_check_prediction(0, True)
        dbscan_check_prediction(DBSCAN_METRIC_PRECOMPUTED, False)
        dbscan_check_method(0, 0, "'euclidean'", "'brute'")
    elif unit == 1:
        try:
            dbscan_check_eps(0.0)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn DBSCAN eps must be positive", "eps refusal wording")
    elif unit == 2:
        try:
            dbscan_check_min_samples(0)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn DBSCAN min_samples must be at least 1", "min_samples refusal")
    elif unit == 3:
        try:
            dbscan_check_max_iterations(-1)
            raise Error("accepted")
        except e:
            require(refused(String(e), "mojolearn DBSCAN max_iterations must be None"), "max_iterations refusal")
    elif unit == 4:
        try:
            dbscan_check_budget(-1)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn DBSCAN max_mbytes_per_batch cannot be negative", "budget refusal")
    elif unit == 5:
        try:
            dbscan_check_prediction(DBSCAN_METRIC_PRECOMPUTED, True)
            raise Error("accepted")
        except e:
            require(refused(String(e), "mojolearn DBSCAN: prediction_data=True with metric='precomputed'"),
                    "precomputed prediction refusal")
    elif unit == 6:
        # rbc (1) with cosine (2) is outside the ball cover's scope
        try:
            dbscan_check_method(1, 2, "'cosine'", "'rbc'")
            raise Error("accepted")
        except e:
            require(refused(String(e), "mojolearn DBSCAN: metric='cosine' with algorithm='rbc' is refused."),
                    "method/metric refusal wording")
    elif unit == 7:
        var x: List[Float32] = [0.0, 1.0, 2.0, 3.0]
        dbscan_check_x(Int(x.unsafe_ptr()), 2, 2, False, 0)
        _ = x
    elif unit == 8:
        var x: List[Float32] = [0.0, Float32.MAX, 2.0, 3.0]
        try:
            dbscan_check_x(Int(x.unsafe_ptr()), 2, 2, False, 0)
            raise Error("accepted")
        except e:
            require(refused(String(e), "mojolearn DBSCAN: X contains a NaN or an infinity"), "finite refusal")
        _ = x
    elif unit == 9:
        # an X the public door already admitted is not scanned again
        var x: List[Float32] = [0.0, Float32.MAX, 2.0, 3.0]
        dbscan_check_x(Int(x.unsafe_ptr()), 2, 2, True, 0)
        _ = x
    elif unit == 10:
        var x: List[Float32] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        try:
            dbscan_check_x(Int(x.unsafe_ptr()), 2, 3, True, DBSCAN_METRIC_PRECOMPUTED)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn DBSCAN: metric='precomputed' needs a square distance matrix, got shape (2, 3)",
                    "precomputed shape refusal wording")
        _ = x
    elif unit == 11:
        var x: List[Float32] = [0.0, -1.0, 1.0, 0.0]
        try:
            dbscan_check_x(Int(x.unsafe_ptr()), 2, 2, True, DBSCAN_METRIC_PRECOMPUTED)
            raise Error("accepted")
        except e:
            require(refused(String(e), "mojolearn DBSCAN: metric='precomputed' refuses a negative distance"),
                    "precomputed sign refusal")
        _ = x
    elif unit == 12:
        var r = dbscan_fit_request(1, 2, 3, 4, 5, 6, 7, 8, 9, 1, DBSCAN_METRIC_PRECOMPUTED, 0.25)
        require(r.method == DBSCAN_EPS_NN_BRUTE_FORCE, "precomputed selects the brute arm")
        require(r.x == 1 and r.labels == 2 and r.weights == 3 and r.core == 4, "fit request addresses")
        require(r.rows == 5 and r.features == 6 and r.min_samples == 7 and r.budget_mb == 8
                and r.max_iter == 9 and r.metric == DBSCAN_METRIC_PRECOMPUTED and r.eps == 0.25,
                "fit request order")
    else:
        var r = dbscan_fit_request(1, 2, 0, 4, 5, 6, 7, 0, 0, 1, 0, 0.5)
        require(r.method == 1, "a non-precomputed metric keeps the named method")
        var c = dbscan_core_request(1, 2, 4, 5, 6)
        require(c.x == 1 and c.labels == 2 and c.core == 4 and c.rows == 5 and c.features == 6,
                "core request fields")


def run_checks() raises:
    var failures = List[Int](length=DBSCAN_DOOR_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL dbscan_door_run fixture", unit, String(error))
    host_parallelize(task, DBSCAN_DOOR_RUN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "dbscan_door_run failed " + String(failed) + " fixtures")
    print("PASS dbscan_door_run:", DBSCAN_DOOR_RUN_FIXTURES, "typed check and request fixtures")
