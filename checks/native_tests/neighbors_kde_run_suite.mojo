# SPDX-License-Identifier: Apache-2.0
"""KernelDensity door RUN (`neighbors/neighbors_kde_run.mojo`) without Python.

Each fixture is one typed check, host copy or request builder: accepted and
refused values, the refusal wording, buffer contents and request fields.
Independent fixtures run across the host's cores.
"""
from core.host_parallel import host_parallelize
from core.kde_native_api import KDE_CAP_RESIDENT, KDE_CAP_IMMUTABLE
from neighbors.neighbors_kde_run import (
    KDE_DEFAULT_LEAF_SIZE, kde_refuse_bandwidth_name, kde_bandwidth_positive, kde_refuse_bandwidth,
    kde_refuse_algorithm, kde_refuse_tree_knobs, kde_refuse_names, kde_check_metric_params_count,
    kde_minkowski_supported, kde_refuse_minkowski, kde_refuse_unfitted, kde_refuse_features,
    kde_resident_capable, kde_immutable_capable, kde_copy_f32, kde_copy_finite_f32, kde_query_finite_status,
    kde_fit_key, kde_prepare_request, kde_query_request, kde_resident_query,
)

comptime NEIGHBORS_KDE_RUN_FIXTURES = 16


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def refused(message: String, prefix: String) -> Bool:
    return message.startswith(prefix)


def fixture(unit: Int) raises:
    if unit == 0:
        require(kde_bandwidth_positive(0.5), "positive bandwidth")
        require(not kde_bandwidth_positive(0.0) and not kde_bandwidth_positive(-1.0), "non-positive bandwidth")
        var nan = Float64(0.0) / Float64(0.0)
        require(not kde_bandwidth_positive(nan), "NaN bandwidth")
        require(KDE_DEFAULT_LEAF_SIZE == 40, "default leaf size")
    elif unit == 1:
        try:
            kde_refuse_bandwidth_name("scott")
            raise Error("accepted")
        except e:
            require(refused(String(e), "NotImplementedError: mojolearn KernelDensity: bandwidth='scott' is refused;"),
                    "bandwidth name refusal")
    elif unit == 2:
        try:
            kde_refuse_bandwidth("-1.0")
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn KernelDensity: bandwidth must be positive, got -1.0", "bandwidth refusal")
    elif unit == 3:
        try:
            kde_refuse_algorithm("'kd_tree'")
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn KernelDensity: algorithm='kd_tree' is refused; this is "
                    + "cuML's brute-force KDE, there is no tree to choose", "algorithm refusal")
    elif unit == 4:
        try:
            kde_refuse_tree_knobs()
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn KernelDensity: atol/rtol/breadth_first/leaf_size "
                    + "are tree-traversal knobs; this is brute force (refused)", "knobs refusal")
        try:
            kde_refuse_names()
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn KernelDensity: kernel and metric are names", "names refusal")
    elif unit == 5:
        kde_check_metric_params_count(1)
        try:
            kde_check_metric_params_count(2)
            raise Error("accepted")
        except e:
            require(refused(String(e), "mojolearn KernelDensity: only metrics with a single argument are supported"),
                    "metric_params count refusal")
    elif unit == 6:
        require(kde_minkowski_supported(2.0), "p=2 served")
        require(not kde_minkowski_supported(1.0), "p=1 refused")
        try:
            kde_refuse_minkowski("{'p': 1.0}", "1.0")
            raise Error("accepted")
        except e:
            require(refused(String(e), "NotImplementedError: mojolearn KernelDensity: metric_params={'p': 1.0} asks for Minkowski p=1.0."),
                    "minkowski refusal wording")
    elif unit == 7:
        try:
            kde_refuse_unfitted()
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn KernelDensity: call fit() first", "unfitted refusal")
        try:
            kde_refuse_features("3", "2")
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn KernelDensity: X has 3 features, fit saw 2", "features refusal")
    elif unit == 8:
        require(kde_resident_capable(KDE_CAP_RESIDENT) and not kde_resident_capable(KDE_CAP_IMMUTABLE),
                "resident capability bit")
        require(kde_immutable_capable(KDE_CAP_IMMUTABLE | KDE_CAP_RESIDENT) and not kde_immutable_capable(0),
                "immutable capability bit")
    elif unit == 9:
        var src: List[Float32] = [1.0, 2.0, -3.0, 4.5]
        var dst = List[Float32](length=4, fill=0)
        kde_copy_f32(Int(dst.unsafe_ptr()), Int(src.unsafe_ptr()), 4)
        _ = src
        require(dst[0] == 1.0 and dst[1] == 2.0 and dst[2] == -3.0 and dst[3] == 4.5, "float32 copy")
    elif unit == 10:
        var src: List[Float32] = [1.0, 2.0, -3.0, 4.5, 0.0, 6.0]
        var dst = List[Float32](length=6, fill=0)
        var ok = kde_copy_finite_f32(Int(src.unsafe_ptr()), Int(dst.unsafe_ptr()), 2, 3)
        _ = src
        require(ok, "finite copy verdict")
        require(dst[0] == 1.0 and dst[2] == -3.0 and dst[3] == 4.5 and dst[5] == 6.0, "finite copy contents")
    elif unit == 11:
        var src: List[Float32] = [1.0, Float32.MAX * 2.0, 3.0, 4.0]
        var dst = List[Float32](length=4, fill=0)
        var ok = kde_copy_finite_f32(Int(src.unsafe_ptr()), Int(dst.unsafe_ptr()), 2, 2)
        _ = src
        _ = dst
        require(not ok, "an infinity is reported")
    elif unit == 12:
        var good: List[Float32] = [1.0, 2.0, 3.0]
        require(kde_query_finite_status(Int(good.unsafe_ptr()), 3) == 0, "finite query")
        _ = good
        var bad: List[Float32] = [1.0, Float32.MAX * 2.0, 3.0]
        require(kde_query_finite_status(Int(bad.unsafe_ptr()), 3) != 0, "non-finite query")
        _ = bad
    elif unit == 13:
        var key = kde_fit_key(11, 5, 3, 0, "gaussian", "euclidean", 0.5)
        require(key.train == 11 and key.rows == 5 and key.features == 3 and key.weights == 0
                and key.kernel == "gaussian" and key.metric == "euclidean" and key.bandwidth == 0.5, "fit key fields")
        var request = kde_prepare_request(key, False)
        require(request.train == 11 and request.weights == 0 and request.rows == 5 and request.features == 3
                and request.bandwidth == 0.5 and not request.has_weights and request.finite_tested
                and request.kernel == "gaussian" and request.metric == "euclidean", "prepare request fields")
    elif unit == 14:
        var r = kde_query_request(1, 2, 3, 4, 5, 6, 7, 0.25, True, False, "tophat", "manhattan")
        require(r.train == 1 and r.query == 2 and r.weights == 3 and r.dst == 4 and r.train_rows == 5
                and r.query_rows == 6 and r.features == 7 and r.bandwidth == 0.25 and r.has_weights
                and not r.want_total and r.kernel == "tophat" and r.metric == "manhattan", "query request order")
    else:
        var r = kde_resident_query(9, 1, 2, 3, 4, 5, 0.75, True, "epanechnikov", "euclidean")
        require(r.handle == 9 and r.query == 1 and r.dst == 2 and r.status == 3 and r.rows == 4
                and r.features == 5 and r.bandwidth == 0.75 and r.want_total and r.kernel == "epanechnikov"
                and r.metric == "euclidean", "resident query order")


def run_checks() raises:
    var failures = List[Int](length=NEIGHBORS_KDE_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL neighbors_kde_run fixture", unit, String(error))
    host_parallelize(task, NEIGHBORS_KDE_RUN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "neighbors_kde_run failed " + String(failed) + " fixtures")
    print("PASS neighbors_kde_run:", NEIGHBORS_KDE_RUN_FIXTURES, "typed check, copy and request fixtures")
