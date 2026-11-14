# SPDX-License-Identifier: Apache-2.0
"""k-NN door RUN (`neighbors/neighbors_knn_run.mojo`) without Python.

Each fixture is one typed check, refusal or host helper: accepted and
refused values, the refusal wording, and the copy and scan answers.
Independent fixtures run across the host's cores.
"""
from core.host_parallel import host_parallelize
from neighbors.neighbors_knn_run import (
    KNN_DIST_LP, KNN_DIST_JS, KNN_DIST_COSINE, KNN_MINKOWSKI_OK, KNN_MINKOWSKI_NOT_POSITIVE,
    KNN_MINKOWSKI_SUBNORMAL, knn_refuse_rbc_vote, knn_refuse_distance_inner_product, knn_refuse_algorithm,
    knn_refuse_metric_name, knn_refuse_metric_unimplemented, knn_refuse_metric_unknown, knn_refuse_inert_p,
    knn_minkowski_status, knn_refuse_minkowski, knn_check_metric_input, knn_metric_reads_host_queries,
    knn_refuse_callable_weights, knn_refuse_weights, knn_weights_code, knn_check_y_rank, knn_check_y_rows,
    knn_check_y_ndim, knn_int32_low_fits, knn_check_int32_labels, knn_float_format, knn_labels_finite,
    KNN_I32_LABEL_RANGE_OLD, knn_labels_already_int32, knn_labels_skip_integral, knn_finite_status, knn_check_width, knn_proba_rows, knn_copy_proba_block,
)

comptime NEIGHBORS_KNN_RUN_FIXTURES = 22


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def refused(message: String, prefix: String) -> Bool:
    return message.startswith(prefix)


def fixture(unit: Int) raises:
    if unit == 0:
        try:
            knn_refuse_rbc_vote("KNeighborsClassifier")
            raise Error("accepted")
        except e:
            require(refused(String(e), "mojolearn KNeighborsClassifier: algorithm='rbc' is refused. The ball cover"),
                    "rbc vote refusal wording")
    elif unit == 1:
        try:
            knn_refuse_distance_inner_product("KNeighborsRegressor")
            raise Error("accepted")
        except e:
            require(refused(String(e), "mojolearn KNeighborsRegressor: weights='distance' is refused with "
                            + "metric='inner_product'"), "inner product refusal wording")
    elif unit == 2:
        try:
            knn_refuse_algorithm("'kd_tree'")
            raise Error("accepted")
        except e:
            require(refused(String(e), "mojolearn NearestNeighbors: algorithm='kd_tree' is refused. Three are "
                            + "accepted"), "algorithm refusal wording")
    elif unit == 3:
        try:
            knn_refuse_metric_name("NearestNeighbors", "<function f>")
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn NearestNeighbors: metric must be a name, got <function f>. A callable "
                    + "metric is not implemented: it would have to run inside a GPU kernel.", "metric name refusal")
    elif unit == 4:
        try:
            knn_refuse_metric_unimplemented("NearestNeighbors", "'haversine'", "cosine, euclidean")
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn NearestNeighbors: metric='haversine' is in cuML's "
                    + "VALID_METRICS['brute'] but is NOT IMPLEMENTED (neighbors/NOT_IMPLEMENTED.tsv). "
                    + "Implemented: cosine, euclidean", "unimplemented metric wording")
        try:
            knn_refuse_metric_unknown("NearestNeighbors", "'nope'", "cosine")
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn NearestNeighbors: unknown metric 'nope'. Implemented: cosine",
                    "unknown metric wording")
    elif unit == 5:
        try:
            knn_refuse_inert_p("NearestNeighbors", "'cosine'", "3")
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn NearestNeighbors: p is read only by metric='minkowski' (or 'lp'); "
                    + "metric='cosine' never reads it, so p=3 would change nothing. Leave p at its default (2) "
                    + "or ask for metric='minkowski'.", "inert p wording")
    elif unit == 6:
        require(knn_minkowski_status(2.0) == KNN_MINKOWSKI_OK, "p=2 accepted")
        require(knn_minkowski_status(0.5) == KNN_MINKOWSKI_OK, "p=0.5 accepted")
        require(knn_minkowski_status(0.0) == KNN_MINKOWSKI_NOT_POSITIVE, "p=0 refused")
        require(knn_minkowski_status(-1.0) == KNN_MINKOWSKI_NOT_POSITIVE, "p<0 refused")
        require(knn_minkowski_status(Float64.MAX * 2.0) == KNN_MINKOWSKI_NOT_POSITIVE, "p=inf refused")
        var nan = Float64(0.0) / Float64(0.0)
        require(knn_minkowski_status(nan) == KNN_MINKOWSKI_NOT_POSITIVE, "p=nan refused")
        require(knn_minkowski_status(1.0e-39) == KNN_MINKOWSKI_SUBNORMAL, "subnormal p refused")
        knn_refuse_minkowski("NearestNeighbors", KNN_MINKOWSKI_OK, "2")
    elif unit == 7:
        try:
            knn_refuse_minkowski("NearestNeighbors", KNN_MINKOWSKI_NOT_POSITIVE, "0")
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn NearestNeighbors: metric='minkowski' needs a finite p > 0, got 0. "
                    + "p = 0 makes 1/p infinite, p < 0 is not a metric, and p = infinity is "
                    + "metric='chebyshev'.", "minkowski p refusal wording")
        try:
            knn_refuse_minkowski("NearestNeighbors", KNN_MINKOWSKI_SUBNORMAL, "1e-39")
            raise Error("accepted")
        except e:
            require(refused(String(e), "mojolearn NearestNeighbors: metric='minkowski' p=1e-39 is subnormal in "
                            + "float32"), "subnormal refusal wording")
    elif unit == 8:
        var x: List[Float32] = [0.0, 1.0, 2.0, 3.0]
        knn_check_metric_input("NearestNeighbors", KNN_DIST_JS, Int(x.unsafe_ptr()), 4)
        var y: List[Float32] = [0.0, -1.0]
        # only Jensen-Shannon reads the sign; an empty block is never read
        knn_check_metric_input("NearestNeighbors", KNN_DIST_LP, Int(y.unsafe_ptr()), 2)
        knn_check_metric_input("NearestNeighbors", KNN_DIST_JS, Int(y.unsafe_ptr()), 0)
        try:
            knn_check_metric_input("NearestNeighbors", KNN_DIST_JS, Int(y.unsafe_ptr()), 2)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn NearestNeighbors: metric='jensenshannon' needs X >= 0 (it takes "
                    + "logarithms of the coordinates)", "jensenshannon refusal wording")
        _ = x
        _ = y
    elif unit == 9:
        require(knn_metric_reads_host_queries(KNN_DIST_JS), "js reads host queries")
        require(knn_metric_reads_host_queries(KNN_DIST_COSINE), "cosine reads host queries")
        require(not knn_metric_reads_host_queries(KNN_DIST_LP), "minkowski does not")
    elif unit == 10:
        try:
            knn_refuse_callable_weights("KNeighborsClassifier")
            raise Error("accepted")
        except e:
            require(refused(String(e), "mojolearn KNeighborsClassifier: weights=<callable> is NOT IMPLEMENTED."),
                    "callable weights wording")
        try:
            knn_refuse_weights("KNeighborsClassifier", "'gauss'")
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn KNeighborsClassifier: weights='gauss' is not a weighting; use "
                    + "'uniform' or 'distance'", "weights wording")
    elif unit == 11:
        require(knn_weights_code("uniform") == 0, "uniform")
        require(knn_weights_code("distance") == 1, "distance")
        require(knn_weights_code("Uniform") == -1 and knn_weights_code("") == -1, "other names")
    elif unit == 12:
        knn_check_y_rank(1)
        knn_check_y_rank(2)
        try:
            knn_check_y_rank(3)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn: y must be 1-D or 2-D, got 3-D", "rank wording")
        try:
            knn_check_y_rank(0)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn: y must be 1-D or 2-D, got 0-D", "scalar rank wording")
    elif unit == 13:
        knn_check_y_rows(5, 5)
        try:
            knn_check_y_rows(4, 5)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn: y has 4 rows, X has 5", "rows wording")
        knn_check_y_ndim(2, 2)
        try:
            knn_check_y_ndim(1, 2)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn: y must be 2-D, got 1-D", "ndim wording")
    elif unit == 14:
        require(knn_int32_low_fits(-(1 << 31)), "int32 minimum fits")
        require(not knn_int32_low_fits(-(1 << 31) - 1), "below int32 minimum")
        knn_check_int32_labels(-(1 << 31), (1 << 31) - 1)
        try:
            knn_check_int32_labels(0, 1 << 31)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn: y does not fit int32", "int32 high wording")
        try:
            knn_check_int32_labels(-(1 << 31) - 1, 0)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn: y does not fit int32", "int32 low wording")
        # wave2-knn-0844-gap: only little-endian int32 labels skip the range
        # check (none at all in the B arm); every other dtype keeps it.
        comptime if KNN_I32_LABEL_RANGE_OLD:
            require(not knn_labels_already_int32("<i4"), "B arm keeps the int32 range check")
        else:
            require(knn_labels_already_int32("<i4"), "int32 labels skip the range check")
        require(not knn_labels_already_int32("<i8") and not knn_labels_already_int32(">i4")
                and not knn_labels_already_int32("|i1") and not knn_labels_already_int32("<u4")
                and not knn_labels_already_int32("<f4"), "other label dtypes keep the range check")
    elif unit == 15:
        require(knn_float_format("<f4") and knn_float_format("f") and knn_float_format("=d")
                and knn_float_format("e") and knn_float_format("@g") and knn_float_format("|f2")
                and knn_float_format("<f8") and knn_float_format("!<f"), "float formats")
        require(not knn_float_format("<i4") and not knn_float_format("") and not knn_float_format("<f16")
                and not knn_float_format("ff") and not knn_float_format("<") and not knn_float_format("f4<"),
                "non-float formats")
    elif unit == 16:
        var f32: List[Float32] = [0.0, 1.5, -2.0]
        require(knn_labels_finite(Int(f32.unsafe_ptr()), 3, False), "finite float32 labels")
        var bad32: List[Float32] = [0.0, Float32.MAX * 2.0]
        require(not knn_labels_finite(Int(bad32.unsafe_ptr()), 2, False), "infinite float32 label")
        require(knn_labels_finite(Int(bad32.unsafe_ptr()), 0, False), "empty block is finite")
        var f64: List[Float64] = [1.0, 2.0]
        require(knn_labels_finite(Int(f64.unsafe_ptr()), 2, True), "finite float64 labels")
        var bad64: List[Float64] = [Float64(0.0) / Float64(0.0)]
        require(not knn_labels_finite(Int(bad64.unsafe_ptr()), 1, True), "nan float64 label")
        _ = f32
        _ = bad32
        _ = f64
        _ = bad64
    elif unit == 17:
        require(knn_labels_skip_integral("<i8", 4, 4), "integer block skips")
        require(knn_labels_skip_integral("<f4", 0, 0), "empty block skips")
        require(knn_labels_skip_integral("<f8", 6, 3), "2-D block skips")
        require(not knn_labels_skip_integral("<f4", 4, 4), "flat float32 block is tested")
        require(not knn_labels_skip_integral("<f8", 4, 4), "flat float64 block is tested")
    elif unit == 18:
        var x: List[Float32] = [1.0, 2.0, 3.0]
        require(knn_finite_status(Int(x.unsafe_ptr()), 3) == 0, "finite status zero")
        var bad: List[Float32] = [1.0, Float32(0.0) / Float32(0.0)]
        require(knn_finite_status(Int(bad.unsafe_ptr()), 2) != 0, "non-finite status")
        _ = x
        _ = bad
    elif unit == 19:
        knn_check_width(4, 4)
        try:
            knn_check_width(3, 4)
            raise Error("accepted")
        except e:
            require(String(e) == "mojolearn: X has 3 features, index has 4", "width wording")
    elif unit == 20:
        require(knn_proba_rows(12, 3) == 4 and knn_proba_rows(10, 5) == 2, "proba rows")
    else:
        # two outputs (2 and 3 classes) over two queries: the second block
        # starts at word 2 * 2
        var flat: List[Float32] = [0.5, 0.5, 1.0, 0.0, 0.1, 0.2, 0.7, 0.3, 0.3, 0.4]
        var block = List[Float32](length=6, fill=-1.0)
        knn_copy_proba_block(Int(block.unsafe_ptr()), Int(flat.unsafe_ptr()), 4, 2, 3)
        for i in range(6):
            require(block[i] == flat[4 + i], "proba block word " + String(i))
        _ = flat
        _ = block


def run_checks() raises:
    var failures = List[Int](length=NEIGHBORS_KNN_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL neighbors_knn_run fixture", unit, String(error))
    host_parallelize(task, NEIGHBORS_KNN_RUN_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = failures^
    require(failed == 0, "neighbors_knn_run failed " + String(failed) + " fixtures")
    print("PASS neighbors_knn_run:", NEIGHBORS_KNN_RUN_FIXTURES, "typed check, refusal and helper fixtures")
