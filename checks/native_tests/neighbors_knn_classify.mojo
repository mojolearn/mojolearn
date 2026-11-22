# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The k-NN search and classifier bodies of the host column on the original's literals (lane py2mojo-D6, 2026-11-22;
ported from python/mojolearn/tests/test_peer1_provider_boundary.py
`test_neighbors_return_completed_distance_label_and_probability_results`).

The bodies are the ones the host image's NB_OP_SEARCH and NB_OP_CLASSIFY run (neighbors/ops_entry_host.mojo:
core/core_host_impl.mojo `knn_search_host`, `knn_classify_host`), with the metric words the door resolves
(neighbors/neighbors_knn_run.mojo `knn_metric_code`: 'euclidean' for NearestNeighbors, the default 'minkowski' with
p = 2 for KNeighborsClassifier). Inputs: X [[0], [2], [5]], queries [[0.5], [4.5]].

    unit 0  NearestNeighbors(n_neighbors=2, metric='euclidean').kneighbors: indices [[0, 1], [2, 1]] exactly,
            distances [[0.5, 1.5], [0.5, 2.5]] to atol 1e-6
    unit 1  KNeighborsClassifier(n_neighbors=1) on y ['cold', 'warm', 'hot'] (the codes of the sorted classes
            ['cold', 'hot', 'warm']: [0, 2, 1]): predict [0, 1] ('cold', 'hot'), the class set [0, 1, 2]
    unit 2  the same, predict_proba [[1, 0, 0], [0, 1, 0]] exactly

What stays in Python: the string labels and their encoding, the returned arrays, and `score` (accuracy of predict, a
public composition). Independent units run across the host cores.
"""
from core.hostptr import f32_ptr, i32_ptr, u32_ptr
from core.core_host_impl import knn_search_host, knn_classify_host
from core.knn_host_predict import KNN_HOST_WEIGHTS_UNIFORM
from neighbors.neighbors_knn_run import knn_metric_code
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases

comptime KNN_CLASSIFY_UNITS = 3


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def knn_unit(unit: Int) raises:
    var index: List[Float32] = [0.0, 2.0, 5.0]
    var queries: List[Float32] = [0.5, 4.5]
    if unit == 0:
        var dist = List[Float32](length=4, fill=-1.0)
        var idx = List[UInt32](length=4, fill=UInt32(99))
        knn_search_host(Int(index.unsafe_ptr()), Int(queries.unsafe_ptr()), f32_ptr(Int(dist.unsafe_ptr())),
                        u32_ptr(Int(idx.unsafe_ptr())), 3, 2, 1, 2, True, knn_metric_code("euclidean"), Float32(2.0))
        var want_idx: List[UInt32] = [0, 1, 2, 1]
        var want_dist: List[Float64] = [0.5, 1.5, 0.5, 2.5]
        for i in range(4):  # small-loop: four neighbour slots
            require(idx[i] == want_idx[i], "kneighbors index slot " + String(i) + ": " + String(idx[i])
                    + " expected " + String(want_idx[i]))
            require(abs(Float64(dist[i]) - want_dist[i]) <= 1e-6, "kneighbors distance slot " + String(i) + ": "
                    + String(dist[i]) + " expected " + String(want_dist[i]))
        _ = dist^
        _ = idx^
    else:
        var proba = unit == 2
        var y: List[Int32] = [0, 2, 1]
        var labels = List[Int32](length=2, fill=-7)
        var probs = List[Float32](length=6, fill=-1.0)
        var uniq = List[Int32](length=3, fill=-7)
        var n_classes: List[Int] = [3]
        knn_classify_host(Int(index.unsafe_ptr()), Int(queries.unsafe_ptr()), Int(y.unsafe_ptr()),
                          i32_ptr(Int(labels.unsafe_ptr())), f32_ptr(Int(probs.unsafe_ptr())),
                          i32_ptr(Int(uniq.unsafe_ptr())), 3, 2, 1, 1, 1, proba, n_classes,
                          knn_metric_code("minkowski"), Float32(2.0), KNN_HOST_WEIGHTS_UNIFORM)
        for c in range(3):  # small-loop: the class set
            require(uniq[c] == Int32(c), "class set word " + String(c) + ": " + String(uniq[c]))
        if proba:
            var want: List[Float32] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            for i in range(6):  # small-loop: two proba rows
                require(probs[i] == want[i], "predict_proba word " + String(i) + ": " + String(probs[i])
                        + " expected " + String(want[i]))
        else:
            require(labels[0] == 0 and labels[1] == 1, "predict: [" + String(labels[0]) + ", " + String(labels[1])
                    + "] expected [0, 1] ('cold', 'hot')")
        _ = y^
        _ = labels^
        _ = probs^
        _ = uniq^
    _ = index^
    _ = queries^


def test_knn_classify() raises:
    check_cases("knn_classify", KNN_CLASSIFY_UNITS, knn_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
