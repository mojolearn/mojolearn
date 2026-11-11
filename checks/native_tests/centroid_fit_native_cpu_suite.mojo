# SPDX-License-Identifier: Apache-2.0
"""Actual MeanShift/AP entries, checked against independent small arithmetic.

CPU fixture units run concurrently. No repeated-output hashes, timing, Python
or duplicate numerical implementation. Broad Python workload tests remain.
"""
from std.math import abs
from core.host_parallel import host_parallelize
from core.cluster_centroid_fit_api import CentroidFitDriver, CentroidFitResultSlot, MeanShiftFitRequest, AffinityFitRequest
from core.kmeans_variants_api import VariantsSpan
from x_cluster.centroid_fit_host import cluster_centroid_fit_host_v1

comptime CENTROID_FIT_NATIVE_FIXTURES = 14


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def span(values: List[Float32]) -> VariantsSpan:
    return VariantsSpan(Int(values.unsafe_ptr()) if len(values) else 0, len(values))


def meanshift_case(driver: CentroidFitDriver, unit: Int) raises:
    var x: List[Float32] = [0, 0, 10, 10]
    var seeds = List[Float32]()
    var all_points = True
    var binned = unit == 2 or unit == 3
    if unit == 1:
        seeds.append(0)
        all_points = False
    var result = CentroidFitResultSlot()
    var expected_failure = unit == 3 or unit == 4
    var refused = False
    try:
        driver.meanshift(MeanShiftFitRequest(span(x), span(seeds), 4, 1, len(seeds), all_points,
            10, binned, 3 if unit == 3 else 1, Float64(0) if unit == 4 else Float64(1)), result)
    except problem:
        require(expected_failure, "unexpected MeanShift refusal: " + String(problem))
        if unit == 3:
            require(String(problem).find("No point was within bandwidth=") >= 0, "bin population refusal")
        else:
            require(String(problem).find("bandwidth needs to be greater than zero") >= 0, "zero estimated bandwidth refusal")
        refused = True
    require(refused == expected_failure, "MeanShift refusal was lost")
    if not expected_failure:
        require(result.storage[0].bandwidth == 1 and len(result.storage[0].labels) == 4, "bandwidth/label result fields")
        if unit == 1:
            require(result.storage[0].clusters == 1 and result.storage[0].centers[0] == 0, "supplied seed neighborhood")
            require(result.storage[0].labels[0] == 0 and result.storage[0].labels[1] == 0
                and result.storage[0].labels[2] == -1 and result.storage[0].labels[3] == -1, "cluster_all false preserves orphan labels")
        else:
            require(result.storage[0].clusters == 2 and result.storage[0].centers[0] == 10
                and result.storage[0].centers[1] == 0, "equal-density centers sort by decreasing coordinate")
            require(result.storage[0].labels[0] == 1 and result.storage[0].labels[1] == 1
                and result.storage[0].labels[2] == 0 and result.storage[0].labels[3] == 0, "nearest labels use sorted centers")
    _ = x^
    _ = seeds^


def affinity_equal_case(driver: CentroidFitDriver, unit: Int) raises:
    var precomputed = unit != 4
    var x: List[Float32] = [2, 2, 2, 2]
    if not precomputed:
        x = [3, 3]
    var preference: List[Float32] = [3, 3]
    var mode = 0 if unit == 3 else (2 if unit == 2 else 1)
    var scalar = Float64(3) if unit == 0 else Float64(1)
    if not precomputed:
        scalar = -1
    var result = CentroidFitResultSlot()
    driver.affinity(AffinityFitRequest(span(x), span(preference), 2, 2 if precomputed else 1,
        precomputed, mode, 10, 2, 7, 0.5, scalar), result)
    var separate = unit == 0 or unit == 2
    require(result.storage[0].iterations == 0 and len(result.storage[0].message_diagonal) == 0,
        "equal similarities shortcut retains empty message diagnostic")
    require(len(result.storage[0].center_indices) == (2 if separate else 1)
        and result.storage[0].center_indices[0] == 0, "equal preference exemplar choice")
    require(result.storage[0].labels[0] == 0 and result.storage[0].labels[1] == Int32(1 if separate else 0),
        "equal preference label choice")
    require(len(result.storage[0].affinity) == 4 and result.storage[0].affinity[1] == (Float32(2) if precomputed else Float32(0)),
        "public affinity retains original similarities before preference/noise")
    if precomputed:
        require(len(result.storage[0].centers) == 0, "precomputed result never publishes exemplar rows")
    else:
        require(len(result.storage[0].centers) == 1 and result.storage[0].centers[0] == 3, "euclidean exemplar row gather")
    _ = x^
    _ = preference^


def affinity_step_case(driver: CentroidFitDriver, unit: Int) raises:
    var precomputed = unit < 2
    # Three unequal off-diagonal distances avoid the equal-similarity shortcut.
    var x: List[Float32] = [0, -4, -36, -4, 0, -16, -36, -16, 0]
    if not precomputed:
        x = [0, 2, 6]
    var preference: List[Float32] = [-1, -1, -1]
    var mode = 0 if unit == 3 else (2 if unit == 1 else 1)
    var result = CentroidFitResultSlot()
    driver.affinity(AffinityFitRequest(span(x), span(preference), 3, 3 if precomputed else 1,
        precomputed, mode, 1, 1, 13, 0.5, -1), result)
    # Starting with A=R=0: R_ii=(p - max_off_diagonal_similarity)/2.
    # Scalar/vector p=-1 gives [1.5,1.5,7.5]; median(S)=-4 gives [0,0,6].
    # Off-diagonal R <= 0, hence A_ii=0. Seeded epsilon perturbations only
    # affect the tolerated last bits (and the two zero-message median ties).
    var first_r = Float32(0) if mode == 0 else Float32(1.5)
    var last_r = Float32(6) if mode == 0 else Float32(7.5)
    require(result.storage[0].iterations == 1 and len(result.storage[0].message_diagonal) == 6, "single message iteration/public diagnostic layout")
    require(abs(result.storage[0].message_diagonal[0]) < 0.0001 and abs(result.storage[0].message_diagonal[1]) < 0.0001
        and abs(result.storage[0].message_diagonal[2]) < 0.0001
        and abs(result.storage[0].message_diagonal[3] - first_r) < 0.0001
        and abs(result.storage[0].message_diagonal[4] - first_r) < 0.0001
        and abs(result.storage[0].message_diagonal[5] - last_r) < 0.0001, "independent availability/responsibility arithmetic")
    require(result.storage[0].affinity[0] == 0 and result.storage[0].affinity[1] == -4
        and result.storage[0].affinity[2] == -36 and result.storage[0].affinity[5] == -16
        and result.storage[0].affinity[8] == 0, "original similarity matrix copy")
    if mode != 0:
        require(len(result.storage[0].center_indices) == 3 and result.storage[0].center_indices[0] == 0
            and result.storage[0].center_indices[1] == 1 and result.storage[0].center_indices[2] == 2
            and result.storage[0].labels[0] == 0 and result.storage[0].labels[1] == 1
            and result.storage[0].labels[2] == 2, "positive diagonal messages choose all exemplars")
        if not precomputed:
            require(result.storage[0].centers[0] == 0 and result.storage[0].centers[1] == 2
                and result.storage[0].centers[2] == 6, "ordered exemplar row gather")
    _ = x^
    _ = preference^


def fixture(driver: CentroidFitDriver, unit: Int) raises:
    if unit < 5:
        meanshift_case(driver, unit)
    elif unit < 10:
        affinity_equal_case(driver, unit - 5)
    else:
        affinity_step_case(driver, unit - 10)


def run_checks() raises:
    var failures = List[Int](length=CENTROID_FIT_NATIVE_FIXTURES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            fixture(CentroidFitDriver(cluster_centroid_fit_host_v1), unit)
        except problem:
            flags[unit] = 1
            print("FAIL centroid_fit_native_cpu fixture", unit, String(problem))
    host_parallelize(run, CENTROID_FIT_NATIVE_FIXTURES)
    var failed = 0
    for value in failures:  # metadata: independent fixture statuses
        failed += value
    require(failed == 0, "centroid fit CPU failed " + String(failed) + " fixtures")
    print("PASS centroid_fit_native_cpu:", CENTROID_FIT_NATIVE_FIXTURES, "actual provider fixtures")
