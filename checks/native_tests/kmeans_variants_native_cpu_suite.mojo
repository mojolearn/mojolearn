# SPDX-License-Identifier: Apache-2.0
"""Actual variant provider fixtures with small, independent arithmetic oracles.

No Python, timing, seeded-output hashes or alternate numerical implementation.
Independent cases run in parallel; the two partial updates within one case
are necessarily ordered because the second consumes the first model state.
"""
from core.host_parallel import host_parallelize
from core.kmeans_variants_api import (
    KMeansVariantsDriver, VariantsSpan, VariantsResultSlot, MiniBatchFitRequest,
    MiniBatchPartialRequest, BisectFitRequest, BisectPredictRequest, VariantsQueryRequest,
)
from x_cluster.variants_host import kmeans_variants_host_v1

comptime KMEANS_VARIANTS_NATIVE_CPU_FIXTURES = 10


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def span(values: List[Float32]) -> VariantsSpan:
    return VariantsSpan(Int(values.unsafe_ptr()), len(values))


def fit_case(weighted: Bool) raises:
    # Each observation equals its supplied center. Random row selection can
    # change counts, but never these centers, predictions or zero objective.
    var x: List[Float32] = [1, 11]
    var auxiliary: List[Float32] = [1, 11]
    if weighted:
        auxiliary.append(1)
        auxiliary.append(3)
    var driver = KMeansVariantsDriver(kmeans_variants_host_v1)
    var output = VariantsResultSlot()
    driver.minibatch_fit(MiniBatchFitRequest(span(x), span(auxiliary),
        2, 1, 2, 1, 2, 10, 2, 1, True, 7, False, weighted, False, 0, 0), output)
    require(output.storage[0].centers[0] == 1 and output.storage[0].centers[1] == 11,
            "exact initial centers remain fixed")
    require(output.storage[0].labels[0] == 0 and output.storage[0].labels[1] == 1
            and output.storage[0].inertia == 0, "fit labels and independent zero objective")
    require(output.storage[0].steps == 1 and output.storage[0].iterations == 1, "one full mini batch")
    _ = x^
    _ = auxiliary^


def partial_case(kind: Int) raises:
    var x: List[Float32] = [1, 11]
    var auxiliary: List[Float32] = [0, 10]
    var weighted = kind != 0
    if weighted:
        x = [0, 2, 10, 14]
        auxiliary = [0, 10, 1, 3, 2, 2]
    var n = len(x)
    var driver = KMeansVariantsDriver(kmeans_variants_host_v1)
    var output = VariantsResultSlot()
    driver.minibatch_partial(MiniBatchPartialRequest(span(x), span(auxiliary), n, 1, 2,
        True, 2, n, n, 0, 0, 7, weighted, 0), output)
    if weighted:
        # Weighted means: (0+3*2)/4 = 1.5, (2*10+2*14)/4 = 12.
        # Objective: 1*1.5^2+3*0.5^2+2*2^2+2*2^2 = 19.
        require(output.storage[0].centers[0] == 1.5 and output.storage[0].centers[1] == 12,
                "weighted partial means")
        require(output.storage[0].counts[0] == 4 and output.storage[0].counts[1] == 4,
                "weighted partial counts")
        require(output.storage[0].inertia == 19, "weighted partial objective")
        require(output.storage[0].labels[0] == 0 and output.storage[0].labels[1] == 0
                and output.storage[0].labels[2] == 1 and output.storage[0].labels[3] == 1,
                "weighted partial partition")
        if kind == 2:
            var next_auxiliary: List[Float32] = [output.storage[0].centers[0], output.storage[0].centers[1],
                output.storage[0].counts[0], output.storage[0].counts[1], 1, 3, 2, 2]
            var request = MiniBatchPartialRequest(span(x), span(next_auxiliary), n, 1, 2,
                False, 2, n, n, output.storage[0].since_reassignment,
                output.storage[0].state_hi, output.storage[0].state_lo, True, 0)
            driver.minibatch_partial(request, output)
            require(output.storage[0].counts[0] == 8 and output.storage[0].counts[1] == 8,
                    "second partial step consumes retained counts")
            require(output.storage[0].centers[0] == 1.5 and output.storage[0].centers[1] == 12
                    and output.storage[0].inertia == 19, "second identical batch preserves weighted means/objective")
            _ = next_auxiliary^
    else:
        require(output.storage[0].centers[0] == 1 and output.storage[0].centers[1] == 11
                and output.storage[0].counts[0] == 1 and output.storage[0].counts[1] == 1,
                "unweighted first step means and counts")
        require(output.storage[0].labels[0] == 0 and output.storage[0].labels[1] == 1
                and output.storage[0].inertia == 0, "unweighted first step labels/objective")
    _ = x^
    _ = auxiliary^


def bisect_case(random_init: Bool) raises:
    var x: List[Float32] = [0, 1, 10, 11]
    var driver = KMeansVariantsDriver(kmeans_variants_host_v1)
    var output = VariantsResultSlot()
    driver.bisect_fit(BisectFitRequest(span(x), VariantsSpan(0, 0),
        4, 1, 2, 1, random_init, 100, 7, random_init, 0), output)
    var first_low = output.storage[0].centers[0] == 0.5
    require((first_low and output.storage[0].centers[1] == 10.5)
            or (output.storage[0].centers[0] == 10.5 and output.storage[0].centers[1] == 0.5),
            "bisect fitted means, independent of seed-dependent label order")
    var low = Int32(0 if first_low else 1)
    var high = Int32(1 if first_low else 0)
    require(output.storage[0].labels[0] == low and output.storage[0].labels[1] == low
            and output.storage[0].labels[2] == high and output.storage[0].labels[3] == high,
            "bisect labels agree with the separated partition")
    require(output.storage[0].inertia == 1, "four deviations of magnitude one half")
    require(len(output.storage[0].tree_centers) == 3 and len(output.storage[0].tree_nodes) == 9,
            "one split owns root plus two leaves")
    require(output.storage[0].tree_centers[0] == 0 and output.storage[0].tree_nodes[2] == -1,
            "existing centered-tree root placeholder and non-leaf label")
    var predicted = VariantsResultSlot()
    driver.bisect_predict(BisectPredictRequest(span(x), span(output.storage[0].tree_centers),
        VariantsSpan(Int(output.storage[0].tree_nodes.unsafe_ptr()), len(output.storage[0].tree_nodes)),
        4, 1), predicted)
    require(predicted.storage[0].labels[0] == low and predicted.storage[0].labels[1] == low
            and predicted.storage[0].labels[2] == high and predicted.storage[0].labels[3] == high,
            "native Int32 tree query preserves fit partition")
    _ = output^
    _ = x^


def query_case(kind: Int) raises:
    var x: List[Float32] = [0, 5, 10]
    var centers: List[Float32] = [0, 10]
    var driver = KMeansVariantsDriver(kmeans_variants_host_v1)
    var output = VariantsResultSlot()
    driver.query(VariantsQueryRequest(span(x), span(centers), 3, 2, 1, kind == 1), kind == 2, output)
    if kind == 2:
        require(len(output.storage[0].distances) == 6 and output.storage[0].distances[0] == 0
                and output.storage[0].distances[1] == 10 and output.storage[0].distances[2] == 5
                and output.storage[0].distances[3] == 5 and output.storage[0].distances[4] == 10
                and output.storage[0].distances[5] == 0, "transform remains Euclidean, not squared")
    else:
        require(output.storage[0].labels[0] == 0 and output.storage[0].labels[1] == 0
                and output.storage[0].labels[2] == 1, "nearest ties retain first center")
        require(output.storage[0].distances[0] == 0 and output.storage[0].distances[1] == 25
                and output.storage[0].distances[2] == 0, "nearest retains squared distances")
        require(output.storage[0].inertia == (Float64(25) if kind == 1 else Float64(0)),
                "optional score sum is preserved")
    _ = x^
    _ = centers^


def fixture(unit: Int) raises:
    if unit < 2:
        fit_case(unit == 1)
    elif unit < 5:
        partial_case(unit - 2)
    elif unit < 7:
        bisect_case(unit == 6)
    else:
        query_case(unit - 7)


def run_checks() raises:
    var failures = List[Int](length=KMEANS_VARIANTS_NATIVE_CPU_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def run(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except problem:
            fp[unit] = 1
            print("FAIL kmeans_variants_native_cpu fixture", unit, String(problem))
    host_parallelize(run, KMEANS_VARIANTS_NATIVE_CPU_FIXTURES)
    var failed = 0
    for value in failures:  # metadata: fixture summaries
        failed += value
    require(failed == 0, "native KMeans variants CPU failed " + String(failed) + " fixtures")
    print("PASS kmeans_variants_native_cpu:", KMEANS_VARIANTS_NATIVE_CPU_FIXTURES, "actual provider fixtures")
