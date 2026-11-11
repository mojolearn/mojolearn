# SPDX-License-Identifier: Apache-2.0
"""Independent OPTICS graph/extraction fixtures over the actual native entry.

Fixtures run independently on CPU. The 180x3 case has three disconnected
60-row duplicate groups with an analytic oracle; the original random workload
remains in Python until separately accepted replacement coverage exists.
"""
from std.math import abs, isinf
from core.host_parallel import host_parallelize
from core.optics_fit_api import OpticsFitDriver, OpticsFitRequest, OpticsFitResultSlot, OpticsMetric, OPTICS_FIT_RUN
from core.kmeans_variants_api import VariantsSpan
from core.native_descriptor_fixed_call import native_descriptor_call_fixed
from x_cluster.optics_fit_host import optics_fit_host_v1

comptime OPTICS_NATIVE_CASES = 9


def require(value: Bool, message: String) raises:
    if not value:
        raise Error(message)


def fixture(driver: OpticsFitDriver, unit: Int) raises:
    if unit == 8:
        var a: InlineArray[Int, 2] = [0, 0]
        var p: InlineArray[Int, 8] = [0, 0, 0, 2, 2, 0, 1, -1]
        var f: InlineArray[Float64, 4] = [1, 0.05, 1, 2]
        var refused = False
        try:
            _ = native_descriptor_call_fixed(driver.entry, OPTICS_FIT_RUN, a, p, f)
        except problem:
            require(String(problem).find("invalid fit request") >= 0, "wrong malformed-slot refusal")
            refused = True
        require(refused, "missing native result owner was accepted")
        return
    var rows = 180 if unit == 6 else (4 if unit == 4 else 6)
    var features = 3 if unit == 6 else (4 if unit == 4 else (2 if unit >= 1 and unit <= 3 else 1))
    var values = List[Float32](length=rows * features, fill=0)
    if unit == 4:
        values = [0, 1, 10, 11, 1, 0, 9, 10, 10, 9, 0, 1, 11, 10, 1, 0]
    else:
        var xp = values.unsafe_ptr()
        def fill(row: Int) {imm xp, imm features, imm unit}:
            var coordinate = Float32((row // 60) * 4) if unit == 6 else Float32((row // 3) * 10 + row % 3)
            for column in range(features):  # fixed test coordinate fields, not a product numerical loop
                xp[row * features + column] = coordinate
        host_parallelize(fill, rows)
    var metric = 1 if unit == 1 else (2 if unit == 2 else (3 if unit == 3 else (5 if unit == 4 else -1)))
    var step = Float32(2) if unit == 1 else (Float32(1.2599210498948732) if unit == 3 else Float32(1))
    var max_eps = Float64(0) if unit == 7 else (Float64(1) if unit == 6 else Float64(step) * 2)
    var eps = Float64(0) if unit == 7 else Float64(step) * 1.1
    var xi = unit == 5 or unit == 6
    var result = OpticsFitResultSlot()
    driver.fit(OpticsFitRequest(VariantsSpan(Int(values.unsafe_ptr()), len(values)), rows, features,
        5 if unit == 6 else 2, 5 if unit == 6 else 2, 0 if xi else 1, True,
        OpticsMetric(metric, 3 if unit == 3 else 2), max_eps, 0.05, eps), result)
    require(len(result.storage[0].core_distances) == rows and len(result.storage[0].reachability) == rows
        and len(result.storage[0].ordering) == rows and len(result.storage[0].predecessor) == rows
        and len(result.storage[0].labels) == rows, "named graph/result lengths")
    var cores = result.storage[0].core_distances.unsafe_ptr()
    var reach = result.storage[0].reachability.unsafe_ptr()
    var order = result.storage[0].ordering.unsafe_ptr()
    var pred = result.storage[0].predecessor.unsafe_ptr()
    var labels = result.storage[0].labels.unsafe_ptr()
    var flags = List[Int](length=rows, fill=0)
    var fp = flags.unsafe_ptr()
    def check(row: Int) {imm cores, imm reach, imm order, imm pred, imm labels, imm fp, imm unit, imm step}:
        var group_size = 60 if unit == 6 else (2 if unit == 4 else 3)
        var first = row % group_size == 0
        var expected_core = Float32(0) if unit == 6 else step
        var expected_pred = -1 if first else (row - row % group_size if unit == 6 else row - 1)
        var ok = Int(order[row]) == row
        if unit == 7:
            ok = ok and isinf(cores[row]) and isinf(reach[row]) and pred[row] == -1 and labels[row] == -1
        else:
            ok = ok and abs(cores[row] - expected_core) < 0.00001
            ok = ok and (isinf(reach[row]) if first else abs(reach[row] - expected_core) < 0.00001)
            ok = ok and Int(pred[row]) == expected_pred and Int(labels[row]) == row // group_size
        fp[row] = 0 if ok else 1
    host_parallelize(check, rows)
    var failed = 0
    for flag in flags:  # metadata: independent row assertion statuses
        failed += flag
    require(failed == 0, "independent graph/order/predecessor/label oracle")
    if xi:
        var groups = 3 if unit == 6 else 2
        var width = 60 if unit == 6 else 3
        require(len(result.storage[0].hierarchy) == groups * 2, "xi hierarchy count")
        for group in range(groups):  # fixed expected cluster intervals
            require(Int(result.storage[0].hierarchy[2 * group]) == group * width
                and Int(result.storage[0].hierarchy[2 * group + 1]) == (group + 1) * width - 1, "xi interval endpoints")
    else:
        require(len(result.storage[0].hierarchy) == 0, "DBSCAN extraction has no hierarchy payload")
    _ = flags
    _ = values
    _ = result


def run_checks() raises:
    var failures = List[Int](length=OPTICS_NATIVE_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def run(unit: Int) {imm flags}:
        try:
            fixture(OpticsFitDriver(optics_fit_host_v1), unit)
        except problem:
            flags[unit] = 1
            print("FAIL optics_native_cpu fixture", unit, String(problem))
    host_parallelize(run, OPTICS_NATIVE_CASES)
    var failed = 0
    for value in failures:  # metadata: independent fixture statuses
        failed += value
    require(failed == 0, "OPTICS CPU failed " + String(failed) + " fixtures")
    print("PASS optics_native_cpu:", OPTICS_NATIVE_CASES, "actual provider fixtures")
