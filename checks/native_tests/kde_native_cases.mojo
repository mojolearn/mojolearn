# SPDX-License-Identifier: Apache-2.0
"""Original 48 KDE profiles through typed native providers and independent refs.

Inputs, kernel/metric/weight matrix, bandwidth and two retained-query passes
are unchanged. Mathematical references use the existing KDE correctness bound;
they are not outputs of either the legacy controller or the product oracle.
"""
from std.math import isfinite
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.kde_native_api import KDENativeDriver, KDEPrepareRequest, KDEQueryRequest, KDEResidentQuery, KDE_MODE, KDE_VENDOR, KDE_CAPABILITIES, KDE_CAP_QUERY
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.native_tests.pca_native_cases import require, guards, unchanged, PCA_TEST_GUARD

comptime KDE_NATIVE_FIXTURES = 48


def input_values(name: String, count: Int) raises -> List[Float32]:
    var values = List[Float32](unsafe_uninit_length=count)
    with open("checks/native_tests/fixtures/kde/" + name + ".f32", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 4 * count, "KDE original input extent")
        var bp = bytes.unsafe_ptr()
        var vp = values.unsafe_ptr()
        def cell(index: Int) {imm bp, imm vp}:
            var at = 4 * index
            var bits = UInt32(bp[at]) | (UInt32(bp[at + 1]) << 8) | (UInt32(bp[at + 2]) << 16) | (UInt32(bp[at + 3]) << 24)
            vp.unsafe_store(index, bitcast[DType.float32](bits))
        host_parallelize(cell, count)
        _ = bytes^
    return values^


def reference_values(name: String) raises -> List[Float64]:
    var values = List[Float64](unsafe_uninit_length=11)
    with open("checks/native_tests/fixtures/kde/" + name + ".f64", "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 88, "KDE independent reference extent")
        var bp = bytes.unsafe_ptr()
        var vp = values.unsafe_ptr()
        def cell(index: Int) {imm bp, imm vp}:
            var bits = UInt64(0)
            for byte in range(8):  # Fixed-width reference decoding only.
                bits |= UInt64(bp[8 * index + byte]) << UInt64(8 * byte)
            vp.unsafe_store(index, bitcast[DType.float64](bits))
        host_parallelize(cell, 11)
        _ = bytes^
    return values^


def kernel_name(index: Int) -> String:
    if index == 0:
        return "gaussian"
    if index == 1:
        return "tophat"
    if index == 2:
        return "epanechnikov"
    if index == 3:
        return "exponential"
    if index == 4:
        return "linear"
    return "cosine"


def metric_name(index: Int) -> String:
    if index == 0:
        return "euclidean"
    if index == 1:
        return "manhattan"
    if index == 2:
        return "chebyshev"
    return "sqeuclidean"


def compare_scores(actual: List[Float32], expected: List[Float64]) raises:
    var count = len(expected)
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = actual.unsafe_ptr() + 1
    var ep = expected.unsafe_ptr()
    def cell(index: Int) {imm fp, imm ap, imm ep}:
        var observed = Float64(ap[index])
        var wanted = ep[index]
        var okay = observed <= Float64(-1e38)
        if isfinite(wanted):
            okay = isfinite(observed) and abs(observed - wanted) <= Float64(2e-4) * max(Float64(1), abs(wanted))
        fp.unsafe_store(index, Int(not okay))
    host_parallelize(cell, count)
    for failure in failures:
        require(failure == 0, "KDE independent density correctness bound")
    _ = actual
    _ = expected


def compare_total(actual: Float64, expected: List[Float64]) raises:
    # One reduction over the eleven mathematical score expectations. This
    # uses Float64 addition, not the product's fixed Float32 reduction tree.
    var total = Float64(0)
    var outside = False
    for value in expected:
        if isfinite(value):
            total += value
        else:
            outside = True
    if outside:
        require(actual <= Float64(-1e38), "KDE total preserves outside-support score convention")
    else:
        require(isfinite(actual) and abs(actual - total) <= Float64(2e-4) * max(Float64(1), abs(total)),
            "KDE total independent mathematical sum")


def fixture(driver: KDENativeDriver, unit: Int, host: Bool) raises:
    if unit == 0:
        require(driver.metadata(KDE_MODE) == Int(GLOBAL_NUMERIC_MODE)
            and (driver.metadata(KDE_CAPABILITIES) & KDE_CAP_QUERY) != 0, "KDE actual provider mode/query capability")
        var vendor = driver.metadata(KDE_VENDOR)
        require(vendor == 1 if host else vendor >= 2 and vendor <= 4, "KDE actual provider vendor")
    var kernel = kernel_name(unit // 8)
    var metric = metric_name((unit // 2) % 4)
    var weighted = unit % 2 != 0
    var profile = kernel + "-" + metric + ("-weighted" if weighted else "-uniform")
    var train = input_values("train", 185)
    var query = input_values("query", 55)
    var weights = input_values("weights", 37)
    var original_train = train.copy()
    var original_query = query.copy()
    var original_weights = weights.copy()
    var expected = reference_values(profile)
    var destination = List[Float32](length=13, fill=PCA_TEST_GUARD)
    var status = List[Int32](length=3, fill=Int32(23117))
    var ta = Int(train.unsafe_ptr())
    var qa = Int(query.unsafe_ptr())
    var wa = Int(weights.unsafe_ptr()) if weighted else 0
    var da = Int(destination.unsafe_ptr()) + 4
    var sa = Int(status.unsafe_ptr()) + 4
    var handle = -1
    try:
        if not host:
            # Also cover the unchanged nonresident device descriptor route.
            require(driver.query(KDEQueryRequest(ta, qa, wa, da, 37, 11, 5, 0.75,
                weighted, False, kernel.copy(), metric.copy())) == Float64(0), "KDE direct device samples return no total")
            compare_scores(destination, expected)
            # Controlled fixture inputs are finite. This mirrors the public
            # fit door's already-admitted finite flag, without an extra scan.
            handle = driver.prepare(KDEPrepareRequest(ta, wa, 37, 5, 0.75, weighted, True,
                kernel.copy(), metric.copy()))
            require(handle >= 0, "KDE real resident handle")
        for pass_number in range(2):  # Two original retained-fit/reuse passes.
            for total_kind in range(2):  # Original score_samples then score.
                var want_total = total_kind != 0
                var total = Float64(0)
                if host:
                    total = driver.query(KDEQueryRequest(ta, qa, wa, da, 37, 11, 5, 0.75,
                        weighted, want_total, kernel.copy(), metric.copy()))
                else:
                    total = driver.resident(KDEResidentQuery(handle, qa, da, sa, 11, 5, 0.75,
                        want_total, kernel.copy(), metric.copy()))
                    require(status[1] == Int32(0) and status[0] == Int32(23117) and status[2] == Int32(23117),
                        "KDE finite-status word and adjacent storage")
                compare_scores(destination, expected)
                if want_total:
                    compare_total(total, expected)
                else:
                    require(total == Float64(0), "KDE sample-only result has no total")
                guards(destination, "KDE score destination")
        if not host:
            var released = handle
            driver.release(handle)
            handle = -1
            if unit == 0:
                var refused = False
                try:
                    _ = driver.resident(KDEResidentQuery(released, qa, da, sa, 11, 5, 0.75,
                        False, kernel.copy(), metric.copy()))
                except error:
                    refused = String(error) == "unknown or released resident KDE fit handle"
                require(refused, "KDE released handle cannot be queried")
        unchanged(train, original_train, "KDE borrowed training rows")
        unchanged(query, original_query, "KDE borrowed query rows")
        unchanged(weights, original_weights, "KDE borrowed weights")
    finally:
        try:
            if handle >= 0:
                driver.release(handle)
        finally:
            _ = train
            _ = query
            _ = weights
            _ = original_train
            _ = original_query
            _ = original_weights
            _ = expected
            _ = destination
            _ = status
