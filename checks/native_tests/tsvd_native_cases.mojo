# SPDX-License-Identifier: Apache-2.0
"""Original literal TSVD query case over actual typed native providers.

Expected integers are analytic dot products of the original 2x3 source and
2x3 components. No fitted output or repeated execution supplies an oracle.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.pca_native_api import PCAFoldStore
from core.tsvd_native_api import TSVDNativeDriver, TSVDQueryRequest, TSVDFitRequest, TSVD_RANDOMIZED

comptime TSVD_NATIVE_FIXTURES = 7
comptime TSVD_GUARD = Float32(27182.5)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def check_words(actual: List[Float32], expected: List[Float32], offset: Int, name: String) raises:
    var count = len(expected)
    require(len(actual) >= count + offset, name + " storage extent")
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var ap = actual.unsafe_ptr() + offset
    var ep = expected.unsafe_ptr()
    def cell(index: Int) {imm fp, imm ap, imm ep}:
        fp.unsafe_store(index, Int(bitcast[DType.uint32](ap[index]) != bitcast[DType.uint32](ep[index])))
    host_parallelize(cell, count)
    for failure in failures:
        require(failure == 0, name + " literal value/storage ownership")
    _ = actual
    _ = expected


def guards(dst: List[Float32]) raises:
    require(dst[0] == TSVD_GUARD and dst[len(dst) - 1] == TSVD_GUARD,
            "TSVD destination guards")


def fixture(driver: TSVDNativeDriver, unit: Int, host: Bool) raises:
    if unit == 0:
        require(driver.metadata(0) == 1 and driver.metadata(2) == (33 if host else 9)
                and driver.metadata(3) == 0, "TSVD estimator mode/capabilities/Q-fix")
        var vendor = driver.metadata(1)
        require(vendor == 1 if host else (vendor >= 2 and vendor <= 4), "TSVD estimator vendor")
        return
    var source: List[Float32] = [1, 2, 3, -1, 1, 2]
    var components: List[Float32] = [1, 0, 1, 0, 2, -1]
    var original_source = source.copy()
    var original_components = components.copy()
    var dst = List[Float32](length=6, fill=TSVD_GUARD)
    var request = TSVDQueryRequest(Int(source.unsafe_ptr()), Int(components.unsafe_ptr()),
        Int(dst.unsafe_ptr()) + 4, 2, 3, 2, False, False, False, 0)
    if unit == 1:
        require(driver.query(request) == 0, "TSVD literal transform status")
        check_words(dst, List[Float32](4, 1, 1, 0), 1, "TSVD original 2x3 transform")
        components[0] = 2
        require(driver.query(request) == 0, "TSVD current borrowed component status")
        check_words(dst, List[Float32](5, 1, 0, 0), 1, "TSVD current borrowed component values")
        components[0] = 1
    elif unit == 2:
        var projected: List[Float32] = [4, 1, 1, 0]
        var original = projected.copy()
        var inverse_dst = List[Float32](length=8, fill=TSVD_GUARD)
        var inverse = TSVDQueryRequest(Int(projected.unsafe_ptr()), Int(components.unsafe_ptr()),
            Int(inverse_dst.unsafe_ptr()) + 4, 2, 3, 2, True, False, False, 0)
        require(driver.query(inverse) == 0, "TSVD literal inverse status")
        check_words(inverse_dst, List[Float32](4, 2, 3, 1, 0, 1), 1, "TSVD literal inverse")
        check_words(projected, original, 0, "TSVD inverse source")
        guards(inverse_dst)
        _ = projected^
    elif unit == 3:
        var replacement: List[Float32] = [2, 0, 1, 0, 2, -1]
        var copied = request.copy()
        copied.components = Int(replacement.unsafe_ptr())
        require(driver.query(copied) == 0, "TSVD rebound component status")
        check_words(dst, List[Float32](5, 1, 0, 0), 1, "TSVD rebound component values")
        require(request.components == Int(components.unsafe_ptr()), "TSVD original query record unchanged")
        _ = replacement^
    elif unit == 4:
        source[2] = bitcast[DType.float32](UInt32(0x7FC00042))
        var nan_source = source.copy()
        request.test_finite = True
        require(driver.query(request) != 0, "TSVD deferred finite status")
        check_words(source, nan_source, 0, "TSVD nonfinite input")
        source[2] = 3
    elif unit == 5:
        var refused = False
        try:
            _ = driver.fit(TSVDFitRequest(0, 0, 0, 0, 0, 2, 3, 2, 10, 0, False, 0,
                False, TSVD_RANDOMIZED, PCAFoldStore()))
        except error:
            var expected = String("TruncatedSVD host estimator does not provide this fit route") if host else String("TruncatedSVD estimator does not provide this fit route")
            refused = String(error) == expected
        require(refused, "TSVD estimator refuses decomposition-image route before buffer access")
    elif host:
        var refused_open = False
        var refused_query = False
        var refused_close = False
        try:
            _ = driver.open()
        except error:
            refused_open = String(error) == "TruncatedSVD host estimator does not own resident leases"
        request.resident = True
        try:
            _ = driver.query(request)
        except error:
            refused_query = String(error) == "TruncatedSVD host estimator does not own resident leases"
        try:
            driver.close(0)
        except error:
            refused_close = String(error) == "TruncatedSVD host estimator does not own resident leases"
        require(refused_open and refused_query and refused_close, "TSVD host resident capability refusal")
    else:
        var lease = driver.open()
        require(lease > 0, "TSVD native resident lease is positive")
        try:
            request.resident = True
            request.lease = lease
            require(driver.query(request) == 0, "TSVD resident transform status")
            check_words(dst, List[Float32](4, 1, 1, 0), 1, "TSVD resident transform")
            var restored = List[Float32](length=8, fill=TSVD_GUARD)
            var inverse = TSVDQueryRequest(Int(dst.unsafe_ptr()) + 4, Int(components.unsafe_ptr()),
                Int(restored.unsafe_ptr()) + 4, 2, 3, 2, True, False, True, lease)
            require(driver.query(inverse) == 0, "TSVD resident inverse status")
            check_words(restored, List[Float32](4, 2, 3, 1, 0, 1), 1, "TSVD resident inverse")
            guards(restored)
        finally:
            driver.close(lease)
    guards(dst)
    check_words(source, original_source, 0, "TSVD original source")
    check_words(components, original_components, 0, "TSVD original components")
    _ = source^
    _ = components^
    _ = dst^


def negative_lease(driver: TSVDNativeDriver) raises:
    # The provider must reject this native token before dereferencing any
    # buffer; the test owns no fake pointer or partially initialized owner.
    var refused = False
    try:
        _ = driver.query(TSVDQueryRequest(0, 0, 0, 2, 3, 2, False, False, True, -1))
    except error:
        refused = String(error) == "tsvd_transform_resident: the lease must be 0 or a lease, got -1"
    require(refused, "TSVD negative lease refusal precedes buffer access")
