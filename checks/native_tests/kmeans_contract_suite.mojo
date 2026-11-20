# SPDX-License-Identifier: Apache-2.0
"""Native KMeans wire and owner contracts, independently checking every slot."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.kmeans_native_api import (
    KMeansNativeKernels, KMeansNativeDriver, KMeansFoldStore, KMeansFitRequest,
    KMeansFitResult, KMeansQueryRequest, KMeansQueryResult, kmeans_native_dispatch,
)
from core.kmeans_owner_state import KMeansResidentState

comptime KMEANS_CONTRACT_FIXTURES = 26


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


struct WireOracle(KMeansNativeKernels):
    @staticmethod
    def mode() -> Int:
        return 1

    @staticmethod
    def capabilities() -> Int:
        return 15

    @staticmethod
    def fit(r: KMeansFitRequest) raises -> KMeansFitResult:
        _require(r.source == 11 and r.centers == 13 and r.labels == 17 and r.weights == 19,
            "fit input/output address slots")
        _require(r.rows == 29 and r.features == 31 and r.clusters == 37 and r.weight_count == 41,
            "fit shape/weight slots")
        _require(r.max_iter == 43 and r.seed == 47 and r.n_init == 53 and r.init == 59 and r.metric == 61,
            "fit algorithm option slots")
        _require(r.test_finite and r.tolerance == 0.125 and r.oversampling == 2.5,
            "fit finite/tolerance/oversampling slots")
        _require(r.distances == 0 or r.distances == 23, "fused transform output slot")
        if r.fold.active():
            _require(r.fold.base == 67 and r.fold.rows == 71 and r.fold.indices == 73, "fit fold identity slots")
        var route = Int(r.distances != 0) + 2 * Int(r.fold.active())
        return KMeansFitResult(-0.0, 79, 1.25, 2.5, -7, route)

    @staticmethod
    def query(r: KMeansQueryRequest) raises -> KMeansQueryResult:
        _require(r.source == 11 and r.centers == 13 and r.dst == 17, "query address slots")
        _require(r.rows == 29 and r.features == 31 and r.clusters == 37 and r.metric == 61 and r.test_finite,
            "query option/shape slots")
        _require(r.handle == -1 or r.handle == 83, "query resident handle slot")
        if r.fold.active():
            _require(r.fold.base == 67 and r.fold.rows == 71 and r.fold.indices == 73, "query fold slots")
        return KMeansQueryResult(Int(r.transform) + 2 * Int(r.handle >= 0) + 4 * Int(r.fold.active()), r.rows)

    @staticmethod
    def prepare(centers: Int, clusters: Int, features: Int) raises -> Int:
        _require(centers == 13 and clusters == 37 and features == 31, "resident prepare slots")
        return 83

    @staticmethod
    def release(handle: Int) raises:
        _require(handle == 83, "resident release slot")


def wire_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    return kmeans_native_dispatch[WireOracle](operation, aa, na, pp, np, ff, nf, result, context, sink)


def _fixture(unit: Int) raises:
    var driver = KMeansNativeDriver(wire_entry)
    if unit == 0:
        _require(driver.metadata(0) == 1 and driver.metadata(1) == 15, "compiled metadata")
    elif unit < 4:
        var fold = KMeansFoldStore(67, 71, 73) if unit == 3 else KMeansFoldStore()
        var result = driver.fit(KMeansFitRequest(11, 13, 17, 19, 23 if unit == 2 else 0,
            29, 31, 37, 41, 43, 47, 53, 59, 61, True, 0.125, 2.5, fold))
        _require(bitcast[DType.uint64](result.inertia) == UInt64(0x8000000000000000), "fit result keeps signed zero")
        _require(result.iterations == 79 and result.sum_scale == 1.25 and result.weight_scale == 2.5
            and result.finite == -7 and result.range_error == unit - 1, "fit result wire layout")
    elif unit < 10:
        var variant = unit - 4
        var fold = KMeansFoldStore(67, 71, 73) if variant >= 4 else KMeansFoldStore()
        var handle = 83 if variant == 2 or variant == 3 else -1
        var result = driver.query(KMeansQueryRequest(11, 13, 17, 29, 31, 37, 61,
            True, variant % 2 == 1, handle, fold))
        _require(result.finite == variant and result.range_error == 29, "query route and result slots")
    elif unit == 10:
        _require(driver.prepare(13, 37, 31) == 83, "prepare native result")
    elif unit == 11:
        driver.release(83)
    elif unit < 20:
        var variant = unit - 12
        var state = KMeansResidentState(101, 103, 107, 5, 37, 31, 83, driver.copy())
        if variant < 6:
            var matches = state.matches(109 if variant == 1 else 101,
                109 if variant == 2 else 103, 109 if variant == 3 else 107, 5,
                41 if variant == 4 else 37, 41 if variant == 5 else 31)
            _require(matches == (variant == 0), "resident ownership/buffer/provider/shape identity")
        elif variant == 6:
            _require(state.take_handle() == 83 and state.take_handle() == -1,
                "resident release transfers the handle exactly once")
            _require(not state.matches(101, 103, 107, 5, 37, 31), "retired owner cannot hit")
        else:
            _require(not state.matches(101, 103, 107, 6, 37, 31), "a different admitted provider cannot reuse the handle")
            _require(state.driver.metadata(0) == 1 and state.driver.prepare(13, 37, 31) == 83,
                "resident owner retains native provider dispatch")
    else:
        var operation = unit - 18
        var expected = String("KMeans native ABI: unknown operation")
        if operation == 2:
            expected = "KMeans native ABI: fit requires 6 addresses, 13 integers and 2 floats"
        elif operation == 3:
            expected = "KMeans native ABI: query requires 4 addresses and 10 integers"
        elif operation == 4:
            expected = "KMeans native ABI: prepare requires 1 address and 2 integers"
        elif operation == 5:
            expected = "KMeans native ABI: release requires 1 integer"
        var refused = False
        try:
            _ = driver.metadata(operation)
        except error:
            refused = String(error) == expected
        _require(refused, "malformed descriptor call refusal")


def run_checks() raises:
    var failures = List[Int](length=KMEANS_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            _fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL kmeans_contract fixture", unit, String(error))
    host_parallelize(task, KMEANS_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _require(failed == 0, "native KMeans contracts failed " + String(failed) + " fixtures")
    print("PASS kmeans_contract:", KMEANS_CONTRACT_FIXTURES, "native wire and owner fixtures")
