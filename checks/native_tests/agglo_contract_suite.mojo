# SPDX-License-Identifier: Apache-2.0
"""Typed Agglomerative requests retain the existing solver/cluster/Xn wires.

Parallel independent fixtures use prescribed metadata and owned output
markers. Fake tensor addresses are never dereferenced; no numerical provider
or device registry is called by these transport contracts.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neighbors_native_driver import XnDriver
from core.agglo_native_api import AggloSolverDriver, AggloClusterDriver, AggloZeroRowsDriver, AggloSolverRequest, AggloClusterRequest, AggloZeroRowsRequest, agglo_reference_request, agglo_query_request, agglo_resident_query
from hierarchy.solver_descriptor_schema import HIERARCHY_SOLVER_LINKAGE
from hierarchy.cluster_descriptor_schema import HIERARCHY_CLUSTER_FIT
from x_neighbors.descriptor_schema import xn_operation_id

comptime AGGLO_CONTRACT_FIXTURES = 28


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def solver_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                 result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        require(operation == HIERARCHY_SOLVER_LINKAGE and na == 4 and np == 5 and nf == 0 and result != 0,
            "Agglomerative solver fixed envelope")
        var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
        var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
        require(a[0] == 11 and a[1] == 13 and a[2] == 17 and a[3] == 19, "Agglomerative solver address order")
        require(p[0] == 23 and p[1] == 29 and p[2] == 31 and p[3] == 1 and (p[4] == 0 or p[4] == 1),
            "Agglomerative solver dimensions/cuML sqrt metric/knn flag")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 9007199254740993 + p[4]
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def cluster_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                  result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        require(operation == HIERARCHY_CLUSTER_FIT and na == 6 and np == 11 and nf == 2 and result != 0,
            "Agglomerative cluster fixed envelope")
        var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
        var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
        var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
        var variant = p[0] - 400
        require(variant >= 0 and variant < 16, "Agglomerative cluster fixture variant")
        require(a[0] == 37 and a[1] == 41 and a[2] == 43 and a[3] == 47 and a[4] == 53 and a[5] > 4096,
            "Agglomerative cluster address order and owned result marker")
        var metrics: List[Int] = [-1, 1, 4, 5]
        require(p[1] == 59 and p[2] == 61 and p[3] == 67 and p[4] == variant % 4
            and p[5] == metrics[variant % 4] and p[6] == 71 and p[7] == 73 and p[8] == variant // 8
            and p[9] == 79 and p[10] == ((variant >> 2) & 1), "Agglomerative all eleven cluster integer slots")
        require(f[0] == Float64(1.5000000000000002), "Agglomerative binary64 Minkowski parameter")
        if p[10] == 0:
            require(bitcast[DType.uint64](f[1]) == UInt64(0x8000000000000000), "Agglomerative signed-zero threshold")
        else:
            require(f[1] == Float64(0.10000000000000002), "Agglomerative binary64 threshold precision")
        var marker = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=a[5])
        marker[0] = Float64(variant + 1)
        marker[1] = 97.0
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 0
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def zero_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
               result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        require(operation == xn_operation_id("row_all_zero") and operation == 36
            and na == 2 and np == 2 and nf == 0 and result != 0, "Agglomerative existing Xn zero-row envelope")
        var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
        var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
        require(a[0] == 83 and a[1] > 4096 and p[0] == 89 and p[1] == 97, "Agglomerative zero-row slots")
        MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=a[1])[] = 1
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 0
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def rejected_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                   result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    native_descriptor_report(Error("Agglomerative native provider refusal"), context, sink)
    return 1


def solver_request(knn: Bool) -> AggloSolverRequest:
    return AggloSolverRequest(11, 13, 17, 19, 23, 29, 31, 1, knn)


def cluster_request(marker: Int, variant: Int) -> AggloClusterRequest:
    var metrics: List[Int] = [-1, 1, 4, 5]
    var threshold = bitcast[DType.float64](UInt64(0x8000000000000000))
    if (variant & 4) != 0:
        threshold = 0.10000000000000002
    return AggloClusterRequest(37, 41, 43, 47, 53, marker, 400 + variant, 59, 61, 67,
        variant % 4, metrics[variant % 4], 71, 73, variant // 8, 79, (variant & 4) != 0,
        1.5000000000000002, threshold)


def fixture(unit: Int) raises:
    if unit < 2:
        require(AggloSolverDriver(solver_entry).fit(solver_request(unit == 1)) == 9007199254740993 + unit,
            "Agglomerative exact native solver result and Boolean")
    elif unit < 18:
        var marker = List[Float64](length=4, fill=-123.5)
        try:
            AggloClusterDriver(cluster_entry).fit(cluster_request(Int(marker.unsafe_ptr()) + 8, unit - 2))
            require(marker[0] == Float64(-123.5) and marker[3] == Float64(-123.5)
                and marker[1] == Float64(unit - 1) and marker[2] == Float64(97),
                "Agglomerative cluster invocation publishes owned native markers within guards")
        finally:
            _ = marker^
    elif unit == 18 or unit == 27:
        var marker = List[Int32](length=3, fill=12345)
        try:
            var request = AggloZeroRowsRequest(83, Int(marker.unsafe_ptr()) + 4, 89, 97)
            if unit == 27:
                var copied = request.copy()
                copied.rows = 101
                copied.source = 103
                require(request.rows == 89 and request.source == 83, "Agglomerative zero-row request copy independence")
            AggloZeroRowsDriver(XnDriver(zero_entry)).run(request)
            require(marker[0] == Int32(12345) and marker[1] == Int32(1) and marker[2] == Int32(12345),
                "Agglomerative zero-row native invocation marker and guards")
        finally:
            _ = marker^
    elif unit == 19:
        var request = agglo_reference_request(107, 109, 113, 127)
        require(request.refs == 107 and request.keys == 109 and request.labels == 109
            and request.rows == 113 and request.features == 127 and request.metric == 0,
            "Agglomerative retained labels supply both tie keys and values; shared Euclidean code0")
    elif unit == 20:
        var request = agglo_query_request(131, 137, 139, 149, 151, 157, 163, 167)
        require(request.refs == 131 and request.keys == 137 and request.labels == 137 and request.query == 139
            and request.dst == 149 and request.chosen == 151 and request.rows == 157 and request.queries == 163
            and request.features == 167 and request.metric == 0 and not request.has_threshold
            and request.eps == Float64(0), "Agglomerative borrowed query uses labels as tie keys and no radius")
    elif unit == 21:
        var request = agglo_resident_query(173, 179, 9007199254740993, 181, 191)
        require(request.query == 173 and request.dst == 179 and request.handle == 9007199254740993
            and request.queries == 181 and request.features == 191 and request.metric == 0
            and not request.has_threshold and request.eps == Float64(0), "Agglomerative resident exact handle and no radius")
    elif unit == 22:
        var request = solver_request(False)
        var copied = request.copy()
        copied.source = 197
        copied.metric = 199
        copied.use_knn = True
        require(request.source == 11 and request.metric == 1 and not request.use_knn,
            "Agglomerative solver request copy independence")
        require(AggloSolverDriver(solver_entry).fit(request) == 9007199254740993, "Agglomerative original solver request retained")
    elif unit == 23:
        var marker = List[Float64](length=2, fill=-123.5)
        try:
            var request = cluster_request(Int(marker.unsafe_ptr()), 7)
            var copied = request.copy()
            copied.source_count = 0
            copied.threshold = 2.0
            copied.use_threshold = False
            require(request.source_count == 407 and request.threshold == Float64(0.10000000000000002)
                and request.use_threshold, "Agglomerative cluster request copy independence")
            AggloClusterDriver(cluster_entry).fit(request)
            require(marker[0] == Float64(8) and marker[1] == Float64(97), "Agglomerative original cluster request retained")
        finally:
            _ = marker^
    else:
        var refused = False
        try:
            if unit == 24:
                _ = AggloSolverDriver(rejected_entry).fit(solver_request(False))
            elif unit == 25:
                AggloClusterDriver(rejected_entry).fit(cluster_request(0, 0))
            else:
                AggloZeroRowsDriver(XnDriver(rejected_entry)).run(AggloZeroRowsRequest(0, 0, 0, 0))
        except error:
            refused = String(error) == "Agglomerative native provider refusal"
        require(refused, "Agglomerative every typed driver propagates native provider errors")


def run_checks() raises:
    var failures = List[Int](length=AGGLO_CONTRACT_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL agglo_contract fixture", unit, String(error))
    host_parallelize(task, AGGLO_CONTRACT_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "agglo_contract failed " + String(failed) + " fixtures")
    print("PASS agglo_contract:", AGGLO_CONTRACT_FIXTURES, "native solver/tree/query wire fixtures")
