# SPDX-License-Identifier: Apache-2.0
"""KernelPCA native route/storage state and fixed neighbor request envelopes.

Route boundaries preserve existing choices. The wire oracle validates fields
only and performs no numerical kernel work; actual numerical suites follow.
"""
from core.host_parallel import host_parallelize
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.neighbors_native_driver import XnDriver
from core.kpca_native_plan import (
    KPCAKernelOptions, KPCAFitRoute, KPCAScratch, KPCAEigenRoom, KPCAAlphaRequest,
    KPCAQueryRequest, KPCANeighborDriver, KPCA_PRECOMPUTED, kpca_shape,
)
from x_neighbors.descriptor_schema import xn_operation_id

comptime KPCA_STATE_FIXTURES = 38


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def plan_entry(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        require(result != 0 and aa != 0 and pp != 0, "KPCA plan requires owned wire spans")
        var a = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)
        var p = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=pp)
        var f = Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)
        if operation == xn_operation_id("kpca_alpha_scale"):
            require(na == 3 and np == 3 and nf == 0, "KPCA alpha envelope")
            require(a[0] == 11 and a[1] == 13 and a[2] == 17 + p[2]
                and p[0] == 19 and p[1] == 23 and (p[2] == 0 or p[2] == 1), "KPCA alpha fields")
        elif operation == xn_operation_id("kpca_transform"):
            require(na == 6 and np == 6 and nf == 3 and ff != 0, "KPCA fused query envelope")
            var pre = p[4] == KPCA_PRECOMPUTED
            require(p[4] == 1 or pre, "KPCA query kind")
            require(a[0] == 29 and a[1] == (29 if pre else 31) and a[2] == 37
                and a[3] == 41 and a[4] == 43 and a[5] == 47 and p[0] == 53
                and p[1] == 59 and p[2] == 61 and p[3] == 67 and p[5] == 3,
                "KPCA fused query fields and precomputed source reuse")
            require(f[0] == (Float64(0) if pre else Float64(0.10000000149011612))
                and f[1] == (Float64(0) if pre else Float64(0.3333333432674408))
                and f[2] == Float64(59), "KPCA query scalar float32 admission")
        elif operation == xn_operation_id("kernel"):
            require(na == 3 and np == 5 and nf == 2 and ff != 0, "KPCA kernel envelope")
            require(a[0] == 11 and a[1] == 13 and a[2] == 17 and p[0] == 19
                and p[1] == 23 and p[2] == 29 and p[3] == 1 and p[4] == 3
                and f[0] == Float64(0.10000000149011612) and f[1] == Float64(0.3333333432674408),
                "KPCA kernel field/scalar rounding")
        elif operation == xn_operation_id("colsum") or operation == xn_operation_id("rowsum"):
            var column = operation == xn_operation_id("colsum")
            require(na == 2 and np == 2 and nf == 0 and a[0] == (11 if column else 31)
                and a[1] == (13 if column else 37) and p[0] == (17 if column else 41)
                and p[1] == (19 if column else 43), "KPCA distinct row/column sum fields")
        elif operation == xn_operation_id("scale_div"):
            require(na == 2 and np == 1 and nf == 1 and ff != 0 and a[0] == 11 and a[1] == 13
                and p[0] == 17 and f[0] == Float64(0.125), "KPCA divide fields")
        elif operation == xn_operation_id("kpca_center"):
            require(na == 5 and np == 2 and nf == 0 and a[0] == 11 and a[1] == 13 and a[2] == 17
                and a[3] == 19 and a[4] == 23 and p[0] == 29 and p[1] == 31, "KPCA center fields")
        elif operation == xn_operation_id("matmul"):
            require(na == 3 and np == 3 and nf == 0 and a[0] == 11 and a[1] == 13 and a[2] == 17
                and p[0] == 19 and p[1] == 23 and p[2] == 29, "KPCA projection fields")
        else:
            raise Error("KPCA plan unexpected operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = 0
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def zeros(value: List[Float32]) raises:
    var count = len(value)
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var vp = value.unsafe_ptr()
    def cell(index: Int) {imm fp, imm vp}:
        fp.unsafe_store(index, Int(vp[index] != Float32(0)))
    host_parallelize(cell, count)
    for failed in failures:
        require(failed == 0, "KPCA native storage starts zeroed")
    _ = value


def move_scratch(var value: KPCAScratch) -> KPCAScratch:
    return value^


def move_eigen(var value: KPCAEigenRoom) -> KPCAEigenRoom:
    return value^


def route_case(unit: Int) raises:
    var rows = 201
    var requested = 3
    var kind = 2
    var auto = True
    var available = True
    var lanczos = False
    var resident = False
    var room = 3
    if unit == 0:
        rows = 200
    elif unit == 1 or unit == 2 or unit == 11:
        requested = 3 if unit == 1 else (9 if unit == 2 else 1)
        room = requested
        lanczos = True
        resident = True
    elif unit == 3:
        requested = 10
        room = 10
    elif unit == 4:
        requested = 0
        room = 0
    elif unit == 5:
        requested = -1
        room = 201
    elif unit == 6:
        kind = 1
        lanczos = True
    elif unit == 7:
        auto = False
    elif unit == 8:
        available = False
        lanczos = True
    elif unit == 9:
        rows = 3
        requested = 9
    else:
        rows = 0
        requested = -1
        room = 0
    var route = KPCAFitRoute(rows, requested, kind, auto, available)
    require(route.lanczos == lanczos and route.resident == resident and route.room == room,
            "KPCA existing route choice and room capacity")


def fixture(unit: Int) raises:
    if unit < 12:
        route_case(unit)
    elif unit < 16:
        var vector = unit == 13 or unit == 15
        var rows = 0 if unit == 15 else 3
        var cols = 0 if unit == 14 else 5
        var shape = kpca_shape(rows, cols, vector)
        require(len(shape) == (1 if vector else 2) and shape[0] == rows,
                "KPCA vector/matrix rank and rows")
        if not vector:
            require(shape[1] == cols, "KPCA zero columns retained")
    elif unit < 19 or unit == 37:
        var rows = 0 if unit == 16 else 3
        var cols = 0 if unit == 17 else 4
        var scratch = KPCAScratch(rows, cols)
        require(scratch.rows == rows and scratch.cols == cols and len(scratch.values) == rows * cols,
                "KPCA typed scratch shape/storage extent")
        require((scratch.address() == 0) == (rows * cols == 0), "KPCA empty scratch address contract")
        zeros(scratch.values)
        if unit == 18 or unit == 37:
            var address = scratch.address()
            var p = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
            def cell(index: Int) {imm p}:
                p.unsafe_store(index, Float32(index + 1))
            host_parallelize(cell, 12)
            require(scratch.values[0] == Float32(1) and scratch.values[11] == Float32(12), "KPCA scratch owns written endpoint words")
            if unit == 37:
                var moved = move_scratch(scratch^)
                require(moved.address() == address and moved.rows == 3 and moved.cols == 4
                    and moved.values[0] == Float32(1) and moved.values[11] == Float32(12), "KPCA scratch move retains storage")
    elif unit == 19 or unit == 20 or unit == 36:
        var rows = -1 if unit == 19 else (Int.MAX if unit == 20 else Int.MAX // 4 + 1)
        var cols = 2 if unit == 20 else 1
        var expected = String("ValueError: negative dimensions are not allowed")
        if unit == 20:
            expected = "ValueError: array element count exceeds native storage"
        elif unit == 36:
            expected = "ValueError: array byte count exceeds native storage"
        var refused = False
        try:
            var scratch = KPCAScratch(rows, cols)
            _ = scratch^
        except error:
            refused = String(error) == expected
        require(refused, "KPCA scratch validates shape before allocation")
    elif unit < 25:
        var rows = 0 if unit == 21 else 4
        var room_count = 0 if unit == 21 or unit == 22 else 2
        var room = KPCAEigenRoom(rows, room_count)
        require(len(room.values) == max(rows, 1) and len(room.vectors) == max(rows * room_count, 1)
            and room.values_address() != 0 and room.vectors_address() != 0
            and room.values_address() != room.vectors_address(), "KPCA eigen rooms and nonnull independent empty dummies")
        zeros(room.values)
        zeros(room.vectors)
        if unit == 24:
            room.values[0] = 7
            room.vectors[3] = 11
            var address = room.values_address()
            var moved = move_eigen(room^)
            require(moved.values_address() == address and moved.values[0] == Float32(7)
                and moved.vectors[3] == Float32(11), "KPCA eigen room move retains independent data")
    else:
        var driver = KPCANeighborDriver(XnDriver(plan_entry))
        var options = KPCAKernelOptions(KPCA_PRECOMPUTED if unit == 29 else 1, 3, 0.1, Float64(1) / Float64(3))
        if unit == 25 or unit == 28 or unit == 29:
            var query = KPCAQueryRequest(29, 31, 37, 41, 43, 47, 53, 59, 61, 67, options^)
            if unit == 25:
                var copied = query.copy()
                copied.options.gamma = 0.5
                copied.options.kind = 2
                copied.source = 101
                require(query.options.gamma == Float64(0.1) and query.options.kind == 1 and query.source == 29,
                        "KPCA query/options copies do not alias")
            else:
                driver.fused_query(query)
        elif unit == 26 or unit == 27:
            driver.alpha(KPCAAlphaRequest(11, 13, 17 if unit == 26 else 18, 19, 23, unit == 27))
        elif unit == 30:
            driver.kernel(11, 13, 17, 19, 23, 29, options)
        elif unit == 31:
            driver.colsum(11, 13, 17, 19)
        elif unit == 32:
            driver.rowsum(31, 37, 41, 43)
        elif unit == 33:
            driver.divide(11, 13, 17, 0.125)
        elif unit == 34:
            driver.center(11, 13, 17, 19, 23, 29, 31)
        else:
            driver.matmul(11, 13, 17, 19, 23, 29)


def run_checks() raises:
    var failures = List[Int](length=KPCA_STATE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL kpca_state fixture", unit, String(error))
    host_parallelize(task, KPCA_STATE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "kpca_state failed " + String(failed) + " fixtures")
    print("PASS kpca_state:", KPCA_STATE_FIXTURES, "native route/storage/query-wire fixtures")
