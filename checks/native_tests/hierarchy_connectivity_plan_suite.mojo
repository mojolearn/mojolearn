# SPDX-License-Identifier: Apache-2.0
"""Independent stored-edge oracles, refusal ordering and native owner contracts.

Each fixture owns its inputs and runs independently. Large setup/assertion
spans use the same established host task policy; no backend is simulated.
These exercise the actual storage plans, not hierarchy numerical kernels.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.hierarchy_connectivity_plan import (
    ConnectivityStorage, ConnectivityEdge, HierarchyConnectivityPlan,
    connectivity_edge, connectivity_pack, connectivity_compressed, connectivity_coo, connectivity_dia,
)

comptime HIERARCHY_CONNECTIVITY_FIXTURES = 21


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def vector(address: Int, count: Int, width: Int, kind: Int) -> ConnectivityStorage:
    return ConnectivityStorage(address, count, 1, 1, width, 0, 0, width, kind, True)


def expect(plan: HierarchyConnectivityPlan, expected: List[Float32]) raises:
    var count = len(expected)
    require(len(plan.words) == count and 3 * plan.edges == count, "COO output length")
    if count == 0:
        require(plan.address() == 0, "empty COO span has no dereferenceable pointer")
        return
    var actual = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=plan.address())
    var wanted = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=Int(expected.unsafe_ptr()))
    var units = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, units)
    var failures = List[Int](length=units, fill=0)
    var fp = failures.unsafe_ptr()
    def check(unit: Int) {imm actual, imm wanted, imm count, imm chunk, imm fp}:
        for at in range(unit * chunk, min((unit + 1) * chunk, count)):
            if bitcast[DType.uint32](actual[at]) != bitcast[DType.uint32](wanted[at]):
                fp[unit] = 1
                return
    host_parallelize(check, units)
    for failed in failures:  # fixture task summaries
        require(failed == 0, "stored COO word or ordering differs from independent oracle")
    _ = expected
    _ = plan


def detached() raises -> HierarchyConnectivityPlan:
    var rows: List[Int32] = [2, 0, 2]
    var cols: List[Int32] = [1, 2, 1]
    var values: List[Float32] = [4, 0, -4]
    var plan = connectivity_coo(vector(Int(rows.unsafe_ptr()), 3, 4, 1),
        vector(Int(cols.unsafe_ptr()), 3, 4, 1), vector(Int(values.unsafe_ptr()), 3, 4, 2), 3)
    _ = rows^
    _ = cols^
    _ = values^
    return plan^


def large_case(kind: Int) raises:
    var count = 262147
    var indices = List[Int32](unsafe_uninit_length=count)
    var data = List[Float32](unsafe_uninit_length=count)
    var ix = rebind[MutPointer[Int32, MutUntrackedOrigin]](indices.unsafe_ptr())
    var dx = rebind[MutPointer[Float32, MutUntrackedOrigin]](data.unsafe_ptr())
    var units = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, units)
    def fill(unit: Int) {imm ix, imm dx, imm kind, imm count, imm chunk}:
        for at in range(unit * chunk, min((unit + 1) * chunk, count)):
            ix[at] = Int32(at % 4) if kind == 0 else Int32(0)
            dx[at] = Float32(at % 7 - 3)
    host_parallelize(fill, units)
    var pointers: List[Int32] = [0, Int32(count), Int32(count), Int32(count), Int32(count)]
    var plan = HierarchyConnectivityPlan()
    if kind == 0:
        # One long compressed row must still split conversion by entries.
        plan = connectivity_compressed(vector(Int(data.unsafe_ptr()), count, 4, 2),
            vector(Int(indices.unsafe_ptr()), count, 4, 1), vector(Int(pointers.unsafe_ptr()), 5, 4, 1), 4, False, False)
    else:
        var matrix = vector(Int(data.unsafe_ptr()), count, 4, 2)
        matrix.size1 = 1
        matrix.stride1 = 4
        # Repeated diagonals are intentional: each stored record survives.
        plan = connectivity_dia(vector(Int(indices.unsafe_ptr()), count, 4, 1), matrix, 4)
    require(plan.edges == count, "large conversion preserves every stored edge")
    var values = Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=plan.address())
    var failures = List[Int](length=units, fill=0)
    var fp = failures.unsafe_ptr()
    def check(unit: Int) {imm values, imm count, imm chunk, imm kind, imm fp}:
        for at in range(unit * chunk, min((unit + 1) * chunk, count)):
            var col = Float32(at % 4) if kind == 0 else Float32(0)
            if values[at] != 0 or values[count + at] != col or values[2 * count + at] != Float32(at % 7 - 3):
                fp[unit] = 1
                return
    host_parallelize(check, units)
    for failed in failures:  # task summaries
        require(failed == 0, "large stored-edge pattern or prefix order")
    _ = plan^
    _ = indices^
    _ = data^
    _ = pointers^


def fixture(unit: Int) raises:
    if unit == 0 or unit == 1:
        var data: List[Float32] = [2, 0, -2, 7, 9]
        var indices: List[Int32] = [2, 1, 2, 0, 99]
        var pointers: List[Int64] = [0, 3, 3, 4]
        var result = connectivity_compressed(vector(Int(data.unsafe_ptr()), 5, 4, 2),
            vector(Int(indices.unsafe_ptr()), 5, 4, 1), vector(Int(pointers.unsafe_ptr()), 4, 8, 1), 3, unit == 1, False)
        if unit == 0:
            expect(result, [0, 0, 0, 2, 2, 1, 2, 0, 2, 0, -2, 7])
        else:
            expect(result, [2, 1, 2, 0, 0, 0, 0, 2, 2, 0, -2, 7])
        _ = data^
        _ = indices^
        _ = pointers^
    elif unit == 2:
        var data: List[Float32] = [1, 2, 3, 4, -1, 0, -3, -4, 8, 9, 10, 11]
        var indices: List[Int32] = [1, 1, 0]
        var pointers: List[Int32] = [0, 2, 3]
        var view = vector(Int(data.unsafe_ptr()), 3, 4, 2)
        view.size1 = 2
        view.size2 = 2
        view.stride0 = 16
        view.stride1 = 8
        view.stride2 = 4
        var result = connectivity_compressed(view, vector(Int(indices.unsafe_ptr()), 3, 4, 1),
            vector(Int(pointers.unsafe_ptr()), 3, 4, 1), 4, False, True)
        expect(result, [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3,
                        2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 0, 1,
                        1, 2, 3, 4, -1, 0, -3, -4, 8, 9, 10, 11])
        _ = data^
        _ = indices^
        _ = pointers^
    elif unit == 3:
        var rows: List[UInt32] = [0x02000000, 0, 0x01000000]
        var cols: List[UInt32] = [0, 0x02000000, 0x01000000]
        var values: List[UInt32] = [0x0000803f, 0x00000080, 0x00002040]
        var r = vector(Int(rows.unsafe_ptr() + 2), 3, 4, 1)
        var c = vector(Int(cols.unsafe_ptr() + 2), 3, 4, 1)
        var d = vector(Int(values.unsafe_ptr() + 2), 3, 4, 2)
        r.native = False
        c.native = False
        d.native = False
        r.stride0 = -4
        c.stride0 = -4
        d.stride0 = -4
        var result = connectivity_coo(r, c, d, 3)
        expect(result, [1, 0, 2, 1, 2, 0, 2.5, bitcast[DType.float32](UInt32(0x80000000)), 1])
        _ = rows^
        _ = cols^
        _ = values^
    elif unit == 4:
        var offsets: List[Int64] = [-1, 1, 5, -4]
        var values: List[Float32] = [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43]
        var d = vector(Int(values.unsafe_ptr()), 4, 4, 2)
        d.size1 = 4
        d.stride0 = 16
        d.stride1 = 4
        var result = connectivity_dia(vector(Int(offsets.unsafe_ptr()), 4, 8, 1), d, 4)
        expect(result, [1, 2, 3, 0, 1, 2, 0, 1, 2, 1, 2, 3, 10, 11, 12, 21, 22, 23])
        _ = offsets^
        _ = values^
    elif unit == 5:
        var records = List[ConnectivityEdge]()
        records.append(connectivity_edge(2, 1, bitcast[DType.uint64](Float64(1.25)), 2, 8, 3))
        records.append(connectivity_edge(0, 2, UInt64(0x80000000), 2, 4, 3))
        records.append(connectivity_edge(2, 1, bitcast[DType.uint64](Int64(-5)), 1, 8, 3))
        var result = connectivity_pack(records)
        expect(result, [2, 0, 2, 1, 2, 1, 1.25, bitcast[DType.float32](UInt32(0x80000000)), -5])
        _ = records^
    elif unit == 6:
        var pointers: List[Int32] = [0, 0, 0, 0]
        var result = connectivity_compressed(vector(0, 0, 4, 2), vector(0, 0, 4, 1),
            vector(Int(pointers.unsafe_ptr()), 4, 4, 1), 3, False, False)
        expect(result, [])
        _ = pointers^
    elif unit == 7:
        var result = connectivity_coo(vector(0, 0, 4, 1), vector(0, 0, 4, 1), vector(0, 0, 4, 2), 3)
        expect(result, [])
    elif unit == 8:
        var offsets: List[Int64] = [-9223372036854775808, 9223372036854775807]
        var values: List[Float32] = [3, 4]
        var d = vector(Int(values.unsafe_ptr()), 2, 4, 2)
        d.stride1 = 4
        var result = connectivity_dia(vector(Int(offsets.unsafe_ptr()), 2, 8, 1), d, 3)
        expect(result, [])
        _ = offsets^
        _ = values^
    elif unit >= 9 and unit <= 16:
        var data: List[Float32] = [1, 2, 3]
        var indices: List[Int32] = [0, 1, 2]
        var pointers: List[Int32] = [0, 1, 2, 3]
        var d = vector(Int(data.unsafe_ptr()), 3, 4, 2)
        var ix = vector(Int(indices.unsafe_ptr()), 3, 4, 1)
        var pp = vector(Int(pointers.unsafe_ptr()), 4, 4, 1)
        var expected = String("")
        if unit == 9:
            pointers[0] = 1
            expected = "sparse connectivity pointer count or origin is invalid"
        elif unit == 10:
            d.size1 = 2
            pointers[0] = 1
            expected = "sparse connectivity blocks must evenly cover its shape"
        elif unit == 11:
            pp.size0 = 3
            expected = "sparse connectivity pointer count or origin is invalid"
        elif unit == 12:
            indices[0] = 99
            pointers[2] = 99
            expected = "sparse connectivity index exceeds its shape"
        elif unit == 13:
            indices[1] = 99
            pointers[2] = -1
            expected = "sparse connectivity pointers exceed stored data or are not ordered"
        elif unit == 14:
            ix.size0 = 2
            expected = "sparse connectivity coordinate counts differ from stored data"
        elif unit == 15:
            indices[0] = -1
            expected = "AgglomerativeClustering: a connectivity edge is outside [0, n)"
        else:
            ix.size0 = 2
            expected = "sparse connectivity diagonal count differs from stored data"
        var message = String("")
        try:
            if unit <= 13:
                _ = connectivity_compressed(d, ix, pp, 3, False, unit == 10)
            elif unit <= 15:
                _ = connectivity_coo(ix, ix, d, 3)
            else:
                _ = connectivity_dia(ix, d, 3)
        except error:
            message = String(error)
        require(message == expected, "sparse admission priority or error text changed")
        _ = data^
        _ = indices^
        _ = pointers^
    elif unit == 17:
        var result = detached()
        var moved = result^
        expect(moved, [2, 0, 2, 1, 2, 1, 4, 0, -4])
    elif unit == 18:
        var records = List[ConnectivityEdge]()
        var result = connectivity_pack(records)
        expect(result, [])
    else:
        large_case(unit - 19)


def run_checks() raises:
    var failures = List[Int](length=HIERARCHY_CONNECTIVITY_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def run(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except error:
            fp[unit] = 1
            print("FAIL hierarchy_connectivity_plan fixture", unit, String(error))
    host_parallelize(run, HIERARCHY_CONNECTIVITY_FIXTURES)
    var failed = 0
    for value in failures:  # metadata: fixture summaries
        failed += value
    require(failed == 0, "native hierarchy connectivity plans failed " + String(failed) + " fixtures")
    print("PASS hierarchy_connectivity_plan:", HIERARCHY_CONNECTIVITY_FIXTURES, "stored-edge/order/ownership/refusal fixtures")
