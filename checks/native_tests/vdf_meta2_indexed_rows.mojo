# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE INDEXED ROWS and THE GATHERED ROWS of the meta member (lane vdf-meta2, core/ensemble_member.mojo), in a
process with no Python.

A meta-estimator in a search (or in a member set that draws rows) is handed the caller's whole X and a row index.
The voting, one-vs-rest and multi-output kinds hand the index (and the caller's resident scope and X's pin) on to
every nested member, which then reads its fold rows where the caller holds X (on the device inside `resident()`);
every other kind gathers the rows once at its entry. Both read the same rows in the same order as the gather the
caller made before, so the bytes are the gathered path's.

    case 0   `_gather_view` (THE GATHERED ROWS) == the oracle host gather, word for word, for a float32 X of 3
             columns and a float64 column, under int32 and int64 row indexes (repeats and out-of-order rows)
    case 1   `_kind_mask` (route-exact): ROW_INDEX only for the indexed kinds whose every nested kind reads one;
             SCOPE only beside it and only when a nested kind keeps X resident; never for stacking, calibration,
             bagging; the family union carries ROW_INDEX (a member set hands the index instead of gathering)

Fail-proof edits (each must fail the named case): in `_gather_bytes` read the index as int32 always (case 0, the
int64 index); in `_gather_view` size the store by rows only (case 0); in `_kind_mask` test `_any` for ROW_INDEX on a
vote (case 1); drop MEMBER_CAN_ROW_INDEX from `_UNION` (case 1).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require
from core.member_abi import (
    MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT,
)
from core.ensemble_member import (
    _gather_view, _kind_mask, _UNION, META_VOTE_CLASSIFIER, META_VOTE_REGRESSOR, META_STACK_CLASSIFIER, META_OVR,
    META_CALIBRATED, META_MULTI_REGRESSOR, META_BAG_CLASSIFIER,
)

comptime N = 23
comptime D = 3


def _rows() -> List[Int]:
    var rows: List[Int] = [7, 0, 22, 7, 13, 1, 22, 5]
    return rows^


def _gather_case() raises:
    var x = List[Float32](length=N * D, fill=0.0)
    var y = List[Float64](length=N, fill=0.0)
    for i in range(N):  # small-loop(fixture: one 23 x 3 block and a column)
        y[i] = Float64(i) * 1.5 - 7.25
        for j in range(D):
            x[i * D + j] = Float32(((i * (j + 5) * 11) % 19) - 9) * Float32(0.125) + Float32(j)
    var rows = _rows()
    var n = len(rows)
    var i32 = List[Int32](length=n, fill=Int32(0))
    var i64 = List[Int64](length=n, fill=Int64(0))
    for i in range(n):  # small-loop(fixture: the row index in both widths)
        i32[i] = Int32(rows[i])
        i64[i] = Int64(rows[i])
    for wide in range(2):  # small-loop(fixture: int32 then int64 rows)
        var index = MemberMatrix(Int(i32.unsafe_ptr()), n, 1, MEMBER_I32)
        if wide == 1:
            index = MemberMatrix(Int(i64.unsafe_ptr()), n, 1, MEMBER_I64)
        var gx = List[Int64]()
        var gy = List[Int64]()
        var vx = _gather_view(MemberMatrix(Int(x.unsafe_ptr()), N, D, MEMBER_F32), index, gx)
        var vy = _gather_view(MemberMatrix(Int(y.unsafe_ptr()), N, 1, MEMBER_F64), index, gy)
        require(vx.rows == n and vx.cols == D and vx.dtype == MEMBER_F32, "gathered X view shape")
        require(vy.rows == n and vy.cols == 1 and vy.dtype == MEMBER_F64, "gathered y view shape")
        var px = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=vx.data)
        var py = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=vy.data)
        for i in range(n):  # small-loop(oracle: the host gather word for word)
            var r = rows[i]
            require(py[i] == y[r], "y row " + String(i) + " (wide " + String(wide) + ")")
            for j in range(D):
                require(px[i * D + j] == x[r * D + j], "X row " + String(i) + " col " + String(j)
                        + " (wide " + String(wide) + ")")
        _ = len(gx)
        _ = len(gy)
    _ = len(x)
    _ = len(y)
    _ = len(i32)
    _ = len(i64)


def _has(mask: Int, bit: Int) -> Bool:
    return (mask & bit) != 0


def _mask_case() raises:
    var plain = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT
    var indexed = plain | MEMBER_CAN_ROW_INDEX
    var resident = indexed | MEMBER_CAN_SCOPE
    var both: List[Int] = [indexed, resident]
    var one_plain: List[Int] = [resident, plain]
    var only_indexed: List[Int] = [indexed]
    var only_resident: List[Int] = [resident]
    var m = _kind_mask(META_VOTE_CLASSIFIER, True, both)
    require(_has(m, MEMBER_CAN_ROW_INDEX) and _has(m, MEMBER_CAN_SCOPE), "vote over index readers: ROW_INDEX and SCOPE")
    m = _kind_mask(META_VOTE_REGRESSOR, False, one_plain)
    require(not _has(m, MEMBER_CAN_ROW_INDEX) and not _has(m, MEMBER_CAN_SCOPE),
            "vote with a member that reads no index: neither bit")
    m = _kind_mask(META_OVR, False, only_indexed)
    require(_has(m, MEMBER_CAN_ROW_INDEX) and not _has(m, MEMBER_CAN_SCOPE), "OvR over an index reader: ROW_INDEX only")
    m = _kind_mask(META_MULTI_REGRESSOR, False, only_resident)
    require(_has(m, MEMBER_CAN_ROW_INDEX) and _has(m, MEMBER_CAN_SCOPE), "multi-output over a resident reader: both")
    var stack: List[Int] = [resident, resident, resident]
    m = _kind_mask(META_STACK_CLASSIFIER, False, stack)
    require(not _has(m, MEMBER_CAN_ROW_INDEX) and not _has(m, MEMBER_CAN_SCOPE), "stacking gathers: neither bit")
    m = _kind_mask(META_CALIBRATED, False, only_resident)
    require(not _has(m, MEMBER_CAN_ROW_INDEX) and not _has(m, MEMBER_CAN_SCOPE), "calibration gathers: neither bit")
    m = _kind_mask(META_BAG_CLASSIFIER, False, only_resident)
    require(not _has(m, MEMBER_CAN_ROW_INDEX), "bagging gathers: no ROW_INDEX")
    require(_has(_UNION, MEMBER_CAN_ROW_INDEX), "the family union carries ROW_INDEX")


def _case(unit: Int) raises:
    if unit == 0:
        _gather_case()
    else:
        _mask_case()


def test_vdf_meta2_indexed_rows() raises:
    check_cases_serial("vdf_meta2_indexed_rows", 2, _case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
