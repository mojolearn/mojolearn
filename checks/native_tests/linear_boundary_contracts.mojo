# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of Mojolearn.
"""Native linear request, planning and storage ownership contracts.

Independent fixtures run through the host-wide host_parallelize pool with
task-local buffers and mock allocation records. No Python imports, objects,
protocols, device kernels, timing or identity checks are involved. The
Python API/protocol suites remain responsible for public admission.
"""
from core.host_parallel import host_parallelize
from core.linear_boundary_types import linear_predict_request_from_spans
from core.linear_fit_types import (
    LinearFitRequest,
    LF_FLAGS, LF_GRAM, LF_RIDGE_RESIDENT, LF_PREP, LF_RIDGE, LF_OLS,
    LF_OLS_RESIDENT, LF_MODE, LF_HAS_OLS_RESIDENT,
    linear_fit_check_spans, ridge_fit_check_spans,
)
from core.linear_fit_run import linear_ridge_run
from core.linear_ols_run import linear_ols_plan
from core.linear_tsqr_run import LinearKit
from core.native_descriptor_abi import NativeDescriptorCall
from checks.native_tests.linear_boundary_fixtures import plan_descriptor, storage_descriptor


def _require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _prediction_spans(fault: Int) raises:
    var source = List[Float64](length=12, fill=1.0)
    var a: List[Int] = [Int(source.unsafe_ptr()), Int(source.unsafe_ptr() + 6), Int(source.unsafe_ptr() + 9)]
    var p: List[Int] = [2, 3, 1]
    var f: List[Float64] = [-2.5]
    var result = List[Int](length=1, fill=0)
    var operation = 0
    var na = 3
    var np = 3
    var nf = 1
    var aa = Int(a.unsafe_ptr())
    var pp = Int(p.unsafe_ptr())
    var ff = Int(f.unsafe_ptr())
    var rr = Int(result.unsafe_ptr())
    if fault == 1:
        operation = -1
    elif fault == 2:
        na -= 1
    elif fault == 3:
        np += 1
    elif fault == 4:
        nf = 0
    elif fault == 5:
        aa = 0
    elif fault == 6:
        pp = 0
    elif fault == 7:
        ff = 0
    elif fault == 8:
        rr = 0
    elif fault >= 9:
        a[fault - 9] = 0
    var refused = False
    try:
        var request = linear_predict_request_from_spans(operation, aa, na, pp, np, ff, nf, rr)
        _require(request.x == a[0] and request.coef == a[1] and request.dst == a[2], "prediction buffer order")
        _require(request.rows == 2 and request.cols == 3 and request.wide, "prediction shape/dtype")
        _require(request.intercept == -2.5, "prediction intercept precision")
    except problem:
        _require(String(problem).find("linear prediction ABI:") >= 0, "unexpected prediction refusal")
        refused = True
    _require(refused == (fault != 0), "prediction malformed spans admitted")
    _ = source
    _ = a
    _ = p
    _ = f
    _ = result


def _fit_spans(unit: Int) raises:
    # Fixture table states the private operation signatures, independently
    # of the span validator. Optional PREP weights get a separate case.
    var operations: List[Int] = [LF_FLAGS, LF_GRAM, LF_RIDGE_RESIDENT, LF_PREP,
                                 LF_RIDGE, LF_OLS, LF_OLS_RESIDENT, LF_MODE]
    var address_counts: List[Int] = [0, 5, 5, 9, 3, 3, 5, 0]
    var integer_counts: List[Int] = [0, 3, 3, 4, 2, 2, 3, 0]
    var float_counts: List[Int] = [0, 1, 1, 0, 1, 0, 1, 0]
    var index = unit // 9
    var fault = unit % 9
    if unit >= 72:
        index = 3
        fault = 0
    var source = List[Float64](length=9, fill=1.0)
    var a = List[Int](length=9, fill=Int(source.unsafe_ptr()))
    var p: List[Int] = [4, 2, 1, 0]
    var f: List[Float64] = [0.25]
    var result = List[Int](length=1, fill=0)
    var operation = operations[index]
    var na = address_counts[index]
    var np = integer_counts[index]
    var nf = float_counts[index]
    var aa = Int(a.unsafe_ptr())
    var pp = Int(p.unsafe_ptr())
    var ff = Int(f.unsafe_ptr())
    var rr = Int(result.unsafe_ptr())
    var expect_refusal = fault > 0
    if fault == 1:
        na += 1
    elif fault == 2:
        np += 1
    elif fault == 3:
        nf += 1
    elif fault == 4:
        rr = 0
    elif fault == 5:
        aa = 0
        expect_refusal = na != 0
    elif fault == 6:
        pp = 0
        expect_refusal = np != 0
    elif fault == 7:
        ff = 0
        expect_refusal = nf != 0
    elif fault == 8:
        a[0] = 0
        expect_refusal = na != 0
    if unit == 72:
        a[2] = 0  # PREP admits an absent weight buffer.
    elif unit == 73:
        operation = -1
        expect_refusal = True
    var refused = False
    try:
        linear_fit_check_spans(operation, aa, na, pp, np, ff, nf, rr)
    except problem:
        _require(String(problem).find("linear fit ABI:") >= 0, "unexpected fit refusal")
        refused = True
    _require(refused == expect_refusal, "fit operation span admission")
    _ = source
    _ = a
    _ = p
    _ = f
    _ = result


def _ridge_spans(fault: Int) raises:
    var source = List[Float32](length=8, fill=1.0)
    var a: List[Int] = [Int(source.unsafe_ptr()), Int(source.unsafe_ptr() + 4), Int(source.unsafe_ptr() + 6)]
    var p: List[Int] = [2, 2, 1, 0]
    var f: List[Float64] = [0.5]
    var result = List[Int](length=1, fill=0)
    var operation = 0
    var na = 3
    var np = 4
    var nf = 1
    var aa = Int(a.unsafe_ptr())
    var pp = Int(p.unsafe_ptr())
    var ff = Int(f.unsafe_ptr())
    var rr = Int(result.unsafe_ptr())
    if fault == 1:
        operation = 1
    elif fault == 2:
        na += 1
    elif fault == 3:
        np -= 1
    elif fault == 4:
        nf = 0
    elif fault == 5:
        aa = 0
    elif fault == 6:
        pp = 0
    elif fault == 7:
        ff = 0
    elif fault == 8:
        rr = 0
    elif 9 <= fault < 12:
        a[fault - 9] = 0
    elif fault >= 12:
        p[(fault - 12) // 2] = 0 if fault % 2 == 0 else -1
    var refused = False
    try:
        ridge_fit_check_spans(operation, aa, na, pp, np, ff, nf, rr)
    except problem:
        _require(String(problem).find("x_linear") >= 0, "unexpected Ridge span refusal")
        refused = True
    _require(refused == (fault != 0), "Ridge shape/span admission")
    _ = source
    _ = a
    _ = p
    _ = f
    _ = result


def _ridge_requests(fault: Int) raises:
    var source = List[Float32](length=16, fill=1.0)
    var address = Int(source.unsafe_ptr())
    var request = LinearFitRequest(address, address + 32, 0, address + 48, address + 56, 4, 2, True, 0.5)
    if fault == 0:
        request.rows = 0
    elif fault == 1:
        request.cols = 0
    elif fault == 2:
        request.alpha = -1.0
    elif fault == 3:
        request.x = 0
    elif fault == 4:
        request.y = 0
    elif fault == 5:
        request.coef = 0
    else:
        request.mean = 0
    var refused = False
    try:
        _ = linear_ridge_run(plan_descriptor[False], plan_descriptor[False], request)
    except problem:
        var message = String(problem)
        if fault < 2:
            refused = message.find("rows and cols must be positive") >= 0
        elif fault == 2:
            refused = message.find("alpha must be non-negative") >= 0
        else:
            refused = message.find("null fit buffer") >= 0
    _require(refused, "invalid typed Ridge request must fail before numerical dispatch")
    for value in source:
        _require(value == 1.0, "refused Ridge request wrote borrowed buffers")


def _plans(unit: Int) raises:
    # Tall/square and factorization-width boundaries; these are API shape
    # limits, not benchmark dimensions. Resident capability supersedes TSQR.
    var rows: List[Int] = [0, 1, 2, 2, 512, 513, 513, 8]
    var cols: List[Int] = [1, 1, 1, 2, 511, 511, 512, 0]
    var wants_tsqr: List[Bool] = [False, False, True, False, True, True, False, False]
    var index = unit % 8
    var resident = unit >= 8
    var refused = False
    try:
        var entry: NativeDescriptorCall = plan_descriptor[False]
        if resident:
            entry = plan_descriptor[True]
        var plan = linear_ols_plan(entry, rows[index], cols[index], "fixture OLS")
        _require(plan.tsqr == (wants_tsqr[index] and not resident), "OLS shape/capability plan")
        _require(plan.flags == (LF_HAS_OLS_RESIDENT if resident else 0), "OLS capability retained")
        _require(plan.numeric_mode == (17 if plan.tsqr else 0), "OLS mode only queried for TSQR")
    except problem:
        _require(String(problem).find("minimum of 2") >= 0, "unexpected OLS plan refusal")
        refused = True
    _require(refused == (index < 2), "OLS sample count refusal")


def _host_ownership(unit: Int) raises:
    var owner: List[Float32] = [-99, 3, -5, 7, -99]
    var kit = LinearKit(storage_descriptor[False])
    _require(not kit.resident, "host capabilities")
    if unit == 0:
        var original = kit.borrowed(Int(owner.unsafe_ptr() + 1), 1, 3)
        var view = kit.transpose(original)
        _require(kit.matrices[view].rows == 3 and kit.matrices[view].cols == 1, "transpose view shape")
        _require(kit.address(view) == Int(owner.unsafe_ptr() + 1), "borrowed transpose must share storage")
        var ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=kit.address(view))
        ptr[1] = 19
        kit.release_host(original)
        kit.release_host(original)
        _require(ptr[0] == 3 and ptr[1] == 19 and ptr[2] == 7, "view survives original release")
        kit.release_host(view)
        _require(owner[2] == 19 and owner[0] == -99 and owner[4] == -99, "borrowed owner survives view release")
    elif unit == 1:
        var original = kit.constant(7.0, 3, 1)
        var view = kit.transpose(original)
        var slot = kit.matrices[view].host
        kit.release_host(original)
        var ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=kit.address(view))
        _require(ptr[0] == 7 and ptr[1] == 7 and ptr[2] == 7, "owned view retains scratch")
        kit.release_host(view)
        _require(len(kit.storage[slot].words) == 0 and kit.storage[slot].address == 0,
                 "last owned view must release native storage")
        _require(kit.storage[slot].references == 0, "released view reference count")
    elif unit == 2:
        var values: List[Float32] = [1, 2, 3, 4, 5, 6]
        var original = kit.borrowed(Int(values.unsafe_ptr()), 2, 3)
        var selected = kit.columns(original, 1, 3)
        kit.release_host(original)
        values[1] = -77
        var ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=kit.address(selected))
        _require(ptr[0] == 2 and ptr[1] == 3 and ptr[2] == 5 and ptr[3] == 6,
                 "column selection owns an independent result")
        kit.release_host(selected)
        _ = values
    else:
        var first = kit.shared_zero()
        var second = kit.shared_zero()
        _require(first == second and kit.word(first) == 0.0, "shared zero retained and reused")
        var matrix = kit.zeros(2, 3)
        var same = kit.zeros(2, 3)
        var row = kit.zeros(1, 3)
        var column = kit.zeros(2, 1)
        var wrong = kit.zeros(3, 2)
        _require(kit.broadcast(matrix, same) == 0, "equal shape broadcast")
        _require(kit.broadcast(matrix, row) == 1, "row broadcast")
        _require(kit.broadcast(matrix, column) == 2, "column broadcast")
        _require(kit.broadcast(matrix, first) == 3, "scalar broadcast")
        var refused = False
        try:
            _ = kit.broadcast(matrix, wrong)
        except problem:
            refused = String(problem).find("invalid internal broadcast") >= 0
        _require(refused, "incompatible shape must be refused")
    kit.close()
    _ = owner
    _ = kit


def _device_ownership(unit: Int) raises:
    var kit = LinearKit(storage_descriptor[True])
    _require(kit.resident, "resident capabilities")
    var first = kit.device_output(2, 3)
    var second = kit.device_output(1, 3)
    var h1 = kit.matrices[first].device
    var h2 = kit.matrices[second].device
    var p1 = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=h1)
    var p2 = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=h2)
    try:
        if unit == 0:
            var view = kit.transpose(second)
            _require(kit.matrices[view].device == h2, "device vector transpose shares its handle")
            _require(kit.matrices[view].rows == 3 and kit.matrices[view].cols == 1, "device view shape")
        elif unit == 1:
            p2[2] = 1
        else:
            var address = kit.address(second)
            var ptr = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=address)
            _require(ptr[0] == 7 and ptr[1] == 8 and ptr[2] == 9, "download retains native host output")
            _require(kit.address(second) == address, "downloaded host storage reused")
            kit.release_host(second)
        var refused = False
        try:
            kit.close()
        except problem:
            refused = String(problem).find("injected native release failure") >= 0
        _require(refused == (unit == 1), "release error propagated")
        _require(p1[0] == 1 and p2[0] == 1, "all handles released once, including after failure")
        kit.close()
        _require(p1[0] == 1 and p2[0] == 1, "close is idempotent")
    finally:
        p1.free()
        p2.free()
        _ = kit


def _upload_failure() raises:
    var owner: List[Float32] = [3, -5, 7]
    var kit = LinearKit(storage_descriptor[True, True])
    var original = kit.borrowed(Int(owner.unsafe_ptr()), 1, 3)
    var refused = False
    try:
        _ = kit.device(original)
    except problem:
        refused = String(problem).find("injected native upload failure") >= 0
    _require(refused, "upload failure propagated")
    _require(len(kit.allocations) == 1, "failed upload retains allocation for cleanup")
    var record = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=kit.allocations[0])
    try:
        _require(kit.address(original) == Int(owner.unsafe_ptr()), "failed upload retains borrowed host view")
        _require(owner[0] == 3 and owner[1] == -5 and owner[2] == 7, "failed upload preserves input")
        kit.close()
        _require(record[0] == 1 and record[3] == 1, "failed transfer allocation released exactly once")
    finally:
        record.free()
        _ = owner
        _ = kit


def main() raises:
    comptime FIXTURES = 133
    var failed = List[Int](length=FIXTURES, fill=0)
    var failures_ptr = failed.unsafe_ptr()
    def task(unit: Int) {imm failures_ptr}:
        try:
            if unit < 12:
                _prediction_spans(unit)
            elif unit < 86:
                _fit_spans(unit - 12)
            elif unit < 102:
                _ridge_spans(unit - 86)
            elif unit < 118:
                _plans(unit - 102)
            elif unit < 122:
                _host_ownership(unit - 118)
            elif unit < 125:
                _device_ownership(unit - 122)
            elif unit == 125:
                _upload_failure()
            else:
                _ridge_requests(unit - 126)
        except problem:
            print("FAIL linear_boundary_contracts fixture", unit, String(problem))
            failures_ptr.unsafe_store(unit, 1)
    host_parallelize(task, FIXTURES)
    var failures = 0
    for value in failed:
        failures += value
    _require(failures == 0, "linear boundary failed " + String(failures) + " of " + String(FIXTURES) + " fixtures")
    print("PASS linear_boundary_contracts:", FIXTURES, "native fixtures, parallel")
