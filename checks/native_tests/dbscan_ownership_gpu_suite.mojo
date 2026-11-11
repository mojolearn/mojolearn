# SPDX-License-Identifier: Apache-2.0
"""Resident reference copies and leases across release, without concurrent GPU work.

Three independent lifecycle graphs run in order on one physical GPU/shared
context. Overlapping leases are metadata owners; all copies/queries remain
ordered. This does not clear the recorded shared-context submission failure.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.host_parallel import host_parallelize
from core.dbscan_native_api import DBSCANNativeDriver, DBSCANReferenceRequest, DBSCANResidentQuery, dbscan_native_v1
from core.dbscan_native_device import DBSCANDeviceKernels
from core.labeled_reference_resident import LabeledReferenceLease, labeled_reference_lease
from checks.native_tests.dbscan_native_cases import require, integer_guards, DBSCAN_I32_GUARD

comptime DBSCAN_OWNERSHIP_FIXTURES = 3


def downloaded(lease: LabeledReferenceLease, expected: List[Float32], keys: List[Int32], labels: List[Int32]) raises:
    require(lease.n_refs == 3 and lease.n_features == 2 and lease.metric == 0, "DBSCAN retained native lease metadata")
    var host_refs = lease.ctx.enqueue_create_host_buffer[DType.float32](6)
    var host_keys = lease.ctx.enqueue_create_host_buffer[DType.int32](3)
    var host_labels = lease.ctx.enqueue_create_host_buffer[DType.int32](3)
    lease.ctx.enqueue_copy(dst_ptr=host_refs.unsafe_ptr(), src_buf=lease.refs)
    lease.ctx.enqueue_copy(dst_ptr=host_keys.unsafe_ptr(), src_buf=lease.keys)
    lease.ctx.enqueue_copy(dst_ptr=host_labels.unsafe_ptr(), src_buf=lease.labels)
    lease.ctx.synchronize()
    var failures = List[Int](length=12, fill=0)
    var fp = failures.unsafe_ptr()
    var rp = host_refs.unsafe_ptr()
    var kp = host_keys.unsafe_ptr()
    var lp = host_labels.unsafe_ptr()
    var ep = expected.unsafe_ptr()
    var ek = keys.unsafe_ptr()
    var el = labels.unsafe_ptr()
    def cell(index: Int) {imm fp, imm rp, imm kp, imm lp, imm ep, imm ek, imm el}:
        var okay = True
        if index < 6:
            okay = rp[index] == ep[index]
        elif index < 9:
            okay = kp[index - 6] == ek[index - 6]
        else:
            okay = lp[index - 9] == el[index - 9]
        fp.unsafe_store(index, Int(not okay))
    host_parallelize(cell, 12)
    for failure in failures:
        require(failure == 0, "DBSCAN lease retains original reference/key/label storage after registry release")
    _ = host_refs
    _ = host_keys
    _ = host_labels
    _ = expected
    _ = keys
    _ = labels
    _ = lease


def query(driver: DBSCANNativeDriver, handle: Int, replacement: Bool) raises:
    var queries: List[Float32] = [1, 0, 2, 2, 100, 100]
    var output = List[Int32](length=5, fill=DBSCAN_I32_GUARD)
    try:
        require(driver.resident(DBSCANResidentQuery(Int(queries.unsafe_ptr()), Int(output.unsafe_ptr()) + 4,
            handle, 3, 2, 0, True, 1.0)) == 0, "DBSCAN owned reference query return")
        require(output[1] == Int32(15 if replacement else 5) and output[2] == Int32(19 if replacement else 9)
            and output[3] == Int32(-1), "DBSCAN analytic tie/nearest/noise with independently owned labels")
        integer_guards(output)
    finally:
        _ = queries^
        _ = output^


def missing(handle: Int) raises:
    var refused = False
    try:
        var unexpected = labeled_reference_lease(handle)
        _ = unexpected
    except error:
        refused = String(error) == "unknown or released resident labeled reference handle"
    require(refused, "DBSCAN registry release removes lookup without revoking active native leases")


def fixture(unit: Int) raises:
    var driver = DBSCANNativeDriver(dbscan_native_v1[DBSCANDeviceKernels])
    var refs: List[Float32] = [0, 0, 2, 0, 2, 2]
    var keys: List[Int32] = [19, 7, 11]
    var labels: List[Int32] = [3, 5, 9]
    var original_refs = refs.copy()
    var original_keys = keys.copy()
    var original_labels = labels.copy()
    var next_labels: List[Int32] = [13, 15, 19]
    var handle = -1
    var replacement = -1
    try:
        handle = driver.prepare(DBSCANReferenceRequest(Int(refs.unsafe_ptr()), Int(keys.unsafe_ptr()),
            Int(labels.unsafe_ptr()), 3, 2, 0))
        require(handle >= 0, "DBSCAN native reference owner was minted")
        var first = labeled_reference_lease(handle)
        var second = labeled_reference_lease(handle)
        var minted = handle
        try:
            if unit == 2:
                replacement = driver.prepare(DBSCANReferenceRequest(Int(refs.unsafe_ptr()), Int(keys.unsafe_ptr()),
                    Int(next_labels.unsafe_ptr()), 3, 2, 0))
                require(replacement >= 0 and replacement != handle, "DBSCAN replacement has an independent registry identity")
            var rp = refs.unsafe_ptr()
            var kp = keys.unsafe_ptr()
            var lp = labels.unsafe_ptr()
            def poison(index: Int) {imm rp, imm kp, imm lp}:
                rp.unsafe_store(index, Float32(-100))
                if index < 3:
                    kp.unsafe_store(index, Int32(-100))
                    lp.unsafe_store(index, Int32(-100))
            host_parallelize(poison, 6)
            if unit == 0:
                query(driver, handle, False)
            var owned = handle
            handle = -1
            require(driver.release(owned) == 0, "DBSCAN native registry release")
            missing(minted)
            downloaded(first, original_refs, original_keys, original_labels)
            downloaded(second, original_refs, original_keys, original_labels)
            if unit == 2:
                query(driver, replacement, True)
                var next_lease = labeled_reference_lease(replacement)
                downloaded(next_lease, original_refs, original_keys, next_labels)
                _ = next_lease
        finally:
            _ = second^
            _ = first^
    finally:
        try:
            if handle >= 0:
                _ = driver.release(handle)
        finally:
            try:
                if replacement >= 0:
                    _ = driver.release(replacement)
            finally:
                _ = refs^
                _ = keys^
                _ = labels^
                _ = original_refs^
                _ = original_keys^
                _ = original_labels^
                _ = next_labels^


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(DBSCAN_OWNERSHIP_FIXTURES):
        try:
            fixture(unit)
        except error:
            failed += 1
            print("FAIL dbscan_ownership_gpu fixture", unit, String(error))
    require(failed == 0, "dbscan_ownership_gpu failed " + String(failed) + " fixtures")
    print("PASS dbscan_ownership_gpu:", DBSCAN_OWNERSHIP_FIXTURES, "native reference lease fixtures")
