# SPDX-License-Identifier: Apache-2.0
"""Native retained KDE leases, ordered GPU work and sequential pool reuse.

The original 131x7 weighted fixture and query lengths9/3 retain the existing
allocation4/reuse1 expectation. Overlapping owners are acquired by metadata
operations only; GPU submissions remain ordered on one physical GPU. This
does not test or clear the separate shared-context concurrent-submit failure.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU
from core.host_parallel import host_parallelize
from core.kde_native_api import KDENativeDriver, KDEPrepareRequest, KDEResidentQuery, kde_native_v1
from core.kde_native_device import KDEDeviceKernels
from kde.resident_fit import KdeFitLease, _kde_fit_lease, _kde_fit_return
from kde.impl.chunk_workspace import KDE_CHUNK_POOL_ON
from kde.checks.kde_fixture import train_fixture, query_fixture, weight_fixture
from checks.native_tests.pca_native_cases import require, guards, PCA_TEST_GUARD

comptime KDE_OWNERSHIP_FIXTURES = 3


def poison(mut values: List[Float32], value: Float32):
    var vp = values.unsafe_ptr()
    def cell(index: Int) {imm vp, imm value}:
        vp.unsafe_store(index, value)
    host_parallelize(cell, len(values))
    _ = values


def downloaded(lease: KdeFitLease, training: Bool, expected: List[Float32]) raises:
    var count = len(expected)
    var host = lease.ctx.enqueue_create_host_buffer[DType.float32](count)
    if training:
        lease.ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=lease.train)
    else:
        lease.ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=lease.weights)
    lease.ctx.synchronize()
    var failures = List[Int](length=count, fill=0)
    var fp = failures.unsafe_ptr()
    var hp = host.unsafe_ptr()
    var ep = expected.unsafe_ptr()
    def cell(index: Int) {imm fp, imm hp, imm ep}:
        fp.unsafe_store(index, Int(hp[index] != ep[index]))
    host_parallelize(cell, count)
    for failure in failures:
        require(failure == 0, "KDE active native lease retains original immutable buffer contents")
    _ = host
    _ = expected
    _ = lease


def warm(driver: KDENativeDriver, handle: Int, train: List[Float32]) raises:
    for step in range(2):  # Original ordered queries9 then3; one GPU job.
        var rows = 9 if step == 0 else 3
        var query = query_fixture(train, 131, rows, 7, 37 + rows)
        var output = List[Float32](length=rows + 2, fill=PCA_TEST_GUARD)
        var status = List[Int32](length=3, fill=Int32(23117))
        try:
            _ = driver.resident(KDEResidentQuery(handle, Int(query.unsafe_ptr()), Int(output.unsafe_ptr()) + 4,
                Int(status.unsafe_ptr()) + 4, rows, 7, 0.25, False, String("gaussian"), String("euclidean")))
            guards(output, "KDE workspace warm query")
            require(status[0] == Int32(23117) and status[1] == Int32(0) and status[2] == Int32(23117),
                "KDE workspace query finite-status bounds")
        finally:
            _ = query
            _ = output
            _ = status
    _ = train


def fixture(unit: Int) raises:
    var driver = KDENativeDriver(kde_native_v1[KDEDeviceKernels])
    var train = train_fixture(131, 7, 29)
    var weights = weight_fixture(131, 41)
    var expected_train = train.copy()
    var expected_weights = weights.copy()
    var handle = -1
    try:
        handle = driver.prepare(KDEPrepareRequest(Int(train.unsafe_ptr()), Int(weights.unsafe_ptr()),
            131, 7, 0.25, True, True, String("gaussian"), String("euclidean")))
        warm(driver, handle, train)
        var minted = handle
        var first = _kde_fit_lease(handle)
        try:
            comptime if KDE_CHUNK_POOL_ON:
                if not first.partial_pool:
                    raise Error("KDE retained workspace missing after two eligible queries")
                require(first.partial_pool.value().allocations == 4 and first.partial_pool.value().reuse_calls == 1,
                    "KDE original sequential allocation4/reuse1 contract")
                require(first.partial_pool.value().retained_bytes() <= first.partial_pool.value().budget_bytes,
                    "KDE retained workspace respects its budget")
            var second = _kde_fit_lease(handle)
            try:
                require(not second.partial_pool, "KDE active leases never alias one cached workspace")
                require(first.n_train == 131 and second.n_train == 131 and first.n_features == 7
                    and second.n_features == 7 and first.has_weights and second.has_weights,
                    "KDE active leases retain complete immutable metadata")
                if unit == 0:
                    poison(train, Float32(-99))
                    poison(weights, Float32(77))
                    driver.release(handle)
                    handle = -1
                    downloaded(first, True, expected_train)
                    downloaded(first, False, expected_weights)
                    downloaded(second, True, expected_train)
                    downloaded(second, False, expected_weights)
                    var refused = False
                    try:
                        var unexpected = _kde_fit_lease(minted)
                        _kde_fit_return(minted, unexpected, False)
                        _ = unexpected
                    except error:
                        refused = String(error) == "unknown or released resident KDE fit handle"
                    require(refused, "KDE release removes registry lookup while active owners survive")
                else:
                    _kde_fit_return(handle, first, unit == 1)
                    require(not first.partial_pool, "KDE return transfers or discards its workspace once")
                    var next_lease = _kde_fit_lease(handle)
                    try:
                        if unit == 1:
                            comptime if KDE_CHUNK_POOL_ON:
                                if not next_lease.partial_pool:
                                    raise Error("KDE clean returned workspace was not retained")
                                require(next_lease.partial_pool.value().allocations == 4
                                    and next_lease.partial_pool.value().reuse_calls == 1,
                                    "KDE clean return keeps the existing workspace and counters")
                            _kde_fit_return(handle, next_lease, True)
                        else:
                            require(not next_lease.partial_pool, "KDE unclean workspace is never recached")
                    finally:
                        _kde_fit_return(minted, next_lease, False)
                        _ = next_lease
                    downloaded(second, True, expected_train)
                    downloaded(second, False, expected_weights)
            finally:
                _kde_fit_return(minted, second, False)
                _ = second
        finally:
            _kde_fit_return(minted, first, False)
            _ = first
    finally:
        try:
            if handle >= 0:
                driver.release(handle)
        finally:
            _ = train
            _ = weights
            _ = expected_train
            _ = expected_weights


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var failed = 0
    for unit in range(KDE_OWNERSHIP_FIXTURES):
        try:
            fixture(unit)
        except error:
            failed += 1
            print("FAIL kde_ownership_gpu fixture", unit, String(error))
    require(failed == 0, "kde_ownership_gpu failed " + String(failed) + " fixtures")
    print("PASS kde_ownership_gpu:", KDE_OWNERSHIP_FIXTURES, "native retained lease/workspace lifecycle fixtures")
