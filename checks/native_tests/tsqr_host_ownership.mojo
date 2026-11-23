# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Temporary TSQR factorizations overlap with a kept one, and every kept factorization is its caller's own owner
(lane linalg-owner, 2026-11-22: no process slot)."""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from core.host_parallel import host_parallelize
from x_decomp.cells import F32Ptr
from x_decomp.tsqr_host import ts_factor_host, ts_apply_host, _TsHost


def values(rows: Int, cols: Int, seed: Int) -> List[Float32]:
    var data = List[Float32](length=rows * cols, fill=0)
    var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(data.unsafe_ptr()))
    def fill(row: Int) {imm dst, imm cols, imm seed}:
        for col in range(cols):
            dst.unsafe_store(row * cols + col, Float32((row * 7 + col * 11 + seed) % 29 - 14) * 0.125)
    host_parallelize(fill, rows)
    return data^


def temporary(fixture: Int) raises:
    var rows = 19 + fixture * 13
    var cols = 2 + fixture % 5
    var a = values(rows, cols, fixture)
    var r = List[Float32](length=cols * cols, fill=0)
    var none = ts_factor_host(F32Ptr(unsafe_from_address=Int(a.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(a.unsafe_ptr())),
                              F32Ptr(unsafe_from_address=Int(r.unsafe_ptr())), rows, cols, 0, False)
    require(none == 0, "a temporary TSQR returns no owner")
    # Independent Gram equality: R'R equals A'A, without choosing QR signs.
    # Fixtures run across the pool; each oracle's tiny dot folds are ordered.
    for i in range(cols):
        for j in range(cols):
            var expected = Float64(0)
            var actual = Float64(0)
            for row in range(rows):
                expected += Float64(a[row * cols + i]) * Float64(a[row * cols + j])
            for row in range(cols):
                actual += Float64(r[row * cols + i]) * Float64(r[row * cols + j])
            require(abs(actual - expected) <= 0.002 + 0.0001 * abs(expected), "temporary TSQR Gram mismatch")
    _ = a
    _ = r


def test_retained_and_temporary_owners() raises:
    """A kept fit, 32 parallel temporary fits beside it, then the kept owner's apply (which frees it) and the
    reconstruction."""
    var a = values(17, 3, 31)
    var r = List[Float32](length=9, fill=0)
    var owner = ts_factor_host(F32Ptr(unsafe_from_address=Int(a.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(a.unsafe_ptr())),
                               F32Ptr(unsafe_from_address=Int(r.unsafe_ptr())), 17, 3, 0, True)
    require(owner != 0, "a kept TSQR returns its owner")
    check_cases("parallel temporary TSQR", 32, temporary)
    var held = MutPointer[_TsHost, MutUntrackedOrigin](unsafe_from_address=owner)
    require(held[].live and held[].m == 17 and held[].n == 3, "temporary fit changed the kept TSQR owner")
    var eye: List[Float32] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    var q = List[Float32](length=51, fill=0)
    ts_apply_host(owner, F32Ptr(unsafe_from_address=Int(eye.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(q.unsafe_ptr())), 17, 3, 3)
    for row in range(17):
        for col in range(3):
            var actual = Float64(0)
            for k in range(3):
                actual += Float64(q[row * 3 + k]) * Float64(r[k * 3 + col])
            require(abs(actual - Float64(a[row * 3 + col])) < 0.0001, "retained QR reconstruction mismatch")
    _ = a
    _ = r
    _ = eye
    _ = q


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
