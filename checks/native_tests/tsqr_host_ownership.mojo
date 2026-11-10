# SPDX-License-Identifier: Apache-2.0
"""Temporary TSQR owners overlap without replacing a retained factorization."""
from core.host_parallel import host_parallelize
from core.hostptr import f32_ptr
from x_decomp.tsqr_host import ts_factor_host, ts_apply_host, TS_HOST_STATE


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


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
    ts_factor_host(f32_ptr(Int(a.unsafe_ptr())), f32_ptr(Int(a.unsafe_ptr())),
                   f32_ptr(Int(r.unsafe_ptr())), rows, cols, 0, False)
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


def main() raises:
    var a = values(17, 3, 31)
    var r = List[Float32](length=9, fill=0)
    ts_factor_host(f32_ptr(Int(a.unsafe_ptr())), f32_ptr(Int(a.unsafe_ptr())),
                   f32_ptr(Int(r.unsafe_ptr())), 17, 3, 0, True)
    var failures = List[Int](length=32, fill=0)
    var failed = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(failures.unsafe_ptr()))
    def task(fixture: Int) {imm failed}:
        try:
            temporary(fixture)
        except problem:
            print("FAIL tsqr_host_ownership fixture", fixture, String(problem))
            failed.unsafe_store(fixture, 1)
    host_parallelize(task, 32)
    for failure in failures:
        require(failure == 0, "parallel temporary TSQR failed")
    var slot = TS_HOST_STATE.get_or_create_ptr()
    require(slot[].live and slot[].m == 17 and slot[].n == 3, "temporary fit replaced retained TSQR")
    var eye: List[Float32] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    var q = List[Float32](length=51, fill=0)
    ts_apply_host(f32_ptr(Int(eye.unsafe_ptr())), f32_ptr(Int(q.unsafe_ptr())), 17, 3, 3)
    require(not slot[].live, "applied TSQR state was not released")
    for row in range(17):
        for col in range(3):
            var actual = Float64(0)
            for k in range(3):
                actual += Float64(q[row * 3 + k]) * Float64(r[k * 3 + col])
            require(abs(actual - Float64(a[row * 3 + col])) < 0.0001, "retained QR reconstruction mismatch")
    _ = a
    _ = r
    _ = failures
    _ = eye
    _ = q
    print("PASS tsqr_host_ownership: 32 parallel temporary owners and retained factor/apply reconstruction")
