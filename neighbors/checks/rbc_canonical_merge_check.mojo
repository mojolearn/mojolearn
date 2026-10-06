# SPDX-License-Identifier: Apache-2.0
"""Exact integer order, duplicates, ragged bounds, dense rows and grid.x scale."""
from std.sys import argv
from max.gpu.host import DeviceContext
from checks.numerics import PIN_CROSS_VENDOR
from neighbors.checks.ball_cover_canonical_order import rbc_canonicalize_row_order


def check(ctx: DeviceContext, sizes: List[Int], duplicates: Bool, extremes: Bool) raises:
    var rows = len(sizes)
    var nnz = 0
    for n in sizes:
        nnz += n
    var hi = ctx.enqueue_create_host_buffer[DType.int32](rows + 1)
    var hx = ctx.enqueue_create_host_buffer[DType.int32](nnz + 17)
    var start = 0
    hi[0] = Int32(0)
    for row in range(rows):
        var n = sizes[row]
        for p in range(n):
            var value = (n - 1 - p + row * 7) % n
            if duplicates:
                value //= 3
            elif extremes:
                if value == 0:
                    value = -2147483648
                elif value == n - 1:
                    value = 2147483647
                else:
                    value -= n // 2
            hx[start + p] = Int32(value)
        start += n
        hi[row + 1] = Int32(start)
    for p in range(nnz, nnz + 17):
        hx[p] = Int32(-123456789)
    var ia = ctx.enqueue_create_buffer[DType.int32](rows + 1)
    var ja = ctx.enqueue_create_buffer[DType.int32](nnz + 17)
    ctx.enqueue_copy(dst_buf=ia, src_ptr=hi.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=ja, src_ptr=hx.unsafe_ptr())
    ctx.synchronize()
    rbc_canonicalize_row_order(ctx, ia, ja, rows, nnz)
    ctx.enqueue_copy(dst_ptr=hx.unsafe_ptr(), src_buf=ja)
    ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=ia)
    ctx.synchronize()
    start = 0
    for row in range(rows):
        if hi[row] != Int32(start):
            raise Error("CSR row boundary changed")
        var n = sizes[row]
        for p in range(n):
            var expected = p
            if duplicates:
                expected //= 3
            elif extremes:
                if p == 0:
                    expected = -2147483648
                elif p == n - 1:
                    expected = 2147483647
                else:
                    expected -= n // 2
            if hx[start + p] != Int32(expected):
                raise Error("canonical value mismatch row=" + String(row) + " p=" + String(p))
        start += n
    if hi[rows] != Int32(nnz):
        raise Error("CSR terminal boundary changed")
    for p in range(nnz, nnz + 17):
        if hx[p] != Int32(-123456789):
            raise Error("wrote beyond live CSR prefix")
    print("CANON_CASE_PASS rows=" + String(rows) + " nnz=" + String(nnz)
          + " duplicates=" + String(duplicates) + " extremes=" + String(extremes))


def main() raises:
    if not PIN_CROSS_VENDOR:
        raise Error("test requires IDENTICAL")
    var ctx = DeviceContext()
    var sizes: List[Int] = [0, 1, 2, 31, 255, 256, 257, 513, 1025]
    check(ctx, sizes, False, False)
    if len(argv()) > 1:
        print("CANON_LEGACY_EQUALITY_PASS")
        return
    check(ctx, [0, 1, 2, 31, 255, 256], True, False)
    check(ctx, sizes, True, False)
    check(ctx, sizes, False, True)
    check(ctx, [0, 65537, 0, 131071, 257], False, False)
    check(ctx, [1048577], True, False)
    check(ctx, List[Int](length=70000, fill=2), True, False)
    check(ctx, [0, 0], False, False)
    print("RBC_CANONICAL_MERGE_PASS cases=8")
