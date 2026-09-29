# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHY THE 512-THREAD TILES NEVER RAN on the M2 Pro (job 1, 19ad540d3):
`f4.sg16.kb16` and `f3.sg16.kb16`, 4 x 4 simdgroups, left every poisoned
cell of every case unwritten, and nothing reported it.

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_int15_apple_tuned_threads_probe.mojo
    MTL_DEBUG_LAYER=1 ... the same, under Metal's validation layer

One case (64 x 64 x 64, codes of every size), the 128-thread tile of the
same kernel as the reference, and the kernel at 256 and 512 threads with
many and with few registers a thread:

    ref.t32      2 x 2 simdgroups, 2 x 2 fragments   128 threads
    sg8.2x2      4 x 2 simdgroups, 2 x 2 fragments   256 threads
    sg16.2x2     4 x 4 simdgroups, 2 x 2 fragments   512 threads (job 1's)
    sg16.1x1     4 x 4 simdgroups, 1 x 1 fragment    512 threads, few registers
    sg16.1x1.f2  the same, form TWO

A tile that writes nothing at 512 threads with FEW registers says the limit
is the threads a threadgroup may have on this chip; one that writes with
few and not with many says the pipeline's limit falls with its registers
(`maxTotalThreadsPerThreadgroup`). The validation layer, where it is on,
names the limit. Lines: `THREADS <tile> <threads> wrote|NEVER-WROTE
equal|DIFFER`. It certifies nothing and times nothing.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys import has_accelerator

from checks.kernel_matrix import COLUMN_APPLE, TARGET_COLUMN
from gemm.checks.gemm_int15_apple_tuned import (
    INT15_TUNED_FORM_FOUR,
    INT15_TUNED_FORM_TWO,
    _launch_tuned,
)
from gemm.checks.gemm_int15_check import _bits, _download, _poisoned, _upload, POISON


def _planes(rows: Int, k: Int, salt: Int) -> Tuple[List[Int8], List[Int8]]:
    var h = List[Int8]()
    var l = List[Int8]()
    var x = salt * 7919 + 13
    for _ in range(rows * k):
        x = (x * 1103515245 + 12345) % 2147483648
        h.append(Int8(((x >> 8) % 256) - 128))
        l.append(Int8((x >> 17) % 128))
    return (h^, l^)


def _report(tag: String, threads: Int, got: List[Float32], want: List[Float32]):
    var unwritten = 0
    var differ = 0
    for i in range(len(got)):
        if _bits(got[i]) == _bits(POISON):
            unwritten += 1
        elif len(want) > 0 and _bits(got[i]) != _bits(want[i]):
            differ += 1
    var w = String("wrote") if unwritten == 0 else String("NEVER-WROTE(") + String(unwritten) + " cells)"
    var d = String("equal") if differ == 0 else String("DIFFER(") + String(differ) + " cells)"
    print("THREADS " + tag + " " + String(threads) + " " + w + " " + d)


def main() raises:
    comptime if TARGET_COLUMN != COLUMN_APPLE or not has_accelerator():
        raise Error("threads probe: Apple only; NOTHING RAN")
    else:
        var ctx = DeviceContext()
        var m = 64
        var n = 64
        var k = 64
        var pa = _planes(m, k, 1)
        var pb = _planes(n, k, 2)
        var ea = List[Int32]()
        for _ in range(m):
            ea.append(Int32(0))
        var eb = List[Int32]()
        for _ in range(n):
            eb.append(Int32(0))
        var dah = _upload[DType.int8](ctx, pa[0])
        var dal = _upload[DType.int8](ctx, pa[1])
        var dbh = _upload[DType.int8](ctx, pb[0])
        var dbl = _upload[DType.int8](ctx, pb[1])
        var dea = _upload[DType.int32](ctx, ea)
        var deb = _upload[DType.int32](ctx, eb)
        comptime S = 4_294_967_296
        var c0 = _poisoned(ctx, m * n)
        _launch_tuned[2, 2, 2, 2, 16, INT15_TUNED_FORM_FOUR, False, False, False, False](
            ctx, c0, dah, dal, dea, dbh, dbl, deb, m, n, k, S
        )
        ctx.synchronize()
        var want = _download[DType.float32](ctx, c0, m * n)
        _report(String("ref.t32"), 128, want, List[Float32]())
        var c1 = _poisoned(ctx, m * n)
        _launch_tuned[4, 2, 2, 2, 16, INT15_TUNED_FORM_FOUR, False, False, False, False](
            ctx, c1, dah, dal, dea, dbh, dbl, deb, m, n, k, S
        )
        ctx.synchronize()
        _report(String("sg8.2x2"), 256, _download[DType.float32](ctx, c1, m * n), want)
        var c2 = _poisoned(ctx, m * n)
        _launch_tuned[4, 4, 2, 2, 16, INT15_TUNED_FORM_FOUR, False, False, False, False](
            ctx, c2, dah, dal, dea, dbh, dbl, deb, m, n, k, S
        )
        ctx.synchronize()
        _report(String("sg16.2x2"), 512, _download[DType.float32](ctx, c2, m * n), want)
        var c3 = _poisoned(ctx, m * n)
        _launch_tuned[4, 4, 1, 1, 32, INT15_TUNED_FORM_FOUR, False, False, False, False](
            ctx, c3, dah, dal, dea, dbh, dbl, deb, m, n, k, S
        )
        ctx.synchronize()
        _report(String("sg16.1x1"), 512, _download[DType.float32](ctx, c3, m * n), want)
        var c4 = _poisoned(ctx, m * n)
        _launch_tuned[4, 4, 1, 1, 32, INT15_TUNED_FORM_TWO, False, False, True, False](
            ctx, c4, dah, dal, dea, dbh, dbl, deb, m, n, k, S
        )
        ctx.synchronize()
        _report(String("sg16.1x1.f2d"), 512, _download[DType.float32](ctx, c4, m * n), want)
        _ = dah
        _ = dal
        _ = dbh
        _ = dbl
        _ = dea
        _ = deb
