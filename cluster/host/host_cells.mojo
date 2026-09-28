# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the cluster family's CPU host bindings.
"""THE CLUSTER FAMILY'S HOST THREAD SPLIT AND SIMD SPELLINGS (lane
cluster-cpu, 2026-09-28). HOST ONLY: no `std.gpu`, no `max.gpu`.

`host_cells(body, n, cost)` runs `body(t)` for every t in [0, n) in
contiguous tasks (`core/host_predict_threads.mojo`'s policy:
MOJOLEARN_CPU_THREADS, else one task per physical core) through
`core/host_parallel.mojo::host_parallelize`, so every task runs in the
caller's floating-point environment (DEVIATION 5900). A caller hands it
only bodies whose indices are independent: index t writes only its own
output cells and reads nothing another index writes in the same call. Each
index's arithmetic is the serial walk's, statement for statement, so the
bits are the same at every thread count. THIS IS NOT A NUMERIC ROW.

`ftz_v` and `mul_add_v` are `checks/numerics.mojo`'s `ftz` and
`identical_mul_add` lane by lane, spelled as vector operations: `ftz_v`
is the same bit test and select (a subnormal becomes its signed zero,
everything else is returned unchanged; nothing under FAST), and
`mul_add_v` is one rounding per lane (`fma`) under IDENTICAL, the naive
chain under FAST. So a SIMD lane computes exactly the scalar statement it
replaces; vectors run ACROSS independent outputs, never along a fold."""
from std.math import fma
from std.memory import bitcast

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count

#: Scalar operations one task should carry at least, so a small call (a
#: k-sized update, a tiny fixture) stays on the calling thread.
comptime HOST_CELLS_GRAIN = 1 << 15


def host_cells[F: def(Int) -> None](ref body: F, n: Int, cost: Int):
    """`body(t)` for every t in [0, n): contiguous tasks over the pool, or
    the plain loop on the calling thread when the work is small or one task
    is asked for. `cost` is the rough operation count of one index."""
    if n <= 0:
        return
    var c = cost if cost > 1 else 1
    var want = (n * c) // HOST_CELLS_GRAIN
    if want > n:
        want = n
    var tasks = host_predict_task_count(want) if want > 1 else 1
    if tasks <= 1:
        for t in range(n):
            body(t)
        return
    var chunk = host_predict_chunk(n, tasks)
    tasks = (n + chunk - 1) // chunk

    def task(ci: Int) {imm body, imm chunk, imm n}:
        var lo = ci * chunk
        var hi = min(lo + chunk, n)
        for t in range(lo, hi):
            body(t)

    host_parallelize(task, tasks)


@always_inline
def ftz_v[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """`ftz`, lane by lane, as one vector select."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var b = bitcast[DType.uint32, w](x)
        var e = b & SIMD[DType.uint32, w](0x7F800000)
        var m = b & SIMD[DType.uint32, w](0x007FFFFF)
        var sub = e.eq(SIMD[DType.uint32, w](0)) & m.ne(SIMD[DType.uint32, w](0))
        var z = bitcast[DType.float32, w](b & SIMD[DType.uint32, w](0x80000000))
        return sub.select(z, x)
    return x


@always_inline
def mul_add_v[w: Int](
    a: SIMD[DType.float32, w], b: SIMD[DType.float32, w], c: SIMD[DType.float32, w]
) -> SIMD[DType.float32, w]:
    """`identical_mul_add`, lane by lane."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return fma(a, b, c)
    return a * b + c
