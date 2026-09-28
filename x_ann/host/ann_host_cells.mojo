# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into the ann family's CPU host binding (_mojolearn_x_ann_host).
"""THE ANN FAMILY'S HOST ROW SPLIT AND SIMD SPELLINGS (lane ann-cpu,
2026-09-28). HOST ONLY: no `std.gpu`, no `max.gpu`.

`ann_rows(body, n, cost)` runs `body(t)` for every t in [0, n). The caller
hands it only bodies whose indices are independent: index t writes only its
own output cells (and its task's scratch, which `body` allocates per call
of the task, never per index) and reads nothing another index writes in the
same call. Each index's arithmetic is the serial walk's, statement for
statement, so the bits are the same at every thread count. THIS IS NOT A
NUMERIC ROW: it changes which thread computes an index, never what it
computes.

The split is contiguous (`core/host_predict_threads.mojo`'s policy:
MOJOLEARN_CPU_THREADS, else one task per physical core, never more tasks
than indices), and every task runs in the caller's floating-point
environment through `core/host_parallel.mojo::host_parallelize` (DEVIATION
5900; the module is carried byte for byte from lane/cpu until it lands on
main). One task runs on the calling thread with no dispatch.

`ftz_v`, `mul_add_v` and `mul_v` are `checks/numerics.mojo`'s `ftz`,
`identical_mul_add` and `identical_mul` lane by lane, spelled as vector
operations: `ftz_v` is the same bit test and select (a subnormal becomes its
signed zero, everything else unchanged), `mul_add_v` is one rounding per
lane (`fma`), `mul_v` one correctly rounded product per lane behind an
arithmetic fence (the host arm of `pinned_mul_f32`, so no neighbor add
fuses with it). So a SIMD lane computes exactly the scalar statement it
replaces; vectors run ACROSS independent cells, never along a fold.
"""
from std.math import fma
from std.memory import bitcast
from std.sys import llvm_intrinsic

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_chunk, host_predict_task_count

#: Scalar operations one task should carry at least, so a small call (a
#: fixture, a handful of queries) stays on the calling thread.
comptime ANN_ROWS_GRAIN = 1 << 15


def ann_task_count(n: Int, cost: Int) -> Int:
    """How many contiguous tasks `ann_rows` splits `n` indices of `cost`
    operations each into (1 means the calling thread alone)."""
    if n <= 1:
        return 1
    var c = cost if cost > 1 else 1
    var want = (n * c) // ANN_ROWS_GRAIN
    if want > n:
        want = n
    if want <= 1:
        return 1
    var tasks = host_predict_task_count(want)
    var chunk = host_predict_chunk(n, tasks)
    return (n + chunk - 1) // chunk


def ann_tasks[F: def(Int) -> None](ref body: F, tasks: Int):
    """`body(c)` for every task c in [0, tasks): the one place the ann host
    paths fan out (module docstring)."""
    if tasks <= 1:
        body(0)
        return
    host_parallelize(body, tasks)


@always_inline
def ann_span(c: Int, tasks: Int, n: Int) -> Tuple[Int, Int]:
    """Task c's contiguous index range [lo, hi) of n indices over `tasks`."""
    var chunk = host_predict_chunk(n, tasks)
    var lo = c * chunk
    var hi = min(lo + chunk, n)
    if lo > n:
        lo = n
    return (lo, hi)


@always_inline
def _flush_bits[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """A zero exponent field becomes the signed zero, every other word is
    returned unchanged: `ftz`'s rule, spelled without a compare or a select
    (a vector select of a bool mask lowered ten times slower than the
    arithmetic around it, measured on the Xeon 8470, 2026-09-28). With
    e = bits & 0x7F800000, `(e + 0x7FFFFFFF) >> 31` is 1 exactly when e != 0
    (no wrap: e <= 0x7F800000), so `0 - that` is the all-ones keep mask."""
    var b = bitcast[DType.uint32, w](x)
    var e = b & SIMD[DType.uint32, w](0x7F800000)
    var keep = SIMD[DType.uint32, w](0) - ((e + SIMD[DType.uint32, w](0x7FFFFFFF)) >> SIMD[DType.uint32, w](31))
    return bitcast[DType.float32, w](b & (keep | SIMD[DType.uint32, w](0x80000000)))


@always_inline
def ftz_v[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """`ftz`, lane by lane (a zero or a subnormal becomes its signed zero,
    which for a zero is itself)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return _flush_bits[w](x)
    return x


@always_inline
def mul_add_v[w: Int](
    a: SIMD[DType.float32, w], b: SIMD[DType.float32, w], c: SIMD[DType.float32, w]
) -> SIMD[DType.float32, w]:
    """`identical_mul_add`, lane by lane."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return fma(a, b, c)
    return a * b + c


@always_inline
def mul_v[w: Int](a: SIMD[DType.float32, w], b: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """`identical_mul`, lane by lane: the host `pinned_mul_f32` (an
    arithmetic fence around the product, so no neighbor add fuses with it).
    Spelled for 8 and 16 lanes."""
    comptime assert w == 8 or w == 16, "mul_v: the fence is spelled for 8 or 16 lanes"
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        comptime if w == 8:
            return rebind[SIMD[DType.float32, w]](llvm_intrinsic[
                "llvm.arithmetic.fence.v8f32", SIMD[DType.float32, 8], has_side_effect=False
            ](rebind[SIMD[DType.float32, 8]](a * b)))
        else:
            return rebind[SIMD[DType.float32, w]](llvm_intrinsic[
                "llvm.arithmetic.fence.v16f32", SIMD[DType.float32, 16], has_side_effect=False
            ](rebind[SIMD[DType.float32, 16]](a * b)))
    return a * b


@always_inline
def div_v[w: Int](a: SIMD[DType.float32, w], b: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """`identical_div` (`portable_divf`: operands flushed, one correctly
    rounded division, result flushed; the flush unconditional), lane by
    lane. The host binding builds IDENTICAL only."""
    return _ftz_always_v[w](_ftz_always_v[w](a) / _ftz_always_v[w](b))


@always_inline
def _ftz_always_v[w: Int](x: SIMD[DType.float32, w]) -> SIMD[DType.float32, w]:
    """`checks/numerics.mojo::_ftz_always`, lane by lane: exponent field
    zero becomes the signed zero, in every mode."""
    return _flush_bits[w](x)
