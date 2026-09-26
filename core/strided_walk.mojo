# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The strided per-thread walk of the pinned one-block folds, with its loads
issued ahead (lane apple-identical-steps, 2026-09-26).

WHAT. Every pinned fold of the `sum_terms_kernel` / `mean_kernel` /
`xty_kernel` family gives thread `tid` the elements `tid, tid + TPB,
tid + 2 TPB, ...` and folds them into ONE accumulator in ascending order,
then hands the per-thread partials to `pinned_block_sum`. That chain is the
contract: it fixes every rounding. What it does not fix is WHEN the loads
are issued. Written as `while i < n: acc = f(acc, v[i]); i += TPB`, the
Metal compiler issues one load, waits for it, folds, and only then issues
the next, so a 256-thread block keeps ~8 cache lines in flight and a 4M-row
fold runs at ~2 GB/s (measured on the M4: `sum_terms` over 4,000,000 floats
8.26 ms, `xty` 4,000,000 x 11 10.96 ms).

HOW. `UNROLL` loads are read into registers first, then folded into the
accumulator one after another, in the same order, with the same function,
as the plain loop would. The loads carry no dependence on the accumulator,
so they can all be in flight at once; the fold is the same chain of
roundings. The tail (`n` not a multiple of `UNROLL * TPB` past `tid`) is the
plain loop. So the bits are the bits of the plain loop on every input, by
construction, and the identity hashes in `bench/apple_identical_steps_main
.mojo` are the evidence.

WHERE. Apple, IDENTICAL only, like every other Apple-scheduling choice: the
NVIDIA, AMD and CPU columns compile the unchanged loop.
`-D MOJOLEARN_APPLE_STEP_UNROLL_OFF` restores the plain loop on Apple.
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)


comptime APPLE_IDENTICAL_STEP_UNROLL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_APPLE_STEP_UNROLL_OFF"]()
)

comptime STRIDED_UNROLL = 8


@always_inline
def strided_ftz_sum[
    stride: Int
](
    v: MutPointer[Float32, MutAnyOrigin],
    row: Int,
    off: Int,
    n: Int,
    start: Int,
    acc_in: Float32,
) -> Float32:
    """`acc = ftz(acc + v[i * row + off])` for `i = start, start + stride,
    ... < n`."""
    comptime U = STRIDED_UNROLL
    var acc = acc_in
    var i = start
    while i + (U - 1) * stride < n:
        var t = InlineArray[Float32, U](fill=Float32(0.0))
        comptime for u in range(U):
            t[u] = v.unsafe_load((i + u * stride) * row + off)
        comptime for u in range(U):
            acc = ftz(acc + t[u])
        i += U * stride
    while i < n:
        acc = ftz(acc + v.unsafe_load(i * row + off))
        i += stride
    return acc


@always_inline
def strided_mul_add[
    stride: Int
](
    x: MutPointer[Float32, MutAnyOrigin],
    x_row: Int,
    x_off: Int,
    y: MutPointer[Float32, MutAnyOrigin],
    y_row: Int,
    y_off: Int,
    n: Int,
    start: Int,
) -> Float32:
    """`acc = identical_mul_add(x[r * x_row + x_off], y[r * y_row + y_off],
    acc)` for `r = start, start + stride, ... < n`, from `acc = 0.0`."""
    comptime U = STRIDED_UNROLL
    var acc = Float32(0.0)
    var r = start
    while r + (U - 1) * stride < n:
        var a = InlineArray[Float32, U](fill=Float32(0.0))
        var b = InlineArray[Float32, U](fill=Float32(0.0))
        comptime for u in range(U):
            a[u] = x.unsafe_load((r + u * stride) * x_row + x_off)
            b[u] = y.unsafe_load((r + u * stride) * y_row + y_off)
        comptime for u in range(U):
            acc = identical_mul_add(a[u], b[u], acc)
        r += U * stride
    while r < n:
        acc = identical_mul_add(
            x.unsafe_load(r * x_row + x_off), y.unsafe_load(r * y_row + y_off), acc
        )
        r += stride
    return acc
