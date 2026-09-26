# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""StandardScaler and MinMaxScaler on the host, for a box with no GPU
(workstream E batch 2, the standard-scaler and minmax-scaler lanes of
tools/identity_break.py, 2026-09-14).

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu` or a `DeviceContext`,
and no GPU binding imports this file. Every kernel of `preprocessing/
standard.mojo` and `preprocessing/minmax.mojo` is spelled a SECOND time
here, in the same statement order, with the arithmetic leaves of
`checks/numerics.mojo` (`ftz`, `identical_mul`, `identical_div`,
`portable_sqrtf`) and DEVIATION 653's slab tree (`metrics/checks/
pinned_sum.mojo::virtual_block_sum` at block 256, `PINNED_SUM_W` 256)
spelled again as `host_tree_sum`.

WHAT IS RESTATED, AND WHERE THE ORIGINAL IS.

  `host_tree_sum`          `virtual_block_sum[256]` over every 256-row chunk
                           of one column, each chunk total flushed, the
                           totals folded ascending from `+0.0` through `ftz`
                           (`standard_chunks_kernel` then the loop of
                           `standard_finalize_kernel`, `standard.mojo:29,
                           58`).
  `host_standard_fit`      `standard_fit`, `standard.mojo:115`:
                           `standard_initialize_kernel` (mean 0, var 0,
                           scale 1), the mean pass (`value = ftz(x)`, the
                           per-chunk count of rows whose flushed value
                           differs from the flushed first row, `mean =
                           ftz(x[0, c])` when none differs else
                           `ftz(identical_div(total, n))`, the constant
                           marker in the variance row), then the variance
                           pass (`value = ftz(identical_mul(ftz(value -
                           ftz(mean)), same))`, zero on a marked column,
                           `var = ftz(identical_div(total, n))`, `scale = 1`
                           at zero variance else `ftz(portable_sqrtf(var))`).
  `host_standard_transform`
                           `standard_transform_kernel`, `standard.mojo:86`,
                           per element in its branch order.
  `host_ordered_key`, `host_key_value`
                           `ordered_key`, `key_value`, `minmax.mojo:26, 31`:
                           the total order on the bits (`-0.0` below
                           `+0.0`).
  `host_minmax_fit`        `minmax_fit`, `minmax.mojo:113`: the per-column
                           extrema of the keys (a selection; the chunking
                           of `extrema_chunks_kernel` cannot move a min or a
                           max), then `extrema_finalize_kernel`'s
                           `data_range = ftz(ftz(max) - ftz(min))`, the
                           `10 * eps` denominator rule, `scale = ftz(
                           identical_div(ftz(ftz(upper) - ftz(lower)),
                           denominator))`, `offset = ftz(ftz(lower) -
                           ftz(identical_mul(ftz(min), scale)))`.
  `host_minmax_transform`  `minmax_transform_kernel`, `minmax.mojo:93`, per
                           element, the clip as `min(max(v, lower), upper)`.

The validation is `preprocessing/estimator.mojo`'s, in its words, spelled
in the binding.

The two transform mirrors split contiguous output rows with the shared
`MOJOLEARN_CPU_THREADS` host policy. A task reads fitted statistics and owns
all cells of its rows; no reduction or arithmetic statement crosses a row,
so the serial and parallel outputs are bitwise identical. Setting the policy
to one retains the original serial walk for qualification and small boxes.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` shifts the slab
tree's chunk boundaries by one value (the standard scaler's mean, variance
and scale move where the column's sums are inexact) and turns the min-max
offset's subtraction into an addition. When data_min is zero, it adds one
to the offset instead: changing the sign of zero left the constant-zero
fixture unchanged. Since lane/inference-linear-svm (2026-09-15) it also makes both
transform kernels read the next column's statistics: saved-model inference
(`mojolearn.host_model`, through the estimators host binding) runs the
transforms alone with stored statistics, and the fit arms never reached it
(measured: before this arm the saved-model check read EQUAL on all five
scaler lanes under the sabotage build). Read back by
`preprocessing_host_sabotage` and `estimators_host_sabotage`.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the two scaler lanes is the measurement.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.algorithm import sync_parallelize

from checks.numerics import ftz, identical_div, identical_mul, portable_sqrtf
from core.host_predict_threads import (
    host_list_ptr,
    host_predict_chunk,
    host_predict_task_count,
)


#: The gate's negative control (see THE NEGATIVE CONTROL above).
comptime SCALER_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: `PINNED_SUM_W`, `metrics/checks/pinned_sum.mojo:70`, the kernels' 256.
comptime PINNED_SUM_W = 256

#: `10 * eps32`, `extrema_finalize_kernel`'s near-constant threshold.
comptime MINMAX_RANGE_FLOOR = Float32(0.0000011920928955078125)


def host_tree_sum(values: List[Float32], n: Int) -> Float32:
    """Module docstring. `values[0:n]` are the per-row terms of one column."""
    var chunks = (n + PINNED_SUM_W - 1) // PINNED_SUM_W
    var slab = List[Float32](length=PINNED_SUM_W, fill=Float32(0.0))
    var total = Float32(0.0)
    for c in range(chunks):
        for t in range(PINNED_SUM_W):
            var i = c * PINNED_SUM_W + t
            comptime if SCALER_ORACLE_HOST_SABOTAGE:
                # THE SABOTAGE ARM: the chunk boundaries shifted by one
                # value. Wrong on purpose; see SCALER_ORACLE_HOST_SABOTAGE.
                slab[t] = ftz(values[(i + 1) % n]) if i < n else Float32(0.0)
            else:
                slab[t] = ftz(values[i]) if i < n else Float32(0.0)
        var step = PINNED_SUM_W // 2
        while step > 0:
            for t in range(step):
                slab[t] = ftz(slab[t] + slab[t + step])
            step //= 2
        total = ftz(total + ftz(slab[0]))
    return total


def host_standard_fit(
    x: List[Float32], n: Int, d: Int, with_mean: Int, with_std: Int
) -> List[Float32]:
    """`standard_fit` (module docstring): `3 * d` floats, the rows mean,
    var, scale."""
    var out = List[Float32](length=3 * d, fill=Float32(0.0))
    for c in range(d):
        out[2 * d + c] = Float32(1.0)
    if with_mean == 0 and with_std == 0:
        return out^
    var column = List[Float32](length=n, fill=Float32(0.0))
    for c in range(d):
        # the mean pass
        var first = ftz(x[c])
        var changed = 0
        for row in range(n):
            var value = ftz(x[row * d + c])
            column[row] = value
            if value != first:
                changed += 1
        var total = host_tree_sum(column, n)
        var mean = first if changed == 0 else ftz(identical_div(total, Float32(n)))
        out[c] = mean
        if with_std != 0:
            # the constant-column marker the mean pass leaves in the
            # variance row, read by the variance pass
            var marker = Float32(1.0) if changed != 0 else Float32(0.0)
            for row in range(n):
                var value = ftz(x[row * d + c])
                if marker == Float32(0.0):
                    value = Float32(0.0)
                else:
                    value = ftz(value - ftz(mean))
                    value = ftz(identical_mul(value, value))
                column[row] = value
            var vtotal = host_tree_sum(column, n)
            var var_value = ftz(identical_div(vtotal, Float32(n)))
            var scale = Float32(1.0)
            if var_value != Float32(0.0):
                scale = ftz(portable_sqrtf(var_value))
            out[d + c] = var_value
            out[2 * d + c] = scale
    return out^


def host_standard_transform(
    x: List[Float32],
    mean: List[Float32],
    scale: List[Float32],
    n: Int,
    d: Int,
    inverse: Int,
    with_mean: Int,
    with_std: Int,
) -> List[Float32]:
    """`standard_transform_kernel` (module docstring), `n * d` floats."""
    var out = List[Float32](length=n * d, fill=Float32(0.0))
    var tasks = host_predict_task_count(n)
    var chunk = host_predict_chunk(n, tasks)
    var xp = host_list_ptr(x)
    var mp = host_list_ptr(mean)
    var sp = host_list_ptr(scale)
    var op = host_list_ptr(out)

    def _rows(task: Int) {imm xp, imm mp, imm sp, imm op, imm chunk, imm n, imm d, imm inverse, imm with_mean, imm with_std}:
        var lo = task * chunk
        var hi = min(lo + chunk, n)
        for i in range(lo * d, hi * d):
            var c = i % d
            comptime if SCALER_ORACLE_HOST_SABOTAGE:
                # THE TRANSFORM SABOTAGE ARM (lane/inference-linear-svm,
                # 2026-09-15): each element reads the NEXT column's statistics.
                c = (c + 1) % d
            var value = xp.unsafe_load(i)
            if inverse != 0:
                if with_std != 0:
                    value = ftz(identical_mul(ftz(value), ftz(sp.unsafe_load(c))))
                if with_mean != 0:
                    value = ftz(ftz(value) + ftz(mp.unsafe_load(c)))
            else:
                if with_mean != 0:
                    value = ftz(ftz(value) - ftz(mp.unsafe_load(c)))
                if with_std != 0:
                    value = ftz(identical_div(ftz(value), ftz(sp.unsafe_load(c))))
            op.unsafe_store(i, value)

    if tasks == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, tasks)
    return out^


def host_ordered_key(value: Float32) -> UInt32:
    var bits = bitcast[DType.uint32](value)
    return ~bits if (bits & UInt32(0x80000000)) != 0 else bits | UInt32(0x80000000)


def host_key_value(key: UInt32) -> Float32:
    var bits = key & UInt32(0x7FFFFFFF) if (key & UInt32(0x80000000)) != 0 else ~key
    return bitcast[DType.float32](bits)


def host_minmax_fit(
    x: List[Float32], n: Int, d: Int, lower: Float32, upper: Float32
) -> List[Float32]:
    """`minmax_fit` (module docstring): `5 * d` floats, the rows min, max,
    range, scale, offset."""
    var out = List[Float32](length=5 * d, fill=Float32(0.0))
    for column in range(d):
        var lo = UInt32(0xFFFFFFFF)
        var hi = UInt32(0)
        for row in range(n):
            var key = host_ordered_key(x[row * d + column])
            if key < lo:
                lo = key
            if key > hi:
                hi = key
        var data_min = host_key_value(lo)
        var data_max = host_key_value(hi)
        var data_range = ftz(ftz(data_max) - ftz(data_min))
        var denominator = Float32(1.0) if data_range < MINMAX_RANGE_FLOOR else data_range
        var scale = ftz(identical_div(ftz(ftz(upper) - ftz(lower)), denominator))
        var offset: Float32
        comptime if SCALER_ORACLE_HOST_SABOTAGE:
            # THE SABOTAGE ARM: the offset's subtraction as an addition.
            # Wrong on purpose; see SCALER_ORACLE_HOST_SABOTAGE.
            offset = ftz(ftz(lower) + ftz(identical_mul(ftz(data_min), scale)))
            if data_min == Float32(0.0):
                # A sign change of zero is inert. Corrupt the actual fitted
                # offset on constant-zero columns as well.
                offset = ftz(offset + Float32(1.0))
        else:
            offset = ftz(ftz(lower) - ftz(identical_mul(ftz(data_min), scale)))
        out[column] = data_min
        out[d + column] = data_max
        out[2 * d + column] = data_range
        out[3 * d + column] = scale
        out[4 * d + column] = offset
    return out^


def host_minmax_transform(
    x: List[Float32],
    scale: List[Float32],
    offset: List[Float32],
    n: Int,
    d: Int,
    inverse: Int,
    clip: Int,
    lower: Float32,
    upper: Float32,
) -> List[Float32]:
    """`minmax_transform_kernel` (module docstring), `n * d` floats."""
    var out = List[Float32](length=n * d, fill=Float32(0.0))
    var tasks = host_predict_task_count(n)
    var chunk = host_predict_chunk(n, tasks)
    var xp = host_list_ptr(x)
    var sp = host_list_ptr(scale)
    var mp = host_list_ptr(offset)
    var op = host_list_ptr(out)

    def _rows(task: Int) {imm xp, imm sp, imm mp, imm op, imm chunk, imm n, imm d, imm inverse, imm clip, imm lower, imm upper}:
        var lo = task * chunk
        var hi = min(lo + chunk, n)
        for i in range(lo * d, hi * d):
            var c = i % d
            comptime if SCALER_ORACLE_HOST_SABOTAGE:
                # THE TRANSFORM SABOTAGE ARM (lane/inference-linear-svm,
                # 2026-09-15): each element reads the NEXT column's statistics.
                c = (c + 1) % d
            var value = ftz(xp.unsafe_load(i))
            if inverse != 0:
                value = ftz(identical_div(ftz(value - ftz(mp.unsafe_load(c))), ftz(sp.unsafe_load(c))))
            else:
                value = ftz(ftz(identical_mul(value, ftz(sp.unsafe_load(c)))) + ftz(mp.unsafe_load(c)))
                if clip != 0:
                    if value < lower:
                        value = lower
                    if value > upper:
                        value = upper
            op.unsafe_store(i, value)

    if tasks == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, tasks)
    return out^
