# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""The small MLP's three device operations on the host (the mlp lane,
2026-09-14): `training/mlp_ops.mojo`'s bias plus optional ReLU, ReLU
backward and ascending row sum, spelled a SECOND time.

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu` or a `DeviceContext`,
and no GPU binding imports this file. The device kernel `_mlp_kernel`
(`training/mlp_ops.mojo:36`) writes every output cell from its own thread
with no fold except the row sum, which runs serially inside one thread per
column, so a serial host loop stores the same values:

  `host_mlp_add`             `_add`: `ftz(identical_mul_add(1, ftz(left),
                             ftz(right)))`, one rounded addition.
  `host_mlp_bias_activation` operations 0 and 1: `_add(x[i], bias[i % cols])`,
                             then `value <= 0 -> +0.0` when the ReLU is on.
  `host_mlp_relu_backward`   operation 2: `ftz(incoming[i])` where
                             `activation[i] > 0`, else `+0.0`.
  `host_mlp_sum_rows`        operation 3: per column, rows ascending from
                             `+0.0` through `_add`.

`host_mlp_validate_shape` and `host_mlp_finite` are `mlp_validate_shape` and
`_finite` in their words, and the host entry checks the inputs before and the
outputs after, as `_mlp_host` does.

THE NEGATIVE CONTROL. `-D MOJOLEARN_HOST_SABOTAGE=1` walks every row sum
DESCENDING, so both bias gradients move in their last bits and every AdamW
update after them. Read back by `training_host_sabotage`.

The restatement is a prediction until measured. The four-column diff of
tools/identity_break.py on the mlp lane is the measurement.
"""
from std.math import isfinite
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add


#: The gate's negative control (see THE NEGATIVE CONTROL above).
comptime MLP_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def host_mlp_validate_shape(rows: Int, cols: Int) raises:
    """`mlp_validate_shape`."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("small MLP operations require IDENTICAL numeric mode")
    if rows < 1 or rows > 256 or cols < 1 or cols > 64:
        raise Error("small MLP operations require rows 1..256 and cols 1..64")


def host_mlp_finite(values: List[Float32]) raises:
    """`_finite`."""
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error("small MLP operation has nonfinite input or output")


@always_inline
def host_mlp_add(left: Float32, right: Float32) -> Float32:
    """`_add`."""
    return ftz(identical_mul_add(Float32(1), ftz(left), ftz(right)))


def host_mlp_bias_activation(
    source: List[Float32], bias: List[Float32], rows: Int, cols: Int, relu_flag: Int,
) raises -> List[Float32]:
    """`mlp_bias_activation_host`: operation `relu_flag` (0 or 1)."""
    host_mlp_validate_shape(rows, cols)
    if relu_flag != 0 and relu_flag != 1:
        raise Error("mlp_bias_activation relu_flag must be 0 or 1")
    host_mlp_finite(source)
    host_mlp_finite(bias)
    var out = List[Float32](length=rows * cols, fill=Float32(0.0))
    for i in range(rows * cols):
        var value = host_mlp_add(source[i], bias[i % cols])
        if relu_flag == 1 and value <= Float32(0):
            value = Float32(0)
        out[i] = value
    host_mlp_finite(out)
    return out^


def host_mlp_relu_backward(
    activation: List[Float32], incoming: List[Float32], rows: Int, cols: Int,
) raises -> List[Float32]:
    """`mlp_relu_backward_host`: operation 2."""
    host_mlp_validate_shape(rows, cols)
    host_mlp_finite(activation)
    host_mlp_finite(incoming)
    var out = List[Float32](length=rows * cols, fill=Float32(0.0))
    for i in range(rows * cols):
        var value = Float32(0)
        if activation[i] > Float32(0):
            value = ftz(incoming[i])
        out[i] = value
    host_mlp_finite(out)
    return out^


def host_mlp_sum_rows(source: List[Float32], rows: Int, cols: Int) raises -> List[Float32]:
    """`mlp_sum_rows_host`: operation 3."""
    host_mlp_validate_shape(rows, cols)
    host_mlp_finite(source)
    var out = List[Float32](length=cols, fill=Float32(0.0))
    for c in range(cols):
        var value = Float32(0)
        comptime if MLP_ORACLE_HOST_SABOTAGE:
            # THE SABOTAGE ARM: rows walked DESCENDING. Wrong on purpose.
            for rr in range(rows):
                value = host_mlp_add(value, source[(rows - 1 - rr) * cols + c])
        else:
            for row in range(rows):
                value = host_mlp_add(value, source[row * cols + c])
        out[c] = value
    host_mlp_finite(out)
    return out^
