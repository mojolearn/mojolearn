# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""The small MLP's bias plus optional ReLU on the host: the CPU-only
inference forward of `MLPInference` (bindings/_mojolearn_neural_host.mojo
`mlp_forward_logits`, core/mojolearn_neural_host_impl.mojo).

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu` or a `DeviceContext`,
and no GPU binding imports this file.

  `host_mlp_add`             `ftz(identical_mul_add(1, ftz(left), ftz(right)))`,
                             one rounded addition.
  `host_mlp_bias_activation` `add(x[i], bias[i % cols])`, then
                             `value <= 0 -> +0.0` when the ReLU is on.

`host_mlp_validate_shape` and `host_mlp_finite` check the inputs before and
the outputs after.

`MLP_ORACLE_HOST_SABOTAGE` (`-D MOJOLEARN_HOST_SABOTAGE=1`) is read back by
`training_host_sabotage`; the same define walks every `gemm_oracle` leaf
descending.
"""
from std.math import isfinite
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add


#: The host column's negative control define (read back by training_host_sabotage).
comptime MLP_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def host_mlp_validate_shape(rows: Int, cols: Int) raises:
    """Rows 1..256, columns 1..64, IDENTICAL only."""
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("small MLP operations require IDENTICAL numeric mode")
    if rows < 1 or rows > 256 or cols < 1 or cols > 64:
        raise Error("small MLP operations require rows 1..256 and cols 1..64")


def host_mlp_finite(values: List[Float32]) raises:
    """Refuse a non-finite input or output."""
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error("small MLP operation has nonfinite input or output")


@always_inline
def host_mlp_add(left: Float32, right: Float32) -> Float32:
    """One rounded FP32 addition with the IDENTICAL flush."""
    return ftz(identical_mul_add(Float32(1), ftz(left), ftz(right)))


def host_mlp_bias_activation(
    source: List[Float32], bias: List[Float32], rows: Int, cols: Int, relu_flag: Int,
) raises -> List[Float32]:
    """Bias plus ReLU (`relu_flag` 1) or bias only (`relu_flag` 0)."""
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


# TOMBSTONE: host_mlp_relu_backward and host_mlp_sum_rows (the CPU twins of training/mlp_ops.mojo's ReLU backward and row sum, and the descending row-sum MOJOLEARN_HOST_SABOTAGE arm in them) deleted 2026-11-09 by lane/m2-neural-leftovers; code recoverable at 0ce88411d3.
# Restore: git apply experiments/removed/M2_SMALL_MLP_PRIMITIVES.patch

