# SPDX-License-Identifier: Apache-2.0
"""Typed parts of the GP kernel public door (RUN): the natural log with the math-module domain refusal,
and the decoded kernel input (shape and float64 values) the door fills once from its Python arguments."""
from std.memory import bitcast
from x_metrics.epilogue import portable_log_c


def _kernel_log(value: Float64) raises -> Float64:
    if value <= 0.0:
        raise Error("ValueError: math domain error")
    if value != value or bitcast[DType.uint64](value) == UInt64(0x7FF0000000000000):
        return value
    return portable_log_c(value)


struct GPKernelInput(Movable):
    var shape: List[Int]
    var values: List[Float64]

    def __init__(out self):
        self.shape = List[Int]()
        self.values = List[Float64]()
