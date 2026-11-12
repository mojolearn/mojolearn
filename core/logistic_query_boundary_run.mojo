# SPDX-License-Identifier: Apache-2.0
"""RUN half of `bindings/logistic_query_boundary.mojo` (wave 8 run split,
2026-11-12): the Torch scalar's dtype word applied to its decoded width. The
door unpacks the tensor scalar (value and dtype name) and decodes the value;
this applies the dtype's weak-conversion precision or integer width. Typed
Mojo only, no Python import (tools/python_free_dirs.sh). Moved word for word."""
from core.logistic_query_state import LogisticQueryWidth


def logistic_torch_width_run(var result: LogisticQueryWidth, dtype: String) raises -> LogisticQueryWidth:
    """The width of a Torch scalar whose value decoded to `result`, with
    the tensor dtype `dtype` (for example "torch.float32")."""
    if result.kind == 3:
        result.kind = 0  # the tensor dtype controls weak scalar conversion
    if dtype == "torch.float16" or dtype == "torch.complex32":
        result.precision = 16
    elif dtype == "torch.bfloat16":
        result.precision = 8
    elif dtype == "torch.float32" or dtype == "torch.complex64":
        result.precision = 32
    elif dtype == "torch.float64" or dtype == "torch.complex128":
        result.precision = 64
    elif dtype == "torch.uint8" or dtype == "torch.uint16" or dtype == "torch.uint32" or dtype == "torch.uint64":
        result.integer_bits = 8 if dtype == "torch.uint8" else (16 if dtype == "torch.uint16" else (32 if dtype == "torch.uint32" else 64))
    elif dtype == "torch.int8" or dtype == "torch.int16" or dtype == "torch.int32" or dtype == "torch.int64":
        result.integer_bits = 8 if dtype == "torch.int8" else (16 if dtype == "torch.int16" else (32 if dtype == "torch.int32" else 64))
        result.integer_signed = True
    elif dtype != "torch.bool":
        raise Error("TypeError: LogisticRegression feature metadata requires a primitive numeric Tensor")
    return result
