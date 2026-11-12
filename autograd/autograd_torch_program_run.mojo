# SPDX-License-Identifier: Apache-2.0
"""The Torch program input dtype table (RUN): cast kind and element width per dtype name."""
from core.cast_in import CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL


def _torch_kind(dtype: String) raises -> Tuple[Int, Int]:
    if dtype == "float32":
        return (CAST_KIND_FLOAT, 4)
    if dtype == "float64":
        return (CAST_KIND_FLOAT, 8)
    if dtype == "float16":
        return (CAST_KIND_FLOAT, 2)
    if dtype == "int64":
        return (CAST_KIND_SIGNED, 8)
    if dtype == "int32":
        return (CAST_KIND_SIGNED, 4)
    if dtype == "int16":
        return (CAST_KIND_SIGNED, 2)
    if dtype == "int8":
        return (CAST_KIND_SIGNED, 1)
    if dtype == "uint8":
        return (CAST_KIND_UNSIGNED, 1)
    if dtype == "bool":
        return (CAST_KIND_BOOL, 1)
    raise Error("TypeError: unsupported Torch program input dtype " + dtype)
