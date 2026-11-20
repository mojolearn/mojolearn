# SPDX-License-Identifier: Apache-2.0
"""RUN of `providers/python/autograd_program_device_pyapi.mojo` (wave 8 run split): the (CastIn kind, item size) of a device input's dtype name, typed Mojo
only. Imports no Python (tools/python_free_dirs.sh); body moved unchanged
(lane rs-sequence-tokenizer, 2026-11-12)."""
from core.cast_in import CAST_KIND_FLOAT, CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL


def _device_kind(dtype: String, what: String) raises -> Tuple[Int, Int]:
    """(CastIn kind, item size) of a dtype name (torch's names, which are
    NumPy's): the same kinds cast_in_kind gives the host path's arrays."""
    if dtype == "float32":
        return (CAST_KIND_FLOAT, 4)
    if dtype == "float16":
        return (CAST_KIND_FLOAT, 2)
    if dtype == "float64":
        return (CAST_KIND_FLOAT, 8)
    if dtype == "int8":
        return (CAST_KIND_SIGNED, 1)
    if dtype == "int16":
        return (CAST_KIND_SIGNED, 2)
    if dtype == "int32":
        return (CAST_KIND_SIGNED, 4)
    if dtype == "int64":
        return (CAST_KIND_SIGNED, 8)
    if dtype == "uint8":
        return (CAST_KIND_UNSIGNED, 1)
    if dtype == "bool":
        return (CAST_KIND_BOOL, 1)
    raise Error("TypeError: " + what + ": dtype " + dtype + " is not supported; pass a float16, float32, "
                + "float64, integer or bool tensor")
