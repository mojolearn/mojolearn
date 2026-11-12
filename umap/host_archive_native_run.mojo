# SPDX-License-Identifier: Apache-2.0
"""Typed UMAP control checks of the host archive loader (RUN): the integer range refusal and the
finite float32 rounding of saved controls."""
from std.math import isfinite


def _umap_whole(value: Int, name: String, minimum: Int, maximum: Int) raises -> Int:
    """UMAP `_integer`'s range refusal for a saved integer control."""
    if value < minimum or value > maximum:
        raise Error("ValueError: UMAP " + name + " must be in [" + String(minimum) + ", " + String(maximum) + "]")
    return value


def _umap_real(value: Float64, name: String) raises -> Float64:
    """UMAP `_scalar`: a finite float32 value, rounded once to binary32."""
    if not isfinite(value) or abs(value) > 3.4028234663852886e38:
        raise Error("ValueError: UMAP " + name + " must be a finite float32 value")
    return Float64(Float32(value))
