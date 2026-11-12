# SPDX-License-Identifier: Apache-2.0
"""Typed scalar helpers of the one-class SVM door (RUN)."""


def _f32_scalar(value: Float64) -> Float64:
    """A Python double rounded once to float32."""
    return Float64(Float32(value))
