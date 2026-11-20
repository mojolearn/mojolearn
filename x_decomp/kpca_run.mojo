# SPDX-License-Identifier: Apache-2.0
"""RUN pieces of bindings/kpca_native.mojo: the float32 rounding of a scalar setting. Typed Mojo only, no Python
import (tools/python_free_dirs.sh); moved unchanged (lane rs-decomposition,
2026-11-12)."""


def _f32_scalar(value: Float64) -> Float64:
    return Float64(Float32(value))
