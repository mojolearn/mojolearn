# SPDX-License-Identifier: Apache-2.0
"""RUN piece of the linalg owner's buffer readings
(bindings/linalg_ops_buffers.mojo): the float32 test on an admitted layout.
Typed Mojo only, no Python import (tools/python_free_dirs.sh); moved
unchanged (lane rs-decomposition, 2026-11-12)."""
from bindings.tensor_view import TensorView, base_format


def lin_is_f32(v: TensorView) -> Bool:
    """`_bufcheck.is_native_f32(pb.format)`."""
    return base_format(v.fmt) == "f"
