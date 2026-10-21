# SPDX-License-Identifier: Apache-2.0
"""Typed Holt-Winters output stores moved word for word from bindings/holtwinters_host_predict.mojo."""
from std.math import isfinite
from std.memory import bitcast
from std.sys.compile import is_defined

comptime HW_PREDICT_SABOTAGE = (
    is_defined["MOJOLEARN_HOST_SABOTAGE"]()
    or is_defined["MOJOLEARN_HW_PREDICT_SABOTAGE"]()
)


@always_inline
def _store(op: MutPointer[Float32, MutUntrackedOrigin], i: Int, v: Float32):
    comptime if HW_PREDICT_SABOTAGE:
        if isfinite(v):
            op.unsafe_store(
                i, bitcast[DType.float32](bitcast[DType.uint32](v) ^ UInt32(1))
            )
            return
    op.unsafe_store(i, v)


def hw_store_values(op: MutPointer[Float32, MutUntrackedOrigin], values: List[Float32], count: Int):
    for i in range(count):
        _store(op, i, values[i])
