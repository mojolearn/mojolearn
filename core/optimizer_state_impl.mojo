# SPDX-License-Identifier: Apache-2.0
# State input conversion; optimizer arithmetic and accumulation order are unchanged.
from std.memory import bitcast
from std.math import isfinite
from core.classical_cast import classical_f32_bits
from core.classical_pandas_impl import _pd_double


def optimizer_state_cast(addr: Int, stride: Int, size: Int, item: Int, kind: Int, native: Bool,
                         dest: MutPointer[UInt32, MutUntrackedOrigin], integer: Bool) raises:
    for i in range(size):
        var bytes = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr + i * stride)
        var raw = UInt64(0)
        for k in range(item):
            raw |= UInt64(bytes[k if native else item - 1 - k]) << UInt64(k * 8)
        if integer:
            var value = _pd_double(raw, kind, item)
            if value != value:
                raise Error("ValueError: cannot convert float NaN to integer")
            if not isfinite(value) or value <= -2147483649.0 or value >= 2147483648.0:
                raise Error("OverflowError: buf_initialized value does not fit int32")
            dest[i] = bitcast[DType.uint32](Int32(value))
        else:
            dest[i] = classical_f32_bits(raw, kind, item)
