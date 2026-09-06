# SPDX-License-Identifier: Apache-2.0
"""Host quantization refusal gate; does not construct a DeviceContext."""
from gbdt.data.quantization import calc_quantization
from gbdt.options.data_processing_options import NAN_MODE_FORBIDDEN
from std.memory import bitcast


def main() raises:
    for budget in range(2):
        var values = List[Float32]()
        values.append(Float32(0))
        values.append(bitcast[DType.float32](UInt32(0x7fc00000)))
        values.append(Float32(1))
        var refused = False
        try:
            var result = calc_quantization(values^, budget, NAN_MODE_FORBIDDEN)
        except:
            refused = True
        if not refused:
            raise Error("Forbidden accepted NaN before border construction")
    var clean = List[Float32]()
    clean.append(Float32(0))
    clean.append(Float32(1))
    clean.append(Float32(2))
    var result = calc_quantization(clean^, 1, NAN_MODE_FORBIDDEN)
    if result[1] != NAN_MODE_FORBIDDEN or len(result[0]) != 1:
        raise Error("clean Forbidden quantization changed")
