# SPDX-License-Identifier: Apache-2.0
"""C30 versioned classical direct Euclidean arithmetic, host and all GPUs.
NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Only opted-in classical callers use this helper; no global numeric mode changes.
"""
from checks.numerics import ftz, identical_mul

@always_inline
def direct_squared_distance(
    x: MutPointer[Float32, MutAnyOrigin], y: MutPointer[Float32, MutAnyOrigin],
    features: Int, x_stride: Int = 1, y_stride: Int = 1,
) -> Float32:
    # One ascending logical feature fold independent of register geometry.
    # Separate multiply prevents contraction of square plus sum on the host.
    var result = Float32(0)
    for f in range(features):
        var delta = ftz(ftz(x[f * x_stride]) - ftz(y[f * y_stride]))
        result = ftz(result + ftz(identical_mul(delta, delta)))
    if result <= Float32(0):
        result = Float32(0)
    return result

@always_inline
def direct_distance_step[w: Int](acc: SIMD[DType.float32,w], x: SIMD[DType.float32,w], y: SIMD[DType.float32,w]) -> SIMD[DType.float32,w]:
    var out = SIMD[DType.float32,w](0)
    comptime for lane in range(w):
        var delta = ftz(ftz(x[lane])-ftz(y[lane]))
        out[lane] = ftz(acc[lane]+ftz(identical_mul(delta,delta)))
    return out
