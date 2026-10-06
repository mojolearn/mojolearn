# SPDX-License-Identifier: Apache-2.0
"""NN59 portable residual-dropout layer graph; no GPU imports.

Philox mapping is core.philox_neural's (block low/high, stream, 0), keyed by
seed low/high. The ten-round integer body below is shared verbatim by this
layer's host and GPU routes, avoiding a host import of a device runtime module.
"""
from std.sys.compile import is_defined
from std.memory import bitcast
from std.math import isfinite
from checks.numerics import GLOBAL_NUMERIC_MODE,NUMERIC_IDENTICAL,ftz,identical_mul,identical_div

comptime NN59_DROPOUT_RESIDUAL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN59_DROPOUT_RESIDUAL"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    and not is_defined["MOJOLEARN_NN59_CONTROL"]()
)


@always_inline
def _nn59_mulhilo(a: UInt32,b: UInt32) -> Tuple[UInt32,UInt32]:
    var product = UInt64(a)*UInt64(b)
    return (UInt32((product >> 32) & 0xffffffff),UInt32(product & 0xffffffff))


@always_inline
def nn59_unit(seed_lo: UInt32,seed_hi: UInt32,stream: UInt32,index: Int) -> Float32:
    var block = UInt64(index//4)
    var c = SIMD[DType.uint32,4](UInt32(block & 0xffffffff),UInt32((block >> 32) & 0xffffffff),stream,UInt32(0))
    var key0 = seed_lo
    var key1 = seed_hi
    for round in range(10):
        var a = _nn59_mulhilo(UInt32(0xd2511f53),c[0])
        var b = _nn59_mulhilo(UInt32(0xcd9e8d57),c[2])
        c = SIMD[DType.uint32,4](b[0]^c[1]^key0,b[1],a[0]^c[3]^key1,a[1])
        if round<9:
            key0 += UInt32(0x9e3779b9)
            key1 += UInt32(0xbb67ae85)
    # 24-bit integer times 2^-24 is exact in FP32 on every column.
    return Float32(Int(c[index%4] >> 8))*bitcast[DType.float32](UInt32(0x33800000))


def residual_dropout_admit(n: Int,offset: Int,p: Float32,
                           seed_lo: Int,seed_hi: Int,stream: Int) raises -> Float32:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("residual_dropout requires IDENTICAL mode")
    if n<0 or n>2147483647 or offset<0 or offset>9223372036854775807-n:
        raise Error("residual_dropout element range exceeds native Philox coordinates")
    if seed_lo<0 or seed_lo>4294967295 or seed_hi<0 or seed_hi>4294967295 or stream<0 or stream>4294967295:
        raise Error("residual_dropout seed words and stream must be uint32")
    if not isfinite(p) or p<Float32(0) or p>=Float32(1):
        raise Error("residual_dropout p must be finite and in [0,1)")
    var scale = ftz(identical_div(Float32(1),ftz(Float32(1)-ftz(p))))
    if not isfinite(scale) or scale<=Float32(0):
        raise Error("residual_dropout scale is not finite and positive")
    return scale


@always_inline
def nn_dropout_cell(value: Float32,p: Float32,scale: Float32,
    seed_lo: UInt32,seed_hi: UInt32,stream: UInt32,index: Int) -> Float32:
    # Probability is part of the portable FP32/FTZ contract too. In particular
    # a subnormal positive threshold must not compare differently against a
    # zero Philox draw on host and a device with denormal-flushing comparisons.
    if nn59_unit(seed_lo,seed_hi,stream,index)>=ftz(p):
        return ftz(identical_mul(ftz(value),ftz(scale)))
    return Float32(0)


@always_inline
def nn_dropout_residual_cell(value: Float32,residual: Float32,p: Float32,scale: Float32,
    seed_lo: UInt32,seed_hi: UInt32,stream: UInt32,index: Int) -> Float32:
    return ftz(ftz(residual)+ftz(nn_dropout_cell(value,p,scale,seed_lo,seed_hi,stream,index)))
