# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Separate Float32 GPU binary prediction postprocessing for sklearn adapters.

Reference CatBoost libs/model/eval_processing.h:186-200: Probability calls
CalcSigmoid; Class compares raw margin > BinclassRawValueBorder. This slice
supports the default zero border and binary Logloss only.
BINARY-PRED-1: Float32 GPU sigmoid/complement replaces the reference's double
host link for this NEW entry only; legacy gbdt_sigmoid is unchanged. IDENTICAL
uses existing portable sigmoid with operand/result FTZ. No clipping.
BINARY-PRED-2: integer sign/magnitude class selection preserves strict raw>0
for positive subnormal margins on devices that flush floating comparisons.
Thus rounded probabilities can tie while raw-margin classification is positive.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast
from std.math import isfinite
from max.gpu.host import DeviceContext
from checks.numerics import ftz, identical_sigmoid
from metrics.checks.device_io import upload_f32


def binary_prediction_kernel[probabilities: Bool, dtype: DType](
    raw: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Scalar[dtype], MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i < Int(n_in):
        var margin = raw.unsafe_load(i)
        comptime if probabilities:
            var positive = ftz(identical_sigmoid(ftz(margin)))
            output.unsafe_store(2*i,Scalar[dtype](ftz(Float32(1)-positive)))
            output.unsafe_store(2*i+1,Scalar[dtype](positive))
        else:
            var bits = bitcast[DType.uint32](margin)
            var positive = (bits & UInt32(0x80000000)) == 0 and (bits & UInt32(0x7fffffff)) != 0
            output.unsafe_store(i,Scalar[dtype](Int32(1) if positive else Int32(0)))


def binary_prediction_host[probabilities: Bool, dtype: DType](
    raw: List[Float32], n: Int,
) raises -> List[Scalar[dtype]]:
    comptime assert (probabilities and dtype == DType.float32) or (not probabilities and dtype == DType.int32)
    if n <= 0 or n > 2147483647 or len(raw) < n:
        raise Error("binary prediction: positive n<=Int32.max required")
    for i in range(n):
        if not isfinite(raw[i]):
            raise Error("binary prediction: finite Float32 margins required")
    var ctx = DeviceContext()
    var device_raw = upload_f32(ctx,raw)
    var count = 2*n if probabilities else n
    var output = ctx.enqueue_create_buffer[dtype](count)
    ctx.enqueue_function[binary_prediction_kernel[probabilities,dtype]](
        device_raw.unsafe_ptr(),output.unsafe_ptr(),Int32(n),
        grid_dim=(n+255)//256,block_dim=256,
    )
    var result = List[Scalar[dtype]]()
    with output.map_to_host() as h:
        for i in range(count):
            result.append(h[i])
    _ = output^
    _ = device_raw^
    _ = ctx^
    return result^
