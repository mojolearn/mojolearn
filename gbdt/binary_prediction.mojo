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
from max.gpu.host import DeviceContext
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL

#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojoGbdtContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoGbdtContextFast"

from checks.numerics import ftz, identical_sigmoid
from checks.soft_f64 import SF64_ONE, sf64_ftz, sf64_sigmoid_f64, sf64_sub
from core.device_scan import device_first_nonfinite


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


def binary_prediction_device[probabilities: Bool, dtype: DType](
    raw_addr: Int, out_addr: Int, n: Int,
) raises:
    """cpu2-l6-bindings: the caller's Float32 margins go to the device from
    their own address (no host List copy), the finiteness refusal is the
    device scan (`core/device_scan.device_first_nonfinite`, one Int32 per
    block read back, never the data) and the kernel's output is copied
    straight into the caller's buffer. Same kernel, same bits as the host
    staging this replaces; the host column restates it in
    `bindings/_mojolearn_gbdt_host.mojo`."""
    comptime assert (probabilities and dtype == DType.float32) or (not probabilities and dtype == DType.int32)
    if n <= 0 or n > 2147483647:
        raise Error("binary prediction: positive n<=Int32.max required")
    if raw_addr == 0 or out_addr == 0:
        raise Error("binary prediction: null buffer")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var device_raw = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(
        dst_buf=device_raw,
        src_ptr=MutPointer[Float32, MutAnyOrigin](unsafe_from_address=raw_addr),
    )
    if device_first_nonfinite(ctx, device_raw, n) >= 0:
        raise Error("binary prediction: finite Float32 margins required")
    var count = 2*n if probabilities else n
    var output = ctx.enqueue_create_buffer[dtype](count)
    ctx.enqueue_function[binary_prediction_kernel[probabilities,dtype]](
        device_raw.unsafe_ptr(),output.unsafe_ptr(),Int32(n),
        grid_dim=(n+255)//256,block_dim=256,
    )
    ctx.enqueue_copy(
        dst_ptr=MutPointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=out_addr),
        src_buf=output,
    )
    ctx.synchronize()
    _ = output^
    _ = device_raw^
    _ = ctx^


def sigmoid_f64_kernel(
    raw: MutPointer[UInt64, MutAnyOrigin], output: MutPointer[UInt64, MutAnyOrigin],
    n_in: Int32, pair: Int32, swap: Int32,
):
    """`gbdt_sigmoid` (`pair == 0`: `out[i] = sigmoid(raw[i])`, soft
    binary64, no flush) and `gbdt_sigmoid_pair` (`pair != 0`: `out[2i] =
    1 - p`, `out[2i + 1] = p` with `p` flushed, swapped under `swap`), the
    host loops' statements word for word: integer soft-f64, the same bits
    on every vendor and on the host column."""
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var r = raw.unsafe_load(i)
    if pair == 0:
        output.unsafe_store(i, sf64_sigmoid_f64(r))
        return
    var p = sf64_ftz(sf64_sigmoid_f64(r))
    var q = sf64_sub(SF64_ONE, p)
    if swap != 0:
        output.unsafe_store(2*i, p)
        output.unsafe_store(2*i+1, q)
    else:
        output.unsafe_store(2*i, q)
        output.unsafe_store(2*i+1, p)


def sigmoid_f64_device(
    raw_addr: Int, out_addr: Int, n: Int, pair: Bool, swap: Bool,
) raises:
    """cpu2-l6-bindings: the Logloss / CrossEntropy probability link on the
    device. `raw_addr` holds `n` float64 margins, `out_addr` receives `n`
    (or `2n` for the pair) float64 values; both are the caller's. One
    upload, one launch, one download straight into the caller's buffer."""
    if n <= 0:
        return
    if n > 2147483647:
        raise Error("gbdt_sigmoid: n must be <= Int32.max")
    if raw_addr == 0 or out_addr == 0:
        raise Error("mojolearn: null buffer address")
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var count = 2*n if pair else n
    var d_raw = ctx.enqueue_create_buffer[DType.uint64](n)
    var d_out = ctx.enqueue_create_buffer[DType.uint64](count)
    ctx.enqueue_copy(
        dst_buf=d_raw,
        src_ptr=MutPointer[UInt64, MutAnyOrigin](unsafe_from_address=raw_addr),
    )
    ctx.enqueue_function[sigmoid_f64_kernel](
        d_raw.unsafe_ptr(), d_out.unsafe_ptr(), Int32(n),
        Int32(1) if pair else Int32(0), Int32(1) if swap else Int32(0),
        grid_dim=(n+255)//256, block_dim=256,
    )
    ctx.enqueue_copy(
        dst_ptr=MutPointer[UInt64, MutAnyOrigin](unsafe_from_address=out_addr),
        src_buf=d_out,
    )
    ctx.synchronize()
    _ = d_out^
    _ = d_raw^
    _ = ctx^
