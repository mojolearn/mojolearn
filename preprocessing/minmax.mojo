# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Dense finite Float32 MinMaxScaler GPU arithmetic.

Source: upstream/cuml-v26.08.00/python/cuml/cuml/_thirdparty/sklearn/
preprocessing/_data.py:378-400 (fit), :423-424 (transform), :447-448 (inverse).
MINMAX-1: sklearn sklearn/preprocessing/_data.py:101-134 near-constant
range<10*eps replaces pinned cuML :72-87 exact-zero handling.
MINMAX-2: 256-row integer total-order extrema replace nanmin/nanmax. This
finite-only slice preserves subnormal values and chooses -0 for min, +0 for
max when both occur, independent of arrival order. No NaN omission support.
MINMAX-3: Float32 arithmetic uses IDENTICAL operand/seam FTZ and portable
 division. Multiplication rounds before offset addition. Host validation
rejects nonfinite statistics/results and nonpositive scale; no CPU reduction.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz, identical_div, identical_mul
from metrics.checks.device_io import download_f32


def ordered_key(value: Float32) -> UInt32:
    var bits = bitcast[DType.uint32](value)
    return ~bits if (bits & UInt32(0x80000000)) != 0 else bits | UInt32(0x80000000)


def key_value(key: UInt32) -> Float32:
    var bits = key & UInt32(0x7fffffff) if (key & UInt32(0x80000000)) != 0 else ~key
    return bitcast[DType.float32](bits)


def extrema_chunks_kernel(
    x: MutPointer[Float32, MutAnyOrigin], n: Int32, d: Int32,
    lows: MutPointer[UInt32, MutAnyOrigin], highs: MutPointer[UInt32, MutAnyOrigin],
):
    var tid = Int(thread_idx.x)
    var chunk = Int(block_idx.x) // Int(d)
    var column = Int(block_idx.x) % Int(d)
    var row = chunk*256+tid
    var lo = UInt32(0xffffffff)
    var hi = UInt32(0)
    if row < Int(n):
        lo = ordered_key(x.unsafe_load(row*Int(d)+column))
        hi = lo
    var slab = stack_allocation[512, UInt32, address_space=AddressSpace.SHARED]()
    slab[unsafe_offset=tid] = lo
    slab[unsafe_offset=256+tid] = hi
    barrier()
    var step = 128
    while step > 0:
        if tid < step:
            slab[unsafe_offset=tid] = min(slab[unsafe_offset=tid],slab[unsafe_offset=tid+step])
            slab[unsafe_offset=256+tid] = max(slab[unsafe_offset=256+tid],slab[unsafe_offset=256+tid+step])
        barrier()
        step //= 2
    if tid == 0:
        var index = chunk*Int(d)+column
        lows.unsafe_store(index,slab[unsafe_offset=0])
        highs.unsafe_store(index,slab[unsafe_offset=256])


def extrema_finalize_kernel(
    lows: MutPointer[UInt32, MutAnyOrigin], highs: MutPointer[UInt32, MutAnyOrigin],
    chunks: Int32, d_in: Int32, lower: Float32, upper: Float32,
    output: MutPointer[Float32, MutAnyOrigin],
):
    var column = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var d = Int(d_in)
    if column < d:
        var lo = UInt32(0xffffffff)
        var hi = UInt32(0)
        for chunk in range(Int(chunks)):
            lo = min(lo,lows.unsafe_load(chunk*d+column))
            hi = max(hi,highs.unsafe_load(chunk*d+column))
        var data_min = key_value(lo)
        var data_max = key_value(hi)
        var data_range = ftz(ftz(data_max)-ftz(data_min))
        var denominator = Float32(1) if data_range < Float32(0.0000011920928955078125) else data_range
        var scale = ftz(identical_div(ftz(ftz(upper)-ftz(lower)),denominator))
        var offset = ftz(ftz(lower)-ftz(identical_mul(ftz(data_min),scale)))
        output.unsafe_store(column,data_min)
        output.unsafe_store(d+column,data_max)
        output.unsafe_store(2*d+column,data_range)
        output.unsafe_store(3*d+column,scale)
        output.unsafe_store(4*d+column,offset)


def minmax_transform_kernel(
    x: MutPointer[Float32, MutAnyOrigin], scale: MutPointer[Float32, MutAnyOrigin],
    offset: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin],
    count: Int32, d: Int32, inverse: Int32, clip: Int32, lower: Float32, upper: Float32,
):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i < Int(count):
        var c = i % Int(d)
        var value = ftz(x.unsafe_load(i))
        if inverse != 0:
            value = ftz(identical_div(ftz(value-ftz(offset.unsafe_load(c))),ftz(scale.unsafe_load(c))))
        else:
            # Explicit rounded multiplication keeps sklearn's two operations.
            value = ftz(ftz(identical_mul(value,ftz(scale.unsafe_load(c))))+ftz(offset.unsafe_load(c)))
            if clip != 0:
                value = min(max(value,lower),upper)
        output.unsafe_store(i,value)


def minmax_fit(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], n: Int, d: Int,
    lower: Float32, upper: Float32,
) raises -> List[Float32]:
    var chunks = (n+255)//256
    var lows = ctx.enqueue_create_buffer[DType.uint32](chunks*d)
    var highs = ctx.enqueue_create_buffer[DType.uint32](chunks*d)
    var output = ctx.enqueue_create_buffer[DType.float32](5*d)
    ctx.enqueue_function[extrema_chunks_kernel](
        x.unsafe_ptr(),Int32(n),Int32(d),lows.unsafe_ptr(),highs.unsafe_ptr(),
        grid_dim=chunks*d,block_dim=256,
    )
    ctx.enqueue_function[extrema_finalize_kernel](
        lows.unsafe_ptr(),highs.unsafe_ptr(),Int32(chunks),Int32(d),lower,upper,output.unsafe_ptr(),
        grid_dim=(d+255)//256,block_dim=256,
    )
    var result = download_f32(ctx,output,5*d)
    _ = output^
    _ = highs^
    _ = lows^
    return result^


def minmax_transform(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32],
    mut scale: DeviceBuffer[DType.float32], mut offset: DeviceBuffer[DType.float32],
    n: Int, d: Int, inverse: Int, clip: Int, lower: Float32, upper: Float32,
) raises -> List[Float32]:
    var output = ctx.enqueue_create_buffer[DType.float32](n*d)
    ctx.enqueue_function[minmax_transform_kernel](
        x.unsafe_ptr(),scale.unsafe_ptr(),offset.unsafe_ptr(),output.unsafe_ptr(),
        Int32(n*d),Int32(d),Int32(inverse),Int32(clip),lower,upper,
        grid_dim=(n*d+255)//256,block_dim=256,
    )
    var result = download_f32(ctx,output,n*d)
    _ = output^
    return result^
