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
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

#: lane/apple-fast-prep (2026-10-02): the row-tiled extrema kernel below is
#: the fit's extrema pass (and `minmax_transform_direct` the transform entry)
#: on the FAST tier on Apple. DEFAULT since the M3 A/B (lane/apple-fast-prep
#: 387211293, n=1, output digests identical: minmax-scaler istella 152 ->
#: 106 ms). `-D MOJOLEARN_PREP_FAST_MINMAX_OFF` restores main's path; the old
#: `-D MOJOLEARN_PREP_FAST_MINMAX` stays harmless. Every other build takes
#: `extrema_chunks_kernel`, so IDENTICAL compiles main's code unchanged.
comptime PREP_FAST_MINMAX = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_PREP_FAST_MINMAX_OFF"]()
)
from checks.numerics import ftz, identical_div, identical_mul

#: lane/apple-fast-gap-cls2 (2026-10-03), FAST + Apple, default ON (below):
#: `-D MOJOLEARN_PREP_FAST_CLS2_MINMAX_FUSED` folds the fit's nonfinite scan
#: (core/device_scan.mojo device_first_nonfinite: a second full read of X, a
#: partials buffer, a host buffer and its own synchronize) into the extrema
#: pass (`extrema_rows_flag_kernel` / `extrema_finalize_flag_kernel`, one
#: flag row more in the one download). `-D MOJOLEARN_PREP_FAST_CLS2_MINMAX_POOL`
#: (preprocessing/estimator.mojo) takes the X buffer from a pool kept between
#: fits instead of a fresh 880 MB allocation per fit at the board's Istella
#: shape. Both need PREP_FAST_MINMAX. Same keys, same partials, same finalize:
#: the same words. Both are the FAST + Apple default since the M3 A/B (n=1,
#: quality identical): minmax-scaler istella 104.7 -> 19.1 ms (FUSED+POOL;
#: POOL alone 106 -> 21.9). `-D MOJOLEARN_PREP_FAST_CLS2_MINMAX_FUSED_OFF` /
#: `_POOL_OFF` turn them off; the old -D names stay harmless.
comptime PREP_CLS2_MINMAX_FUSED = PREP_FAST_MINMAX and not is_defined["MOJOLEARN_PREP_FAST_CLS2_MINMAX_FUSED_OFF"]()
comptime PREP_CLS2_MINMAX_POOL = PREP_FAST_MINMAX and not is_defined["MOJOLEARN_PREP_FAST_CLS2_MINMAX_POOL_OFF"]()
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


#: rows per block of `extrema_rows_fast_kernel`: `extrema_chunks_kernel`'s
#: chunk of 256 rows, so `extrema_finalize_kernel` folds the same partials
comptime FAST_ROWS = 256


def extrema_rows_fast_kernel(
    x: MutPointer[Float32, MutAnyOrigin], n: Int32, d: Int32, tpb: Int32,
    lows: MutPointer[UInt32, MutAnyOrigin], highs: MutPointer[UInt32, MutAnyOrigin],
):
    """lane/apple-fast-prep (2026-10-02), `PREP_FAST_MINMAX` (FAST on Apple,
    default; -D MOJOLEARN_PREP_FAST_MINMAX_OFF turns it off) only, from `minmax_fit_fast`. Block
    (chunk, column group): thread t owns column group*tpb + t over the
    chunk's FAST_ROWS rows, so a simdgroup reads consecutive words of one
    row. `extrema_chunks_kernel` (above) gives
    a block one column: its 256 threads read 256 words d apart, a cache line
    each, for every row of the 220-column Istella block (board minmax-scaler
    Istella 5.7x behind scikit-learn). The same ordered keys folded by the
    same min / max into the same (chunk, column) partials: the same words."""
    var column = Int(block_idx.y) * Int(tpb) + Int(thread_idx.x)
    var chunk = Int(block_idx.x)
    var dd = Int(d)
    if column >= dd:
        return
    var row0 = chunk * FAST_ROWS
    var row1 = min(row0 + FAST_ROWS, Int(n))
    var lo = UInt32(0xffffffff)
    var hi = UInt32(0)
    for row in range(row0, row1):
        var k = ordered_key(x.unsafe_load(row * dd + column))
        lo = min(lo, k)
        hi = max(hi, k)
    var index = chunk * dd + column
    lows.unsafe_store(index, lo)
    highs.unsafe_store(index, hi)


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


def minmax_fit_dev(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], n: Int, d: Int,
    lower: Float32, upper: Float32,
) raises -> DeviceBuffer[DType.float32]:
    """`minmax_fit`'s five rows of d, left on the device."""
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
    _ = highs^
    _ = lows^
    return output^


def minmax_fit(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], n: Int, d: Int,
    lower: Float32, upper: Float32,
) raises -> List[Float32]:
    var output = minmax_fit_dev(ctx,x,n,d,lower,upper)
    var result = download_f32(ctx,output,5*d)
    _ = output^
    return result^


def minmax_fit_fast_dev(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], n: Int, d: Int,
    lower: Float32, upper: Float32,
) raises -> DeviceBuffer[DType.float32]:
    """`minmax_fit` with the row-tiled `extrema_rows_fast_kernel` under
    `PREP_FAST_MINMAX` (lane apple-fast-prep): the same chunk count, the same
    finalize. Any other build: `minmax_fit` itself (no new kernel). The five
    rows stay on the device (`minmax_fit_fast` downloads them)."""
    comptime if not PREP_FAST_MINMAX:
        return minmax_fit_dev(ctx,x,n,d,lower,upper)
    else:
        var chunks = (n + FAST_ROWS - 1) // FAST_ROWS
        var tpb = 256 if d >= 256 else ((d + 31) // 32) * 32
        var cgroups = (d + tpb - 1) // tpb
        var lows = ctx.enqueue_create_buffer[DType.uint32](chunks*d)
        var highs = ctx.enqueue_create_buffer[DType.uint32](chunks*d)
        var output = ctx.enqueue_create_buffer[DType.float32](5*d)
        ctx.enqueue_function[extrema_rows_fast_kernel](
            x.unsafe_ptr(), Int32(n), Int32(d), Int32(tpb), lows.unsafe_ptr(), highs.unsafe_ptr(),
            grid_dim=(chunks, cgroups), block_dim=tpb,
        )
        ctx.enqueue_function[extrema_finalize_kernel](
            lows.unsafe_ptr(),highs.unsafe_ptr(),Int32(chunks),Int32(d),lower,upper,output.unsafe_ptr(),
            grid_dim=(d+255)//256,block_dim=256,
        )
        _ = highs^
        _ = lows^
        return output^


def minmax_fit_fast(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], n: Int, d: Int,
    lower: Float32, upper: Float32,
) raises -> List[Float32]:
    var output = minmax_fit_fast_dev(ctx,x,n,d,lower,upper)
    var result = download_f32(ctx,output,5*d)
    _ = output^
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


def minmax_transform_into(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32],
    mut scale: DeviceBuffer[DType.float32], mut offset: DeviceBuffer[DType.float32],
    mut output: DeviceBuffer[DType.float32], n: Int, d: Int, inverse: Int,
    clip: Int, lower: Float32, upper: Float32,
) raises:
    """Transform into caller-owned device storage without a host materialization."""
    ctx.enqueue_function[minmax_transform_kernel](
        x.unsafe_ptr(),scale.unsafe_ptr(),offset.unsafe_ptr(),output.unsafe_ptr(),
        Int32(n*d),Int32(d),Int32(inverse),Int32(clip),lower,upper,
        grid_dim=(n*d+255)//256,block_dim=256,
    )


def extrema_rows_flag_kernel(
    x: MutPointer[Float32, MutAnyOrigin], n: Int32, d: Int32, tpb: Int32,
    lows: MutPointer[UInt32, MutAnyOrigin], highs: MutPointer[UInt32, MutAnyOrigin],
    bad: MutPointer[UInt32, MutAnyOrigin],
):
    """`extrema_rows_fast_kernel` (the same keys into the same partials) plus
    bad[(chunk, column)] = 1 when one of the chunk's words of the column is a
    NaN or an infinity (by bits, as core/device_scan.mojo's predicate).
    lane/apple-fast-gap-cls2, PREP_CLS2_MINMAX_FUSED only."""
    var column = Int(block_idx.y) * Int(tpb) + Int(thread_idx.x)
    var chunk = Int(block_idx.x)
    var dd = Int(d)
    if column >= dd:
        return
    var row0 = chunk * FAST_ROWS
    var row1 = min(row0 + FAST_ROWS, Int(n))
    var lo = UInt32(0xffffffff)
    var hi = UInt32(0)
    var nf = UInt32(0)
    for row in range(row0, row1):
        var v = x.unsafe_load(row * dd + column)
        var k = ordered_key(v)
        lo = min(lo, k)
        hi = max(hi, k)
        if (bitcast[DType.uint32](v) & UInt32(0x7fffffff)) >= UInt32(0x7f800000):
            nf = UInt32(1)
    var index = chunk * dd + column
    lows.unsafe_store(index, lo)
    highs.unsafe_store(index, hi)
    bad.unsafe_store(index, nf)


def extrema_finalize_flag_kernel(
    lows: MutPointer[UInt32, MutAnyOrigin], highs: MutPointer[UInt32, MutAnyOrigin],
    bad: MutPointer[UInt32, MutAnyOrigin],
    chunks: Int32, d_in: Int32, lower: Float32, upper: Float32,
    output: MutPointer[Float32, MutAnyOrigin],
):
    """`extrema_finalize_kernel`'s five rows (the same arithmetic) and a sixth:
    1 when any chunk flagged the column, else 0."""
    var column = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var d = Int(d_in)
    if column < d:
        var lo = UInt32(0xffffffff)
        var hi = UInt32(0)
        var nf = UInt32(0)
        for chunk in range(Int(chunks)):
            lo = min(lo,lows.unsafe_load(chunk*d+column))
            hi = max(hi,highs.unsafe_load(chunk*d+column))
            nf = nf | bad.unsafe_load(chunk*d+column)
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
        output.unsafe_store(5*d+column,Float32(1) if nf != UInt32(0) else Float32(0))


def minmax_fit_flagged_dev(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], n: Int, d: Int,
    lower: Float32, upper: Float32,
) raises -> DeviceBuffer[DType.float32]:
    """`minmax_fit_fast`'s five rows of d and a sixth row of nonfinite flags,
    from ONE read of X (PREP_CLS2_MINMAX_FUSED only), left on the device."""
    comptime if not PREP_CLS2_MINMAX_FUSED:
        raise Error("MinMaxScaler: minmax_fit_flagged is a FAST Apple switch")
    else:
        var chunks = (n + FAST_ROWS - 1) // FAST_ROWS
        var tpb = 256 if d >= 256 else ((d + 31) // 32) * 32
        var cgroups = (d + tpb - 1) // tpb
        var lows = ctx.enqueue_create_buffer[DType.uint32](chunks*d)
        var highs = ctx.enqueue_create_buffer[DType.uint32](chunks*d)
        var bad = ctx.enqueue_create_buffer[DType.uint32](chunks*d)
        var output = ctx.enqueue_create_buffer[DType.float32](6*d)
        ctx.enqueue_function[extrema_rows_flag_kernel](
            x.unsafe_ptr(), Int32(n), Int32(d), Int32(tpb), lows.unsafe_ptr(), highs.unsafe_ptr(), bad.unsafe_ptr(),
            grid_dim=(chunks, cgroups), block_dim=tpb,
        )
        ctx.enqueue_function[extrema_finalize_flag_kernel](
            lows.unsafe_ptr(),highs.unsafe_ptr(),bad.unsafe_ptr(),Int32(chunks),Int32(d),lower,upper,output.unsafe_ptr(),
            grid_dim=(d+255)//256,block_dim=256,
        )
        _ = bad^
        _ = highs^
        _ = lows^
        return output^


def minmax_fit_flagged(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], n: Int, d: Int,
    lower: Float32, upper: Float32,
) raises -> List[Float32]:
    """`minmax_fit_flagged_dev`, downloaded (one synchronize)."""
    var output = minmax_fit_flagged_dev(ctx,x,n,d,lower,upper)
    var result = download_f32(ctx,output,6*d)
    _ = output^
    return result^
