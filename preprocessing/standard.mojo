# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Dense finite Float32 StandardScaler, GPU statistics and transforms.

References: cuML v26.08.00 python/cuml/cuml/_thirdparty/sklearn/preprocessing/
_data.py:815-849 and utils/extmath.py:123-145; sklearn preprocessing/_data.py
_is_constant_feature and utils/extmath.py _incremental_mean_and_var.
STD-1: fixed 256-row Float32 reductions replace upstream safe Float64 sums.
Variance is mean squared centered residual, without sklearn's correction term.
Exact constant columns retain their first value and zero variance despite sum
rounding. Final chunk folds are ascending, per feature, on GPU.
STD-2: exact zero variance gets scale1 (cuML), not sklearn's Float64 error bound.
STD-3: IDENTICAL operand/seam FTZ, pinned multiply/divide and portable sqrt.
Finite inputs/statistics/output only. Unused statistics are not computed.
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import ftz, identical_mul, identical_div, portable_sqrtf, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from metrics.checks.pinned_sum import virtual_block_sum
from metrics.checks.device_io import download_f32


def standard_initialize_kernel(output: MutPointer[Float32, MutAnyOrigin], d: Int32):
    var c = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if c < Int(d):
        output.unsafe_store(c,Float32(0))
        output.unsafe_store(Int(d)+c,Float32(0))
        output.unsafe_store(2*Int(d)+c,Float32(1))


def standard_chunks_kernel[variance: Bool](
    x: MutPointer[Float32, MutAnyOrigin], stats: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32, d_in: Int32, partials: MutPointer[Float32, MutAnyOrigin],
    differences: MutPointer[Float32, MutAnyOrigin],
):
    var d = Int(d_in)
    var chunk = Int(block_idx.x)//d
    var c = Int(block_idx.x)%d
    var tid = Int(thread_idx.x)
    var row = chunk*256+tid
    var value = Float32(0)
    var different = Float32(0)
    if row < Int(n_in):
        value = ftz(x.unsafe_load(row*d+c))
        comptime if variance:
            # The mean pass stores a constant-column marker in the variance row.
            if stats.unsafe_load(d+c) == 0:
                value = Float32(0)
            else:
                value = ftz(value-ftz(stats.unsafe_load(c)))
                value = ftz(identical_mul(value,value))
        else:
            different = Float32(1) if value != ftz(x.unsafe_load(c)) else Float32(0)
    var total = virtual_block_sum[256](SIMD[DType.float32,1](value))
    if tid == 0:
        partials.unsafe_store(chunk*d+c,ftz(total))
    comptime if not variance:
        var changed = virtual_block_sum[256](SIMD[DType.float32,1](different))
        if tid == 0:
            differences.unsafe_store(chunk*d+c,changed)


def standard_finalize_kernel[variance: Bool](
    x: MutPointer[Float32, MutAnyOrigin], partials: MutPointer[Float32, MutAnyOrigin],
    differences: MutPointer[Float32, MutAnyOrigin], n_in: Int32, d_in: Int32,
    chunks_in: Int32, with_std: Int32, output: MutPointer[Float32, MutAnyOrigin],
):
    var c = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    var d = Int(d_in)
    if c < d:
        var total = Float32(0)
        var changed = Float32(0)
        for chunk in range(Int(chunks_in)):
            total = ftz(total+partials.unsafe_load(chunk*d+c))
            comptime if not variance:
                changed += differences.unsafe_load(chunk*d+c)
        comptime if variance:
            var var_value = ftz(identical_div(total,Float32(n_in)))
            var scale = Float32(1)
            if var_value != 0:
                comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
                    scale = ftz(portable_sqrtf(var_value))
                else:
                    scale = sqrt(var_value)
            output.unsafe_store(d+c,var_value)
            output.unsafe_store(2*d+c,scale)
        else:
            var mean = ftz(x.unsafe_load(c)) if changed == 0 else ftz(identical_div(total,Float32(n_in)))
            output.unsafe_store(c,mean)
            if with_std != 0:
                output.unsafe_store(d+c,Float32(1) if changed != 0 else Float32(0))


def standard_transform_kernel(
    x: MutPointer[Float32, MutAnyOrigin], mean: MutPointer[Float32, MutAnyOrigin],
    scale: MutPointer[Float32, MutAnyOrigin], output: MutPointer[Float32, MutAnyOrigin],
    count: Int32, d: Int32, inverse: Int32, with_mean: Int32, with_std: Int32,
):
    var i = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if i < Int(count):
        var c = i % Int(d)
        var value = x.unsafe_load(i)
        # Both disabled is an exact identity copy, including subnormal bits.
        if inverse != 0:
            if with_std != 0:
                value = ftz(identical_mul(ftz(value),ftz(scale.unsafe_load(c))))
            if with_mean != 0:
                value = ftz(ftz(value)+ftz(mean.unsafe_load(c)))
        else:
            if with_mean != 0:
                value = ftz(ftz(value)-ftz(mean.unsafe_load(c)))
            if with_std != 0:
                value = ftz(identical_div(ftz(value),ftz(scale.unsafe_load(c))))
        output.unsafe_store(i,value)


def standard_fit(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32], n: Int, d: Int,
    with_mean: Int, with_std: Int,
) raises -> List[Float32]:
    var output = ctx.enqueue_create_buffer[DType.float32](3*d)
    ctx.enqueue_function[standard_initialize_kernel](output.unsafe_ptr(),Int32(d),grid_dim=(d+255)//256,block_dim=256)
    if with_mean != 0 or with_std != 0:
        var chunks = (n+255)//256
        var partials = ctx.enqueue_create_buffer[DType.float32](chunks*d)
        var differences = ctx.enqueue_create_buffer[DType.float32](chunks*d)
        ctx.enqueue_function[standard_chunks_kernel[False]](
            x.unsafe_ptr(),output.unsafe_ptr(),Int32(n),Int32(d),partials.unsafe_ptr(),differences.unsafe_ptr(),
            grid_dim=chunks*d,block_dim=256,
        )
        ctx.enqueue_function[standard_finalize_kernel[False]](
            x.unsafe_ptr(),partials.unsafe_ptr(),differences.unsafe_ptr(),Int32(n),Int32(d),Int32(chunks),Int32(with_std),output.unsafe_ptr(),
            grid_dim=(d+255)//256,block_dim=256,
        )
        if with_std != 0:
            ctx.enqueue_function[standard_chunks_kernel[True]](
                x.unsafe_ptr(),output.unsafe_ptr(),Int32(n),Int32(d),partials.unsafe_ptr(),differences.unsafe_ptr(),
                grid_dim=chunks*d,block_dim=256,
            )
            ctx.enqueue_function[standard_finalize_kernel[True]](
                x.unsafe_ptr(),partials.unsafe_ptr(),differences.unsafe_ptr(),Int32(n),Int32(d),Int32(chunks),Int32(with_std),output.unsafe_ptr(),
                grid_dim=(d+255)//256,block_dim=256,
            )
        ctx.synchronize()
        _ = differences^
        _ = partials^
    var result = download_f32(ctx,output,3*d)
    _ = output^
    return result^


def standard_transform(
    ctx: DeviceContext, mut x: DeviceBuffer[DType.float32],
    mut mean: DeviceBuffer[DType.float32], mut scale: DeviceBuffer[DType.float32],
    n: Int, d: Int, inverse: Int, with_mean: Int, with_std: Int,
) raises -> List[Float32]:
    var output = ctx.enqueue_create_buffer[DType.float32](n*d)
    ctx.enqueue_function[standard_transform_kernel](
        x.unsafe_ptr(),mean.unsafe_ptr(),scale.unsafe_ptr(),output.unsafe_ptr(),
        Int32(n*d),Int32(d),Int32(inverse),Int32(with_mean),Int32(with_std),
        grid_dim=(n*d+255)//256,block_dim=256,
    )
    var result = download_f32(ctx,output,n*d)
    _ = output^
    return result^
