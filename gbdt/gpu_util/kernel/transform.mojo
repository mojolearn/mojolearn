"""`GatherWithMask` and `ScatterWithMask`: the two masked permutation moves.

PORT OF `catboost/cuda/cuda_util/kernel/transform.cu:214-274` at CatBoost
`54a8143a` -- `GatherWithMaskImpl`/`GatherWithMask` (`:214-233`) and
`ScatterWithMaskImpl`/`ScatterWithMask` (`:257-276`).

Both are three lines of arithmetic and one grid-stride loop:

    Gather :  dst[i] = src[map[i] & mask]
    Scatter:  dst[map[i] & mask] = src[i]

The MASK is what makes them "WithMask" rather than the plain `Gather` and
`Scatter` fifty lines above them: every index in the CTR block carries a
segment-start flag in bit 31, so the low bits have to be isolated before the
word is used as an address. Every CTR call site passes
`TCtrBinBuilder<T>::GetMask()`, the same `0x3FFFFFFF` that
`TIndexWrapper::Index()` applies (`ctr_bins_builder.h:265`,
`index_wrapper.cuh:22-24`). `gbdt/ctrs/index_wrapper.mojo` carries the
argument for why it is THIRTY bits and not thirty-one.

Where the CTR path uses them, in call order:

    ctr_bins_builder.h:145   ScatterWithMask(dst, tmp, indices, Mask)
                             -- ComputeCurrentBins' last step
    ctr_bins_builder.h:216   GatherWithMask(Bins, DecompressedTempBins,
                                            Indices, Mask)
                             -- the raw cat bins into sorted order
    ctr_calcers.h:71         GatherWithMask(GatheredWeightsWithMask,
                                            weights, Indices, Mask)
    ctr_calcers.h:261        GatherWithMask(GatheredBinarizedSample,
                                            BinarizedSample, Indices, Mask)
                             -- GetGatheredBinSample

`StreamLoad` and `WriteThrough` are CUB thread-load/store hints
(`kernel_helpers.cuh:174-192`) and are plain accesses here; Mojo 1.0 ships
no non-temporal load or store. Same deviation `fill.mojo`,
`split_points.mojo` and `ctrs/kernel/ctr_calcers.mojo` already record.

Their `numBlocks` is `min(CeilDivide(size, 256), TArchProps::MaxBlockCount())`
and the kernels grid-stride, so the cap changes no answer; `TRANSFORM_MAX_BLOCKS`
is the same stand-in constant `fill.mojo` uses for `MaxBlockCount()`.

## ADDRESS NOTE

Their file is `cuda_util/kernel/transform.cu` and this is the mirror address
for it. Only the two masked moves are ported -- `Gather`, `Scatter`,
`Reverse`, and the arithmetic transforms in the same file have no caller
here, and porting a function nothing reaches is the defect
`PORTING_RULES.md` rule 3 names.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext


comptime TRANSFORM_BLOCK_SIZE = 256
"""`const ui64 blockSize = 256` (`transform.cu:226`, `:270`)."""

comptime TRANSFORM_MAX_BLOCKS = 65535
"""Stand-in for `TArchProps::MaxBlockCount()`; both kernels grid-stride, so
any large constant gives the same answer (`fill.mojo` says the same)."""


def _transform_blocks(size: Int) -> Int:
    var blocks = (size + TRANSFORM_BLOCK_SIZE - 1) // TRANSFORM_BLOCK_SIZE
    if blocks > TRANSFORM_MAX_BLOCKS:
        blocks = TRANSFORM_MAX_BLOCKS
    return blocks


def gather_with_mask_u32_kernel(
    dst: MutPointer[UInt32, MutAnyOrigin],
    src: MutPointer[UInt32, MutAnyOrigin],
    map_ptr: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    mask: UInt32,
):
    """`GatherWithMaskImpl<ui32, ui32>` (`transform.cu:214-221`)."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while i < size:
        var m = map_ptr.unsafe_load(i) & mask
        dst.unsafe_store(i, src.unsafe_load(Int(m)))
        i += Int(grid_dim.x) * Int(block_dim.x)


def gather_with_mask_u8_kernel(
    dst: MutPointer[UInt8, MutAnyOrigin],
    src: MutPointer[UInt8, MutAnyOrigin],
    map_ptr: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    mask: UInt32,
):
    """`GatherWithMaskImpl<ui8, ui32>`, the instantiation
    `GetGatheredBinSample` takes (`ctr_calcers.h:258-265`)."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while i < size:
        var m = map_ptr.unsafe_load(i) & mask
        dst.unsafe_store(i, src.unsafe_load(Int(m)))
        i += Int(grid_dim.x) * Int(block_dim.x)


def gather_with_mask_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    map_ptr: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    mask: UInt32,
):
    """`GatherWithMaskImpl<float, ui32>`, the instantiation the
    non-trivial-weights `Reset` takes (`ctr_calcers.h:71`)."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while i < size:
        var m = map_ptr.unsafe_load(i) & mask
        dst.unsafe_store(i, src.unsafe_load(Int(m)))
        i += Int(grid_dim.x) * Int(block_dim.x)


def scatter_with_mask_u32_kernel(
    dst: MutPointer[UInt32, MutAnyOrigin],
    src: MutPointer[UInt32, MutAnyOrigin],
    map_ptr: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    mask: UInt32,
):
    """`ScatterWithMaskImpl<ui32, ui32>` (`transform.cu:257-264`)."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while i < size:
        var m = map_ptr.unsafe_load(i) & mask
        dst.unsafe_store(Int(m), src.unsafe_load(i))
        i += Int(grid_dim.x) * Int(block_dim.x)


def launch_gather_with_mask_u32(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.uint32],
    mut src: DeviceBuffer[DType.uint32],
    mut map_buf: DeviceBuffer[DType.uint32],
    size: Int,
    mask: UInt32,
) raises:
    """`GatherWithMask<ui32, ui32>` (`transform.cu:224-233`)."""
    var blocks = _transform_blocks(size)
    if blocks == 0:
        return
    ctx.enqueue_function[gather_with_mask_u32_kernel](
        dst.unsafe_ptr(), src.unsafe_ptr(), map_buf.unsafe_ptr(),
        Int32(size), mask,
        grid_dim=(blocks, 1, 1),
        block_dim=(TRANSFORM_BLOCK_SIZE, 1, 1),
    )


def launch_gather_with_mask_u8(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.uint8],
    mut src: DeviceBuffer[DType.uint8],
    mut map_buf: DeviceBuffer[DType.uint32],
    size: Int,
    mask: UInt32,
) raises:
    """`GatherWithMask<ui8, ui32>` (`transform.cu:224-233`)."""
    var blocks = _transform_blocks(size)
    if blocks == 0:
        return
    ctx.enqueue_function[gather_with_mask_u8_kernel](
        dst.unsafe_ptr(), src.unsafe_ptr(), map_buf.unsafe_ptr(),
        Int32(size), mask,
        grid_dim=(blocks, 1, 1),
        block_dim=(TRANSFORM_BLOCK_SIZE, 1, 1),
    )


def gather_planes_with_mask_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    map_ptr: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    mask: UInt32,
    n_planes_in: Int32,
    stride_in: Int32,
):
    """`GatherWithMask` over a MULTI-COLUMN buffer.

    Their `Gather` takes a `TCudaBuffer` and moves every column
    (`multiclass_targets.cpp:30`, `pointwise_oracle`'s factory), because a
    column count is a property of the buffer there. Ours is a pointer plus
    a stride, so the plane count is an argument and `block_idx.y` is the
    plane -- the same axis `add_model_value_kernel` grew for the same
    reason.

    `n_planes == 1` reproduces `gather_with_mask_f32_kernel` exactly.
    """
    var size = Int(size_in)
    var stride = Int(stride_in)
    var plane = Int(block_idx.y)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var step = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        var srow = Int(map_ptr.unsafe_load(i) & mask)
        dst.unsafe_store(
            plane * stride + i, src.unsafe_load(plane * stride + srow)
        )
        i += step
    _ = n_planes_in


def launch_gather_planes_with_mask_f32(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    mut map_buf: DeviceBuffer[DType.uint32],
    size: Int,
    mask: UInt32,
    n_planes: Int,
    stride: Int,
) raises:
    """`Gather(dst, src, indices)` over `n_planes` columns."""
    var blocks = _transform_blocks(size)
    if blocks == 0 or n_planes <= 0:
        return
    ctx.enqueue_function[gather_planes_with_mask_f32_kernel](
        dst.unsafe_ptr(), src.unsafe_ptr(), map_buf.unsafe_ptr(),
        Int32(size), mask, Int32(n_planes), Int32(stride),
        grid_dim=(blocks, n_planes, 1),
        block_dim=(TRANSFORM_BLOCK_SIZE, 1, 1),
    )


def launch_gather_with_mask_f32(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    mut map_buf: DeviceBuffer[DType.uint32],
    size: Int,
    mask: UInt32,
) raises:
    """`GatherWithMask<float, ui32>` (`transform.cu:224-233`)."""
    var blocks = _transform_blocks(size)
    if blocks == 0:
        return
    ctx.enqueue_function[gather_with_mask_f32_kernel](
        dst.unsafe_ptr(), src.unsafe_ptr(), map_buf.unsafe_ptr(),
        Int32(size), mask,
        grid_dim=(blocks, 1, 1),
        block_dim=(TRANSFORM_BLOCK_SIZE, 1, 1),
    )


def launch_scatter_with_mask_u32(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.uint32],
    mut src: DeviceBuffer[DType.uint32],
    mut map_buf: DeviceBuffer[DType.uint32],
    size: Int,
    mask: UInt32,
) raises:
    """`ScatterWithMask<ui32, ui32>` (`transform.cu:267-276`)."""
    var blocks = _transform_blocks(size)
    if blocks == 0:
        return
    ctx.enqueue_function[scatter_with_mask_u32_kernel](
        dst.unsafe_ptr(), src.unsafe_ptr(), map_buf.unsafe_ptr(),
        Int32(size), mask,
        grid_dim=(blocks, 1, 1),
        block_dim=(TRANSFORM_BLOCK_SIZE, 1, 1),
    )
