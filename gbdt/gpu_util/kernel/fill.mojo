"""Their `fill.cu`, the parts this port reaches.

`MakeSequence` is how CatBoost materialises an identity (or offset) index ON
THE DEVICE (`cuda_util/kernel/fill.cu:37-55`). The boosting loop resets the
row index every iteration because tree growth permutes it in place; this
port used to UPLOAD a host-built identity array for that, 3.2 MB of H2D per
tree at 800k rows, which is not their design -- theirs never leaves the
device.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

#: `const ui32 blockSize = 512` (`fill.cu:51`).
comptime FILL_BLOCK_SIZE = 512

#: their grid is capped at `TArchProps::MaxBlockCount()`; the cap only
#: matters for very large buffers and any large constant serves, because
#: the kernel grid-strides (`fill.cu:40-43`).
comptime FILL_MAX_BLOCKS = 65535


def make_sequence_kernel(
    offset: UInt32,
    buffer: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`MakeSequenceImpl` (`fill.cu:37-44`). `WriteThrough` is a plain store
    here; no non-temporal store ships in Mojo 1.0 (the same deviation
    `split_points.mojo` records)."""
    var size = Int(size_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    while i < size:
        buffer.unsafe_store(i, offset + UInt32(i))
        i += Int(grid_dim.x) * Int(block_dim.x)


def launch_make_sequence(
    ctx: DeviceContext,
    offset: UInt32,
    mut buffer: DeviceBuffer[DType.uint32],
    size: Int,
) raises:
    """`MakeSequence` (`fill.cu:47-55`)."""
    if size <= 0:
        return
    var blocks = (size + FILL_BLOCK_SIZE - 1) // FILL_BLOCK_SIZE
    if blocks > FILL_MAX_BLOCKS:
        blocks = FILL_MAX_BLOCKS
    ctx.enqueue_function[make_sequence_kernel](
        offset,
        buffer.unsafe_ptr(),
        Int32(size),
        grid_dim=(blocks, 1, 1),
        block_dim=(FILL_BLOCK_SIZE, 1, 1),
    )
