# SPDX-License-Identifier: Apache-2.0
"""Experimental byte-LM block copies; production dispatch is unchanged.

One launch copies nine disjoint tensor ranges. The y grid selects a tensor;
no floating point arithmetic, barriers or atomics are involved. Each tensor
buffer must remain alive until the caller synchronizes the context.
"""
from max.gpu import block_idx, block_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext


def _block_copy_kernel[pack: Bool](
    flat: MutPointer[Float32, MutAnyOrigin],
    p0: MutPointer[Float32, MutAnyOrigin],
    p1: MutPointer[Float32, MutAnyOrigin],
    p2: MutPointer[Float32, MutAnyOrigin],
    p3: MutPointer[Float32, MutAnyOrigin],
    p4: MutPointer[Float32, MutAnyOrigin],
    p5: MutPointer[Float32, MutAnyOrigin],
    p6: MutPointer[Float32, MutAnyOrigin],
    p7: MutPointer[Float32, MutAnyOrigin],
    p8: MutPointer[Float32, MutAnyOrigin],
    o0: Int32,
    o1: Int32,
    o2: Int32,
    o3: Int32,
    o4: Int32,
    o5: Int32,
    o6: Int32,
    o7: Int32,
    o8: Int32,
    o9: Int32,
):
    var tensor = Int(block_idx.y)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if tensor == 0:
        if i < Int(o1 - o0):
            comptime if pack:
                flat.unsafe_store(Int(o0) + i, p0.unsafe_load(i))
            else:
                p0.unsafe_store(i, flat.unsafe_load(Int(o0) + i))
    elif tensor == 1:
        if i < Int(o2 - o1):
            comptime if pack:
                flat.unsafe_store(Int(o1) + i, p1.unsafe_load(i))
            else:
                p1.unsafe_store(i, flat.unsafe_load(Int(o1) + i))
    elif tensor == 2:
        if i < Int(o3 - o2):
            comptime if pack:
                flat.unsafe_store(Int(o2) + i, p2.unsafe_load(i))
            else:
                p2.unsafe_store(i, flat.unsafe_load(Int(o2) + i))
    elif tensor == 3:
        if i < Int(o4 - o3):
            comptime if pack:
                flat.unsafe_store(Int(o3) + i, p3.unsafe_load(i))
            else:
                p3.unsafe_store(i, flat.unsafe_load(Int(o3) + i))
    elif tensor == 4:
        if i < Int(o5 - o4):
            comptime if pack:
                flat.unsafe_store(Int(o4) + i, p4.unsafe_load(i))
            else:
                p4.unsafe_store(i, flat.unsafe_load(Int(o4) + i))
    elif tensor == 5:
        if i < Int(o6 - o5):
            comptime if pack:
                flat.unsafe_store(Int(o5) + i, p5.unsafe_load(i))
            else:
                p5.unsafe_store(i, flat.unsafe_load(Int(o5) + i))
    elif tensor == 6:
        if i < Int(o7 - o6):
            comptime if pack:
                flat.unsafe_store(Int(o6) + i, p6.unsafe_load(i))
            else:
                p6.unsafe_store(i, flat.unsafe_load(Int(o6) + i))
    elif tensor == 7:
        if i < Int(o8 - o7):
            comptime if pack:
                flat.unsafe_store(Int(o7) + i, p7.unsafe_load(i))
            else:
                p7.unsafe_store(i, flat.unsafe_load(Int(o7) + i))
    elif tensor == 8:
        if i < Int(o9 - o8):
            comptime if pack:
                flat.unsafe_store(Int(o8) + i, p8.unsafe_load(i))
            else:
                p8.unsafe_store(i, flat.unsafe_load(Int(o8) + i))


def byte_block_copy[pack: Bool](
    ctx: DeviceContext,
    mut flat: DeviceBuffer[DType.float32],
    mut b0: DeviceBuffer[DType.float32],
    mut b1: DeviceBuffer[DType.float32],
    mut b2: DeviceBuffer[DType.float32],
    mut b3: DeviceBuffer[DType.float32],
    mut b4: DeviceBuffer[DType.float32],
    mut b5: DeviceBuffer[DType.float32],
    mut b6: DeviceBuffer[DType.float32],
    mut b7: DeviceBuffer[DType.float32],
    mut b8: DeviceBuffer[DType.float32],
    offsets: List[Int],
) raises:
    """Pack or unpack nine tensors at absolute flat-buffer offsets.

    Reject invalid ranges before enqueueing. Buffers may be larger than their
    ranges (the check uses tail canaries); ranges must not overlap. Caller
    guarantees the ten device allocations do not alias one another.
    """
    if len(offsets) != 10:
        raise Error("byte block copy: expected ten offsets")
    var widest = 0
    var sizes: List[Int] = [len(b0),len(b1),len(b2),len(b3),len(b4),len(b5),len(b6),len(b7),len(b8)]
    for i in range(10):
        if offsets[i] < 0 or offsets[i] > len(flat) or offsets[i] > 2147483647:
            raise Error("byte block copy: offset out of range")
    for i in range(9):
        var count = offsets[i + 1] - offsets[i]
        if count < 0 or count > sizes[i]:
            raise Error("byte block copy: invalid tensor range")
        widest = max(widest, count)
    if widest == 0:
        return
    ctx.enqueue_function[_block_copy_kernel[pack]](
        flat.unsafe_ptr(),
        b0.unsafe_ptr(),
        b1.unsafe_ptr(),
        b2.unsafe_ptr(),
        b3.unsafe_ptr(),
        b4.unsafe_ptr(),
        b5.unsafe_ptr(),
        b6.unsafe_ptr(),
        b7.unsafe_ptr(),
        b8.unsafe_ptr(),
        Int32(offsets[0]),
        Int32(offsets[1]),
        Int32(offsets[2]),
        Int32(offsets[3]),
        Int32(offsets[4]),
        Int32(offsets[5]),
        Int32(offsets[6]),
        Int32(offsets[7]),
        Int32(offsets[8]),
        Int32(offsets[9]),
        grid_dim=((widest + 255) // 256, 9, 1),
        block_dim=(256, 1, 1),
    )
