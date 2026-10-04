# SPDX-License-Identifier: Apache-2.0
"""Existing scaler kernel used unchanged as an adapter wiring example."""
from max.gpu.host import DeviceBuffer
from experiments.apple_callpath.context_owner import CallpathContext
from preprocessing.minmax import minmax_transform_kernel
from experiments.apple_callpath.resident_slot import ResidentCallSlot


def enqueue_minmax_transform(ctx: CallpathContext, mut slot: ResidentCallSlot,
    mut scale: DeviceBuffer[DType.float32], mut offset: DeviceBuffer[DType.float32],
    columns: Int, inverse: Int32, clip: Int32, lower: Float32, upper: Float32,
) raises:
    """begin -> existing kernel -> seal; caller chooses wait or wait_pair.

    Model scale/offset are pre-uploaded, immutable, and caller-owned. This
    adapter demonstrates the plumbing; it is not a MaxAbs implementation.
    """
    if columns <= 0 or len(scale) < columns or len(offset) < columns:
        raise Error("invalid scaler model shape")
    if slot.input_count != slot.output_count or slot.input_count % columns != 0:
        raise Error("invalid scaler input shape")
    if slot.input_count > 2147483647:
        raise Error("scaler exceeds Int32 kernel indexing")
    slot.begin(ctx)
    if slot.input_count > 0:
        ctx.device.enqueue_function[minmax_transform_kernel](
            slot.device_input.unsafe_ptr(), scale.unsafe_ptr(), offset.unsafe_ptr(),
            slot.device_output.unsafe_ptr(), Int32(slot.input_count), Int32(columns),
            inverse, clip, lower, upper,
            grid_dim=(slot.input_count + 255) // 256, block_dim=256,
        )
    slot.seal(ctx)
