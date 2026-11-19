# SPDX-License-Identifier: Apache-2.0
"""The device's multiprocessor count, from a source every vendor answers.

`DeviceAttribute.MULTIPROCESSOR_COUNT` is a CUDA / HIP attribute
(`cudaDevAttrMultiProcessorCount`, `hipDeviceAttributeMultiprocessorCount`).
The Metal backend refuses it ("Attribute "MULTIPROCESSOR_COUNT" not supported
for Apple GPU"), so a call with no vendor guard raised out of every Apple fit
that sized a grid from it (a downstream user's GitHub macOS CI, 2026-11-19,
fx-user-crash-scorer). NVIDIA and AMD still read the device, exactly as
before (same call, same cost); an Apple build reads the Apple row of the
hardware table (`checks/hardware_matrix.gpu_cores_for`, the core count every
Apple launch in this repo is already sized from).

The count is SCHEDULING on every reader: where it would decide a float fold
(the GBDT partition-stats chunk count) `checks/kernel_matrix.
partition_chunks_sm_for` pins it under `identical`, so the L40S (142 SMs), the
MI300X (304 CUs) and an Apple part all fold the same partition. Bits do not
depend on which source answered.
"""
from max.gpu.host import DeviceContext
from max.gpu.host.device_attribute import DeviceAttribute
from core.accelerator_target import TARGET_HAS_APPLE_GPU
from checks.hardware_matrix import gpu_cores_for
from checks.kernel_matrix import TARGET_COLUMN


def device_multiprocessor_count(ctx: DeviceContext) raises -> Int:
    """SMs (NVIDIA) or CUs (AMD) of `ctx`, read from the device; on an Apple
    build the Apple column's GPU core count (Metal has no such query)."""
    comptime if TARGET_HAS_APPLE_GPU:
        return gpu_cores_for[TARGET_COLUMN]()
    return ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
