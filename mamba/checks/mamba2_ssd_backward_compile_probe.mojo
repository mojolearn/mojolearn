# SPDX-License-Identifier: Apache-2.0
"""Compile-only probe for the first Mamba-2 SSD backward kernel."""

from max.gpu.host import DeviceBuffer, DeviceContext

from mamba.impl.mamba_ssm.ops.mamba2_ssd_backward import (
    Mamba2SSDBackwardState,
    mamba2_reverse_chunk_state_into,
)


def _force_elaborate(
    ctx: DeviceContext,
    mut out: Mamba2SSDBackwardState,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut c: DeviceBuffer[DType.float32],
    mut d: DeviceBuffer[DType.float32],
) raises:
    mamba2_reverse_chunk_state_into(ctx, out, a, b, c, d, 2, 1, 3, 256)


def main():
    print("mamba2 SSD backward reverse-chunk recurrence compile probe")
