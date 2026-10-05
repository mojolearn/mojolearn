# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CTR estimation order on the device, one thread per position (lane
cpu4-gbdt). The order itself is defined in `gbdt/ctrs/ctr_order.mojo`,
which the host column shares."""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.ctrs.ctr_order import ctr_order_key, ctr_order_row
from gbdt.data.permutation import IDENTITY_PERMUTATION_ID

comptime CTR_ORDER_BLOCK = 256


def ctr_order_kernel(
    dst: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
    key: UInt64,
    identity: Int32,
):
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        dst.unsafe_store(
            i, UInt32(ctr_order_row(i, n, key, identity != Int32(0)))
        )
        i += stride


def launch_ctr_estimation_order(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.uint32],
    n_rows: Int,
    permutation_id: Int,
) raises:
    """Write permutation `permutation_id`'s CTR estimation order into
    `dst[0:n_rows]` (`order[i]` = the original row at position `i`, the
    direction `TCtrBinBuilderGpu` consumes)."""
    if n_rows <= 0:
        return
    var blocks = min(
        (n_rows + CTR_ORDER_BLOCK - 1) // CTR_ORDER_BLOCK, 65535
    )
    ctx.enqueue_function[ctr_order_kernel](
        dst.unsafe_ptr(),
        Int32(n_rows),
        ctr_order_key(permutation_id),
        Int32(1) if permutation_id == IDENTITY_PERMUTATION_ID else Int32(0),
        grid_dim=(blocks, 1, 1),
        block_dim=(CTR_ORDER_BLOCK, 1, 1),
    )
