# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CTR estimation order on the device, one thread per position (lane
cpu4-gbdt; seeded from `random_state` by lane cat-ctr, 2026-11-21). The
order itself is defined in `gbdt/ctrs/ctr_order.mojo`, which the host
column shares."""

from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.ctrs.ctr_order import learn_load_key, learn_permutation_key, learn_order_row

comptime CTR_ORDER_BLOCK = 256


def ctr_order_kernel(
    dst: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
    key: UInt64,
    load_key: UInt64,
    permuted: Int32,
):
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        dst.unsafe_store(
            i,
            UInt32(learn_order_row(i, n, 1, permuted != Int32(0), key, -1, load_key)),
        )
        i += stride


def launch_ctr_estimation_order(
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.uint32],
    n_rows: Int,
    permutation_id: Int,
    random_seed: UInt64,
) raises:
    """Write permutation `permutation_id`'s CTR estimation order of a fit
    seeded `random_seed` into `dst[0:n_rows]` (`order[i]` = the original
    row at position `i`, the direction `TCtrBinBuilderGpu` consumes)."""
    if n_rows <= 0:
        return
    var blocks = min(
        (n_rows + CTR_ORDER_BLOCK - 1) // CTR_ORDER_BLOCK, 65535
    )
    ctx.enqueue_function[ctr_order_kernel](
        dst.unsafe_ptr(),
        Int32(n_rows),
        learn_permutation_key(random_seed, permutation_id, 1),
        learn_load_key(random_seed),
        Int32(0) if permutation_id == 0 else Int32(1),
        grid_dim=(blocks, 1, 1),
        block_dim=(CTR_ORDER_BLOCK, 1, 1),
    )
