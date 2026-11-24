# SPDX-License-Identifier: Apache-2.0
"""The RBF affinity exp(-gamma ||x_i - x_j||^2) on the metrics image's device
(lane vdf-misc3, 2026-11-24).

Before this file the spectral embedding's rbf affinity was built by the x_decomp
image (`core/spectral_affinity_plan.mojo`: XD_DEV_SQDIST, then XD_DEV_EW op
OP_SCALE with s = -gamma, then op OP_EXP), downloaded to host storage and
uploaded again by this image's dense spectral route. Here one thread per cell
runs the SAME three cell bodies in the same order (`sqdist_cell`, then
`ew_cell(OP_SCALE)`, then `ew_cell(OP_EXP)`, each on a Float32 value and each
flushing its inputs exactly as the stored-and-reloaded intermediate was), so the
cell's bits are the three-launch chain's and the x_decomp host column's
(`XD_SQDIST` / `XD_EW`, the same cells). The n x n result stays on the device.
"""
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from x_decomp.cells import F32Ptr, OP_EXP, OP_SCALE, ew_cell, sqdist_cell
from core.metrics_store import metrics_store_buffer
from core.fold_x_word import fold_x_slot

comptime RBF_TPB = 128


def rbf_affinity_kernel(x: F32Ptr, dst: F32Ptr, n: Int32, d: Int32, scale: Float32):
    """dst[i * n + j] = exp_c(-gamma * ||x_i - x_j||^2), one thread per cell;
    `scale` is Float32(-gamma), the word the x_decomp EW cell received."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nn = Int(n) * Int(n)
    if t < nn:
        var sq = sqdist_cell(x, x, t // Int(n), t % Int(n), Int(d))
        var scaled = ew_cell(OP_SCALE, sq, Float32(0), Float32(0), scale)
        dst.unsafe_store(t, ew_cell(OP_EXP, scaled, Float32(0), Float32(0), Float32(0)))


def rbf_affinity_device(
    ctx: DeviceContext, x_addr: Int, n: Int, d: Int, gamma: Float64,
    mut dst: DeviceBuffer[DType.float32],
) raises:
    """X (n x d float32 at the host address `x_addr`) uploaded once, the n x n
    affinity written into `dst` (at least max(n * n, 1) words) on the device.
    Enqueued: the caller synchronizes before reading `dst` on the host.

    A NEGATIVE `x_addr` (lane vdf-scope-cluster8, core/fold_x_word.mojo) names
    a slot of the metrics image's resident store holding X (a model-selection
    fold taken on the device, the same context): the kernel reads that slot in
    place, no upload; the same cells on the same words."""
    if n * n > 2147483647:
        raise Error("spectral embedding: the rbf affinity exceeds the Int32 index bound")
    var dx: DeviceBuffer[DType.float32]
    if x_addr < 0:
        dx = metrics_store_buffer(fold_x_slot(x_addr), max(n * d, 1))
    else:
        dx = ctx.enqueue_create_buffer[DType.float32](max(n * d, 1))
        if n * d > 0:
            ctx.enqueue_copy(dst_buf=dx, src_ptr=F32Ptr(unsafe_from_address=x_addr))
    var nn = n * n
    ctx.enqueue_function[rbf_affinity_kernel](
        dx.unsafe_ptr(), dst.unsafe_ptr(), Int32(n), Int32(d), Float32(-gamma),
        grid_dim=(max((nn + RBF_TPB - 1) // RBF_TPB, 1), 1, 1), block_dim=(RBF_TPB, 1, 1),
    )
    # The upload is read by the kernel: keep `dx` alive until the stream has run it.
    ctx.synchronize()
    _ = dx^
