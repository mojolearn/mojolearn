# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device side of `x_cluster/bgmm_device.mojo` (lane
cgr4-device-optim-bgmm): one thread per cell for every step, one block per
component for the Cholesky. GPU binding only (the host column walks the same
cells in `bgmm_host_step`)."""
from std.gpu import block_dim, block_idx, thread_idx
from std.sys import llvm_intrinsic
from std.sys.info import is_apple_gpu
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier

from x_cluster.bodies import FPtr
from x_cluster.bgmm_device import (
    ST_CHOL,
    bgmm_cell,
    bgmm_cells,
    chol_diag_cell,
    chol_inv_cell,
    chol_off_cell,
    chol_regions,
)

comptime BG_TPB = 128


@always_inline
def _bg_barrier():
    """A block barrier that also orders DEVICE memory: on Apple
    `air.wg.barrier(3, 1)` (`barrier()` there orders threadgroup memory
    only; x_decomp/jacobi2.mojo `dev_barrier`), `barrier()` elsewhere."""
    comptime if is_apple_gpu():
        llvm_intrinsic["llvm.air.wg.barrier", NoneType](Int32(3), Int32(1))
    else:
        barrier()


def bgmm_cell_kernel(
    step: Int32, w: FPtr, p1: FPtr, p2: FPtr, p3: FPtr, kc: Int32, d: Int32, cfg: Int32, aux: Int32, n: Int32
):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n):
        bgmm_cell(Int(step), w, p1, p2, p3, Int(kc), Int(d), Int(cfg), Int(aux), t)


def bgmm_chol_kernel(w: FPtr, kc: Int32, d: Int32, aux: Int32):
    """Block k: component k's Cholesky (column j's diagonal by thread 0, then
    the rows below it across the block), then inv(L)^T a column per thread."""
    var k = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var D = Int(d)
    var K = Int(kc)
    var rg = chol_regions(K, D, Int(aux))
    for j in range(D):
        if tid == 0:
            chol_diag_cell(w, K, D, rg[0], k, j)
        _bg_barrier()
        var i = j + 1 + tid
        while i < D:
            chol_off_cell(w, K, D, rg[0], k, j, i)
            i += nt
        _bg_barrier()
    var c = tid
    while c < D:
        chol_inv_cell(w, K, D, rg[1], k, c)
        c += nt


def bgmm_launch(
    ctx: DeviceContext, step: Int, w: FPtr, p1: FPtr, p2: FPtr, p3: FPtr, kc: Int, d: Int, cfg: Int, aux: Int
) raises:
    """Enqueue one step (no synchronize)."""
    if step == ST_CHOL:
        ctx.enqueue_function[bgmm_chol_kernel](w, Int32(kc), Int32(d), Int32(aux), grid_dim=kc, block_dim=BG_TPB)
        return
    var n = bgmm_cells(step, kc, d)
    ctx.enqueue_function[bgmm_cell_kernel](
        Int32(step), w, p1, p2, p3, Int32(kc), Int32(d), Int32(cfg), Int32(aux), Int32(n),
        grid_dim=(n + BG_TPB - 1) // BG_TPB if n > 0 else 1, block_dim=BG_TPB,
    )
