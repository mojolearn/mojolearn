# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple: GaussianProcessClassifier's latent variance on the device
(lane neighbors-apple3, 2026-09-28): gaussian_process/host/gpc_steps.mojo's
`gpc_scale_rows` and `gpc_latent_var` as kernels. Its own module, imported
only where a build selects it (`-D MOJOLEARN_GPC_DEVICE_VAR`)."""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_mul, identical_mul_add
from gaussian_process.gp_var_seg import (
    GP_IDN_VAR_SEG,
    GP_VAR_PTS,
    GP_VAR_SEGS,
    GP_VAR_SMEM_FLOATS,
    GP_VAR_SMEM_LIMIT_BYTES,
    gp_var_seg_len,
    gp_var_tree8,
)

comptime GPC_VAR_TPB = 256


def gpc_scale_rows_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    kcross: MutPointer[Float32, MutAnyOrigin],
    wsr: MutPointer[Float32, MutAnyOrigin],
    n_train_in: Int32,
    n_star_in: Int32,
):
    """`gpc_scale_rows`, one thread per cell."""
    var n_train = Int(n_train_in)
    var n_star = Int(n_star_in)
    var e = Int(block_idx.x) * GPC_VAR_TPB + Int(thread_idx.x)
    if e >= n_train * n_star:
        return
    var i = e // n_star
    var sc = ftz(wsr[unsafe_offset = i])
    dst[unsafe_offset = e] = ftz(identical_mul(sc, ftz(kcross[unsafe_offset = e])))


def gpc_latent_var_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_train_in: Int32,
    n_star_in: Int32,
    kss: Float32,
):
    """`gpc_latent_var`, one thread per column: the fold over i ascending."""
    var n_train = Int(n_train_in)
    var n_star = Int(n_star_in)
    var t = Int(block_idx.x) * GPC_VAR_TPB + Int(thread_idx.x)
    if t >= n_star:
        return
    var acc = Float32(0.0)
    for i in range(n_train):
        var vv = ftz(v[unsafe_offset = i * n_star + t])
        acc = ftz(identical_mul_add(vv, vv, acc))
    dst[unsafe_offset = t] = ftz(ftz(kss) - acc)


def gpc_latent_var_seg_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    v: MutPointer[Float32, MutAnyOrigin],
    n_train_in: Int32,
    n_star_in: Int32,
    kss: Float32,
):
    """GP_IDN_VAR_SEG (fix-kg1-kernel, audit B13): `gpc_latent_var_kernel`
    with the fold over i cut into GP_VAR_SEGS segments (one per
    thread_idx.y) and the fixed tree of `gp_var_seg.mojo`; block
    (GP_VAR_PTS, GP_VAR_SEGS). Not clamped, as the serial kernel. Every
    thread reaches the barrier."""
    comptime assert GP_VAR_SMEM_FLOATS * 4 <= GP_VAR_SMEM_LIMIT_BYTES, "gpc var smem fits"
    var n_train = Int(n_train_in)
    var n_star = Int(n_star_in)
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var t = Int(block_idx.x) * GP_VAR_PTS + tx
    var part = stack_allocation[
        GP_VAR_SMEM_FLOATS, Float32, address_space=AddressSpace.SHARED
    ]()
    var seg_len = gp_var_seg_len(n_train)
    var lo = ty * seg_len
    var hi = min(lo + seg_len, n_train)
    var acc = Float32(0.0)
    if t < n_star:
        for i in range(lo, hi):
            var vv = ftz(v[unsafe_offset = i * n_star + t])
            acc = ftz(identical_mul_add(vv, vv, acc))
    part[unsafe_offset = ty * GP_VAR_PTS + tx] = acc
    barrier()
    if ty != 0 or t >= n_star:
        return
    var total = gp_var_tree8(
        part[unsafe_offset = 0 * GP_VAR_PTS + tx], part[unsafe_offset = 1 * GP_VAR_PTS + tx],
        part[unsafe_offset = 2 * GP_VAR_PTS + tx], part[unsafe_offset = 3 * GP_VAR_PTS + tx],
        part[unsafe_offset = 4 * GP_VAR_PTS + tx], part[unsafe_offset = 5 * GP_VAR_PTS + tx],
        part[unsafe_offset = 6 * GP_VAR_PTS + tx], part[unsafe_offset = 7 * GP_VAR_PTS + tx],
    )
    dst[unsafe_offset = t] = ftz(ftz(kss) - total)


def gpc_latent_var_launch(
    ctx: DeviceContext,
    mut dvar: DeviceBuffer[DType.float32],
    mut dv: DeviceBuffer[DType.float32],
    n_train: Int,
    n_star: Int,
    kss: Float32,
) raises:
    """The latent variance launch every caller uses: the segmented kernel
    under GP_IDN_VAR_SEG (IDENTICAL), the serial one otherwise (FAST and
    the `_OFF` builds keep their bits)."""
    comptime if GP_IDN_VAR_SEG:
        ctx.enqueue_function[gpc_latent_var_seg_kernel](
            dvar.unsafe_ptr(), dv.unsafe_ptr(), Int32(n_train), Int32(n_star), kss,
            grid_dim=((n_star + GP_VAR_PTS - 1) // GP_VAR_PTS, 1, 1),
            block_dim=(GP_VAR_PTS, GP_VAR_SEGS, 1),
        )
    else:
        ctx.enqueue_function[gpc_latent_var_kernel](
            dvar.unsafe_ptr(), dv.unsafe_ptr(), Int32(n_train), Int32(n_star), kss,
            grid_dim=((n_star + GPC_VAR_TPB - 1) // GPC_VAR_TPB, 1, 1),
            block_dim=(GPC_VAR_TPB, 1, 1),
        )
