# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple: GaussianProcessClassifier's latent variance on the device
(lane neighbors-apple3, 2026-09-28): gaussian_process/host/gpc_steps.mojo's
`gpc_scale_rows` and `gpc_latent_var` as kernels. Its own module, imported
only where a build selects it (`-D MOJOLEARN_GPC_DEVICE_VAR`)."""
from std.gpu import block_idx, thread_idx

from checks.numerics import ftz, identical_mul, identical_mul_add

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
