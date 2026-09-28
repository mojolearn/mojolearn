# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple: RBFSampler.transform as one kernel per cell (lane
neighbors-apple3, 2026-09-28). Its own module, imported only where a build
selects it (kernel_methods/estimator.mojo, `-D MOJOLEARN_RBF_FUSED`)."""
from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import ftz, identical_cos, identical_mul, identical_mul_add

comptime RBF_FUSED_MAX_D = 64
comptime RBF_FUSED_TPB = 256


def rbf_fused_transform_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    b_in: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_features_in: Int32,
    n_components_in: Int32,
    scale: Float32,
):
    """`sqrt(2/D) * cos(X @ W + b)`, one thread per cell: the projection as
    one chain over the features ascending, then
    `feature_map_epilogue_kernel`'s statements."""
    var d = Int(n_features_in)
    var dd = Int(n_components_in)
    var total = Int(n_rows_in) * dd
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= total:
        return
    var i = t // dd
    var j = t - i * dd
    var acc = Float32(0.0)
    for f in range(d):
        acc = ftz(identical_mul_add(ftz(x.unsafe_load(i * d + f)), ftz(w.unsafe_load(f * dd + j)), acc))
    var shifted = ftz(acc + ftz(b_in.unsafe_load(j)))
    dst.unsafe_store(t, ftz(identical_mul(identical_cos(shifted), scale)))
