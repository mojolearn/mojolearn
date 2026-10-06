# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST on Apple: RBFSampler.transform as one kernel per cell (lane
neighbors-apple3, 2026-09-28). Its own module, imported only where a build
selects it (kernel_methods/estimator.mojo, `-D MOJOLEARN_RBF_FUSED`)."""
from std.gpu import block_dim, block_idx, thread_idx

from gemm.contract import contract_leaf_size, leaf_count, leaf_begin, leaf_end
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


def rbf_fused_project_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    w: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_features_in: Int32,
    n_components_in: Int32,
):
    """`X @ W` alone, one thread per cell: `rbf_fused_transform_kernel`'s
    chain over the features ascending, stored before the offset. Followed
    by `feature_map_epilogue_kernel` it writes the fused kernel's words
    (the epilogue's `ftz(load)` of a flushed value is that value). The
    IDENTICAL route when a trace or a sabotage arm needs the projection as
    its own stage (`kernel_methods/estimator.mojo::RBF_IDN_FUSED`)."""
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
    dst.unsafe_store(t, acc)


def classical_projection_kernel[FUSED: Bool](dst: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin], w: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32, d_in: Int32, q_in: Int32, scale: Float32):
    """C25: four independent row scores reuse each immutable projection weight.
    GEMM-v1 leaves/tree and the incumbent offset/cos/scale seams are retained.
    NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    """
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var n = Int(n_in)
    var d = Int(d_in)
    var q = Int(q_in)
    var first = (t // q) * 4
    var col = t % q
    if first >= n:
        return
    var leaf = contract_leaf_size(d)
    var levels = InlineArray[Float32, 44](fill=Float32(0))
    var occupied = UInt32(0)
    for part in range(leaf_count(d, leaf)):
        var acc = InlineArray[Float32, 4](fill=Float32(0))
        for f in range(leaf_begin(part, leaf), leaf_end(part, leaf, d)):
            var weight = ftz(w.unsafe_load(f * q + col))
            comptime for row in range(4):
                if first + row < n:
                    acc[row] = ftz(identical_mul_add(ftz(x.unsafe_load((first + row) * d + f)), weight, acc[row]))
        var level = 0
        while (occupied & (UInt32(1) << UInt32(level))) != 0:
            comptime for row in range(4):
                acc[row] = ftz(levels[level * 4 + row] + acc[row])
            occupied = occupied & ~(UInt32(1) << UInt32(level))
            level += 1
        comptime for row in range(4):
            levels[level * 4 + row] = acc[row]
        occupied = occupied | (UInt32(1) << UInt32(level))
    comptime for row in range(4):
        if first + row < n:
            var acc = Float32(0)
            var have = False
            for level in range(11):
                if (occupied & (UInt32(1) << UInt32(level))) != 0:
                    acc = ftz(levels[level * 4 + row] + acc) if have else levels[level * 4 + row]
                    have = True
            comptime if FUSED:
                acc = ftz(identical_mul(identical_cos(ftz(acc + ftz(b.unsafe_load(col)))), scale))
            dst.unsafe_store((first + row) * q + col, acc)
