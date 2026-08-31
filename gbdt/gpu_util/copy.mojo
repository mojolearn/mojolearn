# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# This lane MIRRORS CatBoost (pinned commit 54a8143a). Per-file provenance is in gbdt/DERIVATION_MAP.tsv, in this file's own docstring, and in NOTICE; files with no CatBoost counterpart say so.
"""Device-to-device copy.

MIRRORS the copy operations in `catboost/cuda/cuda_util/`, which are CUB or
`cudaMemcpyAsync` calls.

# =========================================================================
# DEVIATION BLOCK: `enqueue_copy` in Mojo is host-to-device or
# device-to-host. There is no device-to-device form, so this is a kernel.
#
# It exists because of a real bug: the level driver did
#
#     ctx.enqueue_copy(dst_buf=row_index, src_ptr=new_index.unsafe_ptr())
#
# passing a DEVICE pointer where a host source pointer is expected. It does
# not fail. The gathered row index and the gathered stat columns were simply
# never written back, so every level's histogram read the same data and the
# score came out IDENTICAL at every depth (9237.699, feature 0, five levels
# running). The tree still conserved every row and still produced 2^depth
# partitions, which is why nothing else caught it.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx


comptime COPY_BLOCK = 256


def copy_u32_kernel(
    dst: MutPointer[UInt32, MutAnyOrigin],
    src: MutPointer[UInt32, MutAnyOrigin],
    n_in: Int32,
):
    """`dst[i] = src[i]`, grid-stride."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        dst.unsafe_store(i, src.unsafe_load(i))
        i += stride


def copy_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`dst[i] = src[i]`, grid-stride."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    while i < n:
        dst.unsafe_store(i, src.unsafe_load(i))
        i += stride
