# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Ownership/compile gate for the synchronized tensor-search state."""

from max.gpu.host import DeviceContext
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.feature_blocks import blocks_for
from gbdt.methods.greedy_subsets_searcher.greedy_search_helper import (
    TSynchronizedSymmetricLevelState,
)


def main() raises:
    var ctx = DeviceContext()
    var folds: List[Int] = [1, 255]
    var layout = build_layout(folds)
    var blocks = blocks_for(layout, 8)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](16)
    ctx.enqueue_memset(cindex, UInt32(0))
    var state = TSynchronizedSymmetricLevelState(
        ctx, layout^, blocks^, cindex^, 8, 2, 2, False
    )
    state.initialize_tree(ctx, Float32(8.0), Float32(8.0))
    var regenerated = ctx.enqueue_create_buffer[DType.uint32](16)
    ctx.enqueue_memset(regenerated, UInt32(0))
    state.replace_active_cindex(ctx, regenerated^)
    if state.level != 0 or state.live_leaves != 1 or (
        state.max_depth != 2 or len(state.workspace) != 1
    ):
        raise Error("synchronized tensor-search state initialized incorrectly")
    print("tensor synchronized level state PASS")
