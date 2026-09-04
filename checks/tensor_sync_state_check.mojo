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
    var state = TSynchronizedSymmetricLevelState(
        ctx, layout^, blocks^, 8, 2, 2, False
    )
    if state.level != 0 or state.live_leaves != 1 or (
        state.max_depth != 2 or len(state.workspace) != 1
    ):
        raise Error("synchronized tensor-search state initialized incorrectly")
    print("tensor synchronized level state PASS")
