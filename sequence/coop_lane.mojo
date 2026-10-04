# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The source lane of a broadcast inside one COOP cell (lane/ml-k3-amd-fix,
review item K3). `coop_kernel` (sequence/exec_device.mojo) packs one cell per
COOP_CELL_W = 32 consecutive threads. On a 32-lane simdgroup or warp (Apple,
NVIDIA, RDNA) a cell is the whole warp, so lane j of the cell is physical
lane j. On a 64-lane CDNA wave two cells share a wave and the second cell's
lane j is physical lane 32 + j: a shuffle by the bare index j would read the
FIRST cell's values. `coop_src` adds the cell's lane base. Where the warp is
32 wide it compiles to the bare index (same code, same bits as before); on
a 64-lane wave each cell reads only its own lanes, so every cell runs the
values its own lanes loaded, the same chain as on the other columns. Only
the communication scope changes, never the arithmetic or its order."""
from std.gpu import WARP_SIZE
from std.gpu.primitives.id import lane_id

#: threads per COOP cell (the Apple simdgroup width)
comptime COOP_CELL_W = 32


@always_inline
def coop_src(lane: Int, j: Int) -> UInt32:
    """Physical source lane of the cell's lane j for the thread that is the
    cell's lane `lane`. A cell never straddles a warp: the launch block is a
    multiple of WARP_SIZE and WARP_SIZE a multiple of COOP_CELL_W."""
    comptime assert WARP_SIZE % COOP_CELL_W == 0, "a COOP cell must tile the warp"
    comptime if WARP_SIZE == COOP_CELL_W:
        return UInt32(j)
    else:
        return UInt32(Int(lane_id()) - lane + j)
