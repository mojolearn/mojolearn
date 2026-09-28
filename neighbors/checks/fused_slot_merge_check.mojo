# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DEVIATION 5219's seam driver: the fused k-NN's cross-block merge is
per-block candidate slots folded by `fused_l2_knn_merge_kernel` after the
kernel boundary, in the queue's `(distance, index)` order, and no longer a
device-mutex handoff with plain loads and stores (the M3 lost-candidate
defect, trees DEVIATION 5611).

Two checks, both IDENTICAL:
  * `check_fused_griddimx_merge`: at the computed multi-block grid and a
    forced grid_x = 5 every slot matches a host Float64 oracle, the merged
    grid's (distance, index) cells equal the grid_x = 1 launch's BIT FOR
    BIT, and dropping one candidate in the merge moves the output;
  * `check_knn_fused_tie_set_is_geometry_invariant`: a tied fixture at
    1, 40 and 2,000 queries (the 40-query launch takes the x-split now that
    the grid pin is lifted) returns one tie set.

Arm: `neighbors/checks/sabotage/5219_slot_merge_drops_last_block.patch`
(the merge never reads the last column block's slot), which must FAIL here.
Run: `mojo run -I . neighbors/checks/fused_slot_merge_check.mojo`.
"""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from neighbors.checks.knn_check import check_fused_griddimx_merge
from neighbors.checks.knn_identity_check import (
    check_knn_fused_tie_set_is_geometry_invariant,
)


def main() raises:
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        raise Error("the slot-merge seam check requires IDENTICAL")
    check_fused_griddimx_merge()
    check_knn_fused_tie_set_is_geometry_invariant()
    print("fused_slot_merge_check PASS (DEVIATION 5219)")
