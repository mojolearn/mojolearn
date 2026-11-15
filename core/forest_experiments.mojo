# SPDX-License-Identifier: Apache-2.0
"""T31–T39 experiment controls and the forest fold finish.

All new switches are default OFF. NOT COMPILED — NOT TESTED — IDENTITY NOT
VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. No device-specific arithmetic.
"""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import ftz, identical_div, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
# TOMBSTONE: MOJOLEARN_TREES_T31_PACKED_A (dead code) deleted 2026-10-07 by bad9c74ce; code recoverable at bad9c74ce^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T31_PACKED_A.patch; record in docs/TOMBSTONES.md.

# T31_PACKED_A/B removed (trees-cleanup 2026-10-07): packed nodes are already
# the default layout (core/forest_inference.mojo FOREST_PACKED_NODES), so A was
# a no-op and B duplicated MOJOLEARN_FOREST_SEPARATE_NODES.
# TOMBSTONE: MOJOLEARN_TREES_T32_SHARED_ROWS (IDENTICAL opt-in shared-row staging, core/forest_inference.mojo; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at a5502e594d.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T32_SHARED_ROWS.patch; record in docs/TOMBSTONES.md.
comptime T33_COST_SCHEDULE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T33_COST_SCHEDULE"]()
# TOMBSTONE: MOJOLEARN_TREES_T34_CHUNK_FOLD (IDENTICAL opt-in chunk fold version: FOREST_CHUNK, chunk_add, forest_chunk_sum; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at cfad8788a0.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T34_CHUNK_FOLD.patch; record in docs/TOMBSTONES.md.
#: Kept as the folded constant: core/forest_inference_model.mojo imports it.
comptime T34_CHUNK_FOLD = False
# Forest prediction ROUTE: one arms switch (trees-cleanup 2026-10-07; replaces
# T35_LEAF_REUSE / T36_FINITE_STAGE, whose silent precedence over T34 and the
# grove schedule hid T33). `launch_forest_inference` takes exactly one route:
#   -D MOJOLEARN_TREES_FOREST_ROUTE=0  (absent) walk-and-fold per row: grove
#                                      lanes/rows, or the T34 chunk kernel when
#                                      the T34 fold version is on
#   -D MOJOLEARN_TREES_FOREST_ROUTE=1  leaf IDs once, then every output folds
#                                      from them (old T35)
#   -D MOJOLEARN_TREES_FOREST_ROUTE=2  route 1 with the finiteness check fused
#                                      into the leaf-ID pass (old T36)
# Orthogonal controls: T34_CHUNK_FOLD is the fold VERSION (bits) on every
# route; T33_COST_SCHEDULE picks rows-vs-lanes by cache cost in route 0's grove
# launch and in routes 1/2's leaf-ID launch. The one empty cell is route 0
# with T34 on: the chunk kernel has no grove schedule, so T33 is inert there.
comptime FOREST_ROUTE = get_defined_int["MOJOLEARN_TREES_FOREST_ROUTE", 0]() if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else 0
comptime T35_LEAF_REUSE = FOREST_ROUTE == 1 or FOREST_ROUTE == 2
comptime T36_FINITE_STAGE = FOREST_ROUTE == 2
comptime T37_WORKSPACE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T37_WORKSPACE"]()
comptime T38_FUSED_LABELS = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T38_FUSED_LABELS"]()

#: The default fold's finish: the tree-order total divided by the tree count.
@always_inline
def forest_chunk_finish(total: Float32, trees: Int) -> Float32:
    return ftz(identical_div(ftz(total), Float32(trees)))

comptime T44_METADATA_CACHE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T44_METADATA_CACHE"]()
comptime T39_AUXILIARY_WALK = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T39_AUXILIARY_WALK"]()

comptime C50_GB_PACKED = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_C50_GB_PACKED"]()
