# SPDX-License-Identifier: Apache-2.0
"""T31–T39 experiment controls and the forest fold finish.

All new switches are default OFF. NOT COMPILED — NOT TESTED — IDENTITY NOT
VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED. No device-specific arithmetic.
"""
from std.sys.compile import is_defined
from checks.numerics import ftz, identical_div, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
# TOMBSTONE: MOJOLEARN_TREES_T31_PACKED_A (dead code) deleted 2026-10-07 by bad9c74ce; code recoverable at bad9c74ce^.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T31_PACKED_A.patch; record in docs/TOMBSTONES.md.

# T31_PACKED_A/B removed (trees-cleanup 2026-10-07): packed nodes are already
# the default layout (core/forest_inference.mojo FOREST_PACKED_NODES), so A was
# a no-op and B duplicated MOJOLEARN_FOREST_SEPARATE_NODES.
# TOMBSTONE: MOJOLEARN_TREES_T32_SHARED_ROWS (IDENTICAL opt-in shared-row staging, core/forest_inference.mojo; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at a5502e594d.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T32_SHARED_ROWS.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_TREES_T33_COST_SCHEDULE (IDENTICAL opt-in cache-cost grove schedule, core/forest_inference.mojo; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at 1cad80e23b.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T33_COST_SCHEDULE.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_TREES_T34_CHUNK_FOLD (IDENTICAL opt-in chunk fold version: FOREST_CHUNK, chunk_add, forest_chunk_sum; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at cfad8788a0.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T34_CHUNK_FOLD.patch; record in docs/TOMBSTONES.md.
#: Kept as the folded constant: core/forest_inference_model.mojo imports it.
comptime T34_CHUNK_FOLD = False
# TOMBSTONE: MOJOLEARN_TREES_FOREST_ROUTE (forest prediction routes 1/2: leaf IDs once, then every output folds from them, old T35;
# and route 2 with the finiteness check fused into the leaf-ID pass, old T36; no A/B recorded) deleted 2026-11-15 by
# lane/expdel2-03-base; code recoverable at c44dc3c017.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_FOREST_ROUTE.patch; record in docs/TOMBSTONES.md.
#: Kept as the folded constants: core/forest_inference_model.mojo imports them.
comptime T35_LEAF_REUSE = False
comptime T36_FINITE_STAGE = False
comptime T37_WORKSPACE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T37_WORKSPACE"]()
comptime T38_FUSED_LABELS = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T38_FUSED_LABELS"]()

#: The default fold's finish: the tree-order total divided by the tree count.
@always_inline
def forest_chunk_finish(total: Float32, trees: Int) -> Float32:
    return ftz(identical_div(ftz(total), Float32(trees)))

# TOMBSTONE: MOJOLEARN_TREES_T44_METADATA_CACHE (IDENTICAL opt-in metadata cache; this alias had no importer, providers/python/xtrees_api_pyapi_door.mojo reads the define in its own lane; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at 3e9cbbd919.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T44_METADATA_CACHE.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_TREES_T39_AUXILIARY_WALK (IDENTICAL opt-in shared auxiliary walk, core/forest_auxiliary_units.mojo; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at 96ee21ad13.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T39_AUXILIARY_WALK.patch; record in docs/TOMBSTONES.md.

comptime C50_GB_PACKED = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_C50_GB_PACKED"]()
