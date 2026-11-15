# SPDX-License-Identifier: Apache-2.0
"""The forest fold finish, and the folded constants of the deleted T31–T44 /
C50 experiment controls that other modules still import (each card off).
"""
from checks.numerics import ftz, identical_div
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
# TOMBSTONE: MOJOLEARN_TREES_T37_WORKSPACE (IDENTICAL opt-in resident prediction workspace, core/forest_inference_model.mojo; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at 07f7a0733b.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T37_WORKSPACE.patch; record in docs/TOMBSTONES.md.
#: Kept as the folded constant: core/forest_inference_model.mojo imports it.
comptime T37_WORKSPACE = False
# TOMBSTONE: MOJOLEARN_TREES_T38_FUSED_LABELS (IDENTICAL opt-in fused prediction labels; this lane also dropped core/forest_inference.mojo's unused import of it; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at 81c34fb4fb.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T38_FUSED_LABELS.patch; record in docs/TOMBSTONES.md.
#: Kept as the folded constant: core/forest_inference_model.mojo, core/forest_inference_run.mojo and bindings/forest_inference_binding_door.mojo import it.
comptime T38_FUSED_LABELS = False

#: The default fold's finish: the tree-order total divided by the tree count.
@always_inline
def forest_chunk_finish(total: Float32, trees: Int) -> Float32:
    return ftz(identical_div(ftz(total), Float32(trees)))

# TOMBSTONE: MOJOLEARN_TREES_T44_METADATA_CACHE (IDENTICAL opt-in metadata cache; this alias had no importer, providers/python/xtrees_api_pyapi_door.mojo reads the define in its own lane; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at 3e9cbbd919.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T44_METADATA_CACHE.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_TREES_T39_AUXILIARY_WALK (IDENTICAL opt-in shared auxiliary walk, core/forest_auxiliary_units.mojo; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at 96ee21ad13.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_T39_AUXILIARY_WALK.patch; record in docs/TOMBSTONES.md.

# TOMBSTONE: MOJOLEARN_TREES_C50_GB_PACKED (IDENTICAL opt-in packed GBDT predict nodes; no A/B recorded) deleted 2026-11-15 by lane/expdel2-03-base; code recoverable at 03a43eb1ec.
# Restore: git apply experiments/removed/MOJOLEARN_TREES_C50_GB_PACKED.patch; record in docs/TOMBSTONES.md.
#: Kept as the folded constant: core/gbdt_host_predict.mojo, gbdt/resident_model.mojo and gbdt/models/add_non_symmetric_tree_doc_parallel.mojo import it.
comptime C50_GB_PACKED = False
