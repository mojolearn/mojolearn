"""lane/apple-fast-sym-iter (2026-10-03): the per-iteration FIXED cost of the
pointwise SymmetricTree arm (`doc_parallel_boosting.mojo`, Plain boosting,
one permutation) on the Apple FAST tier. Every define of the lane is deleted
(docs/TOMBSTONES.md): SYM_REUSE_PARTITION, SYM_DERIV_FUSED and
SYM_LEAF_FROM_STATS (2026-10-09, DROPPED-noise), SYM_ITER_ALL and
SYM_BUF_ARENA (2026-11-15, NEUTRAL). What is left is the objective predicate
the default estimator and the device rounds read.
"""

from gbdt.targets.pointwise_bodies import BODY_OBJ_RMSE, BODY_OBJ_HUBER

# TOMBSTONE: MOJOLEARN_SYM_ITER_ALL (NEUTRAL: rab4-symiter 1000-taxi +0.2%, quality identical) umbrella of
# SYM_BUF_ARENA deleted 2026-11-15 by lane/expdel-05-trees; code recoverable at a768195223.
# Restore: git apply experiments/removed/MOJOLEARN_SYM_ITER_ALL.patch
#: TOMBSTONE: MOJOLEARN_SYM_LEAF_FROM_STATS (DROPPED-noise: sym-iter-leaf istella/taxi -0.2%, old base) deleted 2026-10-09 on
#: lane/owed-deletions-D1 (one Newton step from the searcher's partition stats, on the device); code recoverable at b639a2bd2.
#: Restore: git apply experiments/removed/MOJOLEARN_SYM_LEAF_FROM_STATS.patch; record in docs/TOMBSTONES.md.
#: TOMBSTONE: MOJOLEARN_SYM_REUSE_PARTITION (DROPPED-noise: sym-iter-reuse taxi +0.3%, old base; F12 2026-10-06 B/A 0.94..1.03 mixed)
#: deleted 2026-10-09 on lane/owed-deletions-D1 (the searcher's final partition as the estimator's); code recoverable at
#: b639a2bd2. Restore: git apply experiments/removed/MOJOLEARN_SYM_REUSE_PARTITION.patch; record in docs/TOMBSTONES.md.
#: TOMBSTONE: MOJOLEARN_SYM_DERIV_FUSED (DROPPED-noise: sym-iter-fused istella/taxi -0.1%, old base) deleted 2026-10-09 on
#: lane/owed-deletions-D1 (the next tree's gradient pass behind this tree's tail drain); code recoverable at b639a2bd2.
#: Restore: git apply experiments/removed/MOJOLEARN_SYM_DERIV_FUSED.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_SYM_BUF_ARENA (NEUTRAL: rab7-symbufarena 1000-taxi -0.2%; F12 M3 B/A 0.93..1.01 mixed)
# (`SymIterPool`, `sym_pool_get`, `sym_split_planes`, `sym_scale_from_mags`, `sym_compute_bins_pooled`)
# deleted 2026-11-15 by lane/expdel-05-trees; code recoverable at 0f457a0c03.
# Restore: git apply experiments/removed/MOJOLEARN_SYM_BUF_ARENA.patch


def sym_objective_is_pointwise(objective: Int) -> Bool:
    """The objectives whose gradient pass is `launch_approximate` (the
    single-dimensional pointwise oracles)."""
    # The native row-target table assigns consecutive tags RMSE through
    # Huber. Querywise and multidimensional losses have other kernels.
    return BODY_OBJ_RMSE <= objective and objective <= BODY_OBJ_HUBER


# TOMBSTONE: MOJOLEARN_SYM_LEAF_FROM_STATS (DROPPED-noise: sym-iter-leaf istella/taxi -0.2%, old base) deleted 2026-10-09 on
# lane/owed-deletions-D1 (one Newton step from the searcher's partition stats, on the device); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_SYM_LEAF_FROM_STATS.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_SYM_DERIV_FUSED (DROPPED-noise: sym-iter-fused istella/taxi -0.1%, old base) deleted 2026-10-09 on
# lane/owed-deletions-D1 (the next tree's gradient pass behind this tree's tail drain); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_SYM_DERIV_FUSED.patch; record in docs/TOMBSTONES.md.
