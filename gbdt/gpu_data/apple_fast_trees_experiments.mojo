"""Tombstones of the deleted Apple FAST tree cards G01--G12 (never run, 2026-11-15).

The sweep of 2026-11-15 removed their False constants, which nothing read.
"""

# G01: one uniform-barrier block per feature scans 128-bin tiles.
# No performance/quality evidence; FAST prefix sums may change rounding.
# TOMBSTONE: MOJOLEARN_AFT_G01 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at 55d610dfe9.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_G01.patch; record in docs/TOMBSTONES.md.
# G02: occupancy and per-partition row budget choose document replication.
# No performance/quality evidence; retain every row and the multiplier cap.
# TOMBSTONE: MOJOLEARN_AFT_G02 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at 957d58e74f.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_G02.patch; record in docs/TOMBSTONES.md.
# G03: one leader folds winners per block and broadcasts through shared memory.
# No performance/quality evidence; preserve original record order and tie rules.
# TOMBSTONE: MOJOLEARN_AFT_G03 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at 868b669683.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_G03.patch; record in docs/TOMBSTONES.md.
# G04: 512-thread partition reduction, without chunk atomics or extra planes.
# No performance/quality evidence; FAST reduction order may change.
# TOMBSTONE: MOJOLEARN_AFT_G04 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at 6df465532f.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_G04.patch; record in docs/TOMBSTONES.md.
# G05: 512-thread quantization block at the original eight rows per lane.
# No performance/quality evidence; identical borders and comparison routine.
# TOMBSTONE: MOJOLEARN_AFT_G05 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at d39d33ceb2.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_G05.patch; record in docs/TOMBSTONES.md.
# G06: 128-thread compressed-word packing with sixteen rows per lane.
# No performance/quality evidence; same 2048 rows/block and bit layout.
# TOMBSTONE: MOJOLEARN_AFT_G06 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at adccdfe434.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_G06.patch; record in docs/TOMBSTONES.md.
# G07: 128-thread Apple leaf walker objective reduction and direction traversal.
# No performance/quality evidence; leave shared/fused statistics contracts intact.
# TOMBSTONE: MOJOLEARN_AFT_G07 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at 5f0ca355ea.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_G07.patch; record in docs/TOMBSTONES.md.
# G08: four independent cursor rows per leaf-update loop iteration.
# No performance/quality evidence; same leaves, learning rate and row coverage.
# TOMBSTONE: MOJOLEARN_AFT_G08 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at 0c740c97bb.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_G08.patch; record in docs/TOMBSTONES.md.
# G09: eight-row numerator preparation tiles for exact ordered CTR sums.
# No performance/quality evidence; preserve segment signs and target predicates.
# TOMBSTONE: MOJOLEARN_AFT_G09 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at 70b5f7a860.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_G09.patch; record in docs/TOMBSTONES.md.
# G10: four coalesced rows per weighted-frequency/mean-scatter CTR worker.
# No performance/quality evidence; preserve priors, index maps and prefix sums.
# TOMBSTONE: MOJOLEARN_AFT_G10 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at d9dedc681d.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_G10.patch; record in docs/TOMBSTONES.md.
# G11: 128-thread ordered batched derivative/weight reduction.
# No performance/quality evidence; same prefix tasks, eight chunks and RNG.
# TOMBSTONE: MOJOLEARN_AFT_G11 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at 07284d9b26.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_G11.patch; record in docs/TOMBSTONES.md.
# G12: four-row ordered apply tiles with cached task descriptor bounds.
# No performance/quality evidence; every row still resolves its own prefix model.
# TOMBSTONE: MOJOLEARN_AFT_G12 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-42-trees; code recoverable at dd541ff51d.
# Restore: git apply experiments/removed/MOJOLEARN_AFT_G12.patch; record in docs/TOMBSTONES.md.
