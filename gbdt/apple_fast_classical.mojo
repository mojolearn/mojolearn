# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Independent Apple FAST classical scheduling experiments, 2026-10-06.

Every card was deleted unmeasured on 2026-11-15; the names remain as False.
"""

# Every card below was deleted 2026-11-15 (lane/expdel2-41-trees, V2 rule g:
# never run); each name stays as the constant False for the files that import it.

# AFCL-T01: halve rows per quantized-histogram replica to shorten each
# block's serial row walk; the existing occupancy floor/cap still applies.
# TOMBSTONE: MOJOLEARN_AFCL_T01 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 382112ba07.
# Restore: git apply experiments/removed/MOJOLEARN_AFCL_T01.patch; record in docs/TOMBSTONES.md.
comptime AFCL_T01 = False

# AFCL-T02: eight features use 16 KiB of histogram shared memory instead
# of Apple's 32 KiB group, trading extra workgroups for resident capacity.
# TOMBSTONE: MOJOLEARN_AFCL_T02 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 4dcdb311b5.
# Restore: git apply experiments/removed/MOJOLEARN_AFCL_T02.patch; record in docs/TOMBSTONES.md.
comptime AFCL_T02 = False

# AFCL-T03: 512 threads retain room for all 256 borders while reducing
# the per-feature binarizer's register footprint per threadgroup.
# TOMBSTONE: MOJOLEARN_AFCL_T03 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 951210db46.
# Restore: git apply experiments/removed/MOJOLEARN_AFCL_T03.patch; record in docs/TOMBSTONES.md.
comptime AFCL_T03 = False

# AFCL-T04: four Apple SIMD groups per partition-metadata block instead
# of eight; all kernels retain their grid stride and visit every row.
# TOMBSTONE: MOJOLEARN_AFCL_T04 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at 358548637a.
# Restore: git apply experiments/removed/MOJOLEARN_AFCL_T04.patch; record in docs/TOMBSTONES.md.
comptime AFCL_T04 = False

# AFCL-T05: four adjacent score records per lane before advancing a
# cooperative block; comparator and final reduction order stay unchanged.
# Both arms require MOJOLEARN_SYM_RESOLVE_BLOCK and the fused search path.
# TOMBSTONE: MOJOLEARN_AFCL_T05 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at c8fa4d47fc.
# Restore: git apply experiments/removed/MOJOLEARN_AFCL_T05.patch; record in docs/TOMBSTONES.md.
comptime AFCL_T05 = False
comptime AFCL_RESOLVE_RECORDS = 1

# AFCL-T06: halve hardware-derived leaf-statistic partial counts to reduce
# scratch/fold traffic; accumulation order may move bits, not solver effort.
# TOMBSTONE: MOJOLEARN_AFCL_T06 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at bc67334f43.
# Restore: git apply experiments/removed/MOJOLEARN_AFCL_T06.patch; record in docs/TOMBSTONES.md.
comptime AFCL_T06 = False

# AFCL-T07: four SIMD groups per resident prediction block rather than
# eight; rows remain independent and each row visits trees in order.
# TOMBSTONE: MOJOLEARN_AFCL_T07 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at f22e75ee88.
# Restore: git apply experiments/removed/MOJOLEARN_AFCL_T07.patch; record in docs/TOMBSTONES.md.
comptime AFCL_T07 = False
comptime AFCL_PREDICT_BLOCK = 256

# AFCL-T12: four SIMD groups per CTR elementwise block rather than eight;
# same sorted categories, prefixes, priors and output destinations.
# TOMBSTONE: MOJOLEARN_AFCL_T12 (DELETE-EXPERIMENT, never-run Apple FAST programme card, rule g) deleted 2026-11-15 by lane/expdel2-41-trees; code recoverable at d985e1b336.
# Restore: git apply experiments/removed/MOJOLEARN_AFCL_T12.patch; record in docs/TOMBSTONES.md.
comptime AFCL_T12 = False
