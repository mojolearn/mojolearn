# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The FAST Apple experiment switches of lane af-hdbscan2 (2026-10-03).

Every switch is a build define read at compile time and compiled ONLY under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`; an
IDENTICAL build, a FAST build off Apple and a FAST Apple build without the
define compile main's code unchanged. docs/apple-fast/notes/hdbscan.md has the profile each one
answers; docs/apple-fast/ab/hdbscan2.md the mechanism and the risk.
"""

from core.accelerator_target import TARGET_HAS_APPLE_GPU

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST


comptime HDB_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and TARGET_HAS_APPLE_GPU
)
"""The FAST tier on the Apple GPU: the only place any switch below is true."""

# TOMBSTONE: MOJOLEARN_HDBSCAN2_ALL (DROP) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: every af-hdbscan2 switch at once; its gain was HDB_SMR_TILED alone (now the default), CORE_TILE and ONE_SYNC lost vs main.
# Restore: git apply experiments/removed/MOJOLEARN_HDBSCAN2_ALL.patch; record in docs/TOMBSTONES.md.

# AFCL-G04 (never run; halved the core tile) was deleted with HDB_CORE_TILE below.

# TOMBSTONE: MOJOLEARN_HDB_SMR_TILED (force-on knob equal to the FAST Apple default; EXP:1911) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at bb7c99f6b4.
# Restore: git apply experiments/removed/MOJOLEARN_HDB_SMR_TILED.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_HDB_SMR_TILED_OFF (old path: the per-pair search on FAST Apple; the tiled kernel is the measured default, EXP:1911) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at bb7c99f6b4.
# Restore: git apply experiments/removed/MOJOLEARN_HDB_SMR_TILED_OFF.patch; record in docs/TOMBSTONES.md.
comptime HDB_SMR_TILED = HDB_FAST_APPLE
"""The sparse arm's tiled search kernel on Apple (sparse_mr_mst.mojo).
FAST + Apple DEFAULT since lane/apple-fast-batchv (2026-10-03), M3 A/B vs
main batchv-hdb-smr-istella: hdbscan istella 44,879 -> 3,800 ms, n_clusters
47 / noise 0.25381 both arms (taxi takes the d <= 64 arm: 426 / 434 ms,
160 / 0.14222 both)."""

# TOMBSTONE: MOJOLEARN_HDB_CORE_TILE (DROP) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: core distances from one tiled kernel with a register top-k (core_tile.mojo, with its AFCL_G04 tile knob); hdbscan istella 44,717 -> 45,590 ms (+2%), taxi +0.8%, clusters identical; F15 mixed (B/A 0.95..1.04).
# Restore: git apply experiments/removed/MOJOLEARN_HDB_CORE_TILE.patch; record in docs/TOMBSTONES.md.

# TOMBSTONE: MOJOLEARN_HDB_DEV_BORUVKA (DROPPED-noise) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
# Tried: the d <= 64 arm's Boruvka rounds driven on the device (fast_mr_mst_device.mojo); hdbscan taxi 434 -> 432.6 ms (-0.3%, noise).
# Restore: git apply experiments/removed/MOJOLEARN_HDB_DEV_BORUVKA.patch; record in docs/TOMBSTONES.md.

# TOMBSTONE: MOJOLEARN_HDB_ONE_SYNC (FAST Apple opt-in: output downloads under one wait; EXP:1914 inconclusive full-data, owed row withdrawn) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at 0aed7700fc.
# Restore: git apply experiments/removed/MOJOLEARN_HDB_ONE_SYNC.patch; record in docs/TOMBSTONES.md.
# Tried: M3 batchv-hdb-onesync taxi / istella 424.7 -> 425.5 / 3809 -> 3829 ms (inconclusive); F15/hdbscan-downloads cold B/A 0.89..0.95, one sample, never timed with the promoted linkage.
# The same staged route stays the IDENTICAL default (idn_switches.IDN_HDB_ONE_SYNC).

# F15/hdbscan-linkage M3 2026-10-06: promote FAST Apple device linkage.
# Full public cold/repeated fits at509x7,997x13,1031x67 (3/4/5 separated
# clusters, min_samples5, seed319): cold B/A0.8085,0.6857,0.6613;
# repeated0.6957,0.6656,0.6698. All captured task quality metrics equal.
# One excluded warmup+one scored task; three planned cases, no universal
# dataset claim. Caller67d0efb29; unchanged reused build/hash provenance:
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/
# hdbscan-linkage. No identity/compile rerun.
# TOMBSTONE: MOJOLEARN_HDB_LINKAGE_DEVICE_OFF (old path: the per-level hook loop on FAST Apple; the device linkage is the recorded winner, EXP:509) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at f23100a40e.
# Restore: git apply experiments/removed/MOJOLEARN_HDB_LINKAGE_DEVICE_OFF.patch; record in docs/TOMBSTONES.md.
comptime HDB_LINKAGE_DEVICE = HDB_FAST_APPLE
"""The dendrogram's per-level hook loop as one lock-free union launch (no
flag readback, dendrogram_union.mojo) and the condense with two status
readbacks instead of eight waits (tree_device.mojo `_condensed_two_reads`)."""

# TOMBSTONE: MOJOLEARN_HDB_SELECT_DEVICE (FAST Apple opt-in: one-readback selection; F15 B/A 0.92..1.03 mixed, EXP:1915 owed row withdrawn) deleted 2026-11-15 by lane/expdel2-11-cluster; code recoverable at aab954d195.
# Restore: git apply experiments/removed/MOJOLEARN_HDB_SELECT_DEVICE.patch; record in docs/TOMBSTONES.md.
# The same route stays the IDENTICAL default (idn_switches.IDN_HDB_SELECT_ONE_READ).
