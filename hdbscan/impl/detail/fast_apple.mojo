# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The FAST Apple experiment switches of lane af-hdbscan2 (2026-10-03).

Every switch is a build define read at compile time and compiled ONLY under
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()`; an
IDENTICAL build, a FAST build off Apple and a FAST Apple build without the
define compile main's code unchanged. `-D MOJOLEARN_HDBSCAN2_ALL` turns every
switch on at once. docs/apple-fast/notes/hdbscan.md has the profile each one
answers; docs/apple-fast/ab/hdbscan2.md the mechanism and the risk.
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST


comptime HDB_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)
"""The FAST tier on the Apple GPU: the only place any switch below is true."""

comptime HDB_ALL = is_defined["MOJOLEARN_HDBSCAN2_ALL"]()

comptime HDB_SMR_TILED = HDB_FAST_APPLE and (
    is_defined["MOJOLEARN_HDB_SMR_TILED"]() or HDB_ALL
    or not is_defined["MOJOLEARN_HDB_SMR_TILED_OFF"]()
)
"""The sparse arm's tiled search kernel on Apple (sparse_mr_mst.mojo).
FAST + Apple DEFAULT since lane/apple-fast-batchv (2026-10-03), M3 A/B vs
main batchv-hdb-smr-istella: hdbscan istella 44,879 -> 3,800 ms, n_clusters
47 / noise 0.25381 both arms (taxi takes the d <= 64 arm: 426 / 434 ms,
160 / 0.14222 both). -D MOJOLEARN_HDB_SMR_TILED_OFF: main's search."""

# INCONCLUSIVE-speed, M3 batchv-hdb-core-taxi / -istella vs main:
# taxi +0.8%; istella 44717 -> 45590 ms (+2%), clusters identical.
# Old-base -11% did not carry; dropped for no demonstrated main gain.
# See docs/apple-fast/EXPERIMENTS.md (HDB_CORE_TILE).
# F15/hdbscan-core M3 2026-10-06: six cold/repeated public fit times,
# B/A0.9529..1.0389, mixed; retain OFF. Same three-case quality contract,
# one warmup/one score, caller67d0efb29; results/F15/hdbscan-core.
comptime HDB_CORE_TILE = HDB_FAST_APPLE and (
    is_defined["MOJOLEARN_HDB_CORE_TILE"]() or HDB_ALL
)
"""Core distances from one tiled kernel with a register top-k (core_tile.mojo)."""

comptime HDB_DEV_BORUVKA = HDB_FAST_APPLE and (
    is_defined["MOJOLEARN_HDB_DEV_BORUVKA"]() or HDB_ALL
)
"""The d <= 64 arm's Boruvka rounds on the device (fast_mr_mst_device.mojo)."""

# INCONCLUSIVE-speed, M3 batchv-hdb-onesync-taxi / -istella vs main:
# 424.7 -> 425.5 / 3809 -> 3829 ms, clusters identical (SMR on both).
# Old-base -5% did not carry; dropped for no demonstrated main gain.
# See docs/apple-fast/EXPERIMENTS.md (HDB_ONE_SYNC).
# F15/hdbscan-downloads M3 2026-10-06: same three public fits as linkage;
# cold B/A0.9506,0.9225,0.8915, repeated0.8997,0.9363,0.8688;
# task quality equal. One scored sample. Promising independently, retain
# OFF pending combined timing with promoted linkage (no combination artifact
# is present). Caller/build evidence: results/F15/hdbscan-downloads under
# the retained root above. Historical full-dataset inconclusive result stays.
comptime HDB_ONE_SYNC = HDB_FAST_APPLE and (
    is_defined["MOJOLEARN_HDB_ONE_SYNC"]() or HDB_ALL
)
"""The extract's and the runner's output downloads under one wait each."""

# F15/hdbscan-linkage M3 2026-10-06: promote FAST Apple device linkage.
# Full public cold/repeated fits at509x7,997x13,1031x67 (3/4/5 separated
# clusters, min_samples5, seed319): cold B/A0.8085,0.6857,0.6613;
# repeated0.6957,0.6656,0.6698. All captured task quality metrics equal.
# One excluded warmup+one scored task; three planned cases, no universal
# dataset claim. Caller67d0efb29; unchanged reused build/hash provenance:
# ~/mojolearn-evidence/ab-overnight-20261006/m3/artifacts/results/F15/
# hdbscan-linkage. No identity/compile rerun. OFF is the baseline/rollback.
# ONE_SYNC remains off: individual wins do not establish a combined win.
comptime HDB_LINKAGE_DEVICE = HDB_FAST_APPLE and (
    not is_defined["MOJOLEARN_HDB_LINKAGE_DEVICE_OFF"]()
)
"""The dendrogram's per-level hook loop as one lock-free union launch (no
flag readback, dendrogram_union.mojo) and the condense with two status
readbacks instead of eight waits (tree_device.mojo `_condensed_two_reads`)."""

# F15/hdbscan-selection M3 2026-10-06: six cold/repeated public fit times,
# B/A0.9170..1.0298, mixed; retain OFF. Same three-case quality contract,
# one warmup/one score, caller67d0efb29; results/F15/hdbscan-selection.
comptime HDB_SELECT_DEVICE = HDB_FAST_APPLE and (
    is_defined["MOJOLEARN_HDB_SELECT_DEVICE"]() or HDB_ALL
)
"""Stabilities, selection, labels, scores and probabilities with no wait in
between and ONE readback at the end (extract.mojo `_extract_one_read`);
epsilon != 0 keeps main's route."""
