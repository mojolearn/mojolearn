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
)
"""The sparse arm's tiled search kernel on Apple (sparse_mr_mst.mojo)."""

comptime HDB_CORE_TILE = HDB_FAST_APPLE and (
    is_defined["MOJOLEARN_HDB_CORE_TILE"]() or HDB_ALL
)
"""Core distances from one tiled kernel with a register top-k (core_tile.mojo)."""

comptime HDB_DEV_BORUVKA = HDB_FAST_APPLE and (
    is_defined["MOJOLEARN_HDB_DEV_BORUVKA"]() or HDB_ALL
)
"""The d <= 64 arm's Boruvka rounds on the device (fast_mr_mst_device.mojo)."""

comptime HDB_ONE_SYNC = HDB_FAST_APPLE and (
    is_defined["MOJOLEARN_HDB_ONE_SYNC"]() or HDB_ALL
)
"""The extract's and the runner's output downloads under one wait each."""

comptime HDB_LINKAGE_DEVICE = HDB_FAST_APPLE and (
    is_defined["MOJOLEARN_HDB_LINKAGE_DEVICE"]() or HDB_ALL
)
"""The dendrogram's per-level hook loop as one lock-free union launch (no
flag readback, dendrogram_union.mojo) and the condense with two status
readbacks instead of eight waits (tree_device.mojo `_condensed_two_reads`)."""

comptime HDB_SELECT_DEVICE = HDB_FAST_APPLE and (
    is_defined["MOJOLEARN_HDB_SELECT_DEVICE"]() or HDB_ALL
)
"""Stabilities, selection, labels, scores and probabilities with no wait in
between and ONE readback at the end (extract.mojo `_extract_one_read`);
epsilon != 0 keeps main's route."""
