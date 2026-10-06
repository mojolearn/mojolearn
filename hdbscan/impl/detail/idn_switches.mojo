# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTICAL speed switches of lane fam2-cluster (2026-10-04), HDBSCAN.

Each is ON by default under `GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL` on
every vendor, has its own `-D MOJOLEARN_<name>_OFF=1`, and is off under the
master `-D MOJOLEARN_IDN_ALL_OFF=1`. None moves a bit: each one runs the
same kernels on the same values and removes waits, readbacks or whole
passes. The FAST Apple switches of `fast_apple.mojo` are not changed; where
a mechanism was written there first (lane af-hdbscan2), the IDENTICAL
switch below ORs into the same `comptime if`.
"""

from std.sys.compile import is_defined
from std.sys.defines import get_defined_int

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime _IDN_HDB_ON = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

comptime IDN_HDB_ONE_SYNC = (
    # I14 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    _IDN_HDB_ON and not is_defined["MOJOLEARN_IDN_HDB_ONE_SYNC_OFF"]()
)
"""The extract's eight output downloads and the runner's six under one wait
each (`td_stage_*` / `td_take_*`, af-hdbscan2's HDB_ONE_SYNC route)."""

comptime IDN_HDB_CONDENSE_TWO_READS = (
    _IDN_HDB_ON
    # I14 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    and not is_defined["MOJOLEARN_IDN_HDB_CONDENSE_TWO_READS_OFF"]()
)
"""The condense with two status readbacks instead of eight waits
(`tree_device.mojo::_condensed_two_reads`): the same kernels in the same
order, the refusals with the same messages and first indices."""

comptime IDN_HDB_SELECT_ONE_READ = (
    # I14 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    _IDN_HDB_ON and not is_defined["MOJOLEARN_IDN_HDB_SELECT_ONE_READ_OFF"]()
)
"""Stabilities, selection, labels, scores and probabilities with no wait in
between and ONE readback (`extract.mojo::_extract_one_read`); a nonzero
`cluster_selection_epsilon` keeps the multi-read route."""

comptime IDN_HDB_MR_FUSED_GUARD = (
    # I14 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    _IDN_HDB_ON and not is_defined["MOJOLEARN_IDN_HDB_MR_FUSED_GUARD_OFF"]()
)
"""Dense arm: the NaN count of the distance matrix (DEVIATION 623) and the
non-finite count of the mutual reachabilities (DEVIATION 1607) taken INSIDE
the mutual-reachability transform's one pass over the m * m cells and read
back together, instead of two more m * m passes with two waits each. Same
counts, same refusals, the distance refusal first."""

comptime IDN_HDB_SOFT_LEAN = (
    # I14 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    _IDN_HDB_ON and not is_defined["MOJOLEARN_IDN_HDB_SOFT_LEAN_OFF"]()
)
"""Soft clustering: the four intermediates are downloaded only when a trace
is being written (they are read by nothing else), and the five kernels run
with one wait at the end instead of one each."""

comptime IDN_HDB_PREDICT_DEVICE_CAST = (
    _IDN_HDB_ON
    # I14 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    and not is_defined["MOJOLEARN_IDN_HDB_PREDICT_DEVICE_CAST_OFF"]()
)
"""approximate_predict: the k-NN's UInt32 indices are narrowed to Int32 by
the fit's device kernel instead of a host loop over nq * k cells."""

comptime IDN_HDB_PREDICT_LEAN = (
    # I14 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    _IDN_HDB_ON and not is_defined["MOJOLEARN_IDN_HDB_PREDICT_LEAN_OFF"]()
)
"""approximate_predict (lane fix-c1-cluster): every host staging buffer is
allocated under one wait, the training matrix, the queries and the model
arrays are staged with bulk copies instead of per-element loops (and without
`_upload_*`'s three waits each), the kernels run with no wait between them,
and the four outputs come back under one wait. The same kernels on the same
values: no bit moves. Implies `IDN_HDB_PREDICT_DEVICE_CAST`'s device cast."""

comptime IDN_HDB_SPARSE_MIN_ROWS = (
    get_defined_int["MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS", 46340]()
    if _IDN_HDB_ON
    else 46340
)
"""CANDIDATE ARM (default 46340 = `PAIRWISE_MAX_ROWS`, no change). Under
`graph = auto` the matrix-free sparse arm (DEVIATION 1620: the same tree,
bit for bit) is taken ABOVE this many rows instead of only past the dense
bound. The dense arm allocates and scans m * m cells every Boruvka round;
the sparse arm prunes. Time `-D MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS=4096`,
`=16384` against the default. A value above 46340 is refused at compile
time. With a trace on, the dense arm's `hdbscan.mr.dists` stage is not
recorded by the sparse arm, so keep the default for card runs."""
