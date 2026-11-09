# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the w4 candidates (x_decomp/w4_fast.mojo): the doors
the GPU binding registers. Each door reads compile-time flags or unpacks ids
and integers, releases the GIL and calls one typed entry in
x_decomp/w4_fast.mojo; no data work happens here."""
from std.python import Python, PythonObject

from x_decomp.w4_fast import RSVD_FAST_DIRECT_IN, RSVD_IDN_DIRECT_IN


def w4_flags_py() raises -> PythonObject:
    """Which w4 candidates this build compiled in (bit 2 RSVD_FAST_DIRECT_IN or
    RSVD_IDN_DIRECT_IN; bit 1, LLE_FAST_DEV_LU, was deleted 2026-11-08; bit 4,
    LLE_FAST_NULL_CANON, was deleted 2026-10-09)."""
    var f = 0
    comptime if RSVD_FAST_DIRECT_IN or RSVD_IDN_DIRECT_IN:
        f |= 2
    return PythonObject(f)


# TOMBSTONE: MOJOLEARN_LLE_FAST_DEV_LU (its door dev_lu_aux_py) deleted 2026-11-08 by ed9ba6e7a1; code recoverable at 41ffcec665.
# Restore: git apply experiments/removed/MOJOLEARN_LLE_FAST_DEV_LU.patch
