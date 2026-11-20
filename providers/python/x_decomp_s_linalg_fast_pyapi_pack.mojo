# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/s_linalg_fast.mojo: the route flags and the
resident LU, upload-scan and orthonormalize doors the GPU binding registers.
Each unpacks addresses, pool ids and integers, releases the GIL and calls
one typed `*_run` entry in x_decomp/s_linalg_fast.mojo; no data work happens
here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from providers.python.x_decomp_resident_pyapi_unpack import _id, _n
from x_decomp.s_linalg_fast import (
    dev_orth_ws_run,
    s_flags_run,
)


def s_flags_py() raises -> PythonObject:
    """Bit 2 DECOMP_FAST_ORTH_WS, bit 4 LU_FAST_RESIDENT, bit 8
    IDN_LU_RESIDENT_PAIR (0 when none; bit 1 was RSVD_FAST_DEVSCAN)."""
    return PythonObject(s_flags_run())






# TOMBSTONE: MOJOLEARN_RSVD_FAST_DEVSCAN (its door dev_upload_scan_py, x_decomp_dev_upload_scan) deleted 2026-11-08 by lane a3-admit-decomp-prep-metrics; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A3_ADMISSION_HELPERS.patch.
# Restored 2026-11-09 by lane r1-restore-measured-winners as behaviour, not as this entry: the define is back and its
# device test runs inside randomized_svd's one call through resident.mojo dev_upload_finite_run (the one upload-finite body).


def dev_orth_ws_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """`dev_orth_py` (dst = the orthonormalized columns of a; a unchanged)
    with `orth_ws_on_device`."""
    var m = _n(p, 0)
    var l = _n(p, 1)
    var ia = _id(a)
    var io = _id(dst)
    var cells = 0
    with GILReleased(Python()):
        cells = dev_orth_ws_run(ia, io, m, l)
    return PythonObject(cells)
