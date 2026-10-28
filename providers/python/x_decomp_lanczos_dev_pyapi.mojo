# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of x_decomp/lanczos_dev.mojo: the device Lanczos door and
the route flags the GPU binding registers. The door unpacks pool ids and
integers and calls `dev_lanczos_run`; no data work happens here."""
from std.python import PythonObject

from x_decomp.lanczos_dev import IPCA_FAST_DEV, KPCA_FAST_LANCZOS_DEV, dev_lanczos_run
from providers.python.x_decomp_resident_pyapi import _id, _n


# TOMBSTONE: the dead x_decomp exports (registrations only where the body stays live in the decomp owners; see docs/TOMBSTONES.md) deleted 2026-10-26 by 25b5daca2; code recoverable at a88745022.
# Restore: git apply experiments/removed/dead_entries_x_decomp.patch


def dev_lanczos_py(a: PythonObject, q: PythonObject, ab: PythonObject, p: PythonObject) raises -> PythonObject:
    """Lanczos steps j0 .. j1-1 (see x_decomp/lanczos_dev.mojo), enqueued.
    a: n x n; q: the basis, rows 0 .. j0 hold q_0 .. q_j0 on entry, with
    room for cap + 1 rows; ab: 2 cap floats. p = [n, j0, j1, cap]."""
    var n = _n(p, 0)
    var j0 = _n(p, 1)
    var j1 = _n(p, 2)
    var cap = _n(p, 3)
    var ia = _id(a)
    var iq = _id(q)
    var iab = _id(ab)
    var steps = dev_lanczos_run(ia, iq, iab, n, j0, j1, cap)
    return PythonObject(steps)

