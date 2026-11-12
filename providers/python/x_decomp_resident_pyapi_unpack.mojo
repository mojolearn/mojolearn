# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python shell of the device-resident matrices (x_decomp/resident.mojo):
the doors the GPU binding registers. Each door unpacks ids and integers and
calls one typed entry in x_decomp/resident.mojo; no data work happens here."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.resident import (
    pool_alloc,
    pool_free,
    pool_live_count,
    dev_upload_run,
    dev_download_run,
    dev_ew_run,
    dev_gemm_run,
    dev_rand_run,
    dev_trisolve_run,
    dev_colsum_run,
    dev_rowsum_run,
    dev_sqdist_run,
    dev_absmax_run,
    dev_orth_run,
    dev_orth_diag_run,
    dev_svd_run,
    dev_lu_run,
    dev_lu_aux_run,
    dev_code_rows_run,
    dev_eigh_run,
    dev_eigh_to_run,
    GRP_CLS2_NOSCAN,
    GRP_CLS2_DEVSCAN,
    GRP_CLS2_LAZY,
    GRP_FAST_FUSED,
    GRP_FAST_SRP_STRAT,
    dev_move_run,
)


def _id(o: PythonObject) raises -> Int:
    return Int(py=o)


def _n(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension out of range")
    return v
