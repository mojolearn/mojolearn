# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The doors of the decomposition owners' drivers (x_decomp/ica_owner.mojo,
...): each unpacks one list of addresses and integers (and one of floats),
releases the GIL and calls one typed driver; no data work happens here. The
native owners in bindings/ (ica_native.mojo, ...) are their only callers."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev
from x_decomp.decomp_owner_kit import DecompDev
from x_decomp.ica_owner import ica_fit_run, ica_project_run
from x_decomp.nmf import nmf_args_checked
from x_decomp.nmf_owner import nmf_inverse_run, nmf_run
from x_decomp.fa_owner import fa_fit_run, fa_query_run
from x_decomp.lda_fit import LdaArgs
from x_decomp.lda_owner import lda_run
from x_decomp.rp_owner import rp_fit_run, rp_project_run


def door_ptr(p: PythonObject, i: Int) raises -> F32Ptr:
    var a = Int(py=p[i])
    if a == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=a)


def door_dim(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension out of range")
    return v
