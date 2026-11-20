# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The doors of TruncatedSVD's decomp-kit routes (x_decomp/tsvd_owner.mojo):
each unpacks addresses and integers, releases the GIL and calls one typed
driver; no data work happens here. The native TruncatedSVD owner
(bindings/tsvd_native.mojo) is their only caller."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.pca_randomized import RDev
from x_decomp.tsvd_owner import tsvd_randomized_fit_run, tsvd_tsqr_components_run


def _ptr(address: PythonObject) raises -> F32Ptr:
    var a = Int(py=address)
    if a == 0:
        raise Error("x_decomp: null float32 buffer address")
    return F32Ptr(unsafe_from_address=a)


def _dim(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension out of range")
    return v
