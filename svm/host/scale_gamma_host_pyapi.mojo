# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python entry points for `svm/host/scale_gamma_host.mojo` (the shell): they
unpack PythonObject arguments, call the typed host kernel and box the result."""

from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from core.py2mojo_linear import py2mojo_linear_flags
from svm.host.scale_gamma_host import scale_gamma_limbs_host


def scale_gamma_limbs_host_binding(
    x_addr: PythonObject, count: PythonObject, out_addr: PythonObject
) raises -> PythonObject:
    """`scale_gamma_limbs(x, count, out)` on the host column: `count` float32
    cells at x, SG_SLOTS int64 words written at out. Returns count."""
    var n = Int(py=count)
    var xa = Int(py=x_addr)
    var oa = Int(py=out_addr)
    if n < 0 or oa == 0 or (n > 0 and xa == 0):
        raise Error("scale_gamma_limbs: null buffer or negative count")
    with GILReleased(Python()):
        scale_gamma_limbs_host(
            MutPointer[UInt32, MutAnyOrigin](unsafe_from_address=xa),
            n,
            MutPointer[Int64, MutAnyOrigin](unsafe_from_address=oa),
        )
    return PythonObject(n)


def py2mojo_linear_flags_binding() raises -> PythonObject:
    return PythonObject(py2mojo_linear_flags())
