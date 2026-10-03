# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""gamma='scale' exact sums, THE HOST COLUMN (lane/apple-fast-py2mojo-linear):
`svm/impl/scale_gamma_limbs.mojo`'s limbs from one host loop, for the CPU
bindings (`_mojolearn_svm_host`, `_mojolearn_kernel_methods_host`)."""

from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from core.py2mojo_linear import py2mojo_linear_flags
from svm.impl.scale_gamma_limbs import (
    SG_CHUNK,
    SG_SLOTS,
    sg_add_cell,
    sg_normalize,
    sg_zero,
)


def scale_gamma_limbs_host(
    x: MutPointer[UInt32, MutAnyOrigin], count: Int, dst: MutPointer[Int64, MutAnyOrigin]
):
    """THE HOST COLUMN: the same limbs from one host loop (exact, so the
    same value as the device grid)."""
    var acc = sg_zero()
    var i = 0
    while i < count:
        var end = min(count, i + SG_CHUNK)
        for j in range(i, end):
            sg_add_cell(acc, x[j])
        sg_normalize(acc)
        i = end
    for k in range(SG_SLOTS):
        dst[k] = acc[k]


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
