# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""gamma='scale' exact sums, THE HOST COLUMN (lane/apple-fast-py2mojo-linear):
`svm/impl/scale_gamma_limbs.mojo`'s limbs from one host loop, for the CPU
bindings (`_mojolearn_svm_host`, `_mojolearn_kernel_methods_host`). The Python
entry points are in `providers/python/svm_host_scale_gamma_host_pyapi.mojo`."""

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

