# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Python entry points for `svm/impl/svc_ovo_layout.mojo` (the shell): they
unpack PythonObject arguments, call the typed device kernels and box the
result. `ovo_pairs_from_python` is shared with the host column's shell."""

from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceContext

from svm.impl.svc_ovo_layout import (
    OvoPairs,
    ovo_check_result,
    ovo_dual_gemv_device,
    ovo_layout_device,
)


def ovo_pairs_from_python(sup_addrs: PythonObject, dual_addrs: PythonObject, meta: PythonObject) raises -> OvoPairs:
    var p = Int(py=meta[2])
    if len(sup_addrs) != p or len(dual_addrs) != p or len(meta) != 3 + 3 * p:
        raise Error("svc_ovo_layout: one support and one dual address per pair, meta [k, n, P, (i, j, m) * P]")
    var out = OvoPairs(List[Int](), List[Int](), List[Int](), List[Int](), List[Int]())
    for q in range(p):  # small-loop(p: OvO class pairs): reads one pair's addresses and sizes, plan entries only
        out.sup.append(Int(py=sup_addrs[q]))
        out.dual.append(Int(py=dual_addrs[q]))
        out.ci.append(Int(py=meta[3 + 3 * q]))
        out.cj.append(Int(py=meta[4 + 3 * q]))
        out.m.append(Int(py=meta[5 + 3 * q]))
        if out.m[q] < 0 or (out.m[q] > 0 and (out.sup[q] == 0 or out.dual[q] == 0)):
            raise Error("svc_ovo_layout: a pair's support or dual buffer is null")
    return out^
