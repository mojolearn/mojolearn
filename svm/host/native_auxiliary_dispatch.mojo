# SPDX-License-Identifier: Apache-2.0
# Existing SVC gamma/one-vs-one/linear-coefficient owners, through the common
# descriptor table. These are the original validation and native host calls.
from core.py2mojo_linear import py2mojo_linear_flags
from svm.host.scale_gamma_host import scale_gamma_limbs_host
from svm.impl.svc_ovo_layout import OvoPairs, ovo_check_result
from svm.host.svc_ovo_host import ovo_dual_gemv_host, ovo_layout_host


def svc_host_auxiliary(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    if op == 8:
        return py2mojo_linear_flags()
    if op == 9:
        var n = p[0]
        if n < 0 or a[1] == 0 or (n > 0 and a[0] == 0):
            raise Error("scale_gamma_limbs: null buffer or negative count")
        scale_gamma_limbs_host(MutPointer[UInt32, MutAnyOrigin](unsafe_from_address=a[0]), n,
                              MutPointer[Int64, MutAnyOrigin](unsafe_from_address=a[1]))
        return n
    if op == 10:
        var k = p[0]
        var n_bound = p[1]
        var count = p[2]
        var cap = p[3]
        if count < 0 or len(p) != 4 + 3 * count or len(a) != 3 + 2 * count:
            raise Error("svc_ovo_layout: one support and one dual address per pair, meta [k, n, P, (i, j, m) * P]")
        var pairs = OvoPairs(List[Int](), List[Int](), List[Int](), List[Int](), List[Int]())
        for q in range(count):  # driver-loop: native class-pair descriptors
            pairs.sup.append(a[3 + q * 2])
            pairs.dual.append(a[4 + q * 2])
            pairs.ci.append(p[4 + q * 3])
            pairs.cj.append(p[5 + q * 3])
            pairs.m.append(p[6 + q * 3])
            if pairs.m[q] < 0 or (pairs.m[q] > 0 and (pairs.sup[q] == 0 or pairs.dual[q] == 0)):
                raise Error("svc_ovo_layout: a pair's support or dual buffer is null")
        if k < 2 or n_bound < 0 or a[0] == 0 or a[1] == 0 or a[2] == 0:
            raise Error("svc_ovo_layout: needs two classes and three output buffers")
        var result = ovo_layout_host( pairs, k, n_bound, cap, a[0], a[1], a[2])
        ovo_check_result(result)
        return result
    if op == 11:
        var n = p[0]
        var d = p[1]
        if n < 0 or d < 0 or a[2] == 0:
            raise Error("svc_dual_gemv: bad dims or null output")
        if d == 0:
            return 0
        if n > 0 and (a[0] == 0 or a[1] == 0):
            raise Error("svc_dual_gemv: null input")
        ovo_dual_gemv_host(a[0], a[1], n, d, p[2], a[2])
        return d
    raise Error("SVC native ABI: unknown auxiliary operation")
