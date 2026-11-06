# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/rotation.mojo's varimax / quartimax rotation on resident device
matrices (lane py-runtime-b): the same statements on `DKit`; each scalar
Python read (max |A|, the singular-value sum) one word home (`_polar`'s steps
inlined in the loop). GPU binding only.

lane perf-manifold-kit (2026-11-06): one sync per iteration for the two scalar
reads (was two). Iteration i's singular-value sum and iteration i + 1's max |A|
come home together: the next iteration's polar input (C^T target(C R_i)) is
enqueued before the stopping test of iteration i is read, and dropped when the
test stops. The returned R, the stopping iteration and every word are the
ones the two-read loop produced (the dropped matrix never feeds R)."""
from x_decomp.cells import F32Ptr
from x_decomp.kit import Mat, OP_CUBE, OP_DIV, OP_MUL, OP_SCALE, OP_SQ, OP_SQRT, OP_SUB, mat_eye, mat_from
from x_decomp.kit_device import DKit, DMat
from x_decomp.rotation import polar_exp, pow2
from x_decomp.select_ops import SEL_MAXABS


def _polar_input(mut k: DKit, C: DMat, R: DMat, nrow: Int, vm: Bool) raises -> DMat:
    """C^T target(C R): the matrix `_polar` factors in one iteration."""
    var cr = k.mm(C, R, False, False)
    var target: DMat
    if vm:
        var tmp = k.ew2(OP_MUL, cr, k.ew1(OP_SCALE, k.colsum(k.ew1(OP_SQ, cr, 0.0)), 1.0 / Float64(nrow)))
        target = k.ew2(OP_SUB, k.ew1(OP_CUBE, cr, 0.0), tmp)
    else:
        target = k.ew1(OP_CUBE, cr, 0.0)
    return k.mm(C, target, True, False)


def ortho_rotation_dev_run(
    pc: F32Ptr, po: F32Ptr, nrow: Int, ncol: Int, vm: Bool, max_iter: Int, tol: Float64
) raises:
    """`ortho_rotation_run` on the resident kit."""
    if nrow * ncol > 2147483647 or ncol < 1:
        raise Error("x_decomp: rotation shape out of range")
    var k = DKit()
    var C = k.upload(mat_from(pc, nrow, ncol))
    var R = k.upload(mat_eye(ncol))
    var var_ = 0.0
    if max_iter > 0:
        # `polar_dev`'s steps, its two scalar reads folded across iterations
        var A0 = _polar_input(k, C, R, nrow, vm)
        var m = k.word(k.reduce(A0, SEL_MAXABS))
        for it in range(max_iter):  # driver-loop: over rotation iterations (max_iter)
            var A = k.copy(A0)
            var e = polar_exp(m)
            if e != 0:
                A = k.ew1(OP_SCALE, A, pow2(-e))
            var eg = k.eigh(k.mm(A, A, True, False))
            var sv = k.ew1(OP_SQRT, eg.wd, 0.0)
            var AV = k.ew2(OP_DIV, k.mm(A, eg.vd, False, False), sv)
            var sum_dev = k.total(sv)
            R = k.mm(AV, eg.vd, False, True)
            if it + 1 == max_iter:
                # the loop ends whatever the test says; the sum is not needed
                break
            # the next polar input, then both words home at one sync
            A0 = _polar_input(k, C, R, nrow, vm)
            # A0 is ncol x ncol with ncol >= 1 (checked above), never empty;
            # the reduce output stays owned until the sync has read it
            var red = k.reduce(A0, SEL_MAXABS)
            var hs = k.get(sum_dev)
            var hm = k.get(red)
            k.sync()
            var var_new = Float64(hs.d[0]) * pow2(e)
            m = Float64(hm.d[0])
            _ = red^
            _ = sum_dev^
            if var_ != 0 and var_new < var_ * (1 + tol):
                break
            var_ = var_new
    var T = k.t(k.mm(C, R, False, False))
    var h = k.get(T)
    k.sync()
    for i in range(h.n()):
        po.unsafe_store(i, h.d[i])
