# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LAMB's one step over packed tensors: its table, its bias corrections and
its launches (MOVED WORD FOR WORD, lane nn-optim 2026-11-23, from
sequence/ops_impl.mojo, so `Sequential.compile(optimizer="lamb")`
(autograd/nn_optim.mojo) runs the same body as the public LAMB optimizer
(sequence/ops_impl.mojo lamb_step_impl) and its resident route
(sequence/opt_resident.mojo) without importing the estimators' module).
The element kernels are sequence/adafactor.mojo's (OP_LAMB_*)."""
from std.memory import bitcast
from checks.numerics import ftz, identical_mul, identical_sqrt
from sequence.exec_trait import Exec
from sequence.ops import (
    FP, Args, OP_FILL, OP_LAMB_APPLY_ALL, OP_LAMB_BLK, OP_LAMB_CLIP, OP_LAMB_SEGFOLD, OP_LAMB_TRUST, OP_LAMB_UPD,
)
from sequence.adafactor import AF_NORM_BLOCK

def lamb_table(offs: List[Int], mut tab: List[Float32]) raises -> Int:
    """LAMB's table (`sequence/adafactor.mojo`, THE TABLE): the nt + 1
    element offsets, then each tensor's first block index of AF_NORM_BLOCK
    values and the block count, as int32 bit patterns in float32 words
    (copied, never converted). Returns the block count."""
    var nt = len(offs) - 1
    tab.clear()
    for k in range(nt + 1):
        tab.append(bitcast[DType.float32](Int32(offs[k])))
    var nb = 0
    for k in range(nt):
        tab.append(bitcast[DType.float32](Int32(nb)))
        nb += (offs[k + 1] - offs[k] + AF_NORM_BLOCK - 1) // AF_NORM_BLOCK
    tab.append(bitcast[DType.float32](Int32(nb)))
    return nb



def lamb_bias(flags: Int, t: Int, t0: Int, b1: Float32, b2: Float32, sc: FP,
              carry: Bool) -> Tuple[Float32, Float32, Float32, Float32]:
    """(bc1, bc2, beta1^t, beta2^t): the host scalars of step t, from the
    running products after step t0 (sc, when carried and t0 > 0) or from
    step 1, one identical_mul per step."""
    var pw1 = Float32(1.0)
    var pw2 = Float32(1.0)
    if (flags & 8) == 0:
        return (Float32(1.0), Float32(1.0), pw1, pw2)
    var k0 = 0
    if carry and t0 > 0:
        pw1 = sc.unsafe_load(0)
        pw2 = sc.unsafe_load(1)
        k0 = t0
    for _ in range(k0, t):
        pw1 = ftz(identical_mul(pw1, b1))
        pw2 = ftz(identical_mul(pw2, b2))
    return (Float32(1.0) - pw1, Float32(1.0) - pw2, pw1, pw2)



def lamb_core[E: Exec](
    mut ex: E, P: FP, G: FP, M: FP, V: FP, U: FP, TAB: FP,
    partsA: FP, partsB: FP, nrm: FP, ratio: FP, scal: FP,
    n: Int, nt: Int, nb: Int, flags: Int,
    lr: Float32, b1: Float32, b2: Float32, eps: Float32, wd: Float32, max_norm: Float32,
    bc1: Float32, bc2: Float32,
) raises:
    """ONE LAMB STEP over every tensor, launches only (lane gap-optimizers):
    nothing is read back. The global clip's per-tensor gradient norms and
    its coefficient on the device (op_lamb_blk, op_lamb_segfold,
    op_lamb_clip; the coefficient was folded on the host from downloaded
    norms), the moments and the update (op_lamb_upd, which applies the
    clip), the trust ratios (op_lamb_blk over p and u, op_lamb_trust) and
    the update of every tensor (op_lamb_apply_all, one launch). Buffers:
    P, G, M, V, U n floats; TAB `lamb_table`'s 2 nt + 2 words; partsA and
    partsB nb; nrm and ratio nt; scal 1. G is read, never written."""
    if (flags & 16) != 0:
        var q = Args()
        q.p0 = G
        q.p2 = TAB
        q.p3 = partsA
        q.i0 = nt
        q.i2 = AF_NORM_BLOCK
        ex.launch[OP_LAMB_BLK](q, nb)
        var s = Args()
        s.p0 = partsA
        s.p1 = TAB
        s.p2 = nrm
        s.i0 = nt
        ex.launch[OP_LAMB_SEGFOLD](s, nt)
        var c = Args()
        c.p0 = nrm
        c.p1 = scal
        c.i0 = nt
        c.f0 = max_norm
        ex.launch[OP_LAMB_CLIP](c, 1)
    var u = Args()
    u.p0 = P
    u.p1 = G
    u.p2 = M
    u.p3 = V
    u.p4 = U
    u.p5 = scal
    u.i0 = 1 if (flags & 16) != 0 else 0
    u.f1 = b1
    u.f2 = b2
    u.f3 = eps
    u.f4 = wd
    u.f5 = bc1
    u.f6 = ftz(identical_sqrt(bc2))
    u.f7 = (Float32(1.0) - b1) if (flags & 4) != 0 else Float32(1.0)
    ex.launch[OP_LAMB_UPD](u, n)
    if wd != Float32(0.0) or (flags & 2) != 0:
        var q = Args()
        q.p0 = P
        q.p1 = U
        q.p2 = TAB
        q.p3 = partsA
        q.p4 = partsB
        q.i0 = nt
        q.i1 = 1
        q.i2 = AF_NORM_BLOCK
        ex.launch[OP_LAMB_BLK](q, nb)
        var r = Args()
        r.p0 = partsA
        r.p1 = partsB
        r.p2 = TAB
        r.p3 = ratio
        r.i0 = flags & 1
        r.i1 = nt
        ex.launch[OP_LAMB_TRUST](r, nt)
    else:
        var f = Args()
        f.p0 = ratio
        f.f0 = Float32(1.0)
        ex.launch[OP_FILL](f, nt)
    var ap = Args()
    ap.p0 = P
    ap.p1 = U
    ap.p2 = ratio
    ap.p3 = TAB
    ap.i0 = nt
    ap.f0 = lr
    ex.launch[OP_LAMB_APPLY_ALL](ap, n)

