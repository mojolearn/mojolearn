# SPDX-License-Identifier: Apache-2.0
"""Adafactor updates over one vector or matrix on the active executor."""
from std.math import sqrt
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul, identical_pow64
from sequence.adafactor import AF_NORM_BLOCK
from sequence.exec_trait import Exec
from sequence.ops import (
    Args, FP, OP_AF_ALPHA, OP_AF_APPLY, OP_AF_BLK_SUMSQ, OP_AF_COL, OP_AF_DENOM, OP_AF_RMEAN,
    OP_AF_ROW, OP_AF_UPDATE_MAT, OP_AF_VEC, OP_CHUNK_SUMSQ, OP_SCALE,
)


#: FAST's two-pass norms (sequence/adafactor.mojo, op_chunk_sumsq) start at
#: this many elements; below it the one-thread fold is as quick.
comptime FAST_NORM_MIN = 65536
comptime _PARTS = 4096      # = sequence.adafactor.SUMSQ_THREADS


def _fast_norms() -> Bool:
    return GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL


#: lane hr-adafactor (2026-10-02, plan HOST_ROUTE_REMOVAL (git history)): under
#: IDENTICAL, Adafactor's two whole-tensor norms of more than AF_NORM_BLOCK
#: values are the blocked order on the device (sequence/adafactor.mojo):
#: block partials, one GPU thread per block, then the partials added
#: ascending. Nothing leaves the device. It replaced one ascending chain
#: over the whole tensor, which a GPU thread walked at memory latency (the
#: board's 16,777,216-value lane, about a second a step on an MI325X) and
#: which tensors of 65,536 values or more folded on the host instead.
def _blocked_norms() -> Bool:
    return GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


def _blk_sumsq[E: Exec](mut ex: E, src: FP, n: Int, parts: FP) raises -> Int:
    """IDENTICAL: the ceil(n / AF_NORM_BLOCK) block partials of src[0:n]'s
    sum of squares into parts[0:]; returns how many."""
    var nb = (n + AF_NORM_BLOCK - 1) // AF_NORM_BLOCK
    var c = Args()
    c.p0 = src
    c.p1 = parts
    c.i0 = n
    c.i1 = AF_NORM_BLOCK
    ex.launch[OP_AF_BLK_SUMSQ](c, nb)
    return nb


def _chunk_sumsq[E: Exec](mut ex: E, src: FP, start: Int, n: Int, parts: FP, slot: Int) raises -> Int:
    """FAST: min(_PARTS, n) strided partial sums of squares of src[start:start+n]
    into parts[slot:]; returns how many."""
    var T = min(_PARTS, n)
    var c = Args()
    c.p0 = src
    c.p1 = parts
    c.i0 = n
    c.i1 = T
    c.i2 = slot
    c.i3 = start
    ex.launch[OP_CHUNK_SUMSQ](c, T)
    return T


@always_inline
def adafactor_partial_words(n: Int) -> Int:
    if _fast_norms() and n >= FAST_NORM_MIN:
        return _PARTS
    return (n + AF_NORM_BLOCK - 1) // AF_NORM_BLOCK if _blocked_norms() and n > AF_NORM_BLOCK else 1


def adafactor_core[E: Exec](mut ex: E, P: FP, G: FP, S1: FP, S2: FP, R: Int, C: Int, t: Int,
                            fp: List[Float64]) raises:
    """The launches of one Adafactor step on device-side P, G, S1 (row_var
    or the variance) and S2 (col_var; unused for a vector), queued, no
    transfer: `adafactor_step_impl`'s body, also the resident step's
    (`sequence/opt_resident.mojo`, AF_RESIDENT)."""
    var n = R * C if C > 0 else R
    var update = ex.alloc(n)
    var scales = ex.alloc(4)
    var parts = ex.alloc(adafactor_partial_words(n))
    adafactor_update(ex, P, G, S1, S2, R, C, t, fp[0], fp[1], fp[2], fp[3], fp[4], fp[5], update, scales, parts)


def adafactor_update[E: Exec](
    mut ex: E, P: FP, G: FP, S1: FP, S2: FP, R: Int, C: Int, t: Int,
    lr: Float64, beta2_decay: Float64, epsilon1: Float64, epsilon2: Float64, clip: Float64, decay: Float64,
    U: FP, sc: FP, parts: FP,
) raises:
    var n = R * C if C > 0 else R
    var w = Float32(identical_pow64(Float64(t), beta2_decay))
    var rho = Float32(min(lr, Float64(1.0) / sqrt(Float64(t))))
    var eps1 = Float32(epsilon1)
    var eps1sq = ftz(identical_mul(eps1, eps1))
    var wd = Float32(decay)
    var fast = _fast_norms() and n >= FAST_NORM_MIN
    var blocked = _blocked_norms() and n > AF_NORM_BLOCK
    # TOMBSTONE (lane/postmerge-act-5, 2026-10-09: -D MOJOLEARN_IDN_AF_VEC_FUSED (Adafactor fused vector step)):
    # the vfuse + vfin vector step was NOISE: NV 13.671 -> 13.738 ms (1.00x, nv2 v1010/v1012), AMD 8.620 -> 8.186 ms (0.95x, a1141/a1143), avg 0.98x, digest aa99a3dc unchanged; recoverable at main 0a7b206f1.
    var a = Args()
    a.p0 = P
    a.p1 = sc
    a.i0 = n
    a.f0 = Float32(epsilon2)
    a.f1 = rho
    if fast:
        a.p2 = parts
        a.i2 = _chunk_sumsq(ex, P, 0, n, parts, 0)
    elif blocked:
        a.p2 = parts
        a.i2 = _blk_sumsq(ex, P, n, parts)
    ex.launch[OP_AF_ALPHA](a, 1)
    if wd != Float32(0.0):
        var s = Args()
        s.p0 = P
        s.f0 = Float32(1.0) - ftz(identical_mul(Float32(lr), wd))
        ex.launch[OP_SCALE](s, n)
    if C > 0:
        var r = Args()
        r.p0 = G
        r.p1 = S1
        r.i0 = C
        r.f0 = w
        ex.launch[OP_AF_ROW](r, R)
        var c = Args()
        c.p0 = G
        c.p1 = S2
        c.i0 = R
        c.i1 = C
        c.f0 = w
        ex.launch[OP_AF_COL](c, C)
        var m = Args()
        m.p0 = S1
        m.p1 = sc
        m.i0 = R
        m.f0 = eps1
        ex.launch[OP_AF_RMEAN](m, 1)
        var u = Args()
        u.p0 = G
        u.p1 = S1
        u.p2 = S2
        u.p3 = sc
        u.p4 = U
        u.i0 = C
        u.f0 = eps1sq
        ex.launch[OP_AF_UPDATE_MAT](u, n)
    else:
        var v = Args()
        v.p0 = G
        v.p1 = S1
        v.p2 = U
        v.f0 = w
        v.f1 = eps1sq
        ex.launch[OP_AF_VEC](v, n)
    var d = Args()
    d.p0 = U
    d.p1 = sc
    d.i0 = n
    d.f0 = Float32(clip)
    if fast:
        d.p2 = parts
        d.i2 = _chunk_sumsq(ex, U, 0, n, parts, 0)
    elif blocked:
        d.p2 = parts
        d.i2 = _blk_sumsq(ex, U, n, parts)
    ex.launch[OP_AF_DENOM](d, 1)
    var ap = Args()
    ap.p0 = P
    ap.p1 = U
    ap.p2 = sc
    ex.launch[OP_AF_APPLY](ap, n)
