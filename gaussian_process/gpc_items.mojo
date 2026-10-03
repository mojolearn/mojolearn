# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Laplace Newton step's per-element statements and the marginal
likelihood's fold, ONE definition for the device kernels
(`gaussian_process/classifier.mojo`) and the host column
(`gaussian_process/host/gpc_steps.mojo`, which loops over these items)
(cpu-gpu-cleanup c-gp-kernel, 2026-10-02: the GPU binding ran every one of
them on the host between its device launches).

The statements are DEVIATION 2831's (gpc_steps.mojo). The likelihood's two
folds (a.f and the softplus sum) were serial ascending chains over all n;
they are now GPC_FOLD-row blocks, each its chain from zero ascending, the
block partials then added ascending (`gpc_lml_fin`). The same words on
every column."""
from checks.numerics import ftz, identical_mul, identical_mul_add, identical_sigmoid, identical_softplus, identical_sqrt

comptime _P = MutPointer[Float32, MutAnyOrigin]
#: rows per partial of the likelihood's folds
comptime GPC_FOLD = 2048


@always_inline
def gpc_fold_blocks(n: Int) -> Int:
    return (n + GPC_FOLD - 1) // GPC_FOLD if n > 0 else 0


@always_inline
def gpc_weight_item(i: Int, f: _P, pi: _P, w: _P, wsr: _P):
    """`pi = expit(f)`, `W = pi (1 - pi)`, `W_sr = sqrt(W)`."""
    var p = ftz(identical_sigmoid(ftz(f.unsafe_load(i))))
    var one_minus = ftz(Float32(1.0) - p)
    var wi = ftz(identical_mul(p, one_minus))
    pi.unsafe_store(i, p)
    w.unsafe_store(i, wi)
    wsr.unsafe_store(i, ftz(identical_sqrt(wi)))


@always_inline
def gpc_rhs_item(i: Int, w: _P, f: _P, y: _P, pi: _P, dst: _P):
    """`b = W f + (y - pi)`."""
    var wf = ftz(identical_mul(ftz(w.unsafe_load(i)), ftz(f.unsafe_load(i))))
    var r = ftz(ftz(y.unsafe_load(i)) - ftz(pi.unsafe_load(i)))
    dst.unsafe_store(i, ftz(wf + r))


@always_inline
def gpc_scale_item(i: Int, wsr: _P, v: _P, dst: _P):
    """`W_sr_i * v_i`."""
    dst.unsafe_store(i, ftz(identical_mul(ftz(wsr.unsafe_load(i)), ftz(v.unsafe_load(i)))))


@always_inline
def gpc_a_item(i: Int, b: _P, wsr: _P, x: _P, dst: _P):
    """`a = b - W_sr x`."""
    var s = ftz(identical_mul(ftz(wsr.unsafe_load(i)), ftz(x.unsafe_load(i))))
    dst.unsafe_store(i, ftz(ftz(b.unsafe_load(i)) - s))


@always_inline
def gpc_residual_item(i: Int, y: _P, pi: _P, dst: _P):
    """`y - pi`."""
    dst.unsafe_store(i, ftz(ftz(y.unsafe_load(i)) - ftz(pi.unsafe_load(i))))


def gpc_lml_part_item(blk: Int, a: _P, f: _P, y: _P, n: Int, pdot: _P, pt2: _P):
    """Block blk's a.f chain and softplus chain, from zero, rows ascending."""
    var lo = blk * GPC_FOLD
    var hi = min(lo + GPC_FOLD, n)
    var dot = Float32(0.0)
    var t2 = Float32(0.0)
    for i in range(lo, hi):
        dot = ftz(identical_mul_add(ftz(a.unsafe_load(i)), ftz(f.unsafe_load(i)), dot))
        # -(2y - 1) f is -f for y = 1 and +f for y = 0, an exact negation.
        var z = ftz(f.unsafe_load(i))
        if y.unsafe_load(i) == Float32(1.0):
            z = -z
        t2 = ftz(t2 + ftz(identical_softplus(z)))
    pdot.unsafe_store(blk, dot)
    pt2.unsafe_store(blk, t2)


def gpc_lml_fin(pdot: _P, pt2: _P, nb: Int, logdet_b: Float32) -> Float32:
    """The partials added ascending; `lml = ftz(ftz(t1 - t2) - t3)`."""
    var dot = Float32(0.0)
    var t2 = Float32(0.0)
    for b in range(nb):
        dot = ftz(dot + pdot.unsafe_load(b))
        t2 = ftz(t2 + pt2.unsafe_load(b))
    var t1 = ftz(identical_mul(Float32(-0.5), dot))
    var t3 = ftz(identical_mul(Float32(0.5), ftz(logdet_b)))
    return ftz(ftz(t1 - t2) - t3)


def gpr_ydot_part_item(blk: Int, y: _P, dual: _P, n: Int, part: _P):
    """GaussianProcessRegressor's `y^T alpha_` (lane/cgr-kernel): block
    blk's fused multiply-add chain from zero, rows ascending. Was one
    serial chain over all n on the host."""
    var lo = blk * GPC_FOLD
    var hi = min(lo + GPC_FOLD, n)
    var acc = Float32(0.0)
    for i in range(lo, hi):
        acc = ftz(identical_mul_add(ftz(y.unsafe_load(i)), ftz(dual.unsafe_load(i)), acc))
    part.unsafe_store(blk, acc)


def gpr_ydot_fin(part: _P, nb: Int) -> Float32:
    """The block partials added ascending."""
    var acc = Float32(0.0)
    for b in range(nb):
        acc = ftz(acc + part.unsafe_load(b))
    return ftz(acc)


def gpr_ydot_host(y: _P, dual: _P, n: Int) -> Float32:
    """The host column's `y^T alpha_`: the device's blocks and order."""
    var nb = gpc_fold_blocks(n)
    var parts = List[Float32](length=max(nb, 1), fill=Float32(0.0))
    var pp = _P(unsafe_from_address=Int(parts.unsafe_ptr()))
    for b in range(nb):
        gpr_ydot_part_item(b, y, dual, n, pp)
    var v = gpr_ydot_fin(pp, nb)
    _ = parts^
    return v
