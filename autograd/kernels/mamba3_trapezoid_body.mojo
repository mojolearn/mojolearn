# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Mamba-3 trapezoid recurrence's math, written ONCE (Andrew, 2026-11-22: one body per operation;
lane neural-blocks-3-m3).

The trapezoid rule's state update of the Mamba-3 SISO core, one state cell (p, n) of one head:

    adt   = A * dt                       alpha = exp(adt)
    sigma = sigmoid(trap)                gamma = sigma * dt
    beta  = ((1 - sigma) * dt) * alpha
    h[t]  = alpha h[t-1] + beta (k[t-1] v[t-1]) + gamma (k[t] v[t])     (three roundings, k*v first)
    y[t]  = sum_n q[t, n] h[t, n] (ascending fused chain) + D v[t]

  body                      callers
  m3_adt, m3_trap_sigma     m3_pre_kernel (the block's S8, mamba/impl/ops/mamba3_siso.mojo),
                            m3_step_core_kernel (the decode step, mamba/impl/modules/mamba3.mojo),
                            the generic op (m3t_scan_cell and the backward cells below)
  m3_trap_gamma,            m3_scale_kernel (S9: gamma and beta' over the working rows),
  m3_trap_beta_pre          m3_step_core_kernel, the generic op
  m3_trap_alpha,            m3_step_core_kernel, the generic op
  m3_trap_beta
  m3_trap_state             m3_step_core_kernel (the per-cell S update), the generic op
  m3_trap_readout,          m3_step_core_kernel, the generic op
  m3_trap_skip
  m3t_*_cell                the generic op's element bodies (autograd/ssm_m3_ops.mojo): one call per
                            element on the host column and one GPU thread per element on the device

The chunked prefill (S14-S20 of mamba3_siso.mojo) spells the same recurrence as a chunked contraction:
that is another summation order, recorded beside those kernels as this rule's exception (they keep
their math; the comparing check is mamba/checks/mamba3_check.mojo's prefix-vs-step relation and
checks/native_tests/mamba3_prefix_step.mojo, a forward equal to its prefix forward plus decode steps).
The generic op is the decode form over a whole sequence: one serial chain per (sequence, head, channel).

Moving these out of the kernels moved no float operation: each body is the kernel's former text (the
products are commutative, so the scale kernel's dt * sigma and the step's sigma * dt are one body).

SABOTAGE: MOJOLEARN_MAMBA3_SABOTAGE_TRAP_LEFT_ONLY (Euler, no trapezoid) is read here as well as at
S15's consumption site: the state update drops the beta leg. The armed generic op differs from its
pinned digest (checks/native_tests/neural_blocks.mojo, trapezoid_scan cases).
"""
from std.sys.compile import is_defined
from checks.numerics import ftz, identical_exp, identical_mul, identical_mul_add, identical_sigmoid

comptime M3P = MutPointer[Float32, MutAnyOrigin]

comptime SAB3_TRAP_LEFT_ONLY = is_defined["MOJOLEARN_MAMBA3_SABOTAGE_TRAP_LEFT_ONLY"]()


@always_inline
def m3_adt(a: Float32, dt: Float32) -> Float32:
    """ADT = A * dt (S8)."""
    return ftz(identical_mul(ftz(a), ftz(dt)))


@always_inline
def m3_trap_sigma(raw: Float32) -> Float32:
    """sigma = sigmoid(trap) (S8)."""
    return ftz(identical_sigmoid(ftz(raw)))


@always_inline
def m3_trap_gamma(dt: Float32, sg: Float32) -> Float32:
    """gamma = dt * sigma (S9)."""
    return ftz(identical_mul(ftz(dt), ftz(sg)))


@always_inline
def m3_trap_beta_pre(dt: Float32, sg: Float32) -> Float32:
    """beta' = dt * (1 - sigma) (S9; the decode step's (1 - sigma) * dt)."""
    return ftz(identical_mul(ftz(dt), ftz(Float32(1.0) - ftz(sg))))


@always_inline
def m3_trap_alpha(adt: Float32) -> Float32:
    return ftz(identical_exp(adt))


@always_inline
def m3_trap_beta(beta_pre: Float32, al: Float32) -> Float32:
    """beta = beta' * alpha, torch's left association."""
    return ftz(identical_mul(beta_pre, al))


@always_inline
def m3_trap_state(
    al: Float32, be: Float32, ga: Float32, h: Float32, kst: Float32, vst: Float32, kn: Float32, vv: Float32,
) -> Float32:
    """S = alpha*S + beta*(K_st*V_st) + gamma*(k*v): three separate roundings, k*v product first inside
    each (the decode step's :125-127)."""
    var s1 = ftz(identical_mul(al, ftz(h)))
    var s2: Float32
    comptime if SAB3_TRAP_LEFT_ONLY:
        # SABOTAGE (Euler): the beta leg is dropped
        s2 = s1
    else:
        s2 = ftz(s1 + ftz(identical_mul(be, ftz(identical_mul(kst, vst)))))
    return ftz(s2 + ftz(identical_mul(ga, ftz(identical_mul(kn, vv)))))


@always_inline
def m3_trap_readout(s: Float32, qn: Float32, acc: Float32) -> Float32:
    """One term of the readout after the update, ascending n."""
    return ftz(identical_mul_add(s, qn, acc))


@always_inline
def m3_trap_skip(out: Float32, d_val: Float32, vv: Float32) -> Float32:
    """y = out + D * v, D last."""
    return ftz(out + ftz(identical_mul(ftz(d_val), vv)))


# ---------------------------------------------------------------- the generic op's element bodies
# Layout: rows M = batch * length (row bb * length + t); v [M, H*P], k and q [M, H*N], dt, a, trap [M, H],
# D [H]; hs [M, H, P, N] the state after each token; g [M, H, P, N] the state's gradient.


@always_inline
def _coef(dt: M3P, a: M3P, tr: M3P, idx: Int) -> Tuple[Float32, Float32, Float32]:
    """(alpha, beta, gamma) of row-head idx."""
    var dtv = ftz(dt.unsafe_load(idx))
    var al = m3_trap_alpha(m3_adt(a.unsafe_load(idx), dtv))
    var sg = m3_trap_sigma(tr.unsafe_load(idx))
    return (al, m3_trap_beta(m3_trap_beta_pre(dtv, sg), al), m3_trap_gamma(dtv, sg))


@always_inline
def m3t_scan_cell(
    cell: Int, y: M3P, hs: M3P, v: M3P, k: M3P, q: M3P, dt: M3P, a: M3P, tr: M3P, dsk: M3P,
    batch: Int, length: Int, nh: Int, pd: Int, ns: Int,
):
    """The forward of one (sequence, head, channel) chain: every token's state row and output."""
    if cell >= batch * nh * pd:
        return
    var bb = cell // (nh * pd)
    var rem = cell - bb * nh * pd
    var hh = rem // pd
    var p = rem - hh * pd
    var dval = dsk.unsafe_load(hh)
    for t in range(length):  # serial by nature: the recurrence
        var mm = bb * length + t
        var c = _coef(dt, a, tr, mm * nh + hh)
        var vv = ftz(v.unsafe_load((mm * nh + hh) * pd + p))
        var vst = Float32(0.0)
        if t > 0:
            vst = ftz(v.unsafe_load(((mm - 1) * nh + hh) * pd + p))
        var out = Float32(0.0)
        var hrow = ((mm * nh + hh) * pd + p) * ns
        var prow = (((mm - 1) * nh + hh) * pd + p) * ns
        for n in range(ns):
            var kn = ftz(k.unsafe_load((mm * nh + hh) * ns + n))
            var qn = ftz(q.unsafe_load((mm * nh + hh) * ns + n))
            var kst = Float32(0.0)
            var hp = Float32(0.0)
            if t > 0:
                kst = ftz(k.unsafe_load(((mm - 1) * nh + hh) * ns + n))
                hp = hs.unsafe_load(prow + n)
            var s3 = m3_trap_state(c[0], c[1], c[2], hp, kst, vst, kn, vv)
            hs.unsafe_store(hrow + n, s3)
            out = m3_trap_readout(s3, qn, out)
        y.unsafe_store((mm * nh + hh) * pd + p, m3_trap_skip(ftz(out), dval, vv))


@always_inline
def m3t_dkv(ga: Float32, gn: Float32, be_next: Float32, g_next: Float32, last: Bool) -> Float32:
    """The gradient of the product k[t] v[t]: gamma[t] g[t] + beta[t+1] g[t+1]."""
    var d = ftz(identical_mul(ga, gn))
    if not last:
        d = ftz(d + ftz(identical_mul(be_next, g_next)))
    return d


@always_inline
def m3t_bwd_cell(
    cell: Int, g: M3P, dv_s: M3P, wal: M3P, wbe: M3P, wga: M3P, dy: M3P, hs: M3P, v: M3P, k: M3P, q: M3P,
    dt: M3P, a: M3P, tr: M3P, batch: Int, length: Int, nh: Int, pd: Int, ns: Int,
):
    """The reverse chain of one (sequence, head, channel): the state gradient g at every token, v's
    scan gradient, and the per-channel terms of d alpha, d beta, d gamma (folded over the channels by
    m3t_coef_cell)."""
    if cell >= batch * nh * pd:
        return
    var bb = cell // (nh * pd)
    var rem = cell - bb * nh * pd
    var hh = rem // pd
    var p = rem - hh * pd
    var t = length - 1
    while t >= 0:  # serial by nature: the reverse recurrence
        var mm = bb * length + t
        var last = t + 1 >= length
        var c = _coef(dt, a, tr, mm * nh + hh)
        var al_next = Float32(0.0)
        var be_next = Float32(0.0)
        if not last:
            var cn = _coef(dt, a, tr, (mm + 1) * nh + hh)
            al_next = cn[0]
            be_next = cn[1]
        var row = (mm * nh + hh) * pd + p
        var dyv = ftz(dy.unsafe_load(row))
        var vv = ftz(v.unsafe_load(row))
        var vst = Float32(0.0)
        if t > 0:
            vst = ftz(v.unsafe_load(((mm - 1) * nh + hh) * pd + p))
        var sdv = Float32(0.0)
        var sal = Float32(0.0)
        var sbe = Float32(0.0)
        var sga = Float32(0.0)
        for n in range(ns):
            var kn = ftz(k.unsafe_load((mm * nh + hh) * ns + n))
            var gn = ftz(identical_mul(dyv, ftz(q.unsafe_load((mm * nh + hh) * ns + n))))
            var g_next = Float32(0.0)
            if not last:
                g_next = g.unsafe_load(((((mm + 1) * nh + hh) * pd + p) * ns) + n)
                gn = ftz(gn + ftz(identical_mul(al_next, g_next)))
            g.unsafe_store(row * ns + n, gn)
            sdv = ftz(identical_mul_add(m3t_dkv(c[2], gn, be_next, g_next, last), kn, sdv))
            if t > 0:
                var kst = ftz(k.unsafe_load(((mm - 1) * nh + hh) * ns + n))
                sal = ftz(identical_mul_add(gn, ftz(hs.unsafe_load(((((mm - 1) * nh + hh) * pd + p) * ns) + n)), sal))
                sbe = ftz(identical_mul_add(gn, ftz(identical_mul(kst, vst)), sbe))
            sga = ftz(identical_mul_add(gn, ftz(identical_mul(kn, vv)), sga))
        dv_s.unsafe_store(row, sdv)
        wal.unsafe_store(row, sal)
        wbe.unsafe_store(row, sbe)
        wga.unsafe_store(row, sga)
        t -= 1


@always_inline
def m3t_dq_cell(cell: Int, dq: M3P, dy: M3P, hs: M3P, m: Int, nh: Int, pd: Int, ns: Int):
    """dq[row, h, n] = sum_p dy[row, h, p] h[row, h, p, n], ascending p."""
    if cell >= m * nh * ns:
        return
    var rh = cell // ns
    var n = cell - rh * ns
    var acc = Float32(0.0)
    for p in range(pd):
        acc = ftz(identical_mul_add(ftz(dy.unsafe_load(rh * pd + p)), ftz(hs.unsafe_load((rh * pd + p) * ns + n)), acc))
    dq.unsafe_store(cell, acc)


@always_inline
def m3t_dk_cell(
    cell: Int, dk: M3P, g: M3P, v: M3P, dt: M3P, a: M3P, tr: M3P, length: Int, m: Int, nh: Int, pd: Int, ns: Int,
):
    """dk[row, h, n] = sum_p dkv[row, h, p, n] v[row, h, p], ascending p."""
    if cell >= m * nh * ns:
        return
    var rh = cell // ns
    var n = cell - rh * ns
    var mm = rh // nh
    var hh = rh - mm * nh
    var last = (mm % length) + 1 >= length
    var c = _coef(dt, a, tr, rh)
    var be_next = Float32(0.0)
    if not last:
        be_next = _coef(dt, a, tr, rh + nh)[1]
    var acc = Float32(0.0)
    for p in range(pd):
        var gn = g.unsafe_load((rh * pd + p) * ns + n)
        var g_next = Float32(0.0)
        if not last:
            g_next = g.unsafe_load(((rh + nh) * pd + p) * ns + n)
        acc = ftz(identical_mul_add(m3t_dkv(c[2], gn, be_next, g_next, last), ftz(v.unsafe_load(rh * pd + p)), acc))
    dk.unsafe_store(cell, acc)
    _ = hh


@always_inline
def m3t_coef_cell(
    cell: Int, ddt: M3P, da: M3P, dtr: M3P, wal: M3P, wbe: M3P, wga: M3P, dt: M3P, a: M3P, tr: M3P,
    m: Int, nh: Int, pd: Int,
):
    """The coefficients' chain for one (row, head): fold the channel terms (ascending p), then
    alpha = exp(A dt), beta = dt (1 - sigma) alpha, gamma = dt sigma, sigma = sigmoid(trap)."""
    if cell >= m * nh:
        return
    var dal = Float32(0.0)
    var dbe = Float32(0.0)
    var dga = Float32(0.0)
    for p in range(pd):
        dal = ftz(dal + ftz(wal.unsafe_load(cell * pd + p)))
        dbe = ftz(dbe + ftz(wbe.unsafe_load(cell * pd + p)))
        dga = ftz(dga + ftz(wga.unsafe_load(cell * pd + p)))
    var dtv = ftz(dt.unsafe_load(cell))
    var av = ftz(a.unsafe_load(cell))
    var al = m3_trap_alpha(m3_adt(av, dtv))
    var sg = m3_trap_sigma(tr.unsafe_load(cell))
    var bp = m3_trap_beta_pre(dtv, sg)
    var be = m3_trap_beta(bp, al)
    # d adt = d alpha * alpha + d beta * beta (beta is beta' alpha)
    var dadt = ftz(ftz(identical_mul(dal, al)) + ftz(identical_mul(dbe, be)))
    var dbp = ftz(identical_mul(dbe, al))
    var one_m = ftz(Float32(1.0) - sg)
    var dd = ftz(identical_mul(dadt, av))
    dd = ftz(dd + ftz(identical_mul(dbp, one_m)))
    dd = ftz(dd + ftz(identical_mul(dga, sg)))
    ddt.unsafe_store(cell, dd)
    da.unsafe_store(cell, ftz(identical_mul(dadt, dtv)))
    var dsg = ftz(ftz(identical_mul(dga, dtv)) - ftz(identical_mul(dbp, dtv)))
    dtr.unsafe_store(cell, ftz(identical_mul(ftz(identical_mul(dsg, sg)), one_m)))


@always_inline
def m3t_dv_cell(cell: Int, dv: M3P, dv_s: M3P, dy: M3P, dsk: M3P, m: Int, nh: Int, pd: Int):
    """v's whole gradient: the D skip term first, then the scan term."""
    if cell >= m * nh * pd:
        return
    var hh = (cell // pd) % nh
    var sk = ftz(identical_mul(ftz(dy.unsafe_load(cell)), ftz(dsk.unsafe_load(hh))))
    dv.unsafe_store(cell, ftz(sk + ftz(dv_s.unsafe_load(cell))))


@always_inline
def m3t_dd_cell(cell: Int, dd: M3P, dy: M3P, v: M3P, m: Int, nh: Int, pd: Int):
    """dD[h] = sum over rows then channels of dy v, ascending."""
    if cell >= nh:
        return
    var acc = Float32(0.0)
    for mm in range(m):
        for p in range(pd):
            var i = (mm * nh + cell) * pd + p
            acc = ftz(identical_mul_add(ftz(dy.unsafe_load(i)), ftz(v.unsafe_load(i)), acc))
    dd.unsafe_store(cell, acc)
