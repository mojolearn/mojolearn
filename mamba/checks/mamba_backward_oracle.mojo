# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host Float32 oracle for the pinned Mamba-1 backward profile."""

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_mul_add,
    identical_rsqrt,
    identical_sigmoid,
    identical_silu,
)
from gemm.checks.gemm_oracle import OP_NN, OP_TN, gemm_oracle
from mamba.checks.mamba_fixture import (
    D_CONV,
    D_STATE,
    MambaDims,
    MambaWeights,
    RMS_EPS,
)
from mamba.checks.mamba_oracle import (
    MambaStages,
    MambaState,
    pinned_mul,
    refuse_nonfinite,
)



comptime BWD_KIND_TOKEN = 0
comptime BWD_KIND_BATCH = 1
comptime BWD_KIND_GLOBAL = 2

comptime BWD_STAGE_COUNT = 26
comptime BWD_ACTIVATION_STAGES = 16


def mamba_backward_stage_names() -> List[String]:
    """The twenty six stage names, card order, plan section 7 verbatim. A future `mamba_backward_check.mojo` must IMPORT this, never restate it."""
    var n: List[String] = [
        String("bwd.dres"),
        String("bwd.dg"),
        String("bwd.dsk"),
        String("bwd.dz"),
        String("bwd.dh"),
        String("bwd.dCm"),
        String("bwd.dBm"),
        String("bwd.du_s"),
        String("bwd.ddelta"),
        String("bwd.ddtp"),
        String("bwd.du"),
        String("bwd.dconv"),
        String("bwd.dhin"),
        String("bwd.dnrm"),
        String("bwd.drstd"),
        String("bwd.dx"),
        String("bwd.dW_in"),
        String("bwd.dW_x"),
        String("bwd.dW_dt"),
        String("bwd.dW_out"),
        String("bwd.dA_log"),
        String("bwd.dD"),
        String("bwd.db_dt"),
        String("bwd.dcw"),
        String("bwd.dcb"),
        String("bwd.dw_norm"),
    ]
    return n^


def mamba_backward_stage_kind(i: Int) -> Int:
    """TOKEN for the sixteen activation gradients, GLOBAL for the ten parameter gradients."""
    if i < BWD_ACTIVATION_STAGES:
        return BWD_KIND_TOKEN
    return BWD_KIND_GLOBAL


def mamba_backward_stage_width(i: Int, dims: MambaDims) -> Int:
    """Cells per token for a TOKEN stage, total cells for a GLOBAL one."""
    var dm = dims.d_model
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    if i == 0:
        return dm  # bwd.dres  [M, dm]
    if i == 1 or i == 2 or i == 3:
        return di  # bwd.dg, bwd.dsk, bwd.dz  [M, di]
    if i == 4:
        return di * D_STATE  # bwd.dh  [M, di, N]
    if i == 5 or i == 6:
        return D_STATE  # bwd.dCm, bwd.dBm  [M, N]
    if i >= 7 and i <= 12:
        return di  # du_s, ddelta, ddtp, du, dconv, dhin  [M, di]
    if i == 13:
        return dm  # bwd.dnrm  [M, dm]
    if i == 14:
        return 1  # bwd.drstd  [M]
    if i == 15:
        return dm  # bwd.dx  [M, dm]
    if i == 16:
        return 2 * di * dm  # bwd.dW_in   [2di, dm]
    if i == 17:
        return xr * di  # bwd.dW_x    [r+2N, di]
    if i == 18:
        return di * r  # bwd.dW_dt   [di, r]
    if i == 19:
        return dm * di  # bwd.dW_out  [dm, di]
    if i == 20:
        return di * D_STATE  # bwd.dA_log  [di, N]
    if i == 21 or i == 22 or i == 24:
        return di  # bwd.dD, bwd.db_dt, bwd.dcb  [di]
    if i == 23:
        return di * D_CONV  # bwd.dcw  [di, K]
    return dm  # bwd.dw_norm  [dm]


def mamba_backward_free_choices() -> String:
    """The seven decisions this file MADE because the plan does not state them, each with what would falsify it."""
    return (
        String("mojolearn.identical.mamba1.bwd.fp32.v1 FREE CHOICES")
        + " (7, none derived, none measured)\n"
        + "FC1 B7 silu' middle term `1 + v*(1-sig)` is ONE"
        + " identical_mul_add. The plan gives the expression and the count"
        + " (3 roundings) but never the fusion; only the fused middle has"
        + " both. FALSIFIED BY: a device spelling it unfused (4 roundings),"
        + " or the plan amending the count.\n"
        + "FC2 B18 ddelta is TWO separate ascending-n folds (B path, then A"
        + " path), each seeded +0.0, joined by ONE unfused flushed add with"
        + " ddelta_B LEFT. Plan section 2 (S7', S5', S14') spells exactly"
        + " this; 3.2's B18 says `interleaving pinned` and never says what"
        + " the interleaving is. FALSIFIED BY: a SAB_BWD_DDELTA_INTERLEAVED"
        + " arm folding both terms in one ascending-n chain (upstream's"
        + " shape, bwd_kernel `ddelta_vals[i] += ddelta_u*u + dx*A*a`)"
        + " agreeing with the device and not with this file.\n"
        + "FC3 the T2 h checkpoint is [B, L+1, di, N] TOKEN-MAJOR, slot t+1"
        + " holding h[t] and slot 0 the flushed entry state."
        + " t2_h_checkpoint_floats writes the SIZE as b*di*(l+1)*N, whose"
        + " factor order reads as [B, di, L+1, N]. Layout moves no bits but"
        + " the card compares per cell. FALSIFIED BY: a device writing the"
        + " other order (a permuted, not a moved, checkpoint).\n"
        + "FC4 bwd.dres is recorded as a RAW COPY, unflushed, and flushed at"
        + " each consuming seam -- the forward's treatment of conv.window."
        + " FALSIFIED BY: a device that stores ftz(dres) and differs on a"
        + " denormal dres.\n"
        + "FC5 T5's batch fold is applied to bwd.dA_log ONLY. Plan 3.4 T5"
        + " says it is `shared by dA, dD, db_dt, dcw and dcb`, but 3.2,"
        + " 4.1 and mamba_backward.mojo's eight RED_* ids route those four"
        + " (and dw_norm) to ones-vector v1 GEMMs at k' = M, which subsume"
        + " the batch axis inside v1's own partition. The two answers"
        + " DIFFER whenever M > 128. FALSIFIED BY: a device computing dD by"
        + " private slots plus a b-fold. THIS IS THE ONE FREE CHOICE THAT"
        + " RESOLVES A CONTRADICTION IN THE PLAN RATHER THAN A SILENCE.\n"
        + "FC6 above the softplus guard (biased > 20) ddtp is a flushed COPY"
        + " of ddelta rather than pinned_mul(ddelta, 1.0). Provably"
        + " bit-inert: fma(x, 1, -0.0) == x at every x including both signed"
        + " zeros. Stated because inert is a claim, not an omission.\n"
        + "FC7 d_arg is RECOMPUTED at both its consumers (ddelta's A path"
        + " and T4's dA fold) rather than stored. A host memory choice, not"
        + " an arithmetic one: the expression is pure and deterministic."
        + " FALSIFIED BY: nothing bitwise; it is a note so a reader does not"
        + " read the second spelling as a second answer.\n"
    )




struct MambaBackwardStages(Movable):
    """Every recorded stage of one backward call, in the card's order. `dh` IS a card stage (`bwd.dh`, `[M, di, N]`, plan section 7); `h_ckpt` is T2's `[B, L+1, di, N]` checkpoint, which the DEVICE gets from the forward and which the host must rebuild, so a gate that wants to price MB9 (upstream's `h[t] - dbu[t]` recovery) has the numbers to price it with."""

    var dres: List[Float32]  # [M, d_model]        B1
    var dg: List[Float32]  # [M, d_inner]          B2
    var dsk: List[Float32]  # [M, d_inner]         B5
    var dz: List[Float32]  # [M, d_inner]          B8
    var dh: List[Float32]  # [M, d_inner, N]       B13, T1
    var dcm: List[Float32]  # [M, N]               B15, T3
    var dbm: List[Float32]  # [M, N]               B16, T3
    var du_s: List[Float32]  # [M, d_inner]        B17
    var ddelta: List[Float32]  # [M, d_inner]      B18
    var ddtp: List[Float32]  # [M, d_inner]        B22
    var du: List[Float32]  # [M, d_inner]          B29, T6
    var dconv: List[Float32]  # [M, d_inner]       B30
    var dhin: List[Float32]  # [M, d_inner]        B31
    var dnrm: List[Float32]  # [M, d_model]        B35
    var drstd: List[Float32]  # [M]                B40
    var dx: List[Float32]  # [M, d_model]          B41, B42, T7, T8
    var dw_in: List[Float32]  # [2*d_inner, d_model]        B36
    var dw_x: List[Float32]  # [r+2N, d_inner]              B28
    var dw_dt: List[Float32]  # [d_inner, r]                B25
    var dw_out: List[Float32]  # [d_model, d_inner]         B3
    var da_log: List[Float32]  # [d_inner, N]               B19, B21, T4, T5
    var dd_skip: List[Float32]  # [d_inner]                 B12
    var db_dt: List[Float32]  # [d_inner]                   B23
    var dcw: List[Float32]  # [d_inner, D_CONV]             B32
    var dcb: List[Float32]  # [d_inner]                     B33
    var dw_norm: List[Float32]  # [d_model]                 B39
    var h_ckpt: List[Float32]  # [B, L+1, d_inner, N]       T2, FREE CHOICE 3

    def __init__(out self):
        self.dres = List[Float32]()
        self.dg = List[Float32]()
        self.dsk = List[Float32]()
        self.dz = List[Float32]()
        self.dh = List[Float32]()
        self.dcm = List[Float32]()
        self.dbm = List[Float32]()
        self.du_s = List[Float32]()
        self.ddelta = List[Float32]()
        self.ddtp = List[Float32]()
        self.du = List[Float32]()
        self.dconv = List[Float32]()
        self.dhin = List[Float32]()
        self.dnrm = List[Float32]()
        self.drstd = List[Float32]()
        self.dx = List[Float32]()
        self.dw_in = List[Float32]()
        self.dw_x = List[Float32]()
        self.dw_dt = List[Float32]()
        self.dw_out = List[Float32]()
        self.da_log = List[Float32]()
        self.dd_skip = List[Float32]()
        self.db_dt = List[Float32]()
        self.dcw = List[Float32]()
        self.dcb = List[Float32]()
        self.dw_norm = List[Float32]()
        self.h_ckpt = List[Float32]()


def backward_oracle_dump(st: MambaBackwardStages) -> List[List[Float32]]:
    """The oracle's stages, card order, matching `mamba_backward_stage_names`."""
    var out = List[List[Float32]]()
    out.append(st.dres.copy())
    out.append(st.dg.copy())
    out.append(st.dsk.copy())
    out.append(st.dz.copy())
    out.append(st.dh.copy())
    out.append(st.dcm.copy())
    out.append(st.dbm.copy())
    out.append(st.du_s.copy())
    out.append(st.ddelta.copy())
    out.append(st.ddtp.copy())
    out.append(st.du.copy())
    out.append(st.dconv.copy())
    out.append(st.dhin.copy())
    out.append(st.dnrm.copy())
    out.append(st.drstd.copy())
    out.append(st.dx.copy())
    out.append(st.dw_in.copy())
    out.append(st.dw_x.copy())
    out.append(st.dw_dt.copy())
    out.append(st.dw_out.copy())
    out.append(st.da_log.copy())
    out.append(st.dd_skip.copy())
    out.append(st.db_dt.copy())
    out.append(st.dcw.copy())
    out.append(st.dcb.copy())
    out.append(st.dw_norm.copy())
    return out^




def _zeros(n: Int) -> List[Float32]:
    var xs = List[Float32]()
    for _ in range(n):
        xs.append(Float32(0.0))
    return xs^


def pinned_silu_prime(v: Float32) -> Float32:
    """`silu'(v) = sig(v) * (1 + v*(1 - sig(v)))`, THREE roundings."""
    var sig = ftz(identical_sigmoid(v))
    var one_minus = ftz(Float32(1.0) - sig)
    var mid = ftz(identical_mul_add(v, one_minus, Float32(1.0)))
    return ftz(pinned_mul(sig, mid))


def _forward_rstd(sumsq: Float32, dm: Int) -> Float32:
    """`rstd` RECOMPUTED with the forward's own spelling. Plan section 6's general rule -- *a recomputed forward quantity must be spelled with the same function the forward used, not with an algebraically equal one* -- is the only reason this is a function and not two inline lines."""
    var mean = ftz(identical_div(sumsq, Float32(dm)))
    return ftz(identical_rsqrt(ftz(mean + RMS_EPS)))


def _forward_da(delta: Float32, a: Float32) -> Float32:
    """`da[t,d,n] = exp(delta[t,d] * A[d,n])`, seams S5 and S6, RECOMPUTED. Upstream's forward and backward both use the exp2 substitution and are self-consistent; DEVIATION 722 already refused it as a different function with an extra rounding, and plan section 6 item 6 says the backward's recomputed `da` must be OUR forward's `da` or it is not a recomputation."""
    return ftz(identical_exp(ftz(pinned_mul(delta, a))))


def _forward_dbb(delta: Float32, bm: Float32) -> Float32:
    """`dbb[t,d,n] = delta[t,d] * Bm[t,n]`, seam S7's `db`, RECOMPUTED."""
    return ftz(pinned_mul(delta, bm))




def mamba_h_checkpoint_oracle(
    u: List[Float32],  # [M, d_inner]  silu.out
    delta: List[Float32],  # [M, d_inner]  softplus.out
    a: List[Float32],  # [d_inner, D_STATE]  A.out
    bmat: List[Float32],  # [M, D_STATE]
    h_in: List[Float32],  # [B, d_inner, D_STATE]  the state ENTERING
    b: Int,
    l: Int,
    d_inner: Int,
) -> List[Float32]:
    """`h[t, d, n]` for every `t` in `[-1, L)`, into `[B, L+1, di, N]`. PINNED AS AN EXPLICIT CHECKPOINT.** The REFUSED alternative is upstream's `a = h[t] - dbu[t]` (`selective_scan_bwd_kernel.cuh:290`)."""
    var out = _zeros(b * (l + 1) * d_inner * D_STATE)
    for bb in range(b):
        for d in range(d_inner):
            var h = List[Float32]()
            for n in range(D_STATE):
                h.append(ftz(h_in[(bb * d_inner + d) * D_STATE + n]))
            for n in range(D_STATE):
                out[((bb * (l + 1) + 0) * d_inner + d) * D_STATE + n] = h[n]
            for li in range(l):
                var t = bb * l + li
                var uv = ftz(u[t * d_inner + d])
                var dl = ftz(delta[t * d_inner + d])
                for n in range(D_STATE):
                    var da = _forward_da(dl, ftz(a[d * D_STATE + n]))
                    var dbb = _forward_dbb(dl, ftz(bmat[t * D_STATE + n]))
                    var dbu = ftz(pinned_mul(dbb, uv))
                    h[n] = ftz(identical_mul_add(da, h[n], dbu))
                for n in range(D_STATE):
                    out[
                        ((bb * (l + 1) + li + 1) * d_inner + d) * D_STATE + n
                    ] = h[n]
    return out^


def _h_at(
    ck: List[Float32], bb: Int, li: Int, d: Int, l: Int, di: Int, n: Int
) -> Float32:
    """`h[li, d, n]` out of the checkpoint. `li = -1` is the entry state."""
    return ck[((bb * (l + 1) + li + 1) * di + d) * D_STATE + n]




def mamba_block_backward_oracle(
    w: MambaWeights,
    x: List[Float32],  # [B, L, d_model] token-major, the FORWARD's input
    dres: List[Float32],  # [M, d_model] the gradient arriving at the block
    st: MambaStages,  # the FORWARD's recorded stages, SAME call
    state_in: MambaState,  # the state ENTERING the forward call
    b: Int,
    l: Int,
) raises -> MambaBackwardStages:
    """One `MambaBlock.forward`'s backward, stage by stage, through the seams. `st` must be the stages the FORWARD oracle recorded for exactly this `(w, x, b, l, state_in)`."""
    refuse_nonfinite("dres", dres)  # plan section 10 item 8, now written
    refuse_nonfinite("state.conv_win", state_in.conv_win)
    refuse_nonfinite("state.h", state_in.h)

    var dims = w.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    var m = b * l
    var bst = MambaBackwardStages()


    var dt_low = List[Float32]()
    var bmat = List[Float32]()
    var cmat = List[Float32]()
    for t in range(m):
        for j in range(r):
            dt_low.append(st.x_proj[t * xr + j])
        for n in range(D_STATE):
            bmat.append(st.x_proj[t * xr + r + n])
        for n in range(D_STATE):
            cmat.append(st.x_proj[t * xr + r + D_STATE + n])

    bst.h_ckpt = mamba_h_checkpoint_oracle(
        st.silu_out, st.softplus_out, st.a_out, bmat, state_in.h, b, l, di
    )
    from std.memory import bitcast

    for bb in range(b):
        for d in range(di):
            for n in range(D_STATE):
                var got = _h_at(bst.h_ckpt, bb, l - 1, d, l, di, n)
                var want = st.scan_h[(bb * di + d) * D_STATE + n]
                if bitcast[DType.uint32](got) != bitcast[DType.uint32](want):
                    raise Error(
                        String("mamba backward oracle: the T2 checkpoint's")
                        + " last slot is not the forward's scan.h at (b="
                        + String(bb)
                        + ", d="
                        + String(d)
                        + ", n="
                        + String(n)
                        + "). The backward is differentiating a DIFFERENT"
                        + " forward than the one that ran."
                    )

    for i in range(m * dm):
        bst.dres.append(dres[i])

    bst.dg = gemm_oracle(bst.dres, w.w_out, OP_NN, m, di, dm)

    bst.dw_out = gemm_oracle(bst.dres, st.gate_out, OP_TN, dm, di, m)

    var silu_z = _zeros(m * di)
    for t in range(m):
        for d in range(di):
            var z = ftz(st.in_proj[t * 2 * di + di + d])
            silu_z[t * di + d] = ftz(identical_silu(z))
            bst.dsk.append(
                ftz(pinned_mul(ftz(bst.dg[t * di + d]), silu_z[t * di + d]))
            )

    for t in range(m):
        for d in range(di):
            var z = ftz(st.in_proj[t * 2 * di + di + d])
            var p1 = ftz(
                pinned_mul(
                    ftz(bst.dg[t * di + d]), ftz(st.skip_out[t * di + d])
                )
            )
            bst.dz.append(ftz(pinned_mul(p1, pinned_silu_prime(z))))

    bst.dh = _zeros(m * di * D_STATE)
    for bb in range(b):
        for d in range(di):
            var dhn = _zeros(D_STATE)  # dh[t+1], T1's +0.0 seed
            var dan = _zeros(D_STATE)  # da[t+1], +0.0 at t = L-1
            for li in range(l - 1, -1, -1):
                var t = bb * l + li
                var dyv = ftz(bst.dsk[t * di + d])
                for n in range(D_STATE):
                    var contrib = ftz(
                        pinned_mul(dyv, ftz(cmat[t * D_STATE + n]))
                    )
                    var v = ftz(identical_mul_add(dan[n], dhn[n], contrib))
                    bst.dh[(t * di + d) * D_STATE + n] = v
                    dhn[n] = v
                var dl = ftz(st.softplus_out[t * di + d])
                for n in range(D_STATE):
                    dan[n] = _forward_da(dl, ftz(st.a_out[d * D_STATE + n]))

    for t in range(m):
        var bb = t // l
        var li = t % l
        var dy_row = List[Float32]()
        var w_row = List[Float32]()
        for d in range(di):
            dy_row.append(bst.dsk[t * di + d])
            w_row.append(
                ftz(
                    pinned_mul(
                        ftz(st.softplus_out[t * di + d]),
                        ftz(st.silu_out[t * di + d]),
                    )
                )
            )
        var h_slab = List[Float32]()
        var dh_slab = List[Float32]()
        for d in range(di):
            for n in range(D_STATE):
                h_slab.append(_h_at(bst.h_ckpt, bb, li, d, l, di, n))
                dh_slab.append(bst.dh[(t * di + d) * D_STATE + n])
        var cm_row = gemm_oracle(dy_row, h_slab, OP_NN, 1, D_STATE, di)
        var bm_row = gemm_oracle(w_row, dh_slab, OP_NN, 1, D_STATE, di)
        for n in range(D_STATE):
            bst.dcm.append(cm_row[n])
            bst.dbm.append(bm_row[n])

    for t in range(m):
        for d in range(di):
            var dl = ftz(st.softplus_out[t * di + d])
            var acc = Float32(0.0)
            for n in range(D_STATE):
                var dbb = _forward_dbb(dl, ftz(bmat[t * D_STATE + n]))
                acc = ftz(
                    identical_mul_add(
                        ftz(bst.dh[(t * di + d) * D_STATE + n]), dbb, acc
                    )
                )
            bst.du_s.append(acc)

    for t in range(m):
        var bb = t // l
        var li = t % l
        for d in range(di):
            var dl = ftz(st.softplus_out[t * di + d])
            var uv = ftz(st.silu_out[t * di + d])
            var acc_b = Float32(0.0)
            for n in range(D_STATE):
                var d_dbb = ftz(
                    pinned_mul(ftz(bst.dh[(t * di + d) * D_STATE + n]), uv)
                )
                acc_b = ftz(
                    identical_mul_add(
                        d_dbb, ftz(bmat[t * D_STATE + n]), acc_b
                    )
                )
            var acc_a = Float32(0.0)
            for n in range(D_STATE):
                var av = ftz(st.a_out[d * D_STATE + n])
                var d_da = ftz(
                    pinned_mul(
                        ftz(bst.dh[(t * di + d) * D_STATE + n]),
                        ftz(_h_at(bst.h_ckpt, bb, li - 1, d, l, di, n)),
                    )
                )
                var d_arg = ftz(pinned_mul(d_da, _forward_da(dl, av)))
                acc_a = ftz(identical_mul_add(d_arg, av, acc_a))
            bst.ddelta.append(ftz(acc_b + acc_a))

    for t in range(m):
        for d in range(di):
            var biased = ftz(st.dt_proj[t * di + d] + ftz(w.b_dt[d]))
            var g = ftz(bst.ddelta[t * di + d])
            if biased <= Float32(20.0):
                bst.ddtp.append(
                    ftz(pinned_mul(g, ftz(identical_sigmoid(biased))))
                )
            else:
                bst.ddtp.append(g)

    var ddtl = gemm_oracle(bst.ddtp, w.w_dt, OP_NN, m, r, di)
    bst.dw_dt = gemm_oracle(bst.ddtp, dt_low, OP_TN, di, r, m)
    var dxp = List[Float32]()
    for t in range(m):
        for j in range(r):
            dxp.append(ddtl[t * r + j])
        for n in range(D_STATE):
            dxp.append(bst.dbm[t * D_STATE + n])
        for n in range(D_STATE):
            dxp.append(bst.dcm[t * D_STATE + n])

    var du_x = gemm_oracle(dxp, w.w_x, OP_NN, m, di, xr)
    bst.dw_x = gemm_oracle(dxp, st.silu_out, OP_TN, xr, di, m)

    var du_d = _zeros(m * di)
    for t in range(m):
        for d in range(di):
            du_d[t * di + d] = ftz(
                pinned_mul(ftz(bst.dsk[t * di + d]), ftz(w.d_skip[d]))
            )
            var s1 = ftz(ftz(du_d[t * di + d]) + bst.du_s[t * di + d])
            bst.du.append(ftz(s1 + du_x[t * di + d]))

    for t in range(m):
        for d in range(di):
            var c = ftz(st.conv_out[t * di + d])
            bst.dconv.append(
                ftz(pinned_mul(ftz(bst.du[t * di + d]), pinned_silu_prime(c)))
            )

    for bb in range(b):
        for p in range(l):
            for d in range(di):
                var acc = Float32(0.0)
                for k in range(D_CONV):
                    var q = p + (D_CONV - 1) - k
                    if q < l:
                        acc = ftz(
                            identical_mul_add(
                                ftz(w.conv_w[d * D_CONV + k]),
                                ftz(bst.dconv[(bb * l + q) * di + d]),
                                acc,
                            )
                        )
                _set_at(bst.dhin, (bb * l + p) * di + d, acc, m * di)

    var dp = List[Float32]()
    for t in range(m):
        for d in range(di):
            dp.append(bst.dhin[t * di + d])
        for d in range(di):
            dp.append(bst.dz[t * di + d])
    bst.dnrm = gemm_oracle(dp, w.w_in, OP_NN, m, dm, 2 * di)
    bst.dw_in = gemm_oracle(dp, st.norm_out, OP_TN, 2 * di, dm, m)

    var dinner = _zeros(m * dm)
    for t in range(m):
        var acc = Float32(0.0)
        for j in range(dm):
            dinner[t * dm + j] = ftz(
                pinned_mul(ftz(bst.dnrm[t * dm + j]), ftz(w.norm_w[j]))
            )
            acc = ftz(
                identical_mul_add(
                    dinner[t * dm + j], ftz(x[t * dm + j]), acc
                )
            )
        bst.drstd.append(acc)

    for t in range(m):
        var rstd = _forward_rstd(st.norm_sumsq[t], dm)
        var c3 = ftz(pinned_mul(ftz(pinned_mul(rstd, rstd)), rstd))
        var s = ftz(identical_div(c3, Float32(dm)))
        for j in range(dm):
            var t2 = ftz(
                pinned_mul(
                    ftz(pinned_mul(s, ftz(x[t * dm + j]))),
                    ftz(bst.drstd[t]),
                )
            )
            var t1 = ftz(pinned_mul(rstd, dinner[t * dm + j]))
            var dx_norm = ftz(t1 - t2)
            bst.dx.append(ftz(ftz(dres[t * dm + j]) + dx_norm))

    var da_partial = _zeros(b * di * D_STATE)
    for bb in range(b):
        for d in range(di):
            for n in range(D_STATE):
                var av = ftz(st.a_out[d * D_STATE + n])
                var acc = Float32(0.0)
                for li in range(l - 1, -1, -1):  # T4: DESCENDING
                    var t = bb * l + li
                    var dl = ftz(st.softplus_out[t * di + d])
                    var d_da = ftz(
                        pinned_mul(
                            ftz(bst.dh[(t * di + d) * D_STATE + n]),
                            ftz(_h_at(bst.h_ckpt, bb, li - 1, d, l, di, n)),
                        )
                    )
                    var d_arg = ftz(pinned_mul(d_da, _forward_da(dl, av)))
                    acc = ftz(identical_mul_add(d_arg, dl, acc))
                da_partial[(bb * di + d) * D_STATE + n] = acc
    for d in range(di):
        for n in range(D_STATE):
            var acc = Float32(0.0)
            for bb in range(b):  # T5: ASCENDING, plain flushed add
                acc = ftz(acc + ftz(da_partial[(bb * di + d) * D_STATE + n]))
            bst.da_log.append(
                ftz(pinned_mul(acc, ftz(st.a_out[d * D_STATE + n])))
            )

    var ones = List[Float32]()
    for _ in range(m):
        ones.append(Float32(1.0))

    var p_d = _zeros(m * di)
    for t in range(m):
        for d in range(di):
            p_d[t * di + d] = ftz(
                pinned_mul(
                    ftz(bst.dsk[t * di + d]), ftz(st.silu_out[t * di + d])
                )
            )
    bst.dd_skip = gemm_oracle(ones, p_d, OP_NN, 1, di, m)

    bst.db_dt = gemm_oracle(ones, bst.ddtp, OP_NN, 1, di, m)

    bst.dcw = _zeros(di * D_CONV)
    for k in range(D_CONV):
        var p_cw = _zeros(m * di)
        for bb in range(b):
            for li in range(l):
                var p = li - (D_CONV - 1) + k
                for d in range(di):
                    var hv: Float32
                    if p >= 0:
                        hv = st.in_proj[(bb * l + p) * 2 * di + d]
                    else:
                        hv = state_in.conv_win[
                            (bb * di + d) * D_CONV + (D_CONV + p)
                        ]
                    p_cw[(bb * l + li) * di + d] = ftz(
                        pinned_mul(
                            ftz(bst.dconv[(bb * l + li) * di + d]), ftz(hv)
                        )
                    )
        var tap = gemm_oracle(ones, p_cw, OP_NN, 1, di, m)
        for d in range(di):
            bst.dcw[d * D_CONV + k] = tap[d]

    bst.dcb = gemm_oracle(ones, bst.dconv, OP_NN, 1, di, m)

    var p_w = _zeros(m * dm)
    for t in range(m):
        var rstd = _forward_rstd(st.norm_sumsq[t], dm)
        for j in range(dm):
            var inner = ftz(pinned_mul(ftz(x[t * dm + j]), rstd))
            p_w[t * dm + j] = ftz(
                pinned_mul(ftz(bst.dnrm[t * dm + j]), inner)
            )
    bst.dw_norm = gemm_oracle(ones, p_w, OP_NN, 1, dm, m)

    return bst^


def _set_at(mut xs: List[Float32], i: Int, v: Float32, size: Int):
    """Write-by-index into a stage list, growing it to `size` zeros first."""
    while len(xs) < size:
        xs.append(Float32(0.0))
    xs[i] = v
