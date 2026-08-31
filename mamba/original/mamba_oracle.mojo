# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host Float32 oracle of one Mamba-1 block under profile
`mojolearn.identical.mamba1.fp32.v1`, and the Float64 tolerance reference.

NOT A PORT -- the reference libraries ship no oracle; the ALGORITHM the
oracle spells is theirs, cited seam by seam in
`mamba/IDENTICAL_MAMBA_CONTRACT.md` and in the ported files. This file is
the contract's arithmetic order written on the CPU through the SAME seams
(`identical_mul_add`, `ftz`, `identical_exp`, `identical_div`,
`identical_rsqrt`, `identical_silu`, `identical_softplus`, GEMM v1's
`gemm_oracle`), so the device card of
`mamba/derived/transformers/models/mamba/modeling_mamba.mojo` is diffed
against it bitwise. The device kernels are an INDEPENDENT transcription of
the same order; nothing below is imported by them except the seam functions
themselves, which are shared BY DESIGN (one arithmetic, two spellings of
the loops around it).

The Float64 reference at the bottom is `selective_scan_ref` semantics in
double precision. It is a TOLERANCE instrument (is the FP32 answer near the
real number), never a bitwise one.
"""

from original.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_mul_add,
    identical_rsqrt,
    identical_silu,
    identical_softplus,
)
from gemm.original.gemm_oracle import OP_NT, gemm_oracle
from mamba.original.mamba_fixture import (
    D_CONV,
    D_STATE,
    MambaDims,
    MambaWeights,
    RMS_EPS,
)


def pinned_mul(a: Float32, b: Float32) -> Float32:
    """DEVIATION 720: a MULTIPLY no codegen may contract into a neighboring
    add. Spelled `identical_mul_add(a, b, -0.0)`: under IDENTICAL that is
    `fma(a, b, -0.0)`, which is bit-equal to the correctly rounded product
    at EVERY input including zero signs (`p + (-0.0) == p` for p of either
    zero sign under round-to-nearest; a `+0.0` addend would launder a
    `-0.0` product, the gemm lane's F6a lesson), and which presents no
    syntactic multiply for a compiler to contract (the gemm README's F3
    scar: `var p = a * b; p + c` WAS contracted across statements on this
    host). Under FAST it is `a * b + (-0.0)`, the plain product. Used at
    every seam the reference rounds as its own multiply: `delta * A`,
    `delta * B`, `(delta*B) * u`, `u * D`, `x * rstd`, `weight * hidden`,
    `skip * silu(z)`."""
    return identical_mul_add(a, b, Float32(-0.0))


def refuse_nonfinite(name: String, values: List[Float32]) raises:
    """Row 39: a NaN or infinity in an input or a parameter is REFUSED BY
    NAME before any recorded stage. IEEE default semantics would carry it
    through every seam (contract section 6 of
    `mamba/IDENTICAL_MAMBA_CONTRACT.md`), and NaN PAYLOADS are vendor-shaped
    (row 39: three payloads for one IEEE answer), so a certified card can
    never contain a computed NaN. Tested BY BITS, not by compares: Metal
    flushes COMPARE operands (row 49's measurement), so a bit test is the
    only spelling with one meaning on every column."""
    from std.memory import bitcast

    for i in range(len(values)):
        var au = bitcast[DType.uint32](values[i]) & UInt32(0x7FFFFFFF)
        if au > UInt32(0x7F800000):
            raise Error(
                String("mamba: NaN in ") + name + " at flat index "
                + String(i) + " REFUSED (row 39: NaN payloads are"
                + " vendor-shaped; no stage may record one)"
            )
        if au == UInt32(0x7F800000):
            raise Error(
                String("mamba: infinity in ") + name + " at flat index "
                + String(i) + " REFUSED (row 39)"
            )


def refuse_bad_inputs(w: MambaWeights, x: List[Float32]) raises:
    refuse_nonfinite("x", x)
    refuse_nonfinite("norm.weight", w.norm_w)
    refuse_nonfinite("in_proj.weight", w.w_in)
    refuse_nonfinite("conv1d.weight", w.conv_w)
    refuse_nonfinite("conv1d.bias", w.conv_b)
    refuse_nonfinite("x_proj.weight", w.w_x)
    refuse_nonfinite("dt_proj.weight", w.w_dt)
    refuse_nonfinite("dt_proj.bias", w.b_dt)
    refuse_nonfinite("A_log", w.a_log)
    refuse_nonfinite("D", w.d_skip)
    refuse_nonfinite("out_proj.weight", w.w_out)


struct MambaState(Copyable, Movable):
    """The recurrent state between calls: the conv WINDOW (the last d_conv
    inputs of the conv, oldest first -- `mamba_simple.py:216-217`'s
    `conv_state` after the roll) and the SSM state h
    (`selective_scan_ref:160`'s `x`). Zeros before the first token
    (`mamba_simple.py:258-266 allocate_inference_cache`). Layouts
    `[B, d_inner, D_CONV]` and `[B, d_inner, D_STATE]`, row-major."""

    var conv_win: List[Float32]
    var h: List[Float32]

    def __init__(out self, b: Int, dims: MambaDims):
        self.conv_win = List[Float32]()
        self.h = List[Float32]()
        for _ in range(b * dims.d_inner * D_CONV):
            self.conv_win.append(0.0)
        for _ in range(b * dims.d_inner * D_STATE):
            self.h.append(0.0)


struct MambaStages(Movable):
    """Every recorded stage of one block call, in the card's order. Token-
    major layouts (`M = B * L` rows): the projections' natural GEMM shapes.
    The corpus's channel-major `[B, d_inner, L]` view of the same values is
    the corpus check's reindexing, not a second computation."""

    var norm_sumsq: List[Float32]  # [M]
    var norm_out: List[Float32]    # [M, d_model]
    var in_proj: List[Float32]     # [M, 2*d_inner]
    var a_out: List[Float32]       # [d_inner, D_STATE]
    var conv_out: List[Float32]    # [M, d_inner]
    var silu_out: List[Float32]    # [M, d_inner]
    var x_proj: List[Float32]      # [M, dt_rank + 2*D_STATE]
    var dt_proj: List[Float32]     # [M, d_inner]  (bias NOT added; MM:429)
    var softplus_out: List[Float32]  # [M, d_inner]  softplus(dt_proj + bias)
    var scan_y: List[Float32]      # [M, d_inner]   sum_n C*h, BEFORE D
    var scan_h: List[Float32]      # [B, d_inner, D_STATE]  final state
    var skip_out: List[Float32]    # [M, d_inner]   scan_y + u*D
    var gate_out: List[Float32]    # [M, d_inner]   skip * silu(z)
    var out_proj: List[Float32]    # [M, d_model]
    var residual_out: List[Float32]  # [M, d_model]
    var conv_win: List[Float32]    # [B, d_inner, D_CONV] window after

    def __init__(out self):
        self.norm_sumsq = List[Float32]()
        self.norm_out = List[Float32]()
        self.in_proj = List[Float32]()
        self.a_out = List[Float32]()
        self.conv_out = List[Float32]()
        self.silu_out = List[Float32]()
        self.x_proj = List[Float32]()
        self.dt_proj = List[Float32]()
        self.softplus_out = List[Float32]()
        self.scan_y = List[Float32]()
        self.scan_h = List[Float32]()
        self.skip_out = List[Float32]()
        self.gate_out = List[Float32]()
        self.out_proj = List[Float32]()
        self.residual_out = List[Float32]()
        self.conv_win = List[Float32]()


def selective_scan_oracle(
    u: List[Float32],       # [M, d_inner] token-major
    delta: List[Float32],   # [M, d_inner] (post-softplus)
    a: List[Float32],       # [d_inner, D_STATE]
    bmat: List[Float32],    # [M, D_STATE]
    cmat: List[Float32],    # [M, D_STATE]
    mut h_state: List[Float32],  # [B, d_inner, D_STATE], in/out
    b: Int,
    l: Int,
    d_inner: Int,
) -> List[Float32]:
    """`selective_scan_ref` (mamba_ssm e9594ce,
    selective_scan_interface.py:160-187) under the contract's pins, WITHOUT
    the D skip and the z gate (the reference applies those after the loop;
    the block oracle below does too). Returns y `[M, d_inner]`; `h_state`
    is left holding the state after token `l - 1` (`:183-184 last_state`).

    Order, per (b, d) independently -- the reference's own semantic order:
      :162  deltaA    = exp(delta * A)          seam S5 (pinned_mul), S6 (identical_exp)
      :167  deltaB_u  = (delta * B) * u         seams S7, S8: the einsum's
            left-to-right pairwise order (delta with B first, then u), which
            is also HF `modeling_mamba.py:188-189`'s explicit spelling
            (`discrete_B = dt * B; deltaB_u = discrete_B * u`). The CUDA
            kernel rounds it the OTHER way (`B * (delta * u)`,
            selective_scan_fwd_kernel.cuh:162,222) -- fixture F5 shows the
            two separate, and the reference's order is the profile's.
      :175  h = deltaA * h + deltaB_u           seam S9: ONE rounding (fma).
            The torch reference rounds twice (mul, add); fusion is pinned
            because only fusion HAS a portable spelling (gemm contract
            section 4), and the CUDA scan op contracts here too.
      :180  y = sum_n h[n] * C[n]               seam S10: serial ascending n
            from +0.0, fma. The einsum's fold order is torch-internal; the
            CUDA kernel walks state_idx 0..15 ascending (:185-266), which is
            the order pinned.
    """
    var y = List[Float32]()
    for _ in range(b * l * d_inner):
        y.append(0.0)
    for bb in range(b):
        for d in range(d_inner):
            var h = List[Float32]()
            for n in range(D_STATE):
                h.append(ftz(h_state[(bb * d_inner + d) * D_STATE + n]))
            for li in range(l):
                var t = bb * l + li
                var uv = ftz(u[t * d_inner + d])
                var dl = ftz(delta[t * d_inner + d])
                for n in range(D_STATE):
                    var da = ftz(
                        identical_exp(ftz(pinned_mul(dl, ftz(a[d * D_STATE + n]))))
                    )
                    var db = ftz(pinned_mul(dl, ftz(bmat[t * D_STATE + n])))
                    var dbu = ftz(pinned_mul(db, uv))
                    h[n] = ftz(identical_mul_add(da, h[n], dbu))
                var acc = Float32(0.0)
                for n in range(D_STATE):
                    acc = ftz(
                        identical_mul_add(ftz(cmat[t * D_STATE + n]), h[n], acc)
                    )
                y[t * d_inner + d] = acc
            for n in range(D_STATE):
                h_state[(bb * d_inner + d) * D_STATE + n] = h[n]
    return y^


def mamba_block_oracle(
    w: MambaWeights,
    x: List[Float32],  # [B, L, d_model] row-major (token-major)
    b: Int,
    l: Int,
    mut state: MambaState,
) raises -> MambaStages:
    """One `MambaBlock.forward` (transformers d56c55b modeling_mamba.py:505-530:
    `residual = hidden; hidden = norm(hidden); hidden = mixer(hidden);
    hidden = residual + hidden`), stage by stage, through the seams.
    Prefill is a fresh zero `MambaState`; the decode step is this same
    function at `l == 1` carrying the state -- ONE spelling for both, which
    is what makes gate D (decode == prefill) a theorem the gate then
    verifies rather than a coincidence."""
    refuse_bad_inputs(w, x)
    refuse_nonfinite("state.conv_win", state.conv_win)
    refuse_nonfinite("state.h", state.h)
    var dims = w.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    var m = b * l
    var st = MambaStages()

    # ---- MambaRMSNorm (modeling_mamba.py:485-503, eps 1e-5 MC:70) --------
    # variance = mean(x^2)  -> serial ascending fma from +0.0 (seam S1);
    # x * rsqrt(variance + eps) (S2, S3); weight * hidden (S4).
    for t in range(m):
        var acc = Float32(0.0)
        for j in range(dm):
            var xj = ftz(x[t * dm + j])
            acc = ftz(identical_mul_add(xj, xj, acc))
        st.norm_sumsq.append(acc)
        var mean = ftz(identical_div(acc, Float32(dm)))
        var rstd = ftz(identical_rsqrt(ftz(mean + RMS_EPS)))
        for j in range(dm):
            var inner = ftz(pinned_mul(ftz(x[t * dm + j]), rstd))
            st.norm_out.append(ftz(pinned_mul(ftz(w.norm_w[j]), inner)))

    # ---- in_proj (MM:371; Linear no bias, use_bias False MC:75) ----------
    # GEMM v1 OP_NT: [M, dm] . [2di, dm]^T. k = dm <= 128 so P == 1: the
    # serial ascending chain, gemm contract section 7.1.
    st.in_proj = gemm_oracle(st.norm_out, w.w_in, OP_NT, m, 2 * di, dm)

    # ---- A = -exp(A_log) (MM:373) ----------------------------------------
    for i in range(di * D_STATE):
        st.a_out.append(-ftz(identical_exp(ftz(w.a_log[i]))))

    # ---- causal depthwise conv1d, kernel 4, padding 3, truncated to L ----
    # (MM:410-419 causal_conv1d_fn = F.conv1d(padding=3, groups=di)[:,:,:L];
    #  bias use_conv_bias True MC:76). Spelling: bias-SEEDED accumulator,
    #  taps k = 0..3 ascending (oldest first), each tap one fma, a
    #  pre-sequence position reads the state WINDOW (zeros on prefill ==
    #  the zero padding). MAX's causal_conv1d.mojo:190-205 is this exact
    #  shape; the decode step (mamba_simple.py:216-220) is the same chain
    #  by construction -- DEVIATION 721 records that its upstream "sum then
    #  + bias" order is NOT mirrored, the prefill's bias-seed is.
    var new_win = List[Float32]()
    for bb in range(b):
        for d in range(di):
            for li in range(l):
                var acc = ftz(w.conv_b[d])
                for k in range(D_CONV):
                    var p = li - (D_CONV - 1) + k
                    var xv: Float32
                    if p >= 0:
                        xv = x_hidden_at(st.in_proj, bb, p, d, l, di)
                    else:
                        xv = state.conv_win[(bb * di + d) * D_CONV + (D_CONV + p)]
                    acc = ftz(
                        identical_mul_add(ftz(w.conv_w[d * D_CONV + k]), ftz(xv), acc)
                    )
                # conv_out is token-major [M, di]; this loop fills by
                # (bb, d, li), so write by index, not append.
                _set_at(st.conv_out, (bb * l + li) * di + d, acc, m * di)
                _set_at(
                    st.silu_out,
                    (bb * l + li) * di + d,
                    ftz(identical_silu(acc)),
                    m * di,
                )
            # the window after this call: the last D_CONV inputs (raw
            # copies -- a copy is not an arithmetic seam)
            for j in range(D_CONV):
                var p = l - D_CONV + j
                if p >= 0:
                    new_win.append(x_hidden_at(st.in_proj, bb, p, d, l, di))
                else:
                    new_win.append(
                        state.conv_win[(bb * di + d) * D_CONV + (D_CONV + p)]
                    )
    state.conv_win = new_win.copy()
    st.conv_win = new_win^

    # ---- x_proj (MM:422-427; Linear no bias) -----------------------------
    st.x_proj = gemm_oracle(st.silu_out, w.w_x, OP_NT, m, xr, di)

    # ---- split (torch.split, MM:422; bit-exact copies, not a stage) ------
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

    # ---- dt_proj WITHOUT bias (MM:429: `dt_proj.weight @ time_step`) -----
    # k = dt_rank (1 here): GEMM v1's one-fma leaf, seeded +0.0 -- v1
    # section 9.2(a) launders a -0.0 product to +0.0 at this stage, an
    # inherited clause, recorded in the contract.
    st.dt_proj = gemm_oracle(dt_low, w.w_dt, OP_NT, m, di, r)

    # ---- delta = softplus(dt + bias) (MM:178-181 in mamba_selective_scan;
    #      selective_scan_ref:145-148) ------------------------------------
    for t in range(m):
        for d in range(di):
            var biased = ftz(st.dt_proj[t * di + d] + ftz(w.b_dt[d]))
            st.softplus_out.append(ftz(identical_softplus(biased)))

    # ---- the scan (selective_scan_ref:160-187) ---------------------------
    st.scan_y = selective_scan_oracle(
        st.silu_out, st.softplus_out, st.a_out, bmat, cmat, state.h, b, l, di
    )
    for i in range(b * di * D_STATE):
        st.scan_h.append(state.h[i])

    # ---- D skip (selective_scan_ref:189 `out = y + u * D`) ---------------
    # u * D is the reference's own rounding step (also the CUDA kernel's,
    # cuh:163): pinned_mul, then the add -- UNFUSED at this seam, both
    # references round the product.
    for t in range(m):
        for d in range(di):
            var p = ftz(pinned_mul(ftz(st.silu_out[t * di + d]), ftz(w.d_skip[d])))
            st.skip_out.append(ftz(st.scan_y[t * di + d] + p))

    # ---- the z gate (selective_scan_ref:190-191 `out = out * silu(z)`;
    #      z is the second half of in_proj, MM:396) ------------------------
    for t in range(m):
        for d in range(di):
            var z = ftz(st.in_proj[t * 2 * di + di + d])
            st.gate_out.append(
                ftz(pinned_mul(st.skip_out[t * di + d], ftz(identical_silu(z))))
            )

    # ---- out_proj (MM:476; Linear no bias) -------------------------------
    st.out_proj = gemm_oracle(st.gate_out, w.w_out, OP_NT, m, dm, di)

    # ---- residual (MM:521-527: `hidden = residual + mixer(norm(residual))`)
    for i in range(m * dm):
        st.residual_out.append(ftz(ftz(x[i]) + st.out_proj[i]))

    return st^


def x_hidden_at(
    in_proj: List[Float32], bb: Int, li: Int, d: Int, l: Int, di: Int
) -> Float32:
    """The conv's input: column `d` of the HIDDEN half of in_proj.out
    (MM:396 `chunk(2, dim=1)`, first half), token `(bb, li)`. RAW value;
    the caller flushes at the seam."""
    return in_proj[(bb * l + li) * 2 * di + d]


def _set_at(mut xs: List[Float32], i: Int, v: Float32, size: Int):
    """Write-by-index into a stage list, growing it to `size` zeros first."""
    while len(xs) < size:
        xs.append(0.0)
    xs[i] = v


# ===========================================================================
# The Float64 tolerance reference (selective_scan_ref semantics, double).
# ===========================================================================


struct MambaRef64(Movable):
    var skip_out: List[Float64]      # scan.y + u*D, the corpus's "scan.y"
    var residual_out: List[Float64]  # the corpus's "block.out"

    def __init__(out self):
        self.skip_out = List[Float64]()
        self.residual_out = List[Float64]()


def mamba_block_ref64(
    w: MambaWeights, x: List[Float32], b: Int, l: Int
) raises -> MambaRef64:
    """The block in Float64, plain spellings (two-rounding mul/add, stdlib
    exp/log/sqrt): a TOLERANCE reference for fixtures the corpus does not
    carry. Bitwise claims never touch this function."""
    from std.math import exp, log, sqrt

    var dims = w.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    var m = b * l
    var out = MambaRef64()

    var norm = List[Float64]()
    for t in range(m):
        var acc = Float64(0.0)
        for j in range(dm):
            acc += Float64(x[t * dm + j]) * Float64(x[t * dm + j])
        var rstd = 1.0 / sqrt(acc / Float64(dm) + Float64(RMS_EPS))
        for j in range(dm):
            norm.append(Float64(w.norm_w[j]) * (Float64(x[t * dm + j]) * rstd))

    var proj = List[Float64]()
    for t in range(m):
        for c in range(2 * di):
            var acc = Float64(0.0)
            for j in range(dm):
                acc += norm[t * dm + j] * Float64(w.w_in[c * dm + j])
            proj.append(acc)

    var u = List[Float64]()
    for _ in range(m * di):
        u.append(0.0)
    for bb in range(b):
        for d in range(di):
            for li in range(l):
                var acc = Float64(w.conv_b[d])
                for k in range(D_CONV):
                    var p = li - (D_CONV - 1) + k
                    var xv = Float64(0.0)
                    if p >= 0:
                        xv = proj[(bb * l + p) * 2 * di + d]
                    acc += Float64(w.conv_w[d * D_CONV + k]) * xv
                u[(bb * l + li) * di + d] = acc / (1.0 + exp(-acc))

    var delta = List[Float64]()
    var bm = List[Float64]()
    var cm = List[Float64]()
    for t in range(m):
        var xp = List[Float64]()
        for c in range(xr):
            var acc = Float64(0.0)
            for j in range(di):
                acc += u[t * di + j] * Float64(w.w_x[c * di + j])
            xp.append(acc)
        for d in range(di):
            var acc = Float64(0.0)
            for j in range(r):
                acc += xp[j] * Float64(w.w_dt[d * r + j])
            var biased = acc + Float64(w.b_dt[d])
            if biased <= 20.0:
                delta.append(log(1.0 + exp(biased)))
            else:
                delta.append(biased)
        for n in range(2 * D_STATE):
            if n < D_STATE:
                bm.append(xp[r + n])
            else:
                cm.append(xp[r + n])

    for bb in range(b):
        for d in range(di):
            var h = List[Float64]()
            for _ in range(D_STATE):
                h.append(0.0)
            for li in range(l):
                var t = bb * l + li
                var dl = delta[t * di + d]
                var uv = u[t * di + d]
                for n in range(D_STATE):
                    var a = -exp(Float64(w.a_log[d * D_STATE + n]))
                    var da = exp(dl * a)
                    h[n] = da * h[n] + (dl * bm[t * D_STATE + n]) * uv
                var y = Float64(0.0)
                for n in range(D_STATE):
                    y += cm[t * D_STATE + n] * h[n]
                _set64(out.skip_out, t * di + d, y + uv * Float64(w.d_skip[d]), m * di)

    out.residual_out = List[Float64]()
    for t in range(m):
        for c in range(dm):
            var acc = Float64(0.0)
            for d in range(di):
                var z = proj[t * 2 * di + di + d]
                var g = out.skip_out[t * di + d] * (z / (1.0 + exp(-z)))
                acc += g * Float64(w.w_out[c * di + d])
            out.residual_out.append(Float64(x[t * dm + c]) + acc)
    return out^


def _set64(mut xs: List[Float64], i: Int, v: Float64, size: Int):
    while len(xs) < size:
        xs.append(0.0)
    xs[i] = v
