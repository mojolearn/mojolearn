# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host kernels for the byte LM's threaded CPU path (DEVIATION 2624).

THE ORACLES' ARITHMETIC, NOT A NEW ONE. Every floating point operation below
is one the host oracles perform, on the same operands, in the same order for
each output value, through the same seams (`identical_mul_add` and its
lane-wise twin `identical_mul_add_simd`, `identical_mul`, `identical_div`,
`identical_rsqrt`, `identical_exp`, `identical_silu`, `identical_fmax`,
`ftz`). What changes is everything around that arithmetic:

  - Nothing is allocated per GEMM cell. `gemm_oracle_cell` builds a partials
    List and `fold_balanced_tree` copies it for every cell; on the M4 profile
    of the reference path the allocator was the largest cost after the cells.
  - Right operands are flushed and packed once per call (`pack_nt`). `ftz` is
    pure and idempotent, so `ftz(B[j*k+p])` read from a flushed copy is the
    value the oracle's per-read `ftz` returns.
  - The cells of one output row advance together, one SIMD lane per cell,
    down the p axis. Each lane of `identical_mul_add_simd` is an IEEE fused
    multiply-add, correctly rounded exactly as the scalar seam is, so every
    cell runs the oracle's chain and no lane reads another.
  - The attention value sum (S19) advances all `head_dim` outputs of one
    query together down the key axis, the same chain per output.
  - `ftz_lanes` flushes every lane without a branch; the argument that it is
    `ftz` is at the function.

The folds the contracts make serial stay serial and scalar: the RMS sum of
squares (S1), the softmax maximum (S14) and denominator (S17). Prefill only,
from absolute position 0 with no window and no plant, which is the only call
`training/byte_lm_host.mojo` makes of `transformer_block_oracle`.

A PREDICTION UNTIL THE GATE RUNS. `tools/byte_lm_host_gate.py` compares this
path's loss bytes against the retained Metal, CUDA and HIP captures and its
logits against the reference path's.
"""

from std.memory import bitcast
from std.sys.info import simd_width_of

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_mul_add_simd,
    identical_rsqrt,
    identical_silu,
)
from gemm.checks.gemm_oracle import contract_leaf_size, leaf_count
from training.checks.loss_oracle import (
    CE_NEG_INF_BITS,
    REDUCTION_MEAN,
    CeConfig,
    ce_count,
    ce_divisor,
    ce_refuse_inputs,
    neg_by_bits,
)
from transformer.checks.transformer_fixture import (
    RMS_EPS,
    TransformerDims,
    attention_scale,
    mask_fill,
    unmasked_fill,
)
from transformer.checks.transformer_oracle import RopeTable, refuse_nonfinite


comptime HOST_FW = simd_width_of[DType.float32]()
comptime F32V = SIMD[DType.float32, HOST_FW]
comptime U32V = SIMD[DType.uint32, HOST_FW]
#: SIMD accumulators `gemm_nt_rows` advances together per p step at one leaf
#: (spelled out as eight locals there). A schedule knob: every lane still
#: runs its own cell's chain, so it changes no bit.
comptime GEMM_CHAINS = 8


def _identical_build_only():
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, "byte LM host kernels: IDENTICAL builds only"


@always_inline
def ftz_lanes(x: F32V) -> F32V:
    """`ftz` on every lane, without a branch.

    Under IDENTICAL, `ftz(x)` is the sign bit alone when the exponent field is
    zero and the mantissa is not, and `x` otherwise. When the exponent field
    and the mantissa are both zero, `x` already IS its sign bit. Selecting the
    sign bit whenever the exponent field is zero is therefore the same map on
    all 2^32 bit patterns."""
    var bits = bitcast[DType.uint32](x)
    var subnormal = (bits & U32V(0x7F800000)).eq(U32V(0))
    return bitcast[DType.float32](subnormal.select(bits & U32V(0x80000000), bits))


@always_inline
def _neg_zero_lanes() -> F32V:
    return bitcast[DType.float32](U32V(0x80000000))


def flushed(values: List[Float32]) -> List[Float32]:
    """`ftz` of every value, once."""
    _identical_build_only()
    var out = List[Float32](capacity=len(values))
    for i in range(len(values)):
        out.append(ftz(values[i]))
    return out^


def pack_nt(b: List[Float32], n: Int, k: Int) -> List[Float32]:
    """The right operand of an OP_NT product, `B [n x k]`, flushed and laid
    out `[k x n]`: `out[p * n + j] = ftz(B[j * k + p])`, so the p-th terms of
    the n cells of one output row are contiguous."""
    _identical_build_only()
    var out = List[Float32](length=n * k, fill=Float32(0.0))
    for j in range(n):
        for p in range(k):
            out[p * n + j] = ftz(b[j * k + p])
    return out^


def gemm_nt_rows(
    a: List[Float32],
    bt: List[Float32],
    n: Int,
    k: Int,
    lo: Int,
    hi: Int,
    mut c: List[Float32],
    reverse: Bool = False,
) raises:
    """Rows `[lo, hi)` of `gemm_oracle(A, B, OP_NT, m, n, k)`, written to
    `c[(i - lo) * n + j]`. `a` is `A [m x k]` as the oracle receives it and
    `bt` is `pack_nt(B, n, k)`.

    Per cell, the oracle's arithmetic: leaves at `contract_leaf_size(k)`, each
    `acc = ftz(identical_mul_add(ftz(a[i*k+p]), ftz(b[j*k+p]), acc))` for p
    ascending from `+0.0`, then `fold_balanced_tree` over the partials,
    pairing `(2q, 2q + 1)` and carrying an odd tail bit for bit. The fold runs
    in place over the level-0 scratch, which the gemm contract names as a
    legal layout: every node is written after both children are read. The
    oracle's `ftz` on each child read and on the output is not repeated,
    because every stored value is already flushed and `ftz` is idempotent.

    `reverse` walks p DESCENDING (DEVIATION 2612's negative control) and is
    admitted only at one leaf, where it changes the fold and nothing else."""
    if hi < lo or len(c) < (hi - lo) * n:
        raise Error("byte LM host kernels: GEMM output span too short")
    var zero = Float32(0.0)
    var cp = c.unsafe_ptr()
    if k <= 0:
        for i in range((hi - lo) * n):
            cp.unsafe_store(i, zero)
        return
    var leaf = contract_leaf_size(k)
    var pcount = leaf_count(k, leaf)
    if reverse and pcount != 1:
        raise Error("byte LM host kernels: the reversed fold is admitted at one leaf only")
    var ap = a.unsafe_ptr()
    var bp = bt.unsafe_ptr()
    if pcount == 1:
        # ONE LEAF (k <= CONTRACT_K_LEAF_MIN): each cell is the serial chain
        # and its output IS the chain's accumulator. Accumulators stay in
        # registers, GEMM_CHAINS SIMD vectors advanced together per p step,
        # instead of a scratch row loaded and stored every step. The step
        # order per lane and the operands are unchanged.
        var arow = List[Float32](length=k, fill=zero)
        var boff = List[Int](length=k, fill=0)
        var arp = arow.unsafe_ptr()
        var bop = boff.unsafe_ptr()
        var zv = F32V(0.0)
        var group = GEMM_CHAINS * HOST_FW
        for i in range(lo, hi):
            for step in range(k):
                var p = step
                if reverse:
                    p = k - 1 - step
                arp.unsafe_store(step, ftz(ap.unsafe_load(i * k + p)))
                bop.unsafe_store(step, p * n)
            var row = (i - lo) * n
            var jb = 0
            while jb + group <= n:
                var c0 = zv
                var c1 = zv
                var c2 = zv
                var c3 = zv
                var c4 = zv
                var c5 = zv
                var c6 = zv
                var c7 = zv
                for step in range(k):
                    var sv = F32V(arp.unsafe_load(step))
                    var base = bop.unsafe_load(step) + jb
                    c0 = ftz_lanes(identical_mul_add_simd[HOST_FW](sv, bp.unsafe_load[width=HOST_FW](base), c0))
                    c1 = ftz_lanes(identical_mul_add_simd[HOST_FW](sv, bp.unsafe_load[width=HOST_FW](base + HOST_FW), c1))
                    c2 = ftz_lanes(identical_mul_add_simd[HOST_FW](sv, bp.unsafe_load[width=HOST_FW](base + 2 * HOST_FW), c2))
                    c3 = ftz_lanes(identical_mul_add_simd[HOST_FW](sv, bp.unsafe_load[width=HOST_FW](base + 3 * HOST_FW), c3))
                    c4 = ftz_lanes(identical_mul_add_simd[HOST_FW](sv, bp.unsafe_load[width=HOST_FW](base + 4 * HOST_FW), c4))
                    c5 = ftz_lanes(identical_mul_add_simd[HOST_FW](sv, bp.unsafe_load[width=HOST_FW](base + 5 * HOST_FW), c5))
                    c6 = ftz_lanes(identical_mul_add_simd[HOST_FW](sv, bp.unsafe_load[width=HOST_FW](base + 6 * HOST_FW), c6))
                    c7 = ftz_lanes(identical_mul_add_simd[HOST_FW](sv, bp.unsafe_load[width=HOST_FW](base + 7 * HOST_FW), c7))
                cp.unsafe_store[width=HOST_FW](row + jb, c0)
                cp.unsafe_store[width=HOST_FW](row + jb + HOST_FW, c1)
                cp.unsafe_store[width=HOST_FW](row + jb + 2 * HOST_FW, c2)
                cp.unsafe_store[width=HOST_FW](row + jb + 3 * HOST_FW, c3)
                cp.unsafe_store[width=HOST_FW](row + jb + 4 * HOST_FW, c4)
                cp.unsafe_store[width=HOST_FW](row + jb + 5 * HOST_FW, c5)
                cp.unsafe_store[width=HOST_FW](row + jb + 6 * HOST_FW, c6)
                cp.unsafe_store[width=HOST_FW](row + jb + 7 * HOST_FW, c7)
                jb += group
            while jb + HOST_FW <= n:
                var cv = zv
                for step in range(k):
                    cv = ftz_lanes(identical_mul_add_simd[HOST_FW](
                        F32V(arp.unsafe_load(step)), bp.unsafe_load[width=HOST_FW](bop.unsafe_load(step) + jb), cv))
                cp.unsafe_store[width=HOST_FW](row + jb, cv)
                jb += HOST_FW
            while jb < n:
                var cs = zero
                for step in range(k):
                    cs = ftz(identical_mul_add(arp.unsafe_load(step), bp.unsafe_load(bop.unsafe_load(step) + jb), cs))
                cp.unsafe_store(row + jb, cs)
                jb += 1
        return
    var body = n - n % HOST_FW
    var scratch = List[Float32](length=pcount * n, fill=zero)
    var sp = scratch.unsafe_ptr()
    for i in range(lo, hi):
        for t in range(pcount):
            var acc_base = t * n
            for jz in range(n):
                sp.unsafe_store(acc_base + jz, zero)
            var p_begin = t * leaf
            var p_end = p_begin + leaf
            if p_end > k:
                p_end = k
            for step in range(p_begin, p_end):
                var p = step
                if reverse:
                    p = k - 1 - step
                var av = ftz(ap.unsafe_load(i * k + p))
                var avv = F32V(av)
                var brow = p * n
                var jv = 0
                while jv < body:
                    var acc = sp.unsafe_load[width=HOST_FW](acc_base + jv)
                    sp.unsafe_store[width=HOST_FW](
                        acc_base + jv,
                        ftz_lanes(identical_mul_add_simd[HOST_FW](
                            avv, bp.unsafe_load[width=HOST_FW](brow + jv), acc)),
                    )
                    jv += HOST_FW
                while jv < n:
                    sp.unsafe_store(
                        acc_base + jv,
                        ftz(identical_mul_add(av, bp.unsafe_load(brow + jv),
                                              sp.unsafe_load(acc_base + jv))),
                    )
                    jv += 1
        var width = pcount
        while width > 1:
            var pairs = width // 2
            for q in range(pairs):
                var dst = q * n
                var left = 2 * q * n
                var right = left + n
                var jf = 0
                while jf < body:
                    sp.unsafe_store[width=HOST_FW](
                        dst + jf,
                        ftz_lanes(sp.unsafe_load[width=HOST_FW](left + jf)
                                  + sp.unsafe_load[width=HOST_FW](right + jf)),
                    )
                    jf += HOST_FW
                while jf < n:
                    sp.unsafe_store(dst + jf, ftz(sp.unsafe_load(left + jf) + sp.unsafe_load(right + jf)))
                    jf += 1
            if width % 2 != 0:
                # THE CARRY, bit for bit (fold_balanced_tree's odd tail).
                var src = (width - 1) * n
                var dst_carry = pairs * n
                for jc in range(n):
                    sp.unsafe_store(dst_carry + jc, sp.unsafe_load(src + jc))
            width = pairs + width % 2
        var row = (i - lo) * n
        for jo in range(n):
            cp.unsafe_store(row + jo, sp.unsafe_load(jo))


def rms_norm_fast(x: List[Float32], wnorm_flushed: List[Float32], m: Int, dm: Int) -> List[Float32]:
    """`rms_norm_into`'s output (S1-S4) for `m` tokens. S1 stays one serial
    scalar fold per token; S3 and S4 are one `identical_mul` per element, so
    they run as lanes."""
    var out = List[Float32](length=m * dm, fill=Float32(0.0))
    var xp = x.unsafe_ptr()
    var wp = wnorm_flushed.unsafe_ptr()
    var op = out.unsafe_ptr()
    var neg0 = _neg_zero_lanes()
    var body = dm - dm % HOST_FW
    for t in range(m):
        var base = t * dm
        var acc = Float32(0.0)
        for j in range(dm):
            var xj = ftz(xp.unsafe_load(base + j))
            acc = ftz(identical_mul_add(xj, xj, acc))
        var mean = ftz(identical_div(acc, Float32(dm)))
        var rstd = ftz(identical_rsqrt(ftz(mean + RMS_EPS)))
        var rv = F32V(rstd)
        var jv = 0
        while jv < body:
            var inner = ftz_lanes(identical_mul_add_simd[HOST_FW](
                ftz_lanes(xp.unsafe_load[width=HOST_FW](base + jv)), rv, neg0))
            op.unsafe_store[width=HOST_FW](
                base + jv,
                ftz_lanes(identical_mul_add_simd[HOST_FW](wp.unsafe_load[width=HOST_FW](jv), inner, neg0)),
            )
            jv += HOST_FW
        while jv < dm:
            var inner_s = ftz(identical_mul(ftz(xp.unsafe_load(base + jv)), rstd))
            op.unsafe_store(base + jv, ftz(identical_mul(wp.unsafe_load(jv), inner_s)))
            jv += 1
    return out^


def rope_fast(src: List[Float32], n_head: Int, hd: Int, l: Int, rope: RopeTable) raises -> List[Float32]:
    """`apply_rope_into`'s output (S9, S10) at `pos0 = 0` for one batch row of
    `l` tokens, scalar and in the oracle's spelling."""
    if l > rope.positions:
        raise Error("byte LM host kernels: length exceeds the rotary table")
    var half = hd // 2
    var width = n_head * hd
    var out = List[Float32](length=l * width, fill=Float32(0.0))
    var sp = src.unsafe_ptr()
    var op = out.unsafe_ptr()
    var cosp = rope.cos.unsafe_ptr()
    var sinp = rope.sin.unsafe_ptr()
    for t in range(l):
        for h in range(n_head):
            var base = t * width + h * hd
            for j in range(hd):
                var ci: Int
                var rot: Float32
                if j < half:
                    ci = j
                    rot = -ftz(sp.unsafe_load(base + j + half))
                else:
                    ci = j - half
                    rot = ftz(sp.unsafe_load(base + j - half))
                var cv = ftz(cosp.unsafe_load(t * half + ci))
                var sv = ftz(sinp.unsafe_load(t * half + ci))
                var pa = ftz(identical_mul(ftz(sp.unsafe_load(base + j)), cv))
                var pb = ftz(identical_mul(rot, sv))
                op.unsafe_store(base + j, ftz(ftz(pa) + ftz(pb)))
    return out^


def block_fast(
    tensors: List[List[Float32]],
    tb: Int,
    x: List[Float32],
    l: Int,
    dims: TransformerDims,
    rope: RopeTable,
) raises -> List[Float32]:
    """`transformer_block_oracle(...).residual2_out` for ONE batch row of `l`
    tokens with a fresh cache, prefill at position 0, no window, no plant.

    `tensors[tb .. tb + 8]` are the block's weights as
    `byte_host_fast_tensors` prepares them: norm1 flushed, q, k, v, o packed,
    norm2 flushed, gate, up, down packed. Their shapes and finiteness were
    refused by `refuse_bad_weights` when they were prepared."""
    var dm = dims.d_model
    var nh = dims.n_heads
    var hd = dims.head_dim
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var inter = dims.intermediate
    var n_rep = dims.n_rep()
    var m = l
    var s = l
    if len(x) != m * dm:
        raise Error("byte LM host kernels: block input has the wrong length")
    refuse_nonfinite("x", x)

    # S1-S4, then S5 q/k/v.
    var n1 = rms_norm_fast(x, tensors[tb], m, dm)
    var q = List[Float32](length=m * qw, fill=Float32(0.0))
    var k = List[Float32](length=m * kw, fill=Float32(0.0))
    var v = List[Float32](length=m * kw, fill=Float32(0.0))
    gemm_nt_rows(n1, tensors[tb + 1], qw, dm, 0, m, q)
    gemm_nt_rows(n1, tensors[tb + 2], kw, dm, 0, m, k)
    gemm_nt_rows(n1, tensors[tb + 3], kw, dm, 0, m, v)

    # S9, S10. The key span is this call's own tokens: key j is token j.
    var qr = rope_fast(q, nh, hd, l, rope)
    var kr = rope_fast(k, dims.n_kv_heads, hd, l, rope)

    var scale = attention_scale(hd)
    var mfill = mask_fill()
    var ufill = unmasked_fill()
    var ctx = List[Float32](length=m * qw, fill=Float32(0.0))
    var qmat = List[Float32](length=l * hd, fill=Float32(0.0))
    var kpack = List[Float32](length=hd * s, fill=Float32(0.0))
    var vpack = List[Float32](length=s * hd, fill=Float32(0.0))
    var cell = List[Float32](length=l * s, fill=Float32(0.0))
    var masked = List[Float32](length=s, fill=Float32(0.0))
    var shifted = List[Float32](length=s, fill=Float32(0.0))
    var aexp = List[Float32](length=s, fill=Float32(0.0))
    var aweights = List[Float32](length=s, fill=Float32(0.0))
    # S13's additive mask per (query, key), `+0.0` where the key is visible
    # and the finite fill where it is not, exactly the `mv` the oracle picks.
    var masks = List[Float32](length=l * s, fill=ufill)
    for qi in range(l):
        for j in range(qi + 1, s):
            masks[qi * s + j] = mfill
    var ctxp = ctx.unsafe_ptr()
    var vp = vpack.unsafe_ptr()
    var cellp = cell.unsafe_ptr()
    var mkp = masks.unsafe_ptr()
    var mp = masked.unsafe_ptr()
    var shp = shifted.unsafe_ptr()
    var ep = aexp.unsafe_ptr()
    var awp = aweights.unsafe_ptr()
    var neg0 = _neg_zero_lanes()
    var scale_v = F32V(scale)
    var hbody = hd - hd % HOST_FW
    var sbody = s - s % HOST_FW
    for h in range(nh):
        var kvh = h // n_rep
        for qi in range(l):
            for d in range(hd):
                qmat[qi * hd + d] = qr[qi * qw + h * hd + d]
        for j in range(s):
            for d in range(hd):
                kpack[d * s + j] = ftz(kr[j * kw + kvh * hd + d])
                vpack[j * hd + d] = ftz(v[j * kw + kvh * hd + d])
        # S11, one gemm per head, k = head_dim.
        gemm_nt_rows(qmat, kpack, s, hd, 0, l, cell)
        for qi in range(l):
            var crow = qi * s
            # S12, S13: one product and one add per key, as lanes.
            var jv = 0
            while jv < sbody:
                var sc = ftz_lanes(identical_mul_add_simd[HOST_FW](
                    ftz_lanes(cellp.unsafe_load[width=HOST_FW](crow + jv)), scale_v, neg0))
                mp.unsafe_store[width=HOST_FW](jv, ftz_lanes(sc + mkp.unsafe_load[width=HOST_FW](crow + jv)))
                jv += HOST_FW
            while jv < s:
                var sc_s = ftz(identical_mul(ftz(cellp.unsafe_load(crow + jv)), scale))
                mp.unsafe_store(jv, ftz(ftz(sc_s) + mkp.unsafe_load(crow + jv)))
                jv += 1
            # S14, serial and scalar.
            var mx = ftz(mp.unsafe_load(0))
            for j in range(1, s):
                mx = identical_fmax(mx, ftz(mp.unsafe_load(j)))
            mx = ftz(mx)
            # S15 as lanes (one subtraction per key), S16 scalar.
            var mxv = F32V(mx)
            jv = 0
            while jv < sbody:
                shp.unsafe_store[width=HOST_FW](jv, ftz_lanes(mp.unsafe_load[width=HOST_FW](jv) - mxv))
                jv += HOST_FW
            while jv < s:
                shp.unsafe_store(jv, ftz(ftz(mp.unsafe_load(jv)) - mx))
                jv += 1
            for j in range(s):
                ep.unsafe_store(j, ftz(identical_exp(shp.unsafe_load(j))))
            # S17, serial ascending from +0.0, scalar.
            var den = Float32(0.0)
            for j in range(s):
                den = ftz(ftz(den) + ftz(ep.unsafe_load(j)))
            den = ftz(den)
            # S18 as lanes: `identical_div` is `portable_divf`, the flush
            # around ONE correctly rounded division, and every operand here is
            # already flushed.
            var denv = F32V(den)
            jv = 0
            while jv < sbody:
                awp.unsafe_store[width=HOST_FW](jv, ftz_lanes(ep.unsafe_load[width=HOST_FW](jv) / denv))
                jv += HOST_FW
            while jv < s:
                awp.unsafe_store(jv, ftz(identical_div(ftz(ep.unsafe_load(jv)), den)))
                jv += 1
            # S19, one serial chain per output, all head_dim outputs as lanes.
            var cbase = qi * qw + h * hd
            for j in range(s):
                var wj = aweights[j]
                var wv = F32V(wj)
                var vrow = j * hd
                var dv = 0
                while dv < hbody:
                    var acc = ctxp.unsafe_load[width=HOST_FW](cbase + dv)
                    ctxp.unsafe_store[width=HOST_FW](
                        cbase + dv,
                        ftz_lanes(identical_mul_add_simd[HOST_FW](
                            wv, vp.unsafe_load[width=HOST_FW](vrow + dv), acc)),
                    )
                    dv += HOST_FW
                while dv < hd:
                    ctxp.unsafe_store(
                        cbase + dv,
                        ftz(identical_mul_add(wj, vp.unsafe_load(vrow + dv), ctxp.unsafe_load(cbase + dv))),
                    )
                    dv += 1

    # S5 o_proj, S22.
    var o = List[Float32](length=m * dm, fill=Float32(0.0))
    gemm_nt_rows(ctx, tensors[tb + 4], dm, qw, 0, m, o)
    var r1 = List[Float32](length=m * dm, fill=Float32(0.0))
    for i in range(m * dm):
        r1[i] = ftz(ftz(x[i]) + ftz(o[i]))

    # S1-S4 again, then the MLP (S5, S20, S21, S5) and S23.
    var n2 = rms_norm_fast(r1, tensors[tb + 5], m, dm)
    var gate = List[Float32](length=m * inter, fill=Float32(0.0))
    var up = List[Float32](length=m * inter, fill=Float32(0.0))
    gemm_nt_rows(n2, tensors[tb + 6], inter, dm, 0, m, gate)
    gemm_nt_rows(n2, tensors[tb + 7], inter, dm, 0, m, up)
    var gated = List[Float32](length=m * inter, fill=Float32(0.0))
    for i in range(m * inter):
        var silu = ftz(identical_silu(ftz(gate[i])))
        gated[i] = ftz(identical_mul(ftz(silu), ftz(up[i])))
    var down = List[Float32](length=m * dm, fill=Float32(0.0))
    gemm_nt_rows(gated, tensors[tb + 8], dm, inter, 0, m, down)
    var r2 = List[Float32](length=m * dm, fill=Float32(0.0))
    for i in range(m * dm):
        r2[i] = ftz(ftz(r1[i]) + ftz(down[i]))
    return r2^


def hidden_fast(
    tensors: List[List[Float32]],
    rope: RopeTable,
    row_ids: List[Int32],
    l: Int,
    dims: TransformerDims,
    layers: Int,
) raises -> List[Float32]:
    """The last block's residual `[l, d_model]` for one batch row whose ids
    were validated by the caller. `tensors[0]` is the flushed embedding, so a
    row copy is `emb_forward_oracle`'s `ftz(ftz(W[id, j]))`."""
    var dm = dims.d_model
    var ep = tensors[0].unsafe_ptr()
    var x = List[Float32](length=l * dm, fill=Float32(0.0))
    var xp = x.unsafe_ptr()
    for t in range(l):
        var base = Int(row_ids[t]) * dm
        for j in range(dm):
            xp.unsafe_store(t * dm + j, ep.unsafe_load(base + j))
    for layer in range(layers):
        x = block_fast(tensors, 1 + 9 * layer, x, l, dims, rope)
    return x^


def ce_causal_mean_loss_fast(logits: List[Float32], targets: List[Int32], vocab: Int) raises -> Float32:
    """`ce_forward_oracle(logits, targets, CeConfig.causal_lm(vocab)).loss[0]`.

    The oracle's refusals (`ce_refuse_inputs`, called as is) and then its
    seams per row in its order: L1 `identical_fmax` from `-inf` over the
    unflushed logits, L2 the flushed shift, L3 `identical_exp` with no outer
    flush, L5 the flushed `identical_log`, L6 the flushed difference, L7
    `neg_by_bits`, and a `+0.0` row where the target is `ignore_index`.

    Both folds (L4's denominator per row and the batch total) are `ce_fold`,
    which is `gemm_oracle(values, ones, OP_NN, 1, 1, count)`. At m = n = 1,
    OP_NN reads `a[p]` and `b[p]`, which is what OP_NT reads there, so each
    fold is `gemm_nt_rows` against a ones operand: the same leaves at
    `contract_leaf_size(count)` and the same balanced tree. The denominators
    of all rows are one call, one output row per logits row.

    The causal LM configuration only: mean reduction, no label smoothing,
    which is the only loss `byte_host_loss` computes."""
    var cfg = CeConfig.causal_lm(vocab)
    if cfg.smoothing_is_spelled() or cfg.reduction != REDUCTION_MEAN or cfg.num_items != 0:
        raise Error("byte LM host kernels: the loss kernel is the causal LM mean with no smoothing only")
    var n = ce_refuse_inputs(logits, targets, cfg)
    var v = cfg.vocab
    var lp = logits.unsafe_ptr()
    var shift = List[Float32](length=n * v, fill=Float32(0.0))
    var expo = List[Float32](length=n * v, fill=Float32(0.0))
    var shp = shift.unsafe_ptr()
    var exp_p = expo.unsafe_ptr()
    for i in range(n):
        var base = i * v
        var mx = bitcast[DType.float32](CE_NEG_INF_BITS)
        for vv in range(v):
            mx = identical_fmax(mx, lp.unsafe_load(base + vv))
        for vv in range(v):
            var s = ftz(ftz(lp.unsafe_load(base + vv)) - ftz(mx))
            shp.unsafe_store(base + vv, s)
            exp_p.unsafe_store(base + vv, identical_exp(s))
    var ones_v = List[Float32](length=v, fill=Float32(1.0))
    var denom = List[Float32](length=n, fill=Float32(0.0))
    gemm_nt_rows(expo, ones_v, 1, v, 0, n, denom)
    var rows = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        var y = Int(targets[i])
        var ignored = y == cfg.ignore_index
        var logdenom = ftz(identical_log(ftz(denom[i])))
        var ty = y
        if ignored:
            ty = 0
        var lp_y = ftz(ftz(shp.unsafe_load(i * v + ty)) - ftz(logdenom))
        var row_loss = neg_by_bits(lp_y)
        if ignored:
            row_loss = Float32(0.0)
        rows[i] = row_loss
    var ones_n = List[Float32](length=n, fill=Float32(1.0))
    var total = List[Float32](length=1, fill=Float32(0.0))
    gemm_nt_rows(rows, ones_n, 1, n, 0, 1, total)
    var divisor = ce_divisor(cfg.reduction, ce_count(targets, cfg.ignore_index), cfg.num_items)
    return ftz(identical_div(ftz(total[0]), divisor))
