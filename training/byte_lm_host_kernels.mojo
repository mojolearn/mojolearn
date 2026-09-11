# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Host kernels for the byte LM's threaded CPU path (DEVIATION 2624).

THE ORACLES' ARITHMETIC, NOT A NEW ONE. Every floating point operation below
is one the host oracles perform, on the same operands, in the same order for
each output value, through the same seams (`identical_mul_add` and its
lane-wise twin `identical_mul_add_simd`, `identical_mul`, `identical_div`,
`identical_rsqrt`, `identical_exp`, `identical_silu`, `identical_fmax`,
`ftz`) or through a lane-wise respelling of one of them that is checked
against it. What changes is everything around that arithmetic:

  - Nothing is allocated per GEMM cell. `gemm_oracle_cell` builds a partials
    List and `fold_balanced_tree` copies it for every cell; on the M4 profile
    of the reference path the allocator was the largest cost after the cells.
  - Operands are flushed and packed once per call, straight from the
    parameters (`pack_nt_span`, `flushed_span`). `ftz` is pure and
    idempotent, so a value read from a flushed copy is the value the
    oracle's per-read `ftz` returns.
  - The cells of one GEMM output row advance together, one SIMD lane per
    cell, down the p axis, in registers, with the flush deferred and an
    exact fallback (`_chain_step`, `gemm_nt_rows`). Each lane of
    `identical_mul_add_simd` is an IEEE fused multiply-add, correctly
    rounded exactly as the scalar seam is.
  - Per-element seams run as lanes: attention scale, mask add, shift,
    exponential and division, SiLU, the gated product, residual adds, and
    the loss's shift and exponential. `expf_lanes` and `silu_lanes` respell
    `portable_expf` and `portable_siluf`; `training/checks/
    byte_lm_host_exp_check.mojo` compares them with the scalar seams on all
    2^32 Float32 bit patterns.
  - The attention value sum (S19) advances all `head_dim` outputs of one
    query together down the key axis, the same chain per output.

The folds the contracts make serial stay serial: the RMS sum of squares
(S1), the softmax maximum (S14) and denominator (S17). Prefill only, from
absolute position 0 with no window and no plant, which is the only call
`training/byte_lm_host.mojo` makes of `transformer_block_oracle`.

A PREDICTION UNTIL IT IS GATED. `tools/byte_lm_host_gate.py` compares this
path's loss bytes against the retained Metal, CUDA and HIP captures and its
logits against the reference path's, and
`training/checks/byte_lm_host_kernels_check.mojo` compares every kernel
with its oracle by bits on shapes and planted values the captures never
reach.
"""

from std.math import floor, fma, min
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
#: (spelled out as eight locals there). A schedule knob: every lane still runs
#: its own cell's chain, so it changes no bit. On the M4 at one thread, 16
#: chains measured 18% slower than 8 at [32, 32].
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
def _chain_step(sv: F32V, bv: F32V, acc: F32V, mut exps: U32V) -> F32V:
    """One p step of one SIMD accumulator with the flush DEFERRED: the raw
    `identical_mul_add(a, b, acc)` per lane, while `exps` keeps the lane-wise
    minimum exponent field of every raw result.

    Why that is still the oracle's chain. The operands are flushed before the
    chain, so only an accumulator can be subnormal. If no raw result in a lane
    is subnormal, `ftz` was the identity at every step of that lane and, by
    induction from the `+0.0` seed, the unflushed chain IS the flushed chain
    bit for bit. A zero minimum exponent field (a subnormal, or an exact zero,
    which is conservative) sends the whole group back through the flush at
    every step (`gemm_nt_rows`). On the M4 at one thread this measured 17%
    faster at [32, 32] than flushing every step, because the accumulator's
    dependency chain is one fused multiply-add per step instead of four
    operations."""
    var raw = identical_mul_add_simd[HOST_FW](sv, bv, acc)
    exps = min(exps, bitcast[DType.uint32](raw) & U32V(0x7F800000))
    return raw


@always_inline
def _neg_zero_lanes() -> F32V:
    return bitcast[DType.float32](U32V(0x80000000))


@always_inline
def _nan_bits(x: F32V) -> SIMD[DType.bool, HOST_FW]:
    """NaN lanes by bits: magnitude above the infinity's."""
    return (bitcast[DType.uint32](x) & U32V(0x7FFFFFFF)).gt(U32V(0x7F800000))


@always_inline
def expf_lanes(x: F32V) -> F32V:
    """`checks/numerics.mojo::portable_expf` on every lane: the same
    operations on the same constants in the same order, with the scalar's
    three early returns (NaN as is, `+inf` above 88.722835, `+0.0` below
    -87.33655) applied last as masks in the scalar's priority. Special lanes
    run the arithmetic on `+0.0`, so no lane converts a NaN or an out-of-range
    value to an integer.

    The NaN lanes are passed through by an INTEGER select on the input's bits.
    Selected as floats, a signaling NaN came back quieted (0x7f800001 as
    0x7fc00001, measured on the M4), where the scalar returns it untouched.
    `training/checks/byte_lm_host_exp_check.mojo` compares this with the
    scalar over all 2^32 bit patterns."""
    var nan = _nan_bits(x)
    var over = x.gt(F32V(88.722835))
    var under = x.lt(F32V(-87.33655))
    var xs = (nan | over | under).select(F32V(0.0), x)
    var t = xs * F32V(1.4426950408889634)
    t = t + F32V(0.5)
    var zf = floor(t)
    var r = fma(zf, F32V(-0.693359375), xs)
    r = fma(zf, F32V(2.12194440e-4), r)
    var q = F32V(1.9875691500e-4)
    q = fma(q, r, F32V(1.3981999507e-3))
    q = fma(q, r, F32V(8.3334519073e-3))
    q = fma(q, r, F32V(4.1665795894e-2))
    q = fma(q, r, F32V(1.6666665459e-1))
    q = fma(q, r, F32V(5.0000001201e-1))
    var r2 = r * r
    var y = fma(q, r2, r)
    y = y + F32V(1.0)
    var k = zf.cast[DType.int32]()
    var k1 = k >> SIMD[DType.int32, HOST_FW](1)
    var k2 = k - k1
    var bias = SIMD[DType.int32, HOST_FW](127)
    var shift = SIMD[DType.int32, HOST_FW](23)
    y = y * bitcast[DType.float32](((k1 + bias) << shift).cast[DType.uint32]())
    y = y * bitcast[DType.float32](((k2 + bias) << shift).cast[DType.uint32]())
    y = y.lt(F32V(1.1754943508222875e-38)).select(F32V(0.0), y)
    y = under.select(F32V(0.0), y)
    y = over.select(bitcast[DType.float32](U32V(0x7F800000)), y)
    return bitcast[DType.float32](nan.select(bitcast[DType.uint32](x), bitcast[DType.uint32](y)))


@always_inline
def silu_lanes(x: F32V) -> F32V:
    """`checks/numerics.mojo::portable_siluf` on every lane: NaN as is (by
    an integer select, as in `expf_lanes`), else
    `portable_divf(x, portable_expf(-x) + 1.0)`, whose flushes are the
    unconditional `_ftz_always`, which is `ftz_lanes` on every bit pattern.
    Compared with the scalar over all 2^32 bit patterns by
    `training/checks/byte_lm_host_exp_check.mojo`."""
    var nan = _nan_bits(x)
    var d = expf_lanes(-x) + F32V(1.0)
    var quotient = ftz_lanes(ftz_lanes(x) / ftz_lanes(d))
    return bitcast[DType.float32](nan.select(bitcast[DType.uint32](x), bitcast[DType.uint32](quotient)))


def all_finite_span(values: List[Float32], lo: Int, hi: Int) -> Bool:
    """Whether every value in `[lo, hi)` is finite, tested by bits (an
    exponent field of all ones is an infinity or a NaN), lane-wise. A screen
    only: a caller that finds a non-finite value raises through the oracle's
    own refusal, so the message is the oracle's."""
    var sp = values.unsafe_ptr()
    var expm = U32V(0x7F800000)
    var i = lo
    while i + HOST_FW <= hi:
        if (bitcast[DType.uint32](sp.unsafe_load[width=HOST_FW](i)) & expm).reduce_max() == UInt32(0x7F800000):
            return False
        i += HOST_FW
    while i < hi:
        if (bitcast[DType.uint32](sp.unsafe_load(i)) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            return False
        i += 1
    return True


def flushed_span(values: List[Float32], lo: Int, hi: Int) -> List[Float32]:
    """`ftz` of every value in `[lo, hi)`, once, as lanes."""
    _identical_build_only()
    var n = hi - lo
    var out = List[Float32](length=n, fill=Float32(0.0))
    var sp = values.unsafe_ptr()
    var op = out.unsafe_ptr()
    var i = 0
    while i + HOST_FW <= n:
        op.unsafe_store[width=HOST_FW](i, ftz_lanes(sp.unsafe_load[width=HOST_FW](lo + i)))
        i += HOST_FW
    while i < n:
        op.unsafe_store(i, ftz(sp.unsafe_load(lo + i)))
        i += 1
    return out^


def flushed(values: List[Float32]) -> List[Float32]:
    """`ftz` of every value, once."""
    return flushed_span(values, 0, len(values))


def pack_nt_span(values: List[Float32], lo: Int, n: Int, k: Int) -> List[Float32]:
    """The right operand of an OP_NT product, the `B [n x k]` stored at
    `values[lo : lo + n * k]`, flushed and laid out `[k x n]`:
    `out[p * n + j] = ftz(B[j * k + p])`, so the p-th terms of the n cells of
    one output row are contiguous."""
    _identical_build_only()
    var out = List[Float32](length=n * k, fill=Float32(0.0))
    var sp = values.unsafe_ptr()
    var op = out.unsafe_ptr()
    for j in range(n):
        var src = lo + j * k
        for p in range(k):
            op.unsafe_store(p * n + j, ftz(sp.unsafe_load(src + p)))
    return out^


def pack_nt(b: List[Float32], n: Int, k: Int) -> List[Float32]:
    """`pack_nt_span` of a whole `B [n x k]`."""
    return pack_nt_span(b, 0, n, k)


def gemm_nt_rows(
    a: List[Float32],
    bt: List[Float32],
    n: Int,
    k: Int,
    lo: Int,
    hi: Int,
    mut c: List[Float32],
    reverse: Bool = False,
    force_redo: Bool = False,
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

    At one leaf, groups of GEMM_CHAINS vectors defer the flush (`_chain_step`)
    and fall back to flushing at every step when a raw result's exponent field
    is zero; `force_redo` takes that fallback for every group, which is how
    `training/checks/byte_lm_host_kernels_check.mojo` reaches it.

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
        var expm = U32V(0x7F800000)
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
                var e0 = expm
                var e1 = expm
                var e2 = expm
                var e3 = expm
                var e4 = expm
                var e5 = expm
                var e6 = expm
                var e7 = expm
                for step in range(k):
                    var sv = F32V(arp.unsafe_load(step))
                    var base = bop.unsafe_load(step) + jb
                    c0 = _chain_step(sv, bp.unsafe_load[width=HOST_FW](base), c0, e0)
                    c1 = _chain_step(sv, bp.unsafe_load[width=HOST_FW](base + HOST_FW), c1, e1)
                    c2 = _chain_step(sv, bp.unsafe_load[width=HOST_FW](base + 2 * HOST_FW), c2, e2)
                    c3 = _chain_step(sv, bp.unsafe_load[width=HOST_FW](base + 3 * HOST_FW), c3, e3)
                    c4 = _chain_step(sv, bp.unsafe_load[width=HOST_FW](base + 4 * HOST_FW), c4, e4)
                    c5 = _chain_step(sv, bp.unsafe_load[width=HOST_FW](base + 5 * HOST_FW), c5, e5)
                    c6 = _chain_step(sv, bp.unsafe_load[width=HOST_FW](base + 6 * HOST_FW), c6, e6)
                    c7 = _chain_step(sv, bp.unsafe_load[width=HOST_FW](base + 7 * HOST_FW), c7, e7)
                var emin = min(min(min(e0, e1), min(e2, e3)), min(min(e4, e5), min(e6, e7)))
                if force_redo or emin.reduce_min() == UInt32(0):
                    # A zero or subnormal raw result (or a forced check): this
                    # group again, with the flush at every step.
                    var jr = jb
                    while jr < jb + group:
                        var cr = zv
                        for step in range(k):
                            cr = ftz_lanes(identical_mul_add_simd[HOST_FW](
                                F32V(arp.unsafe_load(step)),
                                bp.unsafe_load[width=HOST_FW](bop.unsafe_load(step) + jr), cr))
                        cp.unsafe_store[width=HOST_FW](row + jr, cr)
                        jr += HOST_FW
                    jb += group
                    continue
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


@no_inline
def _residual_add(a: List[Float32], b: List[Float32]) -> List[Float32]:
    """S22 and S23: `ftz(ftz(a[i]) + ftz(b[i]))`, one add per element, as
    lanes."""
    var n = len(a)
    var out = List[Float32](length=n, fill=Float32(0.0))
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var op = out.unsafe_ptr()
    var i = 0
    while i + HOST_FW <= n:
        op.unsafe_store[width=HOST_FW](i, ftz_lanes(
            ftz_lanes(ap.unsafe_load[width=HOST_FW](i)) + ftz_lanes(bp.unsafe_load[width=HOST_FW](i))))
        i += HOST_FW
    while i < n:
        op.unsafe_store(i, ftz(ftz(ap.unsafe_load(i)) + ftz(bp.unsafe_load(i))))
        i += 1
    return out^


@no_inline
def _softmax_head(cell: List[Float32], masks: List[Float32], l: Int, s: Int, scale: Float32,
                  mut aweights: List[Float32]):
    """S12-S18 for every query of one head, from `cell` (S11, `[l, s]`) and
    the additive `masks`, into `aweights` `[l, s]`: the scale, mask add,
    shift, exponential and division as lanes per key, the maximum (S14) and
    the denominator (S17) as the oracle's serial scalar folds."""
    var cellp = cell.unsafe_ptr()
    var mkp = masks.unsafe_ptr()
    var awp = aweights.unsafe_ptr()
    var masked = List[Float32](length=s, fill=Float32(0.0))
    var aexp = List[Float32](length=s, fill=Float32(0.0))
    var mp = masked.unsafe_ptr()
    var ep = aexp.unsafe_ptr()
    var neg0 = _neg_zero_lanes()
    var scale_v = F32V(scale)
    var sbody = s - s % HOST_FW
    for qi in range(l):
        var crow = qi * s
        # S12, S13.
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
        # S15, S16.
        var mxv = F32V(mx)
        jv = 0
        while jv < sbody:
            ep.unsafe_store[width=HOST_FW](jv, ftz_lanes(expf_lanes(ftz_lanes(mp.unsafe_load[width=HOST_FW](jv) - mxv))))
            jv += HOST_FW
        while jv < s:
            ep.unsafe_store(jv, ftz(identical_exp(ftz(ftz(mp.unsafe_load(jv)) - mx))))
            jv += 1
        # S17, serial ascending from +0.0, scalar.
        var den = Float32(0.0)
        for j in range(s):
            den = ftz(ftz(den) + ftz(ep.unsafe_load(j)))
        den = ftz(den)
        # S18: `identical_div` is `portable_divf`, the flush around ONE
        # correctly rounded division, and every operand here is flushed.
        var denv = F32V(den)
        jv = 0
        while jv < sbody:
            awp.unsafe_store[width=HOST_FW](crow + jv, ftz_lanes(ep.unsafe_load[width=HOST_FW](jv) / denv))
            jv += HOST_FW
        while jv < s:
            awp.unsafe_store(crow + jv, ftz(identical_div(ftz(ep.unsafe_load(jv)), den)))
            jv += 1


@no_inline
def _value_sum_head(aweights: List[Float32], vpack: List[Float32], l: Int, s: Int, hd: Int,
                    qw: Int, h: Int, mut ctx: List[Float32]):
    """S19 for every query of one head: one serial chain per output over the
    key axis from `+0.0`, all `head_dim` outputs of a query as lanes. At a
    head_dim of one or two SIMD widths the accumulators stay in registers;
    `ctx` holds `+0.0` in this head's slots on entry."""
    var awp = aweights.unsafe_ptr()
    var vp = vpack.unsafe_ptr()
    var ctxp = ctx.unsafe_ptr()
    var hbody = hd - hd % HOST_FW
    for qi in range(l):
        var cbase = qi * qw + h * hd
        var wrow = qi * s
        if hd == 2 * HOST_FW:
            var a0 = F32V(0.0)
            var a1 = F32V(0.0)
            for j in range(s):
                var wv2 = F32V(awp.unsafe_load(wrow + j))
                var vrow2 = j * hd
                a0 = ftz_lanes(identical_mul_add_simd[HOST_FW](wv2, vp.unsafe_load[width=HOST_FW](vrow2), a0))
                a1 = ftz_lanes(identical_mul_add_simd[HOST_FW](wv2, vp.unsafe_load[width=HOST_FW](vrow2 + HOST_FW), a1))
            ctxp.unsafe_store[width=HOST_FW](cbase, a0)
            ctxp.unsafe_store[width=HOST_FW](cbase + HOST_FW, a1)
            continue
        if hd == HOST_FW:
            var a_one = F32V(0.0)
            for j in range(s):
                a_one = ftz_lanes(identical_mul_add_simd[HOST_FW](
                    F32V(awp.unsafe_load(wrow + j)), vp.unsafe_load[width=HOST_FW](j * hd), a_one))
            ctxp.unsafe_store[width=HOST_FW](cbase, a_one)
            continue
        for j in range(s):
            var wj = awp.unsafe_load(wrow + j)
            var wv = F32V(wj)
            var vrow = j * hd
            var dv = 0
            while dv < hbody:
                var acc = ctxp.unsafe_load[width=HOST_FW](cbase + dv)
                ctxp.unsafe_store[width=HOST_FW](
                    cbase + dv,
                    ftz_lanes(identical_mul_add_simd[HOST_FW](wv, vp.unsafe_load[width=HOST_FW](vrow + dv), acc)),
                )
                dv += HOST_FW
            while dv < hd:
                ctxp.unsafe_store(
                    cbase + dv,
                    ftz(identical_mul_add(wj, vp.unsafe_load(vrow + dv), ctxp.unsafe_load(cbase + dv))),
                )
                dv += 1


@no_inline
def _silu_gated(gate: List[Float32], up: List[Float32]) -> List[Float32]:
    """S20 then S21 per element, as lanes: the oracle's
    `silu = ftz(identical_silu(ftz(gate)))` and
    `ftz(identical_mul(ftz(silu), ftz(up)))`."""
    var n = len(gate)
    var out = List[Float32](length=n, fill=Float32(0.0))
    var gatep = gate.unsafe_ptr()
    var upp = up.unsafe_ptr()
    var op = out.unsafe_ptr()
    var neg0 = _neg_zero_lanes()
    var i = 0
    while i + HOST_FW <= n:
        var silu = ftz_lanes(silu_lanes(ftz_lanes(gatep.unsafe_load[width=HOST_FW](i))))
        op.unsafe_store[width=HOST_FW](i, ftz_lanes(identical_mul_add_simd[HOST_FW](
            ftz_lanes(silu), ftz_lanes(upp.unsafe_load[width=HOST_FW](i)), neg0)))
        i += HOST_FW
    while i < n:
        var silu_s = ftz(identical_silu(ftz(gatep.unsafe_load(i))))
        op.unsafe_store(i, ftz(identical_mul(ftz(silu_s), ftz(upp.unsafe_load(i)))))
        i += 1
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
    refused when they were prepared."""
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

    # S13's additive mask per (query, key), `+0.0` where the key is visible
    # and the finite fill where it is not, exactly the `mv` the oracle picks.
    var mfill = mask_fill()
    var masks = List[Float32](length=l * s, fill=unmasked_fill())
    for qi in range(l):
        for j in range(qi + 1, s):
            masks[qi * s + j] = mfill
    var scale = attention_scale(hd)
    var ctx = List[Float32](length=m * qw, fill=Float32(0.0))
    var qmat = List[Float32](length=l * hd, fill=Float32(0.0))
    var kpack = List[Float32](length=hd * s, fill=Float32(0.0))
    var vpack = List[Float32](length=s * hd, fill=Float32(0.0))
    var cell = List[Float32](length=l * s, fill=Float32(0.0))
    var aweights = List[Float32](length=l * s, fill=Float32(0.0))
    var qrp = qr.unsafe_ptr()
    var krp = kr.unsafe_ptr()
    var vvp = v.unsafe_ptr()
    var qmp = qmat.unsafe_ptr()
    var kpp = kpack.unsafe_ptr()
    var vpp = vpack.unsafe_ptr()
    for h in range(nh):
        var kvh = h // n_rep
        for qi in range(l):
            for d in range(hd):
                qmp.unsafe_store(qi * hd + d, qrp.unsafe_load(qi * qw + h * hd + d))
        for j in range(s):
            for d in range(hd):
                kpp.unsafe_store(d * s + j, ftz(krp.unsafe_load(j * kw + kvh * hd + d)))
                vpp.unsafe_store(j * hd + d, ftz(vvp.unsafe_load(j * kw + kvh * hd + d)))
        # S11, one gemm per head, k = head_dim; then S12-S18 and S19.
        gemm_nt_rows(qmat, kpack, s, hd, 0, l, cell)
        _softmax_head(cell, masks, l, s, scale, aweights)
        _value_sum_head(aweights, vpack, l, s, hd, qw, h, ctx)

    # S5 o_proj, S22; S1-S4 again; the MLP (S5, S20, S21, S5); S23.
    var o = List[Float32](length=m * dm, fill=Float32(0.0))
    gemm_nt_rows(ctx, tensors[tb + 4], dm, qw, 0, m, o)
    var r1 = _residual_add(x, o)
    var n2 = rms_norm_fast(r1, tensors[tb + 5], m, dm)
    var gate = List[Float32](length=m * inter, fill=Float32(0.0))
    var up = List[Float32](length=m * inter, fill=Float32(0.0))
    gemm_nt_rows(n2, tensors[tb + 6], inter, dm, 0, m, gate)
    gemm_nt_rows(n2, tensors[tb + 7], inter, dm, 0, m, up)
    var gated = _silu_gated(gate, up)
    var down = List[Float32](length=m * dm, fill=Float32(0.0))
    gemm_nt_rows(gated, tensors[tb + 8], dm, inter, 0, m, down)
    return _residual_add(r1, down)


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
    unflushed logits, L2 the flushed shift and L3 the exponential with no
    outer flush (both as lanes), L5 the flushed `identical_log`, L6 the
    flushed difference, L7 `neg_by_bits`, and a `+0.0` row where the target is
    `ignore_index`.

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
    var vbody = v - v % HOST_FW
    for i in range(n):
        var base = i * v
        var mx = bitcast[DType.float32](CE_NEG_INF_BITS)
        for vv in range(v):
            mx = identical_fmax(mx, lp.unsafe_load(base + vv))
        var mxv = F32V(ftz(mx))
        var jv = 0
        while jv < vbody:
            var sv = ftz_lanes(ftz_lanes(lp.unsafe_load[width=HOST_FW](base + jv)) - mxv)
            shp.unsafe_store[width=HOST_FW](base + jv, sv)
            exp_p.unsafe_store[width=HOST_FW](base + jv, expf_lanes(sv))
            jv += HOST_FW
        while jv < v:
            var s = ftz(ftz(lp.unsafe_load(base + jv)) - ftz(mx))
            shp.unsafe_store(base + jv, s)
            exp_p.unsafe_store(base + jv, identical_exp(s))
            jv += 1
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
