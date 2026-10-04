# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The MoE block on Apple FAST: register-tiled expert products, a tiled
router and the expert grouping on the device (lane apple-fast-moespeed,
2026-10-03).

WHERE THE TIME WENT (M3 board, FAST, synthetic 8,192 x 1,024, E 8, F 2,816,
top-2: 788.7 ms against torch-eager-bf16 37.7 ms). The two expert products
are 94 GFLOP; `moe_hidden_tiled_kernel` / `moe_out_tiled_kernel`
(sequence/moe_tiled.mojo) give each thread ONE output cell of a 16 x 8 tile,
so every fma reads two words from threadgroup memory (the x word and the
weight word): about 120 GFLOP/s. The router (`op_moe_route`) is one thread
per token, a serial 8 x 1,024 chain reading the token's x row strided across
the threads. Between the two, the entry synchronised, downloaded the picks,
grouped the pairs by expert on the host and uploaded four arrays.

MOJOLEARN_MOE_REGTILE (FAST + Apple): each thread holds a 4 x 4 block of
cells (the hidden product 4 x 4 gate AND 4 x 4 up chains), a block 64 pairs x
64 outputs of one expert, the operands staged RK = 16 at a time, so a slab
step is 8 (12) threadgroup reads for 16 (32) fmas. The router's logits are a
tiled kernel (one thread per (token, expert), the token block's x slab
shared), its softmax and top-k one thread per token as before. EVERY CELL'S
CHAIN IS THE ITEM'S: the same `ld` operands in ascending order through the
same `fma3` from +0.0; a short tail slab runs only its `cnt` terms. So the
words equal the tiled kernels' on the same build: no bit moves.

MOJOLEARN_MOE_DEVGROUP (with REGTILE): the pairs are grouped by expert on the
device (counts by atomic add, the offsets by one thread per expert, each pair
its slot by atomic add), so the forward has no host round trip. The slot
order inside an expert varies run to run; no cell's chain reads another
pair, so the words do not. The products' grids are the upper bound
(T k / 64 + E) x tiles; a block past its expert's pairs returns at once.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.atomic import Atomic
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL, ftz, identical_div, identical_exp, identical_silu
from sequence.ops import FP, add, fma3, ld, mul, st, sub

# FAST + Apple default since the M3 A/B (lane/apple-fast-moespeed 2813b2dff,
# n=1, synthetic, output digest identical): REGTILE 789 -> 72.7 ms;
# REGTILE+DEVGROUP 73.4 -> 71.9 ms, and DEVGROUP drops the host sync/sort
# round trip. MOJOLEARN_MOE_REGTILE_OFF turns both off (DEVGROUP needs
# REGTILE); MOJOLEARN_MOE_DEVGROUP_OFF turns DEVGROUP alone off. The old
# -D MOJOLEARN_MOE_REGTILE / MOJOLEARN_MOE_DEVGROUP names are harmless.
# lane/idn-gates (2026-10-04): REGTILE (and DEVGROUP with it) is also the
# IDENTICAL default on every vendor: every cell's chain is the item's (the
# same flushed operands, ascending, through the same fma3 from +0.0) and the
# route tail is the item's statements on the stored logits.
# -D MOJOLEARN_IDN_GATES_OFF (or MOJOLEARN_MOE_REGTILE_OFF) restores
# sequence/moe_tiled.mojo in IDENTICAL.
comptime MOE_REGTILE = (
    (
        (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator())
        or (GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_GATES_OFF"]())
    )
    and not is_defined["MOJOLEARN_MOE_REGTILE_OFF"]()
)
comptime MOE_DEVGROUP = MOE_REGTILE and not is_defined["MOJOLEARN_MOE_DEVGROUP_OFF"]()

#: pairs per block, outputs per block, reduction slab, threads per block
comptime RM = 64
comptime RN = 64
comptime RK = 16
comptime RT = 256
#: threads along each tile axis (RT = RS * RS), cells per thread per axis
comptime RS = 16
comptime RC = 4
#: padded threadgroup row stride (writes transpose: no bank conflicts)
comptime RMP = RM + 1
comptime RNP = RN + 1
#: the router's logits: token block x experts <= RT, slab of RLK
comptime RLK = 8


def moe_reg_blocks(n_pairs: Int, n_experts: Int, n_out: Int) -> Int:
    """The products' grid upper bound: sum over e of ceil(c_e / RM) x tiles
    <= (n_pairs / RM + E) x tiles."""
    return (n_pairs // RM + n_experts) * ((n_out + RN - 1) // RN)


@always_inline
def _block_map(poff: FP, En: Int, nq: Int, b: Int) -> Tuple[Int, Int]:
    """(expert, tile within the expert) of block b, the experts' blocks in
    expert order; (-1, 0) past the last."""
    var base = 0
    for e in range(En):
        var c = Int(poff.unsafe_load(e + 1)) - Int(poff.unsafe_load(e))
        var nb = ((c + RM - 1) // RM) * nq
        if b < base + nb:
            return (e, b - base)
        base += nb
    return (-1, 0)


def moe_hidden_reg_kernel(
    x: FP, gu: FP, order: FP, poff: FP, h: FP,
    d_model: Int32, n_ff: Int32, top_k: Int32, n_experts: Int32,
):
    """h[pair, f] = silu(g) * u, g and u the item's chains over d, a block
    RM pairs x RN features of one expert, RC x RC cells a thread."""
    var D = Int(d_model)
    var F = Int(n_ff)
    var k = Int(top_k)
    var En = Int(n_experts)
    var nq = (F + RN - 1) // RN
    var m = _block_map(poff, En, nq, Int(block_idx.x))
    var e = m[0]
    if e < 0:
        return
    var ti = m[1] // nq
    var qi = m[1] - ti * nq
    var p0 = Int(poff.unsafe_load(e)) + ti * RM
    var p1 = Int(poff.unsafe_load(e + 1))
    var q0 = qi * RN
    var tid = Int(thread_idx.x)
    var tr = tid // RS
    var tc = tid - tr * RS
    var xs = stack_allocation[RK * RMP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var gs = stack_allocation[RK * RNP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var us = stack_allocation[RK * RNP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    # the rows this thread stages: pair rows tr + RS l (their tokens), feature rows tr + RS l
    var ltok = InlineArray[Int, RC](fill=-1)
    comptime for l in range(RC):
        var rp = p0 + tr + RS * l
        if rp < p1:
            ltok[l] = Int(order.unsafe_load(rp)) // k
    var g = InlineArray[Float32, RC * RC](fill=Float32(0.0))
    var u = InlineArray[Float32, RC * RC](fill=Float32(0.0))
    var d0 = 0
    while d0 < D:
        var cnt = min(RK, D - d0)
        comptime for l in range(RC):
            var xv = Float32(0.0)
            if tc < cnt and ltok[l] >= 0:
                xv = ld(x, ltok[l] * D + d0 + tc)
            xs[tc * RMP + tr + RS * l] = xv
            var f = q0 + tr + RS * l
            var gv = Float32(0.0)
            var uv = Float32(0.0)
            if tc < cnt and f < F:
                gv = ld(gu, (e * 2 * F + f) * D + d0 + tc)
                uv = ld(gu, (e * 2 * F + F + f) * D + d0 + tc)
            gs[tc * RNP + tr + RS * l] = gv
            us[tc * RNP + tr + RS * l] = uv
        barrier()
        if cnt == RK:
            comptime for i in range(RK):
                var av = InlineArray[Float32, RC](fill=Float32(0.0))
                var gb = InlineArray[Float32, RC](fill=Float32(0.0))
                var ub = InlineArray[Float32, RC](fill=Float32(0.0))
                comptime for c in range(RC):
                    av[c] = xs[i * RMP + tr + RS * c]
                    gb[c] = gs[i * RNP + tc + RS * c]
                    ub[c] = us[i * RNP + tc + RS * c]
                comptime for a in range(RC):
                    comptime for b in range(RC):
                        g[a * RC + b] = fma3(av[a], gb[b], g[a * RC + b])
                        u[a * RC + b] = fma3(av[a], ub[b], u[a * RC + b])
        else:
            for i in range(cnt):
                var av = InlineArray[Float32, RC](fill=Float32(0.0))
                var gb = InlineArray[Float32, RC](fill=Float32(0.0))
                var ub = InlineArray[Float32, RC](fill=Float32(0.0))
                comptime for c in range(RC):
                    av[c] = xs[i * RMP + tr + RS * c]
                    gb[c] = gs[i * RNP + tc + RS * c]
                    ub[c] = us[i * RNP + tc + RS * c]
                comptime for a in range(RC):
                    comptime for b in range(RC):
                        g[a * RC + b] = fma3(av[a], gb[b], g[a * RC + b])
                        u[a * RC + b] = fma3(av[a], ub[b], u[a * RC + b])
        barrier()
        d0 += RK
    comptime for a in range(RC):
        var rp = p0 + tr + RS * a
        if rp < p1:
            var pair = Int(order.unsafe_load(rp))
            comptime for b in range(RC):
                var f = q0 + tc + RS * b
                if f < F:
                    st(h, pair * F + f, mul(ftz(identical_silu(g[a * RC + b])), u[a * RC + b]))


def moe_out_reg_kernel(
    h: FP, dn: FP, order: FP, poff: FP, s: FP,
    d_model: Int32, n_ff: Int32, n_experts: Int32,
):
    """s[pair, d] = the item's chain h[pair, :] . down[e, d, :] over f
    ascending, a block RM pairs x RN outputs of one expert."""
    var D = Int(d_model)
    var F = Int(n_ff)
    var En = Int(n_experts)
    var nq = (D + RN - 1) // RN
    var m = _block_map(poff, En, nq, Int(block_idx.x))
    var e = m[0]
    if e < 0:
        return
    var ti = m[1] // nq
    var qi = m[1] - ti * nq
    var p0 = Int(poff.unsafe_load(e)) + ti * RM
    var p1 = Int(poff.unsafe_load(e + 1))
    var q0 = qi * RN
    var tid = Int(thread_idx.x)
    var tr = tid // RS
    var tc = tid - tr * RS
    var hs = stack_allocation[RK * RMP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ws = stack_allocation[RK * RNP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var lpair = InlineArray[Int, RC](fill=-1)
    comptime for l in range(RC):
        var rp = p0 + tr + RS * l
        if rp < p1:
            lpair[l] = Int(order.unsafe_load(rp))
    var acc = InlineArray[Float32, RC * RC](fill=Float32(0.0))
    var f0 = 0
    while f0 < F:
        var cnt = min(RK, F - f0)
        comptime for l in range(RC):
            var hv = Float32(0.0)
            if tc < cnt and lpair[l] >= 0:
                hv = ld(h, lpair[l] * F + f0 + tc)
            hs[tc * RMP + tr + RS * l] = hv
            var d = q0 + tr + RS * l
            var wv = Float32(0.0)
            if tc < cnt and d < D:
                wv = ld(dn, (e * D + d) * F + f0 + tc)
            ws[tc * RNP + tr + RS * l] = wv
        barrier()
        if cnt == RK:
            comptime for i in range(RK):
                var av = InlineArray[Float32, RC](fill=Float32(0.0))
                var bv = InlineArray[Float32, RC](fill=Float32(0.0))
                comptime for c in range(RC):
                    av[c] = hs[i * RMP + tr + RS * c]
                    bv[c] = ws[i * RNP + tc + RS * c]
                comptime for a in range(RC):
                    comptime for b in range(RC):
                        acc[a * RC + b] = fma3(av[a], bv[b], acc[a * RC + b])
        else:
            for i in range(cnt):
                var av = InlineArray[Float32, RC](fill=Float32(0.0))
                var bv = InlineArray[Float32, RC](fill=Float32(0.0))
                comptime for c in range(RC):
                    av[c] = hs[i * RMP + tr + RS * c]
                    bv[c] = ws[i * RNP + tc + RS * c]
                comptime for a in range(RC):
                    comptime for b in range(RC):
                        acc[a * RC + b] = fma3(av[a], bv[b], acc[a * RC + b])
        barrier()
        f0 += RK
    comptime for a in range(RC):
        if lpair[a] >= 0:
            comptime for b in range(RC):
                var d = q0 + tc + RS * b
                if d < D:
                    s.unsafe_store(lpair[a] * D + d, acc[a * RC + b])


def moe_logits_reg_kernel(x: FP, wg: FP, logits: FP, n_tok: Int32, d_model: Int32, n_experts: Int32):
    """logits[t, e] = the item's chain x[t, :] . W_g[e, :] over d ascending,
    one thread per (t, e), a block RT // E tokens (E <= RT)."""
    var T = Int(n_tok)
    var D = Int(d_model)
    var En = Int(n_experts)
    var bt = RT // En
    var tid = Int(thread_idx.x)
    var tl = tid // En
    var e = tid - tl * En
    var t0 = Int(block_idx.x) * bt
    var t = t0 + tl
    var live = tl < bt and t < T
    var xs = stack_allocation[RT * RLK, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var wsh = stack_allocation[RT * RLK, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = Float32(0.0)
    var d0 = 0
    while d0 < D:
        var cnt = min(RLK, D - d0)
        var q = tid
        while q < bt * RLK:
            var row = q // RLK
            var col = q - row * RLK
            var v = Float32(0.0)
            if col < cnt and t0 + row < T:
                v = ld(x, (t0 + row) * D + d0 + col)
            xs[q] = v
            q += RT
        q = tid
        while q < En * RLK:
            var row = q // RLK
            var col = q - row * RLK
            var v = Float32(0.0)
            if col < cnt:
                v = ld(wg, row * D + d0 + col)
            wsh[q] = v
            q += RT
        barrier()
        if live:
            for i in range(cnt):
                acc = fma3(xs[tl * RLK + i], wsh[e * RLK + i], acc)
        barrier()
        d0 += RLK
    if live:
        st(logits, t * En + e, acc)


@always_inline
def _div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


def moe_route_tail_kernel(
    logits: FP, sel: FP, w: FP, pr: FP, n_tok: Int32, n_experts: Int32, top_k: Int32, renorm: Int32,
):
    """`op_moe_route` after its logits, one thread per token, from the stored
    logits (the same words the item kept in `s`): the max, the softmax, the
    k picks by strict greater-than (ties to the lower expert), the
    renormalisation."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(n_tok):
        return
    var E = Int(n_experts)
    var k = Int(top_k)
    var mx = Float32(0.0)
    for e in range(E):
        var s = logits.unsafe_load(t * E + e)
        if e == 0 or s > mx:
            mx = s
    var z = Float32(0.0)
    for e in range(E):
        var v = ftz(identical_exp(sub(ld(logits, t * E + e), mx)))
        st(pr, t * E + e, v)
        z = add(z, v)
    for e in range(E):
        st(pr, t * E + e, _div(ld(pr, t * E + e), z))
    var tot = Float32(0.0)
    for j in range(k):
        var best = -1
        var bv = Float32(0.0)
        for e in range(E):
            var taken = False
            for i in range(j):
                if Int(sel.unsafe_load(t * k + i)) == e:
                    taken = True
            if taken:
                continue
            var v = ld(pr, t * E + e)
            if best < 0 or v > bv:
                best = e
                bv = v
        sel.unsafe_store(t * k + j, Float32(best))
        st(w, t * k + j, bv)
        tot = add(tot, bv)
    if renorm != 0:
        for j in range(k):
            st(w, t * k + j, _div(ld(w, t * k + j), tot))


def moe_group_zero_kernel(cnt: FP, n_experts: Int32):
    """counts[0..E) and cursors[E..2E) to zero (int32 words)."""
    var i = Int(thread_idx.x)
    if i < 2 * Int(n_experts):
        cnt.bitcast[Int32]().unsafe_store(i, Int32(0))


def moe_group_count_kernel(sel: FP, cnt: FP, n_pairs: Int32):
    """counts[e] += 1 for each pair's expert, one thread per pair."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_pairs):
        var e = Int(sel.unsafe_load(i))
        _ = Atomic.fetch_add(cnt.bitcast[Int32]() + e, Int32(1))


def moe_group_offsets_kernel(cnt: FP, poff: FP, n_experts: Int32):
    """poff[e] = sum of counts below e, one thread per e (and E itself)."""
    var e = Int(thread_idx.x)
    var En = Int(n_experts)
    if e <= En:
        var ci = cnt.bitcast[Int32]()
        var s = 0
        for j in range(e):
            s += Int(ci.unsafe_load(j))
        poff.unsafe_store(e, Float32(s))


def moe_group_scatter_kernel(sel: FP, cnt: FP, poff: FP, order: FP, n_pairs: Int32, n_experts: Int32):
    """order[poff[e] + slot] = pair, the slot by atomic add on e's cursor."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_pairs):
        var e = Int(sel.unsafe_load(i))
        var slot = Atomic.fetch_add(cnt.bitcast[Int32]() + Int(n_experts) + e, Int32(1))
        order.unsafe_store(Int(poff.unsafe_load(e)) + Int(slot), Float32(i))
