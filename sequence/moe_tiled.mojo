# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The MoE block's two products as TILED device kernels with the items'
chains (lane neural-pass29, 2026-10-01).

`op_moe_hidden` and `op_moe_out` (sequence/moe.mojo) are one thread per
output cell, each thread a serial fma chain over the reduction axis that
reads its operands straight from device memory: no reuse of an x row across
the features it meets nor of a weight row across the tokens that pick its
expert, so at the board's shape (8,192 tokens, 1,024 wide, 8 experts of
2,816, top-2) the two products read about 750 GB for 94 GFLOP and take 957 ms
on the L40S and 6.3 s on the M4. These kernels tile them: the tokens are
grouped by expert on the host after the route (`order`, the pair indices
`tok * k + j` sorted by expert; `poff`, each expert's first pair; `boff`,
each expert's first block), a block holds TILE_P pairs x TILE_Q outputs of
ONE expert, and the operands come through threadgroup memory a slab of
TILE_K at a time. EVERY CELL'S CHAIN IS THE ITEM'S: the same `ld` (flushed)
operands, in the same ascending order, through the same `fma3`, started from
the same `+0.0`; the slabs only change where the operands are read from.
`op_moe_out`'s weighted sum over the picks is `moe_combine_kernel`, the
item's own `y = fma3(w_j, s_j, y)` in pick order from the stored chains.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from std.sys.compile import is_defined
from checks.numerics import ftz, identical_silu, GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# NN44 independent arm: integer upper-bound search, including empty experts.
# OFF, unmeasured. O(log E) lookup reduces scheduling overhead as E grows;
# no floating arithmetic, routing decision, or combine order changes.
comptime NN44_EXPERT_BISECT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN44_EXPERT_BISECT"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
from sequence.ops import FP, fma3, ld, mul, st

#: pairs (token, pick) per block, outputs (features or d) per block, and the
#: reduction slab: TILE_P * TILE_Q threads a block
comptime TILE_P = 16
comptime TILE_Q = 8
comptime TILE_K = 32
comptime MOE_TPB = TILE_P * TILE_Q


@always_inline
def _block_expert(boff: FP, n_experts: Int, b: Int) -> Int:
    """The expert whose blocks hold block b (`boff[e] <= b < boff[e + 1]`)."""
    comptime if NN44_EXPERT_BISECT:
        var lo = 0
        var hi = n_experts
        while lo + 1 < hi:
            var mid = (lo + hi) // 2
            if Int(boff.unsafe_load(mid)) <= b:
                lo = mid
            else:
                hi = mid
        return lo
    var e = 0
    while e + 1 < n_experts and Int(boff.unsafe_load(e + 1)) <= b:
        e += 1
    return e


def moe_hidden_tiled_kernel(
    x: FP, gu: FP, order: FP, poff: FP, boff: FP, h: FP,
    d_model: Int32, n_ff: Int32, top_k: Int32, n_experts: Int32, n_ftiles: Int32,
):
    """`op_moe_hidden` over a block of TILE_P pairs x TILE_Q features of one
    expert: h[p, f] = silu(g) * u with g, u the chains over d."""
    var D = Int(d_model)
    var F = Int(n_ff)
    var k = Int(top_k)
    var En = Int(n_experts)
    var nF = Int(n_ftiles)
    var b = Int(block_idx.x)
    var e = _block_expert(boff, En, b)
    var tile = b - Int(boff.unsafe_load(e))
    var ti = tile // nF
    var fi = tile - ti * nF
    var p0 = Int(poff.unsafe_load(e))
    var p1 = Int(poff.unsafe_load(e + 1))
    var tid = Int(thread_idx.x)
    var r = tid // TILE_Q
    var c = tid - r * TILE_Q
    var pidx = p0 + ti * TILE_P + r
    var pair = -1
    if pidx < p1:
        pair = Int(order.unsafe_load(pidx))
    var tok = pair // k if pair >= 0 else 0
    var f = fi * TILE_Q + c
    var f_ok = f < F
    var xs = stack_allocation[TILE_P * TILE_K, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var gws = stack_allocation[TILE_Q * TILE_K, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var uws = stack_allocation[TILE_Q * TILE_K, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var g = Float32(0.0)
    var u = Float32(0.0)
    var d0 = 0
    while d0 < D:
        var cnt = min(TILE_K, D - d0)
        # the x slab: TILE_P rows of cnt (flushed by `ld`, as the item reads them)
        var q = tid
        while q < TILE_P * TILE_K:
            var row = q // TILE_K
            var col = q - row * TILE_K
            var v = Float32(0.0)
            var rp = p0 + ti * TILE_P + row
            if col < cnt and rp < p1:
                var rt = Int(order.unsafe_load(rp)) // k
                v = ld(x, rt * D + d0 + col)
            xs[q] = v
            q += MOE_TPB
        # the gate and up slabs: TILE_Q rows of cnt
        q = tid
        while q < TILE_Q * TILE_K:
            var row = q // TILE_K
            var col = q - row * TILE_K
            var gv = Float32(0.0)
            var uv = Float32(0.0)
            var rf = fi * TILE_Q + row
            if col < cnt and rf < F:
                gv = ld(gu, (e * 2 * F + rf) * D + d0 + col)
                uv = ld(gu, (e * 2 * F + F + rf) * D + d0 + col)
            gws[q] = gv
            uws[q] = uv
            q += MOE_TPB
        barrier()
        if pair >= 0 and f_ok:
            for i in range(cnt):
                var xv = xs[r * TILE_K + i]
                g = fma3(xv, gws[c * TILE_K + i], g)
                u = fma3(xv, uws[c * TILE_K + i], u)
        barrier()
        d0 += TILE_K
    if pair >= 0 and f_ok:
        st(h, pair * F + f, mul(ftz(identical_silu(g)), u))


def moe_out_tiled_kernel(
    h: FP, dn: FP, order: FP, poff: FP, boff: FP, s: FP,
    d_model: Int32, n_ff: Int32, top_k: Int32, n_experts: Int32, n_dtiles: Int32,
):
    """`op_moe_out`'s inner chain over a block of TILE_P pairs x TILE_Q
    outputs d of one expert: s[p, d] = h[p, :] . down[e, d, :], ascending f."""
    var D = Int(d_model)
    var F = Int(n_ff)
    var En = Int(n_experts)
    var nD = Int(n_dtiles)
    var b = Int(block_idx.x)
    var e = _block_expert(boff, En, b)
    var tile = b - Int(boff.unsafe_load(e))
    var ti = tile // nD
    var di = tile - ti * nD
    var p0 = Int(poff.unsafe_load(e))
    var p1 = Int(poff.unsafe_load(e + 1))
    var tid = Int(thread_idx.x)
    var r = tid // TILE_Q
    var c = tid - r * TILE_Q
    var pidx = p0 + ti * TILE_P + r
    var pair = -1
    if pidx < p1:
        pair = Int(order.unsafe_load(pidx))
    var d = di * TILE_Q + c
    var d_ok = d < D
    var hs = stack_allocation[TILE_P * TILE_K, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dns = stack_allocation[TILE_Q * TILE_K, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var acc = Float32(0.0)
    var f0 = 0
    while f0 < F:
        var cnt = min(TILE_K, F - f0)
        var q = tid
        while q < TILE_P * TILE_K:
            var row = q // TILE_K
            var col = q - row * TILE_K
            var v = Float32(0.0)
            var rp = p0 + ti * TILE_P + row
            if col < cnt and rp < p1:
                var rpair = Int(order.unsafe_load(rp))
                v = ld(h, rpair * F + f0 + col)
            hs[q] = v
            q += MOE_TPB
        q = tid
        while q < TILE_Q * TILE_K:
            var row = q // TILE_K
            var col = q - row * TILE_K
            var v = Float32(0.0)
            var rd = di * TILE_Q + row
            if col < cnt and rd < D:
                v = ld(dn, (e * D + rd) * F + f0 + col)
            dns[q] = v
            q += MOE_TPB
        barrier()
        if pair >= 0 and d_ok:
            for i in range(cnt):
                acc = fma3(hs[r * TILE_K + i], dns[c * TILE_K + i], acc)
        barrier()
        f0 += TILE_K
    if pair >= 0 and d_ok:
        s.unsafe_store(pair * D + d, acc)


def moe_combine_kernel(w: FP, s: FP, y: FP, d_model: Int32, top_k: Int32, n_cells: Int32):
    """`op_moe_out`'s weighted sum: y[tok, d] = fma3(w_j, s_j, y) over the picks
    in pick order from +0.0, one thread per (tok, d)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(n_cells):
        return
    var D = Int(d_model)
    var k = Int(top_k)
    var tok = t // D
    var d = t - tok * D
    var yv = Float32(0.0)
    for j in range(k):
        var p = tok * k + j
        yv = fma3(ld(w, p), ld(s, p * D + d), yv)
    st(y, t, yv)
