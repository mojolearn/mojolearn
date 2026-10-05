# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The MoE expert products on Apple's simdgroup matrix unit (FAST + Apple,
lane apple-fast-gap-misc, 2026-10-03). Opt-in candidates, default OFF.

WHERE THE TIME GOES after REGTILE + DEVGROUP (M3 board, synthetic T 8,192,
D 1,024, F 2,816, E 8, top-2: 72.7 ms vs torch-eager-bf16 37.7 ms). The
products are 16,384 pairs x (2F x D + D x F) = 283 GFLOP (fma = 2); the
register-tiled kernels (sequence/moe_reg.mojo) read 12 (8) threadgroup words
per 32 (16) scalar fmas, about 5 TFLOP/s. The matrix unit takes an 8 x 8 x 8
fragment product per instruction, its operands loaded per simdgroup.

MOJOLEARN_MOE_FAST_MMA: the hidden and out products as 8 x 8 fragment
products. A block is 64 pairs (of one expert, DEVGROUP's grouping) x BN
outputs, four simdgroups 2 x 2, each 32 pairs x BN / 2 outputs; the slab of
KB reduction words staged k-major for the pairs and row-major for the
weights. Each cell is still one fma chain over the reduction index
ascending (the matrix unit's per-cell order), with no per-step flush; FAST
only, quality on the board's rel_fro / max_rel_diff column.
  _KB32: 32-word slabs (half the barriers per reduction).
  _WIDE: the hidden product 64 pairs x 64 features (gate and up 32 fragments
         a simdgroup) instead of 64 x 32.
  _PF:   the next slab's global words read into registers before the current
         slab's fragment products (one slab of load latency hidden).
Every variant implies MOJOLEARN_MOE_FAST_MMA."""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.ffi import external_call
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from core.apple_air import simdgroup_load_legacy_air
from checks.numerics import ftz, identical_silu
from sequence.ops import FP, ld, mul, st
from sequence.moe_reg import MOE_DEVGROUP

#: MOE_FAST_MMA is the FAST + Apple default since M3 A/B rab10-moemma (2026-10-05, afc_ab_def, full board
#: size, 1 run per arm): moe synthetic 71.39 -> 53.48 ms (-25.1%), output digest identical.
#: `-D MOJOLEARN_MOE_FAST_MMA_OFF` turns it off. KB32/WIDE/PF geometry variants stay opt-in:
#: the bundle KB32 + WIDE + PF (rab10-moemmaall, same A/B setup) was slower, moe 71.3 -> 146.8 ms;
#: recorded as a bundle loss (no single-variant A/B yet).
# See docs/apple-fast/EXPERIMENTS.md (MOE_FAST_MMA); neural lane owns validation.
comptime MM_KB32 = is_defined["MOJOLEARN_MOE_FAST_MMA_KB32"]()
comptime MM_WIDE = is_defined["MOJOLEARN_MOE_FAST_MMA_WIDE"]()
comptime MM_PF = is_defined["MOJOLEARN_MOE_FAST_MMA_PF"]()
# Apple simdgroup intrinsics only: an NVIDIA/AMD target cannot link them
# (gfx942 lld: undefined air.simdgroup_matrix_*; box-run-2 compile fix).
comptime MOE_MMA = has_apple_gpu_accelerator() and MOE_DEVGROUP and (
    (not is_defined["MOJOLEARN_MOE_FAST_MMA_OFF"]()) or MM_KB32 or MM_WIDE or MM_PF
)

comptime MM_KB = 32 if MM_KB32 else 16
comptime MM_SGM = 2
comptime MM_SGN = 2
comptime MM_FM = 4
comptime MM_NT = 32 * MM_SGM * MM_SGN
comptime MM_BM = 8 * MM_FM * MM_SGM
#: fragments per simdgroup along the outputs: hidden (gate and up each), out
comptime MM_FNH = 4 if MM_WIDE else 2
comptime MM_FNO = 4
comptime MM_BNH = 8 * MM_FNH * MM_SGN
comptime MM_BNO = 8 * MM_FNO * MM_SGN
#: padded strides: pairs k-major (BM + 4), weights row-major (KB + 4)
comptime MM_AST = MM_BM + 4
comptime MM_BST = MM_KB + 4

comptime _M64 = SIMD[DType.float32, 64]
comptime _V2 = SIMD[DType.int64, 2]


@always_inline
def _mm_load_t(
    p: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    stride: Int,
) -> _M64:
    """Fragment M[r][c] = p[c * stride + r] (gemm_identical `_amma_load_t`)."""
    comptime if simdgroup_load_legacy_air():
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, Int64(stride), _V2(0, 0), True
        )
    else:
        return external_call["air.simdgroup_matrix_8x8_load.v64f32.p3f32", _M64](
            p, _V2(Int64(stride), 8), _V2(Int64(stride), 1), _V2(0, 0)
        )


@always_inline
def _mm_mma(a: _M64, b: _M64, c: _M64) -> _M64:
    return external_call[
        "air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
        _M64,
    ](a, b, c)


def moe_mma_blocks(n_pairs: Int, n_experts: Int, n_out: Int, bn: Int) -> Int:
    """Grid upper bound: sum over e of ceil(c_e / BM) x tiles."""
    return (n_pairs // MM_BM + n_experts) * ((n_out + bn - 1) // bn)


@always_inline
def _mm_block_map(poff: FP, En: Int, nq: Int, b: Int) -> Tuple[Int, Int]:
    var base = 0
    for e in range(En):
        var c = Int(poff.unsafe_load(e + 1)) - Int(poff.unsafe_load(e))
        var nb = ((c + MM_BM - 1) // MM_BM) * nq
        if b < base + nb:
            return (e, b - base)
        base += nb
    return (-1, 0)


@always_inline
def _mm_gload[N: Int](src: FP, rows: InlineArray[Int, N], k0: Int, K: Int, tid: Int) -> InlineArray[Float32, N]:
    """Thread tid's words of a slab: item j is row (tid + NT j) // KB, column
    k0 + (tid + NT j) % KB of `src` (row offsets in `rows`, -1 = a zero row)."""
    var out = InlineArray[Float32, N](fill=Float32(0.0))
    comptime for j in range(N):
        var q = tid + MM_NT * j
        var r = q // MM_KB
        var c = k0 + q - r * MM_KB
        if rows[j] >= 0 and c < K:
            out[j] = ld(src, rows[j] + c)
    return out^


def moe_hidden_mma_kernel(
    x: FP, gu: FP, order: FP, poff: FP, h: FP,
    d_model: Int32, n_ff: Int32, top_k: Int32, n_experts: Int32,
):
    """h[pair, f] = silu(g) * u over a block 64 pairs x MM_BNH features of
    one expert, g and u fragment products over d."""
    comptime FN = MM_FNH
    comptime BN = MM_BNH
    comptime NX = MM_BM * MM_KB // MM_NT
    comptime NW = BN * MM_KB // MM_NT
    var D = Int(d_model)
    var F = Int(n_ff)
    var k = Int(top_k)
    var En = Int(n_experts)
    var nq = (F + BN - 1) // BN
    var m = _mm_block_map(poff, En, nq, Int(block_idx.x))
    var e = m[0]
    if e < 0:
        return
    var ti = m[1] // nq
    var qi = m[1] - ti * nq
    var p0 = Int(poff.unsafe_load(e)) + ti * MM_BM
    var p1 = Int(poff.unsafe_load(e + 1))
    var q0 = qi * BN
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // MM_SGN
    var sgn = sg % MM_SGN
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var at = stack_allocation[MM_KB * MM_AST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var gt = stack_allocation[BN * MM_BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ut = stack_allocation[BN * MM_BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var xrow = InlineArray[Int, NX](fill=-1)
    comptime for j in range(NX):
        var rp = p0 + (tid + MM_NT * j) // MM_KB
        if rp < p1:
            xrow[j] = (Int(order.unsafe_load(rp)) // k) * D
    var grow = InlineArray[Int, NW](fill=-1)
    var urow = InlineArray[Int, NW](fill=-1)
    comptime for j in range(NW):
        var f = q0 + (tid + MM_NT * j) // MM_KB
        if f < F:
            grow[j] = (e * 2 * F + f) * D
            urow[j] = (e * 2 * F + F + f) * D
    var accg = InlineArray[_M64, MM_FM * FN](fill=_M64(0))
    var accu = InlineArray[_M64, MM_FM * FN](fill=_M64(0))
    var xv = _mm_gload[NX](x, xrow, 0, D, tid)
    var gv = _mm_gload[NW](gu, grow, 0, D, tid)
    var uv = _mm_gload[NW](gu, urow, 0, D, tid)
    var d0 = 0
    while d0 < D:
        comptime for j in range(NX):
            var q = tid + MM_NT * j
            var r = q // MM_KB
            at[(q - r * MM_KB) * MM_AST + r] = xv[j]
        comptime for j in range(NW):
            var q = tid + MM_NT * j
            var r = q // MM_KB
            gt[r * MM_BST + q - r * MM_KB] = gv[j]
            ut[r * MM_BST + q - r * MM_KB] = uv[j]
        barrier()
        var d1 = d0 + MM_KB
        comptime if MM_PF:
            if d1 < D:
                xv = _mm_gload[NX](x, xrow, d1, D, tid)
                gv = _mm_gload[NW](gu, grow, d1, D, tid)
                uv = _mm_gload[NW](gu, urow, d1, D, tid)
        comptime for p8 in range(MM_KB // 8):
            var af = InlineArray[_M64, MM_FM](fill=_M64(0))
            comptime for fm in range(MM_FM):
                af[fm] = _mm_load_t(at + (8 * p8) * MM_AST + (sgm * MM_FM + fm) * 8, MM_AST)
            comptime for fq in range(FN):
                var gf = _mm_load_t(gt + ((sgn * FN + fq) * 8) * MM_BST + 8 * p8, MM_BST)
                var uf = _mm_load_t(ut + ((sgn * FN + fq) * 8) * MM_BST + 8 * p8, MM_BST)
                comptime for fm in range(MM_FM):
                    accg[fm * FN + fq] = _mm_mma(af[fm], gf, accg[fm * FN + fq])
                    accu[fm * FN + fq] = _mm_mma(af[fm], uf, accu[fm * FN + fq])
        barrier()
        comptime if not MM_PF:
            if d1 < D:
                xv = _mm_gload[NX](x, xrow, d1, D, tid)
                gv = _mm_gload[NW](gu, grow, d1, D, tid)
                uv = _mm_gload[NW](gu, urow, d1, D, tid)
        d0 = d1
    comptime for fm in range(MM_FM):
        var rp = p0 + (sgm * MM_FM + fm) * 8 + frow
        if rp < p1:
            var pair = Int(order.unsafe_load(rp))
            comptime for fq in range(FN):
                comptime for c2 in range(2):
                    var f = q0 + (sgn * FN + fq) * 8 + fcol + c2
                    if f < F:
                        var g = ftz(accg[fm * FN + fq][c2])
                        var u = ftz(accu[fm * FN + fq][c2])
                        st(h, pair * F + f, mul(ftz(identical_silu(g)), u))


def moe_out_mma_kernel(
    h: FP, dn: FP, order: FP, poff: FP, s: FP,
    d_model: Int32, n_ff: Int32, n_experts: Int32,
):
    """s[pair, d] = h[pair, :] . down[e, d, :] over a block 64 pairs x
    MM_BNO outputs of one expert, fragment products over f."""
    comptime FN = MM_FNO
    comptime BN = MM_BNO
    comptime NX = MM_BM * MM_KB // MM_NT
    comptime NW = BN * MM_KB // MM_NT
    var D = Int(d_model)
    var F = Int(n_ff)
    var En = Int(n_experts)
    var nq = (D + BN - 1) // BN
    var m = _mm_block_map(poff, En, nq, Int(block_idx.x))
    var e = m[0]
    if e < 0:
        return
    var ti = m[1] // nq
    var qi = m[1] - ti * nq
    var p0 = Int(poff.unsafe_load(e)) + ti * MM_BM
    var p1 = Int(poff.unsafe_load(e + 1))
    var q0 = qi * BN
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // MM_SGN
    var sgn = sg % MM_SGN
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var at = stack_allocation[MM_KB * MM_AST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var wt = stack_allocation[BN * MM_BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var hrow = InlineArray[Int, NX](fill=-1)
    comptime for j in range(NX):
        var rp = p0 + (tid + MM_NT * j) // MM_KB
        if rp < p1:
            hrow[j] = Int(order.unsafe_load(rp)) * F
    var wrow = InlineArray[Int, NW](fill=-1)
    comptime for j in range(NW):
        var d = q0 + (tid + MM_NT * j) // MM_KB
        if d < D:
            wrow[j] = (e * D + d) * F
    var acc = InlineArray[_M64, MM_FM * FN](fill=_M64(0))
    var hv = _mm_gload[NX](h, hrow, 0, F, tid)
    var wv = _mm_gload[NW](dn, wrow, 0, F, tid)
    var f0 = 0
    while f0 < F:
        comptime for j in range(NX):
            var q = tid + MM_NT * j
            var r = q // MM_KB
            at[(q - r * MM_KB) * MM_AST + r] = hv[j]
        comptime for j in range(NW):
            var q = tid + MM_NT * j
            var r = q // MM_KB
            wt[r * MM_BST + q - r * MM_KB] = wv[j]
        barrier()
        var f1 = f0 + MM_KB
        comptime if MM_PF:
            if f1 < F:
                hv = _mm_gload[NX](h, hrow, f1, F, tid)
                wv = _mm_gload[NW](dn, wrow, f1, F, tid)
        comptime for p8 in range(MM_KB // 8):
            var af = InlineArray[_M64, MM_FM](fill=_M64(0))
            comptime for fm in range(MM_FM):
                af[fm] = _mm_load_t(at + (8 * p8) * MM_AST + (sgm * MM_FM + fm) * 8, MM_AST)
            comptime for fq in range(FN):
                var bf = _mm_load_t(wt + ((sgn * FN + fq) * 8) * MM_BST + 8 * p8, MM_BST)
                comptime for fm in range(MM_FM):
                    acc[fm * FN + fq] = _mm_mma(af[fm], bf, acc[fm * FN + fq])
        barrier()
        comptime if not MM_PF:
            if f1 < F:
                hv = _mm_gload[NX](h, hrow, f1, F, tid)
                wv = _mm_gload[NW](dn, wrow, f1, F, tid)
        f0 = f1
    comptime for fm in range(MM_FM):
        var rp = p0 + (sgm * MM_FM + fm) * 8 + frow
        if rp < p1:
            var pair = Int(order.unsafe_load(rp))
            comptime for fq in range(FN):
                comptime for c2 in range(2):
                    var d = q0 + (sgn * FN + fq) * 8 + fcol + c2
                    if d < D:
                        st(s, pair * D + d, acc[fm * FN + fq][c2])
