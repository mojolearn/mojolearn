# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SPLIT-K ORDER OF THE SEQUENCE GEMM (lane ag-pass6-gemm, 2026-10-19;
PASS6_IDEAS idea B). BIT VERSION, IDENTICAL, every column together.

Every route of OP_GEMM, OP_GEMM_EPI (the hand MLP) and OP_AGK_GEMM_EPI (the
tape) used to fold a cell's K terms in ONE dependent chain: a product with
few cells and a long K (a matrix-vector product, a weight gradient over a
batch) ran on M N threads, each a K-long chain, and could not fill either
GPU (dW2 256 x 1 x 512: 256 threads on a 142-SM / 304-CU device; dW1 at a
65536-row batch: 16384 cells x a 65536-term chain).

For the products `gemm_split_shape` selects, the order is now:

    R = gemm_split_block(K), S = ceil(K / R)
    p_s = +0.0; p_s = fma3(A(m, k), B(k, n), p_s), k ascending over
          [s R, min(K, (s + 1) R))                    (gemm_dot's chain)
    acc = (accumulate ? C[m, n] : +0.0); acc = add(acc, p_s), s ascending
    C[m, n] = the epilogue of acc (op_gemm_epi's follower, gemm_epi_value)

which is `OP_GEMM_SPLITK`'s two stages (the IDENTICAL recurrent weight
gradients' blocked order, sequence/recurrent.mojo `wgrad_blocked`) with
the block size a function of K alone and the epilogue applied to the
folded accumulator. The rule reads the shape only, so the host column
(`HostExec`, this file's `op_gk_split`) and every device column
(`DeviceExec`, sequence/gemm_tiled.mojo `seq_gemm_split_kernel`, one
launch, the partials in threadgroup memory) choose the same order and
store the same words; the bits move once, on NVIDIA, AMD, Apple and the
host together. The column sums of OP_AGK_GEMM_EPI (i11 = 1, the fused
linear node's bias gradient) are not split: they stay OP_COLSUM_DIV's
chain, k ascending from +0.0 (OP_GK_SPLIT with i11 = 1 runs them in its
own launch on the device, extra blocks one thread a column; the host runs
OP_COLSUM_DIV after the cells), the same word as every other route.

-D MOJOLEARN_IDN_GEMM_SPLITK_OFF (or MOJOLEARN_IDN_ALL_OFF) restores the
one chain a cell on every column (arm A: today's digests). FAST keeps its
own routes (no split here)."""
from std.sys.compile import is_defined

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.ops import FP, Args, OP_GEMM, OP_GEMM_EPI, add, gemm_dot, ld, st
from sequence.mlp import gemm_epi_value
from autograd.ops import OP_AGK_GEMM_EPI

comptime GEMM_SPLITK = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_GEMM_SPLITK_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

#: the split opcode (pass-6 gemm range 2100..2199, registered in
#: sequence/dispatch.mojo): element t = one output cell, the whole split
#: order above. The executors launch it in place of the three GEMM opcodes
#: when `gemm_split_on` holds; the GEMM check launches it directly.
comptime OP_GK_SPLIT = 2100
comptime OP_GK_FIRST = OP_GK_SPLIT
comptime OP_GK_LAST = OP_GK_SPLIT

#: the shortest block: one gemm_dot round of 64 staged terms
#: (GEMM_STAGE_V2_WIDE), so every block but the last is one load round
comptime SK_MIN_BLOCK = 64
#: the most blocks a cell: the device folds a cell's partials inside one
#: 256-thread block (threadgroup memory), so S <= 256
comptime SK_MAX_S = 256
#: the reuse edge: an output edge under 8 gives a tile no operand reuse
#: (GT_MIN_EDGE's reasoning), so splitting K loses nothing to the tiles
comptime SK_REUSE_EDGE = 8
#: occupancy bounds, one per tile family (cost, not a data shape). The
#: split costs 2 K loads a cell with no operand reuse; it pays where the
#: product's own route leaves the device idle:
#:  - a thin product (an edge under SK_REUSE_EDGE: one thread a cell or the
#:    thin tile, no reuse to lose) below 2^15 cells: under 2^15 threads (less
#:    than half a wave64 per SIMD on a 304-CU part, two warps a scheduler on
#:    a 142-SM part) a chain's load rounds cannot be hidden;
#:  - an edge under SK_WIDE_EDGE (the 16 x 16 tile family) below 2^14 cells:
#:    64 tiles, under half a 256-thread block per multiprocessor on either
#:    GPU, so the tile's reuse cannot pay for the idle device;
#:  - both edges >= SK_WIDE_EDGE (the 64 x 64 tile family) below
#:    SK_WIDE_BLOCKS 64 x 64 tiles: under 8 busy multiprocessors of 142 /
#:    304, each folding the whole K. The family is the shape's, never the
#:    -D MOJOLEARN_IDN_SEQ_GEMM_MICRO_OFF arm's, so that define moves no bit.
#: Above every bound the tiles' reuse costs less than the split's loads
#: (an estimate the A/B decides: MOJOLEARN_IDN_GEMM_SPLITK_OFF).
comptime SK_THIN_CELLS = 1 << 15
comptime SK_TILE_CELLS = 1 << 14
comptime SK_WIDE_EDGE = 64
comptime SK_WIDE_BLOCKS = 8


@always_inline
def gemm_split_block(K: Int) -> Int:
    """Terms per block: the smallest power of two R >= SK_MIN_BLOCK with
    R SK_MAX_S >= K (so at most SK_MAX_S blocks). A function of K alone."""
    var r = SK_MIN_BLOCK
    while r * SK_MAX_S < K:
        r *= 2
    return r


@always_inline
def gemm_split_parts(K: Int) -> Int:
    """S = ceil(K / gemm_split_block(K)), 1 <= S <= SK_MAX_S for K >= 1."""
    var r = gemm_split_block(K)
    return (K + r - 1) // r


@always_inline
def gemm_split_lanes(S: Int) -> Int:
    """Threads the device gives one cell: the smallest power of two >= S
    (<= SK_MAX_S), so a 256-thread block holds 256 / lanes whole cells."""
    var w = 1
    while w < S:
        w *= 2
    return w


@always_inline
def gemm_split_shape(M: Int, N: Int, K: Int) -> Bool:
    """True when an M x N x K product takes the split order (the cost rule
    above; shape only, so host and device agree)."""
    comptime if not GEMM_SPLITK:
        return False
    if M <= 0 or N <= 0 or K <= SK_MIN_BLOCK:
        return False
    var cells = M * N
    if M < SK_REUSE_EDGE or N < SK_REUSE_EDGE:
        return cells < SK_THIN_CELLS
    if M < SK_WIDE_EDGE or N < SK_WIDE_EDGE:
        return cells < SK_TILE_CELLS
    var tiles = ((M + SK_WIDE_EDGE - 1) // SK_WIDE_EDGE) * ((N + SK_WIDE_EDGE - 1) // SK_WIDE_EDGE)
    return tiles < SK_WIDE_BLOCKS


@always_inline
def gemm_split_on[OP: Int](a: Args) -> Bool:
    """The rule for one launch of OP_GEMM, OP_GEMM_EPI or OP_AGK_GEMM_EPI
    (i0 = M, i1 = N, i2 = K). The epilogue opcodes index a dense C (i8 = i1);
    any other layout keeps its chain."""
    comptime assert OP == OP_GEMM or OP == OP_GEMM_EPI or OP == OP_AGK_GEMM_EPI, "gemm_split_on: a GEMM opcode"
    comptime if not GEMM_SPLITK:
        return False
    comptime if OP != OP_GEMM:
        if a.i8 != a.i1:
            return False
    return gemm_split_shape(a.i0, a.i1, a.i2)


@always_inline
def gemm_split_args[OP: Int](a: Args) -> Args:
    """OP_GK_SPLIT's arguments for a launch of OP: the GEMM's own (i0..i8,
    p0..p4, i9 / i10 the epilogue, f0 / f1, and for OP_AGK_GEMM_EPI i11 the
    column sums into p4); OP_GEMM carries no epilogue and no column sums,
    so its i9..i11 are cleared (op_gemm never reads them; FAST VAR's coop
    flag lives in i11), and OP_GEMM_EPI has no column sums."""
    var g = a
    comptime if OP == OP_GEMM:
        g.i9 = 0
        g.i10 = 0
    comptime if OP != OP_AGK_GEMM_EPI:
        g.i11 = 0
    return g


@always_inline
def gemm_split_partial(
    pa: FP, sam: Int, sak: Int, pb: FP, sbk: Int, sbn: Int, K: Int, m: Int, n: Int, s: Int, R: Int,
) -> Float32:
    """Block s of cell (m, n) of A(m, k) = pa[m sam + k sak], B(k, n) =
    pb[k sbk + n sbn]: gemm_dot's chain from +0.0 over the terms
    [s R, min(K, (s + 1) R)). The device kernel and the host body both
    call this, so a partial is the same word on every column."""
    var k0 = s * R
    var kc = min(K, k0 + R) - k0
    return gemm_dot(pa, m * sam + k0 * sak, sak, pb, n * sbn + k0 * sbk, sbk, kc, Float32(0.0))


def op_gk_split(t: Int, a: Args):
    """OP_GK_SPLIT, cell t = m N + n: the split order (module docstring),
    then gemm_epi_value (i9 = epi, i10 = its argument, p3, f0, f1) and one
    store at C[m i8 + n]. The host column's body (i11's column sums are
    HostExec's OP_COLSUM_DIV launch after the cells); the device runs
    sequence/gemm_tiled.mojo seq_gemm_split_kernel (the same partials and
    the same ascending fold)."""
    var N = a.i1
    var m = t // N
    var n = t - m * N
    var ci = m * a.i8 + n
    var R = gemm_split_block(a.i2)
    var S = (a.i2 + R - 1) // R
    var acc = Float32(0.0)
    if a.i7 != 0:
        acc = ld(a.p2, ci)
    for s in range(S):
        acc = add(acc, gemm_split_partial(a.p0, a.i3, a.i4, a.p1, a.i5, a.i6, a.i2, m, n, s, R))
    st(a.p2, ci, gemm_epi_value(acc, ci, n, a.i9, a.i10, a.p3, a.f0, a.f1))
