# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MOJOLEARN_IDN_NORM_FIXED_LANES (lane norm-folds, 2026-10-28; idea 12 of
the Oct 28 neural report, roadmap C4b / D10). IDENTICAL only, default OFF.

BITS CHANGE across versions (allowed: every column moves together in one
commit). The transformer/LM RMSNorm row statistic (contract S1, `sum x_j^2`),
the norm variants' LayerNorm mean and squared-deviation sums, and the
RMSNorm backward dot (`c = sum dh_j x_j`) stop being ONE ascending chain
of d_model terms and become ONE FIXED 32-LANE FOLD:

    slot s (0 <= s < 32) folds j = s, s + 32, s + 64, ... ascending from
    +0.0 with the site's own operation (`ftz(fma(a_j, b_j, acc))` for S1
    and the dot, `ftz(acc + x_j)` for a plain sum); then the pairwise tree
    at offsets 16, 8, 4, 2, 1 (`s[l] = ftz(s[l] + s[l + off])`, l < off);
    s[0] is the row total.

That is the fold order of sequence/fold32.mojo (`tree32`, the series fits
and the IDENTICAL sequence LayerNorm since 2026-10-10), so the library has
one row-fold definition. It is spelled here, not imported, to keep the
transformer packages free of a sequence-package dependency; the statements
are the same.

The order never depends on the vendor, the warp width or the launch: the
device cell (`nfl_cell_*`, one 32-thread cell per row; `shuffle_xor` with
strides below 32 never leaves a 32-lane half, so a 64-wide CDNA wave holds
two independent cells) and the one-thread spelling (`nfl_dot`, `nfl_sum`,
`nfl_square`: the host oracles, the byte-LM host kernels and the one-thread
device kernels that keep their geometry) run the same 32 slot chains and
the same tree. NVIDIA == AMD == Apple == host by construction.

Cost reasoning (no shape rule): the one-thread row chain is d_model
dependent FMAs per row (~4 d_model cycles of latency) while the row's
4 d_model bytes need only ~d_model/32 coalesced loads per lane; at any
d_model >= 32 the chain, not memory, bounds the kernel. The cell cuts the
dependent chain to ceil(d_model / 32) + 5 and makes every row load
coalesced. Thirty-two lanes, not 64 or a block, so a cell is one warp /
simdgroup / half-wave on every vendor with no shared memory or barrier and
the tree shape never depends on the row width.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_mul_add

comptime IDN_NORM_FIXED_LANES = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_NORM_FIXED_LANES"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
#: logical lanes of the fold (the tree has log2(NFL_LANES) = 5 levels)
comptime NFL_LANES = 32
#: threads a row-statistic kernel launches per token row: one 32-lane cell
#: under the define, the one-thread row owner otherwise (arm B unchanged)
comptime NORM_ROW_LANES = NFL_LANES if IDN_NORM_FIXED_LANES else 1


@always_inline
def norm_row_threads(rows: Int) -> Int:
    """Threads a row kernel needs for `rows` rows (rows itself when off)."""
    return rows * NORM_ROW_LANES


@always_inline
def nfl_cell(rows: Int) -> Tuple[Int, Int, Bool]:
    """(row, lane, live) of this thread in a cell launch. A cell is 32
    consecutive threads; the launch block is a multiple of 32, so a cell
    never straddles a warp and its lanes leave together when not live."""
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var row = g // NFL_LANES
    return (row, g - row * NFL_LANES, row < rows)


@always_inline
def _tree(mut s: InlineArray[Float32, NFL_LANES]) -> Float32:
    """The slots' pairwise tree, serially (sequence/fold32.mojo `tree32`)."""
    comptime for e in range(5):
        comptime off = NFL_LANES >> (e + 1)
        comptime for l in range(off):
            s[l] = ftz(s[l] + s[l + off])
    return s[0]


def nfl_dot(
    a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin],
    abase: Int, bbase: Int, width: Int,
) -> Float32:
    """sum a[abase + j] b[bbase + j], j < width, as the fixed 32-lane fold:
    slot j % 32 runs `ftz(fma(ftz(a_j), ftz(b_j), acc))` ascending from +0.0,
    then the tree. One thread; the same bits as `nfl_cell_dot`."""
    var p = InlineArray[Float32, NFL_LANES](fill=Float32(0.0))
    var k = 0
    while k < width:  # device-loop: the fold's 32-wide strides over one row
        var n = min(NFL_LANES, width - k)
        for s in range(n):
            p[s] = ftz(identical_mul_add(
                ftz(a.unsafe_load(abase + k + s)), ftz(b.unsafe_load(bbase + k + s)), p[s]))
        k += NFL_LANES
    return _tree(p)


def nfl_sum(x: MutPointer[Float32, MutAnyOrigin], base: Int, width: Int) -> Float32:
    """sum x[base + j] as the fixed fold (slot op `ftz(acc + ftz(x_j))`)."""
    var p = InlineArray[Float32, NFL_LANES](fill=Float32(0.0))
    var k = 0
    while k < width:  # device-loop: the fold's 32-wide strides over one row
        var n = min(NFL_LANES, width - k)
        for s in range(n):
            p[s] = ftz(p[s] + ftz(x.unsafe_load(base + k + s)))
        k += NFL_LANES
    return _tree(p)


def nfl_square(
    x: MutPointer[Float32, MutAnyOrigin], base: Int, width: Int, mean: Float32,
) -> Float32:
    """sum (x[base + j] - mean)^2 as the fixed fold: slot op
    `d = ftz(ftz(x_j) - mean); acc = ftz(fma(d, d, acc))`."""
    var p = InlineArray[Float32, NFL_LANES](fill=Float32(0.0))
    var k = 0
    while k < width:  # device-loop: the fold's 32-wide strides over one row
        var n = min(NFL_LANES, width - k)
        for s in range(n):
            var d = ftz(ftz(x.unsafe_load(base + k + s)) - mean)
            p[s] = ftz(identical_mul_add(d, d, p[s]))
        k += NFL_LANES
    return _tree(p)


@always_inline
def nfl_cell_total(v: Float32) -> Float32:
    """The tree on a 32-lane cell: level h (16, 8, 4, 2, 1) gives lane l
    `ftz(t_l + t_(l^h))`; for l < h that is the tree's `s[l] + s[l + h]`,
    for l >= h the same two operands (IEEE add commutes), so after five
    levels every lane holds s[0]. Strides below 32 stay in a 32-lane half."""
    var t = v
    comptime for lv in range(5):
        comptime h = NFL_LANES >> (lv + 1)
        t = ftz(t + shuffle_xor(t, UInt32(h)))
    return t


@always_inline
def nfl_cell_dot(
    a: MutPointer[Float32, MutAnyOrigin], b: MutPointer[Float32, MutAnyOrigin],
    abase: Int, bbase: Int, width: Int, lane: Int,
) -> Float32:
    """`nfl_dot` on a cell: lane `lane` is slot `lane` (coalesced loads),
    then the shuffle tree; the total on every lane."""
    var acc = Float32(0.0)
    var j = lane
    while j < width:  # device-loop: this lane's slot of one row
        acc = ftz(identical_mul_add(ftz(a.unsafe_load(abase + j)), ftz(b.unsafe_load(bbase + j)), acc))
        j += NFL_LANES
    return nfl_cell_total(acc)
