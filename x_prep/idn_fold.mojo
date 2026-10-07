# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LANE ml-prep-nb (2026-10-04): THE LANE-TREE ORDER of three x_prep folds
that ran as ONE device thread walking every row under IDENTICAL
(`te_global`, `te_enc`, `ii_gram`; fastprep2.mojo's notes on the FAST
forms). This leaf module holds the switches and the order; the units
(x_prep/target.mojo, x_prep/iterative.mojo), the host column
(x_prep/host/target.mojo) and the device kernels (x_prep/idn_tree.mojo)
all spell it, so the four columns share one arithmetic and one order.
No accelerator import: the CPU-only host binding compiles this file.

THE ORDER (`LT`, lane tree): a sequence of terms c_r, r = 0 .. L-1 (r a
position; a skipped position adds nothing) is folded into TREE_W lanes, lane
l = r mod TREE_W taking its positions ascending from zero by `add`; then the
lanes are folded by the halving tree: for w = TREE_W/2, .., 1, lane l < w
becomes add(lane l, lane l + w). The result is lane 0. A device threadgroup
of TREE_W threads (thread l strides r = l, l + TREE_W, ...; the tree in
shared memory) computes exactly these words; the host and the units run the
same loop serially (`lt_tree`). With L <= 1 it is the old serial chain.

  te_global  r = the row index; terms: the target of rows outside fold fi
             (sum; count exact); then the squared deviations from the mean.
  te_enc     r = the row's rank among ITS CATEGORY's rows (all folds,
             ascending row order: the position in `te_bucket`'s bucket);
             the same two folds per category.
  ii_gram    the rows cut into IIG_ROWS-row chunks; chunk c's partial is
             the old serial chain over its observed rows (ascending, from
             zero); then r = c over the chunk partials.

BITS: a new order for every fold longer than one position (te_global /
te_enc at more than 1 row of a category or fold; ii_gram at n > IIG_ROWS).
All four columns run it together.

SWITCHES (IDENTICAL only, ON by default; MOJOLEARN_IDN_ALL_OFF turns all off):
  -D MOJOLEARN_IDN_TE_GLOBAL_TREE_OFF  te_global's old serial unit
  -D MOJOLEARN_IDN_TE_ENC_TREE_OFF     te_enc's old serial unit / host pass
  -D MOJOLEARN_IDN_II_GRAM_TILE_OFF    ii_gram's old serial unit
  -D MOJOLEARN_IDN_II_CONV_TREE_OFF    ii_conv's one-thread unit on the device
                                       (no bits: a max is order-free)

THE BLOCKED ORDER (`BLT`, lane classical-te-gmm 2026-10-07, IDENTICAL,
default OFF, A/B define -D MOJOLEARN_CLASSICAL_TE_BLOCKED_FOLD): the
positions r are cut into TEB-wide blocks m = r // TEB. Block m's partial
is the lane tree LT over its positions, lane r mod TREE_W (TEB is a
multiple of TREE_W, so the lane is the position's, not the block's); the
word is LT over the block partials ascending in m, lane m mod TREE_W. A
block with no taken position contributes its tree of zeros (+0.0), and a
sequence shorter than one block is LT over its own block then a one-lane
tree: the old word only when the positions sit in lane 0 of the outer tree.

  te_global  r = the row index: blocks of TEB rows, every (fold, target
             column) unit; the count is exact; then the squared deviations
             from the mean the same way.
  te_enc     r = the row's POSITION in its column's bucket array (every
             category of the column, ascending category, ascending row:
             `te_bucket`'s START[cat] + rank). A category's blocks are the
             TEB-aligned segments its positions cross, so one block of the
             array holds the segments of every category that crosses it:
             a segmented blocked reduction over the whole column, however
             skewed the categories.

WHY (cost, no shape rule): the lane tree gives one TREE_W group per unit,
so te_global runs (F+1)*T groups over all n rows and te_enc gives a
dominant category most of n in one group. Blocked, the device runs n/TEB
groups per unit (part kernel) and one group per unit over n/TEB partials
(fold kernel): TEB = 4096 positions is 16 per thread of a group, enough to
amortize the tree's eight barrier rounds, and few enough that the part
grid fills a device from ~10^5 rows up; the work is unchanged. The order
holds for any n and any category skew.
BITS: a new order for every fold longer than one position (both device
vendors and the host column change together: x_prep/target.mojo
`te_global_blt`, `te_enc_blt_fold`, x_prep/host/target.mojo, the device
kernels in x_prep/idn_blocked.mojo). It REPLACES the lane-tree order
above for te_global and te_enc (the two TREE_OFF defines are moot with it).
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.prims import add

comptime _IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
#: lane classical-te-gmm: te_global / te_enc in the blocked order `BLT` (docstring), default OFF
comptime IDN_TE_BLOCKED = _IDN and is_defined["MOJOLEARN_CLASSICAL_TE_BLOCKED_FOLD"]()
comptime IDN_TE_GLOBAL_TREE = _IDN and not is_defined["MOJOLEARN_IDN_TE_GLOBAL_TREE_OFF"]() and not IDN_TE_BLOCKED
comptime IDN_TE_ENC_TREE = _IDN and not is_defined["MOJOLEARN_IDN_TE_ENC_TREE_OFF"]() and not IDN_TE_BLOCKED
comptime IDN_II_GRAM_TILE = _IDN and not is_defined["MOJOLEARN_IDN_II_GRAM_TILE_OFF"]()
#: K14's non-eigh part: ii_conv's max over the row sums by a threadgroup tree
#: (device only: a max is exact, the unit's word, so the host is unchanged)
comptime IDN_II_CONV_TREE = _IDN and not is_defined["MOJOLEARN_IDN_II_CONV_TREE_OFF"]()

#: lanes of the tree (the device threadgroup width); a power of two
comptime TREE_W = 256
#: rows per chunk of the tiled ii_gram
comptime IIG_ROWS = 512
#: positions per block of the blocked order `BLT` (a multiple of TREE_W)
comptime TEB = 4096

comptime LTLanes = InlineArray[Float32, TREE_W]


@always_inline
def lt_zero(mut a: LTLanes):
    for l in range(TREE_W):
        a[l] = Float32(0)


@always_inline
def lt_tree(mut a: LTLanes) -> Float32:
    """The halving tree over the lanes (destroys them): lane 0's word."""
    var w = TREE_W // 2
    while w >= 1:
        for l in range(w):
            a[l] = add(a[l], a[l + w])
        w //= 2
    return a[0]


@always_inline
def iig_chunks(n: Int) -> Int:
    """ii_gram's chunks of IIG_ROWS rows (at least one)."""
    return max(1, (n + IIG_ROWS - 1) // IIG_ROWS)


@always_inline
def blt_close(mut a: LTLanes, mut b: LTLanes, m: Int):
    """`BLT`: block m's lanes `a` close into lane m mod TREE_W of the outer
    lanes `b` (the block's tree, then one `add`); `a` is zero afterwards.
    The serial spelling the units and the host column share; the device
    kernels (x_prep/idn_blocked.mojo) spell the same two trees."""
    b[m % TREE_W] = add(b[m % TREE_W], lt_tree(a))
    lt_zero(a)
