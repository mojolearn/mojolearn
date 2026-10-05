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
"""
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from x_prep.prims import add

comptime _IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
comptime IDN_TE_GLOBAL_TREE = _IDN and not is_defined["MOJOLEARN_IDN_TE_GLOBAL_TREE_OFF"]()
comptime IDN_TE_ENC_TREE = _IDN and not is_defined["MOJOLEARN_IDN_TE_ENC_TREE_OFF"]()
comptime IDN_II_GRAM_TILE = _IDN and not is_defined["MOJOLEARN_IDN_II_GRAM_TILE_OFF"]()
#: K14's non-eigh part: ii_conv's max over the row sums by a threadgroup tree
#: (device only: a max is exact, the unit's word, so the host is unchanged)
comptime IDN_II_CONV_TREE = _IDN and not is_defined["MOJOLEARN_IDN_II_CONV_TREE_OFF"]()

#: lanes of the tree (the device threadgroup width); a power of two
comptime TREE_W = 256
#: rows per chunk of the tiled ii_gram
comptime IIG_ROWS = 512

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
