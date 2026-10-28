# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The attention key-axis fold v2 (`-D MOJOLEARN_IDN_ATTN_FOLD_V2=1`, default
OFF; idea 10 of the 2026-10-28 neural speed report).

v1 (contract 5.3 and S19) folds the softmax denominator and every context
cell as ONE serial ascending chain over the key axis, so a row cannot be
split across threads. v2 splits the key axis into LEAVES OF 128 KEYS AT
ABSOLUTE POSITIONS (leaf i holds keys 128 i .. 128 i + 127 of the cache) and
combines the leaf partials with the GEMM contract's balanced carry tree
(`gemm/contract.mojo` section 7.2: node(d, q) = ftz(ftz(node(d-1, 2q)) +
ftz(node(d-1, 2q+1))), an unpaired last node is CARRIED bit for bit, no
padding). Every column computes the same fold:

    row (query t) sees keys [lo, hi]                (the mask's rule)
    P = hi // 128 + 1                               leaves 0 .. P-1
    leaf i partial = +0.0 when leaf i holds no key of [lo, hi];
                     else the v1 chain from +0.0 over the keys of leaf i
                     inside [lo, hi], ascending (S17: ftz(ftz(acc) + ftz(e));
                     S19: ftz(fma(ftz(w), ftz(v), acc)))
    result = ftz(tree(partial 0 .. P-1))

WHY DECODE STILL EQUALS PREFILL AND LENGTH STILL DOES NOT MATTER (contract
7.2, 7.3, and the reason 5.3 gave for refusing the GEMM topology). The GEMM's
P is a function of the contraction length k, which differs between prefill
(L keys) and decode (t + 1 keys). Here P and every leaf boundary are pure
functions of the ROW's absolute visible range [lo, hi], which is the same in
both paths and for every sequence length that contains the row; keys past
`hi` are not part of the fold at all, so no masked tail meets the tree. Keys
before `lo` (a sliding window) contribute exactly `+0.0` (S16 makes a masked
exponent exactly `+0.0`, its weight `+0.0 / den` is `+0.0`); a `+0.0`-seeded
chain over `+0.0` terms stays `+0.0`, so a leaf partly before `lo` may start
its chain at `lo`, and a leaf wholly before `lo` is the constant `+0.0` node.
The fused kernels therefore skip every masked cell EXACTLY: there is no
`-0.0` corner left in the forward fold (v1's FUSED_CORNER case was a chain
holding `-0.0` meeting a masked tail; v2 has no masked tail).

The fold is the same on NVIDIA, AMD, Apple and the host column: the per-term
seams are v1's, the leaf boundaries are absolute, and the tree reads no
launch geometry. The max (S14), the exponent (S15/S16) and the division
(S18) are unchanged. Bits change against v1 for every row past its first 128
keys (P >= 2); rows with hi < 128 keep v1's bits (P == 1, the tree is the
leaf).

The tree is evaluated here as a stream over the leaves in ascending order
(`AttnFoldV2Tree`): a binary counter merges two equal aligned blocks as soon
as both exist, and the remaining blocks are folded right to left at the end.
That is node for node the level-wise carry tree: an aligned block of 2^d
leaves is node(d, q), and the right-to-left finish is exactly the carry
chain of the last, partial nodes (P = 13: B8 + (B4 + L12) = node(4, 0)).
"""

from std.collections import InlineArray
from std.math import max, min
from std.sys.compile import is_defined

from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
)


comptime ATTN_FOLD_V2 = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_ATTN_FOLD_V2"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
"""The v2 key-axis fold on every column (device eager, fused, host oracle,
byte-LM host kernels). OFF: v1's serial chains, byte for byte."""

comptime ATTN_V2_LEAF = 128
"""Keys per leaf, a PROFILE constant (the GEMM contract's leaf minimum, for
the same reason: long enough that the leaf, not the tree, holds the work).
Changing it changes bits on every column."""

comptime ATTN_V2_STACK = 40
"""Depth of the streaming tree's stack: one entry per set bit of the leaf
count plus one, so 40 covers any key axis an `Int` index can address at a
128-key leaf."""


@always_inline
def attn_v2_leaf_count(hi: Int) -> Int:
    """P, the leaves of a row whose last visible key is `hi` (0 when the row
    sees no key)."""
    if hi < 0:
        return 0
    return hi // ATTN_V2_LEAF + 1


@always_inline
def attn_v2_leaf_visible(i: Int, lo: Int, hi: Int) -> Bool:
    """Leaf `i` holds at least one key of `[lo, hi]`."""
    var a = i * ATTN_V2_LEAF
    return a <= hi and a + ATTN_V2_LEAF - 1 >= lo


struct AttnFoldV2Tree(Copyable, Movable):
    """The balanced carry tree over leaf partials pushed in ascending leaf
    order. Host and device use this one spelling."""

    var stk: InlineArray[Float32, ATTN_V2_STACK]
    var sp: Int
    var n: Int

    @always_inline
    def __init__(out self):
        self.stk = InlineArray[Float32, ATTN_V2_STACK](fill=Float32(0.0))
        self.sp = 0
        self.n = 0

    @always_inline
    def push(mut self, x: Float32):
        self.stk[self.sp] = x
        self.sp += 1
        self.n += 1
        var c = self.n
        while c % 2 == 0:
            self.stk[self.sp - 2] = ftz(ftz(self.stk[self.sp - 2]) + ftz(self.stk[self.sp - 1]))
            self.sp -= 1
            c = c // 2

    @always_inline
    def result(self) -> Float32:
        """`ftz(tree)`; `+0.0` for an empty tree (a row with no key)."""
        if self.sp == 0:
            return Float32(0.0)
        var acc = self.stk[self.sp - 1]
        var i = self.sp - 2
        while i >= 0:
            acc = ftz(ftz(self.stk[i]) + ftz(acc))
            i -= 1
        return ftz(acc)


@always_inline
def attn_v2_denom_row(
    e: MutPointer[Float32, MutAnyOrigin], base: Int, lo: Int, hi: Int
) -> Float32:
    """S17 under v2 for one row: `e[base + j]` is the row's exponent at
    absolute key `j`. Reads only `[lo, hi]`."""
    var tree = AttnFoldV2Tree()
    for i in range(attn_v2_leaf_count(hi)):
        var acc = Float32(0.0)
        if attn_v2_leaf_visible(i, lo, hi):
            var a = max(i * ATTN_V2_LEAF, lo)
            var z = min(i * ATTN_V2_LEAF + ATTN_V2_LEAF - 1, hi)
            for j in range(a, z + 1):
                acc = ftz(ftz(acc) + ftz(e.unsafe_load(base + j)))
        tree.push(ftz(acc))
    return tree.result()


@always_inline
def attn_v2_context_cell(
    w: MutPointer[Float32, MutAnyOrigin],
    wbase: Int,
    v: MutPointer[Float32, MutAnyOrigin],
    vbase: Int,
    hd: Int,
    d: Int,
    lo: Int,
    hi: Int,
) -> Float32:
    """S19 under v2 for one output cell: `w[wbase + j]` is the row's weight
    at absolute key `j`, `v[vbase + j * hd + d]` the value. Reads only
    `[lo, hi]`."""
    var tree = AttnFoldV2Tree()
    for i in range(attn_v2_leaf_count(hi)):
        var acc = Float32(0.0)
        if attn_v2_leaf_visible(i, lo, hi):
            var a = max(i * ATTN_V2_LEAF, lo)
            var z = min(i * ATTN_V2_LEAF + ATTN_V2_LEAF - 1, hi)
            for j in range(a, z + 1):
                acc = ftz(identical_mul_add(
                    ftz(w.unsafe_load(wbase + j)), ftz(v.unsafe_load(vbase + j * hd + d)), acc))
        tree.push(acc)
    return tree.result()
