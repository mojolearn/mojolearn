# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Blocked column moments: the scalar arithmetic and the shape rules shared by
the device kernels (`core/blocked_moments.mojo`) and their host column
(`core/blocked_moments_host.mojo`). GPU free. Lane classical-decomp
(2026-10-07); IDENTICAL only, every switch that reaches it is default OFF.

THE PROFILE (one version, every column):
- Rows are cut into LEAVES of `L` consecutive rows. `L` is a pure function of
  the row count and the per-leaf partial width (`bm_*_leaf_rows` below),
  never of a launch, a core count or a vendor.
- Inside a leaf a value is a plain ascending chain per (sub-chain, cell),
  every operand and result flushed; sub-chains are combined in ascending
  sub-chain order (`bm_sub_chains`: a function of the column count only).
- Leaves are folded as a BINARY COUNTER folds them: adjacent pairs at each
  level, left before right; an odd node at the end of a level is that
  level's remainder, and the remainders are combined right to left
  (ascending level, `rem[L2] + acc`). The device runs each level as one
  parallel launch; the host runs the same levels in a loop.
- Moment folds (`chan`) carry (rows, mean, M2) and merge with Chan's
  pairwise update, so no raw `sum(x^2) - n mean^2` cancellation is formed.
"""
from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add
from gemm.contract import contract_leaf_size

#: Threads per block of every blocked-moment kernel (a version constant).
comptime BM_TPB = 256
#: Output tile edge of the leaf Gram kernel (BM_TILE x BM_TILE cells).
comptime BM_TILE = 16
#: Rows staged per shared-memory pass of the leaf Gram kernel.
comptime BM_ROW_TILE = 32
#: Sub-chains per column for a leaf's own means in the leaf Gram kernel.
comptime BM_GRAM_MEAN_CHAINS = 8
#: Base leaf length of the one-pass profiles (PCA one-pass covariance, TSVD
#: fused statistics): long enough that the leaf, not the fold, holds the work.
comptime BM_LEAF_ROWS = 1024
#: Storage cap (float words) for one fold's leaf partials. A memory bound,
#: not a shape rule: at P leaves of `cells` partials the leaf doubles until
#: P * cells fits, so every (n, d) gets the same rule.
comptime BM_PART_WORDS = 1 << 24
#: Fold levels a mask can hold (leaf counts below 2^31).
comptime BM_MAX_LEVELS = 31


def bm_leaf_count(n: Int, leaf: Int) -> Int:
    if n <= 0:
        return 0
    return (n + leaf - 1) // leaf


def bm_mean_leaf_rows(n: Int) -> Int:
    """Column-mean leaves: the GEMM contract's own leaf rule (at most
    1024 leaves, at least `CONTRACT_K_LEAF_MIN` rows)."""
    return contract_leaf_size(n)


def bm_onepass_leaf_rows(n: Int, cells: Int) -> Int:
    """`BM_LEAF_ROWS`, doubled while the leaf partials exceed
    `BM_PART_WORDS` words (a storage bound)."""
    var leaf = BM_LEAF_ROWS
    var c = max(cells, 1)
    while leaf < n and bm_leaf_count(n, leaf) * c > BM_PART_WORDS:
        leaf *= 2
    return leaf


def bm_sub_chains(m: Int) -> Int:
    """Sub-chains per column inside one leaf block: `BM_TPB // m`, at least 1."""
    if m >= BM_TPB:
        return 1
    return BM_TPB // max(m, 1)


def bm_node_rows(n: Int, leaf: Int, level: Int, p: Int) -> Int:
    """Rows under node `p` of fold level `level` (2^level leaves of `leaf` rows)."""
    var span = leaf << level
    var r0 = p * span
    return min(n, r0 + span) - r0


def bm_tile_pair(tp: Int, t: Int) -> Tuple[Int, Int]:
    """The `tp`-th upper tile pair (I, J), I <= J, row-major over T tiles."""
    var rest = tp
    var i = 0
    while rest >= t - i:
        rest -= t - i
        i += 1
    return (i, i + rest)


@always_inline
def bm_sub(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) - ftz(b))


@always_inline
def bm_add(a: Float32, b: Float32) -> Float32:
    """One flushed add, left operand first: the only way two partials meet."""
    return ftz(ftz(a) + ftz(b))


@always_inline
def bm_fma(a: Float32, b: Float32, acc: Float32) -> Float32:
    return ftz(identical_mul_add(a, b, acc))


@always_inline
def bm_div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def bm_chan_mean(ma: Float32, mb: Float32, na: Int, nb: Int) -> Float32:
    """Merged mean of (na, ma) then (nb, mb): ma + (mb - ma) * nb / (na + nb)."""
    var fb = bm_div(Float32(nb), Float32(na + nb))
    return bm_add(ma, ftz(identical_mul(bm_sub(mb, ma), fb)))


@always_inline
def bm_chan_m2(
    m2a: Float32, m2b: Float32,
    mai: Float32, mbi: Float32, maj: Float32, mbj: Float32,
    na: Int, nb: Int,
) -> Float32:
    """Merged centered cross moment of cell (i, j):
    M2a + M2b + (mb_i - ma_i)(mb_j - ma_j) * na nb / (na + nb)."""
    var w = bm_div(ftz(identical_mul(Float32(na), Float32(nb))), Float32(na + nb))
    var di = bm_sub(mbi, mai)
    var dj = bm_sub(mbj, maj)
    var corr = ftz(identical_mul(ftz(identical_mul(di, dj)), w))
    return bm_add(bm_add(m2a, m2b), corr)
