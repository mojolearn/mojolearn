# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The IDENTICAL FP32 GEMM oracle: the contract, written out, on the host.

**NOT A PORT, and it replaces no upstream call.** RAFT's standalone matrix
product is `raft/linalg/gemm.hpp` -> `detail/cublaslt_wrappers.hpp` ->
cuBLASLt, a CLOSED library with no source to mirror (`PORTING_RULES.md` 0b-i:
"where the path their dispatch actually takes calls a CLOSED library we
cannot read or port -- cuBLAS, cuSOLVER -- call the MAX equivalent, because
there is nothing to port"). There is therefore no upstream reference
implementation of a matrix product ANYWHERE in cuML, cuVS or RAFT to check
against, and this file is what stands in for one. The call it replaces, in
this repository's own terms, is `core/gemm.mojo::gemm_nt` / `gemm_tn` /
`gemv_n` under `NUMERIC_IDENTICAL` -- their pinned arms -- and, once Phase 2
lands, the scalable kernel.

What it DOES mirror is the ORDER. RAFT's own contraction
(`raft/distance/detail/pairwise_distance_base.cuh:139-149`, `:223-241`) walks
`kidx` ascending from 0 to k in steps of `Kblk` and, inside each block, `ki`
ascending in steps of `Veclen`, with ONE thread block owning the entire k
range of its output tile. No split-K, no cross-block combination. The
contract's "ascending k, one leaf at a time" is that order, generalized so
that a partition is allowed and NAMED rather than left to a library.

WHAT THIS FILE IS FOR
---------------------
`gemm/IDENTICAL_FP32_CONTRACT.md` is the contract in prose. This is the
same contract in code, and the two must be read together: every clause below
cites its section. Performance is irrelevant here -- it is `O(m n k)` scalar
Mojo on the host, single threaded, and it is meant to stay that way. Its jobs
are exactly three:

1. **Be the definition.** The NORMATIVE answer of profile
   `mojolearn.identical.gemm.fp32.v1` is `gemm_oracle(...)`: logical leaves
   at `contract_leaf_size(k)`, combined by `fold_balanced_tree`'s FIXED
   BALANCED TREE. Not "close to"; the same bits.
2. **Be the diagnostic reference.** `gemm_oracle_serial(...)` is the whole-K
   ascending chain, the simplest thing anybody can check by hand and what
   `core/gemm.mojo`'s two shipped pinned kernels compute. **It is NOT the v1
   answer when `P > 1`**, and the two coincide only at
   `k <= CONTRACT_K_LEAF_MIN`. Clause 4 of the Phase 2 contract call names
   both and names which is which.
3. **Be the instrument that proves a fixture separates.** The partition
   count is a PARAMETER here, so "does this input distinguish P = 8 from
   P = 16" is a question this file answers rather than one the kernel is
   trusted about. `gemm_oracle_check.mojo` uses it that way throughout.
4. **Be the tree's address space.** The balanced fold's structure is a pure
   function of `P`, computed on the HOST by `fold_level_width`,
   `fold_level_base`, `fold_node_addr` and `fold_node_is_carry`, so Phase 2b
   can address a node from a device kernel without inventing a second
   opinion about the topology. See the block comment above
   `fold_level_width`.

BUILT FROM THE DECLARED HELPERS, ON PURPOSE
--------------------------------------------
Every multiply-add is `original.numerics.identical_mul_add` and every seam
is `original.numerics.ftz`. Not local copies of them: the actual helpers, so
this file cannot drift into an independent opinion about what IDENTICAL
means. The consequence is that under `NUMERIC_FAST` both helpers compile away
and THIS FILE IS NOT THE CONTRACT -- it is the FAST spelling of the same
loops. That is correct and it is why every check prints the mode it compiled
in, and why the adversarial spellings in `gemm_oracle_check.mojo` are written
with an EXPLICIT `fma` and an explicit flush instead: a separation proof has
to hold in both modes or it is a proof about a build.

NO DEVICE KERNEL LIVES HERE, DELIBERATELY. Phase 1 is the oracle and its
fixtures; the scalable kernel is Phase 2's brief and must be built against a
contract that has been reviewed. A host-only oracle also runs with no GPU
present, which is a real virtue in a reference.

`[[mojo-string-float-roundtrip]]`: nothing here prints; the check does, and
it prints hex bits beside every decimal.
"""

from original.numerics import ftz, identical_mul_add


# ===========================================================================
# THE PROFILE CONSTANTS (contract section 6)
# ===========================================================================
# These two integers, and `k`, are the ONLY inputs to the logical k
# partition. Not the core count, not the occupancy, not the vendor, not the
# free memory, not the batch size, not the launch geometry, and -- the one
# that is easy to miss and is section 6's whole point -- **not m and not n**.

#: The shortest leaf the contract will produce, except for the ragged last
#: one. Below this the partition is not worth having: the fold's own rounding
#: steps start to rival the leaf's.
#:
#: 128 is `PINNED_GRAM_SPLITK_CHUNKS`'s sibling number and is chosen for the
#: same reason it was: it is small enough that the shipped Gram shapes get
#: real k-parallelism and large enough that the fold stays short. It is a
#: PROFILE constant, so changing it changes the answer's bits and is a
#: contract revision, not a tuning knob.
comptime CONTRACT_K_LEAF_MIN = 128

#: The cap on the number of leaves.
#:
#: ITS JUSTIFICATION CHANGED WITH THE v1 FOLD AND THE OLD ONE IS DELETED.
#: While the fold was serial, the cap was a CONDITIONING argument: an
#: unbounded `ceil(k/128)` at k = 4,000,000 is 31,250 partials, and a 31,250-
#: term serial fp32 chain is worse conditioned than the leaves the partition
#: was introduced to fix. **Under the balanced tree that argument is void** --
#: 31,250 partials fold in 15 levels, which is better conditioned than 1,024
#: leaves of 3,907, not worse. The cap survives on two DIFFERENT grounds:
#: it bounds the per-cell fold SCRATCH (`fold_node_total(P)` nodes) and the
#: number of ARITHMETIC tree levels a staged implementation may have to launch
#: (10 at P = 1024 against 15 at 31,250), and it keeps the leaf long enough
#: that the
#: leaf loop, not the fold, is where the work is.
#:
#: It is a PROFILE constant either way: changing it changes the answer's bits
#: and is a contract revision, not a tuning knob.
comptime CONTRACT_MAX_LEAVES = 1024


def contract_leaf_size(k: Int) -> Int:
    """The logical k leaf size, `L`. **A pure function of `k` and the two
    profile constants above.** Contract section 6.

    THE RULE:

        k <= 0                  -> 1        (no leaves; see leaf_count)
        k <= K_LEAF_MIN         -> k        (one leaf, the serial chain)
        ceil(k/K_LEAF_MIN) <= MAX_LEAVES
                                -> K_LEAF_MIN
        otherwise               -> ceil(k / MAX_LEAVES)

    `L` is the primitive and `P = ceil(k / L)` is DERIVED (`leaf_count`
    below), never the other way round. That ordering is what guarantees no
    empty leaf can exist: `P = ceil(k/L)` implies `(P-1)*L < k`, so every
    leaf index in `[0, P)` has at least one element. A kernel that instead
    fixed `P` and derived `L = ceil(k/P)` can produce trailing empty leaves
    at some `k`, and an empty leaf is a `+0.0` partial that a reader has to
    reason about; this way there is nothing to reason about.

    In the capped branch `P` may come out BELOW `MAX_LEAVES` (at k =
    4,000,000, `L = 3907` and `ceil(k/L) = 1024`; at other k it can be
    1023). That is fine and is why `MAX_LEAVES` is documented as a cap on
    the count rather than the count.
    """
    if k <= 0:
        return 1
    if k <= CONTRACT_K_LEAF_MIN:
        return k
    var p0 = (k + CONTRACT_K_LEAF_MIN - 1) // CONTRACT_K_LEAF_MIN
    if p0 <= CONTRACT_MAX_LEAVES:
        return CONTRACT_K_LEAF_MIN
    return (k + CONTRACT_MAX_LEAVES - 1) // CONTRACT_MAX_LEAVES


def leaf_count(k: Int, leaf: Int) -> Int:
    """`P = ceil(k / L)`, the number of leaves. `k == 0` gives 0 leaves and
    a `+0.0` product (contract section 8)."""
    if k <= 0:
        return 0
    var el = leaf
    if el < 1:
        el = 1
    return (k + el - 1) // el


def contract_leaf_count(k: Int) -> Int:
    """`P` at the contract's own leaf size. The number Phase 2's kernel must
    launch its partials against."""
    return leaf_count(k, contract_leaf_size(k))


def leaf_begin(j: Int, leaf: Int) -> Int:
    """Leaf `j` covers `[j*L, min((j+1)*L, k))`. Contract section 8
    (ragged k): only the LAST leaf is ever short, and it is never empty."""
    return j * leaf


def leaf_end(j: Int, leaf: Int, k: Int) -> Int:
    var e = (j + 1) * leaf
    if e > k:
        e = k
    return e


# ===========================================================================
# THE THREE OPERAND ORIENTATIONS (contract section 3)
# ===========================================================================
# ONE numerical implementation, three ADDRESSINGS. The accumulation below is
# character for character the same loop in all three cases; only `_a_at` and
# `_b_at` differ. That is what makes "NN, NT and TN agree bit for bit on the
# same logical matrices" a property rather than a coincidence, and it is the
# charter's requirement that the variants share one documented contract
# instead of being three kernels.
#
# Every matrix is ROW-MAJOR and CONTIGUOUS. No leading dimension, no stride,
# no sub-view. Contract section 2.

#: `C[m x n] = A[m x k] . B[k x n]`.
comptime OP_NN = 0
#: `C[m x n] = A[m x k] . B[n x k]^T`. `core/gemm.mojo::gemm_nt`'s shape.
comptime OP_NT = 1
#: `C[m x n] = A[k x m]^T . B[k x n]`. `core/gemm.mojo::gemm_tn`'s shape.
comptime OP_TN = 2


def _a_at(
    a: List[Float32], op: Int, i: Int, p: Int, m: Int, k: Int
) -> Float32:
    """`A_eff[i, p]`, the (i, p) entry of the LOGICAL left operand."""
    if op == OP_TN:
        return a[p * m + i]  # A is k x m row-major; A^T[i, p] = A[p, i]
    return a[i * k + p]  # A is m x k row-major


def _b_at(
    b: List[Float32], op: Int, p: Int, j: Int, n: Int, k: Int
) -> Float32:
    """`B_eff[p, j]`, the (p, j) entry of the LOGICAL right operand."""
    if op == OP_NT:
        return b[j * k + p]  # B is n x k row-major; B^T[p, j] = B[j, p]
    return b[p * n + j]  # B is k x n row-major


def op_name(op: Int) -> String:
    if op == OP_NN:
        return String("NN")
    if op == OP_NT:
        return String("NT")
    if op == OP_TN:
        return String("TN")
    return String("OP?")


# ===========================================================================
# THE ARITHMETIC (contract sections 4, 5, 7)
# ===========================================================================


def oracle_leaf_partial(
    a: List[Float32],
    b: List[Float32],
    op: Int,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
    k: Int,
    p_begin: Int,
    p_end: Int,
) -> Float32:
    """One leaf's partial for cell (i, j): `p` ASCENDING over `[p_begin,
    p_end)`, seeded `+0.0`, one `identical_mul_add` per step, every seam
    flushed.

    This loop is character for character `core/gemm.mojo::
    pinned_gemm_nt_kernel`'s, on purpose: a second spelling of the same
    arithmetic is a second thing that can be wrong. Contract sections 4
    (multiply-add), 5 (flush) and 7 (ordering).

    The `+0.0` seed is contract section 9's first half. It is what makes a
    leaf of all-zero products return `+0.0` and never `-0.0`, at every leaf
    length, on every vendor: `fma(x, +-0, +0.0)` is `+-0.0 + (+0.0)`, and a
    sum of two zeros of opposite sign is `+0` in round-to-nearest.
    """
    var acc = Float32(0.0)
    for p in range(p_begin, p_end):
        acc = ftz(
            identical_mul_add(
                ftz(_a_at(a, op, i, p, m, k)),
                ftz(_b_at(b, op, p, j, n, k)),
                acc,
            )
        )
    # The seam a real split-K kernel writes the partial through. Bitwise a
    # no-op given the flush inside the loop; here because the contract names
    # it as a seam and a reader should not have to derive that it is
    # redundant.
    return ftz(acc)



# ===========================================================================
# THE FIXED BALANCED FOLD TREE (contract section 7.2) AND ITS ADDRESSING
# ===========================================================================
# The structure below is a PURE FUNCTION OF `P`. It reads no launch geometry,
# no block size, no warp width, no vendor and no occupancy, and that is the
# whole point of naming it: **a physical block may calculate any node in any
# order once its dependencies are complete, and the bits do not move.**
#
# THE LOGICAL ADDRESS SPACE, which Phase 2b's kernel has to address from a
# device.
#
#     level 0     the P real leaf partials, in ASCENDING LOGICAL LEAF ORDER
#     level d     N_d = ceil(P / 2^d) nodes, for d = 1 .. D
#     D           the smallest d with N_d == 1; D = 0 when P == 1
#
#     node(d, q) for d >= 1 and 2q + 1 <  N_{d-1}   ARITHMETIC:
#         ftz( ftz(node(d-1, 2q)) + ftz(node(d-1, 2q+1)) )
#
#     node(d, q) for d >= 1 and 2q + 1 == N_{d-1}   CARRY:
#         node(d-1, 2q), copied BIT FOR BIT. No arithmetic, no padding.
#
#     output = ftz( node(D, 0) )
#
# A carry can only occur at the LAST node of a level whose predecessor had an
# ODD width, so there is at most one carry per level. `+0.0` padding is NOT
# an allowed spelling of it: `x + (+0.0)` is not the identity at `x = -0.0`,
# and fixture F7 is that difference measured. (`-0.0` padding IS bitwise
# equal to the carry at every node -- also measured, in F7's third arm -- and
# is still forbidden, because it is one character away from the spelling that
# is not, and it buys nothing.)
#
# THE FLAT ADDRESS. Levels are laid out low to high, level-major:
#
#     fold_level_base(P, d) = sum of N_0 .. N_{d-1}
#     fold_node_addr(P, d, q) = fold_level_base(P, d) + q
#     fold_node_total(P) = fold_level_base(P, D + 1)
#
# and for a whole `m x n` output the cell block is
#
#     (i * n + j) * fold_node_total(P) + fold_node_addr(P, d, q).
#
# **The (d, q) pair is the normative address; the flat integer is one legal
# layout of it.** A Phase 2b kernel that reduces IN PLACE over the level-0
# scratch, or that keeps only two levels live, is free to do so: what it may
# not change is which node is added to which, or the ascending logical order
# the pairing is taken in.


def fold_level_width(p: Int, d: Int) -> Int:
    """`N_d = ceil(P / 2^d)`, the number of nodes at level `d`.

    Repeated ceiling-halving IS a single ceiling division -- `ceil(ceil(x/2)
    /2) == ceil(x/4)` -- so the closed form and the level-by-level
    construction agree, and `check_fold_tree_addressing` asserts that against
    an iterative walk rather than trusting the identity.

    `d` past the top of the tree keeps returning 1, which is what makes the
    carry test below total.
    """
    if p <= 0:
        return 0
    if d <= 0:
        return p
    if d >= 40:
        return 1
    var denom = 1 << d
    return (p + denom - 1) // denom


def fold_level_count(p: Int) -> Int:
    """The number of levels INCLUDING level 0, i.e. `D + 1`.

    `P == 0` has no tree at all (0). `P == 1` is one level and performs NO
    fold addition: the single leaf reaches the output through the declared
    output seam and nothing else. That is contract section 7.2 and it is not
    a bypass of anything -- a one-node tree HAS no internal node to skip.
    """
    if p <= 0:
        return 0
    var levels = 1
    var w = p
    while w > 1:
        w = (w + 1) // 2
        levels += 1
    return levels


def fold_level_base(p: Int, d: Int) -> Int:
    """The flat address of node `(d, 0)`: the widths of every level below."""
    var base = 0
    for dd in range(d):
        base += fold_level_width(p, dd)
    return base


def fold_node_total(p: Int) -> Int:
    """Every node in the tree, level 0 included. The scratch a fully staged
    Phase 2b implementation would size per output cell."""
    return fold_level_base(p, fold_level_count(p))


def fold_node_addr(p: Int, d: Int, q: Int) -> Int:
    """The flat logical address of node `(d, q)`. One legal layout of the
    normative `(d, q)` pair; see the block comment above."""
    return fold_level_base(p, d) + q


def fold_node_is_carry(p: Int, d: Int, q: Int) -> Bool:
    """True when node `(d, q)` is the unpaired ODD tail of level `d-1` and is
    therefore a BIT-FOR-BIT COPY with no arithmetic in it.

    Level 0 is never a carry: its nodes are leaf partials.
    """
    if d < 1:
        return False
    return 2 * q + 1 >= fold_level_width(p, d - 1)


def fold_balanced_tree(partials: List[Float32]) -> Float32:
    """**THE CONTRACT'S FOLD**: the fixed balanced tree of contract section
    7.2, over the `P` real leaf partials in ASCENDING LOGICAL LEAF ORDER.

        current = partials
        while len(current) > 1:
            next[q] = ftz( ftz(current[2q]) + ftz(current[2q+1]) )
            if len(current) is odd: carry current[-1] unchanged
            current = next
        output = ftz(current[0])

    Four things this is NOT, each of which is a real alternative somebody
    will reach for and each of which fixture F5, F7, F8 or F9 separates:

    - it is NOT the serial ascending fold `acc = ftz(acc + ftz(p[t]))` that
      `core/gram_splitk.mojo::gram_splitk_reduce_kernel` ships and that this
      contract required before the v1 call. That fold has an `O(P)`
      dependency chain per output cell; this one has `O(log P)`. Fixture F5.
    - it is NOT the STRIDE pairing `red[t] += red[t + step]` that
      `core/pinned_reduce.mojo::pinned_block_sum` ships. That is also a
      balanced tree of the same depth, and it pairs DIFFERENT leaves, so it
      is a different answer. Fixture F8.
    - it does NOT pad an odd level to the next power of two. Fixture F7.
    - it does NOT let a block size, a warp width, an occupancy or a launch
      count define a level. Nothing here reads any of them; fixture F9 runs
      three unrelated evaluation schedules over the same tree and requires
      identical bits from all of them.

    `P == 0` (k == 0) returns `+0.0`. `P == 1` performs NO addition and
    returns `ftz(partials[0])` -- contract sections 7.2 and 8.

    The `ftz` on each child read is bitwise redundant (every node value is
    already flushed: a leaf partial by `oracle_leaf_partial`'s own output
    seam, an arithmetic node by this function, a carry node by inheritance)
    and it is written anyway, because contract section 5's seam table names
    it and a reader should not have to derive that two of the seven seams are
    no-ops.
    """
    var p = len(partials)
    if p == 0:
        return Float32(0.0)
    var current = partials.copy()
    while len(current) > 1:
        var width = len(current)
        var pairs = width // 2
        var nxt = List[Float32]()
        for q in range(pairs):
            nxt.append(ftz(ftz(current[2 * q]) + ftz(current[2 * q + 1])))
        if width % 2 != 0:
            # THE CARRY. Bit for bit, no arithmetic, no padding. Contract
            # section 7.2.
            nxt.append(current[width - 1])
        current = nxt^
    # The output seam, contract section 5g.
    return ftz(current[0])


def gemm_oracle_cell(
    a: List[Float32],
    b: List[Float32],
    op: Int,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
    k: Int,
    leaf: Int,
) -> Float32:
    """`C[i, j]` at an EXPLICIT leaf size, folded by the contract's balanced
    tree.

    `leaf >= k` is one leaf, which at `P == 1` makes this the whole-K
    ascending chain -- the same value as `gemm_oracle_serial_cell`, and
    `check_serial_oracle_is_the_one_leaf_case` asserts it. `leaf =
    contract_leaf_size(k)` is the contract's answer. Any other value is an
    adversary, and that is what the parameter is for.
    """
    var pcount = leaf_count(k, leaf)
    var el = leaf
    if el < 1:
        el = 1
    var partials = List[Float32]()
    for t in range(pcount):
        partials.append(
            oracle_leaf_partial(
                a,
                b,
                op,
                i,
                j,
                m,
                n,
                k,
                leaf_begin(t, el),
                leaf_end(t, el, k),
            )
        )
    return fold_balanced_tree(partials)


def gemm_oracle_at_leaf(
    a: List[Float32],
    b: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    leaf: Int,
) -> List[Float32]:
    """The whole `m x n` product at an explicit leaf size, row-major."""
    var c = List[Float32]()
    for i in range(m):
        for j in range(n):
            c.append(gemm_oracle_cell(a, b, op, i, j, m, n, k, leaf))
    return c^


def gemm_oracle(
    a: List[Float32], b: List[Float32], op: Int, m: Int, n: Int, k: Int
) -> List[Float32]:
    """**THE NORMATIVE ANSWER of `mojolearn.identical.gemm.fp32.v1`.**

    Logical leaves at `contract_leaf_size(k)`, combined by the fixed balanced
    tree of `fold_balanced_tree`. Row-major `m x n`.

    This is the value Phase 2's kernel must reproduce bit for bit, on Apple,
    NVIDIA and AMD, at every legal launch geometry and every batch
    composition. **It is `gemm_oracle`, not `gemm_oracle_serial`, that the
    scalable kernel must agree with** whenever `P > 1`; the two coincide only
    at `k <= CONTRACT_K_LEAF_MIN`.
    """
    return gemm_oracle_at_leaf(a, b, op, m, n, k, contract_leaf_size(k))


def gemm_oracle_serial_cell(
    a: List[Float32],
    b: List[Float32],
    op: Int,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
    k: Int,
) -> Float32:
    """One cell of the WHOLE-K ASCENDING CHAIN. **Diagnostic, NOT normative.**

    Written out as its own loop rather than as `gemm_oracle_cell` at
    `leaf = k`, so that the diagnostic reference is a second, independent
    spelling. `check_serial_oracle_is_the_one_leaf_case` requires the two to
    agree bit for bit; if they ever stop agreeing, the one-leaf case of the
    tree has grown an arithmetic step it should not have.
    """
    var acc = Float32(0.0)
    for p in range(k):
        acc = ftz(
            identical_mul_add(
                ftz(_a_at(a, op, i, p, m, k)),
                ftz(_b_at(b, op, p, j, n, k)),
                acc,
            )
        )
    return ftz(acc)


def gemm_oracle_serial(
    a: List[Float32], b: List[Float32], op: Int, m: Int, n: Int, k: Int
) -> List[Float32]:
    """The WHOLE-K ASCENDING CHAIN, one `p` loop per cell, no partition and
    no fold. **A DIAGNOSTIC REFERENCE. It is NOT the v1 answer when P > 1.**

    Contract clause 4 of the Phase 2 call names both references and this is
    the one that is only a reference: it is the simplest thing a person can
    check by hand, it is what `core/gemm.mojo`'s two shipped pinned kernels
    compute today, and it is what a reader means by "the obvious answer".

    Equal to `gemm_oracle` when and only when `k <= CONTRACT_K_LEAF_MIN`
    (there `P == 1` and the tree has no arithmetic node). Above that the two
    are DIFFERENT ANSWERS, the contract's is the partitioned one, and
    fixture F1 is that difference measured. Do not describe this function as
    "the right answer" at large `k`.
    """
    var c = List[Float32]()
    for i in range(m):
        for j in range(n):
            c.append(gemm_oracle_serial_cell(a, b, op, i, j, m, n, k))
    return c^
