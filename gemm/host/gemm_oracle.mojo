# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""The IDENTICAL FP32 GEMM oracle: the contract, written out, on the host.

This historical module name is the compatibility and normative-reference
location. Production host code imports ``gemm.host.identical_gemm`` so its
role as the CPU implementation is not confused with check-only code.

**NO REFERENCE FILE, and it replaces no reference call.** RAFT's standalone matrix
product is `raft/linalg/gemm.hpp` -> `detail/cublaslt_wrappers.hpp` ->
cuBLASLt, a CLOSED library with no source to mirror (`CONTRIBUTING.md` (Algorithms and references):
"where the path their dispatch actually takes calls a CLOSED library we
cannot read or implementation -- cuBLAS, cuSOLVER -- call the MAX equivalent, because
there is nothing to implement"). There is therefore no reference
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


from checks.numerics import ftz, identical_mul_add
from gemm.contract import (
    CONTRACT_K_LEAF_MIN,
    CONTRACT_MAX_LEAVES,
    GEMM_ORACLE_HOST_SABOTAGE,
    GEMM_ORACLE_SABOTAGE_LEGACY_ORDER,
    GEMM_ORACLE_SABOTAGE_ORDER_ARM,
    GEMM_ORACLE_SABOTAGE_VALUE_ARM,
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_count,
    contract_leaf_size,
    fold_level_base,
    fold_level_count,
    fold_level_width,
    fold_node_addr,
    fold_node_is_carry,
    fold_node_total,
    gemm_oracle_sabotage_value_flip,
    leaf_begin,
    leaf_count,
    leaf_end,
    op_name,
)

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
# no sub-view. Contract section 2. The OP_* ids and op_name are in
# gemm/contract.mojo.


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
    comptime if GEMM_ORACLE_SABOTAGE_ORDER_ARM:
        # THE OLD ORDER ARM, compiled only as a witness: the same leaf walked
        # DESCENDING. Inert wherever the leaf adds exactly, `ties` included,
        # which is the whole reason it is no longer the arm. See
        # GEMM_ORACLE_SABOTAGE_LEGACY_ORDER.
        for q in range(p_end - p_begin):
            var p = p_end - 1 - q
            acc = ftz(
                identical_mul_add(
                    ftz(_a_at(a, op, i, p, m, k)),
                    ftz(_b_at(b, op, p, j, n, k)),
                    acc,
                )
            )
    else:
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
    comptime if GEMM_ORACLE_SABOTAGE_VALUE_ARM:
        # THE SABOTAGE ARM: this leaf's own arithmetic, then a value whose
        # bits differ from it on EVERY fixture, exact ones included. The flip
        # is applied after the seam so no `ftz` can fold it back. See
        # GEMM_ORACLE_HOST_SABOTAGE.
        return gemm_oracle_sabotage_value_flip(ftz(acc))
    return ftz(acc)



# ===========================================================================
# THE FIXED BALANCED FOLD TREE (contract section 7.2)
# ===========================================================================
# Its address space (fold_level_width / _count / _base, fold_node_total /
# _addr / _is_carry) lives in gemm/contract.mojo with the rest of the contract
# the device kernels share; the oracle fold below walks the same tree.


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
    # A one-leaf tree has no arithmetic node.  Return that leaf through the
    # profile's output seam directly instead of allocating a one-element
    # partial list and copying it in `fold_balanced_tree`.  Keep the final
    # `ftz` explicit: it is contract section 5g even though every ordinary
    # leaf is already flushed, and it also keeps the output spelling exact
    # under either host sabotage arm.
    if pcount == 1:
        return ftz(
            oracle_leaf_partial(
                a, b, op, i, j, m, n, k, 0, k
            )
        )
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
    # Hoist the one-leaf decision out of the cell loop.  Besides avoiding
    # the partial-list allocation, this keeps a dense short-k product from
    # recomputing the same leaf count for every output cell.
    if leaf_count(k, leaf) == 1:
        for i in range(m):
            for j in range(n):
                c.append(
                    ftz(oracle_leaf_partial(a, b, op, i, j, m, n, k, 0, k))
                )
        return c^
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


def oracle_leaf_partial_right_zero_padded(
    a: List[Float32],
    b: List[Float32],
    op: Int,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
    k: Int,
    real_k: Int,
    p_begin: Int,
    p_end: Int,
) -> Float32:
    """One v1 leaf when both operands are exact +0.0 at ``p >= real_k``.

    The logical ``k``, leaf boundaries and fold tree do not change. Only the
    repeated zero products are compressed. In the normal order a padded tail
    still performs one zero FMA: it observably turns a real prefix ending at
    -0.0 into +0.0. Every later zero FMA is then bitwise inert. In the legacy
    descending sabotage order the padding precedes the real terms and leaves
    the +0.0 seed unchanged, so there is no trailing operation to simulate.
    """
    var active_end = p_end
    if active_end > real_k:
        active_end = real_k
    if active_end < p_begin:
        active_end = p_begin
    var acc = Float32(0.0)
    comptime if GEMM_ORACLE_SABOTAGE_ORDER_ARM:
        for q in range(active_end - p_begin):
            var p = active_end - 1 - q
            acc = ftz(
                identical_mul_add(
                    ftz(_a_at(a, op, i, p, m, k)),
                    ftz(_b_at(b, op, p, j, n, k)),
                    acc,
                )
            )
    else:
        for p in range(p_begin, active_end):
            acc = ftz(
                identical_mul_add(
                    ftz(_a_at(a, op, i, p, m, k)),
                    ftz(_b_at(b, op, p, j, n, k)),
                    acc,
                )
            )
        if active_end < p_end:
            acc = ftz(
                identical_mul_add(Float32(0.0), Float32(0.0), acc)
            )
    comptime if GEMM_ORACLE_SABOTAGE_VALUE_ARM:
        return gemm_oracle_sabotage_value_flip(ftz(acc))
    return ftz(acc)


def gemm_oracle_right_zero_padded(
    a: List[Float32],
    b: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    real_k: Int,
) raises -> List[Float32]:
    """The normative ``m x n`` v1 product with a shared exact-zero suffix.

    Bitwise equal to ``gemm_oracle(..., k)`` when both logical operands read
    +0.0 for every contraction position in ``[real_k, k)``. This is not a
    smaller GEMM: leaf size and the balanced fold remain functions of ``k``.
    """
    if real_k < 0 or real_k > k:
        raise Error("gemm_oracle_right_zero_padded: real_k must be in [0, k]")
    var leaf = contract_leaf_size(k)
    var pcount = leaf_count(k, leaf)
    var out = List[Float32]()
    for i in range(m):
        for j in range(n):
            if pcount == 1:
                out.append(
                    ftz(
                        oracle_leaf_partial_right_zero_padded(
                            a, b, op, i, j, m, n, k, real_k, 0, k
                        )
                    )
                )
                continue
            var partials = List[Float32]()
            for t in range(pcount):
                partials.append(
                    oracle_leaf_partial_right_zero_padded(
                        a, b, op, i, j, m, n, k, real_k,
                        leaf_begin(t, leaf), leaf_end(t, leaf, k),
                    )
                )
            out.append(fold_balanced_tree(partials))
    return out^


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
