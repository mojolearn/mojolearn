# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The IDENTICAL GEMM profile contract that the device kernels and the CPU
oracle share: the profile constants, the logical k-leaf partition, the three
operand orientations, the fixed balanced fold tree's address space, the
low-bit profiles' k bounds and versions, and the negative control's value
flip.

Everything here is a pure function of `k`, `P` and the profile constants
(index arithmetic over at most `log2(P)` levels). Nothing here reads or
writes a matrix. The device GEMMs (`gemm/checks/gemm_identical.mojo`, the
low-bit kernels) import THIS module, not the CPU oracle in
`gemm/host/gemm_oracle.mojo`, so a GPU binding does not link the CPU
reference. The oracle imports the same definitions from here, so the two
cannot drift (cpu-gpu-cleanup lane n-gemm, 2026-10-02: moved out of
`gemm/host/gemm_oracle.mojo` and the two low-bit oracles unchanged).
"""

from std.memory import bitcast
from std.sys.compile import is_defined


#: THE NEGATIVE CONTROL OF THE CPU IDENTITY GATE (the CPU training lane,
#: 2026-09-13, brief section 3.4). `-D MOJOLEARN_HOST_SABOTAGE=1` makes
#: `oracle_leaf_partial` return a partial whose bits DIFFER from the one its
#: own arithmetic produced, so a host binding built with it and every lane
#: that reaches this oracle must read DIVERGENT against the GPU columns. A
#: gate that cannot fail proves nothing. The define is passed by the host
#: build scripts only (`bindings/build_*_host.sh` through
#: MOJOLEARN_BUILD_EXTRA_DEFINES); a GPU binding never carries it, and a host
#: binding that carries it says so through `<prefix>_sabotage()` and is
#: refused by `python/mojolearn/_backend.py::load_host_module` outside the
#: gate.
#:
#: THIS WAS AN ORDER ARM UNTIL 2026-09-16, AND THAT ARM COULD NOT FAIL ON THE
#: `ties` FIXTURE (a 2026-09-16 sabotage audit). It
#: walked each leaf DESCENDING. Reversing a sum whose values add EXACTLY
#: cannot change its result, and `ties` is integer valued
#: (`tools/identity_break.py`, `rng.integers(0, 6)`), so on that fixture
#: `gemm-pinned` and `gemm-transposed` read UNMOVED under a build that was
#: supposed to be wrong. The reach is wider than those two lanes: this site is
#: the ONLY arm `linalg`, `mamba` and `transformer` reach, and one of two for
#: `training` and `neural`, so on `ties` those five families had no working
#: negative control at all. The remedy is the one `lane/ties-sabotage` already
#: applied to the neighbor and IVF families: perturb a VALUE, which no
#: fixture can make exact, rather than an ORDER, which an exact fixture folds
#: away.
comptime GEMM_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: The old order arm, kept behind a define of its own so the defect above can
#: be WATCHED to fail rather than believed. A build carrying BOTH defines
#: moves `base` and leaves `ties` unmoved, which is the measurement that
#: justifies the value flip. No build script and no gate sets this; it exists
#: for the witness run recorded in
#: `bench/results/identity_break/2026-09-16_sabotage-evidence/`.
comptime GEMM_ORACLE_SABOTAGE_LEGACY_ORDER = is_defined[
    "MOJOLEARN_GEMM_ORACLE_SABOTAGE_LEGACY_ORDER"
]()

#: The two arms, named so the `comptime if`s below read as one condition each.
comptime GEMM_ORACLE_SABOTAGE_ORDER_ARM = (
    GEMM_ORACLE_HOST_SABOTAGE and GEMM_ORACLE_SABOTAGE_LEGACY_ORDER
)
comptime GEMM_ORACLE_SABOTAGE_VALUE_ARM = (
    GEMM_ORACLE_HOST_SABOTAGE and not GEMM_ORACLE_SABOTAGE_LEGACY_ORDER
)


@always_inline
def gemm_oracle_sabotage_value_flip(v: Float32) -> Float32:
    """A float32 whose bits always differ from `v`'s. A magnitude below the
    smallest normal (either zero, or a subnormal) becomes the smallest
    positive normal, which the seam's `ftz` cannot fold back to zero; every
    other value steps its mantissa by one unit. Compiled only under
    GEMM_ORACLE_SABOTAGE_VALUE_ARM; the caller guards it.

    This is `core/knn_host_predict.mojo::host_sabotage_value_flip` spelled a
    second time rather than imported. This oracle depends on
    `checks.numerics` and nothing else, and a negative control that dragged a
    neighbors import into `linalg`, `mamba` and `transformer` would be a worse
    thing than ten repeated lines.
    """
    var bits = bitcast[DType.uint32](v)
    if (bits & UInt32(0x7FFFFFFF)) < UInt32(0x00800000):
        return bitcast[DType.float32](UInt32(0x00800000))
    return bitcast[DType.float32](bits + UInt32(1))


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
# I04 supported research version: every device and host oracle imports
# this constant. The profile cap/fold remain unchanged; partitions still
# depend only on k. Requires an explicit IDENTICAL build and is default off.
# I04 experiment: NEVER RUN — PENDING MEASUREMENT; incumbent defaults retained.
# I04 PENDING: compile evidence alone does not qualify device correctness, quality or speed.
# Leaf64 requires IDENTICAL + MOJOLEARN_IDN_GEMM_FOLD_LEAF_64; otherwise leaf128.
# I04 NVIDIA L40S 2026-10-06 component LOSS: leaf64 0.304/0.281 ms
# versus leaf128 0.280/0.255 ms, M,N,K=1024,1024,2048 and1023,1025,2049.
# One warmup/score; no full-workload promotion.
# Evidence: overnight-ab-20261006/nvidia/default-repair-normalized-measurements.json.
comptime CONTRACT_K_LEAF_MIN = (
    64 if (is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]()
           and is_defined["MOJOLEARN_IDN_GEMM_FOLD_LEAF_64"]()) else 128
)

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


def op_name(op: Int) -> String:
    if op == OP_NN:
        return String("NN")
    if op == OP_NT:
        return String("NT")
    if op == OP_TN:
        return String("TN")
    return String("OP?")


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



# ===========================================================================
# THE LOW-BIT PROFILES' BOUNDS AND VERSIONS
# ===========================================================================
# Moved from gemm/host/gemm_lowbit_oracle.mojo and gemm/host/gemm_int15_oracle.mojo
# (which import them back from here); see those files for the derivations.

#: The largest `k` the int8 profile accepts: `127 * 127 * k < 2^31` holds
#: up to 133,152, and the profile stops at the power of two below it so the
#: bound is a number a reader can check. Contract L-7.
comptime INT8_MAX_K = 131072

#: The int8 / bf16 profile version the bindings read back.
comptime LOWBIT_PROFILE_VERSION = 1

#: The largest `k` the int15 profile accepts (see gemm_int15_oracle.mojo).
comptime INT15_MAX_K = 65536

#: The int15 profile version the bindings read back.
comptime INT15_PROFILE_VERSION = 1
