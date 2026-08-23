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

1. **Be the definition.** The answer the Phase 2 kernel must produce is
   `gemm_oracle(...)` at `leaf = contract_leaf_size(k)`. Not "close to"; the
   same bits.
2. **Be the serial reference.** At `leaf >= k` there is one leaf and the
   whole product is one ascending fp32 chain, which is the simplest thing
   anybody can check by hand.
3. **Be the instrument that proves a fixture separates.** The partition
   count is a PARAMETER here, so "does this input distinguish P = 8 from
   P = 16" is a question this file answers rather than one the kernel is
   trusted about. `gemm_oracle_check.mojo` uses it that way throughout.

BUILT FROM THE DECLARED HELPERS, ON PURPOSE
--------------------------------------------
Every multiply-add is `mojo_only.numerics.identical_mul_add` and every seam
is `mojo_only.numerics.ftz`. Not local copies of them: the actual helpers, so
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

from mojo_only.numerics import ftz, identical_mul_add


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

#: The cap on the number of leaves. At k = 4,000,000 an unbounded
#: `ceil(k / 128)` would be 31,250 partials to fold serially, which is a
#: worse-conditioned sum than the leaves it was meant to fix. With the cap
#: the two levels are ~3,907 and 1,024 steps instead of 4,000,000.
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


def fold_partials(partials: List[Float32]) -> Float32:
    """The partial fold: SERIAL, ASCENDING by leaf index, seeded `+0.0`.
    Contract section 7.

    `acc = ftz(acc + ftz(partial[j]))` for `j = 0 .. P-1`. This is
    `core/gram_splitk.mojo::gram_splitk_reduce_kernel` exactly, which is
    deliberate: the contract picks the topology that already ships rather
    than a second one.

    **THE FOLD IS UNCONDITIONAL. `P == 1` IS A ONE-TERM FOLD, NOT A BYPASS**,
    and that sentence is contract section 9's second half rather than a
    pedantic aside. The one-term fold has exactly one arithmetic effect: it
    turns a `-0.0` partial into `+0.0`. Skip it at `P == 1` and the SIGN OF A
    ZERO becomes a function of the partition count -- `-0.0` at `P = 1` and
    `+0.0` at `P = 2` for the same inputs -- which is IDENTITY_PATHS row 13's
    defect (`-0.0` and `+0.0` compare equal, so which one survives is decided
    by order and the sign reaches the model) reappearing in a GEMM. Running
    the fold at every `P` costs one add and removes the class.

    `P == 0` (k == 0) returns `+0.0`.
    """
    var acc = Float32(0.0)
    for j in range(len(partials)):
        acc = ftz(acc + ftz(partials[j]))
    return acc


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
    """`C[i, j]` at an EXPLICIT leaf size.

    `leaf >= k` is the serial reference (one leaf). `leaf =
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
    return ftz(fold_partials(partials))


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
    """**THE CONTRACT'S ANSWER.** The `m x n` product at
    `contract_leaf_size(k)`, row-major.

    This is the value Phase 2's kernel must reproduce bit for bit, on Apple,
    NVIDIA and AMD, at every legal launch geometry and every batch
    composition.
    """
    return gemm_oracle_at_leaf(a, b, op, m, n, k, contract_leaf_size(k))


def gemm_oracle_serial(
    a: List[Float32], b: List[Float32], op: Int, m: Int, n: Int, k: Int
) -> List[Float32]:
    """The SERIAL reference: one leaf, one ascending fp32 chain per cell.

    Equal to `gemm_oracle` when and only when `k <= CONTRACT_K_LEAF_MIN`.
    Above that the two are DIFFERENT ANSWERS and the contract's is the
    partitioned one -- see contract section 7's note on why a serial fp32
    chain is not the target at large k.
    """
    var el = k
    if el < 1:
        el = 1
    return gemm_oracle_at_leaf(a, b, op, m, n, k, el)
