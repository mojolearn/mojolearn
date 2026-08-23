"""The GEMM oracle, and the fixtures that make each of its clauses FALSIFIABLE.

Phase 1 of `IDENTICAL_GEMM_PLAN.md`'s lane charter. The contract is
`gemm/IDENTICAL_FP32_CONTRACT.md`; the oracle is
`gemm/mojo_only/gemm_oracle.mojo`; this file is the evidence that the
oracle's clauses are choices rather than defaults.

    pixi run mojo run -I . gemm/mojo_only/gemm_oracle_check.mojo

HOST ONLY. No `DeviceContext`, no kernel, no GPU required. Phase 1 is the
oracle and its fixtures; the scalable kernel and the device-side invariance
gates are Phase 2's and Phase 3's briefs.

WHY A RANDOM-INPUT HASH IS NOT ACCEPTABLE HERE, IN THIS REPOSITORY'S OWN WORDS
------------------------------------------------------------------------------
`check-ieee-arith` scored Metal through MAX as UNFUSED on 2^20 hashed
patterns. IDENTITY_PATHS row 9's correction, 2026-08-23, found that **ZERO of
those 2^20 patterns separate a fused `a*b+c` from an unfused one**: random
exponents put the product and the addend so far apart that both spellings
round the same way, and the tie arm was written `if got == unfused` first, so
1,046,394 ties were counted as evidence for UNFUSED. A backend that contracts
everything scored identically. The verdict propagated into the ledger and from
there into a design claim, and it was wrong for a month.

So every fixture below is BUILT TO SEPARATE, and every one of them carries its
own proof: the check computes BOTH alternatives and REFUSES to pass unless
their bits differ. A fixture that cannot distinguish the alternatives is not
evidence, and a check that would pass on such a fixture is worse than no check
because it converts a checkable property into a belief.

THE ADVERSARIES ARE WRITTEN WITH EXPLICIT SPELLINGS, NOT WITH THE HELPERS
--------------------------------------------------------------------------
`identical_mul_add` and `ftz` are comptime-gated on `GLOBAL_NUMERIC_MODE`, so
a fixture written with them proves something about a BUILD. The alternatives
here use an explicit `std.math.fma` against an explicit two-statement multiply
then add, and an explicit flush model against the identity function. The
separation proofs therefore hold in both modes, and are printed in both.

What the MODE does change is one thing, checked separately by
`check_oracle_matches_the_contract_spelling`: whether the gated oracle IS the
contract. Under IDENTICAL it must be, bit for bit — that is the reach proof
for the two helpers. Under FAST it is the FAST spelling and the check reports
rather than asserts.

`[[mojo-string-float-roundtrip]]`: `String(Float32)` does not round trip, so
every float printed here carries `/0x<bits>` beside it and every comparison is
on the bits.

EVERY CHECK PRINTS THE MODE THIS BINARY COMPILED IN. Four sessions share this
checkout, `GLOBAL_NUMERIC_MODE` is a comptime constant in a shared file, and a
run that compiled inside another session's flip window would otherwise report
one arm's numbers under the other arm's label. That has happened three times
in one day. Use `tools/with_identical_mode.sh` (which takes the shared build
lock, DEVIATION 514) for the IDENTICAL arm and `tools/with_build_lock.sh` for
the FAST arm, and never quote a number whose mode line you did not see the
binary print.
"""

from std.math import fma
from std.memory import bitcast

from gemm.mojo_only.gemm_oracle import (
    CONTRACT_K_LEAF_MIN,
    CONTRACT_MAX_LEAVES,
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_count,
    contract_leaf_size,
    gemm_oracle_at_leaf,
    gemm_oracle_cell,
    leaf_begin,
    leaf_count,
    leaf_end,
    op_name,
)
from core.gemm import (
    GEMM_MBLK,
    GEMM_NBLK,
    GEMM_SMEM_PAGE_X,
    GEMM_SMEM_PAGE_Y,
    GEMM_SMEM_STRIDE,
    GEMM_THREADS,
)
from gemm.ported.linalg.contractions import (
    Policy2x8Float,
    Policy4x4Float,
    Policy4x4FloatCol,
    Policy4x4SkinnyFloat,
    assert_col_policy_square,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

#: The smallest normal binary32, `2^-126`. `ftz`'s own boundary.
comptime MIN_NORMAL_F32 = Float32(1.1754943508222875e-38)

#: `2^24`. The smallest float32 whose ulp is 2, so `2^24 + 1` is exactly
#: halfway between two representables and round-half-to-even discards it.
#: Every "swallow" fixture below is built on that one fact.
comptime TWO_P24 = Float32(16777216.0)

# ---- the fold spellings under test ---------------------------------------
#: SERIAL ASCENDING, seeded `+0.0`. Contract section 7.2. THE CONTRACT.
comptime FOLD_SERIAL_ZERO_SEED = 0
#: A fixed BALANCED HALVING tree, `pinned_block_sum`'s shape. The adversary
#: for contract section 7.2's topology choice.
comptime FOLD_HALVING = 1
#: SERIAL ASCENDING seeded with `partials[0]` instead of `+0.0` -- i.e. the
#: spelling in which `P == 1` is a BYPASS rather than a one-term fold. The
#: adversary for contract section 9.2(b).
comptime FOLD_SERIAL_FIRST_SEED = 2


def _mode_name() -> String:
    """The mode THIS BINARY WAS COMPILED IN, read from the comptime constant
    the kernels compile against. See the module header."""
    comptime if IDENTICAL_BUILD:
        return String("IDENTICAL")
    return String("FAST")


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _f32(b: UInt32) -> Float32:
    return bitcast[DType.float32](b)


def _show(x: Float32) -> String:
    return String(x) + "/" + hex(_bits(x))


# ===========================================================================
# THE FOUR AXES, EACH WITH BOTH SPELLINGS WRITTEN OUT
# ===========================================================================
# Not the gated helpers. See the module header: an adversary written with
# `identical_mul_add` proves something about a build, not about arithmetic.


def _ftz_on(x: Float32) -> Float32:
    """`mojo_only/numerics.ftz`'s IDENTICAL arm, UNGATED. Flush a subnormal
    to a zero of its own sign; leave zeros, normals, infinities and NaN
    alone.

    `x != 0.0` is False for BOTH zeros (`-0.0 == +0.0` in IEEE), so a `-0.0`
    passes through unchanged rather than being normalized. That is deliberate
    and it is what makes fixture F6a possible.
    """
    if abs(x) < MIN_NORMAL_F32 and x != Float32(0.0):
        if x < Float32(0.0):
            return Float32(-0.0)
        return Float32(0.0)
    return x


def _ftz_off(x: Float32) -> Float32:
    """Gradual underflow: the subnormal survives. What CUDA does by default."""
    return x


def _flush(x: Float32, on: Bool) -> Float32:
    if on:
        return _ftz_on(x)
    return _ftz_off(x)


def _fused(a: Float32, b: Float32, c: Float32) -> Float32:
    """ONE rounding of `a*b + c`. Contract section 4."""
    return fma(a, b, c)


def _rounded_product(a: Float32, b: Float32) -> Float32:
    """`fl(a*b)`, the correctly-rounded product, WITH NO MULTIPLY IN IT.

    `fma(a, b, +0.0)` is `round(a*b + 0)` which is `round(a*b)`. Same value as
    `a * b` on every backend, for every input except one: when the exact
    product is `-0.0`, `a * b` is `-0.0` and `fma(a, b, +0.0)` is `+0.0`. No
    fixture here has a zero product, so the substitution is exact where it is
    used, and it is written this way for the reason below.

    THE REASON. The obvious spelling of an unfused multiply-add is a
    standalone multiply into a named local followed by an add, which is what
    `mojo_only/ieee_arith_check.mojo`'s separating arm relies on ("ONE
    multiply, standing alone: nothing for a compiler to contract it into").
    **On the HOST, in this toolchain, that is not true.** MEASURED HERE,
    2026-08-23: F3's earlier spelling refused itself on 1,621 of 4,096
    patterns because `var prod = a * b; return prod + c` came back as the
    FUSED answer -- Mojo contracted across the two statements. That is the
    mojotrees plateau-tie incident's class, reproduced on the host, and it is
    a finding about Mojo codegen rather than a defect in the fixture.

    So the unfused spelling is built out of an operation there is nothing to
    contract: an `fma` whose addend is a literal zero, and then a plain add.
    No multiply appears syntactically, and no compiler may rewrite
    `fma(a,b,0) + c` into `fma(a,b,c)` because they are different values.
    """
    return fma(a, b, Float32(0.0))


def _unfused(a: Float32, b: Float32, c: Float32) -> Float32:
    """TWO roundings: `fl(fl(a*b) + c)`. See `_rounded_product` for why the
    product is spelled the way it is, and for the measurement that forced
    it."""
    return _rounded_product(a, b) + c


def _mul_add(a: Float32, b: Float32, c: Float32, use_fma: Bool) -> Float32:
    if use_fma:
        return _fused(a, b, c)
    return _unfused(a, b, c)


def _a_at(
    a: List[Float32], op: Int, i: Int, p: Int, m: Int, k: Int
) -> Float32:
    """A local copy of the oracle's addressing, so the adversary evaluator
    does not import a gated file's opinion about anything. Contract
    section 3."""
    if op == OP_TN:
        return a[p * m + i]
    return a[i * k + p]


def _b_at(
    b: List[Float32], op: Int, p: Int, j: Int, n: Int, k: Int
) -> Float32:
    if op == OP_NT:
        return b[j * k + p]
    return b[p * n + j]


def _leaf_partial(
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
    use_fma: Bool,
    use_ftz: Bool,
) -> Float32:
    var acc = Float32(0.0)
    for p in range(p_begin, p_end):
        acc = _flush(
            _mul_add(
                _flush(_a_at(a, op, i, p, m, k), use_ftz),
                _flush(_b_at(b, op, p, j, n, k), use_ftz),
                acc,
                use_fma,
            ),
            use_ftz,
        )
    return _flush(acc, use_ftz)


def _fold(
    partials: List[Float32], mode: Int, use_ftz: Bool
) raises -> Float32:
    """The three fold spellings. Only `FOLD_SERIAL_ZERO_SEED` is the
    contract."""
    var p = len(partials)
    if mode == FOLD_HALVING:
        # `pinned_block_sum`'s shape: `red[t] += red[t + step]` for
        # `step = P/2 ... 1`. Defined for power-of-two P only, which is all
        # the adversary is ever used at; a general P would need a padding
        # rule and that would be a second thing to specify.
        if p == 0:
            return Float32(0.0)
        var q = p
        while q > 1:
            if q % 2 != 0:
                raise Error(
                    "gemm_oracle_check: the halving adversary needs a"
                    " power-of-two partial count, got " + String(p)
                )
            q //= 2
        var red = partials.copy()
        var step = p // 2
        while step > 0:
            for t in range(step):
                red[t] = _flush(red[t] + _flush(red[t + step], use_ftz), use_ftz)
            step //= 2
        return red[0]
    if mode == FOLD_SERIAL_FIRST_SEED:
        if p == 0:
            return Float32(0.0)
        var acc = _flush(partials[0], use_ftz)
        for t in range(1, p):
            acc = _flush(acc + _flush(partials[t], use_ftz), use_ftz)
        return acc
    var acc2 = Float32(0.0)
    for t in range(p):
        acc2 = _flush(acc2 + _flush(partials[t], use_ftz), use_ftz)
    return acc2


def _eval_cell(
    a: List[Float32],
    b: List[Float32],
    op: Int,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
    k: Int,
    leaf: Int,
    use_fma: Bool,
    use_ftz: Bool,
    fold_mode: Int,
) raises -> Float32:
    """One output cell under an EXPLICIT choice on each of the four axes.

    `(leaf = contract_leaf_size(k), use_fma = True, use_ftz = True,
    fold_mode = FOLD_SERIAL_ZERO_SEED)` is the contract. Every other
    combination is an adversary, and each fixture below moves exactly ONE
    axis so that a difference in bits names its cause.
    """
    var el = leaf
    if el < 1:
        el = 1
    var pcount = leaf_count(k, el)
    var partials = List[Float32]()
    for t in range(pcount):
        partials.append(
            _leaf_partial(
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
                use_fma,
                use_ftz,
            )
        )
    return _flush(_fold(partials, fold_mode, use_ftz), use_ftz)


def _contract_cell(
    a: List[Float32],
    b: List[Float32],
    op: Int,
    i: Int,
    j: Int,
    m: Int,
    n: Int,
    k: Int,
) raises -> Float32:
    """The contract's answer, spelled explicitly. `gemm_oracle` must equal
    this under IDENTICAL."""
    return _eval_cell(
        a,
        b,
        op,
        i,
        j,
        m,
        n,
        k,
        contract_leaf_size(k),
        True,
        True,
        FOLD_SERIAL_ZERO_SEED,
    )


def _separates(
    tag: String, label_a: String, va: Float32, label_b: String, vb: Float32
) raises:
    """THE PROOF, and the refusal. Print both bit patterns and REFUSE unless
    they differ."""
    print(
        "    "
        + tag
        + ":  "
        + label_a
        + " = "
        + _show(va)
        + "   |   "
        + label_b
        + " = "
        + _show(vb)
    )
    if _bits(va) == _bits(vb):
        raise Error(
            "gemm_oracle_check: THE FIXTURE DOES NOT SEPARATE. "
            + tag
            + ": "
            + label_a
            + " and "
            + label_b
            + " produced the same bits (0x"
            + hex(_bits(va))
            + "). A fixture that cannot distinguish the alternatives is not"
            " evidence -- see IDENTITY_PATHS row 9's correction. Fix the"
            " fixture; do NOT relax the check."
        )


def _zeros(n: Int) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n):
        v.append(Float32(0.0))
    return v^


def _ones(n: Int) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n):
        v.append(Float32(1.0))
    return v^


def _hash64(i: Int, f: Int) -> UInt64:
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(f + 1) * UInt64(
        0xBF58476D1CE4E5B9
    )
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


def _spread_half(i: Int, f: Int) -> Float32:
    """A value in `[1, 2)` whose mantissa is the TOP 12 BITS ONLY.

    Two such significands are 13 bits each, so their exact product needs up
    to 26 bits and float32 keeps 24: the product is INEXACT, which is the
    whole requirement of F3. Full-width mantissas do not reliably separate
    the two spellings -- measured elsewhere in this repository at 0 of 256 --
    because the accumulator's own rounding swallows the tail. Same generator
    and same reasoning as `mojo_only/ieee_arith_check.mojo`'s ARM 6.
    """
    return Float32(1.0) + Float32(Int(_hash64(i, f) & UInt64(0xFFF))) / Float32(
        4096.0
    )


# ===========================================================================
# THE PRELIMINARIES
# ===========================================================================


def check_ieee_zero_assumptions() raises:
    """Contract section 9.2 rests on THREE IEEE facts. Assert them rather
    than assume them.

    The contract argues that a `+0.0` seed makes a `-0.0` unreachable from
    products alone, on every backend, because the facts below are IEEE-754
    and not codegen choices. If Mojo's HOST constant folding disagrees with
    any of them, the whole of section 9.2 is unsound and this check is where
    that shows up, before a device ever runs. This is the host half only; the
    device half is Phase 3's.
    """
    # Through a List, so the values arrive at the arithmetic as runtime
    # loads rather than as literals a constant folder can evaluate with a
    # different rounding path than the one that runs on real data.
    var box = List[Float32]()
    box.append(Float32(0.0))
    box.append(Float32(-0.0))
    box.append(Float32(3.5))
    var pz = box[0]
    var nz = box[1]
    if _bits(pz) != UInt32(0x00000000):
        raise Error("check_ieee_zero_assumptions: +0.0 is not 0x00000000")
    if _bits(nz) != UInt32(0x80000000):
        raise Error("check_ieee_zero_assumptions: -0.0 is not 0x80000000")

    # (1) `(+0) + (-0) == +0` in round-to-nearest. The fold seed's whole
    #     mechanism.
    var s1 = pz + nz
    # (2) `(-0) + (-0) == -0`. Without this the FOLD_SERIAL_FIRST_SEED
    #     adversary would not be an adversary and F6a would be vacuous.
    var s2 = nz + nz
    # (3) `fma(x, -0.0, +0.0) == +0.0`: a negative-zero PRODUCT does not
    #     reach a `+0.0`-seeded accumulator.
    var s3 = fma(box[2], nz, pz)
    print(
        "    IEEE zeros:  (+0)+(-0) = "
        + _show(s1)
        + "   (-0)+(-0) = "
        + _show(s2)
        + "   fma(3.5,-0,+0) = "
        + _show(s3)
    )
    if _bits(s1) != UInt32(0x00000000):
        raise Error(
            "check_ieee_zero_assumptions: (+0)+(-0) is not +0.0 on this host."
            " Contract section 9.2(a) is unsound as written."
        )
    if _bits(s2) != UInt32(0x80000000):
        raise Error(
            "check_ieee_zero_assumptions: (-0)+(-0) is not -0.0 on this host."
        )
    if _bits(s3) != UInt32(0x00000000):
        raise Error(
            "check_ieee_zero_assumptions: fma(x, -0, +0) is not +0.0 on this"
            " host. Contract section 9.2(a) is unsound as written."
        )
    print("check_ieee_zero_assumptions OK [" + _mode_name() + "]")


def _eq(tag: String, got: Int, want: Int) raises:
    if got != want:
        raise Error(
            "check_ported_policy: " + tag + " = " + String(got)
            + ", upstream's formula gives " + String(want)
        )


def check_ported_policy_matches_upstream() raises:
    """`gemm/ported/linalg/contractions.mojo` against RAFT's own arithmetic,
    and against `core/gemm.mojo`'s independent flattening of the same policy.

    Two reasons this check exists rather than the file just sitting there.

    (1) A `ported/` file that nothing compiles is a file that rots. This is
    the only thing in the tree that imports it, so it is the only thing that
    would catch a transcription that stopped parsing.

    (2) `core/gemm.mojo:195-208` carries `Policy4x4<float, 4>` flattened into
    seven hand-copied integers. Two transcriptions of one upstream table is
    two chances to mis-copy it, so the parameterized one is required to agree
    with the flattened one. If they ever disagree, ONE of them is wrong and
    upstream decides which -- `PORTING_RULES.md` 0c.
    """
    # Policy4x4<float, 4>, the one RAFT's float distance kernels instantiate.
    _eq("Policy4x4 Nthreads", Policy4x4Float.nthreads, 256)
    _eq("Policy4x4 Mblk", Policy4x4Float.mblk, 64)
    _eq("Policy4x4 Nblk", Policy4x4Float.nblk, 64)
    _eq("Policy4x4 LdgThRow", Policy4x4Float.ldg_th_row, 8)
    _eq("Policy4x4 LdgPerThX", Policy4x4Float.ldg_per_th_x, 2)
    _eq("Policy4x4 LdgPerThY", Policy4x4Float.ldg_per_th_y, 2)
    _eq("Policy4x4 LdgRowsX", Policy4x4Float.ldg_rows_x, 32)
    _eq("Policy4x4 LdgRowsY", Policy4x4Float.ldg_rows_y, 32)
    _eq("Policy4x4 SmemStride", Policy4x4Float.smem_stride, 36)
    _eq("Policy4x4 SmemPageX", Policy4x4Float.smem_page_x, 2304)
    _eq("Policy4x4 SmemPage", Policy4x4Float.smem_page, 4608)
    _eq("Policy4x4 SmemSize", Policy4x4Float.smem_size, 36864)

    # The same policy as `core/gemm.mojo` flattened it, independently.
    _eq("vs core/gemm Nthreads", Policy4x4Float.nthreads, GEMM_THREADS)
    _eq("vs core/gemm Mblk", Policy4x4Float.mblk, GEMM_MBLK)
    _eq("vs core/gemm Nblk", Policy4x4Float.nblk, GEMM_NBLK)
    _eq("vs core/gemm SmemStride", Policy4x4Float.smem_stride, GEMM_SMEM_STRIDE)
    _eq("vs core/gemm SmemPageX", Policy4x4Float.smem_page_x, GEMM_SMEM_PAGE_X)
    _eq("vs core/gemm SmemPageY", Policy4x4Float.smem_page_y, GEMM_SMEM_PAGE_Y)

    # Policy4x4Skinny<float, 4>: their small-k policy, Kblk 8 and 8x8 threads.
    _eq("Skinny Nthreads", Policy4x4SkinnyFloat.nthreads, 64)
    _eq("Skinny Mblk", Policy4x4SkinnyFloat.mblk, 32)
    _eq("Skinny SmemStride", Policy4x4SkinnyFloat.smem_stride, 12)
    _eq("Skinny SmemSize", Policy4x4SkinnyFloat.smem_size, 6144)

    # Policy2x8<float, 1>: the shape `fused_l2_knn` is instantiated at
    # (FKNN_KBLK 16, MBLK 16, NBLK 256, THREADS 256).
    _eq("Policy2x8 Nthreads", Policy2x8Float.nthreads, 256)
    _eq("Policy2x8 Kblk", Policy2x8Float.kblk, 16)
    _eq("Policy2x8 Mblk", Policy2x8Float.mblk, 16)
    _eq("Policy2x8 Nblk", Policy2x8Float.nblk, 256)
    _eq("Policy2x8 LdgPerThY", Policy2x8Float.ldg_per_th_y, 16)

    # The COLUMN-MAJOR policy and their relocated static_assert.
    _eq("Col SmemStride", Policy4x4FloatCol.smem_stride, 68)
    _eq("Col SmemSize", Policy4x4FloatCol.smem_size, 34816)
    assert_col_policy_square[4, 4, 16, 16]()
    var raised = False
    try:
        assert_col_policy_square[4, 2, 16, 16]()
    except:
        raised = True
    if not raised:
        raise Error(
            "check_ported_policy: assert_col_policy_square did not raise on"
            " Mblk != Nblk, so RAFT's static_assert is not actually enforced."
        )

    print(
        "    Policy4x4<float,4>: Nthreads 256, Mblk/Nblk 64, SmemStride 36,"
        " SmemSize 36864 B; agrees with core/gemm.mojo's flattening"
    )
    print("check_ported_policy_matches_upstream OK [" + _mode_name() + "]")


def check_leaf_partition_is_a_pure_function_of_k() raises:
    """Contract section 6. The partition table, asserted at named `k`.

    This cannot prove that `contract_leaf_size` does not read a machine
    number -- nothing running on one machine can. What it CAN do, and what
    row 7's class of defect needs, is pin the VALUES, so that a later edit
    which makes the leaf size depend on the core count, the occupancy or
    (section 6.1's specific hazard) on `m` and `n` moves a number here and
    fails. Phase 3 owes the sabotage that demonstrates it fails.

    The `m`/`n` independence is asserted directly: `contract_leaf_size` takes
    no `m` and no `n`, so the assertion is that the SAME `k` gives the same
    `P` at wildly different output shapes, which is the batch-invariance
    clause in its weakest testable form.
    """
    var ks = List[Int]()
    var want_l = List[Int]()
    ks.append(0)
    want_l.append(1)
    ks.append(1)
    want_l.append(1)
    ks.append(64)
    want_l.append(64)
    ks.append(128)
    want_l.append(128)
    ks.append(129)
    want_l.append(128)
    ks.append(1024)
    want_l.append(128)
    ks.append(131072)
    want_l.append(128)
    # 131072/128 = 1024 leaves exactly: the last k the uncapped branch serves.
    ks.append(131200)
    want_l.append(129)
    # 131200/128 = 1025 > MAX_LEAVES, so the capped branch takes over:
    # ceil(131200/1024) = 129.
    ks.append(4000000)
    want_l.append(3907)

    for t in range(len(ks)):
        var k = ks[t]
        var got = contract_leaf_size(k)
        var p = contract_leaf_count(k)
        print(
            "    k = "
            + String(k)
            + "  ->  L = "
            + String(got)
            + "  P = "
            + String(p)
        )
        if got != want_l[t]:
            raise Error(
                "check_leaf_partition: at k = "
                + String(k)
                + " the contract leaf size is "
                + String(got)
                + ", the profile requires "
                + String(want_l[t])
                + ". Contract section 6."
            )
        if p > CONTRACT_MAX_LEAVES:
            raise Error(
                "check_leaf_partition: at k = "
                + String(k)
                + " the leaf COUNT is "
                + String(p)
                + ", above the profile cap "
                + String(CONTRACT_MAX_LEAVES)
                + ". Contract section 6.3."
            )
        # Section 6.2: P = ceil(k/L) implies (P-1)*L < k, so NO EMPTY LEAF.
        if p > 0:
            if leaf_begin(p - 1, got) >= k:
                raise Error(
                    "check_leaf_partition: at k = "
                    + String(k)
                    + " the last leaf is EMPTY. Contract section 6.2 says the"
                    " derivation order forbids that."
                )
            if leaf_end(p - 1, got, k) != k:
                raise Error(
                    "check_leaf_partition: at k = "
                    + String(k)
                    + " the leaves do not cover k."
                )
    print(
        "    profile: K_LEAF_MIN = "
        + String(CONTRACT_K_LEAF_MIN)
        + "  MAX_LEAVES = "
        + String(CONTRACT_MAX_LEAVES)
    )
    print("check_leaf_partition_is_a_pure_function_of_k OK [" + _mode_name() + "]")


def check_oracle_matches_the_contract_spelling() raises:
    """THE REACH PROOF FOR THE TWO HELPERS.

    `gemm_oracle` is written with `identical_mul_add` and `ftz`, which are
    comptime-gated. `_contract_cell` is the same arithmetic written with an
    explicit `fma` and an explicit flush. Under IDENTICAL they must agree bit
    for bit -- that is what says the helpers actually compile to the contract
    rather than to something that happens to be close.

    Under FAST they are two different arithmetics and the check REPORTS. The
    report is worth reading: on Apple the `fma` seam agrees anyway (row 9's
    correction -- Metal contracts), so a difference here under FAST comes
    from the FLUSH, not from the contraction.

    THE FIXTURE HAD TO BE REPLACED ONCE AND THAT IS WORTH RECORDING. The
    first version put the subnormal in the accumulator and then added `1.0`
    to it, which swallows the subnormal: both spellings returned `1.0` and
    the check reported "agree" in FAST mode having tested nothing. It is F4a's
    fixture now, where the subnormal IS the answer, so the flush axis decides
    the output and the FAST report is a real one. `_separates` is called on
    the two arms explicitly, so a fixture that stops separating fails loudly
    instead of passing quietly.
    """
    var k = 2
    var a = List[Float32]()
    a.append(_f32(UInt32(0x00C00000)))  # 1.5 * 2^-126, NORMAL
    a.append(_f32(UInt32(0x00800000)))  # 2^-126, NORMAL
    var b = List[Float32]()
    b.append(Float32(1.0))
    b.append(Float32(-1.0))

    # First: prove the fixture can see the flush axis at all. If these two
    # agree the check below is vacuous whatever it reports.
    _separates(
        "flush axis live",
        "contract (flush)",
        _eval_cell(
            a, b, OP_NT, 0, 0, 1, 1, k, k, True, True, FOLD_SERIAL_ZERO_SEED
        ),
        "no flush",
        _eval_cell(
            a, b, OP_NT, 0, 0, 1, 1, k, k, True, False, FOLD_SERIAL_ZERO_SEED
        ),
    )

    var oracle = gemm_oracle_cell(a, b, OP_NT, 0, 0, 1, 1, k, contract_leaf_size(k))
    var contract = _contract_cell(a, b, OP_NT, 0, 0, 1, 1, k)
    print(
        "    gated oracle = "
        + _show(oracle)
        + "   explicit contract spelling = "
        + _show(contract)
    )
    comptime if IDENTICAL_BUILD:
        if _bits(oracle) != _bits(contract):
            raise Error(
                "check_oracle_matches_the_contract_spelling: under IDENTICAL"
                " the oracle built from `identical_mul_add`/`ftz` does NOT"
                " equal the contract written out explicitly. Either a helper"
                " is not compiling to what its docstring says or the oracle"
                " has drifted from the contract."
            )
        print(
            "check_oracle_matches_the_contract_spelling OK ["
            + _mode_name()
            + "]: the gated helpers ARE the contract"
        )
    else:
        var same = _bits(oracle) == _bits(contract)
        print(
            "check_oracle_matches_the_contract_spelling REPORT ["
            + _mode_name()
            + "]: agree = "
            + String(same)
            + " (under FAST the oracle is the FAST spelling; a difference"
            " here is the FLUSH axis, since Apple's FAST arm contracts"
            " anyway -- IDENTITY_PATHS row 9's correction)"
        )


def check_orientations_agree() raises:
    """Contract section 3: NN, NT and TN are ONE numerical implementation.

    Build one logical `A_eff` (m x k) and one logical `B_eff` (k x n), lay
    them out three ways, and require all three orientations to return the
    same bits for every cell. This is a property of the ORACLE (the device
    version is Phase 3's), and it is the check that would fail if somebody
    "optimized" one orientation's inner loop.

    `k = 200 > K_LEAF_MIN`, so the partitioned path is exercised rather than
    the single-leaf one.
    """
    var m = 3
    var n = 5
    var k = 200
    var ae = List[Float32]()  # m x k, row-major -- the LOGICAL left operand
    var be = List[Float32]()  # k x n, row-major -- the LOGICAL right operand
    for i in range(m):
        for p in range(k):
            ae.append(_spread_half(i * 977 + p, 11) * Float32(1 + (p % 7)))
    for p in range(k):
        for j in range(n):
            be.append(_spread_half(p * 31 + j, 23) * Float32(1 + (p % 5)))

    # NN takes them as they are.
    var a_nn = ae.copy()
    var b_nn = be.copy()
    # NT wants B as n x k.
    var b_nt = List[Float32]()
    for j in range(n):
        for p in range(k):
            b_nt.append(be[p * n + j])
    # TN wants A as k x m.
    var a_tn = List[Float32]()
    for p in range(k):
        for i in range(m):
            a_tn.append(ae[i * k + p])

    var c_nn = gemm_oracle_at_leaf(
        a_nn, b_nn, OP_NN, m, n, k, contract_leaf_size(k)
    )
    var c_nt = gemm_oracle_at_leaf(
        a_nn, b_nt, OP_NT, m, n, k, contract_leaf_size(k)
    )
    var c_tn = gemm_oracle_at_leaf(
        a_tn, b_nn, OP_TN, m, n, k, contract_leaf_size(k)
    )
    # THE SABOTAGE THAT GIVES THE AGREEMENT ITS TEETH. Run OP_NT against the
    # NN-shaped `B` -- the wrong layout, the exact mistake this check exists
    # to catch -- and require the answer to MOVE. Without this, three
    # orientations that all silently read the same addresses would also pass.
    var c_wrong = gemm_oracle_at_leaf(
        a_nn, b_nn, OP_NT, m, n, k, contract_leaf_size(k)
    )
    _separates(
        "orientation sabotage",
        "NT on n x k B",
        c_nt[0],
        "NT on k x n B (wrong)",
        c_wrong[0],
    )

    var bad = 0
    for cell in range(m * n):
        if _bits(c_nn[cell]) != _bits(c_nt[cell]):
            bad += 1
        if _bits(c_nn[cell]) != _bits(c_tn[cell]):
            bad += 1
    print(
        "    cell (0,0):  "
        + op_name(OP_NN)
        + " = "
        + _show(c_nn[0])
        + "   "
        + op_name(OP_NT)
        + " = "
        + _show(c_nt[0])
        + "   "
        + op_name(OP_TN)
        + " = "
        + _show(c_tn[0])
    )
    if bad != 0:
        raise Error(
            "check_orientations_agree: "
            + String(bad)
            + " of "
            + String(2 * m * n)
            + " comparisons differ. Contract section 3 requires the three"
            " orientations to be one numerical implementation."
        )
    print(
        "check_orientations_agree OK ["
        + _mode_name()
        + "]: "
        + String(m * n)
        + " cells, NN == NT == TN bit for bit"
    )


# ===========================================================================
# F1 / F2: THE PARTITION AXIS
# ===========================================================================
#
# THE CONSTRUCTION, once, because F1, F2 and F5 all use it.
#
# `2^24` is the smallest float32 whose ulp is 2. So `2^24 + 1` is exactly
# halfway between `2^24` and `2^24 + 2`, and round-half-to-even returns
# `2^24`: adding 1 to it is a NO-OP. Put `2^24` at `p = 0` and `1.0`
# everywhere else and the SERIAL chain swallows every one of the `k-1`
# additions and returns `2^24` exactly.
#
# Partition the same k and the ones no longer meet `2^24`: each leaf after
# the first accumulates its own ones into a small exact integer (`L`, which
# is a power of two and exactly representable), and the FOLD adds those to
# `2^24` -- where they ARE representable, because `L` is a multiple of the
# ulp. So the split answer is `2^24 + L*(P-1)`.
#
# Nothing about this is delicate: every quantity is an exact integer, no tie
# can hide it, and the difference is `L*(P-1)`, not one ulp.


def _swallow_fixture(k: Int) -> List[Float32]:
    """`A = [2^24, 1, 1, ..., 1]`. Paired with `B = [1, 1, ..., 1]`."""
    var a = List[Float32]()
    a.append(TWO_P24)
    for _ in range(1, k):
        a.append(Float32(1.0))
    return a^


def check_f1_serial_vs_splitk() raises:
    """FIXTURE F1: SERIAL summation separated from SPLIT-K summation.

    One axis moved: the leaf size. `leaf = k` is one leaf (the serial
    reference, `gemm_oracle_serial`); `leaf = contract_leaf_size(k)` is the
    contract's partition.
    """
    var k = 1024
    var a = _swallow_fixture(k)
    var b = _ones(k)
    var el = contract_leaf_size(k)
    var p = leaf_count(k, el)

    var serial = _eval_cell(
        a, b, OP_NT, 0, 0, 1, 1, k, k, True, True, FOLD_SERIAL_ZERO_SEED
    )
    var split = _eval_cell(
        a, b, OP_NT, 0, 0, 1, 1, k, el, True, True, FOLD_SERIAL_ZERO_SEED
    )
    print(
        "    k = "
        + String(k)
        + ", contract L = "
        + String(el)
        + ", P = "
        + String(p)
        + "; predicted split - serial = L*(P-1) = "
        + String(el * (p - 1))
    )
    _separates("F1", "serial (P=1)", serial, "split (P=" + String(p) + ")", split)
    # The prediction is part of the proof: if the difference is not exactly
    # `L*(P-1)` then the fixture is separating for a reason nobody stated,
    # and an unexplained separation is not evidence either.
    var want = TWO_P24 + Float32(el * (p - 1))
    if _bits(split) != _bits(want):
        raise Error(
            "check_f1: the split answer is "
            + _show(split)
            + " but the construction predicts "
            + _show(want)
            + ". The fixture separates for an unexplained reason."
        )
    if _bits(serial) != _bits(TWO_P24):
        raise Error(
            "check_f1: the serial answer is "
            + _show(serial)
            + " but the construction predicts "
            + _show(TWO_P24)
            + " (every +1 swallowed)."
        )
    print("check_f1_serial_vs_splitk OK [" + _mode_name() + "]")


def check_f2_partition_count() raises:
    """FIXTURE F2: ONE partition count separated from ANOTHER.

    Same fixture, same everything, two leaf sizes -- and NEITHER is the
    serial one, so this is strictly stronger than F1. It is the fixture a
    Phase 3 sabotage of `CONTRACT_K_LEAF_MIN` or `CONTRACT_MAX_LEAVES` must
    fail on.
    """
    var k = 1024
    var a = _swallow_fixture(k)
    var b = _ones(k)

    var p8 = leaf_count(k, 128)
    var p16 = leaf_count(k, 64)
    var v8 = _eval_cell(
        a, b, OP_NT, 0, 0, 1, 1, k, 128, True, True, FOLD_SERIAL_ZERO_SEED
    )
    var v16 = _eval_cell(
        a, b, OP_NT, 0, 0, 1, 1, k, 64, True, True, FOLD_SERIAL_ZERO_SEED
    )
    _separates(
        "F2",
        "L=128 (P=" + String(p8) + ")",
        v8,
        "L=64 (P=" + String(p16) + ")",
        v16,
    )
    var w8 = TWO_P24 + Float32(128 * (p8 - 1))
    var w16 = TWO_P24 + Float32(64 * (p16 - 1))
    if _bits(v8) != _bits(w8) or _bits(v16) != _bits(w16):
        raise Error(
            "check_f2: the two partitions do not match the construction's"
            " prediction ("
            + _show(w8)
            + " / "
            + _show(w16)
            + ")."
        )
    # A third count, so the check is not a two-point coincidence.
    var v4 = _eval_cell(
        a, b, OP_NT, 0, 0, 1, 1, k, 256, True, True, FOLD_SERIAL_ZERO_SEED
    )
    _separates("F2b", "L=128", v8, "L=256", v4)
    print("check_f2_partition_count OK [" + _mode_name() + "]")


# ===========================================================================
# F3: THE CONTRACTION AXIS
# ===========================================================================


def check_f3_fused_vs_unfused() raises:
    """FIXTURE F3: FUSED multiply-add separated from UNFUSED.

    Lifted from `mojo_only/ieee_arith_check.mojo`'s ARM 6, the BUILT-TO-
    SEPARATE arm that corrected IDENTITY_PATHS row 9, and embedded in a
    `k = 2` GEMM so that the thing separated is this contract's inner loop
    rather than a bare expression.

    THE CONSTRUCTION. Take `a0`, `b0` with half-width mantissas, so the exact
    product `a0*b0` needs more bits than float32 keeps. Let `q = fl(a0*b0)`.
    Lay the k axis out as

        A = [ 1.0, a0 ]        B = [ -q,  b0 ]

    Then the accumulator after `p = 0` is exactly `-q` under either spelling,
    and at `p = 1`

        unfused:  fl(a0*b0) + (-q)  ==  q - q  ==  +0.0, EXACTLY, always
        fused:    a0*b0 - q         ==  the rounding error, NONZERO

    so one bit of the answer IS the contraction. Nothing about the magnitudes
    is delicate and no tie can hide it.

    THE SELF-CHECK THAT MATTERS, AND IT HAS ALREADY EARNED ITS KEEP. On this
    fixture the unfused answer must be exactly `+0.0` BY CONSTRUCTION, so the
    check verifies that rather than assuming it. The first version of
    `_unfused` here wrote the multiply on its own statement, and the check
    refused itself on 1,621 of 4,096 patterns: Mojo contracted across the two
    statements on the HOST. `_rounded_product` records the measurement and
    the contraction-proof spelling that replaced it. Had the check merely
    reported a separation, it would have reported a real difference and
    attributed it to the wrong axis.

    Patterns whose product happens to be exact carry no information and are
    skipped and counted, exactly as ARM 6 does.
    """
    var trials = 4096
    var separating = 0
    var contracted_host = 0
    var first_a = Float32(0.0)
    var first_b = Float32(0.0)
    var first_fused = Float32(0.0)
    var first_unfused = Float32(0.0)

    for t in range(trials):
        var a0 = _spread_half(t, 0xC0FFEE)
        var b0 = _spread_half(t, 0xBEEF01)
        # `fl(a0*b0)` with no multiply in it -- see `_rounded_product`.
        var q = _rounded_product(a0, b0)
        var a = List[Float32]()
        a.append(Float32(1.0))
        a.append(a0)
        var b = List[Float32]()
        b.append(-q)
        b.append(b0)

        var vf = _eval_cell(
            a, b, OP_NT, 0, 0, 1, 1, 2, 2, True, True, FOLD_SERIAL_ZERO_SEED
        )
        var vu = _eval_cell(
            a, b, OP_NT, 0, 0, 1, 1, 2, 2, False, True, FOLD_SERIAL_ZERO_SEED
        )
        if _bits(vf) == UInt32(0x00000000):
            continue  # the product was exact; this pattern says nothing
        if _bits(vu) != UInt32(0x00000000):
            contracted_host += 1
            continue
        if separating == 0:
            first_a = a0
            first_b = b0
            first_fused = vf
            first_unfused = vu
        separating += 1

    print(
        "    "
        + String(separating)
        + " of "
        + String(trials)
        + " patterns separate; "
        + String(contracted_host)
        + " had a non-zero unfused answer (host contraction)"
    )
    if contracted_host != 0:
        raise Error(
            "check_f3: on "
            + String(contracted_host)
            + " patterns the UNFUSED spelling did not return exactly +0.0,"
            " which by construction it must. The host contracted"
            " `_unfused`'s standalone multiply into an fma, so this fixture"
            " cannot attribute a difference to the contraction axis. That is"
            " a finding about Mojo codegen, not a reason to relax the check."
        )
    if separating == 0:
        raise Error(
            "check_f3: NO pattern separated fused from unfused. That is"
            " exactly the failure IDENTITY_PATHS row 9 records for the 2^20"
            " hashed patterns -- the generator is producing exact products."
            " Fix the generator; do not conclude anything about a backend."
        )
    print(
        "    a0 = "
        + _show(first_a)
        + "   b0 = "
        + _show(first_b)
        + "   c = -fl(a0*b0)"
    )
    _separates("F3", "fused", first_fused, "unfused", first_unfused)
    print("check_f3_fused_vs_unfused OK [" + _mode_name() + "]")


# ===========================================================================
# F4: THE DENORMAL AXIS
# ===========================================================================


def check_f4_ftz_vs_gradual_underflow() raises:
    """FIXTURE F4: FLUSH-TO-ZERO separated from GRADUAL UNDERFLOW.

    Two sub-fixtures, because contract section 5 has two kinds of seam and
    they fail differently.

    F4a -- THE RESULT SEAM, and it is the one that matters most because
    `ftz`'s own docstring warns that intermediates inside an expression are
    unreachable on a non-FTZ backend. Both operands are NORMAL (`1.5*2^-126`
    and `2^-126`, either side of the boundary but neither below it), so
    nothing is flushed on the way in. Their difference is `2^-127`, a
    SUBNORMAL, and it is the running accumulator. Flush it and the answer is
    `+0.0`; carry it and the answer is `0x00400000`. This is the exact shape
    of a divergence between Metal (which flushed on the spot) and CUDA (which
    carried it).

    F4b -- THE OPERAND SEAM. A subnormal operand `2^-127` times `2^20`. With
    the flush the product is zero; without it the product is `2^-107`, a
    perfectly ordinary NORMAL number. So the divergence is not confined to
    the subnormal range: one flushed operand puts a full-magnitude value into
    a cell or does not.
    """
    # ---- F4a: the accumulator underflows -----------------------------
    var a4 = List[Float32]()
    a4.append(_f32(UInt32(0x00C00000)))  # 1.5 * 2^-126   NORMAL
    a4.append(_f32(UInt32(0x00800000)))  # 1.0 * 2^-126   NORMAL (the min)
    var b4 = List[Float32]()
    b4.append(Float32(1.0))
    b4.append(Float32(-1.0))

    var ftz_on = _eval_cell(
        a4, b4, OP_NT, 0, 0, 1, 1, 2, 2, True, True, FOLD_SERIAL_ZERO_SEED
    )
    var ftz_off = _eval_cell(
        a4, b4, OP_NT, 0, 0, 1, 1, 2, 2, True, False, FOLD_SERIAL_ZERO_SEED
    )
    print(
        "    F4a operands: "
        + _show(a4[0])
        + " and "
        + _show(a4[1])
        + " (both NORMAL); their difference is 2^-127, subnormal"
    )
    _separates("F4a", "FTZ", ftz_on, "gradual", ftz_off)
    if _bits(ftz_on) != UInt32(0x00000000):
        raise Error(
            "check_f4: the FTZ answer is " + _show(ftz_on) + ", expected +0.0."
        )
    if _bits(ftz_off) != UInt32(0x00400000):
        raise Error(
            "check_f4: the gradual-underflow answer is "
            + _show(ftz_off)
            + ", expected 0x00400000 = 2^-127."
        )

    # ---- F4b: an operand is subnormal --------------------------------
    var a4b = List[Float32]()
    a4b.append(_f32(UInt32(0x00400000)))  # 2^-127, SUBNORMAL
    var b4b = List[Float32]()
    b4b.append(Float32(1048576.0))  # 2^20, exact
    var on_b = _eval_cell(
        a4b, b4b, OP_NT, 0, 0, 1, 1, 1, 1, True, True, FOLD_SERIAL_ZERO_SEED
    )
    var off_b = _eval_cell(
        a4b, b4b, OP_NT, 0, 0, 1, 1, 1, 1, True, False, FOLD_SERIAL_ZERO_SEED
    )
    print(
        "    F4b operand "
        + _show(a4b[0])
        + " (SUBNORMAL) times 2^20: the unflushed product is a NORMAL number"
    )
    _separates("F4b", "FTZ", on_b, "gradual", off_b)
    print("check_f4_ftz_vs_gradual_underflow OK [" + _mode_name() + "]")


# ===========================================================================
# F5: THE FOLD TOPOLOGY AXIS
# ===========================================================================


def check_f5_balanced_vs_serial_fold() raises:
    """FIXTURE F5: a BALANCED FOLD separated from an ASCENDING SERIAL FOLD.

    The leaf partition is held FIXED at the contract's, so the only thing
    that moves is the topology of the fold over the partials. That is the
    choice contract section 7.2 makes and this is the fixture that says the
    choice is a choice.

    THE CONSTRUCTION. Make the partials `[2^24, 1, 1, 1, 1, 1, 1, 1]` by
    putting a single `2^24` in leaf 0 and a single `1.0` in each later leaf,
    with zeros everywhere else (`fma(0, 1, acc) == acc`, exactly, so the
    zeros are inert).

        serial:  2^24, then +1 seven times, each swallowed  ->  2^24
        halving: [2^24+1 -> 2^24,  2, 2, 2]
                 [2^24+2,          4]
                 [2^24+6]                                   ->  2^24 + 6

    Every intermediate is an exact integer: `2^24 + 2` and `2^24 + 6` are
    representable because ulp(2^24) is 2, and `2^24 + 1` is the halfway case
    round-half-to-even discards. The separation is 6, not one ulp.
    """
    var k = 1024
    var el = contract_leaf_size(k)  # 128
    var p = leaf_count(k, el)  # 8
    var a = _zeros(k)
    a[0] = TWO_P24
    for j in range(1, p):
        a[leaf_begin(j, el)] = Float32(1.0)
    var b = _ones(k)

    var serial = _eval_cell(
        a, b, OP_NT, 0, 0, 1, 1, k, el, True, True, FOLD_SERIAL_ZERO_SEED
    )
    var halving = _eval_cell(
        a, b, OP_NT, 0, 0, 1, 1, k, el, True, True, FOLD_HALVING
    )
    print(
        "    partials = [2^24, 1 x " + String(p - 1) + "], P = " + String(p)
    )
    _separates("F5", "serial ascending", serial, "balanced halving", halving)
    if _bits(serial) != _bits(TWO_P24):
        raise Error(
            "check_f5: the serial fold gave "
            + _show(serial)
            + ", the construction predicts "
            + _show(TWO_P24)
            + "."
        )
    if _bits(halving) != _bits(TWO_P24 + Float32(6.0)):
        raise Error(
            "check_f5: the halving fold gave "
            + _show(halving)
            + ", the construction predicts "
            + _show(TWO_P24 + Float32(6.0))
            + "."
        )
    print("check_f5_balanced_vs_serial_fold OK [" + _mode_name() + "]")


# ===========================================================================
# F6: SIGNED ZERO AND CANCELLATION
# ===========================================================================


def _minus_zero_leaves(k: Int, el: Int) -> List[Float32]:
    """`A` such that EVERY leaf's partial is exactly `-0.0`.

    A leaf's partial can only be `-0.0` through `ftz` of a negative subnormal
    (contract section 9.2(a): a `+0.0` seed makes it unreachable from
    products), so each leaf ends with `2^-126` then `-1.5*2^-126`, giving
    `-2^-127` which the flush turns into `-0.0`.

    THE PLACEMENT IS AT THE END OF THE LEAF AND THAT IS NOT COSMETIC. Put the
    pair at the START and the trailing zero products erase the sign:
    `fma(0, 1, -0.0)` is `+0.0 + (-0.0)`, which is `+0.0`. The fixture would
    then be all `+0.0` partials and would separate nothing -- which is itself
    a small demonstration of section 9.2's mechanism.
    """
    var a = _zeros(k)
    var p = leaf_count(k, el)
    for j in range(p):
        var e = leaf_end(j, el, k)
        a[e - 2] = _f32(UInt32(0x00800000))  # 2^-126
        a[e - 1] = -_f32(UInt32(0x00C00000))  # -1.5 * 2^-126
    return a^


def check_f6_signed_zero_and_cancellation() raises:
    """FIXTURE F6: the signed-zero and cancellation cases.

    IDENTITY_PATHS row 13 is a real defect in this repository where `-0.0`
    and `+0.0` compare equal, so which one survived a fold was decided by
    ORDER and the sign reached the model. Contract section 9.2 answers it
    with a `+0.0` seed and an UNCONDITIONAL fold. These are the fixtures that
    make both halves falsifiable.

    F6a -- THE SEED. All `P` partials are exactly `-0.0`. The contract's fold
    (seeded `+0.0`) returns `+0.0`; the same fold seeded with `partials[0]`
    returns `-0.0`. Two spellings of "sum these partials", two different
    output SIGNS, from identical inputs.

    F6a-P1 -- THE UNCONDITIONAL CLAUSE. The same thing at `P == 1`, where the
    seeded-with-first spelling IS "skip the fold, one leaf needs no folding".
    This is the exact defect section 9.2(b) forbids, and it is the reason the
    contract says `P == 1` is a one-term fold and not a bypass.

    F6a-noftz -- the third answer. With gradual underflow the partials are
    `-2^-127` each and neither fold returns a zero at all, so the sign
    question does not even arise the same way. Printed because a reader
    should see that the flush is what CREATES the negative zero here.

    F6b -- CANCELLATION. `[2^24, 1, ..., -2^24, -1, ...]` whose EXACT sum is
    zero. Serial: the `+1` is swallowed by `2^24`, the `-2^24` cancels, and
    the `-1` survives -> `-1.0`. Partitioned at the contract's `L = 128`:
    leaf 0 folds to `2^24` and leaf 4 folds to `-2^24` (its `-1` swallowed
    the same way), and the fold cancels them -> `+0.0`. Same exact answer,
    two computed answers a whole unit apart. Cancellation is where the
    partition choice stops being a last-bit question.
    """
    # ---- F6a: the fold seed decides the SIGN --------------------------
    var k = 1024
    var el = contract_leaf_size(k)
    var p = leaf_count(k, el)
    var a = _minus_zero_leaves(k, el)
    var b = _ones(k)

    var seeded_zero = _eval_cell(
        a, b, OP_NT, 0, 0, 1, 1, k, el, True, True, FOLD_SERIAL_ZERO_SEED
    )
    var seeded_first = _eval_cell(
        a, b, OP_NT, 0, 0, 1, 1, k, el, True, True, FOLD_SERIAL_FIRST_SEED
    )
    print("    F6a: all " + String(p) + " partials are exactly -0.0")
    _separates(
        "F6a", "+0.0 seed (contract)", seeded_zero, "first-partial seed",
        seeded_first,
    )
    if _bits(seeded_zero) != UInt32(0x00000000):
        raise Error(
            "check_f6: the contract fold gave "
            + _show(seeded_zero)
            + ", section 9.2(a) requires +0.0 (0x00000000)."
        )
    if _bits(seeded_first) != UInt32(0x80000000):
        raise Error(
            "check_f6: the first-seeded fold gave "
            + _show(seeded_first)
            + ", the construction predicts -0.0 (0x80000000). If this is"
            " +0.0 the fixture's partials are not -0.0 and F6a is vacuous."
        )

    # ---- F6a-P1: the UNCONDITIONAL clause, at one leaf ----------------
    var k1 = CONTRACT_K_LEAF_MIN
    var el1 = contract_leaf_size(k1)  # == k1, so P == 1
    var p1 = leaf_count(k1, el1)
    if p1 != 1:
        raise Error(
            "check_f6: F6a-P1 needs P == 1 at k = "
            + String(k1)
            + ", got "
            + String(p1)
        )
    var a1 = _minus_zero_leaves(k1, el1)
    var b1 = _ones(k1)
    var folded = _eval_cell(
        a1, b1, OP_NT, 0, 0, 1, 1, k1, el1, True, True, FOLD_SERIAL_ZERO_SEED
    )
    var bypassed = _eval_cell(
        a1, b1, OP_NT, 0, 0, 1, 1, k1, el1, True, True, FOLD_SERIAL_FIRST_SEED
    )
    print("    F6a-P1: ONE leaf, k = " + String(k1))
    _separates(
        "F6a-P1", "fold runs (contract)", folded, "fold bypassed", bypassed
    )

    # ---- F6a-noftz: the flush is what makes the -0.0 ------------------
    var no_flush = _eval_cell(
        a, b, OP_NT, 0, 0, 1, 1, k, el, True, False, FOLD_SERIAL_ZERO_SEED
    )
    print(
        "    F6a-noftz: with gradual underflow the same fixture gives "
        + _show(no_flush)
        + " (partials are -2^-127, not -0.0, so no sign question arises)"
    )
    if _bits(no_flush) == UInt32(0x00000000) or _bits(
        no_flush
    ) == UInt32(0x80000000):
        raise Error(
            "check_f6: with the flush OFF the answer is still a zero ("
            + _show(no_flush)
            + "), so the fixture does not show that the flush is what creates"
            " the negative zero."
        )

    # ---- F6b: cancellation ------------------------------------------
    var kc = 1024
    var elc = contract_leaf_size(kc)
    var pc = leaf_count(kc, elc)
    var ac = _zeros(kc)
    ac[0] = TWO_P24
    ac[1] = Float32(1.0)
    ac[leaf_begin(pc // 2, elc)] = -TWO_P24
    ac[leaf_begin(pc // 2, elc) + 1] = Float32(-1.0)
    var bc = _ones(kc)

    var serial_c = _eval_cell(
        ac, bc, OP_NT, 0, 0, 1, 1, kc, kc, True, True, FOLD_SERIAL_ZERO_SEED
    )
    var split_c = _eval_cell(
        ac, bc, OP_NT, 0, 0, 1, 1, kc, elc, True, True, FOLD_SERIAL_ZERO_SEED
    )
    print(
        "    F6b: exact sum is 0; contract L = "
        + String(elc)
        + ", P = "
        + String(pc)
    )
    _separates("F6b", "serial (P=1)", serial_c, "contract partition", split_c)
    if _bits(serial_c) != _bits(Float32(-1.0)):
        raise Error(
            "check_f6: the serial cancellation answer is "
            + _show(serial_c)
            + ", the construction predicts -1.0."
        )
    if _bits(split_c) != UInt32(0x00000000):
        raise Error(
            "check_f6: the partitioned cancellation answer is "
            + _show(split_c)
            + ", the construction predicts +0.0."
        )
    print("check_f6_signed_zero_and_cancellation OK [" + _mode_name() + "]")


def main() raises:
    print("== gemm/mojo_only/gemm_oracle_check.mojo [" + _mode_name() + "] ==")
    print(
        "   contract: gemm/IDENTICAL_FP32_CONTRACT.md   oracle:"
        " gemm/mojo_only/gemm_oracle.mojo"
    )
    check_ieee_zero_assumptions()
    check_ported_policy_matches_upstream()
    check_leaf_partition_is_a_pure_function_of_k()
    check_oracle_matches_the_contract_spelling()
    check_orientations_agree()
    check_f1_serial_vs_splitk()
    check_f2_partition_count()
    check_f3_fused_vs_unfused()
    check_f4_ftz_vs_gradual_underflow()
    check_f5_balanced_vs_serial_fold()
    check_f6_signed_zero_and_cancellation()
    print("gemm oracle + fixtures: all green [" + _mode_name() + "]")
