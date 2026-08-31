# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device GEMM's gates: oracle agreement, launch invariance, batch
invariance -- and the sabotages that show each of them can fail.

Phase 2b of `IDENTICAL_GEMM_PLAN.md`. The kernel is
`gemm/original/gemm_identical.mojo`; the answer is
`gemm/original/gemm_oracle.mojo::gemm_oracle`; the contract is
`gemm/IDENTICAL_FP32_CONTRACT.md`, profile
`mojolearn.identical.gemm.fp32.v1`.

WHY EVERY GATE HERE HAS A SABOTAGE BESIDE IT
---------------------------------------------
`IDENTICAL_GEMM_PLAN.md`'s Phase 3 brief: *"A passing test whose fixture
cannot distinguish the alternatives is not evidence."* And row 9's correction
is the repository's own worst case -- 2^20 patterns that scored a contracting
backend as unfused because not one of them separated the two spellings. So
each gate below is run against five deliberately broken builds, selected by
`-D MOJOLEARN_GEMM_SABOTAGE_*`, and the report records the failure text. A
check that has never failed is a check nobody has tested.

    tools/with_identical_mode.sh pixi run mojo run \
        -D MOJOLEARN_GEMM_SABOTAGE_FOLD_STRIDE=1 \
        -I . gemm/original/gemm_device_check.mojo

MAIN RUNS EVERY GATE AND REPORTS EVERY VERDICT before it raises. That is
deliberate and it is not a softening: the process still exits non-zero if
anything failed. Stopping at the first failure under a sabotage build would
show one gate's opinion when the useful evidence is WHICH gates a given
defect reaches and which it walks past -- and "walks past" is exactly the
defect class `check_pinned_gemm_is_batch_invariant`'s first version had.

WHAT IS ASSERTED IN WHICH MODE
-------------------------------
- `check_device_is_launch_invariant` and `check_device_is_batch_invariant`
  are ASSERTED IN BOTH MODES. They are properties of the kernel's SHAPE --
  no float crosses a thread boundary, and the leaf partition comes from `k`
  alone -- not of the arithmetic pins, so FAST has no excuse for failing
  them and a FAST failure is a real defect.
- `check_device_matches_oracle` is ASSERTED under IDENTICAL and REPORTED
  under FAST, for the reason `core/gemm_identity_check.mojo` gives at the
  same seam: under FAST both sides are the unpinned spelling, the host CPU
  and the device backend are free to contract and to flush differently, and
  whether they agree is a measurement rather than a bug.

`[[mojo-string-float-roundtrip]]`: every float printed here carries its hex
bits beside its decimal.
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.memory import bitcast

from bench.gemm_shapes import (
    GEMM_SHAPE_COUNT,
    gemm_shape_k,
    gemm_shape_m,
    gemm_shape_n,
    gemm_shape_name,
    gemm_shape_op,
)
from bench.gemm_shapes import OP_NN as TBL_OP_NN
from bench.gemm_shapes import OP_NT as TBL_OP_NT
from bench.gemm_shapes import OP_TN as TBL_OP_TN
from gemm.original.gemm_identical import (
    GEMM_FOLD_LEVELS,
    GEMM_FOLD_SLOTS,
    GEMM_PLAN_COUNT,
    PLAN_FLAT,
    PLAN_SPLITK,
    _fold_drain,
    _fold_push,
    choose_gemm_plan,
    contract_partition,
    gemm_operand_strides,
    gemm_plan_name,
    gemm_sabotage_name,
    identical_gemm_into,
    identical_gemm_with_plan,
    identical_gemm_workspace_floats,
    identical_gemm_workspace_max_floats,
)
from gemm.original.gemm_oracle import (
    CONTRACT_MAX_LEAVES,
    OP_NN,
    OP_NT,
    OP_TN,
    contract_leaf_count,
    contract_leaf_size,
    fold_balanced_tree,
    gemm_oracle,
    op_name,
)
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, numeric_mode_name

comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

#: The value written into `C` before every launch. A cell still holding it
#: afterwards was NEVER WRITTEN, and nothing downstream of it is a comparison
#: of products. Contract section 8 makes this load-bearing at `k == 0`, where
#: the required answer `+0.0` must be STORED rather than the store skipped.
comptime POISON = Float32(-987654.0)


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _bits(x: Float32) -> UInt32:
    return bitcast[DType.uint32](x)


def _show(x: Float32) -> String:
    return String(x) + "/" + hex(_bits(x))


# ===========================================================================
# FIXTURE VALUES
# ===========================================================================


def _hash64(i: Int, salt: Int) -> UInt64:
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(salt + 1) * UInt64(
        0xBF58476D1CE4E5B9
    )
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


def _val(i: Int, salt: Int) -> Float32:
    """A signed value whose SIGNIFICAND IS 13 BITS and whose exponent spreads
    over eight binades.

    Two 13-bit significands need up to 26 bits to multiply exactly and
    float32 keeps 24, so **every product in these fixtures is inexact** --
    which is the requirement `gemm_oracle_check.mojo::_spread_half` documents
    and the reason full-width random mantissas are the wrong generator here:
    they separate nothing, because the accumulator's own rounding swallows
    the tail. The exponent spread is what makes the ORDER of the additions
    matter, so a wrong leaf boundary or a wrong pairing moves bits instead of
    cancelling out.
    """
    var h = _hash64(i, salt)
    var mant = Float32(1.0) + Float32(Int(h & UInt64(0xFFF))) / Float32(4096.0)
    var e = Int((h >> UInt64(13)) & UInt64(7)) - 4
    var scale = Float32(1.0)
    if e >= 0:
        for _ in range(e):
            scale = scale * Float32(2.0)
    else:
        for _ in range(-e):
            scale = scale * Float32(0.5)
    var v = mant * scale
    if (h >> UInt64(20)) & UInt64(1) == UInt64(1):
        return -v
    return v


def _val_subnormal(i: Int, salt: Int) -> Float32:
    """The same generator scaled so PRODUCTS land in the subnormal range.

    `2^-126` is the smallest normal. These operands are around `2^-64`, so
    every product is around `2^-128` -- below it -- and the running
    accumulator spends the whole leaf in the range contract section 5's row
    5c is about. On Metal the hardware flushes; on CUDA's default it would
    not; `ftz` is what makes the two the same array of bits, and this is the
    only fixture here that reaches that seam ON A DEVICE.
    """
    var v = _val(i, salt)
    var scale = Float32(1.0)
    for _ in range(64):
        scale = scale * Float32(0.5)
    return v * scale


def _fill(n_elems: Int, salt: Int, subnormal: Bool) -> List[Float32]:
    var v = List[Float32]()
    for i in range(n_elems):
        if subnormal:
            v.append(_val_subnormal(i, salt))
        else:
            v.append(_val(i, salt))
    return v^


def _a_elems(op: Int, m: Int, n: Int, k: Int) -> Int:
    return m * k


def _b_elems(op: Int, m: Int, n: Int, k: Int) -> Int:
    if op == OP_NT:
        return n * k
    return k * n


# ===========================================================================
# HOST PRELIMINARIES
# ===========================================================================


def check_fold_stack_depth_covers_the_profile() raises:
    """`GEMM_FOLD_LEVELS` slots are enough for every legal `P`.

    ASSERTED AGAINST `CONTRACT_MAX_LEAVES`, not against a comment. `D =
    ceil(log2 P)` levels plus level 0 means `floor(log2 P) + 1` occupied
    slots at worst, and the stack must hold every set bit of `P`
    simultaneously. If `MAX_LEAVES` ever moves, this fails here rather than
    silently dropping a leaf on the deepest shape in the table.
    """
    var needed = 0
    var w = CONTRACT_MAX_LEAVES
    while w > 0:
        needed += 1
        w = w // 2
    if GEMM_FOLD_LEVELS < needed:
        raise Error(
            "check_fold_stack_depth_covers_the_profile: the register fold"
            " stack has "
            + String(GEMM_FOLD_LEVELS)
            + " levels but CONTRACT_MAX_LEAVES = "
            + String(CONTRACT_MAX_LEAVES)
            + " needs "
            + String(needed)
        )
    if GEMM_FOLD_SLOTS < needed:
        raise Error(
            "check_fold_stack_depth_covers_the_profile: the SIMD register"
            " holding the fold stack has "
            + String(GEMM_FOLD_SLOTS)
            + " lanes and CONTRACT_MAX_LEAVES = "
            + String(CONTRACT_MAX_LEAVES)
            + " needs "
            + String(needed)
        )
    print(
        "check_fold_stack_depth_covers_the_profile OK:",
        GEMM_FOLD_LEVELS,
        "levels in",
        GEMM_FOLD_SLOTS,
        "slots covers P <=",
        CONTRACT_MAX_LEAVES,
        "which needs",
        needed,
    )


def _stack_fold(partials: List[Float32]) raises -> Float32:
    """The kernel's fold, driven from the host over the SAME `_fold_push` /
    `_fold_drain` the device calls."""
    var stack = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
    var occ = 0
    for t in range(len(partials)):
        if not _fold_push(stack, occ, partials[t]):
            raise Error(
                "_stack_fold: the fold stack OVERFLOWED at leaf "
                + String(t)
                + " of "
                + String(len(partials))
            )
    return _fold_drain(stack, occ)


def check_stack_fold_is_the_contract_tree() raises:
    """**THE LOAD-BEARING PRELIMINARY.** The kernel's register-stack fold is
    bit for bit `gemm_oracle.fold_balanced_tree`, at every `P` from 1 to
    2049.

    The kernel does NOT build the tree level by level in the TILE_* and FLAT
    plans; it merges a leaf partial into an occupancy-indexed register stack
    the moment the partial is finished. That is a different SPELLING of the
    contract's tree, and a different spelling is a second thing that can be
    wrong -- so it is proven against the oracle's own function rather than
    argued from `_fold_push`'s docstring.

    THE RANGE MATTERS AND IS NOT DECORATION. `P = 5` is the smallest `P` that
    carries TWICE and `P = 7` looks like it should and carries once
    (contract 7.2.2); the sweep covers every `P` up to twice the profile cap,
    so every carry pattern, every level width and every set-bit combination
    of `P` is exercised, including the ones no real shape produces
    (`bench/gemm_shapes.mojo`: not one of its twenty rows has an odd `P`).

    The partials are the same 13-bit-significand generator the device
    fixtures use, so the additions are inexact and a wrong pairing shows.
    """
    var bad = 0
    var first_bad = -1
    var got0 = Float32(0.0)
    var want0 = Float32(0.0)
    for p in range(1, 2050):
        var v = List[Float32]()
        for t in range(p):
            v.append(_val(t * 31 + p, 909))
        var want = fold_balanced_tree(v)
        var got = _stack_fold(v)
        if _bits(got) != _bits(want):
            bad += 1
            if first_bad < 0:
                first_bad = p
                got0 = got
                want0 = want
    if bad != 0:
        raise Error(
            "check_stack_fold_is_the_contract_tree: the kernel's register"
            " fold DISAGREES with gemm_oracle.fold_balanced_tree at "
            + String(bad)
            + " of 2049 leaf counts. First at P = "
            + String(first_bad)
            + ": stack = "
            + _show(got0)
            + "   tree = "
            + _show(want0)
            + ". The kernel is not computing the contract's fold."
        )
    print(
        "check_stack_fold_is_the_contract_tree OK: register stack == the"
        " contract's balanced tree, bit for bit, at every P in 1..2049"
    )


def check_strides_match_the_contract_addressing() raises:
    """`gemm_operand_strides` reproduces contract section 3's four lines.

    The kernel addresses its operands as `a[i*a_si + p*a_sp]` and
    `b[p*b_sp + j*b_sj]` so that the inner loop is character for character
    the same in all three orientations (section 3's requirement that the
    three ARE one numerical implementation). That lifts an `if` out of the k
    loop, and lifting an `if` is exactly where an addressing bug hides -- so
    the stride pair is asserted element by element against the contract's own
    spelling at a shape where `m`, `n` and `k` are all DIFFERENT, because
    equal dimensions make a transposed index look correct.
    """
    var m = 5
    var n = 7
    var k = 11
    for oi in range(3):
        var op = OP_NN
        if oi == 1:
            op = OP_NT
        elif oi == 2:
            op = OP_TN
        var st = gemm_operand_strides(op, m, n, k)
        for i in range(m):
            for p in range(k):
                # Contract section 3: A_eff[i,p] = A[i*k + p], or A[p*m + i]
                # for OP_TN.
                var want = i * k + p
                if op == OP_TN:
                    want = p * m + i
                var got = i * st[0] + p * st[1]
                if got != want:
                    raise Error(
                        "check_strides_match_the_contract_addressing: op "
                        + op_name(op)
                        + " A_eff["
                        + String(i)
                        + ", "
                        + String(p)
                        + "] addressed "
                        + String(got)
                        + ", contract says "
                        + String(want)
                    )
        for p in range(k):
            for j in range(n):
                # Contract section 3: B_eff[p,j] = B[p*n + j], or B[j*k + p]
                # for OP_NT.
                var want2 = p * n + j
                if op == OP_NT:
                    want2 = j * k + p
                var got2 = p * st[2] + j * st[3]
                if got2 != want2:
                    raise Error(
                        "check_strides_match_the_contract_addressing: op "
                        + op_name(op)
                        + " B_eff["
                        + String(p)
                        + ", "
                        + String(j)
                        + "] addressed "
                        + String(got2)
                        + ", contract says "
                        + String(want2)
                    )
    print(
        "check_strides_match_the_contract_addressing OK: NN, NT and TN"
        " address every A_eff and B_eff entry the way contract section 3"
        " spells it (m=5, n=7, k=11, all different)"
    )


# ===========================================================================
# THE DEVICE HARNESS
# ===========================================================================


def _run_device(
    ctx: DeviceContext,
    ha: List[Float32],
    hb: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    plan: Int,
    tag: String,
) raises -> List[Float32]:
    """One launch on one named plan, poisoned first, returned as host floats.

    A surviving poison RAISES here rather than being compared, because a
    never-written cell that happens to compare equal to something is not
    evidence of anything.
    """
    var na = len(ha)
    var nb = len(hb)
    var mn = m * n
    # A zero-length operand is legal (`k == 0`, contract section 8) and a
    # zero-length device buffer is not, so the ALLOCATION is clamped while
    # the copy loops still run over the real element count.
    var na_buf = na
    if na_buf < 1:
        na_buf = 1
    var nb_buf = nb
    if nb_buf < 1:
        nb_buf = 1
    var da = ctx.enqueue_create_buffer[DType.float32](na_buf)
    var db = ctx.enqueue_create_buffer[DType.float32](nb_buf)
    var dc = ctx.enqueue_create_buffer[DType.float32](mn)
    # THE WORKSPACE MUST COVER THE PLAN THAT WILL ACTUALLY RUN. Sizing it
    # for `plan` when `plan < 0` means the dispatcher chooses is an
    # out-of-bounds write, and it is the defect this harness shipped for one
    # run: `identical_gemm_workspace_floats(m, n, k, -1)` returns 0, so a
    # SPLITK dispatch at 64x64x4096 wrote 512 KB of partials into a one-float
    # buffer and whole regions of C came back +0.0.
    # `check_device_is_batch_invariant` caught it and nothing else did.
    var nws = identical_gemm_workspace_floats(m, n, k, plan)
    if plan < 0:
        nws = identical_gemm_workspace_max_floats(m, n, k)
    if nws < 1:
        nws = 1
    var dw = ctx.enqueue_create_buffer[DType.float32](nws)
    var hA = ctx.enqueue_create_host_buffer[DType.float32](na_buf)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](nb_buf)
    var hC = ctx.enqueue_create_host_buffer[DType.float32](mn)
    ctx.synchronize()
    for i in range(na):
        hA.unsafe_ptr().unsafe_store(i, ha[i])
    for i in range(nb):
        hB.unsafe_ptr().unsafe_store(i, hb[i])
    for i in range(mn):
        hC.unsafe_ptr().unsafe_store(i, POISON)
    ctx.enqueue_copy(dst_buf=da, src_ptr=hA.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=db, src_ptr=hB.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dc, src_ptr=hC.unsafe_ptr())
    ctx.synchronize()
    if plan < 0:
        identical_gemm_into(ctx, dc, da, db, dw, m, n, k, op)
    else:
        identical_gemm_with_plan(ctx, dc, da, db, dw, m, n, k, op, plan)
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hC.unsafe_ptr(), src_buf=dc)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(mn):
        var v = hC.unsafe_ptr().unsafe_load(i)
        if _bits(v) == _bits(POISON):
            raise Error(
                "POISON SURVIVED at cell ("
                + String(i // n)
                + ", "
                + String(i % n)
                + ") of "
                + tag
                + ": the kernel never wrote it, so nothing downstream is a"
                " comparison of products. Contract section 8 requires the"
                " value to be STORED, including the +0.0 at k == 0."
            )
        out.append(v)
    _ = da
    _ = db
    _ = dc
    _ = dw
    _ = hA
    _ = hB
    _ = hC
    return out^


# ===========================================================================
# GATE 1: THE DEVICE COMPUTES `gemm_oracle`
# ===========================================================================


def _tbl_op(i: Int) -> Int:
    """`bench/gemm_shapes.mojo` numbers the orientations `NT=0, TN=1, NN=2`
    and `gemm_oracle.mojo` numbers them `NN=0, NT=1, TN=2`. **Two files, two
    encodings of the same three words**, so the mapping is written once, here,
    rather than assumed at each use -- an off-by-one in this map would run
    every table row against the wrong addressing and still produce a green
    oracle comparison, because both sides would be wrong together."""
    var o = gemm_shape_op(i)
    if o == TBL_OP_NT:
        return OP_NT
    if o == TBL_OP_TN:
        return OP_TN
    return OP_NN


def _oracle_case(
    ctx: DeviceContext,
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    label: String,
    subnormal: Bool,
    mut cases: Int,
    mut fails: Int,
) raises:
    var ha = _fill(_a_elems(op, m, n, k), 11, subnormal)
    var hb = _fill(_b_elems(op, m, n, k), 22, subnormal)
    _oracle_case_vals(ctx, ha, hb, op, m, n, k, label, cases, fails)


def _oracle_case_vals(
    ctx: DeviceContext,
    ha: List[Float32],
    hb: List[Float32],
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    label: String,
    mut cases: Int,
    mut fails: Int,
) raises:
    var want = gemm_oracle(ha, hb, op, m, n, k)
    var plan = -1  # the shipped dispatcher
    var got = _run_device(ctx, ha, hb, op, m, n, k, plan, label)
    var bad = 0
    var first = -1
    for c in range(m * n):
        if _bits(got[c]) != _bits(want[c]):
            bad += 1
            if first < 0:
                first = c
    var part = contract_partition(k)
    cases += 1
    if bad == 0:
        print(
            "    OK   "
            + label
            + "  "
            + op_name(op)
            + " "
            + String(m)
            + "x"
            + String(n)
            + "x"
            + String(k)
            + "  L="
            + String(part[0])
            + " P="
            + String(part[1])
            + "  plan="
            + gemm_plan_name(_dispatch_plan(m, n, k))
        )
        return
    fails += 1
    print(
        "    FAIL "
        + label
        + "  "
        + op_name(op)
        + " "
        + String(m)
        + "x"
        + String(n)
        + "x"
        + String(k)
        + "  L="
        + String(part[0])
        + " P="
        + String(part[1])
        + ": "
        + String(bad)
        + " of "
        + String(m * n)
        + " cells differ from gemm_oracle. First at ("
        + String(first // n)
        + ", "
        + String(first % n)
        + "): device = "
        + _show(got[first])
        + "   oracle = "
        + _show(want[first])
    )


def _dispatch_plan(m: Int, n: Int, k: Int) -> Int:
    return choose_gemm_plan(m, n, k)


def _minus_zero_leaves(k: Int, el: Int) -> List[Float32]:
    """One row of `A` such that EVERY leaf's partial is exactly `-0.0`, when
    `B` is all ones. Lifted from `gemm_oracle_check.mojo::_minus_zero_leaves`
    so the device fixture and the host fixture are the same construction.

    Contract 9.2(a): a `+0.0`-seeded leaf can NEVER reach `-0.0` from
    products, so the only route is `ftz` of a negative subnormal accumulator.
    Each leaf therefore ends with `2^-126` then `-1.5 * 2^-126`, giving
    `-2^-127`, which the flush turns into `-0.0`.

    THE PLACEMENT AT THE END OF THE LEAF IS LOAD-BEARING. Put the pair at the
    START and the trailing `0 * 1` products erase the sign --
    `fma(0, 1, -0.0)` is `+0.0 + (-0.0)` is `+0.0` -- and the fixture becomes
    all `+0.0` partials, which separates nothing.
    """
    var a = List[Float32]()
    for _ in range(k):
        a.append(Float32(0.0))
    var p = contract_leaf_count(k)
    for j in range(p):
        var e = (j + 1) * el
        if e > k:
            e = k
        a[e - 2] = bitcast[DType.float32](UInt32(0x00800000))
        a[e - 1] = -bitcast[DType.float32](UInt32(0x00C00000))
    return a^


def _minus_zero_case(
    ctx: DeviceContext,
    m: Int,
    n: Int,
    k: Int,
    label: String,
    mut cases: Int,
    mut fails: Int,
) raises:
    """`OP_NT` with every leaf partial `-0.0` and `B` all ones.

    **THIS IS THE ONLY FIXTURE HERE THAT CAN TELL A CARRY FROM `+0.0`
    PADDING**, and it exists because the first version of this gate could
    not. `x + (+0.0) == x` for every finite float, every infinity and every
    NaN -- it differs on exactly ONE value, `x = -0.0`, where
    `(-0) + (+0) = +0`. So the `-D MOJOLEARN_GEMM_SABOTAGE_PAD_PLUS_ZERO=1`
    build passed all 42 shapes of the first sweep, EXIT CODE 0, because not
    one of them ever produced a negative zero. That is IDENTITY_PATHS row 9's
    failure with a different subject and it is why this function is here.

    The fixture REFUSES ITSELF if the oracle does not come back `-0.0`: a
    vacuous fixture that reports green is worse than no fixture.
    """
    var el = contract_leaf_size(k)
    var p_count = contract_leaf_count(k)
    var last = k - (p_count - 1) * el
    if p_count < 1 or last < 2:
        raise Error(
            "_minus_zero_case ["
            + label
            + "]: this construction needs EVERY leaf to hold at least two"
            " elements (it writes 2^-126 then -1.5*2^-126 at the leaf's"
            " end), and the last leaf of k = "
            + String(k)
            + " holds "
            + String(last)
            + ". Pick a k whose ragged tail is >= 2."
        )
    var row = _minus_zero_leaves(k, el)
    var ha = List[Float32]()
    for _ in range(m):
        for p in range(k):
            ha.append(row[p])
    var hb = List[Float32]()
    for _ in range(n * k):
        hb.append(Float32(1.0))
    var want = gemm_oracle(ha, hb, OP_NT, m, n, k)
    comptime if IDENTICAL_BUILD:
        for c in range(m * n):
            if _bits(want[c]) != UInt32(0x80000000):
                raise Error(
                    "_minus_zero_case ["
                    + label
                    + "]: the oracle returned "
                    + _show(want[c])
                    + " at cell "
                    + String(c)
                    + ", the construction predicts -0.0 (0x80000000). The"
                    " fixture's partials are not -0.0, so it separates a"
                    " carry from +0.0 padding on nothing and reporting it"
                    " green would be reporting the opposite of the truth."
                )
    _oracle_case_vals(ctx, ha, hb, OP_NT, m, n, k, label, cases, fails)


def _oracle_all_plans_case(
    ctx: DeviceContext,
    op: Int,
    m: Int,
    n: Int,
    k: Int,
    label: String,
    mut cases: Int,
    mut fails: Int,
) raises:
    """The oracle comparison on EVERY named plan, not only the one the
    dispatcher picks.

    Everywhere else this gate runs `plan = -1`, the shipped dispatcher, which
    is the right default because it is what a caller gets. The consequence is
    that a plan the dispatcher never chooses -- `PLAN_SPLITK_STAGED` is never
    chosen, it exists for contract 13.6.2 and as the most distant invariance
    witness -- would only be checked TRANSITIVELY, through
    `check_device_is_launch_invariant` agreeing with `PLAN_FLAT` at a
    different shape set. Transitive is not the same as checked. This arm
    closes it directly.
    """
    var ha = _fill(_a_elems(op, m, n, k), 11, False)
    var hb = _fill(_b_elems(op, m, n, k), 22, False)
    for plan in range(GEMM_PLAN_COUNT):
        var lbl = label + " [" + gemm_plan_name(plan) + "]"
        var want = gemm_oracle(ha, hb, op, m, n, k)
        var got = _run_device(ctx, ha, hb, op, m, n, k, plan, lbl)
        var bad = 0
        var first = -1
        for c in range(m * n):
            if _bits(got[c]) != _bits(want[c]):
                bad += 1
                if first < 0:
                    first = c
        cases += 1
        if bad == 0:
            print("    OK   " + lbl)
        else:
            fails += 1
            print(
                "    FAIL "
                + lbl
                + ": "
                + String(bad)
                + " of "
                + String(m * n)
                + " cells differ from gemm_oracle. First at ("
                + String(first // n)
                + ", "
                + String(first % n)
                + "): device = "
                + _show(got[first])
                + "   oracle = "
                + _show(want[first])
            )


def check_device_matches_oracle() raises:
    """**GATE 1.** Every cell's bits equal `gemm_oracle`'s, over the shape
    table plus the synthetic partition cases the table cannot reach.

    THE TABLE ROWS ARE RUN AT REDUCED `m` AND `n`, AND THAT IS SOUND RATHER
    THAN A SHORTCUT. The oracle is `O(m n k)` scalar host Mojo; a table row
    at `512 x 128256 x 4096` is 2.7e11 host operations. What the numerical
    plan sees is `k` and the profile and NOTHING ELSE (contract 6.1 and 0.3),
    so clamping `m` and `n` preserves every leaf boundary, every tree level
    and every fold pairing of the row while making the oracle runnable. The
    `m` and `n` axes are covered instead by GATE 3, which is the check that
    is actually about them.

    THE SYNTHETIC CASES ARE WHERE THE CONTRACT LIVES, and
    `bench/gemm_shapes.mojo`'s header is why: **not one of its twenty real
    rows produces an odd leaf count**, so a gate built only from real shapes
    would exercise the even path and report the carry clause as covered. The
    list below therefore forces, by construction:

        k = 0          P = 0, and every cell must be a STORED +0.0
        k = 1, 64, 128 P = 1, the fold that performs no addition (7.3)
        k = 129, 256   P = 2, ragged and exact
        k = 300        P = 3, the ragged odd case contract 12.1 names
        k = 517        P = 5, the smallest P that carries TWICE
        k = 800        P = 7, looks like it carries twice and carries once
        k = 4097       P = 33, odd with a ONE-ELEMENT last leaf
        k = 130900     P = 1023, odd, ragged, ten levels of carries
        k = 131073     P = 1017, odd and ragged PAST the leaf rule's
                       crossover, where L stops being 128 and becomes 129
        k = 4000000    L = 3907, P = 1024, the profile cap

    plus ragged `m` and `n` that no tile shape divides (5x7, 17x33, 33x17),
    and one SUBNORMAL fixture whose every product lands below `2^-126` --
    the only case here that reaches contract seam 5c on a device.
    """
    print(
        "check_device_matches_oracle ["
        + _mode_name()
        + "]: per-cell bits against gemm_oracle"
    )
    var cases = 0
    var fails = 0
    with DeviceContext() as ctx:
        print("  -- the shape table, at reduced m and n (see docstring) --")
        for i in range(GEMM_SHAPE_COUNT):
            var op = _tbl_op(i)
            var m = gemm_shape_m(i)
            var n = gemm_shape_n(i)
            if m > 4:
                m = 4
            if n > 4:
                n = 4
            _oracle_case(
                ctx,
                op,
                m,
                n,
                gemm_shape_k(i),
                gemm_shape_name(i),
                False,
                cases,
                fails,
            )
        print("  -- one real row at its SHIPPED m and n --")
        _oracle_case(
            ctx, OP_NT, 4096, 64, 64, String("kmeans.dist.full"), False, cases, fails
        )
        print("  -- synthetic: the partition cases the table cannot reach --")
        _oracle_case(ctx, OP_NT, 3, 3, 0, String("k0.P0"), False, cases, fails)
        _oracle_case(ctx, OP_NT, 1, 1, 1, String("k1.P1"), False, cases, fails)
        _oracle_case(ctx, OP_NN, 4, 4, 64, String("k64.P1"), False, cases, fails)
        _oracle_case(ctx, OP_TN, 4, 4, 128, String("k128.P1"), False, cases, fails)
        _oracle_case(ctx, OP_NT, 4, 4, 129, String("k129.P2.ragged"), False, cases, fails)
        _oracle_case(ctx, OP_NT, 4, 4, 256, String("k256.P2.exact"), False, cases, fails)
        _oracle_case(ctx, OP_NN, 4, 4, 300, String("k300.P3.odd.ragged"), False, cases, fails)
        _oracle_case(ctx, OP_TN, 4, 4, 517, String("k517.P5.odd.2carries"), False, cases, fails)
        _oracle_case(ctx, OP_NT, 4, 4, 800, String("k800.P7.odd"), False, cases, fails)
        _oracle_case(ctx, OP_NT, 4, 4, 1024, String("k1024.P8.even"), False, cases, fails)
        _oracle_case(ctx, OP_NT, 3, 3, 4097, String("k4097.P33.odd.tail1"), False, cases, fails)
        _oracle_case(ctx, OP_NT, 2, 2, 130900, String("k130900.P1023.odd"), False, cases, fails)
        _oracle_case(ctx, OP_NT, 2, 2, 131073, String("k131073.P1017.past.crossover"), False, cases, fails)
        _oracle_case(ctx, OP_NT, 1, 1, 4000000, String("k4M.P1024.cap"), False, cases, fails)
        print("  -- synthetic: ragged m and n that no tile divides --")
        _oracle_case(ctx, OP_NT, 5, 7, 300, String("5x7.P3"), False, cases, fails)
        _oracle_case(ctx, OP_NT, 17, 33, 300, String("17x33.P3"), False, cases, fails)
        _oracle_case(ctx, OP_NN, 33, 17, 517, String("33x17.P5"), False, cases, fails)
        _oracle_case(ctx, OP_TN, 17, 33, 4097, String("17x33.P33"), False, cases, fails)
        _oracle_case(ctx, OP_NT, 64, 64, 64, String("64x64.P1.manyblocks"), False, cases, fails)
        print(
            "  -- EVERY named plan against the oracle, not only the"
            " dispatcher's --"
        )
        _oracle_all_plans_case(
            ctx, OP_NT, 17, 33, 517, String("allplans.17x33x517.P5.odd"), cases, fails
        )
        _oracle_all_plans_case(
            ctx, OP_TN, 5, 7, 4200, String("allplans.5x7x4200.P33.odd"), cases, fails
        )
        print(
            "  -- synthetic: EVERY LEAF PARTIAL IS -0.0 (contract 9.2, F7)"
            " -- the ONLY input that separates a CARRY from +0.0 PADDING --"
        )
        _minus_zero_case(ctx, 5, 7, 300, String("minuszero.k300.P3.carry"), cases, fails)
        _minus_zero_case(ctx, 5, 7, 517, String("minuszero.k517.P5.2carries"), cases, fails)
        _minus_zero_case(ctx, 5, 7, 128, String("minuszero.k128.P1.nofold"), cases, fails)
        # k = 4200, not 4097: `L = 128` and `P = 33` in both, but 4097's
        # last leaf holds ONE element and this construction needs TWO. The
        # k = 4097 arm was written first and the fixture REFUSED ITSELF --
        # `a[e-2]` had landed in the previous leaf and the one-element last
        # leaf came back a normal `-1.5*2^-126`, not `-0.0`. Kept as a
        # comment because the refusal is the interesting part: the guard in
        # `_minus_zero_case` is what turned a silently vacuous fixture into
        # an error.
        _minus_zero_case(ctx, 3, 3, 4200, String("minuszero.k4200.P33.odd"), cases, fails)
        print("  -- synthetic: SUBNORMAL products, contract seam 5c --")
        _oracle_case(ctx, OP_NT, 4, 4, 300, String("subnormal.k300"), True, cases, fails)
        _oracle_case(ctx, OP_NT, 4, 4, 4096, String("subnormal.k4096"), True, cases, fails)

    if fails == 0:
        print(
            "check_device_matches_oracle ["
            + _mode_name()
            + "] OK: "
            + String(cases)
            + " shapes, every cell bit-identical to gemm_oracle"
        )
        return
    comptime if IDENTICAL_BUILD:
        raise Error(
            "check_device_matches_oracle [IDENTICAL]: "
            + String(fails)
            + " of "
            + String(cases)
            + " shapes DISAGREE with gemm_oracle (per-shape lines above)."
            " The device is not computing mojolearn.identical.gemm.fp32.v1."
        )
    print(
        "check_device_matches_oracle [FAST] REPORT: "
        + String(fails)
        + " of "
        + String(cases)
        + " shapes differ from the host oracle. Under FAST both sides are"
        " the unpinned spelling and the host CPU and the device backend are"
        " free to contract and to flush differently, so this is a"
        " measurement and not an assertion. Under IDENTICAL it is an"
        " assertion and any difference raises."
    )


# ===========================================================================
# GATE 2: LAUNCH INVARIANCE
# ===========================================================================


def check_device_is_launch_invariant() raises:
    """**GATE 2, AND THE PROPERTY THE WHOLE PROFILE EXISTS FOR.** The SAME
    logical problem, computed under seven DIFFERENT execution plans, gives
    bit-identical output.

    The seven are not seven tunings of one kernel. They differ in:

        block size          16, 64 and 256 threads
        grid rank           1-D over cells, 1-D over tiles, 2-D over tiles
        tile shape          none, 16x16, 8x32, 32x8, 4x4
        staging             none, and threadgroup windows of KS = 8, 16, 32
        tile order          natural, reversed, transposed
        where the fold runs REGISTERS (one thread owns the whole tree) or a
                            SECOND KERNEL over a global workspace, level by
                            level in threadgroup memory

    The last row is the one that matters most: `PLAN_SPLITK` materializes
    `m*n*P` named partials and folds them level-wise with an explicit
    `(d, q)` walk, while `PLAN_FLAT` never writes a partial anywhere and
    merges into an occupancy-indexed register stack. **Two structurally
    unrelated realizations of one arithmetic DAG**, required to agree bit for
    bit. If the numerical plan had leaked into the execution plan anywhere,
    these two are where it would show.

    `KS = 8` against `KS = 32` is the specific proof that a STAGING WINDOW IS
    NOT A LEAF: the same leaf is walked in four times as many threadgroup
    windows and must accumulate into the same register in the same order.

    The tile-order permutations are the charter's *"scheduling of logical
    work"*, and the reversed and transposed orders mean a given output tile
    is computed by a different block, at a different point in the launch, in
    each of them.

    ASSERTED IN BOTH MODES. Nothing here depends on `fma` or on `ftz`: it is
    a claim about the SHAPE of the kernel, and FAST has the same shape.
    """
    print(
        "check_device_is_launch_invariant ["
        + _mode_name()
        + "]: "
        + String(GEMM_PLAN_COUNT)
        + " execution plans, one numerical plan"
    )
    var fails = 0
    var shapes_run = 0
    with DeviceContext() as ctx:
        for si in range(6):
            var op = OP_NT
            var m = 64
            var n = 64
            var k = 64
            var label = String("64x64x64   P=1  (the fold has no node)")
            if si == 1:
                op = OP_NN
                m = 17
                n = 33
                k = 300
                label = String("17x33x300  P=3  odd, ragged, no tile divides")
            elif si == 2:
                op = OP_TN
                m = 33
                n = 17
                k = 517
                label = String("33x17x517  P=5  two carries")
            elif si == 3:
                op = OP_NT
                m = 130
                n = 70
                k = 4096
                label = String("130x70x4096 P=32 even")
            elif si == 4:
                op = OP_NT
                m = 12
                n = 9
                k = 130900
                label = String("12x9x130900 P=1023 odd, ten levels")
            elif si == 5:
                op = OP_NT
                m = 5
                n = 7
                k = 0
                label = String("5x7x0      P=0  every cell a stored +0.0")
            # `si == 0` keeps the initializers above. It was an `else` for one
            # revision, which swallowed si == 0 into the k == 0 case and
            # printed the 64x64x64 label over it -- so the P == 1 shape never
            # ran and the transcript said it had. A label is not evidence.

            var ha = _fill(_a_elems(op, m, n, k), 31, False)
            var hb = _fill(_b_elems(op, m, n, k), 47, False)
            var reference = _run_device(
                ctx, ha, hb, op, m, n, k, PLAN_FLAT, label
            )
            shapes_run += 1
            print("  " + label + "  op=" + op_name(op))
            for plan in range(GEMM_PLAN_COUNT):
                if plan == PLAN_FLAT:
                    continue
                var got = _run_device(ctx, ha, hb, op, m, n, k, plan, label)
                var bad = 0
                var first = -1
                for c in range(m * n):
                    if _bits(got[c]) != _bits(reference[c]):
                        bad += 1
                        if first < 0:
                            first = c
                if bad == 0:
                    print("      OK   " + gemm_plan_name(plan))
                else:
                    fails += 1
                    print(
                        "      FAIL "
                        + gemm_plan_name(plan)
                        + ": "
                        + String(bad)
                        + " of "
                        + String(m * n)
                        + " cells differ from "
                        + gemm_plan_name(PLAN_FLAT)
                        + ". First at ("
                        + String(first // n)
                        + ", "
                        + String(first % n)
                        + "): this plan = "
                        + _show(got[first])
                        + "   FLAT = "
                        + _show(reference[first])
                    )
    if fails != 0:
        raise Error(
            "check_device_is_launch_invariant ["
            + _mode_name()
            + "]: "
            + String(fails)
            + " plan/shape combinations produced DIFFERENT BITS from the"
            " same logical problem (lines above). Launch geometry has"
            " reached the arithmetic, which is the one thing profile"
            " mojolearn.identical.gemm.fp32.v1 exists to prevent."
        )
    print(
        "check_device_is_launch_invariant ["
        + _mode_name()
        + "] OK: "
        + String(shapes_run)
        + " shapes x "
        + String(GEMM_PLAN_COUNT)
        + " execution plans, all bit-identical"
    )


# ===========================================================================
# GATE 3: BATCH INVARIANCE
# ===========================================================================


def _batch_a(i: Int, p: Int, k: Int) -> Float32:
    """`A`'s row `i` is a pure function of `(i, p)` -- NOT of `m`. That is
    what makes two launches of different sizes comparable at all."""
    return _val(i * 7919 + p, 5150)


def _batch_b(j: Int, p: Int, k: Int) -> Float32:
    return _val(j * 104729 + p, 5151)


def _batch_run(
    ctx: DeviceContext, m: Int, n: Int, k: Int
) raises -> List[Float32]:
    """`C[m x n] = A[m x k] . B[n x k]^T` through the SHIPPED dispatcher.

    Through the dispatcher on purpose: `choose_gemm_plan` reads `m` and `n`
    and is ALLOWED to (contract 6.1), so at `n = 64` this shape takes the
    SPLITK arm and at `n = 256` it takes a 16x16 tiled arm. If the profile
    holds, the cell's bits do not notice.
    """
    var ha = List[Float32]()
    for i in range(m):
        for p in range(k):
            ha.append(_batch_a(i, p, k))
    var hb = List[Float32]()
    for j in range(n):
        for p in range(k):
            hb.append(_batch_b(j, p, k))
    return _run_device(
        ctx,
        ha,
        hb,
        OP_NT,
        m,
        n,
        k,
        -1,
        String("batch ") + String(m) + "x" + String(n) + "x" + String(k),
    )


def _batch_arm(
    ctx: DeviceContext,
    k: Int,
    m_ref: Int,
    n_ref: Int,
    m_alt: Int,
    n_alt: Int,
    m_ov: Int,
    n_ov: Int,
    tag: String,
    mut fails: Int,
) raises:
    var a = _batch_run(ctx, m_ref, n_ref, k)
    var b = _batch_run(ctx, m_alt, n_alt, k)
    var bad = 0
    var first_i = -1
    var first_j = -1
    for i in range(m_ov):
        for j in range(n_ov):
            var va = a[i * n_ref + j]
            var vb = b[i * n_alt + j]
            if _bits(va) != _bits(vb):
                bad += 1
                if first_i < 0:
                    first_i = i
                    first_j = j
    if bad == 0:
        print(
            "      OK   "
            + tag
            + ": "
            + String(m_ov * n_ov)
            + " overlapping cells identical  ["
            + String(m_ref)
            + "x"
            + String(n_ref)
            + " -> "
            + gemm_plan_name(_dispatch_plan(m_ref, n_ref, k))
            + "  |  "
            + String(m_alt)
            + "x"
            + String(n_alt)
            + " -> "
            + gemm_plan_name(_dispatch_plan(m_alt, n_alt, k))
            + "]"
        )
        return
    fails += 1
    var va2 = a[first_i * n_ref + first_j]
    var vb2 = b[first_i * n_alt + first_j]
    print(
        "      FAIL "
        + tag
        + ": "
        + String(bad)
        + " of "
        + String(m_ov * n_ov)
        + " overlapping cells CHANGED BITS with the launch width. First at"
        " cell ("
        + String(first_i)
        + ", "
        + String(first_j)
        + "): "
        + String(m_ref)
        + "x"
        + String(n_ref)
        + " gave "
        + _show(va2)
        + "   "
        + String(m_alt)
        + "x"
        + String(n_alt)
        + " gave "
        + _show(vb2)
    )


def _batch_loop_arm(
    ctx: DeviceContext, k: Int, m_ov: Int, n_ov: Int, mut fails: Int
) raises:
    """THE SHARED-WORKSPACE COMPOSITION. Several INDEPENDENT GEMMs of
    different shapes, enqueued back to back with NO synchronize between them
    through ONE workspace buffer (`identical_gemm_into`, the asynchronous
    form a batched caller would use), against the same cells computed in a
    launch of their own.

    What this sees that `_batch_arm` cannot: a plan that reads a partial it
    did not write -- a workspace slot left over from the PREVIOUS product in
    the queue -- or any state that leaks from one launch into the next. Each
    GEMM in the loop takes whatever plan `choose_gemm_plan` gives its shape
    (SPLITK for `64 x 4` and `64 x 64`, a tile for `130 x 64`), so the
    workspace is sized for the largest and is DIRTY for every launch after
    the first. The reference is `_batch_run`'s solo `64 x 4`, which had a
    clean workspace of its own.
    """
    var shapes_m = [64, 64, 130, 64]
    var shapes_n = [4, 64, 64, 4]
    var count = len(shapes_m)
    var solo = _batch_run(ctx, m_ov, n_ov, k)

    # One workspace for the whole loop, sized for the largest plan in it.
    var nws = 1
    var ma = 0
    var mb = 0
    var mc = 0
    for q in range(count):
        var w = identical_gemm_workspace_max_floats(shapes_m[q], shapes_n[q], k)
        if w > nws:
            nws = w
        if shapes_m[q] * k > ma:
            ma = shapes_m[q] * k
        if shapes_n[q] * k > mb:
            mb = shapes_n[q] * k
        if shapes_m[q] * shapes_n[q] > mc:
            mc = shapes_m[q] * shapes_n[q]
    var dw = ctx.enqueue_create_buffer[DType.float32](nws)
    # Operands are built ONCE at the largest extent and every GEMM reads a
    # prefix of them: `_batch_a` is a function of `(i, p)` and `_batch_b` of
    # `(j, p)`, so row `i` of the 130-row `A` IS row `i` of the 64-row `A`,
    # and the loop's inputs are the reference's inputs by construction.
    var hA = ctx.enqueue_create_host_buffer[DType.float32](ma)
    var hB = ctx.enqueue_create_host_buffer[DType.float32](mb)
    ctx.synchronize()
    var big_m = ma // k
    var big_n = mb // k
    for i in range(big_m):
        for pp in range(k):
            hA.unsafe_ptr().unsafe_store(i * k + pp, _batch_a(i, pp, k))
    for j in range(big_n):
        for pp in range(k):
            hB.unsafe_ptr().unsafe_store(j * k + pp, _batch_b(j, pp, k))
    var das = List[DeviceBuffer[DType.float32]]()
    var dbs = List[DeviceBuffer[DType.float32]]()
    var dcs = List[DeviceBuffer[DType.float32]]()
    var hcs = List[HostBuffer[DType.float32]]()
    for q in range(count):
        var mq = shapes_m[q]
        var nq = shapes_n[q]
        # `_batch_run` lays `A` out as `m x k` row-major with row stride k,
        # so a prefix of rows of the big `A` is exactly the small `A`. `B` is
        # `n x k` the same way.
        var da = ctx.enqueue_create_buffer[DType.float32](mq * k)
        var db = ctx.enqueue_create_buffer[DType.float32](nq * k)
        var dc = ctx.enqueue_create_buffer[DType.float32](mq * nq)
        var hc = ctx.enqueue_create_host_buffer[DType.float32](mq * nq)
        ctx.synchronize()
        for i in range(mq * nq):
            hc.unsafe_ptr().unsafe_store(i, POISON)
        ctx.enqueue_copy(dst_buf=da, src_ptr=hA.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=db, src_ptr=hB.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dc, src_ptr=hc.unsafe_ptr())
        das.append(da^)
        dbs.append(db^)
        dcs.append(dc^)
        hcs.append(hc^)
    ctx.synchronize()
    # THE LOOP: no synchronize between launches, one workspace throughout.
    for q in range(count):
        identical_gemm_into(
            ctx, dcs[q], das[q], dbs[q], dw, shapes_m[q], shapes_n[q], k, OP_NT
        )
    ctx.synchronize()
    for q in range(count):
        ctx.enqueue_copy(dst_ptr=hcs[q].unsafe_ptr(), src_buf=dcs[q])
    ctx.synchronize()

    var total_bad = 0
    for q in range(count):
        var mq = shapes_m[q]
        var nq = shapes_n[q]
        var bad = 0
        var first_i = -1
        var first_j = -1
        var va2 = Float32(0.0)
        var vb2 = Float32(0.0)
        for i in range(m_ov):
            for j in range(n_ov):
                var vb = hcs[q].unsafe_ptr().unsafe_load(i * nq + j)
                if _bits(vb) == _bits(POISON):
                    raise Error(
                        "POISON SURVIVED in the shared-workspace loop, GEMM "
                        + String(q)
                        + " ("
                        + String(mq)
                        + "x"
                        + String(nq)
                        + "), cell ("
                        + String(i)
                        + ", "
                        + String(j)
                        + ")"
                    )
                var va = solo[i * n_ov + j]
                if _bits(va) != _bits(vb):
                    bad += 1
                    if first_i < 0:
                        first_i = i
                        first_j = j
                        va2 = va
                        vb2 = vb
        var tag = (
            String("loop[")
            + String(q)
            + "] "
            + String(mq)
            + "x"
            + String(nq)
            + " -> "
            + gemm_plan_name(_dispatch_plan(mq, nq, k))
            + ", shared ws of "
            + String(nws)
            + " floats"
        )
        if bad == 0:
            print(
                "      OK   "
                + tag
                + ": "
                + String(m_ov * n_ov)
                + " overlapping cells identical to the solo launch"
            )
        else:
            total_bad += 1
            print(
                "      FAIL "
                + tag
                + ": "
                + String(bad)
                + " of "
                + String(m_ov * n_ov)
                + " overlapping cells CHANGED BITS against the solo launch."
                " First at cell ("
                + String(first_i)
                + ", "
                + String(first_j)
                + "): solo "
                + _show(va2)
                + "   in-loop "
                + _show(vb2)
            )
    if total_bad != 0:
        fails += 1
    _ = dw
    _ = hA
    _ = hB
    _ = das
    _ = dbs
    _ = dcs
    _ = hcs


def check_device_is_batch_invariant() raises:
    """**GATE 3.** A cell's bits do not depend on how many OTHER cells were
    in the launch.

    THE FIXTURE IS BUILT FROM `core/gemm_identity_check.mojo::
    check_pinned_gemm_is_batch_invariant`'s CORRECTION, and the correction is
    the reason this docstring is long. That check's first version compared
    cell `(0, 0)` at three launch sizes and PASSED a deliberate
    grid-dependent sabotage, because `(0, 0)` is linear index 0 and therefore
    sits in block 0 at every launch size: the sabotage could not reach the
    only cell being read. A check a real order-dependence walks straight
    through does not report a weak result, it reports the opposite of the
    truth.

    So, both of that correction's lessons:

    1. **VARY `n`, NOT `m`.** The output is row-major, so cell `(i, j)` lives
       at `i*n + j` and changing `m` moves no index. `n` is what moves a cell
       between blocks, and here it also moves the TILE a cell sits in and its
       position inside that tile.
    2. **COMPARE THE WHOLE OVERLAP**, not one cell. Every `(i, j)` with
       `i < 64, j < 4` exists in all three launches, so 256 cells spread over
       many blocks are compared bitwise.

    And two things that check could not do, because its kernel had one plan:

    3. **THE DISPATCHER IS PART OF THE FIXTURE.** `choose_gemm_plan` reads
       `m` and `n`, so `64 x 4` and `64 x 64` take the SPLITK arm while
       `64 x 256` takes a tiled arm. The comparison therefore crosses a
       DISPATCH BOUNDARY, which is the failure mode contract 13.5's last
       sentence forbids ("the numerical tree cannot change at a performance
       dispatch boundary").
    4. **`m` IS VARIED TOO, as a second arm.** It moves no index in the
       row-major output, but it does move which TILE ROW a cell falls in and
       whether the last tile row is ragged, and the tiled plans are new here.

    Two `k` are used: `k = 4096` (`P = 32`, even) and `k = 4097` (`P = 33`,
    ODD with a one-element last leaf), because a carry is where a
    launch-dependent fold would most easily hide.

    BATCH COMPOSITION, added 2026-08-23, two arms:

    5. **`n: 4 vs 4096`** -- the same `(i, j, k)` cell inside a launch of
       256 cells and inside a launch of 262,144 cells. At `n = 4096` every
       cell of the overlap sits in a DIFFERENT BLOCK than it does at `n = 4`
       (a 16x16 tile grid 256 tiles wide against one SPLITK grid), so a
       kernel whose leaf order or fold depends on the block it was scheduled
       in cannot pass both. This is the charter's "batch composition" in its
       literal form: the same cell computed inside a small launch and a
       large one.
    6. **The shared-workspace loop** (`_batch_loop_arm`): four independent
       GEMMs of three shapes, enqueued back to back through ONE workspace
       with no synchronize between them, each compared to the solo launch.

    SABOTAGE, shown to fail: `-D MOJOLEARN_GEMM_SABOTAGE_LEAF_ROTATE=1`
    rotates the leaf start by the block index in every plan (the leaf
    arithmetic and the tree untouched; only WHICH leaf stands at which tree
    position moves with the block). Measured 2026-08-23 on Apple, IDENTICAL:
    arms 1-5 all FAIL (`n: 4 vs 4096`: 144 of 256 cells), and in arm 6 the
    two GEMMs whose shape differs from the solo launch FAIL (144 of 256 each)
    while the two of the SAME shape agree -- as they must, since a rotation
    that is a pure function of the block index reproduces itself at the same
    geometry, which is why the arm has more than one shape in it. The oracle
    gate fails 32 of 62 shapes and the launch-invariance gate fails too.

    ASSERTED IN BOTH MODES, for the reason GATE 2 gives.
    """
    print(
        "check_device_is_batch_invariant ["
        + _mode_name()
        + "]: a cell's bits against the launch width"
    )
    var fails = 0
    with DeviceContext() as ctx:
        for arm in range(2):
            var k = 4096
            if arm == 1:
                k = 4097
            var part = contract_partition(k)
            print(
                "  k = "
                + String(k)
                + "  L = "
                + String(part[0])
                + "  P = "
                + String(part[1])
            )
            # (1) and (2) and (3): vary n, compare the whole overlap, cross
            # the dispatch boundary.
            _batch_arm(ctx, k, 64, 4, 64, 64, 64, 4, String("n: 4 vs 64"), fails)
            _batch_arm(
                ctx, k, 64, 4, 64, 256, 64, 4, String("n: 4 vs 256"), fails
            )
            _batch_arm(
                ctx, k, 64, 64, 64, 256, 64, 64, String("n: 64 vs 256"), fails
            )
            # (4): vary m as well, at a fixed n.
            _batch_arm(
                ctx, k, 64, 64, 130, 64, 64, 64, String("m: 64 vs 130"), fails
            )
            # (5): the same cell inside a small launch and a LARGE one.
            _batch_arm(
                ctx, k, 64, 4, 64, 4096, 64, 4, String("n: 4 vs 4096"), fails
            )
            # (6): several independent GEMMs through one dirty workspace.
            _batch_loop_arm(ctx, k, 64, 4, fails)
    if fails != 0:
        raise Error(
            "check_device_is_batch_invariant ["
            + _mode_name()
            + "]: "
            + String(fails)
            + " arms found a cell whose BITS depend on how many other cells"
            " shared the launch (lines above). That is the defect the"
            " serving world calls batch non-invariance and IDENTITY_PATHS"
            " rows 3 and 7 call 'a block count is a summation order'."
        )
    print(
        "check_device_is_batch_invariant ["
        + _mode_name()
        + "] OK: every overlapping cell identical across launch widths and"
        " across a dispatch boundary"
    )


# ===========================================================================


def _gate(name: String, mut ran: Int, mut failed: Int, e: String):
    ran += 1
    if e.byte_length() > 0:
        failed += 1
        print("!! GATE FAILED: " + name)
        print("   " + e)


def main() raises:
    print(
        "== gemm/original/gemm_device_check.mojo ["
        + _mode_name()
        + "]  sabotage: "
        + gemm_sabotage_name()
        + " =="
    )
    print(
        "   profile: mojolearn.identical.gemm.fp32.v1   contract:"
        " gemm/IDENTICAL_FP32_CONTRACT.md"
    )
    print(
        "   kernel: gemm/original/gemm_identical.mojo   oracle:"
        " gemm/original/gemm_oracle.mojo::gemm_oracle"
    )
    var ran = 0
    var failed = 0

    try:
        check_fold_stack_depth_covers_the_profile()
        _gate(String("check_fold_stack_depth_covers_the_profile"), ran, failed, String(""))
    except e:
        _gate(String("check_fold_stack_depth_covers_the_profile"), ran, failed, String(e))
    try:
        check_strides_match_the_contract_addressing()
        _gate(String("check_strides_match_the_contract_addressing"), ran, failed, String(""))
    except e:
        _gate(String("check_strides_match_the_contract_addressing"), ran, failed, String(e))
    try:
        check_stack_fold_is_the_contract_tree()
        _gate(String("check_stack_fold_is_the_contract_tree"), ran, failed, String(""))
    except e:
        _gate(String("check_stack_fold_is_the_contract_tree"), ran, failed, String(e))
    try:
        check_device_matches_oracle()
        _gate(String("check_device_matches_oracle"), ran, failed, String(""))
    except e:
        _gate(String("check_device_matches_oracle"), ran, failed, String(e))
    try:
        check_device_is_launch_invariant()
        _gate(String("check_device_is_launch_invariant"), ran, failed, String(""))
    except e:
        _gate(String("check_device_is_launch_invariant"), ran, failed, String(e))
    try:
        check_device_is_batch_invariant()
        _gate(String("check_device_is_batch_invariant"), ran, failed, String(""))
    except e:
        _gate(String("check_device_is_batch_invariant"), ran, failed, String(e))

    if failed != 0:
        raise Error(
            "gemm_device_check ["
            + _mode_name()
            + "]: "
            + String(failed)
            + " of "
            + String(ran)
            + " gates FAILED (sabotage: "
            + gemm_sabotage_name()
            + ")"
        )
    print(
        "gemm device kernel + gates: all green ["
        + _mode_name()
        + "]  ("
        + String(ran)
        + " gates, sabotage: "
        + gemm_sabotage_name()
        + ")"
    )
