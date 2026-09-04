# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host Float32 oracle of the Mamba-1 BACKWARD pass under the proposed
profile `mojolearn.identical.mamba1.bwd.fp32.v1`.

**NORMATIVE.** This file is not a second implementation to cross-check
against; it DEFINES the answer. When the device backward and this file
disagree, the device is wrong by definition, per stage and per cell. It is
`archive/plans/mamba/IDENTICAL_BACKWARD_PLAN.md` sections 2, 3.2, 3.4 and 3.5 written on
the CPU through the SAME pinned seam functions the device must call
(`ftz`, `identical_mul_add`, `pinned_mul`, `identical_exp`,
`identical_sigmoid`, `identical_silu`, `identical_div`, `identical_rsqrt`,
and GEMM v1's `gemm_oracle`), so a device card is diffed against it
bitwise by `tools/identity_trace_diff.py`. The plan's own summary of the
toolbox is that **no new transcendental is needed**; every derivative is a
rational function of quantities the forward already computes. Nothing
below calls a primitive the forward lane has not already certified.

NOT A PORT. Upstream ships a backward (`selective_scan_bwd_kernel.cuh`)
and it is deliberately NOT the model: it accumulates `dA`, `dB`, `dC`,
`dD` and `ddelta_bias` with five float `gpuAtomicAdd` calls and picks its
block shape from `params.seqlen` through a table that DIFFERS under
`USE_ROCM`, so its summation order is a function of the sequence length,
the arrival order of blocks and the vendor. Its algorithm is cited seam by
seam in the plan's section 6; its ORDER is refused there and here. The
derivation this file spells is a theorem of the FORWARD's pinned order
(`mamba/IDENTICAL_MAMBA_CONTRACT.md` S1-S17, `mamba_oracle.mojo`) and not
a second source of truth -- plan section 10 item 6 is why: `torch.autograd`
over einsums has no fixed contraction path to transcribe.

**NOTHING IN THIS FILE HAS BEEN RUN.** It has not been compiled. No gate
of plan section 7 (MB1-MB10) exists, no device backward kernel exists, and
no gradient has ever been printed on any vendor. Every sentence below
about what is identical is a PREDICTION.

WHAT IS IMPLEMENTED, and the deviations it spends (all already SPENT in
the plan; this file allocates none of its own):

    1070  T1  the reverse recurrence: t DESCENDING, seed +0.0, FUSED,
              `da` read at t+1
    1071  T2  `h[t-1]` from an explicit checkpoint, never upstream's
              `h[t] - dbu[t]` subtraction
    1072  T3  `dBm` / `dCm` as per-token gemm v1 contractions at
              `(1, D_STATE, d_inner)` over materialized `h` and `dh`
    1073  T4  the `dA` fold over t, DESCENDING
    1074  T5  the parameter fold over b, private slots, ASCENDING, no atomic
    1075  T6  the three-way join at `du`: D-skip, scan, x_proj
    1076  T7  the RMSNorm backward closed form, `rstd^3` left-associated
    1077  T8  the residual join at `dx`, `dres` LEFT -- THE ABSORPTION SITE
    1078  B22 softplus' as a MULTIPLY by `identical_sigmoid`, guard <= 20
    1079      the fourteen gemm routings, entered through
              `mamba_backward.mojo`'s tables (see FREE CHOICE 5)
    1080      the six token-axis reductions as ones-vector v1 GEMMs

**THE FOURTEEN ROUTED SHAPES ARE SPELLED OUT HERE, NOT IMPORTED FROM
`mamba_backward.mojo`, AND THAT IS ON PURPOSE.** Every other quantity in
this repository has ONE producer and this one deliberately has two. Gate
MB1's whole job is to assert that `mamba_backward_proj_a_call`,
`mamba_backward_proj_b_call` and `mamba_reduction_width` return the shapes
plan 3.2 names; if this file obtained its shapes by calling them, a router
that put `dC` on the wrong side would move the oracle and the device
together and MB1 would be comparing a mistake with itself. Each call site
below therefore states its `(op', m', n', k')` in a comment WITH the
derivation, and MB1 compares the two independently derived tables. The
gemm lane named a router that assumes `dC` is always the left operand as
the most likely error, so the check has to be able to see it.

**THE FMA RE-CONTRACTION TRAP, and it applies to the HOST.** `pinned_mul`
is `identical_mul_add(a, b, -0.0)` precisely so that no syntactic multiply
is presented to a compiler that would contract it into a neighboring add;
the gemm README's F3 scar records `var p = a * b; p + c` being contracted
ACROSS STATEMENTS on this host, so storing an intermediate in a `var` is
NOT a guard on its own. Every difference of two products in this file is
written `ftz(ftz(pinned_mul(a, b)) - ftz(pinned_mul(c, d)))` -- three
roundings where a re-contraction would give one. T7's `t1 - t2` is the
only such site and it is spelled that way. The forward lane hit this on
2026-09-03 and the fix is the same fix.

**FREE CHOICES.** Where the plan does not pin an order or a seed, this
file picks one, says so in a `FREE CHOICE n` block at the site, and names
what would falsify it. There are seven and they are collected in
`mamba_backward_free_choices()` so a gate can print them beside the card.
A free choice is a CHOICE and not a derivation; none of them is evidence
of anything until an arm separates it from its alternative.

THE CARD IS TWENTY SIX STAGES, plan section 7, sixteen activation
gradients then ten parameter gradients, and the split is exactly the split
plan section 4.1 draws: **the sixteen are batch-composition invariant and
the ten are DEFINED as sums over the batch and cannot be.** MB5 asserts
the positive half over the sixteen and asserts the EXCLUSION over the ten.

**PER STAGE OR NOT AT ALL.** T8 is the reason. IDENTITY_PATHS row 55
records that at the shape where a forward sabotage was strongest, thirteen
of sixteen intermediate stages moved, `out_proj.out` differed on 23 of 64
cells, and `residual.out` was STILL BIT IDENTICAL, because the residual
add put a value of order 1e-3 beside one of order 1. `bwd.dx` is the
backward's `residual.out`. A gate that compares only `bwd.dx` will report
a broken norm backward as green.
"""

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_mul_add,
    identical_rsqrt,
    identical_sigmoid,
    identical_silu,
)
from gemm.checks.gemm_oracle import OP_NN, OP_TN, gemm_oracle
from mamba.checks.mamba_fixture import (
    D_CONV,
    D_STATE,
    MambaDims,
    MambaWeights,
    RMS_EPS,
)
from mamba.checks.mamba_oracle import (
    MambaStages,
    MambaState,
    pinned_mul,
    refuse_nonfinite,
)


# ===========================================================================
# STAGE LAYOUT KINDS. Copied in VALUE from `mamba_check.mojo` rather than
# imported, because a check file has a `main()` and importing one into an
# oracle inverts the dependency the forward lane established. The three
# constants are the same three integers.
# ===========================================================================

#: `[M, W]`, `M = B * L` rows -- the sixteen activation gradients.
comptime BWD_KIND_TOKEN = 0
#: `[B, W]`, one block per sequence. **No backward stage has this kind**, and
#: the fact that none does is plan section 4.1's positive half: not one
#: activation gradient reduces over the batch.
comptime BWD_KIND_BATCH = 1
#: One buffer for the whole call -- the ten parameter gradients.
comptime BWD_KIND_GLOBAL = 2

#: The card, plan section 7. Sixteen activation stages then ten parameter
#: stages, in the backward's own direction, upstream to downstream.
comptime BWD_STAGE_COUNT = 26
#: Stages `[0, BWD_ACTIVATION_STAGES)` are the activation gradients, the group
#: plan section 4.1 says IS batch-composition invariant. The rest are the
#: parameter gradients, which are DEFINED as sums over the batch. **MB5 must
#: assert the exclusion, not merely skip the ten.**
comptime BWD_ACTIVATION_STAGES = 16


def mamba_backward_stage_names() -> List[String]:
    """The twenty six stage names, card order, plan section 7 verbatim.

    THE ONE PRODUCER. The forward lane put `stage_names()` in
    `mamba_check.mojo`; this lane puts it in the oracle instead, because the
    backward check does not exist yet and a name list that lives in an
    unwritten file is a name list two people will invent twice. A future
    `mamba_backward_check.mojo` must IMPORT this, never restate it.
    """
    var n: List[String] = [
        String("bwd.dres"),
        String("bwd.dg"),
        String("bwd.dsk"),
        String("bwd.dz"),
        String("bwd.dh"),
        String("bwd.dCm"),
        String("bwd.dBm"),
        String("bwd.du_s"),
        String("bwd.ddelta"),
        String("bwd.ddtp"),
        String("bwd.du"),
        String("bwd.dconv"),
        String("bwd.dhin"),
        String("bwd.dnrm"),
        String("bwd.drstd"),
        String("bwd.dx"),
        String("bwd.dW_in"),
        String("bwd.dW_x"),
        String("bwd.dW_dt"),
        String("bwd.dW_out"),
        String("bwd.dA_log"),
        String("bwd.dD"),
        String("bwd.db_dt"),
        String("bwd.dcw"),
        String("bwd.dcb"),
        String("bwd.dw_norm"),
    ]
    return n^


def mamba_backward_stage_kind(i: Int) -> Int:
    """TOKEN for the sixteen activation gradients, GLOBAL for the ten
    parameter gradients. Nothing is BATCH, and that is the point."""
    if i < BWD_ACTIVATION_STAGES:
        return BWD_KIND_TOKEN
    return BWD_KIND_GLOBAL


def mamba_backward_stage_width(i: Int, dims: MambaDims) -> Int:
    """Cells per token for a TOKEN stage, total cells for a GLOBAL one."""
    var dm = dims.d_model
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    if i == 0:
        return dm  # bwd.dres  [M, dm]
    if i == 1 or i == 2 or i == 3:
        return di  # bwd.dg, bwd.dsk, bwd.dz  [M, di]
    if i == 4:
        return di * D_STATE  # bwd.dh  [M, di, N]
    if i == 5 or i == 6:
        return D_STATE  # bwd.dCm, bwd.dBm  [M, N]
    if i >= 7 and i <= 12:
        return di  # du_s, ddelta, ddtp, du, dconv, dhin  [M, di]
    if i == 13:
        return dm  # bwd.dnrm  [M, dm]
    if i == 14:
        return 1  # bwd.drstd  [M]
    if i == 15:
        return dm  # bwd.dx  [M, dm]
    if i == 16:
        return 2 * di * dm  # bwd.dW_in   [2di, dm]
    if i == 17:
        return xr * di  # bwd.dW_x    [r+2N, di]
    if i == 18:
        return di * r  # bwd.dW_dt   [di, r]
    if i == 19:
        return dm * di  # bwd.dW_out  [dm, di]
    if i == 20:
        return di * D_STATE  # bwd.dA_log  [di, N]
    if i == 21 or i == 22 or i == 24:
        return di  # bwd.dD, bwd.db_dt, bwd.dcb  [di]
    if i == 23:
        return di * D_CONV  # bwd.dcw  [di, K]
    return dm  # bwd.dw_norm  [dm]


def mamba_backward_free_choices() -> String:
    """The seven decisions this file MADE because the plan does not state
    them, each with what would falsify it. Print this beside the card.

    A gate that prints a green banner without this block is reporting seven
    unmeasured choices as if they were derivations. The plan's own lesson,
    3.5's `adv_softplus_guard`, is that an arm which runs and moves nothing
    is not coverage; a choice nobody has separated from its alternative is
    not even that.
    """
    return (
        String("mojolearn.identical.mamba1.bwd.fp32.v1 FREE CHOICES")
        + " (7, none derived, none measured)\n"
        + "FC1 B7 silu' middle term `1 + v*(1-sig)` is ONE"
        + " identical_mul_add. The plan gives the expression and the count"
        + " (3 roundings) but never the fusion; only the fused middle has"
        + " both. FALSIFIED BY: a device spelling it unfused (4 roundings),"
        + " or the plan amending the count.\n"
        + "FC2 B18 ddelta is TWO separate ascending-n folds (B path, then A"
        + " path), each seeded +0.0, joined by ONE unfused flushed add with"
        + " ddelta_B LEFT. Plan section 2 (S7', S5', S14') spells exactly"
        + " this; 3.2's B18 says `interleaving pinned` and never says what"
        + " the interleaving is. FALSIFIED BY: a SAB_BWD_DDELTA_INTERLEAVED"
        + " arm folding both terms in one ascending-n chain (upstream's"
        + " shape, bwd_kernel `ddelta_vals[i] += ddelta_u*u + dx*A*a`)"
        + " agreeing with the device and not with this file.\n"
        + "FC3 the T2 h checkpoint is [B, L+1, di, N] TOKEN-MAJOR, slot t+1"
        + " holding h[t] and slot 0 the flushed entry state."
        + " t2_h_checkpoint_floats writes the SIZE as b*di*(l+1)*N, whose"
        + " factor order reads as [B, di, L+1, N]. Layout moves no bits but"
        + " the card compares per cell. FALSIFIED BY: a device writing the"
        + " other order (a permuted, not a moved, checkpoint).\n"
        + "FC4 bwd.dres is recorded as a RAW COPY, unflushed, and flushed at"
        + " each consuming seam -- the forward's treatment of conv.window."
        + " FALSIFIED BY: a device that stores ftz(dres) and differs on a"
        + " denormal dres.\n"
        + "FC5 T5's batch fold is applied to bwd.dA_log ONLY. Plan 3.4 T5"
        + " says it is `shared by dA, dD, db_dt, dcw and dcb`, but 3.2,"
        + " 4.1 and mamba_backward.mojo's eight RED_* ids route those four"
        + " (and dw_norm) to ones-vector v1 GEMMs at k' = M, which subsume"
        + " the batch axis inside v1's own partition. The two answers"
        + " DIFFER whenever M > 128. FALSIFIED BY: a device computing dD by"
        + " private slots plus a b-fold. THIS IS THE ONE FREE CHOICE THAT"
        + " RESOLVES A CONTRADICTION IN THE PLAN RATHER THAN A SILENCE.\n"
        + "FC6 above the softplus guard (biased > 20) ddtp is a flushed COPY"
        + " of ddelta rather than pinned_mul(ddelta, 1.0). Provably"
        + " bit-inert: fma(x, 1, -0.0) == x at every x including both signed"
        + " zeros. Stated because inert is a claim, not an omission.\n"
        + "FC7 d_arg is RECOMPUTED at both its consumers (ddelta's A path"
        + " and T4's dA fold) rather than stored. A host memory choice, not"
        + " an arithmetic one: the expression is pure and deterministic."
        + " FALSIFIED BY: nothing bitwise; it is a note so a reader does not"
        + " read the second spelling as a second answer.\n"
    )


# ===========================================================================
# THE STAGE RECORD
# ===========================================================================


struct MambaBackwardStages(Movable):
    """Every recorded stage of one backward call, in the card's order.

    Token-major layouts, `M = B * L` rows, the same layouts the forward
    oracle records so a per-cell diff is a list index and nothing else. The
    parameter gradients carry their PARAMETER's shape, so `bwd.dW_out` is
    `[d_model, d_inner]` exactly as `w_out` is.

    `h_ckpt` and `dh_slabs` are COMPARE-ONLY and are NOT card stages. `dh`
    IS a card stage (`bwd.dh`, `[M, di, N]`, plan section 7); `h_ckpt` is
    T2's `[B, L+1, di, N]` checkpoint, which the DEVICE gets from the
    forward and which the host must rebuild, so a gate that wants to price
    MB9 (upstream's `h[t] - dbu[t]` recovery) has the numbers to price it
    with.
    """

    # ---- the sixteen activation gradients ---------------------------------
    var dres: List[Float32]  # [M, d_model]        B1
    var dg: List[Float32]  # [M, d_inner]          B2
    var dsk: List[Float32]  # [M, d_inner]         B5
    var dz: List[Float32]  # [M, d_inner]          B8
    var dh: List[Float32]  # [M, d_inner, N]       B13, T1
    var dcm: List[Float32]  # [M, N]               B15, T3
    var dbm: List[Float32]  # [M, N]               B16, T3
    var du_s: List[Float32]  # [M, d_inner]        B17
    var ddelta: List[Float32]  # [M, d_inner]      B18
    var ddtp: List[Float32]  # [M, d_inner]        B22
    var du: List[Float32]  # [M, d_inner]          B29, T6
    var dconv: List[Float32]  # [M, d_inner]       B30
    var dhin: List[Float32]  # [M, d_inner]        B31
    var dnrm: List[Float32]  # [M, d_model]        B35
    var drstd: List[Float32]  # [M]                B40
    var dx: List[Float32]  # [M, d_model]          B41, B42, T7, T8
    # ---- the ten parameter gradients --------------------------------------
    var dw_in: List[Float32]  # [2*d_inner, d_model]        B36
    var dw_x: List[Float32]  # [r+2N, d_inner]              B28
    var dw_dt: List[Float32]  # [d_inner, r]                B25
    var dw_out: List[Float32]  # [d_model, d_inner]         B3
    var da_log: List[Float32]  # [d_inner, N]               B19, B21, T4, T5
    var dd_skip: List[Float32]  # [d_inner]                 B12
    var db_dt: List[Float32]  # [d_inner]                   B23
    var dcw: List[Float32]  # [d_inner, D_CONV]             B32
    var dcb: List[Float32]  # [d_inner]                     B33
    var dw_norm: List[Float32]  # [d_model]                 B39
    # ---- compare-only, NOT card stages ------------------------------------
    var h_ckpt: List[Float32]  # [B, L+1, d_inner, N]       T2, FREE CHOICE 3

    def __init__(out self):
        self.dres = List[Float32]()
        self.dg = List[Float32]()
        self.dsk = List[Float32]()
        self.dz = List[Float32]()
        self.dh = List[Float32]()
        self.dcm = List[Float32]()
        self.dbm = List[Float32]()
        self.du_s = List[Float32]()
        self.ddelta = List[Float32]()
        self.ddtp = List[Float32]()
        self.du = List[Float32]()
        self.dconv = List[Float32]()
        self.dhin = List[Float32]()
        self.dnrm = List[Float32]()
        self.drstd = List[Float32]()
        self.dx = List[Float32]()
        self.dw_in = List[Float32]()
        self.dw_x = List[Float32]()
        self.dw_dt = List[Float32]()
        self.dw_out = List[Float32]()
        self.da_log = List[Float32]()
        self.dd_skip = List[Float32]()
        self.db_dt = List[Float32]()
        self.dcw = List[Float32]()
        self.dcb = List[Float32]()
        self.dw_norm = List[Float32]()
        self.h_ckpt = List[Float32]()


def backward_oracle_dump(st: MambaBackwardStages) -> List[List[Float32]]:
    """The oracle's stages, card order, matching `mamba_backward_stage_names`.

    Twenty six entries. `h_ckpt` is deliberately absent: it is not a card
    stage and a dump that carried it would make the card twenty seven long
    on one side of a diff and twenty six on the other.
    """
    var out = List[List[Float32]]()
    out.append(st.dres.copy())
    out.append(st.dg.copy())
    out.append(st.dsk.copy())
    out.append(st.dz.copy())
    out.append(st.dh.copy())
    out.append(st.dcm.copy())
    out.append(st.dbm.copy())
    out.append(st.du_s.copy())
    out.append(st.ddelta.copy())
    out.append(st.ddtp.copy())
    out.append(st.du.copy())
    out.append(st.dconv.copy())
    out.append(st.dhin.copy())
    out.append(st.dnrm.copy())
    out.append(st.drstd.copy())
    out.append(st.dx.copy())
    out.append(st.dw_in.copy())
    out.append(st.dw_x.copy())
    out.append(st.dw_dt.copy())
    out.append(st.dw_out.copy())
    out.append(st.da_log.copy())
    out.append(st.dd_skip.copy())
    out.append(st.db_dt.copy())
    out.append(st.dcw.copy())
    out.append(st.dcb.copy())
    out.append(st.dw_norm.copy())
    return out^


# ===========================================================================
# SMALL PINNED SPELLINGS
# ===========================================================================


def _zeros(n: Int) -> List[Float32]:
    var xs = List[Float32]()
    for _ in range(n):
        xs.append(Float32(0.0))
    return xs^


def pinned_silu_prime(v: Float32) -> Float32:
    """`silu'(v) = sig(v) * (1 + v*(1 - sig(v)))`, THREE roundings.

    B7 and B30, plan section 3.5. The alternative that is wrong is
    `sig + v*sig*(1-sig)`, which is algebraically equal and a different
    float. `identical_sigmoid` is the forward's own row-52 seam (B6 is a NEW
    SITE for an EXISTING primitive, plan 3.2 category (a)) and the derivative
    is spelled around it rather than around a fresh `1/(1+exp(-v))`.

    FREE CHOICE 1. The plan pins the expression and the ROUNDING COUNT (3);
    it never says which of the three operations is the fused one. Only the
    middle term fused as a single `identical_mul_add(v, 1-sig, 1.0)` gives
    three: subtract, fma, multiply. Written unfused it is four. FALSIFIED BY
    a device that spells the middle unfused and disagrees on `bwd.dz` or
    `bwd.dconv`.

    Re-contraction: `Float32(1.0) - sig` presents no multiply to fuse into,
    the middle is an explicit `fma`, and the outer product is `pinned_mul`,
    which presents no syntactic multiply at all.
    """
    var sig = ftz(identical_sigmoid(v))
    var one_minus = ftz(Float32(1.0) - sig)
    var mid = ftz(identical_mul_add(v, one_minus, Float32(1.0)))
    return ftz(pinned_mul(sig, mid))


def _forward_rstd(sumsq: Float32, dm: Int) -> Float32:
    """`rstd` RECOMPUTED with the forward's own spelling.

    `mamba_oracle.mojo`'s norm block: `mean = identical_div(sumsq, dm)` then
    `rstd = identical_rsqrt(mean + RMS_EPS)`, each flushed. Plan section 6's
    general rule -- *a recomputed forward quantity must be spelled with the
    same function the forward used, not with an algebraically equal one* --
    is the only reason this is a function and not two inline lines.
    """
    var mean = ftz(identical_div(sumsq, Float32(dm)))
    return ftz(identical_rsqrt(ftz(mean + RMS_EPS)))


def _forward_da(delta: Float32, a: Float32) -> Float32:
    """`da[t,d,n] = exp(delta[t,d] * A[d,n])`, seams S5 and S6, RECOMPUTED.

    `identical_exp(pinned_mul(delta, A))`, NOT `exp2f(delta*A*M_LOG2E)`.
    Upstream's forward and backward both use the exp2 substitution and are
    self-consistent; DEVIATION 722 already refused it as a different function
    with an extra rounding, and plan section 6 item 6 says the backward's
    recomputed `da` must be OUR forward's `da` or it is not a recomputation.
    """
    return ftz(identical_exp(ftz(pinned_mul(delta, a))))


def _forward_dbb(delta: Float32, bm: Float32) -> Float32:
    """`dbb[t,d,n] = delta[t,d] * Bm[t,n]`, seam S7's `db`, RECOMPUTED.

    The reference's left-to-right pairwise order (delta with B first), which
    is what the forward oracle spells. The CUDA kernel rounds the OTHER way
    (`B * (delta * u)`); fixture F5 shows the two separate and the
    reference's order is the profile's.
    """
    return ftz(pinned_mul(delta, bm))


# ===========================================================================
# T2, DEVIATION 1071: THE `h` CHECKPOINT
# ===========================================================================


def mamba_h_checkpoint_oracle(
    u: List[Float32],  # [M, d_inner]  silu.out
    delta: List[Float32],  # [M, d_inner]  softplus.out
    a: List[Float32],  # [d_inner, D_STATE]  A.out
    bmat: List[Float32],  # [M, D_STATE]
    h_in: List[Float32],  # [B, d_inner, D_STATE]  the state ENTERING
    b: Int,
    l: Int,
    d_inner: Int,
) -> List[Float32]:
    """`h[t, d, n]` for every `t` in `[-1, L)`, into `[B, L+1, di, N]`.

    **T2, DEVIATION 1071. PINNED AS AN EXPLICIT CHECKPOINT.** The REFUSED
    alternative is upstream's `a = h[t] - dbu[t]`
    (`selective_scan_bwd_kernel.cuh:290`). Since `h[t] = round(da*h[t-1] +
    dbu[t])` that subtraction is exact only when no rounding occurred, and it
    suffers unbounded relative cancellation whenever `|da*h[t-1]| <<
    |dbu[t]|`, which is the NORMAL case early in a sequence where `h` is near
    zero and `dbu` is not. It is a memory optimization with an accuracy cost
    upstream does not state. **Do not port their bugs.** The price is
    `D_STATE = 16` times the activation footprint, priced in plan 5.3.
    `SAB_BWD_H_SUBTRACT` keeps upstream's spelling so MB9 can MEASURE the ulp
    distance rather than assert that it matters -- and if MB9 records zero
    ulps everywhere, T2's refusal downgrades to a preference and this
    function's memory is spent for nothing. Say so if it does.

    The recurrence below is `selective_scan_oracle`'s, character for
    character, minus the `y` fold: the same `pinned_mul`, the same
    `identical_exp`, the same ONE-rounding `identical_mul_add` at seam S9.
    It is a re-derivation of the forward's own order, not a second order, and
    `mamba_block_backward_oracle` REFUSES to continue unless the checkpoint's
    last slot equals the forward's recorded `scan.h` bit for bit.

    FREE CHOICE 3: the layout is TOKEN-MAJOR `[B, L+1, di, N]`, slot `t+1`
    holding `h[t]`, so that T3's per-token `[di, N]` slab is contiguous and
    the card's `bwd.dh [M, di, N]` and this buffer index the same way.
    `t2_h_checkpoint_floats` in `mamba_backward.mojo` writes the SIZE as
    `b * d_inner * (l + 1) * D_STATE`, whose factor order reads as
    `[B, di, L+1, N]`. The two are the same number of floats and a different
    permutation; layout moves no bits but the per-cell card diff does not
    know that, so the device must match THIS one.

    Slot 0 stores the entry state ALREADY FLUSHED, because `ftz` at the read
    is what the forward recurrence consumed.
    """
    var out = _zeros(b * (l + 1) * d_inner * D_STATE)
    for bb in range(b):
        for d in range(d_inner):
            var h = List[Float32]()
            for n in range(D_STATE):
                h.append(ftz(h_in[(bb * d_inner + d) * D_STATE + n]))
            for n in range(D_STATE):
                out[((bb * (l + 1) + 0) * d_inner + d) * D_STATE + n] = h[n]
            for li in range(l):
                var t = bb * l + li
                var uv = ftz(u[t * d_inner + d])
                var dl = ftz(delta[t * d_inner + d])
                for n in range(D_STATE):
                    var da = _forward_da(dl, ftz(a[d * D_STATE + n]))
                    var dbb = _forward_dbb(dl, ftz(bmat[t * D_STATE + n]))
                    var dbu = ftz(pinned_mul(dbb, uv))
                    h[n] = ftz(identical_mul_add(da, h[n], dbu))
                for n in range(D_STATE):
                    out[
                        ((bb * (l + 1) + li + 1) * d_inner + d) * D_STATE + n
                    ] = h[n]
    return out^


def _h_at(
    ck: List[Float32], bb: Int, li: Int, d: Int, l: Int, di: Int, n: Int
) -> Float32:
    """`h[li, d, n]` out of the checkpoint. `li = -1` is the entry state."""
    return ck[((bb * (l + 1) + li + 1) * di + d) * D_STATE + n]


# ===========================================================================
# THE BACKWARD
# ===========================================================================


def mamba_block_backward_oracle(
    w: MambaWeights,
    x: List[Float32],  # [B, L, d_model] token-major, the FORWARD's input
    dres: List[Float32],  # [M, d_model] the gradient arriving at the block
    st: MambaStages,  # the FORWARD's recorded stages, SAME call
    state_in: MambaState,  # the state ENTERING the forward call
    b: Int,
    l: Int,
) raises -> MambaBackwardStages:
    """One `MambaBlock.forward`'s backward, stage by stage, through the seams.

    `st` must be the stages the FORWARD oracle recorded for exactly this
    `(w, x, b, l, state_in)`. `state_in` must be the state BEFORE the forward
    call, not after: `mamba_block_oracle` mutates its `MambaState` argument,
    so the caller has to keep a copy. The backward reads the incoming conv
    WINDOW (for `dcw`'s pre-sequence taps, forward seam S13) and the incoming
    `h` (for the checkpoint's slot 0) out of it, and passing the post-call
    state instead produces a plausible gradient with no shape symptom.

    **THE GRADIENT OF THE CARRIED STATE IS DROPPED**, plan section 2.4.
    `selective_scan_fn`'s docstring says "the gradient of the last state is
    not considered in the backward pass" and the kernel seeds its reverse
    scan with `make_float2(1.f, 0.f)`. This lane adopts that, so
    `dh[L, d, n] = 0` and no gradient flows out of a block call into the
    state a previous call produced. The same is true of the conv window:
    `dhin` at a pre-sequence position would flow into the PREVIOUS call's
    output and is dropped. **Nobody should read "the backward is correct" as
    "the backward does truncated BPTT correctly", because it does no BPTT at
    all.**

    **THERE IS NO DECODE BACKWARD**, plan section 4.2. The forward's clause
    (d) is prefill == decode per token and holds because the recurrence is
    CAUSAL. This recurrence runs the other way: `dh[t]` depends on every
    token from `t` to `L - 1`. Truncate the sequence and `dh[t]` is a
    DIFFERENT NUMBER, not a differently rounded one. There is no prefix
    property to preserve and therefore nothing to pin, and this function has
    no `l == 1` special case for the same reason it has no streaming form.
    """
    refuse_nonfinite("dres", dres)  # plan section 10 item 8, now written
    refuse_nonfinite("state.conv_win", state_in.conv_win)
    refuse_nonfinite("state.h", state_in.h)

    var dims = w.dims.copy()
    var dm = dims.d_model
    var di = dims.d_inner
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    var m = b * l
    var bst = MambaBackwardStages()

    # =======================================================================
    # RECOMPUTED FORWARD QUANTITIES
    # =======================================================================
    # Plan section 6's general rule: a recomputed forward quantity must be
    # spelled with the same function the forward used, not with an
    # algebraically equal one. Upstream breaks it once, recomputing `silu(z)`
    # as `z * sigmoid(z)` where its forward wrote `z / (1 + expf(-z))`, which
    # makes their gradient inconsistent with their own OUTPUT. This
    # repository has MEASURED that exact difference: `SAB_S12_MUL_SIGMOID`
    # bit `gate.out` on 15 of 64 cells, IDENTITY_PATHS row 55. B4 calls
    # `identical_silu`, full stop.

    # The x_proj split (`torch.split`, MM:422) -- bit-exact copies.
    var dt_low = List[Float32]()
    var bmat = List[Float32]()
    var cmat = List[Float32]()
    for t in range(m):
        for j in range(r):
            dt_low.append(st.x_proj[t * xr + j])
        for n in range(D_STATE):
            bmat.append(st.x_proj[t * xr + r + n])
        for n in range(D_STATE):
            cmat.append(st.x_proj[t * xr + r + D_STATE + n])

    # T2's checkpoint, and the assertion that it IS the forward's own scan.
    bst.h_ckpt = mamba_h_checkpoint_oracle(
        st.silu_out, st.softplus_out, st.a_out, bmat, state_in.h, b, l, di
    )
    from std.memory import bitcast

    for bb in range(b):
        for d in range(di):
            for n in range(D_STATE):
                var got = _h_at(bst.h_ckpt, bb, l - 1, d, l, di, n)
                var want = st.scan_h[(bb * di + d) * D_STATE + n]
                if bitcast[DType.uint32](got) != bitcast[DType.uint32](want):
                    raise Error(
                        String("mamba backward oracle: the T2 checkpoint's")
                        + " last slot is not the forward's scan.h at (b="
                        + String(bb)
                        + ", d="
                        + String(d)
                        + ", n="
                        + String(n)
                        + "). The backward is differentiating a DIFFERENT"
                        + " forward than the one that ran."
                    )

    # =======================================================================
    # B1. `bwd.dres`  [M, dm]        stage 0
    # =======================================================================
    # A copy. FREE CHOICE 4: recorded RAW, unflushed, and flushed at each
    # consuming seam -- the forward oracle's own treatment of `conv.window`
    # ("raw copies -- a copy is not an arithmetic seam").
    for i in range(m * dm):
        bst.dres.append(dres[i])

    # =======================================================================
    # B2. `bwd.dg = dO . W_out`  [M, di]        stage 1
    # =======================================================================
    # gemm `dA` for PROJ_OUT. The forward is `OP_NT(gate.out, W_out)` at
    # `(M, dm, di)`, so `gemm_backward_a_call` returns `OP_NN(dC, W)` at
    # `(M, di, dm)` with `dC` LEFT and `k' = dm`. `W_out` is `[dm, di]`
    # row-major, which is exactly `OP_NN`'s `[k, n]`, so nothing is
    # transposed and nothing is copied. `k' = dm` is a LAYER WIDTH, bounded,
    # so at every gate shape `P == 1` and this is v1's serial ascending
    # chain. DEVIATION 1079.
    bst.dg = gemm_oracle(bst.dres, w.w_out, OP_NN, m, di, dm)

    # =======================================================================
    # B3. `bwd.dW_out = dO^T . g`  [dm, di]        stage 19
    # =======================================================================
    # gemm `dB` for PROJ_OUT: `OP_TN(dC, A)` at `(dm, di, M)`, `dC` LEFT.
    # **`k'` IS THE TOKEN COUNT.** At `M > 128` the partition has more than
    # one leaf and v1's balanced tree is exercised through a Mamba entry
    # point for the first time anywhere in this profile, so `SAB_FOLD_STRIDE`,
    # `SAB_PAD_PLUS_ZERO`, `SAB_FOLD_SERIAL` and `SAB_NODE_ORDER` become
    # reachable here. It is also why plan 4.1 says a microbatched Mamba
    # training step is not bit equal to an unsplit one. Computed here, beside
    # its `dA` twin, because the two read the same `dO` and write disjoint
    # buffers; recorded at card position 19.
    bst.dw_out = gemm_oracle(bst.dres, st.gate_out, OP_TN, dm, di, m)

    # =======================================================================
    # B4, B5. `bwd.dsk`  [M, di]        stage 2
    # =======================================================================
    # `gate.out = skip.out * silu(z)` (seam S12), so `dsk = dg * silu(z)`.
    # B4 RECOMPUTES `silu(z)` through `identical_silu`, the forward's own
    # call, and B5 is `pinned_mul(dg, silu_z)`. The wrong spelling is
    # upstream's `pinned_mul(dg, pinned_mul(z, sig))`, plan 3.5.
    var silu_z = _zeros(m * di)
    for t in range(m):
        for d in range(di):
            var z = ftz(st.in_proj[t * 2 * di + di + d])
            silu_z[t * di + d] = ftz(identical_silu(z))
            bst.dsk.append(
                ftz(pinned_mul(ftz(bst.dg[t * di + d]), silu_z[t * di + d]))
            )

    # =======================================================================
    # B6, B7, B8. `bwd.dz`  [M, di]        stage 3
    # =======================================================================
    # `dz = ((dg * sk) * silu'(z))`, LEFT ASSOCIATED, two `pinned_mul`s.
    # The alternative that is wrong is `dg * (sk * silu')`, and upstream's is
    # a four factor chain. Plan 3.5 row B8.
    for t in range(m):
        for d in range(di):
            var z = ftz(st.in_proj[t * 2 * di + di + d])
            var p1 = ftz(
                pinned_mul(
                    ftz(bst.dg[t * di + d]), ftz(st.skip_out[t * di + d])
                )
            )
            bst.dz.append(ftz(pinned_mul(p1, pinned_silu_prime(z))))

    # =======================================================================
    # B9, B13. `bwd.dh`  [M, di, N]        stage 4        T1, DEVIATION 1070
    # =======================================================================
    # `dy = dsk` is a copy (B9). Then T1:
    #
    #     contrib   = ftz(pinned_mul(dy[t,d], Cm[t,n]))
    #     dh[t,d,n] = ftz(identical_mul_add(da[t+1,d,n], dh[t+1,d,n], contrib))
    #
    # REVERSE over `t` from `L - 1` to `0`, seed `dh[L,d,n] = +0.0`, ONE
    # rounding per step, FUSED. FUSED mirrors S9, whose clause says fusion is
    # pinned because only fusion has a portable spelling. **If this is wrong,
    # every `dh` from `t = L-2` downward moves and nine card stages move with
    # it. The cost of being wrong is the whole lane.**
    #
    # **THE OFF-BY-ONE ON `da`.** `h[t]` influences `y[t]` through `Cm[t]`
    # and `h[t+1]` through `da[t+1]`, NOT through `da[t]`. Upstream spells
    # the shift explicitly as `thread_reverse_data[i-1].x = delta_a_exp`
    # (`selective_scan_bwd_kernel.cuh:255`). Getting it wrong produces a
    # plausible gradient that is BIT IDENTICAL on any fixture where all `da`
    # are equal, which is why MB7's fixture must plant unequal `delta` across
    # tokens and the gate must REFUSE to pass if it does not.
    #
    # At `t = L - 1` there is no successor, so `da_next` is `+0.0` and the
    # fma degenerates to `contrib + 0.0`, which is `contrib` at every input
    # including both signed zeros. That is the seed of plan T1 written as a
    # register rather than as a branch, so the loop body has ONE spelling.
    #
    # This whole walk stays inside one `(b, d)` pair holding `N = 16` values
    # in registers, exactly as the forward holds `N` values of `h`. No
    # cross-thread communication, no shared memory, no atomic: **it inherits
    # the forward's structural launch invariance and batch-composition
    # invariance unchanged.** Plan 5.1 -- the seam that looked hardest is not.
    bst.dh = _zeros(m * di * D_STATE)
    for bb in range(b):
        for d in range(di):
            var dhn = _zeros(D_STATE)  # dh[t+1], T1's +0.0 seed
            var dan = _zeros(D_STATE)  # da[t+1], +0.0 at t = L-1
            for li in range(l - 1, -1, -1):
                var t = bb * l + li
                var dyv = ftz(bst.dsk[t * di + d])
                for n in range(D_STATE):
                    var contrib = ftz(
                        pinned_mul(dyv, ftz(cmat[t * D_STATE + n]))
                    )
                    var v = ftz(identical_mul_add(dan[n], dhn[n], contrib))
                    bst.dh[(t * di + d) * D_STATE + n] = v
                    dhn[n] = v
                var dl = ftz(st.softplus_out[t * di + d])
                for n in range(D_STATE):
                    dan[n] = _forward_da(dl, ftz(st.a_out[d * D_STATE + n]))

    # =======================================================================
    # B15, B16. `bwd.dCm`, `bwd.dBm`  [M, N]      stages 5, 6
    # =======================================================================
    # **T3, DEVIATION 1072. THE HARD SEAM.** In Mamba-1 `A` and `D` are per
    # channel but `B` and `C` come out of `x_proj` per TOKEN and are SHARED
    # by every one of the `d_inner` channels. The forward reads them, which
    # costs nothing; the backward SUMS over the channels that read them, and
    # every term of that sum lives in a different thread under the forward's
    # grid. `d_inner` is an axis NO FLOAT CROSSES IN THE FORWARD.
    #
    # Four structures were considered and three refused (plan 5.2): a float
    # `atomicAdd` per term is arrival order, REFUSED by IDENTITY_PATHS rows 1
    # and 2; a threadgroup tree over `d` is REFUSED on Metal, which has no
    # threadgroup float atomics and 32 KB of threadgroup memory; a fixed-point
    # Int32 accumulator is REJECTED, not refused, because gradients have a far
    # wider dynamic range than a histogram bin. PINNED is structure 4:
    # materialize `h` and `dh` and contract PER TOKEN, which is gemm v1 at
    # `(1, D_STATE, d_inner)` exactly.
    #
    # **So the fold order here is v1's certified one and not a hand-declared
    # direction**, which matters at width: if `d_inner > 128` this is NOT one
    # chain, it is `contract_leaf_size(d_inner)` leaves combined by v1's
    # balanced tree, and an implementation that treats it as a single serial
    # chain is a DIFFERENT ANSWER at every real model width. Calling
    # `gemm_oracle` rather than writing the loop is how this file refuses to
    # hold a second opinion about that. **`d_inner` is the reduced length,
    # never a launch quantity.**
    #
    # `dBm`'s leaf multiplies a PRE-FORMED `w[t,d] = pinned_mul(delta, u)` by
    # `dh[t,d,n]`, so the leaf is ONE fma. The alternative pre-forms
    # `R[t,d,n] = pinned_mul(dh, u)`, which costs `N` times the memory and one
    # extra rounding per `(t,d,n)` rather than per `(t,d)`.
    #
    # The price of structure 4 is plan 5.3 and it is stated in the same
    # breath as the claim: `h` and `dh` are each `M * d_inner * N * 4` bytes,
    # which is 2.1 MB at the corpus's largest case and 3.2 GB for a 130M
    # parameter block at `L = 2048, B = 8`. **The declared T3 spelling is a
    # GATE-SCALE construction**; a blocked variant is plan phase K and its
    # tile size must be `contract_leaf_size(d_inner)`'s leaf boundary and
    # NEVER a VRAM budget, or the answer becomes a function of the machine.
    for t in range(m):
        var bb = t // l
        var li = t % l
        var dy_row = List[Float32]()
        var w_row = List[Float32]()
        for d in range(di):
            dy_row.append(bst.dsk[t * di + d])
            w_row.append(
                ftz(
                    pinned_mul(
                        ftz(st.softplus_out[t * di + d]),
                        ftz(st.silu_out[t * di + d]),
                    )
                )
            )
        # `h[t]` is the POST-update state, checkpoint slot `li + 1`, because
        # the forward's `y[t] = sum_n C[t,n] * h[t,n]` reads it after S9.
        var h_slab = List[Float32]()
        var dh_slab = List[Float32]()
        for d in range(di):
            for n in range(D_STATE):
                h_slab.append(_h_at(bst.h_ckpt, bb, li, d, l, di, n))
                dh_slab.append(bst.dh[(t * di + d) * D_STATE + n])
        var cm_row = gemm_oracle(dy_row, h_slab, OP_NN, 1, D_STATE, di)
        var bm_row = gemm_oracle(w_row, dh_slab, OP_NN, 1, D_STATE, di)
        for n in range(D_STATE):
            bst.dcm.append(cm_row[n])
            bst.dbm.append(bm_row[n])

    # =======================================================================
    # B17. `bwd.du_s`  [M, di]        stage 7
    # =======================================================================
    # `du_s[t,d] = sum over n of dh[t,d,n] * dbb[t,d,n]`, a fold over `N`.
    # Category c-INHERITED: the order is S10's ascending fused chain, seeded
    # `+0.0`, one `identical_mul_add` per term, reused verbatim rather than
    # re-declared. `dbb` is RECOMPUTED through `_forward_dbb`.
    # `SAB_BWD_S10_N_DESCENDING` is the arm, predicted INERT at `N == 1` or
    # with fewer than two nonzero terms.
    for t in range(m):
        for d in range(di):
            var dl = ftz(st.softplus_out[t * di + d])
            var acc = Float32(0.0)
            for n in range(D_STATE):
                var dbb = _forward_dbb(dl, ftz(bmat[t * D_STATE + n]))
                acc = ftz(
                    identical_mul_add(
                        ftz(bst.dh[(t * di + d) * D_STATE + n]), dbb, acc
                    )
                )
            bst.du_s.append(acc)

    # =======================================================================
    # B18. `bwd.ddelta`  [M, di]        stage 8
    # =======================================================================
    # The two paths of plan section 2:
    #
    #   S7'  ddelta_B = sum over n of d_dbb[t,d,n] * Bm[t,n],
    #                   d_dbb = pinned_mul(dh, u)
    #   S5'  ddelta_A = sum over n of d_arg[t,d,n] * A[d,n],
    #                   d_arg = pinned_mul(pinned_mul(dh, h[t-1]), da[t,d,n])
    #   S14' ddelta   = ddelta_B + ddelta_A
    #
    # `d_arg`'s `da` factor is S6's "exp' is exp": `d/d(arg) exp(arg) =
    # exp(arg) = da`, so no new transcendental appears. `h[t-1]` is T2's
    # checkpoint at slot `li` (NOT upstream's subtraction).
    #
    # **A backward with only the `B` path is a plausible wrong answer** and
    # plan section 6 lists the two-term `ddelta` among the things we must
    # COPY from upstream.
    #
    # FREE CHOICE 2. Plan 3.2's B18 says "interleaving pinned" and never says
    # what the interleaving IS. Pinned here: TWO separate ascending-`n` folds
    # each seeded `+0.0`, then ONE unfused flushed add with `ddelta_B` on the
    # left, which is section 2's literal spelling. The alternative is a
    # single ascending-`n` chain accumulating both terms per `n`, which is
    # upstream's shape (`ddelta_vals[i] += ddelta_u * u + dx * A * a`).
    # FALSIFIED BY a `SAB_BWD_DDELTA_INTERLEAVED` arm agreeing with the
    # device where this file does not.
    for t in range(m):
        var bb = t // l
        var li = t % l
        for d in range(di):
            var dl = ftz(st.softplus_out[t * di + d])
            var uv = ftz(st.silu_out[t * di + d])
            var acc_b = Float32(0.0)
            for n in range(D_STATE):
                var d_dbb = ftz(
                    pinned_mul(ftz(bst.dh[(t * di + d) * D_STATE + n]), uv)
                )
                acc_b = ftz(
                    identical_mul_add(
                        d_dbb, ftz(bmat[t * D_STATE + n]), acc_b
                    )
                )
            var acc_a = Float32(0.0)
            for n in range(D_STATE):
                var av = ftz(st.a_out[d * D_STATE + n])
                var d_da = ftz(
                    pinned_mul(
                        ftz(bst.dh[(t * di + d) * D_STATE + n]),
                        ftz(_h_at(bst.h_ckpt, bb, li - 1, d, l, di, n)),
                    )
                )
                var d_arg = ftz(pinned_mul(d_da, _forward_da(dl, av)))
                acc_a = ftz(identical_mul_add(d_arg, av, acc_a))
            bst.ddelta.append(ftz(acc_b + acc_a))

    # =======================================================================
    # B22. `bwd.ddtp`  [M, di]        stage 9        DEVIATION 1078
    # =======================================================================
    # `delta = softplus(biased)`, `biased = dt_proj.out[t,d] + b_dt[d]`
    # RECOMPUTED with the forward's own spelling, so `ddtp = ddelta *
    # softplus'(biased)` and `softplus'(x) = sigmoid(x)`.
    #
    #     PINNED    ddtp = ftz(pinned_mul(ddelta, identical_sigmoid(biased)))
    #     upstream  ddtp = ddelta / (1 + expf(-biased))   bwd_kernel:454-457
    #
    # Upstream spells ONE DIVISION where `identical_sigmoid` is
    # `portable_divf(1, 1+e)` then a product -- two roundings against one.
    # The guard is the same in both, `biased <= 20`, and the operand is the
    # PRE-softplus biased value in both. The multiply is pinned because the
    # forward already owns `identical_sigmoid` as row 52's seam.
    #
    # **`SAB_BWD_S14_DIVISION`'s distinguishing band is `biased` in roughly
    # `[8, 14]`; a fixture that only straddles 20 passes this VACUOUSLY. That
    # has already bitten this lane once**, on `adv_softplus_guard`, where 256
    # inputs spanned `[19.87, 20.10]` with ZERO cells in the distinguishing
    # band and a rebuild under `S14_THRESHOLD_10` produced 23 of 23
    # byte-identical dumps. BITWISE INERT. Do not repeat it.
    #
    # FREE CHOICE 6: above the guard the derivative is exactly `1.0` and this
    # file writes a flushed COPY rather than `pinned_mul(ddelta, 1.0)`. The
    # two are bit equal at every input -- `fma(x, 1, -0.0) == x` including
    # both signed zeros -- so this is stated to be inert, not omitted.
    for t in range(m):
        for d in range(di):
            var biased = ftz(st.dt_proj[t * di + d] + ftz(w.b_dt[d]))
            var g = ftz(bst.ddelta[t * di + d])
            if biased <= Float32(20.0):
                bst.ddtp.append(
                    ftz(pinned_mul(g, ftz(identical_sigmoid(biased))))
                )
            else:
                bst.ddtp.append(g)

    # =======================================================================
    # B24, B25, B26. `ddtl`, `bwd.dW_dt`, `dXP`
    # =======================================================================
    # B24 is gemm `dA` for PROJ_DT. The forward is `OP_NT(dt_low, W_dt)` at
    # `(M, di, r)`, so the backward is `OP_NN(dC, W)` at `(M, r, di)` and
    # **`k' = di`, NOT `r`**. Plan section 2's S17c' annotation says
    # "gemm dA, k' = r"; that annotation is WRONG and 3.2's B24 ("reduces
    # over `di`") is right. `gemm_backward_a_call` in `mamba_backward.mojo`
    # is THE ONE DOOR for this shape and it returns `di`.
    var ddtl = gemm_oracle(bst.ddtp, w.w_dt, OP_NN, m, r, di)
    # B25, gemm `dB`: `OP_TN(dC, A)` at `(di, r, M)`. `k'` is the token count.
    bst.dw_dt = gemm_oracle(bst.ddtp, dt_low, OP_TN, di, r, m)
    # B26, a copy: `dXP = concat(ddtl | dBm | dCm)` at `[M, r + 2N]`, the
    # x_proj split's own order.
    var dxp = List[Float32]()
    for t in range(m):
        for j in range(r):
            dxp.append(ddtl[t * r + j])
        for n in range(D_STATE):
            dxp.append(bst.dbm[t * D_STATE + n])
        for n in range(D_STATE):
            dxp.append(bst.dcm[t * D_STATE + n])

    # B27, gemm `dA` for PROJ_X: `OP_NN(dXP, W_x)` at `(M, di, xr)`.
    var du_x = gemm_oracle(dxp, w.w_x, OP_NN, m, di, xr)
    # B28, gemm `dB`: `OP_TN(dXP, silu.out)` at `(xr, di, M)`.  stage 17
    bst.dw_x = gemm_oracle(dxp, st.silu_out, OP_TN, xr, di, m)

    # =======================================================================
    # B10, B29. `bwd.du`  [M, di]        stage 10      T6, DEVIATION 1075
    # =======================================================================
    #     du[t,d] = ftz(ftz(ftz(du_D[t,d]) + du_s[t,d]) + du_x[t,d])
    #
    # D-skip first, then the scan path, then the x_proj path, following the
    # forward's own data flow since `u` is consumed by S11, then the scan,
    # then S17b. **Any permutation is a different answer.**
    # `SAB_BWD_JOIN_ORDER` permutes it and is **predicted INERT on any
    # fixture where one contribution dominates the other two**, so MB4's
    # fixture must plant the three at comparable magnitude and the gate must
    # raise VACUOUS rather than pass if it cannot.
    #
    # `du_D = pinned_mul(dsk, D[d])` (B10). The wrong spelling is fusing it
    # into this join, plan 3.5.
    var du_d = _zeros(m * di)
    for t in range(m):
        for d in range(di):
            du_d[t * di + d] = ftz(
                pinned_mul(ftz(bst.dsk[t * di + d]), ftz(w.d_skip[d]))
            )
            var s1 = ftz(ftz(du_d[t * di + d]) + bst.du_s[t * di + d])
            bst.du.append(ftz(s1 + du_x[t * di + d]))

    # =======================================================================
    # B30. `bwd.dconv`  [M, di]        stage 11
    # =======================================================================
    # `u = silu(conv.out)` (seam S13's silu), so `dconv = du * silu'(conv)`,
    # the SAME three-rounding composite as B7. The wrong spelling is an
    # `identical_div` of the silu quotient form, plan 3.5.
    for t in range(m):
        for d in range(di):
            var c = ftz(st.conv_out[t * di + d])
            bst.dconv.append(
                ftz(pinned_mul(ftz(bst.du[t * di + d]), pinned_silu_prime(c)))
            )

    # =======================================================================
    # B31. `bwd.dhin`  [M, di]        stage 12
    # =======================================================================
    #     dhin[b,p,d] = sum over k of cw[d,k] * dconv[b, p+3-k, d]
    #
    # Category c-INHERITED: S13's tap order (`k` ASCENDING, oldest first, one
    # `identical_mul_add` per tap), with the index reversed. The forward's
    # accumulator is BIAS-SEEDED; there is no bias in the backward, so this
    # one is seeded `+0.0`.
    #
    # **THE INDEX ARITHMETIC, derived rather than recalled** (plan 2.3).
    # Forward, `conv[b,l,d]` reads `hin[b, l-3+k, d]` for `k` in `[0, 4)`, so
    # `hin[b,p,d]` is read by output position `l = p + 3 - k`, in range only
    # for `k` with `0 <= p + 3 - k < L`. The backward is a CORRELATION in the
    # opposite direction and **the tap dropped at the START of the forward
    # sequence is dropped at the END of the backward one.** `p >= 0` and
    # `3 - k >= 0` make the lower bound unreachable, so only `q >= L` drops.
    # **A reversed-tap implementation is bit identical on a symmetric fixture
    # and wrong on every other**, which is why `SAB_BWD_S13_TAPS_REVERSED`
    # exists and why MB2 must DEMONSTRATE the arm passing on a `k`-symmetric
    # weight and print that as the reason it demands an asymmetric one.
    for bb in range(b):
        for p in range(l):
            for d in range(di):
                var acc = Float32(0.0)
                for k in range(D_CONV):
                    var q = p + (D_CONV - 1) - k
                    if q < l:
                        acc = ftz(
                            identical_mul_add(
                                ftz(w.conv_w[d * D_CONV + k]),
                                ftz(bst.dconv[(bb * l + q) * di + d]),
                                acc,
                            )
                        )
                _set_at(bst.dhin, (bb * l + p) * di + d, acc, m * di)

    # =======================================================================
    # B34, B35, B36. `dP`, `bwd.dnrm`, `bwd.dW_in`      stages 13, 16
    # =======================================================================
    # B34 is a copy: `dP = concat(dhin | dz)` at `[M, 2*di]`, the hidden half
    # first, mirroring `in_proj.out`'s `chunk(2, dim=1)` (MM:396).
    var dp = List[Float32]()
    for t in range(m):
        for d in range(di):
            dp.append(bst.dhin[t * di + d])
        for d in range(di):
            dp.append(bst.dz[t * di + d])
    # B35, gemm `dA` for PROJ_IN: `OP_NN(dP, W_in)` at `(M, dm, 2*di)`.
    bst.dnrm = gemm_oracle(dp, w.w_in, OP_NN, m, dm, 2 * di)
    # B36, gemm `dB`: `OP_TN(dP, norm.out)` at `(2*di, dm, M)`.
    bst.dw_in = gemm_oracle(dp, st.norm_out, OP_TN, 2 * di, dm, m)

    # =======================================================================
    # B37, B40. `bwd.drstd`  [M]        stage 14
    # =======================================================================
    # `dinner = pinned_mul(dnrm, w_norm[j])` (B37; the wrong spelling is
    # folding `w_norm` into B35's gemm), then
    #
    #     drstd[t] = sum over j of dinner[t,j] * x[t,j]
    #
    # a fold over `d_model`, category c-INHERITED: S1's ascending fused chain
    # seeded `+0.0`. **The only reduction the RMSNorm backward has is this
    # one**, over the FEATURE axis, per row -- the same axis and the same
    # length as S1's own fold. That is the whole reason T7 has a closed form.
    # `SAB_BWD_S1B_FOLD_DESCENDING` is the arm, INERT at `dm == 1`.
    var dinner = _zeros(m * dm)
    for t in range(m):
        var acc = Float32(0.0)
        for j in range(dm):
            dinner[t * dm + j] = ftz(
                pinned_mul(ftz(bst.dnrm[t * dm + j]), ftz(w.norm_w[j]))
            )
            acc = ftz(
                identical_mul_add(
                    dinner[t * dm + j], ftz(x[t * dm + j]), acc
                )
            )
        bst.drstd.append(acc)

    # =======================================================================
    # B41, B42. `bwd.dx`  [M, dm]      stage 15    T7 + T8, DEVIATIONS 1076-7
    # =======================================================================
    # **T7, the RMSNorm backward closed form.** With `nrm = w * x * rstd`,
    # `rstd = (ss/dm + eps)^{-1/2}` and `ss = sum_j x^2`, we have
    # `d rstd / d ss = -0.5 * rstd^3 / dm` and `d ss / d x_j = 2 x_j`, so
    # `d rstd / d x_j = -rstd^3 * x_j / dm`, giving
    #
    #     dx_norm[t,j] = rstd*dinner[t,j] - (rstd^3 / dm) * x[t,j] * R[t]
    #
    # spelled exactly as plan 3.4 T7 spells it:
    #
    #     c3 = ftz(pinned_mul(ftz(pinned_mul(rstd, rstd)), rstd))
    #     s  = ftz(identical_div(c3, Float32(dm)))
    #     t2 = ftz(pinned_mul(ftz(pinned_mul(s, x[t,j])), drstd[t]))
    #     t1 = ftz(pinned_mul(rstd, dinner[t,j]))
    #     dx_norm = ftz(t1 - t2)
    #
    # `rstd^3` is `(rstd*rstd)*rstd`, LEFT ASSOCIATED, never `portable_powf`
    # (`SAB_BWD_RSTD3_ASSOC` right-associates it, INERT when `rstd` is a power
    # of two). `c3 / dm` is HOISTED per row because it is loop invariant in
    # `j` and a hoisted flushed division moves no bits. **The subtraction is
    # UNFUSED** because the two terms are separately rounded products in every
    # reference spelling; the alternative folds `-1/dm` into the fold as a
    # negative scale, which is one fewer rounding and a different answer.
    #
    # **RE-CONTRACTION.** `ftz(t1 - t2)` is a difference of two products and
    # is exactly the shape a compiler turns into one `fma`. Both operands are
    # `pinned_mul` results -- `identical_mul_add(a, b, -0.0)` -- so there is
    # no syntactic multiply beside the subtract to contract, and both are
    # flushed first. THREE roundings where a re-contraction would give one.
    # Storing them in `var`s is NOT what makes this safe; the gemm README's
    # F3 scar records contraction ACROSS statements on this host.
    #
    # **T8, THE ABSORPTION SITE.** `dx = ftz(ftz(dres) + dx_norm)`, one add,
    # `dres` on the LEFT. This is the exact twin of S16 and IDENTITY_PATHS
    # row 55 records what happened there: at the shape where a sabotage arm
    # was STRONGEST, thirteen of sixteen stages moved, `out_proj.out`
    # differed on 23 of 64 cells, and `residual.out` was STILL BIT IDENTICAL.
    # An output-only gate called that arm INERT. **Any gate that compares
    # only `bwd.dx` will report a broken norm backward as green.** MB8 is the
    # recording gate for exactly this and its prediction is that arms moving
    # ten or more intermediate stages leave `bwd.dx` bit identical on most
    # cells.
    for t in range(m):
        var rstd = _forward_rstd(st.norm_sumsq[t], dm)
        var c3 = ftz(pinned_mul(ftz(pinned_mul(rstd, rstd)), rstd))
        var s = ftz(identical_div(c3, Float32(dm)))
        for j in range(dm):
            var t2 = ftz(
                pinned_mul(
                    ftz(pinned_mul(s, ftz(x[t * dm + j]))),
                    ftz(bst.drstd[t]),
                )
            )
            var t1 = ftz(pinned_mul(rstd, dinner[t * dm + j]))
            var dx_norm = ftz(t1 - t2)
            bst.dx.append(ftz(ftz(dres[t * dm + j]) + dx_norm))

    # =======================================================================
    # B19, B21. `bwd.dA_log`  [di, N]     stage 20    T4 + T5, DEV 1073-4
    # =======================================================================
    # **T4, the `dA` fold over `t`, PINNED DESCENDING**, the direction the
    # reverse pass already walks, from `+0.0`, FUSED:
    #
    #     for t from L-1 down to 0:
    #         acc = ftz(identical_mul_add(d_arg[t,d,n], delta[t,d], acc))
    #
    # **This is the one place either profile pins a DESCENDING fold** and
    # every other fold in both is ascending. Ascending would need either a
    # second pass over `M * di * N` stored per-token products or a second
    # traversal. `SAB_BWD_DA_ASCENDING` is the other direction and **MB4 must
    # show it bites, otherwise this deviation is unmeasured and should be
    # re-decided in favor of consistency.**
    #
    # **`dA` IS ROUTABLE TO GEMM v1 AND IS DELIBERATELY NOT ROUTED**, the one
    # place the plan declines a category (b) answer. With `q[t, d*N+n] =
    # pinned_mul(d_arg, delta)` materialized at `[M, di*N]`, `dA` is a
    # ones-vector `OP_NN` at `(1, di*N, M)` and its fold order would be v1's
    # certified one rather than a hand-declared direction. Declined because
    # `q` is a THIRD buffer of `M * di * N` floats on top of T2's `h` and
    # T3's `dh`. **If a blocked variant ever makes the memory affordable this
    # decision should be revisited and the routed form is strictly
    # preferable**, because it deletes a declared order in favor of a
    # certified one.
    #
    # **T5, the parameter fold over `b`.** The per-`(b, d, n)` result goes to
    # a PRIVATE SLOT with NO ATOMIC ANYWHERE, and a second pass folds over `b`
    # ASCENDING from `+0.0` with a plain flushed add. Upstream instead calls
    # float `gpuAtomicAdd` five times, which is arrival order and therefore
    # not reproducible run to run on ONE device; that is refused for
    # IDENTITY_PATHS rows 1 and 2's reason. **Metal is the forcing function**:
    # with no threadgroup float atomics and 32 KB of threadgroup memory a fold
    # needing scratch proportional to `d_inner` would not fit at any real
    # width, so routing through global private slots and a second pass is the
    # spelling with no ceiling. `SAB_BWD_PARAM_ATOMIC` is the arm and **only
    # MB4 can see it**, because an atomic accumulation is bit identical to a
    # pinned fold on any run whose arrival order happens to match.
    #
    # FREE CHOICE 5. T5's prose says the `b` fold is "shared by `dA`, `dD`,
    # `db_dt`, `dcw` and `dcb`", but 3.2, 4.1 and `mamba_backward.mojo`'s
    # eight `RED_*` ids route those four (and `dw_norm`) to ones-vector v1
    # GEMMs at `k' = M`, which subsume the batch axis inside v1's own
    # partition and leave no per-`b` partial for T5 to fold. Pinned here:
    # **T4 + T5 for `dA_log` only; the routed answer for the other five.**
    # The two differ whenever `M > 128`. This is the one free choice that
    # resolves a CONTRADICTION in the plan rather than a silence.
    #
    # The `b` fold seeds at `+0.0` and performs `B` adds, so at `B == 1` a
    # `-0.0` partial is laundered to `+0.0`. That is gemm v1 section 9.2(a)'s
    # inherited clause, the same laundering `dt_proj`'s `k = 1` leaf already
    # carries in the forward, and it is recorded rather than avoided.
    #
    # B21 closes it: `A = -exp(A_log)` (seam S15), so `dA_log = dA * A`. The
    # alternative is recomputing `-exp(A_log)`, which is bit equal and a
    # second transcendental call for nothing, plan 3.5.
    var da_partial = _zeros(b * di * D_STATE)
    for bb in range(b):
        for d in range(di):
            for n in range(D_STATE):
                var av = ftz(st.a_out[d * D_STATE + n])
                var acc = Float32(0.0)
                for li in range(l - 1, -1, -1):  # T4: DESCENDING
                    var t = bb * l + li
                    var dl = ftz(st.softplus_out[t * di + d])
                    var d_da = ftz(
                        pinned_mul(
                            ftz(bst.dh[(t * di + d) * D_STATE + n]),
                            ftz(_h_at(bst.h_ckpt, bb, li - 1, d, l, di, n)),
                        )
                    )
                    var d_arg = ftz(pinned_mul(d_da, _forward_da(dl, av)))
                    acc = ftz(identical_mul_add(d_arg, dl, acc))
                da_partial[(bb * di + d) * D_STATE + n] = acc
    for d in range(di):
        for n in range(D_STATE):
            var acc = Float32(0.0)
            for bb in range(b):  # T5: ASCENDING, plain flushed add
                acc = ftz(acc + ftz(da_partial[(bb * di + d) * D_STATE + n]))
            bst.da_log.append(
                ftz(pinned_mul(acc, ftz(st.a_out[d * D_STATE + n])))
            )

    # =======================================================================
    # THE FIVE ONES-VECTOR TOKEN-AXIS REDUCTIONS      DEVIATION 1080
    # =======================================================================
    # Each is `sum over the M token rows of a [M, W] matrix` giving `[W]`,
    # which `mamba_backward.mojo::mamba_backward_reduce_into` spells as
    # `identical_gemm_backward_bias_into`, which is `identical_gemm_into(out,
    # ones, src, ws, 1, W, M, OP_NN)`. DEVIATION 851's argument is that
    # `fma(ftz(1.0), ftz(x), acc)` is ONE rounding of `x + acc`, so the ones
    # vector turns the reduction into the gemm contract's own ascending
    # flushed leaf chain and its own balanced tree, **under the same
    # certificate as the projections** instead of a second fold shape needing
    # its own clause, its own fixtures and its own sabotages. The host
    # spelling is therefore `gemm_oracle(ones, src, OP_NN, 1, W, M)` and not
    # a loop.
    #
    # **`k'` IS `M` IN ALL FIVE.** Plan 4.1: these are as microbatch
    # sensitive as the four weight matmuls, and they are the ones a reader is
    # most likely to assume are exempt, because they look like reductions
    # rather than matmuls. `m == 0` gives `k' == 0` and every entry is a
    # STORED `+0.0`, which gemm contract section 8 requires to be written
    # rather than skipped.
    #
    # Five of the eight `RED_*` ids take a PRE-PRODUCT the caller
    # materializes; `RED_DT_BIAS` and `RED_CONV_BIAS` reduce a recorded stage
    # directly. A caller that passes a raw stage where a pre-product belongs
    # computes a plausible column sum of the wrong quantity and **no shape
    # check can see it**, which is why MB2's finite-difference arm is the
    # gate for this.
    var ones = List[Float32]()
    for _ in range(m):
        ones.append(Float32(1.0))

    # ---- B11, B12. `bwd.dD`  [di]        stage 21 -------------------------
    # `dD[d] = sum over tokens of dsk[t,d] * u[t,d]`. B11's pre-product is
    # `pinned_mul(dsk, u)` and the wrong spelling is fusing it into the
    # ones-vector leaf, plan 3.5.
    var p_d = _zeros(m * di)
    for t in range(m):
        for d in range(di):
            p_d[t * di + d] = ftz(
                pinned_mul(
                    ftz(bst.dsk[t * di + d]), ftz(st.silu_out[t * di + d])
                )
            )
    bst.dd_skip = gemm_oracle(ones, p_d, OP_NN, 1, di, m)

    # ---- B23. `bwd.db_dt`  [di]        stage 22 ---------------------------
    # `d(dt_proj.bias)[d] = sum over tokens of ddtp[t,d]`, no pre-product.
    bst.db_dt = gemm_oracle(ones, bst.ddtp, OP_NN, 1, di, m)

    # ---- B32. `bwd.dcw`  [di, K]        stage 23 --------------------------
    # `d(conv1d.weight)[d,k] = sum over tokens of dconv[t,d] * hin[t-3+k, d]`,
    # FOUR separate reductions, one per tap, each a contiguous `[di]` vector
    # scattered into the tap column. The scatter is a copy, not a seam.
    # **The shifted read must take a pre-sequence position from the conv
    # WINDOW exactly as forward seam S13 does** -- zeros on prefill, the
    # carried window otherwise -- and reading zero there instead would be a
    # silently different gradient on every decode-resumed call.
    # `SAB_BWD_CONV_TAP_SLOT` reverses the slot, which puts correct values in
    # the wrong tap and is BITWISE INERT when the conv weight is symmetric in
    # `k`; MB1 must assert the fixture is asymmetric before counting that arm
    # as covered.
    bst.dcw = _zeros(di * D_CONV)
    for k in range(D_CONV):
        var p_cw = _zeros(m * di)
        for bb in range(b):
            for li in range(l):
                var p = li - (D_CONV - 1) + k
                for d in range(di):
                    var hv: Float32
                    if p >= 0:
                        hv = st.in_proj[(bb * l + p) * 2 * di + d]
                    else:
                        hv = state_in.conv_win[
                            (bb * di + d) * D_CONV + (D_CONV + p)
                        ]
                    p_cw[(bb * l + li) * di + d] = ftz(
                        pinned_mul(
                            ftz(bst.dconv[(bb * l + li) * di + d]), ftz(hv)
                        )
                    )
        var tap = gemm_oracle(ones, p_cw, OP_NN, 1, di, m)
        for d in range(di):
            bst.dcw[d * D_CONV + k] = tap[d]

    # ---- B33. `bwd.dcb`  [di]        stage 24 -----------------------------
    # `d(conv1d.bias)[d] = sum over tokens of dconv[t,d]`, no pre-product.
    # The forward's accumulator is BIAS-SEEDED (DEVIATION 721), so the bias
    # enters each output exactly once with coefficient one and its gradient
    # is the plain column sum.
    bst.dcb = gemm_oracle(ones, bst.dconv, OP_NN, 1, di, m)

    # ---- B38, B39. `bwd.dw_norm`  [dm]        stage 25 --------------------
    # `d(norm.weight)[j] = sum over tokens of dnrm[t,j] * inner[t,j]`, where
    # `inner` is S3's output, BEFORE the weight multiply. **Recovering it by
    # dividing `norm.out` by the weight is a division and a different
    # answer** (plan 3.5 row B38), so it is RECOMPUTED with the forward's own
    # `pinned_mul(x, rstd)`.
    var p_w = _zeros(m * dm)
    for t in range(m):
        var rstd = _forward_rstd(st.norm_sumsq[t], dm)
        for j in range(dm):
            var inner = ftz(pinned_mul(ftz(x[t * dm + j]), rstd))
            p_w[t * dm + j] = ftz(
                pinned_mul(ftz(bst.dnrm[t * dm + j]), inner)
            )
    bst.dw_norm = gemm_oracle(ones, p_w, OP_NN, 1, dm, m)

    return bst^


def _set_at(mut xs: List[Float32], i: Int, v: Float32, size: Int):
    """Write-by-index into a stage list, growing it to `size` zeros first.
    The forward oracle's helper, same shape, because `bwd.dhin` is filled by
    `(bb, p, d)` and recorded token-major."""
    while len(xs) < size:
        xs.append(Float32(0.0))
    xs[i] = v


# ===========================================================================
# RUN OWED
# ===========================================================================
# **NOTHING BELOW HAS BEEN RUN AND THIS FILE HAS NEVER BEEN THROUGH A
# COMPILER.** It is a library module with no `main()`, so `mojo build` on it
# fails with "module does not define a `main` function" before it type checks
# anything -- `pixi.toml` records that exact failure for 21 of 22 check files
# and `mamba_backward_compile_probe.mojo` exists for the routing layer for
# exactly this reason. The orchestrator runs these; a lane does not.
#
# ---------------------------------------------------------------------------
# OWED 1. THE COMPILE PROBE. Create the driver, then build it.
# ---------------------------------------------------------------------------
#   cat > /Users/andrewhendel/CascadeProjects/mojolearn/mamba/checks/mamba_backward_oracle_compile_probe.mojo <<'EOF'
#   # SPDX-License-Identifier: Apache-2.0
#   # COMPILE + SMOKE ONLY. It asserts no bits and is NOT one of MB1-MB10.
#   from mamba.checks.mamba_fixture import (
#       MambaDims, corpus_case, corpus_case_seed, corpus_case_weights,
#       corpus_case_x, corpus_tensor,
#   )
#   from mamba.checks.mamba_oracle import MambaState, mamba_block_oracle
#   from mamba.checks.mamba_backward_oracle import (
#       BWD_STAGE_COUNT, backward_oracle_dump, mamba_backward_free_choices,
#       mamba_backward_stage_names, mamba_backward_stage_width,
#       mamba_block_backward_oracle,
#   )
#
#   def main() raises:
#       var k = 1                      # corpus case base_b2_l4_d8
#       var c = corpus_case(k)
#       var dims = MambaDims.of(c.d_model)
#       var w = corpus_case_weights(k)
#       var x = corpus_case_x(k)
#       var m = c.b * c.l
#       var st_in = MambaState(c.b, dims)
#       var st_entry = st_in.copy()    # the state BEFORE the forward call
#       var fwd = mamba_block_oracle(w, x, c.b, c.l, st_in)
#       var dres = corpus_tensor(
#           corpus_case_seed(k) + 0x777, 12, m * c.d_model, -1.0, 1.0
#       )
#       var bwd = mamba_block_backward_oracle(
#           w, x, dres, fwd, st_entry, c.b, c.l
#       )
#       print(mamba_backward_free_choices())
#       var names = mamba_backward_stage_names()
#       var dump = backward_oracle_dump(bwd)
#       if len(names) != BWD_STAGE_COUNT or len(dump) != BWD_STAGE_COUNT:
#           raise Error("the card is not 26 stages")
#       for i in range(BWD_STAGE_COUNT):
#           var wdt = mamba_backward_stage_width(i, dims)
#           print(names[i], "len", len(dump[i]), "width", wdt, dump[i][0])
#       print("h_ckpt", len(bwd.h_ckpt))
#   EOF
#
#   \
#     pixi run mojo build -I . \
#       mamba/checks/mamba_backward_oracle_compile_probe.mojo \
#       -o /tmp/mojolearn_mb_oracle_probe
#
#   nice -n 19 /tmp/mojolearn_mb_oracle_probe
#
# WHAT A PASS MEANS AND DOES NOT. It means the file compiles, the card is 26
# stages long, every stage's recorded length is a whole multiple of the width
# this file declares, and the T2 checkpoint's last slot equals the forward's
# recorded `scan.h` bit for bit (the oracle RAISES if it does not). **It says
# nothing whatever about whether any gradient is correct.** Plan section 7's
# MB1 through MB10 are what say that and not one of them exists.
#
# ---------------------------------------------------------------------------
# OWED 2. THE STAGE-LENGTH ASSERTION, which the probe above only prints.
# ---------------------------------------------------------------------------
# For each `i`, `len(dump[i])` must equal `m * width(i)` for a TOKEN stage and
# `width(i)` for a GLOBAL one. Worth an explicit `raise` in the eventual
# check, because a stage that comes back the right SHAPE and the wrong LENGTH
# is the failure `_set_at`'s grow-to-size spelling makes silent.
#
# ---------------------------------------------------------------------------
# OWED 3. THE PIXI TASK, once the probe exists.
# ---------------------------------------------------------------------------
#   # COMPILE + SMOKE ONLY, and the name says so. Builds the host backward
#   # oracle and prints its 26-stage card shape; asserts no gradient.
#   build-mamba-backward-oracle-probe = "mojo build -I . mamba/checks/mamba_backward_oracle_compile_probe.mojo -o /tmp/mojolearn_mb_oracle_probe"
#
# ---------------------------------------------------------------------------
# OWED 4. THE GATES. None exists. Plan section 7, in the order the plan's
# phase ladder (section 8) puts them:
# ---------------------------------------------------------------------------
#   MB1 check_backward_routing_is_the_table          host, needs this file
#   MB2 check_backward_is_the_derivative             host, needs this file
#   MB3 check_backward_device_matches_oracle         device
#   MB4 check_backward_is_launch_invariant           device
#   MB5 check_backward_activation_gradients_are_batch_invariant   device
#   MB6 check_parameter_gradients_are_not_microbatch_invariant    device
#   MB7 check_the_recurrence_is_actually_reversed    device
#   MB8 record_the_absorption_at_the_backward_residual            device
#   MB9 price_the_upstream_h_recovery                either
#   MB10 the card and the three vendor leg           device
#
# MB1 and MB2 are HOST ONLY and are unblocked by this file. **MB2 is the only
# gate that asks whether the answer is the RIGHT derivative**, which matters
# because a transposed conv backward is bit identical on three boxes, and it
# cannot be tolerance free the way the gemm lane's was: `exp`, `softplus`,
# `silu` and `rsqrt` make no step size exact. Its split is fourteen BILINEAR
# sub-seams asserted BITWISE at a central difference of `h = 1` on an integer
# fixture, and the transcendental seams against a Float64 directional
# derivative of `mamba_block_ref64` at a tolerance calibrated PER CASE from
# that function's own self test, **never a fixed epsilon** -- `adv_gate_
# saturation` passed only at `rtol=1e-3` and the tolerance was the defect,
# not the block.
#
# **NO Float64 BACKWARD REFERENCE IS WRITTEN HERE, DELIBERATELY.** The
# forward oracle carries `mamba_block_ref64` at its bottom and MB2's
# transcendental arm takes a DIRECTIONAL DERIVATIVE of that existing
# function. A second Float64 backward would be a second derivation to keep
# correct and would not be independent of this one.
#
# ---------------------------------------------------------------------------
# OWED 5. THE FIXTURE REQUIREMENTS, which are gate guards and not defaults.
# ---------------------------------------------------------------------------
# Each is a plan section 7 guard that this file cannot enforce and that a
# gate must REFUSE to pass without:
#   - `delta` VARIES ACROSS `t`, or T1's `da` off-by-one is bitwise inert.
#   - `L >= 8` for MB7, which must REFUSE to run at `L == 1`.
#   - `L == 1` PRESENT AND MARKED in MB3's sweep, since four of the fourteen
#     sabotages are predicted inert there and a ledger that omits it reports
#     four false negatives.
#   - the conv weight ASYMMETRIC in `k`, or `SAB_BWD_S13_TAPS_REVERSED` and
#     `SAB_BWD_CONV_TAP_SLOT` are both inert. MB2 must DEMONSTRATE this by
#     running the arm on a symmetric weight, showing it PASSES, and printing
#     that as the reason.
#   - `biased` with cells in `[8, 14]` for `SAB_BWD_S14_DIVISION`. A fixture
#     that only straddles 20 passes VACUOUSLY and this lane has already been
#     bitten by exactly that.
#   - the three `du` contributions at COMPARABLE MAGNITUDE for T6, or
#     `SAB_BWD_JOIN_ORDER` is inert and the gate must raise VACUOUS.
#   - `B >= 2` for T5, and `d_inner >= 2` for T3's two atomic arms, which
#     only MB4 can see at all.
