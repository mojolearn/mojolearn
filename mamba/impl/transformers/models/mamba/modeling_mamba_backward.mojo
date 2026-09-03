# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The BACKWARD of `MambaMixer.forward` and `MambaBlock.forward`, on device.

The block half of `mojolearn.identical.mamba1.bwd.fp32.v1`. Its forward twin
is `mamba/impl/transformers/models/mamba/modeling_mamba.mojo`, and the split
is the same one: that file owns forward seams S1-S4 (RMSNorm), S12 (the z
gate), S13 (the conv tap chain), S14 (the softplus), S15, S16 (the residual)
and the four S17 projections, and this file owns their derivatives.
`mamba/impl/mamba_ssm/ops/selective_scan_backward.mojo` owns S5'-S11', the
scan. `mamba/IDENTICAL_BACKWARD_PLAN.md` is the plan and its topology names
T6, T7, T8 are used below without restating their prices.

WHAT IS HERE, BY THE PLAN'S OPERATION NUMBERS
----------------------------------------------
    B4-B8   S12', the z gate: `silu(z)` recomputed, `dsk`, `silu'(z)`, `dz`
    B10     `du_D = pinned_mul(dsk, D[d])`
    B11     `pD = pinned_mul(dsk, u)`, the pre-product `dD` reduces
    B22     S14', softplus' as a MULTIPLY          DEVIATION 1078
    B26     `dXP = concat(ddtl, dBm, dCm)`, a copy
    B29     T6, the three way join at `du`         DEVIATION 1075
    B30     `dconv = du * silu'(conv)`
    B31     S13', the four tap correlation, reversed index
    B32     the four `d(conv1d.weight)` pre-products
    B34     `dP = concat(dhin, dz)`, a copy
    B37     `dinner = pinned_mul(dnrm, w_norm)`
    B38     `pW = pinned_mul(dnrm, inner)`, the pre-product `dw_norm` reduces
    B40     `drstd[t] = sum_j dinner * x`, S1's ascending chain
    B41     T7, the RMSNorm backward closed form   DEVIATION 1076
    B42     T8, the residual join, THE ABSORPTION SITE  DEVIATION 1077

NEW DEVIATIONS THIS FILE SPENDS
--------------------------------
**DEVIATION 1085, `silu'` HAS EXACTLY ONE SPELLING IN THIS REPOSITORY AND IT
IS NOT NEW.** `transformer/checks/transformer_backward.mojo:713`
(`bwd_silu_backward_kernel`, DEVIATION 1411) already writes the SiLU
derivative under this repository's pins, and it writes

    sg = identical_sigmoid(x);  r1 = ftz(1.0 - sg);  r2 = pinned_mul(x, r1)
    r3 = ftz(1.0 + r2);         r4 = pinned_mul(sg, r3)

which is `sig * (1 + x*(1 - sig))`, left to right, FOUR roundings after the
sigmoid. `_silu_prime` below is that function, transcribed from that file
rather than re-derived, and both of this file's SiLU-derivative sites (B7 at
`z`, B30 at `conv.out`) call it. **The plan's section 3.2 row B7 says "3
roundings" and that is an undercount**: the subtraction `1 - sig` is a
rounding too. Corrected in the plan in the same commit. That lane's arm
`B20_SILU_DERIV_ALT_ASSOC` (`sig + x*sig*(1-sig)`) is plan section 3.5's own
listed alternative for B7, and it is carried here under this lane's name.

**DEVIATION 1086, `SAB_BWD_RSTD3_ASSOC` AS THE PLAN SPELLS IT IS VACUOUS BY
CONSTRUCTION, AND IT IS RE-POINTED.** Plan section 7's table lists an arm
"`rstd^3` right associated", predicted inert only when "`rstd` is a power of
two". That prediction is wrong and the arm as written can never fire:
`(r*r)*r` and `r*(r*r)` are the SAME two operations -- one rounded product
`p = r*r`, then one rounded product `p*r` -- because IEEE multiplication is
commutative, so the two associations are bit identical on EVERY input,
including both zero signs and every NaN-free special. An arm that cannot be
witnessed may not be credited; that is `mamba/impl/mamba_ssm/ops/
mamba3_siso.mojo`'s DEVIATION 834 rule and it applies here. The arm is
therefore re-pointed at the spelling T7 actually forbids, `identical_pow(rstd,
3.0)`, which is a different function with a different rounding, and the plan's
table row is corrected. **This was found by reading, not by running**; no
fixture has been built and nothing has been measured.

**DEVIATION 1087, THIS REPOSITORY NOW HAS TWO RMSNORM BACKWARDS AND THEY ARE
NOT THE SAME NUMBER.** `transformer/checks/transformer_backward.mojo`'s `bwd_norm_dh_kernel` (:801),
`bwd_norm_dot_kernel` (:824) and `bwd_norm_dx_kernel` (:932) compute the same
mathematical node as T7 between them, and spell it

    r3 = (rstd*rstd)*rstd;  cr3 = c*r3;  da = (-0.5)*cr3;  dv = da/dm
    dx = ftz( (dh_j*rstd) + (dv * (2.0*x_j)) )

against T7's

    c3 = (rstd*rstd)*rstd;  s = c3/dm;  t2 = (s*x_j)*c;  t1 = rstd*dinner_j
    dx = ftz(t1 - t2)

The `-0.5` and the `2.0` are exact scalings and move no bits away from the
overflow and subnormal edges, so the difference that matters is the
ASSOCIATION: the transformer folds `c` into the row coefficient before
touching `x_j`, and T7 folds `x_j` into the row coefficient before touching
`c`. `(s*x)*c` and `((c*r3/dm)*2*x/2)` are different roundings of the same
real number. **Neither is wrong and no gate in this repository can tell them
apart**, because the two lanes have separate oracles and separate cards. T7 is
what this profile pins, because DEVIATION 1076 pins it; the divergence is
recorded here so that a future lane unifying the two knows there is a choice
to make and that making it silently would move one lane's card.

WHAT THIS FILE IS NOT
----------------------
Not a gate, not a driver and not an orchestrator: there is no function here
that runs a whole backward pass, because the order the launches go in is the
caller's and `mamba/checks/mamba_backward.mojo` is where the routed calls
live. **Nothing here has been compiled or run**, no gradient has ever come
out of it, and every sentence about behavior is a prediction. Gates MB1-MB10
are SPECIFIED AND NOT BUILT and the host backward oracle they compare against
does not exist.

`refuse_nonfinite` HAS NO BACKWARD COUNTERPART AND IS NOT WRITTEN. Plan
section 10 item 8 names it: `dres` needs the same by-bits refusal the forward
gives its inputs, and a NaN or infinity arriving in an incoming gradient will
propagate through every kernel below without being named. That is an OWED
item, not a decision.

`[[mojo-buffer-freed-at-last-use]]`: every launcher is an `_into` form. It
enqueues and returns. THE CALLER OWNS EVERY BUFFER and must keep each alive
past its own `ctx.synchronize()`.

DEVIATIONS 1075-1078 (T6, T7, T8 and the softplus derivative, declared in
`mamba/checks/mamba_backward.mojo`), 1085-1087 (spent here).
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined

from max.gpu.host import DeviceBuffer, DeviceContext

from mamba.checks.mamba_fixture import D_CONV, D_STATE, MambaDims, RMS_EPS
from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_mul_add,
    identical_pow,
    identical_rsqrt,
    identical_sigmoid,
    identical_silu,
)

# THE FORWARD'S OWN SPELLINGS, IMPORTED RATHER THAN RE-DECLARED. `pinned_mul`
# is DEVIATION 720's device construction, and the two forward arms are
# imported so that a sabotaged forward gets a matching derivative: a backward
# that differentiates a function the forward did not compute is not a
# backward. (`transformer/checks/transformer_backward.mojo:166` imports
# `pinned_mul` from the same place, for the same reason.)
from mamba.impl.transformers.models.mamba.modeling_mamba import (
    SAB_S12_MUL_SIGMOID,
    SAB_S14_THRESHOLD_10,
    pinned_mul,
)


# ===========================================================================
# LAUNCH GEOMETRY. An EXECUTION plan quantity (DEVIATION 725's rule): it
# decides which thread owns a cell and nothing else. No kernel below reads
# `block_dim` or `block_idx` in any expression that reaches a fold boundary,
# an accumulator seed or a tap index. Declared here rather than imported from
# the forward module so a reader can see in one place that this file's
# geometry is its own.
# ===========================================================================

comptime BWD_TPB = 128


def _grid(n: Int) -> Int:
    var g = (n + BWD_TPB - 1) // BWD_TPB
    if g < 1:
        return 1
    return g


# ===========================================================================
# THE SABOTAGE SWITCHES THIS FILE OWNS
# ===========================================================================
# OFF in every build that does not name them. Each carries its own INERT
# prediction, because this lane has already paid for an arm that ran, reached
# its branch and moved nothing (`adv_softplus_guard`: 256 softplus inputs
# spanning [19.87, 20.10], ZERO cells in the distinguishing band, 23 of 23
# byte-identical dumps under `S14_THRESHOLD_10`). A gate must assert its arm's
# predicted cell count, not merely that the arm ran.

#: B4 recomputes `silu(z)` as `z * sigmoid(z)` instead of calling
#: `identical_silu`. **THIS IS UPSTREAM'S ACTUAL BUG, NOT A HYPOTHETICAL**:
#: their forward computes the gate as `out_vals[r][i] *= z_val / (1 +
#: expf(-z_val))` (`selective_scan_fwd_kernel.cuh:298`) and their backward
#: recomputes the same quantity as `z_sigmoid_val = 1.0f/(1.0f+expf(-z_val));
#: z_silu_vals[i] = z_val * z_sigmoid_val;` (`selective_scan_bwd_kernel.cuh:
#: 193-194`), so their gradient multiplies by a `silu(z)` their own forward
#: never produced. The forward side of this repository has MEASURED the same
#: difference: `SAB_S12_MUL_SIGMOID` bit `gate.out` on 15 of 64 cells
#: (IDENTITY_PATHS row 55). INERT at `z == 0`.
comptime SAB_BWD_S12_MUL_SIGMOID = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S12_MUL_SIGMOID"
]()

#: B7's other association, plan section 3.5: `sig + z*sig*(1-sig)` instead of
#: `sig * (1 + z*(1 - sig))`. Five operations against four, equal in the reals.
#: The transformer lane carries the same arm as `B20_SILU_DERIV_ALT_ASSOC`
#: and it has never been shown able to fail there either.
comptime SAB_BWD_SILU_DERIV_ALT_ASSOC = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_SILU_DERIV_ALT_ASSOC"
]()

#: B8's other association, plan section 3.5: `dg * (sk * silu')` instead of
#: `(dg * sk) * silu'`. INERT wherever any of the three factors is exactly
#: representable against the others, which a hashed fixture will not arrange.
comptime SAB_BWD_S12B_ASSOC = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S12B_ASSOC"
]()

#: B22 spelled upstream's way, ONE division `ddelta / (1 + expf(-biased))`
#: (`selective_scan_bwd_kernel.cuh:454-457`), instead of a MULTIPLY by
#: `identical_sigmoid`. Two roundings against one. DEVIATION 1078.
#: **THE DISTINGUISHING BAND IS `biased` IN ROUGHLY [8, 14]**, and a fixture
#: that only straddles the guard at 20 passes this VACUOUSLY. That has
#: already bitten this lane once.
comptime SAB_BWD_S14_DIVISION = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S14_DIVISION"
]()

#: T6's three way join permuted to x_proj, scan, D-skip. **PREDICTED BITWISE
#: INERT on any fixture where one contribution dominates the other two**, so
#: the gate's fixture must plant the three at comparable magnitude and must
#: raise VACUOUS rather than pass if it cannot.
comptime SAB_BWD_JOIN_ORDER = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_JOIN_ORDER"
]()

#: B31's correlation index reversed -- the tap dropped at the END of the
#: backward sequence dropped at the START instead. **BIT IDENTICAL ON A
#: SYMMETRIC FIXTURE AND WRONG ON EVERY OTHER**, which is why gate MB2 must
#: DEMONSTRATE the requirement: run this arm on a `k`-symmetric conv weight,
#: show it PASSES, and print that as the reason the asymmetric fixture is
#: mandatory.
comptime SAB_BWD_S13_TAPS_REVERSED = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S13_TAPS_REVERSED"
]()

#: B40's `j` fold run DESCENDING. The forward's own `SAB_S1_FOLD_DESCENDING`
#: is the same mistake on the same axis one seam earlier. INERT at
#: `d_model == 1`, which this profile never runs.
comptime SAB_BWD_S1B_FOLD_DESCENDING = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S1B_FOLD_DESCENDING"
]()

#: T7's `rstd^3` as `identical_pow(rstd, 3.0)`. **THIS IS NOT THE ARM THE
#: PLAN NAMED**; see DEVIATION 1086 in the header for why the plan's
#: "right associated" arm is vacuous by construction and this one replaces it.
#: `identical_pow` is a different function with a different rounding and T7
#: forbids it by name.
comptime SAB_BWD_RSTD3_POW = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_RSTD3_POW"
]()

#: B38's alternative, plan section 3.5: recover `inner` by dividing the
#: recorded `norm.out` by `w_norm[j]` instead of recomputing S3's product.
#: A division where the profile has a product, and `0/0` wherever a weight is
#: zero. INERT wherever `w_norm[j]` is exactly a power of two.
comptime SAB_BWD_INNER_FROM_NRM = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_INNER_FROM_NRM"
]()

#: TRUE when `silu(z)`'s RECOMPUTE takes the `z * sigmoid(z)` spelling, for
#: either of two different reasons. Named once, at module scope, so the two
#: reasons are visible together and so no `comptime if` has to carry an `or`.
comptime SILU_RECOMPUTE_IS_MUL_SIGMOID = (
    SAB_S12_MUL_SIGMOID or SAB_BWD_S12_MUL_SIGMOID
)


comptime ANY_BWD_BLOCK_SABOTAGE = (
    SAB_BWD_S12_MUL_SIGMOID
    or SAB_BWD_SILU_DERIV_ALT_ASSOC
    or SAB_BWD_S12B_ASSOC
    or SAB_BWD_S14_DIVISION
    or SAB_BWD_JOIN_ORDER
    or SAB_BWD_S13_TAPS_REVERSED
    or SAB_BWD_S1B_FOLD_DESCENDING
    or SAB_BWD_RSTD3_POW
    or SAB_BWD_INNER_FROM_NRM
)


def mamba_backward_block_sabotage_name() -> String:
    """Which arm this binary compiled with, for a gate's banner. A backward
    binary can carry FORWARD, SCAN, GEMM and ROUTING sabotages as well, and a
    banner naming only one of five mislabels the run."""
    comptime if SAB_BWD_S12_MUL_SIGMOID:
        return String("BWD_S12_MUL_SIGMOID")
    comptime if SAB_BWD_SILU_DERIV_ALT_ASSOC:
        return String("BWD_SILU_DERIV_ALT_ASSOC")
    comptime if SAB_BWD_S12B_ASSOC:
        return String("BWD_S12B_ASSOC")
    comptime if SAB_BWD_S14_DIVISION:
        return String("BWD_S14_DIVISION")
    comptime if SAB_BWD_JOIN_ORDER:
        return String("BWD_JOIN_ORDER")
    comptime if SAB_BWD_S13_TAPS_REVERSED:
        return String("BWD_S13_TAPS_REVERSED")
    comptime if SAB_BWD_S1B_FOLD_DESCENDING:
        return String("BWD_S1B_FOLD_DESCENDING")
    comptime if SAB_BWD_RSTD3_POW:
        return String("BWD_RSTD3_POW")
    comptime if SAB_BWD_INNER_FROM_NRM:
        return String("BWD_INNER_FROM_NRM")
    return String("none")


# ===========================================================================
# THE SiLU DERIVATIVE. DEVIATION 1085.
# ===========================================================================


def _silu_prime(x: Float32) -> Float32:
    """`silu'(x) = sig(x) * (1 + x*(1 - sig(x)))`, left to right.

    TRANSCRIBED, NOT DERIVED, from
    `transformer/checks/transformer_backward.mojo:713`
    (`bwd_silu_backward_kernel`, DEVIATION 1411), which already writes this
    exact chain under this repository's pins. Four roundings after the
    sigmoid, each intermediate flushed and STORED so no backend can reach
    inside the expression:

        sg = identical_sigmoid(x)      DEVIATION 743, portable_sigmoidf
        r1 = ftz(1.0 - sg)             SUBTRACT
        r2 = pinned_mul(x, r1)         PRODUCT
        r3 = ftz(1.0 + r2)             UNFUSED ADD
        r4 = pinned_mul(sg, r3)        PRODUCT

    **`sg` IS COMPUTED FROM `x`, NEVER RECONSTRUCTED FROM `silu(x)`.**
    `silu(x) = x * sigmoid(x)` is true in the reals and FALSE in Float32
    under this profile: seam S12's forward is ONE division
    (`identical_silu` is `portable_siluf`, `x / (1 + expf(-x))`), so
    `silu(x)/x` is a different number and is `0/0` at `x = 0`.
    `portable_sigmoidf` and `portable_siluf` share the same
    `d = portable_expf(-x) + 1.0` and differ only in the numerator, so the
    recomputed sigmoid is exactly `1/d` against the forward's `x/d`.

    **NEITHER REFERENCE SPELLING HAS BEEN READ BY THIS FILE, AND THAT IS A
    STATED GAP RATHER THAN A DECISION ON EVIDENCE.** There is no PyTorch
    checkout in `/Users/andrewhendel/CascadeProjects/upstream/`, so ATen's
    `silu_backward` is unread; and `selective_scan_bwd_kernel.cuh` was read
    by `mamba/IDENTICAL_BACKWARD_PLAN.md` section 6 and not by this file, so
    every upstream claim here is that document's reading repeated, never a
    line quoted from a file this author opened. **What section 6 records
    about upstream's SiLU is one thing only**: their backward recomputes
    `silu(z)` as a reciprocal-then-product at `:193-194` where their forward
    wrote a single quotient at `fwd_kernel.cuh:298`. It records nothing about
    how they associate the derivative's own chain, so nothing is claimed
    about it here.
    """
    var sg = ftz(identical_sigmoid(x))
    var r1 = ftz(ftz(Float32(1.0)) - ftz(sg))
    comptime if SAB_BWD_SILU_DERIV_ALT_ASSOC:
        # SABOTAGE: `sig + x*sig*(1-sig)`. Equal in the reals, different in
        # the last bit, five operations instead of four.
        var p1 = ftz(pinned_mul(x, sg))
        var p2 = ftz(pinned_mul(p1, r1))
        return ftz(ftz(sg) + ftz(p2))
    var r2 = ftz(pinned_mul(x, r1))
    var r3 = ftz(ftz(Float32(1.0)) + ftz(r2))
    return ftz(pinned_mul(sg, r3))


def _silu_recomputed(z: Float32) -> Float32:
    """`silu(z)` as the FORWARD computed it, for B4.

    The profile's answer is `identical_silu(z)`, seam S12's own call
    (`modeling_mamba.mojo:999`). **A recomputed forward quantity must be
    spelled with the same function the forward used, not with an
    algebraically equal one**, and that rule is the general form of the one
    thing upstream gets wrong in its own backward.

    Two different arms take the other branch and they mean different things.
    `SAB_S12_MUL_SIGMOID` is the FORWARD's arm: when it is armed the forward
    genuinely computed `z * sigmoid(z)`, so following it here keeps the
    backward consistent with the function that ran. `SAB_BWD_S12_MUL_SIGMOID`
    is THIS lane's arm: it makes the backward inconsistent with an unmodified
    forward, which is exactly upstream's defect, on purpose. The body is
    `modeling_mamba.mojo:991-998` verbatim.
    """
    comptime if SILU_RECOMPUTE_IS_MUL_SIGMOID:
        var s = ftz(
            identical_div(
                Float32(1.0), ftz(Float32(1.0) + ftz(identical_exp(-z)))
            )
        )
        return ftz(pinned_mul(z, s))
    return ftz(identical_silu(z))


# ===========================================================================
# S12's BACKWARD: B4, B5, B7, B8, B10, B11.
# ===========================================================================


def mamba_bwd_gate_kernel(
    dsk_ptr: MutPointer[Float32, MutAnyOrigin],
    dz_ptr: MutPointer[Float32, MutAnyOrigin],
    du_d_ptr: MutPointer[Float32, MutAnyOrigin],
    pd_ptr: MutPointer[Float32, MutAnyOrigin],
    dg_ptr: MutPointer[Float32, MutAnyOrigin],
    sk_ptr: MutPointer[Float32, MutAnyOrigin],
    u_ptr: MutPointer[Float32, MutAnyOrigin],
    in_proj_ptr: MutPointer[Float32, MutAnyOrigin],
    d_skip_ptr: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    di_in: Int32,
):
    """S12 is `gate.out = pinned_mul(skip.out, silu(z))`, so its backward is
    two products, and S11's `out = y + u*D` adds two more. One thread per
    `[M, d_inner]` cell, four outputs.

        silu_z = identical_silu(z)                       B4, S12's own call
        dsk    = ftz(pinned_mul(dg, silu_z))             B5
        dz     = ftz(pinned_mul(ftz(pinned_mul(dg, sk)), silu'(z)))   B8
        du_D   = ftz(pinned_mul(dsk, D[d]))              B10
        pD     = ftz(pinned_mul(dsk, u))                 B11, a PRE-PRODUCT

    `dy = dsk` IS A COPY (B9) and gets no buffer and no instruction: the scan
    backward takes `dsk` as its `dy`.

    **B8's FOUR FACTORS ARE ASSOCIATED `((dg * sk) * silu')`, AND THE PLAN
    CALLS THE ALTERNATIVE "upstream's four factor chain".** Pinned this way
    so that `silu'` is ONE function with ONE spelling shared with B30 and
    with the transformer lane, and so that the two incoming factors are
    multiplied together first -- which is the order S12's forward data flow
    reads in. `SAB_BWD_S12B_ASSOC` is `dg * (sk * silu')`, plan section 3.5's
    listed alternative. **The CUDA backward's own association has NOT been
    read by this file** (see `_silu_prime`), so no line is cited for it.

    **B10 IS ITS OWN PRODUCT AND IS NOT FUSED INTO T6's JOIN.** Plan section
    3.5 lists that fusion as the wrong answer for B10: `du` is a three-term
    sum of three separately rounded contributions, and folding `dsk * D` into
    the first add would make it two roundings where the profile has three.

    `z` is the SECOND half of `in_proj.out`, read at column offset `d_inner`
    (DEVIATION 730's addressing, `modeling_mamba.mojo:989`), never copied out.
    """
    var m = Int(m_in)
    var di = Int(di_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= m * di:
        return
    var t = i // di
    var d = i - t * di

    var z = ftz(in_proj_ptr.unsafe_load(t * 2 * di + di + d))
    var dgv = ftz(dg_ptr.unsafe_load(i))

    # B4 then B5.
    var silu_z = _silu_recomputed(z)
    var dskv = ftz(pinned_mul(dgv, silu_z))
    dsk_ptr.unsafe_store(i, dskv)

    # B6 and B7 inside `_silu_prime`, then B8.
    var sp = _silu_prime(z)
    var skv = ftz(sk_ptr.unsafe_load(i))
    comptime if SAB_BWD_S12B_ASSOC:
        # SABOTAGE: `dg * (sk * silu')`.
        var q = ftz(pinned_mul(skv, sp))
        dz_ptr.unsafe_store(i, ftz(pinned_mul(dgv, q)))
    else:
        var p1 = ftz(pinned_mul(dgv, skv))
        dz_ptr.unsafe_store(i, ftz(pinned_mul(p1, sp)))

    # B10 and B11.
    du_d_ptr.unsafe_store(
        i, ftz(pinned_mul(dskv, ftz(d_skip_ptr.unsafe_load(d))))
    )
    pd_ptr.unsafe_store(
        i, ftz(pinned_mul(dskv, ftz(u_ptr.unsafe_load(i))))
    )


def mamba_bwd_gate_into(
    ctx: DeviceContext,
    mut dsk: DeviceBuffer[DType.float32],
    mut dz: DeviceBuffer[DType.float32],
    mut du_D: DeviceBuffer[DType.float32],
    mut pD: DeviceBuffer[DType.float32],
    mut dg: DeviceBuffer[DType.float32],
    mut skip_out: DeviceBuffer[DType.float32],
    mut silu_out: DeviceBuffer[DType.float32],
    mut in_proj_out: DeviceBuffer[DType.float32],
    mut d_skip: DeviceBuffer[DType.float32],
    m: Int,
    d_inner: Int,
) raises:
    """B4, B5, B7, B8, B10, B11 in one launch. ASYNCHRONOUS.

    `silu_out` is the forward stage `silu.out`, which is the scan's `u`.
    `skip_out` is `skip.out`, S11's output. `dg` is the activation gradient
    the out_proj `dA` routing produced.
    """
    var n = m * d_inner
    _require(len(dsk), n, "dsk", "[M, di]")
    _require(len(dz), n, "dz", "[M, di]")
    _require(len(du_D), n, "du_D", "[M, di]")
    _require(len(pD), n, "pD", "[M, di]")
    _require(len(dg), n, "dg", "[M, di]")
    _require(len(skip_out), n, "skip.out", "[M, di]")
    _require(len(silu_out), n, "silu.out", "[M, di]")
    _require(len(in_proj_out), m * 2 * d_inner, "in_proj.out", "[M, 2di]")
    _require(len(d_skip), d_inner, "D", "[di]")
    if n < 1:
        return
    ctx.enqueue_function[mamba_bwd_gate_kernel](
        dsk.unsafe_ptr(),
        dz.unsafe_ptr(),
        du_D.unsafe_ptr(),
        pD.unsafe_ptr(),
        dg.unsafe_ptr(),
        skip_out.unsafe_ptr(),
        silu_out.unsafe_ptr(),
        in_proj_out.unsafe_ptr(),
        d_skip.unsafe_ptr(),
        Int32(m),
        Int32(d_inner),
        grid_dim=(_grid(n), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )


# ===========================================================================
# S14's BACKWARD: B22, THE SOFTPLUS DERIVATIVE. DEVIATION 1078.
# ===========================================================================


def mamba_bwd_ddtp_kernel(
    ddtp_ptr: MutPointer[Float32, MutAnyOrigin],
    ddelta_ptr: MutPointer[Float32, MutAnyOrigin],
    dt_proj_ptr: MutPointer[Float32, MutAnyOrigin],
    b_dt_ptr: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    di_in: Int32,
):
    """`ddtp = ddelta * softplus'(biased)`, one thread per cell.

    `softplus'(x) = sigmoid(x)`, and the guard is the forward's own:
    `identical_softplus` is `x <= 20 ? log1p(exp(x)) : x`, so above the
    threshold the derivative is EXACTLY `1.0` and `ddtp` is `ddelta`
    unchanged -- not multiplied by a `sigmoid` that rounds to one.

    **THE OPERAND IS THE PRE-SOFTPLUS BIASED VALUE, RECOMPUTED, NOT
    INVERTED.** `biased = ftz(ftz(dt_proj.out) + ftz(b_dt[d]))` is S14's own
    spelling (`modeling_mamba.mojo:951`), read off the recorded `dt_proj.out`
    stage and the bias. Recovering it from `softplus.out` by inverting the
    softplus would be a second transcendental and a different number, and
    upstream does not do it either: it reloads and re-biases (`bwd_kernel.
    cuh:454-457`).

    **THE GUARD TRAVELS WITH THE FORWARD.** `SAB_S14_THRESHOLD_10` is the
    forward's arm, imported: when it is armed the forward's softplus really
    did break at 10, and a derivative that broke at 20 would be the
    derivative of a function nobody computed.

    DEVIATION 1078: the PINNED spelling is a MULTIPLY by `identical_sigmoid`,
    which is `portable_divf(1, 1+e)` then a product, two roundings. Upstream
    spells ONE DIVISION, `ddelta / (1 + expf(-biased))`.
    `SAB_BWD_S14_DIVISION` is theirs, and **the distinguishing band is
    `biased` in roughly [8, 14]**.
    """
    var m = Int(m_in)
    var di = Int(di_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= m * di:
        return
    var d = i - (i // di) * di

    var biased = ftz(
        ftz(dt_proj_ptr.unsafe_load(i)) + ftz(b_dt_ptr.unsafe_load(d))
    )
    var dv = ftz(ddelta_ptr.unsafe_load(i))

    var thr = Float32(20.0)
    comptime if SAB_S14_THRESHOLD_10:
        thr = Float32(10.0)

    if biased <= thr:
        comptime if SAB_BWD_S14_DIVISION:
            # SABOTAGE: upstream's single division, bwd_kernel.cuh:454-457.
            ddtp_ptr.unsafe_store(
                i,
                ftz(
                    identical_div(
                        dv,
                        ftz(Float32(1.0) + ftz(identical_exp(-biased))),
                    )
                ),
            )
        else:
            ddtp_ptr.unsafe_store(
                i, ftz(pinned_mul(dv, ftz(identical_sigmoid(biased))))
            )
    else:
        # `softplus(x) = x` above the guard, so the derivative is exactly
        # 1.0 and this is a COPY, not a product by a rounded sigmoid.
        ddtp_ptr.unsafe_store(i, dv)


def mamba_bwd_ddtp_into(
    ctx: DeviceContext,
    mut ddtp: DeviceBuffer[DType.float32],
    mut ddelta: DeviceBuffer[DType.float32],
    mut dt_proj_out: DeviceBuffer[DType.float32],
    mut b_dt: DeviceBuffer[DType.float32],
    m: Int,
    d_inner: Int,
) raises:
    """B22. ASYNCHRONOUS."""
    var n = m * d_inner
    _require(len(ddtp), n, "ddtp", "[M, di]")
    _require(len(ddelta), n, "ddelta", "[M, di]")
    _require(len(dt_proj_out), n, "dt_proj.out", "[M, di]")
    _require(len(b_dt), d_inner, "dt_proj.bias", "[di]")
    if n < 1:
        return
    ctx.enqueue_function[mamba_bwd_ddtp_kernel](
        ddtp.unsafe_ptr(),
        ddelta.unsafe_ptr(),
        dt_proj_out.unsafe_ptr(),
        b_dt.unsafe_ptr(),
        Int32(m),
        Int32(d_inner),
        grid_dim=(_grid(n), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )


# ===========================================================================
# T6 AND B30: THE THREE WAY JOIN AT `du`, THEN THE CONV's SiLU.
# DEVIATION 1075.
# ===========================================================================


def mamba_bwd_du_join_kernel(
    du_ptr: MutPointer[Float32, MutAnyOrigin],
    dconv_ptr: MutPointer[Float32, MutAnyOrigin],
    du_d_ptr: MutPointer[Float32, MutAnyOrigin],
    du_s_ptr: MutPointer[Float32, MutAnyOrigin],
    du_x_ptr: MutPointer[Float32, MutAnyOrigin],
    conv_ptr: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """T6 then B30, one thread per `[M, d_inner]` cell.

        du    = ftz(ftz(ftz(du_D) + du_s) + du_x)      THREE flushed adds
        dconv = ftz(pinned_mul(du, silu'(conv.out)))

    **THE ORDER IS D-SKIP, THEN SCAN, THEN X_PROJ, AND IT IS STATED ONCE AND
    NEVER A SCHEDULER'S CHOICE.** It follows the forward's own data flow:
    `u` (the stage `silu.out`) is consumed by S11's `D * u`, then by the scan,
    then by S17b's x_proj. Any permutation is a different number.
    `SAB_BWD_JOIN_ORDER` permutes it and is **PREDICTED BITWISE INERT on any
    fixture where one contribution dominates the other two.**

    **THERE IS NO `+0.0` SEED.** Three terms, two adds, left associated, the
    first term installed rather than accumulated. `+0.0 + x` is `x` for
    every value except `x = -0.0`, so a seed would launder a negative-zero
    D-skip contribution -- IDENTITY_PATHS row 39 -- and a negative zero is
    reachable here, because `du_D = dsk * D[d]` is a `pinned_mul` and
    DEVIATION 720's whole point is that it preserves one.

    B30's `silu'` is `_silu_prime`, the SAME function B7 calls, evaluated on
    `conv.out` rather than on `z`. Plan section 3.5 lists the alternative as
    "`identical_div` of the silu quotient form" and it is not taken: see
    `_silu_prime`'s note on why `silu(x)/x` is not `sigmoid(x)` here.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return

    var vd = ftz(du_d_ptr.unsafe_load(i))
    var vs = ftz(du_s_ptr.unsafe_load(i))
    var vx = ftz(du_x_ptr.unsafe_load(i))

    var duv: Float32
    comptime if SAB_BWD_JOIN_ORDER:
        # SABOTAGE: x_proj, scan, D-skip.
        var s1s = ftz(ftz(vx) + ftz(vs))
        duv = ftz(ftz(s1s) + ftz(vd))
    else:
        var s1 = ftz(ftz(vd) + ftz(vs))
        duv = ftz(ftz(s1) + ftz(vx))
    du_ptr.unsafe_store(i, duv)

    var sp = _silu_prime(ftz(conv_ptr.unsafe_load(i)))
    dconv_ptr.unsafe_store(i, ftz(pinned_mul(duv, sp)))


def mamba_bwd_du_join_into(
    ctx: DeviceContext,
    mut du: DeviceBuffer[DType.float32],
    mut dconv: DeviceBuffer[DType.float32],
    mut du_D: DeviceBuffer[DType.float32],
    mut du_s: DeviceBuffer[DType.float32],
    mut du_x: DeviceBuffer[DType.float32],
    mut conv_out: DeviceBuffer[DType.float32],
    m: Int,
    d_inner: Int,
) raises:
    """T6 and B30. ASYNCHRONOUS.

    `du_x` is the activation gradient the x_proj `dA` routing produced,
    which is `d(silu.out)` through the projection. All three contributions
    must be FINISHED before this launch; on one `DeviceContext` MAX enforces
    that by ordering, and on two contexts nothing does.
    """
    var n = m * d_inner
    _require(len(du), n, "du", "[M, di]")
    _require(len(dconv), n, "dconv", "[M, di]")
    _require(len(du_D), n, "du_D", "[M, di]")
    _require(len(du_s), n, "du_s", "[M, di]")
    _require(len(du_x), n, "du_x", "[M, di]")
    _require(len(conv_out), n, "conv.out", "[M, di]")
    if n < 1:
        return
    ctx.enqueue_function[mamba_bwd_du_join_kernel](
        du.unsafe_ptr(),
        dconv.unsafe_ptr(),
        du_D.unsafe_ptr(),
        du_s.unsafe_ptr(),
        du_x.unsafe_ptr(),
        conv_out.unsafe_ptr(),
        Int32(n),
        grid_dim=(_grid(n), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )


# ===========================================================================
# S13's BACKWARD: B31, THE FOUR TAP CORRELATION.
# ===========================================================================
# Forward, `conv[b,l,d]` reads `hin[b, l-3+k, d]` for `k` in `[0, 4)`, so
# `hin[b,p,d]` is read by output position `l = p + 3 - k`, in range only for
# `k` with `0 <= p + 3 - k < L`. Since `p >= 0` and `k <= 3` the lower bound
# is never binding, so the ONLY drop is at the END of the backward sequence.
# **THE BACKWARD IS A CORRELATION IN THE OPPOSITE DIRECTION AND THE TAP
# DROPPED AT THE START OF THE FORWARD SEQUENCE IS DROPPED AT THE END OF THE
# BACKWARD ONE.** A reversed-tap implementation is bit identical on a
# `k`-symmetric fixture and wrong on every other, which is why
# `SAB_BWD_S13_TAPS_REVERSED` exists and why gate MB2 must demonstrate the
# fixture requirement instead of asserting it.
#
# `causal_conv1d` IS NOT ON DISK. It is a separate repository
# (Dao-AILab/causal-conv1d) and is not checked out, so nothing about
# `causal_conv1d_bwd_function`'s rounding order, its bias-gradient fold or its
# tap order has been READ. Everything here is derived from forward seam S13
# and from the index arithmetic above and is marked INFERRED.


def mamba_bwd_dhin_kernel(
    dhin_ptr: MutPointer[Float32, MutAnyOrigin],
    dconv_ptr: MutPointer[Float32, MutAnyOrigin],
    conv_w_ptr: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    di_in: Int32,
):
    """`dhin[b,p,d] = sum_k cw[d,k] * dconv[b, p+3-k, d]`, taps `k` ASCENDING.

    S13's TAP ORDER INHERITED, its INDEX REVERSED. The forward walks
    `k = 0..3` oldest first (`modeling_mamba.mojo:794-806`, MAX
    `causal_conv1d.mojo:190-205`, and the CUDA `causal_conv1d` kernel), and
    this walks the same `k` ascending; only the position it reads moves the
    other way.

    **THE SEED IS `+0.0` AND THAT IS A CHOICE, NOT AN INHERITANCE.** The
    forward's chain is BIAS-SEEDED (DEVIATION 721), and the backward has no
    bias to seed with: `d(conv1d.bias)` is a separate token-axis reduction
    (B33) and contributes nothing to `dhin`. The plan is silent, so `+0.0` is
    pinned here and stated as a choice. The consequence is the usual one:
    where every in-range tap contributes a negative zero, `dhin` is `+0.0`.

    **AN OUT-OF-RANGE TAP IS A STRUCTURAL OMISSION, NEVER A PRODUCT BY
    ZERO.** The forward's out-of-range positions read the conv WINDOW, which
    holds real values on decode and zeros on prefill, so the forward always
    performs four `fma`s. The backward's out-of-range positions have no
    window analogue -- `dconv[b, l, d]` for `l >= L` is a gradient of an
    output that does not exist -- so the term is not formed at all. Padding
    it with `0.0 * cw` would be a different answer at a `-0.0` weight and is
    what gemm contract section 8 forbids for the same reason.

    THE CROSS-CALL TAP IS DROPPED. Plan section 2.4: `dhin` at a pre-sequence
    position would flow into the PREVIOUS call's output, and this lane drops
    it exactly as it drops `dh[L]`. **Nobody should read "the backward is
    correct" as "the backward does truncated BPTT correctly"**, because it
    does no BPTT at all.
    """
    var b = Int(b_in)
    var l = Int(l_in)
    var di = Int(di_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * l * di:
        return
    var bb = cell // (l * di)
    var rest = cell - bb * (l * di)
    var p = rest // di
    var d = rest - p * di

    var acc = Float32(0.0)
    for kk in range(D_CONV):
        var k = kk
        comptime if SAB_BWD_S13_TAPS_REVERSED:
            # SABOTAGE: the correlation index reversed, so the tap dropped
            # at the end of the backward sequence is dropped at the start.
            # BIT IDENTICAL on a conv weight symmetric in `k`.
            k = D_CONV - 1 - kk
        var lpos = p + (D_CONV - 1) - k
        if lpos < l:
            acc = ftz(
                identical_mul_add(
                    ftz(conv_w_ptr.unsafe_load(d * D_CONV + k)),
                    ftz(dconv_ptr.unsafe_load((bb * l + lpos) * di + d)),
                    acc,
                )
            )
    dhin_ptr.unsafe_store((bb * l + p) * di + d, acc)


def mamba_bwd_dhin_into(
    ctx: DeviceContext,
    mut dhin: DeviceBuffer[DType.float32],
    mut dconv: DeviceBuffer[DType.float32],
    mut conv_w: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    d_inner: Int,
) raises:
    """B31. `dhin [M, d_inner]`. ASYNCHRONOUS."""
    var n = b * l * d_inner
    _require(len(dhin), n, "dhin", "[M, di]")
    _require(len(dconv), n, "dconv", "[M, di]")
    _require(len(conv_w), d_inner * D_CONV, "conv1d.weight", "[di, 4]")
    if n < 1:
        return
    ctx.enqueue_function[mamba_bwd_dhin_kernel](
        dhin.unsafe_ptr(),
        dconv.unsafe_ptr(),
        conv_w.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(d_inner),
        grid_dim=(_grid(n), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )


def mamba_bwd_conv_tap_product_kernel(
    p_ptr: MutPointer[Float32, MutAnyOrigin],
    dconv_ptr: MutPointer[Float32, MutAnyOrigin],
    in_proj_ptr: MutPointer[Float32, MutAnyOrigin],
    win_ptr: MutPointer[Float32, MutAnyOrigin],
    b_in: Int32,
    l_in: Int32,
    di_in: Int32,
    k_in: Int32,
):
    """B32's pre-product for ONE tap: `p[t,d] = dconv[t,d] * hin[b,l-3+k,d]`.

    `d(conv1d.weight)[d,k] = sum over tokens of dconv * hin_shifted`, which
    is a Hadamard product and then a token-axis reduction. The reduction is
    `mamba/checks/mamba_backward.mojo::mamba_backward_reduce_into` at
    `RED_CONV_W_TAP<k>`, a ones-vector `OP_NN` v1 GEMM at `(1, d_inner, M)`;
    the PRODUCT is arithmetic and is here, because a routing layer with a
    float multiply in it is not a routing layer.

    **THE SHIFTED READ TAKES A PRE-SEQUENCE POSITION FROM THE CONV WINDOW,
    EXACTLY AS FORWARD SEAM S13 DOES** (`modeling_mamba.mojo:798-803`). It
    is not zero-padded and it is not dropped: on prefill the window is zeros
    and IS `F.conv1d`'s `padding = 3`, and on decode it holds the previous
    call's last inputs, so the same expression serves both paths. `win` is
    the OLD window, the one the forward READ, never the one it wrote.

    `hin` is column `d` of the HIDDEN half of `in_proj.out`, read at column
    offset 0 with row stride `2 * d_inner` (DEVIATION 730), never copied out.

    THIS COSTS TWO ROUNDINGS PER TERM WHERE A HAND-WRITTEN
    `fma(dconv, hin, acc)` WOULD COST ONE. Bought deliberately and recorded
    rather than hidden: it is the same trade the transformer lane's
    DEVIATION 1410 states for `dW_norm`, and it buys one arithmetic under one
    certificate instead of a fifth hand-declared fold.
    """
    var b = Int(b_in)
    var l = Int(l_in)
    var di = Int(di_in)
    var k = Int(k_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= b * l * di:
        return
    var bb = cell // (l * di)
    var rest = cell - bb * (l * di)
    var li = rest // di
    var d = rest - li * di

    var pidx = li - (D_CONV - 1) + k
    var xv: Float32
    if pidx >= 0:
        xv = in_proj_ptr.unsafe_load((bb * l + pidx) * 2 * di + d)
    else:
        xv = win_ptr.unsafe_load((bb * di + d) * D_CONV + (D_CONV + pidx))

    p_ptr.unsafe_store(
        (bb * l + li) * di + d,
        ftz(
            pinned_mul(
                ftz(dconv_ptr.unsafe_load((bb * l + li) * di + d)), ftz(xv)
            )
        ),
    )


def mamba_bwd_conv_tap_product_into(
    ctx: DeviceContext,
    mut p_out: DeviceBuffer[DType.float32],
    mut dconv: DeviceBuffer[DType.float32],
    mut in_proj_out: DeviceBuffer[DType.float32],
    mut old_window: DeviceBuffer[DType.float32],
    b: Int,
    l: Int,
    d_inner: Int,
    k: Int,
) raises:
    """One of B32's four pre-products. ASYNCHRONOUS.

    **`k` IS THE FORWARD TAP INDEX, NOT A SLOT.** The slot the reduced vector
    is scattered into is `mamba_backward_reduce_into`'s business and
    `SAB_BWD_CONV_TAP_SLOT` is the arm that breaks it; this argument selects
    which SHIFTED READ forms the product and a wrong value here is a wrong
    number rather than a misplaced one.
    """
    if k < 0 or k >= D_CONV:
        raise Error(
            "mamba_bwd_conv_tap_product_into: tap k must be in [0, "
            + String(D_CONV)
            + "), got "
            + String(k)
        )
    var n = b * l * d_inner
    _require(len(p_out), n, "p_out", "[M, di]")
    _require(len(dconv), n, "dconv", "[M, di]")
    _require(len(in_proj_out), n * 2, "in_proj.out", "[M, 2di]")
    _require(
        len(old_window), b * d_inner * D_CONV, "conv.window", "[B, di, 4]"
    )
    if n < 1:
        return
    ctx.enqueue_function[mamba_bwd_conv_tap_product_kernel](
        p_out.unsafe_ptr(),
        dconv.unsafe_ptr(),
        in_proj_out.unsafe_ptr(),
        old_window.unsafe_ptr(),
        Int32(b),
        Int32(l),
        Int32(d_inner),
        Int32(k),
        grid_dim=(_grid(n), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )


# ===========================================================================
# B26 AND B34: THE TWO CONCATENATIONS. COPIES, NOT SEAMS.
# ===========================================================================
# The forward materializes the x_proj split (DEVIATION 728) because gemm v1
# takes only contiguous row-major operands; the backward materializes the
# JOIN for the same reason. A copy moves bits untouched and neither kernel
# below contains an arithmetic operation.


def mamba_bwd_concat_xp_kernel(
    dxp_ptr: MutPointer[Float32, MutAnyOrigin],
    ddtl_ptr: MutPointer[Float32, MutAnyOrigin],
    dbm_ptr: MutPointer[Float32, MutAnyOrigin],
    dcm_ptr: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    r_in: Int32,
    xr_in: Int32,
):
    """`dXP[t] = concat(ddtl[t] | dBm[t] | dCm[t])` at `[M, r + 2*D_STATE]`.

    THE INVERSE OF `split_x_proj_kernel` (`modeling_mamba.mojo:899`),
    column for column: `[0, r)` is dt, `[r, r+16)` is B, `[r+16, r+32)` is C.
    One thread per token row, so the three writes for a row are one thread's
    and a partial concatenation cannot be observed by the gemm that reads it.
    """
    var m = Int(m_in)
    var r = Int(r_in)
    var xr = Int(xr_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m:
        return
    for j in range(r):
        dxp_ptr.unsafe_store(t * xr + j, ddtl_ptr.unsafe_load(t * r + j))
    for n in range(D_STATE):
        dxp_ptr.unsafe_store(
            t * xr + r + n, dbm_ptr.unsafe_load(t * D_STATE + n)
        )
        dxp_ptr.unsafe_store(
            t * xr + r + D_STATE + n, dcm_ptr.unsafe_load(t * D_STATE + n)
        )


def mamba_bwd_concat_p_kernel(
    dp_ptr: MutPointer[Float32, MutAnyOrigin],
    dhin_ptr: MutPointer[Float32, MutAnyOrigin],
    dz_ptr: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    di_in: Int32,
):
    """`dP[t] = concat(dhin[t] | dz[t])` at `[M, 2*d_inner]`.

    The inverse of `chunk(2, dim=1)` (MM:396), which the forward never
    materializes: DEVIATION 730 reads the hidden half at column offset 0 and
    `z` at column offset `d_inner` in place. The backward has to materialize
    it, because `dP` is the left operand of in_proj's `dA` and `dB` routings
    and gemm v1 takes no sub-view.
    """
    var m = Int(m_in)
    var di = Int(di_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m:
        return
    for d in range(di):
        dp_ptr.unsafe_store(t * 2 * di + d, dhin_ptr.unsafe_load(t * di + d))
        dp_ptr.unsafe_store(
            t * 2 * di + di + d, dz_ptr.unsafe_load(t * di + d)
        )


def mamba_bwd_concat_xp_into(
    ctx: DeviceContext,
    mut dXP: DeviceBuffer[DType.float32],
    mut ddtl: DeviceBuffer[DType.float32],
    mut dBm: DeviceBuffer[DType.float32],
    mut dCm: DeviceBuffer[DType.float32],
    m: Int,
    dims: MambaDims,
) raises:
    """B26. ASYNCHRONOUS."""
    var r = dims.dt_rank
    var xr = dims.x_proj_rows()
    _require(len(dXP), m * xr, "dXP", "[M, r+2N]")
    _require(len(ddtl), m * r, "ddtl", "[M, r]")
    _require(len(dBm), m * D_STATE, "dBm", "[M, 16]")
    _require(len(dCm), m * D_STATE, "dCm", "[M, 16]")
    if m < 1:
        return
    ctx.enqueue_function[mamba_bwd_concat_xp_kernel](
        dXP.unsafe_ptr(),
        ddtl.unsafe_ptr(),
        dBm.unsafe_ptr(),
        dCm.unsafe_ptr(),
        Int32(m),
        Int32(r),
        Int32(xr),
        grid_dim=(_grid(m), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )


def mamba_bwd_concat_p_into(
    ctx: DeviceContext,
    mut dP: DeviceBuffer[DType.float32],
    mut dhin: DeviceBuffer[DType.float32],
    mut dz: DeviceBuffer[DType.float32],
    m: Int,
    d_inner: Int,
) raises:
    """B34. ASYNCHRONOUS."""
    _require(len(dP), m * 2 * d_inner, "dP", "[M, 2di]")
    _require(len(dhin), m * d_inner, "dhin", "[M, di]")
    _require(len(dz), m * d_inner, "dz", "[M, di]")
    if m < 1 or d_inner < 1:
        return
    ctx.enqueue_function[mamba_bwd_concat_p_kernel](
        dP.unsafe_ptr(),
        dhin.unsafe_ptr(),
        dz.unsafe_ptr(),
        Int32(m),
        Int32(d_inner),
        grid_dim=(_grid(m), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )


# ===========================================================================
# S1-S4's BACKWARD PLUS S16's: B37, B38, B40, T7, T8.
# DEVIATIONS 1076, 1077, 1086, 1087.
# ===========================================================================
# ONE KERNEL, ONE THREAD PER TOKEN ROW, and the reason is contract S1's own:
# "one fold per row, no block fold", so the row's fold never leaves this
# thread's registers and no launch geometry can reorder it. The row's
# coefficients (`rstd`, `drstd`, `c3/dm`) must be finished before any cell of
# `dx` can be written, and a thread that owns the whole row has them.
#
# The derivation, done rather than recalled. With `nrm = w * (x * rstd)`,
# `rstd = (ss/dm + eps)^{-1/2}` and `ss = sum_j x^2`:
#     d rstd / d ss   = -0.5 * rstd^3 / dm
#     d ss   / d x_j  = 2 * x_j
# so with `dinner_j = dnrm_j * w_j` and `R = sum_j dinner_j * x_j`,
#     dx_norm[t,j] = rstd*dinner_j - (rstd^3 / dm) * x_j * R
# The `-0.5` and the `2` cancel exactly and the profile does not spell them.
# **THE ONLY REDUCTION IS `R`, OVER THE FEATURE AXIS, PER ROW**, the same axis
# and the same length as S1's own fold, which is why B40 is category
# c-inherited and not a new topology.


def mamba_bwd_norm_kernel(
    dx_ptr: MutPointer[Float32, MutAnyOrigin],
    drstd_ptr: MutPointer[Float32, MutAnyOrigin],
    pw_ptr: MutPointer[Float32, MutAnyOrigin],
    dres_ptr: MutPointer[Float32, MutAnyOrigin],
    dnrm_ptr: MutPointer[Float32, MutAnyOrigin],
    x_ptr: MutPointer[Float32, MutAnyOrigin],
    nrm_ptr: MutPointer[Float32, MutAnyOrigin],
    w_ptr: MutPointer[Float32, MutAnyOrigin],
    sumsq_ptr: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    dm_in: Int32,
):
    """B37, B38, B40, T7 and T8 for one token row.

        rstd  = ftz(identical_rsqrt(ftz(identical_div(sumsq, dm) + eps)))
        dinner_j = ftz(pinned_mul(dnrm_j, w_j))                   B37
        drstd = fold over j ASCENDING of fma(dinner_j, x_j, acc)  B40
        c3    = ftz(pinned_mul(ftz(pinned_mul(rstd, rstd)), rstd))
        s     = ftz(identical_div(c3, dm))            HOISTED, once per row
        inner_j = ftz(pinned_mul(x_j, rstd))          a recompute of S3
        pW_j  = ftz(pinned_mul(dnrm_j, inner_j))                  B38
        t2    = ftz(pinned_mul(ftz(pinned_mul(s, x_j)), drstd))
        t1    = ftz(pinned_mul(rstd, dinner_j))
        dx_norm = ftz(ftz(t1) - ftz(t2))              UNFUSED                T7
        dx    = ftz(ftz(dres_j) + ftz(dx_norm))       ONE add, dres LEFT     T8

    **`rstd` IS RECOMPUTED FROM THE RECORDED `norm.sumsq`, NOT RE-FOLDED.**
    `identical_div` and `identical_rsqrt` are pure functions of bits the card
    already holds, so the recompute is bit exact BY CONSTRUCTION, and it is
    S2's own spelling (`modeling_mamba.mojo:689-690`) rather than an
    algebraically equal one. `inner` is likewise S3's own product
    (`modeling_mamba.mojo:695`) and NOT `norm.out / w`, which is a division,
    a different number, and `0/0` at a zero weight.

    **THE SUBTRACTION IS UNFUSED AND EACH OPERAND IS SEPARATELY FLUSHED AND
    STORED.** That is not decoration. `pinned_mul(a,b)` is
    `identical_mul_add(a, b, -0.0)`, which is mathematically `a*b`, so a
    backend is free to simplify it and then re-contract `a*b - c*d` into a
    single fused multiply-add -- ONE rounding where the profile asks for
    three. Writing `ftz(ftz(t1) - ftz(t2))` over two flushed local variables
    is what prevents it. Under FAST `ftz` is the identity, so the guard costs
    nothing there.

    **`c3 / dm` IS HOISTED, AND HOISTING IT MOVES NO BITS** because it is
    loop invariant in `j` and a flushed division of two loop-invariant values
    is the same value every time. `rstd^3` is `(rstd*rstd)*rstd` and NEVER
    `portable_powf`; see DEVIATION 1086 in this file's header for why the
    plan's "right associated" sabotage cannot fire and what replaced it.

    **T8 IS THE ABSORPTION SITE AND THIS KERNEL IS WHY THE GATE IS PER
    STAGE.** IDENTITY_PATHS row 55 records that at the shape where a forward
    sabotage was STRONGEST, thirteen of sixteen intermediate stages moved,
    `out_proj.out` differed on 23 of 64 cells, and `residual.out` was STILL
    BIT IDENTICAL, because the residual add put a value of order 1e-3 beside
    one of order 1 and rounded the difference away. An output-only gate
    called that arm inert. `dx` is the backward's `residual.out`, so **any
    backward gate that compares only `bwd.dx` is blind in exactly the same
    way**, and `bwd.drstd` and `bwd.dnrm` are recorded separately for no
    other reason.
    """
    var m = Int(m_in)
    var dm = Int(dm_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= m:
        return

    # ---- rstd, recomputed from the recorded sumsq (S2's spelling) -----
    var mean = ftz(identical_div(ftz(sumsq_ptr.unsafe_load(t)), Float32(dm)))
    var rstd = ftz(identical_rsqrt(ftz(mean + RMS_EPS)))

    # ---- B37 then B40, contract S1's ascending fused chain ------------
    var c = Float32(0.0)
    comptime if SAB_BWD_S1B_FOLD_DESCENDING:
        # SABOTAGE: the row fold run descending. The forward's own
        # SAB_S1_FOLD_DESCENDING is this mistake one seam earlier.
        for jj in range(dm):
            var jd = dm - 1 - jj
            var dind = ftz(
                pinned_mul(
                    ftz(dnrm_ptr.unsafe_load(t * dm + jd)),
                    ftz(w_ptr.unsafe_load(jd)),
                )
            )
            c = ftz(
                identical_mul_add(
                    dind, ftz(x_ptr.unsafe_load(t * dm + jd)), c
                )
            )
    else:
        for j in range(dm):
            var din = ftz(
                pinned_mul(
                    ftz(dnrm_ptr.unsafe_load(t * dm + j)),
                    ftz(w_ptr.unsafe_load(j)),
                )
            )
            c = ftz(
                identical_mul_add(din, ftz(x_ptr.unsafe_load(t * dm + j)), c)
            )
    drstd_ptr.unsafe_store(t, c)

    # ---- T7's row-invariant coefficient -------------------------------
    var c3: Float32
    comptime if SAB_BWD_RSTD3_POW:
        # SABOTAGE: the spelling T7 forbids by name. A different function
        # with a different rounding, NOT a re-association (which would be
        # bit identical -- DEVIATION 1086).
        c3 = ftz(identical_pow(rstd, Float32(3.0)))
    else:
        var r2 = ftz(pinned_mul(rstd, rstd))
        c3 = ftz(pinned_mul(r2, rstd))
    var s = ftz(identical_div(c3, Float32(dm)))

    # ---- B38, T7 and T8, per cell -------------------------------------
    for j in range(dm):
        var xj = ftz(x_ptr.unsafe_load(t * dm + j))
        var dnv = ftz(dnrm_ptr.unsafe_load(t * dm + j))
        var wj = ftz(w_ptr.unsafe_load(j))
        var dinner = ftz(pinned_mul(dnv, wj))

        var inner: Float32
        comptime if SAB_BWD_INNER_FROM_NRM:
            # SABOTAGE: `inner` recovered by dividing the recorded norm.out
            # by the weight. A division where the profile has a product,
            # and 0/0 at a zero weight. INERT where `w_j` is a power of two.
            inner = ftz(
                identical_div(ftz(nrm_ptr.unsafe_load(t * dm + j)), wj)
            )
        else:
            inner = ftz(pinned_mul(xj, rstd))
        pw_ptr.unsafe_store(t * dm + j, ftz(pinned_mul(dnv, inner)))

        # THE INNER FLUSHES BELOW ARE LOAD BEARING. Each product is formed,
        # flushed and STORED before the subtraction sees it, so no backend
        # can re-contract `a*b - c*d` into one fused multiply-add.
        var sx = ftz(pinned_mul(s, xj))
        var t2 = ftz(pinned_mul(sx, c))
        var t1 = ftz(pinned_mul(rstd, dinner))
        var dxn = ftz(ftz(t1) - ftz(t2))

        dx_ptr.unsafe_store(
            t * dm + j,
            ftz(ftz(dres_ptr.unsafe_load(t * dm + j)) + ftz(dxn)),
        )


def mamba_bwd_norm_into(
    ctx: DeviceContext,
    mut dx: DeviceBuffer[DType.float32],
    mut drstd: DeviceBuffer[DType.float32],
    mut pW: DeviceBuffer[DType.float32],
    mut dres: DeviceBuffer[DType.float32],
    mut dnrm: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut norm_out: DeviceBuffer[DType.float32],
    mut norm_w: DeviceBuffer[DType.float32],
    mut sumsq: DeviceBuffer[DType.float32],
    m: Int,
    d_model: Int,
) raises:
    """B37, B38, B40, T7 and T8. ASYNCHRONOUS.

    `dres` is the incoming gradient at the block's OUTPUT, which S16 sends
    down both legs unchanged (B1's copy): it is the `dO` the out_proj
    routings consume AND the left operand of T8's residual add. One buffer,
    read twice, never copied.

    `norm_out` is the recorded stage `norm.out` and **THE PROFILE NEVER READS
    IT**; it is passed only so `SAB_BWD_INNER_FROM_NRM` has a body. That is
    the same relationship `bwd_silu_backward_kernel`'s `silu_out` argument
    has with `B20_SIGMOID_FROM_SILU` in the transformer lane.
    """
    var n = m * d_model
    _require(len(dx), n, "dx", "[M, dm]")
    _require(len(drstd), m, "drstd", "[M]")
    _require(len(pW), n, "pW", "[M, dm]")
    _require(len(dres), n, "dres", "[M, dm]")
    _require(len(dnrm), n, "dnrm", "[M, dm]")
    _require(len(x), n, "x", "[M, dm]")
    _require(len(norm_out), n, "norm.out", "[M, dm]")
    _require(len(norm_w), d_model, "norm.weight", "[dm]")
    _require(len(sumsq), m, "norm.sumsq", "[M]")
    if m < 1:
        return
    ctx.enqueue_function[mamba_bwd_norm_kernel](
        dx.unsafe_ptr(),
        drstd.unsafe_ptr(),
        pW.unsafe_ptr(),
        dres.unsafe_ptr(),
        dnrm.unsafe_ptr(),
        x.unsafe_ptr(),
        norm_out.unsafe_ptr(),
        norm_w.unsafe_ptr(),
        sumsq.unsafe_ptr(),
        Int32(m),
        Int32(d_model),
        grid_dim=(_grid(m), 1, 1),
        block_dim=(BWD_TPB, 1, 1),
    )


def _require(got: Int, want: Int, name: String, shape: String) raises:
    """A buffer that is the wrong size is a CALLER bug, caught by name here
    rather than by an out-of-bounds device read that returns plausible bits.
    The forward scan file's own helper, same spelling."""
    if got < want:
        raise Error(
            "modeling_mamba_backward: buffer '"
            + name
            + "' holds "
            + String(got)
            + " floats, needs "
            + String(want)
            + " for "
            + shape
        )
