# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The BACKWARD of `selective_scan_ref`, seams S5'-S11', on device.

`mamba_ssm/ops/selective_scan_interface.py::SelectiveScanFn.backward` reaches
`csrc/selective_scan/selective_scan_bwd_kernel.cuh`. This file is the
mojolearn answer to that kernel under the profile
`mojolearn.identical.mamba1.bwd.fp32.v1`, which CONSUMES
`mojolearn.identical.mamba1.fp32.v1` and adds clauses of its own. The forward
contract `mamba/IDENTICAL_MAMBA_CONTRACT.md` is consumed here and NEVER
edited; the plan is `archive/plans/mamba/IDENTICAL_BACKWARD_PLAN.md` and its topology names
T1-T5 are used below without restating their prices.

**THIS FILE OWNS EXACTLY WHAT THE FORWARD SCAN FILE OWNS, MIRRORED.**
`selective_scan_interface.mojo` owns forward seams S5-S11; this file owns
their backward and nothing else. S1'-S4', S12'-S16' and the conv live in
`mamba/impl/transformers/models/mamba/modeling_mamba_backward.mojo`, exactly
as their forward halves live in `modeling_mamba.mojo`. Splitting the backward
the way the forward is split is the whole reason there are two files: a seam
and its derivative should be readable side by side.

WHAT IS HERE, BY THE PLAN'S OPERATION NUMBERS
----------------------------------------------
    B13   T1, the reverse state recurrence           DEVIATION 1070
    B14   T2, `h[t-1]` from an explicit checkpoint   DEVIATION 1071
    B15   T3, `dCm[t,n] = sum_d dy*h`                DEVIATION 1072
    B16   T3, `dBm[t,n] = sum_d w*dh`                DEVIATION 1072
    B17   `du_s[t,d] = sum_n dh*dbb`, S10's chain
    B18   `ddelta`, the two paths INTERLEAVED over n
    B19   T4, the `dA` fold over `t`, DESCENDING     DEVIATION 1073
    B20   T5, the parameter fold over `b`            DEVIATION 1074
    B21   `dA_log = pinned_mul(dA, A)`

NEW DEVIATIONS THIS FILE SPENDS
--------------------------------
**DEVIATION 1081, the checkpoint producer and the four buffer layouts.**
T2 says the forward "stores `h[t,d,n]` for every `t` in `[-1, L)` into a
caller-owned buffer" and does not say WHO writes it. It is NOT written by
`selective_scan_fwd_kernel`, because that kernel is one of three columns of a
CLOSED certificate (`mamba.identical.card` md5 `f072dd22`, 18 records,
byte-identical on Apple M4, NVIDIA and AMD MI325X) and changing its signature
to add an output would put a live three-vendor result at risk to save a
recompute. `selective_scan_checkpoint_kernel` below RE-RUNS the forward
recurrence and stores every state instead. The cost is honest and stated:
**it is a SECOND SPELLING of seams S5 through S9, and a second spelling is a
second thing that can drift.** Its only defense is a gate, and that gate is
cheap: the checkpoint's LAST slot per `(b, d)` must equal the forward's
recorded `scan.h` stage BITWISE, at every shape, and if it does not, one of
the two transcriptions is wrong. That comparison is OWED (`mamba_backward_
check.mojo` does not exist) and until it prints, "the checkpoint is the
forward's `h`" is a prediction. The kernel honors the forward scan file's own
three arms (`SAB_S5_EXP2`, `SAB_S8_CUDA_PAIRING`, `SAB_S9_UNFUSED`) by
IMPORTING them rather than re-deriving them from `is_defined`, so a sabotaged
forward build gets a checkpoint that matches the forward it sabotaged.

**DEVIATION 1082, T1's seed is STRUCTURAL and the plan does not say so.**
T1 reads `dh[t] = fma(da[t+1], dh[t+1], contrib)` with `dh[L] = +0.0`. At
`t = L - 1` there is no `da[L]`: the forward computed `L` values of `da` and
the recurrence asks for an `L + 1`-th. The plan leaves this open and it is not
cosmetic, because the two closures differ on a real input class:

    omitted (PINNED)   dh[L-1] = ftz(contrib)
    seeded             dh[L-1] = ftz(ftz(+0.0) + contrib)

and `(+0.0) + (-0.0)` is `+0.0`, so the seeded form LAUNDERS a negative zero
wherever `dy[L-1,d] * Cm[L-1,n]` is a negative zero. That is the gemm lane's
F6a lesson and IDENTITY_PATHS row 39 in one line. The omitted form is pinned:
an operation that is not performed is the strongest possible statement of a
`+0.0` seed, which is the argument `gemm_identical.mojo::_fold_drain` already
makes for `P == 1` ("performs NO addition"). `SAB_BWD_T1_SEED_ADD` is the
other closure. **PREDICTED BITWISE INERT on any fixture with no negative-zero
`dy*Cm` product at the last token**, which is every hashed fixture, so the
gate must PLANT one and must raise VACUOUS rather than pass if it does not.
This is not a measured claim; nothing here has run.

**DEVIATION 1083, `ddelta` is ONE interleaved fold and the plan contradicts
itself about that.** Plan section 2.2 spells `ddelta = ddelta_B + ddelta_A`,
two folds joined at S14'. Plan section 3.2 row B18 spells
`ddelta = sum_n (B path + A path)`, one fold over `N`, category c-inherited,
"interleaving pinned". Those are different numbers. **The interleaved form is
pinned here** -- one accumulator, ascending `n`, per step the `B` term then
the `A` term, two `fma`s into one register -- because it is the single
ascending fused chain B18 inherits from S10, because it is upstream's own
shape (`ddelta_vals[i] += ddelta_u * u + dx * A * a`, one statement per state
index), and because the two-fold form needs a second `N`-length accumulator
for no numerical reason. `SAB_BWD_DDELTA_TWO_FOLDS` is section 2.2's reading.
**The plan text is corrected in the same commit as this file**; a document
that spells a seam two ways is a document that has not decided.

**DEVIATION 1084, T4 is a SECOND PASS over `t`, not a fold inside T1.**
T4 folds `d_arg * delta` over `t` DESCENDING. The obvious place for it is
inside T1's own descending walk, and that spelling makes `SAB_BWD_DA_
ASCENDING` unimplementable without a second traversal anyway -- so the fold
is its own kernel, reading the materialized `dh` T1 already wrote. The price
is that `da[t,d,n]` and `d_arg[t,d,n]` are computed TWICE, once in
`selective_scan_bwd_scan_kernel` for `ddelta`'s A path and once here. Both
computations are the same expression on the same bits and are therefore bit
equal, so the cost is `M * d_inner * D_STATE` extra `identical_exp` calls and
NOT a second answer. Bought deliberately: it keeps T4's direction a
one-line change at one site, which is what `SAB_BWD_DA_ASCENDING` has to
falsify, and it keeps the fold out of a kernel that already carries three
accumulators.

WHAT THIS FILE IS NOT
----------------------
Not a gate. **Nothing here has been compiled or run**, no gradient has ever
been produced by it, and every sentence about behavior is a prediction.
`archive/plans/mamba/IDENTICAL_BACKWARD_PLAN.md` section 7's gates MB1-MB10 are still
SPECIFIED AND NOT BUILT, and the host backward oracle they compare against
does not exist, so there is nothing yet to compare these bits to. The blocked
T3 of plan section 5.3 (the model-scale tiling) is NOT here either: the T3
spelling below is GATE SCALE and its `h` plus `dh` footprint is
`2 * M * d_inner * 16 * 4` bytes.

`[[mojo-buffer-freed-at-last-use]]`: every launcher is an `_into` form. It
enqueues and returns without waiting. THE CALLER OWNS EVERY BUFFER and must
keep each alive past its own `ctx.synchronize()`.

DEVIATIONS 1070-1074 (T1-T5, declared in `mamba/checks/mamba_backward.mojo`),
1081-1084 (spent here).
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.math import exp2
from std.sys.compile import is_defined

from max.gpu.host import DeviceBuffer, DeviceContext

from gemm.checks.gemm_identical import (
    GEMM_FOLD_SLOTS,
    _fold_drain,
    _fold_push,
    _leaf_at,
    _leaf_bounds,
    contract_partition,
)
from checks.kernel_matrix import TARGET_COLUMN, column_has_float_atomics
from checks.numerics import ftz, identical_exp, identical_mul_add

# THE FORWARD'S OWN SPELLINGS, IMPORTED RATHER THAN RE-DECLARED. `pinned_mul`
# is DEVIATION 720's device construction and this lane may not hold a second
# opinion about it; the three scan sabotage arms are imported for DEVIATION
# 1081's reason (a sabotaged forward must get a matching checkpoint).
from mamba.impl.mamba_ssm.ops.selective_scan_interface import (
    MAX_DSTATE,
    SAB_S5_EXP2,
    SAB_S8_CUDA_PAIRING,
    SAB_S9_UNFUSED,
    pinned_mul,
)


# ===========================================================================
# THE BUFFER LAYOUTS. DEVIATION 1081.
# ===========================================================================
# Stated once, here, because an index expression that appears in four kernels
# with three spellings is a bug waiting for a fixture that cannot see it.
# `M = B * L` token-major throughout, exactly the forward's layouts.
#
#     dy, du_s, ddelta, w, delta, u   [M, di]          i = t * di + d
#     cmat, bmat                      [M, N]           i = t * N + n
#     a  (stage `A.out`)              [di, N]          i = d * N + n
#     dh                              [M, di, N]       i = (t*di + d)*N + n
#     h_ckpt                          [B, L+1, di, N]
#                                     i = ((b*(L+1) + s)*di + d)*N + n
#     dA_partial                      [B, di, N]       i = (b*di + d)*N + n
#     dA, dA_log                      [di, N]          i = d * N + n
#
# **THE CHECKPOINT'S SLOT INDEX IS `s = li + 1`, NOT `li`.** Slot 0 holds
# `h[-1]`, the state CARRIED INTO the call, which is zeros on prefill
# (contract section 5) and the previous call's state on decode. So the state
# AFTER token `li` is at slot `li + 1` and the state BEFORE it -- which is
# what S6' needs -- is at slot `li`. Getting this off by one produces a
# plausible gradient that is bit identical wherever `h` is constant in `t`,
# which is every all-zero prefix, and `bwd.ddelta` is the first stage that
# moves. `dh` has NO carry slot, because T1's seed is structural
# (DEVIATION 1082) and there is nothing to store at `t = L`.
#
# `dh` is token-major `[M, di, N]` and not `[B, di, L, N]` on purpose: it is a
# CARD STAGE and plan section 7 names its shape `bwd.dh [M,di,N]`. T3 then
# reads it with stride `N` in `d`, which is the axis T3 contracts.


# ===========================================================================
# LAUNCH GEOMETRY. An EXECUTION plan quantity (DEVIATION 725's rule, applied
# here): it decides which thread owns a cell and nothing else. No kernel below
# reads `block_dim` or `block_idx` in any expression that reaches a fold
# boundary, an accumulator seed, a tap index or a leaf size. The ONE place a
# launch quantity could leak into the arithmetic is T3's `(leaf, p_count)`,
# and those come from `contract_partition(d_inner)` and from nowhere else --
# gemm contract section 6's one-door rule, borrowed rather than re-spelled.
# ===========================================================================

comptime BWD_SCAN_TPB = 64
"""Threads per block for the per-`(b, d)` kernels. The forward's launcher
takes `block_size` as an argument and defaults it to 64; these take it the
same way, and gate MB4 varies it across 32, 64, 128 and 256 precisely to
assert it cannot move a bit."""

comptime BWD_CELL_TPB = 128
"""Threads per block for the per-cell kernels."""


def _grid(n: Int, tpb: Int) -> Int:
    var g = (n + tpb - 1) // tpb
    if g < 1:
        return 1
    return g


# ===========================================================================
# THE SABOTAGE SWITCHES THIS FILE OWNS
# ===========================================================================
# OFF in every build that does not name them. Each is one compile-time
# alternative spelling of ONE clause, reachable by a plausible implementer,
# and each carries its own INERT prediction -- because the lesson this lane
# has already paid for is `adv_softplus_guard`, where an arm ran, reached its
# branch, and moved nothing, and 23 of 23 dumps came back byte identical.
# An arm that has never been shown to fail is an arm nobody has tested.
#
#     tools/with_identical_mode.sh pixi run mojo build \
#         -D MOJOLEARN_MAMBA_SABOTAGE_BWD_S9B_FORWARD=1 \
#         -I . mamba/checks/mamba_backward_compile_probe.mojo

#: T1 walked ASCENDING in `t`. The reverse recurrence run forwards is the
#: single most damaging thing that can be wrong here and it is the one gate
#: MB7 exists for. INERT at `L == 1`.
comptime SAB_BWD_S9B_FORWARD = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S9B_FORWARD"
]()

#: T1 multiplies by `da[t]` where `da[t+1]` belongs. `h[t]` influences
#: `h[t+1]` through `da[t+1]`, never through `da[t]`; upstream spells the
#: shift explicitly as `thread_reverse_data[i-1].x = delta_a_exp`
#: (`selective_scan_bwd_kernel.cuh:255`). **INERT at `L == 1` AND wherever
#: every `delta` is equal across tokens**, so MB7's fixture must plant
#: `delta` varying in `t` and the gate must REFUSE to pass if it does not.
comptime SAB_BWD_S9B_DA_OFFSET = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S9B_DA_OFFSET"
]()

#: T1's `fma` split into a multiply then an add -- the same relationship
#: `SAB_S9_UNFUSED` has with forward seam S9, and the forward measured that
#: one AGREEING at every `L = 1` shape and first failing at `L = 4`. INERT at
#: `L == 1` for the mirror-image reason: the seed is `dh[L] = +0.0`, so the
#: product both spellings round is a product by zero.
comptime SAB_BWD_S9B_UNFUSED = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S9B_UNFUSED"
]()

#: DEVIATION 1082's other closure: T1's seed spelled as a stored `+0.0` that
#: is FOLDED IN rather than as an omitted operation. Differs only where the
#: last token's `dy * Cm` product is a NEGATIVE ZERO, which a hashed fixture
#: cannot reach. The gate must plant one and must raise VACUOUS otherwise.
comptime SAB_BWD_T1_SEED_ADD = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_T1_SEED_ADD"
]()

#: T2 refused: `h[t-1]`'s contribution recovered upstream's way, from
#: `a = h[t] - dbu[t]` (`selective_scan_bwd_kernel.cuh:290`), which is exact
#: only when S9's fused rounding did nothing and which suffers unbounded
#: relative cancellation whenever `|da*h[t-1]| << |dbu[t]|`. Spelled here in
#: UPSTREAM'S OWN SHAPE -- their `a` is `da*h[t-1]`, so their `d_arg` is one
#: multiply (`dh * a`) where ours is two (`(dh * h[t-1]) * da`) -- so gate MB9
#: MEASURES the ulp distance instead of asserting that it matters. **If MB9
#: records zero ulps everywhere, DEVIATION 1071's refusal must be downgraded
#: to a preference.** INERT at `t == 0`, where `h[-1]` is zero on prefill.
comptime SAB_BWD_H_SUBTRACT = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_H_SUBTRACT"
]()

#: `du_s` and `ddelta` folded `n` DESCENDING. The forward's `SAB_S10_
#: DESCENDING` is the same mistake one seam earlier and it failed at
#: `b=1 l=1 d_model=8` by 1 ulp on `scan.y`. INERT at `N == 1`, which this
#: profile never runs (`D_STATE` is 16), and wherever a row has fewer than
#: two nonzero terms.
comptime SAB_BWD_S10_N_DESCENDING = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_S10_N_DESCENDING"
]()

#: DEVIATION 1083's other reading: `ddelta` as TWO folds over `n` joined by
#: one add, plan section 2.2's spelling, instead of one interleaved chain.
#: INERT wherever one of the two paths is zero for every `n`, which includes
#: `t == 0` on prefill (the A path's `h[-1]` is zero there).
comptime SAB_BWD_DDELTA_TWO_FOLDS = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_DDELTA_TWO_FOLDS"
]()

#: T4's `t` fold reversed to ASCENDING, which is the direction every OTHER
#: fold in both profiles runs. DEVIATION 1073 is the one place this lane pins
#: a descending fold and this arm is what has to bite for that deviation to be
#: MEASURED rather than merely argued. INERT at `L == 1`. **If it does not
#: bite, DEVIATION 1073 should be re-decided in favor of consistency.**
comptime SAB_BWD_DA_ASCENDING = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_DA_ASCENDING"
]()

#: T3 by float `atomicAdd` instead of the per-token `(1, N, d_inner)`
#: contraction -- upstream's answer (`:486`, `:491`, `:306`, `:307`).
#: **THIS ARM CANNOT BE SEEN BY A SINGLE-LAUNCH ORACLE COMPARISON**: an
#: atomic accumulation is bit identical to a pinned fold on any run whose
#: arrival order happens to match, so MB3 can pass with it armed and only
#: MB4's repeat-launch and launch-geometry arms can see it. INERT at
#: `d_inner == 1`.
comptime SAB_BWD_DBDC_ATOMIC = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_DBDC_ATOMIC"
]()

#: T5's batch fold by float `atomicAdd` instead of private slots plus a second
#: kernel -- upstream's `dA` accumulation (`:483`). MB4, not MB3, for the same
#: reason. INERT at `B == 1`, where there is one term and no order.
comptime SAB_BWD_PARAM_ATOMIC = is_defined[
    "MOJOLEARN_MAMBA_SABOTAGE_BWD_PARAM_ATOMIC"
]()

comptime ANY_BWD_SCAN_SABOTAGE = (
    SAB_BWD_S9B_FORWARD
    or SAB_BWD_S9B_DA_OFFSET
    or SAB_BWD_S9B_UNFUSED
    or SAB_BWD_T1_SEED_ADD
    or SAB_BWD_H_SUBTRACT
    or SAB_BWD_S10_N_DESCENDING
    or SAB_BWD_DDELTA_TWO_FOLDS
    or SAB_BWD_DA_ASCENDING
    or SAB_BWD_DBDC_ATOMIC
    or SAB_BWD_PARAM_ATOMIC
)


def mamba_backward_scan_sabotage_name() -> String:
    """Which arm this binary compiled with, for a gate's banner.

    A backward binary can carry a FORWARD sabotage and a GEMM sabotage as
    well, so a banner printing only this one mislabels the run. A check must
    print this beside `mamba_scan_sabotage_name()`,
    `mamba_block_sabotage_name()`, `mamba_backward_block_sabotage_name()`,
    `mamba_backward_sabotage_name()` and `gemm_sabotage_name()`.
    """
    comptime if SAB_BWD_S9B_FORWARD:
        return String("BWD_S9B_FORWARD")
    comptime if SAB_BWD_S9B_DA_OFFSET:
        return String("BWD_S9B_DA_OFFSET")
    comptime if SAB_BWD_S9B_UNFUSED:
        return String("BWD_S9B_UNFUSED")
    comptime if SAB_BWD_T1_SEED_ADD:
        return String("BWD_T1_SEED_ADD")
    comptime if SAB_BWD_H_SUBTRACT:
        return String("BWD_H_SUBTRACT")
    comptime if SAB_BWD_S10_N_DESCENDING:
        return String("BWD_S10_N_DESCENDING")
    comptime if SAB_BWD_DDELTA_TWO_FOLDS:
        return String("BWD_DDELTA_TWO_FOLDS")
    comptime if SAB_BWD_DA_ASCENDING:
        return String("BWD_DA_ASCENDING")
    comptime if SAB_BWD_DBDC_ATOMIC:
        return String("BWD_DBDC_ATOMIC")
    comptime if SAB_BWD_PARAM_ATOMIC:
        return String("BWD_PARAM_ATOMIC")
    return String("none")


# ===========================================================================
# T2's PRODUCER: THE FORWARD RECURRENCE, RE-RUN, EVERY STATE STORED.
# DEVIATION 1081.
# ===========================================================================


def selective_scan_checkpoint_kernel[
    DSTATE: Int
](
    hck_ptr: MutPointer[Float32, MutAnyOrigin],
    h_in_ptr: MutPointer[Float32, MutAnyOrigin],
    u_ptr: MutPointer[Float32, MutAnyOrigin],
    delta_ptr: MutPointer[Float32, MutAnyOrigin],
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    batch_in: Int32,
    seqlen_in: Int32,
    dim_in: Int32,
):
    """Every `h[t,d,n]` for `t` in `[-1, L)`, into `hck_ptr`.

    **THE BODY IS `selective_scan_fwd_kernel`'s S5-S9 CHAIN, TRANSCRIBED,
    AND THE TRANSCRIPTION IS THE RISK.** Line for line against
    `mamba/impl/mamba_ssm/ops/selective_scan_interface.mojo:374-412`:

        :382  da_arg = ftz(pinned_mul(dl, a_vals[n]))        S5
        :388  da     = ftz(identical_exp(da_arg))            S6
        :399  db     = ftz(pinned_mul(dl, bv))               S7
        :402  dbu    = ftz(pinned_mul(db, uv))               S8
        :412  state  = ftz(identical_mul_add(da, state, dbu)) S9, ONE rounding

    Nothing about S10 or S11 is here, because `y` and `out` are the forward's
    outputs and the backward reads the recorded stages for them. **What this
    kernel does NOT do is the whole point**: it stores `state` at every
    token instead of only after the last one.

    THE THREE FORWARD ARMS ARE HONORED, by import (DEVIATION 1081). A build
    carrying `-D MOJOLEARN_MAMBA_SABOTAGE_S9_UNFUSED=1` has a forward whose
    `h` rounds twice, and a checkpoint that rounded once would then be a
    checkpoint of a different forward. The arms travel together.

    **AND THEY TRAVEL ONLY THIS FAR**, which is the part worth stating.
    `SAB_S8_CUDA_PAIRING` and `SAB_S9_UNFUSED` are honored HERE and NOWHERE
    else in this lane, because they change how the forward ROUNDS `h` and not
    what the derivative of the exact function is. Plan section 5.4 is the
    argument: `d/d(da) = h[t-1]` and `d/d(dbu) = 1` whether or not the
    forward fused S9, and the exact bilinear form `delta*B*u` has the same
    partials however the forward paired it, so the backward differentiates
    the exact form and rounds its OWN products through `pinned_mul`.
    `SAB_S5_EXP2` is different and is honored everywhere `da` is recomputed,
    because it changes `da`'s VALUE and `da` is a multiplier the backward
    consumes directly rather than a rounding of something it re-derives.

    THE SHAPE IS THE FORWARD'S: one `(batch, dim)` pair per thread, `DSTATE`
    values in registers, the whole sequence walked in order. No float crosses
    a thread boundary, there is no shared memory, no reduction and no atomic,
    so `block_size` and the grid decide only WHICH thread holds a pair. That
    is the forward's structural launch-invariance argument, inherited
    verbatim, and it is the LAST place in this lane where it can be made
    without qualification -- T3 and T5 both cross threads.
    """
    var batch = Int(batch_in)
    var seqlen = Int(seqlen_in)
    var dim = Int(dim_in)

    var thread_id = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if thread_id >= batch * dim:
        return
    var bb = thread_id // dim
    var d = thread_id - bb * dim
    if bb >= batch or d >= dim:
        return

    var slots = seqlen + 1

    var state = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        state[n] = ftz(h_in_ptr.unsafe_load((bb * dim + d) * DSTATE + n))

    # Slot 0 IS `h[-1]`, the carried state, stored as read. A copy, not a
    # seam: the backward reads it at `t = 0` and on prefill it is the zeros
    # `allocate_inference_cache` hands over.
    comptime for n in range(DSTATE):
        hck_ptr.unsafe_store(
            ((bb * slots + 0) * dim + d) * DSTATE + n, state[n]
        )

    var a_vals = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        a_vals[n] = ftz(a_ptr.unsafe_load(d * DSTATE + n))

    for li in range(seqlen):
        var t = bb * seqlen + li
        var uv = ftz(u_ptr.unsafe_load(t * dim + d))
        var dl = ftz(delta_ptr.unsafe_load(t * dim + d))

        comptime for n in range(DSTATE):
            var da_arg = ftz(pinned_mul(dl, a_vals[n]))
            var da: Float32
            comptime if SAB_S5_EXP2:
                da = ftz(
                    exp2(
                        ftz(pinned_mul(da_arg, Float32(1.4426950408889634)))
                    )
                )
            else:
                da = ftz(identical_exp(da_arg))

            var bv = ftz(b_ptr.unsafe_load(t * DSTATE + n))
            var dbu: Float32
            comptime if SAB_S8_CUDA_PAIRING:
                var du = ftz(pinned_mul(dl, uv))
                dbu = ftz(pinned_mul(bv, du))
            else:
                var db = ftz(pinned_mul(dl, bv))
                dbu = ftz(pinned_mul(db, uv))

            comptime if SAB_S9_UNFUSED:
                state[n] = ftz(ftz(pinned_mul(da, state[n])) + dbu)
            else:
                state[n] = ftz(identical_mul_add(da, state[n], dbu))

        comptime for n in range(DSTATE):
            hck_ptr.unsafe_store(
                ((bb * slots + (li + 1)) * dim + d) * DSTATE + n, state[n]
            )


def selective_scan_checkpoint_fn(
    ctx: DeviceContext,
    mut h_ckpt: DeviceBuffer[DType.float32],
    mut h_in: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut delta: DeviceBuffer[DType.float32],
    mut A: DeviceBuffer[DType.float32],
    mut B: DeviceBuffer[DType.float32],
    batch: Int,
    seqlen: Int,
    dim: Int,
    block_size: Int = BWD_SCAN_TPB,
) raises:
    """T2's checkpoint, `B * (L+1) * dim * D_STATE` floats. ASYNCHRONOUS.

    `h_in` is the state CARRIED INTO the block call, which is the same buffer
    the forward's `selective_scan_fn` received -- **not the `scan.h` it left
    behind**, which is the state AFTER the last token. A caller that passes
    the post-call state gets a checkpoint whose slot 0 is wrong and whose
    every other slot is wrong with it, and no shape check can see it. The
    forward's `h_state` is in-and-out on one buffer, so a training step must
    keep a copy of the incoming state before it calls the forward.

    Argument order mirrors `selective_scan_fn`'s so the two read alike.
    """
    if batch < 0 or seqlen < 0 or dim < 0:
        raise Error(
            "selective_scan_checkpoint_fn: negative shape (batch="
            + String(batch)
            + " seqlen="
            + String(seqlen)
            + " dim="
            + String(dim)
            + ")"
        )
    if block_size < 1:
        raise Error(
            "selective_scan_checkpoint_fn: block_size must be >= 1, got "
            + String(block_size)
        )
    var m = batch * seqlen
    _require(len(u), m * dim, "u", "[M, dim]")
    _require(len(delta), m * dim, "delta", "[M, dim]")
    _require(len(A), dim * MAX_DSTATE, "A", "[dim, 16]")
    _require(len(B), m * MAX_DSTATE, "B", "[M, 16]")
    _require(len(h_in), batch * dim * MAX_DSTATE, "h_in", "[B, dim, 16]")
    _require(
        len(h_ckpt),
        batch * (seqlen + 1) * dim * MAX_DSTATE,
        "h_ckpt",
        "[B, L+1, dim, 16]",
    )

    var total = batch * dim
    if total > 0:
        comptime kern = selective_scan_checkpoint_kernel[MAX_DSTATE]
        ctx.enqueue_function[kern](
            h_ckpt.unsafe_ptr(),
            h_in.unsafe_ptr(),
            u.unsafe_ptr(),
            delta.unsafe_ptr(),
            A.unsafe_ptr(),
            B.unsafe_ptr(),
            Int32(batch),
            Int32(seqlen),
            Int32(dim),
            grid_dim=(_grid(total, block_size), 1, 1),
            block_dim=(block_size, 1, 1),
        )


# ===========================================================================
# T1 + B17 + B18: THE REVERSE RECURRENCE AND THE TWO `N` FOLDS.
# DEVIATIONS 1070, 1082, 1083.
# ===========================================================================


def selective_scan_bwd_scan_kernel[
    DSTATE: Int
](
    dh_ptr: MutPointer[Float32, MutAnyOrigin],
    du_s_ptr: MutPointer[Float32, MutAnyOrigin],
    ddelta_ptr: MutPointer[Float32, MutAnyOrigin],
    w_ptr: MutPointer[Float32, MutAnyOrigin],
    dy_ptr: MutPointer[Float32, MutAnyOrigin],
    c_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    delta_ptr: MutPointer[Float32, MutAnyOrigin],
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    u_ptr: MutPointer[Float32, MutAnyOrigin],
    hck_ptr: MutPointer[Float32, MutAnyOrigin],
    batch_in: Int32,
    seqlen_in: Int32,
    dim_in: Int32,
):
    """One `(batch, dim)` pair per thread, the whole sequence in REVERSE.

    THE FORWARD'S SHAPE, RUN THE OTHER WAY. `DSTATE` values of `dh` live in
    registers exactly as `DSTATE` values of `h` do in the forward, plus
    `DSTATE` values of the carried `da[t+1]`. **No float crosses a thread
    boundary in this kernel**, so it inherits the forward's structural launch
    invariance and its batch-composition invariance unchanged. Plan section
    5.1 predicted this seam would be the crux and it is not; the crux is T3,
    which is a different kernel for exactly that reason.

    FOUR OUTPUTS, in the plan's own operation numbers:

        dh[t,d,n]     T1, B13     reverse recurrence, FUSED, one rounding
        du_s[t,d]     B17         sum over n of dh * dbb, ascending, FUSED
        ddelta[t,d]   B18         ONE interleaved chain, B term then A term
        w[t,d]        T3's pre-form, `pinned_mul(delta, u)`

    `dy` IS `dsk` AND THERE IS NO SEPARATE BUFFER. Plan operation B9 is a
    copy (`dy = dsk`), so the caller passes `bwd.dsk` here and the copy is
    realized as "no instruction at all".

    THE TWO `n` LOOPS ARE SEPARATE ON PURPOSE. T1's step for state index `n`
    touches only `dh[n]`, so the recurrence and the folds could share one
    loop; they do not, because `SAB_BWD_S10_N_DESCENDING` has to reverse the
    FOLDS without reversing anything about T1, and a fused loop would make
    that arm reverse two clauses at once and mislabel whatever it moved.

    `da[t]` IS COMPUTED ONCE AND CARRIED. The recurrence at step `t` needs
    `da[t+1]`, which the previous (higher) step already computed; carrying it
    in a register is bit equal to recomputing it and saves `M * di * N`
    `identical_exp` calls. `da[t]` is ALSO needed by the A path of `ddelta`
    at the same step, so the carry is read before it is overwritten.
    """
    var batch = Int(batch_in)
    var seqlen = Int(seqlen_in)
    var dim = Int(dim_in)

    var thread_id = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if thread_id >= batch * dim:
        return
    var bb = thread_id // dim
    var d = thread_id - bb * dim
    if bb >= batch or d >= dim:
        return

    var slots = seqlen + 1

    # A's row for this dim, hoisted -- the forward hoists the same row and a
    # load plus a flush is loop invariant, so this moves no bits.
    var a_vals = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        a_vals[n] = ftz(a_ptr.unsafe_load(d * DSTATE + n))

    var dh_state = SIMD[DType.float32, DSTATE](0.0)
    var da_carry = SIMD[DType.float32, DSTATE](0.0)

    for step in range(seqlen):
        var li = seqlen - 1 - step
        comptime if SAB_BWD_S9B_FORWARD:
            # SABOTAGE: the reverse recurrence run forwards. `dh[t]` then
            # depends on tokens BEFORE `t` instead of after it, which is a
            # different function and not a differently rounded one. Gate MB7
            # is the structural gate for this and MB3 alone would only see
            # it through an oracle a shared misconception could corrupt.
            li = step
        var t = bb * seqlen + li

        var dyv = ftz(dy_ptr.unsafe_load(t * dim + d))
        var uv = ftz(u_ptr.unsafe_load(t * dim + d))
        var dl = ftz(delta_ptr.unsafe_load(t * dim + d))

        # ---- T1, the recurrence -------------------------------------
        comptime for n in range(DSTATE):
            var cv = ftz(c_ptr.unsafe_load(t * DSTATE + n))
            # S10's own product, differentiated: `y = sum_n C * h`, so
            # `dh[t,n]` picks up `dy * C[t,n]`.
            var contrib = ftz(pinned_mul(dyv, cv))

            # `da[t]` in the forward's spelling, S5 then S6. Needed by the A
            # path below and carried to the next (lower) step for T1.
            var da_arg = ftz(pinned_mul(dl, a_vals[n]))
            var da_t: Float32
            comptime if SAB_S5_EXP2:
                da_t = ftz(
                    exp2(
                        ftz(pinned_mul(da_arg, Float32(1.4426950408889634)))
                    )
                )
            else:
                da_t = ftz(identical_exp(da_arg))

            # DEVIATION 1082: the seed is an OMITTED operation at the first
            # step of the walk, never a stored `+0.0` folded in.
            var dh_n = ftz(contrib)
            comptime if SAB_BWD_T1_SEED_ADD:
                dh_n = ftz(ftz(Float32(0.0)) + ftz(contrib))
            if step != 0:
                var afac = da_carry[n]
                comptime if SAB_BWD_S9B_DA_OFFSET:
                    # SABOTAGE: `da[t]` where `da[t+1]` belongs. Bitwise
                    # inert wherever every `delta` is equal across tokens.
                    afac = da_t
                comptime if SAB_BWD_S9B_UNFUSED:
                    # SABOTAGE: two roundings where T1 pins one.
                    dh_n = ftz(ftz(pinned_mul(afac, dh_state[n])) + contrib)
                else:
                    dh_n = ftz(
                        identical_mul_add(afac, dh_state[n], contrib)
                    )

            dh_state[n] = dh_n
            da_carry[n] = da_t
            dh_ptr.unsafe_store((t * dim + d) * DSTATE + n, dh_n)

        # ---- B17 and B18, the two folds over `n` --------------------
        # S10's ascending fused chain, seeded `+0.0`, inherited (plan 3.2
        # calls both c-inherited). `du_s` is its own accumulator because it
        # is its own card stage; `ddelta` is ONE accumulator taking the B
        # term then the A term per `n` (DEVIATION 1083).
        var acc_us = Float32(0.0)
        var acc_dd = Float32(0.0)
        var acc_dd_a = Float32(0.0)

        comptime for nn in range(DSTATE):
            var n = nn
            comptime if SAB_BWD_S10_N_DESCENDING:
                n = DSTATE - 1 - nn

            var bv = ftz(b_ptr.unsafe_load(t * DSTATE + n))
            # S7's own product, `delta * B`, which is what `dbu` is
            # differentiable in for `u`.
            var dbb = ftz(pinned_mul(dl, bv))

            # B17: `du_s[t,d] = sum_n dh * dbb`.
            acc_us = ftz(identical_mul_add(dh_state[n], dbb, acc_us))

            # B18, the B path: `d(dbb) = dh * u`, contracted against `B`.
            var ddbb = ftz(pinned_mul(dh_state[n], uv))
            acc_dd = ftz(identical_mul_add(ddbb, bv, acc_dd))

            # B18, the A path. `d_arg` is `dh * h[t-1] * da`, and the two
            # spellings below differ in HOW `da * h[t-1]` is obtained.
            var d_arg: Float32
            comptime if SAB_BWD_H_SUBTRACT:
                # SABOTAGE, DEVIATION 1071's refused alternative, in
                # upstream's own shape: `a = h[t] - dbu[t]` recovers
                # `da*h[t-1]` in ONE subtraction, so their `d_arg` is one
                # multiply where ours is two. Unbounded relative
                # cancellation when `|da*h[t-1]| << |dbu[t]|`, which is the
                # normal case early in a sequence. MB9 prices it.
                var dbu_h = ftz(pinned_mul(dbb, uv))
                var hcur = ftz(
                    hck_ptr.unsafe_load(
                        ((bb * slots + (li + 1)) * dim + d) * DSTATE + n
                    )
                )
                var arec = ftz(ftz(hcur) - ftz(dbu_h))
                d_arg = ftz(pinned_mul(dh_state[n], arec))
            else:
                # T2: `h[t-1]` READ from the checkpoint, slot `li`.
                var hprev = ftz(
                    hck_ptr.unsafe_load(
                        ((bb * slots + li) * dim + d) * DSTATE + n
                    )
                )
                # S6' then the exp node: `exp' is exp`, so `d_arg` is
                # `(dh * h[t-1]) * da`, two separately rounded products.
                var d_da = ftz(pinned_mul(dh_state[n], hprev))
                d_arg = ftz(pinned_mul(d_da, da_carry[n]))

            comptime if SAB_BWD_DDELTA_TWO_FOLDS:
                # SABOTAGE: plan section 2.2's two-fold reading.
                acc_dd_a = ftz(
                    identical_mul_add(d_arg, a_vals[n], acc_dd_a)
                )
            else:
                acc_dd = ftz(identical_mul_add(d_arg, a_vals[n], acc_dd))

        var ddelta_v = acc_dd
        comptime if SAB_BWD_DDELTA_TWO_FOLDS:
            ddelta_v = ftz(ftz(acc_dd) + ftz(acc_dd_a))

        du_s_ptr.unsafe_store(t * dim + d, acc_us)
        ddelta_ptr.unsafe_store(t * dim + d, ddelta_v)

        # T3's pre-form, `w[t,d] = pinned_mul(delta, u)`, written here
        # because this thread already holds both operands flushed. The
        # alternative pre-forms `R[t,d,n] = pinned_mul(dh, u)`, which costs
        # `DSTATE` times the memory and one extra rounding per `(t,d,n)`
        # rather than per `(t,d)`. DEVIATION 1072.
        w_ptr.unsafe_store(t * dim + d, ftz(pinned_mul(dl, uv)))


def selective_scan_bwd_scan_into(
    ctx: DeviceContext,
    mut dh: DeviceBuffer[DType.float32],
    mut du_s: DeviceBuffer[DType.float32],
    mut ddelta: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut C: DeviceBuffer[DType.float32],
    mut B: DeviceBuffer[DType.float32],
    mut delta: DeviceBuffer[DType.float32],
    mut A: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut h_ckpt: DeviceBuffer[DType.float32],
    batch: Int,
    seqlen: Int,
    dim: Int,
    block_size: Int = BWD_SCAN_TPB,
) raises:
    """T1, B17 and B18 in one launch. ASYNCHRONOUS, caller-owned buffers."""
    if batch < 0 or seqlen < 0 or dim < 0:
        raise Error(
            "selective_scan_bwd_scan_into: negative shape (batch="
            + String(batch)
            + " seqlen="
            + String(seqlen)
            + " dim="
            + String(dim)
            + ")"
        )
    if block_size < 1:
        raise Error(
            "selective_scan_bwd_scan_into: block_size must be >= 1, got "
            + String(block_size)
        )
    var m = batch * seqlen
    _require(len(dy), m * dim, "dy", "[M, dim]")
    _require(len(u), m * dim, "u", "[M, dim]")
    _require(len(delta), m * dim, "delta", "[M, dim]")
    _require(len(A), dim * MAX_DSTATE, "A", "[dim, 16]")
    _require(len(B), m * MAX_DSTATE, "B", "[M, 16]")
    _require(len(C), m * MAX_DSTATE, "C", "[M, 16]")
    _require(len(dh), m * dim * MAX_DSTATE, "dh", "[M, dim, 16]")
    _require(len(du_s), m * dim, "du_s", "[M, dim]")
    _require(len(ddelta), m * dim, "ddelta", "[M, dim]")
    _require(len(w), m * dim, "w", "[M, dim]")
    _require(
        len(h_ckpt),
        batch * (seqlen + 1) * dim * MAX_DSTATE,
        "h_ckpt",
        "[B, L+1, dim, 16]",
    )

    var total = batch * dim
    if total > 0:
        comptime kern = selective_scan_bwd_scan_kernel[MAX_DSTATE]
        ctx.enqueue_function[kern](
            dh.unsafe_ptr(),
            du_s.unsafe_ptr(),
            ddelta.unsafe_ptr(),
            w.unsafe_ptr(),
            dy.unsafe_ptr(),
            C.unsafe_ptr(),
            B.unsafe_ptr(),
            delta.unsafe_ptr(),
            A.unsafe_ptr(),
            u.unsafe_ptr(),
            h_ckpt.unsafe_ptr(),
            Int32(batch),
            Int32(seqlen),
            Int32(dim),
            grid_dim=(_grid(total, block_size), 1, 1),
            block_dim=(block_size, 1, 1),
        )


# ===========================================================================
# T3: `dBm` AND `dCm`, THE `d_inner` CONTRACTION. DEVIATION 1072.
# ===========================================================================
# THE HARD SEAM, and plan section 5.2 is why. `B` and `C` come out of x_proj
# per TOKEN and are shared by every one of the `d_inner` channels; the forward
# READS them, which costs nothing, and the backward SUMS over the channels
# that read them. Every term of that sum lives in a different thread under the
# forward's `(b, d)` grid.
#
# **THE FOLD IS gemm v1's, NOT A NEW ONE.** `dCm[t,:]` is `dy[t,:]` at
# `1 x d_inner` times `h[t]` at `d_inner x D_STATE`, which is
# `mojolearn.identical.gemm.fp32.v1` at `(m', n', k') = (1, D_STATE,
# d_inner)` exactly. So this kernel does not declare a fold: it IMPORTS
# `contract_partition`, `_leaf_bounds`, `_leaf_at`, `_fold_push` and
# `_fold_drain` from `gemm/checks/gemm_identical.mojo` and calls them
# unchanged, which is the same-spelling form of reuse rather than the
# same-math form. If `d_inner > 128` this is NOT one serial chain, it is
# `contract_partition(d_inner)` leaves combined by v1's balanced tree, and an
# implementation that treated it as one chain would be a different answer at
# every real model width.
#
# **TWO OF gemm's FOUR FOLD ARMS REACH THIS KERNEL AND TWO DO NOT**, and a
# gate must print that rather than discover it. `SAB_FOLD_STRIDE` lives inside
# `_fold_push` and `SAB_LEAF_ROTATE` inside `_leaf_at`, so both bite here.
# `SAB_LEAF_READS_LAUNCH`, `SAB_PAD_PLUS_ZERO` and `SAB_FOLD_SERIAL` live in
# the bodies of gemm's own kernels and are NOT reachable through this entry
# point. Running them against this lane and reporting green measures nothing.


def mamba_bwd_dbc_kernel[
    DSTATE: Int
](
    dcm_ptr: MutPointer[Float32, MutAnyOrigin],
    dbm_ptr: MutPointer[Float32, MutAnyOrigin],
    dy_ptr: MutPointer[Float32, MutAnyOrigin],
    hck_ptr: MutPointer[Float32, MutAnyOrigin],
    w_ptr: MutPointer[Float32, MutAnyOrigin],
    dh_ptr: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    l_in: Int32,
    di_in: Int32,
    leaf_in: Int32,
    p_in: Int32,
):
    """One `(token, state index)` cell per thread, contracting `d` ASCENDING.

        dCm[t,n] = fold over d of fma(dy[t,d],  h[t,d,n],  acc)
        dBm[t,n] = fold over d of fma(w[t,d],   dh[t,d,n], acc)
        with     w[t,d] = ftz(pinned_mul(delta[t,d], u[t,d]))

    Both leaves are ONE `identical_mul_add` on flushed operands with the
    accumulator flushed after every step, which is gemm contract 7.1's
    ascending chain, and the leaf partials go through v1's balanced tree.

    `leaf_in` and `p_in` come from `contract_partition(d_inner)` on the host
    and from nowhere else. **`d_inner` IS THE REDUCED LENGTH, NEVER A LAUNCH
    QUANTITY**: this kernel cannot compute a leaf boundary from anything but
    the two arguments, exactly as `identical_gemm_flat_kernel` cannot.

    `w` IS PRE-FORMED AND `R = pinned_mul(dh, u)` IS NOT. The two are
    algebraically equal and differ in the last bit; the chosen one costs
    `D_STATE` times less memory and one rounding per `(t,d)` instead of per
    `(t,d,n)`. DEVIATION 1072.
    """
    var m = Int(m_in)
    var l = Int(l_in)
    var di = Int(di_in)
    var leaf = Int(leaf_in)
    var p_count = Int(p_in)

    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * DSTATE:
        return
    var t = cell // DSTATE
    var n = cell - t * DSTATE
    if l < 1:
        return
    var bb = t // l
    var li = t - bb * l
    var slots = l + 1

    var stack_c = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
    var occ_c = 0
    var stack_b = SIMD[DType.float32, GEMM_FOLD_SLOTS](0.0)
    var occ_b = 0

    for tt in range(p_count):
        var bounds = _leaf_bounds(_leaf_at(tt, p_count), leaf, di)
        var acc_c = Float32(0.0)
        var acc_b = Float32(0.0)
        for d in range(bounds[0], bounds[1]):
            acc_c = ftz(
                identical_mul_add(
                    ftz(dy_ptr.unsafe_load(t * di + d)),
                    ftz(
                        hck_ptr.unsafe_load(
                            ((bb * slots + (li + 1)) * di + d) * DSTATE + n
                        )
                    ),
                    acc_c,
                )
            )
            acc_b = ftz(
                identical_mul_add(
                    ftz(w_ptr.unsafe_load(t * di + d)),
                    ftz(dh_ptr.unsafe_load((t * di + d) * DSTATE + n)),
                    acc_b,
                )
            )
        _ = _fold_push(stack_c, occ_c, ftz(acc_c))
        _ = _fold_push(stack_b, occ_b, ftz(acc_b))

    dcm_ptr.unsafe_store(cell, ftz(_fold_drain(stack_c, occ_c)))
    dbm_ptr.unsafe_store(cell, ftz(_fold_drain(stack_b, occ_b)))


def mamba_bwd_dbc_atomic_kernel[
    DSTATE: Int
](
    dcm_ptr: MutPointer[Float32, MutAnyOrigin],
    dbm_ptr: MutPointer[Float32, MutAnyOrigin],
    dy_ptr: MutPointer[Float32, MutAnyOrigin],
    hck_ptr: MutPointer[Float32, MutAnyOrigin],
    w_ptr: MutPointer[Float32, MutAnyOrigin],
    dh_ptr: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    l_in: Int32,
    di_in: Int32,
):
    """SABOTAGE ONLY (`SAB_BWD_DBDC_ATOMIC`). Upstream's answer.

    One `(token, channel)` per thread, each term added into the shared cell
    by a float `atomicAdd`. The SET of terms is the profile's; the ORDER is
    the arrival order of blocks, which is what IDENTITY_PATHS rows 1 and 2
    refuse. The output buffers must be pre-zeroed by the launcher, and the
    result is not reproducible run to run on ONE device -- which is exactly
    why only MB4 can see this arm and MB3 can pass with it armed.
    """
    var m = Int(m_in)
    var l = Int(l_in)
    var di = Int(di_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= m * di:
        return
    var t = cell // di
    var d = cell - t * di
    if l < 1:
        return
    var bb = t // l
    var li = t - bb * l
    var slots = l + 1

    var dyv = ftz(dy_ptr.unsafe_load(t * di + d))
    var wv = ftz(w_ptr.unsafe_load(t * di + d))
    comptime for n in range(DSTATE):
        var hv = ftz(
            hck_ptr.unsafe_load(
                ((bb * slots + (li + 1)) * di + d) * DSTATE + n
            )
        )
        var dhv = ftz(dh_ptr.unsafe_load((t * di + d) * DSTATE + n))
        _ = Atomic.fetch_add(
            dcm_ptr.unsafe_offset(t * DSTATE + n), ftz(pinned_mul(dyv, hv))
        )
        _ = Atomic.fetch_add(
            dbm_ptr.unsafe_offset(t * DSTATE + n), ftz(pinned_mul(wv, dhv))
        )


def mamba_bwd_dbc_into(
    ctx: DeviceContext,
    mut dCm: DeviceBuffer[DType.float32],
    mut dBm: DeviceBuffer[DType.float32],
    mut dy: DeviceBuffer[DType.float32],
    mut h_ckpt: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut dh: DeviceBuffer[DType.float32],
    batch: Int,
    seqlen: Int,
    dim: Int,
    block_size: Int = BWD_CELL_TPB,
) raises:
    """T3. `dCm` and `dBm`, each `[M, D_STATE]`. ASYNCHRONOUS.

    `contract_partition(dim)` is called HERE, on the host, and its two
    results are passed through as kernel arguments. That is gemm contract
    section 6's one-door discipline arriving in this lane: to move a leaf
    boundary you would have to change `contract_partition`, and there is
    exactly one line to look at.

    **MEMORY.** This is the GATE-SCALE spelling. `h_ckpt` and `dh` together
    are `(2*M + B) * dim * 16 * 4` bytes, which is 2.1 MB at the corpus's
    largest case and 3.2 GB for a 130M-parameter block at `L = 2048, B = 8`.
    Plan section 5.3 prices it and calls the blocked variant the largest open
    item of the lane. **When that variant is built its tile size must be
    `contract_partition(d_inner)`'s leaf boundary and never a VRAM budget**,
    or the answer becomes a function of the machine and every claim here is
    void.
    """
    if block_size < 1:
        raise Error(
            "mamba_bwd_dbc_into: block_size must be >= 1, got "
            + String(block_size)
        )
    var m = batch * seqlen
    _require(len(dCm), m * MAX_DSTATE, "dCm", "[M, 16]")
    _require(len(dBm), m * MAX_DSTATE, "dBm", "[M, 16]")
    _require(len(dy), m * dim, "dy", "[M, dim]")
    _require(len(w), m * dim, "w", "[M, dim]")
    _require(len(dh), m * dim * MAX_DSTATE, "dh", "[M, dim, 16]")
    _require(
        len(h_ckpt),
        batch * (seqlen + 1) * dim * MAX_DSTATE,
        "h_ckpt",
        "[B, L+1, dim, 16]",
    )
    if m < 1 or dim < 1:
        return

    comptime if SAB_BWD_DBDC_ATOMIC:
        if not column_has_float_atomics(TARGET_COLUMN):
            raise Error(
                "mamba_bwd_dbc_into: SAB_BWD_DBDC_ATOMIC is armed and this"
                " kernel_matrix column has no float atomicAdd"
                " (checks/kernel_matrix.mojo::column_has_float_atomics)."
                " REFUSED BY NAME rather than silently skipped: an arm that"
                " cannot run on a column has not been tested on it"
            )
        dCm.enqueue_fill(Float32(0.0))
        dBm.enqueue_fill(Float32(0.0))
        ctx.synchronize()
        comptime akern = mamba_bwd_dbc_atomic_kernel[MAX_DSTATE]
        ctx.enqueue_function[akern](
            dCm.unsafe_ptr(),
            dBm.unsafe_ptr(),
            dy.unsafe_ptr(),
            h_ckpt.unsafe_ptr(),
            w.unsafe_ptr(),
            dh.unsafe_ptr(),
            Int32(m),
            Int32(seqlen),
            Int32(dim),
            grid_dim=(_grid(m * dim, block_size), 1, 1),
            block_dim=(block_size, 1, 1),
        )
        return

    var part = contract_partition(dim)
    comptime kern = mamba_bwd_dbc_kernel[MAX_DSTATE]
    ctx.enqueue_function[kern](
        dCm.unsafe_ptr(),
        dBm.unsafe_ptr(),
        dy.unsafe_ptr(),
        h_ckpt.unsafe_ptr(),
        w.unsafe_ptr(),
        dh.unsafe_ptr(),
        Int32(m),
        Int32(seqlen),
        Int32(dim),
        Int32(part[0]),
        Int32(part[1]),
        grid_dim=(_grid(m * MAX_DSTATE, block_size), 1, 1),
        block_dim=(block_size, 1, 1),
    )


# ===========================================================================
# T4 AND T5: THE `dA` FOLD OVER `t`, THEN OVER `b`.
# DEVIATIONS 1073, 1074, 1084.
# ===========================================================================


def mamba_bwd_da_partial_kernel[
    DSTATE: Int
](
    dap_ptr: MutPointer[Float32, MutAnyOrigin],
    dh_ptr: MutPointer[Float32, MutAnyOrigin],
    delta_ptr: MutPointer[Float32, MutAnyOrigin],
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    u_ptr: MutPointer[Float32, MutAnyOrigin],
    b_ptr: MutPointer[Float32, MutAnyOrigin],
    hck_ptr: MutPointer[Float32, MutAnyOrigin],
    batch_in: Int32,
    seqlen_in: Int32,
    dim_in: Int32,
):
    """T4: `dA[d,n] = sum over t of d_arg[t,d,n] * delta[t,d]`, per `(b,d)`.

    **DESCENDING IN `t`, AND IT IS THE ONLY DESCENDING FOLD IN EITHER
    PROFILE.** Every other fold in `mojolearn.identical.mamba1.fp32.v1` and
    in this backward is ascending. The direction chosen is the one the
    reverse pass already walks; ascending would need either a second pass
    over `M * di * N` stored per-token products or a second traversal.
    `SAB_BWD_DA_ASCENDING` is the other direction and **MB4 must show it
    bites, otherwise DEVIATION 1073 is unmeasured and should be re-decided in
    favor of consistency.** Seed `+0.0`, FUSED, one `identical_mul_add` per
    term.

    `dA` IS ROUTABLE TO gemm v1 AND IS DELIBERATELY NOT ROUTED. With
    `q[t, d*N + n] = pinned_mul(d_arg, delta)` materialized at `[M, di*N]`
    this would be a ones-vector `OP_NN` at `(1, di*N, M)` whose fold order is
    v1's certified one rather than a hand-declared direction. Declined
    because `q` is a THIRD buffer of `M * di * N` floats on top of T2's `h`
    and T3's `dh`. **If a blocked variant ever makes that memory affordable
    the routed form is strictly preferable**, because it deletes a declared
    order in favor of a certified one.

    `d_arg` IS RECOMPUTED HERE, DEVIATION 1084, from the `dh` the recurrence
    already materialized. It is the same expression on the same bits as the
    copy inside `selective_scan_bwd_scan_kernel`'s A path, so the two are bit
    equal and the cost is `identical_exp` calls, not a second answer.

    T5, DEVIATION 1074: the result goes to a PRIVATE SLOT `partial[b,d,n]`
    with NO ATOMIC ANYWHERE, and a second kernel folds over `b`. Upstream
    does this with five `gpuAtomicAdd` calls, which makes its `dA` a function
    of block arrival order and therefore not reproducible run to run on one
    device. Metal check, done rather than assumed: this fold never enters
    threadgroup memory -- Metal has no threadgroup float atomics and 32 KB of
    threadgroup memory, so a fold needing scratch proportional to `d_inner`
    inside a threadgroup would not fit at any real width. Global private
    slots plus a second kernel is the spelling with no ceiling.
    """
    var batch = Int(batch_in)
    var seqlen = Int(seqlen_in)
    var dim = Int(dim_in)

    var thread_id = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if thread_id >= batch * dim:
        return
    var bb = thread_id // dim
    var d = thread_id - bb * dim
    if bb >= batch or d >= dim:
        return

    var slots = seqlen + 1

    var a_vals = SIMD[DType.float32, DSTATE](0.0)
    comptime for n in range(DSTATE):
        a_vals[n] = ftz(a_ptr.unsafe_load(d * DSTATE + n))

    var acc = SIMD[DType.float32, DSTATE](0.0)

    for step in range(seqlen):
        var li = seqlen - 1 - step
        comptime if SAB_BWD_DA_ASCENDING:
            # SABOTAGE: T4 folded the direction every other fold runs.
            li = step
        var t = bb * seqlen + li
        var dl = ftz(delta_ptr.unsafe_load(t * dim + d))

        comptime for n in range(DSTATE):
            var dhv = ftz(dh_ptr.unsafe_load((t * dim + d) * DSTATE + n))
            var d_arg: Float32
            comptime if SAB_BWD_H_SUBTRACT:
                var uv = ftz(u_ptr.unsafe_load(t * dim + d))
                var bv = ftz(b_ptr.unsafe_load(t * DSTATE + n))
                var dbb = ftz(pinned_mul(dl, bv))
                var dbu_h = ftz(pinned_mul(dbb, uv))
                var hcur = ftz(
                    hck_ptr.unsafe_load(
                        ((bb * slots + (li + 1)) * dim + d) * DSTATE + n
                    )
                )
                var arec = ftz(ftz(hcur) - ftz(dbu_h))
                d_arg = ftz(pinned_mul(dhv, arec))
            else:
                var da_arg = ftz(pinned_mul(dl, a_vals[n]))
                var da_t: Float32
                comptime if SAB_S5_EXP2:
                    da_t = ftz(
                        exp2(
                            ftz(
                                pinned_mul(
                                    da_arg, Float32(1.4426950408889634)
                                )
                            )
                        )
                    )
                else:
                    da_t = ftz(identical_exp(da_arg))
                var hprev = ftz(
                    hck_ptr.unsafe_load(
                        ((bb * slots + li) * dim + d) * DSTATE + n
                    )
                )
                var d_da = ftz(pinned_mul(dhv, hprev))
                d_arg = ftz(pinned_mul(d_da, da_t))

            acc[n] = ftz(identical_mul_add(d_arg, dl, acc[n]))

    comptime if SAB_BWD_PARAM_ATOMIC:
        # SABOTAGE: `dap_ptr` is the FINAL `[di, N]` buffer and the batch
        # fold happens by float atomicAdd instead of by private slots plus a
        # second kernel. The launcher pre-zeroes it and skips the fold. Only
        # MB4 can see this arm; MB3 can pass with it armed.
        comptime for n in range(DSTATE):
            _ = Atomic.fetch_add(dap_ptr.unsafe_offset(d * DSTATE + n), acc[n])
        return

    comptime for n in range(DSTATE):
        dap_ptr.unsafe_store((bb * dim + d) * DSTATE + n, acc[n])


def mamba_bwd_param_fold_kernel(
    out_ptr: MutPointer[Float32, MutAnyOrigin],
    partial_ptr: MutPointer[Float32, MutAnyOrigin],
    batch_in: Int32,
    w_in: Int32,
):
    """T5's second kernel: fold `partial[b, w]` over `b` ASCENDING from
    `+0.0` with a plain flushed add. DEVIATION 1074.

    One thread per output element, so the fold never leaves a thread's
    registers and no launch geometry can reorder it. There is no atomic, no
    shared memory and no tree: the term count is the batch size, which is a
    caller-declared quantity, and gemm contract 6.1's rule that a crossing
    fold's partition may be a pure function of the length of the axis it
    reduces and of nothing else is satisfied trivially by a serial chain.

    **THE `+0.0` SEED LAUNDERS A NEGATIVE ZERO FIRST TERM**, and that is the
    declared behavior rather than an oversight: `T5_SEED_IS_POSITIVE_ZERO` is
    True, and it is the same trade forward seams S1 and S10 make, where the
    contract's section 6 blesses it so that an all-zero row sums to `+0.0` on
    every vendor. At `B == 1` this fold is one addition of `+0.0` to one
    value, which is the identity on everything except a `-0.0`.

    **THIS FOLD'S ONLY SITE TODAY IS `dA`.** Plan section 3.4 says T5 is
    shared by `dA`, `dD`, `db_dt`, `dcw` and `dcb`, and that sentence does
    not survive contact with the plan's own section 3.2: those four other
    parameters are category (b), routed to ones-vector v1 GEMMs at `k' = M`,
    and `M = B * L` already folds the batch. Only `dA` needs T5, because only
    `dA` declines the routing (T4). The kernel is written width-generic
    anyway so a future unrouted parameter has a home, and the plan text is
    corrected in the same commit as this file.
    """
    var w = Int(w_in)
    var nb = Int(batch_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= w:
        return
    var acc = Float32(0.0)
    for bb in range(nb):
        acc = ftz(ftz(acc) + ftz(partial_ptr.unsafe_load(bb * w + i)))
    out_ptr.unsafe_store(i, acc)


def mamba_bwd_da_into(
    ctx: DeviceContext,
    mut dA: DeviceBuffer[DType.float32],
    mut dA_partial: DeviceBuffer[DType.float32],
    mut dh: DeviceBuffer[DType.float32],
    mut delta: DeviceBuffer[DType.float32],
    mut A: DeviceBuffer[DType.float32],
    mut u: DeviceBuffer[DType.float32],
    mut B: DeviceBuffer[DType.float32],
    mut h_ckpt: DeviceBuffer[DType.float32],
    batch: Int,
    seqlen: Int,
    dim: Int,
    block_size: Int = BWD_SCAN_TPB,
) raises:
    """T4 then T5: `dA [dim, D_STATE]` through `dA_partial [B, dim, D_STATE]`.

    TWO LAUNCHES, and the second one is the price DEVIATION 1074 pays to
    refuse an atomic. They are enqueued on ONE `DeviceContext` and MAX runs
    them in order, so the fold cannot start before every partial is written.
    **It is not safe across two contexts or two streams**, and a second
    stream would be a change to this function's contract.

    ASYNCHRONOUS, caller-owned buffers.
    """
    if block_size < 1:
        raise Error(
            "mamba_bwd_da_into: block_size must be >= 1, got "
            + String(block_size)
        )
    var m = batch * seqlen
    _require(len(dA), dim * MAX_DSTATE, "dA", "[dim, 16]")
    _require(len(dh), m * dim * MAX_DSTATE, "dh", "[M, dim, 16]")
    _require(len(delta), m * dim, "delta", "[M, dim]")
    _require(len(u), m * dim, "u", "[M, dim]")
    _require(len(A), dim * MAX_DSTATE, "A", "[dim, 16]")
    _require(len(B), m * MAX_DSTATE, "B", "[M, 16]")
    _require(
        len(h_ckpt),
        batch * (seqlen + 1) * dim * MAX_DSTATE,
        "h_ckpt",
        "[B, L+1, dim, 16]",
    )

    var total = batch * dim
    if total < 1:
        return

    comptime if SAB_BWD_PARAM_ATOMIC:
        if not column_has_float_atomics(TARGET_COLUMN):
            raise Error(
                "mamba_bwd_da_into: SAB_BWD_PARAM_ATOMIC is armed and this"
                " kernel_matrix column has no float atomicAdd"
                " (checks/kernel_matrix.mojo::column_has_float_atomics)."
                " REFUSED BY NAME rather than silently skipped"
            )
        dA.enqueue_fill(Float32(0.0))
        ctx.synchronize()
        comptime akern = mamba_bwd_da_partial_kernel[MAX_DSTATE]
        ctx.enqueue_function[akern](
            dA.unsafe_ptr(),
            dh.unsafe_ptr(),
            delta.unsafe_ptr(),
            A.unsafe_ptr(),
            u.unsafe_ptr(),
            B.unsafe_ptr(),
            h_ckpt.unsafe_ptr(),
            Int32(batch),
            Int32(seqlen),
            Int32(dim),
            grid_dim=(_grid(total, block_size), 1, 1),
            block_dim=(block_size, 1, 1),
        )
        return

    _require(
        len(dA_partial),
        batch * dim * MAX_DSTATE,
        "dA_partial",
        "[B, dim, 16]",
    )
    comptime kern = mamba_bwd_da_partial_kernel[MAX_DSTATE]
    ctx.enqueue_function[kern](
        dA_partial.unsafe_ptr(),
        dh.unsafe_ptr(),
        delta.unsafe_ptr(),
        A.unsafe_ptr(),
        u.unsafe_ptr(),
        B.unsafe_ptr(),
        h_ckpt.unsafe_ptr(),
        Int32(batch),
        Int32(seqlen),
        Int32(dim),
        grid_dim=(_grid(total, block_size), 1, 1),
        block_dim=(block_size, 1, 1),
    )
    var width = dim * MAX_DSTATE
    ctx.enqueue_function[mamba_bwd_param_fold_kernel](
        dA.unsafe_ptr(),
        dA_partial.unsafe_ptr(),
        Int32(batch),
        Int32(width),
        grid_dim=(_grid(width, BWD_CELL_TPB), 1, 1),
        block_dim=(BWD_CELL_TPB, 1, 1),
    )


def mamba_bwd_param_fold_into(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut partial: DeviceBuffer[DType.float32],
    batch: Int,
    width: Int,
) raises:
    """T5 on its own, for a parameter that is not `dA`. ASYNCHRONOUS.

    Nothing in this lane calls it today (see `mamba_bwd_param_fold_kernel`'s
    note on the plan's over-broad claim about T5's sharing). It is exported
    so that a future unrouted parameter gradient has one fold to use rather
    than a second one to declare.
    """
    _require(len(out), width, "out", "[W]")
    _require(len(partial), batch * width, "partial", "[B, W]")
    if width < 1:
        return
    ctx.enqueue_function[mamba_bwd_param_fold_kernel](
        out.unsafe_ptr(),
        partial.unsafe_ptr(),
        Int32(batch),
        Int32(width),
        grid_dim=(_grid(width, BWD_CELL_TPB), 1, 1),
        block_dim=(BWD_CELL_TPB, 1, 1),
    )


# ===========================================================================
# B21: `dA_log = pinned_mul(dA, A)`. SEAM S15's BACKWARD.
# ===========================================================================


def mamba_bwd_da_log_kernel(
    dalog_ptr: MutPointer[Float32, MutAnyOrigin],
    da_ptr: MutPointer[Float32, MutAnyOrigin],
    a_ptr: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """S15 is `A = -exp(A_log)`, so `dA_log = dA * A`. ONE `pinned_mul`.

    `A` is READ from the recorded stage `A.out`, never recomputed as
    `-identical_exp(A_log)`. The two are bit equal -- `identical_exp` is a
    pure function of bits the card already holds -- so this is not an
    accuracy choice; it is one call instead of two and one place for a
    recompute to drift instead of two. Plan section 3.5 lists the recompute
    as the alternative and calls it "bit equal but a second call".
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    dalog_ptr.unsafe_store(
        i,
        ftz(
            pinned_mul(ftz(da_ptr.unsafe_load(i)), ftz(a_ptr.unsafe_load(i)))
        ),
    )


def mamba_bwd_da_log_into(
    ctx: DeviceContext,
    mut dA_log: DeviceBuffer[DType.float32],
    mut dA: DeviceBuffer[DType.float32],
    mut A: DeviceBuffer[DType.float32],
    dim: Int,
) raises:
    """B21. `dA_log [dim, D_STATE]`. ASYNCHRONOUS."""
    var n = dim * MAX_DSTATE
    _require(len(dA_log), n, "dA_log", "[dim, 16]")
    _require(len(dA), n, "dA", "[dim, 16]")
    _require(len(A), n, "A", "[dim, 16]")
    if n < 1:
        return
    ctx.enqueue_function[mamba_bwd_da_log_kernel](
        dA_log.unsafe_ptr(),
        dA.unsafe_ptr(),
        A.unsafe_ptr(),
        Int32(n),
        grid_dim=(_grid(n, BWD_CELL_TPB), 1, 1),
        block_dim=(BWD_CELL_TPB, 1, 1),
    )


# ===========================================================================
# SIZING HELPERS
# ===========================================================================
# The two buffers plan section 5.3 prices, so a caller can allocate them
# without re-deriving the shape. `mamba/checks/mamba_backward.mojo` carries
# the same two numbers as `t2_h_checkpoint_floats` and `t3_dh_floats`; these
# are spelled in terms of `dim` rather than `MambaDims` because this file
# knows nothing about a block.


def bwd_h_checkpoint_floats(batch: Int, seqlen: Int, dim: Int) -> Int:
    """`B * (L+1) * dim * D_STATE`. The `+1` is `h[-1]`, slot 0."""
    return batch * (seqlen + 1) * dim * MAX_DSTATE


def bwd_dh_floats(batch: Int, seqlen: Int, dim: Int) -> Int:
    """`M * dim * D_STATE`. No carry slot; T1's seed is structural."""
    return batch * seqlen * dim * MAX_DSTATE


def bwd_da_partial_floats(batch: Int, dim: Int) -> Int:
    """`B * dim * D_STATE`, T5's private slots. Not needed when
    `SAB_BWD_PARAM_ATOMIC` is armed, which is the point of that arm."""
    return batch * dim * MAX_DSTATE


def _require(got: Int, want: Int, name: String, shape: String) raises:
    """A buffer that is the wrong size is a CALLER bug, and it is caught by
    name here rather than by an out-of-bounds device read that returns
    plausible bits. The forward scan file's own helper, same spelling."""
    if got < want:
        raise Error(
            "selective_scan_backward: buffer '"
            + name
            + "' holds "
            + String(got)
            + " floats, needs "
            + String(want)
            + " for "
            + shape
        )
