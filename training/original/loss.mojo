# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device cross-entropy of profile `mojolearn.identical.loss.ce.fp32.v1`.

**THIS BANNER WAS FALSE AND IS CORRECTED. COMPILED, RUN AND GATED ON ONE
DEVICE.** Until 2026-08-31 this header read "THIS FILE HAS NEVER BEEN COMPILED
AND HAS NEVER BEEN EXECUTED", and added that no GPU had run a kernel from it,
no gate had ever failed against it, and no sabotage arm had ever been built.
**Commit `ecd1a436` is the first execution of this file together with
`loss_check.mojo`, `loss_fixture.mojo` and `loss_oracle.mojo`**, and it
falsified every clause: six clauses passed, 24 cases and 61,925 cells matched
the oracle BITWISE, and clause (f) MEASURED a real defect here, that this file
never called `ce_refuse_inputs` so a planted NaN reached 40 recorded cells. The
refusal was added at `b90f52ab` (DEVIATION 1495) and calls the oracle's own
function. Written by the training lane, DEVIATIONS 1150-1169. The contract is
`training/IDENTICAL_LOSS_CONTRACT.md`; the answer is
`training/original/loss_oracle.mojo`, bit for bit.

**WHAT IS STILL UNPAID: THE OTHER TWO VENDORS.** Everything above was measured
on one device.

WHAT IS OWED
------------
  - ~~`training/original/loss_check.mojo` and
    `training/original/loss_fixture.mojo` DO NOT EXIST, so every sabotage
    switch below has never been compiled, let alone shown to fail a gate.~~
    **Both exist and both ran, `ecd1a436`.** The sabotage arms were compiled
    and fired there, and firing them is what corrected DEVIATION 1490, where
    the assertion about `L_MAX_SEED_ZERO`'s inert case had the claim inverted.
    A switch that has never fired is still a comment; these have fired on one
    device and on no other.
  - `pinned_block_fmax` below is a LOCAL fold (DEVIATION 1165). The clean fix
    is to give `core/pinned_reduce.mojo::pinned_block_max` `identical_fmax`
    as its combine step, which is `portable_fmaxf`'s own suggestion and that
    file's owner's call. The transformer lane's S14 wants the identical edit,
    so two lanes now want it once. Contract OWED item 3.
  - `neg_by_bits` and `refuse_nonfinite` are imported from
    `loss_oracle.mojo` rather than copied, but both belong in
    `original/numerics.mojo`. Contract OWED items 1 and 2.
  - No card is emitted from here. `core/identity_trace.mojo` recording is the
    check file's job and the check file does not exist.
  - No performance number exists and contract section 11 forbids quoting one.

ONE SOURCE, THREE VENDORS, AND NO `if apple` ANYWHERE
------------------------------------------------------
`[[always-gpu-agnostic]]`. There is not one branch on a vendor in this file
and there is not one place where there could be. The only place a column is
consulted at all is `_ce_max_tpb`, which resolves a BLOCK SIZE, and the block
size is a SCHEDULING row here for a reason that is structural rather than
promised:

  1. **The row maximum's fold shape is FREE.** `identical_fmax` is
     commutative and associative over all of Float32 including both zeros and
     NaN (`portable_fmaxf`'s docstring, DEVIATION 825), so a halving fold of
     any width returns the same bits. Contract 5.1.
  2. **Every other reduction in this profile is ROUTED to gemm v1**, whose
     `(L, P)` come from the contraction length and the two profile constants
     and from nothing else (`contract_leaf_size`, gemm contract section 6).
     No block size, no grid shape, no occupancy and no lane width can reach
     them. Contract 5.3 and 7.2.
  3. **Everything else is elementwise.** One thread owns one cell, reads it,
     and writes it. There is no cross-thread float anywhere outside (1).

So `original/kernel_matrix.mojo` is consulted for the one thing it is for --
`column_max_block_size`, the vendor's dispatch cap, which the portable
baseline column exposed as a real limit on 2026-08-21 -- and for
`IDENTITY_FLOOR_BLOCK`, so that a `NUMERIC_IDENTICAL` build launches ONE
geometry on every vendor the way `block_size_for` does. DEVIATION 1168. A
vendor divergence in this lane would be a kernel-matrix ROW, never an inline
branch, and today there is no such row because there is no such divergence.

WHAT `NUMERIC_FAST` DOES HERE
------------------------------
Everything, with the pins compiled away. `ftz` becomes the identity,
`identical_exp` / `identical_log` / `identical_div` / `identical_fmax` become
the stdlib's device paths, and `identical_mul` becomes a plain product. The
kernels, the launches, the routing and the geometry are UNCHANGED -- so FAST
is a correct cross-entropy, it is the same code on the same path (not dead
code and not a separate kernel), and it makes no identity claim at all.

`[[mojo-buffer-freed-at-last-use]]`: every launcher here is the `_into` form.
It enqueues and returns. The CALLER owns every buffer -- including the ones
vector and the workspace -- and must keep every one of them alive past its
own `ctx.synchronize()`. A buffer created in a caller's frame is dead at its
`.unsafe_ptr()`, so a training step that allocates a workspace, calls one of
these, and returns without waiting has freed the workspace before the kernel
ran. There is deliberately no synchronizing form, because a training step
chains many of these and one wait per stage is the wrong shape.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import exp, log
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

# `exp` and `log` above are the STDLIB's, imported at module scope for the
# `SAB_EXP_STDLIB` and `SAB_LOG_STDLIB` arms and used on NO normative path.
# They are imported here rather than inside the `comptime if` bodies that use
# them because `original/numerics.mojo`'s own two-arm functions put their
# stdlib import at FUNCTION scope after the comptime branch, and an import
# INSIDE a comptime-if body is a spelling nothing in this tree has compiled.
from original.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_log,
    identical_mul,
    identical_mul_add,
)
from original.kernel_matrix import (
    IDENTITY_FLOOR_BLOCK,
    TARGET_COLUMN,
    column_max_block_size,
)
from gemm.original.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.original.gemm_oracle import OP_NN
from training.original.loss_oracle import (
    CE_NEG_INF_BITS,
    CeConfig,
    REDUCTION_NONE,
    ce_divisor,
    ce_one_minus_eps,
    ce_refuse_inputs,
    ce_smoothing_targets,
    neg_by_bits,
)


# ===========================================================================
# THE SABOTAGE SWITCHES (contract 10.1)
# ===========================================================================
# OFF in every build that does not name them, exactly as `gemm_identical.mojo`
# and `gemm_backward.mojo` do it. Each one is a specific way to get a clause
# wrong that a plausible implementation could reach by accident, and each
# exists so a gate can be SHOWN to fail. A gate that has never failed is a
# gate nobody has tested.
#
# EVERY SWITCH CARRIES ITS PREDICTED **INERT** SET, and the check must assert
# the inert set as a mask rather than merely observing that the arm moved
# something. `IDENTICAL_BACKWARD_PLAN.md` section 5.0 is the discipline: two
# of its six routes are inert under `BWD_UNTRANSPOSED` for structural
# reasons, and a gate that reported a 4-of-6 result as a 6-of-6 one would be
# `[[reached-but-inert]]` in its purest form.
#
# Build with, e.g.:
#     tools/with_identical_mode.sh pixi run mojo run \
#         -D MOJOLEARN_LOSS_SABOTAGE_MAX_PLAIN_COMPARE=1 \
#         -I . training/original/loss_check.mojo
#
# FOUR CLAUSES ARE **NOT** SABOTAGED HERE AND MUST BE REACHED THROUGH
# `gemm_identical.mojo`'s OWN SIX SWITCHES, because this file routes its
# folds rather than spelling them and a second copy of a sabotage is a second
# thing that can be wrong:
#     L_DENOM_HALVING_TREE     -> MOJOLEARN_GEMM_SABOTAGE_FOLD_STRIDE
#     L_DENOM_PAD_PLUS_ZERO    -> MOJOLEARN_GEMM_SABOTAGE_PAD_PLUS_ZERO
#     L_VOCAB_FOLD_READS_LAUNCH-> MOJOLEARN_GEMM_SABOTAGE_LEAF_READS_LAUNCH
#     (the leaf rotation)      -> MOJOLEARN_GEMM_SABOTAGE_LEAF_ROTATE
# Showing those six fail THROUGH this lane's entry points is the proof that
# the routing lands on the contract's arithmetic rather than on some other
# path that happens to agree. Contract 7.2.

#: Contract 5.1. The row maximum folds with `a > b ? a : b` instead of
#: `identical_fmax`. **INERT on every row that does not carry both zero
#: signs**, because a plain compare and a total-order selection agree at
#: every other input. The gate must plant a `-0.0` beside a `+0.0` AND must
#: fold an ordinary row and show the arm inert there, or it is a smoke test.
comptime SAB_MAX_PLAIN_COMPARE = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_MAX_PLAIN_COMPARE"
]()
#: Contract 5.1. The fold seeds `+0.0` instead of `-inf`. **INERT on any row
#: containing a positive logit**, and catastrophic on an all-negative row --
#: which is most rows of a trained head. Note that it is wrong on EVERY
#: vendor identically, so bit-identity cannot see it and only the oracle can.
comptime SAB_MAX_SEED_ZERO = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_MAX_SEED_ZERO"
]()
#: Contract 5.2. The maximum is taken over the first 32 classes only.
#: **INERT on a row whose maximum is inside the first 32.** The fixture must
#: put an argmax in the tail.
comptime SAB_MAX_TOPK_PREFIX = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_MAX_TOPK_PREFIX"
]()
#: Contract section 4, seam L3. `std.math.exp` instead of `identical_exp`.
#: INERT under `NUMERIC_FAST` (where `identical_exp` IS the stdlib) and inert
#: at `shift == +0.0`, which is the whole exact-fixture family of contract
#: 12.1 -- so this arm must run on an ORDINARY fixture.
comptime SAB_EXP_STDLIB = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_EXP_STDLIB"
]()
#: Contract section 4, seam L5. `std.math.log` instead of `identical_log`.
#: `[[mojo-log-breaks-ties]]`: the stdlib log carries about `5e-8` absolute
#: error and re-decides plateau ties, which is why row 12 exists at all.
#: INERT under FAST, and inert at `denom == 1.0` where both return `+0.0`.
comptime SAB_LOG_STDLIB = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_LOG_STDLIB"
]()
#: Contract 4.2(a). `nll = (m + logdenom) - x_y` instead of
#: `logdenom - shift[y]`. **INERT on a centered row** -- logits in `[-1, 1]`
#: give the add-back nothing to lose. The separating fixture adds a large
#: common offset (`+1e6`) to every logit of a row, where the pinned spelling
#: does not move a bit and this one moves many.
comptime SAB_NLL_VIA_ADDBACK = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_NLL_VIA_ADDBACK"
]()
#: Contract 4.2(b). `nll = -log(w[y])`. **INERT wherever `e[y]` is normal.**
#: The separating fixture puts `shift[y]` below `-87.33655`, where
#: `portable_expf` returns exactly `+0.0`, so this arm returns `+inf` and the
#: pinned spelling returns an ordinary finite number near 87.
comptime SAB_NLL_VIA_LOG_W = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_NLL_VIA_LOG_W"
]()
#: Contract 4.3. The negation spelled `0.0 - x`. **INERT at every input
#: except `x == +0.0`**, where IEEE negation gives `-0.0` and this gives
#: `+0.0`. Contract 8.1 shows `lp_y == +0.0` is reachable at `V == 1` and
#: wherever every non-target exponential underflows. A fixture without such a
#: row makes this arm look broken.
comptime SAB_NEG_VIA_ZERO_SUB = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_NEG_VIA_ZERO_SUB"
]()
#: Contract 6.2(a). `smooth` multiplied by a host-folded `eps / V` constant
#: instead of divided by `V` and then multiplied by `eps`. **INERT at
#: `eps == 0`** (where the smoothing arm is not spelled at all).
comptime SAB_SMOOTH_FOLDED_CONSTANT = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_SMOOTH_FOLDED_CONSTANT"
]()
#: Contract 6.2(b). The combine fused into one `identical_mul_add`. **INERT
#: at `eps == 0`.**
comptime SAB_SMOOTH_FUSED_COMBINE = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_SMOOTH_FUSED_COMBINE"
]()
#: Contract 6.2(c), **the strongest reach-per-branch arm in this lane.** The
#: smoothing path is spelled even at `eps == 0`. **INERT at every `eps != 0`
#: AND at every row whose `nll` is not `-0.0`**, because
#: `ftz(nll + (+0.0)) == nll` everywhere else. It moves exactly one thing:
#: a `-0.0` row loss laundered to `+0.0`. Without a `V == 1` row in the
#: fixture it looks inert and somebody deletes it.
comptime SAB_SMOOTH_ALWAYS_SPELLED = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_SMOOTH_ALWAYS_SPELLED"
]()
#: Contract 5.5, seam L13. `total * (1/divisor)`. **INERT when the divisor is
#: an exact power of two**, where the reciprocal is exact and the two
#: spellings agree. The gate must use a divisor that is not -- `count = 3` is
#: the cheapest one.
comptime SAB_MEAN_RECIPROCAL_MUL = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_MEAN_RECIPROCAL_MUL"
]()
#: Contract 7.3. An ignored row writes `-0.0` instead of `+0.0`. **It MUST
#: move `ce.row` and it MUST NOT move `ce.total`**, because L12's leaf
#: accumulator is seeded `+0.0` and `(+0.0) + (-0.0)` is `+0.0` in
#: round-to-nearest. A gate that compares only the final loss calls this
#: inert, and it is not -- it is a divergence at the stage that produced it.
#: This is transformer 5.1's "the card is the only instrument that can see
#: this clause at all" at a second site.
comptime SAB_IGNORED_ROW_NEG_ZERO = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_IGNORED_ROW_NEG_ZERO"
]()
#: Contract 7.3. An ignored row's stores are SKIPPED rather than written.
#: Never inert if the gate pre-fills the buffers with a poison pattern, and
#: ALWAYS inert if it pre-fills them with zeros -- which is what a fresh
#: allocation may or may not contain. The gate must poison.
comptime SAB_IGNORED_ROW_SKIPPED = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_IGNORED_ROW_SKIPPED"
]()
#: Contract 6.4, seam L14. `w = exp(logp)` instead of `e / denom`. Requires
#: the smoothing path's `logp` buffer, so under this arm the backward
#: demands it. Never inert on an ordinary row.
comptime SAB_W_VIA_EXP_LOGP = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_W_VIA_EXP_LOGP"
]()
#: Contract 6.4, seam L16. `t - w` instead of `w - t`. **INERT only on a row
#: where `w == t` at every class**, which the uniform exact fixture of
#: contract 12.1 reaches at `V == 1`. Everywhere else it flips every sign.
comptime SAB_GRAD_SIGN = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_GRAD_SIGN"
]()
#: Contract 6.4, seam L15. The target vector is applied at `y + 1` mod V.
#: **INERT at `V == 1`.** At `V > 1` it moves exactly two cells per row,
#: which is why the check must compare CELLS and not a row norm.
comptime SAB_GRAD_TARGET_OFF_BY_ONE = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_GRAD_TARGET_OFF_BY_ONE"
]()
#: Contract 5.5. The backward divides by `Float32(N)` instead of by the
#: divisor `ce_divisor` produced. **INERT on any fixture with no ignored row
#: under MEAN**, and inert on every SUM. It is the falsifier for "the divisor
#: has exactly one producer" and it needs a fixture built for it.
comptime SAB_GRAD_DIVISOR_IS_N = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_GRAD_DIVISOR_IS_N"
]()
#: Contract 6.4, seam L16. `d * (1/divisor)`. **INERT when the divisor is an
#: exact power of two**, exactly as `SAB_MEAN_RECIPROCAL_MUL` is.
comptime SAB_GRAD_RECIPROCAL_MUL = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_GRAD_RECIPROCAL_MUL"
]()
#: Contract 5.3. The vocabulary denominator is folded by a hand-written
#: WHOLE-AXIS ASCENDING CHAIN instead of routed to gemm v1. This is the
#: spelling `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 5.3 pins for ITS
#: softmax, so this arm is also the instrument that PRINTS the difference
#: between the two lanes rather than arguing about it.
#: **INERT at every `V <= 128`**, where `P == 1` and the tree performs no
#: addition (gemm contract 7.3). `V >= 129` is mandatory.
comptime SAB_DENOM_SERIAL_CHAIN = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_DENOM_SERIAL_CHAIN"
]()
#: Contract 5.4. The batch reduction folded by the same hand-written chain.
#: **INERT at every `N <= 128`.**
comptime SAB_REDUCE_SERIAL = is_defined[
    "MOJOLEARN_LOSS_SABOTAGE_REDUCE_SERIAL"
]()

comptime ANY_LOSS_SABOTAGE = (
    SAB_MAX_PLAIN_COMPARE
    or SAB_MAX_SEED_ZERO
    or SAB_MAX_TOPK_PREFIX
    or SAB_EXP_STDLIB
    or SAB_LOG_STDLIB
    or SAB_NLL_VIA_ADDBACK
    or SAB_NLL_VIA_LOG_W
    or SAB_NEG_VIA_ZERO_SUB
    or SAB_SMOOTH_FOLDED_CONSTANT
    or SAB_SMOOTH_FUSED_COMBINE
    or SAB_SMOOTH_ALWAYS_SPELLED
    or SAB_MEAN_RECIPROCAL_MUL
    or SAB_IGNORED_ROW_NEG_ZERO
    or SAB_IGNORED_ROW_SKIPPED
    or SAB_W_VIA_EXP_LOGP
    or SAB_GRAD_SIGN
    or SAB_GRAD_TARGET_OFF_BY_ONE
    or SAB_GRAD_DIVISOR_IS_N
    or SAB_GRAD_RECIPROCAL_MUL
    or SAB_DENOM_SERIAL_CHAIN
    or SAB_REDUCE_SERIAL
)


def loss_sabotage_name() -> String:
    """Which sabotage this binary compiled with, for a check's banner.

    A check MUST print this AND `gemm_identical.mojo::gemm_sabotage_name` AND
    `gemm_backward.mojo::gemm_backward_sabotage_name`, because a loss binary
    can carry a GEMM sabotage as well -- four of this lane's clauses are
    reached that way on purpose -- and a banner that names only one of them
    mislabels the run. That is `gemm_backward_sabotage_name`'s own warning
    with a third file added to it.
    """
    comptime if SAB_MAX_PLAIN_COMPARE:
        return String("MAX_PLAIN_COMPARE")
    comptime if SAB_MAX_SEED_ZERO:
        return String("MAX_SEED_ZERO")
    comptime if SAB_MAX_TOPK_PREFIX:
        return String("MAX_TOPK_PREFIX")
    comptime if SAB_EXP_STDLIB:
        return String("EXP_STDLIB")
    comptime if SAB_LOG_STDLIB:
        return String("LOG_STDLIB")
    comptime if SAB_NLL_VIA_ADDBACK:
        return String("NLL_VIA_ADDBACK")
    comptime if SAB_NLL_VIA_LOG_W:
        return String("NLL_VIA_LOG_W")
    comptime if SAB_NEG_VIA_ZERO_SUB:
        return String("NEG_VIA_ZERO_SUB")
    comptime if SAB_SMOOTH_FOLDED_CONSTANT:
        return String("SMOOTH_FOLDED_CONSTANT")
    comptime if SAB_SMOOTH_FUSED_COMBINE:
        return String("SMOOTH_FUSED_COMBINE")
    comptime if SAB_SMOOTH_ALWAYS_SPELLED:
        return String("SMOOTH_ALWAYS_SPELLED")
    comptime if SAB_MEAN_RECIPROCAL_MUL:
        return String("MEAN_RECIPROCAL_MUL")
    comptime if SAB_IGNORED_ROW_NEG_ZERO:
        return String("IGNORED_ROW_NEG_ZERO")
    comptime if SAB_IGNORED_ROW_SKIPPED:
        return String("IGNORED_ROW_SKIPPED")
    comptime if SAB_W_VIA_EXP_LOGP:
        return String("W_VIA_EXP_LOGP")
    comptime if SAB_GRAD_SIGN:
        return String("GRAD_SIGN")
    comptime if SAB_GRAD_TARGET_OFF_BY_ONE:
        return String("GRAD_TARGET_OFF_BY_ONE")
    comptime if SAB_GRAD_DIVISOR_IS_N:
        return String("GRAD_DIVISOR_IS_N")
    comptime if SAB_GRAD_RECIPROCAL_MUL:
        return String("GRAD_RECIPROCAL_MUL")
    comptime if SAB_DENOM_SERIAL_CHAIN:
        return String("DENOM_SERIAL_CHAIN")
    comptime if SAB_REDUCE_SERIAL:
        return String("REDUCE_SERIAL")
    return String("none")


# ===========================================================================
# THE ONE SCHEDULING ROW (DEVIATION 1168)
# ===========================================================================

#: SCHEDULING. What this lane would like a block to be, before the column's
#: cap and the identity floor are applied. 256 is `FLAT_TPB`'s value in
#: `gemm_identical.mojo` and is chosen for the same non-reason: it is what
#: fits, it has never been measured here, and **do not read it as evidence
#: that it is optimal anywhere.** `lib_block_size`'s docstring makes exactly
#: this disclaimer about its own uniform rows.
comptime CE_TPB_WANT = 256


def _ce_max_tpb[column: Int]() -> Int:
    """Threads per block, resolved the way
    `original/kernel_matrix.mojo::block_size_for` resolves its own.

    SCHEDULING, and the module docstring's three structural facts are why: it
    changes which thread does which work and never what is added to what.

    Two bounds, both from the kernel matrix and neither from a vendor branch:

      - **the identity floor.** Under `NUMERIC_IDENTICAL` the block is capped
        at `IDENTITY_FLOOR_BLOCK` so every column launches ONE geometry, the
        gate `block_size_for` grew on 2026-08-22 after an audit found
        IDENTITY_PATHS row 3 closed at the runtime REPORT and open at the
        comptime accessor the kernels actually compile against. This lane
        does not strictly need it -- the max fold is associative and
        everything else is elementwise -- and it is applied anyway, because
        "this particular fold does not care" is exactly the reasoning that
        put the hole in `block_size_for` in the first place.
      - **the vendor's dispatch cap**, `column_max_block_size`. Slack on all
        three founding columns (their caps are 1024), and NOT slack on the
        portable-baseline column, which guarantees only 128 invocations per
        workgroup and exposed the omission on its first run.

    `[[mojotrees-code-not-source-of-truth]]`: none of this has been measured.
    """
    comptime want = CE_TPB_WANT
    comptime identical = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    comptime floored = (
        IDENTITY_FLOOR_BLOCK if identical
        and IDENTITY_FLOOR_BLOCK < want else want
    )
    comptime hard = column_max_block_size(column)
    return floored if floored < hard else hard


#: The resolved block. Every kernel below is launched at exactly this width
#: and `pinned_block_fmax` is instantiated at it.
comptime CE_TPB = _ce_max_tpb[TARGET_COLUMN]()


# ===========================================================================
# THE ONE CROSS-THREAD FLOAT OPERATION IN THIS FILE (DEVIATION 1165)
# ===========================================================================


@always_inline
def pinned_block_fmax[block_size: Int](value: Float32) -> Float32:
    """A block-wide maximum whose combine step is `identical_fmax`.

    Contract 5.1. **DEVIATION 1165, and it is a DEBT rather than a design.**
    `core/pinned_reduce.mojo::pinned_block_max` already has this shape and
    the wrong combine -- a plain `other > red[tid]` compare (:159-190), which
    is precisely the spelling IDENTITY_PATHS row 13 closed everywhere else --
    and its own block comment (:145-155) says a caller whose inputs can carry
    `+-0.0` or NaN must state why before using it. **An attention or
    cross-entropy row cannot say why**, which is `portable_fmaxf`'s docstring
    verbatim. That docstring also names the fix: *"The clean fix is to give
    that fold this function as its combine step, which is the block's owner's
    call and not this file's."* The transformer lane's S14 wants the same
    edit. Contract OWED item 3.

    THE CONTRACT, and it is `pinned_block_sum`'s own:
      - EVERY thread of the block must call this. Threads with no data pass
        `-inf` (`_ce_neg_inf()`), which is the true identity of
        `identical_fmax` on a NaN-free row -- `_total_order_key(-inf)` is
        `0x007FFFFF`, strictly below the key of every finite value, and
        contract section 8 refuses a NaN input.
      - Only thread 0's return value is meaningful.
      - `block_size` must be a power of two.
      - The trailing `barrier()` protects the slab so back-to-back calls in
        one kernel are safe.

    **NO MODE GATE, and that is a real difference from `pinned_block_sum`.**
    `identical_fmax` is itself mode-gated, so this fold inherits both arms
    from it. Under FAST the combine is the stdlib `max`, whose signed-zero
    answer is the vendor's own (row 39 measured `-0.0` on Apple against
    `+0.0` on NVIDIA and AMD), and the FAST arm therefore makes no claim.

    **THE FOLD SHAPE IS FREE AND THE HALVING TREE HERE IS NOT PART OF THE
    CONTRACT.** A two-pass launch, an atomic, a warp shuffle or a different
    width would all be legal and would all return the same bits, because the
    combine is exactly commutative and associative. Contract clause (d)'s
    gate is what turns that from a claim into a measurement, and it is not
    written.
    """
    var tid = Int(thread_idx.x)
    var red = stack_allocation[
        block_size,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    red[tid] = value
    barrier()
    var step = block_size // 2
    while step > 0:
        if tid < step:
            var other = red[tid + step]
            comptime if SAB_MAX_PLAIN_COMPARE:
                # SABOTAGE: the spelling row 13 closed. INERT unless the row
                # carries both zero signs.
                if other > red[tid]:
                    red[tid] = other
            else:
                red[tid] = identical_fmax(red[tid], other)
        barrier()
        step //= 2
    var total = red[0]
    barrier()
    return total


@always_inline
def _ce_neg_inf() -> Float32:
    """`-inf` by bits. Contract 5.1's seed, and `pinned_block_fmax`'s
    identity element."""
    return bitcast[DType.float32](CE_NEG_INF_BITS)


@always_inline
def _ce_neg(x: Float32) -> Float32:
    """Seams L7 and L10's negation, with the one sabotage arm attached.

    The clean path is `loss_oracle.mojo::neg_by_bits`, IMPORTED rather than
    copied so the host and the device cannot come to two opinions about what
    negation is. DEVIATION 1154, contract 4.3.
    """
    comptime if SAB_NEG_VIA_ZERO_SUB:
        # SABOTAGE: `0.0 - x`. INERT at every input except `x == +0.0`.
        return Float32(0.0) - x
    return neg_by_bits(x)


# ===========================================================================
# K1: THE ROW MAXIMUM (seam L1, contract 5.1)
# ===========================================================================


def ce_row_max_kernel[
    BLOCK: Int
](
    max_out: MutPointer[Float32, MutAnyOrigin],
    logits: MutPointer[Float32, MutAnyOrigin],
    vocab_in: Int32,
):
    """`max_out[row]` for one row, one block per row.

    `block_idx.x` IS the row. The block strides over the vocabulary, folds
    with `identical_fmax`, and thread 0 stores. Threads with no element pass
    `-inf`.

    **THE LOOP ORDER AND THE STRIDE ARE SCHEDULING.** `identical_fmax` is
    commutative and associative over all of Float32 including both zeros and
    NaN, so this kernel's answer is a pure function of the row's multiset and
    of nothing else -- not the block width, not the stride, not the vendor.
    That is the one place in this profile where an execution plan may choose
    its own tree and it may because the operation is exactly associative
    rather than because the difference is thought to be small. Contract 5.1.

    THE SEED IS `-inf`, contract 5.1 and DEVIATION 1151. `+0.0` would clamp
    an all-negative row and be wrong on every vendor identically, which
    bit-identity cannot see -- only the oracle can. `-FLT_MAX` would be
    correct only because section 8 refuses an infinite logit somewhere else.
    """
    var vocab = Int(vocab_in)
    var row = Int(block_idx.x)
    var base = row * vocab
    var tid = Int(thread_idx.x)
    var nth = Int(block_dim.x)

    var limit = vocab
    comptime if SAB_MAX_TOPK_PREFIX:
        # SABOTAGE: the maximum over the first 32 classes only. INERT on a
        # row whose argmax is inside the prefix. Contract 5.2.
        if limit > 32:
            limit = 32

    var acc = _ce_neg_inf()
    comptime if SAB_MAX_SEED_ZERO:
        # SABOTAGE: `+0.0` seed. INERT on any row with a positive logit.
        acc = Float32(0.0)

    var v = tid
    while v < limit:
        comptime if SAB_MAX_PLAIN_COMPARE:
            var x = logits.unsafe_load(base + v)
            if x > acc:
                acc = x
        else:
            acc = identical_fmax(acc, logits.unsafe_load(base + v))
        v += nth
    var m = pinned_block_fmax[BLOCK](acc)
    if tid == 0:
        max_out.unsafe_store(row, m)


# ===========================================================================
# K2: THE SHIFT AND THE EXPONENTIAL (seams L2 and L3)
# ===========================================================================


def ce_shift_exp_kernel(
    shift: MutPointer[Float32, MutAnyOrigin],
    expo: MutPointer[Float32, MutAnyOrigin],
    logits: MutPointer[Float32, MutAnyOrigin],
    max_in: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    vocab_in: Int32,
):
    """`shift[i, v] = ftz(ftz(x) - ftz(m))` and `expo = identical_exp(shift)`.

    One thread owns one cell. No cross-thread anything, so the launch decides
    only which linear index this thread holds and batch composition is a
    property of the kernel's SHAPE rather than of a check that happens to
    pass. Contract 7.1.

    THE FLUSHES ARE ROW 10's CHECKLIST UNIT -- every operand flushed as
    loaded, every result flushed as written. `ftz` on `identical_exp`'s
    output is bitwise redundant (`portable_expf` never returns a subnormal
    and bakes the policy in unconditionally), and it is spelled anyway
    because "the seam a kernel writes for another kernel to read" is the unit
    the checklist is written in and a reader should not have to derive which
    seams are no-ops. That is the gemm contract's own argument for its 5d and
    5e.

    BY CONTRACT 8.2(a) the shift is provably non-positive, so
    `identical_exp` never takes its `x > 88.722835` overflow branch and
    `expo` is always in `[+0.0, 1.0]`.
    """
    var n_rows = Int(n_rows_in)
    var vocab = Int(vocab_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= n_rows * vocab:
        return
    var row = cell // vocab
    var s = ftz(ftz(logits.unsafe_load(cell)) - ftz(max_in.unsafe_load(row)))
    shift.unsafe_store(cell, s)
    comptime if SAB_EXP_STDLIB:
        # SABOTAGE: the vendor's own exp. INERT under FAST and at `s == 0`.
        expo.unsafe_store(cell, exp(s))
        return
    expo.unsafe_store(cell, ftz(identical_exp(s)))


# ===========================================================================
# K3: log(denom) (seam L5)
# ===========================================================================


def ce_logdenom_kernel(
    logdenom: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
):
    """`logdenom[i] = ftz(identical_log(ftz(denom[i])))`, one thread per row.

    BY CONTRACT 8.2(b) the argument is provably in `[1.0, Float32(V)]` -- the
    argmax contributes `identical_exp(+0.0)`, which is exactly `1.0`, every
    other term is non-negative, and the `+0.0`-seeded leaf chain cannot lose
    it -- so `portable_logf`'s zero, negative, subnormal and infinite
    branches are all unreachable from this profile and the result is in
    `[+0.0, 11.7621]` at the shipped vocabulary. That is worth knowing and it
    is NOT an argument for deleting those branches.

    `[[mojo-log-breaks-ties]]`: `std.math.log` carries about `5e-8` absolute
    error, which is enough to re-decide a plateau tie. `identical_log` routes
    through row 12's Cephes polynomial under IDENTICAL. Sabotage
    `SAB_LOG_STDLIB`.
    """
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(n_rows_in):
        return
    var d = ftz(denom.unsafe_load(row))
    comptime if SAB_LOG_STDLIB:
        logdenom.unsafe_store(row, log(d))
        return
    logdenom.unsafe_store(row, ftz(identical_log(d)))


# ===========================================================================
# K4: THE TARGET LOG-PROBABILITY AND THE NLL (seams L6 and L7)
# ===========================================================================


def ce_nll_kernel(
    logp_target: MutPointer[Float32, MutAnyOrigin],
    nll: MutPointer[Float32, MutAnyOrigin],
    shift: MutPointer[Float32, MutAnyOrigin],
    logdenom: MutPointer[Float32, MutAnyOrigin],
    max_in: MutPointer[Float32, MutAnyOrigin],
    logits: MutPointer[Float32, MutAnyOrigin],
    expo: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    vocab_in: Int32,
    ignore_index_in: Int32,
):
    """`lp_y = ftz(ftz(shift[y]) - ftz(logdenom))` and `nll = -lp_y`.

    Contract 4.2. **The pinned spelling never forms a quantity larger than
    `max(|shift[y]|, logdenom)`**, which is the whole point of having
    subtracted the maximum, and the two refused alternatives both undo it.

    AN IGNORED ROW still records a log-probability and an nll, computed at
    class 0 as a placeholder, and only `ce.row` is forced to `+0.0`. They are
    real numbers, they are on the card, and recording them is what keeps
    `SAB_IGNORED_ROW_NEG_ZERO` localizable to exactly one stage. Contract
    7.3.

    `max_in`, `logits`, `expo` and `denom` are read ONLY by the sabotage arms
    and are unused on the clean path. That is deliberate and it is the cost
    of making the two refused spellings buildable rather than describable --
    `IDENTICAL_BACKWARD_PLAN.md` section 5's rule is that a pin with no
    fixture separating it from the unpinned spelling, plus a demonstrated
    failure when the pin is removed, is not a pin.
    """
    var vocab = Int(vocab_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(n_rows_in):
        return
    var y = Int(targets.unsafe_load(row))
    if y == Int(ignore_index_in):
        y = 0
    var ld = ftz(logdenom.unsafe_load(row))

    comptime if SAB_NLL_VIA_ADDBACK:
        # SABOTAGE: `(m + logdenom) - x_y`. INERT on a centered row.
        var back = ftz(ftz(max_in.unsafe_load(row)) + ld)
        var lp_bad = ftz(ftz(logits.unsafe_load(row * vocab + y)) - back)
        logp_target.unsafe_store(row, lp_bad)
        nll.unsafe_store(row, _ce_neg(lp_bad))
        return
    comptime if SAB_NLL_VIA_LOG_W:
        # SABOTAGE: `-log(w[y])`. INERT wherever `e[y]` is normal; returns
        # `+inf` where `e[y]` has underflowed to exactly `+0.0`.
        var w_bad = ftz(
            identical_div(
                ftz(expo.unsafe_load(row * vocab + y)),
                ftz(denom.unsafe_load(row)),
            )
        )
        var lp_w = ftz(identical_log(w_bad))
        logp_target.unsafe_store(row, lp_w)
        nll.unsafe_store(row, _ce_neg(lp_w))
        return

    var lp = ftz(ftz(shift.unsafe_load(row * vocab + y)) - ld)
    logp_target.unsafe_store(row, lp)
    nll.unsafe_store(row, _ce_neg(lp))


# ===========================================================================
# K5: EVERY LOG-PROBABILITY (seam L8, smoothing only)
# ===========================================================================


def ce_logp_kernel(
    logp: MutPointer[Float32, MutAnyOrigin],
    shift: MutPointer[Float32, MutAnyOrigin],
    logdenom: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    vocab_in: Int32,
):
    """`logp[i, v] = ftz(ftz(shift[i, v]) - ftz(logdenom[i]))`.

    THE SAME EXPRESSION AS SEAM L6 AT EVERY `v`, deliberately, so that a
    divergence at `ce.logp` and NOT at `ce.logp_target` is a vocabulary
    indexing defect and nothing else. Splitting one arithmetic into two
    spellings is how a lane comes to have two answers, which is
    `[[read-their-source-against-ours]]`'s standing class of finding.
    """
    var vocab = Int(vocab_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(n_rows_in) * vocab:
        return
    var row = cell // vocab
    logp.unsafe_store(
        cell,
        ftz(ftz(shift.unsafe_load(cell)) - ftz(logdenom.unsafe_load(row))),
    )


# ===========================================================================
# K6: THE SMOOTHING AVERAGE (seam L10, smoothing only)
# ===========================================================================


def ce_smooth_kernel(
    smooth: MutPointer[Float32, MutAnyOrigin],
    logp_sum: MutPointer[Float32, MutAnyOrigin],
    vocab_f: Float32,
    eps_over_vocab: Float32,
    n_rows_in: Int32,
):
    """`smooth[i] = -(logp_sum[i] / V)`, ONE division by `V` FIRST.

    Contract 6.2(a). The alternative is a host-precomputed `eps / V` folded
    into L11's product, which saves a division per row and **is a different
    answer** -- it rounds `eps/V` once on the host and then rounds one
    product, where the pinned spelling rounds one quotient and then one
    product, and the two quotients are not the same number.

    `vocab_f` is `Float32(V)`, exact for every `V` under the section-3
    ceiling. `eps_over_vocab` is passed for the sabotage arm ALONE and is
    unused on the clean path.
    """
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(n_rows_in):
        return
    comptime if SAB_SMOOTH_FOLDED_CONSTANT:
        # SABOTAGE: multiply by a host-folded `eps / V`, which moves the
        # division to the host and changes where it rounds. The clean path's
        # `eps` product happens at L11; under this arm L11 must NOT apply it
        # again, which is why the launcher passes `+1.0` for `eps` there.
        smooth.unsafe_store(
            row, _ce_neg(ftz(identical_mul(logp_sum.unsafe_load(row),
                                           eps_over_vocab)))
        )
        return
    smooth.unsafe_store(
        row,
        _ce_neg(ftz(identical_div(ftz(logp_sum.unsafe_load(row)), vocab_f))),
    )


# ===========================================================================
# K7a AND K7b: THE ROW LOSS (seam L11) -- TWO KERNELS, ONE HOST BRANCH
# ===========================================================================
# Contract 6.2(c), DEVIATION 1155. `eps == 0` selects a DIFFERENT KERNEL
# rather than a bit-inert arm inside one kernel, and the branch is taken on
# the HOST from a configuration constant. Two kernels rather than a flag,
# because a flag is a data-dependent branch and this decision is not data.


def ce_row_nll_kernel(
    row_out: MutPointer[Float32, MutAnyOrigin],
    nll: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    ignore_index_in: Int32,
):
    """`row[i] = nll[i]`, or `+0.0` for an ignored row. The `eps == 0` path.

    THE IGNORED ROW IS `+0.0` AND NOT `-0.0`, contract 7.3, and it is
    STORED. L12's leaf accumulator is seeded `+0.0` and
    `acc + (+0.0) == acc` for every `acc` a `+0.0`-seeded chain can hold, so
    an ignored row is BITWISE INERT in the batch fold -- the transformer
    contract's 7.1 theorem with a row where it had a masked key. It is NOT
    inert in `P` (the fold still has `N` terms) and it is NOT inert on the
    card.
    """
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(n_rows_in):
        return
    if Int(targets.unsafe_load(row)) == Int(ignore_index_in):
        comptime if SAB_IGNORED_ROW_SKIPPED:
            # SABOTAGE: the store skipped. ALWAYS INERT unless the gate
            # poisons the buffer first, which is why contract 10.1 says the
            # gate must poison.
            return
        comptime if SAB_IGNORED_ROW_NEG_ZERO:
            # SABOTAGE: `-0.0`. MUST move `ce.row` and MUST NOT move
            # `ce.total`.
            row_out.unsafe_store(row, Float32(-0.0))
            return
        row_out.unsafe_store(row, Float32(0.0))
        return
    row_out.unsafe_store(row, nll.unsafe_load(row))


def ce_row_smooth_kernel(
    row_out: MutPointer[Float32, MutAnyOrigin],
    nll: MutPointer[Float32, MutAnyOrigin],
    smooth: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    one_minus_eps: Float32,
    eps: Float32,
    n_rows_in: Int32,
    ignore_index_in: Int32,
):
    """Seam L11 with smoothing, contract 6.2(b).

        ftz( ftz(identical_mul(ONE_MINUS_EPS, nll))
           + ftz(identical_mul(EPS, smooth)) )

    **UNFUSED.** Two rounded products, then one add of two already-rounded
    values. An `identical_mul_add(eps, smooth, first)` is one rounding where
    this is three, and it is the natural thing for a kernel to write. That is
    the transformer contract's S10 clause at a different seam.

    **`identical_mul` and not `*`.** DEVIATION 826. It presents no syntactic
    multiply for a codegen to contract into the following add -- the gemm
    README's F3 scar records `var p = a * b; p + c` being contracted ACROSS
    STATEMENTS on this host, so a separate statement is not a barrier -- and
    its `-0.0` addend preserves a negative-zero product where a `+0.0`
    addend would launder it (the gemm lane's F6a lesson).

    This is the ONLY `+` between two Float32 values in this file on a
    normative path. If a second appears, a fold has grown by hand.
    """
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(n_rows_in):
        return
    if Int(targets.unsafe_load(row)) == Int(ignore_index_in):
        comptime if SAB_IGNORED_ROW_SKIPPED:
            return
        comptime if SAB_IGNORED_ROW_NEG_ZERO:
            row_out.unsafe_store(row, Float32(-0.0))
            return
        row_out.unsafe_store(row, Float32(0.0))
        return
    var a = ftz(identical_mul(one_minus_eps, nll.unsafe_load(row)))
    var b = ftz(identical_mul(eps, smooth.unsafe_load(row)))
    comptime if SAB_SMOOTH_FUSED_COMBINE:
        # SABOTAGE: one rounding where the contract has three. INERT at
        # `eps == 0`, where this kernel is not launched at all.
        row_out.unsafe_store(
            row,
            ftz(identical_mul_add(eps, ftz(smooth.unsafe_load(row)), a)),
        )
        return
    row_out.unsafe_store(row, ftz(ftz(a) + ftz(b)))


# ===========================================================================
# K8: THE REDUCTION DIVIDE (seam L13)
# ===========================================================================


def ce_divide_kernel(
    loss_out: MutPointer[Float32, MutAnyOrigin],
    total: MutPointer[Float32, MutAnyOrigin],
    divisor: Float32,
):
    """`loss = ftz(identical_div(ftz(total), divisor))`, one thread.

    ONE DIVISION, never a reciprocal multiplied in. `e * (1/d)` rounds twice
    where `e / d` rounds once and they differ in the last bit on ordinary
    inputs. Contract 5.5, which is
    `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 5.4's paragraph at a
    second site, and it is UNVERIFIED AGAINST THE REFERENCE for the same
    reason -- there is no PyTorch checkout.

    `divisor` comes from `loss_oracle.mojo::ce_divisor` and from nowhere
    else. At `REDUCTION_SUM` with no `num_items_in_batch` it is exactly
    `+1.0` and the division is bitwise inert at every value `total` can hold;
    it is spelled anyway so there is ONE code path and no branch whose two
    arms have to be shown to agree.
    """
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return
    var t = ftz(total.unsafe_load(0))
    comptime if SAB_MEAN_RECIPROCAL_MUL:
        # SABOTAGE: two roundings. INERT when the divisor is an exact power
        # of two.
        var r = ftz(identical_div(Float32(1.0), divisor))
        loss_out.unsafe_store(0, ftz(identical_mul(t, r)))
        return
    loss_out.unsafe_store(0, ftz(identical_div(t, divisor)))


# ===========================================================================
# K9 AND K10: THE BACKWARD (seams L14, L15, L16)
# ===========================================================================


def ce_weights_kernel(
    weights: MutPointer[Float32, MutAnyOrigin],
    expo: MutPointer[Float32, MutAnyOrigin],
    denom: MutPointer[Float32, MutAnyOrigin],
    logp: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    vocab_in: Int32,
):
    """`w[i, v] = ftz(identical_div(e[i, v], denom[i]))`, seam L14.

    **TRANSFORMER DEVIATION 806 AT A SECOND SITE.** ONE division per weight,
    never `e * (1/denom)`. The transformer contract pinned this for
    `attn.weights` and this lane's brief required consistency with it; the
    two must agree, and `SAB_GRAD_RECIPROCAL_MUL` is the falsifier at seam
    L16 while this one is falsified by `SAB_W_VIA_EXP_LOGP`.

    `w = exp(logp)` is the refused alternative -- one more transcendental
    where a division suffices, and a different answer. `logp` is passed for
    that arm alone and is unused on the clean path; under `NUMERIC_FAST` on
    a build with no smoothing it may be an empty buffer, which is why the
    launcher refuses the arm unless `logp` was actually produced.
    """
    var vocab = Int(vocab_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(n_rows_in) * vocab:
        return
    var row = cell // vocab
    comptime if SAB_W_VIA_EXP_LOGP:
        weights.unsafe_store(cell, ftz(identical_exp(logp.unsafe_load(cell))))
        return
    weights.unsafe_store(
        cell,
        ftz(
            identical_div(
                ftz(expo.unsafe_load(cell)), ftz(denom.unsafe_load(row))
            )
        ),
    )


def ce_dlogits_kernel(
    dlogits: MutPointer[Float32, MutAnyOrigin],
    weights: MutPointer[Float32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    t_target: Float32,
    t_other: Float32,
    divisor: Float32,
    n_rows_in: Int32,
    vocab_in: Int32,
    ignore_index_in: Int32,
):
    """`dl[i, v] = ftz(identical_div(ftz(ftz(w) - ftz(t)), divisor))`, L16.

    `t_target` and `t_other` are the two HOST constants of seam L15
    (`ce_smoothing_targets`), selected by an INTEGER compare. Integers do not
    flush anywhere, so there is no row-49 reasoning at this seam at all.

    **THE DIVISION LIVES PER CELL, NOT PER ROW.** `d / divisor` rounds once
    where `d * (1/divisor)` rounds twice, and the same paragraph that pins
    L13 pins this. At `divisor == +1.0` it is bitwise inert and it is spelled
    anyway, for `ce_divide_kernel`'s reason.

    **AT `eps == 0` THE TARGET CONSTANTS ARE EXACTLY `1.0` AND `+0.0`**, so
    the backward's smoothing arm is bitwise inert at eps zero and needs no
    branch, unlike the forward's. That asymmetry is
    `[[reached-but-inert]]` in the other direction and contract 6.3 names it:
    a sabotage that deletes a branch the gradient does not have moves
    nothing, and that is not evidence about the gradient.

    AN IGNORED ROW writes `V` copies of `+0.0`, STORED and not skipped, for
    `ce_row_nll_kernel`'s reason and because a `-0.0` gradient cell would
    flow into whatever the optimizer does next.

    **THIS IS THE STAGE THE EXACT-ANALYTIC GATE ASSERTS BY BITS.** Contract
    12 -- on a uniform row at a power-of-two `V` with a power-of-two divisor
    every value here is a dyadic rational a person can write down, so there
    is no epsilon in the comparison. Contract 12.1 also says what that
    family cannot do: being exact it separates NO spelling from any other,
    so run alone it would pass every sabotage in this file and gate nothing
    about the arithmetic.
    """
    var vocab = Int(vocab_in)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= Int(n_rows_in) * vocab:
        return
    var row = cell // vocab
    var v = cell - row * vocab
    var y = Int(targets.unsafe_load(row))
    if y == Int(ignore_index_in):
        comptime if SAB_IGNORED_ROW_SKIPPED:
            return
        dlogits.unsafe_store(cell, Float32(0.0))
        return
    comptime if SAB_GRAD_TARGET_OFF_BY_ONE:
        # SABOTAGE: the target vector applied at `y + 1`. INERT at `V == 1`;
        # at `V > 1` it moves exactly two cells per row, which is why the
        # gate must compare CELLS and not a row norm.
        y = (y + 1) % vocab
    var t = t_other
    if v == y:
        t = t_target
    var w = ftz(weights.unsafe_load(cell))
    var d = ftz(ftz(w) - ftz(t))
    comptime if SAB_GRAD_SIGN:
        # SABOTAGE: `t - w`. INERT only where `w == t` at every class.
        d = ftz(ftz(t) - ftz(w))
    comptime if SAB_GRAD_RECIPROCAL_MUL:
        var r = ftz(identical_div(Float32(1.0), divisor))
        dlogits.unsafe_store(cell, ftz(identical_mul(d, r)))
        return
    dlogits.unsafe_store(cell, ftz(identical_div(d, divisor)))


# ===========================================================================
# THE SABOTAGE FOLD (contract 10.1's DENOM_SERIAL_CHAIN and REDUCE_SERIAL)
# ===========================================================================


def ce_serial_fold_kernel(
    out_buf: MutPointer[Float32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    span_in: Int32,
):
    """**SABOTAGE ONLY. NEVER ON A NORMATIVE PATH.**

    The WHOLE-AXIS ASCENDING CHAIN, one thread per output, `acc = ftz(ftz(acc)
    + ftz(v))` seeded `+0.0`. It is `ce_fold_serial_diagnostic`'s device
    twin and it is the spelling
    `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 5.3 pins for ITS softmax
    denominator.

    It exists so that the DEPARTURE of DEVIATION 1152 is a value a gate can
    print rather than an argument in a document. **It is INERT at every span
    of 128 or fewer**, where `P == 1` and gemm v1's tree performs no addition
    (gemm contract 7.3), and contract 5.3 makes `V >= 129` and `N >= 129`
    mandatory gate shapes for exactly that reason.

    The `+` here is the second of the two on non-normative paths in this
    lane; the other is `ce_fold_serial_diagnostic`'s.
    """
    var span = Int(span_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(n_rows_in):
        return
    var base = row * span
    var acc = Float32(0.0)
    for t in range(span):
        acc = ftz(ftz(acc) + ftz(values.unsafe_load(base + t)))
    out_buf.unsafe_store(row, ftz(acc))


# ===========================================================================
# SIZING (contract 5.3, and `identical_gemm_into`'s own hazard)
# ===========================================================================


def identical_ce_ones_floats(n_rows: Int, vocab: Int) -> Int:
    """How long the ones vector is: `max(vocab, n_rows)`.

    ONE vector serves all three folds. The vocabulary folds put it on the
    RIGHT (`OP_NN` at `(n_rows, 1, vocab)`, so it is the `k x n` operand) and
    the batch fold puts it on the LEFT (`OP_NN` at `(1, 1, n_rows)`, the
    `m x k` operand). **Operand side is part of the routing and not a
    convention that can be assumed** -- that is `gemm_backward.mojo`'s
    `BWD_DC_LEFT` / `BWD_DC_RIGHT` lesson, and `SAB_BWD_OPERAND_ORDER` is
    what it grew from.

    The caller allocates it, fills it with EXACTLY `Float32(1.0)` in every
    entry, and keeps it alive past its own synchronize. It is a constant of
    the shape, not per step.

    **A WRONG VALUE HERE IS A WRONG ANSWER WITH NO SYMPTOM**, because any
    vector produces a plausible weighted sum. That is
    `identical_gemm_backward_bias_ones_floats`'s warning verbatim. The gate
    for it is not "the buffer was allocated"; it is contract 12's
    EXACT-ANALYTIC arm, which fails if the ones are anything else.
    """
    var w = vocab
    if n_rows > w:
        w = n_rows
    if w < 1:
        return 1
    return w


def identical_ce_workspace_max_floats(
    n_rows: Int, vocab: Int, reduction: Int
) -> Int:
    """Floats of scratch the routed GEMMs may need, at this shape.

    **SIZING A WORKSPACE FROM THE WRONG SHAPE IS AN OUT-OF-BOUNDS WRITE.**
    `identical_gemm_into`'s docstring records what that mistake cost the GEMM
    lane: a 1-float workspace passed to a SPLITK dispatch at `64 x 4` still
    returned the right answer because the allocation had slack, and only at
    `64 x 64` did whole regions of the output come back `+0.0`. Use this
    helper; do not guess.

    Three calls share one workspace and the maximum of the three is what is
    returned. Sharing is safe because they are enqueued on ONE
    `DeviceContext` and MAX runs them in order, so the second call's leaf
    kernel cannot start before the first call's fold kernel has finished
    reading -- `identical_gemm_backward_workspace_max_floats`'s own argument,
    and the forward gate already asserts the stronger form of it
    (`check_device_is_batch_invariant` runs four GEMMs through one DIRTY
    shared workspace and requires the same bits). **It is not safe across two
    contexts or two streams.**

    The vocabulary folds land at `(n_rows, 1, vocab)`. `m * n` there is
    `n_rows`, so `choose_gemm_plan`'s SPLITK condition (`m * n <= 4096` with
    `P >= 4`) is met at every `n_rows <= 4096` with `vocab > 384`, which is
    most real shapes -- and the workspace is `n_rows * P(vocab)`, up to
    `n_rows * 1024` floats. At `n_rows = 4096` that is 16 MB. Not large, and
    not zero either, so it must be allocated.
    """
    var w = identical_gemm_workspace_max_floats(n_rows, 1, vocab)
    if reduction != REDUCTION_NONE:
        var wb = identical_gemm_workspace_max_floats(1, 1, n_rows)
        if wb > w:
            w = wb
    if w < 1:
        return 1
    return w


def _grid_for(count: Int) -> Int:
    """Blocks needed to cover `count` items at `CE_TPB` threads each.
    SCHEDULING, and the only arithmetic in this file that reads a block
    size."""
    if count < 1:
        return 1
    return (count + CE_TPB - 1) // CE_TPB


# ===========================================================================
# THE LAUNCHERS
# ===========================================================================
# Both are the `_into` form: caller-owned buffers, caller-owned workspace,
# ASYNCHRONOUS, nothing waits. A training step enqueues the forward, the
# backward and the `lm_head` backward back to back and synchronizes once,
# which is the shape the whole file exists for.


def ce_refuse_device_inputs(
    ctx: DeviceContext,
    mut logits: DeviceBuffer[DType.float32],
    mut targets: DeviceBuffer[DType.int32],
    n_rows: Int,
    cfg: CeConfig,
) raises:
    """Contract section 8 ON THE DEVICE ENTRY POINT, which is where it was
    missing. DEVIATION 1495.

    **THE GAP THIS CLOSES WAS MEASURED, NOT SUSPECTED.** `loss_check.mojo`
    clause (f), first execution 2026-08-25, planted a quiet NaN in `logits`
    and it reached **40 recorded cells, first at `ce.max`**. Contract section
    8 says non-finite inputs are "REFUSED BY NAME before any recorded stage";
    that was true of `ce_refuse_inputs` in `loss_oracle.mojo` and FALSE of
    `identical_ce_forward_into`, which never called it. A caller reaching the
    device entry directly put a **vendor-shaped NaN payload into a certified
    stage** -- IDENTITY_PATHS row 39 measured three different payloads for one
    IEEE answer (`0x7fc00000` Apple, `0x7fffffff` NVIDIA, `0xffc00000` AMD),
    so a stage hash containing one cannot match across vendors and the
    profile's whole claim fails on that input.

    **IT CALLS THE ORACLE'S OWN `ce_refuse_inputs` AND DOES NOT RESTATE IT.**
    A second copy of the refusal is a second thing to keep in step, and the
    property that matters is that the two sides fail IDENTICALLY -- same
    order, same name, same message. The transformer lane's
    `llama_refuse_bad_inputs` is the precedent and it downloads for the same
    reason.

    **THE COST IS A FULL DOWNLOAD OF `logits`, AND IT IS REAL.** At the
    shipped vocabulary that is `n_rows * 128256` floats crossing the bus
    before any arithmetic. This is correctness before speed and it is stated
    rather than hidden. A device-side scan writing one count would be
    cheaper; it is OWED, and it must produce the SAME message for the SAME
    first offending cell or it is a different refusal wearing this one's name.

    **THE REFUSAL COVERS INPUTS AND NOT INTERMEDIATES**, the same stated gap
    the transformer lane carries at its DEVIATION 815. A computed non-finite
    is caught by the card, since a stage hash holding a vendor-shaped payload
    cannot match.
    """
    var nv = n_rows * cfg.vocab
    var hl = ctx.enqueue_create_host_buffer[DType.float32](nv if nv > 0 else 1)
    var ht = ctx.enqueue_create_host_buffer[DType.int32](
        n_rows if n_rows > 0 else 1
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=logits)
    ctx.enqueue_copy(dst_ptr=ht.unsafe_ptr(), src_buf=targets)
    ctx.synchronize()
    var hx = List[Float32]()
    for i in range(nv):
        hx.append(hl.unsafe_ptr().unsafe_load(i))
    var ht_l = List[Int32]()
    for i in range(n_rows):
        ht_l.append(ht.unsafe_ptr().unsafe_load(i))
    _ = ce_refuse_inputs(hx, ht_l, cfg)
    _ = hl
    _ = ht


def identical_ce_forward_into(
    ctx: DeviceContext,
    mut max_v: DeviceBuffer[DType.float32],
    mut shift: DeviceBuffer[DType.float32],
    mut expo: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32],
    mut logdenom: DeviceBuffer[DType.float32],
    mut logp_target: DeviceBuffer[DType.float32],
    mut nll: DeviceBuffer[DType.float32],
    mut logp: DeviceBuffer[DType.float32],
    mut logp_sum: DeviceBuffer[DType.float32],
    mut smooth: DeviceBuffer[DType.float32],
    mut row: DeviceBuffer[DType.float32],
    mut total: DeviceBuffer[DType.float32],
    mut loss: DeviceBuffer[DType.float32],
    mut logits: DeviceBuffer[DType.float32],
    mut targets: DeviceBuffer[DType.int32],
    mut ones: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    n_rows: Int,
    count: Int,
    cfg: CeConfig,
) raises:
    """Seams L1 through L13, enqueued. Nothing waits.

    EVERY BUFFER IS THE CALLER'S. `[[mojo-buffer-freed-at-last-use]]`: a
    `DeviceBuffer` is dead at its `.unsafe_ptr()`, so every one of these --
    including `ones` and `ws` -- must outlive the caller's
    `ctx.synchronize()`. `logp`, `logp_sum` and `smooth` may be
    single-element placeholders when `cfg.smoothing_is_spelled()` is false;
    they are not touched then.

    **`count` IS A HOST INTEGER AND IS THE CALLER'S**, contract 5.5. Counting
    the unignored rows on device would be an integer reduction -- exact,
    order-free and perfectly legal as an execution plan -- and v1 takes it
    from the host so that `ce_divisor`, the ONE producer of the quantity the
    forward and the backward must agree about, stays host-side and testable
    with no GPU present. That is `gemm_backward_a_call`'s "pure, host-side
    and testable with no device" discipline applied to a scalar.

    THE ORDER OF THE ENQUEUES IS THE ORDER OF THE SEAMS. Each kernel reads
    only buffers an earlier one wrote, and MAX runs one context in order, so
    no barrier is spelled here. A second stream would break that and would be
    a change to this function's contract rather than to its arithmetic.

    ROW INDEPENDENCE, contract 7.1. Nothing between L1 and L11 reads
    `n_rows` except as a bound, so a caller may split the rows into chunks of
    any size, call this per chunk, and concatenate -- which is the escape
    from contract 5.3's memory cost, since `[N, V]` at the shipped Llama-3
    head is 2.1 GB per buffer at `N = 4096`. **Only L12 folds over `n_rows`**,
    and it folds `[N]` floats rather than `[N, V]`.
    """
    # DEVIATION 1495: contract section 8, on the DEVICE path. Measured
    # missing by loss_check clause (f) -- a planted NaN reached 40
    # recorded cells, first at `ce.max`. This is the FIRST statement in
    # the body because "before any recorded stage" is the clause.
    ce_refuse_device_inputs(ctx, logits, targets, n_rows, cfg)

    if n_rows < 1 or cfg.vocab < 1:
        return
    var vocab = cfg.vocab
    var cells = n_rows * vocab
    var smoothing = cfg.smoothing_is_spelled()
    comptime if SAB_SMOOTH_ALWAYS_SPELLED:
        # SABOTAGE: the smoothing path spelled even at `eps == 0`. INERT
        # everywhere except a row whose `nll` is `-0.0`. Contract 6.2(c).
        smoothing = True

    # ---- L1, one block per row. The fold shape is FREE (contract 5.1).
    # `[[mojo-syntax]]`: the parametric kernel is bound to a `comptime` local
    # first, which is `_launch_tiled`'s spelling in `gemm_identical.mojo` and
    # the only one this tree has compiled.
    comptime max_kern = ce_row_max_kernel[CE_TPB]
    ctx.enqueue_function[max_kern](
        max_v.unsafe_ptr(),
        logits.unsafe_ptr(),
        Int32(vocab),
        grid_dim=(n_rows, 1, 1),
        block_dim=(CE_TPB, 1, 1),
    )

    # ---- L2 and L3, elementwise.
    ctx.enqueue_function[ce_shift_exp_kernel](
        shift.unsafe_ptr(),
        expo.unsafe_ptr(),
        logits.unsafe_ptr(),
        max_v.unsafe_ptr(),
        Int32(n_rows),
        Int32(vocab),
        grid_dim=(_grid_for(cells), 1, 1),
        block_dim=(CE_TPB, 1, 1),
    )

    # ---- L4. ROUTED. The ones vector is the RIGHT operand here.
    comptime if SAB_DENOM_SERIAL_CHAIN:
        ctx.enqueue_function[ce_serial_fold_kernel](
            denom.unsafe_ptr(),
            expo.unsafe_ptr(),
            Int32(n_rows),
            Int32(vocab),
            grid_dim=(_grid_for(n_rows), 1, 1),
            block_dim=(CE_TPB, 1, 1),
        )
    else:
        identical_gemm_into(
            ctx, denom, expo, ones, ws, n_rows, 1, vocab, OP_NN
        )

    # ---- L5.
    ctx.enqueue_function[ce_logdenom_kernel](
        logdenom.unsafe_ptr(),
        denom.unsafe_ptr(),
        Int32(n_rows),
        grid_dim=(_grid_for(n_rows), 1, 1),
        block_dim=(CE_TPB, 1, 1),
    )

    # ---- L6 and L7.
    ctx.enqueue_function[ce_nll_kernel](
        logp_target.unsafe_ptr(),
        nll.unsafe_ptr(),
        shift.unsafe_ptr(),
        logdenom.unsafe_ptr(),
        max_v.unsafe_ptr(),
        logits.unsafe_ptr(),
        expo.unsafe_ptr(),
        denom.unsafe_ptr(),
        targets.unsafe_ptr(),
        Int32(n_rows),
        Int32(vocab),
        Int32(cfg.ignore_index),
        grid_dim=(_grid_for(n_rows), 1, 1),
        block_dim=(CE_TPB, 1, 1),
    )

    var one_minus = ce_one_minus_eps(cfg.eps)
    var tv = ce_smoothing_targets(cfg.eps, vocab)

    if smoothing:
        # ---- L8.
        ctx.enqueue_function[ce_logp_kernel](
            logp.unsafe_ptr(),
            shift.unsafe_ptr(),
            logdenom.unsafe_ptr(),
            Int32(n_rows),
            Int32(vocab),
            grid_dim=(_grid_for(cells), 1, 1),
            block_dim=(CE_TPB, 1, 1),
        )
        # ---- L9. ROUTED, the same call at the same shape as L4.
        identical_gemm_into(
            ctx, logp_sum, logp, ones, ws, n_rows, 1, vocab, OP_NN
        )
        # ---- L10. `tv[1]` is `T_OTHER`, which is `eps / V` -- the SABOTAGE
        # arm's folded constant, passed here and unused on the clean path.
        ctx.enqueue_function[ce_smooth_kernel](
            smooth.unsafe_ptr(),
            logp_sum.unsafe_ptr(),
            Float32(vocab),
            tv[1],
            Int32(n_rows),
            grid_dim=(_grid_for(n_rows), 1, 1),
            block_dim=(CE_TPB, 1, 1),
        )
        # ---- L11, the smoothing kernel. Under SAB_SMOOTH_FOLDED_CONSTANT
        # the `eps` product has already been applied inside K6, so this arm
        # passes `+1.0` rather than applying it twice; that is what makes the
        # sabotage a DIFFERENT SPELLING of the same quantity rather than a
        # different quantity, which is the only kind of sabotage that proves
        # anything about a rounding decision.
        var eps_here = ftz(cfg.eps)
        comptime if SAB_SMOOTH_FOLDED_CONSTANT:
            eps_here = Float32(1.0)
        ctx.enqueue_function[ce_row_smooth_kernel](
            row.unsafe_ptr(),
            nll.unsafe_ptr(),
            smooth.unsafe_ptr(),
            targets.unsafe_ptr(),
            one_minus,
            eps_here,
            Int32(n_rows),
            Int32(cfg.ignore_index),
            grid_dim=(_grid_for(n_rows), 1, 1),
            block_dim=(CE_TPB, 1, 1),
        )
    else:
        # ---- L11, the `eps == 0` path. A DIFFERENT KERNEL and not a
        # bit-inert arm inside one kernel. Contract 6.2(c), DEVIATION 1155.
        ctx.enqueue_function[ce_row_nll_kernel](
            row.unsafe_ptr(),
            nll.unsafe_ptr(),
            targets.unsafe_ptr(),
            Int32(n_rows),
            Int32(cfg.ignore_index),
            grid_dim=(_grid_for(n_rows), 1, 1),
            block_dim=(CE_TPB, 1, 1),
        )

    if cfg.reduction == REDUCTION_NONE:
        return

    # ---- L12. ROUTED. The ones vector is the LEFT operand here, which is
    # `identical_gemm_backward_bias_into`'s own shape at `n == 1`.
    comptime if SAB_REDUCE_SERIAL:
        ctx.enqueue_function[ce_serial_fold_kernel](
            total.unsafe_ptr(),
            row.unsafe_ptr(),
            Int32(1),
            Int32(n_rows),
            grid_dim=(1, 1, 1),
            block_dim=(CE_TPB, 1, 1),
        )
    else:
        identical_gemm_into(ctx, total, ones, row, ws, 1, 1, n_rows, OP_NN)

    # ---- L13. The divisor's ONE producer.
    var divisor = ce_divisor(cfg.reduction, count, cfg.num_items)
    ctx.enqueue_function[ce_divide_kernel](
        loss.unsafe_ptr(),
        total.unsafe_ptr(),
        divisor,
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )


def identical_ce_backward_into(
    ctx: DeviceContext,
    mut weights: DeviceBuffer[DType.float32],
    mut dlogits: DeviceBuffer[DType.float32],
    mut expo: DeviceBuffer[DType.float32],
    mut denom: DeviceBuffer[DType.float32],
    mut logp: DeviceBuffer[DType.float32],
    mut targets: DeviceBuffer[DType.int32],
    n_rows: Int,
    count: Int,
    cfg: CeConfig,
) raises:
    """Seams L14 through L16, enqueued. Nothing waits.

    `expo` and `denom` must be the buffers `identical_ce_forward_into` wrote
    on the SAME inputs and the SAME config. **The backward recomputes
    nothing**, which is deliberate: a second spelling of the softmax is a
    second thing that can be wrong, and the gemm oracle's own header makes
    the same argument about its serial diagnostic.

    **REDUCTION_NONE HAS NO BACKWARD** and raises, contract section 11. A
    per-row upstream vector adds a second product per cell whose placement --
    before or after L16's division -- is a real decision with two different
    answers, and this lane has no caller for it. An unused wrapper is an
    ungated one.

    `dlogits` may alias nothing that the forward still needs; it is written
    once per cell and read by nobody here.

    THE DIVISOR COMES FROM `ce_divisor`, the same call the forward made, with
    the same three arguments. Sabotage `SAB_GRAD_DIVISOR_IS_N` substitutes
    `Float32(n_rows)` **and is bit-inert whenever no row is ignored under
    MEAN, and on every SUM** -- the gate must predict that and assert the
    inertness as a mask, or the arm will look broken on an ordinary fixture.
    """
    if n_rows < 1 or cfg.vocab < 1:
        return
    if cfg.reduction == REDUCTION_NONE:
        raise Error(
            String("ce: REDUCTION_NONE has no backward (contract section 11)")
        )
    var vocab = cfg.vocab
    var cells = n_rows * vocab
    var tv = ce_smoothing_targets(cfg.eps, vocab)
    var divisor = ce_divisor(cfg.reduction, count, cfg.num_items)
    comptime if SAB_GRAD_DIVISOR_IS_N:
        divisor = Float32(n_rows)

    ctx.enqueue_function[ce_weights_kernel](
        weights.unsafe_ptr(),
        expo.unsafe_ptr(),
        denom.unsafe_ptr(),
        logp.unsafe_ptr(),
        Int32(n_rows),
        Int32(vocab),
        grid_dim=(_grid_for(cells), 1, 1),
        block_dim=(CE_TPB, 1, 1),
    )
    ctx.enqueue_function[ce_dlogits_kernel](
        dlogits.unsafe_ptr(),
        weights.unsafe_ptr(),
        targets.unsafe_ptr(),
        tv[0],
        tv[1],
        divisor,
        Int32(n_rows),
        Int32(vocab),
        Int32(cfg.ignore_index),
        grid_dim=(_grid_for(cells), 1, 1),
        block_dim=(CE_TPB, 1, 1),
    )
