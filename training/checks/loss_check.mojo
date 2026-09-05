# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device loss validation against independent fixtures, including forward values, gradients, reductions, and edge cases."""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace, read_trace_lines
from mamba.checks.mamba_fixture import corpus_splitmix64
from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_log,
    identical_mul,
)
from gemm.checks.gemm_identical import (
    PLAN_FLAT,
    PLAN_SPLITK,
    PLAN_SPLITK_STAGED,
    PLAN_TILE_16_16_32,
    gemm_plan_name,
    gemm_sabotage_name,
    identical_gemm_with_plan,
    identical_gemm_workspace_max_floats,
)
from gemm.checks.gemm_backward import gemm_backward_sabotage_name
from gemm.checks.gemm_oracle import OP_NN, contract_leaf_size
from transformer.checks.transformer_fixture import fixture_splitmix64
from training.checks.loss import (
    ANY_LOSS_SABOTAGE,
    identical_ce_backward_into,
    identical_ce_forward_into,
    identical_ce_ones_floats,
    identical_ce_workspace_max_floats,
    loss_sabotage_name,
)
from training.checks.loss_fixture import (
    BITS_POS_INF,
    BITS_QNAN,
    CE_CASE_COUNT,
    CE_CASE_SHIPPED_V,
    CE_EXACT_CASE_COUNT,
    CeCase,
    CeExactCase,
    assert_no_zero_or_subnormal_logits,
    assert_row_has_a_positive_logit,
    bits32_hex,
    bits_of,
    case_fold_has_carry,
    case_has_both_zero_signs,
    case_leaf_count,
    case_row_nll_is_negative_zero,
    case_target_exp_underflows,
    ce_case,
    ce_case_by_name,
    ce_case_config,
    ce_case_divisor,
    ce_case_has_backward,
    ce_case_logits,
    ce_case_targets,
    ce_case_unignored_count,
    ce_eps_smooth,
    ce_exact_case,
    ce_exact_config,
    ce_exact_guard,
    ce_exact_logits,
    ce_exact_targets,
    ce_poison,
    ce_profile_constants_are_intact,
    ce_splitmix64,
    count_poison,
    describe_case,
    f32_from_bits,
    is_exact_power_of_two,
    is_nonfinite_bits,
    mode_is_identical,
    mode_name,
    row_max_oracle,
    row_max_seeded_zero,
    row_max_topk_prefix,
)
from training.checks.loss_oracle import (
    CeConfig,
    IGNORE_INDEX_DEFAULT,
    REDUCTION_MEAN,
    REDUCTION_NONE,
    REDUCTION_SUM,
    ce_backward_oracle,
    ce_count,
    ce_divisor,
    ce_exact_saturating_gradient,
    ce_exact_uniform_gradient,
    ce_forward_f64,
    ce_forward_oracle,
    ce_ones,
    ce_refuse_inputs,
    ce_smoothing_targets,
)


# Card output follows the caller-selected runtime path; standalone runs use the fallback.

comptime TRACE_PATH = "/tmp/mojolearn_loss_ce.trace"


def card_path() -> String:
    """`MOJOLEARN_IDENTITY_TRACE` when the caller set it, else
    `TRACE_PATH`. The same precedence `bench/gemm_card_main.mojo:553`,
    `mamba/checks/mamba_check.mojo::card_path` and
    `transformer/checks/transformer_check.mojo::card_path` use. Read at
    RUN time, never at compile time, because the harness chooses the
    directory."""
    var p = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if p.byte_length() > 0:
        return p^
    return String(TRACE_PATH)


comptime TAG_PREFIX = "ce"

#: Contract section 10 clause (b)'s launch count.
comptime CLAUSE_B_LAUNCHES = 8


def env_on(name: String) -> Bool:
    return String(getenv(name)) != ""


def env_str(name: String) -> String:
    return String(getenv(name))


def env_case(name: String, dflt: Int) raises -> Int:
    """A fixture case chosen by NAME or by index, with a default.

    A name that does not exist RAISES rather than silently falling back. A
    gate that quietly ran a different case from the one the operator asked
    for would report a green for a shape nobody chose, which is the same
    class of defect as the card-path alias above."""
    var s = env_str(name)
    if s == "":
        return dflt
    var all_digits = s.byte_length() > 0
    for i in range(s.byte_length()):
        var c = ord(String(s[byte=i]))
        if c < 48 or c > 57:
            all_digits = False
    if all_digits:
        var v = 0
        for i in range(s.byte_length()):
            v = v * 10 + (ord(String(s[byte=i])) - 48)
        if v < 0 or v >= CE_CASE_COUNT:
            raise Error(
                String("loss_check: ")
                + name
                + " is case "
                + String(v)
                + " and there are "
                + String(CE_CASE_COUNT)
            )
        return v
    return ce_case_by_name(s)


# ===========================================================================
# THE STAGE TABLE (contract section 9)
#
# The tag strings live HERE and nowhere else in this lane, because
# `loss.mojo` emits no card at all -- its own header says so -- and
# `loss_oracle.mojo` carries the stages as struct fields rather than as
# tags. That makes this file the only producer of the tag sequence, which
# removes the failure mode `tools/identity_trace_diff.py` fears most: it
# aligns two traces by their TAG SEQUENCES before it compares a single
# hash, so a tag that disagrees between two producers does not make the
# diff smaller, it makes the ALIGNMENT wrong and reports a plausible answer
# for a pair of stages that were never the same stage.
#
# DEVIATION 1451, the consequence, stated because it is a real weakness:
# **the card this gate writes is a hash of what THIS GATE DOWNLOADED**, not
# of what the device recorded on its own. A device that computed the right
# bits and a download that read the wrong region would be indistinguishable
# here. The transformer lane does not have this problem because its block
# records its own stages. Closing it means adding `IdentityTrace` recording
# to `loss.mojo`, which this lane may not touch, and it is in the OWED
# block.
# ===========================================================================

comptime CE_STAGE_COUNT = 20

#: Indices, so no clause has to count.
comptime ST_INPUT_LOGITS = 0
comptime ST_INPUT_TARGETS = 1
comptime ST_MAX = 2
comptime ST_SHIFT = 3
comptime ST_EXP = 4
comptime ST_DENOM = 5
comptime ST_LOGDENOM = 6
comptime ST_LOGP_TARGET = 7
comptime ST_NLL = 8
comptime ST_LOGP = 9
comptime ST_LOGP_SUM = 10
comptime ST_SMOOTH = 11
comptime ST_ROW = 12
comptime ST_COUNT = 13
comptime ST_DIVISOR = 14
comptime ST_TOTAL = 15
comptime ST_LOSS = 16
comptime ST_TARGET_VEC = 17
comptime ST_WEIGHTS = 18
comptime ST_DLOGITS = 19


def ce_stage_tag(i: Int) raises -> String:
    """Contract section 9's tags, in section 9's order."""
    if i == ST_INPUT_LOGITS:
        return String("input.logits")
    if i == ST_INPUT_TARGETS:
        return String("input.targets")
    if i == ST_MAX:
        return String("ce.max")
    if i == ST_SHIFT:
        return String("ce.shift")
    if i == ST_EXP:
        return String("ce.exp")
    if i == ST_DENOM:
        return String("ce.denom")
    if i == ST_LOGDENOM:
        return String("ce.logdenom")
    if i == ST_LOGP_TARGET:
        return String("ce.logp_target")
    if i == ST_NLL:
        return String("ce.nll")
    if i == ST_LOGP:
        return String("ce.logp")
    if i == ST_LOGP_SUM:
        return String("ce.logp_sum")
    if i == ST_SMOOTH:
        return String("ce.smooth")
    if i == ST_ROW:
        return String("ce.row")
    if i == ST_COUNT:
        return String("ce.count")
    if i == ST_DIVISOR:
        return String("ce.divisor")
    if i == ST_TOTAL:
        return String("ce.total")
    if i == ST_LOSS:
        return String("ce.loss")
    if i == ST_TARGET_VEC:
        return String("ce.target_vec")
    if i == ST_WEIGHTS:
        return String("ce.weights")
    if i == ST_DLOGITS:
        return String("ce.dlogits")
    raise Error(
        String("loss_check: stage ")
        + String(i)
        + " is not one of contract section 9's "
        + String(CE_STAGE_COUNT)
    )


def stage_index_of(tag: String) raises -> Int:
    for i in range(CE_STAGE_COUNT):
        if ce_stage_tag(i) == tag:
            return i
    raise Error(
        String("loss_check: '") + tag + "' is not one of section 9's tags"
    )


def stage_is_int(i: Int) -> Bool:
    """`input.targets` and `ce.count` are INTEGERS and are compared as
    integers, never bitcast into floats.

    A gate that pushed them through the float path would work -- `Int32`
    and `Float32` are both 32 bits -- and it would be wrong in a way
    nothing catches, because an integer that happens to be a NaN bit
    pattern would then be "non-finite" to every audit in this file.
    Contract section 4: integer work is not a seam and it is not compared
    like one."""
    return i == ST_INPUT_TARGETS or i == ST_COUNT


def stage_is_smoothing_only(i: Int) -> Bool:
    """Contract section 9's three SMOOTHING ONLY stages. At `EPS == 0` they
    are ABSENT, and absent is not the same claim as "present and zero" --
    `CeStages`'s own docstring says the differ must treat a zero-length
    stage as absent rather than as agreeing with an absent one."""
    return i == ST_LOGP or i == ST_LOGP_SUM or i == ST_SMOOTH


def stage_is_reduced_only(i: Int) -> Bool:
    """`ce.divisor`, `ce.total` and `ce.loss` do not exist under
    `REDUCTION_NONE` (contract 5.5's table: the divisor is "not formed")."""
    return i == ST_DIVISOR or i == ST_TOTAL or i == ST_LOSS


def stage_is_backward(i: Int) -> Bool:
    return i == ST_WEIGHTS or i == ST_DLOGITS


def stage_is_host_constant(i: Int) -> Bool:
    """`ce.count`, `ce.divisor` and `ce.target_vec` are HOST quantities.

    **THEY HAVE NO DEVICE COPY AND CLAUSE (a) CANNOT SEPARATE ONE.**
    `identical_ce_forward_into` calls `ce_divisor` and
    `ce_smoothing_targets` itself, on the host, and passes the results into
    kernels as scalars; there is no buffer to download. So for these three
    the "device" value this file compares is the CHECK's own call of the
    same producer with the same arguments.

    What that CAN catch, and it is not nothing: the two passes supplying
    DIFFERENT arguments -- a forward that counted the unignored rows one
    way and a backward that counted them another, a `num_items` that
    reached one and not the other, a `vocab` that disagreed. That is
    exactly what contract 5.5's "the divisor has exactly ONE PRODUCER"
    clause is about, and `L_GRAD_DIVISOR_IS_N` is its falsifier -- an arm
    that substitutes `N` for `count` INSIDE the backward, which is
    invisible at `ce.divisor` and visible at `ce.dlogits`.

    What it CANNOT catch is `ce_divisor` itself being wrong, because both
    sides call it. Only a hand-written expected value can, and contract
    12's EXACT-ANALYTIC arm is where one exists."""
    return i == ST_COUNT or i == ST_DIVISOR or i == ST_TARGET_VEC


def stage_is_per_row(i: Int) -> Bool:
    """Is this stage indexed by ROW, so that clause (c) can cut it?

    The four that are not are `ce.count`, `ce.divisor`, `ce.total` and
    `ce.loss` -- and `ce.target_vec`, which is a two-element configuration
    constant. **`ce.total` and `ce.loss` being outside is the CLAUSE and
    not an omission**: contract 7.1 says nothing per row reads `N` EXCEPT
    L12, whose fold length IS the batch. A gate that quietly row-sliced
    `ce.total` would be asserting something contract 5.4 says is false."""
    if i == ST_COUNT or i == ST_DIVISOR or i == ST_TOTAL:
        return False
    if i == ST_LOSS or i == ST_TARGET_VEC:
        return False
    return True


def stage_row_width(i: Int, vocab: Int) raises -> Int:
    """Cells per ROW for a per-row stage. `[N, V]` stages are `V` wide and
    `[N]` stages are 1."""
    if i == ST_INPUT_LOGITS or i == ST_SHIFT or i == ST_EXP:
        return vocab
    if i == ST_LOGP or i == ST_WEIGHTS or i == ST_DLOGITS:
        return vocab
    if i == ST_INPUT_TARGETS:
        return 1
    if i == ST_MAX or i == ST_DENOM or i == ST_LOGDENOM:
        return 1
    if i == ST_LOGP_TARGET or i == ST_NLL or i == ST_LOGP_SUM:
        return 1
    if i == ST_SMOOTH or i == ST_ROW:
        return 1
    raise Error(
        String("loss_check: stage ")
        + String(i)
        + " is not per-row and has no row width"
    )


def stage_present(i: Int, c: CeCase) -> Bool:
    """Does this case produce this stage at all?

    Three rules and each is a contract clause rather than a convenience:
      * the three smoothing stages exist only at `EPS != 0` (6.2(c));
      * `ce.divisor`, `ce.total` and `ce.loss` exist only under SUM and
        MEAN (5.5);
      * `ce.weights` and `ce.dlogits` exist only when the backward ran, and
        `REDUCTION_NONE` HAS NO BACKWARD (section 11).
    """
    if stage_is_smoothing_only(i) and not c.eps_on:
        return False
    if stage_is_reduced_only(i) and c.reduction == REDUCTION_NONE:
        return False
    if stage_is_backward(i) and c.reduction == REDUCTION_NONE:
        return False
    return True


# ===========================================================================
# THE DEVICE SIDE
#
# DEVIATION 1459. `loss.mojo` exports no `_upload` and no `_download`, so
# unlike `transformer_check.mojo` -- which imports the block's own copies
# specifically so that the gate's plumbing IS the block's plumbing -- this
# file has to spell its own. That is a real weakness and it is named rather
# than hidden: a gate whose plumbing is not the code's plumbing is a gate
# that can pass because its plumbing is different. What limits the damage
# here is that `loss.mojo` has NO plumbing of its own to differ from -- it
# is an `_into` launcher over caller-owned buffers, so the caller's upload
# and download are the only ones there are.
# ===========================================================================


def _upload_f32(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    """One host list onto the device, contiguous, in index order."""
    var n = len(values)
    var n_buf = n
    if n_buf < 1:
        n_buf = 1
    var dev = ctx.enqueue_create_buffer[DType.float32](n_buf)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n_buf)
    ctx.synchronize()
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    for i in range(n, n_buf):
        host.unsafe_ptr().unsafe_store(i, Float32(0.0))
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def _upload_i32(
    ctx: DeviceContext, values: List[Int32]
) raises -> DeviceBuffer[DType.int32]:
    var n = len(values)
    var n_buf = n
    if n_buf < 1:
        n_buf = 1
    var dev = ctx.enqueue_create_buffer[DType.int32](n_buf)
    var host = ctx.enqueue_create_host_buffer[DType.int32](n_buf)
    ctx.synchronize()
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    for i in range(n, n_buf):
        host.unsafe_ptr().unsafe_store(i, Int32(0))
    ctx.enqueue_copy(dst_buf=dev, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return dev^


def _download_f32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    """The first `n` elements of a device buffer, as a host list.

    `[[mojo-buffer-freed-at-last-use]]`: `buf` is a `mut` REFERENCE, so it
    is the CALLER's owner that has to stay alive, and every caller below
    keeps its buffers in locals until after its last download. A
    `DeviceBuffer` is dead at its `.unsafe_ptr()` and this repository has
    lost a night to that."""
    if n <= 0:
        return List[Float32]()
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    if n == len(buf):
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    else:
        var view = buf.create_sub_buffer[DType.float32](0, n)
        ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^
    return out^


def _plant_bits_f32(
    ctx: DeviceContext,
    mut buf: DeviceBuffer[DType.float32],
    count: Int,
    idx: Int,
    pattern: UInt32,
) raises:
    """Write one bit pattern into one cell of a device buffer, as a HOST
    ROUND TRIP.

    `transformer/impl/.../modeling_llama.mojo::_plant_bits`'s spelling
    and its reason: a plant kernel would need an `Int32` index buffer and a
    `UInt32` bit buffer, which are element types nothing else in this lane
    uses, and a download-store-upload uses only forms already exercised
    here. It costs two drains and a full buffer copy per plant, on the
    handful of plants clause (f) performs, in a profile that publishes no
    timing number.

    **AN INDEX THAT MISSES IS REFUSED.** A plant landing outside the USED
    region is worse than no plant: the gate goes green and proves nothing.
    The bound is `count`, the used cell count, not `len(buf)`."""
    if idx < 0 or idx >= count:
        raise Error(
            String("loss_check: plant index ")
            + String(idx)
            + " is outside the used region of "
            + String(count)
            + " cells. A plant that misses makes the gate green and proves"
            + " nothing ([[verify-reach-not-output]])."
        )
    var host = ctx.enqueue_create_host_buffer[DType.float32](count)
    ctx.synchronize()
    var view = buf.create_sub_buffer[DType.float32](0, count)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    host.unsafe_ptr().unsafe_store(idx, f32_from_bits(pattern))
    ctx.enqueue_copy(dst_buf=view, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    _ = view


def nonfinite_cells(values: List[Float32]) -> Int:
    """How many cells are NaN or infinity, BY BITS.

    NOT BY COMPARES. Metal FLUSHES COMPARE OPERANDS (IDENTITY_PATHS row 49,
    DEVIATION 746(i)), so `v != v` is a test with two meanings across
    columns while a mask-and-compare on the exponent field has one. This
    must return exactly 1 for a single plant on every column or clause (f)
    is measuring the toolchain rather than the refusal."""
    var n = 0
    for i in range(len(values)):
        if is_nonfinite_bits(values[i]):
            n += 1
    return n


# ===========================================================================
# ONE DEVICE RUN
# ===========================================================================


struct DeviceRun(Movable):
    """One device loss call's stages, card order, plus the two things a
    verdict needs beside them.

    `poison` is the number of cells that came back still holding
    `BITS_POISON`, i.e. cells NO KERNEL WROTE. On a CLEAN build that must
    be zero: a stage the card records and nothing wrote is a stage whose
    hash is whatever the allocator left there. Under
    `SAB_IGNORED_ROW_SKIPPED` it must NOT be zero, and that is the whole
    arm -- `loss.mojo`'s own switch docstring says the arm is ALWAYS INERT
    against a zero-filled buffer and never inert against a poisoned one
    (DEVIATION 1456).

    `count` is the host integer the forward was given, kept so that clause
    (c) can prove it passed the SAME one to every chunk."""

    var dump: List[List[Float32]]
    var poison: Int
    var count: Int

    def __init__(
        out self,
        var dump: List[List[Float32]],
        poison: Int,
        count: Int,
    ):
        self.dump = dump^
        self.poison = poison
        self.count = count


def run_device_rows(
    ctx: DeviceContext,
    c: CeCase,
    logits: List[Float32],
    targets: List[Int32],
    count_for_divisor: Int,
) raises -> DeviceRun:
    """ONE whole loss call on the device, from FRESH everything, stages
    returned on the host.

    `logits` and `targets` are passed in rather than read from the case, so
    that clause (c) can hand this function a ROW CHUNK of a larger case
    with the larger case's `count`. `c` supplies only the shape fields that
    do not change under chunking -- `vocab`, `reduction`, `eps_on`,
    `num_items` -- and `n_rows` is taken from `len(targets)`, never from
    `c.n_rows`, because under chunking those two differ and reading the
    wrong one is how a chunk silently runs the whole case.

    FRESH BUFFERS ON EVERY CALL. That is what clause (b) means by "each run
    its own fresh state, stage buffers and kernel dispatches", and building
    them here rather than hoisting them is what lets clause (b) be a loop
    over this function.

    EVERY OUTPUT BUFFER IS POISONED BEFORE THE CALL (DEVIATION 1456). A
    fresh `enqueue_create_buffer` may or may not be zeroed and the profile
    may not depend on which; more to the point, a zero-filled buffer makes
    `SAB_IGNORED_ROW_SKIPPED` bit-inert, because the value the arm declines
    to store is `+0.0`.

    `[[mojo-buffer-freed-at-last-use]]`: every device object is a LOCAL and
    every one is still alive when it is downloaded, because
    `ctx.synchronize()` runs before the downloads and the explicit `_ =`
    moves at the foot keep the owners alive past the last `.unsafe_ptr()`
    any of them hands out. A `DeviceBuffer` is dead at `.unsafe_ptr()`.
    """
    var cfg = ce_case_config(c)
    var v = c.vocab
    var n = len(targets)
    if n < 1:
        raise Error(String("loss_check: run_device_rows with zero rows"))
    if len(logits) != n * v:
        raise Error(
            String("loss_check: run_device_rows got ")
            + String(len(logits))
            + " logits for N*V = "
            + String(n * v)
        )
    var cells = n * v
    var smoothing = cfg.smoothing_is_spelled()
    # The smoothing buffers are ONE ELEMENT when smoothing is off, which is
    # what `identical_ce_forward_into`'s docstring permits ("may be
    # single-element placeholders ... they are not touched then"). The
    # placeholder is still POISONED, so a kernel that touched it anyway
    # would be caught by the survivor count rather than by nothing.
    var smooth_cells = 1
    var smooth_rows = 1
    if smoothing:
        smooth_cells = cells
        smooth_rows = n

    var d_logits = _upload_f32(ctx, logits)
    var d_targets = _upload_i32(ctx, targets)
    var d_ones = _upload_f32(ctx, ce_ones(identical_ce_ones_floats(n, v)))
    var nws = identical_ce_workspace_max_floats(n, v, cfg.reduction)
    var d_ws = _upload_f32(ctx, ce_poison(nws))

    var d_max = _upload_f32(ctx, ce_poison(n))
    var d_shift = _upload_f32(ctx, ce_poison(cells))
    var d_expo = _upload_f32(ctx, ce_poison(cells))
    var d_denom = _upload_f32(ctx, ce_poison(n))
    var d_logdenom = _upload_f32(ctx, ce_poison(n))
    var d_lpt = _upload_f32(ctx, ce_poison(n))
    var d_nll = _upload_f32(ctx, ce_poison(n))
    var d_logp = _upload_f32(ctx, ce_poison(smooth_cells))
    var d_lpsum = _upload_f32(ctx, ce_poison(smooth_rows))
    var d_smooth = _upload_f32(ctx, ce_poison(smooth_rows))
    var d_row = _upload_f32(ctx, ce_poison(n))
    var d_total = _upload_f32(ctx, ce_poison(1))
    var d_loss = _upload_f32(ctx, ce_poison(1))
    var d_weights = _upload_f32(ctx, ce_poison(cells))
    var d_dlogits = _upload_f32(ctx, ce_poison(cells))

    identical_ce_forward_into(
        ctx, d_max, d_shift, d_expo, d_denom, d_logdenom, d_lpt, d_nll,
        d_logp, d_lpsum, d_smooth, d_row, d_total, d_loss, d_logits,
        d_targets, d_ones, d_ws, n, count_for_divisor, cfg,
    )
    if cfg.reduction != REDUCTION_NONE:
        # `SAB_W_VIA_EXP_LOGP` reads `d_logp`, which is a ONE-ELEMENT
        # placeholder when smoothing is off. `ce_weights_kernel`'s
        # docstring claims the launcher refuses the arm in that case and
        # **the launcher contains no such refusal** (DEVIATION 1458). This
        # file refuses instead, at the one call site, so an armed build
        # cannot walk off the end of a one-cell allocation and call the
        # resulting garbage a sabotage.
        if loss_sabotage_name() == "W_VIA_EXP_LOGP" and not smoothing:
            raise Error(
                String("loss_check: SAB_W_VIA_EXP_LOGP reads the `logp`")
                + " buffer, which is a ONE-ELEMENT placeholder when"
                + " smoothing is off, so on case "
                + String(c.name)
                + " the arm would read "
                + String(cells)
                + " cells out of a 1-cell allocation. That is an"
                + " out-of-bounds read, not a sabotage."
                + " `ce_weights_kernel`'s docstring says the launcher"
                + " refuses this and `identical_ce_backward_into` does"
                + " NOT (DEVIATION 1458). Run this arm on a SMOOTHING"
                + " case."
            )
        identical_ce_backward_into(
            ctx, d_weights, d_dlogits, d_expo, d_denom, d_logp, d_targets,
            n, count_for_divisor, cfg,
        )
    ctx.synchronize()

    # ---- the dump, card order, index-aligned with `ce_stage_tag` --------
    var out = List[List[Float32]]()
    for _i in range(CE_STAGE_COUNT):
        out.append(List[Float32]())
    out[ST_INPUT_LOGITS] = _download_f32(ctx, d_logits, cells)
    # ST_INPUT_TARGETS is an INTEGER stage and stays empty here; it is
    # compared as integers by `compare_int_stage`.
    out[ST_MAX] = _download_f32(ctx, d_max, n)
    out[ST_SHIFT] = _download_f32(ctx, d_shift, cells)
    out[ST_EXP] = _download_f32(ctx, d_expo, cells)
    out[ST_DENOM] = _download_f32(ctx, d_denom, n)
    out[ST_LOGDENOM] = _download_f32(ctx, d_logdenom, n)
    out[ST_LOGP_TARGET] = _download_f32(ctx, d_lpt, n)
    out[ST_NLL] = _download_f32(ctx, d_nll, n)
    if smoothing:
        out[ST_LOGP] = _download_f32(ctx, d_logp, cells)
        out[ST_LOGP_SUM] = _download_f32(ctx, d_lpsum, n)
        out[ST_SMOOTH] = _download_f32(ctx, d_smooth, n)
    out[ST_ROW] = _download_f32(ctx, d_row, n)

    # The three HOST constants. `stage_is_host_constant`'s docstring is the
    # argument for why these are here and what they can and cannot catch.
    if cfg.reduction != REDUCTION_NONE:
        var dv = List[Float32]()
        dv.append(ce_divisor(cfg.reduction, count_for_divisor, cfg.num_items))
        out[ST_DIVISOR] = dv^
        out[ST_TOTAL] = _download_f32(ctx, d_total, 1)
        out[ST_LOSS] = _download_f32(ctx, d_loss, 1)
        out[ST_WEIGHTS] = _download_f32(ctx, d_weights, cells)
        out[ST_DLOGITS] = _download_f32(ctx, d_dlogits, cells)
    var tv = ce_smoothing_targets(cfg.eps, v)
    var tvl = List[Float32]()
    tvl.append(tv[0])
    tvl.append(tv[1])
    out[ST_TARGET_VEC] = tvl^

    # ---- the poison survivors, over every stage that MUST be written ----
    var survivors = 0
    for i in range(CE_STAGE_COUNT):
        if stage_is_int(i) or stage_is_host_constant(i):
            continue
        if i == ST_INPUT_LOGITS:
            continue  # an input, never written by a kernel
        survivors += count_poison(out[i])

    _ = d_logits^
    _ = d_targets^
    _ = d_ones^
    _ = d_ws^
    _ = d_max^
    _ = d_shift^
    _ = d_expo^
    _ = d_denom^
    _ = d_logdenom^
    _ = d_lpt^
    _ = d_nll^
    _ = d_logp^
    _ = d_lpsum^
    _ = d_smooth^
    _ = d_row^
    _ = d_total^
    _ = d_loss^
    _ = d_weights^
    _ = d_dlogits^
    return DeviceRun(out^, survivors, count_for_divisor)


def run_host_case(
    c: CeCase, logits: List[Float32], targets: List[Int32]
) raises -> List[List[Float32]]:
    """The ORACLE's stages for the same call, card order, index-aligned
    with `ce_stage_tag`.

    `ce_forward_oracle` REFUSES its inputs first (`ce_refuse_inputs`), so a
    case whose logits carry a non-finite value never reaches a stage here.
    The device has no such refusal at all, which clause (f) MEASURES."""
    var cfg = ce_case_config(c)
    var st = ce_forward_oracle(logits, targets, cfg)
    if cfg.reduction != REDUCTION_NONE:
        ce_backward_oracle(st, targets, cfg)
    var out = List[List[Float32]]()
    for _i in range(CE_STAGE_COUNT):
        out.append(List[Float32]())
    out[ST_INPUT_LOGITS] = logits.copy()
    out[ST_MAX] = st.max_v.copy()
    out[ST_SHIFT] = st.shift.copy()
    out[ST_EXP] = st.expo.copy()
    out[ST_DENOM] = st.denom.copy()
    out[ST_LOGDENOM] = st.logdenom.copy()
    out[ST_LOGP_TARGET] = st.logp_target.copy()
    out[ST_NLL] = st.nll.copy()
    out[ST_LOGP] = st.logp.copy()
    out[ST_LOGP_SUM] = st.logp_sum.copy()
    out[ST_SMOOTH] = st.smooth.copy()
    out[ST_ROW] = st.row.copy()
    out[ST_DIVISOR] = st.divisor.copy()
    out[ST_TOTAL] = st.total.copy()
    out[ST_LOSS] = st.loss.copy()
    out[ST_TARGET_VEC] = st.target_vec.copy()
    out[ST_WEIGHTS] = st.weights.copy()
    out[ST_DLOGITS] = st.dlogits.copy()
    _ = st^
    return out^


def host_count(c: CeCase, targets: List[Int32]) -> Int:
    return ce_count(targets, IGNORE_INDEX_DEFAULT)


# ===========================================================================
# COMPARING
# ===========================================================================


@fieldwise_init
struct StageDiff(Copyable, Movable):
    var name: String
    var n_cells: Int
    var n_diff: Int
    var first: Int


def compare_stage(
    name: String, host: List[Float32], dev: List[Float32], loud: Bool
) raises -> StageDiff:
    """Bitwise, cell by cell.

    A LENGTH mismatch is reported as such rather than compared to the
    shorter of the two, because a stage that is the wrong SIZE is a
    different defect from a stage that is the wrong VALUE, and comparing to
    the shorter of two lists is how a truncated stage passes.

    BY BITS AND NEVER BY COMPARES. `host[i] == dev[i]` would call `+0.0`
    and `-0.0` equal, which would launder the one three-vendor split
    IDENTITY_PATHS row 39 measured (`max(+0.0, -0.0)` is `-0.0` on Apple
    and `+0.0` on NVIDIA and AMD) -- and that split is not hypothetical
    here: contract 5.1 says a row of 128,256 logits reaches both zero signs
    easily, and `L_IGNORED_ROW_NEG_ZERO` is an arm whose ENTIRE effect is a
    zero sign. Metal also flushes compare operands (row 49), so a
    compare-written gate has a different meaning on different columns.
    Everything here goes through `bitcast[DType.uint32]`.

    `loud` prints per stage. It is False wherever a difference is EXPECTED
    (every negative control) or wherever the caller reports the failure
    itself with more context, so the only lines on stdout are lines a
    reader should act on."""
    if len(host) != len(dev):
        raise Error(
            String("loss_check: stage ")
            + name
            + " has "
            + String(len(host))
            + " cells on one side and "
            + String(len(dev))
            + " on the other. A stage of the wrong SHAPE is a different"
            + " defect from a stage of the wrong VALUE and is not compared"
            + " to the shorter of the two."
        )
    var n_diff = 0
    var first = -1
    for i in range(len(host)):
        if bits_of(host[i]) != bits_of(dev[i]):
            n_diff += 1
            if first < 0:
                first = i
    if n_diff == 0:
        if loud:
            print("  OK    " + name + "  (" + String(len(host)) + " cells)")
    elif loud:
        print(
            "  MOVED "
            + name
            + "  "
            + String(n_diff)
            + " of "
            + String(len(host))
            + " cells, first cell "
            + String(first)
            + "  a "
            + bits32_hex(host[first])
            + "  b "
            + bits32_hex(dev[first])
        )
    return StageDiff(name, len(host), n_diff, first)


def compare_dumps(
    a: List[List[Float32]], b: List[List[Float32]], loud: Bool
) raises -> List[StageDiff]:
    """Contract section 9's stages, in section 9's order, by tag.

    THE FIRST ASSERTION IS THE LENGTH OF THE DUMPS THEMSELVES. Two dumps of
    different lengths mean the two halves of this lane disagree about what
    a stage list IS, and every per-stage comparison after that would be
    comparing misaligned tags -- `core/identity_trace.mojo`'s own header
    calls that the worst thing the instrument can do."""
    if len(a) != CE_STAGE_COUNT or len(b) != CE_STAGE_COUNT:
        raise Error(
            String("loss_check: the host dump has ")
            + String(len(a))
            + " stages and the device dump has "
            + String(len(b))
            + ", and contract section 9 lists "
            + String(CE_STAGE_COUNT)
            + " (see DEVIATION 1463 -- the section says NINETEEN and lists"
            + " TWENTY)"
        )
    var out = List[StageDiff]()
    for i in range(CE_STAGE_COUNT):
        if stage_is_int(i):
            # Not a float stage. `compare_int_stage` handles the two
            # integer stages and a zero-length row here keeps the index
            # alignment with `ce_stage_tag` intact.
            out.append(StageDiff(ce_stage_tag(i), 0, 0, -1))
            continue
        out.append(compare_stage(ce_stage_tag(i), a[i], b[i], loud))
    return out^


def compare_int_stage(
    name: String, host: List[Int32], dev: List[Int32]
) raises -> StageDiff:
    """The two INTEGER stages. Integers do not flush and do not round, so
    this is an ordinary comparison and it is written separately precisely
    so that nobody is tempted to bitcast an `Int32` through the float
    path -- where a target that happened to hold a NaN bit pattern would
    become "non-finite" to every audit in this file."""
    if len(host) != len(dev):
        raise Error(
            String("loss_check: integer stage ")
            + name
            + " has "
            + String(len(host))
            + " cells on one side and "
            + String(len(dev))
            + " on the other"
        )
    var n_diff = 0
    var first = -1
    for i in range(len(host)):
        if host[i] != dev[i]:
            n_diff += 1
            if first < 0:
                first = i
    return StageDiff(name, len(host), n_diff, first)


def count_moved(diffs: List[StageDiff]) -> Int:
    var n = 0
    for i in range(len(diffs)):
        if diffs[i].n_diff > 0:
            n += 1
    return n


def total_cells(dump: List[List[Float32]]) -> Int:
    var n = 0
    for i in range(len(dump)):
        n += len(dump[i])
    return n


def first_moved_index(diffs: List[StageDiff]) -> Int:
    for i in range(len(diffs)):
        if diffs[i].n_diff > 0:
            return i
    return -1


def first_moved(diffs: List[StageDiff]) -> String:
    """The contract's own report shape, verbatim: `<tag> on X of Y cells`.

    Contract 10.1's discipline is that an arm must move the stage its OWN
    clause writes "and no earlier one", so the FIRST moved stage is the
    finding and the count is the evidence that it is not a single-bit
    coincidence."""
    for i in range(len(diffs)):
        if diffs[i].n_diff > 0:
            return (
                diffs[i].name
                + " on "
                + String(diffs[i].n_diff)
                + " of "
                + String(diffs[i].n_cells)
                + " cells"
            )
    return String("")


# ===========================================================================
# THE CARD
# ===========================================================================


def write_card(
    mut trace: IdentityTrace,
    prefix: String,
    c: CeCase,
    dump: List[List[Float32]],
    targets: List[Int32],
    count: Int,
) raises -> Int:
    """Contract section 9's stages onto the card, in section 9's order,
    each exactly once, each carrying this driver's prefix. Returns how many
    records were emitted.

    **THE ABSENT STAGES ARE ABSENT AND ARE NOT WRITTEN AS ZEROS.**
    `CeStages`'s own docstring asks for this in as many words -- a check
    that hashes an empty list records a zero-length stage, which the differ
    must treat as ABSENT rather than as agreeing with an absent one. So a
    run at `EPS == 0` emits seventeen records and a run with smoothing
    emits twenty, and `check_card_tags` knows which to expect from the
    case. A card that always emitted twenty, three of them zero-length,
    would let `tools/identity_trace_diff.py` align a smoothing run against
    a non-smoothing one and report a plausible answer for a pair of stages
    that were never the same stage.

    `ce.count` and `input.targets` go through `record_list_i32`, which
    hashes four-byte integers, and NOT through the float recorder. The
    dtype is part of the record (`_dtype_name`), so a card that recorded a
    target as a float would not even align against one that recorded it as
    an integer -- which is the failure being loud instead of silent."""
    var emitted = 0
    for i in range(CE_STAGE_COUNT):
        if not stage_present(i, c):
            continue
        var tag = prefix + "." + ce_stage_tag(i)
        if i == ST_INPUT_TARGETS:
            trace.record_list_i32(tag, targets)
            emitted += 1
            continue
        if i == ST_COUNT:
            var cl = List[Int32]()
            cl.append(Int32(count))
            trace.record_list_i32(tag, cl)
            emitted += 1
            continue
        trace.record_list_f32(tag, dump[i])
        emitted += 1
    return emitted


def expected_card_records(c: CeCase) -> Int:
    var n = 0
    for i in range(CE_STAGE_COUNT):
        if stage_present(i, c):
            n += 1
    return n


def check_card_tags(path: String, c: CeCase) raises -> Int:
    """The card holds this case's stages, in contract section 9's order,
    each exactly once, each carrying this driver's prefix.

    THIS IS A CHECK ON THE COMPOSITION AND NOT ON THE ARITHMETIC.
    `IdentityTrace` enforces tag UNIQUENESS and raises, which catches two
    writers claiming one tag; nothing but this function catches a MISSING
    tag, because a card with sixteen records is a card, and
    `tools/identity_trace_diff.py` would align it against a seventeen-record
    card and report a plausible WRONG ANSWER.

    AND IT IS THE CHECK THAT CATCHES A CARD THAT WAS NEVER WRITTEN, which
    was DEVIATION 970's actual damage: `read_trace_lines` on a path nothing
    wrote raises rather than returning an empty list that a lenient gate
    would call zero differences."""
    var want = expected_card_records(c)
    var lines = read_trace_lines(path)
    print(
        "card: "
        + String(len(lines))
        + " records at "
        + path
        + ", case "
        + String(c.name)
        + " wants "
        + String(want)
        + " of contract section 9's "
        + String(CE_STAGE_COUNT)
    )
    if len(lines) != want:
        raise Error(
            String("loss_check: the card has ")
            + String(len(lines))
            + " records and case "
            + String(c.name)
            + " produces "
            + String(want)
            + " stages"
        )
    var k = 0
    for i in range(CE_STAGE_COUNT):
        if not stage_present(i, c):
            continue
        var fields = lines[k].split("\t")
        if len(fields) < 2:
            raise Error(
                String("loss_check: malformed trace record: ") + lines[k]
            )
        var got = String(fields[1])
        var expect = String(TAG_PREFIX) + "." + ce_stage_tag(i)
        if got != expect:
            raise Error(
                String("loss_check: card record ")
                + String(k)
                + " is '"
                + got
                + "', contract section 9 wants '"
                + expect
                + "'. A renamed or reordered tag does not make"
                + " identity_trace_diff.py's diff smaller, it makes its"
                + " ALIGNMENT wrong."
            )
        k += 1
    print(
        "card: "
        + String(want)
        + "/"
        + String(want)
        + " tags in contract section 9's order, all unique"
    )
    return len(lines)


# ===========================================================================
# PREFLIGHT: the assertions the contract, the fixture and the oracle ASK
# THIS FILE FOR
#
# DEVIATION 1454. Every one of these is about a constant or an identity
# that is cheap to check here and catastrophic to get wrong, because a
# wrong constant does not look like a wrong constant -- it looks like a
# kernel bug on every stage downstream of it. They run BEFORE any device
# call, so a build with a bad constant fails in a second rather than after
# a case sweep.
# ===========================================================================


def preflight() raises:
    print("preflight: the assertions the contract and the fixture asked for")

    # ---- 1. The fixture's frozen bit patterns ---------------------------
    if not ce_profile_constants_are_intact():
        raise Error(
            String("loss_check: a fixture constant has the wrong bits.")
            + " eps_smooth "
            + bits32_hex(ce_eps_smooth())
            + " wants 0x3dcccccd. `[[mojo-string-float-roundtrip]]`: this"
            + " toolchain's Float32 text is lossy and a constant that was"
            + " right when it was typed is not a constant the toolchain"
            + " agrees with."
        )
    print(
        "  fixture constants OK: eps_smooth "
        + bits32_hex(ce_eps_smooth())
        + " (and it is NOT one tenth -- every smoothing clause is about"
        + " rounding)"
    )

    # ---- 2. THE THREE splitmix64 COPIES AGREE ---------------------------
    # `loss_fixture.mojo::ce_splitmix64` is the THIRD copy of this hash in
    # the repository, after the mamba corpus's and the transformer
    # fixture's. Copying was DEVIATION 1000's decision and its stated cost
    # is "two copies of a hash have two chances to be edited apart"; this
    # lane makes it three.
    #
    # `[[mojo-amp-plus-is-bitwise-and]]` is what this actually guards.
    # Mojo's `&+` computes `x & k` with NO COMPILE ERROR, and a `+` "fixed"
    # into a `&+` in ONE copy and not the others is exactly the edit this
    # catches. It has happened here twice.
    var seeds: List[UInt64] = [
        UInt64(0),
        UInt64(1),
        UInt64(0x9E3779B97F4A7C15),
        UInt64(0xFFFFFFFFFFFFFFFF),
        UInt64(0x43654C6F73734631),
    ]
    for i in range(len(seeds)):
        var a = ce_splitmix64(seeds[i])
        if a != corpus_splitmix64(seeds[i]):
            raise Error(
                String("loss_check: the loss fixture's splitmix64 and the")
                + " mamba corpus's disagree at seed "
                + String(i)
                + " ([[mojo-amp-plus-is-bitwise-and]] is the usual way)"
            )
        if a != fixture_splitmix64(seeds[i]):
            raise Error(
                String("loss_check: the loss fixture's splitmix64 and the")
                + " transformer fixture's disagree at seed "
                + String(i)
            )
    print(
        "  splitmix64: all THREE copies (loss, transformer, mamba corpus)"
        " agree on "
        + String(len(seeds))
        + " seeds"
    )

    # ---- 3. THE STAGE COUNT, and the contract's own arithmetic ---------
    # DEVIATION 1463. Printed on every run rather than buried, because a
    # gate that had quietly used the contract's stated NINETEEN would have
    # dropped a stage and reported a green.
    print(
        "  stages: this file uses "
        + String(CE_STAGE_COUNT)
        + " and contract section 9 LISTS "
        + String(CE_STAGE_COUNT)
        + " while its prose says NINETEEN (DEVIATION 1463 -- the prose is"
        + " off by one; the SEVENTEEN it also quotes is right, being"
        + " twenty minus the three smoothing-only stages)"
    )
    var seen = List[String]()
    for i in range(CE_STAGE_COUNT):
        var t = ce_stage_tag(i)
        for j in range(len(seen)):
            if seen[j] == t:
                raise Error(
                    String("loss_check: stage tag '")
                    + t
                    + "' appears twice in ce_stage_tag"
                )
        seen.append(t)

    # ---- 4. THE LEAF RULE, restated in the fixture, checked here -------
    # `case_leaf_count` is a RESTATEMENT of `contract_leaf_size` and a
    # restatement can drift. Every fold-arm prediction in this file is
    # built on it, so it is checked against the GEMM oracle's own function
    # at every length the case set uses -- which is the same mitigation the
    # transformer lane applied to its copied splitmix64.
    var lens: List[Int] = [
        1, 2, 3, 4, 5, 6, 8, 16, 64, 128, 129, 256, 300, 512, 1024, 128256
    ]
    for i in range(len(lens)):
        var k = lens[i]
        var want_leaf = contract_leaf_size(k)
        var want_p = (k + want_leaf - 1) // want_leaf
        if k <= 0:
            want_p = 0
        if case_leaf_count(k) != want_p:
            raise Error(
                String("loss_check: loss_fixture.case_leaf_count(")
                + String(k)
                + ") is "
                + String(case_leaf_count(k))
                + " and gemm_oracle.contract_leaf_size says P = "
                + String(want_p)
                + ". The fixture's restatement of the leaf rule has"
                + " drifted, and every fold-arm prediction in this file"
                + " rests on it."
            )
    print(
        "  the leaf rule: loss_fixture's restatement agrees with"
        " gemm_oracle.contract_leaf_size on "
        + String(len(lens))
        + " lengths including 128, 129, 300 and the shipped 128256"
    )

    # ---- 5. The two theorems of contract 8.2 that the exact families use
    # `identical_exp(+0.0)` must be EXACTLY 1.0 and `identical_exp(-200)`
    # EXACTLY +0.0. Contract 8.2(b) and 12.2 both rest on these and neither
    # has ever been run on this toolchain. They are HOST assertions here;
    # the DEVICE's own answers are what clause (a) compares, so a device
    # `identical_exp` that disagreed would show at `ce.exp`.
    var e0 = identical_exp(Float32(0.0))
    if bits_of(e0) != UInt32(0x3F800000):
        raise Error(
            String("loss_check: identical_exp(+0.0) is ")
            + bits32_hex(e0)
            + " and contract 8.2(b) needs EXACTLY 1.0 (0x3f800000). The"
            + " proof that `denom >= 1.0`, the proof that `identical_log`"
            + " never sees a pathological argument, and both EXACT"
            + " families all rest on it."
        )
    var e200 = identical_exp(Float32(-200.0))
    if bits_of(e200) != UInt32(0x00000000):
        raise Error(
            String("loss_check: identical_exp(-200.0) is ")
            + bits32_hex(e200)
            + " and contract 12.2 needs EXACTLY +0.0 (portable_expf"
            + " returns +0.0 below -87.33655). The SATURATING exact family"
            + " and L_NLL_VIA_LOG_W's separating fixture both rest on it."
        )
    var l1 = identical_log(Float32(1.0))
    if bits_of(l1) != UInt32(0x00000000):
        raise Error(
            String("loss_check: identical_log(1.0) is ")
            + bits32_hex(l1)
            + " and L_LOG_STDLIB's INERT half at base_n1_v1 needs it to be"
            + " EXACTLY +0.0"
        )
    print(
        "  contract 8.2: identical_exp(+0.0) = "
        + bits32_hex(e0)
        + ", identical_exp(-200) = "
        + bits32_hex(e200)
        + ", identical_log(1.0) = "
        + bits32_hex(l1)
    )

    # ---- 6. THE FIXTURE'S OWN CLAIMS, over every case ------------------
    # These are the claims every INERT half rests on, and a claim that is
    # only argued is a claim that is wrong on one case out of twenty-five --
    # which is the hardest kind of flake to read, because the arm simply
    # looks inert.
    # `L_MAX_SEED_ZERO`'s inert condition, asserted on its DECLARED INERT
    # CASE and on no other (DEVIATION 1491).
    assert_row_has_a_positive_logit(ce_case(ce_case_by_name("base_n4_v8")))
    var checked = 0
    for k in range(CE_CASE_COUNT):
        if k == CE_CASE_SHIPPED_V and not env_on(
            "MOJOLEARN_LOSS_CHECK_SHIPPED_V"
        ):
            continue
        var c = ce_case(k)
        checked += assert_no_zero_or_subnormal_logits(c)
        # DEVIATION 1491, AND THE FIRST TWO RUNS OF THIS GATE FOUND IT.
        # `assert_row_has_a_positive_logit` was called HERE, over every case,
        # and it raised twice for two DIFFERENT and both-correct reasons:
        # `base_n1_v1` (where the arm is a WITNESS, not inert -- DEVIATION
        # 1490) and then `long_n300_v8` row 40.
        #
        # The second is the one that condemns the call site. The property is
        # PROBABILISTIC over a hashed fixture: a row of 8 hashed logits is
        # all-negative with probability about 1/256, so across 300 rows the
        # invariant holds only about 31% of the time. It is not a property of
        # the fixture, it is a property of the seed, and asserting it over
        # every case would make this gate fail or pass on a coin flip.
        #
        # It is `L_MAX_SEED_ZERO`'s INERT condition and nothing else, so it
        # belongs on that arm's declared inert case ALONE. Moved below.
        _ = c
        if ce_case_has_backward(c):
            if ce_case_unignored_count(c) < 1:
                raise Error(
                    String("loss_check: case ")
                    + String(c.name)
                    + " has count 0 under a reduction that needs a"
                    + " divisor; ce_divisor REFUSES that by name"
                    + " (contract 5.5, DEVIATION 1156)"
                )
        # The row maximum the fixture predicts with must be the ORACLE's.
        # `row_max_oracle` is a COPY of `loss_oracle::_row_max`'s loop and
        # this is what turns the copy from a risk into a cross-check.
        var x = ce_case_logits(c)
        var host = run_host_case(c, x, ce_case_targets(c))
        for r in range(c.n_rows):
            var m = row_max_oracle(x, r * c.vocab, c.vocab)
            if bits_of(m) != bits_of(host[ST_MAX][r]):
                raise Error(
                    String("loss_check: loss_fixture.row_max_oracle and")
                    + " loss_oracle._row_max disagree on case "
                    + String(c.name)
                    + " row "
                    + String(r)
                    + ": "
                    + bits32_hex(m)
                    + " against "
                    + bits32_hex(host[ST_MAX][r])
                    + ". Every L_MAX_* prediction in this file is built on"
                    + " the fixture's copy."
                )
    print(
        "  fixture: "
        + String(checked)
        + " logits carry no signed zero and no subnormal outside the two"
        + " cases that plant them, every row of every other case has a"
        + " positive logit, and the fixture's row maximum IS the oracle's"
    )

    # ---- 6b. THE SEPARATING CONDITIONS OF THE ARM TABLE, COMPUTED ------
    # DEVIATION 1464. Clause (g)'s expectation table names, for each arm, a
    # WITNESS case and an INERT case. **Those pairings are CLAIMS about the
    # fixture and every one of them can be checked on the host before a
    # device exists.** Checking them here is what turns the table from a
    # list of names into a set of measured predicates: if
    # `adv_signed_zeros` did not actually carry both zero signs, or if
    # `base_n1_v1`'s `nll` were not actually `-0.0`, the corresponding arm
    # would report inert and a reader would conclude the pin was vacuous --
    # which is the exact failure mode contract 10.1's last paragraph warns
    # about for three of its arms.
    #
    # Two of these go further than a condition and SIMULATE the arm on the
    # host, which is legitimate for exactly two of them and for no others:
    # `L_MAX_SEED_ZERO` and `L_MAX_TOPK_PREFIX` change only the SEED and
    # the RANGE of a fold whose combine (`identical_fmax`) is exactly
    # commutative and associative, so a host ascending loop and the
    # device's block halving tree return the same bits and the simulation
    # is a prediction rather than a guess. `L_MAX_PLAIN_COMPARE` is NOT
    # simulated, because a plain `>` fold is not associative in the
    # presence of signed zeros and the two schedules can legitimately
    # disagree -- so only its INERTNESS is predicted here.
    var zc_w = ce_case(ce_case_by_name(String("adv_signed_zeros")))
    var zc_i = ce_case(ce_case_by_name(String("base_n4_v8")))
    if not case_has_both_zero_signs(zc_w):
        raise Error(
            String("loss_check: adv_signed_zeros does NOT carry both zero")
            + " signs in any row, so L_MAX_PLAIN_COMPARE has no witness and"
            + " the arm would look inert ([[reached-but-inert]])."
        )
    if case_has_both_zero_signs(zc_i):
        raise Error(
            String("loss_check: base_n4_v8 DOES carry both zero signs, so")
            + " it is not L_MAX_PLAIN_COMPARE's inert half and the arm's"
            + " reach proof has no second side."
        )
    print(
        "  L_MAX_PLAIN_COMPARE: adv_signed_zeros carries both zero signs,"
        " base_n4_v8 carries neither (row 39 measured max(+0.0, -0.0) as"
        " -0.0 on Apple and +0.0 on NVIDIA and AMD)"
    )

    var neg_w = ce_case(ce_case_by_name(String("base_n1_v1")))
    if not case_row_nll_is_negative_zero(neg_w):
        raise Error(
            String("loss_check: base_n1_v1's row `nll` is NOT exactly")
            + " -0.0, so L_NEG_VIA_ZERO_SUB and L_SMOOTH_ALWAYS_SPELLED"
            + " have NO WITNESS IN THIS FIXTURE. Contract 8.1 says a"
            + " V == 1 row reaches lp_y == +0.0 -- if that is false on this"
            + " toolchain, contract 8.1 is the thing that is wrong."
        )
    if case_row_nll_is_negative_zero(zc_i):
        raise Error(
            String("loss_check: base_n4_v8 HAS a -0.0 row loss, so it is")
            + " not the inert half of L_SMOOTH_ALWAYS_SPELLED"
        )
    print(
        "  L_NEG_VIA_ZERO_SUB / L_SMOOTH_ALWAYS_SPELLED: base_n1_v1's nll"
        " IS exactly -0.0 and base_n4_v8's is not"
    )

    var uf_w = ce_case(ce_case_by_name(String("adv_underflow_y")))
    if not case_target_exp_underflows(uf_w):
        raise Error(
            String("loss_check: adv_underflow_y's e[y] is NOT exactly")
            + " +0.0, so L_NLL_VIA_LOG_W has no witness -- the refused"
            + " spelling only returns +inf where the target exponential"
            + " has underflowed"
        )
    if case_target_exp_underflows(zc_i):
        raise Error(
            String("loss_check: base_n4_v8 has an underflowed e[y], so it")
            + " is not L_NLL_VIA_LOG_W's inert half"
        )
    print(
        "  L_NLL_VIA_LOG_W: adv_underflow_y's e[y] IS exactly +0.0 and"
        " base_n4_v8's is normal"
    )

    # `L_MAX_SEED_ZERO`, SIMULATED on the host.
    var sz_w = ce_case(ce_case_by_name(String("adv_all_negative")))
    var sz_moved = 0
    var sz_inert = 0
    for side in range(2):
        # `CeCase` is not ImplicitlyCopyable; name the copy.
        var cc = sz_w.copy()
        if side == 1:
            cc = zc_i.copy()
        var xs = ce_case_logits(cc)
        for r in range(cc.n_rows):
            var a = row_max_oracle(xs, r * cc.vocab, cc.vocab)
            var b = row_max_seeded_zero(xs, r * cc.vocab, cc.vocab)
            if bits_of(a) != bits_of(b):
                if side == 0:
                    sz_moved += 1
                else:
                    sz_inert += 1
    if sz_moved == 0:
        raise Error(
            String("loss_check: a +0.0 SEED changes nothing on")
            + " adv_all_negative, so L_MAX_SEED_ZERO has no witness. Every"
            + " logit there has its sign bit set, so this cannot be true"
            + " unless identical_fmax's total order disagrees with contract"
            + " 5.1."
        )
    if sz_inert != 0:
        raise Error(
            String("loss_check: a +0.0 SEED changes ")
            + String(sz_inert)
            + " rows of base_n4_v8, so it is not L_MAX_SEED_ZERO's inert"
            + " half. `assert_row_has_a_positive_logit` is supposed to make"
            + " that impossible."
        )
    print(
        "  L_MAX_SEED_ZERO [SIMULATED on the host, legitimate because the"
        " fold is exactly associative]: a +0.0 seed moves "
        + String(sz_moved)
        + " of "
        + String(sz_w.n_rows)
        + " rows of adv_all_negative and 0 rows of base_n4_v8"
    )

    # `L_MAX_TOPK_PREFIX`, SIMULATED the same way.
    var tk_w = ce_case(ce_case_by_name(String("adv_argmax_tail")))
    var tk_i = ce_case(ce_case_by_name(String("adv_argmax_head")))
    var tk_moved = 0
    var tk_inert = 0
    for side in range(2):
        # `CeCase` is not ImplicitlyCopyable; name the copy.
        var cc = tk_w.copy()
        if side == 1:
            cc = tk_i.copy()
        var xs = ce_case_logits(cc)
        for r in range(cc.n_rows):
            var a = row_max_oracle(xs, r * cc.vocab, cc.vocab)
            var b = row_max_topk_prefix(xs, r * cc.vocab, cc.vocab)
            if bits_of(a) != bits_of(b):
                if side == 0:
                    tk_moved += 1
                else:
                    tk_inert += 1
    if tk_moved == 0:
        raise Error(
            String("loss_check: truncating the maximum to the first 32")
            + " classes changes nothing on adv_argmax_tail, so"
            + " L_MAX_TOPK_PREFIX has no witness. The case plants its"
            + " maximum at index V-1 = 63, so this cannot be true."
        )
    if tk_inert != 0:
        raise Error(
            String("loss_check: truncating to 32 changes ")
            + String(tk_inert)
            + " rows of adv_argmax_head, whose maximum is planted at index"
            + " 0, so it is not the arm's inert half"
        )
    print(
        "  L_MAX_TOPK_PREFIX [SIMULATED]: a 32-class prefix moves "
        + String(tk_moved)
        + " of "
        + String(tk_w.n_rows)
        + " rows of adv_argmax_tail and 0 rows of adv_argmax_head, which"
        " differ ONLY in where the maximum was planted"
    )

    # The two reciprocal arms' condition: EXACT at a power-of-two divisor.
    var pow2_case = ce_case(ce_case_by_name(String("base_n4_v8")))
    var odd_case = ce_case(ce_case_by_name(String("base_n3_v3")))
    if not is_exact_power_of_two(ce_case_divisor(pow2_case)):
        raise Error(
            String("loss_check: base_n4_v8's divisor ")
            + bits32_hex(ce_case_divisor(pow2_case))
            + " is NOT a power of two, so it is not the inert half of"
            + " L_MEAN_RECIPROCAL_MUL or L_GRAD_RECIPROCAL_MUL"
        )
    if is_exact_power_of_two(ce_case_divisor(odd_case)):
        raise Error(
            String("loss_check: base_n3_v3's divisor IS a power of two, so")
            + " x * (1/d) is EXACT there and the reciprocal arms have no"
            + " witness ([[reached-but-inert]])."
        )
    print(
        "  L_*_RECIPROCAL_MUL: base_n4_v8's divisor "
        + bits32_hex(ce_case_divisor(pow2_case))
        + " IS a power of two (inert) and base_n3_v3's "
        + bits32_hex(ce_case_divisor(odd_case))
        + " is not (witness)"
    )

    # `L_GRAD_DIVISOR_IS_N`'s condition: `count != N`.
    var ign = ce_case(ce_case_by_name(String("ign_n6_v8_c3")))
    if ce_case_unignored_count(ign) == ign.n_rows:
        raise Error(
            String("loss_check: ign_n6_v8_c3 has count == N == ")
            + String(ign.n_rows)
            + ", so substituting N for count changes nothing and"
            + " L_GRAD_DIVISOR_IS_N has no witness"
        )
    if ce_case_unignored_count(pow2_case) != pow2_case.n_rows:
        raise Error(
            String("loss_check: base_n4_v8 has count != N, so it is not")
            + " L_GRAD_DIVISOR_IS_N's inert half"
        )
    print(
        "  L_GRAD_DIVISOR_IS_N: ign_n6_v8_c3 has count "
        + String(ce_case_unignored_count(ign))
        + " against N "
        + String(ign.n_rows)
        + " (witness); base_n4_v8 has count == N == "
        + String(pow2_case.n_rows)
        + " (inert)"
    )

    # `G_PAD_PLUS_ZERO`'s condition: does the tree ever CARRY an odd level?
    if not case_fold_has_carry(300):
        raise Error(
            "loss_check: the fold over V = 300 has NO odd level width, so"
            " G_PAD_PLUS_ZERO has nothing to pad and no witness"
        )
    if case_fold_has_carry(256) or case_fold_has_carry(1024):
        raise Error(
            "loss_check: the fold over V = 256 or V = 1024 CARRIES an odd"
            " level, so those are not G_PAD_PLUS_ZERO's inert halves"
        )
    print(
        "  G_PAD_PLUS_ZERO: V=300 (P="
        + String(case_leaf_count(300))
        + ") CARRIES an odd level; V=256 (P="
        + String(case_leaf_count(256))
        + ") and V=1024 (P="
        + String(case_leaf_count(1024))
        + ") have every level width even"
    )

    # ---- 7. The exact families' vacuity guards -------------------------
    # Contract 12.4's guards 1 and 2, ENFORCED by the fixture and run here
    # so a bad exact case fails before a device exists.
    for k in range(CE_EXACT_CASE_COUNT):
        var e = ce_exact_case(k)
        var d = ce_exact_guard(e)
        print(
            "  exact case "
            + String(e.name)
            + ": V="
            + String(e.vocab)
            + " N="
            + String(e.n_rows)
            + " divisor="
            + bits32_hex(d)
            + " -- V != N, V != divisor, both powers of two"
            + " (contract 12.4 guards 1 and 2)"
        )


# ===========================================================================
# THE CASE SET
# ===========================================================================


def clause_a_cases() raises -> List[Int]:
    """The default clause-(a) set: every case except the shipped
    vocabulary.

    WHAT THE SET BUYS, because a set chosen for what it excludes is a set
    nobody checked for what it INCLUDES. Each line names the clause that
    would go untested without it.

      0  base_n4_v8       divisor 4, a power of two -- the INERT half of
                          both reciprocal arms and of L_GRAD_DIVISOR_IS_N
      1  base_n1_v1       `nll` EXACTLY `-0.0` -- the ONLY witness for
                          L_NEG_VIA_ZERO_SUB and L_SMOOTH_ALWAYS_SPELLED,
                          and the INERT half of L_GRAD_SIGN and
                          L_GRAD_TARGET_OFF_BY_ONE and L_LOG_STDLIB
      2  base_n3_v3       divisor 3 -- the witness for both reciprocal arms
      3  wide_v300        P = 3, ragged 44-element leaf, and a CARRY --
                          the witness for every fold arm
      4  wide_v256        P = 2, every level width even -- the INERT half
                          of G_PAD_PLUS_ZERO
      5  wide_v1024       P = 8, four even levels -- a deeper second
                          inert half
      6  wide_v129        P = 2 with a ONE-ELEMENT ragged leaf
      7  long_n300_v8     N = 300 -- L_REDUCE_SERIAL's witness
      8  long_n512_sum    SUM with divisor exactly +1.0, the bitwise-inert
                          divide contract 5.5 spells anyway
      9  sum_num_items7   the OTHER branch of ce_divisor, so the "one
                          producer" clause is checked on both arms
      10 ign_n6_v8_c3     count 3 of N 6 -- L_GRAD_DIVISOR_IS_N's witness
                          and an ignored row in the MIDDLE of the batch
                          fold
      11 ign_n5_v8_c2     count 2 of N 5 -- count != N with a POWER-OF-TWO
                          divisor, which separates L_GRAD_DIVISOR_IS_N
                          from the reciprocal arms on one case
      12 smooth_n4_v8     EPS = 0.1 -- the smoothing arms' witness
      13 smooth_v1_eps    EPS != 0 on a `-0.0` row -- the INERT half of
                          L_SMOOTH_ALWAYS_SPELLED
      14 smooth_v300      smoothing over a real fold, so `ce.logp_sum` is
                          a tree and not a leaf
      15 none_reduction   REDUCTION_NONE -- the absent stages must be
                          ABSENT and not zero-filled
      16 adv_signed_zeros L_MAX_PLAIN_COMPARE's witness
      17 adv_all_negative L_MAX_SEED_ZERO's witness
      18 adv_argmax_tail  L_MAX_TOPK_PREFIX's witness
      19 adv_argmax_head  the same shape, argmax at 0 -- its INERT half
      20 adv_big_offset   L_NLL_VIA_ADDBACK's witness
      21 adv_centered     its INERT half
      22 adv_underflow_y  L_NLL_VIA_LOG_W's witness, and contract 8.1's
                          `ce.exp` row reached
      23 adv_subnormal    IDENTITY_PATHS row 10's flush unit at the LOAD
                          seam

    `shipped_v128256` joins on `MOJOLEARN_LOSS_CHECK_SHIPPED_V`. **A GATE
    THAT HAS NEVER RUN IT HAS NEVER RUN THE SHAPE THE PROFILE IS FOR**, and
    no amount of `V = 300` closes that; the SCOPE line says so."""
    var out = List[Int]()
    for k in range(CE_CASE_COUNT):
        if k == CE_CASE_SHIPPED_V:
            continue
        out.append(k)
    if env_on("MOJOLEARN_LOSS_CHECK_SHIPPED_V"):
        out.append(CE_CASE_SHIPPED_V)
    return out^


@fieldwise_init
struct CaseVerdict(Copyable, Movable):
    """One case's clause-(a) result, kept so that the sabotage expectation
    table can be evaluated ACROSS cases in one binary.

    `moved` is a per-stage bitmask as a list of Bools rather than a count,
    because two arms in this lane are stated as "must move stage X and must
    NOT move stage Y ON THE SAME CASE" -- `L_IGNORED_ROW_NEG_ZERO`
    (`ce.row` yes, `ce.total` no) and `L_GRAD_DIVISOR_IS_N` (`ce.dlogits`
    yes, `ce.divisor` no). A count cannot express either."""

    var name: String
    var n_moved: Int
    var first_index: Int
    var first: String
    var cells: Int
    var moved: List[Bool]
    var poison: Int


def stage_moved(v: CaseVerdict, i: Int) -> Bool:
    if i < 0 or i >= len(v.moved):
        return False
    return v.moved[i]


# ===========================================================================
# CLAUSE (a): device equals host oracle, bitwise, every stage, every shape
# ===========================================================================


def clause_a_case(
    ctx: DeviceContext, k: Int, mut trace: IdentityTrace, prefix: String
) raises -> CaseVerdict:
    """Contract section 10 clause (a) at ONE fixture case.

    WHAT WOULD MAKE THIS PASS WHILE GATING NOTHING. Three things, and two
    of them are closed here:

      1. A device dump and an oracle dump that are the same object. They
         are not: they come from two functions in two files, one reading
         `CeStages` fields and one reading device buffers.
      2. Two dumps of different LENGTHS compared to the shorter of the two.
         `compare_dumps` checks the stage COUNT and `compare_stage` raises
         on a per-stage length mismatch.
      3. **Our oracle being wrong in the same way as our device.** THIS IS
         NOT CLOSED AND CANNOT BE CLOSED HERE. Both halves are ours -- and
         in this lane they are closer than usual, because
         `identical_ce_forward_into` calls `ce_divisor`,
         `ce_one_minus_eps` and `ce_smoothing_targets` OUT OF THE ORACLE
         FILE, so three of the twenty stages are literally the same host
         code on both sides (`stage_is_host_constant`). Only an
         independent reference can see that class of error and
         `training/corpus/` does not exist (contract OWED item 6). Contract
         12's EXACT-ANALYTIC arm is the one place a hand-written expected
         value exists, and it covers the gradient on two exact families and
         nothing else.

    AND THE FOURTH, WHICH IS THIS LANE'S OWN. The card is written from what
    THIS GATE DOWNLOADED, because `loss.mojo` records nothing (DEVIATION
    1451). A device that computed the right bits and a download that read
    the wrong region are indistinguishable here."""
    var c = ce_case(k)
    var logits = ce_case_logits(c)
    var targets = ce_case_targets(c)
    var count = host_count(c, targets)

    var host = run_host_case(c, logits, targets)
    var run = run_device_rows(ctx, c, logits, targets, count)
    var diffs = compare_dumps(host, run.dump, False)

    # The two INTEGER stages. `input.targets` is the same list on both
    # sides by construction (it is the fixture's), so what this really
    # gates is `ce.count`, which the forward and the backward must agree
    # about -- contract 5.5's one-producer clause.
    var cl_host = List[Int32]()
    cl_host.append(Int32(count))
    var cl_dev = List[Int32]()
    cl_dev.append(Int32(run.count))
    var cdiff = compare_int_stage(ce_stage_tag(ST_COUNT), cl_host, cl_dev)
    if cdiff.n_diff != 0:
        raise Error(
            String("loss_check: CLAUSE (a) FAILED at ce.count on case ")
            + String(c.name)
            + ": the oracle counted "
            + String(count)
            + " unignored rows and the device call was given "
            + String(run.count)
            + ". Contract 5.5 says the divisor has exactly ONE PRODUCER"
            + " and this is the argument to it."
        )

    var moved = List[Bool]()
    for i in range(CE_STAGE_COUNT):
        moved.append(diffs[i].n_diff > 0)
    var n_moved = count_moved(diffs)
    var fi = first_moved_index(diffs)
    var fname = String("")
    if fi >= 0:
        fname = ce_stage_tag(fi)
    var cells = total_cells(host)

    # ---- THE POISON SURVIVORS -------------------------------------------
    # DEVIATION 1456. On a CLEAN build every stage cell must have been
    # written by some kernel. A survivor means the card is hashing whatever
    # the allocator left there, which is a green with no evidence behind
    # it. Under an ARMED build it is a finding rather than a failure and
    # `clause_g` evaluates it, so this only raises when nothing is armed.
    # A PLAIN `if` and not a `comptime if`. `ANY_LOSS_SABOTAGE` is a
    # comptime Bool and is perfectly usable at run time, and the
    # `comptime if not X` spelling is one this tree has never compiled.
    # The branch costs one predictable test per case and buys a
    # spelling every other file here already uses.
    if not ANY_LOSS_SABOTAGE:
        if run.poison != 0:
            raise Error(
                String("loss_check: case ")
                + String(c.name)
                + " came back with "
                + String(run.poison)
                + " UNWRITTEN cells (still holding the poison pattern"
                + " 0x7fc0dead) on a CLEAN build. A stage the card records"
                + " that no kernel wrote is a stage whose hash is whatever"
                + " the allocator left there."
            )

    var line = "  case " + String(k) + " " + describe_case(c) + ": "
    if n_moved == 0:
        print(
            line
            + String(expected_card_records(c))
            + "/"
            + String(expected_card_records(c))
            + " stages bit-identical ("
            + String(cells)
            + " cells)"
        )
    else:
        print(
            line
            + String(n_moved)
            + " stages MOVED, first at "
            + first_moved(diffs)
        )
        # The per-stage detail, printed only when something moved, because
        # twenty OK lines per case across twenty-four cases is 480 lines of
        # nothing and the one line that matters would be lost in it.
        _ = compare_dumps(host, run.dump, True)

    _ = write_card(trace, prefix, c, run.dump, targets, count)
    return CaseVerdict(
        String(c.name), n_moved, fi, fname, cells, moved^, run.poison
    )


def find_verdict(
    verdicts: List[CaseVerdict], name: String
) raises -> CaseVerdict:
    for i in range(len(verdicts)):
        if verdicts[i].name == name:
            return verdicts[i].copy()
    raise Error(
        String("loss_check: the sabotage expectation names case '")
        + name
        + "' and it was not in the clause-(a) set this build ran. The arm"
        + " cannot be evaluated, which is NOT the same as the arm passing"
        + " ([[reached-but-inert]])."
    )


# ===========================================================================
# CLAUSE (b): the same bits on every one of eight repeated launches
# ===========================================================================


def clause_b(ctx: DeviceContext, k: Int) raises:
    """Contract section 10 clause (b). Eight launches, each with its OWN
    fresh buffers, ones vector, workspace and kernel dispatches, every
    stage compared to the first on every cell.

    Fresh EVERYTHING and not just a fresh call, because the failure this
    clause is for is an execution plan that is not a pure function of the
    input -- a scratch buffer read before it is written, an accumulator
    that survives a call, a launch geometry chosen from a clock. Re-calling
    one set of buffers would hide all three. The workspace matters most:
    `identical_ce_workspace_max_floats`'s docstring says three GEMM calls
    SHARE one workspace within a call and are safe only because MAX runs
    one context in order, and a stale workspace read across calls is
    exactly what a fresh one per launch would expose.

    WHAT WOULD MAKE THIS PASS WHILE GATING NOTHING: comparing only
    `ce.loss`. Contract 5.3's own theorem says an ignored row is BITWISE
    INERT in the batch fold, and contract 7.3 turns that into a whole
    sabotage (`L_IGNORED_ROW_NEG_ZERO`) whose effect is invisible in
    `ce.total` and visible in `ce.row`. Every stage is compared.

    AND THE HOLE NO CONTROL CLOSES: eight identical runs of a
    DETERMINISTICALLY WRONG loss pass it. Clause (b) is an invariance claim
    and says nothing about correctness; clause (a) is what covers that and
    clause (e) is what covers the gradient. Stated because "8/8 launches
    identical" reads like a strong result and is not one on its own."""
    var c = ce_case(k)
    var logits = ce_case_logits(c)
    var targets = ce_case_targets(c)
    var count = host_count(c, targets)
    print(
        "clause (b): "
        + String(CLAUSE_B_LAUNCHES)
        + " repeated launches of "
        + String(c.name)
        + ", every stage, every cell, fresh state each time"
    )
    var base = run_device_rows(ctx, c, logits, targets, count)
    var cells = total_cells(base.dump)
    for run in range(2, CLAUSE_B_LAUNCHES + 1):
        var got = run_device_rows(ctx, c, logits, targets, count)
        var diffs = compare_dumps(base.dump, got.dump, False)
        if count_moved(diffs) != 0:
            raise Error(
                String("loss_check: CLAUSE (b) FAILED, launch ")
                + String(run)
                + " differs from launch 1 at "
                + first_moved(diffs)
            )
        if got.poison != base.poison:
            raise Error(
                String("loss_check: CLAUSE (b) FAILED, launch ")
                + String(run)
                + " left "
                + String(got.poison)
                + " cells unwritten and launch 1 left "
                + String(base.poison)
                + ". WHICH cells a run writes is part of the answer."
            )
    print(
        "clause (b): PASS, launches 2.."
        + String(CLAUSE_B_LAUNCHES)
        + " bit-identical to launch 1 on all "
        + String(cells)
        + " cells of every stage"
    )


# ===========================================================================
# CLAUSE (c): BATCH COMPOSITION INVARIANCE, which here is ROW CHUNKING
# ===========================================================================


def row_slice(
    values: List[Float32], i: Int, row: Int, vocab: Int
) raises -> List[Float32]:
    """The cells of per-row stage `i` belonging to row `row`.

    THE POINT OF CLAUSE (c): a row's bits must not depend on who else
    shares the launch, and the only way to SAY that is to cut the row out
    of buffers of different lengths. A whole-buffer compare cannot even be
    spelled at `N = 1` against `N = 6`."""
    var w = stage_row_width(i, vocab)
    var out = List[Float32]()
    for j in range(w):
        out.append(values[row * w + j])
    return out^


def clause_c(ctx: DeviceContext, k: Int) raises:
    """Contract section 10 clause (c) and contract 7.1.

    A row's bits are identical whether it is computed alone, in a chunk of
    2, in a chunk of 3, or in a launch of all `N`. Nothing in L1 through
    L11 or L14 through L16 reads `N` -- not the row chunk, not the launch
    geometry, not the block size, not the vendor -- so this is true BY
    CONSTRUCTION and the gate exists to catch the construction being
    violated by an execution plan. `archive/plans/IDENTICAL_GEMM_PLAN.md:86-93` is the
    in-repo statement of why this is the same problem one layer down.

    **THE EXCEPTION IS L12 AND ONLY L12, AND EXCLUDING IT IS THE CLAUSE
    RATHER THAN AN OMISSION.** The batch fold's `k` IS `N`, so `ce.total`
    and `ce.loss` are functions of the whole launch by construction
    (contract 5.4: "the microbatch schedule is part of a training run's
    numerical specification, not an execution detail"). `stage_is_per_row`
    returns False for them and a gate that quietly row-sliced `ce.total`
    would be asserting something contract 5.4 says is FALSE.

    **THE `count` IS THE WHOLE BATCH'S ON EVERY CHUNK**, and that is not a
    convenience either. `count` feeds `ce_divisor`, which feeds L13 and
    L16, so a chunk that counted only its own unignored rows would produce
    a different divisor and `ce.dlogits` would differ for a reason that has
    nothing to do with row independence. Contract 7.1's escape hatch is
    stated in exactly these terms -- a caller may split the rows and
    concatenate -- and a caller doing that must carry the whole batch's
    `count`. This clause is also, therefore, the only place that property
    is written down as code.

    **AND ITS NEGATIVE CONTROL, WITHOUT WHICH IT IS WORTHLESS.** If
    `row_slice` were wrong -- if it returned row 0 whatever row it was
    asked for -- every comparison below would compare a row to ITSELF and
    pass for ever, on every vendor, hiding any batch dependence there is.
    So the clause first proves the slicer can tell two rows apart: rows 0
    and 1 of the full run have different targets and different logits and
    must differ somewhere. A zero there RAISES and calls the clause VACUOUS
    rather than passing it (`[[verify-reach-not-output]]`)."""
    var c = ce_case(k)
    if c.n_rows < 4:
        raise Error(
            String("loss_check: clause (c) needs a case with N >= 4 and ")
            + String(c.name)
            + " has N="
            + String(c.n_rows)
        )
    var logits = ce_case_logits(c)
    var targets = ce_case_targets(c)
    var count = host_count(c, targets)
    var v = c.vocab
    var n = c.n_rows

    print(
        "clause (c): "
        + String(c.name)
        + ", the same row alone and in chunks, count="
        + String(count)
        + " carried to every chunk"
    )

    var full = run_device_rows(ctx, c, logits, targets, count)

    # ---- THE NEGATIVE CONTROL -------------------------------------------
    var control = 0
    var sliceable = 0
    for i in range(CE_STAGE_COUNT):
        if not stage_present(i, c) or stage_is_int(i):
            continue
        if not stage_is_per_row(i):
            continue
        sliceable += 1
        var a0 = row_slice(full.dump[i], i, 0, v)
        var a1 = row_slice(full.dump[i], i, 1, v)
        var d = compare_stage(
            ce_stage_tag(i) + " CONTROL row0 vs row1", a0, a1, False
        )
        if d.n_diff > 0:
            control += 1
    if control == 0:
        raise Error(
            "loss_check: CLAUSE (c) IS VACUOUS. Rows 0 and 1 of the full"
            " run are bit-identical on every per-row stage, which cannot be"
            " true of two rows with different logits and different targets."
            " `row_slice` is not cutting distinct rows, so every comparison"
            " below is a row against itself ([[reached-but-inert]])."
        )
    print(
        "clause (c) control: rows 0 and 1 of the full run differ on "
        + String(control)
        + " of "
        + String(sliceable)
        + " per-row stages, so the row slicer distinguishes rows"
    )

    # ---- THE CLAUSE ------------------------------------------------------
    var chunks: List[Int] = [1, 2, 3]
    var cells = 0
    var bad = 0
    var first_bad = String("")
    for ci in range(len(chunks)):
        var w = chunks[ci]
        var start = 0
        while start < n:
            var take = w
            if start + take > n:
                take = n - start
            var sub_logits = List[Float32]()
            for r in range(take):
                for j in range(v):
                    sub_logits.append(logits[(start + r) * v + j])
            var sub_targets = List[Int32]()
            for r in range(take):
                sub_targets.append(targets[start + r])
            var got = run_device_rows(ctx, c, sub_logits, sub_targets, count)
            for i in range(CE_STAGE_COUNT):
                if not stage_present(i, c) or stage_is_int(i):
                    continue
                if not stage_is_per_row(i):
                    continue
                for r in range(take):
                    var a = row_slice(full.dump[i], i, start + r, v)
                    var b = row_slice(got.dump[i], i, r, v)
                    cells += len(a)
                    var d = compare_stage(
                        ce_stage_tag(i)
                        + " row "
                        + String(start + r)
                        + " full vs chunk "
                        + String(w),
                        a,
                        b,
                        False,
                    )
                    if d.n_diff > 0:
                        bad += 1
                        if first_bad == "":
                            first_bad = (
                                ce_stage_tag(i)
                                + " at row "
                                + String(start + r)
                                + " in chunks of "
                                + String(w)
                                + ", "
                                + String(d.n_diff)
                                + " of "
                                + String(d.n_cells)
                                + " cells"
                            )
            start += take
    if bad != 0:
        raise Error(
            String("loss_check: CLAUSE (c) FAILED on ")
            + String(bad)
            + " stage-rows, first at "
            + first_bad
            + ": a row's bits depend on who shares its launch. Contract 7.1"
            + " makes this true BY CONSTRUCTION -- nothing per row reads"
            + " N -- so a failure here is a finding about the EXECUTION"
            + " PLAN and not about the gate."
        )
    print(
        "clause (c): PASS, every row identical alone and in chunks of 1, 2"
        " and 3, on all "
        + String(cells)
        + " compared cells. `ce.total` and `ce.loss` are EXCLUDED because"
        " L12's fold length IS the batch (contract 5.4), which is the"
        " clause and not an omission."
    )


# ===========================================================================
# CLAUSE (d): VOCABULARY FOLD INVARIANCE ACROSS NAMED GEMM PLANS
# ===========================================================================


def clause_d(ctx: DeviceContext, k: Int) raises:
    """Contract section 10 clause (d): `ce.denom` identical under at least
    three unrelated GEMM execution plans, reaching
    `identical_gemm_with_plan`'s named plans through this lane's routing.

    **WHAT THIS REACHES AND WHAT IT DOES NOT, because the difference
    matters and a reader is entitled to it (DEVIATION 1455).**
    `identical_ce_forward_into` calls `identical_gemm_into`, which asks
    `choose_gemm_plan` for a plan; there is no argument by which a caller
    can name one. So this clause CANNOT drive the loss entry point on four
    plans. What it does instead is take the DEVICE'S OWN `ce.exp` buffer --
    the exact operand the loss handed to the GEMM -- and re-run the SAME
    call, `(N, 1, V)` at `OP_NN` against the SAME ones vector, under four
    named plans, requiring all four to equal the `ce.denom` the loss itself
    produced.

    That is a real reach proof of the fold and it is NOT a reach proof of
    the dispatcher. `choose_gemm_plan`'s own choice is exercised by exactly
    one plan (whichever it picked), and the gemm lane's
    `check_device_is_launch_invariant` is where the dispatcher is gated.
    Saying so is the point; a clause that quietly claimed to cover both
    would be the more useful-sounding and less true sentence.

    **THE CONTROL.** Four plans that all returned the same wrong number
    would pass, and so would a "sweep" that ran one plan under four names.
    So the clause also folds the SAME buffer at `k = V - 1` and REQUIRES
    the answer to move. If it does not, the comparison cannot see a
    difference at all and the clause raises VACUOUS rather than passing.
    (`V - 1` drops the last term of every row; contract 8.2(a) says every
    term is in `[+0.0, 1.0]`, so this moves unless the last exponential is
    exactly `+0.0`, and the case chosen has none -- which the control
    checks rather than assumes.)"""
    var c = ce_case(k)
    if case_leaf_count(c.vocab) < 2:
        raise Error(
            String("loss_check: clause (d) needs a case whose vocabulary")
            + " fold has P >= 2 and "
            + String(c.name)
            + " has V="
            + String(c.vocab)
            + " (P=1, and at P == 1 the tree performs NO addition -- gemm"
            + " contract 7.3 -- so every plan trivially agrees and the"
            + " clause would gate nothing)"
        )
    var logits = ce_case_logits(c)
    var targets = ce_case_targets(c)
    var count = host_count(c, targets)
    var v = c.vocab
    var n = c.n_rows

    print(
        "clause (d): "
        + String(c.name)
        + ", the vocabulary fold at (N="
        + String(n)
        + ", 1, V="
        + String(v)
        + "), P="
        + String(case_leaf_count(v))
        + ", under named GEMM plans"
    )

    var run = run_device_rows(ctx, c, logits, targets, count)
    var expo = run.dump[ST_EXP].copy()
    var want = run.dump[ST_DENOM].copy()

    var plans: List[Int] = [
        PLAN_FLAT, PLAN_TILE_16_16_32, PLAN_SPLITK, PLAN_SPLITK_STAGED
    ]
    var nws = identical_gemm_workspace_max_floats(n, 1, v)
    var bad = 0
    var first_bad = String("")
    for pi in range(len(plans)):
        var plan = plans[pi]
        var d_expo = _upload_f32(ctx, expo)
        var d_ones = _upload_f32(ctx, ce_ones(v))
        var d_out = _upload_f32(ctx, ce_poison(n))
        var d_ws = _upload_f32(ctx, ce_poison(nws))
        identical_gemm_with_plan(
            ctx, d_out, d_expo, d_ones, d_ws, n, 1, v, OP_NN, plan
        )
        ctx.synchronize()
        var got = _download_f32(ctx, d_out, n)
        var diff = compare_stage(
            String("ce.denom under ") + gemm_plan_name(plan),
            want,
            got,
            False,
        )
        if diff.n_diff > 0:
            bad += 1
            if first_bad == "":
                first_bad = (
                    gemm_plan_name(plan)
                    + " on "
                    + String(diff.n_diff)
                    + " of "
                    + String(diff.n_cells)
                    + " cells, first cell "
                    + String(diff.first)
                    + " want "
                    + bits32_hex(want[diff.first])
                    + " got "
                    + bits32_hex(got[diff.first])
                )
        _ = d_expo^
        _ = d_ones^
        _ = d_out^
        _ = d_ws^

    # ---- THE CONTROL: a DIFFERENT fold length MUST move -----------------
    #
    # DEVIATION 1492, AND THE FIRST RUN OF THIS CLAUSE FOUND IT. The control
    # used to fold `expo` itself over `v - 1` terms and require a move. It
    # DID NOT MOVE, and this clause correctly raised VACUOUS. The cause is
    # ABSORPTION, the same mechanism the mamba lane measured: at
    # `wide_v300` the dropped term is `exp(x[v-1] - max)`, which at the tail
    # of a shifted-exponential row is small enough that adding it to a
    # running sum of up to 300 terms rounds away entirely. Dropping it is a
    # no-op IN THE BITS, so four plans agreeing said nothing.
    #
    # The replacement drops a term that CANNOT be absorbed. `expo` is a
    # shifted exponential, so its maximum is exactly `exp(0) = 1.0` and the
    # whole sum is at most `V`. A control buffer is built with `1.0` planted
    # in the LAST slot, and both arms fold THAT buffer -- `v` terms against
    # `v - 1`. The difference is one whole unit out of at most 300, a
    # relative change near 3e-3, which is four orders above FP32 epsilon and
    # cannot round away at any partition.
    #
    # Both arms must run on the SAME buffer or the control is comparing two
    # different sums and proves nothing about the fold.
    var expo_ctl = expo.copy()
    for r in range(n):
        expo_ctl[r * v + (v - 1)] = Float32(1.0)
    var d_expo_full = _upload_f32(ctx, expo_ctl)
    var d_ones_full = _upload_f32(ctx, ce_ones(v))
    var d_out_full = _upload_f32(ctx, ce_poison(n))
    var d_ws_full = _upload_f32(ctx, ce_poison(nws))
    identical_gemm_with_plan(
        ctx, d_out_full, d_expo_full, d_ones_full, d_ws_full,
        n, 1, v, OP_NN, PLAN_FLAT,
    )
    ctx.synchronize()
    var ctl_full = _download_f32(ctx, d_out_full, n)
    _ = d_expo_full^
    _ = d_ones_full^
    _ = d_out_full^
    _ = d_ws_full^

    var d_expo2 = _upload_f32(ctx, expo_ctl)
    var d_ones2 = _upload_f32(ctx, ce_ones(v))
    var d_out2 = _upload_f32(ctx, ce_poison(n))
    var d_ws2 = _upload_f32(ctx, ce_poison(nws))
    identical_gemm_with_plan(
        ctx, d_out2, d_expo2, d_ones2, d_ws2, n, 1, v - 1, OP_NN, PLAN_FLAT
    )
    ctx.synchronize()
    var shorter = _download_f32(ctx, d_out2, n)
    var cdiff = compare_stage(String("control"), ctl_full, shorter, False)
    _ = d_expo2^
    _ = d_ones2^
    _ = d_out2^
    _ = d_ws2^
    if cdiff.n_diff == 0:
        raise Error(
            "loss_check: CLAUSE (d) IS VACUOUS. Folding the SAME buffer"
            " over V-1 terms instead of V returned bit-identical answers,"
            " so this comparison cannot see a difference in the fold at all"
            " and the four plans agreeing means nothing"
            " ([[reached-but-inert]])."
        )
    print(
        "clause (d) control: the same buffer folded over V-1 terms moves on "
        + String(cdiff.n_diff)
        + " of "
        + String(cdiff.n_cells)
        + " cells, so the comparison can see a fold difference"
    )

    if bad != 0:
        raise Error(
            String("loss_check: CLAUSE (d) FAILED on ")
            + String(bad)
            + " of "
            + String(len(plans))
            + " plans, first at "
            + first_bad
            + ". Contract 7.2 says `(L, P)` come from the contraction"
            + " length and the two profile constants and from NOTHING"
            + " else -- no block size, no grid shape, no occupancy, no lane"
            + " width -- so a plan-dependent denominator is a finding about"
            + " gemm v1 reached through this lane's routing."
        )
    print(
        "clause (d): PASS, "
        + String(len(plans))
        + " named plans (FLAT, TILE_16_16_32, SPLITK, SPLITK_STAGED) all"
        " reproduce the loss's own ce.denom bit for bit. NOTE: this gates"
        " the FOLD and not the DISPATCHER -- `identical_ce_forward_into`"
        " calls `identical_gemm_into`, which picks a plan itself, and no"
        " argument lets a caller name one (DEVIATION 1455)."
    )


# ===========================================================================
# CLAUSE (e): THE GRADIENT, BITWISE, AGAINST A HAND-WRITTEN CLOSED FORM
#
# Contract section 12, and DEVIATION 1163's demotion of finite differences
# is why there is no epsilon anywhere below. On a fixture chosen so the
# derivative is EXACTLY REPRESENTABLE the derivative can be WRITTEN DOWN
# and compared to, and comparing two exact numbers is simpler and stronger
# than comparing two exact differences of exact numbers. A central
# difference has a step size, and a step size is a tolerance wearing a
# different hat.
#
# **THIS CLAUSE IS ALWAYS ON.** Every other clause here is about SPELLING
# -- whether our device says the same thing as our oracle -- and only this
# one asks whether either of them is RIGHT. A run without it has checked
# that two halves of one lane agree and nothing else.
#
# **AND IT IS ASSERTED IN BOTH MODES.** Contract section 10: on an exactly
# representable fixture a contracted multiply-add and an uncontracted one
# produce the same bits, so a FAST failure here is a real routing defect
# and not a mode difference. `IDENTICAL_BACKWARD_PLAN.md`'s G2 makes the
# same argument.
# ===========================================================================


def exact_as_case(e: CeExactCase) raises -> CeCase:
    """A `CeCase` view of an exact case, for the four fields
    `run_device_rows` actually reads (`vocab`, `reduction`, `eps_on`,
    `num_items`) plus the name.

    **IT IS NOT IN THE CASE TABLE AND MUST NOT BE PASSED TO ANYTHING THAT
    LOOKS ITS INDEX UP.** `ce_case_logits`, `ce_case_targets`,
    `ce_case_divisor` and `describe_case` all resolve a case's SEED through
    `ce_case_by_name`, which walks the real table and RAISES on a name that
    is not in it. That raise is the desired behavior -- it is what stops a
    synthetic case from silently drawing the seed of whatever case happens
    to share its name -- and it is why clause (e) hands logits and targets
    to `run_device_rows` explicitly."""
    return CeCase(
        e.name, e.n_rows, e.vocab, e.reduction, False, e.ignore_stride,
        e.num_items, 0, e.high_count,
    )


def clause_e(ctx: DeviceContext) raises:
    """Contract 12.3's arms A1 (exact-analytic, BOTH modes) and A3 (float64
    tolerance, REPORTED), plus contract 12.4's guard 4.

    **WHAT THIS FAMILY GATES.** The formula, the sign, the target COLUMN,
    the divisor's identity, the target-vector constants, and the routing of
    every buffer. A transposed index, an off-by-one target, a flipped sign
    or a divisor taken from the wrong producer all fail it.

    **WHAT IT CANNOT GATE, and saying so is the point of the subsection
    contract 12.1 devotes to it.** Every quantity in it is EXACT, so it
    separates NO spelling from any other. `identical_div(1.0, 4.0)` and
    `1.0 * (1/4)` agree; a serial fold and a balanced tree of ones agree;
    `identical_exp` and `std.math.exp` agree at `+0.0`. **Run alone this
    family would pass every arm in contract 10.1 and gate nothing about the
    arithmetic.** It is the CORRECTNESS half and it needs clause (a)'s
    per-stage comparison beside it. `check_the_exact_fixture_is_vacuous`
    below DEMONSTRATES that rather than asserting it, which is contract
    12.4's guard 4 and DEVIATION 1053's pattern at a new site."""
    print(
        "clause (e): the EXACT-ANALYTIC gradient, "
        + String(CE_EXACT_CASE_COUNT)
        + " cases, BY BITS, no epsilon anywhere, asserted in "
        + mode_name()
    )
    var total_cells_checked = 0
    var had_ignored = False
    var had_clean = False
    for k in range(CE_EXACT_CASE_COUNT):
        var e = ce_exact_case(k)
        var divisor = ce_exact_guard(e)
        var sc = exact_as_case(e)
        var logits = ce_exact_logits(e)
        var targets = ce_exact_targets(e)
        var count = ce_count(targets, IGNORE_INDEX_DEFAULT)
        if e.ignore_stride > 0:
            had_ignored = True
        else:
            had_clean = True

        var host = run_host_case(sc, logits, targets)
        var run = run_device_rows(ctx, sc, logits, targets, count)

        # A2 at the exact shape, so that a divergence between the two
        # halves is reported HERE rather than being blamed on the closed
        # form below.
        var diffs = compare_dumps(host, run.dump, False)
        if count_moved(diffs) != 0:
            raise Error(
                String("loss_check: CLAUSE (e) -- device and oracle differ")
                + " on exact case "
                + String(e.name)
                + " at "
                + first_moved(diffs)
                + ", before the closed form was even consulted"
            )

        # ---- the exact denominator and weights -----------------------
        # `denom` is exactly `Float32(V)` for the uniform family and
        # `Float32(high_count)` for the saturating one, IN EVERY FOLD
        # ORDER, which is why this assertion gates the ROUTING (is the
        # right buffer being folded?) and NOT the fold topology.
        var want_denom = Float32(e.vocab)
        if e.high_count > 0:
            want_denom = Float32(e.high_count)
        for r in range(e.n_rows):
            if bits_of(host[ST_DENOM][r]) != bits_of(want_denom):
                raise Error(
                    String("loss_check: CLAUSE (e) FAILED at ce.denom on ")
                    + String(e.name)
                    + " row "
                    + String(r)
                    + ": "
                    + bits32_hex(host[ST_DENOM][r])
                    + ", the closed form says "
                    + bits32_hex(want_denom)
                )

        # ---- the closed form, cell by cell ---------------------------
        var v = e.vocab
        for r in range(e.n_rows):
            var y = Int(targets[r])
            var ignored = y == IGNORE_INDEX_DEFAULT
            for vv in range(v):
                var is_high = e.high_count == 0 or vv < e.high_count
                var is_target = (not ignored) and vv == y
                var want_w = Float32(0.0)
                if is_high:
                    if e.high_count > 0:
                        want_w = Float32(1.0) / Float32(e.high_count)
                    else:
                        want_w = Float32(1.0) / Float32(v)
                var want_dl = Float32(0.0)
                if not ignored:
                    if e.high_count > 0:
                        want_dl = ce_exact_saturating_gradient(
                            e.high_count, is_high, is_target, divisor
                        )
                    else:
                        want_dl = ce_exact_uniform_gradient(
                            v, y, is_target, divisor
                        )
                var cell = r * v + vv
                for side in range(2):
                    var w_got = host[ST_WEIGHTS][cell]
                    var d_got = host[ST_DLOGITS][cell]
                    var who = String("oracle")
                    if side == 1:
                        w_got = run.dump[ST_WEIGHTS][cell]
                        d_got = run.dump[ST_DLOGITS][cell]
                        who = String("device")
                    if bits_of(w_got) != bits_of(want_w):
                        raise Error(
                            String("loss_check: CLAUSE (e) FAILED at")
                            + " ce.weights on "
                            + String(e.name)
                            + " cell "
                            + String(cell)
                            + " ("
                            + who
                            + "): "
                            + bits32_hex(w_got)
                            + ", the closed form says "
                            + bits32_hex(want_w)
                            + ". A `+0.0` weight whose SIGN came out"
                            + " negative fails here too, which is the"
                            + " point of comparing by bits."
                        )
                    if bits_of(d_got) != bits_of(want_dl):
                        raise Error(
                            String("loss_check: CLAUSE (e) FAILED at")
                            + " ce.dlogits on "
                            + String(e.name)
                            + " row "
                            + String(r)
                            + " class "
                            + String(vv)
                            + " (target "
                            + String(y)
                            + ", "
                            + who
                            + "): "
                            + bits32_hex(d_got)
                            + ", the hand-written closed form says "
                            + bits32_hex(want_dl)
                            + ". Contract 12.1: every cell here is a"
                            + " dyadic rational a person can write down,"
                            + " so this is not a tolerance failing."
                        )
                total_cells_checked += 2
        print(
            "  "
            + String(e.name)
            + ": N="
            + String(e.n_rows)
            + " V="
            + String(v)
            + " count="
            + String(count)
            + " divisor="
            + bits32_hex(divisor)
            + " -- oracle AND device equal the closed form on every cell"
        )

    # Contract 12.4 guard 3, ENFORCED rather than hoped for.
    if not had_ignored or not had_clean:
        raise Error(
            String("loss_check: CLAUSE (e) needs at least one exact case")
            + " WITH an ignored row and one WITHOUT (contract 12.4 guard"
            + " 3), or L_GRAD_DIVISOR_IS_N's predicted inert mask is"
            + " untested on the only family whose gradient is asserted"
            + " against a hand-written number"
        )
    print(
        "clause (e) A1: PASS, "
        + String(total_cells_checked)
        + " cell comparisons against a hand-written closed form, on both"
        " halves, with no epsilon anywhere"
    )
    check_the_exact_fixture_is_vacuous()


def check_the_exact_fixture_is_vacuous() raises:
    """Contract 12.4's guard 4. **A DEMONSTRATION, NOT AN ASSERTION.**

    Without this arm a reader is entitled to think clause (e) gates the
    SPELLING, and it does not. So the arm re-spells the gradient's division
    LOCALLY as a reciprocal-multiply -- which is exactly what
    `L_GRAD_RECIPROCAL_MUL` does -- and shows that on the exact fixture's
    power-of-two divisor the two spellings agree BIT FOR BIT, in a CLEAN
    build. Then it shows the same two spellings DISAGREEING at a divisor
    that is not a power of two, which proves the demonstration is capable
    of seeing a difference at all.

    A one-sided version of this arm -- "they agreed" with no control --
    would be indistinguishable from a broken comparison, which is the same
    defect it exists to warn about.

    This is `check_the_square_fixture_is_vacuous`'s pattern (DEVIATION
    1053) at a new site."""
    print(
        "clause (e) guard 4: DEMONSTRATING that the exact fixture separates"
        " no spelling"
    )
    # The `d = w - t` values the exact family actually produces, written
    # out here rather than read from the oracle so this arm does not depend
    # on the thing it is characterizing.
    var ds: List[Float32] = [
        Float32(0.125) - Float32(1.0),  # uniform V=8, target
        Float32(0.125),                 # uniform V=8, other
        Float32(0.25) - Float32(1.0),   # uniform V=4, target
        Float32(0.25),                  # uniform V=4, other
        Float32(0.25) - Float32(1.0),   # saturating c=4, high target
        Float32(0.0) - Float32(1.0),    # saturating c=4, LOW target
        Float32(0.0),                   # saturating c=4, low other
    ]
    var pow2 = Float32(2.0)
    var agree = 0
    for i in range(len(ds)):
        var a = ftz(identical_div(ds[i], pow2))
        var r = ftz(identical_div(Float32(1.0), pow2))
        var b = ftz(identical_mul(ds[i], r))
        if bits_of(a) == bits_of(b):
            agree += 1
    if agree != len(ds):
        raise Error(
            String("loss_check: guard 4 -- the reciprocal-multiply and the")
            + " true divide DISAGREED on "
            + String(len(ds) - agree)
            + " of "
            + String(len(ds))
            + " exact-family values at a POWER-OF-TWO divisor. Contract 4c"
            + " and 12.4 both say `x * (1/d)` is EXACT there, so either"
            + " that is false on this toolchain or these are not the"
            + " fixture's values."
        )
    var three = Float32(3.0)
    var disagree = 0
    for i in range(len(ds)):
        var a = ftz(identical_div(ds[i], three))
        var r = ftz(identical_div(Float32(1.0), three))
        var b = ftz(identical_mul(ds[i], r))
        if bits_of(a) != bits_of(b):
            disagree += 1
    if disagree == 0:
        raise Error(
            String("loss_check: guard 4's CONTROL is dead. The")
            + " reciprocal-multiply and the true divide agreed on ALL "
            + String(len(ds))
            + " values at divisor 3.0 as well, so this arm cannot see a"
            + " spelling difference anywhere and its 'they agree at a"
            + " power of two' half proves nothing"
            + " ([[reached-but-inert]])."
        )
    print(
        "clause (e) guard 4: the reciprocal-multiply agrees with the true"
        " divide on "
        + String(agree)
        + "/"
        + String(len(ds))
        + " exact-family values at a power-of-two divisor, and DISAGREES on "
        + String(disagree)
        + "/"
        + String(len(ds))
        + " at divisor 3.0. **The exact family therefore gates CORRECTNESS"
        " and separates NO spelling**, exactly as contract 12.1 says, and"
        " clause (a) is what covers the spelling."
    )


def clause_e_float64_report(k: Int) raises:
    """Contract 12.3's arm A3. **REPORTED, NEVER ASSERTED.**

    `ce_forward_f64` is the mamba oracle's Float64 arm at a second site --
    "a TOLERANCE instrument, never a bitwise one" -- and it exists so that
    a systematic error which the closed form's algebra and the oracle's
    seams SHARE would be visible. Neither of the other two arms can see a
    mistake they both make.

    **AND IT IS WEAKER THAN THE PHRASE "float64 reference" SUGGESTS.** It
    is OUR Float64 code, in this repository, written by this lane's
    neighbours on the same day, and it shares this lane's algebra
    completely -- the same log-sum-exp shape, the same smoothing formula.
    What it does NOT share is the pins: it calls `std.math`'s `exp` and
    `log` rather than the portable polynomials, and it takes no partition.
    So it can catch a wrong POLYNOMIAL and a wrong FOLD and it cannot catch
    a wrong FORMULA. The thing that could is `training/corpus/`, which does
    not exist (contract OWED item 6)."""
    var c = ce_case(k)
    var logits = ce_case_logits(c)
    var targets = ce_case_targets(c)
    var cfg = ce_case_config(c)
    var host = run_host_case(c, logits, targets)
    var ref64 = ce_forward_f64(logits, targets, cfg)
    var worst = Float64(0.0)
    var worst_row = -1
    for r in range(len(ref64)):
        var got = Float64(host[ST_ROW][r])
        var want = ref64[r]
        var d = got - want
        if d < Float64(0.0):
            d = -d
        var scale = want
        if scale < Float64(0.0):
            scale = -scale
        if scale > Float64(1.0):
            d = d / scale
        if d > worst:
            worst = d
            worst_row = r
    print(
        "clause (e) A3 [REPORTED, NOT ASSERTED]: "
        + String(c.name)
        + " -- the FP32 per-row loss differs from the Float64 reference by"
        " at most "
        + String(worst)
        + " (row "
        + String(worst_row)
        + "). This asserts NOTHING. It exists so a systematic error the"
        " closed form and the oracle SHARE would be visible, and it cannot"
        " see one that this repository's Float64 code shares too."
    )


# ===========================================================================
# CLAUSE (f): THE ROW-39 AUDIT OF CONTRACT SECTION 8
#
# A NaN or an infinity in `logits` must be REFUSED BY NAME before any
# recorded stage, because NaN payloads are VENDOR SHAPED -- IDENTITY_PATHS
# row 39 measured three payloads for one IEEE answer, `0x7fc00000` on
# Apple, `0x7fffffff` on NVIDIA, `0xffc00000` on AMD -- and a certified
# stage may never hold one. Every test below is BY BITS AND NEVER BY A
# COMPARE, because Metal flushes compare operands (row 49) so a
# compare-written test has two meanings across columns.
#
# **AND THE FINDING THIS CLAUSE EXISTS TO MAKE, DEVIATION 1460.**
# `training/checks/loss.mojo` NEVER CALLS `ce_refuse_inputs`. The
# refusal lives entirely in `ce_forward_oracle`, on the host, and
# `identical_ce_forward_into` has no refusal of any kind -- not for a
# non-finite logit, not for an out-of-range target, not for `count == 0`.
# So contract section 8's "REFUSED BY NAME before any recorded stage" is a
# property of the ORACLE and not of the profile, and a caller who reaches
# the device entry point directly gets a vendor-shaped NaN into a certified
# stage. This clause MEASURES that gap rather than describing it: it plants
# a NaN, calls the device WITHOUT the refusal, reads the stages back, and
# requires a non-finite cell to be there. A gate that only tested the
# oracle would have reported clause (f) GREEN with the hole wide open.
# ===========================================================================


def clause_f_names() -> List[String]:
    """The refusals of contract section 8 and section 3, in
    `ce_refuse_inputs`'s and `ce_divisor`'s order, with the substring each
    message must contain.

    THE ORDER MATTERS: a refusal late in the walk only fires if the walk
    got past the earlier ones, so the audit tests the WALK and not only the
    test. The target-range scan is LAST inside `ce_refuse_inputs` and is
    therefore the sharpest of them -- it is behind the shape checks, the
    eps checks and the whole non-finite scan."""
    var n: List[String] = [
        String("vocab"),
        String("N < 1"),
        String("logits hold"),
        String("label_smoothing"),
        String("logits"),
        String("target"),
        String("MEAN over zero unignored rows"),
        String("num_items_in_batch"),
    ]
    return n^


def clause_f(ctx: DeviceContext, k: Int) raises:
    """The audit, its control, and the measured device gap."""
    var c = ce_case(k)
    var logits = ce_case_logits(c)
    var targets = ce_case_targets(c)
    var cfg = ce_case_config(c)
    var v = c.vocab
    var n = c.n_rows
    var cells = n * v
    print(
        "clause (f): the section 8 audit on "
        + String(c.name)
        + ", every test BY BITS, never by a compare"
    )

    # ---- THE CONTROL, and it is the one that matters --------------------
    # If the refusal fired UNCONDITIONALLY -- a stray raise, a scan over
    # the wrong buffer, a mask that matched every finite value -- then
    # every plant below would be "refused by name" and the clause would
    # pass FOR EVER while gating nothing. That is the arima lane's shape of
    # blind gate: the arm fine, the gate incapable of failing.
    var ctrl_raised = False
    try:
        _ = ce_refuse_inputs(logits, targets, cfg)
    except e:
        ctrl_raised = True
    if ctrl_raised:
        raise Error(
            "loss_check: CLAUSE (f) IS VACUOUS. `ce_refuse_inputs` fired on"
            " CLEAN inputs, so every plant below would be 'refused'"
            " whatever it held."
        )
    var ctrl_host = run_host_case(c, logits, targets)
    if len(ctrl_host[ST_ROW]) != n:
        raise Error(
            String("loss_check: CLAUSE (f) IS VACUOUS. A clean oracle call")
            + " produced "
            + String(len(ctrl_host[ST_ROW]))
            + " row losses instead of "
            + String(n)
            + ", so 'the call raised' below does not distinguish a refusal"
            + " from a broken call."
        )
    print(
        "clause (f) control: a clean call does NOT raise and produces all "
        + String(n)
        + " row losses, so the refusal is not unconditional"
    )

    # ---- THE PLANTED REFUSALS -------------------------------------------
    var names = clause_f_names()
    var patterns: List[UInt32] = [BITS_QNAN, BITS_POS_INF]
    var pat_names: List[String] = [String("NaN"), String("infinity")]
    var checked = 0
    for idx in range(len(names)):
        var reps = 1
        if names[idx] == "logits":
            reps = len(patterns)
        for pk in range(reps):
            var bad_logits = logits.copy()
            var bad_targets = targets.copy()
            var bad_cfg = ce_case_config(c)
            var what = names[idx]
            if idx == 0:
                bad_cfg = CeConfig(0, IGNORE_INDEX_DEFAULT, c.reduction,
                                   bad_cfg.eps, c.num_items)
            elif idx == 1:
                bad_targets = List[Int32]()
                bad_logits = List[Float32]()
            elif idx == 2:
                # A logits buffer one cell short of `N * V`. The shape
                # refusal, and it is worth having because a short buffer is
                # the one input that would otherwise read past the end.
                var short = List[Float32]()
                for i in range(cells - 1):
                    short.append(logits[i])
                bad_logits = short^
            elif idx == 3:
                bad_cfg = CeConfig(v, IGNORE_INDEX_DEFAULT, c.reduction,
                                   Float32(1.5), c.num_items)
            elif idx == 4:
                # **THE PLANT IS AT CELL `len / 2`, NOT AT CELL 0.** A
                # plant at index 0 is the one a loop that skips its first
                # element would still catch, which makes it the weakest
                # possible plant site. BY BITS, never by writing a
                # `Float32("nan")`.
                bad_logits[cells // 2] = f32_from_bits(patterns[pk])
                what = String("logits")
            elif idx == 5:
                # A target that is neither `ignore_index` nor in `[0, V)`.
                # An INTEGER refusal, so there is no flush reasoning at
                # this seam at all.
                bad_targets[len(bad_targets) // 2] = Int32(v + 7)
            elif idx == 6:
                # `count == 0` under MEAN. Torch returns NaN here
                # (`0.0/0.0`) and this profile REFUSES, which contract
                # section 11 records as a KNOWING DEPARTURE from the
                # reference.
                bad_cfg = CeConfig(v, IGNORE_INDEX_DEFAULT, REDUCTION_MEAN,
                                   bad_cfg.eps, 0)
                bad_targets = List[Int32]()
                for _r in range(n):
                    bad_targets.append(Int32(IGNORE_INDEX_DEFAULT))
            else:
                bad_cfg = CeConfig(v, IGNORE_INDEX_DEFAULT, REDUCTION_SUM,
                                   bad_cfg.eps, -3)

            var raised = False
            var msg = String("")
            try:
                var st = ce_forward_oracle(bad_logits, bad_targets, bad_cfg)
                _ = st^
            except e:
                raised = True
                msg = String(e)
            if not raised:
                raise Error(
                    String("loss_check: CLAUSE (f) FAILED. The input")
                    + " planted for '"
                    + what
                    + "' was NOT refused. Contract section 8 says a"
                    + " vendor-shaped payload may not enter a certified"
                    + " stage and section 3's refusals are by name."
                )
            if msg.find(what) < 0:
                raise Error(
                    String("loss_check: CLAUSE (f) FAILED. The input")
                    + " planted for '"
                    + what
                    + "' was refused and the refusal does not NAME it: "
                    + msg
                )
            checked += 1
    print(
        "clause (f): "
        + String(checked)
        + " planted inputs, each REFUSED BY NAME by the host oracle,"
        " including both NaN and infinity at cell len/2"
    )

    # ---- REACH, MEASURED ON THE DEVICE ----------------------------------
    # `[[verify-reach-not-output]]`. A plant that did not survive the
    # upload proves nothing about anything downstream of it, so the bits
    # are READ BACK OFF THE DEVICE and counted BY BITS before the call.
    var planted = logits.copy()
    planted[cells // 2] = f32_from_bits(BITS_QNAN)
    var d_check = _upload_f32(ctx, planted)
    var back = _download_f32(ctx, d_check, cells)
    _ = d_check^
    var reached = nonfinite_cells(back)
    if reached != 1:
        raise Error(
            String("loss_check: CLAUSE (f) IS VACUOUS for the device arm.")
            + " The planted NaN did NOT arrive on the device ("
            + String(reached)
            + " non-finite cells read back, expected exactly 1). Anything"
            + " measured after this would be measuring something else"
            + " ([[reached-but-inert]])."
        )
    print(
        "clause (f) reach: the planted NaN is on the device, exactly 1"
        " non-finite cell read back by bits"
    )

    # ---- THE MEASURED GAP, DEVIATION 1460 -------------------------------
    var run = run_device_rows(ctx, c, planted, targets, host_count(c, targets))
    var leaked = 0
    var first_stage = String("")
    for i in range(CE_STAGE_COUNT):
        if stage_is_int(i) or not stage_present(i, c):
            continue
        if i == ST_INPUT_LOGITS:
            continue
        var here = nonfinite_cells(run.dump[i])
        if here > 0:
            leaked += here
            if first_stage == "":
                first_stage = ce_stage_tag(i) + " (" + String(here) + " cells)"
    if leaked == 0:
        raise Error(
            String("loss_check: CLAUSE (f) -- a NaN was planted in the")
            + " logits, MEASURED on the device, and NOT ONE recorded stage"
            + " came back non-finite. Either `loss.mojo` has grown a"
            + " refusal since this file was written (in which case"
            + " DEVIATION 1460 is closed and this arm must be rewritten as"
            + " an assertion that it refuses BY NAME), or the NaN was"
            + " LAUNDERED somewhere -- and a laundered NaN is a worse"
            + " finding than a propagated one, because it means a stage is"
            + " not a function of its input."
        )
    print(
        "clause (f) DEVICE GAP, MEASURED (DEVIATION 1460): `loss.mojo`"
        " never calls `ce_refuse_inputs`, so a planted NaN reached "
        + String(leaked)
        + " recorded cells, first at "
        + first_stage
        + ". Contract section 8's 'REFUSED BY NAME before any recorded"
        " stage' is a property of the ORACLE and NOT of the device entry"
        " point. A caller that reaches `identical_ce_forward_into` directly"
        " puts a vendor-shaped payload into a certified stage. The fix is a"
        " refusal in `loss.mojo`, which this lane may not edit; it is in"
        " the OWED block."
    )


# ===========================================================================
# CLAUSE (g): every clause above falsifiable by a NAMED sabotage
#
# DEVIATION 1461. The expectation table below is contract 10.1's,
# DUPLICATED into code, and a duplicated table is a table that can drift.
# The alternative was to have the check read the contract, which is a
# markdown file, which would make the gate depend on parsing prose. The
# duplication is accepted and the mitigation is that every row cites the
# contract and that the ONE place this file's table knowingly departs from
# the contract's is argued in the header (`L_NLL_VIA_MAX_LOGSOFTMAX`, which
# has no switch at all).
# ===========================================================================


@fieldwise_init
struct ArmExpectation(Copyable, Movable):
    """What one sabotage arm must do.

    `first_stage` is the tag the arm's OWN SEAM writes, and the discipline
    contract 10.1 states is that the arm must move THAT stage "and no
    earlier one". An arm that moves an EARLIER stage is not aimed where it
    says it is; an arm that moves a LATER one has been absorbed on the way
    and its clause is being gated by the wrong stage.

    `inert_a` and `inert_b` are cases on which the arm must move NOTHING,
    and they are the half that turns a smoke test into a reach proof. Two
    slots because `L_SMOOTH_ALWAYS_SPELLED` has two DIFFERENT reasons to be
    inert -- `base_n4_v8` has `EPS == 0` and no `-0.0` row, `smooth_v1_eps`
    has a `-0.0` row and `EPS != 0` -- and an arm that was inert for one
    reason and not the other would be a finding.

    `unmoved_stage` is a stage that must NOT move ON THE WITNESS CASE, in
    the same run in which `first_stage` moves. Two arms in this lane are
    stated that way and neither can be expressed any other way:
      * `L_IGNORED_ROW_NEG_ZERO` must move `ce.row` and must NOT move
        `ce.total`, because L12's leaf accumulator is seeded `+0.0` and
        `(+0.0) + (-0.0)` is `+0.0`. **A gate that compared only the final
        loss would call this arm inert, and it is not** -- it is a
        divergence at the stage that produced it. Contract 7.3.
      * `L_GRAD_DIVISOR_IS_N` must move `ce.dlogits` and must NOT move
        `ce.divisor`, because the substitution happens INSIDE the backward
        and the recorded divisor is still the one producer's. Contract 5.5.

    `needs_poison` marks the one arm whose entire effect is an UNWRITTEN
    cell, so the verdict has to look at the survivor count and not only at
    the values."""

    var arm: String
    var first_stage: String
    var witness: String
    var inert_a: String
    var inert_b: String
    var unmoved_stage: String
    var needs_poison: Bool
    var note: String


def arm_expectation(arm: String) raises -> ArmExpectation:
    if arm == "MAX_PLAIN_COMPARE":
        return ArmExpectation(
            arm, String("ce.max"), String("adv_signed_zeros"),
            String("base_n4_v8"), String("wide_v300"), String(""), False,
            String(
                "the order-free maximum. A plain `>` and a total-order"
                " selection agree at every input EXCEPT a row carrying"
                " both zero signs, and row 39 measured max(+0.0, -0.0) as"
                " -0.0 on Apple against +0.0 on NVIDIA and AMD"
            ),
        )
    if arm == "MAX_SEED_ZERO":
        return ArmExpectation(
            arm, String("ce.max"), String("adv_all_negative"),
            String("base_n4_v8"), String("wide_v300"), String(""), False,
            String(
                "the -inf seed. A +0.0 seed clamps an ALL-NEGATIVE row --"
                " which is most rows of a trained head -- and is wrong on"
                " EVERY VENDOR IDENTICALLY, so bit-identity cannot see it"
                " and only the oracle can. That is the sharpest argument"
                " in this lane for gating against an oracle rather than"
                " against a second device"
            ),
        )
    if arm == "MAX_TOPK_PREFIX":
        return ArmExpectation(
            arm, String("ce.max"), String("adv_argmax_tail"),
            String("adv_argmax_head"), String("base_n4_v8"), String(""),
            False,
            String(
                "the max over the WHOLE row. The two cases are the same"
                " shape and the same seed with the argmax moved from index"
                " 63 to index 0, which is what makes the pair a controlled"
                " comparison"
            ),
        )
    if arm == "EXP_STDLIB":
        return ArmExpectation(
            arm, String("ce.exp"), String("base_n4_v8"), String(""),
            String(""), String(""), False,
            String(
                "row 12's exp. NO INERT CASE IS ASSERTED: the arm is inert"
                " under FAST (where identical_exp IS the stdlib) and at"
                " shift == +0.0, and this file REFUSES to evaluate it"
                " under FAST rather than pretending a FAST inert result is"
                " evidence"
            ),
        )
    if arm == "LOG_STDLIB":
        return ArmExpectation(
            arm, String("ce.logdenom"), String("wide_v300"),
            String("base_n1_v1"), String(""), String(""), False,
            String(
                "row 12's log. `[[mojo-log-breaks-ties]]`: the stdlib log"
                " carries about 5e-8 absolute error and re-decides plateau"
                " ties. INERT at base_n1_v1, where V == 1 makes denom"
                " exactly 1.0 and both spellings return exactly +0.0 --"
                " which preflight MEASURES for identical_log rather than"
                " assuming"
            ),
        )
    if arm == "NLL_VIA_ADDBACK":
        return ArmExpectation(
            arm, String("ce.logp_target"), String("adv_big_offset"),
            String("adv_centered"), String(""), String(""), False,
            String(
                "contract 4.2(a). `(m + logdenom) - x_y` re-adds the row"
                " maximum the shift existed to remove. INERT on a CENTERED"
                " row, which is contract 4.2's own 'what would make L6 pass"
                " while gating nothing'"
            ),
        )
    if arm == "NLL_VIA_LOG_W":
        return ArmExpectation(
            arm, String("ce.logp_target"), String("adv_underflow_y"),
            String("base_n4_v8"), String(""), String(""), False,
            String(
                "contract 4.2(b). `-log(w[y])` returns +inf where e[y] has"
                " underflowed to exactly +0.0 and the pin returns an"
                " ordinary finite number near 200. INERT wherever e[y] is"
                " normal"
            ),
        )
    if arm == "NEG_VIA_ZERO_SUB":
        return ArmExpectation(
            arm, String("ce.nll"), String("base_n1_v1"),
            String("base_n4_v8"), String("wide_v300"), String(""), False,
            String(
                "contract 4.3. `0.0 - x` is WRONG at x == +0.0, where IEEE"
                " negation gives -0.0. INERT at every other input, and"
                " base_n1_v1 is the ONLY case in the set whose lp_y is"
                " exactly +0.0"
            ),
        )
    if arm == "SMOOTH_FOLDED_CONSTANT":
        return ArmExpectation(
            arm, String("ce.smooth"), String("smooth_n4_v8"),
            String("base_n4_v8"), String(""), String(""), False,
            String(
                "contract 6.2(a). A host-folded eps/V rounds the quotient"
                " once on the host and once in the product; the pin rounds"
                " one quotient then one product, and the two quotients are"
                " not the same number. INERT at EPS == 0, where the arm is"
                " not spelled at all"
            ),
        )
    if arm == "SMOOTH_FUSED_COMBINE":
        return ArmExpectation(
            arm, String("ce.row"), String("smooth_n4_v8"),
            String("base_n4_v8"), String(""), String(""), False,
            String(
                "contract 6.2(b). One rounding where the contract has"
                " three. INERT at EPS == 0"
            ),
        )
    if arm == "SMOOTH_ALWAYS_SPELLED":
        return ArmExpectation(
            arm, String("ce.row"), String("base_n1_v1"),
            String("base_n4_v8"), String("smooth_v1_eps"), String(""),
            False,
            String(
                "contract 6.2(c), THE STRONGEST REACH-PER-BRANCH ARM IN"
                " THIS LANE. It moves exactly one thing: a -0.0 row loss"
                " laundered to +0.0 by ftz(nll + (+0.0)). TWO inert cases"
                " for TWO DIFFERENT REASONS -- base_n4_v8 has EPS == 0 and"
                " no -0.0 row, smooth_v1_eps has a -0.0 row and EPS != 0"
                " -- and an arm inert for one reason and not the other"
                " would be a finding"
            ),
        )
    if arm == "MEAN_RECIPROCAL_MUL":
        return ArmExpectation(
            arm, String("ce.loss"), String("base_n3_v3"),
            String("base_n4_v8"), String("long_n512_sum"), String(""),
            False,
            String(
                "contract 5.5. `total * (1/divisor)` is EXACT, and"
                " therefore inert, when the divisor is a power of two."
                " base_n4_v8 has divisor 4 and long_n512_sum has divisor"
                " exactly 1.0"
            ),
        )
    if arm == "IGNORED_ROW_NEG_ZERO":
        return ArmExpectation(
            arm, String("ce.row"), String("ign_n6_v8_c3"),
            String("base_n4_v8"), String(""), String("ce.total"), False,
            String(
                "contract 7.3. It MUST move ce.row and MUST NOT move"
                " ce.total, because L12's leaf accumulator is seeded +0.0"
                " and (+0.0) + (-0.0) is +0.0 in round-to-nearest. **A"
                " gate that compared only the final loss would call this"
                " arm inert and would be wrong** -- it is a divergence at"
                " the stage that produced it, and the CARD is the only"
                " instrument that can see it"
            ),
        )
    if arm == "IGNORED_ROW_SKIPPED":
        return ArmExpectation(
            arm, String("ce.row"), String("ign_n6_v8_c3"),
            String("base_n4_v8"), String(""), String(""), True,
            String(
                "contract 7.3, the store is REQUIRED. ALWAYS INERT against"
                " a zero-filled buffer and never inert against a poisoned"
                " one, which is why every output buffer here is pre-filled"
                " with 0x7fc0dead (DEVIATION 1456)"
            ),
        )
    if arm == "W_VIA_EXP_LOGP":
        return ArmExpectation(
            arm, String("ce.weights"), String("smooth_n4_v8"), String(""),
            String(""), String(""), False,
            String(
                "seam L14's division. **RESTRICTED TO SMOOTHING CASES**:"
                " the arm reads the `logp` buffer, which is a ONE-ELEMENT"
                " placeholder when smoothing is off, and"
                " `identical_ce_backward_into` contains none of the"
                " refusal its docstring claims (DEVIATION 1458). No inert"
                " case is asserted"
            ),
        )
    if arm == "GRAD_SIGN":
        return ArmExpectation(
            arm, String("ce.dlogits"), String("base_n4_v8"),
            String("base_n1_v1"), String(""), String(""), False,
            String(
                "contract 6.4. `t - w` instead of `w - t`. INERT only"
                " where w == t at every class, which base_n1_v1 reaches"
                " exactly: V == 1 makes w exactly 1.0 and T_TARGET exactly"
                " 1.0 at EPS == 0"
            ),
        )
    if arm == "GRAD_TARGET_OFF_BY_ONE":
        return ArmExpectation(
            arm, String("ce.dlogits"), String("base_n4_v8"),
            String("base_n1_v1"), String(""), String(""), False,
            String(
                "seam L15's target column. INERT at V == 1, where"
                " (y + 1) mod 1 is y. At V > 1 it moves exactly TWO cells"
                " per row, which is why the check compares CELLS and never"
                " a row norm"
            ),
        )
    if arm == "GRAD_DIVISOR_IS_N":
        return ArmExpectation(
            arm, String("ce.dlogits"), String("ign_n6_v8_c3"),
            String("base_n4_v8"), String("ign_n5_v8_c2"),
            String("ce.divisor"), False,
            String(
                "contract 5.5's ONE PRODUCER. It substitutes N for count"
                " INSIDE the backward, so it must move ce.dlogits and must"
                " NOT move ce.divisor. **INERT ON ANY FIXTURE WITH NO"
                " IGNORED ROW**, which is why base_n4_v8 (count == N == 4)"
                " is the inert half -- and note ign_n5_v8_c2 is NOT inert"
                " (count 2, N 5) and is listed as a second inert slot only"
                " if a future edit makes it one, so it is deliberately left"
                " EMPTY here"
            ),
        )
    if arm == "GRAD_RECIPROCAL_MUL":
        return ArmExpectation(
            arm, String("ce.dlogits"), String("base_n3_v3"),
            String("base_n4_v8"), String("long_n512_sum"), String(""),
            False,
            String(
                "seam L16's division, EXACT and therefore inert at a"
                " power-of-two divisor. This is transformer DEVIATION 806"
                " at a second site and the two must agree"
            ),
        )
    if arm == "DENOM_SERIAL_CHAIN":
        return ArmExpectation(
            arm, String("ce.denom"), String("wide_v300"),
            String("base_n4_v8"), String("base_n3_v3"), String(""), False,
            String(
                "contract 5.3, DEVIATION 1152. A whole-axis ascending"
                " chain instead of gemm v1's leaf-and-tree. **INERT AT"
                " EVERY V <= 128**, where P == 1 and the tree performs no"
                " addition (gemm 7.3). This arm is also the instrument"
                " that PRINTS the difference between this lane's fold and"
                " the transformer lane's rather than arguing about it"
            ),
        )
    if arm == "REDUCE_SERIAL":
        return ArmExpectation(
            arm, String("ce.total"), String("long_n300_v8"),
            String("base_n4_v8"), String("base_n3_v3"), String(""), False,
            String(
                "contract 5.4, the BATCH fold. INERT at every N <= 128"
            ),
        )
    # ---- the five reached through gemm_identical.mojo's own switches ----
    # Contract 7.2: showing these fail THROUGH this lane's entry points is
    # the proof that the routing lands on the CONTRACT's arithmetic rather
    # than on some other path that happens to agree. The HOST oracle is
    # unaffected by them -- `gemm_oracle.mojo` carries no `is_defined` at
    # all -- which is exactly what makes clause (a) able to see them.
    if arm == "G_FOLD_STRIDE":
        return ArmExpectation(
            arm, String("ce.denom"), String("wide_v300"),
            String("base_n4_v8"), String(""), String(""), False,
            String(
                "contract 10.1's L_DENOM_HALVING_TREE, reached through the"
                " routing. `pinned_block_sum`'s STRIDE pairing, which gemm"
                " contract 7.2 clause 1 names as a DIFFERENT ANSWER from"
                " v1's adjacent pairing. INERT at every V <= 128"
            ),
        )
    if arm == "G_PAD_PLUS_ZERO":
        return ArmExpectation(
            arm, String("ce.denom"), String("wide_v300"),
            String("wide_v256"), String("wide_v1024"), String(""), False,
            String(
                "contract 10.1's L_DENOM_PAD_PLUS_ZERO. The odd tail padded"
                " with a +0.0 instead of carried. **INERT ON ANY V WHOSE"
                " EVERY LEVEL WIDTH IS EVEN**, which is why the two inert"
                " cases are V = 256 (P = 2, widths 2, 1) and V = 1024"
                " (P = 8, widths 8, 4, 2, 1) while V = 300 (P = 3) carries"
                " -- `case_fold_has_carry` computes exactly that and"
                " preflight checks it against the GEMM oracle's own leaf"
                " rule"
            ),
        )
    if arm == "G_LEAF_READS_LAUNCH":
        return ArmExpectation(
            arm, String("ce.denom"), String("wide_v300"), String(""),
            String(""), String(""), False,
            String(
                "contract 10.1's L_VOCAB_FOLD_READS_LAUNCH and contract"
                " 7.2: (L, P) come from V and the two profile constants"
                " and from NOTHING else. Contract 10.1's inert column says"
                " 'nothing', so none is asserted"
            ),
        )
    if arm == "G_LEAF_ROTATE":
        return ArmExpectation(
            arm, String("ce.denom"), String("wide_v300"),
            String("base_n4_v8"), String(""), String(""), False,
            String(
                "the leaf ROTATION by block index -- which leaf stands at"
                " which tree position depends on the physical block. Not"
                " in contract 10.1's table and reachable through this"
                " lane's routing anyway, so it is here"
            ),
        )
    if arm == "G_FOLD_SERIAL":
        return ArmExpectation(
            arm, String("ce.denom"), String("wide_v300"),
            String("base_n4_v8"), String(""), String(""), False,
            String("gemm fixture F5's serial fold, through the routing"),
        )
    if arm == "G_NODE_ORDER":
        return ArmExpectation(
            arm, String("ce.denom"), String("wide_v300"),
            String("base_n4_v8"), String(""), String(""), False,
            String(
                "gemm contract 7.2.2: the pairing is over LOGICAL leaf"
                " indices, never over physical block, warp or thread"
                " indices"
            ),
        )
    raise Error(
        String("loss_check: '")
        + arm
        + "' is not one of the arms this file knows. Contract 10.1 lists"
        + " twenty-five and `loss.mojo` implements twenty-one of them;"
        + " L_NLL_VIA_MAX_LOGSOFTMAX (contract 4.2(c)) HAS NO SWITCH"
        + " ANYWHERE (DEVIATION 1457) and the other three are reached"
        + " through gemm_identical.mojo's own switches. If an arm was"
        + " added to loss.mojo, this table owes it a row."
    )


def armed_arm_name() raises -> String:
    """Which arm this BINARY carries, across all three switch files.

    A loss binary can carry a GEMM sabotage as well -- four of this lane's
    clauses are reached that way ON PURPOSE -- so a banner that named only
    one of them would mislabel the run. That is
    `gemm_backward_sabotage_name`'s own warning with a third file added to
    it, and `loss_sabotage_name`'s docstring says a check MUST print all
    three.

    **TWO ARMS AT ONCE IS REFUSED.** A binary carrying a loss arm and a
    GEMM arm produces a verdict nobody can attribute: if `ce.denom` moves,
    which one moved it? The whole discipline of clause (g) is that an arm
    moves the stage its OWN seam writes AND NO EARLIER ONE, and that
    sentence has no meaning for two seams at once."""
    var l = loss_sabotage_name()
    var g = gemm_sabotage_name()
    var b = gemm_backward_sabotage_name()
    var n_armed = 0
    var name = String("none")
    if l != "none":
        n_armed += 1
        name = l
    if g != "none":
        n_armed += 1
        name = String("G_") + g
    if b != "none":
        n_armed += 1
        name = String("B_") + b
    if n_armed > 1:
        raise Error(
            String("loss_check: this binary carries ")
            + String(n_armed)
            + " sabotage arms at once (loss='"
            + l
            + "', gemm='"
            + g
            + "', gemm_backward='"
            + b
            + "'). A verdict cannot be attributed to a seam when two seams"
            + " are broken, and clause (g)'s whole discipline is that an"
            + " arm moves the stage its OWN seam writes and no earlier"
            + " one. Build one arm at a time."
        )
    return name^


def clause_g(
    ctx: DeviceContext, arm: String, verdicts: List[CaseVerdict]
) raises:
    """The INVERTED verdict of a sabotage build.

    When an arm is armed this file INVERTS: a clean compare is the FAILURE,
    because it means the sabotage was reached and made no difference, or
    was never reached at all. Both are `[[reached-but-inert]]` and both are
    reported as such rather than as a pass."""
    var exp = arm_expectation(arm)
    print("clause (g): arm " + arm + " -- " + exp.note)

    if arm == "EXP_STDLIB" and not mode_is_identical():
        raise Error(
            String("loss_check: arm EXP_STDLIB cannot be evaluated under ")
            + mode_name()
            + ". Under FAST `identical_exp` IS the stdlib, so the arm is"
            + " bit-inert BY CONSTRUCTION and an inert result here is"
            + " evidence about the build mode and not about the seam"
            + " ([[reached-but-inert]])."
        )
    if arm == "LOG_STDLIB" and not mode_is_identical():
        raise Error(
            String("loss_check: arm LOG_STDLIB cannot be evaluated under ")
            + mode_name()
            + " for EXP_STDLIB's reason."
        )

    # ---- THE WITNESS -----------------------------------------------------
    var wv = find_verdict(verdicts, exp.witness)
    if exp.needs_poison:
        if wv.poison == 0:
            raise Error(
                String("loss_check: SABOTAGE ")
                + arm
                + " IS ARMED and its witness "
                + exp.witness
                + " came back with ZERO unwritten cells. This arm's entire"
                + " effect is a store that does not happen, so a zero"
                + " survivor count means the store happened -- the branch"
                + " was never reached ([[reached-but-inert]])."
            )
        print(
            "clause (g): "
            + arm
            + " left "
            + String(wv.poison)
            + " cells UNWRITTEN on "
            + exp.witness
            + ", which is the arm's whole effect and is invisible against"
            + " a zero-filled buffer"
        )
    if wv.n_moved == 0:
        raise Error(
            String("loss_check: SABOTAGE ")
            + arm
            + " IS ARMED AND MOVED NO BIT on its witness case "
            + exp.witness
            + ". Either its branch was never reached at this shape or it"
            + " is inert there ([[reached-but-inert]]). It falsifies"
            + " NOTHING and must not be reported as a passing arm."
        )
    var want = stage_index_of(exp.first_stage)
    if wv.first_index != want:
        raise Error(
            String("loss_check: SABOTAGE ")
            + arm
            + " moved '"
            + wv.first
            + "' FIRST on "
            + exp.witness
            + ", and contract 10.1 says its own clause writes '"
            + exp.first_stage
            + "'. Each arm must move the stage its OWN seam writes and no"
            + " EARLIER one; an earlier stage means the arm is not aimed"
            + " where it says it is, and a LATER one means it was absorbed"
            + " on the way and the wrong stage is gating its clause."
        )
    print(
        "clause (g): "
        + arm
        + " BIT on "
        + exp.witness
        + ": "
        + String(wv.n_moved)
        + " stages moved, FIRST at "
        + exp.first_stage
        + ", which is the stage its own seam writes"
    )

    # ---- THE STAGE THAT MUST **NOT** MOVE, ON THE SAME CASE -------------
    if exp.unmoved_stage != "":
        var idx = stage_index_of(exp.unmoved_stage)
        if stage_moved(wv, idx):
            raise Error(
                String("loss_check: SABOTAGE ")
                + arm
                + " moved '"
                + exp.unmoved_stage
                + "' on "
                + exp.witness
                + " and the contract requires it NOT to. For"
                + " IGNORED_ROW_NEG_ZERO that is contract 7.3's proof that"
                + " a `+0.0`-seeded chain launders a `-0.0` -- if"
                + " `ce.total` moved, the proof is false and THAT is the"
                + " finding. For GRAD_DIVISOR_IS_N it is contract 5.5's"
                + " one-producer clause: the substitution is inside the"
                + " backward and the recorded divisor must still be the"
                + " producer's."
            )
        print(
            "clause (g): "
            + arm
            + " left '"
            + exp.unmoved_stage
            + "' UNMOVED on the same case, which is the half a"
            + " loss-only gate cannot see at all"
        )

    # ---- THE INERT HALVES ------------------------------------------------
    var inerts: List[String] = [exp.inert_a, exp.inert_b]
    var shown = 0
    for i in range(len(inerts)):
        if inerts[i] == "":
            continue
        var iv = find_verdict(verdicts, inerts[i])
        if iv.n_moved != 0:
            raise Error(
                String("loss_check: SABOTAGE ")
                + arm
                + " moved "
                + String(iv.n_moved)
                + " stages on "
                + inerts[i]
                + ", first at "
                + iv.first
                + ", and contract 10.1 requires it to move NOTHING there."
                + " **An arm that moves everywhere is a smoke test; the"
                + " inert case is what makes it a REACH PROOF**"
                + " ([[verify-reach-not-output]])."
            )
        if iv.poison != 0:
            raise Error(
                String("loss_check: SABOTAGE ")
                + arm
                + " left "
                + String(iv.poison)
                + " cells unwritten on the INERT case "
                + inerts[i]
            )
        shown += 1
        print(
            "clause (g): "
            + arm
            + " is INERT on "
            + inerts[i]
            + ", every stage unmoved"
        )
    if shown == 0:
        print(
            "clause (g): "
            + arm
            + " carries NO inert case, so this run is a SMOKE TEST for it"
            + " and not a reach proof. Contract 10.1's inert column for"
            + " this arm says 'nothing', and that is the reason -- not an"
            + " omission in the table."
        )


# ===========================================================================


def main() raises:
    var armed = armed_arm_name()

    print(
        "=== cross-entropy identity gate, profile"
        " mojolearn.identical.loss.ce.fp32.v1"
    )
    print(
        "=== NOTHING IN THIS FILE, IN loss.mojo, IN loss_oracle.mojo OR IN"
        " loss_fixture.mojo HAD EVER BEEN COMPILED OR RUN BEFORE THIS"
        " PROCESS. Read the header."
    )
    print(
        "mode "
        + mode_name()
        + "   loss sabotage: "
        + loss_sabotage_name()
        + "   gemm: "
        + gemm_sabotage_name()
        + "   gemm_backward: "
        + gemm_backward_sabotage_name()
    )
    print(
        "NOTE: contract 10.1 lists 25 arms. `loss.mojo` implements 21,"
        " three more are reached through gemm_identical.mojo's own"
        " switches, and **L_NLL_VIA_MAX_LOGSOFTMAX (contract 4.2(c)) HAS"
        " NO SWITCH ANYWHERE** (DEVIATION 1457), so contract 4.2(c) is a"
        " pinned decision with no falsifier in the tree. Adding the switch"
        " is an edit to loss.mojo, which this lane may not make."
    )

    # ---- THE LEDGER OF WHICH ARM THIS BINARY WAS BUILT WITH -------------
    # DEVIATION 1461, and it is `tools/gemm_ladder.sh:71`'s scar written
    # down. A `-D MOJOLEARN_LOSS_SABOTAGE_...` with a typo in it is
    # SILENTLY IGNORED by the compiler: `is_defined` returns False and the
    # build is clean. The operator then sees a green gate and records it as
    # "arm X did not bite", which is the exact inverse of the truth. So the
    # operator may state what they expect and the binary checks itself.
    var expect = env_str("MOJOLEARN_LOSS_EXPECT_SABOTAGE")
    if expect != "":
        if expect != armed:
            raise Error(
                String("loss_check: the caller expected sabotage '")
                + expect
                + "' and this BINARY was built with '"
                + armed
                + "'. A misspelled -D is silently ignored by the compiler"
                + " and produces a clean build that a caller reads as 'the"
                + " arm did not bite'. Fix the -D or the expectation."
                + " (GEMM arms are named with a G_ prefix here, e.g."
                + " G_PAD_PLUS_ZERO for"
                + " MOJOLEARN_GEMM_SABOTAGE_PAD_PLUS_ZERO.)"
            )
        print(
            "ledger: the caller expected '"
            + expect
            + "' and the binary agrees, so the -D was not silently dropped"
        )
    elif armed == "none":
        print(
            "ledger: this binary is CLEAN -- no sabotage arm is compiled"
            " in. Set MOJOLEARN_LOSS_EXPECT_SABOTAGE to have the binary"
            " check its own -D."
        )
    else:
        print(
            "ledger: this binary carries sabotage '"
            + armed
            + "' and the caller did not say so. Set"
            " MOJOLEARN_LOSS_EXPECT_SABOTAGE to close the misspelled -D"
            " hole."
        )

    preflight()

    var ctx = DeviceContext()

    # ---- CLAUSE (a) ------------------------------------------------------
    var cases = clause_a_cases()
    if armed == "W_VIA_EXP_LOGP":
        # DEVIATION 1458. The arm reads the `logp` buffer, which is a
        # ONE-ELEMENT placeholder when smoothing is off. Restricting the
        # set is not a convenience: running the arm on a non-smoothing case
        # is an OUT-OF-BOUNDS READ, and calling whatever comes back a
        # sabotage verdict would be worse than not running the arm at all.
        var only_smooth = List[Int]()
        for i in range(len(cases)):
            if ce_case(cases[i]).eps_on:
                only_smooth.append(cases[i])
        print(
            "clause (a): RESTRICTED to the "
            + String(len(only_smooth))
            + " SMOOTHING cases, because arm W_VIA_EXP_LOGP reads the"
            " `logp` buffer and `identical_ce_backward_into` contains none"
            " of the refusal `ce_weights_kernel`'s docstring claims"
            " (DEVIATION 1458)"
        )
        cases = only_smooth^
    print(
        "clause (a): "
        + String(len(cases))
        + " fixture cases, every stage, device vs host oracle, BITWISE"
    )
    var verdicts = List[CaseVerdict]()
    var cpath = card_path()
    for ci in range(len(cases)):
        if ci == 0:
            # ONLY THE FIRST CASE WRITES THE CARD. `IdentityTrace` enforces
            # tag uniqueness within one trace, so twenty-four cases would
            # emit `ce.input.logits` twenty-four times and raise. A
            # per-case prefix would be the alternative and it is
            # deliberately not taken: the card `tools/e3_round_judge.sh`
            # reads must have contract section 9's tags and nothing else,
            # and a 400-record card is not that card.
            var trace = IdentityTrace.to_path(cpath)
            verdicts.append(clause_a_case(ctx, cases[ci], trace, TAG_PREFIX))
        else:
            var off = IdentityTrace.disabled()
            var pfx = "case" + String(cases[ci])
            verdicts.append(clause_a_case(ctx, cases[ci], off, pfx))

    _ = check_card_tags(cpath, ce_case(cases[0]))

    var moved_cases = 0
    var first_case = String("")
    var all_cells = 0
    for i in range(len(verdicts)):
        all_cells += verdicts[i].cells
        if verdicts[i].n_moved > 0:
            moved_cases += 1
            if first_case == "":
                first_case = verdicts[i].name + " at " + verdicts[i].first

    if armed != "none":
        clause_g(ctx, armed, verdicts)
        print(
            "clauses (b), (c), (d) and (f) are NOT run under a sabotage"
            " build: (b), (c) and (d) are INVARIANCE claims and a"
            " DETERMINISTIC sabotage satisfies all three, and (f)'s"
            " refusal is upstream of every sabotaged seam. Clause (e) is"
            " not run either, and that one is worth a sentence: it is the"
            " CORRECTNESS arm, so under an armed build it would fail for"
            " the arm's own reason and the failure would prove nothing"
            " about the gate."
        )
        return

    if moved_cases != 0:
        var n = verdicts[0].n_moved
        var f = verdicts[0].first
        for i in range(len(verdicts)):
            if verdicts[i].n_moved > 0:
                n = verdicts[i].n_moved
                f = verdicts[i].first
                break
        if mode_is_identical():
            raise Error(
                String("loss_check: CLAUSE (a) FAILED, ")
                + String(n)
                + " stages differ from the oracle, first at "
                + f
                + "  (case "
                + first_case
                + ")"
            )
        else:
            # FAST arms of (a) are RECORDED, not asserted, where they are
            # vendor-shaped -- contract section 10's second paragraph, and
            # the metrics lane's leg-11 lesson. Under FAST every
            # `identical_*` compiles away to the platform's own spelling
            # and the two halves of this lane are two different platforms'
            # libm, so a difference here is EXPECTED and means nothing
            # about the profile.
            print(
                "clause (a) [FAST]: RECORDED, NOT ASSERTED. "
                + String(moved_cases)
                + " of "
                + String(len(verdicts))
                + " cases differ from the oracle, first at "
                + first_case
                + ". FAST is unversioned and makes no identity claim"
                + " (contract section 0)."
            )
    else:
        print(
            "clause (a): PASS, "
            + String(len(verdicts))
            + " cases bit-identical to the oracle on all "
            + String(all_cells)
            + " cells"
        )

    # ---- CLAUSE (e) IS ALWAYS ON ---------------------------------------
    # It is the ONLY correctness arm in the file. Every other clause here
    # asks whether our device says the same thing as our oracle; this one
    # asks whether either of them is right, against a number written by
    # hand. A run without it has checked spelling and not answers.
    clause_e(ctx)
    clause_e_float64_report(ce_case_by_name(String("base_n4_v8")))

    if env_on("MOJOLEARN_LOSS_CHECK_CLAUSE_B"):
        clause_b(ctx, ce_case_by_name(String("base_n4_v8")))
    else:
        print("clause (b): SKIPPED (set MOJOLEARN_LOSS_CHECK_CLAUSE_B=1)")

    if env_on("MOJOLEARN_LOSS_CHECK_CLAUSE_C"):
        var ck = env_case(
            "MOJOLEARN_LOSS_CHECK_C_CASE",
            ce_case_by_name(String("ign_n6_v8_c3")),
        )
        clause_c(ctx, ck)
    else:
        print(
            "clause (c): SKIPPED (set MOJOLEARN_LOSS_CHECK_CLAUSE_C=1)."
            " Its default case is ign_n6_v8_c3 rather than a plain one,"
            " because an IGNORED ROW is what makes the 'carry the whole"
            " batch's count into every chunk' half of contract 7.1's"
            " escape hatch mean anything."
        )

    if env_on("MOJOLEARN_LOSS_CHECK_CLAUSE_D"):
        var dk = env_case(
            "MOJOLEARN_LOSS_CHECK_D_CASE",
            ce_case_by_name(String("wide_v300")),
        )
        clause_d(ctx, dk)
    else:
        print(
            "clause (d): SKIPPED (set MOJOLEARN_LOSS_CHECK_CLAUSE_D=1)."
            " NOTE: without it, the four fold arms reached through"
            " gemm_identical.mojo are gated only by clause (a), which sees"
            " ONE execution plan -- whichever `choose_gemm_plan` picked."
        )

    if env_on("MOJOLEARN_LOSS_CHECK_CLAUSE_F"):
        clause_f(ctx, ce_case_by_name(String("base_n4_v8")))
    else:
        print(
            "clause (f): SKIPPED (set MOJOLEARN_LOSS_CHECK_CLAUSE_F=1)."
            " NOTE: it carries the MEASURED finding that `loss.mojo` never"
            " calls `ce_refuse_inputs` at all (DEVIATION 1460), so a leg"
            " that skips it has not looked at the refusal on the DEVICE"
            " side even once."
        )

    print(
        "SCOPE: this build, this column, "
        + mode_name()
        + " only. What is NOT closed by anything printed above: an"
        " INDEPENDENT reference (training/corpus/ does not exist, so every"
        " clause here is our device against our oracle, and three of the"
        " twenty stages are literally the SAME HOST CODE on both sides --"
        " see `stage_is_host_constant`); the SHIPPED vocabulary V=128256"
        " unless MOJOLEARN_LOSS_CHECK_SHIPPED_V was set; FAST mode; every"
        " column that is not this one; contract 4.2(c), which has no"
        " switch at all; and the "
        + String(len(cases))
        + "-case set's every arm this binary is not. Contract section 11:"
        " nothing cross-vendor until a leg runs, and two backends agreeing"
        " closes nothing -- Apple and AMD agreed bit for bit through 302"
        " GEMM stages while NVIDIA diverged at tree001.winners.scores."
    )
