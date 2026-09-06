# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate file of profile `mojolearn.identical.optimizer.fp32.v1`, the
file `training/IDENTICAL_OPTIMIZER_CONTRACT.md` section 11 names and
section 16 item 1 calls **"the largest owed item by a wide margin"**.

NOT A PORT. It runs the device optimizer step
(`training/checks/optimizer.mojo`) against the host oracle
(`training/checks/optimizer_oracle.mojo`) and compares every recorded
stage BY BITS, at every step of a multi-step run.

Recorded executions and stage cards live under `bench/results/`. Each run
establishes evidence only for its build, selected fixtures, enabled clauses
and device. A device-to-oracle comparison does not independently validate
the mathematics or establish cross-vendor identity.

The gate's non-finite-input audit exposed the clipping-off parameter
refusal defect recorded in DEVIATION 1496. Sabotage arms without a
non-vacuous control remain limited as described below.

WHY THE CASES CARRY A **STEP COUNT** AND THE GATE COMPARES AT EVERY STEP
--------------------------------------------------------------------------
DEVIATION 1474, and it is a finding this file makes that contract section
12's table does not contain.

`m`, `v` and the momentum buffer are RUNNING STATE. Contract section 12
lists the fixture property each arm needs, and for `OPT_SAB_MOMENT_LERP` it
says "`g` and `m` in the same binade with no low bits". That is true and it
is not the whole condition. **The arm is bit-inert whenever `m_prev` is
exactly `+0.0`, which is EVERY FIRST STEP FROM FRESH STATE**: `m + c1*(g -
m)` at `m = 0` is `c1*g`, and the pinned `fma(c1, g, ftz(beta1 * 0))` is
`c1*g` too. An optimizer gate that ran ONE step -- which is the obvious
shape, since the profile pins one step -- could not see that arm at all,
would report it inert, and somebody would delete it.

Two more clauses need the same thing. Contract 5.1's `beta^t` spellings
agree EXACTLY through `t = 6` and first differ at `t = 7`, so a fixture
that stops at `t = 4` is vacuous. And contract 11(d)'s step-count
invariance is a statement about two runs of DIFFERENT LENGTHS; there is
nothing to compare in one step.

THE SABOTAGE LEDGER, **NOT MEASURED**
---------------------------------------
Contract section 12 lists twenty-two arms. `optimizer.mojo` implements
TWENTY of them. The two it does not are `OPT_SAB_RESUME_REINIT` and
`OPT_SAB_MICROBATCH_SERIAL`, and DEVIATION 1473 is what this file does
about that: **both are modelled in the GATE ITSELF**, because both describe
a defect in the CALLER's behavior rather than in a kernel -- a checkpoint
that drops a flag, and an accumulation loop that folds serially -- and a
`comptime` switch inside `optimizer.mojo` could not express either. They
are driven by `MOJOLEARN_OPT_GATE_ARM` rather than by a `-D`, and the
report says which kind an arm is so nobody counts twenty-two `-D` builds
and finds twenty.

    arm                        first stage        witness / inert
    OPT_SAB_POW_RUNNING        sched.pow1         adam_t7 / adam_t6
    OPT_SAB_POW_EXPLOG         sched.pow1         adam_t1 / --
    OPT_SAB_EPS_INSIDE_SQRT    adam.denom         adam_dead_v /
                                                  adam_ordinary_v
    OPT_SAB_RSQRT              adam.denom         adam_hashed_4096 / --
    OPT_SAB_RECIP_MUL          adam.q             adam_j5_t8_noclip / --
    OPT_SAB_ADAMW_AS_ADAM      param.out          adamw_wd_t8 /
                                                  adamw_wd0_t8
    OPT_SAB_DECAY_ADD_FORM     param.out          adamw_decay_sep /
                                                  adamw_decay_pow2
    OPT_SAB_MOMENT_LERP        adam.m             adam_t8 / adam_t1
    OPT_SAB_SQ_ASSOC           adam.v             adam_j5_t8_noclip /
                                                  adam_unit_grad
    OPT_SAB_MHAT_FORM          adam.denom         adam_j5_t8_noclip / --
    OPT_SAB_UNFUSED_UPDATE     param.out          adam_fused_sep /
                                                  adam_fused_inert
    OPT_SAB_FTZ_LATE           adam.v             adam_tiny_grad /
                                                  adam_ordinary_v
    OPT_SAB_CLIP_SKIP_AT_ONE   clip.grad          clip_subnormal /
                                                  clip_normal
    OPT_SAB_CLIP_FLAT_NORM     clip.total_sumsq   clip_spread_j3 /
                                                  clip_j1
    OPT_SAB_CLIP_PARAM_ORDER   clip.total_sumsq   clip_j5 / clip_j2
    OPT_SAB_CLIP_SERIAL_FOLD   clip.sumsq         clip_ragged_j3 /
                                                  clip_small_j3
    OPT_SAB_CLIP_BLOCK_PARTITION clip.sumsq       clip_ragged_j3 /
                                                  clip_small_j3
    OPT_SAB_SCALARS_PER_ELEMENT (none, EXPECTED   adam_j5_t8_noclip
                                INERT -- a REACH   (reported, never
                                probe)             counted as a pass)
    OPT_SAB_MOMENTUM_FIRST_STEP sgd.buf           sgd_damp_t1 /
                                                  sgd_nodamp_t1
    OPT_SAB_NESTEROV_ORDER     sgd.dir            sgd_nesterov_t2 /
                                                  sgd_nesterov_t1
    GATE_RESUME_REINIT         sgd.buf            sgd_damp_t3 /
                                                  sgd_mom0_t3
    GATE_MICROBATCH_SERIAL     (host fold)        A = 4 / A = 2

**THE LAST ROW OF CONTRACT SECTION 12 IS DELIBERATELY A NON-BITER AND THIS
FILE KEEPS IT THAT WAY.** `OPT_SAB_SCALARS_PER_ELEMENT` recomputes the host
scalars inside the step through the SAME pinned primitives, so the bits do
not move. Contract section 12 labels it "a REACH probe, not a bit probe --
its bit result must be reported as INERT and not as a pass", and
`clause_g` treats it exactly that way: it REQUIRES the arm to move nothing
and prints that the arm proved reachability and not arithmetic. An arm
whose predicted result is "no bits move" is worth having ONLY when it is
labelled that way in advance, which is what the row does.

TWO ARMS THIS FILE COULD **NOT** BUILD A SEPARATING FIXTURE FOR
-----------------------------------------------------------------
* **`OPT_SAB_RECIP_MUL`.** Contract 4c: `x * (1/d)` is EXACT when `d` is a
  power of two, so an inert half needs the Adam denominator
  `ftz(ftz(identical_div(ftz(identical_sqrt(v)), rt_bc2)) + eps)` to come
  out an exact power of two. Every term is data dependent -- `v` is running
  state, `rt_bc2` is `sqrt(1 - beta2^t)`, `eps` is `1e-8` -- and there is
  no assignment of the fixture's free variables that makes the SUM a power
  of two except by search. The arm therefore has a witness and NO inert
  half, it is a SMOKE TEST rather than a reach proof, and the verdict says
  the word.
* **`OPT_SAB_MHAT_FORM`.** Contract section 12 says it about itself:
  "nothing known to make it inert, but it is a 1-ulp-class arm -- report
  the cell count, do not assume". The verdict reports the count.

FOUR FINDINGS ABOUT FILES THIS LANE MAY NOT EDIT
--------------------------------------------------
1. **`adam.gwd` AND `adamw.pdec` HAVE NO PRODUCER.** DEVIATION 1477.
   Contract section 10 lists both as card stages. `AdamElement` carries
   `p`, `m`, `v`, `denom` and `q`; the decayed gradient and the decayed
   parameter are LOCALS inside `adam_element_oracle` and
   `adam_update_kernel`. So a card written to the contract is two records
   short of the contract, and the one seam where Adam and AdamW differ
   (contract 7.4, "the entire difference between the two algorithms") has
   no stage of its own -- it is visible only after it has propagated into
   `adam.m` or `param.out`.
2. **THREE STAGES NEED A `-D` NOBODY IS OBLIGED TO PASS.** `adam.denom`,
   `adam.q` and `sgd.dir` are written only when
   `MOJOLEARN_OPT_RECORD` is defined (`OPT_RECORD_INTERMEDIATES`). A build
   without it gates 20 stages instead of 23, and TWO ARMS lose their
   first-stage address entirely: `OPT_SAB_EPS_INSIDE_SQRT` and
   `OPT_SAB_MHAT_FORM` both write `adam.denom` first. This file REFUSES to
   evaluate those two arms without the define rather than quietly reporting
   them at `param.out`.
3. **THE DEVICE'S REFUSAL IS WEAKER THAN CONTRACT 8a.** DEVIATION 1478.
   `identical_clip_grad_norm` refuses only the SCALAR `clip.total_norm`,
   and it does not run at all when clipping is off;
   `optimizer_step_oracle` refuses all four buffers by name.
   `identical_optimizer_step` has no refusal of its own. `optimizer.mojo`'s
   own docstring says this ("the device path is therefore weaker than the
   contract and the gap is a device-side refusal pass, owed") and clause
   (f) MEASURES it instead of repeating it.
4. **THE GEOMETRY SWEEP OF CONTRACT 11(b) IS NOT REACHABLE FROM HERE.**
   DEVIATION 1479. `OPT_TPB` is a `comptime` literal 256 in
   `optimizer.mojo`, so "the same bits across a sweep of grid and block
   geometries" needs one BUILD per geometry, not one run. Clause (b) does
   the eight repeated launches and says out loud that the geometry half is
   unrun. Contract 16.6 already owes a `kernel_matrix.mojo` row for
   `OPT_TPB`; until it exists the literal 256 is also REFUSED on the
   portable-floor column, whose cap is 128.

WHAT WOULD MAKE EACH CLAUSE PASS WHILE GATING NOTHING
-------------------------------------------------------
* **(a)** One step. See DEVIATION 1474 above: three arms are invisible on
  the step that creates the state they change.
* **(b)** Re-calling one set of buffers, and a deterministic wrongness
  passes regardless. Clause (b) is an invariance claim.
* **(c)** Running it WITH CLIPPING ON. Then one parameter's update depends
  on every other parameter in the model BY THE REFERENCE'S OWN SEMANTICS
  (contract 3.5), the clause is false, and a gate that passed it would be
  measuring something else. This file REFUSES a clipping-on case for clause
  (c) by name.
* **(d)** A "resume" that never actually reconstructed anything -- the same
  process, the same buffers, no round trip. The control is a resume with
  the momentum-initialized flag DROPPED, which must DIFFER.
* **(e)** Reporting only the aligned `A` values. An alignment predicate
  that returned True for everything would pass. The clause reports the
  MISALIGNED ones too and requires them to move.
* **(f)** An unconditional refusal. The control is a clean call that must
  not raise.
* **(g)** A misspelled `-D`, silently ignored, reported as "the arm did not
  bite". `MOJOLEARN_OPT_EXPECT_SABOTAGE` closes it.

RUNNING IT
-----------
    MOJOLEARN_IDENTITY_TRACE=/tmp/opt.card \\
    tools/with_identical_mode.sh pixi run mojo run -I . \\
        -D MOJOLEARN_OPT_RECORD=1 \\
        training/checks/optimizer_check.mojo

and one arm at a time, each of which MUST fail:

    MOJOLEARN_OPT_EXPECT_SABOTAGE=POW_RUNNING \\
    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 \\
        -D MOJOLEARN_OPT_RECORD=1 \\
        -D MOJOLEARN_OPT_SABOTAGE_POW_RUNNING=1 \\
        -I . training/checks/optimizer_check.mojo

and the two GATE arms, which have no `-D` at all:

    MOJOLEARN_OPT_GATE_ARM=GATE_RESUME_REINIT ... (as above, no sabotage -D)

ENVIRONMENT
------------
    MOJOLEARN_IDENTITY_TRACE          where the card goes
    MOJOLEARN_OPT_EXPECT_SABOTAGE     guard against a misspelled -D
    MOJOLEARN_OPT_GATE_ARM            the two gate-local arms (1473)
    MOJOLEARN_OPT_CHECK_T1000         add the thousand-step case
    MOJOLEARN_OPT_CHECK_CLAUSE_B      run clause (b)
    MOJOLEARN_OPT_CHECK_CLAUSE_C      run clause (c)
    MOJOLEARN_OPT_CHECK_CLAUSE_D      run clause (d)
    MOJOLEARN_OPT_CHECK_CLAUSE_F      run clause (f)
    MOJOLEARN_OPT_CHECK_C_CASE        clause (c)'s case
    MOJOLEARN_OPT_CHECK_D_CASE        clause (d)'s case

Clause (e) is ALWAYS ON: it is host-only, it costs nothing, and it is the
one clause whose answer contract 9.3 says the existing measurement CANNOT
supply.
"""

from std.memory import bitcast
from std.os import getenv

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace, read_trace_lines
from mamba.checks.mamba_fixture import corpus_splitmix64
from checks.numerics import ftz, identical_mul, identical_sqrt
from gemm.checks.gemm_identical import gemm_sabotage_name
from gemm.checks.gemm_backward import gemm_backward_sabotage_name
from gemm.checks.gemm_oracle import OP_NT, contract_leaf_size, gemm_oracle
from transformer.checks.transformer_fixture import fixture_splitmix64
from training.checks.loss_fixture import ce_splitmix64
from training.checks.optimizer import (
    OPT_RECORD_INTERMEDIATES,
    device_step_scalars,
    identical_optimizer_step,
    identical_optimizer_workspace_floats,
    optimizer_sabotage_name,
    SAB_CHUNKS,
)
from training.checks.optimizer_fixture import (
    BITS_POISON,
    BITS_POS_INF,
    BITS_QNAN,
    OPT_CASE_COUNT,
    OPT_CASE_T1000,
    OPT_SGD_CASE_COUNT,
    OptCase,
    assert_binade_spread,
    assert_no_zero_or_subnormal,
    bits32_hex,
    bits_of,
    count_poison,
    describe_case,
    f32_from_bits,
    is_nonfinite_bits,
    mode_is_identical,
    mode_name,
    opt_buf_initialized,
    opt_case,
    opt_case_by_name,
    opt_case_grad,
    opt_case_m,
    opt_case_param,
    opt_case_v,
    opt_config,
    opt_offsets,
    opt_poison,
    opt_profile_constants_are_intact,
    opt_sgd_case,
    opt_sgd_case_by_name,
    opt_splitmix64,
    opt_total,
)
from training.checks.optimizer_oracle import (
    OPT_ADAM,
    OPT_ADAMW,
    OPT_SGD,
    OptimizerConfig,
    OptimizerStages,
    microbatch_split_is_identical,
    optimizer_step_oracle,
    refuse_nonfinite,
    sched_field_name,
    step_scalars,
)


# Card output follows the caller-selected runtime path; standalone runs use
# the fallback below.

comptime TRACE_PATH = "/tmp/mojolearn_optimizer_step.trace"


def card_path() -> String:
    var p = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if p.byte_length() > 0:
        return p^
    return String(TRACE_PATH)


comptime TAG_PREFIX = "opt"

#: Contract 11(b)'s launch count.
comptime CLAUSE_B_LAUNCHES = 8


def env_on(name: String) -> Bool:
    return String(getenv(name)) != ""


def env_str(name: String) -> String:
    return String(getenv(name))


# ===========================================================================
# THE STAGE TABLE (contract section 10)
#
# **THE PER-TENSOR TAGS ARE FLATTENED AND THAT IS A DELIBERATE DEPARTURE.**
# DEVIATION 1471. Contract section 10 writes `input.param.<j>`,
# `clip.grad.<j>`, `adam.m.<j>` and so on -- one tag per parameter tensor.
# This file emits ONE tag per stage over the flat buffer instead, and the
# reason is clause (c): a card whose RECORD COUNT depends on `J` cannot be
# diffed against a card at a different `J`, and comparing a run at `J = 5`
# against the same tensor stepped alone at `J = 1` is exactly what contract
# 11(c) asks for. `tools/identity_trace_diff.py` aligns two traces by their
# TAG SEQUENCES before it compares a single hash, so two cards of different
# lengths do not produce a smaller diff -- they produce a WRONG ALIGNMENT
# that pairs one run's stage against another run's different stage and
# reports a plausible answer.
#
# The cost is stated: a divergence localizes to a STAGE and not to a
# TENSOR. `offsets` is in the printed report so a reader can convert a cell
# index into a `param_id` by hand, and the per-tensor `clip.sumsq` and
# `clip.norm` stages ARE per-tensor (they are `[J]` buffers), so the clip's
# own tensor-level divergences localize anyway.
# ===========================================================================

comptime OPT_STAGE_COUNT = 23

comptime OS_STEP_T = 0
comptime OS_IN_PARAM = 1
comptime OS_IN_GRAD = 2
comptime OS_CLIP_SUMSQ = 3
comptime OS_CLIP_NORM = 4
comptime OS_CLIP_TOTAL_SUMSQ = 5
comptime OS_CLIP_TOTAL_NORM = 6
comptime OS_CLIP_COEF = 7
comptime OS_CLIP_GRAD = 8
comptime OS_SCHED_POW1 = 9
comptime OS_SCHED_POW2 = 10
comptime OS_SCHED_BC1 = 11
comptime OS_SCHED_BC2 = 12
comptime OS_SCHED_STEP_SIZE = 13
comptime OS_SCHED_RT_BC2 = 14
comptime OS_SCHED_DECAY_MUL = 15
comptime OS_ADAM_M = 16
comptime OS_ADAM_V = 17
comptime OS_ADAM_DENOM = 18
comptime OS_ADAM_Q = 19
comptime OS_SGD_BUF = 20
comptime OS_SGD_DIR = 21
comptime OS_PARAM_OUT = 22


def opt_stage_tag(i: Int) raises -> String:
    if i == OS_STEP_T:
        return String("step.t")
    if i == OS_IN_PARAM:
        return String("input.param")
    if i == OS_IN_GRAD:
        return String("input.grad")
    if i == OS_CLIP_SUMSQ:
        return String("clip.sumsq")
    if i == OS_CLIP_NORM:
        return String("clip.norm")
    if i == OS_CLIP_TOTAL_SUMSQ:
        return String("clip.total_sumsq")
    if i == OS_CLIP_TOTAL_NORM:
        return String("clip.total_norm")
    if i == OS_CLIP_COEF:
        return String("clip.coef")
    if i == OS_CLIP_GRAD:
        return String("clip.grad")
    if i >= OS_SCHED_POW1 and i <= OS_SCHED_DECAY_MUL:
        return sched_field_name(i - OS_SCHED_POW1)
    if i == OS_ADAM_M:
        return String("adam.m")
    if i == OS_ADAM_V:
        return String("adam.v")
    if i == OS_ADAM_DENOM:
        return String("adam.denom")
    if i == OS_ADAM_Q:
        return String("adam.q")
    if i == OS_SGD_BUF:
        return String("sgd.buf")
    if i == OS_SGD_DIR:
        return String("sgd.dir")
    if i == OS_PARAM_OUT:
        return String("param.out")
    raise Error(
        String("optimizer_check: stage ")
        + String(i)
        + " is not one of contract section 10's "
        + String(OPT_STAGE_COUNT)
    )

def stage_index_of(tag: String) raises -> Int:
    for i in range(OPT_STAGE_COUNT):
        if opt_stage_tag(i) == tag:
            return i
    raise Error(
        String("optimizer_check: '")
        + tag
        + "' is not a stage this file knows. NOTE: contract section 10 also"
        + " lists `adam.gwd` and `adamw.pdec`, and NEITHER"
        + " optimizer_oracle.mojo NOR optimizer.mojo produces either"
        + " (DEVIATION 1477), so they are not here."
    )


def stage_is_int(i: Int) -> Bool:
    """`step.t` is an INTEGER and is compared as one. Integers do not flush
    and do not round, so it has no seam -- and pushing it through the float
    path would make a step index that happened to hold a NaN bit pattern
    'non-finite' to every audit in this file."""
    return i == OS_STEP_T


def stage_is_clip(i: Int) -> Bool:
    return i >= OS_CLIP_SUMSQ and i <= OS_CLIP_GRAD


def stage_is_adam(i: Int) -> Bool:
    return i >= OS_ADAM_M and i <= OS_ADAM_Q


def stage_is_sgd(i: Int) -> Bool:
    return i == OS_SGD_BUF or i == OS_SGD_DIR


def stage_needs_record(i: Int) -> Bool:
    """`adam.denom`, `adam.q` and `sgd.dir` are written ONLY when the build
    defines `MOJOLEARN_OPT_RECORD`.

    DEVIATION 1476. `adam_update_kernel` and `sgd_update_kernel` carry the
    pointers in their signatures either way -- which is the right design,
    one kernel with one signature -- and store into them under a `comptime
    if`. So a build without the define leaves those buffers UNWRITTEN, and
    a gate that recorded them anyway would be hashing the allocator's
    leftovers and calling it a card. Every output buffer here is poisoned,
    so that is a measurement rather than a hope."""
    return i == OS_ADAM_DENOM or i == OS_ADAM_Q or i == OS_SGD_DIR


def stage_present(i: Int, cfg: OptimizerConfig) -> Bool:
    """Does this configuration produce this stage at all?

    The `clip.*` stages are ABSENT when clipping is off, not filled with a
    coefficient of `1.0`. `optimizer_step_oracle` is explicit about it --
    "an empty stage and a stage that says 'no clipping happened' are
    different claims and a card should make the difference visible" -- and
    a gate that zero-filled them would be asserting the second claim while
    the profile makes the first."""
    if stage_needs_record(i) and not OPT_RECORD_INTERMEDIATES:
        return False
    if stage_is_clip(i) and cfg.max_norm <= Float32(0.0):
        return False
    if stage_is_adam(i) and cfg.kind == OPT_SGD:
        return False
    if stage_is_sgd(i) and cfg.kind != OPT_SGD:
        return False
    if i == OS_SCHED_DECAY_MUL and cfg.kind != OPT_ADAMW:
        # Contract section 10 marks `sched.decay_mul` "AdamW only".
        # `step_scalars` computes it unconditionally -- two host flops and
        # one fewer branch -- so the VALUE exists for Adam and SGD too and
        # it is simply not a stage there.
        return False
    return True


def expected_card_records(cfg: OptimizerConfig) -> Int:
    var n = 0
    for i in range(OPT_STAGE_COUNT):
        if stage_present(i, cfg):
            n += 1
    return n


# Device transfer helpers for caller-owned optimizer buffers.


def _upload_f32(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
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


def _download_f32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    """`[[mojo-buffer-freed-at-last-use]]`: `buf` is a `mut` REFERENCE, so
    it is the CALLER's owner that must stay alive. Every caller below keeps
    its buffers in locals across the whole step loop, which is also what
    makes the state a RUNNING state rather than a fresh one per step."""
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


def _overwrite_f32(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32],
    values: List[Float32],
) raises:
    """Replace a device buffer's contents in place, so that a buffer whose
    IDENTITY matters (the running `param`, `m` and `v`) survives a step
    while its CONTENTS are re-seeded. Used by clause (d)'s resume, where
    the whole point is that the state came from somewhere else."""
    var n = len(values)
    if n <= 0:
        return
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    var view = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_buf=view, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    _ = view


def nonfinite_cells(values: List[Float32]) -> Int:
    """NaN or infinity, BY BITS. NOT BY COMPARES: Metal flushes compare
    operands (row 49), so `v != v` has two meanings across columns while a
    mask on the exponent field has one. Contract 8a."""
    var n = 0
    for i in range(len(values)):
        if is_nonfinite_bits(values[i]):
            n += 1
    return n


# ===========================================================================
# ONE STEP, BOTH HALVES
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
    shorter of the two, because a stage of the wrong SIZE is a different
    defect from a stage of the wrong VALUE.

    BY BITS AND NEVER BY COMPARES. `host[i] == dev[i]` would call `+0.0`
    and `-0.0` equal, and contract 8b makes signed zero an ADMITTED value
    at several named sites here -- `identical_mul`'s `-0.0` addend keeps a
    negative-zero product, and contract 7's opening paragraph says the
    `-0.0` addend is LOAD BEARING and a `+0.0` would be a bug. A
    compare-written gate cannot see the difference it exists to protect."""
    if len(host) != len(dev):
        raise Error(
            String("optimizer_check: stage ")
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


def host_dump(
    stages: OptimizerStages,
    param_in: List[Float32],
    grad_in: List[Float32],
    cfg: OptimizerConfig,
) raises -> List[List[Float32]]:
    """The ORACLE's stages for one step, card order, index-aligned with
    `opt_stage_tag`."""
    var out = List[List[Float32]]()
    for _i in range(OPT_STAGE_COUNT):
        out.append(List[Float32]())
    out[OS_IN_PARAM] = param_in.copy()
    out[OS_IN_GRAD] = grad_in.copy()
    if cfg.max_norm > Float32(0.0):
        out[OS_CLIP_SUMSQ] = stages.clip_sumsq.copy()
        out[OS_CLIP_NORM] = stages.clip_norm.copy()
        var ts = List[Float32]()
        var tn = List[Float32]()
        var co = List[Float32]()
        if len(stages.clip_total) == 3:
            ts.append(stages.clip_total[0])
            tn.append(stages.clip_total[1])
            co.append(stages.clip_total[2])
        out[OS_CLIP_TOTAL_SUMSQ] = ts^
        out[OS_CLIP_TOTAL_NORM] = tn^
        out[OS_CLIP_COEF] = co^
        out[OS_CLIP_GRAD] = stages.clip_grad.copy()
    for f in range(7):
        var one = List[Float32]()
        one.append(stages.sched[f])
        out[OS_SCHED_POW1 + f] = one^
    out[OS_ADAM_M] = stages.adam_m.copy()
    out[OS_ADAM_V] = stages.adam_v.copy()
    out[OS_ADAM_DENOM] = stages.adam_denom.copy()
    out[OS_ADAM_Q] = stages.adam_q.copy()
    out[OS_SGD_BUF] = stages.sgd_buf.copy()
    out[OS_SGD_DIR] = stages.sgd_dir.copy()
    out[OS_PARAM_OUT] = stages.param_out.copy()
    return out^


struct DeviceOptState(Movable):
    """Every device buffer of a running optimizer, held across the STEP
    LOOP.

    **HELD, NOT REBUILT.** `param`, `m_state` and `v_state` are the running
    state, and a fresh buffer per step would make an eight-step run into
    eight first steps -- which is exactly the shape that hides
    `OPT_SAB_MOMENT_LERP` (inert at `m_prev == +0.0`),
    `OPT_SAB_NESTEROV_ORDER` (inert at `t = 1`, where `b == g`) and
    `OPT_SAB_POW_RUNNING` (inert at `t <= 6`). DEVIATION 1474.

    `[[mojo-buffer-freed-at-last-use]]`: these are struct FIELDS, so they
    are alive as long as the struct is, which is the whole loop. A
    `DeviceBuffer` is dead at its `.unsafe_ptr()` when it is a local whose
    last use that was, and this repository has lost a night to that."""

    var param: DeviceBuffer[DType.float32]
    var m_state: DeviceBuffer[DType.float32]
    var v_state: DeviceBuffer[DType.float32]
    var denom_out: DeviceBuffer[DType.float32]
    var q_out: DeviceBuffer[DType.float32]
    var sumsq: DeviceBuffer[DType.float32]
    var norms: DeviceBuffer[DType.float32]
    var total_cell: DeviceBuffer[DType.float32]
    var out2: DeviceBuffer[DType.float32]
    var ws: DeviceBuffer[DType.float32]
    var sab_partials: DeviceBuffer[DType.float32]
    var buf_initialized: List[Bool]

    def __init__(
        out self,
        ctx: DeviceContext,
        c: OptCase,
        param: List[Float32],
        m0: List[Float32],
        v0: List[Float32],
    ) raises:
        var off = opt_offsets(c.shape)
        var j = len(off) - 1
        var n = off[j]
        self.param = _upload_f32(ctx, param)
        self.m_state = _upload_f32(ctx, m0)
        self.v_state = _upload_f32(ctx, v0)
        self.denom_out = _upload_f32(ctx, opt_poison(n))
        self.q_out = _upload_f32(ctx, opt_poison(n))
        self.sumsq = _upload_f32(ctx, opt_poison(j))
        self.norms = _upload_f32(ctx, opt_poison(j))
        self.total_cell = _upload_f32(ctx, opt_poison(1))
        self.out2 = _upload_f32(ctx, opt_poison(2))
        self.ws = _upload_f32(
            ctx, opt_poison(identical_optimizer_workspace_floats(off))
        )
        self.sab_partials = _upload_f32(ctx, opt_poison(SAB_CHUNKS))
        self.buf_initialized = opt_buf_initialized(c)


def device_step(
    ctx: DeviceContext,
    mut st: DeviceOptState,
    c: OptCase,
    cfg: OptimizerConfig,
    grad: List[Float32],
    t: Int,
) raises -> List[List[Float32]]:
    """ONE device step, stages returned on the host, card order.

    The intermediate buffers are RE-POISONED before the step (DEVIATION
    1476), so a cell that comes back holding `0x7fc0dead` is a cell NO
    KERNEL WROTE at THIS step -- not one left over from the last one. That
    distinction is the whole reason the poison is refreshed rather than
    written once.

    `input.param` and `input.grad` are DOWNLOADED FROM THE DEVICE before
    the step rather than taken from the host lists that were uploaded. That
    is `[[verify-reach-not-output]]` at the cheapest possible site: it costs
    one copy and it turns "we uploaded the right numbers" from an assumption
    into a measurement. A gate that reported its own input back to itself
    would have two stages that cannot fail."""
    var off = opt_offsets(c.shape)
    var j = len(off) - 1
    var n = off[j]

    _overwrite_f32(ctx, st.denom_out, opt_poison(n))
    _overwrite_f32(ctx, st.q_out, opt_poison(n))
    _overwrite_f32(ctx, st.sumsq, opt_poison(j))
    _overwrite_f32(ctx, st.norms, opt_poison(j))
    _overwrite_f32(ctx, st.total_cell, opt_poison(1))
    _overwrite_f32(ctx, st.out2, opt_poison(2))

    var out = List[List[Float32]]()
    for _i in range(OPT_STAGE_COUNT):
        out.append(List[Float32]())
    out[OS_IN_PARAM] = _download_f32(ctx, st.param, n)
    var d_grad = _upload_f32(ctx, grad)
    out[OS_IN_GRAD] = _download_f32(ctx, d_grad, n)

    identical_optimizer_step(
        ctx, st.param, d_grad, st.m_state, st.v_state, st.denom_out,
        st.q_out, st.sumsq, st.norms, st.total_cell, st.out2, st.ws,
        st.sab_partials, st.buf_initialized, off, cfg, t,
    )
    ctx.synchronize()

    if cfg.max_norm > Float32(0.0):
        out[OS_CLIP_SUMSQ] = _download_f32(ctx, st.sumsq, j)
        out[OS_CLIP_NORM] = _download_f32(ctx, st.norms, j)
        out[OS_CLIP_TOTAL_SUMSQ] = _download_f32(ctx, st.total_cell, 1)
        var two = _download_f32(ctx, st.out2, 2)
        var tn = List[Float32]()
        var co = List[Float32]()
        tn.append(two[0])
        co.append(two[1])
        out[OS_CLIP_TOTAL_NORM] = tn^
        out[OS_CLIP_COEF] = co^
        # The gradient buffer AFTER the step. `identical_clip_grad_norm`
        # rescales IN PLACE, so this is seam C8's output and it is the
        # stage `OPT_SAB_CLIP_SKIP_AT_ONE` must be read at -- contract
        # 3.4c: the difference is CARD VISIBLE and DOWNSTREAM INERT,
        # because O1 flushes the gradient on load, so a gate that compared
        # `param.out` would report that arm inert and the report would be
        # TRUE AND USELESS.
        out[OS_CLIP_GRAD] = _download_f32(ctx, d_grad, n)

    # ---- THE SCHED SCALARS ----------------------------------------------
    # These have no device buffer: `identical_optimizer_step` computes them
    # on the host through `device_step_scalars` and passes them into the
    # kernel as arguments. So the "device" value here is the CHECK's own
    # call of the same function -- which is NOT a tautology, because
    # `device_step_scalars` is where `SAB_POW_RUNNING` and `SAB_POW_EXPLOG`
    # live. Against the ORACLE's `step_scalars`, which carries no sabotage
    # switch of any kind, the two pow arms are visible at `sched.pow1` and
    # nowhere earlier. What this CANNOT catch is the step function passing
    # a DIFFERENT `t` than the gate does, and clause (a) compares `step.t`
    # for exactly that reason.
    var sc = device_step_scalars(cfg, t)
    var vals: List[Float32] = [
        sc.b1t, sc.b2t, sc.bc1, sc.bc2, sc.step_size, sc.rt_bc2,
        sc.decay_mul,
    ]
    for f in range(7):
        var one = List[Float32]()
        one.append(vals[f])
        out[OS_SCHED_POW1 + f] = one^

    if cfg.kind == OPT_SGD:
        out[OS_SGD_BUF] = _download_f32(ctx, st.m_state, n)
        if OPT_RECORD_INTERMEDIATES:
            out[OS_SGD_DIR] = _download_f32(ctx, st.denom_out, n)
    else:
        out[OS_ADAM_M] = _download_f32(ctx, st.m_state, n)
        out[OS_ADAM_V] = _download_f32(ctx, st.v_state, n)
        if OPT_RECORD_INTERMEDIATES:
            out[OS_ADAM_DENOM] = _download_f32(ctx, st.denom_out, n)
            out[OS_ADAM_Q] = _download_f32(ctx, st.q_out, n)
    out[OS_PARAM_OUT] = _download_f32(ctx, st.param, n)
    _ = d_grad^
    return out^


def dump_poison(dump: List[List[Float32]], cfg: OptimizerConfig) -> Int:
    """Cells NO KERNEL WROTE at this step, over every stage that must be
    written. `input.param` and `input.grad` are excluded: they are inputs
    and a kernel is not supposed to write them."""
    var n = 0
    for i in range(OPT_STAGE_COUNT):
        if i == OS_IN_PARAM or i == OS_IN_GRAD or stage_is_int(i):
            continue
        if not stage_present(i, cfg):
            continue
        n += count_poison(dump[i])
    return n


# ===========================================================================
# THE CARD
# ===========================================================================


def write_card(
    mut trace: IdentityTrace,
    prefix: String,
    cfg: OptimizerConfig,
    dump: List[List[Float32]],
    t: Int,
) raises -> Int:
    """Contract section 10's stages onto the card, in section 10's order,
    each exactly once, each carrying this driver's prefix.

    **THE ABSENT STAGES ARE ABSENT AND ARE NOT WRITTEN AS ZEROS.** A run
    with clipping off emits no `clip.*` record at all, because
    `optimizer_step_oracle` is explicit that "an empty stage and a stage
    that says 'no clipping happened' are different claims and a card should
    make the difference visible".

    `step.t` goes through `record_list_i32`. The dtype is part of the
    record (`_dtype_name`), so a card that recorded the step index as a
    float would not even ALIGN against one that recorded it as an integer,
    which is the failure being loud instead of silent."""
    var emitted = 0
    for i in range(OPT_STAGE_COUNT):
        if not stage_present(i, cfg):
            continue
        var tag = prefix + "." + opt_stage_tag(i)
        if i == OS_STEP_T:
            var tl = List[Int32]()
            tl.append(Int32(t))
            trace.record_list_i32(tag, tl)
            emitted += 1
            continue
        trace.record_list_f32(tag, dump[i])
        emitted += 1
    return emitted


def check_card_tags(path: String, cfg: OptimizerConfig) raises -> Int:
    """The card holds this configuration's stages, in contract section 10's
    order, each exactly once, each carrying this driver's prefix.

    THIS IS A CHECK ON THE COMPOSITION AND NOT ON THE ARITHMETIC.
    `IdentityTrace` enforces tag UNIQUENESS and raises, which catches two
    writers claiming one tag; nothing but this function catches a MISSING
    tag, because a card with twenty records is a card, and
    `tools/identity_trace_diff.py` would align it against a twenty-three
    record card and report a plausible WRONG ANSWER.

    AND IT IS THE CHECK THAT CATCHES A CARD THAT WAS NEVER WRITTEN, which
    was DEVIATION 970's actual damage."""
    var want = expected_card_records(cfg)
    var lines = read_trace_lines(path)
    print(
        "card: "
        + String(len(lines))
        + " records at "
        + path
        + ", this configuration wants "
        + String(want)
        + " of contract section 10's "
        + String(OPT_STAGE_COUNT)
        + " (and section 10 also lists adam.gwd and adamw.pdec, which have"
        + " NO PRODUCER -- DEVIATION 1477)"
    )
    if len(lines) != want:
        raise Error(
            String("optimizer_check: the card has ")
            + String(len(lines))
            + " records and this configuration produces "
            + String(want)
        )
    var k = 0
    for i in range(OPT_STAGE_COUNT):
        if not stage_present(i, cfg):
            continue
        var fields = lines[k].split("\t")
        if len(fields) < 2:
            raise Error(
                String("optimizer_check: malformed trace record: ")
                + lines[k]
            )
        var got = String(fields[1])
        var expect = String(TAG_PREFIX) + "." + opt_stage_tag(i)
        if got != expect:
            raise Error(
                String("optimizer_check: card record ")
                + String(k)
                + " is '"
                + got
                + "', contract section 10 wants '"
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
        + " tags in contract section 10's order, all unique"
    )
    return len(lines)


# ===========================================================================
# PREFLIGHT
# ===========================================================================


def preflight() raises:
    print("preflight: the assertions the contract and the fixture asked for")

    if not opt_profile_constants_are_intact():
        raise Error(
            String("optimizer_check: a contract section 1 constant has the")
            + " wrong bits. `[[mojo-string-float-roundtrip]]`: a constant"
            + " that was right when it was typed is not the same thing as a"
            + " constant the toolchain agrees with, and contract section 1"
            + " makes every hyperparameter a BIT PATTERN for exactly that"
            + " reason."
        )
    var cfg = opt_config(0)
    print(
        "  contract section 1 constants OK: lr "
        + bits32_hex(cfg.lr)
        + " beta1 "
        + bits32_hex(cfg.beta1)
        + " beta2 "
        + bits32_hex(cfg.beta2)
        + " eps "
        + bits32_hex(cfg.eps)
    )

    # ---- CONTRACT 5.3, PRINTED BACK ------------------------------------
    # The contract's PREDICTED table, DERIVED OFF-REPOSITORY and NEVER
    # MEASURED, says `1 - 0.9` is `0x3DCCCCD0` here against PyTorch's
    # float64 route's `0x3DCCCCCD`, and `1 - 0.999` is `0x3A831200` against
    # `0x3A83126F` -- about 1.3e-5 RELATIVE, which is the THIRD SIGNIFICANT
    # DECIMAL of the coefficient that drives `v`. It is the larger of the
    # two reasons PyTorch parity is unavailable (contract 5.2 and 5.3), and
    # a reader who sees `1.0 - beta2` in the oracle will assume it is the
    # obvious thing. **This is the first time those numbers will have been
    # produced by a machine.** They are PRINTED and not asserted: the
    # contract labels them PREDICTED, and a gate that asserted a prediction
    # would convert an honest guess into a false measurement.
    var sc1 = step_scalars(cfg, 1)
    print(
        "  contract 5.3 [PREDICTED, now PRINTED, still not asserted]:"
        " c1 = ftz(1 - beta1) = "
        + bits32_hex(sc1.c1)
        + " (contract predicts 0x3dcccd0, PyTorch's float64 route"
        + " 0x3dcccccd);  c2 = ftz(1 - beta2) = "
        + bits32_hex(sc1.c2)
        + " (contract predicts 0x3a831200, PyTorch's 0x3a83126f)"
    )

    # ---- CONTRACT 5.1's t = 7 PREDICTION, PRINTED ----------------------
    # The contract PREDICTS that `pow_int_f32` and the running product
    # first differ at `t = 7` for both betas. That is the single number
    # `OPT_SAB_POW_RUNNING`'s witness/inert pair rests on, so it is
    # computed here on the host and printed. **If it is wrong, the pair is
    # pointed at the wrong steps and the arm would look broken** -- so this
    # print is the thing a reader checks first when that arm misbehaves.
    var first_diff = -1
    for t in range(1, 33):
        var pinned = step_scalars(cfg, t)
        var running = Float32(1.0)
        for _i in range(t):
            # `identical_mul`, NOT a plain `*`, because the arm this
            # predicts (`SAB_POW_RUNNING`) is spelled
            # `ftz(identical_mul(b1t, cfg.beta1))` in
            # `device_step_scalars`. A prediction spelled with a
            # different multiply is a prediction about a different
            # arm.
            running = ftz(identical_mul(running, cfg.beta1))
        if bits_of(running) != bits_of(pinned.b1t):
            first_diff = t
            break
    print(
        "  contract 5.1 [PREDICTED t = 7, now COMPUTED]: the binary"
        " exponentiation and a running product for beta1 first differ at"
        " t = "
        + String(first_diff)
        + ". OPT_SAB_POW_RUNNING's witness is adam_t7 and its inert half is"
        + " adam_t6, and BOTH are wrong if this number is not 7."
    )
    if first_diff <= 6 or first_diff < 0:
        raise Error(
            String("optimizer_check: the running product first differs from")
            + " pow_int_f32 at t = "
            + String(first_diff)
            + ", and contract 5.1 predicts 7. adam_t6 is OPT_SAB_POW_"
            + "RUNNING's INERT half and it is not inert at this number, so"
            + " the arm's pair is pointed at the wrong steps. **THE"
            + " CONTRACT'S NUMBER IS THE THING THAT IS WRONG HERE, NOT THE"
            + " GATE**, and it is a PREDICTED value that had never been"
            + " computed before this run."
        )

    # ---- THE FOUR splitmix64 COPIES AGREE ------------------------------
    # `opt_splitmix64` is the FOURTH copy in this repository. The argument
    # for copying is DEVIATION 1000's and the cost is the one that
    # docstring names, multiplied. `[[mojo-amp-plus-is-bitwise-and]]` is
    # what this actually guards: Mojo's `&+` computes `x & k` with NO
    # COMPILE ERROR, and a `+` "fixed" into a `&+` in ONE copy is exactly
    # the edit this catches. It has happened here twice.
    var seeds: List[UInt64] = [
        UInt64(0),
        UInt64(1),
        UInt64(0x9E3779B97F4A7C15),
        UInt64(0xFFFFFFFFFFFFFFFF),
        UInt64(0x4F70744D6F6A6F31),
    ]
    for i in range(len(seeds)):
        var a = opt_splitmix64(seeds[i])
        if a != corpus_splitmix64(seeds[i]):
            raise Error(
                String("optimizer_check: the optimizer fixture's")
                + " splitmix64 and the mamba corpus's disagree at seed "
                + String(i)
            )
        if a != fixture_splitmix64(seeds[i]):
            raise Error(
                String("optimizer_check: the optimizer fixture's")
                + " splitmix64 and the transformer fixture's disagree"
            )
        if a != ce_splitmix64(seeds[i]):
            raise Error(
                String("optimizer_check: the optimizer fixture's")
                + " splitmix64 and the loss fixture's disagree"
            )
    print(
        "  splitmix64: all FOUR copies (optimizer, loss, transformer, mamba"
        " corpus) agree on "
        + String(len(seeds))
        + " seeds"
    )

    # ---- THE RECORD DEFINE, and what a build without it does NOT gate --
    if OPT_RECORD_INTERMEDIATES:
        print(
            "  MOJOLEARN_OPT_RECORD is DEFINED, so all "
            + String(OPT_STAGE_COUNT)
            + " stages are reachable"
        )
    else:
        print(
            "  **MOJOLEARN_OPT_RECORD IS NOT DEFINED** (DEVIATION 1476)."
            " adam.denom, adam.q and sgd.dir are NOT WRITTEN by the"
            " kernels, so this build gates 20 stages and not 23 -- and TWO"
            " ARMS lose their first-stage address entirely"
            " (OPT_SAB_EPS_INSIDE_SQRT and OPT_SAB_MHAT_FORM both write"
            " adam.denom first). Clause (g) REFUSES to evaluate those two"
            " rather than reporting them at param.out."
        )

    # ---- THE FIXTURE'S OWN CLAIMS --------------------------------------
    var checked = 0
    for k in range(OPT_CASE_COUNT):
        if k == OPT_CASE_T1000 and not env_on("MOJOLEARN_OPT_CHECK_T1000"):
            continue
        var c = opt_case(k)
        checked += assert_no_zero_or_subnormal(c, 1)
    for k in range(OPT_SGD_CASE_COUNT):
        checked += assert_no_zero_or_subnormal(opt_sgd_case(k), 1)
    print(
        "  fixture: "
        + String(checked)
        + " gradient and parameter cells carry no signed zero, no subnormal"
        " and nothing non-finite outside the cases that PLANT them"
    )

    # `OPT_SAB_CLIP_FLAT_NORM`'s separating property, MEASURED rather than
    # argued. Contract 3.1 asks for "norms that differ by several binades"
    # and a fixture whose norms are all within a binade makes the arm inert
    # for a reason the case does not claim.
    var spread_cases: List[String] = [
        String("clip_spread_j3"), String("clip_j5"),
    ]
    for i in range(len(spread_cases)):
        var c = opt_case(opt_case_by_name(spread_cases[i]))
        var span = assert_binade_spread(c)
        print(
            "  "
            + spread_cases[i]
            + ": per-tensor gradient norms span "
            + String(span)
            + " binades (contract 3.1 wants SEVERAL, and at a narrow spread"
            + " the two-level clip norm and a flat one agree)"
        )

    # The leaf rule, at the lengths the registry uses, so that every
    # fold-arm prediction in this file is anchored to gemm_oracle's own
    # function rather than to a comment.
    var lens: List[Int] = [7, 8, 64, 128, 130, 200, 300, 1025, 4096]
    var line = String("  gemm v1 partition at this lane's tensor lengths:")
    for i in range(len(lens)):
        var kk = lens[i]
        var leaf = contract_leaf_size(kk)
        var p = (kk + leaf - 1) // leaf
        line += (
            " N=" + String(kk) + "->P=" + String(p) + ";"
        )
    print(line)
    print(
        "  (at P == 1 the balanced tree performs NO ADDITION -- gemm"
        " contract 7.3 -- which is why clip_small_j3, whose tensors are"
        " all <= 128, is OPT_SAB_CLIP_SERIAL_FOLD's INERT half)"
    )


# ===========================================================================
# THE CASE SET AND THE VERDICT
# ===========================================================================


@fieldwise_init
struct CaseVerdict(Copyable, Movable):
    """One case's clause-(a) result, aggregated ACROSS ITS STEPS.

    `moved` is a per-stage mask ORed over every step, not a count, because
    two arms in this lane are stated as "must move stage X and must NOT
    move stage Y" and a count cannot express either.

    `first_step` is the FIRST STEP at which anything moved, and it is a
    finding in its own right rather than bookkeeping: `OPT_SAB_POW_RUNNING`
    is predicted to first move at `t = 7`, `OPT_SAB_MOMENT_LERP` at `t = 2`
    (it is inert at `t = 1` where `m_prev` is `+0.0`), and
    `OPT_SAB_NESTEROV_ORDER` at `t = 2` (it is inert at `t = 1` where
    `b == g`). An arm that moved at the WRONG step is an arm that is not
    doing what its clause says, even if it moved."""

    var name: String
    var n_moved: Int
    var first_step: Int
    var first_index: Int
    var first: String
    var cells: Int
    var moved: List[Bool]
    var poison: Int
    var steps: Int


def stage_moved(v: CaseVerdict, i: Int) -> Bool:
    if i < 0 or i >= len(v.moved):
        return False
    return v.moved[i]


def find_verdict(
    verdicts: List[CaseVerdict], name: String
) raises -> CaseVerdict:
    for i in range(len(verdicts)):
        if verdicts[i].name == name:
            return verdicts[i].copy()
    raise Error(
        String("optimizer_check: the sabotage expectation names case '")
        + name
        + "' and it was not in the clause-(a) set this build ran. The arm"
        + " cannot be evaluated, which is NOT the same as the arm passing"
        + " ([[reached-but-inert]])."
    )


# ===========================================================================
# CLAUSE (a): device equals host oracle, bitwise, every stage, EVERY STEP
# ===========================================================================


def clause_a_case(
    ctx: DeviceContext,
    c: OptCase,
    mut trace: IdentityTrace,
    prefix: String,
    write_the_card: Bool,
) raises -> CaseVerdict:
    """Contract 11(a) at one fixture case, across all of its steps.

    THE TWO HALVES ARE STEPPED IN LOCKSTEP FROM THE SAME FRESH STATE and
    each one carries its OWN state forward. That is what makes an
    eight-step run eight DIFFERENT steps rather than eight first steps, and
    DEVIATION 1474 is the reason it matters.

    WHAT WOULD MAKE THIS PASS WHILE GATING NOTHING. Four things, three of
    them closed here:

      1. One step. Closed: the case carries a step count and the comparison
         runs at every one.
      2. A device dump and an oracle dump that are the same object. They
         are not: one reads `OptimizerStages` fields, the other reads
         device buffers.
      3. Two dumps of different LENGTHS compared to the shorter of the two.
         `compare_stage` raises on a per-stage length mismatch.
      4. **Our oracle being wrong in the same way as our device.** NOT
         CLOSED AND NOT CLOSEABLE HERE, and in this lane the two halves are
         unusually close: `identical_optimizer_step` calls
         `device_step_scalars`, which calls the ORACLE's own `step_scalars`
         on the clean path, so seven of the twenty-three stages are the
         same host code on both sides. Only an independent reference could
         see it, and contract 16.10 says the corpus does not exist. What
         DOES separate them there is the two pow arms, which live in
         `device_step_scalars` and not in `step_scalars` -- so those seven
         stages are gated against a SABOTAGE and not against a second
         implementation, which is a weaker thing and is said out loud."""
    var cfg = opt_config(c.hp)
    var off = opt_offsets(c.shape)
    var n = off[len(off) - 1]

    var h_param = opt_case_param(c)
    var h_m = opt_case_m(c)
    var h_v = opt_case_v(c)
    var h_flags = opt_buf_initialized(c)

    var st = DeviceOptState(ctx, c, h_param, h_m, h_v)

    var moved = List[Bool]()
    for _i in range(OPT_STAGE_COUNT):
        moved.append(False)
    var n_moved = 0
    var first_step = -1
    var first_index = -1
    var first_name = String("")
    var cells = 0
    var poison = 0

    for t in range(1, c.steps + 1):
        var grad = opt_case_grad(c, t)
        var param_in = h_param.copy()
        var grad_in = grad.copy()

        # The ORACLE. It MUTATES `h_param`, `grad`, `h_m`, `h_v` and
        # `h_flags` in place, which is how the state moves forward.
        var stages = optimizer_step_oracle(
            h_param, grad, h_m, h_v, h_flags, off, cfg, t
        )
        var host = host_dump(stages, param_in, grad_in, cfg)
        var dev = device_step(ctx, st, c, cfg, grad_in, t)
        poison += dump_poison(dev, cfg)

        # `step.t` first. It is an INTEGER stage and it is the only thing
        # that can catch the two halves disagreeing about WHICH step this
        # is -- which would make every `sched.*` comparison agree for the
        # wrong reason, since both sides derive those from `t`.
        for i in range(OPT_STAGE_COUNT):
            if stage_is_int(i) or not stage_present(i, cfg):
                continue
            var d = compare_stage(
                opt_stage_tag(i) + " step " + String(t),
                host[i],
                dev[i],
                False,
            )
            cells += d.n_cells
            if d.n_diff > 0:
                if not moved[i]:
                    moved[i] = True
                    n_moved += 1
                if first_step < 0:
                    first_step = t
                    first_index = i
                    first_name = (
                        opt_stage_tag(i)
                        + " on "
                        + String(d.n_diff)
                        + " of "
                        + String(d.n_cells)
                        + " cells"
                    )
        if write_the_card and t == 1:
            _ = write_card(trace, prefix, cfg, dev, t)
        _ = stages^

    if n_moved == 0:
        print(
            "  "
            + describe_case(c)
            + ": "
            + String(expected_card_records(cfg))
            + " stages bit-identical across "
            + String(c.steps)
            + " steps ("
            + String(cells)
            + " cells)"
        )
    else:
        print(
            "  "
            + describe_case(c)
            + ": "
            + String(n_moved)
            + " stages MOVED, first at step "
            + String(first_step)
            + ", "
            + first_name
        )
    _ = st^
    return CaseVerdict(
        String(c.name), n_moved, first_step, first_index, first_name,
        cells, moved^, poison, c.steps,
    )


def clause_a_cases() raises -> List[Int]:
    """The default Adam-family clause-(a) set: every case except the
    thousand-step one.

    `adam_t1000` joins on `MOJOLEARN_OPT_CHECK_T1000`. Contract 5.1 says
    the gate "must run to at least `t = 8` and should run to `t = 1000`":
    `adam_t7` and `adam_t8` close the `t <= 6` vacuity, and what `t = 1000`
    adds is the DRIFT, which grows with `t`, plus the approach to the
    PREDICTED flush of `beta1^t` at `t = 829`. It is off by default for
    `[[no-heavy-local-compute]]`'s reason and the SCOPE line names it."""
    var out = List[Int]()
    for k in range(OPT_CASE_COUNT):
        if k == OPT_CASE_T1000:
            continue
        out.append(k)
    if env_on("MOJOLEARN_OPT_CHECK_T1000"):
        out.append(OPT_CASE_T1000)
    return out^


# ===========================================================================
# CLAUSE (b): the same bits on eight repeated launches
# ===========================================================================


def clause_b(ctx: DeviceContext, k: Int) raises:
    """Contract 11(b), the LAUNCH half.

    Eight whole runs of the same case, each from its own fresh device
    state, every stage compared to the first on every cell of every step.

    For the ELEMENTWISE update this is structural rather than empirical:
    `adam_update_kernel` reads element `i` and writes element `i`, with no
    shared memory, no `barrier`, no atomic, no cross-lane primitive and no
    reduction, so `block_dim` and `grid_dim` decide WHICH THREAD does
    element `i` and cannot decide what element `i` is. The gate verifies
    what the construction promises rather than being the only thing holding
    the claim up. For the CLIP's reductions it is inherited from the v1
    GEMM, whose own `check_device_is_launch_invariant` covers it.

    **THE GEOMETRY HALF OF CONTRACT 11(b) IS NOT RUN AND CANNOT BE RUN FROM
    HERE.** DEVIATION 1479. `OPT_TPB` is a `comptime` literal 256 in
    `optimizer.mojo`, so "the same bits across a sweep of grid and block
    geometries" needs one BUILD per geometry rather than one run. Contract
    16.6 already owes a `kernel_matrix.mojo` row for it -- and notes that
    `column_max_block_size(COLUMN_SPEC_BASELINE)` is 128, so the literal
    256 would be REFUSED on the portable-floor column rather than resolved
    down to it. Until that row exists, clause (b) here is the LAUNCH half
    and only the launch half, and this print says so rather than letting
    "clause (b) PASS" stand for both.

    AND THE HOLE NO CONTROL CLOSES: eight identical runs of a
    deterministically wrong optimizer pass. Clause (b) is an invariance
    claim and says nothing about correctness."""
    var c = opt_case(k)
    var cfg = opt_config(c.hp)
    var off = opt_offsets(c.shape)
    var n = off[len(off) - 1]
    print(
        "clause (b): "
        + String(CLAUSE_B_LAUNCHES)
        + " repeated runs of "
        + String(c.name)
        + " ("
        + String(c.steps)
        + " steps each), every stage, every cell, fresh device state each"
        " time"
    )
    var base = List[List[Float32]]()
    var cells = 0
    for run in range(1, CLAUSE_B_LAUNCHES + 1):
        var st = DeviceOptState(
            ctx, c, opt_case_param(c), opt_case_m(c), opt_case_v(c)
        )
        var last = List[List[Float32]]()
        for t in range(1, c.steps + 1):
            last = device_step(ctx, st, c, cfg, opt_case_grad(c, t), t)
        if run == 1:
            base = last^
            for i in range(OPT_STAGE_COUNT):
                cells += len(base[i])
        else:
            for i in range(OPT_STAGE_COUNT):
                if stage_is_int(i) or not stage_present(i, cfg):
                    continue
                var d = compare_stage(
                    opt_stage_tag(i), base[i], last[i], False
                )
                if d.n_diff > 0:
                    raise Error(
                        String("optimizer_check: CLAUSE (b) FAILED, run ")
                        + String(run)
                        + " differs from run 1 at "
                        + opt_stage_tag(i)
                        + " on "
                        + String(d.n_diff)
                        + " of "
                        + String(d.n_cells)
                        + " cells"
                    )
        _ = st^
    print(
        "clause (b): PASS on the LAUNCH half -- runs 2.."
        + String(CLAUSE_B_LAUNCHES)
        + " bit-identical to run 1 on all "
        + String(cells)
        + " cells. **THE GEOMETRY HALF IS NOT RUN**: OPT_TPB is a comptime"
        " literal, so a geometry sweep is one BUILD per geometry"
        " (DEVIATION 1479, contract 16.6)."
    )


# ===========================================================================
# CLAUSE (c): PARAMETER-COUNT INVARIANCE, **WITH CLIPPING OFF**
# ===========================================================================


def clause_c(ctx: DeviceContext, k: Int) raises:
    """Contract 11(c). One tensor's `param.out`, `adam.m` and `adam.v` bits
    are identical whether it is stepped ALONE or alongside the others.

    **IT MUST RUN WITH CLIPPING OFF AND THIS FUNCTION REFUSES A CLIPPING-ON
    CASE BY NAME.** Contract 3.5 is unambiguous: with clipping ON, `coef_c`
    is a function of EVERY GRADIENT IN THE MODEL, so adding a parameter
    tensor, removing one, or changing one element of one of them changes
    every parameter's update. **That is not a defect and it is not
    repairable -- it is what a GLOBAL norm clip means** -- so a gate that
    ran this clause with clipping on and PASSED would be measuring
    something other than what it claims, and one that ran it and FAILED
    would be reporting the reference's own semantics as a bug.

    THE NEGATIVE CONTROL. If the slicer were wrong -- if it returned tensor
    0 whatever tensor it was asked for -- every comparison would be a
    tensor against itself and would pass for ever on every vendor. So the
    clause first proves the slice can tell two tensors apart: tensors 0 and
    1 of the full run have different gradients (different lengths, and
    `opt_binade_shift` puts them binades apart) and their `param.out`
    prefixes must differ. A zero there raises VACUOUS, not FAILED."""
    var c = opt_case(k)
    var cfg = opt_config(c.hp)
    if cfg.max_norm > Float32(0.0):
        raise Error(
            String("optimizer_check: clause (c) REFUSES case ")
            + String(c.name)
            + ", which has clipping ON (max_norm "
            + bits32_hex(cfg.max_norm)
            + "). Contract 3.5: with a GLOBAL norm clip one parameter's"
            + " update depends on every other parameter in the model, BY"
            + " THE REFERENCE'S OWN SEMANTICS, so parameter-count"
            + " invariance is FALSE there and contract 11(c) says the gate"
            + " must run with clipping off."
        )
    var off = opt_offsets(c.shape)
    var j_count = len(off) - 1
    if j_count < 3:
        raise Error(
            String("optimizer_check: clause (c) needs J >= 3 and ")
            + String(c.name)
            + " has J="
            + String(j_count)
        )
    print(
        "clause (c): "
        + String(c.name)
        + ", J="
        + String(j_count)
        + ", clipping OFF, each tensor stepped ALONE and compared to its"
        " slice of the full run"
    )

    # ---- the full run ----------------------------------------------------
    var full_st = DeviceOptState(
        ctx, c, opt_case_param(c), opt_case_m(c), opt_case_v(c)
    )
    var full = List[List[Float32]]()
    for t in range(1, c.steps + 1):
        full = device_step(ctx, full_st, c, cfg, opt_case_grad(c, t), t)

    # ---- THE NEGATIVE CONTROL -------------------------------------------
    var w0 = off[1] - off[0]
    var w1 = off[2] - off[1]
    var w = w0
    if w1 < w:
        w = w1
    var control = 0
    for i in range(w):
        if bits_of(full[OS_PARAM_OUT][off[0] + i]) != bits_of(
            full[OS_PARAM_OUT][off[1] + i]
        ):
            control += 1
    if control == 0:
        raise Error(
            String("optimizer_check: CLAUSE (c) IS VACUOUS. The first ")
            + String(w)
            + " cells of tensor 0 and tensor 1 of the full run are"
            + " bit-identical, which cannot be true of two tensors whose"
            + " gradients are drawn from different hashes AND placed"
            + " binades apart by opt_binade_shift. The comparison cannot"
            + " tell two tensors apart ([[reached-but-inert]])."
        )
    print(
        "clause (c) control: tensors 0 and 1 of the full run differ on "
        + String(control)
        + " of "
        + String(w)
        + " overlapping cells, so the slice distinguishes tensors"
    )

    # ---- THE CLAUSE ------------------------------------------------------
    # Each tensor is stepped ALONE by handing the step a registry of ONE.
    # That is the shape contract 11(c) describes -- "whether it is stepped
    # alone or alongside two others" -- and it exercises the real entry
    # point rather than a special path.
    var bad = 0
    var first_bad = String("")
    var cells = 0
    for j in range(j_count):
        var begin = off[j]
        var count = off[j + 1] - begin
        var solo_off = List[Int]()
        solo_off.append(0)
        solo_off.append(count)

        var p_all = opt_case_param(c)
        var m_all = opt_case_m(c)
        var v_all = opt_case_v(c)
        var p_j = List[Float32]()
        var m_j = List[Float32]()
        var v_j = List[Float32]()
        for i in range(count):
            p_j.append(p_all[begin + i])
            m_j.append(m_all[begin + i])
            v_j.append(v_all[begin + i])

        var d_param = _upload_f32(ctx, p_j)
        var d_m = _upload_f32(ctx, m_j)
        var d_v = _upload_f32(ctx, v_j)
        var d_denom = _upload_f32(ctx, opt_poison(count))
        var d_q = _upload_f32(ctx, opt_poison(count))
        var d_sumsq = _upload_f32(ctx, opt_poison(1))
        var d_norms = _upload_f32(ctx, opt_poison(1))
        var d_total = _upload_f32(ctx, opt_poison(1))
        var d_out2 = _upload_f32(ctx, opt_poison(2))
        var d_ws = _upload_f32(
            ctx, opt_poison(identical_optimizer_workspace_floats(solo_off))
        )
        var d_sab = _upload_f32(ctx, opt_poison(SAB_CHUNKS))
        var flags = List[Bool]()
        flags.append(False)

        for t in range(1, c.steps + 1):
            var g_all = opt_case_grad(c, t)
            var g_j = List[Float32]()
            for i in range(count):
                g_j.append(g_all[begin + i])
            var d_grad = _upload_f32(ctx, g_j)
            identical_optimizer_step(
                ctx, d_param, d_grad, d_m, d_v, d_denom, d_q, d_sumsq,
                d_norms, d_total, d_out2, d_ws, d_sab, flags, solo_off,
                cfg, t,
            )
            ctx.synchronize()
            _ = d_grad^

        var solo_param = _download_f32(ctx, d_param, count)
        var solo_m = _download_f32(ctx, d_m, count)
        var solo_v = _download_f32(ctx, d_v, count)

        var which: List[Int] = [OS_PARAM_OUT, OS_ADAM_M, OS_ADAM_V]
        for wi in range(len(which)):
            var stage = which[wi]
            if not stage_present(stage, cfg):
                continue
            var slice_full = List[Float32]()
            for i in range(count):
                slice_full.append(full[stage][begin + i])
            var solo = solo_param.copy()
            if stage == OS_ADAM_M:
                solo = solo_m.copy()
            elif stage == OS_ADAM_V:
                solo = solo_v.copy()
            cells += count
            var d = compare_stage(
                opt_stage_tag(stage) + " tensor " + String(j),
                slice_full,
                solo,
                False,
            )
            if d.n_diff > 0:
                bad += 1
                if first_bad == "":
                    first_bad = (
                        opt_stage_tag(stage)
                        + " on tensor "
                        + String(j)
                        + ", "
                        + String(d.n_diff)
                        + " of "
                        + String(d.n_cells)
                        + " cells"
                    )
        _ = d_param^
        _ = d_m^
        _ = d_v^
        _ = d_denom^
        _ = d_q^
        _ = d_sumsq^
        _ = d_norms^
        _ = d_total^
        _ = d_out2^
        _ = d_ws^
        _ = d_sab^

    if bad != 0:
        raise Error(
            String("optimizer_check: CLAUSE (c) FAILED on ")
            + String(bad)
            + " tensor-stages, first at "
            + first_bad
            + ". With clipping OFF the update is ELEMENTWISE and reads only"
            + " that element's own gradient and state, so a failure here is"
            + " a finding about the execution plan and not about the gate."
        )
    print(
        "clause (c): PASS, every tensor identical alone and in a registry"
        " of "
        + String(j_count)
        + ", on all "
        + String(cells)
        + " compared cells, with clipping OFF"
    )
    _ = full_st^


# ===========================================================================
# CLAUSE (d): STEP-COUNT INVARIANCE, AND THE CHECKPOINT ROUND TRIP
# ===========================================================================


def clause_d(ctx: DeviceContext, c: OptCase) raises -> Int:
    """Contract 11(d). A run that reaches step `2t` continuously and a run
    resumed from a checkpoint written after step `t` produce BITWISE
    IDENTICAL `param.out`, `adam.m`, `adam.v` and `sgd.buf` at step `2t`.
    Returns the number of differing stage-cells on the CONTROL, which
    clause (g) inverts for the gate-local resume arm.

    **WHY THIS IS STRUCTURAL AND THE GATE STILL EXISTS.** Contract 5.1
    makes `beta^t` a PURE FUNCTION of the integer `t` by binary
    exponentiation, so there is no running `beta_pow` state to checkpoint,
    nothing to reconstruct on resume, and no way for a resumed run to
    disagree with a continuous one about what `beta1^t` is. The clause then
    verifies what the construction promises rather than being the only
    thing standing between the profile and a silent drift. The alternative
    the clause exists to refuse -- a running product `b_t = b_{t-1} * beta`
    -- is `t - 1` sequential roundings, is a different number, and IS
    checkpoint state.

    **THE CHECKPOINT IS A REAL ROUND TRIP AND THAT IS THE WHOLE POINT.**
    The resumed run gets a NEW `DeviceOptState` built from downloaded host
    lists. A "resume" that reused the same buffers in the same process
    would be a value against itself and would pass for ever.

    **TWO NEGATIVE CONTROLS, AND EACH ONE IS A CHECKPOINT FIELD CONTRACT
    11(d) NAMES.**

      1. **The step index.** A resume that forgot `t` and restarted at 1
         MUST differ, because `step_size` carries `1/bc1` and `rt_bc2`
         carries `sqrt(bc2)` and both are functions of `t`. This control
         works for every algorithm and is the one that proves the
         comparison can see a difference at all.
      2. **The momentum-initialized flag**, for SGD with momentum. Contract
         7.3b: a resume that reinitializes it recomputes `b` as a COPY of
         the current gradient instead of continuing the recurrence, and the
         two runs diverge at that step and NEVER RECONVERGE. **It is the
         half a checkpoint format forgets, because the flag is a `Bool` and
         everything around it is a tensor.** This is also
         `OPT_SAB_RESUME_REINIT`, which contract section 12 lists as a
         switch and `optimizer.mojo` does not implement -- DEVIATION 1473
         models it here instead."""
    var cfg = opt_config(c.hp)
    var off = opt_offsets(c.shape)
    var n = off[len(off) - 1]
    var t_half = c.steps
    if t_half < 8:
        raise Error(
            String("optimizer_check: clause (d) needs steps >= 8 and ")
            + String(c.name)
            + " has "
            + String(t_half)
            + ". Contract 11(d): 't = 8 at minimum, because section 5.1's"
            + " spellings agree through t = 6' -- a shorter round trip"
            + " cannot tell a correct bias correction from a running"
            + " product."
        )
    var t_full = 2 * t_half
    print(
        "clause (d): "
        + String(c.name)
        + " -- "
        + String(t_full)
        + " steps continuously against a checkpoint at step "
        + String(t_half)
        + " and a resume through step "
        + String(t_full)
    )

    # ---- the continuous run ---------------------------------------------
    var cont = DeviceOptState(
        ctx, c, opt_case_param(c), opt_case_m(c), opt_case_v(c)
    )
    var cont_last = List[List[Float32]]()
    for t in range(1, t_full + 1):
        cont_last = device_step(ctx, cont, c, cfg, opt_case_grad(c, t), t)

    # ---- the checkpointed run -------------------------------------------
    var first_half = DeviceOptState(
        ctx, c, opt_case_param(c), opt_case_m(c), opt_case_v(c)
    )
    for t in range(1, t_half + 1):
        _ = device_step(ctx, first_half, c, cfg, opt_case_grad(c, t), t)
    # THE CHECKPOINT, off the device and back onto it. Everything contract
    # 11(d) says a checkpoint must carry: the parameters, `m`, `v` and the
    # per-tensor momentum-initialized flag. `t` is carried by the loop
    # bound below and `param_id` by `off`, which is the same registry.
    var ck_param = _download_f32(ctx, first_half.param, n)
    var ck_m = _download_f32(ctx, first_half.m_state, n)
    var ck_v = _download_f32(ctx, first_half.v_state, n)
    var ck_flags = first_half.buf_initialized.copy()

    var resumed = DeviceOptState(ctx, c, ck_param, ck_m, ck_v)
    resumed.buf_initialized = ck_flags.copy()
    var res_last = List[List[Float32]]()
    for t in range(t_half + 1, t_full + 1):
        res_last = device_step(ctx, resumed, c, cfg, opt_case_grad(c, t), t)

    var which: List[Int] = [
        OS_PARAM_OUT, OS_ADAM_M, OS_ADAM_V, OS_SGD_BUF
    ]
    var bad = 0
    var first_bad = String("")
    var cells = 0
    for wi in range(len(which)):
        var stage = which[wi]
        if not stage_present(stage, cfg):
            continue
        cells += len(cont_last[stage])
        var d = compare_stage(
            opt_stage_tag(stage), cont_last[stage], res_last[stage], False
        )
        if d.n_diff > 0:
            bad += 1
            if first_bad == "":
                first_bad = (
                    opt_stage_tag(stage)
                    + " on "
                    + String(d.n_diff)
                    + " of "
                    + String(d.n_cells)
                    + " cells"
                )
    if bad != 0:
        raise Error(
            String("optimizer_check: CLAUSE (d) FAILED on ")
            + String(bad)
            + " stages, first at "
            + first_bad
            + ". Contract 5.1 makes step-count invariance STRUCTURAL --"
            + " `beta^t` is a pure function of the integer `t` -- so a"
            + " failure here means something in the step is carrying state"
            + " the checkpoint does not, and contract 11(d) lists what a"
            + " checkpoint must hold."
        )
    print(
        "clause (d): PASS, the resumed run is bit-identical to the"
        " continuous one at step "
        + String(t_full)
        + " on all "
        + String(cells)
        + " cells"
    )

    # ---- CONTROL 1: a resume that FORGOT `t` ---------------------------
    # Restarting `t` tests Adam bias correction. SGD never reads `t`, so its
    # corresponding resume falsifier is the momentum-state control below.
    if cfg.kind == OPT_SGD:
        print(
            "  clause (d) control 1: SKIPPED -- it corrupts the step index,"
            " and contract 4.2's SGD update never reads `t`, so bias"
            " correction (the mechanism this control names) does not exist"
            " here. Control 2 is SGD's falsifier. DEVIATION 1494."
        )
        return 0
    var forgot = DeviceOptState(ctx, c, ck_param, ck_m, ck_v)
    forgot.buf_initialized = ck_flags.copy()
    var forgot_last = List[List[Float32]]()
    for t in range(1, t_half + 1):
        forgot_last = device_step(
            ctx, forgot, c, cfg, opt_case_grad(c, t_half + t), t
        )
    var ctrl1 = 0
    for wi in range(len(which)):
        var stage = which[wi]
        if not stage_present(stage, cfg):
            continue
        var d = compare_stage(
            opt_stage_tag(stage), cont_last[stage], forgot_last[stage],
            False,
        )
        ctrl1 += d.n_diff
    if ctrl1 == 0:
        raise Error(
            "optimizer_check: CLAUSE (d) IS VACUOUS. A resume that"
            " RESTARTED THE STEP INDEX AT 1 -- and therefore used the wrong"
            " bias correction on every remaining step -- produced"
            " bit-identical state. The comparison cannot see a difference"
            " at all, so the aligned result above is a value against itself"
            " ([[reached-but-inert]])."
        )
    print(
        "clause (d) control 1: a resume that FORGOT the step index differs"
        " on "
        + String(ctrl1)
        + " cells, so the comparison can see a checkpoint field going"
        " missing. `t` is one of the fields contract 11(d) lists."
    )

    # ---- CONTROL 2: SGD's momentum-initialized flag --------------------
    var ctrl2 = 0
    if cfg.kind == OPT_SGD and cfg.momentum != Float32(0.0):
        var reinit = DeviceOptState(ctx, c, ck_param, ck_m, ck_v)
        # THE DROPPED FLAG. `DeviceOptState.__init__` already sets every
        # flag False, so this control is what a checkpoint format that
        # forgot the `Bool` actually produces -- not a synthetic
        # perturbation.
        var re_last = List[List[Float32]]()
        for t in range(t_half + 1, t_full + 1):
            re_last = device_step(ctx, reinit, c, cfg, opt_case_grad(c, t), t)
        for wi in range(len(which)):
            var stage = which[wi]
            if not stage_present(stage, cfg):
                continue
            var d = compare_stage(
                opt_stage_tag(stage), cont_last[stage], re_last[stage],
                False,
            )
            ctrl2 += d.n_diff
        if ctrl2 == 0:
            raise Error(
                "optimizer_check: CLAUSE (d) CONTROL 2 IS DEAD. A resume"
                " that DROPPED the momentum-initialized flag -- so the"
                " first resumed step recomputed the buffer as a COPY of"
                " the gradient instead of continuing the recurrence --"
                " produced bit-identical state. Contract 7.3b says the two"
                " runs diverge at that step and NEVER RECONVERGE, so"
                " either that is false or the flag is not reaching the"
                " kernel at all ([[reached-but-inert]])."
            )
        print(
            "clause (d) control 2 (= OPT_SAB_RESUME_REINIT, DEVIATION"
            " 1473): dropping the momentum-initialized flag on resume"
            " differs on "
            + String(ctrl2)
            + " cells. It is the half a checkpoint format forgets, because"
            " the flag is a Bool and everything around it is a tensor."
        )
    else:
        print(
            "clause (d) control 2: SKIPPED -- it needs SGD with momentum,"
            " and this case is "
            + String(c.name)
            + ". The flag does not exist for Adam, so the arm has nothing"
            " to drop there."
        )
    _ = cont^
    _ = first_half^
    _ = resumed^
    _ = forgot^
    return ctrl2


# ===========================================================================
# CLAUSE (e): MICROBATCH ALIGNMENT -- HOST ONLY, AND ALWAYS ON
#
# Contract 11(e) and section 9. This clause is host-only because the thing
# it is about is the CALLER's accumulation, not a kernel: the optimizer
# step imposes no alignment requirement on its own arithmetic and imposes
# one on whoever produced the gradient. It is always on because it costs
# nothing and because contract 9.3 says the EXISTING measurement CANNOT
# supply its answer.
# ===========================================================================


def _hashed_vec(seed: UInt64, n: Int) -> List[Float32]:
    """`n` hashed values in `[1, 2)` -- exactly representable, never zero,
    never subnormal, and all in ONE binade so the fold has to work for its
    differences rather than being dominated by one term."""
    var out = List[Float32]()
    for i in range(n):
        var h = opt_splitmix64(seed + UInt64(i))
        var frac = Int((h >> 41) & UInt64(0x7FFFFF))
        out.append(Float32(1.0 + Float64(frac) * 1.1920928955078125e-07))
    return out^


def _tree_combine(parts: List[Float32]) -> Float32:
    """v1's BALANCED TREE over adjacent pieces with the odd tail CARRIED,
    `ftz(ftz(x) + ftz(y))` at every node.

    This is contract 9.2 condition 5's accumulator, written here so the
    clause can compare it against the serial alternative. It is a SECOND
    SPELLING of the GEMM's own tree and that is deliberate: the point of
    the clause is to compare two CALLER-side accumulators, and routing one
    of them through the code under test would make the comparison
    circular."""
    var level = parts.copy()
    while len(level) > 1:
        var nxt = List[Float32]()
        var i = 0
        while i + 1 < len(level):
            nxt.append(ftz(ftz(level[i]) + ftz(level[i + 1])))
            i += 2
        if i < len(level):
            nxt.append(level[i])  # the odd tail, CARRIED and not padded
        level = nxt^
    return level[0]


def _serial_combine(parts: List[Float32]) -> Float32:
    """A running serial sum. **THE ARM.** `GATE_MICROBATCH_SERIAL`, and
    contract section 12 lists `OPT_SAB_MICROBATCH_SERIAL` as a switch that
    `optimizer.mojo` does not implement -- because it is a defect in a
    TRAINER's accumulation loop and not in any kernel. DEVIATION 1473."""
    var acc = Float32(0.0)
    for i in range(len(parts)):
        acc = ftz(ftz(acc) + ftz(parts[i]))
    return acc


def clause_e() raises:
    """Contract 11(e) and contract 9.3's sharpening.

    **CONTRACT 9.2 CONDITION 5 IS NOT ESTABLISHED BY THE MEASURED EVIDENCE
    AND THIS CLAUSE IS WHAT CLOSES IT.** `IDENTICAL_BACKWARD_PLAN.md` 5.2's
    gate G5 measured five accumulation splits on 2026-08-25 -- 512 as
    256/256 and 384 as 256/128 each moved 0 of 35 gradient cells, while
    150/150 of 300, 200/312 of 512 and 192/192 of 384 moved 31, 30 and 31.
    **EVERY ALIGNED CASE MEASURED IS `A = 2`, and over TWO pieces a serial
    running sum and a balanced tree are the SAME OPERATION.** So a reader
    who concludes "the accumulator just has to be the flushed add" is
    over-reading it. At `A = 4` they separate: the tree computes
    `ftz(ftz(p0+p1) + ftz(p2+p3))` and a running sum computes
    `ftz(ftz(ftz(p0+p1) + p2) + p3)`.

    So this clause does three things, and the third is the one nothing else
    does:
      1. prints `microbatch_split_is_identical` over a sweep of `(T, A)`,
      2. MEASURES that at an ALIGNED `A` the tree combination reproduces the
         unsplit fold bit for bit,
      3. MEASURES that at `A = 4` the SERIAL combination does NOT, while at
         `A = 2` it does -- which is exactly why G5's evidence cannot see
         condition 5.

    The whole clause is host arithmetic through `gemm_oracle`, which is the
    NORMATIVE fold and carries no sabotage switch of any kind."""
    print(
        "clause (e): microbatch alignment (contract 9.2 and 9.3), host"
        " only, always on"
    )
    var ts: List[Int] = [512, 384, 300, 256]
    var as_: List[Int] = [1, 2, 3, 4, 8]
    for ti in range(len(ts)):
        var t = ts[ti]
        var line = String("  T=") + String(t) + ":"
        for ai in range(len(as_)):
            var a = as_[ai]
            line += " A=" + String(a) + "="
            if microbatch_split_is_identical(t, a):
                line += "aligned"
            else:
                line += "NO"
        print(line)

    # ---- THE MEASUREMENT, at T = 512 ------------------------------------
    var t_tokens = 512
    # DEVIATION 1493, AND THE FIRST RUN OF THIS CLAUSE FOUND IT. These two
    # were `_hashed_vec`, whose docstring says the values sit in ONE BINADE
    # "so the fold has to work for its differences rather than being
    # dominated by one term". That is a good instinct and it is exactly what
    # made this clause BLIND. At `A = 4` all four leaf sums come out near
    # equal, so `ftz(ftz(p0+p1) + ftz(p2+p3))` and
    # `ftz(ftz(ftz(p0+p1) + p2) + p3)` round the same way and AGREE. The
    # clause correctly raised VACUOUS: contract 9.3's condition 5 had no
    # falsifier, which is the ONLY reason the `A = 4` fixture is required at
    # all (G5's measurement used two pieces, where a running sum and a tree
    # are the same operation).
    #
    # A separator needs the OPPOSITE of one binade: one huge partial and
    # three tiny ones. With `b` all `1.0`, `a[0:128] = 2^17` gives leaf 0 a
    # sum of exactly `2^24`, and `a[128:512] = 2^-7` gives each remaining
    # leaf exactly `1.0`. Then
    #
    #     tree   (2^24 + 1) + (1 + 1) = 16777218   0x4b800001
    #     serial ((2^24 + 1) + 1) + 1 = 16777216   0x4b800000
    #
    # one ULP apart, because `2^24 + 1` is not representable and rounds to
    # `2^24` under round-half-even while `2^24 + 2` is. EVERY VALUE HERE IS
    # EXACTLY REPRESENTABLE and every leaf sum is exact, so the separation is
    # the COMBINATION ORDER and nothing else. Verified on the host before
    # being written here.
    #
    # The hashed pair is kept and REPORTED below, because "the fold agrees at
    # near-equal partials" is a true and useful fact -- it just is not a
    # falsifier.
    var a_vec = List[Float32]()
    var b_vec = List[Float32]()
    for i in range(t_tokens):
        b_vec.append(Float32(1.0))
        if i < 128:
            a_vec.append(Float32(131072.0))
        else:
            a_vec.append(Float32(0.0078125))
    var a_hashed = _hashed_vec(UInt64(0x4D42414C31), t_tokens)
    var b_hashed = _hashed_vec(UInt64(0x4D42414C32), t_tokens)
    var wh = gemm_oracle(a_hashed, b_hashed, OP_NT, 1, 1, t_tokens)
    var hashed_parts = List[Float32]()
    for piece in range(4):
        var ha = List[Float32]()
        var hb = List[Float32]()
        for i in range(t_tokens // 4):
            ha.append(a_hashed[piece * (t_tokens // 4) + i])
            hb.append(b_hashed[piece * (t_tokens // 4) + i])
        var hg = gemm_oracle(ha, hb, OP_NT, 1, 1, t_tokens // 4)
        hashed_parts.append(hg[0])
    print(
        "  [REPORTED, NOT ASSERTED] the ONE-BINADE hashed pair at A = 4:"
        " tree and serial agree = "
        + String(
            bits_of(_tree_combine(hashed_parts))
            == bits_of(_serial_combine(hashed_parts))
        )
        + ", which is why DEVIATION 1493 replaced it as the falsifier."
    )
    _ = wh
    var whole = gemm_oracle(a_vec, b_vec, OP_NT, 1, 1, t_tokens)
    if len(whole) == 0:
        raise Error("optimizer_check: clause (e) got an empty unsplit fold")
    var unsplit = whole[0]

    for ai in range(len(as_)):
        var a = as_[ai]
        if not microbatch_split_is_identical(t_tokens, a):
            continue
        if a == 1:
            continue
        var per = t_tokens // a
        var parts = List[Float32]()
        for piece in range(a):
            var pa = List[Float32]()
            var pb = List[Float32]()
            for i in range(per):
                pa.append(a_vec[piece * per + i])
                pb.append(b_vec[piece * per + i])
            var got = gemm_oracle(pa, pb, OP_NT, 1, 1, per)
            parts.append(got[0])
        var tree = _tree_combine(parts)
        var serial = _serial_combine(parts)
        var tree_ok = bits_of(tree) == bits_of(unsplit)
        var serial_ok = bits_of(serial) == bits_of(unsplit)
        print(
            "  T=512 A="
            + String(a)
            + " (ALIGNED): unsplit "
            + bits32_hex(unsplit)
            + "   tree "
            + bits32_hex(tree)
            + " ("
            + (String("MATCH") if tree_ok else String("MOVED"))
            + ")   serial "
            + bits32_hex(serial)
            + " ("
            + (String("MATCH") if serial_ok else String("MOVED"))
            + ")"
        )
        if not tree_ok:
            raise Error(
                String("optimizer_check: CLAUSE (e) FAILED. At T=512, A=")
                + String(a)
                + " -- which `microbatch_split_is_identical` calls ALIGNED"
                + " -- the BALANCED TREE combination of the pieces does NOT"
                + " reproduce the unsplit fold. Contract 9.2's whole"
                + " predicate is that conditions 1 through 4 make each"
                + " piece a COMPLETE SUBTREE of the unsplit tree, so this"
                + " is a finding about the predicate."
            )
        if a == 2 and not serial_ok:
            raise Error(
                "optimizer_check: CLAUSE (e) -- at A = 2 the SERIAL"
                " combination differed from the unsplit fold. Contract 9.3"
                " says a serial running sum and a balanced tree are the"
                " SAME OPERATION over two pieces, which is why G5's"
                " measured evidence cannot see condition 5. If that is"
                " false, contract 9.3's argument is wrong and THAT is the"
                " finding."
            )
        if a == 4 and serial_ok:
            raise Error(
                "optimizer_check: CLAUSE (e) -- at A = 4 the SERIAL"
                " combination MATCHED the unsplit fold. Contract 9.3 says"
                " they separate there --"
                " `ftz(ftz(p0+p1) + ftz(p2+p3))` against"
                " `ftz(ftz(ftz(p0+p1) + p2) + p3)` -- and it is the ONLY"
                " reason the A = 4 fixture is required. If they agree at"
                " this data, condition 5 has no falsifier here"
                " ([[reached-but-inert]])."
            )

    # ---- a MISALIGNED A, REPORTED and not asserted ---------------------
    # Contract 11(e): "at an `A` that does not satisfy 9.2, the number of
    # cells that move is REPORTED, not asserted, and the host oracle is
    # what says the number before the device is asked."
    var a_bad = 3
    var per_bad = t_tokens // a_bad
    if per_bad * a_bad == t_tokens:
        var parts_bad = List[Float32]()
        for piece in range(a_bad):
            var pa = List[Float32]()
            var pb = List[Float32]()
            for i in range(per_bad):
                pa.append(a_vec[piece * per_bad + i])
                pb.append(b_vec[piece * per_bad + i])
            var got = gemm_oracle(pa, pb, OP_NT, 1, 1, per_bad)
            parts_bad.append(got[0])
        var tb = _tree_combine(parts_bad)
        print(
            "  T=512 A=3 (MISALIGNED, REPORTED not asserted): tree "
            + bits32_hex(tb)
            + " against unsplit "
            + bits32_hex(unsplit)
            + " -- "
            + (
                String("MATCH")
                if bits_of(tb) == bits_of(unsplit)
                else String("MOVED")
            )
            + ". Contract 9.4: **the microbatch count is part of a run's"
            " numerical specification unless clause 9.2 holds**, and a"
            " reproducibility claim must name A."
        )
    else:
        print(
            "  T=512 A=3: 3 does not divide 512, so the split does not"
            " exist and there is nothing to report"
        )


# ===========================================================================
# CLAUSE (f): THE ROW-39 AUDIT OF CONTRACT SECTION 8
#
# A non-finite value in a gradient, a parameter or a state must be REFUSED
# BY NAME before any recorded stage, because NaN PAYLOADS ARE VENDOR SHAPED
# -- IDENTITY_PATHS row 39 measured `0x7fc00000` on Apple, `0x7fffffff` on
# NVIDIA and `0xffc00000` on AMD for ONE IEEE answer -- so a certified card
# may never contain a computed NaN, and a cross-vendor gate would otherwise
# have to compare NaN cells as "is NaN" rather than by bits, which is a hole
# in a bitwise claim.
#
# Every test is BY BITS AND NEVER BY A COMPARE (row 49).
#
# **AND THE MEASURED FINDING, DEVIATION 1478.** The device path is WEAKER
# than the contract. `identical_clip_grad_norm` refuses only the SCALAR
# `clip.total_norm` -- one float to copy back -- and it does not run at all
# when clipping is off. `identical_optimizer_step` has no refusal of its
# own. So a non-finite PARAMETER, or a non-finite `m` or `v`, or ANY
# non-finite input on a run with clipping off, reaches `param.out`.
# `optimizer.mojo`'s own docstring says so; this clause MEASURES it.
# ===========================================================================


def clause_f(ctx: DeviceContext) raises:
    var c = opt_case(opt_case_by_name(String("adam_j5_t8_noclip")))
    var cfg = opt_config(c.hp)
    var off = opt_offsets(c.shape)
    var n = off[len(off) - 1]
    print(
        "clause (f): the section 8 audit, every test BY BITS, never by a"
        " compare"
    )

    # ---- THE CONTROL ----------------------------------------------------
    var clean_param = opt_case_param(c)
    var clean_grad = opt_case_grad(c, 1)
    var clean_m = opt_case_m(c)
    var clean_v = opt_case_v(c)
    var ctrl_raised = False
    try:
        refuse_nonfinite(String("input.param"), clean_param)
        refuse_nonfinite(String("input.grad"), clean_grad)
        refuse_nonfinite(String("state.m"), clean_m)
        refuse_nonfinite(String("state.v"), clean_v)
    except e:
        ctrl_raised = True
    if ctrl_raised:
        raise Error(
            "optimizer_check: CLAUSE (f) IS VACUOUS. `refuse_nonfinite`"
            " fired on CLEAN inputs, so every plant below would be"
            " 'refused' whatever it held and this clause gates nothing."
        )
    print(
        "clause (f) control: a clean call does NOT raise, so the refusal is"
        " not unconditional"
    )

    # ---- THE PLANTS, ON THE ORACLE --------------------------------------
    var names: List[String] = [
        String("input.param"), String("input.grad"), String("state.m"),
        String("state.v"),
    ]
    var patterns: List[UInt32] = [BITS_QNAN, BITS_POS_INF]
    var pat_names: List[String] = [String("NaN"), String("infinity")]
    var checked = 0
    for pk in range(len(patterns)):
        for idx in range(len(names)):
            var p = opt_case_param(c)
            var g = opt_case_grad(c, 1)
            var m = opt_case_m(c)
            var v = opt_case_v(c)
            var flags = opt_buf_initialized(c)
            # **AT CELL `len / 2`, NOT AT CELL 0.** A plant at index 0 is
            # the one a loop that skips its first element would still
            # catch, which makes it the weakest possible plant site.
            var val = f32_from_bits(patterns[pk])
            if idx == 0:
                p[len(p) // 2] = val
            elif idx == 1:
                g[len(g) // 2] = val
            elif idx == 2:
                m[len(m) // 2] = val
            else:
                v[len(v) // 2] = val
            var raised = False
            var msg = String("")
            try:
                var stages = optimizer_step_oracle(
                    p, g, m, v, flags, off, cfg, 1
                )
                _ = stages^
            except e:
                raised = True
                msg = String(e)
            if not raised:
                raise Error(
                    String("optimizer_check: CLAUSE (f) FAILED. A ")
                    + pat_names[pk]
                    + " planted in "
                    + names[idx]
                    + " was NOT refused by the host oracle. Contract 8a:"
                    + " a certified card may never contain a computed NaN,"
                    + " because the payload is vendor shaped."
                )
            if msg.find(names[idx]) < 0:
                raise Error(
                    String("optimizer_check: CLAUSE (f) FAILED. A ")
                    + pat_names[pk]
                    + " planted in "
                    + names[idx]
                    + " was refused and the refusal does not NAME it: "
                    + msg
                )
            checked += 1
    print(
        "clause (f): "
        + String(checked)
        + " plants, each REFUSED BY NAME by the host oracle, both bit"
        " patterns, each at cell len/2"
    )

    # ---- THE MEASURED GAP, part 1: CLIPPING OFF ------------------------
    var bad_param = opt_case_param(c)
    var at = len(bad_param) // 2
    bad_param[at] = f32_from_bits(BITS_QNAN)
    var st = DeviceOptState(ctx, c, bad_param, opt_case_m(c), opt_case_v(c))
    # REACH, MEASURED: the plant is read back off the device before the
    # step, so a refusal (or a leak) after this cannot be attributed to
    # something else. `[[verify-reach-not-output]]`.
    var back = _download_f32(ctx, st.param, n)
    if nonfinite_cells(back) != 1:
        raise Error(
            String("optimizer_check: CLAUSE (f) IS VACUOUS for the device")
            + " arm -- the planted NaN did NOT arrive on the device ("
            + String(nonfinite_cells(back))
            + " non-finite cells read back, expected exactly 1)."
        )
    # Require a named refusal. Merely seeing finite output cannot distinguish
    # a valid refusal from a laundered NaN.
    var refused = False
    var refuse_msg = String("")
    var leaked = -1
    try:
        var dev = device_step(ctx, st, c, cfg, opt_case_grad(c, 1), 1)
        leaked = nonfinite_cells(dev[OS_PARAM_OUT])
    except e:
        refused = True
        refuse_msg = String(e)
    if not refused:
        if leaked == 0:
            raise Error(
                String("optimizer_check: CLAUSE (f) -- a NaN was planted in")
                + " a PARAMETER, the device did NOT refuse, and `param.out`"
                + " came back entirely finite. That is a LAUNDERED NaN and"
                + " it is a worse finding than a propagated one, because it"
                + " means a stage is not a function of its input."
            )
        raise Error(
            String("optimizer_check: CLAUSE (f) -- DEVIATION 1496's refusal")
            + " did not fire. A NaN planted in a PARAMETER reached "
            + String(leaked)
            + " cells of param.out with clipping OFF. Contract 8a is again a"
            + " property of the oracle and not of the profile."
        )
    if refuse_msg.find("param") < 0:
        raise Error(
            String("optimizer_check: CLAUSE (f) -- the device refused, but")
            + " NOT BY THE NAME contract 8a requires. A refusal that does"
            + " not name the offending buffer cannot be compared against"
            + " the oracle's, which is the only thing that makes two"
            + " refusals the same refusal. Got: "
            + refuse_msg
        )
    print(
        "clause (f) DEVICE REFUSAL, ASSERTED (DEVIATION 1478 CLOSED by"
        " 1496): with clipping OFF, a NaN planted in a PARAMETER is now"
        " REFUSED BY NAME by the device entry point, not merely by the"
        " oracle. The refusal names the buffer: "
        + refuse_msg
    )
    return
    print(
        "clause (f) DEVICE GAP, MEASURED (DEVIATION 1478): with clipping"
        " OFF, a NaN planted in a PARAMETER reached "
        + String(leaked)
        + " cells of param.out. `identical_optimizer_step` has no refusal"
        " of its own and `identical_clip_grad_norm` -- the only refusal on"
        " the device side -- does not run at all when clipping is off."
        " Contract 8a's 'REFUSED BY NAME before any recorded stage' is a"
        " property of `optimizer_step_oracle` and NOT of the profile."
    )
    _ = st^

    # ---- THE MEASURED GAP, part 2: the ONE refusal the device HAS ------
    var cc = opt_case(opt_case_by_name(String("adam_j5_t8")))
    var ccfg = opt_config(cc.hp)
    if ccfg.max_norm <= Float32(0.0):
        raise Error(
            "optimizer_check: clause (f) part 2 needs a case with clipping"
            " ON and adam_j5_t8 does not have it"
        )
    var cst = DeviceOptState(
        ctx, cc, opt_case_param(cc), opt_case_m(cc), opt_case_v(cc)
    )
    var bad_grad = opt_case_grad(cc, 1)
    bad_grad[len(bad_grad) // 2] = f32_from_bits(BITS_QNAN)
    var raised2 = False
    var msg2 = String("")
    try:
        _ = device_step(ctx, cst, cc, ccfg, bad_grad, 1)
    except e:
        raised2 = True
        msg2 = String(e)
    if not raised2:
        raise Error(
            String("optimizer_check: CLAUSE (f) FAILED. With clipping ON, a")
            + " NaN in a gradient did NOT make the device refuse."
            + " `identical_clip_grad_norm` calls"
            + " `refuse_nonfinite_scalar('clip.total_norm', ...)` after"
            + " reading the scalar back, and a NaN gradient reaches"
            + " `total_norm` through the sum of squares, so this is the ONE"
            + " refusal the device path has and it did not fire."
        )
    if msg2.find(String("clip.total_norm")) < 0:
        raise Error(
            String("optimizer_check: CLAUSE (f) FAILED. The device refused")
            + " a NaN gradient and the refusal does not name"
            + " clip.total_norm: "
            + msg2
        )
    print(
        "clause (f): with clipping ON, a NaN gradient IS refused by the"
        " device, by name, at clip.total_norm -- which is the ONE refusal"
        " the device path has, and it covers neither a non-finite PARAMETER"
        " nor a non-finite `m` or `v` nor any input at all on a run with"
        " clipping off."
    )
    _ = cst^


# ===========================================================================
# CLAUSE (g): every clause above falsifiable by a NAMED sabotage
# ===========================================================================


@fieldwise_init
struct ArmExpectation(Copyable, Movable):
    """What one arm must do.

    `first_step` is the step at which the arm must FIRST move something, or
    `-1` for "any step". Three arms in this lane have a predicted step and
    it is a real prediction rather than bookkeeping:
      * `OPT_SAB_POW_RUNNING` at `t = 7` (contract 5.1, PREDICTED and never
        measured until preflight computed it),
      * `OPT_SAB_MOMENT_LERP` at `t = 2`, because at `t = 1` `m_prev` is
        exactly `+0.0` and the two spellings coincide (DEVIATION 1474, not
        in contract section 12's table),
      * `OPT_SAB_NESTEROV_ORDER` at `t = 2`, because at `t = 1` the buffer
        is a COPY of `g` so `b == g` (contract 7.3c).
    An arm that moved at the WRONG step is an arm that is not doing what
    its clause says, even though it moved.

    `expect_inert` marks contract section 12's last row,
    `OPT_SAB_SCALARS_PER_ELEMENT`, whose predicted result is NO BITS MOVE
    and which is in the set as a REACH probe. An arm whose predicted answer
    is "nothing happens" is worth having ONLY when it is labelled that way
    in advance.

    `is_gate_arm` marks the two arms `optimizer.mojo` does not implement
    and this file models itself (DEVIATION 1473)."""

    var arm: String
    var first_stage: String
    var witness: String
    var inert_a: String
    var inert_b: String
    var first_step: Int
    var expect_inert: Bool
    var needs_record: Bool
    var is_gate_arm: Bool
    var note: String


def arm_expectation(arm: String) raises -> ArmExpectation:
    if arm == "POW_RUNNING":
        return ArmExpectation(
            arm, String("sched.pow1"), String("adam_t7"), String("adam_t6"),
            String("adam_t1"), 7, False, False, False,
            String(
                "contract 5.1. `beta^t` by a running product is t-1"
                " sequential roundings. **INERT AT t <= 6** -- the"
                " contract's PREDICTED first difference is t = 7 and"
                " preflight COMPUTES it, because if that number is wrong"
                " the witness and the inert half are pointed at the wrong"
                " steps and the arm looks broken"
            ),
        )
    if arm == "POW_EXPLOG":
        return ArmExpectation(
            arm, String("sched.pow1"), String("adam_t1"), String(""),
            String(""), 1, False, False, False,
            String(
                "contract 5.1. `exp(t * log(beta))` through two Cephes"
                " polynomials, not exact even at t = 1, so it separates"
                " IMMEDIATELY. Contract section 12: 'nothing; it separates"
                " at t = 1. Cheap arm, keep it'"
            ),
        )
    if arm == "EPS_INSIDE_SQRT":
        return ArmExpectation(
            arm, String("adam.denom"), String("adam_dead_v"),
            String("adam_ordinary_v"), String(""), -1, False, True, False,
            String(
                "contract 4d. `sqrt(v_hat + eps)` instead of"
                " `sqrt(v_hat) + eps`. **INERT wherever v is much larger"
                " than eps^2**, and with eps = 1e-8 that is 1e-16, so a"
                " fixture of ordinary gradients (v around 1e-4) cannot see"
                " it. adam_dead_v plants v in the 1e-20 to 1e-12 band a"
                " real run reaches on a dead unit"
            ),
        )
    if arm == "RSQRT":
        return ArmExpectation(
            arm, String("adam.denom"), String("adam_hashed_4096"),
            String(""), String(""), -1, False, True, False,
            String(
                "contract 4b. **NO INERT CASE**: DEVIATION 741 measured"
                " `identical_rsqrt` off the correctly rounded rsqrt on"
                " 134,858 of 520,133 positive normals, so about three"
                " quarters of inputs AGREE and a handful of round v values"
                " will pass. The witness is 4096 hashed lanes and the"
                " verdict reports the CELL COUNT, because a low count here"
                " is the interesting number rather than a pass"
            ),
        )
    if arm == "RECIP_MUL":
        return ArmExpectation(
            arm, String("adam.q"), String("adam_j5_t8_noclip"), String(""),
            String(""), -1, False, True, False,
            String(
                "contract 4c. **THIS ARM IS A SMOKE TEST AND NOT A REACH"
                " PROOF, and this file could not make it one.** `x * (1/d)`"
                " is EXACT at a power-of-two denominator, so an inert half"
                " needs the Adam denominator -- a sum of a quotient of a"
                " square root and an eps, every term data dependent -- to"
                " come out an exact power of two, and there is no"
                " assignment of the fixture's free variables that does it"
                " except by search"
            ),
        )
    if arm == "ADAMW_AS_ADAM":
        return ArmExpectation(
            arm, String("param.out"), String("adamw_wd_t8"),
            String("adamw_wd0_t8"), String(""), 1, False, False, False,
            String(
                "contract 7.4, **THE SINGLE MOST LIKELY VACUOUS GATE IN"
                " THE LANE**. At weight_decay == 0 -- which is"
                " torch.optim.Adam's DEFAULT -- Adam and AdamW are the"
                " SAME ARITHMETIC and the arm changes nothing. Its first"
                " stage is param.out and not adamw.pdec, because contract"
                " section 10 lists adamw.pdec and NOTHING PRODUCES IT"
                " (DEVIATION 1477), so the one seam where the two"
                " algorithms differ has no stage of its own"
            ),
        )
    if arm == "DECAY_ADD_FORM":
        return ArmExpectation(
            arm, String("param.out"), String("adamw_decay_sep"),
            String("adamw_decay_pow2"), String(""), 1, False, False, False,
            String(
                "contract 7.4a. `p - lr*wd*p` instead of `p * (1 - lr*wd)`."
                " INERT when `lr*wd` is a power of two, where the product"
                " is exact -- adamw_decay_pow2 sets lr*wd to exactly 2^-7"
            ),
        )
    if arm == "MOMENT_LERP":
        return ArmExpectation(
            arm, String("adam.m"), String("adam_t8"), String("adam_t1"),
            String(""), 2, False, False, False,
            String(
                "contract 7.2a. `m + c1*(g - m)` instead of a PRODUCT then"
                " an FMA. **INERT AT m_prev == +0.0, WHICH IS EVERY FIRST"
                " STEP FROM FRESH STATE** -- DEVIATION 1474, and it is NOT"
                " in contract section 12's table. An optimizer gate that"
                " ran one step could not see this arm at all"
            ),
        )
    if arm == "SQ_ASSOC":
        return ArmExpectation(
            arm, String("adam.v"), String("adam_j5_t8_noclip"),
            String("adam_unit_grad"), String(""), 1, False, False, False,
            String(
                "contract 7.2b. `(c2*g)*g` instead of `c2*(g*g)`. INERT on"
                " round g -- adam_unit_grad makes every gradient EXACTLY"
                " 1.0, where both readings are exactly c2"
            ),
        )
    if arm == "MHAT_FORM":
        return ArmExpectation(
            arm, String("adam.denom"), String("adam_j5_t8_noclip"),
            String(""), String(""), -1, False, True, False,
            String(
                "contract 7.2c. The documented-pseudocode shape -- explicit"
                " m_hat and v_hat, `lr *` at the end -- one more rounding."
                " **NO INERT CASE, and contract section 12 says so about"
                " itself**: 'nothing known to make it inert, but it is a"
                " 1-ulp-class arm -- report the cell count, do not assume'"
            ),
        )
    if arm == "UNFUSED_UPDATE":
        return ArmExpectation(
            arm, String("param.out"), String("adam_fused_sep"),
            String("adam_fused_inert"), String(""), 1, False, False, False,
            String(
                "contract 7.2d, and its non-vacuity note is the sharpest"
                " sentence in the contract: **check-ieee-arith scored Metal"
                " as UNFUSED over 2^20 HASHED patterns and the verdict was"
                " WRONG, because ZERO of those patterns separate a fused"
                " a*b+c from an unfused one.** adam_fused_sep does not draw"
                " its parameters, it COMPUTES them as the ROUNDED product"
                " step_size*q, so the unfused spelling gives EXACTLY +0.0"
                " and the fused one gives the rounding residual. That is"
                " the whole value, not a last bit (DEVIATION 1472)"
            ),
        )
    if arm == "FTZ_LATE":
        return ArmExpectation(
            arm, String("adam.v"), String("adam_tiny_grad"),
            String("adam_ordinary_v"), String(""), 1, False, False, False,
            String(
                "contract section 6, seam O7. **INERT unless an"
                " INTERMEDIATE lands subnormal**, and a gradient at 1e-25"
                " is a perfectly ordinary normal whose SQUARE is 1e-50 --"
                " not representable as a normal at all. Flush it and v"
                " picks up exactly c2*0 on every column; carry it and Metal"
                " disagrees with CUDA from that step onward FOREVER,"
                " because v is running state"
            ),
        )
    if arm == "CLIP_SKIP_AT_ONE":
        return ArmExpectation(
            arm, String("clip.grad"), String("clip_subnormal"),
            String("clip_normal"), String(""), 1, False, False, False,
            String(
                "contract 3.4c. **THE VERDICT MUST BE READ AT clip.grad AND"
                " NOT AT param.out**: the difference is CARD VISIBLE and"
                " DOWNSTREAM INERT, because the optimizer's own first act"
                " is ftz on the gradient load, so a gate that compared"
                " param.out would report the arm inert and the report would"
                " be TRUE AND USELESS. It needs a SUBNORMAL gradient cell"
                " and a clamp that actually fires"
            ),
        )
    if arm == "CLIP_FLAT_NORM":
        return ArmExpectation(
            arm, String("clip.total_sumsq"), String("clip_spread_j3"),
            String("clip_j1"), String(""), 1, False, False, False,
            String(
                "contract 3.1. One flat sum of squares instead of the"
                " reference's TWO-LEVEL form. In exact arithmetic they are"
                " the same number and in Float32 they are not, because each"
                " per-tensor sqrt rounds and each result is squared again."
                " **INERT AT J == 1**, and it needs norms several binades"
                " apart -- which preflight MEASURES rather than assumes"
            ),
        )
    if arm == "CLIP_PARAM_ORDER":
        return ArmExpectation(
            arm, String("clip.total_sumsq"), String("clip_j5"),
            String("clip_j2"), String(""), 1, False, False, False,
            String(
                "contract 3.3, the CROSS-TENSOR SUMMATION ORDER. **A"
                " FIXTURE WITH J == 2 CANNOT SEE IT AT ALL**, because"
                " reversing two elements swaps the two children of ONE tree"
                " node and a+b == b+a bitwise. That is why clip_j2 is the"
                " inert half and why there is no other way to have one"
            ),
        )
    if arm == "CLIP_SERIAL_FOLD":
        return ArmExpectation(
            arm, String("clip.sumsq"), String("clip_ragged_j3"),
            String("clip_small_j3"), String(""), 1, False, False, False,
            String(
                "contract 3.2. A hand-written fold instead of the v1 GEMM."
                " **INERT AT EVERY N <= 128**, where P == 1, the tree has"
                " no arithmetic node, and the v1 answer IS the serial"
                " ascending chain. clip_ragged_j3 carries N = 300 -- P = 3"
                " with a 44-element ragged last leaf"
            ),
        )
    if arm == "CLIP_BLOCK_PARTITION":
        return ArmExpectation(
            arm, String("clip.sumsq"), String("clip_ragged_j3"),
            String("clip_small_j3"), String(""), 1, False, False, False,
            String(
                "contract 3.2 and gemm contract section 6's first sentence:"
                " the partition is a pure function of N and the PROFILE,"
                " never of the launch. This arm derives it from OPT_TPB,"
                " which is an EXECUTION constant"
            ),
        )
    if arm == "SCALARS_PER_ELEMENT":
        return ArmExpectation(
            arm, String(""), String("adam_j5_t8_noclip"), String(""),
            String(""), -1, True, False, False,
            String(
                "contract 7.1's ban, and **contract section 12's last row"
                " says the predicted result is NO BITS MOVE**. It"
                " recomputes the host scalars inside the step through the"
                " SAME pinned primitives, so it is a REACH probe and not a"
                " bit probe: it proves the kernel COULD reach the scalars."
                " Its bit result is reported as INERT and is never counted"
                " as a pass. An arm whose predicted answer is 'nothing"
                " happens' is worth having only when it is labelled that"
                " way in advance"
            ),
        )
    if arm == "MOMENTUM_FIRST_STEP":
        return ArmExpectation(
            arm, String("sgd.buf"), String("sgd_damp_t1"),
            String("sgd_nodamp_t1"), String("sgd_mom0_t3"), 1, False,
            False, False,
            String(
                "contract 7.3a. `b_1 = c_damp * g` instead of a COPY."
                " **INERT AT dampening == 0, WHICH IS THE DEFAULT**, where"
                " c_damp is exactly 1.0 and identical_mul(1.0, g) returns"
                " g. sgd_damp_t3 is in the set so the divergence can be"
                " shown to PERSIST rather than wash out"
            ),
        )
    if arm == "NESTEROV_ORDER":
        return ArmExpectation(
            arm, String("sgd.dir"), String("sgd_nesterov_t2"),
            String("sgd_nesterov_t1"), String("sgd_mom0_t3"), 2, False,
            True, False,
            String(
                "contract 7.3c. `b + momentum*g` instead of"
                " `g + momentum*b`. INERT at momentum == 0 AND **at t = 1,"
                " where the buffer is a COPY of g so b == g and the two"
                " readings are the SAME EXPRESSION** -- which is exactly"
                " the case a fixture built for this arm would choose by"
                " accident"
            ),
        )
    if arm == "GATE_RESUME_REINIT":
        return ArmExpectation(
            arm, String("sgd.buf"), String("sgd_damp_t8"), String(""),
            String(""), -1, False, False, True,
            String(
                "contract 7.3b and 11(d). **A GATE-LOCAL ARM (DEVIATION"
                " 1473)**: contract section 12 lists"
                " OPT_SAB_RESUME_REINIT as a switch and `optimizer.mojo`"
                " implements no such switch, because it describes a defect"
                " in a CHECKPOINT FORMAT and not in a kernel. Clause (d)'s"
                " control 2 IS the arm"
            ),
        )
    if arm == "GATE_MICROBATCH_SERIAL":
        return ArmExpectation(
            arm, String(""), String(""), String(""), String(""), -1, False,
            False, True,
            String(
                "contract 9.2 condition 5 and 9.3. **A GATE-LOCAL ARM**:"
                " a serial running accumulation across microbatches instead"
                " of the balanced tree. INERT AT A <= 2 -- over two pieces"
                " the two are the SAME OPERATION, which is why every"
                " aligned case gate G5 measured (all A = 2) cannot see"
                " condition 5. Clause (e) IS the arm and it is always on"
            ),
        )
    raise Error(
        String("optimizer_check: '")
        + arm
        + "' is not an arm this file knows. Contract section 12 lists 22;"
        + " `optimizer.mojo` implements 20 and the two it does not"
        + " (OPT_SAB_RESUME_REINIT and OPT_SAB_MICROBATCH_SERIAL) are"
        + " modelled in this gate under the names GATE_RESUME_REINIT and"
        + " GATE_MICROBATCH_SERIAL (DEVIATION 1473)."
    )


def armed_arm_name() raises -> String:
    """Which arm this binary or this run carries, across the three switch
    files AND the gate-local environment variable.

    An optimizer binary can carry a GEMM sabotage as well -- the clip's
    reductions are DELEGATED to gemm v1 entire, so contract 3.2's clause is
    reachable that way -- so a banner naming only one of them would
    mislabel the run.

    **TWO ARMS AT ONCE IS REFUSED.** A verdict cannot be attributed to a
    seam when two seams are broken, and clause (g)'s whole discipline is
    that an arm moves the stage its OWN seam writes and no earlier one."""
    var o = optimizer_sabotage_name()
    var g = gemm_sabotage_name()
    var b = gemm_backward_sabotage_name()
    var gate = env_str("MOJOLEARN_OPT_GATE_ARM")
    var n_armed = 0
    var name = String("none")
    if o != "none":
        n_armed += 1
        name = o
    if g != "none":
        n_armed += 1
        name = String("G_") + g
    if b != "none":
        n_armed += 1
        name = String("B_") + b
    if gate != "":
        n_armed += 1
        name = gate
    if n_armed > 1:
        raise Error(
            String("optimizer_check: this run carries ")
            + String(n_armed)
            + " arms at once (optimizer='"
            + o
            + "', gemm='"
            + g
            + "', gemm_backward='"
            + b
            + "', gate='"
            + gate
            + "'). Build and run one arm at a time."
        )
    return name^


def clause_g(
    ctx: DeviceContext, arm: String, verdicts: List[CaseVerdict]
) raises:
    """The INVERTED verdict of a sabotage build.

    When an arm is armed this file INVERTS: a clean compare is the FAILURE,
    because it means the sabotage was reached and made no difference, or
    was never reached at all. Both are `[[reached-but-inert]]`.

    **THE ONE EXCEPTION IS `SCALARS_PER_ELEMENT`, AND THE EXCEPTION IS THE
    POINT.** Contract section 12's last row predicts NO BITS MOVE, so for
    that arm this function requires INERTNESS and reports the arm as a
    REACH probe rather than as a pass. `[[sabotage-when-required]]` says a
    sabotage is required when a bound was chosen or a path is new, and
    clause 7.1's ban is a DESIGN rule rather than a numerical one."""
    var exp = arm_expectation(arm)
    print("clause (g): arm " + arm + " -- " + exp.note)

    if exp.is_gate_arm:
        print(
            "clause (g): "
            + arm
            + " is a GATE-LOCAL arm (DEVIATION 1473). It has no `-D` and no"
            " comptime switch in `optimizer.mojo`, because it models a"
            " defect in the CALLER -- a checkpoint that drops a field, or"
            " an accumulation loop that folds serially -- and a switch"
            " inside a kernel could not express either. Its verdict is"
            " produced by clause (d)'s control 2 and by clause (e)"
            " respectively, both of which RAISE if the arm turns out to be"
            " inert. Run those clauses; there is nothing for this function"
            " to invert."
        )
        return

    if exp.needs_record and not OPT_RECORD_INTERMEDIATES:
        raise Error(
            String("optimizer_check: arm ")
            + arm
            + " writes '"
            + exp.first_stage
            + "' FIRST, and that stage is only written when the build"
            + " defines MOJOLEARN_OPT_RECORD (DEVIATION 1476). Without it"
            + " the arm's effect would first be VISIBLE at param.out, which"
            + " is a LATER stage, and reporting that as the arm's first"
            + " stage would be recording an absorbed divergence as an"
            + " aimed one. Rebuild with -D MOJOLEARN_OPT_RECORD=1."
        )

    var wv = find_verdict(verdicts, exp.witness)

    if exp.expect_inert:
        if wv.n_moved != 0:
            raise Error(
                String("optimizer_check: SABOTAGE ")
                + arm
                + " MOVED "
                + String(wv.n_moved)
                + " stages on "
                + exp.witness
                + ", first at step "
                + String(wv.first_step)
                + " ("
                + wv.first
                + "). Contract section 12's last row PREDICTS that it moves"
                + " NOTHING, because it recomputes the host scalars through"
                + " the SAME pinned primitives. **A MOVE HERE IS A FINDING"
                + " ABOUT THE PROFILE**, not a passing sabotage: it means"
                + " the in-kernel spelling and `step_scalars` are not the"
                + " same arithmetic after all."
            )
        print(
            "clause (g): "
            + arm
            + " moved NOTHING on "
            + exp.witness
            + ", which is what contract section 12 predicts. **THIS IS NOT"
            + " A PASS AND IT IS NOT A FAILURE.** It is a REACH probe: it"
            + " shows the scalars are recomputable from inside the step"
            + " through the profile's own primitives, which is what makes"
            + " clause 7.1's ban a DESIGN rule rather than a numerical one."
        )
        return

    if wv.n_moved == 0:
        raise Error(
            String("optimizer_check: SABOTAGE ")
            + arm
            + " IS ARMED AND MOVED NO BIT on its witness case "
            + exp.witness
            + " across all "
            + String(wv.steps)
            + " of its steps. Either its branch was never reached at this"
            + " shape or it is inert there ([[reached-but-inert]]). It"
            + " falsifies NOTHING and must not be reported as a passing"
            + " arm."
        )
    var want = stage_index_of(exp.first_stage)
    if wv.first_index != want:
        raise Error(
            String("optimizer_check: SABOTAGE ")
            + arm
            + " moved '"
            + wv.first
            + "' FIRST on "
            + exp.witness
            + ", and its own seam writes '"
            + exp.first_stage
            + "'. Each arm must move the stage its OWN seam writes and no"
            + " EARLIER one; an earlier stage means the arm is not aimed"
            + " where it says it is, and a LATER one means it was ABSORBED"
            + " on the way and the wrong stage is gating its clause."
        )
    if exp.first_step > 0 and wv.first_step != exp.first_step:
        raise Error(
            String("optimizer_check: SABOTAGE ")
            + arm
            + " first moved at STEP "
            + String(wv.first_step)
            + " on "
            + exp.witness
            + " and it is predicted to first move at step "
            + String(exp.first_step)
            + ". For POW_RUNNING that number is contract 5.1's PREDICTED"
            + " t = 7, which preflight computes; for MOMENT_LERP and"
            + " NESTEROV_ORDER it is step 2, because both are inert at"
            + " step 1 for structural reasons (m_prev is +0.0; b == g)."
            + " **An arm that moves at the wrong step is not doing what"
            + " its clause says, even though it moved.**"
        )
    print(
        "clause (g): "
        + arm
        + " BIT on "
        + exp.witness
        + ": "
        + String(wv.n_moved)
        + " stages moved, FIRST at step "
        + String(wv.first_step)
        + ", "
        + wv.first
    )

    var inerts: List[String] = [exp.inert_a, exp.inert_b]
    var shown = 0
    for i in range(len(inerts)):
        if inerts[i] == "":
            continue
        var iv = find_verdict(verdicts, inerts[i])
        if iv.n_moved != 0:
            raise Error(
                String("optimizer_check: SABOTAGE ")
                + arm
                + " moved "
                + String(iv.n_moved)
                + " stages on "
                + inerts[i]
                + " (first at step "
                + String(iv.first_step)
                + ", "
                + iv.first
                + ") and its clause requires it to move NOTHING there."
                + " **An arm that moves everywhere is a smoke test; the"
                + " inert case is what makes it a REACH PROOF**"
                + " ([[verify-reach-not-output]])."
            )
        shown += 1
        print(
            "clause (g): "
            + arm
            + " is INERT on "
            + inerts[i]
            + ", every stage unmoved across all "
            + String(iv.steps)
            + " steps"
        )
    if shown == 0:
        print(
            "clause (g): "
            + arm
            + " carries NO INERT CASE, so this run is a **SMOKE TEST** for"
            + " it and not a reach proof. That is recorded rather than"
            + " glossed: for RECIP_MUL this file could not construct one"
            + " (the Adam denominator would have to be an exact power of"
            + " two), for MHAT_FORM and RSQRT contract section 12 says none"
            + " is known, and for POW_EXPLOG the arm separates at t = 1 so"
            + " there is nothing to be inert on."
        )


# ===========================================================================


def main() raises:
    var armed = armed_arm_name()

    print(
        "=== optimizer-step identity gate, profile"
        " mojolearn.identical.optimizer.fp32.v1"
    )
    print(
        "=== Fixture-based device-to-oracle comparisons for this build"
        " and device. No independent corpus reference; cross-vendor"
        " identity requires matching recorded runs."
    )
    print(
        "mode "
        + mode_name()
        + "   optimizer sabotage: "
        + optimizer_sabotage_name()
        + "   gemm: "
        + gemm_sabotage_name()
        + "   gemm_backward: "
        + gemm_backward_sabotage_name()
        + "   gate arm: "
        + env_str("MOJOLEARN_OPT_GATE_ARM")
    )
    print(
        "NOTE: contract section 12 lists 22 arms. `optimizer.mojo`"
        " implements 20; OPT_SAB_RESUME_REINIT and OPT_SAB_MICROBATCH_"
        "SERIAL have NO comptime switch and are modelled in this gate"
        " (DEVIATION 1473). Contract section 10 also lists `adam.gwd` and"
        " `adamw.pdec` and **NOTHING PRODUCES EITHER** (DEVIATION 1477)."
    )

    var expect = env_str("MOJOLEARN_OPT_EXPECT_SABOTAGE")
    if expect != "":
        if expect != armed:
            raise Error(
                String("optimizer_check: the caller expected sabotage '")
                + expect
                + "' and this run carries '"
                + armed
                + "'. **A misspelled -D is SILENTLY IGNORED by the"
                + " compiler**: `is_defined` returns False and the build is"
                + " clean, and the operator then records a green gate as"
                + " 'arm X did not bite', which is the exact inverse of the"
                + " truth. That is `tools/gemm_ladder.sh:71`'s scar. Fix"
                + " the -D or the expectation."
            )
        print(
            "ledger: the caller expected '"
            + expect
            + "' and the run agrees, so the -D was not silently dropped"
        )
    elif armed == "none":
        print(
            "ledger: this binary is CLEAN -- no arm is compiled in and no"
            " gate arm is named. Set MOJOLEARN_OPT_EXPECT_SABOTAGE to have"
            " the binary check its own -D."
        )
    else:
        print(
            "ledger: this run carries arm '"
            + armed
            + "' and the caller did not say so. Set"
            " MOJOLEARN_OPT_EXPECT_SABOTAGE to close the misspelled -D"
            " hole."
        )

    preflight()

    var ctx = DeviceContext()

    # ---- CLAUSE (a), BOTH FAMILIES --------------------------------------
    # The Adam table and the SGD table are BOTH run, because eight of the
    # arms have their witness or their inert half in one and the rest in
    # the other, and an arm whose inert half was not in the set cannot be
    # evaluated -- which is NOT the same as the arm passing.
    var cases = clause_a_cases()
    print(
        "clause (a): "
        + String(len(cases))
        + " Adam-family cases and "
        + String(OPT_SGD_CASE_COUNT)
        + " SGD cases, every stage, EVERY STEP, device vs host oracle,"
        " BITWISE"
    )
    var verdicts = List[CaseVerdict]()
    var cpath = card_path()
    var card_cfg = opt_config(opt_case(cases[0]).hp)
    for ci in range(len(cases)):
        var c = opt_case(cases[ci])
        if ci == 0:
            # ONLY THE FIRST CASE WRITES THE CARD, and only at its FIRST
            # STEP. `IdentityTrace` enforces tag uniqueness within one
            # trace, so a card written at every step of every case would
            # emit `opt.param.out` hundreds of times and raise. A per-step
            # prefix would be the alternative and it is deliberately not
            # taken: the card a leg's judge reads must carry contract
            # section 10's tags and nothing else.
            var trace = IdentityTrace.to_path(cpath)
            verdicts.append(clause_a_case(ctx, c, trace, TAG_PREFIX, True))
        else:
            var off_trace = IdentityTrace.disabled()
            verdicts.append(
                clause_a_case(
                    ctx, c, off_trace, "case" + String(cases[ci]), False
                )
            )
    for sk in range(OPT_SGD_CASE_COUNT):
        var off_trace2 = IdentityTrace.disabled()
        verdicts.append(
            clause_a_case(
                ctx, opt_sgd_case(sk), off_trace2, "sgd" + String(sk), False
            )
        )

    _ = check_card_tags(cpath, card_cfg)

    var moved_cases = 0
    var first_case = String("")
    var all_cells = 0
    var poison_total = 0
    for i in range(len(verdicts)):
        all_cells += verdicts[i].cells
        poison_total += verdicts[i].poison
        if verdicts[i].n_moved > 0:
            moved_cases += 1
            if first_case == "":
                first_case = (
                    verdicts[i].name
                    + " at step "
                    + String(verdicts[i].first_step)
                    + ", "
                    + verdicts[i].first
                )

    if armed == "none" and poison_total != 0:
        raise Error(
            String("optimizer_check: ")
            + String(poison_total)
            + " cells came back UNWRITTEN (still holding the poison pattern"
            + " 0x7fc0dead) on a CLEAN build. A stage the card records that"
            + " no kernel wrote is a stage whose hash is whatever the"
            + " allocator left there. If MOJOLEARN_OPT_RECORD is not"
            + " defined this is expected for adam.denom, adam.q and"
            + " sgd.dir -- and `stage_present` already excludes those, so a"
            + " survivor here is a DIFFERENT stage (DEVIATION 1476)."
        )

    if armed != "none":
        clause_g(ctx, armed, verdicts)
        print(
            "clauses (b), (c), (d) and (f) are NOT run under an armed"
            " build: (b), (c) and (d) are INVARIANCE claims and a"
            " DETERMINISTIC sabotage satisfies all three, and (f)'s refusal"
            " is upstream of every sabotaged seam. Clause (e) IS run,"
            " because it is host-only and because its own arm is one of the"
            " two gate-local ones."
        )
        clause_e()
        return

    if moved_cases != 0:
        if mode_is_identical():
            raise Error(
                String("optimizer_check: CLAUSE (a) FAILED on ")
                + String(moved_cases)
                + " of "
                + String(len(verdicts))
                + " cases, first at "
                + first_case
            )
        else:
            # FAST arms of (a) are RECORDED, not asserted, where they are
            # vendor-shaped -- contract 11(a), and the metrics lane's
            # leg-11 lesson. Under FAST every `identical_*` compiles away
            # to the platform's own spelling, and `identical_sqrt` in
            # particular becomes the APPROXIMATE PTX sqrt on NVIDIA
            # (DEVIATION 258, 180,714 of 2^20 patterns off by one ulp), so
            # a difference here is EXPECTED and means nothing about the
            # profile.
            print(
                "clause (a) [FAST]: RECORDED, NOT ASSERTED. "
                + String(moved_cases)
                + " of "
                + String(len(verdicts))
                + " cases differ from the oracle, first at "
                + first_case
                + ". FAST is unversioned and makes no identity claim."
            )
    else:
        print(
            "clause (a): PASS, "
            + String(len(verdicts))
            + " cases bit-identical to the oracle across every step, on all "
            + String(all_cells)
            + " compared cells"
        )

    # ---- CLAUSE (e) IS ALWAYS ON ---------------------------------------
    clause_e()

    if env_on("MOJOLEARN_OPT_CHECK_CLAUSE_B"):
        clause_b(ctx, opt_case_by_name(String("adam_t8")))
    else:
        print("clause (b): SKIPPED (set MOJOLEARN_OPT_CHECK_CLAUSE_B=1)")

    if env_on("MOJOLEARN_OPT_CHECK_CLAUSE_C"):
        var ck = env_str("MOJOLEARN_OPT_CHECK_C_CASE")
        var kk = opt_case_by_name(String("adam_j5_t8_noclip"))
        if ck != "":
            kk = opt_case_by_name(ck)
        clause_c(ctx, kk)
    else:
        print(
            "clause (c): SKIPPED (set MOJOLEARN_OPT_CHECK_CLAUSE_C=1). Its"
            " default case is adam_j5_t8_NOCLIP and not the plain one,"
            " because contract 3.5 says parameter-count invariance is FALSE"
            " with a global norm clip on -- and this clause REFUSES a"
            " clipping-on case by name rather than quietly measuring"
            " something else."
        )

    if env_on("MOJOLEARN_OPT_CHECK_CLAUSE_D"):
        var dk = env_str("MOJOLEARN_OPT_CHECK_D_CASE")
        var dcase = opt_case(opt_case_by_name(String("adam_t8")))
        if dk != "":
            dcase = opt_case(opt_case_by_name(dk))
        _ = clause_d(ctx, dcase)
        # The SGD half, which is the ONLY place control 2 -- the dropped
        # momentum flag, i.e. GATE_RESUME_REINIT -- can run at all.
        # sgd_damp_t8 and not sgd_damp_t3: clause (d) runs `2t` steps
        # against a checkpoint at `t` and contract 11(d) fixes `t = 8` at
        # minimum, because contract 5.1's two `beta^t` spellings agree
        # through `t = 6`. The three-step case would be REFUSED by
        # clause (d) by name, which is the right behavior and is why the
        # eight-step SGD case exists.
        _ = clause_d(ctx, opt_sgd_case(opt_sgd_case_by_name(
            String("sgd_damp_t8")
        )))
    else:
        print(
            "clause (d): SKIPPED (set MOJOLEARN_OPT_CHECK_CLAUSE_D=1)."
            " NOTE: its control 2 IS the GATE_RESUME_REINIT arm (DEVIATION"
            " 1473), so a leg that skips clause (d) has one fewer arm and"
            " not merely one fewer clause."
        )

    if env_on("MOJOLEARN_OPT_CHECK_CLAUSE_F"):
        clause_f(ctx)
    else:
        print(
            "clause (f): SKIPPED (set MOJOLEARN_OPT_CHECK_CLAUSE_F=1)."
            " NOTE: it carries the MEASURED finding that the DEVICE path"
            " refuses only `clip.total_norm` and does not refuse at all"
            " when clipping is off (DEVIATION 1478), so a leg that skips it"
            " has not looked at the device-side refusal even once."
        )

    print(
        "SCOPE: this build, this column, "
        + mode_name()
        + " only. What is NOT closed by anything printed above: an"
        " INDEPENDENT reference (contract 16.10's corpus does not exist, so"
        " every clause here is our device against our oracle -- and seven"
        " of the twenty-three stages are the SAME HOST CODE on both sides,"
        " because `device_step_scalars` calls the oracle's own"
        " `step_scalars` on the clean path); **PyTorch parity, which"
        " contract 5.2 and 5.3 make IMPOSSIBLE by construction and which"
        " nothing here should be read as claiming**; the GEOMETRY half of"
        " clause (b) (OPT_TPB is a comptime literal -- DEVIATION 1479);"
        " the thousand-step case unless MOJOLEARN_OPT_CHECK_T1000 was set;"
        " FAST mode; every column that is not this one; and every arm this"
        " run is not. Contract 14.3: nothing cross-vendor until a leg runs."
        " The v1 GEMM this contract DELEGATES its reductions to HAS run on"
        " three vendors at leg 11 (144aa5b) -- **that measurement is the"
        " GEMM's, not this lane's**."
    )
