# SPDX-License-Identifier: Apache-2.0
"""BLOCK STEP OPS of the operation session (lane x1-executor-state-and-step-ops,
2026-11-08; plan EXECUTOR_CONSOLIDATION_AND_CLEANUP (git history) 2.1, the API note
plan EXECUTOR_X1_API (git history)).

Four session operation codes run one block of a recurrent or attention model
over tokens, as ONE node of a session or a captured program:

  OPS_M1_STEP   81  Mamba-1 block      hs [x, weights, state]           ints [b, d_model]
  OPS_M2_STEP   82  Mamba-2 block      hs [x, weights, state, words]    ints [b, d_model]   floats [dt_lo, dt_hi]
  OPS_M3_STEP   83  Mamba-3 block      hs [x, weights, state, words]    ints [b, d_model]
  OPS_ATTN_STEP 84  Llama-style block  hs [x, weights, kv, words]       ints [b, d_model, n_heads, n_kv,
                    with a KV cache                                           head_dim, intermediate, s_max,
                                                                              window, (17 option words)]

`x` holds b * L * d_model values (any shape with that count; the result has
x's shape): L = 1 is a decode step, L > 1 a prefill or a chunked
continuation, the same op. `weights` is ONE uploaded session tensor holding
the block's weights packed in `block_weight_sizes` order (each piece starts at
a `block_slot` boundary), read IN PLACE at every run: the program's parameter
is the resident copy. `state` / `kv` is ONE session tensor declared state
(`OpsSession.declare_state`) holding the carried float state packed the same
way (`block_state_sizes`), written IN PLACE by the op; `words` is a state
tensor of `block_int_words` int32 words holding the host-side integers of the
carried state (Mamba-2 [buf_len], Mamba-3 [buf_len, pending], attention
[cached tokens]). `OpsSession.upload_block` packs a block's host arrays into
these tensors.

The bodies are the resident decode sessions' (mamba/impl/mamba_shell_impl.mojo
`_m1_session_step_run`, `_m2_session_step_run`, `_m3_session_step_run`;
transformer/transformer_binding_impl.mojo `_session_step_run_sync` and
`_session_forward_run`) with the copies to and from the caller's host arrays
replaced by device copies to and from session tensors: the same structs (built
over views of the session tensors through `MambaArena(over=...)` and the
options constructor of `LlamaDeviceWeights`), the same block forward entry,
the same launch set and order, so the same words. The structs are built once
per (weights, state) pair and kept in `BlockStepScratch`, so the block's
first-call weight refusal runs once, as in a session; a write to the weights
or the state from outside a step (`load_into`, an optimizer `step`, a new
`declare_state`) drops the entry, and the next run rebuilds it and reads the
integer words back once.

Dispatch is one table (`comptime for` over the codes), not one more `elif` per
op in `OpsSession._op_run_tensors` (lessons 8, 12 of NATIVE_ENTRY_POINTS_AFTER_0866).

The host arm (lane x7b-step-ops-host-arm, 2026-11-09): an `OpsSession` on the
host executor runs the same four ops through the host block bodies, the
oracles the CPU identity column runs (`transformer_block_oracle` and
`transformer_block_backward_oracle`, `mamba_block_oracle`,
`mamba2_block_oracle`, `mamba3_block_oracle`), over the SAME packed
layouts: the weights are read from the weights tensor at its `block_offsets`
slots at every run, the carried state is read from the state tensor and
written back after a successful run, the integers are the words tensor's.
So one captured program runs on the GPU columns and the host column with
one algorithm. The host arm lives in autograd/block_step_host.mojo (a host
module: no GPU binding imports it) and its entries in the host executor
(`HostExec.steps`, `HostStepScratch`); a host session reaches it through the
executor (`Exec.step_run`, `step_backward`, `step_forget`); it keeps only the
rotary table, a training run's stages and the reports. Every
cross-row fold inside the bodies is a gemm v1 call and every per-row stage
runs over host tasks, so a run uses every core; consecutive layers are
dependent and run in order.

Recording: in a recording session (`OpsSession._block_step`, lanes x7 and
x56d) every step op is the zero-state prefill with its stages kept in the
pair's entry, recorded as ONE custom node over [x, weights]
(`attn_step_train_run`, `mamba_step_train_run` below); `OpsSession.backward`
differentiates it through StepOpRules (autograd/rules_step_ops.mojo). The
packed weights take a gradient when made with
`OpsSession.upload_block_grad(..., requires_grad=True)`.

Public names (lane fx-ops-session-blocks, 2026-11-18): `Session.mamba_block`
/ `Session.mamba_step` (codes 81-83) and `Session.attention_block` /
`Session.attention_step` (code 84, the default block options), through
autograd/ops_block_frontend.mojo.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from core.identity_trace import IdentityTrace
from core.hostptr import f32_ptr
from mamba.checks.mamba_fixture import MambaDims
from mamba.checks.mamba2_fixture import Mamba2Dims, M2_CHUNK_SIZE, M2_HEADDIM, M2_D_STATE
from mamba.checks.mamba3_fixture import Mamba3Dims, M3_CHUNK_SIZE, M3_HEADDIM, M3_D_STATE, M3_NUM_ROPE_ANGLES
from mamba.impl.modules.afn_arena import MambaArena
from mamba.impl.modeling.modeling_mamba import (
    MambaDeviceWeights, MambaDeviceState, MambaDeviceStages, mamba_block_forward, mamba_zeros, MAMBA_GUARD,
    _m1_weight_sizes, _m1_state_sizes, _m1_stage_sizes,
)
from mamba.impl.modules.mamba_simple import mamba_step
from mamba.impl.modules.mamba2 import (
    Mamba2DeviceWeights, Mamba2DeviceState, Mamba2DeviceStages, mamba2_block_forward,
    _m2_weight_sizes, _m2_state_sizes, _m2_stage_sizes,
)
from mamba.impl.modules.mamba3 import (
    Mamba3DeviceWeights, Mamba3DeviceState, Mamba3DeviceStages, mamba3_block_forward,
    _m3_weight_sizes, _m3_state_sizes, _m3_stage_sizes,
)
from transformer.impl.llama.modeling_llama import (
    LlamaDims, LlamaDeviceWeights, LlamaKVCache, LlamaRopeTable, LlamaDeviceStages, llama_decoder_layer_forward,
    residual_next_norm_fusion_enabled, LLAMA_FAST_REFUSE_WATERMARK, llama_kv_cache_rewind,
)
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_APPLE
from transformer.block_options import BlockOptions, BLOCK_OPTION_PARAMS
from transformer.lean_stages import _lean_for
from autograd.kernels.transformer import LlamaBackwardStages
from autograd.rules_transformer import llama_decoder_layer_backward_autograd
from autograd.rules_mamba import (
    Mamba1HandBuffers, Mamba1TapeStack, Mamba2TapeStack, mamba1_step_forward, mamba1_step_backward, mamba2_step_forward,
    mamba2_step_backward,
)
from autograd.rules_mamba3 import mamba3_step_backward
from mamba.impl.modules.afn_defines import IDN_M3_SESSION_STAGE_REUSE, IDN_MAMBA3_REPORTS_ON_REQUEST

# Session operation codes (autograd/ops_api.mojo OPS_*; autograd/program.mojo
# PROGRAM_MAX_CODE covers them, record version 3).
comptime OPS_M1_STEP = 81
comptime OPS_M2_STEP = 82
comptime OPS_M3_STEP = 83
comptime OPS_ATTN_STEP = 84
comptime OPS_STEP_FIRST = OPS_M1_STEP
comptime OPS_STEP_LAST = OPS_ATTN_STEP
comptime N_STEP_OPS = OPS_STEP_LAST - OPS_STEP_FIRST + 1

# The parts of a block that `OpsSession.upload_block` packs.
comptime BLOCK_PART_WEIGHTS = 0
comptime BLOCK_PART_STATE = 1
comptime BLOCK_PART_WORDS = 2

# The attention op's own integer arguments before the options tail.
comptime ATTN_STEP_INTS = 8

# TOMBSTONE: MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE (unmeasured opt-in arm) deleted 2026-11-15 by lane/expdel2-23-neural; code recoverable at 74754fe939.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE.patch; record in docs/TOMBSTONES.md.

#: Lane restore-neural-call-paths (2026-11-13; AGENTS.md "A measured winner
#: is never deleted on reasoning"): the per-call Mamba door's call path.
#: Default (A): the block reads the caller's x in place from the op's x
#: tensor (the Mamba-1/2/3 block forwards only read x: "residual =
#: hidden_states, a NAME, not a copy", modeling_mamba.mojo
#: mamba_block_forward, and the S22 / out_proj residual of Mamba-2/3), and a
#: per-call door (`mamba_step_run_host`, MambaBlockProgram.run) downloads the
#: block output straight from the stages' `residual_out` into the caller's
#: array: one upload, the block, one download, one wait, as 0.8.55's
#: `_mamba1_run` / `_mamba2_run` (bindings/_mojolearn_mamba.mojo at
#: 0e4de0f46e:354 / :920) did. Board on record (NV L40S, candidate
#: 4c2340386abc vs 0.8.55): mamba3-forward 3.95 -> 6.75 ms, mamba2-forward
#: 5.57 -> 7.58, mamba1-forward 3.43 -> 4.08 (AMD MI325X 2.91 / 4.25 / 3.29).
#: A poison build (MAMBA_GUARD > 0) keeps the copy into a zeroed dx: the
#: guard band lives only in `mamba_zeros` buffers.
# TOMBSTONE: MOJOLEARN_MAMBA_CALL_PROGRAM (unmeasured opt-in arm) deleted 2026-11-15 by lane/expdel2-23-neural; code recoverable at ea4327182b.
# Restore: git apply experiments/removed/MOJOLEARN_MAMBA_CALL_PROGRAM.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
# TOMBSTONE: MOJOLEARN_MAMBA_CALL_PROGRAM (removed without measurement: plumbing, rule 2026-11-22; lane x5's route: x copied D2D into a zeroed dx, the output D2D into the op's y tensor then downloaded; same kernels, same input bytes) deleted 2026-11-23 by lane/ab-prep-mamba; code recoverable at 4d76a9e16e.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_MAMBA_CALL_PROGRAM.ab-prep.patch; record in docs/TOMBSTONES.md.
comptime _MAMBA_X_IN_PLACE = MAMBA_GUARD == 0

#: Lane fix-mamba12-forward-exec (2026-11-13): what this switch decides is
#: whether the Mamba-1/2 per-call forward pays per-call plumbing the kernels
#: never needed. Default (A): a session whose `BlockStepScratch.keep_stages`
#: is set (the per-call block programs, mamba/impl/mamba_block_program.mojo)
#: keeps its last L > 1 Mamba-1/2 stages in the resident entry and reuses
#: them for the next run of the same (B, L[, buf_len]), refilled with the
#: constructor's zeros (`rezero`): no 19 / 28 allocations and, for Mamba-1,
#: no constructor wait per call (0.8.55's `_mamba1_run` / `_mamba2_run` also
#: built them per call; the held-stages pattern is the Samba resident lane's
#: IDN_M3_SESSION_STAGE_REUSE). A held Mamba-2 run's h_last report is read
#: straight from the held stages (no device-to-device copy into the entry's
#: report buffer). The public doors run a stateless forward as the fresh
#: (discard-state) call: one device zero fill of the state tensor instead of
#: a host zero state uploaded and downloaded per call, and the door's y (and
#: Mamba-2 h_last) arrays are allocated unfilled (bindings/
#: causal_lm_public_door.mojo, bindings/neural_block_public_door.mojo,
#: mamba/impl/mamba_block_program.mojo). Same kernels, same launch order,
#: same input bytes: no output bit moves on any vendor. A poison build
#: (MAMBA_GUARD > 0) keeps the per-call stages (its fills cover a body view
#: only).
# TOMBSTONE: MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-23-neural; code recoverable at 449d72a027.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
# TOMBSTONE: MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD (removed without measurement: plumbing, rule 2026-11-22) deleted 2026-11-23 by lane/ab-prep-mamba; code recoverable at 4d76a9e16e (per-call stages, the h_last D2D copy, the host zero state round trip and zero-filled door arrays around the same kernels in the same launch order).
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD.ab-prep.patch; record in docs/TOMBSTONES.md.
comptime _M12_HOLD = _MAMBA_X_IN_PLACE

#: Lane fix-mamba-forward-doors (2026-11-14): what this switch decides is how
#: a held Mamba-1/2 stage set (`_M12_HOLD` above) is built and re-zeroed.
#: Default (A): 0.8.55's arena form (`_mamba1_run_arena` /
#: `_mamba2_run_arena`, bindings/_mojolearn_mamba.mojo at 0e4de0f46e:422 /
#: :1018, on in identical device builds through IDN_MAMBA_ARENA): the held
#: set is views of ONE held `MambaArena` (one `enqueue_create_buffer` and one
#: fill when built, no wait), and a reuse re-zeroes it with ONE
#: `enqueue_fill` of the arena instead of 19 (Mamba-1) / 28 (Mamba-2) fills,
#: one per stage. A reused Mamba-2 run of a fresh (discard-state) call also
#: skips the post-run buf_len word write (`BlockStepScratch.skip_words`: the
#: per-call program rewrites the word before any reader,
#: mamba/impl/mamba_block_program.mojo `zero_state` / `load_state`). Same
#: kernels, same launch order, same input bytes and the same zeros under
#: every stage: no output bit moves on any vendor or the host column. The
#: same switch also decides the stateless forward's door
#: (bindings/causal_lm_public_door.mojo `causal_mamba_stateless_run`: one
#: direct call from the neural door with a held table entry). Every held
#: set (`_M12_HOLD`) is an arena set.
# TOMBSTONE: MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-23-neural; code recoverable at ddf9b022e1.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
# TOMBSTONE: MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD (removed without measurement: plumbing, rule 2026-11-22; lane fix-mamba12-forward-exec's held form: separately allocated stages, one fill per stage on reuse, the word upload and write on every fresh call, the four-hop door chain; same kernels, same launch order, same zeros) deleted 2026-11-23 by lane/ab-prep-mamba; code recoverable at 4d76a9e16e.
# Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD.ab-prep.patch; record in docs/TOMBSTONES.md.

#: Lane fx-mamba-decode-arena (2026-11-18, ideas-neural-speed ideas 2 and 7):
#: what this switch decides is whether a Mamba-2 / Mamba-3 decode token (an
#: L = 1 run) allocates, zero-fills and frees its stage buffers. The old form
#: (now only a guard-band or poison build): 28 (Mamba-2) / 33
#: (Mamba-3) `mamba_zeros` per token, parked and freed at the caller's wait.
#: THE DEFAULT since 2026-11-19 (the experiment
#: `MOJOLEARN_MAMBA_FAST_DECODE_STAGE_ARENA`, promoted; numbers beside the
#: constant below): the resident entry holds ONE `MambaArena` with
#: room for the stages at the largest carried buf_len the state contract
#: allows (`_m2_check_words` / `_m3_check_words`: a full open chunk; the
#: extent comes from the contract, never from a board row). Each token
#: re-takes its stage views at the current (B, 1, buf_len), so every view
#: has the length the per-call buffer had, and queues ONE fill of the used
#: prefix (`MambaArena.reset`) ahead of the token's launches on the same
#: in-order queue. No stale value: every word of every stage view lies in
#: that prefix and is zero when the token's first kernel runs, exactly as
#: under `mamba_zeros`; the previous token's readers (the output, report and
#: state copies) were queued before the fill. Same kernels, same launch
#: order, same input bytes, same zeros: no output bit moves on any vendor or
#: the host column (the host column never runs this file's device stages).
#: The same switch (idea 7) builds the held Mamba-3 L > 1 stages
#: (`inference_only`, and under this switch also a `keep_stages` session,
#: as Mamba-1/2 hold theirs) as views of one held arena, re-zeroed with ONE
#: fill instead of `rezero`'s 33. Off in a poison build (its fills cover a
#: body view and its band is per allocation).
#: Not NI47 (`MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE`, deleted unmeasured
#: 2026-11-15, docs/TOMBSTONES.md): NI47 kept 33 separately allocated
#: capacity-sized buffers, rebound their logical window and re-zeroed them
#: with 33 fills per token (so buffer lengths differed from the per-call
#: form), Mamba-3 only. Here: one allocation, exact-length views, one fill,
#: Mamba-2 and Mamba-3, and the held L > 1 sets.
# MOJOLEARN_MAMBA_FAST_DECODE_STAGE_ARENA: PROMOTED 2026-11-19 (lane fx-verdicts3). The arena is the default on
# every column that runs these device stages (confirmed and its _OFF switch deleted 2026-11-20, round 3 below;
# the old opt-in name and _OFF are refused, core/experiment_guards_neural.mojo; the constant
# keeps its experiment name). Evidence: A/B round 1, NVIDIA L40S (host bd9c355c19c0, 14-core cgroup), wheel 0c68d6cd27
# and its B set, base and arm back to back in ONE pod job (nvc1-0006, tag nvc-ab-20261119T125554Z-6f7afc), one scored
# run per arm; digests and quality equal on all 6 cells. Base -> arm ms: mamba3-decode 1066.9 -> 849.3 (0.80),
# mamba2-decode 1251.9 -> 1053.2 (0.84), samba-decode 1433.0 -> 1315.8 (0.92), mamba1-decode 373.0 -> 343.3 (0.92),
# mamba3-forward 2.327 -> 2.339 (1.01, median of 20), samba-forward 4.649 -> 4.586 (0.99, median of 20).
# How to read it: mamba1-decode is the control (no code path of it changes) and it moved 0.92, so this pair carries
# about 8 % of drift toward the arm; net of the control mamba3-decode is about 0.87 and mamba2-decode 0.91, and
# samba-decode is not separable from the drift. The two rows the arm was written for are faster beyond that drift;
# the held L > 1 arena (the forwards) is neither faster nor slower. Less than the -400 ms a row the driver-call
# arithmetic expected: re-time owed. Same kernels, launch order, input bytes and zeros: no bit moves.
# General mechanism: the arena extent comes from the state contract's largest carried buf_len, never from a shape.
# Limits: NVIDIA only (AGENTS.md part 8), one run per arm. OWED (does not hold the flip): the neighbouring shape
# (d_model 256, B 4) and a non-board model (a 6-layer Samba at L 300) against the _OFF arm, with mamba1-decode again.
# A/B round 3, 2026-11-20 (lane fx-verdicts5; NVIDIA L40S host 2ef132a6de04, wheel 28a780604c and its _OFF set, base
# and arm back to back in one pod job, tag nvc-ab-20261120T063324Z-c630ad; digests and quality equal on all 6 cells).
# Arena (default) -> per-token stages ms: mamba3-decode 855.7 -> 1,063.3 (1.243), mamba2-decode 1,062.2 -> 1,245.6
# (1.173), samba-decode 1,312.8 -> 1,431.8 (1.091; its other base in the job 1,302.9); the control mamba1-decode
# 348.5 -> 351.0 (1.007: no drift in this pair); mamba3-forward 2.324 -> 2.325 and samba-forward 4.570 -> 4.575
# (1.001, medians of 20; samba-forward's three bases 4.561 to 4.570). CONFIRMED a second time with a still control:
# the arena stands and its switch is deleted. Not timed: the d_model 256 and 6-layer Samba cases named above.
# The per-token stages themselves stay for a guard-band or poison build (below).
# TOMBSTONE: MOJOLEARN_MAMBA_FAST_DECODE_STAGE_ARENA_OFF (old path's switch; the promotion confirmed by A/B rounds 1 and 3, numbers above) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
# Restore: git apply experiments/removed/MOJOLEARN_MAMBA_FAST_DECODE_STAGE_ARENA_OFF.patch; record in docs/TOMBSTONES.md.
comptime MAMBA_FAST_DECODE_STAGE_ARENA = (
    MAMBA_GUARD == 0
    and not is_defined["MOJOLEARN_MAMBA_POISON"]()
)


def m2_report_floats(b: Int, dm: Int) raises -> Int:
    """The Mamba-2 h_last report: b * heads * 64 * 128 floats."""
    return b * Mamba2Dims.of(dm).nheads * M2_HEADDIM * M2_D_STATE


def m3_report_floats(b: Int, dm: Int) raises -> List[Int]:
    """The four Mamba-3 reports (h_last, k_last, v_last, theta_last), in
    that order, as float counts."""
    var nh = Mamba3Dims.of(dm).nheads
    var out: List[Int] = [b * nh * M3_HEADDIM * M3_D_STATE, b * nh * M3_D_STATE, b * nh * M3_HEADDIM,
                          b * nh * M3_NUM_ROPE_ANGLES]
    return out^


def is_block_step(code: Int) -> Bool:
    return code >= OPS_STEP_FIRST and code <= OPS_STEP_LAST


struct StepOpInfo(ImplicitlyCopyable, Movable):
    """One row of the step-op table: what the op is called and what it takes."""
    var code: Int
    var name: StaticString
    #: tensor operands: x, weights, float state, and (when n_words > 0) words
    var n_tensors: Int
    #: int32 words of the carried state's host integers
    var n_words: Int
    #: float arguments
    var n_floats: Int

    def __init__(out self, code: Int, name: StaticString, n_tensors: Int, n_words: Int, n_floats: Int):
        self.code = code
        self.name = name
        self.n_tensors = n_tensors
        self.n_words = n_words
        self.n_floats = n_floats


def step_op_info(code: Int) raises -> StepOpInfo:
    """The table row of step op `code` (program_tools' names come from here)."""
    comptime for i in range(N_STEP_OPS):  # driver-loop: over the step-op table (four rows, compile time)
        comptime CODE = OPS_STEP_FIRST + i
        if code == CODE:
            return _info[CODE]()
    raise Error("ops: " + String(code) + " is not a block step op")


def _info[CODE: Int]() -> StepOpInfo:
    comptime if CODE == OPS_M1_STEP:
        return StepOpInfo(CODE, "mamba1_step", 3, 0, 0)
    elif CODE == OPS_M2_STEP:
        return StepOpInfo(CODE, "mamba2_step", 4, 1, 2)
    elif CODE == OPS_M3_STEP:
        return StepOpInfo(CODE, "mamba3_step", 4, 2, 0)
    else:
        return StepOpInfo(CODE, "attention_step", 4, 1, 0)


def block_int_words(code: Int) raises -> Int:
    return step_op_info(code).n_words


# ---------------------------------------------------------------- layouts
def block_guard(code: Int) -> Int:
    """The guard band after every piece: the Mamba poison build's band (0 in
    production); none for attention."""
    if code == OPS_ATTN_STEP:
        return 0
    return MAMBA_GUARD


def block_slot(n: Int, guard: Int) -> Int:
    """The floats one packed piece of logical length n takes (MambaArena's
    rule: at least one element, the guard band, 256-byte aligned)."""
    return MambaArena.slot(n, guard)


def block_offsets(sizes: List[Int], guard: Int) -> List[Int]:
    """Start of every piece and, last, the packed total."""
    var out = List[Int](capacity=len(sizes) + 1)
    var at = 0
    for i in range(len(sizes)):  # small-loop(sizes: pieces of one block): layout arithmetic, no data
        out.append(at)
        at += block_slot(sizes[i], guard)
    out.append(at)
    return out^


def _check_ints(code: Int, ints: List[Int]) raises:
    if code == OPS_ATTN_STEP:
        if len(ints) != ATTN_STEP_INTS and len(ints) != ATTN_STEP_INTS + BLOCK_OPTION_PARAMS:
            raise Error("ValueError: ops: attention_step takes " + String(ATTN_STEP_INTS) + " integer arguments"
                        + " (b, d_model, n_heads, n_kv, head_dim, intermediate, s_max, window) or those and the "
                        + String(BLOCK_OPTION_PARAMS) + " block option words")
    elif len(ints) != 2:
        raise Error("ValueError: ops: " + String(step_op_info(code).name) + " takes two integer arguments (b, d_model)")
    if ints[0] < 1 or ints[1] < 1:
        raise Error("ValueError: ops: a block step needs a positive batch and d_model")


def _attn_dims(ints: List[Int]) raises -> LlamaDims:
    var dims = LlamaDims(ints[1], ints[2], ints[3], ints[4], ints[5])
    dims.validate()
    return dims^


def _attn_opts(ints: List[Int]) raises -> BlockOptions:
    var opts = BlockOptions.from_params(ints, ATTN_STEP_INTS)
    opts.validate(ints[4])
    return opts^


def _attn_cache_floats(ints: List[Int]) raises -> Int:
    """One of the two caches: b * n_kv * capacity * head_dim, capacity =
    window, or s_max for a full causal cache."""
    var smax = ints[6]
    var window = ints[7]
    if smax < 1 or window < 0:
        raise Error("ValueError: ops: attention_step needs s_max >= 1 and window >= 0")
    var cap = window if window > 0 else smax
    return ints[0] * ints[3] * cap * ints[4]


def block_weight_sizes(code: Int, ints: List[Int]) raises -> List[Int]:
    """Logical length of every weight piece, in packing order: the block
    struct's field order (Mamba `_m*_weight_sizes`; attention the options
    constructor's twenty, an absent optional piece 0)."""
    _check_ints(code, ints)
    if code == OPS_M1_STEP:
        return _m1_weight_sizes(MambaDims.of(ints[1]))
    if code == OPS_M2_STEP:
        return _m2_weight_sizes(Mamba2Dims.of(ints[1]))
    if code == OPS_M3_STEP:
        return _m3_weight_sizes(Mamba3Dims.of(ints[1]))
    var dims = _attn_dims(ints)
    var opts = _attn_opts(ints)
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var hd = dims.head_dim
    var s: List[Int] = [
        dm, dm, qw * dm, kw * dm, kw * dm, dm * qw, it * dm if opts.gated() else 0, it * dm, dm * it,
        qw if opts.qkv_bias else 0, kw if opts.qkv_bias else 0, kw if opts.qkv_bias else 0,
        dm if opts.o_bias else 0, dm if opts.norm_bias else 0, dm if opts.norm_bias else 0,
        it if opts.mlp_bias else 0, dm if opts.mlp_bias else 0, it if opts.has_gate_bias() else 0,
        hd if opts.qk_norm else 0, hd if opts.qk_norm else 0,
    ]
    return s^


def block_state_sizes(code: Int, ints: List[Int]) raises -> List[Int]:
    """Logical length of every float state piece, in packing order (Mamba
    `_m*_state_sizes`; attention [keys, values])."""
    _check_ints(code, ints)
    if code == OPS_M1_STEP:
        return _m1_state_sizes(ints[0], MambaDims.of(ints[1]))
    if code == OPS_M2_STEP:
        return _m2_state_sizes(ints[0], Mamba2Dims.of(ints[1]))
    if code == OPS_M3_STEP:
        return _m3_state_sizes(ints[0], Mamba3Dims.of(ints[1]))
    var n = _attn_cache_floats(ints)
    var s: List[Int] = [n, n]
    return s^


def block_part_floats(code: Int, ints: List[Int], part: Int) raises -> Int:
    """The float count of the session tensor holding `part` of a block."""
    if part == BLOCK_PART_WORDS:
        return block_int_words(code)
    var sizes = block_weight_sizes(code, ints) if part == BLOCK_PART_WEIGHTS else block_state_sizes(code, ints)
    var offs = block_offsets(sizes, block_guard(code))
    return offs[len(offs) - 1]


# ---------------------------------------------------------------- device views
def _view(ex: DeviceExec, p: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
    """The executor's storage [p, p + n) as a device buffer (no copy)."""
    var found = ex._find(p, n)
    return ex._sub(found[0], found[1], n)


def _d2d(ctx: DeviceContext, dst: DeviceBuffer[DType.float32], src: DeviceBuffer[DType.float32], n: Int) raises:
    """dst[0:n] = src[0:n], device to device (autograd/rules_mamba.mojo _d2d)."""
    if n <= 0:
        return
    ctx.enqueue_copy(
        dst_buf=dst.create_sub_buffer[DType.float32](0, n),
        src_buf=src.create_sub_buffer[DType.float32](0, n),
    )


def _read_words(mut ex: DeviceExec, p: FP, k: Int) raises -> List[Int]:
    """The k int32 words at p, read back once (a new scratch entry only)."""
    var host = List[Float32](length=k, fill=Float32(0))
    ex.download(FP(unsafe_from_address=Int(host.unsafe_ptr())), p, k)
    var out = List[Int](capacity=k)
    for i in range(k):  # small-loop(k: at most two state words): word decode, no data
        out.append(Int(bitcast[DType.int32](host[i])))
    return out^


def _m2_check_words(words: List[Int]) raises:
    """The Mamba-2 carried integer: buf_len in [0, CHUNK) (contract section 5:
    the open-chunk buffer holds at most CHUNK_SIZE - 1 rows between calls)."""
    if len(words) != 1 or words[0] < 0 or words[0] >= M2_CHUNK_SIZE:
        raise Error("ValueError: mamba2: buf_len must be in [0, " + String(M2_CHUNK_SIZE) + "), got "
                    + (String(words[0]) if len(words) > 0 else String("nothing"))
                    + "; the two sides of this boundary disagree about the state")


def _m3_check_words(words: List[Int]) raises:
    """The Mamba-3 carried integers [buf_len, pending] (`_m3_session_copy_in`'s
    checks, word for word): buf_len in [0, Q] inclusive (the buffer never
    empties, DEVIATION 832(i)), pending 0 or 1, and a pending Input_States
    continuation only on a fresh state."""
    if len(words) != 2:
        raise Error("ValueError: mamba3: the state carries buf_len and pending")
    var q0 = words[0]
    var pend = words[1]
    if q0 < 0 or q0 > M3_CHUNK_SIZE:
        raise Error("ValueError: mamba3: buf_len must be in [0, " + String(M3_CHUNK_SIZE) + "] (INCLUSIVE -- the "
                    + "buffer never empties, DEVIATION 832(i)), got " + String(q0)
                    + "; the two sides of this boundary disagree about the state")
    if pend != 0 and pend != 1:
        raise Error("ValueError: mamba3: pending must be 0 or 1, got " + String(pend)
                    + "; the two sides of this boundary disagree about the state")
    if pend == 1 and q0 != 0:
        raise Error("ValueError: mamba3: Input_States pending requires a fresh state (buf_len 0)")


def _write_words(mut ex: DeviceExec, p: FP, words: List[Int]) raises:
    """Queue the carried state's host integers into their int32 words: one
    device fill of each word's bit pattern (lane x4b: was an executor
    upload, which stages a host buffer per call that only the executor's
    next `sync` frees; a decode stack that never syncs its layers' own
    executors grew one per token per layer). No wait, no host staging."""
    for i in range(len(words)):  # small-loop(words: at most two state words): word encode, no data
        var cell = _view(ex, p + i, 1)
        cell.enqueue_fill(bitcast[DType.float32](Int32(words[i])))


# ---------------------------------------------------------------- resident entries
struct _M1Block(Movable):
    var w_addr: Int
    var s_addr: Int
    var b: Int
    var dm: Int
    var arena_w: MambaArena
    var arena_s: MambaArena
    var w: MambaDeviceWeights
    var st: MambaDeviceState
    #: the L = 1 stages and x buffer, as `Mamba1DecodeSession` keeps them
    var stages: MambaDeviceStages
    var dx: DeviceBuffer[DType.float32]
    #: lane fix-mamba12-forward-exec: a `keep_stages` session's last L > 1
    #: run's stages, kept for the next run of the same (B, L) and refilled
    #: with the constructor's zeros (`_m1_held_take`), and their float count
    var held: Optional[MambaDeviceStages]
    var held_n: Int
    #: lane fix-mamba-forward-doors: the one
    #: arena `held`'s stages are views of (re-zeroed with one fill)
    var held_arena: Optional[MambaArena]
    #: lane x56d: the last RECORDED run (`mamba_step_train_run`): its token
    #: count (0: none) and its one-block tape stack (weights over the
    #: weights tensor, input copy, stages, backward workspaces), kept until
    #: the backward reads it
    var train_l: Int
    var train: Optional[Mamba1TapeStack]
    #: lane fix-wave2-leftovers (audit NOTE rules_mamba.mojo:959): the hand
    #: backward's six intermediates, kept across recorded runs and backward
    #: calls (rebuilt when (B, L) changes; `drop_training` keeps them: they
    #: hold no recorded values). Empty until the first hand backward.
    var hand: List[Mamba1HandBuffers]

    def __init__(out self, ctx: DeviceContext, ex: DeviceExec, w: FP, nw: Int, s: FP, ns: Int, b: Int, dm: Int) raises:
        var dims = MambaDims.of(dm)
        var aw = MambaArena(over=_view(ex, w, nw), cap=nw, guard=MAMBA_GUARD)
        var dw = MambaDeviceWeights(ctx, dims, aw)
        var as_ = MambaArena(over=_view(ex, s, ns), cap=ns, guard=MAMBA_GUARD)
        var ds = MambaDeviceState(ctx, b, dims, as_)
        self.w_addr = Int(w)
        self.s_addr = Int(s)
        self.b = b
        self.dm = dm
        self.stages = MambaDeviceStages(ctx, b, 1, dims)
        self.dx = mamba_zeros(ctx, b * dm)
        self.held = None
        self.held_n = 0
        self.held_arena = None
        self.train_l = 0
        self.train = None
        self.hand = List[Mamba1HandBuffers]()
        self.arena_w = aw^
        self.arena_s = as_^
        self.w = dw^
        self.st = ds^


struct _M2Block(Movable):
    var w_addr: Int
    var s_addr: Int
    var q_addr: Int
    var b: Int
    var dm: Int
    var arena_w: MambaArena
    var arena_s: MambaArena
    var w: Mamba2DeviceWeights
    var st: Mamba2DeviceState
    var dx: DeviceBuffer[DType.float32]
    #: the last run's h_last report (stage S17 after the final padded chunk,
    #: NOT the resumption state), copied out of the call-shaped stages
    var h_last: DeviceBuffer[DType.float32]
    #: lane fix-mamba12-forward-exec: a `keep_stages` session's last L > 1
    #: run's stages (kept per (B, L, buf_len), `_m2_held_take`), their float
    #: count, and whether the last run's h_last report is the held stages'
    #: `h_last` (no copy into `h_last` above) rather than `h_last`
    var held: Optional[Mamba2DeviceStages]
    var held_n: Int
    var h_in_held: Bool
    #: lane fix-mamba-forward-doors: the one arena `held`'s stages are views of
    var held_arena: Optional[MambaArena]
    #: lane fx-mamba-decode-arena (MAMBA_FAST_DECODE_STAGE_ARENA, the default;
    #: under its _OFF define both stay empty): the one arena an L = 1 run's stage views are
    #: re-taken from (`_m2_decode_views`) and its float count
    var dec_arena: Optional[MambaArena]
    var dec_n: Int
    #: lane x56d: the last recorded run's token count and tape stack
    var train_l: Int
    var train: Optional[Mamba2TapeStack]

    def __init__(out self, ctx: DeviceContext, mut ex: DeviceExec, w: FP, nw: Int, s: FP, ns: Int, q: FP,
                 b: Int, dm: Int) raises:
        var dims = Mamba2Dims.of(dm)
        var words = _read_words(ex, q, 1)
        _m2_check_words(words)
        var aw = MambaArena(over=_view(ex, w, nw), cap=nw, guard=MAMBA_GUARD)
        var dw = Mamba2DeviceWeights(ctx, dims, aw)
        # The state fields are VIEWS of the session tensor and are never
        # replaced: replacing a Mamba2DeviceState field faults on AMD (HIP
        # memory fault, the B2/L4/DM32 note of the retired `_mamba2_run`).
        # Every state write goes into the session tensor in place
        # (`OpsSession.load_block`).
        var as_ = MambaArena(over=_view(ex, s, ns), cap=ns, guard=MAMBA_GUARD)
        var ds = Mamba2DeviceState(ctx, b, dims, as_)
        ds.buf_len = words[0]
        self.w_addr = Int(w)
        self.s_addr = Int(s)
        self.q_addr = Int(q)
        self.b = b
        self.dm = dm
        self.dx = mamba_zeros(ctx, b * dm)
        self.h_last = mamba_zeros(ctx, m2_report_floats(b, dm))
        self.held = None
        self.held_n = 0
        self.h_in_held = False
        self.held_arena = None
        self.dec_arena = None
        self.dec_n = 0
        self.train_l = 0
        self.train = None
        self.arena_w = aw^
        self.arena_s = as_^
        self.w = dw^
        self.st = ds^


struct _M3Block(Movable):
    var w_addr: Int
    var s_addr: Int
    var q_addr: Int
    var b: Int
    var dm: Int
    var arena_w: MambaArena
    var arena_s: MambaArena
    var w: Mamba3DeviceWeights
    var st: Mamba3DeviceState
    var dx: DeviceBuffer[DType.float32]
    #: the last run's four reports (h_last, k_last, v_last, theta_last: the
    #: contract's REPORT stages, distinct from the resumption state)
    var reports: List[DeviceBuffer[DType.float32]]
    #: lane restore-samba-resident-forward: an inference-only session's
    #: (`BlockStepScratch.inference_only`) last L > 1 run's stages and input
    #: copy, kept for the next run of the same (B, L, buf_len) and refilled
    #: with the constructor's zeros (IDN_M3_SESSION_STAGE_REUSE)
    var held_stages: Optional[Mamba3DeviceStages]
    var held_x: Optional[DeviceBuffer[DType.float32]]
    #: lane fx-mamba-decode-arena (MAMBA_FAST_DECODE_STAGE_ARENA, the default;
    #: under its _OFF define all four stay empty): the one arena `held_stages` are views of
    #: and its float count; the one arena an L = 1 run's stage views are
    #: re-taken from (`_m3_decode_views`) and its float count
    var held_arena: Optional[MambaArena]
    var held_n: Int
    var dec_arena: Optional[MambaArena]
    var dec_n: Int
    #: lane x56d: the last recorded run's token count, its zero-state stages
    #: and its input copy, kept until the backward reads them
    var train_l: Int
    var train_st: Optional[Mamba3DeviceStages]
    var train_x: Optional[DeviceBuffer[DType.float32]]

    def __init__(out self, ctx: DeviceContext, mut ex: DeviceExec, w: FP, nw: Int, s: FP, ns: Int, q: FP,
                 b: Int, dm: Int) raises:
        var dims = Mamba3Dims.of(dm)
        var words = _read_words(ex, q, 2)
        var q0 = words[0]
        var pend = words[1]
        _m3_check_words(words)
        var aw = MambaArena(over=_view(ex, w, nw), cap=nw, guard=MAMBA_GUARD)
        var dw = Mamba3DeviceWeights(ctx, dims, aw)
        var as_ = MambaArena(over=_view(ex, s, ns), cap=ns, guard=MAMBA_GUARD)
        var ds = Mamba3DeviceState(ctx, b, dims, as_)
        ds.buf_len = q0
        ds.pending = pend == 1
        self.w_addr = Int(w)
        self.s_addr = Int(s)
        self.q_addr = Int(q)
        self.b = b
        self.dm = dm
        self.dx = mamba_zeros(ctx, b * dm)
        var sizes = m3_report_floats(b, dm)
        self.reports = List[DeviceBuffer[DType.float32]](capacity=4)
        for i in range(4):  # small-loop(the four Mamba-3 report buffers)
            self.reports.append(mamba_zeros(ctx, sizes[i]))
        self.held_stages = None
        self.held_x = None
        self.held_arena = None
        self.held_n = 0
        self.dec_arena = None
        self.dec_n = 0
        self.train_l = 0
        self.train_st = None
        self.train_x = None
        self.arena_w = aw^
        self.arena_s = as_^
        self.w = dw^
        self.st = ds^


struct _AttnBlock(Movable):
    var w_addr: Int
    var s_addr: Int
    var q_addr: Int
    #: the op's integer arguments the entry was built for
    var ints: List[Int]
    var w: LlamaDeviceWeights
    var kv: LlamaKVCache
    var rope: LlamaRopeTable
    #: the L = 1 stages and x buffer, as `TransformerDecodeSession` keeps them
    var stages: LlamaDeviceStages
    var dx: DeviceBuffer[DType.float32]
    var lean: Bool
    #: lane x4b: built over weights the caller handed in (`attn_step_adopt`,
    #: the fixed15_v1 planes), not over views of the weights tensor; the
    #: weights operand is then a one-float placeholder the op never reads.
    var adopted: Bool
    #: lane x7: the last RECORDED run (a training forward on the tape,
    #: `attn_step_train_run`): its token count (0: none since the entry was
    #: built), its L > 1 stages and input, kept until the backward reads
    #: them (`attn_step_backward`), and the backward's own stages.
    var train_l: Int
    var kept: AttnPrefill
    var bst: Optional[LlamaBackwardStages]
    var bkey: List[Int]
    #: lane r1 (ATTN_STEP_NEXT_NORM): the entry whose recorded run followed
    #: this one's last time (its state and word addresses; 0: none), the
    #: spare stages the previous layer's run prepares and writes this
    #: entry's norm1 into, and that run's output address (0: none pending).
    var next_s: Int
    var next_q: Int
    var spare: AttnPrefill
    var ahead_y: Int

    def __init__(out self, ctx: DeviceContext, mut ex: DeviceExec, w: FP, nw: Int, s: FP, ns: Int, q: FP,
                 ints: List[Int]) raises:
        var dims = _attn_dims(ints)
        var opts = _attn_opts(ints)
        var b = ints[0]
        var smax = ints[6]
        var window = ints[7]
        var words = _read_words(ex, q, 1)
        if words[0] < 0 or words[0] > smax:
            raise Error("ValueError: ops: attention_step: cached tokens must be in [0, " + String(smax)
                        + "], got " + String(words[0]))
        var sizes = block_weight_sizes(OPS_ATTN_STEP, ints)
        var at = block_offsets(sizes, 0)
        var v = List[DeviceBuffer[DType.float32]]()
        for i in range(len(sizes)):  # driver-loop: over the block's twenty weight pieces
            v.append(_view(ex, w + at[i], sizes[i] if sizes[i] > 0 else 1))
        self.w = LlamaDeviceWeights(
            ctx, dims, opts, v[0].copy(), v[1].copy(), v[2].copy(), v[3].copy(), v[4].copy(), v[5].copy(),
            v[6].copy(), v[7].copy(), v[8].copy(), v[9].copy(), v[10].copy(), v[11].copy(), v[12].copy(),
            v[13].copy(), v[14].copy(), v[15].copy(), v[16].copy(), v[17].copy(), v[18].copy(), v[19].copy(),
        )
        self.kv = _attn_kv_views(ctx, ex, s, ints, dims, opts, words[0])
        self.rope = LlamaRopeTable(ctx, dims, opts, smax)
        # `_lean_for_w` with no fixed15 planes (these weights never carry them).
        var lean = _lean_for(dims.head_dim, opts)
        self.lean = lean
        self.stages = LlamaDeviceStages(ctx, b, 1, smax, dims, window, lean=lean)
        self.dx = ctx.enqueue_create_buffer[DType.float32](b * dims.d_model)
        self.w_addr = Int(w)
        self.s_addr = Int(s)
        self.q_addr = Int(q)
        self.ints = ints.copy()
        self.adopted = False
        self.train_l = 0
        self.kept = AttnPrefill()
        self.bst = Optional[LlamaBackwardStages]()
        self.bkey = List[Int]()
        self.next_s = 0
        self.next_q = 0
        self.spare = AttnPrefill()
        self.ahead_y = 0
        _ = ns

    def __init__(out self, ctx: DeviceContext, mut ex: DeviceExec, var weights: LlamaDeviceWeights, w: FP, s: FP,
                 q: FP, ints: List[Int], s0: Int, lean: Bool) raises:
        """Lane x4b: the entry over weights built by the caller (the
        fixed15_v1 decode session's profile constructor: norms, biases and
        the seven projection planes), keyed by the placeholder weights
        tensor `w`; the caches and the word are the op's usual state
        operands. `lean` is the caller's (`_lean_for_w`: never lean over
        planes)."""
        var dims = _attn_dims(ints)
        var opts = _attn_opts(ints)
        var smax = ints[6]
        if s0 < 0 or s0 > smax:
            raise Error("ValueError: ops: attention_step: cached tokens must be in [0, " + String(smax)
                        + "], got " + String(s0))
        self.w = weights^
        self.kv = _attn_kv_views(ctx, ex, s, ints, dims, opts, s0)
        self.rope = LlamaRopeTable(ctx, dims, opts, smax)
        self.lean = lean
        self.stages = LlamaDeviceStages(ctx, ints[0], 1, smax, dims, ints[7], lean=lean)
        self.dx = ctx.enqueue_create_buffer[DType.float32](ints[0] * dims.d_model)
        self.w_addr = Int(w)
        self.s_addr = Int(s)
        self.q_addr = Int(q)
        self.ints = ints.copy()
        self.adopted = True
        self.train_l = 0
        self.kept = AttnPrefill()
        self.bst = Optional[LlamaBackwardStages]()
        self.bkey = List[Int]()
        self.next_s = 0
        self.next_q = 0
        self.spare = AttnPrefill()
        self.ahead_y = 0


def _attn_kv_views(ctx: DeviceContext, ex: DeviceExec, s: FP, ints: List[Int], dims: LlamaDims,
                   opts: BlockOptions, s0: Int) raises -> LlamaKVCache:
    """The KV cache struct over the state tensor's two pieces at their
    `block_offsets` starts (lane x4b: the value cache started at `s +
    cache_n`, which is the packed layout only when cache_n is a whole
    256-byte slot; `AttnStepArm` and `upload_block` pack at the slot)."""
    var ssz = block_state_sizes(OPS_ATTN_STEP, ints)
    var sat = block_offsets(ssz, block_guard(OPS_ATTN_STEP))
    var kv = LlamaKVCache(ctx, ints[0], dims, ints[6], ints[7], opts.max_positions)
    kv.k = _view(ex, s + sat[0], ssz[0])
    kv.v = _view(ex, s + sat[1], ssz[1])
    kv.s = s0
    return kv^


# Lane x45c: a run's call-shaped stages (and x copy) are parked here instead
# of a wait inside the op (the per-layer synchronize X6 measured on every
# L > 1 run, and on every Mamba-2/3 run); a caller that drains the stream
# (a download, a loop's status read) releases them. A session that never
# drains holds at most STEP_PARK_CAP parked runs: the next run past it waits
# once and releases them all (bounded memory, one wait per that many runs).
comptime STEP_PARK_CAP = 64


struct BlockStepScratch(Movable):
    """The resident block structs of one session's step ops, one entry per
    (weights, state) pair, built at the pair's first run, and the parked
    stages of runs the stream may still be executing."""
    var m1: List[_M1Block]
    var m2: List[_M2Block]
    var m3: List[_M3Block]
    var attn: List[_AttnBlock]
    var parked_m1: List[MambaDeviceStages]
    var parked_m2: List[Mamba2DeviceStages]
    var parked_m3: List[Mamba3DeviceStages]
    var parked_attn: List[AttnPrefill]
    var parked_x: List[DeviceBuffer[DType.float32]]
    #: lane fix-mamba-forward-doors: the arenas of parked held Mamba-1/2
    #: stage sets (their views sit in parked_m1 / parked_m2)
    var parked_arena: List[MambaArena]
    #: lane r1 (ATTN_STEP_NEXT_NORM): the last recorded attention run's
    #: state and word addresses and its output address (0: none).
    var last_train_s: Int
    var last_train_q: Int
    var last_train_y: Int
    #: lane restore-samba-resident-forward (2026-11-13): the session's runs
    #: are never read by a backward (the Samba stack program,
    #: training/samba_program.mojo). Its Mamba-3 L > 1 runs keep their
    #: stages and input copy in the entry (`_run_m3`), and its attention runs
    #: launch `forward_only=True` on the entry's kept L > 1 stages
    #: (`_run_attn`), as the 0.8.55 resident forward did
    #: (training/samba_resident.mojo `_forward_blocks`).
    var inference_only: Bool
    #: lane fx-kv-inplace: the full-causal attention entries a FAILED run
    #: dropped (`block_step_run`), as (state address, word address, the used
    #: positions the entry's cache held, the op's integers). A bounded loop
    #: that ran bodies past its stop takes such a cache back to its restored
    #: word from this record (`attn_append_restore`), since the entry that
    #: knew how far those bodies appended is gone. Written only when a run
    #: fails; dropped when the cache's buffers are forgotten or a new entry
    #: is built over them.
    var lost_s: List[Int]
    var lost_q: List[Int]
    var lost_used: List[Int]
    var lost_ints: List[List[Int]]
    #: the current run requests no Mamba-3 reports (the caller reads none
    #: before the next run): `_run_m3` skips the four report copies
    #: (IDN_MAMBA3_REPORTS_ON_REQUEST). Set per run by the caller.
    var skip_reports: Bool
    #: lane fix-mamba12-forward-exec:
    #: the session's Mamba-1/2 L > 1 runs are never read by a backward (a
    #: recorded run goes through `mamba_step_train_run`, which keeps its own
    #: tape stack), so `_run_m1` / `_run_m2` keep their stages in the entry
    #: (`held`) for the next run of the same shape. Set by the per-call
    #: block programs (mamba/impl/mamba_block_program.mojo).
    var keep_stages: Bool
    #: lane fix-mamba-forward-doors: the
    #: current run is a fresh (discard-state) per-call forward whose caller
    #: rewrites the carried words before any reader (`zero_state` /
    #: `load_state`, mamba/impl/mamba_block_program.mojo), so a held
    #: Mamba-2 run skips its post-run buf_len word write. Set per run by
    #: that caller; never set by a session that reads the words back.
    var skip_words: Bool
    #: lane fx-llama-refuse-watermark (MOJOLEARN_LLAMA_FAST_REFUSE_WATERMARK,
    #: the IDENTICAL default; read only by `_attn_watermark_grant`, never under its _OFF define):
    #: the owner of this scratch states that the state tensors of its
    #: attention entries are written only by the step runs of this file and
    #: by writers that reset the entry's watermark or drop the entry (the
    #: list is in `_attn_watermark_grant`'s docstring). False (every generic
    #: `OpsSession`): the whole-cache refusal scans on every call.
    var attn_cache_private: Bool
    #: lane fx-ops-session-blocks (2026-11-19): this session's executor is
    #: NOT on the x_cnn context (`cnn_ctx()`), whose queue the CNN block ops'
    #: entries launch on and wait for (autograd/cnn_block_op.mojo). Set for
    #: a public `mojolearn.ops.Session` (the sequence context); the session
    #: then drains its own queue before a CNN entry, forward or backward, so
    #: the entry reads finished operands (the entry's own wait finishes its
    #: results). False for the CNN model's sessions, which are opened over
    #: `cnn_ctx()` and share one in-order queue: no wait is added there.
    var cnn_other_queue: Bool

    def __init__(out self):
        self.attn_cache_private = False
        self.cnn_other_queue = False
        self.m1 = List[_M1Block]()
        self.m2 = List[_M2Block]()
        self.m3 = List[_M3Block]()
        self.attn = List[_AttnBlock]()
        self.parked_m1 = List[MambaDeviceStages]()
        self.parked_m2 = List[Mamba2DeviceStages]()
        self.parked_m3 = List[Mamba3DeviceStages]()
        self.parked_attn = List[AttnPrefill]()
        self.parked_x = List[DeviceBuffer[DType.float32]]()
        self.parked_arena = List[MambaArena]()
        self.last_train_s = 0
        self.last_train_q = 0
        self.last_train_y = 0
        self.inference_only = False
        self.skip_reports = False
        self.keep_stages = False
        self.skip_words = False
        self.lost_s = List[Int]()
        self.lost_q = List[Int]()
        self.lost_used = List[Int]()
        self.lost_ints = List[List[Int]]()

    def drop_lost(mut self, addr: Int):
        """Forget the lost-append record over the state or word at `addr`."""
        var i = len(self.lost_s) - 1
        while i >= 0:  # driver-loop: over lost-append records (at most one per cache)
            if self.lost_s[i] == addr or self.lost_q[i] == addr:
                _ = self.lost_s.pop(i)
                _ = self.lost_q.pop(i)
                _ = self.lost_used.pop(i)
                _ = self.lost_ints.pop(i)
            i -= 1

    def note_lost(mut self, s_addr: Int, q_addr: Int, used: Int, ints: List[Int]):
        """A failed run dropped the entry of the full-causal cache at
        `s_addr`, which held `used` positions (see `lost_s`)."""
        self.drop_lost(s_addr)
        self.lost_s.append(s_addr)
        self.lost_q.append(q_addr)
        self.lost_used.append(used)
        self.lost_ints.append(ints.copy())

    def held_floats(self) -> Int:
        """The device floats of the Mamba-1/2 stages the entries hold
        (`keep_stages`), for the per-call program cache's byte bound."""
        var n = 0
        for i in range(len(self.m1)):  # driver-loop: over resident Mamba-1 blocks
            n += self.m1[i].held_n
        for i in range(len(self.m2)):  # driver-loop: over resident Mamba-2 blocks
            n += self.m2[i].held_n
        comptime if MAMBA_FAST_DECODE_STAGE_ARENA:
            # lane fx-mamba-decode-arena: the decode arenas and the held
            # Mamba-3 stages count toward the same bound
            for i in range(len(self.m2)):  # driver-loop: over resident Mamba-2 blocks
                n += self.m2[i].dec_n
            for i in range(len(self.m3)):  # driver-loop: over resident Mamba-3 blocks
                n += self.m3[i].held_n + self.m3[i].dec_n
        return n

    def parked(self) -> Int:
        return (len(self.parked_m1) + len(self.parked_m2) + len(self.parked_m3) + len(self.parked_attn)
                + len(self.parked_x) + len(self.parked_arena))

    def release_parked(mut self):
        """Free the parked stages. Only after the stream drained (a download
        or a synchronize the caller made)."""
        self.parked_m1.clear()
        self.parked_m2.clear()
        self.parked_m3.clear()
        self.parked_attn.clear()
        self.parked_x.clear()
        # after the views above (their storage is the arena's)
        self.parked_arena.clear()

    def make_room(mut self, ctx: DeviceContext) raises:
        """Before a run that parks: past STEP_PARK_CAP parked runs, one wait
        and release."""
        if self.parked() >= STEP_PARK_CAP:
            ctx.synchronize()
            self.release_parked()

    def count(self) -> Int:
        return len(self.m1) + len(self.m2) + len(self.m3) + len(self.attn)

    def park_arenas_m1(mut self, i: Int):
        """Entry `i` is about to be dropped: its arena moves to
        `parked_arena`, which `release_parked` frees after every parked view.
        fx-leftovers-product (2026-11-19): dropping the entry freed the arena
        while a parked run's views of it (and, under
        MAMBA_FAST_DECODE_STAGE_ARENA, every parked decode token's stages)
        were still parked: storage freed before its views, and before the
        stream that reads them drained."""
        if self.m1[i].held_arena:
            self.parked_arena.append(self.m1[i].held_arena.take())

    def park_arenas_m2(mut self, i: Int):
        """`park_arenas_m1` for a Mamba-2 entry (held and decode arenas)."""
        if self.m2[i].held_arena:
            self.parked_arena.append(self.m2[i].held_arena.take())
        if self.m2[i].dec_arena:
            self.parked_arena.append(self.m2[i].dec_arena.take())

    def park_arenas_m3(mut self, i: Int):
        """`park_arenas_m1` for a Mamba-3 entry (held and decode arenas)."""
        if self.m3[i].held_arena:
            self.parked_arena.append(self.m3[i].held_arena.take())
        if self.m3[i].dec_arena:
            self.parked_arena.append(self.m3[i].dec_arena.take())

    def forget(mut self, addr: Int):
        """Drop every entry built over the session buffer at `addr` (its
        weights or its state were written from outside a step, or freed)."""
        if len(self.lost_s) > 0:
            self.drop_lost(addr)
        if self.count() == 0:
            return
        var i = len(self.m1) - 1
        while i >= 0:  # driver-loop: over resident Mamba-1 blocks
            if self.m1[i].w_addr == addr or self.m1[i].s_addr == addr:
                self.park_arenas_m1(i)
                _ = self.m1.pop(i)
            i -= 1
        i = len(self.m2) - 1
        while i >= 0:  # driver-loop: over resident Mamba-2 blocks
            if self.m2[i].w_addr == addr or self.m2[i].s_addr == addr or self.m2[i].q_addr == addr:
                self.park_arenas_m2(i)
                _ = self.m2.pop(i)
            i -= 1
        i = len(self.m3) - 1
        while i >= 0:  # driver-loop: over resident Mamba-3 blocks
            if self.m3[i].w_addr == addr or self.m3[i].s_addr == addr or self.m3[i].q_addr == addr:
                self.park_arenas_m3(i)
                _ = self.m3.pop(i)
            i -= 1
        i = len(self.attn) - 1
        while i >= 0:  # driver-loop: over resident attention blocks
            if self.attn[i].w_addr == addr or self.attn[i].s_addr == addr or self.attn[i].q_addr == addr:
                _ = self.attn.pop(i)
            i -= 1

    def clear(mut self):
        # the arenas outlive the parked views of them (`park_arenas_m1`)
        for i in range(len(self.m1)):  # driver-loop: over resident Mamba-1 blocks
            self.park_arenas_m1(i)
        for i in range(len(self.m2)):  # driver-loop: over resident Mamba-2 blocks
            self.park_arenas_m2(i)
        for i in range(len(self.m3)):  # driver-loop: over resident Mamba-3 blocks
            self.park_arenas_m3(i)
        self.m1.clear()
        self.m2.clear()
        self.m3.clear()
        self.attn.clear()
        self.lost_s.clear()
        self.lost_q.clear()
        self.lost_used.clear()
        self.lost_ints.clear()

    def set_words(mut self, addr: Int, words: List[Int]) raises:
        """The carried integers of every Mamba entry whose state words live at
        `addr` were written in place (`OpsSession.load_block`): the entries
        keep their structs (the float state is a view, so it needs nothing)
        and take the new host integers, checked as a new entry checks them."""
        for i in range(len(self.m2)):  # driver-loop: over resident Mamba-2 blocks
            if self.m2[i].q_addr == addr:
                _m2_check_words(words)
                self.m2[i].st.buf_len = words[0]
        for i in range(len(self.m3)):  # driver-loop: over resident Mamba-3 blocks
            if self.m3[i].q_addr == addr:
                _m3_check_words(words)
                self.m3[i].st.buf_len = words[0]
                self.m3[i].st.pending = words[1] == 1
        # integ-w3 (lanes x5 + x4b): an attention entry takes the loaded
        # cached-token count in place, as `attn_step_restart` does (its caches
        # are views of the state tensor, so the loaded floats need nothing);
        # dropping it would lose an adopted entry's fixed15 weights.
        for i in range(len(self.attn)):  # driver-loop: over resident attention blocks
            if self.attn[i].q_addr == addr:
                if len(words) < 1 or words[0] < 0 or words[0] > self.attn[i].ints[6]:
                    raise Error("ValueError: ops: attention_step: cached tokens must be in [0, "
                                + String(self.attn[i].ints[6]) + "]")
                self.attn[i].kv.s = words[0]
                comptime if LLAMA_FAST_REFUSE_WATERMARK:
                    # a loaded position (and whatever floats came with it)
                    self.attn[i].kv.wm_reset()

    def attn_cache_written(mut self, addr: Int):
        """The floats of the state tensor at `addr` were written in place
        from outside a step run (`OpsSession.load_block` of a state part):
        the attention entries over it keep their structs (the caches are
        views) and lose their refusal watermark, so the next run scans the
        whole cache (MOJOLEARN_LLAMA_FAST_REFUSE_WATERMARK; nothing to do on
        a build without it)."""
        comptime if LLAMA_FAST_REFUSE_WATERMARK:
            for i in range(len(self.attn)):  # driver-loop: over resident attention blocks
                if self.attn[i].s_addr == addr:
                    self.attn[i].kv.wm_reset()

    def drop_training(mut self, w_addr: Int):
        """Free the recorded run kept by every Mamba entry over the weights at
        `w_addr` (lane x56d: a one-shot block VJP holds no training stages
        past its call)."""
        for i in range(len(self.m1)):  # driver-loop: over resident Mamba-1 blocks
            if self.m1[i].w_addr == w_addr:
                self.m1[i].train = None
                self.m1[i].train_l = 0
        for i in range(len(self.m2)):  # driver-loop: over resident Mamba-2 blocks
            if self.m2[i].w_addr == w_addr:
                self.m2[i].train = None
                self.m2[i].train_l = 0
        for i in range(len(self.m3)):  # driver-loop: over resident Mamba-3 blocks
            if self.m3[i].w_addr == w_addr:
                self.m3[i].train_st = None
                self.m3[i].train_x = None
                self.m3[i].train_l = 0

    def weights_changed(mut self, addr: Int):
        """The weights at `addr` were rewritten in place: the block's own
        refusal of non-finite weights runs again at the next step."""
        # lane x56d: a recorded run's kept stages were computed over the old
        # weights: they go (a backward after the write is refused), as the
        # attention entry's do.
        for i in range(len(self.m1)):  # driver-loop: over resident Mamba-1 blocks
            if self.m1[i].w_addr == addr:
                self.m1[i].w.weights_checked = False
                self.m1[i].train = None
                self.m1[i].train_l = 0
        for i in range(len(self.m2)):  # driver-loop: over resident Mamba-2 blocks
            if self.m2[i].w_addr == addr:
                self.m2[i].w.weights_checked = False
                self.m2[i].train = None
                self.m2[i].train_l = 0
        for i in range(len(self.m3)):  # driver-loop: over resident Mamba-3 blocks
            if self.m3[i].w_addr == addr:
                self.m3[i].w.weights_checked = False
                self.m3[i].train_st = None
                self.m3[i].train_x = None
                self.m3[i].train_l = 0
        # integ-w3: an adopted entry (lane x4b) never reads the weights
        # tensor (a placeholder), so a write there does not drop it.
        for i in range(len(self.attn) - 1, -1, -1):  # driver-loop: over resident attention blocks
            if self.attn[i].w_addr == addr and not self.attn[i].adopted:
                _ = self.attn.pop(i)


# ---------------------------------------------------------------- the runs
def _tokens(code: Int, nx: Int, b: Int, dm: Int) raises -> Int:
    if nx % (b * dm) != 0 or nx == 0:
        raise Error("ValueError: ops: " + String(step_op_info(code).name) + ": x holds " + String(nx)
                    + " values, not a multiple of b * d_model = " + String(b * dm))
    return nx // (b * dm)


def _expect(code: Int, what: String, have: Int, want: Int) raises:
    if have != want:
        raise Error("ValueError: ops: " + String(step_op_info(code).name) + ": the " + what + " tensor holds "
                    + String(have) + " values, the block's layout " + String(want)
                    + " (pack it with upload_block)")


def _m1_entry(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], b: Int,
             dm: Int) raises -> Int:
    """The index in `sc.m1` of the resident entry over (weights, state),
    built at the pair's first run."""
    for i in range(len(sc.m1)):  # driver-loop: over resident Mamba-1 blocks
        if sc.m1[i].w_addr == Int(bufs[1]) and sc.m1[i].s_addr == Int(bufs[2]) and sc.m1[i].b == b and sc.m1[i].dm == dm:
            return i
    var ctx = ex.ctx.copy()
    sc.m1.append(_M1Block(ctx, ex, bufs[1], counts[1], bufs[2], counts[2], b, dm))
    return len(sc.m1) - 1


def _m2_entry(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], b: Int,
             dm: Int) raises -> Int:
    """The index in `sc.m2` of the resident entry over (weights, state,
    words), built at the triple's first run."""
    for i in range(len(sc.m2)):  # driver-loop: over resident Mamba-2 blocks
        if (sc.m2[i].w_addr == Int(bufs[1]) and sc.m2[i].s_addr == Int(bufs[2]) and sc.m2[i].q_addr == Int(bufs[3])
                and sc.m2[i].b == b and sc.m2[i].dm == dm):
            return i
    var ctx = ex.ctx.copy()
    sc.m2.append(_M2Block(ctx, ex, bufs[1], counts[1], bufs[2], counts[2], bufs[3], b, dm))
    return len(sc.m2) - 1


def _m3_entry(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], b: Int,
             dm: Int) raises -> Int:
    """The index in `sc.m3` of the resident entry over (weights, state,
    words), built at the triple's first run."""
    for i in range(len(sc.m3)):  # driver-loop: over resident Mamba-3 blocks
        if (sc.m3[i].w_addr == Int(bufs[1]) and sc.m3[i].s_addr == Int(bufs[2]) and sc.m3[i].q_addr == Int(bufs[3])
                and sc.m3[i].b == b and sc.m3[i].dm == dm):
            return i
    var ctx = ex.ctx.copy()
    sc.m3.append(_M3Block(ctx, ex, bufs[1], counts[1], bufs[2], counts[2], bufs[3], b, dm))
    return len(sc.m3) - 1


def _block_out(ctx: DeviceContext, ex: DeviceExec, y_out: FP, y_host: Int, src: DeviceBuffer[DType.float32],
               n: Int) raises:
    """A Mamba block's output `src[0:n]`: straight into the caller's host
    array at `y_host` when it is non-zero (queued; the caller waits: the
    per-call door), else device-to-device into the op's y
    tensor at `y_out`."""
    if y_host != 0:
        ctx.enqueue_copy(dst_ptr=f32_ptr(y_host), src_buf=src.create_sub_buffer[DType.float32](0, n))
    else:
        _d2d(ctx, _view(ex, y_out, n), src, n)


def _sum_floats(sizes: List[Int]) -> Int:
    var n = 0
    for i in range(len(sizes)):  # small-loop(sizes: a stage set's field lengths, model structure): size sum, no data
        n += sizes[i]
    return n


def _m1_held_take(ctx: DeviceContext, mut sc: BlockStepScratch, e: Int, b: Int, l: Int,
                  dims: MambaDims) raises -> MambaDeviceStages:
    """Lane fix-mamba12-forward-exec: the stages of a `keep_stages` L > 1
    Mamba-1 run: the entry's held set when it was built for (B, L), refilled
    with the constructor's zeros in stream order (no allocation, no wait),
    else a new set (a held set of another shape is parked, not freed, while
    the stream may still read it). Lane fix-mamba-forward-doors
    (header): the set is views of one held
    arena, built with one allocation and one fill and re-zeroed with one
    fill (0.8.55's `_mamba1_run_arena`)."""
    if sc.m1[e].held:
        var prev = sc.m1[e].held.take()
        sc.m1[e].held_n = 0
        if prev.b == b and prev.l == l:
            if sc.m1[e].held_arena:
                # ONE fill: every stage view's zeros (the arena's own
                # constructor fill), in stream order, no wait
                sc.m1[e].held_arena.value().base.enqueue_fill(Float32(0.0))
            else:
                prev.rezero()
            return prev^
        sc.make_room(ctx)
        sc.parked_m1.append(prev^)
    # a previous set's arena (another shape, or a run that failed between
    # take and hold) is parked, not freed, while the stream may read it
    if sc.m1[e].held_arena:
        sc.make_room(ctx)
        sc.parked_arena.append(sc.m1[e].held_arena.take())
    var arena = MambaArena(ctx, MambaArena.total(_m1_stage_sizes(b, l, dims), MAMBA_GUARD), MAMBA_GUARD)
    var fresh = MambaDeviceStages(ctx, b, l, dims, arena)
    sc.m1[e].held_arena = arena^
    return fresh^


def _m2_held_take(ctx: DeviceContext, mut sc: BlockStepScratch, e: Int, b: Int, l: Int, q0: Int,
                  dims: Mamba2Dims) raises -> Mamba2DeviceStages:
    """`_m1_held_take` for Mamba-2: the held set is reused for the same
    (B, L, buf_len), the shape its working buffers were sized for."""
    if sc.m2[e].held:
        var prev = sc.m2[e].held.take()
        sc.m2[e].held_n = 0
        sc.m2[e].h_in_held = False
        if prev.b == b and prev.l == l and prev.q0 == q0:
            if sc.m2[e].held_arena:
                # ONE fill for the 28 stage views (fix-mamba-forward-doors)
                sc.m2[e].held_arena.value().base.enqueue_fill(Float32(0.0))
            else:
                prev.rezero()
            return prev^
        sc.make_room(ctx)
        sc.parked_m2.append(prev^)
    # lane fix-mamba-forward-doors: one arena (0.8.55's `_mamba2_run_arena`)
    if sc.m2[e].held_arena:
        sc.make_room(ctx)
        sc.parked_arena.append(sc.m2[e].held_arena.take())
    var arena = MambaArena(ctx, MambaArena.total(_m2_stage_sizes(b, l, q0, dims), MAMBA_GUARD), MAMBA_GUARD)
    var fresh = Mamba2DeviceStages(ctx, b, l, q0, dims, arena)
    sc.m2[e].held_arena = arena^
    return fresh^


def _m2_call_stages(ctx: DeviceContext, mut blk: _M2Block, b: Int, l: Int, dims: Mamba2Dims) raises -> Mamba2DeviceStages:
    """The stages of one Mamba-2 run that holds none: built per call at the
    open chunk's length, or, under MAMBA_FAST_DECODE_STAGE_ARENA, an L = 1
    run's views of the entry's decode arena."""
    comptime if MAMBA_FAST_DECODE_STAGE_ARENA:
        if l == 1:
            return _m2_decode_views(ctx, blk, b, dims)
    return Mamba2DeviceStages(ctx, b, l, blk.st.buf_len, dims)


def _m2_decode_views(ctx: DeviceContext, mut blk: _M2Block, b: Int, dims: Mamba2Dims) raises -> Mamba2DeviceStages:
    """Lane fx-mamba-decode-arena (MAMBA_FAST_DECODE_STAGE_ARENA, header):
    the 28 stages of one decode token as views of the entry's ONE arena, at
    the lengths `Mamba2DeviceStages(ctx, b, 1, buf_len, dims)` allocates. No
    allocation and no free per token: the arena is built at the entry's
    first token with room for the largest carried buf_len the contract
    allows (`_m2_check_words`: CHUNK - 1 rows), and every later token queues
    ONE zero fill of the prefix its views cover, in stream order ahead of
    its launches (the views are laid out from offset 0, so the prefix holds
    every word of every view; the first token's zeros are the arena
    constructor's fill). The caller parks the views as it parks per-call
    stages; the arena stays with the entry. A buf_len past the arena's
    extent (the contract refuses it at load) takes per-call stages.
    Serial by nature: one token's stage layout, 28 integer offsets."""
    var q0 = blk.st.buf_len
    comptime cap_q = M2_CHUNK_SIZE - 1
    if q0 > cap_q:
        return Mamba2DeviceStages(ctx, b, 1, q0, dims)
    if blk.dec_arena:
        var arena = blk.dec_arena.take()
        arena.off = 0
        arena.views = 0
        var stages = Mamba2DeviceStages(ctx, b, 1, q0, dims, arena)
        # ONE fill of the used prefix; `reset` keeps its fill view until the next
        # token's and leaves the arena at offset 0
        var used = arena.off
        arena.reset(used)
        blk.dec_arena = arena^
        return stages^
    var cap = MambaArena.total(_m2_stage_sizes(b, 1, cap_q, dims), MAMBA_GUARD)
    var built = MambaArena(ctx, cap, MAMBA_GUARD)
    var first = Mamba2DeviceStages(ctx, b, 1, q0, dims, built)
    blk.dec_n = cap
    blk.dec_arena = built^
    return first^


def _m3_decode_views(ctx: DeviceContext, mut blk: _M3Block, b: Int, dims: Mamba3Dims) raises -> Mamba3DeviceStages:
    """`_m2_decode_views` for Mamba-3: the 33 stages of one decode token as
    views of the entry's ONE arena, sized for the largest carried buf_len
    (`_m3_check_words`: CHUNK rows inclusive), ONE prefix fill per token."""
    var q0 = blk.st.buf_len
    comptime cap_q = M3_CHUNK_SIZE
    if q0 > cap_q:
        return Mamba3DeviceStages(ctx, b, 1, q0, dims)
    if blk.dec_arena:
        var arena = blk.dec_arena.take()
        arena.off = 0
        arena.views = 0
        var stages = Mamba3DeviceStages(ctx, b, 1, q0, dims, arena)
        var used = arena.off
        arena.reset(used)
        blk.dec_arena = arena^
        return stages^
    var cap = MambaArena.total(_m3_stage_sizes(b, 1, cap_q, dims), MAMBA_GUARD)
    var built = MambaArena(ctx, cap, MAMBA_GUARD)
    var first = Mamba3DeviceStages(ctx, b, 1, q0, dims, built)
    blk.dec_n = cap
    blk.dec_arena = built^
    return first^


def _run_m1(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
            y_out: FP, y_host: Int = 0) raises:
    var ctx = ex.ctx.copy()
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_M1_STEP, counts[0], b, dm)
    _expect(OPS_M1_STEP, "weights", counts[1], block_part_floats(OPS_M1_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_M1_STEP, "state", counts[2], block_part_floats(OPS_M1_STEP, ints, BLOCK_PART_STATE))
    var e = _m1_entry(ex, sc, bufs, counts, b, dm)
    var n = b * l * dm
    var x = _view(ex, bufs[0], n)
    var trace = IdentityTrace.disabled()
    comptime if _M12_HOLD:
        # lane fix-mamba12-forward-exec (`_M12_HOLD`,
        # header): the entry keeps its L > 1 stages; x read in place.
        if l > 1 and sc.keep_stages:
            var dims = MambaDims.of(dm)
            var held = _m1_held_take(ctx, sc, e, b, l, dims)
            ref hb = sc.m1[e]
            mamba_block_forward(ctx, held, hb.st, hb.w, x, b, l, trace, String("ops.step"))
            _block_out(ctx, ex, y_out, y_host, held.residual_out, n)
            hb.held_n = _sum_floats(_m1_stage_sizes(b, l, dims))
            hb.held = held^
            return
    ref blk = sc.m1[e]
    if l == 1:
        # `_m1_session_step_run`: mamba_step on the resident stages.
        comptime if _MAMBA_X_IN_PLACE:
            # the block reads x in place (_MAMBA_X_IN_PLACE, header)
            mamba_step(ctx, blk.stages, blk.st, blk.w, x, b, trace, String("ops.step"))
        else:
            _d2d(ctx, blk.dx, x, n)
            mamba_step(ctx, blk.stages, blk.st, blk.w, blk.dx, b, trace, String("ops.step"))
        _block_out(ctx, ex, y_out, y_host, blk.stages.residual_out, n)
    else:
        var dims = MambaDims.of(dm)
        var stages = MambaDeviceStages(ctx, b, l, dims)
        comptime if _MAMBA_X_IN_PLACE:
            mamba_block_forward(ctx, stages, blk.st, blk.w, x, b, l, trace, String("ops.step"))
            _block_out(ctx, ex, y_out, y_host, stages.residual_out, n)
            # lane x45c: parked, no wait (the stream may still read them)
            sc.make_room(ctx)
            sc.parked_m1.append(stages^)
        else:
            var dx = mamba_zeros(ctx, n)
            _d2d(ctx, dx, x, n)
            mamba_block_forward(ctx, stages, blk.st, blk.w, dx, b, l, trace, String("ops.step"))
            _block_out(ctx, ex, y_out, y_host, stages.residual_out, n)
            # lane x45c: parked, no wait (the stream may still read them)
            sc.make_room(ctx)
            sc.parked_m1.append(stages^)
            sc.parked_x.append(dx^)


def _run_m2(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
            fs: List[Float32], y_out: FP, y_host: Int = 0) raises:
    var ctx = ex.ctx.copy()
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_M2_STEP, counts[0], b, dm)
    _expect(OPS_M2_STEP, "weights", counts[1], block_part_floats(OPS_M2_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_M2_STEP, "state", counts[2], block_part_floats(OPS_M2_STEP, ints, BLOCK_PART_STATE))
    _expect(OPS_M2_STEP, "state words", counts[3], 1)
    var e = _m2_entry(ex, sc, bufs, counts, b, dm)
    var dims = Mamba2Dims.of(dm)
    var n = b * l * dm
    var x = _view(ex, bufs[0], n)
    var trace = IdentityTrace.disabled()
    var h_n = m2_report_floats(b, dm)
    # this run's report is copied into the entry's h_last unless it stays in
    # held stages (set below)
    sc.m2[e].h_in_held = False
    comptime if _M12_HOLD:
        # lane fix-mamba12-forward-exec (`_M12_HOLD`,
        # header): the entry keeps its L > 1 stages, x read in place, and the
        # h_last report is read from the held stages, not copied.
        if l > 1 and sc.keep_stages:
            var skip_words = sc.skip_words
            var q0 = sc.m2[e].st.buf_len
            var held = _m2_held_take(ctx, sc, e, b, l, q0, dims)
            ref hb = sc.m2[e]
            mamba2_block_forward(ctx, held, hb.st, hb.w, x, b, l, fs[0], fs[1], trace, String("ops.step"))
            _block_out(ctx, ex, y_out, y_host, held.residual_out, n)
            # lane fix-mamba-forward-doors: a fresh per-call forward's
            # caller rewrites the word before any reader (`skip_words`)
            if not skip_words:
                var held_words: List[Int] = [hb.st.buf_len]
                _write_words(ex, bufs[3], held_words)
            hb.held_n = _sum_floats(_m2_stage_sizes(b, l, q0, dims))
            hb.h_in_held = True
            hb.held = held^
            return
    ref blk = sc.m2[e]
    # `_m2_session_step_run`: call-shaped stages at the open chunk's length
    # (an L = 1 run's are views of the entry's decode arena under
    # MAMBA_FAST_DECODE_STAGE_ARENA, `_m2_call_stages`).
    var stages = _m2_call_stages(ctx, blk, b, l, dims)
    comptime if _MAMBA_X_IN_PLACE:
        # the block reads x in place (_MAMBA_X_IN_PLACE, header)
        mamba2_block_forward(ctx, stages, blk.st, blk.w, x, b, l, fs[0], fs[1], trace,
                             String("ops.step"))
        _block_out(ctx, ex, y_out, y_host, stages.residual_out, n)
        _d2d(ctx, blk.h_last, stages.h_last, h_n)
    else:
        if l == 1:
            _d2d(ctx, blk.dx, x, n)
            mamba2_block_forward(ctx, stages, blk.st, blk.w, blk.dx, b, 1, fs[0], fs[1], trace,
                                 String("ops.step"))
            _block_out(ctx, ex, y_out, y_host, stages.residual_out, n)
            _d2d(ctx, blk.h_last, stages.h_last, h_n)
        else:
            var dx = mamba_zeros(ctx, n)
            _d2d(ctx, dx, x, n)
            mamba2_block_forward(ctx, stages, blk.st, blk.w, dx, b, l, fs[0], fs[1], trace,
                                 String("ops.step"))
            _block_out(ctx, ex, y_out, y_host, stages.residual_out, n)
            _d2d(ctx, blk.h_last, stages.h_last, h_n)
            sc.parked_x.append(dx^)
    var words: List[Int] = [blk.st.buf_len]
    _write_words(ex, bufs[3], words)
    # lane x45c: the call-shaped stages parked, no wait
    sc.make_room(ctx)
    sc.parked_m2.append(stages^)


def _run_m3(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
            y_out: FP, y_host: Int = 0) raises:
    var ctx = ex.ctx.copy()
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_M3_STEP, counts[0], b, dm)
    _expect(OPS_M3_STEP, "weights", counts[1], block_part_floats(OPS_M3_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_M3_STEP, "state", counts[2], block_part_floats(OPS_M3_STEP, ints, BLOCK_PART_STATE))
    _expect(OPS_M3_STEP, "state words", counts[3], 2)
    var e = _m3_entry(ex, sc, bufs, counts, b, dm)
    var dims = Mamba3Dims.of(dm)
    var n = b * l * dm
    # lane restore-samba-resident-forward (2026-11-13): an inference-only
    # session (`sc.inference_only`, the Samba stack program) keeps its L > 1
    # stages and input copy in the entry, as the 0.8.55 resident forward kept
    # them per (B, L) (training/samba_resident.mojo `_ensure_shape`): no 43
    # fresh zero-filled buffers and no input allocation per layer per call.
    # A kept set is refilled with the constructor's zeros (`rezero`,
    # IDN_M3_SESSION_STAGE_REUSE), so the forward starts from the same bytes
    # and no bit moves. A set kept for another (B, L, buf_len) is parked, not
    # freed, while the stream may still read it (lane x45c).
    var hold = False
    comptime if IDN_M3_SESSION_STAGE_REUSE and MAMBA_GUARD == 0:
        comptime if MAMBA_FAST_DECODE_STAGE_ARENA:
            # lane fx-mamba-decode-arena (idea 7): a `keep_stages` session
            # (the per-call block programs; a recorded run goes through
            # `mamba_step_train_run`, never through here) holds its L > 1
            # Mamba-3 stages too, as Mamba-1/2 hold theirs
            hold = (sc.inference_only or sc.keep_stages) and l > 1
        else:
            hold = sc.inference_only and l > 1
    var kept = Optional[Mamba3DeviceStages]()
    var kept_x = Optional[DeviceBuffer[DType.float32]]()
    if hold:
        var q0 = sc.m3[e].st.buf_len
        if sc.m3[e].held_stages:
            var prev = sc.m3[e].held_stages.take()
            if prev.b == b and prev.l == l and prev.q0 == q0:
                comptime if MAMBA_FAST_DECODE_STAGE_ARENA:
                    if sc.m3[e].held_arena:
                        # ONE fill for the 33 stage views (the arena's own
                        # constructor fill), in stream order, no wait
                        sc.m3[e].held_arena.value().base.enqueue_fill(Float32(0.0))
                    else:
                        prev.rezero()
                else:
                    prev.rezero()
                kept = prev^
            else:
                sc.make_room(ctx)
                sc.parked_m3.append(prev^)
        comptime if MAMBA_FAST_DECODE_STAGE_ARENA:
            # lane fx-mamba-decode-arena (idea 7): a new held set is views of
            # ONE held arena (one allocation, one fill, no wait). A previous
            # set's arena (another shape, or a run that failed between take
            # and hold) is parked, not freed, while the stream may read it;
            # `release_parked` frees the parked views before it.
            if kept:
                pass
            else:
                if sc.m3[e].held_arena:
                    sc.make_room(ctx)
                    sc.parked_arena.append(sc.m3[e].held_arena.take())
                var held_cap = MambaArena.total(_m3_stage_sizes(b, l, q0, dims), MAMBA_GUARD)
                var held_arena = MambaArena(ctx, held_cap, MAMBA_GUARD)
                var held_new = Mamba3DeviceStages(ctx, b, l, q0, dims, held_arena)
                sc.m3[e].held_arena = held_arena^
                sc.m3[e].held_n = held_cap
                kept = held_new^
        if sc.m3[e].held_x:
            var px = sc.m3[e].held_x.take()
            if len(px) == n:
                kept_x = px^
            else:
                sc.make_room(ctx)
                sc.parked_x.append(px^)
    ref blk = sc.m3[e]
    var x = _view(ex, bufs[0], n)
    var trace = IdentityTrace.disabled()
    # The Mamba-3 decode session's body (the one body since lane
    # x5-mamba-on-executor): stages built per call at the open chunk's
    # length.
    var stages: Mamba3DeviceStages
    if kept:
        stages = kept.take()
    elif hold:
        stages = Mamba3DeviceStages(ctx, b, l, blk.st.buf_len, dims)
    else:
        stages = _m3_call_stages(ctx, blk, b, l, dims)
    var report_n = m3_report_floats(b, dm)
    var held_x = Optional[DeviceBuffer[DType.float32]]()
    comptime if _MAMBA_X_IN_PLACE:
        # the block reads x in place (_MAMBA_X_IN_PLACE, header)
        mamba3_block_forward(ctx, stages, blk.st, blk.w, x, b, l, trace, String("ops.step"))
        _block_out(ctx, ex, y_out, y_host, stages.residual_out, n)
    else:
        if l == 1:
            _d2d(ctx, blk.dx, x, n)
            mamba3_block_forward(ctx, stages, blk.st, blk.w, blk.dx, b, 1, trace, String("ops.step"))
            _block_out(ctx, ex, y_out, y_host, stages.residual_out, n)
        else:
            var dx: DeviceBuffer[DType.float32]
            if kept_x:
                dx = kept_x.take()
            elif hold:
                # overwritten whole by the copy below, so no zero fill
                dx = ctx.enqueue_create_buffer[DType.float32](n)
            else:
                dx = mamba_zeros(ctx, n)
            _d2d(ctx, dx, x, n)
            mamba3_block_forward(ctx, stages, blk.st, blk.w, dx, b, l, trace, String("ops.step"))
            _block_out(ctx, ex, y_out, y_host, stages.residual_out, n)
            if hold:
                held_x = dx^
            else:
                sc.parked_x.append(dx^)
    # The four reports, unless the caller requests none for this run
    # (`sc.skip_reports`, IDN_MAMBA3_REPORTS_ON_REQUEST: the stateless Samba
    # forward reads none; the 0.8.55 resident forward copied none).
    var copy_reports = True
    comptime if IDN_MAMBA3_REPORTS_ON_REQUEST:
        copy_reports = not sc.skip_reports
    if copy_reports:
        _d2d(ctx, blk.reports[0], stages.h_last, report_n[0])
        _d2d(ctx, blk.reports[1], stages.k_last, report_n[1])
        _d2d(ctx, blk.reports[2], stages.v_last, report_n[2])
        _d2d(ctx, blk.reports[3], stages.theta_last, report_n[3])
    var words: List[Int] = [blk.st.buf_len, 1 if blk.st.pending else 0]
    _write_words(ex, bufs[3], words)
    if hold:
        # kept by the entry; the next run refills them in stream order
        sc.m3[e].held_stages = stages^
        sc.m3[e].held_x = held_x^
        return
    # lane x45c: no wait; the stages are parked
    sc.make_room(ctx)
    sc.parked_m3.append(stages^)


def _m3_call_stages(ctx: DeviceContext, mut blk: _M3Block, b: Int, l: Int, dims: Mamba3Dims) raises -> Mamba3DeviceStages:
    """The stages of one run that holds none: built per call at the open
    chunk's length, or, under MAMBA_FAST_DECODE_STAGE_ARENA, an L = 1 run's
    views of the entry's decode arena."""
    comptime if MAMBA_FAST_DECODE_STAGE_ARENA:
        if l == 1:
            return _m3_decode_views(ctx, blk, b, dims)
    return Mamba3DeviceStages(ctx, b, l, blk.st.buf_len, dims)


def _same_ints(a: List[Int], b: List[Int]) -> Bool:
    return a == b


struct AttnPrefill(Movable):
    """The L > 1 stages and input buffer of attention step runs (lane x4b).
    A run handed one keeps them across runs and reuses them for the next
    compatible shape (reset to the constructor's zeros first, as the block
    session and the CausalLM prefill did); the op's own table run passes a
    fresh one, so its stages live for that run alone, as before. The
    backward reads the forward's stages and input from here."""
    var stages: Optional[LlamaDeviceStages]
    var dx: Optional[DeviceBuffer[DType.float32]]
    var lean: Bool
    #: the shape of the last forward these stages hold (for the backward)
    var b: Int
    var l: Int
    var pos0: Int
    #: lane r1 (ATTN_STEP_NEXT_NORM): the previous layer's run prepared
    #: (reset) these stages and wrote their norm1 (`ahead`); `prepare` then
    #: keeps them as they are for a run of the same shape (`ahead_used`).
    var ahead: Bool
    var ahead_used: Bool

    def __init__(out self):
        self.stages = Optional[LlamaDeviceStages]()
        self.dx = Optional[DeviceBuffer[DType.float32]]()
        self.lean = False
        self.b = 0
        self.l = 0
        self.pos0 = -1
        self.ahead = False
        self.ahead_used = False

    def clear(mut self):
        self.stages = None
        self.dx = None
        self.l = 0
        self.pos0 = -1
        self.ahead = False
        self.ahead_used = False

    def prepare(mut self, ctx: DeviceContext, b: Int, l: Int, smax: Int, dims: LlamaDims, window: Int,
                lean: Bool, keep_bytes: Bool = False) raises:
        """`keep_bytes` (lane fx-transformer-forward; promoted 2026-11-19 for
        the stateless forward, `attn_step_fresh_prefill`): the caller's
        promise that the run it prepares is forward-only, so reused stages of
        the same shape get their host flags back
        (`LlamaDeviceStages.reset_flags`) and keep their bytes; the forward
        writes each stage whole before reading it (modeling_llama
        `LLAMA_FAST_STAGE_KEEP`). False: the constructor's zeros again
        (`reset`: thirty fills and a wait)."""
        var reused = False
        self.ahead_used = False
        if self.stages:
            ref st = self.stages.value()
            reused = (st.b == b and st.l == l and st.s_max == smax and st.window == window
                      and st.dims.d_model == dims.d_model and st.dims.n_heads == dims.n_heads
                      and st.dims.n_kv == dims.n_kv and st.dims.head_dim == dims.head_dim
                      and st.dims.intermediate == dims.intermediate and self.lean == lean)
        if reused and self.ahead:
            # lane r1: already reset by the previous layer's run, which wrote
            # norm1 into them behind the reset (same stream).
            self.ahead_used = True
        elif reused:
            if keep_bytes:
                self.stages.value().reset_flags()
            else:
                # Every constructor zero and route flag back before the next run.
                self.stages.value().reset(ctx)
            self.stages.value().int15_on = False
        else:
            # Drop an incompatible owner before allocating its replacement.
            self.stages = None
            self.stages = LlamaDeviceStages(ctx, b, l, smax, dims, window, lean=lean)
            self.lean = lean
        self.ahead = False
        var n = b * l * dims.d_model
        if not self.dx or len(self.dx.value()) != n:
            self.dx = ctx.enqueue_create_buffer[DType.float32](n)
        self.b = b
        self.l = l


def attn_step_entry(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int],
                    ints: List[Int]) raises -> Int:
    """The index in `sc.attn` of the resident entry over these operands,
    built at the triple's first run (lane x4b: shared by the runs and the
    session arms that read the entry's structs)."""
    var e = -1
    for i in range(len(sc.attn)):  # driver-loop: over resident attention blocks
        if (sc.attn[i].w_addr == Int(bufs[1]) and sc.attn[i].s_addr == Int(bufs[2])
                and sc.attn[i].q_addr == Int(bufs[3]) and _same_ints(sc.attn[i].ints, ints)):
            e = i
    if e >= 0 and sc.attn[e].adopted:
        _expect(OPS_ATTN_STEP, "kv", counts[2], block_part_floats(OPS_ATTN_STEP, ints, BLOCK_PART_STATE))
        _expect(OPS_ATTN_STEP, "state words", counts[3], 1)
        return e
    _expect(OPS_ATTN_STEP, "weights", counts[1], block_part_floats(OPS_ATTN_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_ATTN_STEP, "kv", counts[2], block_part_floats(OPS_ATTN_STEP, ints, BLOCK_PART_STATE))
    _expect(OPS_ATTN_STEP, "state words", counts[3], 1)
    if e < 0:
        var ctx = ex.ctx.copy()
        if len(sc.lost_s) > 0:
            # a new entry reads its position from the word: an older failed
            # run's record over this cache is void
            sc.drop_lost(Int(bufs[2]))
        sc.attn.append(_AttnBlock(ctx, ex, bufs[1], counts[1], bufs[2], counts[2], bufs[3], ints))
        e = len(sc.attn) - 1
        comptime if LLAMA_FAST_REFUSE_WATERMARK:
            _attn_watermark_grant(ctx, sc, e)
    return e


def _attn_watermark_grant(ctx: DeviceContext, mut sc: BlockStepScratch, e: Int) raises:
    """MOJOLEARN_LLAMA_FAST_REFUSE_WATERMARK, once when entry `e` is built
    (never per run): give its cache the refusal watermark and the held scan
    scratch (`LlamaKVCache.wm_grant`), when the scratch's owner vouches for
    the cache's writers (`attn_cache_private`) and no other resident entry
    is built over the same state tensor. Two entries over one state tensor
    (the same caches under other weights or integers, or a Mamba entry)
    would each append behind the other's watermark: both lose it for good
    and scan the whole cache on every call.

    EVERY WRITER OF A GRANTED CACHE, and what it does to the watermark:
      the block forward's append (modeling_llama `kv_append2_kernel` and the
        two copies back): bit copies of scanned rows plus the new row, which
        the next call scans (L = 1) or the next whole scan covers (L > 1);
      `attn_append_restore` (a loop that ran bodies past its stop, lane
        fx-kv-inplace: `llama_kv_cache_rewind` repacks the
        overrun): reset, and the loop then drops the entry;
      a new entry (`_AttnBlock`, `attn_step_adopt`; after `forget`, `clear`,
        a failed run's eviction, a loop that ran bodies past its stop): the
        watermark starts at "nothing known";
      `attn_step_restart` (a caller's caches loaded before it, a zero fill,
        a rewind to position 0 for a training run): reset;
      `attn_step_fresh_prefill` (zero fill, position 0): reset;
      `BlockStepScratch.set_words` (`OpsSession.load_block` of the position
        word; the Samba program's `zero_state` fills the state and then
        loads the word): reset;
      `BlockStepScratch.attn_cache_written` (`OpsSession.load_block` of the
        state floats): reset;
      `OpsSession.load_into`, `declare_state`, `free`, the optimizer's
        weight write: `steps_forget`, the entry is dropped.
    The rotary inv_freq is written only when the entry's table is built."""
    if not sc.attn_cache_private:
        return
    var s_addr = sc.attn[e].s_addr
    var shared = False
    for i in range(len(sc.attn)):  # driver-loop: over resident attention blocks
        if i != e and sc.attn[i].s_addr == s_addr:
            sc.attn[i].kv.wm_revoke()
            shared = True
    for i in range(len(sc.m1)):  # driver-loop: over resident Mamba-1 blocks
        shared = shared or sc.m1[i].s_addr == s_addr
    for i in range(len(sc.m2)):  # driver-loop: over resident Mamba-2 blocks
        shared = shared or sc.m2[i].s_addr == s_addr
    for i in range(len(sc.m3)):  # driver-loop: over resident Mamba-3 blocks
        shared = shared or sc.m3[i].s_addr == s_addr
    if shared:
        return
    var d_model = sc.attn[e].ints[1]
    sc.attn[e].kv.wm_grant(ctx, d_model)


def attn_step_adopt(mut ex: DeviceExec, mut sc: BlockStepScratch, var weights: LlamaDeviceWeights, w: FP, s: FP,
                    q: FP, ints: List[Int], s0: Int, lean: Bool) raises:
    """Lane x4b: register the entry of the triple (w, s, q) over weights the
    caller built (the fixed15_v1 planes), so the op runs them; the word at
    q is written to s0."""
    _check_ints(OPS_ATTN_STEP, ints)
    sc.forget(Int(w))
    sc.forget(Int(s))
    var ctx = ex.ctx.copy()
    sc.attn.append(_AttnBlock(ctx, ex, weights^, w, s, q, ints, s0, lean))
    comptime if LLAMA_FAST_REFUSE_WATERMARK:
        _attn_watermark_grant(ctx, sc, len(sc.attn) - 1)
    var words: List[Int] = [s0]
    _write_words(ex, q, words)


def attn_step_restart(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int],
                      ints: List[Int], s0: Int, zero: Bool) raises:
    """Lane x4b: set the carried position to s0 (and, with `zero`, the two
    caches to the zeros a fresh `LlamaKVCache` holds, on the device) without
    rebuilding the resident entry: the entry's cache struct and the state
    word move together. The caller's own host bytes, when it loads caches,
    go into the state tensor before this call (same stream order)."""
    if s0 < 0 or s0 > ints[6]:
        raise Error("ValueError: ops: attention_step: cached tokens must be in [0, " + String(ints[6])
                    + "], got " + String(s0))
    if zero:
        var view = _view(ex, bufs[2], counts[2])
        view.enqueue_fill(Float32(0.0))
    for i in range(len(sc.attn)):  # driver-loop: over resident attention blocks
        if sc.attn[i].s_addr == Int(bufs[2]) and sc.attn[i].q_addr == Int(bufs[3]):
            sc.attn[i].kv.s = s0
        comptime if LLAMA_FAST_REFUSE_WATERMARK:
            # every entry over this state tensor, whatever its word: the
            # caller wrote the caches before this call, or `zero` did
            if sc.attn[i].s_addr == Int(bufs[2]):
                sc.attn[i].kv.wm_reset()
    var words: List[Int] = [s0]
    _write_words(ex, bufs[3], words)


def attn_append_restore(mut ex: DeviceExec, mut sc: BlockStepScratch, kvs: List[Int], qs: List[Int]) raises:
    """Lane fx-kv-inplace (2026-11-18): a bounded loop ran bodies past its
    stop over the append-only caches at `kvs` (their position words at
    `qs`, already restored on the device by the loop's gate). The gate does
    not restore such a cache, and a full-causal cache is repacked at the new
    stride by every append (`LlamaKVCache`), so the bytes the extra bodies
    left sit at a stride the restored word does not describe. Each cache is
    taken back to its restored word (`llama_kv_cache_rewind`: the committed
    slots repacked at the word's stride, the overrun span zeroed), so the
    state a loop leaves does not depend on how many bodies were enqueued
    between two status reads.

    How far the extra bodies appended is the resident entry's used length,
    or, when one of those bodies failed and its entry was dropped
    (`block_step_run`), the record that failure left
    (`BlockStepScratch.lost_s`); the repack then runs through two temporary
    buffers and waits once more. One wait for all the words (the caller is
    leaving the loop and reads the state next); the repacks of resident
    entries are queued. A cache that never ran past its word costs nothing."""
    var n = len(kvs)
    if n == 0 or len(qs) != n or (len(sc.attn) == 0 and len(sc.lost_s) == 0):
        return
    var host = List[Float32](length=n, fill=Float32(0))
    for k in range(n):  # driver-loop: over append-only caches (one queued word read each)
        ex.download_async(FP(unsafe_from_address=Int(host.unsafe_ptr()) + 4 * k), FP(unsafe_from_address=qs[k]), 1)
    ex.sync()
    var ctx = ex.ctx.copy()
    for k in range(n):  # driver-loop: over append-only caches (one queued repack each)
        var word = Int(bitcast[DType.int32](host[k]))
        var e = _attn_by_state(sc, kvs[k], qs[k])
        if e >= 0:
            if sc.attn[e].ints[7] != 0 or word < 0 or word >= sc.attn[e].kv.s:
                continue
            ref blk = sc.attn[e]
            llama_kv_cache_rewind(ctx, blk.stages.k_cache, blk.stages.v_cache, blk.kv, word)
            continue
        var li = -1
        for i in range(len(sc.lost_s)):  # driver-loop: over lost-append records
            if sc.lost_s[i] == kvs[k] and sc.lost_q[i] == qs[k]:
                li = i
        if li < 0:
            continue
        var used = sc.lost_used[li]
        var ints = sc.lost_ints[li].copy()
        sc.drop_lost(kvs[k])
        if word < 0 or word >= used:
            continue
        var dims = _attn_dims(ints)
        var opts = _attn_opts(ints)
        var kv = _attn_kv_views(ctx, ex, FP(unsafe_from_address=kvs[k]), ints, dims, opts, used)
        var span = ints[0] * ints[3] * used * ints[4]
        var tk = ctx.enqueue_create_buffer[DType.float32](span)
        var tv = ctx.enqueue_create_buffer[DType.float32](span)
        llama_kv_cache_rewind(ctx, tk, tv, kv, word)
        ex.sync()
        _ = len(tk)  # keep-alive past the queued repack and copies
        _ = len(tv)
    _ = len(host)  # keep-alive past the queued word reads


comptime _NP = MutPointer[Float32, MutAnyOrigin]


def _run_attn_on(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
                 y_out: FP, mut pre: AttnPrefill, forward_only: Bool, norm1_ready: Bool = False,
                 next_sumsq: Int = 0, next_out: Int = 0, next_w: Int = 0, next_eps: Float32 = 0.0,
                 write_word: Bool = True) raises:
    """One run of the attention step. Lane r1 (ATTN_STEP_NEXT_NORM, L > 1
    stages only): `norm1_ready` when the previous layer's run wrote this
    run's norm1 into `pre` (`pre.ahead`; honoured only when `prepare` kept
    those stages), and `next_*` (device addresses, 0: none) the next layer's
    norm1 sumsq / out destinations, weight and eps: this run's residual2 add
    then computes them in the same launch (`residual_rms_norm_kernel`), as
    the byte LM's hand step did (training/byte_lm.mojo at 5b5490960d)."""
    var ctx = ex.ctx.copy()
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_ATTN_STEP, counts[0], b, dm)
    var e = attn_step_entry(ex, sc, bufs, counts, ints)
    ref blk = sc.attn[e]
    var n = b * l * dm
    var x = _view(ex, bufs[0], n)
    var y = _view(ex, y_out, n)
    var trace = IdentityTrace.disabled()
    var pos0 = blk.kv.s
    if l == 1:
        # `_session_step_run_sync`: x into the resident buffer, the block on
        # the resident L = 1 stages; the cache advances on the device.
        _d2d(ctx, blk.dx, x, n)
        llama_decoder_layer_forward(
            ctx, blk.stages, blk.kv, blk.rope, blk.w, blk.dx,
            b, 1, pos0, trace, String("ops.step"), forward_only=forward_only)
        if n != len(blk.stages.residual2):
            raise Error("ops: attention_step: the L = 1 stages hold a different output length")
        _d2d(ctx, y, blk.stages.residual2, n)
    else:
        # `_session_forward_run`: L-shaped stages, the caller's (kept) or
        # this run's own.
        var dims = _attn_dims(ints)
        # TOMBSTONE: MOJOLEARN_LLAMA_FAST_STAGE_KEEP, this site's use (`forward_only` passed as `keep_bytes`: a forward with the caller's caches and an inference-only stack layer kept their stage bytes; LOSS, A/B round 2 NVIDIA L40S: samba-forward 4.609 -> 4.751 ms, 1.03, bases 4.609-4.635) deleted 2026-11-19 by lane/fx-verdicts4; code recoverable at 8f3f639b13.
        # Restore: git apply experiments/removed/MOJOLEARN_LLAMA_FAST_STAGE_KEEP-stack-layer.patch; record in docs/TOMBSTONES.md.
        pre.prepare(ctx, b, l, ints[6], dims, ints[7], blk.lean)
        _d2d(ctx, pre.dx.value(), x, n)
        var n1r = norm1_ready and pre.ahead_used
        pre.ahead_used = False
        if next_sumsq != 0 and next_out != 0 and next_w != 0:
            llama_decoder_layer_forward(
                ctx, pre.stages.value(), blk.kv, blk.rope, blk.w, pre.dx.value(), b, l, pos0, trace,
                String("ops.step.forward"), norm1_ready=n1r,
                next_norm_sumsq=Optional(_NP(unsafe_from_address=next_sumsq)),
                next_norm_out=Optional(_NP(unsafe_from_address=next_out)),
                next_norm_weight=Optional(_NP(unsafe_from_address=next_w)),
                next_norm_eps=Optional(next_eps), forward_only=forward_only)
        else:
            llama_decoder_layer_forward(
                ctx, pre.stages.value(), blk.kv, blk.rope, blk.w, pre.dx.value(), b, l, pos0, trace,
                String("ops.step.forward"), norm1_ready=n1r, forward_only=forward_only)
        _d2d(ctx, y, pre.stages.value().residual2, n)
        pre.pos0 = pos0
    # `write_word` False (lane fix-transformer-decode-door): the caller
    # writes the position word itself, once, before anything can read it
    # (`attn_step_word`; the word is read only when an entry is built).
    if write_word:
        var words: List[Int] = [blk.kv.s]
        _write_words(ex, bufs[3], words)


def attn_step_word(mut ex: DeviceExec, q: FP, s: Int) raises:
    """Queue the op's position word at `q` to `s` (lane
    fix-transformer-decode-door): the deferred write of a caller that ran
    steps with `write_word=False`. The word is a mirror of the resident
    entry's position, read only when an entry is built over it
    (`_AttnBlock`), so writing it once before any such build is the same
    state as writing it after every step. No wait."""
    var words: List[Int] = [s]
    _write_words(ex, q, words)


def _run_attn(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
              y_out: FP) raises:
    if sc.inference_only:
        # lane restore-samba-resident-forward (2026-11-13): an inference-only
        # session (the Samba stack program) runs its attention layers as the
        # 0.8.55 resident forward did (training/samba_resident.mojo
        # `_forward_blocks`): `forward_only=True` (no backward reads these
        # stages) on L > 1 stages kept in the entry across runs (reset to the
        # constructor's zeros by `prepare`), not a fresh `AttnPrefill` per
        # call. The kept stages move out for the run (it borrows the scratch
        # mutably) and back in after it, as `attn_step_train_run` does. Bits:
        # the forward-only launch set is the resident forward's, so these
        # layers' bytes are the 0.8.55 resident ones on every vendor.
        var e = attn_step_entry(ex, sc, bufs, counts, ints)
        var kept = AttnPrefill()
        swap(kept, sc.attn[e].kept)
        _run_attn_on(ex, sc, bufs, counts, ints, y_out, kept, True)
        swap(kept, sc.attn[e].kept)
        _ = kept^
        return
    var pre = AttnPrefill()
    _run_attn_on(ex, sc, bufs, counts, ints, y_out, pre, False)
    if pre.stages:
        # lane x45c: the run's own L-shaped stages parked, no wait
        sc.make_room(ex.ctx)
        sc.parked_attn.append(pre^)
    else:
        _ = pre^


def _step_run[CODE: Int](
    mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
    fs: List[Float32], y_out: FP,
) raises:
    comptime if CODE == OPS_M1_STEP:
        _run_m1(ex, sc, bufs, counts, ints, y_out)
    elif CODE == OPS_M2_STEP:
        _run_m2(ex, sc, bufs, counts, ints, fs, y_out)
    elif CODE == OPS_M3_STEP:
        _run_m3(ex, sc, bufs, counts, ints, y_out)
    else:
        _run_attn(ex, sc, bufs, counts, ints, y_out)


def attn_step_run(
    mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP,
) raises:
    """`block_step_run` of OPS_ATTN_STEP alone (lane x4-transformer-on-
    executor): the transformer binding's decode session runs the step op
    through this, so that binding instantiates the attention body only, not
    the Mamba bodies of the table. Same checks, same body, same eviction of
    the pair's entry on a failed run."""
    _check_ints(OPS_ATTN_STEP, ints)
    if len(bufs) != 4 or len(counts) != 4:
        raise Error("ops: attention_step takes four tensors (x, weights, kv, words)")
    try:
        _run_attn(ex, sc, bufs, counts, ints, y_out)
    except err:
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err


def attn_step_run_kept(
    mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP,
    mut pre: AttnPrefill, forward_only: Bool, write_word: Bool = True,
) raises:
    """`attn_step_run` with the caller's L > 1 stages (lane x4b): the block
    session keeps them across calls and its backward reads them; the
    CausalLM stack shares one across its layers. `forward_only` is the
    block forward's promise that no backward reads this run's stages
    (`llama_decoder_layer_forward`'s forward-only fusions). Same body, same
    eviction on a failed run (the kept stages are dropped too)."""
    _check_ints(OPS_ATTN_STEP, ints)
    if len(bufs) != 4 or len(counts) != 4:
        raise Error("ops: attention_step takes four tensors (x, weights, kv, words)")
    try:
        _run_attn_on(ex, sc, bufs, counts, ints, y_out, pre, forward_only, write_word=write_word)
    except err:
        pre.clear()
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err


def attn_step_run_kept_host(
    mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
    px: Int, py: Int, mut pre: AttnPrefill, forward_only: Bool, write_word: Bool = True,
) raises:
    """One decode token (L = 1) from and to the caller's host arrays (lane
    fable-perf-neural-0855, 2026-11-13): x at `px` is copied straight into
    the resident entry's `dx` and `residual2` comes straight back to `py`,
    as 0.8.55's `_session_step_run` did (bindings/_mojolearn_transformer.mojo
    at 4e927c9249:2515: upload into `s.dx`, the block, download from
    `residual2`, one wait). The generic run went through the arm's `xio` /
    `yio` buffers and queued two device-to-device copies per token on top
    (`_run_attn_on`'s L = 1 branch); the board's `transformer-decode/gaussian`
    (2048 steps) read 1105 -> 1406 ms on the L40S against 0.8.55. Same
    `llama_decoder_layer_forward` on the same stages, same state word, so
    the bits are the same; only the two copies go. `bufs[0]` names no
    storage this run reads (the entry is matched on bufs[1..3]);
    `counts[0]` is B * d_model. Same eviction on a failed run."""
    _check_ints(OPS_ATTN_STEP, ints)
    if len(bufs) != 4 or len(counts) != 4:
        raise Error("ops: attention_step takes four tensors (x, weights, kv, words)")
    if px == 0 or py == 0:
        raise Error("ops: attention_step: null x or y host address")
    try:
        var ctx = ex.ctx.copy()
        var b = ints[0]
        var dm = ints[1]
        var n = b * dm
        if counts[0] != n:
            raise Error("ops: attention_step: a decode step takes one token per row")
        var e = attn_step_entry(ex, sc, bufs, counts, ints)
        ref blk = sc.attn[e]
        var trace = IdentityTrace.disabled()
        var pos0 = blk.kv.s
        if n > len(blk.dx):
            raise Error("ops: attention_step: the L = 1 stages hold a different input length")
        ctx.enqueue_copy(dst_buf=blk.dx.create_sub_buffer[DType.float32](0, n), src_ptr=f32_ptr(px))
        llama_decoder_layer_forward(
            ctx, blk.stages, blk.kv, blk.rope, blk.w, blk.dx,
            b, 1, pos0, trace, String("ops.step"), forward_only=forward_only)
        if n != len(blk.stages.residual2):
            raise Error("ops: attention_step: the L = 1 stages hold a different output length")
        ctx.enqueue_copy(dst_ptr=f32_ptr(py), src_buf=blk.stages.residual2.create_sub_buffer[DType.float32](0, n))
        # `write_word` False: the caller (AttnStepArm.run) writes the
        # position word lazily (`attn_step_word`), not one fill per token.
        if write_word:
            var words: List[Int] = [blk.kv.s]
            _write_words(ex, bufs[3], words)
    except err:
        pre.clear()
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err


#: Lane r1-restore-measured-winners (2026-11-09): the attention step op's
#: stateless FRESH PREFILL arm. Default: `attn_step_fresh_prefill` below, the
#: launch sequence of 0.8.66's block session prefill
#: (`_transformer_run_session[discard_cache=True]`, restored from git at
#: 616dbdf75f^ / v0.8.66 transformer/transformer_binding_impl.mojo; the path
#: `TransformerBlock.forward` with no state took on every GPU binding before
#: lanes x4 / x4b). `-D MOJOLEARN_TRANSFORMER_GENERIC_PREFILL` is lane x4b's
#: generic op run (`AttnStepArm.block_forward`: x up into the arm's resident
#: buffer, `attn_step_run_kept`, the two device copies x -> stages input and
#: residual2 -> y, y down), kept for the boxes' A/B. Both arms run the same
#: `llama_decoder_layer_forward` (forward_only) on the same zero cache, so
#: the bits are the same; the restore is a measured-winner rule (AGENTS.md
#: 2026-11-09: no measurement showed the generic run at least as fast).
# TOMBSTONE: MOJOLEARN_TRANSFORMER_GENERIC_PREFILL (unmeasured opt-in arm) deleted 2026-11-15 by lane/expdel2-23-neural; code recoverable at 29b23a6003.
# Restore: git apply experiments/removed/MOJOLEARN_TRANSFORMER_GENERIC_PREFILL.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
comptime TRANSFORMER_GENERIC_PREFILL = is_defined["MOJOLEARN_TRANSFORMER_GENERIC_PREFILL"]()


def attn_step_fresh_prefill(
    mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
    px: Int, py: Int, l: Int, mut pre: AttnPrefill,
) raises -> Int:
    """The stateless prefill of `l` tokens per row on the op's resident entry
    over (bufs[1] weights, bufs[2] kv, bufs[3] word): the caches zero-filled
    on the device (the bytes a fresh `LlamaKVCache` holds, nothing of them
    crosses the host), the caller's x (host `px`) straight into the L-shaped
    stages' input, one forward (forward_only: no backward reads these
    stages), residual2 straight down to the caller's y (host `py`), one wait.
    Returns the cached token count (`l`). 0.8.66's
    `_transformer_run_session[True]` body, with the entry's views in place of
    the session workspace's buffers; its two waits (after the uploads, after
    the download) are kept as they were. bufs[0] (x) is not read."""
    _check_ints(OPS_ATTN_STEP, ints)
    if len(bufs) != 4 or len(counts) != 4:
        raise Error("ops: attention_step takes four tensors (x, weights, kv, words)")
    if l < 1:
        raise Error("transformer_forward_fresh: B and L must be positive")
    try:
        var ctx = ex.ctx.copy()
        var b = ints[0]
        var dm = ints[1]
        var n = b * l * dm
        # A discarded cache starts at position 0: a new entry reads the word
        # it is built over, which still holds the previous call's position
        # (a block session whose shape changed keeps its word buffer, so a
        # 16-token forward then a 1-token forward refused "cached tokens must
        # be in [0, 1], got 16"). Zero the word before a new entry reads it.
        var known = False
        for i in range(len(sc.attn)):  # driver-loop: over resident attention blocks
            if (sc.attn[i].w_addr == Int(bufs[1]) and sc.attn[i].s_addr == Int(bufs[2])
                    and sc.attn[i].q_addr == Int(bufs[3]) and _same_ints(sc.attn[i].ints, ints)):
                known = True
        if not known:
            var zero: List[Int] = [0]
            _write_words(ex, bufs[3], zero)
        var e = attn_step_entry(ex, sc, bufs, counts, ints)
        ref blk = sc.attn[e]
        var dims = _attn_dims(ints)
        # The L-shaped stages (0.8.66: the session workspace's), reset to the
        # constructor's zeros when reused, as `_stage_reset_on()` did by default.
        # This run is forward-only (below), so `prepare` keeps the stage bytes
        # (LLAMA_FAST_STAGE_KEEP, the default since 2026-11-19, confirmed 2026-11-20).
        pre.prepare(ctx, b, l, ints[6], dims, ints[7], blk.lean, True)
        # The bytes a fresh `LlamaKVCache` holds, on the device, no upload.
        # TOMBSTONE: MOJOLEARN_IDN_TF_SESSION_LEAN (loser: transformer-forward 1.350) deleted 2026-10-11 by 68d9654bb; code recoverable at 68d9654bb^.
        # Restore: git apply experiments/removed/MOJOLEARN_IDN_TF_SESSION_LEAN.patch
        blk.kv.k.enqueue_fill(Float32(0.0))
        blk.kv.v.enqueue_fill(Float32(0.0))
        ctx.enqueue_copy(dst_buf=pre.dx.value(), src_ptr=f32_ptr(px))
        ctx.synchronize()
        blk.kv.s = 0
        comptime if LLAMA_FAST_REFUSE_WATERMARK:
            blk.kv.wm_reset()
        var trace = IdentityTrace.disabled()
        # forward_only: no backward reads this entry's stages.
        llama_decoder_layer_forward(ctx, pre.stages.value(), blk.kv, blk.rope, blk.w, pre.dx.value(),
                                    b, l, 0, trace, String("py"), forward_only=True)
        pre.pos0 = 0
        # One wait for the download (the fresh entry's pattern). The op's
        # state word follows the entry's cache position (one queued word,
        # so a later op run over this triple reads the right position).
        ctx.enqueue_copy(dst_ptr=f32_ptr(py), src_buf=pre.stages.value().residual2.create_sub_buffer[DType.float32](0, n))
        var words: List[Int] = [blk.kv.s]
        _write_words(ex, bufs[3], words)
        ctx.synchronize()
        return blk.kv.s
    except err:
        pre.clear()
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err


# ---------------------------------------------------------------- training through the attention step
# Lane x7-byte-lm-on-executor (2026-11-09; plan row X7, EXECUTOR_X1_API.md
# section 2 "Training through a step op"). A RECORDED run of OPS_ATTN_STEP
# (the session's tape is recording: Program.fit's replay) is the
# zero-state prefill at positions [0, L) through the op's own body with its
# stages kept in the pair's entry (not forward-only), and the session
# records it as ONE custom tape node over [x, weights] with tag
# ATTN_STEP_TAG. Its backward rule (autograd/rules_step_ops.mojo) is the
# transformer stage rule (`llama_decoder_layer_backward_autograd`, the one
# the block session's backward and the byte LM trainer ran) over those kept
# stages: d_x into the input's gradient and the nine weight gradients into
# the packed weights tensor's gradient at their `block_offsets` slots (the
# gaps stay the tape's zeros). Same launches as `AttnStepArm.block_backward`,
# so a block trained through the op has that backward's bits.
comptime ATTN_STEP_TAG = 108

# Lane r1-restore-measured-winners (2026-11-09; AGENTS.md "A measured winner
# is never deleted on reasoning"): the byte LM hand step's cross-layer
# fusion, restored inside the step op. The hand step passed each layer the
# next layer's norm1 destinations (`next_norm_*`) and the next layer
# `norm1_ready` (training/byte_lm.mojo `_byte_layer_forward_deferred` at
# 5b5490960d), so a layer's residual2 add and the next layer's input RMSNorm
# are one launch (`residual_rms_norm_kernel`: the exact S22 add, then S1-S4
# with S1's ascending fold unchanged, so no bit moves). Lane x7 put each
# layer behind its own OPS_ATTN_STEP and dropped it. Here a recorded run
# whose input is the previous recorded run's output links the two entries
# (`next_s` / `next_q`); at the next pass the earlier layer prepares the
# later one's SPARE stages, writes its norm1 into them behind their reset,
# and the later layer runs on those stages with norm1_ready (its old kept
# stages become the spare: their backward ran in the previous pass). A run
# whose input is not that output, a different shape or a refused fusion
# (`residual_next_norm_fusion_enabled`: the measured route's own rule, Apple
# column, RMSNorm without bias, within RESIDUAL_NORM_FUSE_MAX_ELEMS) computes
# its norm1 itself, as before. Cost: one more stage set per fused layer.
# Only the column the route runs on compiles it.
# TOMBSTONE: MOJOLEARN_BYTE_LM_GENERIC_OPS (unmeasured opt-in arm) deleted 2026-11-15 by lane/expdel2-23-neural; code recoverable at b4f0e643f7.
# Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_GENERIC_OPS.patch; record in docs/TOMBSTONES.md.
# TOMBSTONE: MOJOLEARN_BYTE_LM_GENERIC_OPS (decided B arm, generic OPS_EMBEDDING and no next-norm fusion; L40S ba08dac006 lm-train-step bytes 35.71 -> 48.54 ms, B 1.36x slower) deleted 2026-11-17 by lane/fb12-loser-arms; code recoverable at 2203db7094.
# Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_GENERIC_OPS.fb12-loser-arms.patch; record in docs/TOMBSTONES.md.
comptime ATTN_STEP_NEXT_NORM = TARGET_COLUMN == COLUMN_APPLE


def _attn_by_state(sc: BlockStepScratch, s_addr: Int, q_addr: Int) -> Int:
    """The index of the attention entry over (state, word) addresses, or -1."""
    for i in range(len(sc.attn)):  # driver-loop: over resident attention blocks
        if sc.attn[i].s_addr == s_addr and sc.attn[i].q_addr == q_addr:
            return i
    return -1


def attn_step_train_run(
    mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP,
) raises -> Int:
    """A recorded (training) run of OPS_ATTN_STEP: the carried position goes
    to 0 (no cache fill: a prefill from position 0 reads only the keys and
    values it appends), the block runs with its stages kept in the pair's
    entry for the backward, and the word records the L tokens appended.
    Returns L. A failed run drops the pair's entry, as every run does."""
    _check_ints(OPS_ATTN_STEP, ints)
    if len(bufs) != 4 or len(counts) != 4:
        raise Error("ops: attention_step takes four tensors (x, weights, kv, words)")
    var l = _tokens(OPS_ATTN_STEP, counts[0], ints[0], ints[1])
    if l > ints[6]:
        raise Error("ValueError: ops: attention_step: a training run of " + String(l)
                    + " tokens needs s_max >= " + String(l) + ", got " + String(ints[6]))
    try:
        attn_step_restart(ex, sc, bufs, counts, ints, 0, False)
        var e = attn_step_entry(ex, sc, bufs, counts, ints)
        if sc.attn[e].adopted:
            raise Error("ValueError: ops: attention_step: an adopted (fixed15) block has no backward")
        # The entry's kept stages move out for the run (the run borrows the
        # scratch mutably) and back in after it.
        var pre = AttnPrefill()
        swap(pre, sc.attn[e].kept)
        var consume = False
        var n_sumsq = 0
        var n_out = 0
        var n_w = 0
        var n_eps = Float32(0.0)
        var ne_s = 0
        var ne_q = 0
        var npre = AttnPrefill()
        comptime if ATTN_STEP_NEXT_NORM:
            var xa = Int(bufs[0])
            var follows = l > 1 and sc.last_train_y != 0 and sc.last_train_y == xa
            if follows:
                # the previous recorded run fed this one: link the two entries
                var pe = _attn_by_state(sc, sc.last_train_s, sc.last_train_q)
                if pe >= 0 and pe != e:
                    sc.attn[pe].next_s = Int(bufs[2])
                    sc.attn[pe].next_q = Int(bufs[3])
            var has_spare = False
            if sc.attn[e].spare.stages:
                has_spare = True
            consume = follows and sc.attn[e].ahead_y == xa and has_spare
            sc.attn[e].ahead_y = 0
            if consume:
                # run on the stages the previous layer prepared; the old kept
                # stages (their backward ran) become the spare
                var ahead = AttnPrefill()
                swap(ahead, sc.attn[e].spare)
                swap(pre, ahead)
                swap(ahead, sc.attn[e].spare)
                pre.ahead = True
            else:
                pre.ahead = False
            if l > 1 and sc.attn[e].next_s != 0:
                var ne = _attn_by_state(sc, sc.attn[e].next_s, sc.attn[e].next_q)
                if (ne >= 0 and ne != e and not sc.attn[ne].adopted and _same_ints(sc.attn[ne].ints, ints)
                        and residual_next_norm_fusion_enabled(ints[0] * l, ints[1], sc.attn[ne].w.opts.norm_kind,
                                                              sc.attn[ne].w.opts.norm_bias)):
                    swap(npre, sc.attn[ne].spare)
                    npre.prepare(ex.ctx, ints[0], l, ints[6], _attn_dims(ints), ints[7], sc.attn[ne].lean)
                    n_sumsq = Int(npre.stages.value().norm1_sumsq.unsafe_ptr())
                    n_out = Int(npre.stages.value().norm1_out.unsafe_ptr())
                    n_w = Int(sc.attn[ne].w.norm1_w.unsafe_ptr())
                    n_eps = sc.attn[ne].w.eps
                    ne_s = sc.attn[ne].s_addr
                    ne_q = sc.attn[ne].q_addr
        _run_attn_on(ex, sc, bufs, counts, ints, y_out, pre, False, consume, n_sumsq, n_out, n_w, n_eps)
        e = attn_step_entry(ex, sc, bufs, counts, ints)
        swap(pre, sc.attn[e].kept)
        sc.attn[e].train_l = l
        comptime if ATTN_STEP_NEXT_NORM:
            if n_sumsq != 0:
                var ne2 = _attn_by_state(sc, ne_s, ne_q)
                if ne2 >= 0:
                    npre.ahead = True
                    swap(npre, sc.attn[ne2].spare)
                    sc.attn[ne2].ahead_y = Int(y_out)
                else:
                    # the next entry went away: park the written stages (the
                    # stream may still be writing them)
                    sc.make_room(ex.ctx)
                    sc.parked_attn.append(npre^)
                    npre = AttnPrefill()
            sc.last_train_s = Int(bufs[2])
            sc.last_train_q = Int(bufs[3])
            sc.last_train_y = Int(y_out)
        _ = pre^
        _ = npre^
    except err:
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        sc.last_train_y = 0
        raise err
    return l


def attn_step_backward(
    mut ex: DeviceExec, mut sc: BlockStepScratch, w: FP, s: FP, q: FP, d_out: FP, d_x: FP, d_w: FP,
) raises:
    """The backward of the last recorded run over (w, s, q): the transformer
    stage rule over the entry's kept stages, d(residual2) `d_out`; writes
    d_x (b * L * d_model floats) and the nine weight gradients into `d_w`,
    the packed weights' gradient (block_offsets layout). Enqueued on the
    executor's stream; the stage rule's own refusal scan of d_out waits."""
    var e = -1
    for i in range(len(sc.attn)):  # driver-loop: over resident attention blocks
        if sc.attn[i].w_addr == Int(w) and sc.attn[i].s_addr == Int(s) and sc.attn[i].q_addr == Int(q):
            e = i
    if e < 0 or sc.attn[e].train_l < 1:
        raise Error("ops: attention_step backward: the recorded forward's stages are gone"
                    + " (its weights or state were written outside the step after the run)")
    var ints = sc.attn[e].ints.copy()
    var b = ints[0]
    var dm = ints[1]
    var l = sc.attn[e].train_l
    var dims = _attn_dims(ints)
    var n = b * l * dm
    var ctx = ex.ctx.copy()
    var lean = sc.attn[e].lean
    var key: List[Int] = [b, l, ints[1], ints[2], ints[3], ints[4], ints[5], ints[7], Int(lean)]
    # The backward stages move out of the entry for the call (the rule
    # borrows the entry's forward stages, weights and rotary table).
    var bst = Optional[LlamaBackwardStages]()
    swap(bst, sc.attn[e].bst)
    if not bst or not (key == sc.attn[e].bkey):
        bst = None
        bst = LlamaBackwardStages(ctx, b, l, l, dims, lean=lean)
        sc.attn[e].bkey = key^
    var dout = _view(ex, d_out, n)
    var off = IdentityTrace.disabled()
    try:
        ref blk = sc.attn[e]
        if l == 1:
            # L = 1 ran on the entry's resident stages and input buffer.
            llama_decoder_layer_backward_autograd(
                ctx, bst.value(), blk.stages, blk.w, blk.rope.cos, blk.rope.sin, blk.dx,
                dout, b, l, 0, off, String("ops.step.backward"))
        else:
            llama_decoder_layer_backward_autograd(
                ctx, bst.value(), blk.kept.stages.value(), blk.w, blk.rope.cos, blk.rope.sin,
                blk.kept.dx.value(), dout, b, l, 0, off, String("ops.step.backward"))
        ref g = bst.value()
        _d2d(ctx, _view(ex, d_x, n), g.d_x, n)
        var sizes = block_weight_sizes(OPS_ATTN_STEP, ints)
        var at = block_offsets(sizes, block_guard(OPS_ATTN_STEP))
        # block_weight_sizes order: norm1, norm2, q, k, v, o, gate, up, down
        _d2d(ctx, _view(ex, d_w + at[0], sizes[0]), g.dw_norm1, sizes[0])
        _d2d(ctx, _view(ex, d_w + at[1], sizes[1]), g.dw_norm2, sizes[1])
        _d2d(ctx, _view(ex, d_w + at[2], sizes[2]), g.dw_q, sizes[2])
        _d2d(ctx, _view(ex, d_w + at[3], sizes[3]), g.dw_k, sizes[3])
        _d2d(ctx, _view(ex, d_w + at[4], sizes[4]), g.dw_v, sizes[4])
        _d2d(ctx, _view(ex, d_w + at[5], sizes[5]), g.dw_o, sizes[5])
        _d2d(ctx, _view(ex, d_w + at[6], sizes[6]), g.dw_gate, sizes[6])
        _d2d(ctx, _view(ex, d_w + at[7], sizes[7]), g.dw_up, sizes[7])
        _d2d(ctx, _view(ex, d_w + at[8], sizes[8]), g.dw_down, sizes[8])
    except err:
        swap(bst, sc.attn[e].bst)
        raise err
    swap(bst, sc.attn[e].bst)
    _ = bst^


# ---------------------------------------------------------------- training through the Mamba steps
# Lane x56d-mamba-samba-train-and-host (2026-11-09; the attention step's
# contract above, for the three Mamba codes). A RECORDED run of OPS_M1_STEP /
# OPS_M2_STEP / OPS_M3_STEP is the zero-state prefill through the op's
# block forward: the state tensor is zeroed and the words set to [0] / [0, 0]
# first, the L-token stages and an input copy are kept in the pair's entry
# (not parked), the state tensor holds the post-prefill state afterwards and
# the words the post-prefill integers. The session records it as ONE custom
# node over [x, weights] with tag M{1,2,3}_STEP_TAG; its VJP (StepOpRules,
# autograd/rules_step_ops.mojo) is `mamba_step_backward`: the Mamba stage
# rules (autograd/rules_mamba.mojo, rules_mamba3.mojo) over the KEPT stages.
# The rules ran the block forward themselves before this lane
# (`mamba{1,2,3}_tape_backward`): the forward a training step ran is no
# longer computed twice.
comptime M1_STEP_TAG = 109
comptime M2_STEP_TAG = 110
comptime M3_STEP_TAG = 111


def step_train_tag(code: Int) raises -> Int:
    """The tape tag of a recorded run of step op `code`."""
    if code == OPS_M1_STEP:
        return M1_STEP_TAG
    if code == OPS_M2_STEP:
        return M2_STEP_TAG
    if code == OPS_M3_STEP:
        return M3_STEP_TAG
    if code == OPS_ATTN_STEP:
        return ATTN_STEP_TAG
    raise Error("ops: " + String(code) + " is not a block step op")


def step_tag_code(tag: Int) -> Int:
    """The step op of a recorded-run tag (0: not a step tag)."""
    if tag == M1_STEP_TAG:
        return OPS_M1_STEP
    if tag == M2_STEP_TAG:
        return OPS_M2_STEP
    if tag == M3_STEP_TAG:
        return OPS_M3_STEP
    if tag == ATTN_STEP_TAG:
        return OPS_ATTN_STEP
    return 0


def mamba_step_train_run(
    mut ex: DeviceExec, mut sc: BlockStepScratch, code: Int, bufs: List[FP], counts: List[Int], ints: List[Int],
    fs: List[Float32], y_out: FP,
) raises -> Int:
    """A recorded (training) run of a Mamba step op: the zero-state prefill
    with its stages kept in the pair's entry for the backward. Returns L. A
    failed run drops the pair's entry, as every run does."""
    _ = block_step_check(code, len(bufs), ints, fs)
    var ctx = ex.ctx.copy()
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(code, counts[0], b, dm)
    _expect(code, "weights", counts[1], block_part_floats(code, ints, BLOCK_PART_WEIGHTS))
    _expect(code, "state", counts[2], block_part_floats(code, ints, BLOCK_PART_STATE))
    var n = b * l * dm
    try:
        # the zero state of a training run (the carried state is not an input
        # of the recorded graph)
        var sv = _view(ex, bufs[2], counts[2])
        sv.enqueue_fill(Float32(0.0))
        var x = _view(ex, bufs[0], n)
        if code == OPS_M1_STEP:
            var e = _m1_entry(ex, sc, bufs, counts, b, dm)
            var dims = MambaDims.of(dm)
            var aw = MambaArena(over=_view(ex, bufs[1], counts[1]), cap=counts[1], guard=MAMBA_GUARD)
            var w = MambaDeviceWeights(ctx, dims, aw)
            sc.m1[e].train = None
            sc.m1[e].train_l = 0
            var S = Mamba1TapeStack(ctx, w^, x, b, l)
            ref blk = sc.m1[e]
            mamba1_step_forward(S, blk.st, y_out)
            blk.train = S^
            blk.train_l = l
        elif code == OPS_M2_STEP:
            _expect(code, "state words", counts[3], 1)
            var zero: List[Int] = [0]
            _write_words(ex, bufs[3], zero)
            var e = _m2_entry(ex, sc, bufs, counts, b, dm)
            var dims = Mamba2Dims.of(dm)
            var aw = MambaArena(over=_view(ex, bufs[1], counts[1]), cap=counts[1], guard=MAMBA_GUARD)
            var w = Mamba2DeviceWeights(ctx, dims, aw)
            sc.m2[e].train = None
            sc.m2[e].train_l = 0
            var S = Mamba2TapeStack(ctx, w^, x, b, l, fs[0], fs[1])
            ref blk = sc.m2[e]
            blk.st.buf_len = 0
            mamba2_step_forward(S, blk.st, y_out)
            _d2d(ctx, blk.h_last, S.stages[0].h_last, m2_report_floats(b, dm))
            blk.h_in_held = False
            var words: List[Int] = [blk.st.buf_len]
            _write_words(ex, bufs[3], words)
            blk.train = S^
            blk.train_l = l
        elif code == OPS_M3_STEP:
            _expect(code, "state words", counts[3], 2)
            var zero: List[Int] = [0, 0]
            _write_words(ex, bufs[3], zero)
            var e = _m3_entry(ex, sc, bufs, counts, b, dm)
            var dims = Mamba3Dims.of(dm)
            sc.m3[e].train_st = None
            sc.m3[e].train_x = None
            sc.m3[e].train_l = 0
            ref blk = sc.m3[e]
            blk.st.buf_len = 0
            blk.st.pending = False
            var stages = Mamba3DeviceStages(ctx, b, l, 0, dims)
            var dx = mamba_zeros(ctx, n)
            _d2d(ctx, dx, x, n)
            var trace = IdentityTrace.disabled()
            mamba3_block_forward(ctx, stages, blk.st, blk.w, dx, b, l, trace, String("ops.step.train"))
            var y = _view(ex, y_out, n)
            _d2d(ctx, y, stages.residual_out, n)
            var report_n = m3_report_floats(b, dm)
            _d2d(ctx, blk.reports[0], stages.h_last, report_n[0])
            _d2d(ctx, blk.reports[1], stages.k_last, report_n[1])
            _d2d(ctx, blk.reports[2], stages.v_last, report_n[2])
            _d2d(ctx, blk.reports[3], stages.theta_last, report_n[3])
            var words: List[Int] = [blk.st.buf_len, 1 if blk.st.pending else 0]
            _write_words(ex, bufs[3], words)
            blk.train_st = stages^
            blk.train_x = dx^
            blk.train_l = l
        else:
            raise Error("ops: " + String(code) + " is not a Mamba step op")
    except err:
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's three or four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err
    return l


def mamba_step_backward(
    mut ex: DeviceExec, mut sc: BlockStepScratch, code: Int, w: FP, s: FP, d_out: FP, d_x: FP, d_w: FP,
) raises:
    """The backward of the last recorded run of Mamba step op `code` over
    (w, s): the stage rules over the entry's kept stages, d(block output)
    `d_out`; writes d_x (b * L * d_model floats) and every weight gradient
    into `d_w`, the packed weights' gradient (block_offsets layout; the gaps
    are the caller's zeros)."""
    var ctx = ex.ctx.copy()
    var gone = String("ops: ") + String(step_op_info(code).name) + (
        " backward: the recorded forward's stages are gone"
        + " (its weights or state were written outside the step after the run)")
    if code == OPS_M1_STEP:
        for i in range(len(sc.m1)):  # driver-loop: over resident Mamba-1 blocks
            if sc.m1[i].w_addr == Int(w) and sc.m1[i].s_addr == Int(s) and sc.m1[i].train_l > 0 and sc.m1[i].train:
                var ints: List[Int] = [sc.m1[i].b, sc.m1[i].dm]
                var at = block_offsets(block_weight_sizes(code, ints), block_guard(code))
                ref blk = sc.m1[i]
                mamba1_step_backward(blk.train.value(), blk.hand, d_out, d_x, d_w, at)
                return
    elif code == OPS_M2_STEP:
        for i in range(len(sc.m2)):  # driver-loop: over resident Mamba-2 blocks
            if sc.m2[i].w_addr == Int(w) and sc.m2[i].s_addr == Int(s) and sc.m2[i].train_l > 0 and sc.m2[i].train:
                var ints: List[Int] = [sc.m2[i].b, sc.m2[i].dm]
                var at = block_offsets(block_weight_sizes(code, ints), block_guard(code))
                ref blk = sc.m2[i]
                mamba2_step_backward(blk.train.value(), d_out, d_x, d_w, at)
                return
    elif code == OPS_M3_STEP:
        for i in range(len(sc.m3)):  # driver-loop: over resident Mamba-3 blocks
            if (sc.m3[i].w_addr == Int(w) and sc.m3[i].s_addr == Int(s) and sc.m3[i].train_l > 0
                    and sc.m3[i].train_st and sc.m3[i].train_x):
                var ints: List[Int] = [sc.m3[i].b, sc.m3[i].dm]
                var at = block_offsets(block_weight_sizes(code, ints), block_guard(code))
                var b = sc.m3[i].b
                var l = sc.m3[i].train_l
                ref blk = sc.m3[i]
                mamba3_step_backward(ctx, blk.w, blk.train_st.value(), blk.train_x.value(), d_out, d_x, d_w, at,
                                     b, l)
                return
    else:
        raise Error("ops: " + String(code) + " is not a Mamba step op")
    raise Error(gone)


def block_step_check(code: Int, n_tensors: Int, ints: List[Int], fs: List[Float32]) raises -> StepOpInfo:
    """The operand counts of step op `code`, refused by name."""
    var info = step_op_info(code)
    if n_tensors != info.n_tensors:
        raise Error("ValueError: ops: " + String(info.name) + " takes " + String(info.n_tensors) + " tensors, got "
                    + String(n_tensors))
    if len(fs) != info.n_floats:
        raise Error("ValueError: ops: " + String(info.name) + " takes " + String(info.n_floats)
                    + " float arguments, got " + String(len(fs)))
    _check_ints(code, ints)
    return info


def block_step_run(
    mut ex: DeviceExec, mut sc: BlockStepScratch, code: Int, bufs: List[FP], counts: List[Int], ints: List[Int],
    fs: List[Float32], y_out: FP,
) raises:
    """Run step op `code` (operands checked by the session): `bufs` / `counts`
    are the tensor operands' storage and element counts in the table's
    order, `y_out` the result's b * L * d_model floats. A failed run drops the
    pair's entry (its state is what the failed call left), so the next run
    rebuilds it from the session tensors."""
    try:
        comptime for i in range(N_STEP_OPS):  # driver-loop: over the step-op table (four rows, compile time)
            comptime CODE = OPS_STEP_FIRST + i
            if code == CODE:
                _step_run[CODE](ex, sc, bufs, counts, ints, fs, y_out)
                return
    except err:
        # lane fx-kv-inplace: a full-causal attention entry is about to go;
        # keep how many positions its cache held (`BlockStepScratch.lost_s`).
        var lost_used = -1
        if code == OPS_ATTN_STEP and len(bufs) == 4:
            var le = _attn_by_state(sc, Int(bufs[2]), Int(bufs[3]))
            if le >= 0 and sc.attn[le].ints[7] == 0:
                lost_used = sc.attn[le].kv.s
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's three or four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        if lost_used >= 0:
            sc.note_lost(Int(bufs[2]), Int(bufs[3]), lost_used, ints)
        raise err
    raise Error("ops: " + String(code) + " is not a block step op")


def mamba_step_run_host(
    mut ex: DeviceExec, mut sc: BlockStepScratch, code: Int, bufs: List[FP], counts: List[Int], ints: List[Int],
    fs: List[Float32], y_host: Int,
) raises:
    """A Mamba step op's run for a per-call door (
    lane restore-neural-call-paths): the same body as `block_step_run`, with
    the block output enqueued straight from the stages into the caller's
    host array at `y_host` (no y tensor, no device-to-device copy); the
    caller waits. `bufs` / `counts` are [x, weights, state, (words)] as the
    session's `_block_step` passes them, checked here the same way
    (`block_step_check`). Same eviction of the pair's entry on a failed
    run."""
    _ = block_step_check(code, len(bufs), ints, fs)
    if y_host == 0:
        raise Error("ops: mamba step: null output address")
    try:
        if code == OPS_M1_STEP:
            _run_m1(ex, sc, bufs, counts, ints, bufs[0], y_host)
        elif code == OPS_M2_STEP:
            _run_m2(ex, sc, bufs, counts, ints, fs, bufs[0], y_host)
        elif code == OPS_M3_STEP:
            _run_m3(ex, sc, bufs, counts, ints, bufs[0], y_host)
        else:
            raise Error("ops: " + String(code) + " is not a Mamba step op")
    except err:
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's three or four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err


# ---------------------------------------------------------------- resident readers
# Lane x5-mamba-on-executor (2026-11-09): what a family's program reads back
# from the resident entry of a (weights, state) pair after a run, with no
# device round trip for the host integers.
def block_buffer_view(ex: DeviceExec, p: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
    """The executor's storage [p, p + n) as a device buffer (no copy)."""
    return _view(ex, p, n)


def block_step_words(sc: BlockStepScratch, code: Int, w_addr: Int, s_addr: Int) raises -> List[Int]:
    """The carried host integers of the resident entry over (weights, state):
    Mamba-2 [buf_len], Mamba-3 [buf_len, pending], Mamba-1 none."""
    if code == OPS_M2_STEP:
        for i in range(len(sc.m2)):  # driver-loop: over resident Mamba-2 blocks
            if sc.m2[i].w_addr == w_addr and sc.m2[i].s_addr == s_addr:
                var out: List[Int] = [sc.m2[i].st.buf_len]
                return out^
    elif code == OPS_M3_STEP:
        for i in range(len(sc.m3)):  # driver-loop: over resident Mamba-3 blocks
            if sc.m3[i].w_addr == w_addr and sc.m3[i].s_addr == s_addr:
                var out: List[Int] = [sc.m3[i].st.buf_len, 1 if sc.m3[i].st.pending else 0]
                return out^
    elif code == OPS_M1_STEP:
        return List[Int]()
    raise Error("ops: " + String(code) + ": no resident block over these weights and state (run the step first)")


def block_step_reports_to_host(ctx: DeviceContext, sc: BlockStepScratch, code: Int, w_addr: Int, s_addr: Int,
                               dsts: List[Int]) raises:
    """Enqueue the last run's reports of the resident entry over (weights,
    state) into the caller's host arrays (a 0 address skips that report):
    Mamba-2 [h_last], Mamba-3 [h_last, k_last, v_last, theta_last]. No wait;
    the caller synchronizes before it reads them."""
    if code == OPS_M2_STEP:
        for i in range(len(sc.m2)):  # driver-loop: over resident Mamba-2 blocks
            if sc.m2[i].w_addr == w_addr and sc.m2[i].s_addr == s_addr:
                if len(dsts) != 1:
                    raise Error("ops: mamba2_step has one report (h_last)")
                if dsts[0] != 0:
                    var n = m2_report_floats(sc.m2[i].b, sc.m2[i].dm)
                    var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dsts[0])
                    # lane fix-mamba12-forward-exec: a held run's report is
                    # the held stages' h_last (S17 after the final padded
                    # chunk), never copied into the entry
                    if sc.m2[i].h_in_held and sc.m2[i].held:
                        ctx.enqueue_copy(dst_ptr=dst,
                                         src_buf=sc.m2[i].held.value().h_last.create_sub_buffer[DType.float32](0, n))
                    else:
                        ctx.enqueue_copy(dst_ptr=dst, src_buf=sc.m2[i].h_last.create_sub_buffer[DType.float32](0, n))
                return
    elif code == OPS_M3_STEP:
        for i in range(len(sc.m3)):  # driver-loop: over resident Mamba-3 blocks
            if sc.m3[i].w_addr == w_addr and sc.m3[i].s_addr == s_addr:
                if len(dsts) != 4:
                    raise Error("ops: mamba3_step has four reports (h_last, k_last, v_last, theta_last)")
                var sizes = m3_report_floats(sc.m3[i].b, sc.m3[i].dm)
                for k in range(4):  # small-loop(the four Mamba-3 report buffers)
                    if dsts[k] != 0:
                        ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dsts[k]),
                                         src_buf=sc.m3[i].reports[k].create_sub_buffer[DType.float32](0, sizes[k]))
                return
    else:
        raise Error("ops: " + String(code) + " keeps no reports")
    raise Error("ops: " + String(code) + ": no resident block over these weights and state (run the step first)")
