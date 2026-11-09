# SPDX-License-Identifier: Apache-2.0
"""BLOCK STEP OPS of the operation session (lane x1-executor-state-and-step-ops,
2026-11-08; docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md 2.1, the API note
docs/plans/EXECUTOR_X1_API.md).

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
one algorithm; the host entry (`BlockStepScratch.hattn` / `.hmamba`) keeps
only the rotary table, a training run's stages and the reports. Every
cross-row fold inside the bodies is a gemm v1 call and every per-row stage
runs over host tasks, so a run uses every core; consecutive layers are
dependent and run in order.

Recording: the ops run in an inference session; a family lane that trains through a step op
records it with `Tape.custom_nd` over the stage buffers its rule reads
(`rules_mamba`, `rules_mamba3`, `rules_transformer`), see the API note.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from core.identity_trace import IdentityTrace
from mamba.checks.mamba_fixture import MambaDims
from mamba.checks.mamba2_fixture import Mamba2Dims, M2_CHUNK_SIZE, M2_HEADDIM, M2_D_STATE
from mamba.checks.mamba3_fixture import Mamba3Dims, M3_CHUNK_SIZE, M3_HEADDIM, M3_D_STATE, M3_NUM_ROPE_ANGLES
from mamba.impl.modules.afn_arena import MambaArena
from mamba.impl.modeling.modeling_mamba import (
    MambaDeviceWeights, MambaDeviceState, MambaDeviceStages, mamba_block_forward, mamba_zeros, MAMBA_GUARD,
    _m1_weight_sizes, _m1_state_sizes,
)
from mamba.impl.modules.mamba_simple import mamba_step
from mamba.impl.modules.mamba2 import (
    Mamba2DeviceWeights, Mamba2DeviceState, Mamba2DeviceStages, mamba2_block_forward,
    _m2_weight_sizes, _m2_state_sizes,
)
from mamba.impl.modules.mamba3 import (
    Mamba3DeviceWeights, Mamba3DeviceState, Mamba3DeviceStages, mamba3_block_forward,
    _m3_weight_sizes, _m3_state_sizes,
)
from transformer.impl.llama.modeling_llama import (
    LlamaDims, LlamaDeviceWeights, LlamaKVCache, LlamaRopeTable, LlamaDeviceStages, llama_decoder_layer_forward,
)
from transformer.block_options import BlockOptions, BLOCK_OPTION_PARAMS
from transformer.lean_stages import _lean_for
from autograd.kernels.transformer import LlamaBackwardStages
from autograd.rules_transformer import llama_decoder_layer_backward_autograd
# The host arm (lane x7b-step-ops-host-arm): the host block bodies, the same
# oracles the CPU identity column has always run (each equal to its device
# block's bits, gemm v1 folds inside, rows over host tasks).
from std.memory import memcpy
from core.host_lanes import host_f32_uninit
from transformer.checks.transformer_fixture import TransformerDims, TransformerWeights, ScorePlant
from transformer.checks.transformer_oracle import (
    RopeTable, TransformerKVCache, TransformerStages, build_rope_table_opts, transformer_block_oracle,
)
from transformer.checks.transformer_backward_oracle import transformer_block_backward_oracle
from mamba.checks.mamba_fixture import MambaWeights
from mamba.checks.mamba_oracle import MambaState, mamba_block_oracle
from mamba.checks.mamba2_fixture import Mamba2Weights
from mamba.checks.mamba2_oracle import Mamba2State, mamba2_block_oracle
from mamba.checks.mamba3_fixture import Mamba3Weights
from mamba.checks.mamba3_oracle import Mamba3State, mamba3_block_oracle

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

# NI47 same-graph boundary scheduling (moved here with the Mamba-3 decode
# session body, lane x5-mamba-on-executor 2026-11-09; the define and its arm
# are unchanged): retained capacity for one physical open chunk plus a new
# token, so an L = 1 step reuses the previous step's stages instead of
# building them at the carried buf_len. The extent comes from the model state
# contract, never a board row. Default OFF; its A/B is owed (EXPERIMENTS.md).
comptime IDN_M3_DECODE_WINDOW_REUSE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
    and MAMBA_GUARD == 0
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
    #: IDN_M3_DECODE_WINDOW_REUSE: the previous L = 1 step's stages
    var decode_stages: Optional[Mamba3DeviceStages]

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
        self.decode_stages = None
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


struct BlockStepScratch(Movable):
    """The resident block structs of one session's step ops, one entry per
    (weights, state) pair, built at the pair's first run."""
    var m1: List[_M1Block]
    var m2: List[_M2Block]
    var m3: List[_M3Block]
    var attn: List[_AttnBlock]
    #: lane x7b: the host arm's entries (a host session's runs)
    var hattn: List[_HostAttnBlock]
    var hmamba: List[_HostMambaBlock]

    def __init__(out self):
        self.m1 = List[_M1Block]()
        self.m2 = List[_M2Block]()
        self.m3 = List[_M3Block]()
        self.attn = List[_AttnBlock]()
        self.hattn = List[_HostAttnBlock]()
        self.hmamba = List[_HostMambaBlock]()

    def count(self) -> Int:
        return len(self.m1) + len(self.m2) + len(self.m3) + len(self.attn) + len(self.hattn) + len(self.hmamba)

    def forget(mut self, addr: Int):
        """Drop every entry built over the session buffer at `addr` (its
        weights or its state were written from outside a step, or freed)."""
        if self.count() == 0:
            return
        var i = len(self.m1) - 1
        while i >= 0:  # driver-loop: over resident Mamba-1 blocks
            if self.m1[i].w_addr == addr or self.m1[i].s_addr == addr:
                _ = self.m1.pop(i)
            i -= 1
        i = len(self.m2) - 1
        while i >= 0:  # driver-loop: over resident Mamba-2 blocks
            if self.m2[i].w_addr == addr or self.m2[i].s_addr == addr or self.m2[i].q_addr == addr:
                _ = self.m2.pop(i)
            i -= 1
        i = len(self.m3) - 1
        while i >= 0:  # driver-loop: over resident Mamba-3 blocks
            if self.m3[i].w_addr == addr or self.m3[i].s_addr == addr or self.m3[i].q_addr == addr:
                _ = self.m3.pop(i)
            i -= 1
        i = len(self.attn) - 1
        while i >= 0:  # driver-loop: over resident attention blocks
            if self.attn[i].w_addr == addr or self.attn[i].s_addr == addr or self.attn[i].q_addr == addr:
                _ = self.attn.pop(i)
            i -= 1
        i = len(self.hattn) - 1
        while i >= 0:  # driver-loop: over host attention entries
            if self.hattn[i].w_addr == addr or self.hattn[i].s_addr == addr or self.hattn[i].q_addr == addr:
                _ = self.hattn.pop(i)
            i -= 1
        i = len(self.hmamba) - 1
        while i >= 0:  # driver-loop: over host Mamba entries
            if self.hmamba[i].w_addr == addr or self.hmamba[i].s_addr == addr or self.hmamba[i].q_addr == addr:
                _ = self.hmamba.pop(i)
            i -= 1

    def clear(mut self):
        self.m1.clear()
        self.m2.clear()
        self.m3.clear()
        self.attn.clear()
        self.hattn.clear()
        self.hmamba.clear()

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

    def weights_changed(mut self, addr: Int):
        """The weights at `addr` were rewritten in place: the block's own
        refusal of non-finite weights runs again at the next step."""
        for i in range(len(self.m1)):  # driver-loop: over resident Mamba-1 blocks
            if self.m1[i].w_addr == addr:
                self.m1[i].w.weights_checked = False
        for i in range(len(self.m2)):  # driver-loop: over resident Mamba-2 blocks
            if self.m2[i].w_addr == addr:
                self.m2[i].w.weights_checked = False
        for i in range(len(self.m3)):  # driver-loop: over resident Mamba-3 blocks
            if self.m3[i].w_addr == addr:
                self.m3[i].w.weights_checked = False
        # integ-w3: an adopted entry (lane x4b) never reads the weights
        # tensor (a placeholder), so a write there does not drop it.
        for i in range(len(self.attn) - 1, -1, -1):  # driver-loop: over resident attention blocks
            if self.attn[i].w_addr == addr and not self.attn[i].adopted:
                _ = self.attn.pop(i)
        # lane x7b: a host entry reads the weights tensor at every run; only
        # a training run's kept stages (computed over the old weights) go.
        for i in range(len(self.hattn) - 1, -1, -1):  # driver-loop: over host attention entries
            if self.hattn[i].w_addr == addr:
                _ = self.hattn.pop(i)


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


def _run_m1(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
            y_out: FP) raises:
    var ctx = ex.ctx.copy()
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_M1_STEP, counts[0], b, dm)
    _expect(OPS_M1_STEP, "weights", counts[1], block_part_floats(OPS_M1_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_M1_STEP, "state", counts[2], block_part_floats(OPS_M1_STEP, ints, BLOCK_PART_STATE))
    var e = -1
    for i in range(len(sc.m1)):  # driver-loop: over resident Mamba-1 blocks
        if sc.m1[i].w_addr == Int(bufs[1]) and sc.m1[i].s_addr == Int(bufs[2]) and sc.m1[i].b == b and sc.m1[i].dm == dm:
            e = i
    if e < 0:
        sc.m1.append(_M1Block(ctx, ex, bufs[1], counts[1], bufs[2], counts[2], b, dm))
        e = len(sc.m1) - 1
    ref blk = sc.m1[e]
    var n = b * l * dm
    var x = _view(ex, bufs[0], n)
    var y = _view(ex, y_out, n)
    var trace = IdentityTrace.disabled()
    if l == 1:
        # `_m1_session_step_run`: x into the resident buffer, mamba_step on
        # the resident stages; the output stays on the device.
        _d2d(ctx, blk.dx, x, n)
        mamba_step(ctx, blk.stages, blk.st, blk.w, blk.dx, b, trace, String("ops.step"))
        _d2d(ctx, y, blk.stages.residual_out, n)
    else:
        var dims = MambaDims.of(dm)
        var stages = MambaDeviceStages(ctx, b, l, dims)
        var dx = mamba_zeros(ctx, n)
        _d2d(ctx, dx, x, n)
        mamba_block_forward(ctx, stages, blk.st, blk.w, dx, b, l, trace, String("ops.step"))
        _d2d(ctx, y, stages.residual_out, n)
        ctx.synchronize()  # the call-shaped stages and x go at return
        _ = stages^
        _ = dx^


def _run_m2(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
            fs: List[Float32], y_out: FP) raises:
    var ctx = ex.ctx.copy()
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_M2_STEP, counts[0], b, dm)
    _expect(OPS_M2_STEP, "weights", counts[1], block_part_floats(OPS_M2_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_M2_STEP, "state", counts[2], block_part_floats(OPS_M2_STEP, ints, BLOCK_PART_STATE))
    _expect(OPS_M2_STEP, "state words", counts[3], 1)
    var e = -1
    for i in range(len(sc.m2)):  # driver-loop: over resident Mamba-2 blocks
        if (sc.m2[i].w_addr == Int(bufs[1]) and sc.m2[i].s_addr == Int(bufs[2]) and sc.m2[i].q_addr == Int(bufs[3])
                and sc.m2[i].b == b and sc.m2[i].dm == dm):
            e = i
    if e < 0:
        sc.m2.append(_M2Block(ctx, ex, bufs[1], counts[1], bufs[2], counts[2], bufs[3], b, dm))
        e = len(sc.m2) - 1
    ref blk = sc.m2[e]
    var dims = Mamba2Dims.of(dm)
    var n = b * l * dm
    var x = _view(ex, bufs[0], n)
    var y = _view(ex, y_out, n)
    var trace = IdentityTrace.disabled()
    # `_m2_session_step_run`: call-shaped stages at the open chunk's length.
    var stages = Mamba2DeviceStages(ctx, b, l, blk.st.buf_len, dims)
    var h_n = m2_report_floats(b, dm)
    if l == 1:
        _d2d(ctx, blk.dx, x, n)
        mamba2_block_forward(ctx, stages, blk.st, blk.w, blk.dx, b, 1, fs[0], fs[1], trace,
                             String("ops.step"))
        _d2d(ctx, y, stages.residual_out, n)
        _d2d(ctx, blk.h_last, stages.h_last, h_n)
        ctx.synchronize()  # the call-shaped stages go at return
    else:
        var dx = mamba_zeros(ctx, n)
        _d2d(ctx, dx, x, n)
        mamba2_block_forward(ctx, stages, blk.st, blk.w, dx, b, l, fs[0], fs[1], trace,
                             String("ops.step"))
        _d2d(ctx, y, stages.residual_out, n)
        _d2d(ctx, blk.h_last, stages.h_last, h_n)
        ctx.synchronize()
        _ = dx^
    _ = stages^
    var words: List[Int] = [blk.st.buf_len]
    _write_words(ex, bufs[3], words)


def _run_m3(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
            y_out: FP) raises:
    var ctx = ex.ctx.copy()
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_M3_STEP, counts[0], b, dm)
    _expect(OPS_M3_STEP, "weights", counts[1], block_part_floats(OPS_M3_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_M3_STEP, "state", counts[2], block_part_floats(OPS_M3_STEP, ints, BLOCK_PART_STATE))
    _expect(OPS_M3_STEP, "state words", counts[3], 2)
    var e = -1
    for i in range(len(sc.m3)):  # driver-loop: over resident Mamba-3 blocks
        if (sc.m3[i].w_addr == Int(bufs[1]) and sc.m3[i].s_addr == Int(bufs[2]) and sc.m3[i].q_addr == Int(bufs[3])
                and sc.m3[i].b == b and sc.m3[i].dm == dm):
            e = i
    if e < 0:
        sc.m3.append(_M3Block(ctx, ex, bufs[1], counts[1], bufs[2], counts[2], bufs[3], b, dm))
        e = len(sc.m3) - 1
    ref blk = sc.m3[e]
    var dims = Mamba3Dims.of(dm)
    var n = b * l * dm
    var x = _view(ex, bufs[0], n)
    var y = _view(ex, y_out, n)
    var trace = IdentityTrace.disabled()
    # The Mamba-3 decode session's body (the one body since lane
    # x5-mamba-on-executor): stages built per call at the open chunk's
    # length, or, under IDN_M3_DECODE_WINDOW_REUSE, the previous L = 1
    # step's stages re-windowed at the carried buf_len.
    var stages: Mamba3DeviceStages
    comptime if IDN_M3_DECODE_WINDOW_REUSE:
        if l == 1 and blk.decode_stages:
            stages = blk.decode_stages.take()
        elif l == 1:
            stages = Mamba3DeviceStages(ctx, b, 1, M3_CHUNK_SIZE, dims)
        else:
            stages = Mamba3DeviceStages(ctx, b, l, blk.st.buf_len, dims)
        if l == 1:
            stages.reuse_decode_window(blk.st.buf_len)
    else:
        stages = Mamba3DeviceStages(ctx, b, l, blk.st.buf_len, dims)
    var report_n = m3_report_floats(b, dm)
    if l == 1:
        _d2d(ctx, blk.dx, x, n)
        mamba3_block_forward(ctx, stages, blk.st, blk.w, blk.dx, b, 1, trace, String("ops.step"))
        _d2d(ctx, y, stages.residual_out, n)
    else:
        var dx = mamba_zeros(ctx, n)
        _d2d(ctx, dx, x, n)
        mamba3_block_forward(ctx, stages, blk.st, blk.w, dx, b, l, trace, String("ops.step"))
        _d2d(ctx, y, stages.residual_out, n)
        ctx.synchronize()
        _ = dx^
    _d2d(ctx, blk.reports[0], stages.h_last, report_n[0])
    _d2d(ctx, blk.reports[1], stages.k_last, report_n[1])
    _d2d(ctx, blk.reports[2], stages.v_last, report_n[2])
    _d2d(ctx, blk.reports[3], stages.theta_last, report_n[3])
    ctx.synchronize()
    comptime if IDN_M3_DECODE_WINDOW_REUSE:
        if l == 1:
            # Every report copy above completed before capacity is retained.
            blk.decode_stages = stages^
        else:
            _ = stages^
    else:
        _ = stages^
    var words: List[Int] = [blk.st.buf_len, 1 if blk.st.pending else 0]
    _write_words(ex, bufs[3], words)


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

    def __init__(out self):
        self.stages = Optional[LlamaDeviceStages]()
        self.dx = Optional[DeviceBuffer[DType.float32]]()
        self.lean = False
        self.b = 0
        self.l = 0
        self.pos0 = -1

    def clear(mut self):
        self.stages = None
        self.dx = None
        self.l = 0
        self.pos0 = -1

    def prepare(mut self, ctx: DeviceContext, b: Int, l: Int, smax: Int, dims: LlamaDims, window: Int,
                lean: Bool) raises:
        var reused = False
        if self.stages:
            ref st = self.stages.value()
            reused = (st.b == b and st.l == l and st.s_max == smax and st.window == window
                      and st.dims.d_model == dims.d_model and st.dims.n_heads == dims.n_heads
                      and st.dims.n_kv == dims.n_kv and st.dims.head_dim == dims.head_dim
                      and st.dims.intermediate == dims.intermediate and self.lean == lean)
        if reused:
            # Every constructor zero and route flag back before the next run.
            self.stages.value().reset(ctx)
            self.stages.value().int15_on = False
        else:
            # Drop an incompatible owner before allocating its replacement.
            self.stages = None
            self.stages = LlamaDeviceStages(ctx, b, l, smax, dims, window, lean=lean)
            self.lean = lean
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
        sc.attn.append(_AttnBlock(ctx, ex, bufs[1], counts[1], bufs[2], counts[2], bufs[3], ints))
        e = len(sc.attn) - 1
    return e


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
    var words: List[Int] = [s0]
    _write_words(ex, bufs[3], words)


def _run_attn_on(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
                 y_out: FP, mut pre: AttnPrefill, forward_only: Bool) raises:
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
        pre.prepare(ctx, b, l, ints[6], dims, ints[7], blk.lean)
        _d2d(ctx, pre.dx.value(), x, n)
        llama_decoder_layer_forward(
            ctx, pre.stages.value(), blk.kv, blk.rope, blk.w, pre.dx.value(), b, l, pos0, trace,
            String("ops.step.forward"), forward_only=forward_only)
        _d2d(ctx, y, pre.stages.value().residual2, n)
        pre.pos0 = pos0
    var words: List[Int] = [blk.kv.s]
    _write_words(ex, bufs[3], words)


def _run_attn(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
              y_out: FP) raises:
    var pre = AttnPrefill()
    _run_attn_on(ex, sc, bufs, counts, ints, y_out, pre, False)
    if pre.stages:
        # The run's own L-shaped stages go at return (as before).
        ex.ctx.synchronize()
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
    mut pre: AttnPrefill, forward_only: Bool,
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
        _run_attn_on(ex, sc, bufs, counts, ints, y_out, pre, forward_only)
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
        _run_attn_on(ex, sc, bufs, counts, ints, y_out, pre, False)
        e = attn_step_entry(ex, sc, bufs, counts, ints)
        swap(pre, sc.attn[e].kept)
        sc.attn[e].train_l = l
        _ = pre^
    except err:
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
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


# ---------------------------------------------------------------- the host arm
# Lane x7b-step-ops-host-arm (2026-11-09). A host session (`OpsSession[HostExec]`)
# runs the four step ops through the host block bodies over the same packed
# tensors, so one captured program has one algorithm on every column (module
# note). Session storage is host memory here: a tensor's FP is its floats.
# Nothing is resident but the rotary table, a training run's stages and the
# Mamba reports: the weights are read at every run (a write from outside a
# step needs no re-check), the state is read in and written back after a
# successful run (a failed run leaves it as it was), the words are the words
# tensor's.


def _host_take(p: FP, at: Int, n: Int) raises -> List[Float32]:
    """The n floats at p + at, as an owned list (the oracles take lists)."""
    var out = host_f32_uninit(n)
    if n > 0:
        memcpy(dest=FP(unsafe_from_address=Int(out.unsafe_ptr())), src=p + at, count=n)
    return out^


def _host_put(p: FP, at: Int, values: List[Float32], n: Int, what: String) raises:
    """p[at : at + n] = values (refused unless it holds n floats)."""
    if len(values) != n:
        raise Error("ops: host step op: " + what + " holds " + String(len(values)) + " values, the layout "
                    + String(n))
    if n > 0:
        memcpy(dest=p + at, src=FP(unsafe_from_address=Int(values.unsafe_ptr())), count=n)


def _host_words(p: FP, k: Int) -> List[Int]:
    """The k int32 words of a host words tensor."""
    var out = List[Int](capacity=k)
    for i in range(k):  # small-loop(k: at most two state words): word decode, no data
        out.append(Int(bitcast[DType.int32](p[i])))
    return out^


def _host_set_words(p: FP, words: List[Int]):
    for i in range(len(words)):  # small-loop(words: at most two state words): word encode, no data
        p[i] = bitcast[DType.float32](Int32(words[i]))


def _host_pieces(p: FP, code: Int, ints: List[Int], part: Int) raises -> List[List[Float32]]:
    """Every piece of `part` (weights or float state) of the packed tensor
    at p, in `block_*_sizes` order (an absent optional piece empty)."""
    var sizes = block_weight_sizes(code, ints) if part == BLOCK_PART_WEIGHTS else block_state_sizes(code, ints)
    var at = block_offsets(sizes, block_guard(code))
    var out = List[List[Float32]](capacity=len(sizes))
    for i in range(len(sizes)):  # driver-loop: over the block's weight or state pieces
        out.append(_host_take(p, at[i], sizes[i]))
    return out^


def _host_store_state(p: FP, code: Int, ints: List[Int], pieces: List[List[Float32]]) raises:
    """The carried float state back into the state tensor at p, piece by
    piece at its slot (the gaps keep their zeros)."""
    var sizes = block_state_sizes(code, ints)
    var at = block_offsets(sizes, block_guard(code))
    if len(pieces) != len(sizes):
        raise Error("ops: host step op: the state has " + String(len(sizes)) + " pieces")
    for i in range(len(sizes)):  # driver-loop: over the block's state pieces
        _host_put(p, at[i], pieces[i], sizes[i], "state piece " + String(i))


struct _HostAttnBlock(Movable):
    """A host attention entry: the rotary table of its integer arguments and
    the last RECORDED run's stages and weights (for the backward)."""
    var w_addr: Int
    var s_addr: Int
    var q_addr: Int
    var ints: List[Int]
    var dims: TransformerDims
    var rope: RopeTable
    #: the last recorded run's token count (0: none) and its kept stages
    var train_l: Int
    var kept: Optional[TransformerStages]
    var kept_w: Optional[TransformerWeights]

    def __init__(out self, w: FP, s: FP, q: FP, ints: List[Int]) raises:
        var d = _attn_dims(ints)
        var opts = _attn_opts(ints)
        # RoPE positions = s_max, as the device table `LlamaRopeTable(ctx,
        # dims, opts, s_max)` of the device entry.
        self.dims = TransformerDims(d.d_model, d.n_heads, d.n_kv, d.head_dim, d.intermediate, ints[6])
        self.rope = build_rope_table_opts(self.dims, opts)
        self.w_addr = Int(w)
        self.s_addr = Int(s)
        self.q_addr = Int(q)
        self.ints = ints.copy()
        self.train_l = 0
        self.kept = Optional[TransformerStages]()
        self.kept_w = Optional[TransformerWeights]()


def _host_attn_weights(w: FP, ints: List[Int], dims: TransformerDims) raises -> TransformerWeights:
    """The block's weights from the packed tensor (`block_weight_sizes`
    order: the options constructor's twenty, `LlamaDeviceWeights`)."""
    var p = _host_pieces(w, OPS_ATTN_STEP, ints, BLOCK_PART_WEIGHTS)
    var tw = TransformerWeights(dims)
    tw.opts = _attn_opts(ints)
    tw.norm1_w = p[0].copy()
    tw.norm2_w = p[1].copy()
    tw.w_q = p[2].copy()
    tw.w_k = p[3].copy()
    tw.w_v = p[4].copy()
    tw.w_o = p[5].copy()
    tw.w_gate = p[6].copy()
    tw.w_up = p[7].copy()
    tw.w_down = p[8].copy()
    tw.b_q = p[9].copy()
    tw.b_k = p[10].copy()
    tw.b_v = p[11].copy()
    tw.b_o = p[12].copy()
    tw.norm1_b = p[13].copy()
    tw.norm2_b = p[14].copy()
    tw.b_up = p[15].copy()
    tw.b_down = p[16].copy()
    tw.b_gate = p[17].copy()
    tw.qn_w = p[18].copy()
    tw.kn_w = p[19].copy()
    return tw^


def _host_attn_entry(mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int]) raises -> Int:
    """The host entry over (weights, kv, words), built at the triple's
    first run; the operand counts are checked at every run."""
    _expect(OPS_ATTN_STEP, "weights", counts[1], block_part_floats(OPS_ATTN_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_ATTN_STEP, "kv", counts[2], block_part_floats(OPS_ATTN_STEP, ints, BLOCK_PART_STATE))
    _expect(OPS_ATTN_STEP, "state words", counts[3], 1)
    for i in range(len(sc.hattn)):  # driver-loop: over host attention entries
        if (sc.hattn[i].w_addr == Int(bufs[1]) and sc.hattn[i].s_addr == Int(bufs[2])
                and sc.hattn[i].q_addr == Int(bufs[3]) and _same_ints(sc.hattn[i].ints, ints)):
            return i
    sc.hattn.append(_HostAttnBlock(bufs[1], bufs[2], bufs[3], ints))
    return len(sc.hattn) - 1


def _host_attn_on(mut blk: _HostAttnBlock, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP,
                  pos0: Int, keep: Bool) raises -> Int:
    """One block call at carried position pos0 (`transformer_block_oracle`:
    prefill, chunked continuation and the L = 1 step are one function), the
    cache read from and written back to the state tensor, the words to the
    new position. With `keep` the stages and weights stay for the backward.
    Returns L."""
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_ATTN_STEP, counts[0], b, dm)
    var smax = ints[6]
    if pos0 < 0 or pos0 > smax:
        raise Error("ValueError: ops: attention_step: cached tokens must be in [0, " + String(smax)
                    + "], got " + String(pos0))
    var n = b * l * dm
    var w = _host_attn_weights(bufs[1], ints, blk.dims)
    var ssz = block_state_sizes(OPS_ATTN_STEP, ints)
    var sat = block_offsets(ssz, block_guard(OPS_ATTN_STEP))
    var cache = TransformerKVCache(b, blk.dims, smax, ints[7])
    if len(cache.k) != ssz[0] or len(cache.v) != ssz[1]:
        raise Error("ops: attention_step: the host cache layout differs from the state tensor's")
    cache.k = _host_take(bufs[2], sat[0], ssz[0])
    cache.v = _host_take(bufs[2], sat[1], ssz[1])
    cache.used = pos0
    var x = _host_take(bufs[0], 0, n)
    var st = transformer_block_oracle(w, x, b, l, cache, blk.rope, ScorePlant.none())
    _host_put(y_out, 0, st.residual2_out, n, "the block output")
    _host_put(bufs[2], sat[0], cache.k, ssz[0], "the key cache")
    _host_put(bufs[2], sat[1], cache.v, ssz[1], "the value cache")
    var words: List[Int] = [cache.used]
    _host_set_words(bufs[3], words)
    if keep:
        blk.kept = st^
        blk.kept_w = w^
        blk.train_l = l
    return l


def _host_run_attn(mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP) raises:
    var e = _host_attn_entry(sc, bufs, counts, ints)
    var pos0 = _host_words(bufs[3], 1)[0]
    _ = _host_attn_on(sc.hattn[e], bufs, counts, ints, y_out, pos0, False)


def host_attn_step_train_run(
    mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP,
) raises -> Int:
    """`attn_step_train_run` on the host: the zero-state prefill at
    position 0 with its stages kept in the entry for the backward. Returns
    L. A failed run drops the triple's entry."""
    _check_ints(OPS_ATTN_STEP, ints)
    if len(bufs) != 4 or len(counts) != 4:
        raise Error("ops: attention_step takes four tensors (x, weights, kv, words)")
    var l = _tokens(OPS_ATTN_STEP, counts[0], ints[0], ints[1])
    if l > ints[6]:
        raise Error("ValueError: ops: attention_step: a training run of " + String(l)
                    + " tokens needs s_max >= " + String(l) + ", got " + String(ints[6]))
    try:
        var e = _host_attn_entry(sc, bufs, counts, ints)
        _ = _host_attn_on(sc.hattn[e], bufs, counts, ints, y_out, 0, True)
    except err:
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err
    return l


def host_attn_step_backward(
    mut sc: BlockStepScratch, w: FP, s: FP, q: FP, d_out: FP, d_x: FP, d_w: FP,
) raises:
    """`attn_step_backward` on the host: `transformer_block_backward_oracle`
    over the entry's kept stages and weights, d(residual2) `d_out`; writes
    d_x and the nine weight gradients at their `block_offsets` slots of
    `d_w` (the optional pieces' gradients are the device rule's: none)."""
    var e = -1
    for i in range(len(sc.hattn)):  # driver-loop: over host attention entries
        if sc.hattn[i].w_addr == Int(w) and sc.hattn[i].s_addr == Int(s) and sc.hattn[i].q_addr == Int(q):
            e = i
    if e < 0 or sc.hattn[e].train_l < 1 or not sc.hattn[e].kept or not sc.hattn[e].kept_w:
        raise Error("ops: attention_step backward: the recorded forward's stages are gone"
                    + " (its weights or state were written outside the step after the run)")
    ref blk = sc.hattn[e]
    var ints = blk.ints.copy()
    var b = ints[0]
    var l = blk.train_l
    var n = b * l * ints[1]
    var dout = _host_take(d_out, 0, n)
    var g = transformer_block_backward_oracle(blk.kept_w.value(), blk.kept.value(), dout, b, l, 0, blk.rope, ints[7])
    _host_put(d_x, 0, g.d_x, n, "d_x")
    var sizes = block_weight_sizes(OPS_ATTN_STEP, ints)
    var at = block_offsets(sizes, block_guard(OPS_ATTN_STEP))
    # block_weight_sizes order: norm1, norm2, q, k, v, o, gate, up, down
    _host_put(d_w, at[0], g.dw_norm1, sizes[0], "dw_norm1")
    _host_put(d_w, at[1], g.dw_norm2, sizes[1], "dw_norm2")
    _host_put(d_w, at[2], g.dw_q, sizes[2], "dw_q")
    _host_put(d_w, at[3], g.dw_k, sizes[3], "dw_k")
    _host_put(d_w, at[4], g.dw_v, sizes[4], "dw_v")
    _host_put(d_w, at[5], g.dw_o, sizes[5], "dw_o")
    _host_put(d_w, at[6], g.dw_gate, sizes[6], "dw_gate")
    _host_put(d_w, at[7], g.dw_up, sizes[7], "dw_up")
    _host_put(d_w, at[8], g.dw_down, sizes[8], "dw_down")


struct _HostMambaBlock(Movable):
    """A host Mamba entry: the last run's reports (Mamba-2 [h_last],
    Mamba-3 [h_last, k_last, v_last, theta_last]; Mamba-1 none)."""
    var code: Int
    var w_addr: Int
    var s_addr: Int
    var q_addr: Int
    var b: Int
    var dm: Int
    var reports: List[List[Float32]]

    def __init__(out self, code: Int, w: Int, s: Int, q: Int, b: Int, dm: Int):
        self.code = code
        self.w_addr = w
        self.s_addr = s
        self.q_addr = q
        self.b = b
        self.dm = dm
        self.reports = List[List[Float32]]()


def _host_mamba_reports(mut sc: BlockStepScratch, code: Int, bufs: List[FP], b: Int, dm: Int,
                        var reports: List[List[Float32]]):
    var q = Int(bufs[3]) if len(bufs) > 3 else 0
    for i in range(len(sc.hmamba)):  # driver-loop: over host Mamba entries
        if (sc.hmamba[i].code == code and sc.hmamba[i].w_addr == Int(bufs[1]) and sc.hmamba[i].s_addr == Int(bufs[2])
                and sc.hmamba[i].q_addr == q):
            sc.hmamba[i].b = b
            sc.hmamba[i].dm = dm
            sc.hmamba[i].reports = reports^
            return
    var blk = _HostMambaBlock(code, Int(bufs[1]), Int(bufs[2]), q, b, dm)
    blk.reports = reports^
    sc.hmamba.append(blk^)


def _host_run_m1(mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP) raises:
    """`mamba_block_oracle` over the packed weights and state (`_m1_weight_sizes`
    / `_m1_state_sizes` order: MambaWeights' and MambaState's fields)."""
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_M1_STEP, counts[0], b, dm)
    _expect(OPS_M1_STEP, "weights", counts[1], block_part_floats(OPS_M1_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_M1_STEP, "state", counts[2], block_part_floats(OPS_M1_STEP, ints, BLOCK_PART_STATE))
    var dims = MambaDims.of(dm)
    var p = _host_pieces(bufs[1], OPS_M1_STEP, ints, BLOCK_PART_WEIGHTS)
    var w = MambaWeights(dims)
    w.norm_w = p[0].copy()
    w.w_in = p[1].copy()
    w.conv_w = p[2].copy()
    w.conv_b = p[3].copy()
    w.w_x = p[4].copy()
    w.w_dt = p[5].copy()
    w.b_dt = p[6].copy()
    w.a_log = p[7].copy()
    w.d_skip = p[8].copy()
    w.w_out = p[9].copy()
    var q = _host_pieces(bufs[2], OPS_M1_STEP, ints, BLOCK_PART_STATE)
    var st = MambaState(b, dims)
    st.conv_win = q[0].copy()
    st.h = q[1].copy()
    var n = b * l * dm
    var x = _host_take(bufs[0], 0, n)
    var stages = mamba_block_oracle(w, x, b, l, st)
    _host_put(y_out, 0, stages.residual_out, n, "the block output")
    var out = List[List[Float32]]()
    out.append(st.conv_win.copy())
    out.append(st.h.copy())
    _host_store_state(bufs[2], OPS_M1_STEP, ints, out)


def _host_run_m2(mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], fs: List[Float32],
                 y_out: FP) raises:
    """`mamba2_block_oracle` over the packed weights and state (Mamba2Weights'
    and Mamba2State's fields), buf_len the state word; the h_last report kept."""
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_M2_STEP, counts[0], b, dm)
    _expect(OPS_M2_STEP, "weights", counts[1], block_part_floats(OPS_M2_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_M2_STEP, "state", counts[2], block_part_floats(OPS_M2_STEP, ints, BLOCK_PART_STATE))
    _expect(OPS_M2_STEP, "state words", counts[3], 1)
    var words = _host_words(bufs[3], 1)
    _m2_check_words(words)
    var dims = Mamba2Dims.of(dm)
    var p = _host_pieces(bufs[1], OPS_M2_STEP, ints, BLOCK_PART_WEIGHTS)
    var w = Mamba2Weights(dims)
    w.norm_w = p[0].copy()
    w.w_in = p[1].copy()
    w.conv_w = p[2].copy()
    w.conv_b = p[3].copy()
    w.dt_bias = p[4].copy()
    w.a_log = p[5].copy()
    w.d_skip = p[6].copy()
    w.gnorm_w = p[7].copy()
    w.w_out = p[8].copy()
    var q = _host_pieces(bufs[2], OPS_M2_STEP, ints, BLOCK_PART_STATE)
    var st = Mamba2State(b, dims)
    st.conv_win = q[0].copy()
    st.h = q[1].copy()
    st.buf_xbc = q[2].copy()
    st.buf_dtraw = q[3].copy()
    st.buf_len = words[0]
    var n = b * l * dm
    var x = _host_take(bufs[0], 0, n)
    var stages = mamba2_block_oracle(w, x, b, l, fs[0], fs[1], st)
    _host_put(y_out, 0, stages.residual_out, n, "the block output")
    var out = List[List[Float32]]()
    out.append(st.conv_win.copy())
    out.append(st.h.copy())
    out.append(st.buf_xbc.copy())
    out.append(st.buf_dtraw.copy())
    _host_store_state(bufs[2], OPS_M2_STEP, ints, out)
    var after: List[Int] = [st.buf_len]
    _host_set_words(bufs[3], after)
    var reports = List[List[Float32]]()
    reports.append(stages.h_last.copy())
    _host_mamba_reports(sc, OPS_M2_STEP, bufs, b, dm, reports^)


def _host_run_m3(mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], y_out: FP) raises:
    """`mamba3_block_oracle` over the packed weights and state (Mamba3Weights'
    and Mamba3State's fields), [buf_len, pending] the state words; the four
    reports kept."""
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_M3_STEP, counts[0], b, dm)
    _expect(OPS_M3_STEP, "weights", counts[1], block_part_floats(OPS_M3_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_M3_STEP, "state", counts[2], block_part_floats(OPS_M3_STEP, ints, BLOCK_PART_STATE))
    _expect(OPS_M3_STEP, "state words", counts[3], 2)
    var words = _host_words(bufs[3], 2)
    _m3_check_words(words)
    var dims = Mamba3Dims.of(dm)
    var p = _host_pieces(bufs[1], OPS_M3_STEP, ints, BLOCK_PART_WEIGHTS)
    var w = Mamba3Weights(dims)
    w.norm_w = p[0].copy()
    w.w_in = p[1].copy()
    w.dt_bias = p[2].copy()
    w.bnorm_w = p[3].copy()
    w.cnorm_w = p[4].copy()
    w.b_bias = p[5].copy()
    w.c_bias = p[6].copy()
    w.d_skip = p[7].copy()
    w.w_out = p[8].copy()
    var q = _host_pieces(bufs[2], OPS_M3_STEP, ints, BLOCK_PART_STATE)
    var st = Mamba3State(b, dims)
    st.theta = q[0].copy()
    st.h = q[1].copy()
    st.buf_qrot = q[2].copy()
    st.buf_krot = q[3].copy()
    st.buf_v = q[4].copy()
    st.buf_dt = q[5].copy()
    st.buf_sig = q[6].copy()
    st.buf_adt = q[7].copy()
    st.pend_k = q[8].copy()
    st.pend_v = q[9].copy()
    st.buf_len = words[0]
    st.pending = words[1] == 1
    var n = b * l * dm
    var x = _host_take(bufs[0], 0, n)
    var stages = mamba3_block_oracle(w, x, b, l, st)
    _host_put(y_out, 0, stages.residual_out, n, "the block output")
    var out = List[List[Float32]]()
    out.append(st.theta.copy())
    out.append(st.h.copy())
    out.append(st.buf_qrot.copy())
    out.append(st.buf_krot.copy())
    out.append(st.buf_v.copy())
    out.append(st.buf_dt.copy())
    out.append(st.buf_sig.copy())
    out.append(st.buf_adt.copy())
    out.append(st.pend_k.copy())
    out.append(st.pend_v.copy())
    _host_store_state(bufs[2], OPS_M3_STEP, ints, out)
    var after: List[Int] = [st.buf_len, 1 if st.pending else 0]
    _host_set_words(bufs[3], after)
    var reports = List[List[Float32]]()
    reports.append(stages.h_last.copy())
    reports.append(stages.k_last.copy())
    reports.append(stages.v_last.copy())
    reports.append(stages.theta_last.copy())
    _host_mamba_reports(sc, OPS_M3_STEP, bufs, b, dm, reports^)


def host_block_step_run(
    mut sc: BlockStepScratch, code: Int, bufs: List[FP], counts: List[Int], ints: List[Int], fs: List[Float32],
    y_out: FP,
) raises:
    """`block_step_run` on a host session: the same operands (host storage),
    the host block bodies. A failed run writes no state (the state goes back
    only after the body returns) and drops the triple's entries."""
    try:
        if code == OPS_M1_STEP:
            _host_run_m1(sc, bufs, counts, ints, y_out)
        elif code == OPS_M2_STEP:
            _host_run_m2(sc, bufs, counts, ints, fs, y_out)
        elif code == OPS_M3_STEP:
            _host_run_m3(sc, bufs, counts, ints, y_out)
        elif code == OPS_ATTN_STEP:
            _host_run_attn(sc, bufs, counts, ints, y_out)
        else:
            raise Error("ops: " + String(code) + " is not a block step op")
    except err:
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's three or four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err


def host_block_step_reports(sc: BlockStepScratch, code: Int, w_addr: Int, s_addr: Int, dsts: List[Int]) raises:
    """`block_step_reports_to_host` of a host session: the last run's reports
    of the host entry over (weights, state) into the caller's arrays (a 0
    address skips that report)."""
    var want = 1 if code == OPS_M2_STEP else 4
    if code != OPS_M2_STEP and code != OPS_M3_STEP:
        raise Error("ops: " + String(code) + " keeps no reports")
    if len(dsts) != want:
        raise Error("ops: " + String(step_op_info(code).name) + " has " + String(want) + " reports")
    for i in range(len(sc.hmamba)):  # driver-loop: over host Mamba entries
        if sc.hmamba[i].code == code and sc.hmamba[i].w_addr == w_addr and sc.hmamba[i].s_addr == s_addr:
            for k in range(want):  # small-loop(the one or four report buffers)
                if dsts[k] != 0:
                    ref r = sc.hmamba[i].reports[k]
                    _host_put(FP(unsafe_from_address=dsts[k]), 0, r, len(r), "report " + String(k))
            return
    raise Error("ops: " + String(code) + ": no resident block over these weights and state (run the step first)")


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
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's three or four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err
    raise Error("ops: " + String(code) + " is not a block step op")


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
                    ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=dsts[0]),
                                     src_buf=sc.m2[i].h_last.create_sub_buffer[DType.float32](0, n))
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
