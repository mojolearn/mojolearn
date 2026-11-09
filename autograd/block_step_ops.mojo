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

Device only: an `OpsSession` on the host executor refuses these codes
(`Exec.IS_DEVICE`), so a host build never instantiates them. Recording: the
ops run in an inference session; a family lane that trains through a step op
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
from transformer.transformer_binding_impl import _lean_for

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
    """Queue the carried state's host integers into their int32 words (the
    executor stages the bytes at once; no wait)."""
    var host = List[Float32](length=len(words), fill=Float32(0))
    for i in range(len(words)):  # small-loop(words: at most two state words): word encode, no data
        host[i] = bitcast[DType.float32](Int32(words[i]))
    ex.upload(p, FP(unsafe_from_address=Int(host.unsafe_ptr())), len(words))
    _ = len(host)  # keep-alive past the staging copy


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
        var cache_n = _attn_cache_floats(ints)
        var kv = LlamaKVCache(ctx, b, dims, smax, window, opts.max_positions)
        kv.k = _view(ex, s, cache_n)
        kv.v = _view(ex, s + cache_n, cache_n)
        kv.s = words[0]
        self.kv = kv^
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
        _ = ns


struct BlockStepScratch(Movable):
    """The resident block structs of one session's step ops, one entry per
    (weights, state) pair, built at the pair's first run."""
    var m1: List[_M1Block]
    var m2: List[_M2Block]
    var m3: List[_M3Block]
    var attn: List[_AttnBlock]

    def __init__(out self):
        self.m1 = List[_M1Block]()
        self.m2 = List[_M2Block]()
        self.m3 = List[_M3Block]()
        self.attn = List[_AttnBlock]()

    def count(self) -> Int:
        return len(self.m1) + len(self.m2) + len(self.m3) + len(self.attn)

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

    def clear(mut self):
        self.m1.clear()
        self.m2.clear()
        self.m3.clear()
        self.attn.clear()

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
        # an attention entry's cached-token count is read back by a rebuild
        for i in range(len(self.attn) - 1, -1, -1):  # driver-loop: over resident attention blocks
            if self.attn[i].q_addr == addr:
                _ = self.attn.pop(i)

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
        for i in range(len(self.attn) - 1, -1, -1):  # driver-loop: over resident attention blocks
            if self.attn[i].w_addr == addr:
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


def _run_attn(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
              y_out: FP) raises:
    var ctx = ex.ctx.copy()
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_ATTN_STEP, counts[0], b, dm)
    _expect(OPS_ATTN_STEP, "weights", counts[1], block_part_floats(OPS_ATTN_STEP, ints, BLOCK_PART_WEIGHTS))
    _expect(OPS_ATTN_STEP, "kv", counts[2], block_part_floats(OPS_ATTN_STEP, ints, BLOCK_PART_STATE))
    _expect(OPS_ATTN_STEP, "state words", counts[3], 1)
    var e = -1
    for i in range(len(sc.attn)):  # driver-loop: over resident attention blocks
        if (sc.attn[i].w_addr == Int(bufs[1]) and sc.attn[i].s_addr == Int(bufs[2])
                and sc.attn[i].q_addr == Int(bufs[3]) and _same_ints(sc.attn[i].ints, ints)):
            e = i
    if e < 0:
        sc.attn.append(_AttnBlock(ctx, ex, bufs[1], counts[1], bufs[2], counts[2], bufs[3], ints))
        e = len(sc.attn) - 1
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
            b, 1, pos0, trace, String("ops.step"))
        if n != len(blk.stages.residual2):
            raise Error("ops: attention_step: the L = 1 stages hold a different output length")
        _d2d(ctx, y, blk.stages.residual2, n)
    else:
        # `_session_forward_run`: call-shaped stages for this call alone.
        var dims = _attn_dims(ints)
        var stages = LlamaDeviceStages(ctx, b, l, ints[6], dims, ints[7], lean=blk.lean)
        var dx = ctx.enqueue_create_buffer[DType.float32](n)
        _d2d(ctx, dx, x, n)
        llama_decoder_layer_forward(
            ctx, stages, blk.kv, blk.rope, blk.w, dx, b, l, pos0, trace,
            String("ops.step.forward"))
        _d2d(ctx, y, stages.residual2, n)
        ctx.synchronize()
        _ = stages^
        _ = dx^
    var words: List[Int] = [blk.kv.s]
    _write_words(ex, bufs[3], words)


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
