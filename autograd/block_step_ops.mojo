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
from max.gpu.host import DeviceBuffer, DeviceContext
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from core.identity_trace import IdentityTrace
from mamba.checks.mamba_fixture import MambaDims
from mamba.checks.mamba2_fixture import Mamba2Dims, M2_CHUNK_SIZE
from mamba.checks.mamba3_fixture import Mamba3Dims, M3_CHUNK_SIZE
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

    def __init__(out self, ctx: DeviceContext, mut ex: DeviceExec, w: FP, nw: Int, s: FP, ns: Int, q: FP,
                 b: Int, dm: Int) raises:
        var dims = Mamba2Dims.of(dm)
        var words = _read_words(ex, q, 1)
        if words[0] < 0 or words[0] >= M2_CHUNK_SIZE:
            raise Error("ValueError: ops: mamba2_step: buf_len must be in [0, " + String(M2_CHUNK_SIZE)
                        + "), got " + String(words[0]))
        var aw = MambaArena(over=_view(ex, w, nw), cap=nw, guard=MAMBA_GUARD)
        var dw = Mamba2DeviceWeights(ctx, dims, aw)
        var as_ = MambaArena(over=_view(ex, s, ns), cap=ns, guard=MAMBA_GUARD)
        var ds = Mamba2DeviceState(ctx, b, dims, as_)
        ds.buf_len = words[0]
        self.w_addr = Int(w)
        self.s_addr = Int(s)
        self.q_addr = Int(q)
        self.b = b
        self.dm = dm
        self.dx = mamba_zeros(ctx, b * dm)
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

    def __init__(out self, ctx: DeviceContext, mut ex: DeviceExec, w: FP, nw: Int, s: FP, ns: Int, q: FP,
                 b: Int, dm: Int) raises:
        var dims = Mamba3Dims.of(dm)
        var words = _read_words(ex, q, 2)
        var q0 = words[0]
        var pend = words[1]
        # `_m3_session_copy_in`'s checks, word for word.
        if q0 < 0 or q0 > M3_CHUNK_SIZE:
            raise Error("ValueError: ops: mamba3_step: buf_len out of range")
        if pend != 0 and pend != 1:
            raise Error("ValueError: ops: mamba3_step: pending must be 0 or 1")
        if pend == 1 and q0 != 0:
            raise Error("ValueError: ops: mamba3_step: Input_States pending requires a fresh state")
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
            out: FP) raises:
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
    var y = _view(ex, out, n)
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
            fs: List[Float32], out: FP) raises:
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
    var y = _view(ex, out, n)
    var trace = IdentityTrace.disabled()
    # `_m2_session_step_run`: call-shaped stages at the open chunk's length.
    var stages = Mamba2DeviceStages(ctx, b, l, blk.st.buf_len, dims)
    if l == 1:
        _d2d(ctx, blk.dx, x, n)
        mamba2_block_forward(ctx, stages, blk.st, blk.w, blk.dx, b, 1, fs[0], fs[1], trace,
                             String("ops.step"))
        _d2d(ctx, y, stages.residual_out, n)
        ctx.synchronize()  # the call-shaped stages go at return
    else:
        var dx = mamba_zeros(ctx, n)
        _d2d(ctx, dx, x, n)
        mamba2_block_forward(ctx, stages, blk.st, blk.w, dx, b, l, fs[0], fs[1], trace,
                             String("ops.step"))
        _d2d(ctx, y, stages.residual_out, n)
        ctx.synchronize()
        _ = dx^
    _ = stages^
    var words: List[Int] = [blk.st.buf_len]
    _write_words(ex, bufs[3], words)


def _run_m3(mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
            out: FP) raises:
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
    var y = _view(ex, out, n)
    var trace = IdentityTrace.disabled()
    # `_m3_session_step_run` (its default arm: stages built per call at the
    # open chunk's length).
    var stages = Mamba3DeviceStages(ctx, b, l, blk.st.buf_len, dims)
    if l == 1:
        _d2d(ctx, blk.dx, x, n)
        mamba3_block_forward(ctx, stages, blk.st, blk.w, blk.dx, b, 1, trace, String("ops.step"))
        _d2d(ctx, y, stages.residual_out, n)
        ctx.synchronize()
    else:
        var dx = mamba_zeros(ctx, n)
        _d2d(ctx, dx, x, n)
        mamba3_block_forward(ctx, stages, blk.st, blk.w, dx, b, l, trace, String("ops.step"))
        _d2d(ctx, y, stages.residual_out, n)
        ctx.synchronize()
        _ = dx^
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
                 out: FP, mut pre: AttnPrefill, forward_only: Bool) raises:
    var ctx = ex.ctx.copy()
    var b = ints[0]
    var dm = ints[1]
    var l = _tokens(OPS_ATTN_STEP, counts[0], b, dm)
    var e = attn_step_entry(ex, sc, bufs, counts, ints)
    ref blk = sc.attn[e]
    var n = b * l * dm
    var x = _view(ex, bufs[0], n)
    var y = _view(ex, out, n)
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
              out: FP) raises:
    var pre = AttnPrefill()
    _run_attn_on(ex, sc, bufs, counts, ints, out, pre, False)
    if pre.stages:
        # The run's own L-shaped stages go at return (as before).
        ex.ctx.synchronize()
    _ = pre^


def _step_run[CODE: Int](
    mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int],
    fs: List[Float32], out: FP,
) raises:
    comptime if CODE == OPS_M1_STEP:
        _run_m1(ex, sc, bufs, counts, ints, out)
    elif CODE == OPS_M2_STEP:
        _run_m2(ex, sc, bufs, counts, ints, fs, out)
    elif CODE == OPS_M3_STEP:
        _run_m3(ex, sc, bufs, counts, ints, out)
    else:
        _run_attn(ex, sc, bufs, counts, ints, out)


def attn_step_run(
    mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], out: FP,
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
        _run_attn(ex, sc, bufs, counts, ints, out)
    except err:
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err


def attn_step_run_kept(
    mut ex: DeviceExec, mut sc: BlockStepScratch, bufs: List[FP], counts: List[Int], ints: List[Int], out: FP,
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
        _run_attn_on(ex, sc, bufs, counts, ints, out, pre, forward_only)
    except err:
        pre.clear()
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err


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
    fs: List[Float32], out: FP,
) raises:
    """Run step op `code` (operands checked by the session): `bufs` / `counts`
    are the tensor operands' storage and element counts in the table's
    order, `out` the result's b * L * d_model floats. A failed run drops the
    pair's entry (its state is what the failed call left), so the next run
    rebuilds it from the session tensors."""
    try:
        comptime for i in range(N_STEP_OPS):  # driver-loop: over the step-op table (four rows, compile time)
            comptime CODE = OPS_STEP_FIRST + i
            if code == CODE:
                _step_run[CODE](ex, sc, bufs, counts, ints, fs, out)
                return
    except err:
        for k in range(1, len(bufs)):  # small-loop(bufs: the op's three or four operands): entry eviction, no data
            sc.forget(Int(bufs[k]))
        raise err
    raise Error("ops: " + String(code) + " is not a block step op")
