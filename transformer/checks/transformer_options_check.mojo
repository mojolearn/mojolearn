# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the block OPTIONS (lane/block-options, 2026-09-17):
`TransformerBlock(..., rope_theta, rope_scaling, rope_dim, max_positions,
qkv_bias, o_bias, norm, norm_eps, norm_bias, mlp, mlp_bias, qk_norm,
attn_softcap)`, device against host oracle, BITWISE, at every one of
contract section 9's thirty stages, for every option axis.

    pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \\
        transformer/checks/transformer_options_check.mojo

**NOTHING IN THIS FILE HAS BEEN RUN.** Written on a box that may not run
it (`[[no-heavy-local-compute]]`); it compiled as an object. Every
"passes" below is a PREDICTION until the RUN OWED line in the lane report
is discharged.

WHAT IT CHECKS, AND WHY A CHECK THAT CANNOT FAIL IS NOT EVIDENCE
------------------------------------------------------------------
Every case is a triple (record ON, record REFERENCE, the first card stage
that must move between them). Three assertions per case:

  1. DEVICE(ON) == ORACLE(ON) at all thirty stages, by bits and by length.
     This is clause (a) of the transformer contract under the record.
  2. ORACLE(ON) != ORACLE(REFERENCE), and the FIRST stage that differs is
     exactly the one the option's seam writes and no earlier one. This is
     the reach proof, `gemm/IDENTICAL_FP32_CONTRACT.md`'s sabotage
     discipline applied to an option instead of a sabotage: an option
     that is wired but inert would pass assertion 1 for ever (a kernel
     that ignores its flag agrees with an oracle that ignores its flag),
     so each option must be shown to MOVE the stage it claims to own.
     Where the reference is not the default record it is the NEAREST
     alternative (layernorm with bias against layernorm without, gelu_tanh
     against gelu, mlp_bias under an ungated MLP against the ungated MLP),
     so the moved stage separates that one option from its neighbour.
  3. For the axes that touch positions (rope, softcap, qk norm), a
     prefill of L tokens whole equals a prefill of the first L-2 tokens
     followed by a call of the last 2, on the device, for the last two
     tokens' block output and the packed caches: contract clause (d)
     under the record.

The control is the DEFAULT record through the OPTIONS constructor and the
options rope table: it must equal the oracle at every stage (the options
plumbing moves no default bit), and it is the one case whose assertion 2
is that NOTHING moves against `build_rope_table` and the old weight
constructor's path.

The ceiling axis is a REFUSAL fixture: `max_positions` below the call's
reach refuses by name on both sides, and a 9000-position table is refused
at the default record and ADMITTED under a linear factor of 2 (its
largest angle is 4499.5 < 8192), on both sides, which is what DEVIATION
2933 claims.

THE SHAPE. The gate shape of contract section 3 (d_model 32, n_heads 2,
head_dim 16) with n_kv 1 (n_rep 2, the GQA map live), intermediate 64,
B 2, L 6, a cache of 8, a 512-position table; one case at head_dim 24
(d_model 48) so the attention scale is inexact and the partial rotary's
width is not a power of two.
"""

from std.memory import bitcast, unsafe_memcpy

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from checks.numerics import numeric_mode_name
from transformer.block_options import (
    BlockOptions,
    MLP_GEGLU,
    MLP_GEGLU_TANH,
    MLP_GELU,
    MLP_GELU_TANH,
    NORM_LAYERNORM,
    NORM_RMSNORM_OFFSET,
    ROPE_SCALING_LINEAR,
    ROPE_SCALING_LLAMA3,
)
from transformer.checks.transformer_fixture import (
    RMS_EPS,
    ScorePlant,
    TransformerDims,
    TransformerWeights,
    bits32_hex,
    fixture_tensor,
)
from transformer.checks.transformer_oracle import (
    TRANSFORMER_STAGE_COUNT,
    TransformerKVCache,
    build_rope_table,
    build_rope_table_opts,
    oracle_dump,
    stage_tag,
    transformer_block_oracle,
)
from transformer.impl.llama.modeling_llama import (
    PLANT_AT_NONE,
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaKVCache,
    LlamaRopeTable,
    _upload,
    _zeros,
    llama_decoder_layer_forward_planted,
)


comptime SEED: UInt64 = 0x424C4B4F50545331  # "BLKOPTS1"
comptime TID_X = 1
comptime TID_NORM1_W = 2
comptime TID_NORM2_W = 3
comptime TID_W_Q = 4
comptime TID_W_K = 5
comptime TID_W_V = 6
comptime TID_W_O = 7
comptime TID_W_GATE = 8
comptime TID_W_UP = 9
comptime TID_W_DOWN = 10
comptime TID_B_Q = 20
comptime TID_B_K = 21
comptime TID_B_V = 22
comptime TID_B_O = 23
comptime TID_NORM1_B = 24
comptime TID_NORM2_B = 25
comptime TID_B_UP = 26
comptime TID_B_DOWN = 27
comptime TID_B_GATE = 28
comptime TID_QN_W = 29
comptime TID_KN_W = 30


def f32(bits: UInt32) -> Float32:
    return bitcast[DType.float32](bits)


def hexbits(v: Float32) -> String:
    """The fixture's `bits32_hex`, the tree's one formatter for a float's
    bits (`[[mojo-string-not-indexable]]`: no per-character String
    indexing here)."""
    return bits32_hex(v)


@fieldwise_init
struct Shape(Copyable, Movable):
    var b: Int
    var l: Int
    var d_model: Int
    var n_heads: Int
    var n_kv: Int
    var head_dim: Int
    var intermediate: Int
    var cap: Int
    var positions: Int

    def dims(self) raises -> TransformerDims:
        var d = TransformerDims(
            self.d_model, self.n_heads, self.n_kv, self.head_dim,
            self.intermediate, self.positions,
        )
        d.validate()
        return d^

    def ldims(self) -> LlamaDims:
        return LlamaDims(
            self.d_model, self.n_heads, self.n_kv, self.head_dim, self.intermediate
        )


def gate_shape() -> Shape:
    return Shape(2, 6, 32, 2, 1, 16, 64, 8, 512)


def hd24_shape() -> Shape:
    return Shape(2, 6, 48, 2, 1, 24, 64, 8, 512)


def weights_for(sh: Shape, opts: BlockOptions) raises -> TransformerWeights:
    """The base nine from one seed (the SAME bits for every record at a
    shape, so a record's reference case shares its weights) plus every
    optional tensor the record switches on, each from its own tensor id.
    An ungated record drops the gate projection."""
    var dims = sh.dims()
    var w = TransformerWeights(dims)
    w.opts = opts.copy()
    var dm = dims.d_model
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var it = dims.intermediate
    var hd = dims.head_dim
    w.norm1_w = fixture_tensor(SEED, TID_NORM1_W, dm, 0.5, 1.5)
    w.norm2_w = fixture_tensor(SEED, TID_NORM2_W, dm, 0.5, 1.5)
    w.w_q = fixture_tensor(SEED, TID_W_Q, qw * dm, -0.5, 0.5)
    w.w_k = fixture_tensor(SEED, TID_W_K, kw * dm, -0.5, 0.5)
    w.w_v = fixture_tensor(SEED, TID_W_V, kw * dm, -0.5, 0.5)
    w.w_o = fixture_tensor(SEED, TID_W_O, dm * qw, -0.25, 0.25)
    if opts.gated():
        w.w_gate = fixture_tensor(SEED, TID_W_GATE, it * dm, -0.25, 0.25)
    w.w_up = fixture_tensor(SEED, TID_W_UP, it * dm, -0.25, 0.25)
    w.w_down = fixture_tensor(SEED, TID_W_DOWN, dm * it, -0.125, 0.125)
    if opts.qkv_bias:
        w.b_q = fixture_tensor(SEED, TID_B_Q, qw, -0.25, 0.25)
        w.b_k = fixture_tensor(SEED, TID_B_K, kw, -0.25, 0.25)
        w.b_v = fixture_tensor(SEED, TID_B_V, kw, -0.25, 0.25)
    if opts.o_bias:
        w.b_o = fixture_tensor(SEED, TID_B_O, dm, -0.25, 0.25)
    if opts.norm_bias:
        w.norm1_b = fixture_tensor(SEED, TID_NORM1_B, dm, -0.5, 0.5)
        w.norm2_b = fixture_tensor(SEED, TID_NORM2_B, dm, -0.5, 0.5)
    if opts.mlp_bias:
        w.b_up = fixture_tensor(SEED, TID_B_UP, it, -0.25, 0.25)
        w.b_down = fixture_tensor(SEED, TID_B_DOWN, dm, -0.25, 0.25)
        if opts.gated():
            w.b_gate = fixture_tensor(SEED, TID_B_GATE, it, -0.25, 0.25)
    if opts.qk_norm:
        w.qn_w = fixture_tensor(SEED, TID_QN_W, hd, 0.5, 1.5)
        w.kn_w = fixture_tensor(SEED, TID_KN_W, hd, 0.5, 1.5)
    return w^


def x_for(sh: Shape) -> List[Float32]:
    return fixture_tensor(SEED, TID_X, sh.b * sh.l * sh.d_model, -2.0, 2.0)


def slice_tokens(x: List[Float32], b: Int, l: Int, dm: Int, t0: Int, t1: Int) -> List[Float32]:
    """Tokens `[t0, t1)` of every batch row, token-major."""
    var out = List[Float32]()
    for bb in range(b):
        for t in range(t0, t1):
            for j in range(dm):
                out.append(x[(bb * l + t) * dm + j])
    return out^


# ===========================================================================
# THE HOST SIDE
# ===========================================================================


def run_host(
    sh: Shape, w: TransformerWeights, x: List[Float32], l: Int,
    mut cache: TransformerKVCache,
) raises -> List[List[Float32]]:
    var dims = sh.dims()
    var rope = build_rope_table_opts(dims, w.opts)
    var st = transformer_block_oracle(w, x, sh.b, l, cache, rope, ScorePlant.none())
    var out = oracle_dump(st)
    _ = st^
    return out^


# ===========================================================================
# THE DEVICE SIDE, through the OPTIONS constructor
# ===========================================================================


def _opt_upload(ctx: DeviceContext, values: List[Float32]) raises -> DeviceBuffer[DType.float32]:
    """A present optional tensor uploaded; an absent one as the placeholder
    the options constructor expects."""
    if len(values) == 0:
        return _zeros[False](ctx, 1)
    return _upload(ctx, values)


struct DeviceRun(Movable):
    var dw: LlamaDeviceWeights
    var kv: LlamaKVCache
    var rope: LlamaRopeTable
    var stages: LlamaDeviceStages
    var dx: DeviceBuffer[DType.float32]

    def __init__(
        out self, ctx: DeviceContext, sh: Shape, w: TransformerWeights,
        x: List[Float32], l: Int,
    ) raises:
        var ld = sh.ldims()
        var opts = w.opts.copy()
        self.dw = LlamaDeviceWeights(
            ctx, ld, opts,
            _upload(ctx, w.norm1_w), _upload(ctx, w.norm2_w),
            _upload(ctx, w.w_q), _upload(ctx, w.w_k), _upload(ctx, w.w_v),
            _upload(ctx, w.w_o), _opt_upload(ctx, w.w_gate),
            _upload(ctx, w.w_up), _upload(ctx, w.w_down),
            _opt_upload(ctx, w.b_q), _opt_upload(ctx, w.b_k),
            _opt_upload(ctx, w.b_v), _opt_upload(ctx, w.b_o),
            _opt_upload(ctx, w.norm1_b), _opt_upload(ctx, w.norm2_b),
            _opt_upload(ctx, w.b_up), _opt_upload(ctx, w.b_down),
            _opt_upload(ctx, w.b_gate), _opt_upload(ctx, w.qn_w),
            _opt_upload(ctx, w.kn_w),
        )
        self.kv = LlamaKVCache(ctx, sh.b, ld, sh.cap, 0, opts.max_positions)
        self.rope = LlamaRopeTable(ctx, ld, opts, sh.positions)
        self.stages = LlamaDeviceStages(ctx, sh.b, l, sh.cap, ld, 0)
        self.dx = _upload(ctx, x)

    def forward(mut self, ctx: DeviceContext, sh: Shape, l: Int, pos0: Int, tag: String) raises:
        var trace = IdentityTrace.disabled()
        llama_decoder_layer_forward_planted(
            ctx, self.stages, self.kv, self.rope, self.dw, self.dx, sh.b, l,
            pos0, PLANT_AT_NONE, List[Int](), List[UInt32](), trace, tag,
        )

    def dump(mut self, ctx: DeviceContext, sh: Shape, l: Int, s: Int) raises -> List[List[Float32]]:
        """Every stage back on the host in card order, at the USED counts
        (`transformer_check.mojo::device_dump`'s shape with the rotary
        table at `rope.half`, which is `rope_dim / 2`). An ungated MLP's
        `gate_proj.out` and `mlp.gated` are EMPTY, as the oracle records
        them."""
        var d = sh.dims()
        var dm = d.d_model
        var qw = d.q_width()
        var kw = d.kv_width()
        var it = d.intermediate
        var nh = d.n_heads
        var nkv = d.n_kv_heads
        var hd = d.head_dim
        var half = self.rope.half
        var m = sh.b * l
        var cells = sh.b * nh * l * s
        var gated = self.dw.opts.gated()
        var counts = List[Int]()
        counts.append(m * dm)  # 0 input.x
        counts.append(m)  # 1 norm1.sumsq
        counts.append(m * dm)  # 2 norm1.out
        counts.append(m * qw)  # 3 q_proj.out
        counts.append(m * kw)  # 4 k_proj.out
        counts.append(m * kw)  # 5 v_proj.out
        counts.append(half)  # 6 rope.inv_freq
        counts.append(self.rope.p_max * half)  # 7 rope.cos
        counts.append(self.rope.p_max * half)  # 8 rope.sin
        counts.append(m * qw)  # 9 q_rope.out
        counts.append(m * kw)  # 10 k_rope.out
        counts.append(sh.b * nkv * s * hd)  # 11 kv.k_cache
        counts.append(sh.b * nkv * s * hd)  # 12 kv.v_cache
        counts.append(cells)  # 13 attn.scores
        counts.append(cells)  # 14 attn.masked
        counts.append(sh.b * nh * l)  # 15 attn.max
        counts.append(cells)  # 16 attn.exp
        counts.append(sh.b * nh * l)  # 17 attn.denom
        counts.append(cells)  # 18 attn.weights
        counts.append(m * qw)  # 19 attn.ctx
        counts.append(m * dm)  # 20 o_proj.out
        counts.append(m * dm)  # 21 residual1.out
        counts.append(m)  # 22 norm2.sumsq
        counts.append(m * dm)  # 23 norm2.out
        counts.append(m * it if gated else 0)  # 24 gate_proj.out
        counts.append(m * it)  # 25 up_proj.out
        counts.append(m * it)  # 26 silu.out
        counts.append(m * it if gated else 0)  # 27 mlp.gated
        counts.append(m * dm)  # 28 down_proj.out
        counts.append(m * dm)  # 29 residual2.out
        var out = List[List[Float32]]()
        for i in range(TRANSFORMER_STAGE_COUNT):
            var n = counts[i]
            var values = List[Float32](length=n, fill=Float32(0.0))
            if n > 0:
                var host = ctx.enqueue_create_host_buffer[DType.float32](n)
                if i == 0:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.dx.create_sub_buffer[DType.float32](0, n))
                elif i == 1:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.norm1_sumsq.create_sub_buffer[DType.float32](0, n))
                elif i == 2:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.norm1_out.create_sub_buffer[DType.float32](0, n))
                elif i == 3:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.q_proj.create_sub_buffer[DType.float32](0, n))
                elif i == 4:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.k_proj.create_sub_buffer[DType.float32](0, n))
                elif i == 5:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.v_proj.create_sub_buffer[DType.float32](0, n))
                elif i == 6:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.rope.inv_freq.create_sub_buffer[DType.float32](0, n))
                elif i == 7:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.rope.cos.create_sub_buffer[DType.float32](0, n))
                elif i == 8:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.rope.sin.create_sub_buffer[DType.float32](0, n))
                elif i == 9:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.q_rope.create_sub_buffer[DType.float32](0, n))
                elif i == 10:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.k_rope.create_sub_buffer[DType.float32](0, n))
                elif i == 11:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.k_cache.create_sub_buffer[DType.float32](0, n))
                elif i == 12:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.v_cache.create_sub_buffer[DType.float32](0, n))
                elif i == 13:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.scores.create_sub_buffer[DType.float32](0, n))
                elif i == 14:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.masked.create_sub_buffer[DType.float32](0, n))
                elif i == 15:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.amax.create_sub_buffer[DType.float32](0, n))
                elif i == 16:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.aexp.create_sub_buffer[DType.float32](0, n))
                elif i == 17:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.denom.create_sub_buffer[DType.float32](0, n))
                elif i == 18:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.weights.create_sub_buffer[DType.float32](0, n))
                elif i == 19:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.ctxv.create_sub_buffer[DType.float32](0, n))
                elif i == 20:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.o_proj.create_sub_buffer[DType.float32](0, n))
                elif i == 21:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.residual1.create_sub_buffer[DType.float32](0, n))
                elif i == 22:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.norm2_sumsq.create_sub_buffer[DType.float32](0, n))
                elif i == 23:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.norm2_out.create_sub_buffer[DType.float32](0, n))
                elif i == 24:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.gate_proj.create_sub_buffer[DType.float32](0, n))
                elif i == 25:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.up_proj.create_sub_buffer[DType.float32](0, n))
                elif i == 26:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.silu_out.create_sub_buffer[DType.float32](0, n))
                elif i == 27:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.gated.create_sub_buffer[DType.float32](0, n))
                elif i == 28:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.down_proj.create_sub_buffer[DType.float32](0, n))
                else:
                    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=self.stages.residual2.create_sub_buffer[DType.float32](0, n))
                ctx.synchronize()
                unsafe_memcpy(dest=values.unsafe_ptr(), src=host.unsafe_ptr(), count=n)
                _ = host^
            out.append(values^)
        return out^


def run_device(
    ctx: DeviceContext, sh: Shape, w: TransformerWeights, x: List[Float32], l: Int,
) raises -> List[List[Float32]]:
    var run = DeviceRun(ctx, sh, w, x, l)
    run.forward(ctx, sh, l, 0, String("opt"))
    var out = run.dump(ctx, sh, l, l)
    _ = run^
    return out^


# ===========================================================================
# COMPARING
# ===========================================================================


def first_moved(a: List[List[Float32]], b: List[List[Float32]]) raises -> Int:
    """The index of the first stage whose length or bits differ; -1 when
    every stage agrees."""
    if len(a) != TRANSFORMER_STAGE_COUNT or len(b) != TRANSFORMER_STAGE_COUNT:
        raise Error(
            String("transformer_options_check: a dump holds ")
            + String(len(a))
            + " / "
            + String(len(b))
            + " stages, not "
            + String(TRANSFORMER_STAGE_COUNT)
        )
    for i in range(TRANSFORMER_STAGE_COUNT):
        if len(a[i]) != len(b[i]):
            return i
        for j in range(len(a[i])):
            if bitcast[DType.uint32](a[i][j]) != bitcast[DType.uint32](b[i][j]):
                return i
    return -1


def describe_diff(a: List[List[Float32]], b: List[List[Float32]], i: Int) raises -> String:
    if len(a[i]) != len(b[i]):
        return (
            stage_tag(i) + " has " + String(len(a[i])) + " cells on one side and "
            + String(len(b[i])) + " on the other"
        )
    var n_diff = 0
    var first = -1
    for j in range(len(a[i])):
        if bitcast[DType.uint32](a[i][j]) != bitcast[DType.uint32](b[i][j]):
            n_diff += 1
            if first < 0:
                first = j
    return (
        stage_tag(i) + " moved on " + String(n_diff) + " of " + String(len(a[i]))
        + " cells starting at cell " + String(first) + ": " + hexbits(a[i][first])
        + " vs " + hexbits(b[i][first])
    )


def assert_device_equals_host(
    ctx: DeviceContext, name: String, sh: Shape, w: TransformerWeights,
) raises -> List[List[Float32]]:
    """Assertion 1. Returns the ORACLE dump for the reach comparison."""
    var x = x_for(sh)
    var dims = sh.dims()
    var cache = TransformerKVCache(sh.b, dims, sh.cap, 0)
    var host = run_host(sh, w, x, sh.l, cache)
    var dev = run_device(ctx, sh, w, x, sh.l)
    var moved = first_moved(dev, host)
    if moved >= 0:
        raise Error(
            String("transformer_options_check: case '") + name
            + "': DEVICE != ORACLE, first at " + describe_diff(dev, host, moved)
        )
    print("  " + name + ": device == oracle at all 30 stages")
    return host^


def assert_reach(
    name: String, on: List[List[Float32]], reference: List[List[Float32]], want: Int,
) raises:
    """Assertion 2: the first stage that differs between the record ON
    and its reference is `want`, and it is not -1."""
    var moved = first_moved(on, reference)
    if moved < 0:
        raise Error(
            String("transformer_options_check: case '") + name
            + "' is INERT: the option moved no stage against its reference"
            + " ([[reached-but-inert]]; a wired option that changes nothing"
            + " gates nothing)"
        )
    if moved != want:
        raise Error(
            String("transformer_options_check: case '") + name
            + "' first moved " + stage_tag(moved) + " but its seam owns "
            + stage_tag(want) + " (" + describe_diff(on, reference, moved) + ")"
        )
    print("  " + name + ": first moved " + stage_tag(moved) + ", as its seam says")


def assert_clause_d(ctx: DeviceContext, name: String, sh: Shape, w: TransformerWeights) raises:
    """Assertion 3: the last two tokens of a whole prefill equal the same
    two tokens computed after a prefill of the first L-2, on the device,
    for `residual2.out` and both packed caches."""
    var x = x_for(sh)
    var l = sh.l
    var dm = sh.d_model
    var whole = DeviceRun(ctx, sh, w, x, l)
    whole.forward(ctx, sh, l, 0, String("whole"))
    var full = whole.dump(ctx, sh, l, l)
    _ = whole^
    var head = slice_tokens(x, sh.b, l, dm, 0, l - 2)
    var tail = slice_tokens(x, sh.b, l, dm, l - 2, l)
    var split = DeviceRun(ctx, sh, w, head, l - 2)
    split.forward(ctx, sh, l - 2, 0, String("head"))
    # The second call at L = 2 on the carried cache: call-shaped stages,
    # the same weights, cache and table.
    var ld = sh.ldims()
    var stages2 = LlamaDeviceStages(ctx, sh.b, 2, sh.cap, ld, 0)
    var dx2 = _upload(ctx, tail)
    var trace = IdentityTrace.disabled()
    llama_decoder_layer_forward_planted(
        ctx, stages2, split.kv, split.rope, split.dw, dx2, sh.b, 2, l - 2,
        PLANT_AT_NONE, List[Int](), List[UInt32](), trace, String("tail"),
    )
    # residual2 of the tail call, and the cache after it.
    var n_out = sh.b * 2 * dm
    var y2 = List[Float32](length=n_out, fill=Float32(0.0))
    var host = ctx.enqueue_create_host_buffer[DType.float32](n_out)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=stages2.residual2.create_sub_buffer[DType.float32](0, n_out))
    ctx.synchronize()
    unsafe_memcpy(dest=y2.unsafe_ptr(), src=host.unsafe_ptr(), count=n_out)
    _ = host^
    var n_cache = sh.b * sh.n_kv * l * sh.head_dim
    var k2 = List[Float32](length=n_cache, fill=Float32(0.0))
    var hostk = ctx.enqueue_create_host_buffer[DType.float32](n_cache)
    ctx.enqueue_copy(dst_ptr=hostk.unsafe_ptr(), src_buf=stages2.k_cache.create_sub_buffer[DType.float32](0, n_cache))
    ctx.synchronize()
    unsafe_memcpy(dest=k2.unsafe_ptr(), src=hostk.unsafe_ptr(), count=n_cache)
    _ = hostk^
    var whole_y = full[29].copy()
    var whole_tail = slice_tokens(whole_y, sh.b, l, dm, l - 2, l)
    for i in range(n_out):
        if bitcast[DType.uint32](whole_tail[i]) != bitcast[DType.uint32](y2[i]):
            raise Error(
                String("transformer_options_check: case '") + name
                + "': clause (d) FAILED, residual2.out of the split tail differs"
                + " at cell " + String(i) + ": " + hexbits(whole_tail[i]) + " vs "
                + hexbits(y2[i])
            )
    var whole_k = full[11].copy()
    for i in range(n_cache):
        if bitcast[DType.uint32](whole_k[i]) != bitcast[DType.uint32](k2[i]):
            raise Error(
                String("transformer_options_check: case '") + name
                + "': clause (d) FAILED, kv.k_cache after the split differs at cell "
                + String(i)
            )
    print("  " + name + ": clause (d), split prefill == whole prefill on the device")
    _ = stages2^
    _ = dx2^
    _ = split^


def run_case(
    ctx: DeviceContext, name: String, sh: Shape, on: BlockOptions,
    reference: BlockOptions, want: Int, clause_d: Bool,
) raises:
    print("case " + name + ": " + on.describe())
    var w_on = weights_for(sh, on)
    var w_ref = weights_for(sh, reference)
    var host_on = assert_device_equals_host(ctx, name, sh, w_on)
    var x = x_for(sh)
    var dims = sh.dims()
    var cache = TransformerKVCache(sh.b, dims, sh.cap, 0)
    var host_ref = run_host(sh, w_ref, x, sh.l, cache)
    assert_reach(name, host_on, host_ref, want)
    if clause_d:
        assert_clause_d(ctx, name, sh, w_on)


def expect_refusal(name: String, got: Bool, needle: String) raises:
    if not got:
        raise Error(
            String("transformer_options_check: '") + name
            + "' was expected to be REFUSED by name (" + needle + ") and was accepted"
        )
    print("  " + name + ": refused by name")


def main() raises:
    print("=== transformer block OPTIONS gate (lane/block-options), mode " + numeric_mode_name())
    var ctx = DeviceContext()
    var sh = gate_shape()
    var dflt = BlockOptions()

    # ---- the control: the default record through the OPTIONS path ------
    # Assertion 1 through the options constructor and the options rope
    # table; assertion 2 inverted: NOTHING moves against the frozen path
    # (`build_rope_table`, the default weights' own dump).
    print("control: the default record through the options constructor")
    var w0 = weights_for(sh, dflt)
    var host0 = assert_device_equals_host(ctx, String("default"), sh, w0)
    var x0 = x_for(sh)
    var dims0 = sh.dims()
    var cache0 = TransformerKVCache(sh.b, dims0, sh.cap, 0)
    var rope_frozen = build_rope_table(dims0)
    var st0 = transformer_block_oracle(w0, x0, sh.b, sh.l, cache0, rope_frozen, ScorePlant.none())
    var frozen = oracle_dump(st0)
    _ = st0^
    var moved0 = first_moved(host0, frozen)
    if moved0 >= 0:
        raise Error(
            String("transformer_options_check: the DEFAULT record moved ")
            + stage_tag(moved0) + " against the frozen path; the options"
            + " plumbing may not move a default bit"
        )
    print("  default: the options path moved no stage against the frozen path")

    # ---- axis 1: the rotary table --------------------------------------
    var theta = BlockOptions()
    theta.rope_theta = Float32(500000.0)
    run_case(ctx, String("rope_theta_500000"), sh, theta, dflt, 6, True)

    var linear = BlockOptions()
    linear.rope_scaling = ROPE_SCALING_LINEAR
    linear.rope_factor = Float32(4.0)
    run_case(ctx, String("rope_scaling_linear_x4"), sh, linear, dflt, 6, True)

    var llama3 = BlockOptions()
    llama3.rope_theta = Float32(500000.0)
    llama3.rope_scaling = ROPE_SCALING_LLAMA3
    llama3.rope_factor = Float32(8.0)
    llama3.rope_low_freq_factor = Float32(1.0)
    llama3.rope_high_freq_factor = Float32(4.0)
    llama3.rope_original_max_positions = 8192
    # Against theta alone, so the smoothing itself is what must move it.
    run_case(ctx, String("rope_scaling_llama3"), sh, llama3, theta, 6, True)

    # DEVIATION 2933, the ceiling axis, as refusals and one admission.
    print("case max_positions: refusals and the angle-domain admission")
    var low = BlockOptions()
    low.max_positions = 4
    var refused = False
    try:
        var wl = weights_for(sh, low)
        var xl = x_for(sh)
        var cl = TransformerKVCache(sh.b, dims0, sh.cap, 0)
        var dl = run_host(sh, wl, xl, sh.l, cl)
        _ = dl^
    except:
        refused = True
    expect_refusal(String("host max_positions=4 under a 6-token call"), refused, String("max_positions"))
    refused = False
    try:
        var wl = weights_for(sh, low)
        var xl = x_for(sh)
        var dr = run_device(ctx, sh, wl, xl, sh.l)
        _ = dr^
    except:
        refused = True
    expect_refusal(String("device max_positions=4 under a 6-token call"), refused, String("max_positions"))
    var big = Shape(1, 2, 32, 2, 1, 16, 64, 9000, 9000)
    var big_default = BlockOptions()
    big_default.max_positions = 16384
    refused = False
    try:
        var t = build_rope_table_opts(big.dims(), big_default)
        _ = t^
    except:
        refused = True
    expect_refusal(String("host 9000-position table at the default record (angle 8999)"), refused, String("Cody-Waite"))
    refused = False
    try:
        var t = LlamaRopeTable(ctx, big.ldims(), big_default, 9000)
        _ = t^
    except:
        refused = True
    expect_refusal(String("device 9000-position table at the default record (angle 8999)"), refused, String("Cody-Waite"))
    var big_linear = BlockOptions()
    big_linear.max_positions = 16384
    big_linear.rope_scaling = ROPE_SCALING_LINEAR
    big_linear.rope_factor = Float32(2.0)
    var th = build_rope_table_opts(big.dims(), big_linear)
    var td = LlamaRopeTable(ctx, big.ldims(), big_linear, 9000)
    if len(th.cos) != 9000 * 8 or td.p_max != 9000:
        raise Error("transformer_options_check: the admitted 9000-position table has the wrong size")
    _ = th^
    _ = td^
    print("  9000-position table under linear factor 2 (largest angle 4499.5): admitted on both sides")

    # ---- axis 2: the projection biases ---------------------------------
    var qkv = BlockOptions()
    qkv.qkv_bias = True
    run_case(ctx, String("qkv_bias"), sh, qkv, dflt, 3, True)
    var ob = BlockOptions()
    ob.o_bias = True
    run_case(ctx, String("o_bias"), sh, ob, dflt, 20, False)

    # ---- axis 3: the norm -----------------------------------------------
    var ln = BlockOptions()
    ln.norm_kind = NORM_LAYERNORM
    run_case(ctx, String("layernorm"), sh, ln, dflt, 1, False)
    var lnb = BlockOptions()
    lnb.norm_kind = NORM_LAYERNORM
    lnb.norm_bias = True
    run_case(ctx, String("layernorm_bias"), sh, lnb, ln, 2, False)
    var eps = BlockOptions()
    eps.norm_eps = Float32(1e-5)
    run_case(ctx, String("norm_eps_1e-5"), sh, eps, dflt, 2, False)
    var off = BlockOptions()
    off.norm_kind = NORM_RMSNORM_OFFSET
    run_case(ctx, String("rmsnorm_offset"), sh, off, dflt, 2, False)

    # ---- axis 4: the MLP ------------------------------------------------
    var gelu = BlockOptions()
    gelu.mlp_kind = MLP_GELU
    run_case(ctx, String("mlp_gelu"), sh, gelu, dflt, 24, False)
    var gelu_t = BlockOptions()
    gelu_t.mlp_kind = MLP_GELU_TANH
    run_case(ctx, String("mlp_gelu_tanh"), sh, gelu_t, gelu, 26, False)
    var geglu = BlockOptions()
    geglu.mlp_kind = MLP_GEGLU
    run_case(ctx, String("mlp_geglu"), sh, geglu, dflt, 26, False)
    var geglu_t = BlockOptions()
    geglu_t.mlp_kind = MLP_GEGLU_TANH
    run_case(ctx, String("mlp_geglu_tanh"), sh, geglu_t, geglu, 26, False)
    var mb = BlockOptions()
    mb.mlp_bias = True
    run_case(ctx, String("mlp_bias_swiglu"), sh, mb, dflt, 24, False)
    var mbu = BlockOptions()
    mbu.mlp_kind = MLP_GELU
    mbu.mlp_bias = True
    run_case(ctx, String("mlp_bias_gelu"), sh, mbu, gelu, 25, False)
    var mbu_t = BlockOptions()
    mbu_t.mlp_kind = MLP_GELU_TANH
    mbu_t.mlp_bias = True
    run_case(ctx, String("mlp_bias_gelu_tanh"), sh, mbu_t, gelu_t, 25, False)
    var mbg = BlockOptions()
    mbg.mlp_kind = MLP_GEGLU
    mbg.mlp_bias = True
    run_case(ctx, String("mlp_bias_geglu"), sh, mbg, geglu, 24, False)
    var mbg_t = BlockOptions()
    mbg_t.mlp_kind = MLP_GEGLU_TANH
    mbg_t.mlp_bias = True
    run_case(ctx, String("mlp_bias_geglu_tanh"), sh, mbg_t, geglu_t, 24, False)

    # ---- axis 5: the q/k norm -------------------------------------------
    var qkn = BlockOptions()
    qkn.qk_norm = True
    run_case(ctx, String("qk_norm"), sh, qkn, dflt, 3, True)
    var qkn_off = BlockOptions()
    qkn_off.qk_norm = True
    qkn_off.norm_kind = NORM_RMSNORM_OFFSET
    run_case(ctx, String("qk_norm_offset_form"), sh, qkn_off, off, 3, False)

    # ---- axis 6: the softcap --------------------------------------------
    var cap = BlockOptions()
    cap.attn_softcap = Float32(50.0)
    run_case(ctx, String("attn_softcap_50"), sh, cap, dflt, 13, True)
    # At head_dim 24 the scale is inexact and the tanh argument is not a
    # power-of-two multiple of the score.
    var sh24 = hd24_shape()
    run_case(ctx, String("attn_softcap_50_hd24"), sh24, cap, dflt, 13, False)

    # ---- axis 7: the partial rotary ------------------------------------
    var part = BlockOptions()
    part.rope_dim = 8
    run_case(ctx, String("rope_dim_8_of_16"), sh, part, dflt, 6, True)
    var part24 = BlockOptions()
    part24.rope_dim = 12
    run_case(ctx, String("rope_dim_12_of_24"), sh24, part24, dflt, 6, True)

    # ---- everything at once, the Gemma-shaped and Qwen-shaped records ---
    var qwen2 = BlockOptions()
    qwen2.qkv_bias = True
    qwen2.rope_theta = Float32(1000000.0)
    qwen2.norm_eps = Float32(1e-6)
    run_case(ctx, String("qwen2_shaped"), sh, qwen2, dflt, 3, True)
    var gemma = BlockOptions()
    gemma.norm_kind = NORM_RMSNORM_OFFSET
    gemma.mlp_kind = MLP_GEGLU_TANH
    gemma.attn_softcap = Float32(50.0)
    gemma.qk_norm = True
    gemma.rope_dim = 8
    gemma.rope_scaling = ROPE_SCALING_LINEAR
    gemma.rope_factor = Float32(2.0)
    run_case(ctx, String("gemma_shaped_everything"), sh, gemma, dflt, 2, True)

    # ---- the refusals by name, both sides -------------------------------
    print("refusals")
    var bad = BlockOptions()
    bad.norm_bias = True
    refused = False
    try:
        bad.validate(16)
    except:
        refused = True
    expect_refusal(String("norm_bias without layernorm"), refused, String("norm_bias"))
    var bad2 = BlockOptions()
    bad2.qk_norm = True
    bad2.norm_kind = NORM_LAYERNORM
    refused = False
    try:
        bad2.validate(16)
    except:
        refused = True
    expect_refusal(String("qk_norm with layernorm"), refused, String("qk_norm"))
    var bad3 = BlockOptions()
    bad3.rope_dim = 7  # odd; 6 is even and legal (half 3), the rule pairs halves
    refused = False
    try:
        bad3.validate(16)
    except:
        refused = True
    expect_refusal(String("odd rope_dim"), refused, String("rope_dim"))
    # A bias present with its flag off: the weights refusal on both sides.
    var stray = weights_for(sh, dflt)
    stray.b_o = fixture_tensor(SEED, TID_B_O, sh.d_model, -0.25, 0.25)
    refused = False
    try:
        var xs = x_for(sh)
        var cs = TransformerKVCache(sh.b, dims0, sh.cap, 0)
        var ds = run_host(sh, stray, xs, sh.l, cs)
        _ = ds^
    except:
        refused = True
    expect_refusal(String("host o_proj.bias present with o_bias off"), refused, String("o_proj.bias"))
    refused = False
    try:
        var xs = x_for(sh)
        var dsd = run_device(ctx, sh, stray, xs, sh.l)
        _ = dsd^
    except:
        refused = True
    expect_refusal(String("device o_proj.bias present with o_bias off"), refused, String("o_proj.bias"))
    # The params round trip, every field by bits.
    var rt = BlockOptions.from_params(gemma.to_params(), 0)
    var p_in = gemma.to_params()
    var p_out = rt.to_params()
    if len(p_in) != len(p_out):
        raise Error("transformer_options_check: BlockOptions.to_params/from_params change the tail length")
    for i in range(len(p_in)):
        if p_in[i] != p_out[i]:
            raise Error(
                String("transformer_options_check: BlockOptions.to_params/from_params do not round-trip at +")
                + String(i)
            )
    print("  BlockOptions params round-trip: every field by bits")

    print("=== transformer_options_check: PASS (every case device == oracle, every option reaches its stage)")
