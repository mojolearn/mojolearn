# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The NORMATIVE host Float32 oracle of one Llama-shaped decoder block under
profile `mojolearn.identical.transformer.fp32.v1`.

NO REFERENCE FILE. HuggingFace ships no oracle; the ALGORITHM this file spells is
theirs, cited seam by seam below and in
`transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` against
huggingface/transformers `d56c55bf564ddb176759eb6ec199442682564916`, whose
line numbers were re-read out of
`/Users/andrewhendel/CascadeProjects/upstream/transformers/` on 2026-08-25
and agree with the contract's. This file is the contract's arithmetic order
written on the CPU through the SAME seams (`identical_mul_add`,
`identical_mul`, `ftz`, `identical_exp`, `identical_div`, `identical_rsqrt`,
`identical_sin`, `identical_cos`, `identical_fmax`, `identical_silu`,
`portable_powf`, and gemm v1's `gemm_oracle`), so that the device card of
`transformer/impl/llama/modeling_llama.mojo` can be
diffed against it bitwise. The device kernels will be an INDEPENDENT
transcription of the same order; nothing here is imported by them except the
seam functions themselves, which are shared BY DESIGN -- one arithmetic, two
spellings of the loops around it.

**NOTHING IN THIS FILE HAS BEEN COMPILED. NOTHING HAS BEEN RUN. NO BIT
PRODUCED BY IT HAS BEEN OBSERVED ON ANY DEVICE OR ANY HOST.** Written
2026-08-25. Every sentence below that says what a seam DOES is a statement
about what the source says, and every sentence that says what two runs will
AGREE ON is a PREDICTION. The word "identical" appears in this file only as
the name of the profile and of the imported functions. It is not a claim
this lane has earned; contract section 11's last two bullets are the reason,
and the GEMM lane's own history is the standing warning -- Apple and AMD
agreed bit for bit through 302 stages while NVIDIA diverged at
`tree001.winners.scores`, so two backends agreeing closes nothing and zero
backends agreeing closes less.

## It is slow on purpose

Scalar, host-side, no parallelism, no device, no vectorization, one loop per
clause. The point is that a reader can put contract section 4's
twenty-three-row table beside this file and check it line by line. Where a
faster spelling exists and would give the same bits, this file still writes
the slow one, because "would give the same bits" is exactly the kind of
sentence an oracle exists to stop people from having to believe.

## The three things this file adds that the contract did not pin

Numbered in this lane's fresh range 1000-1019 (contract section 12.3 assigns
800-819 and every decision it already made keeps its own number there).

  DEVIATION 1002  the rotary table is sized from the CONFIGURATION and not
                  from L, `transformer_fixture.mojo::TransformerDims`.
  DEVIATION 1003  this file imports `core/identity_trace.mojo`, and that
                  pulls `max.gpu.host` into a host-only oracle.
  DEVIATION 1004  the card records the three rope stages on EVERY call.

DEVIATION 1001 (the score plant) and DEVIATION 1005 (its second application
point, and the contract defect that forced it) are argued at length in
`transformer_fixture.mojo` where the plants live.

## DEVIATION 816, resolved, and NOT the way the contract expected

Contract section 0's correction 1 and section 12.3's DEVIATION 816 say this
lane needs "a fourth copy or an import" of `pinned_mul`, on the grounds that
it is not in `checks/numerics.mojo` and that three identical copies live
inside `mamba/`. **That was true when the contract was written on 2026-08-24
and it is no longer true.** The numerics lane landed DEVIATION 826,
`checks/numerics.mojo::identical_mul`, the same day, with a docstring
that names itself "the canonical home" and names all three mamba copies.

So this lane takes neither option. It calls `identical_mul` and makes NO
fourth copy and NO `transformer/ -> mamba/` dependency. The reasoning, since
an undocumented choice here would be the wrong kind of quiet:

  - A fourth copy would make four spellings of one multiply, and four copies
    of an arithmetic have four chances to be edited apart. That risk is not
    hypothetical in this repository; it is the reason DEVIATION 826 exists.
  - An import from `mamba/checks/mamba_oracle.mojo` would point a
    dependency arrow from this lane into a lane under concurrent edit, for
    one line, and would make the transformer oracle fail to compile whenever
    the mamba oracle does.
  - `checks/numerics.mojo` is the file every directory already imports.
    The arrow points the right way.

The cost if this is wrong: `identical_mul` is newer than the three mamba
copies and has not been through a three-vendor leg under that name, so if it
ever drifts from `pinned_mul` this lane's bits move and the mamba lane's do
not. It is four tokens of body (`identical_mul_add(a, b, Float32(-0.0))`) so
drift is unlikely, and the check file owes one assertion that
`identical_mul(a, b)` equals `identical_mul_add(a, b, -0.0)` over a planted
set including both zero signs. THE NAME `pinned_mul` DOES NOT APPEAR IN THIS
FILE, so a reader matching the contract's table against the code should read
every `identical_mul` as the contract's `pinned_mul`.

## What is NOT here

No Float64 tolerance reference. `mamba_oracle.mojo` carries one
(`mamba_block_ref64`) and this file deliberately does not, because contract
section 0 says the transformer corpus "is a later phase and does not exist",
and a float64 reference written now would be a second unreviewed
implementation of a block whose first implementation has not been run. When
`transformer/corpus/` is opened it should carry the float64 reference, in
torch, on the `mamba/corpus/` pattern, so that the tolerance instrument is
not our own code twice.
"""

from core.host_lanes import all_finite
from std.math import min
from std.memory import bitcast

from core.identity_trace import IdentityTrace
from gemm.checks.gemm_oracle import OP_NT, gemm_oracle
# lane/lowbit-blocks (2026-09-29): the profile's host answer, Lane C's.
from gemm.host.gemm_int15_oracle import gemm_int15_from_f32_oracle
from std.time import perf_counter_ns

from core.host_lanes import HostF32Ptr, host_block_timing_on, host_f32_uninit, host_tick
from core.host_parallel import host_parallelize
from core.host_lanes import (
    host_row_tasks,
    span_add,
    span_div,
    span_exp_shift,
    span_fmax_fold,
    span_mul,
    span_scale,
    span_silu,
)
from gemm.host.gemm_host_rows import (
    GHR_FW,
    GHR_G,
    GhrF,
    gemm_host_rows,
    ghr_ftz_lanes,
    ghr_pack_a,
    ghr_pack_b,
    ghr_panel_count,
    ghr_tile,
)
from std.sys.compile import is_defined
from checks.numerics import (
    ftz,
    identical_cos,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_gelu_erf,
    identical_gelu_tanh,
    identical_mul,
    identical_mul_add,
    identical_mul_add_simd,
    identical_rsqrt,
    identical_silu,
    identical_sin,
    identical_tanh,
    portable_powf,
)
# lane/block-options (2026-09-17): the options record. Every option's seam
# below cites its DEVIATION number from `transformer/block_options.mojo`.
from transformer.block_options import (
    BlockOptions,
    MLP_SWIGLU,
    NORM_LAYERNORM,
    NORM_RMSNORM,
    NORM_RMSNORM_OFFSET,
    ROPE_SCALING_LINEAR,
    ROPE_SCALING_LLAMA3,
    ROPE_SCALING_NONE,
    rope_angle_domain,
)
from transformer.checks.transformer_fixture import (
    MAX_ABS_POSITION,
    PLANT_AT_MASKED,
    PLANT_AT_SCORES,
    RMS_EPS,
    ROPE_THETA,
    ScorePlant,
    TransformerDims,
    TransformerWeights,
    attention_scale,
    f32_from_bits,
    mask_fill,
    unmasked_fill,
)


# ===========================================================================
# THE REFUSALS (contract section 8)
# ===========================================================================


def refuse_nonfinite(name: String, values: List[Float32]) raises:
    """Contract section 8, first bullet, and IDENTITY_PATHS row 39: a NaN or
    an infinity in an INPUT or a PARAMETER is REFUSED BY NAME before any
    recorded stage.

    Not because NaN arithmetic is nondeterministic (it is not) but because
    NaN PAYLOADS are vendor-shaped. Row 39 measured three payloads for one
    IEEE answer, `0x7fc00000` on Apple, `0x7fffffff` on NVIDIA, `0xffc00000`
    on AMD, so a certified stage hash can never contain a computed NaN and
    the honest thing is to refuse the input rather than to certify a card
    that cannot be compared.

    **TESTED BY BITS, NOT BY COMPARES.** `mamba_oracle.mojo:57`'s spelling,
    for its reason: Metal FLUSHES COMPARE OPERANDS (row 49, DEVIATION 746
    (i)), so `v != v` is a test with two meanings across columns while a
    mask-and-compare on the exponent field has one. The device transcription
    of this function must keep the bit test.

    CONTRACT SECTION 8's SECOND BULLET IS A STATED GAP AND THIS FUNCTION
    DOES NOT CLOSE IT (DEVIATION 815). A nonfinite INTERMEDIATE is
    reachable: S13 can overflow a score to `-inf` (contract 4.1(b)) and at
    extreme weights `identical_exp` saturates. Those are deterministic and
    the same on every vendor so they do not break identity, but a computed
    NaN would, and this profile does not test for one at every stage. What
    catches it is the card, since a stage hash containing a vendor-shaped
    payload cannot match. That is a detection, not a prevention."""
    if all_finite(values):
        return  # the bit test below, as lanes (lane neural-cpu); nothing to refuse
    for i in range(len(values)):
        var au = bitcast[DType.uint32](values[i]) & UInt32(0x7FFFFFFF)
        if au > UInt32(0x7F800000):
            raise Error(
                String("transformer: NaN in ") + name + " at flat index "
                + String(i) + " REFUSED (row 39: NaN payloads are"
                + " vendor-shaped; no stage may record one)"
            )
        if au == UInt32(0x7F800000):
            raise Error(
                String("transformer: infinity in ") + name
                + " at flat index " + String(i) + " REFUSED (row 39)"
            )


def _refuse_wrong_length(name: String, got: Int, want: Int) raises:
    if got != want:
        raise Error(
            String("transformer: ") + name + " has " + String(got)
            + " elements and the config wants " + String(want)
            + " REFUSED. A mis-sized weight does not fail loudly inside the"
            + " GEMM; it reads a neighbouring row and returns a plausible"
            + " number, which is the failure mode"
            + " [[uniform-test-data-hides-permutation]] is about."
        )


def refuse_bad_weights(w: TransformerWeights) raises:
    """Shapes first, then values.

    The shape check is here rather than left to `gemm_oracle` because the
    GEMM takes m, n and k as arguments and cannot know that the buffer it
    was handed is the wrong one; a `w_k` passed where `w_q` belongs at the
    same shape is a silent wrong answer, and a `w_k` of the wrong shape
    silently reads past its rows. Refusing by name is the difference between
    a five-second diagnosis and an afternoon."""
    var d = w.dims.copy()
    var dm = d.d_model
    var o = w.opts.copy()
    o.validate(d.head_dim)
    _refuse_wrong_length("input_layernorm.weight", len(w.norm1_w), dm)
    _refuse_wrong_length("post_attention_layernorm.weight", len(w.norm2_w), dm)
    _refuse_wrong_length("q_proj.weight", len(w.w_q), d.q_width() * dm)
    _refuse_wrong_length("k_proj.weight", len(w.w_k), d.kv_width() * dm)
    _refuse_wrong_length("v_proj.weight", len(w.w_v), d.kv_width() * dm)
    _refuse_wrong_length("o_proj.weight", len(w.w_o), dm * d.q_width())
    # lane/block-options: an ungated MLP carries no gate projection.
    if o.gated():
        _refuse_wrong_length("gate_proj.weight", len(w.w_gate), d.intermediate * dm)
    else:
        _refuse_optional("gate_proj.weight", len(w.w_gate), 0, False)
    _refuse_wrong_length("up_proj.weight", len(w.w_up), d.intermediate * dm)
    _refuse_wrong_length("down_proj.weight", len(w.w_down), dm * d.intermediate)
    # The eleven optional tensors: present iff their flag is on.
    _refuse_optional("q_proj.bias", len(w.b_q), d.q_width(), o.qkv_bias)
    _refuse_optional("k_proj.bias", len(w.b_k), d.kv_width(), o.qkv_bias)
    _refuse_optional("v_proj.bias", len(w.b_v), d.kv_width(), o.qkv_bias)
    _refuse_optional("o_proj.bias", len(w.b_o), dm, o.o_bias)
    _refuse_optional("input_layernorm.bias", len(w.norm1_b), dm, o.norm_bias)
    _refuse_optional(
        "post_attention_layernorm.bias", len(w.norm2_b), dm, o.norm_bias
    )
    _refuse_optional("up_proj.bias", len(w.b_up), d.intermediate, o.mlp_bias)
    _refuse_optional("down_proj.bias", len(w.b_down), dm, o.mlp_bias)
    _refuse_optional(
        "gate_proj.bias", len(w.b_gate), d.intermediate, o.has_gate_bias()
    )
    _refuse_optional("q_norm.weight", len(w.qn_w), d.head_dim, o.qk_norm)
    _refuse_optional("k_norm.weight", len(w.kn_w), d.head_dim, o.qk_norm)
    refuse_nonfinite("input_layernorm.weight", w.norm1_w)
    refuse_nonfinite("post_attention_layernorm.weight", w.norm2_w)
    refuse_nonfinite("q_proj.weight", w.w_q)
    refuse_nonfinite("k_proj.weight", w.w_k)
    refuse_nonfinite("v_proj.weight", w.w_v)
    refuse_nonfinite("o_proj.weight", w.w_o)
    refuse_nonfinite("gate_proj.weight", w.w_gate)
    refuse_nonfinite("up_proj.weight", w.w_up)
    refuse_nonfinite("down_proj.weight", w.w_down)
    refuse_nonfinite("q_proj.bias", w.b_q)
    refuse_nonfinite("k_proj.bias", w.b_k)
    refuse_nonfinite("v_proj.bias", w.b_v)
    refuse_nonfinite("o_proj.bias", w.b_o)
    refuse_nonfinite("input_layernorm.bias", w.norm1_b)
    refuse_nonfinite("post_attention_layernorm.bias", w.norm2_b)
    refuse_nonfinite("up_proj.bias", w.b_up)
    refuse_nonfinite("down_proj.bias", w.b_down)
    refuse_nonfinite("gate_proj.bias", w.b_gate)
    refuse_nonfinite("q_norm.weight", w.qn_w)
    refuse_nonfinite("k_norm.weight", w.kn_w)


def _refuse_optional(name: String, got: Int, want: Int, on: Bool) raises:
    """An optional tensor is PRESENT (at its exact length) iff its option is
    on. Both mismatches are refused by the tensor's name, because a bias
    that is silently ignored and a bias that is silently zero are the same
    plausible-wrong-answer class as a mis-sized weight."""
    if on:
        if got != want:
            raise Error(
                String("transformer: ")
                + name
                + " is required by its option and has "
                + String(got)
                + " elements where the config wants "
                + String(want)
                + " REFUSED"
            )
    elif got != 0:
        raise Error(
            String("transformer: ")
            + name
            + " is present ("
            + String(got)
            + " elements) but its option is off REFUSED (pass the option"
            + " or drop the tensor; a silently ignored tensor is a wrong"
            + " model that looks like a right one)"
        )


# ===========================================================================
# THE ROTARY TABLE (contract seams S6, S7, S8; DEVIATIONS 809, 810, 812)
# ===========================================================================


struct RopeTable(Copyable, Movable):
    """The precomputed sin and cos table, indexed by ABSOLUTE POSITION.

    Built ONCE per configuration, never per call, which is both what the
    reference effectively does (`LlamaRotaryEmbedding` holds `inv_freq` as
    a buffer and recomputes only the position-dependent part) and what MAX
    does (`max/kernels/src/nn/rope.mojo` takes `freqs_cis` as a TENSOR of
    shape `[max_seq_len, rope_dim]` and computes no angle inside any
    kernel). Contract 12.1 item 2 calls that "FOLLOWED, both halves".

    THE TABLE IS STORED DEDUPLICATED, `[positions, head_dim/2]`, not
    `[positions, head_dim]`. The reference builds `emb = cat(freqs, freqs)`
    (modeling_llama.py:123) so that `cos[j]` and `cos[j + head_dim/2]` are
    the SAME value; contract section 4's preamble lists that concatenation
    among the COPIES, and a copy is not a seam. Storing it once and reading
    `ci = j % (head_dim/2)` is the same bits with half the memory. The card
    stage `rope.cos` is `[P_max, head_dim/2]` for the same reason (contract
    section 9), so a device kernel that materializes the doubled table must
    still record the halved one or the tag will not align.

    ROW 39 IS NOT CHECKED ON THIS TABLE and it does not need to be: every
    entry is `identical_cos` or `identical_sin` of a finite angle, both of
    which return a value in [-1, 1] on their domain. The angle's domain IS
    checked, by `build_rope_table`, and that is DEVIATION 812."""

    var head_dim: Int
    var positions: Int
    var rope_dim: Int
    """DEVIATION 2948 (lane/block-options): how many leading columns of a
    head are rotated. `head_dim` for the full rotary (every table built
    before this lane); a smaller even number for the partial rotary of the
    GPT-NeoX/Phi families. The table is `[positions, rope_dim/2]`."""
    var inv_freq: List[Float32]
    var cos: List[Float32]
    var sin: List[Float32]

    def __init__(out self, head_dim: Int, positions: Int):
        self.head_dim = head_dim
        self.positions = positions
        self.rope_dim = head_dim
        self.inv_freq = List[Float32]()
        self.cos = List[Float32]()
        self.sin = List[Float32]()

    def __init__(out self, head_dim: Int, positions: Int, rope_dim: Int):
        self.head_dim = head_dim
        self.positions = positions
        self.rope_dim = rope_dim
        self.inv_freq = List[Float32]()
        self.cos = List[Float32]()
        self.sin = List[Float32]()

    def half(self) -> Int:
        return self.rope_dim // 2


def build_rope_table(dims: TransformerDims) raises -> RopeTable:
    """Today's table: `build_rope_table_opts` at the default record, which
    is theta 10000.0, no scaling, the full rotary. Every pre-existing
    caller (the byte LM host, the host forward, the gates) reaches this and
    gets the bits it always got."""
    return build_rope_table_opts(dims, BlockOptions())


def rope_inv_freq_scaled(opts: BlockOptions, rope_dim: Int) raises -> List[Float32]:
    """Seam S6 with the options, ORACLE spelling (the device file carries
    its own transcription, `llama_rope_inv_freq_scaled_host`).

    DEVIATION 2930, `rope_theta` as a parameter: the base is `opts.rope_theta`
    where it was the frozen `ROPE_THETA`; the arithmetic is S6's unchanged,
    `e = 2i / rope_dim`, `inv = 1 / theta**e`, through `identical_div` and
    `portable_powf`. `modeling_rope_utils.py::_compute_default_rope_parameters`
    (huggingface/transformers d56c55b, :95-135, `dim = head_dim *
    partial_rotary_factor`).

    DEVIATION 2931, `rope_scaling = linear`:
    `_compute_linear_scaling_rope_parameters` (:138-176) is the default
    inverse frequencies then `inv_freq /= factor`, ONE division, and an
    `attention_factor` of exactly 1.0. Spelled `ftz(identical_div(inv, factor))`.

    DEVIATION 2932, `rope_scaling = llama3`: `_compute_llama3_parameters`
    (:384-432), in the reference's own operation order with every
    intermediate a Float32 and every division through `identical_div`:

        low_wl  = old / low_freq_factor
        high_wl = old / high_freq_factor
        wl      = 2*pi / inv                        (2*pi = 0x40C90FDB)
        inv_l   = inv / factor if wl > low_wl else inv
        medium  = not (wl < high_wl) and not (wl > low_wl)
        if medium:
            smooth = (old / wl - low_freq_factor) / (high_freq_factor - low_freq_factor)
            inv_l  = ((1 - smooth) * inv_l) / factor + smooth * inv_l

    The reference evaluates `(1 - s) * inv_l / factor` left to right, so the
    product is rounded before the division; both products are `identical_mul`
    (uncontractible) and the final add is a plain add of two rounded
    terms, UNFUSED, for the same reason S10 is. The comparisons are exact
    Float32 compares on flushed values. `attention_factor` is 1.0 for
    llama3 too (:431), so no scaling of cos and sin is spelled. **This is
    OUR spelling of the reference's arithmetic, not agreement with torch**:
    torch mixes Python float64 scalars into these expressions and rounds
    where its promotion rules say; contract section 11's "not agreement
    with HuggingFace" covers it.
    """
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise Error("transformer: rope needs an even positive rope_dim")
    var half = rope_dim // 2
    var out = List[Float32]()
    var theta = opts.rope_theta
    for i in range(half):
        var e = ftz(identical_div(Float32(2 * i), Float32(rope_dim)))
        var inv = ftz(identical_div(Float32(1.0), portable_powf(theta, e)))
        if opts.rope_scaling == ROPE_SCALING_LINEAR:
            inv = ftz(identical_div(inv, ftz(opts.rope_factor)))
        elif opts.rope_scaling == ROPE_SCALING_LLAMA3:
            var factor = ftz(opts.rope_factor)
            var lo_f = ftz(opts.rope_low_freq_factor)
            var hi_f = ftz(opts.rope_high_freq_factor)
            var old = Float32(opts.rope_original_max_positions)
            var low_wl = ftz(identical_div(old, lo_f))
            var high_wl = ftz(identical_div(old, hi_f))
            var two_pi = bitcast[DType.float32](UInt32(0x40C90FDB))
            var wl = ftz(identical_div(two_pi, inv))
            var inv_l = inv
            if wl > low_wl:
                inv_l = ftz(identical_div(inv, factor))
            var medium = (not (wl < high_wl)) and (not (wl > low_wl))
            if medium:
                var ratio = ftz(identical_div(old, wl))
                var num = ftz(ratio - lo_f)
                var den = ftz(hi_f - lo_f)
                var smooth = ftz(identical_div(num, den))
                var one_minus = ftz(Float32(1.0) - smooth)
                var t1 = ftz(identical_mul(one_minus, inv_l))
                t1 = ftz(identical_div(t1, factor))
                var t2 = ftz(identical_mul(smooth, inv_l))
                inv_l = ftz(ftz(t1) + ftz(t2))
            inv = inv_l
        out.append(inv)
    return out^


def refuse_rope_angle_domain(
    inv_freq: List[Float32], positions: Int, what: String
) raises:
    """DEVIATION 2933: the ceiling is the ANGLE's domain. The largest angle
    the table will hold is `(positions - 1) * inv_freq[i]` at the column
    whose inverse frequency is largest, spelled exactly as S7 spells the
    angle; if it is at or above 8192.0 the Cody-Waite reduction of
    `_cephes_sincosf_core` has no certified answer and the table is REFUSED
    BY NAME rather than filled with numbers the reduction lost. At the
    default options `inv_freq[0] == 1.0` and this is DEVIATION 812's
    `positions > 8192`, exactly; under a linear factor `f` it admits
    `f` times as many positions, and a wider reduction in
    `checks/numerics.mojo` (the numerics lane's edit, not this lane's) is
    the only way past it."""
    var worst = Float32(0.0)
    var worst_i = 0
    for i in range(len(inv_freq)):
        var angle = ftz(identical_mul(Float32(positions - 1), ftz(inv_freq[i])))
        if angle > worst:
            worst = angle
            worst_i = i
    if worst >= rope_angle_domain():
        raise Error(
            what
            + ": a rotary table of "
            + String(positions)
            + " positions puts the RoPE angle at column "
            + String(worst_i)
            + " at "
            + String(worst)
            + ", at or beyond the Cody-Waite domain of _cephes_sincosf_core"
            + " (8192.0; DEVIATION 812 restated as DEVIATION 2933). Fewer"
            + " positions, a rope_scaling factor, or a wider reduction in"
            + " checks/numerics.mojo are the exits; a clamped angle is a"
            + " wrong answer that looks like a right one"
        )


def build_rope_table_opts(dims: TransformerDims, opts: BlockOptions) raises -> RopeTable:
    """Seams S6, S7 and S8, on the HOST, once.

    S6, the inverse frequencies (`compute_default_rope_parameters`,
    modeling_llama.py:93-109, the reference's line :108 being
    `inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))`):

        e   = ftz(identical_div(Float32(2*i), Float32(head_dim)))
        inv = ftz(identical_div(1.0, portable_powf(theta, e)))

    **`portable_powf` AND NOT THE HOST'S `pow`**, DEVIATION 809. This is
    IDENTITY_PATHS row 18 applied to a CONSTANT, and the reason it is worth
    a deviation number is that the hazard changes shape when the value is
    baked at configuration time: cross-vendor becomes cross-HOST. A table
    built by glibc's `pow` on the machine that wrote the checkpoint and a
    table built by Apple's `pow` on the machine that reads it are two
    tables, and no amount of kernel pinning downstream can recover the
    difference. `portable_powf` is `exp(p * log(x))` through the two row-12
    functions, so it is the same bits on every host and every device.

    S7, the angle (`LlamaRotaryEmbedding.forward`, :113-127, whose :114-122
    is a `k = 1` matmul in FP32 with autocast explicitly disabled):

        angle = identical_mul(Float32(absolute_position), inv_freq[i])

    The contract's note on why a `k = 1` GEMM cell and a pinned product
    coincide here is worth restating, because it is the only place in this
    profile where a GEMM call was replaced by something cheaper: at `k = 1`
    the gemm v1 leaf is `ftz(fma(a, b, +0.0))`, and `fma(a, b, +0.0)`
    differs from `fma(a, b, -0.0)` only when the product is a zero of
    negative sign. Both operands here are non-negative (`position >= 0` and
    `inv_freq > 0`), so the product is never a negative zero and the two
    spellings are bit-equal at every reachable input. THAT ARGUMENT DEPENDS
    ON `position >= 0` AND ON NOTHING ELSE, so if a future profile ever
    admits a negative position the substitution stops being free.

    S8, the rotation's cosine and sine:

        cos = identical_cos(angle)      DEVIATION 258
        sin = identical_sin(angle)      DEVIATION 820

    Both route into `_cephes_sincosf_core`, so BOTH HALVES OF ONE ROTATION
    COME FROM ONE REDUCTION rather than from two functions wearing one name.
    The reference's `* attention_scaling` (:124-125) is exactly `1.0` under
    `rope_type = "default"` (contract section 3) and multiplying by 1.0 is
    bit-inert on every finite input and on both zero signs, so it is NOT
    spelled. That is a deliberate omission, not an oversight: spelling it
    would put a `pinned_mul` on the path whose only effect would be to make
    a reader wonder what it was for.

    DEVIATION 812, the domain, enforced here and nowhere else. Cephes's
    Cody-Waite reduction is valid on |x| < 8192 and `_cephes_sincosf_core`
    DOES NOT GUARD above it -- `portable_sinf`'s docstring says so in as
    many words and asks this lane to bound the angle or ask for a wider
    reduction. `inv_freq[0]` is `theta ** 0`, which `portable_powf` returns
    as exactly `1.0`, so the angle at `i = 0` IS the position and the
    position is therefore the binding constraint. A refusal, not a clamp: a
    clamped angle is a wrong answer that looks like a right one."""
    dims.validate()
    opts.validate(dims.head_dim)
    var hd = dims.head_dim
    # DEVIATION 2948: the rotated width. `half` columns of table, and the
    # card stage `rope.cos` is `[P_max, rope_dim/2]` (head_dim/2 today).
    var rd = opts.rope_dim_of(hd)
    var half = rd // 2
    var t = RopeTable(hd, dims.rope_positions, rd)
    if dims.rope_positions > opts.max_positions:
        raise Error(
            String("transformer: rope table of ")
            + String(dims.rope_positions)
            + " positions exceeds the model's declared max_positions "
            + String(opts.max_positions)
            + " REFUSED (DEVIATION 2933)"
        )

    # ---- S6, with DEVIATIONS 2930-2932 ----------------------------------
    t.inv_freq = rope_inv_freq_scaled(opts, rd)
    # DEVIATION 2933: the ceiling, on the ANGLE. Was `rope_positions >=
    # MAX_ABS_POSITION` here, which refused a table of exactly 8192
    # positions whose largest angle (8191) is inside the domain while the
    # device admitted it; the two sides now agree by construction.
    refuse_rope_angle_domain(t.inv_freq, dims.rope_positions, "transformer")

    # ---- S7 then S8, position major, so the table's flat order is the
    #      card stage's `[P_max, rope_dim/2]` -----------------------------
    for p in range(dims.rope_positions):
        for i in range(half):
            var angle = ftz(identical_mul(Float32(p), ftz(t.inv_freq[i])))
            t.cos.append(ftz(identical_cos(angle)))
            t.sin.append(ftz(identical_sin(angle)))
    return t^


# ===========================================================================
# THE KV CACHE (contract section 7, DEVIATION 814)
# ===========================================================================


struct TransformerKVCache(Copyable, Movable):
    """The recurrent state between calls, `k_cache` and `v_cache`, each
    `[B, n_kv_heads, cap, head_dim]` row-major, appended to at
    modeling_llama.py:261-262 through `past_key_values.update`.

    **SLOT j HOLDS THE TOKEN WHOSE ABSOLUTE POSITION IS j.** That identity
    is the whole of DEVIATION 814 and it is what makes contract section
    5.5's "one axis, one direction, one origin" checkable rather than
    aspirational. It has two consequences a reader should hold onto:

      - A call's tokens have absolute positions `used .. used + L - 1`,
        where `used` is read BEFORE the append. So a prefill of 129 tokens
        followed by a call of 4 gives the second call positions 129..132,
        and RoPE reads the table there. An implementation that indexes the
        table from the start of the current slice reads 0..3, which is
        sabotage `S07_ROPE_RELATIVE_POSITION`, and it is INVISIBLE in a
        single-call gate because a single call always starts at 0. That is
        contract section 10's warning about gates on the gates.
      - The causal mask is `key_slot > query_absolute_position`, with no
        offset arithmetic anywhere. Prefill and decode use the same
        expression.

    `cap` and `used` are SEPARATE and the difference is load-bearing. Every
    fold in this file walks `[0, used)`. Contract section 5.3 names the
    alternative, folding over the ALLOCATION, as the thing a v2 that wants a
    reduction tree would be forced into, and names it as a research question
    rather than an exit. Nothing here folds over `cap`."""

    var b: Int
    var n_kv: Int
    var head_dim: Int
    var cap: Int
    var used: Int
    var window: Int
    var ring: Int
    var k: List[Float32]
    var v: List[Float32]

    def __init__(
        out self, b: Int, dims: TransformerDims, cap: Int, window: Int = 0
    ) raises:
        """`window == 0`: slot `j` is absolute position `j`, `cap` slots.
        `window > 0`: sliding-window causal attention; the buffers are a
        RING of `window` slots per (batch, kv head), slot `= j % window`,
        and `cap` still bounds absolute positions."""
        if b <= 0 or cap <= 0:
            raise Error("transformer: KV cache needs positive B and capacity")
        if window < 0:
            raise Error("transformer: KV cache window must be >= 0")
        self.b = b
        self.n_kv = dims.n_kv_heads
        self.head_dim = dims.head_dim
        self.cap = cap
        self.used = 0
        self.window = window
        self.ring = cap
        if window > 0:
            self.ring = window
        self.k = List[Float32]()
        self.v = List[Float32]()
        for _ in range(b * dims.n_kv_heads * self.ring * dims.head_dim):
            self.k.append(Float32(0.0))
            self.v.append(Float32(0.0))

    def slot(self, bb: Int, kv: Int, j: Int, d: Int) -> Int:
        var jj = j
        if self.window > 0:
            jj = j % self.window
        return ((bb * self.n_kv + kv) * self.ring + jj) * self.head_dim + d

    def key_lo(self, pos0: Int) -> Int:
        """First absolute key a call starting at `pos0` reads: 0 for full
        causal, `max(0, pos0 - window + 1)` under a window."""
        if self.window <= 0:
            return 0
        var lo = pos0 - self.window + 1
        if lo < 0:
            lo = 0
        return lo

    def plant_slots(mut self, first: Int, values: List[Float32]) raises:
        """Write known bits into slots `[first, cap)`, k block then v block.

        For the hot-tail fixture, and ONLY for it. See
        `transformer_fixture.mojo::fixture_cache_tail` for the honest
        account of what a hot tail can and cannot see; the one-line version
        is that it gates the STAGE SHAPES and the mask's reach, and it must
        NOT move `residual2.out`."""
        var n = self.cap - first
        if n <= 0:
            raise Error("transformer: plant_slots with an empty range")
        var need = 2 * self.b * self.n_kv * n * self.head_dim
        if len(values) != need:
            raise Error(
                String("transformer: plant_slots wanted ")
                + String(need)
                + " values and got "
                + String(len(values))
            )
        var w = 0
        for bb in range(self.b):
            for kv in range(self.n_kv):
                for j in range(first, self.cap):
                    for d in range(self.head_dim):
                        self.k[self.slot(bb, kv, j, d)] = values[w]
                        w += 1
        for bb in range(self.b):
            for kv in range(self.n_kv):
                for j in range(first, self.cap):
                    for d in range(self.head_dim):
                        self.v[self.slot(bb, kv, j, d)] = values[w]
                        w += 1


# ===========================================================================
# THE STAGES (contract section 9, DEVIATION 817)
# ===========================================================================

comptime TRANSFORMER_STAGE_COUNT = 30


def stage_tag(i: Int) raises -> String:
    """Contract section 9's thirty tags, in the contract's order.

    **THE STRINGS AND THE ORDER ARE BOTH PART OF THE INSTRUMENT.**
    `tools/identity_trace_diff.py` aligns two traces by their TAG SEQUENCES
    (`difflib.SequenceMatcher` over the tag lists) BEFORE it compares a
    single hash, so a renamed tag or a reordered pair does not produce a
    smaller diff, it produces a WRONG ALIGNMENT that pairs one run's stage
    against another run's different stage and reports a plausible answer.
    `core/identity_trace.mojo`'s own header calls that the worst thing the
    instrument can do.

    A device transcription must emit exactly these, in exactly this order,
    with its own driver prefix. Nothing may be inserted, nothing may be
    skipped conditionally, and a stage that is empty for a given shape must
    still be recorded (there is no such stage in this profile; the sentence
    is here so nobody invents one)."""
    if i == 0:
        return String("input.x")
    if i == 1:
        return String("norm1.sumsq")
    if i == 2:
        return String("norm1.out")
    if i == 3:
        return String("q_proj.out")
    if i == 4:
        return String("k_proj.out")
    if i == 5:
        return String("v_proj.out")
    if i == 6:
        return String("rope.inv_freq")
    if i == 7:
        return String("rope.cos")
    if i == 8:
        return String("rope.sin")
    if i == 9:
        return String("q_rope.out")
    if i == 10:
        return String("k_rope.out")
    if i == 11:
        return String("kv.k_cache")
    if i == 12:
        return String("kv.v_cache")
    if i == 13:
        return String("attn.scores")
    if i == 14:
        return String("attn.masked")
    if i == 15:
        return String("attn.max")
    if i == 16:
        return String("attn.exp")
    if i == 17:
        return String("attn.denom")
    if i == 18:
        return String("attn.weights")
    if i == 19:
        return String("attn.ctx")
    if i == 20:
        return String("o_proj.out")
    if i == 21:
        return String("residual1.out")
    if i == 22:
        return String("norm2.sumsq")
    if i == 23:
        return String("norm2.out")
    if i == 24:
        return String("gate_proj.out")
    if i == 25:
        return String("up_proj.out")
    if i == 26:
        return String("silu.out")
    if i == 27:
        return String("mlp.gated")
    if i == 28:
        return String("down_proj.out")
    if i == 29:
        return String("residual2.out")
    raise Error(
        String("transformer: no stage ")
        + String(i)
        + " (there are "
        + String(TRANSFORMER_STAGE_COUNT)
        + ", contract section 9)"
    )


struct TransformerStages(Movable):
    """Every recorded stage of one block call, in the card's order.

    Layouts, with `M = B * L` token-major and `S` the kv length AFTER the
    append. These are contract section 9's shapes verbatim.

        input_x        [M, d_model]
        norm1_sumsq    [M]
        norm1_out      [M, d_model]
        q_proj_out     [M, n_heads*head_dim]
        k_proj_out     [M, n_kv*head_dim]
        v_proj_out     [M, n_kv*head_dim]
        rope_inv_freq  [head_dim/2]
        rope_cos       [P_max, head_dim/2]
        rope_sin       [P_max, head_dim/2]
        q_rope_out     [M, n_heads*head_dim]
        k_rope_out     [M, n_kv*head_dim]
        kv_k_cache     [B, n_kv, S, head_dim]
        kv_v_cache     [B, n_kv, S, head_dim]
        attn_scores    [B, n_heads, L, S]
        attn_masked    [B, n_heads, L, S]
        attn_max       [B, n_heads, L]
        attn_exp       [B, n_heads, L, S]
        attn_denom     [B, n_heads, L]
        attn_weights   [B, n_heads, L, S]
        attn_ctx       [M, n_heads*head_dim]
        o_proj_out     [M, d_model]
        residual1_out  [M, d_model]
        norm2_sumsq    [M]
        norm2_out      [M, d_model]
        gate_proj_out  [M, intermediate]
        up_proj_out    [M, intermediate]
        silu_out       [M, intermediate]
        mlp_gated      [M, intermediate]
        down_proj_out  [M, d_model]
        residual2_out  [M, d_model]

    `kv_k_cache` and `kv_v_cache` hold the USED prefix `[0, S)` and NOT the
    allocation. `core/identity_trace.mojo`'s rule 3 is the reason and it is
    not a style point: hashing a machine-sized scratch reports a difference
    where there is none and buries the one that matters.

    **Six materialized `[B, n_heads, L, S]` buffers is the cost of the eager
    path and it is paid on purpose.** Contract section 6: the same shape
    MAX's own reference kernel allocates (`mha_gpu_naive`,
    mha.mojo:6328-6501), and it is what makes `attn.scores`, `attn.masked`,
    `attn.max`, `attn.exp`, `attn.denom` and `attn.weights` separately
    recordable. A lane whose only instrument is the per-stage card should
    not begin by fusing the stages away."""

    var input_x: List[Float32]
    var norm1_sumsq: List[Float32]
    var norm1_out: List[Float32]
    var q_proj_out: List[Float32]
    var k_proj_out: List[Float32]
    var v_proj_out: List[Float32]
    var rope_inv_freq: List[Float32]
    var rope_cos: List[Float32]
    var rope_sin: List[Float32]
    var q_rope_out: List[Float32]
    var k_rope_out: List[Float32]
    var kv_k_cache: List[Float32]
    var kv_v_cache: List[Float32]
    var attn_scores: List[Float32]
    var attn_masked: List[Float32]
    var attn_max: List[Float32]
    var attn_exp: List[Float32]
    var attn_denom: List[Float32]
    var attn_weights: List[Float32]
    var attn_ctx: List[Float32]
    var o_proj_out: List[Float32]
    var residual1_out: List[Float32]
    var norm2_sumsq: List[Float32]
    var norm2_out: List[Float32]
    var gate_proj_out: List[Float32]
    var up_proj_out: List[Float32]
    var silu_out: List[Float32]
    var mlp_gated: List[Float32]
    var down_proj_out: List[Float32]
    var residual2_out: List[Float32]

    def __init__(out self):
        self.input_x = List[Float32]()
        self.norm1_sumsq = List[Float32]()
        self.norm1_out = List[Float32]()
        self.q_proj_out = List[Float32]()
        self.k_proj_out = List[Float32]()
        self.v_proj_out = List[Float32]()
        self.rope_inv_freq = List[Float32]()
        self.rope_cos = List[Float32]()
        self.rope_sin = List[Float32]()
        self.q_rope_out = List[Float32]()
        self.k_rope_out = List[Float32]()
        self.kv_k_cache = List[Float32]()
        self.kv_v_cache = List[Float32]()
        self.attn_scores = List[Float32]()
        self.attn_masked = List[Float32]()
        self.attn_max = List[Float32]()
        self.attn_exp = List[Float32]()
        self.attn_denom = List[Float32]()
        self.attn_weights = List[Float32]()
        self.attn_ctx = List[Float32]()
        self.o_proj_out = List[Float32]()
        self.residual1_out = List[Float32]()
        self.norm2_sumsq = List[Float32]()
        self.norm2_out = List[Float32]()
        self.gate_proj_out = List[Float32]()
        self.up_proj_out = List[Float32]()
        self.silu_out = List[Float32]()
        self.mlp_gated = List[Float32]()
        self.down_proj_out = List[Float32]()
        self.residual2_out = List[Float32]()


def oracle_dump(st: TransformerStages) -> List[List[Float32]]:
    """The oracle's thirty stages in CARD ORDER, index-aligned with
    `stage_tag`.

    One function pairs tags with values so that the pairing exists in ONE
    place. `mamba_check.mojo:638`'s `oracle_dump` is the same shape for the
    same reason. A device dump must produce a list of the same length in the
    same order, and the check file's first assertion should be that the two
    lengths agree, before any hash is compared."""
    var out = List[List[Float32]]()
    out.append(st.input_x.copy())
    out.append(st.norm1_sumsq.copy())
    out.append(st.norm1_out.copy())
    out.append(st.q_proj_out.copy())
    out.append(st.k_proj_out.copy())
    out.append(st.v_proj_out.copy())
    out.append(st.rope_inv_freq.copy())
    out.append(st.rope_cos.copy())
    out.append(st.rope_sin.copy())
    out.append(st.q_rope_out.copy())
    out.append(st.k_rope_out.copy())
    out.append(st.kv_k_cache.copy())
    out.append(st.kv_v_cache.copy())
    out.append(st.attn_scores.copy())
    out.append(st.attn_masked.copy())
    out.append(st.attn_max.copy())
    out.append(st.attn_exp.copy())
    out.append(st.attn_denom.copy())
    out.append(st.attn_weights.copy())
    out.append(st.attn_ctx.copy())
    out.append(st.o_proj_out.copy())
    out.append(st.residual1_out.copy())
    out.append(st.norm2_sumsq.copy())
    out.append(st.norm2_out.copy())
    out.append(st.gate_proj_out.copy())
    out.append(st.up_proj_out.copy())
    out.append(st.silu_out.copy())
    out.append(st.mlp_gated.copy())
    out.append(st.down_proj_out.copy())
    out.append(st.residual2_out.copy())
    return out^


def record_transformer_card(
    mut trace: IdentityTrace, prefix: String, st: TransformerStages
) raises:
    """Write all thirty stages to an identity trace, in card order.

    DEVIATION 1003: this function is why `transformer_oracle.mojo` imports
    `core/identity_trace.mojo`, which imports `max.gpu.host` for
    `DeviceBuffer` and `DeviceContext`. A host-only oracle acquiring a GPU
    dependency is a real cost and the alternative was to put the recorder in
    `transformer_check.mojo`. It is here because the brief for this lane
    requires the ORACLE to record the card, and because putting the recorder
    beside `oracle_dump` and `stage_tag` keeps the tag-to-value pairing in
    one file instead of two. The cost if wrong is that the oracle cannot be
    compiled on a box with no MAX GPU package; `mamba_check.mojo` imports
    the same module and runs host-only in phases 1-2, so this is believed
    fine and has NOT been verified, because this lane compiles nothing.

    DEVIATION 1004: the three rope stages are recorded on EVERY call, where
    contract section 9 says they are "recorded once per configuration rather
    than per call". The contract's intent is preserved (the table is built
    once by `build_rope_table` and these are the same bits every time) and
    the departure is in the card's shape. The reason is the tag-alignment
    invariant above: a card that carries thirty tags after a prefill and
    twenty-seven after a decode step gives `identity_trace_diff.py` two
    different tag sequences to align, and the alignment it produces is
    plausible and wrong. Three redundant records per call, each of at most
    `P_max * head_dim/2` floats, is the price. If a future driver wants them
    once, it must drop them from EVERY call and record them separately under
    their own prefix, not conditionally.

    `prefix` becomes `prefix + "." + tag`, or the bare tag when empty. Tags
    must be UNIQUE WITHIN A TRACE (`core/identity_trace.mojo` enforces it and
    raises), so a driver making a prefill call and then a decode call MUST
    give them different prefixes. This function does not check that, because
    the trace already does and its error message is better."""
    # `[[mojo-len-string]]`: `len(String)` is not supported in this
    # toolchain; `byte_length()` is the spelling.
    var values = oracle_dump(st)
    for i in range(TRANSFORMER_STAGE_COUNT):
        var tag = stage_tag(i)
        if prefix.byte_length() > 0:
            tag = prefix + String(".") + tag
        # A LOCAL COPY rather than `values[i]` at the call site. One host
        # copy of a stage on a path that already drains the queue for every
        # record is not a cost, and it keeps a mutable-argument borrow off a
        # list element, which is the shape that produced the origin
        # unification errors this repository has been bitten by.
        var one = values[i].copy()
        trace.record_list_f32(tag, one)
        _ = one^


# ===========================================================================
# THE SEAMS
# ===========================================================================


@always_inline
def _hp(values: List[Float32]) -> HostF32Ptr:
    """A raw pointer to a list's cells for the task closures below (lane
    neural-pass21): the list outlives every region that reads it."""
    return HostF32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


@always_inline
def _row_split(rows: Int, work_per_row: Int) -> Tuple[Int, Int]:
    """(tasks, rows per task) for a stage whose rows are independent (lane
    neural-pass21): `host_row_tasks`' count, a schedule knob that moves no
    bit; every row's statements run unchanged in whichever task owns it."""
    var tasks = host_row_tasks(rows, work_per_row)
    return (tasks, (rows + tasks - 1) // tasks)


def _span_par(n: Int, work_per_cell: Int) -> Tuple[Int, Int]:
    """(tasks, cells per task) for an elementwise span: the per-cell lane
    statement is the same whichever task runs the cell."""
    var tasks = host_row_tasks(n, work_per_cell)
    return (tasks, (n + tasks - 1) // tasks)


def _span_add_par(a: List[Float32], b: List[Float32], n: Int, mut dst: List[Float32]):
    """`span_add` over blocks of cells on host tasks (lane neural-pass21):
    the same per-cell statement whichever block a cell falls in."""
    var sp = _span_par(n, 1)
    var tasks = sp[0]
    var chunk = sp[1]
    def _blocks(t: Int) {imm a, imm b, mut dst, imm n, imm chunk}:
        var c0 = t * chunk
        var c1 = min((t + 1) * chunk, n)
        if c1 > c0:
            span_add(a, c0, b, c0, c1 - c0, dst, c0)
    if tasks <= 1:
        _blocks(0)
    else:
        host_parallelize(_blocks, tasks)


def _span_mul_par(a: List[Float32], b: List[Float32], n: Int, mut dst: List[Float32]):
    """`span_mul` over blocks of cells on host tasks (lane neural-pass21)."""
    var sp = _span_par(n, 1)
    var tasks = sp[0]
    var chunk = sp[1]
    def _blocks(t: Int) {imm a, imm b, mut dst, imm n, imm chunk}:
        var c0 = t * chunk
        var c1 = min((t + 1) * chunk, n)
        if c1 > c0:
            span_mul(a, c0, b, c0, c1 - c0, dst, c0)
    if tasks <= 1:
        _blocks(0)
    else:
        host_parallelize(_blocks, tasks)


def _span_silu_par(a: List[Float32], n: Int, mut dst: List[Float32]):
    """`span_silu` over blocks of cells on host tasks (lane neural-pass21)."""
    var sp = _span_par(n, 8)
    var tasks = sp[0]
    var chunk = sp[1]
    def _blocks(t: Int) {imm a, mut dst, imm n, imm chunk}:
        var c0 = t * chunk
        var c1 = min((t + 1) * chunk, n)
        if c1 > c0:
            span_silu(a, c0, c1 - c0, dst, c0)
    if tasks <= 1:
        _blocks(0)
    else:
        host_parallelize(_blocks, tasks)


def rms_norm_into(
    src: List[Float32],
    wnorm: List[Float32],
    m: Int,
    dm: Int,
    mut sumsq: List[Float32],
    mut out: List[Float32],
):
    """Seams S1 through S4, `LlamaRMSNorm.forward` (modeling_llama.py:62-67).

        variance = hidden_states.pow(2).mean(-1)                      :65
        hidden   = hidden_states * torch.rsqrt(variance + eps)        :66
        return     self.weight * hidden                               :67

    pinned as

        S1  acc  = ftz(fma(x_j, x_j, acc)), serial ASCENDING j from +0.0,
            ONE FOLD PER ROW, no block fold and no tree.
        S2  mean = ftz(identical_div(acc, d_model))
            rstd = ftz(identical_rsqrt(ftz(mean + eps)))
        S3  inner = identical_mul(x_j, rstd)
        S4  out   = identical_mul(w_j, inner)

    Four things about this that are decisions rather than transcription.

    **S1's fold is SERIAL and the contract 12.1 table records that as a
    DEPARTURE from `archive/plans/IDENTICAL_GEMM_PLAN.md`'s sketch, which asked for a
    pinned TREE.** The serial chain is the mamba contract's S1 unchanged, it
    gives this block ONE fold shape instead of two (the same shape as S17's
    denominator), and it keeps the norm out of every launch-geometry
    argument. `core/pinned_reduce.mojo::pinned_block_sum` is the tree and
    this lane does not use it anywhere. Sabotage `S1_FOLD_DESCENDING` must
    move `norm1.sumsq` and nothing earlier.

    **S1 is FUSED and S3/S4 are not.** Contract section 4: FMA contraction
    is PER SEAM. The sum of squares is one rounding per term through `fma`;
    the two scalings are each one rounding through `pinned_mul`, which is
    `identical_mul_add(a, b, -0.0)` and which presents no syntactic multiply
    for a codegen to contract into a neighbouring add. The gemm README's F3
    scar is that `var p = a * b; p + c` WAS contracted across statements on
    this host, so a separate statement is not a barrier and the `-0.0`
    addend is.

    **The `-0.0` addend in `pinned_mul` is load-bearing at S3 and S4.** A
    `+0.0` addend would launder a negative-zero product into a positive one
    (the gemm lane's F6a lesson), and a whole `-0.0` token reaches this seam
    in fixture case `adv_signed_zeros`.

    **eps is `1e-6` and NOT the mamba lane's `1e-5`.** Contract section 3,
    `LlamaConfig.rms_norm_eps`, configuration_llama.py:73. See
    `transformer_fixture.mojo::RMS_EPS` for why that makes the mamba DEVICE
    kernel unusable here until DEVIATION 801 lifts eps to an argument, and
    why the HOST oracle needs nothing from the mamba lane.

    `mean + eps` at :66 is a plain add and is spelled as one. It is not a
    seam the reference rounds twice and there is no product to contract it
    into."""
    # ROWS OVER HOST TASKS (lane neural-pass21): each row's two serial
    # folds and its scalings are the statements above, unchanged; the lists
    # are sized once and written by index.
    sumsq = host_f32_uninit(m)
    out = host_f32_uninit(m * dm)
    var sp = _hp(sumsq)
    var op = _hp(out)
    var split = _row_split(m, 4 * dm)
    var ntasks = split[0]
    var chunk = split[1]
    def _rows(task: Int) {imm src, imm wnorm, imm sp, imm op, imm m, imm dm, imm chunk}:
        for t in range(task * chunk, min((task + 1) * chunk, m)):
            var acc = Float32(0.0)
            for j in range(dm):
                var xj = ftz(src[t * dm + j])
                acc = ftz(identical_mul_add(xj, xj, acc))
            sp.unsafe_store(t, acc)
            var mean = ftz(identical_div(acc, Float32(dm)))
            var rstd = ftz(identical_rsqrt(ftz(mean + RMS_EPS)))
            for j in range(dm):
                var inner = ftz(identical_mul(ftz(src[t * dm + j]), rstd))
                op.unsafe_store(t * dm + j, ftz(identical_mul(ftz(wnorm[j]), inner)))
    if ntasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, ntasks)


def norm_into(
    src: List[Float32],
    wnorm: List[Float32],
    bnorm: List[Float32],
    m: Int,
    dm: Int,
    eps: Float32,
    kind: Int,
    has_bias: Bool,
    mut sumsq: List[Float32],
    mut out: List[Float32],
) raises:
    """The block's normalization under the options (lane/block-options,
    2026-09-17). `rms_norm_into` above is UNTOUCHED and is the frozen
    profile's spelling; this function's `NORM_RMSNORM` arm is that
    arithmetic verbatim with `eps` an argument, so the default record
    reproduces its bits by construction, and the block oracle routes every
    call through here so that one function owns the norm's variants.

    DEVIATION 2937, `norm_eps` as a parameter: S2's `mean + eps` reads the
    record's eps (today's `RMS_EPS`, 1e-6, at the default). No other bit
    moves.

    DEVIATION 2936, `norm = layernorm` (+ `norm_bias`): `nn.LayerNorm`
    (`torch.nn.functional.layer_norm`, biased variance over the last axis)

        mean  = sum_j x_j / d                        plain adds, serial ascending from +0.0
        dev_j = x_j - mean
        var   = sum_j dev_j^2 / d                    fma chain, serial ascending from +0.0
        rstd  = rsqrt(var + eps)
        y_j   = w_j * (dev_j * rstd) + b_j

    pinned, per row: `acc = ftz(ftz(acc) + ftz(x_j))`; `mean =
    ftz(identical_div(acc, d))`; `dev_j = ftz(ftz(x_j) - mean)`; `acc2 =
    ftz(fma(dev_j, dev_j, acc2))`; the recorded `norm*.sumsq` stage IS
    `acc2` (the sum of squared deviations; the mean is an internal);
    `rstd = ftz(identical_rsqrt(ftz(var + eps)))` with `var =
    ftz(identical_div(acc2, d))`; `inner = identical_mul(dev_j, rstd)`;
    `y = identical_mul(w_j, inner)`; with a bias `y = ftz(ftz(y) + ftz(b_j))`,
    one plain add of two rounded values. Two serial folds per row, no
    tree, one thread per row on the device. The two-pass (mean, then
    deviations) form is chosen over `E[x^2] - E[x]^2` because the latter
    cancels catastrophically and no reference spells it.

    DEVIATION 2938, `norm = rmsnorm_offset`: `GemmaRMSNorm.forward`
    (modeling_gemma.py, `output * (1.0 + self.weight.float())`): S4's
    weight is `ftz(Float32(1.0) + ftz(w_j))`, one plain add, and the rest is
    S1-S4 unchanged. Gemma keeps the whole norm in float32, which is this
    profile's dtype anyway.
    """
    # ROWS OVER HOST TASKS (lane neural-pass21), both arms: each row's
    # serial folds and scalings are the statements above, unchanged; the
    # lists are sized once and written by index.
    sumsq = host_f32_uninit(m)
    out = host_f32_uninit(m * dm)
    var sp = _hp(sumsq)
    var op = _hp(out)
    var split = _row_split(m, 6 * dm)
    var ntasks = split[0]
    var chunk = split[1]
    if kind == NORM_LAYERNORM:
        def _ln_rows(task: Int) {imm src, imm wnorm, imm bnorm, imm sp, imm op, imm m, imm dm, imm eps, imm has_bias, imm chunk}:
            for t in range(task * chunk, min((task + 1) * chunk, m)):
                var acc = Float32(0.0)
                for j in range(dm):
                    acc = ftz(ftz(acc) + ftz(src[t * dm + j]))
                var mean = ftz(identical_div(acc, Float32(dm)))
                var acc2 = Float32(0.0)
                for j in range(dm):
                    var dev = ftz(ftz(src[t * dm + j]) - mean)
                    acc2 = ftz(identical_mul_add(dev, dev, acc2))
                sp.unsafe_store(t, acc2)
                var variance = ftz(identical_div(acc2, Float32(dm)))
                var rstd = ftz(identical_rsqrt(ftz(variance + eps)))
                for j in range(dm):
                    var dev = ftz(ftz(src[t * dm + j]) - mean)
                    var inner = ftz(identical_mul(dev, rstd))
                    var y = ftz(identical_mul(ftz(wnorm[j]), inner))
                    if has_bias:
                        y = ftz(ftz(y) + ftz(bnorm[j]))
                    op.unsafe_store(t * dm + j, y)
        if ntasks <= 1:
            _ln_rows(0)
        else:
            host_parallelize(_ln_rows, ntasks)
        return
    if kind != NORM_RMSNORM and kind != NORM_RMSNORM_OFFSET:
        raise Error(
            String("transformer: norm kind ") + String(kind) + " is not spelled"
        )
    if has_bias:
        raise Error("transformer: an RMSNorm carries no bias (norm_bias needs layernorm)")
    def _rms_rows(task: Int) {imm src, imm wnorm, imm sp, imm op, imm m, imm dm, imm eps, imm kind, imm chunk}:
        for t in range(task * chunk, min((task + 1) * chunk, m)):
            var acc = Float32(0.0)
            for j in range(dm):
                var xj = ftz(src[t * dm + j])
                acc = ftz(identical_mul_add(xj, xj, acc))
            sp.unsafe_store(t, acc)
            var mean = ftz(identical_div(acc, Float32(dm)))
            var rstd = ftz(identical_rsqrt(ftz(mean + eps)))
            for j in range(dm):
                var inner = ftz(identical_mul(ftz(src[t * dm + j]), rstd))
                var wj = ftz(wnorm[j])
                if kind == NORM_RMSNORM_OFFSET:
                    wj = ftz(Float32(1.0) + wj)
                op.unsafe_store(t * dm + j, ftz(identical_mul(wj, inner)))
    if ntasks <= 1:
        _rms_rows(0)
    else:
        host_parallelize(_rms_rows, ntasks)


def add_bias_into(mut buf: List[Float32], bias: List[Float32], rows: Int, width: Int) raises:
    """DEVIATIONS 2934, 2935 and 2945: a projection's bias, `nn.Linear`'s
    `y = x @ W^T + b`, spelled as the profile's GEMM cell followed by ONE
    plain add per output cell, `ftz(ftz(y) + ftz(b_col))`. The reference's
    `addmm` may fold the bias into its accumulator's seed or its epilogue,
    which is cuBLAS's decision and not a spelling this profile can pin; the
    one-add-after form is OURS, and every side spells it. In place."""
    if len(bias) != width:
        raise Error(
            String("transformer: a bias of ")
            + String(len(bias))
            + " elements cannot be added across "
            + String(width)
            + " columns"
        )
    # ROWS OVER HOST TASKS (lane neural-pass21): the same add per cell.
    var bp = _hp(buf)
    var split = _row_split(rows, width)
    var ntasks = split[0]
    var chunk = split[1]
    def _rows(task: Int) {imm bp, imm bias, imm rows, imm width, imm chunk}:
        for t in range(task * chunk, min((task + 1) * chunk, rows)):
            for j in range(width):
                var i = t * width + j
                bp.unsafe_store(i, ftz(ftz(bp.unsafe_load(i)) + ftz(bias[j])))
    if ntasks <= 1:
        _rows(0)
    else:
        host_parallelize(_rows, ntasks)


def apply_rope_into(
    src: List[Float32],
    n_head: Int,
    head_dim: Int,
    b: Int,
    l: Int,
    pos0: Int,
    rope: RopeTable,
    mut out: List[Float32],
) raises:
    """Seams S9 and S10, `apply_rotary_pos_emb` (modeling_llama.py:137-160)
    with `rotate_half` (:130-134).

        rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])             :130-134
        q_embed = (q * cos) + (rotate_half(q) * sin)                  :158

    pinned as, per element j of one head vector at absolute position p,

        ci   = j if j < d/2 else j - d/2      the DEDUPLICATED table column
        rot  = -a[j + d/2] if j < d/2 else a[j - d/2]
        S9a  = identical_mul(a[j], cos[p, ci])
        S9b  = identical_mul(rot,  sin[p, ci])
        S10  = ftz(ftz(S9a) + ftz(S9b))       **UNFUSED**

    **S10 IS UNFUSED AND THAT IS THE ONE DECISION HERE.** DEVIATION 811. The
    reference computes two tensors and adds them, so it rounds THREE times:
    once per product and once for the add. An `fma` here rounds ONCE, and an
    fma is the natural thing for a kernel author to write when they see
    `a*c + b*s`. The two answers differ in the last bit on ordinary inputs.
    Sabotage `S10_ROPE_FUSED` must move `q_rope.out` and nothing earlier.

    **The pairing is `j` with `j + head_dim/2`, the HALVES, not adjacent
    even/odd elements.** This is the single most commonly mistransribed
    thing about RoPE, because the ORIGINAL RoFormer paper and several
    kernels pair `(2i, 2i+1)` and HuggingFace's checkpoint layout permutes
    the head dimension so that the halves spelling gives the same rotation.
    MAX makes the same choice and names it
    (`max/kernels/src/nn/rope.mojo::get_safetensors_idx`, :51-53, "the
    rotate-half pairing"). Sabotage `S09_ROPE_HALVES_SWAPPED` must move
    `q_rope.out`.

    **The negation is exact and is NOT a seam.** `-x` flips a sign bit; it
    rounds nothing. It is applied BEFORE the product, as the reference does
    (`cat((-x2, x1))` then multiply), and the order matters at exactly one
    input: `-(-0.0)` is `+0.0`, so a negated negative zero multiplied by a
    sine of either sign gives a different zero sign than a positive zero
    would. Doing the negation after the product would give the same
    magnitude and the same sign in every case, but writing it in the
    reference's order costs nothing and removes the question.

    **The position is ABSOLUTE, always** (contract 7.2). `pos0 + li`, where
    `pos0` is the KV cache's `used` count before this call's append. An
    implementation that indexes the table by the local index `li` agrees
    with this one on every single-call fixture and disagrees on every decode
    step, which is sabotage `S07_ROPE_RELATIVE_POSITION` and which is why
    contract section 10 says it will look inert and be deleted if clause (d)
    is written late."""
    # DEVIATION 2948 (lane/block-options): the PARTIAL rotary. Only the
    # first `rope.rope_dim` columns of a head are rotated; the rest pass
    # through, FLUSHED ON LOAD like every other operand (DEVIATION 1026's
    # per-load rule; a pass-through is a copy and not a seam, and the flush
    # is what every consumer of the column would apply anyway). `rope_dim
    # == head_dim` is today's full rotation and takes the `j < rd` branch
    # for every column. GPT-NeoX's `rotary_ndims = head_size *
    # rotary_pct` (modeling_gpt_neox.py) is the reference shape.
    var rd = rope.rope_dim
    var half = rope.half()
    var width = n_head * head_dim
    var m = b * l
    if rd > head_dim or rd <= 0 or rd % 2 != 0:
        raise Error(
            String("transformer: the rotary table's rope_dim ")
            + String(rd)
            + " does not fit head_dim "
            + String(head_dim)
            + " REFUSED"
        )
    for t in range(m):
        var p = pos0 + t % l
        if p < 0 or p >= rope.positions:
            raise Error(
                String("transformer: absolute position ")
                + String(p)
                + " outside the rotary table of "
                + String(rope.positions)
                + " positions REFUSED (DEVIATION 1002: the table is a"
                + " configuration quantity, so a call that overruns it is a"
                + " misconfiguration and not something to grow into)"
            )
    # TOKENS OVER HOST TASKS (lane neural-pass21): each token's cells are
    # the statements above, unchanged; the list is sized once and written
    # by index (the refusal above ran first, so no task raises).
    out = host_f32_uninit(m * width)
    var op = _hp(out)
    var split = _row_split(m, 6 * width)
    var ntasks = split[0]
    var chunk = split[1]
    def _tokens(task: Int) {imm src, imm rope, imm op, imm m, imm l, imm pos0, imm n_head, imm head_dim, imm width, imm rd, imm half, imm chunk}:
        for t in range(task * chunk, min((task + 1) * chunk, m)):
            var li = t % l
            var p = pos0 + li
            for h in range(n_head):
                var base = t * width + h * head_dim
                for j in range(head_dim):
                    if j >= rd:
                        op.unsafe_store(base + j, ftz(src[base + j]))
                        continue
                    var ci: Int
                    var rot: Float32
                    if j < half:
                        ci = j
                        rot = -ftz(src[base + j + half])
                    else:
                        ci = j - half
                        rot = ftz(src[base + j - half])
                    var c = ftz(rope.cos[p * half + ci])
                    var s = ftz(rope.sin[p * half + ci])
                    var pa = ftz(identical_mul(ftz(src[base + j]), c))
                    var pb = ftz(identical_mul(rot, s))
                    op.unsafe_store(base + j, ftz(ftz(pa) + ftz(pb)))
    if ntasks <= 1:
        _tokens(0)
    else:
        host_parallelize(_tokens, ntasks)


@always_inline
def _visible_keys(own0: Int, window: Int, qi: Int, s: Int) -> Tuple[Int, Int]:
    """The keys query `qi` attends to, `[jlo, jhi]` inclusive, the mask's
    own rule read back: key `j` is visible when `key_lo + j <= p` and, under
    a window, `key_lo + j > p - window`, with `p = pos0 + qi` and
    `own0 = pos0 - key_lo`. `own0 < 0` means every key (the mask-free
    callers). The range is never empty: key `own0 + qi` is always visible."""
    if own0 < 0:
        return (0, s - 1)
    var jhi = own0 + qi
    if jhi > s - 1:
        jhi = s - 1
    var jlo = 0
    if window > 0:
        jlo = own0 + qi - window + 1
        if jlo < 0:
            jlo = 0
    return (jlo, jhi)


@always_inline
def span_add_scalar(src: List[Float32], sb: Int, n: Int, fill: Float32, mut dst: List[Float32], db: Int):
    """`dst[db + j] = ftz(ftz(src[sb + j]) + fill)` for j < n, as lanes: one
    plain add per cell, the same rounding on every lane as the scalar."""
    if n <= 0:
        return
    var sp = src.unsafe_ptr()
    var dp = dst.unsafe_ptr()
    var fv = GhrF(fill)
    var j = 0
    while j + GHR_FW <= n:
        dp.unsafe_store(db + j, ghr_ftz_lanes(ghr_ftz_lanes(sp.unsafe_load[width=GHR_FW](sb + j)) + fv))
        j += GHR_FW
    while j < n:
        dp.unsafe_store(db + j, ftz(ftz(sp.unsafe_load(sb + j)) + fill))
        j += 1


#: Lanes of `head_dim` the 64-wide value-sum arm keeps in flight at once.
comptime ATTN_VALUE_64 = 64 // GHR_FW


def attn_value_sum_lanes(
    weights: List[Float32],
    wbase: Int,
    values: List[Float32],
    vbase: Int,
    mut out: List[Float32],
    obase: Int,
    s: Int,
    hd: Int,
):
    """S19 for one query row: `out[obase + d]` is the chain
    `acc = ftz(identical_mul_add(ftz(w[wbase + j]), ftz(v[vbase + j*hd + d]), acc))`
    over j ascending from `+0.0`, for every d. The `hd` chains advance
    together, one SIMD lane per d (lane neural-cpu, 2026-09-28); every lane is
    its own output's chain, so the bits are the scalar walk's."""
    var wp = weights.unsafe_ptr()
    var vp = values.unsafe_ptr()
    var op = out.unsafe_ptr()
    comptime G = 4 * GHR_FW
    var d = 0
    # head_dim 64 (the byte LM's, the board's): every lane of the head in
    # flight in ONE pass over the keys (lane neural-pass10) instead of four
    # passes of four vectors; each lane's chain is the same statements in
    # the same order.
    if hd == 64:
        var accs = InlineArray[GhrF, ATTN_VALUE_64](fill=GhrF(0.0))
        for j in range(s):
            var wv = GhrF(ftz(wp.unsafe_load(wbase + j)))
            var row = vbase + j * hd
            comptime for v in range(ATTN_VALUE_64):
                accs[v] = ghr_ftz_lanes(identical_mul_add_simd[GHR_FW](
                    wv, ghr_ftz_lanes(vp.unsafe_load[width=GHR_FW](row + v * GHR_FW)), accs[v]))
        comptime for v in range(ATTN_VALUE_64):
            op.unsafe_store(obase + v * GHR_FW, accs[v])
        return
    while d + G <= hd:
        var a0 = GhrF(0.0)
        var a1 = GhrF(0.0)
        var a2 = GhrF(0.0)
        var a3 = GhrF(0.0)
        for j in range(s):
            var wv = GhrF(ftz(wp.unsafe_load(wbase + j)))
            var row = vbase + j * hd + d
            a0 = ghr_ftz_lanes(identical_mul_add_simd[GHR_FW](
                wv, ghr_ftz_lanes(vp.unsafe_load[width=GHR_FW](row)), a0))
            a1 = ghr_ftz_lanes(identical_mul_add_simd[GHR_FW](
                wv, ghr_ftz_lanes(vp.unsafe_load[width=GHR_FW](row + GHR_FW)), a1))
            a2 = ghr_ftz_lanes(identical_mul_add_simd[GHR_FW](
                wv, ghr_ftz_lanes(vp.unsafe_load[width=GHR_FW](row + 2 * GHR_FW)), a2))
            a3 = ghr_ftz_lanes(identical_mul_add_simd[GHR_FW](
                wv, ghr_ftz_lanes(vp.unsafe_load[width=GHR_FW](row + 3 * GHR_FW)), a3))
        op.unsafe_store(obase + d, a0)
        op.unsafe_store(obase + d + GHR_FW, a1)
        op.unsafe_store(obase + d + 2 * GHR_FW, a2)
        op.unsafe_store(obase + d + 3 * GHR_FW, a3)
        d += G
    while d + GHR_FW <= hd:
        var a = GhrF(0.0)
        for j in range(s):
            a = ghr_ftz_lanes(identical_mul_add_simd[GHR_FW](
                GhrF(ftz(wp.unsafe_load(wbase + j))),
                ghr_ftz_lanes(vp.unsafe_load[width=GHR_FW](vbase + j * hd + d)), a))
        op.unsafe_store(obase + d, a)
        d += GHR_FW
    while d < hd:
        var acc = Float32(0.0)
        for j in range(s):
            acc = ftz(identical_mul_add(
                ftz(wp.unsafe_load(wbase + j)), ftz(vp.unsafe_load(vbase + j * hd + d)), acc))
        op.unsafe_store(obase + d, acc)
        d += 1


def _set_at(mut xs: List[Float32], i: Int, v: Float32, size: Int):
    """Write-by-index into a stage list, growing it to `size` zeros first.

    `mamba_oracle.mojo:391`'s helper verbatim. Needed wherever the loop
    order that is natural for the ARITHMETIC differs from the LAYOUT the
    card stage requires; `attn.ctx` is the only such stage here, because it
    is token-major `[M, n_heads*head_dim]` while S19's fold is naturally
    walked `(batch, head, query, depth)`."""
    while len(xs) < size:
        xs.append(Float32(0.0))
    xs[i] = v


#: lane/neural-pass75 (2026-10-01): the host forward's attention ran about
#: twenty parallel regions per block (per head: the q/k gather, the scores
#: product, the scale; then the mask, the softmax and the value sum over all
#: rows), each writing a full [heads x l x s] array that the next one read
#: back. It is now one region over (batch, head, query) rows: a task takes
#: its rows through the scores chain (`ghr_tile` on K panels packed once
#: per (batch, kv head): the leaf, chain and fold `gemm_host_rows` ran), the
#: scale, the mask, the softmax and the value sum, the same per-element
#: statements in the same order, and the stage arrays are still written.
#: The plants are applied row by row at their stage (the refusal of a bad
#: index comes first, serial). The int15 profile and a softcap keep the
#: staged code. `-D MOJOLEARN_TRANSFORMER_HOST_STAGED_ATTENTION=1` restores
#: the staged code everywhere.
comptime ATTN_HOST_FUSED = not is_defined["MOJOLEARN_TRANSFORMER_HOST_STAGED_ATTENTION"]()


def _plant_refuse(n: Int, plant: ScorePlant, at: Int) raises:
    """`_apply_plant`'s index refusal alone, for a plant applied row by row."""
    if plant.at != at:
        return
    for i in range(len(plant.idx)):
        var j = plant.idx[i]
        if j < 0 or j >= n:
            raise Error(
                String("transformer: score plant index ")
                + String(j)
                + " outside a buffer of "
                + String(n)
                + " REFUSED (a plant that misses is worse than no plant:"
                + " the gate goes green and proves nothing)"
            )


def _apply_plant_row(
    mut buf: List[Float32], idx: List[Int], bits: List[UInt32], pat: Int, at: Int, base: Int, n: Int
):
    """`_apply_plant` for the cells `[base, base + n)` of `buf` alone."""
    if pat != at:
        return
    for i in range(len(idx)):
        var j = idx[i]
        if j >= base and j < base + n:
            buf[j] = f32_from_bits(bits[i])


def _apply_plant(
    mut buf: List[Float32], plant: ScorePlant, at: Int
) raises:
    """DEVIATION 1001's application. A no-op unless the plant names this
    point and carries indices.

    Argued in `transformer_fixture.mojo`'s plant section; the summary is
    that contract 4.1(a) requires a `-0.0` score, no assignment of finite
    weights produces one, and the contract's own sentence is "the fixture
    has to plant it by bits"."""
    if plant.at != at:
        return
    for i in range(len(plant.idx)):
        var j = plant.idx[i]
        if j < 0 or j >= len(buf):
            raise Error(
                String("transformer: score plant index ")
                + String(j)
                + " outside a buffer of "
                + String(len(buf))
                + " REFUSED (a plant that misses is worse than no plant:"
                + " the gate goes green and proves nothing)"
            )
        buf[j] = f32_from_bits(plant.bits[i])


# ===========================================================================
# THE BLOCK
# ===========================================================================


def _oracle_product(
    a: List[Float32], b: List[Float32], m: Int, n: Int, k: Int, int15: Bool
) raises -> List[Float32]:
    """`C = A . B^T` (OP_NT). Under the default, gemm.fp32.v1's host rows,
    exactly the call that stood at every site before lane/lowbit-blocks.
    With `int15` (`numeric_profile="fixed15_v1"`), gemm.int15i64.v1's host
    oracle: both operands quantized row by row along `k` by
    `quantize_rows_int15` (clause W-9; a weight's codes are a function of
    its own rows, so quantizing it here per call gives the codes a caller
    packs once), then the exact integer sum. The block's P.V (S19) is not a
    call of this function and stays fp32 under both."""
    if int15:
        return gemm_int15_from_f32_oracle(a, b, m, n, k)
    return gemm_host_rows(a, b, OP_NT, m, n, k)


def transformer_block_oracle(
    w: TransformerWeights,
    x: List[Float32],
    b: Int,
    l: Int,
    mut cache: TransformerKVCache,
    rope: RopeTable,
    plant: ScorePlant,
    int15: Bool = False,
) raises -> TransformerStages:
    """One `LlamaDecoderLayer.forward` (modeling_llama.py:295-324), stage by
    stage, through the seams.

        residual  = x
        h         = residual + self_attn(rmsnorm(x, w_norm1))         :305-317
        residual2 = h
        out       = residual2 + mlp(rmsnorm(h, w_norm2))              :320-323

    Prefill is a fresh zero `TransformerKVCache`. The decode step is THIS
    SAME FUNCTION at `l == 1` carrying the cache. ONE spelling for both,
    which is what makes contract clause (d) a theorem the gate then verifies
    rather than a coincidence it hopes for. Contract section 7.2's structural
    argument in three lines:

      - RoPE reads the table at the ABSOLUTE position, identical in both
        paths.
      - S11 contracts over `head_dim`, whose LENGTH is the same in both
        paths, which is precisely why it may be a gemm v1 call.
      - S17 and S19 contract over the key axis, whose length is `t + 1` in
        decode and `L` in prefill, and the extra terms are exactly `+0.0`
        (section 7.1) and bitwise inert in a serial ascending chain seeded
        `+0.0`. **Under gemm v1's leaf-and-tree topology they would NOT be
        inert**, because `P` is a pure function of `k` and `k` differs. That
        sentence is DEVIATION 807 and it is why S19 is hand-written here
        while S11 is a GEMM call. The asymmetry looks like an inconsistency
        and is the opposite of one.

    Inference only. No bias anywhere (contract section 2 REFUSES a nonzero
    bias rather than specifying where it would round), no dropout, no
    training, no backward.

    The eager attention path and nothing else. FlashAttention is an ONLINE
    softmax whose rescale count is the KV TILE COUNT, which is an
    execution-plan quantity, which is IDENTITY_PATHS rows 3 and 7 moved
    inside the softmax where a contract cannot reach it by pinning a fold
    topology; SDPA picks its backend at run time. Contract section 6 is the
    argument and it is the most important scoping decision in the
    document."""
    var dims = w.dims.copy()
    dims.validate()
    var dm = dims.d_model
    var nh = dims.n_heads
    var nkv = dims.n_kv_heads
    var hd = dims.head_dim
    var qw = dims.q_width()
    var kw = dims.kv_width()
    var inter = dims.intermediate
    var n_rep = dims.n_rep()
    var m = b * l
    var st = TransformerStages()

    # ---- refusals, before ANY recorded stage (contract section 8) --------
    if l <= 0 or b <= 0:
        raise Error("transformer: B and L must be positive")
    if len(x) != m * dm:
        raise Error(
            String("transformer: x has ")
            + String(len(x))
            + " elements and the shape [B, L, d_model] wants "
            + String(m * dm)
        )
    if b != cache.b or nkv != cache.n_kv or hd != cache.head_dim:
        raise Error("transformer: the KV cache does not match the config")
    if rope.head_dim != hd:
        raise Error("transformer: the rotary table does not match head_dim")
    # lane/block-options: the record every option below reads.
    var opts = w.opts.copy()
    opts.validate(hd)
    if rope.rope_dim != opts.rope_dim_of(hd):
        raise Error(
            String("transformer: the rotary table rotates ")
            + String(rope.rope_dim)
            + " columns but the options say rope_dim "
            + String(opts.rope_dim_of(hd))
            + " REFUSED"
        )
    refuse_nonfinite("x", x)
    refuse_bad_weights(w)
    # THE WHOLE ALLOCATION, not just the used prefix. Stricter than the
    # contract asks for (section 8 covers "any input or weight" and the
    # unused tail is neither), and deliberately so: the hot-tail fixture
    # writes real bits past `used`, and a cache whose tail holds a NaN would
    # be a card that cannot be compared the moment any implementation reads
    # one slot too far. Cheap, and it turns a class of confusing failure
    # into a named refusal.
    refuse_nonfinite("k_cache", cache.k)
    refuse_nonfinite("v_cache", cache.v)

    # `pos0` IS THE ORIGIN OF EVERY POSITION AND EVERY KEY INDEX IN THIS
    # CALL, and it is read BEFORE the append. Contract section 5.5: one
    # axis, one direction, one origin.
    var pos0 = cache.used
    var s_abs = pos0 + l
    if s_abs > cache.cap:
        raise Error(
            String("transformer: this call would use ")
            + String(s_abs)
            + " KV slots and the cache holds "
            + String(cache.cap)
            + " REFUSED"
        )
    # DEVIATION 2933: the model's declared ceiling. The angle domain itself
    # was enforced when the table was built (`refuse_rope_angle_domain`),
    # and a position past the table is refused by `apply_rope_into`.
    if s_abs > opts.max_positions:
        raise Error(
            String("transformer: absolute position ")
            + String(s_abs - 1)
            + " is at or beyond the model's max_positions "
            + String(opts.max_positions)
            + " (DEVIATION 2933; DEVIATION 812's 8192 at the default)"
        )
    # THE KEY SPAN THIS CALL READS: absolute keys `[key_lo, pos0 + l)`.
    # Full causal has `key_lo == 0` and `s == pos0 + l` exactly as before;
    # a window `W` has `key_lo = max(0, pos0 - W + 1)`. Every fold below
    # walks the span ascending from `+0.0`, and the masked head and tail
    # of it are exactly `+0.0` (contract 7.1), so a query row's bits are a
    # pure function of (its position, its visible keys).
    var key_lo = cache.key_lo(pos0)
    var s = s_abs - key_lo
    var window = cache.window
    # This call's own first key index in the span (DEVIATION 807's `own0`
    # in the backward): query `qi` sits at key `own0 + qi`.
    var own0 = pos0 - key_lo

    st.input_x = x.copy()
    var hton = host_block_timing_on()
    var htk = Int(perf_counter_ns())

    # ---- S1-S4, input_layernorm (LDL:306, LRN:62-67) --------------------
    # LOCALS THEN ASSIGN, never `mut st.field` at a call site. A struct
    # field passed as a mutable argument is the shape that produces the
    # `MutUntrackedOrigin` vs `MutAnyOrigin` unification errors this
    # repository has lost time to, and this oracle cannot be compiled by its
    # author to find out. The cost is one move per stage.
    var n1_sumsq = List[Float32]()
    var n1_out = List[Float32]()
    # lane/block-options: `norm_into`'s RMSNorm arm is `rms_norm_into`
    # verbatim with eps an argument (DEVIATIONS 2936-2938 for the others).
    norm_into(
        x, w.norm1_w, w.norm1_b, m, dm, opts.norm_eps, opts.norm_kind,
        opts.norm_bias, n1_sumsq, n1_out,
    )
    st.norm1_sumsq = n1_sumsq^
    st.norm1_out = n1_out^
    host_tick(hton, htk, "fwd.norm1")

    # ---- S5, q/k/v (LlamaAttention.forward :250-252) ---------------------
    # `nn.Linear` stores weight [out, in] and applies y = x @ W^T, which is
    # a gemm v1 OP_NT cell with no transpose anywhere. `k = d_model = 32`
    # in every fixture here, which is <= CONTRACT_K_LEAF_MIN (128), so
    # P == 1 and the GEMM is its whole-K serial ascending chain -- the
    # balanced fold tree sits UNEXERCISED inside this block exactly as it
    # does in the mamba lane. Fixture `wide_inter300` is the answer: at
    # intermediate 300 the down_proj has k = 300, P = 3, a ragged 44-element
    # last leaf and one carry.
    #
    # Sabotage `S05_OP_NUMBERING` swaps OP_NT for another op constant. It
    # must move `q_proj.out` and it exists because the op numbering is three
    # bare integers (gemm_oracle.mojo:194-198) and a transposed read of a
    # square-ish weight produces a plausible number.
    st.q_proj_out = _oracle_product(st.norm1_out, w.w_q, m, qw, dm, int15)
    st.k_proj_out = _oracle_product(st.norm1_out, w.w_k, m, kw, dm, int15)
    st.v_proj_out = _oracle_product(st.norm1_out, w.w_v, m, kw, dm, int15)
    host_tick(hton, htk, "fwd.qkv_proj")
    # DEVIATION 2934, `qkv_bias` (Qwen2's `attention_bias=True`): one plain
    # add per cell AFTER the GEMM, recorded INTO the `*_proj.out` stages so
    # the card keeps its thirty tags.
    if opts.qkv_bias:
        var qb = st.q_proj_out.copy()
        add_bias_into(qb, w.b_q, m, qw)
        st.q_proj_out = qb^
        var kb = st.k_proj_out.copy()
        add_bias_into(kb, w.b_k, m, kw)
        st.k_proj_out = kb^
        var vb = st.v_proj_out.copy()
        add_bias_into(vb, w.b_v, m, kw)
        st.v_proj_out = vb^
    # DEVIATION 2946, `qk_norm` (Qwen3's `q_norm`/`k_norm`, Gemma3's): an
    # RMSNorm over EACH HEAD's `head_dim` vector of q and of k, BEFORE RoPE
    # (`Qwen3Attention.forward`: `query_states = self.q_norm(self.q_proj(
    # hidden_states).view(hidden_shape))` then `apply_rotary_pos_emb`). The
    # token-major `[M, n_h*head_dim]` layout already has each head's vector
    # contiguous, so it is `norm_into` over `M * n_h` rows of width
    # `head_dim` with the shared `[head_dim]` weight; the form (plain or
    # Gemma's offset) follows the block's norm, `qk_norm_kind`. The sum of
    # squares is an internal (no card stage); the normed q and k are
    # recorded INTO `q_proj.out` / `k_proj.out`.
    if opts.qk_norm:
        var qn_sumsq = List[Float32]()
        var qn_out = List[Float32]()
        norm_into(
            st.q_proj_out, w.qn_w, w.qn_w, m * nh, hd, opts.norm_eps,
            opts.qk_norm_kind(), False, qn_sumsq, qn_out,
        )
        st.q_proj_out = qn_out^
        var kn_sumsq = List[Float32]()
        var kn_out = List[Float32]()
        norm_into(
            st.k_proj_out, w.kn_w, w.kn_w, m * nkv, hd, opts.norm_eps,
            opts.qk_norm_kind(), False, kn_sumsq, kn_out,
        )
        st.k_proj_out = kn_out^

    # ---- S6-S8, the rotary table, computed once, recorded here ----------
    # DEVIATION 1004: recorded on every call. The values are `rope`'s and
    # this function does not recompute them, so two calls record the same
    # bits; what is repeated is the RECORD, not the arithmetic.
    st.rope_inv_freq = rope.inv_freq.copy()
    st.rope_cos = rope.cos.copy()
    st.rope_sin = rope.sin.copy()

    # ---- S9, S10, RoPE on q and k (LAR:137-160) --------------------------
    # The head reshape (`view(hidden_shape).transpose(1, 2)`, :250-252) is a
    # COPY and not a seam, and this oracle does not perform it at all: the
    # token-major `[M, n_heads*head_dim]` layout already has head `h`'s
    # vector contiguous at offset `h*head_dim`, so the "reshape" is address
    # arithmetic. RoPE is NOT applied to v (`apply_rotary_pos_emb` takes q
    # and k only, :158-159), which is easy to get wrong and impossible to
    # see in the output because a rotated v is still a plausible v.
    var q_rope = List[Float32]()
    var k_rope = List[Float32]()
    apply_rope_into(st.q_proj_out, nh, hd, b, l, pos0, rope, q_rope)
    apply_rope_into(st.k_proj_out, nkv, hd, b, l, pos0, rope, k_rope)
    st.q_rope_out = q_rope^
    st.k_rope_out = k_rope^
    host_tick(hton, htk, "fwd.rope")

    # ---- the key span, BEFORE the append. A COPY. -----------------------
    # The stage `kv.k_cache` is `[B, n_kv, S, head_dim]`: the keys this
    # call's attention reads, work index `j` at absolute `key_lo + j`.
    # Positions before `pos0` come from the cache, this call's own from
    # `k_rope_out` / `v_proj_out`. Gathered BEFORE the append because, under
    # a window, this call's tokens may evict positions it still attends to.
    st.kv_k_cache = host_f32_uninit(b * nkv * s * hd)
    st.kv_v_cache = host_f32_uninit(b * nkv * s * hd)
    # (batch, kv head, key) ROWS OVER HOST TASKS (lane neural-pass21): a
    # copy, each cell read from one place and written to one place.
    var kv_rows = b * nkv * s
    var kvc_k = _hp(st.kv_k_cache)
    var kvc_v = _hp(st.kv_v_cache)
    var k_rope_p = _hp(st.k_rope_out)
    var v_proj_p = _hp(st.v_proj_out)
    var kv_split = _row_split(kv_rows, 2 * hd)
    var kv_tasks = kv_split[0]
    var kv_chunk = kv_split[1]
    def _gather_rows(task: Int) {imm cache, imm kvc_k, imm kvc_v, imm k_rope_p, imm v_proj_p, imm kv_rows, imm nkv, imm s, imm hd, imm l, imm kw, imm key_lo, imm pos0, imm kv_chunk}:
        for r in range(task * kv_chunk, min((task + 1) * kv_chunk, kv_rows)):
            var j = r % s
            var kv = (r // s) % nkv
            var bb = r // (s * nkv)
            var pos = key_lo + j
            var dst = ((bb * nkv + kv) * s + j) * hd
            for d in range(hd):
                if pos < pos0:
                    var ix = cache.slot(bb, kv, pos, d)
                    kvc_k.unsafe_store(dst + d, cache.k[ix])
                    kvc_v.unsafe_store(dst + d, cache.v[ix])
                else:
                    var src = (bb * l + (pos - pos0)) * kw + kv * hd + d
                    kvc_k.unsafe_store(dst + d, k_rope_p.unsafe_load(src))
                    kvc_v.unsafe_store(dst + d, v_proj_p.unsafe_load(src))
    if kv_tasks <= 1:
        _gather_rows(0)
    else:
        host_parallelize(_gather_rows, kv_tasks)

    # ---- the KV append (:261-262 past_key_values.update). A COPY. --------
    # In position order, so under a ring the highest position wins a slot.
    # (kv head, d) COLUMNS OVER HOST TASKS (lane neural-pass21), the tokens
    # in position order inside each: a slot's cell (kv, d) is written by the
    # one task that owns that column, in the same t order as before, so under
    # a ring the highest position still wins it.
    var cols = nkv * hd
    var ck = _hp(cache.k)
    var cv = _hp(cache.v)
    var ap_split = _row_split(cols, 2 * m)
    var ap_tasks = ap_split[0]
    var ap_chunk = ap_split[1]
    def _append_cols(task: Int) {imm cache, imm ck, imm cv, imm k_rope_p, imm v_proj_p, imm cols, imm m, imm l, imm nkv, imm hd, imm kw, imm pos0, imm ap_chunk}:
        for col in range(task * ap_chunk, min((task + 1) * ap_chunk, cols)):
            var kv = col // hd
            var d = col % hd
            for t in range(m):
                var bb = t // l
                var li = t % l
                var src = t * kw + kv * hd + d
                var dst = cache.slot(bb, kv, pos0 + li, d)
                ck.unsafe_store(dst, k_rope_p.unsafe_load(src))
                cv.unsafe_store(dst, v_proj_p.unsafe_load(src))
    if ap_tasks <= 1:
        _append_cols(0)
    else:
        host_parallelize(_append_cols, ap_tasks)
    cache.used = s_abs
    host_tick(hton, htk, "fwd.kv_gather")

    # ---- S11, S12: the scores (EAF:204) ---------------------------------
    # `torch.matmul(query, key_states.transpose(2, 3)) * scaling`.
    #
    # S11 is a gemm v1 OP_NT cell with k = head_dim, ONE CALL PER
    # (batch, head), DEVIATION 808. `repeat_kv` (:179-188) is `expand` then
    # `reshape`, which is a COPY, so attention head `h` simply reads kv head
    # `h // n_rep` and no data is duplicated here. DEVIATION 813 admits GQA
    # on exactly that ground and states the cost: at `n_rep == 1` a broken
    # head-to-kv-head map is INVISIBLE, so the gates must carry both
    # `n_rep == 1` and `n_rep == 2`. Fixture cases 0 and 1.
    #
    # S12 scales the FINISHED dot, not q. Pre-scaling q is a different
    # answer (one rounding per q ELEMENT rather than one per SCORE) and is
    # sabotage `S12_SCALE_INTO_Q`.

    # Every attention stage is built in a LOCAL and moved into `st` at the
    # end of the section. Same reason as the norm above: no `mut st.field`
    # at a call site, and no read of a field being written in the same
    # statement.
    # CPU SPEED (lane neural-cpu, 2026-09-28): every stage list is sized
    # once and written by index, in the order the appends made it, and the
    # per-key statements run as lanes (`core/host_lanes.mojo`, each helper
    # naming the statement it equals). Every stage still completes over the
    # whole array before the next one reads it, so the plants land where they
    # did.
    var scores = host_f32_uninit(b * nh * l * s)
    var masked = host_f32_uninit(b * nh * l * s)
    var amax = List[Float32](length=b * nh * l, fill=Float32(0.0))
    var aexp = host_f32_uninit(b * nh * l * s)
    var adenom = List[Float32](length=b * nh * l, fill=Float32(0.0))
    var aweights = host_f32_uninit(b * nh * l * s)
    var actx = List[Float32]()

    var scale = attention_scale(hd)
    var fused_attn = ATTN_HOST_FUSED and not int15 and not opts.has_softcap()
    if fused_attn:
        _plant_refuse(len(scores), plant, PLANT_AT_SCORES)
        _plant_refuse(len(masked), plant, PLANT_AT_MASKED)
        var pidx = plant.idx.copy()
        var pbits = plant.bits.copy()
        var pat = plant.at
        var npan = ghr_panel_count(s)
        var plen = npan * hd * GHR_G
        var kpan = host_f32_uninit(b * nkv * plen)
        var kp_p = _hp(kpan)
        var kvk_f = _hp(st.kv_k_cache)
        var punits = b * nkv * npan
        var pk_split = _row_split(punits, hd * GHR_G)
        var pk_tasks = pk_split[0]
        var pk_chunk = pk_split[1]
        def _pack_k(task: Int) {imm kp_p, imm kvk_f, imm punits, imm pk_chunk, imm npan, imm plen, imm s, imm hd}:
            for u in range(task * pk_chunk, min((task + 1) * pk_chunk, punits)):
                var head = u // npan
                var g = u % npan
                ghr_pack_b(kvk_f.unsafe_offset(head * s * hd), OP_NT, s, hd, kp_p.unsafe_offset(head * plen), g, g + 1)
        if pk_tasks <= 1:
            _pack_k(0)
        else:
            host_parallelize(_pack_k, pk_tasks)
        actx = host_f32_uninit(m * qw)
        var q_rope_f = _hp(st.q_rope_out)
        var vcache_f = st.kv_v_cache.copy()
        var arows = b * nh * l
        var a_split = _row_split(arows, s * hd + 6 * s)
        var a_tasks = a_split[0]
        var a_chunk = a_split[1]
        var mfill_f = mask_fill()
        var ufill_f = unmasked_fill()
        def _attn_row(task: Int) {imm q_rope_f, imm kp_p, imm vcache_f, mut scores, mut masked, mut amax, mut aexp, mut adenom, mut aweights, mut actx, imm pidx, imm pbits, imm pat, imm arows, imm a_chunk, imm l, imm s, imm hd, imm nh, imm nkv, imm n_rep, imm qw, imm npan, imm plen, imm scale, imm own0, imm window, imm mfill_f, imm ufill_f}:
            var qrow = host_f32_uninit(hd)
            var cell = host_f32_uninit(s)
            var q_p = _hp(qrow)
            var c_p = _hp(cell)
            for r in range(task * a_chunk, min((task + 1) * a_chunk, arows)):
                var qi = r % l
                var h = (r // l) % nh
                var bb = r // (l * nh)
                var kv = h // n_rep
                var base = r * s
                ghr_pack_a(q_rope_f.unsafe_offset((bb * l + qi) * qw + h * hd), OP_NT, 1, hd, q_p)
                ghr_tile(q_p, kp_p.unsafe_offset((bb * nkv + kv) * plen), c_p, s, hd, 0, 1, 0, npan)
                span_scale(cell, 0, s, scale, scores, base)
                _apply_plant_row(scores, pidx, pbits, pat, PLANT_AT_SCORES, base, s)
                var vis = _visible_keys(own0, window, qi, s)
                var jlo = vis[0]
                var jhi = vis[1]
                span_add_scalar(scores, base, jlo, mfill_f, masked, base)
                span_add_scalar(scores, base + jlo, jhi - jlo + 1, ufill_f, masked, base + jlo)
                span_add_scalar(scores, base + jhi + 1, s - jhi - 1, mfill_f, masked, base + jhi + 1)
                _apply_plant_row(masked, pidx, pbits, pat, PLANT_AT_MASKED, base, s)
                var mx = span_fmax_fold(masked, base, s)
                amax[r] = ftz(mx)
                for j in range(jlo):
                    aexp[base + j] = Float32(0.0)
                    aweights[base + j] = Float32(0.0)
                for j in range(jhi + 1, s):
                    aexp[base + j] = Float32(0.0)
                    aweights[base + j] = Float32(0.0)
                span_exp_shift(masked, base + jlo, jhi - jlo + 1, mx, aexp, base + jlo)
                var acc = Float32(0.0)
                for j in range(jlo, jhi + 1):
                    acc = ftz(ftz(acc) + ftz(aexp[base + j]))
                acc = ftz(acc)
                adenom[r] = acc
                span_div(aexp, base + jlo, jhi - jlo + 1, acc, aweights, base + jlo)
                attn_value_sum_lanes(
                    aweights, base + jlo,
                    vcache_f, (bb * nkv + kv) * s * hd + jlo * hd,
                    actx, (bb * l + qi) * qw + h * hd, jhi - jlo + 1, hd,
                )
            _ = qrow^
            _ = cell^
        if a_tasks <= 1:
            _attn_row(0)
        else:
            host_parallelize(_attn_row, a_tasks)
        _ = kpan^
        _ = vcache_f^
        host_tick(hton, htk, "fwd.attention")
    else:
        var q_rope_p = _hp(st.q_rope_out)
        var kvk_p = _hp(st.kv_k_cache)
        for bb in range(b):
            for h in range(nh):
                var kv = h // n_rep
                # the head's query and key matrices, rows over tasks (lane
                # neural-pass21): copies, one source cell to one destination
                var qmat = host_f32_uninit(l * hd)
                var kmat = host_f32_uninit(s * hd)
                var qm_p = _hp(qmat)
                var km_p = _hp(kmat)
                var g_split = _row_split(l + s, hd)
                var g_tasks = g_split[0]
                var g_chunk = g_split[1]
                def _gather(task: Int) {imm q_rope_p, imm kvk_p, imm qm_p, imm km_p, imm l, imm s, imm hd, imm qw, imm nkv, imm bb, imm h, imm kv, imm g_chunk}:
                    for r in range(task * g_chunk, min((task + 1) * g_chunk, l + s)):
                        if r < l:
                            var qi = r
                            for d in range(hd):
                                qm_p.unsafe_store(qi * hd + d, q_rope_p.unsafe_load((bb * l + qi) * qw + h * hd + d))
                        else:
                            var j = r - l
                            for d in range(hd):
                                km_p.unsafe_store(j * hd + d, kvk_p.unsafe_load(((bb * nkv + kv) * s + j) * hd + d))
                if g_tasks <= 1:
                    _gather(0)
                else:
                    host_parallelize(_gather, g_tasks)
                var cell = _oracle_product(qmat, kmat, l, s, hd, int15)
                var sbase = (bb * nh + h) * l * s
                if not opts.has_softcap():
                    # the scale over query rows as spans (the same per-cell lane
                    # statement whichever task runs the row)
                    var sc_split = _row_split(l, s)
                    var sc_tasks = sc_split[0]
                    var sc_chunk = sc_split[1]
                    def _scale_rows(task: Int) {imm cell, mut scores, imm l, imm s, imm scale, imm sbase, imm sc_chunk}:
                        var r0 = task * sc_chunk
                        var r1 = min((task + 1) * sc_chunk, l)
                        if r1 > r0:
                            span_scale(cell, r0 * s, (r1 - r0) * s, scale, scores, sbase + r0 * s)
                    if sc_tasks <= 1:
                        _scale_rows(0)
                    else:
                        host_parallelize(_scale_rows, sc_tasks)
                    continue
                for qi in range(l):
                    for j in range(s):
                        var sc = ftz(identical_mul(ftz(cell[qi * s + j]), scale))
                        # DEVIATION 2947, `attn_softcap` (Gemma2's
                        # `attn_logit_softcapping`, `eager_attention_forward` in
                        # modeling_gemma2.py: `attn_weights = attn_weights /
                        # softcap; attn_weights = torch.tanh(attn_weights);
                        # attn_weights = attn_weights * softcap`), AFTER the scale
                        # and BEFORE the mask: one `identical_div`, one
                        # `identical_tanh` (DEVIATION 821's portable tanh), one
                        # `identical_mul`, each flushed. Recorded into
                        # `attn.scores`.
                        if opts.has_softcap():
                            var cap = ftz(opts.attn_softcap)
                            var th = ftz(identical_tanh(ftz(identical_div(sc, cap))))
                            sc = ftz(identical_mul(th, cap))
                        scores[sbase + qi * s + j] = sc
        _apply_plant(scores, plant, PLANT_AT_SCORES)
        host_tick(hton, htk, "fwd.scores")

        # ---- S13: the additive causal mask (EAF:205-206) ---------------------
        # `attn_weights = attn_weights + attention_mask`, with the mask built at
        # masking_utils.py:601-603 as `torch.finfo(dtype).min` where False and
        # `0.0` where True.
        #
        # **AN ADD, NEVER A SELECT** (DEVIATION 803, contract 4.1(a)). The
        # `+0.0` at an unmasked cell is not a no-op: `x + (+0.0) == x` for every
        # finite x, every infinity and every NaN EXCEPT `x = -0.0`, where it
        # gives `+0.0`. An implementation that skips the add where the mask is
        # true keeps a `-0.0` the reference launders, and it passes every
        # fixture that does not PLANT a negative zero. Sabotage
        # `S13_MASK_SELECT`, fixture `adv_score_neg_zero`.
        #
        # **FINITE, NEVER `-inf`** (contract 4.1(b)). With `-inf` a fully masked
        # row gives `-inf - (-inf) = NaN` and a computed NaN's payload is
        # vendor-shaped. The causal mask never produces a fully masked row
        # (position t always attends to itself) so the difference is not
        # reachable through the mask alone; it IS reachable through an extreme
        # score, which is fixture `adv_score_extreme`. Sabotage
        # `S13_MASK_NEG_INF`.
        #
        # MAX's own reference kernels are inconsistent with themselves here and
        # it is worth knowing: `_bmm0_bs` stamps out-of-range scores with
        # `min_or_neg_inf` (mha.mojo:6639), `softmax_kernel` seeds its max with
        # the FINITE `Scalar[dtype].MIN` (softmax.mojo:958), and
        # `_softmax_warp_kernel` seeds with `min_or_neg_inf` (softmax.mojo:1063).
        # Two spellings of one reduction in one file.
        var mfill = mask_fill()
        var ufill = unmasked_fill()
        # ROWS OVER HOST TASKS (lane neural-pass7): a (batch, head, query) row's
        # cells read that row of `scores` and the row's own position; the same
        # statement per cell whichever task runs it.
        var mrows = b * nh * l
        var mtasks = host_row_tasks(mrows, 3 * s)
        var mchunk = (mrows + mtasks - 1) // mtasks
        # Each row is three spans (lane neural-pass12): the masked keys before
        # the window, the visible keys, the masked keys after the query, each
        # the same per-cell statement `ftz(ftz(score) + fill)` as lanes
        # (`span_add_scalar`), the fill the one the per-cell test picked.
        def _mask_rows(t: Int) {imm scores, mut masked, imm mrows, imm mchunk, imm l, imm s, imm own0, imm window, imm mfill, imm ufill}:
            for r in range(t * mchunk, min((t + 1) * mchunk, mrows)):
                var qi = r % l
                var base = r * s
                var vis = _visible_keys(own0, window, qi, s)
                var jlo = vis[0]
                var jhi = vis[1]
                span_add_scalar(scores, base, jlo, mfill, masked, base)
                span_add_scalar(scores, base + jlo, jhi - jlo + 1, ufill, masked, base + jlo)
                span_add_scalar(scores, base + jhi + 1, s - jhi - 1, mfill, masked, base + jhi + 1)
        if mtasks <= 1:
            _mask_rows(0)
        else:
            host_parallelize(_mask_rows, mtasks)
        _apply_plant(masked, plant, PLANT_AT_MASKED)
        host_tick(hton, htk, "fwd.mask")

        # ---- S14 through S18: the softmax (EAF:208) --------------------------
        # `nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)`,
        # whose internals are ATen's and which contract section 5.4 records as
        # THE ONE THING THIS LANE COULD NOT READ. There is no PyTorch checkout
        # in `/Users/andrewhendel/CascadeProjects/upstream/`, verified again on
        # 2026-08-25, so whether ATen's CUDA softmax divides or multiplies by a
        # reciprocal is not known here. The profile pins the DIVISION because it
        # is the spelling with one rounding. **That is a stated gap, not a
        # decision made on evidence**, and it is the first thing to check when a
        # PyTorch checkout lands.
        # The rows (batch, head, query) are independent: row r = (bb*nh + h)*l + qi
        # owns amax[r], adenom[r] and the s entries at r*s of aexp and aweights,
        # and reads only masked's row r. They split over host tasks
        # (`core/host_parallel.mojo`, lane neural-cpu); each row's statements are
        # the serial walk's.
        var srows = b * nh * l
        var stasks = host_row_tasks(srows, s)
        var schunk = (srows + stasks - 1) // stasks

        # THE MASKED KEYS ARE SKIPPED, AND THAT IS THE CONTRACT'S OWN THEOREM
        # (lane neural-pass10). A masked cell is exactly `-FLT_MAX` (contract
        # 7.1: a finite score plus the mask fill rounds to the fill), so its
        # exponential is exactly `+0.0` (the scalar's early return below
        # -87.33655, `expf_lanes` the same on every bit pattern), its weight
        # is `+0.0 / denom = +0.0`, and a `+0.0` term is bitwise inert in the
        # `+0.0`-seeded serial ascending denominator chain (section 7.2's
        # argument for decode == prefill, and the backward's `z` fold). So the
        # exponential, the denominator and the weights run over the visible
        # keys `[jlo, jhi]` only, and the masked cells are WRITTEN `+0.0`,
        # which is the value the full walk stored. The row maximum stays over
        # EVERY element of the row (contract 5.2).
        def _softmax_rows(t: Int) {imm masked, mut amax, mut aexp, mut adenom, mut aweights, imm s, imm l, imm own0, imm window, imm srows, imm schunk}:
            for r in range(t * schunk, min((t + 1) * schunk, srows)):
                var base = r * s
                var vis = _visible_keys(own0, window, r % l, s)
                var jlo = vis[0]
                var jhi = vis[1]

                # S14, the row maximum, over EVERY element of the row
                # INCLUDING the masked ones (contract 5.2, and it is not
                # "the unmasked prefix" -- section 7 rests on that).
                #
                # `identical_fmax` (DEVIATION 825), which canonicalizes a
                # NaN, flushes both operands and then SELECTS ON A TOTAL
                # ORDER KEY rather than on a float compare. There is no
                # hardware max instruction in it.
                #
                # **`core/pinned_reduce.mojo::pinned_block_max` MAY NOT BE
                # USED** and that refusal is the trap in this seam, because
                # it is the deterministic-looking helper already in the
                # tree. Its fold is a plain `other > red[tid]` compare
                # (:159-190), which is exactly the spelling IDENTITY_PATHS
                # row 13 closed everywhere else, and its own block comment
                # tells a caller whose inputs can carry +-0.0 to say why
                # before using it. This caller cannot say why. The refusal
                # is not about determinism; `pinned_block_max` is perfectly
                # deterministic and computes a different answer.
                #
                # **THE FOLD SHAPE IS FREE AND IT IS THE ONLY PLACE IN THIS
                # PROFILE WHERE AN EXECUTION PLAN MAY CHOOSE ITS OWN TREE**
                # (contract 5.1), because `identical_fmax` is exactly
                # commutative and associative over all of Float32 including
                # both zeros and NaN. Not because the difference is thought
                # to be small. This oracle folds serial ascending with no
                # seed because an oracle should be the simplest legal
                # spelling, and a halving tree over `identical_fmax` on a
                # device is equally legal.
                # S14, the row maximum (`span_fmax_fold`: the serial
                # `identical_fmax` chain, or under IDENTICAL its free-shape
                # lanes, `fmax_fold_span`).
                var mx = span_fmax_fold(masked, base, s)
                # The trailing `ftz` is contract section 4's preamble
                # ("every seam's RESULT passes ftz") and is BIT-INERT under
                # IDENTICAL, because `portable_fmaxf` returns one of its own
                # already-flushed operands. It is written anyway, because a
                # seam that skips the checklist unit because the author
                # reasoned it was inert is exactly how row 10's checklist
                # stops being a checklist.
                amax[r] = ftz(mx)
                # S15 and S16.
                for j in range(jlo):
                    aexp[base + j] = Float32(0.0)
                    aweights[base + j] = Float32(0.0)
                for j in range(jhi + 1, s):
                    aexp[base + j] = Float32(0.0)
                    aweights[base + j] = Float32(0.0)
                span_exp_shift(masked, base + jlo, jhi - jlo + 1, mx, aexp, base + jlo)
                # S17, the denominator: A SERIAL ASCENDING CHAIN over the
                # ABSOLUTE key index, seeded `+0.0`, plain adds. There is
                # nothing to fuse because `e[j]` is not a product.
                #
                # **`core/pinned_reduce.mojo::pinned_block_sum` MAY NOT BE
                # USED** (DEVIATION 805), and saying so is the point of the
                # clause. It is a halving tree; its own docstring says a
                # halving tree and CUB's warp-then-block shape combine
                # different partials; and a halving tree is not a serial
                # ascending chain either. Reaching for the deterministic
                # block fold BECAUSE it is the deterministic block fold is
                # the single most likely way to get this wrong, and it would
                # be wrong in a way that passes every launch-invariance gate,
                # because the tree is perfectly launch-invariant. It is
                # simply a different sum. Sabotage `S17_DENOM_HALVING_TREE`
                # must move `attn.denom` at any kv length of 3 or more.
                #
                # The load-bearing reason for the chain is not tidiness. It
                # is that a tail of exactly-`+0.0` terms is bitwise inert in
                # a chain seeded `+0.0` and is NOT inert under gemm v1's
                # leaf-and-tree topology, where `P = f(k)`. That is what
                # makes decode equal prefill and makes the answer independent
                # of sequence length. Contract 7.1 and 7.3.
                #
                # The price, stated rather than hidden: a row's fold may not
                # be split across threads, so v1's kv length is bounded by
                # what one thread will walk. This profile is reference
                # quality and slow by construction.
                var acc = Float32(0.0)
                for j in range(jlo, jhi + 1):
                    acc = ftz(ftz(acc) + ftz(aexp[base + j]))
                acc = ftz(acc)
                adenom[r] = acc
                # S18, ONE DIVISION PER WEIGHT, never a reciprocal
                # multiplied in (DEVIATION 806). `e * (1/denom)` rounds
                # twice where `e / denom` rounds once and they differ in the
                # last bit on ordinary inputs. MAX's `softmax_kernel`
                # multiplies by a reciprocal, which is evidence about MAX
                # and not about the reference. Sabotage `S18_RECIPROCAL_MUL`
                # must move `attn.weights`.
                span_div(aexp, base + jlo, jhi - jlo + 1, acc, aweights, base + jlo)

        if stasks <= 1:
            _softmax_rows(0)
        else:
            host_parallelize(_softmax_rows, stasks)
        host_tick(hton, htk, "fwd.softmax")

        # ---- S19: the attention-weighted value sum (EAF:210) -----------------
        # `torch.matmul(attn_weights, value_states)`.
        #
        # **DELIBERATELY NOT A gemm v1 CALL** (DEVIATION 807), and this is the
        # least obvious decision in the contract. Serial ascending over the
        # ABSOLUTE key index from `+0.0`, one `fma` per term.
        #
        # The reason is contract 7.2 and it is structural, not aesthetic. Gemm
        # v1's per-cell arithmetic is a pure function of the contraction length
        # `k` and the profile: `P = ceil(k / contract_leaf_size(k))`. S11
        # contracts over `head_dim`, whose length is the SAME in prefill and
        # decode, so routing it through the GEMM is decode-safe. S19 contracts
        # over the KEY axis, whose length is `t + 1` in decode and `L` in
        # prefill, so the GEMM would build two different trees for the same row
        # and give two different answers. Under a serial ascending chain seeded
        # `+0.0` the masked tail is exactly `+0.0` (contract 7.1) and therefore
        # bitwise inert, and decode equals prefill.
        #
        # Sabotage `S19_VALUE_SUM_VIA_GEMM` routes this through `identical_gemm`
        # and must break clause (d) past the first 128 keys while leaving clause
        # (a) GREEN at a fixed length. Clause (a) staying green is the point: a
        # single-path gate cannot see this failure at all, which is why fixture
        # `long_l257` exists and why writing clause (d) late would make the
        # sabotage look pointless and get it deleted.
        #
        # FUSED, unlike S10. The reference's matmul contracts and the fold is
        # ours; there is no two-rounding reference spelling to mirror here the
        # way there is at `(q*cos) + (rotate_half(q)*sin)`.
        #
        # CPU SPEED (lane neural-cpu, 2026-09-28): the chains above, `head_dim` of
        # them per query advanced together down the key axis, one SIMD lane per
        # output (`attn_value_sum_lanes`). Each lane runs its output's chain: the
        # same operands, j ascending from `+0.0`, one fused multiply-add and one
        # flush per step. `actx` is sized once; every cell is written.
        actx = host_f32_uninit(m * qw)
        var vrows = b * nh * l
        var vtasks = host_row_tasks(vrows, s * hd)
        var vchunk = (vrows + vtasks - 1) // vtasks
        var vcache = st.kv_v_cache.copy()

        # The value sum over the visible keys only (the same theorem: a `+0.0`
        # weight's fma term is inert in the `+0.0`-seeded chain, S19).
        def _value_rows(t: Int) {imm aweights, imm vcache, mut actx, imm s, imm hd, imm qw, imm l, imm nh, imm nkv, imm n_rep, imm own0, imm window, imm vrows, imm vchunk}:
            for r in range(t * vchunk, min((t + 1) * vchunk, vrows)):
                var qi = r % l
                var h = (r // l) % nh
                var bb = r // (l * nh)
                var kv = h // n_rep
                var vis = _visible_keys(own0, window, qi, s)
                attn_value_sum_lanes(
                    aweights, r * s + vis[0],
                    vcache, (bb * nkv + kv) * s * hd + vis[0] * hd,
                    actx, (bb * l + qi) * qw + h * hd, vis[1] - vis[0] + 1, hd,
                )

        if vtasks <= 1:
            _value_rows(0)
        else:
            host_parallelize(_value_rows, vtasks)
        _ = vcache^
        host_tick(hton, htk, "fwd.value_sum")

    st.attn_scores = scores^
    st.attn_masked = masked^
    st.attn_max = amax^
    st.attn_exp = aexp^
    st.attn_denom = adenom^
    st.attn_weights = aweights^
    st.attn_ctx = actx^

    # ---- S5, o_proj (:280). The flatten before it is a COPY (:279). ------
    st.o_proj_out = _oracle_product(st.attn_ctx, w.w_o, m, dm, qw, int15)
    host_tick(hton, htk, "fwd.o_proj")
    # DEVIATION 2935, `o_bias`: one plain add per cell after the GEMM.
    if opts.o_bias:
        var ob = st.o_proj_out.copy()
        add_bias_into(ob, w.b_o, m, dm)
        st.o_proj_out = ob^

    # ---- S22, the first residual (LDL:317) -------------------------------
    # `hidden_states = residual + hidden_states`, where `residual` is the
    # BLOCK INPUT and not the normalized one (:305 captures it before :306
    # normalizes). One plain add of two already-rounded values.
    st.residual1_out = host_f32_uninit(m * dm)
    _span_add_par(x, st.o_proj_out, m * dm, st.residual1_out)

    # ---- S1-S4 again, post_attention_layernorm (LDL:321) -----------------
    var n2_sumsq = List[Float32]()
    var n2_out = List[Float32]()
    norm_into(
        st.residual1_out, w.norm2_w, w.norm2_b, m, dm, opts.norm_eps,
        opts.norm_kind, opts.norm_bias, n2_sumsq, n2_out,
    )
    st.norm2_sumsq = n2_sumsq^
    st.norm2_out = n2_out^
    host_tick(hton, htk, "fwd.residual1_norm2")

    # ---- S5, S20, S21, S5: the MLP (LMLP:174-176) ------------------------
    # `down_proj(act_fn(gate_proj(x)) * up_proj(x))` at the default record.
    #
    # lane/block-options, the MLP axis, all of it in one place:
    #   DEVIATION 2939  mlp = gelu       ungated: down(gelu_erf(up(x)))
    #   DEVIATION 2943  mlp = gelu_tanh  ungated: down(gelu_tanh(up(x)))
    #   DEVIATION 2944  mlp = geglu / geglu_tanh: gated, GELU for SiLU
    #                   (Gemma's `hidden_act = gelu_pytorch_tanh` is
    #                   geglu_tanh: `down(act(gate(x)) * up(x))`)
    #   DEVIATION 2945  mlp_bias: one plain add after gate, up and down
    # The activations are `identical_gelu_erf` (DEVIATION 823,
    # `GELUActivation` verbatim) and `identical_gelu_tanh` (DEVIATION 824,
    # `GELUTanh`/`NewGELUActivation` verbatim); nothing is spelled here. The
    # card keeps its thirty tags: under an ungated MLP `gate_proj.out` and
    # `mlp.gated` are recorded EMPTY (the `stage_tag` docstring's rule: a
    # stage that is empty for a shape is still recorded), `silu.out` holds
    # the activation of `up_proj.out` and feeds `down_proj` directly.
    var gated = opts.gated()
    if gated:
        st.gate_proj_out = _oracle_product(st.norm2_out, w.w_gate, m, inter, dm, int15)
        if opts.mlp_bias:
            var gb = st.gate_proj_out.copy()
            add_bias_into(gb, w.b_gate, m, inter)
            st.gate_proj_out = gb^
    st.up_proj_out = _oracle_product(st.norm2_out, w.w_up, m, inter, dm, int15)
    host_tick(hton, htk, "fwd.gate_up_proj")
    if opts.mlp_bias:
        var ub = st.up_proj_out.copy()
        add_bias_into(ub, w.b_up, m, inter)
        st.up_proj_out = ub^

    # S20: `identical_silu` (DEVIATION 744), which is ATen's `z / (1 +
    # exp(-z))`, ONE DIVISION. **Not** `x * sigmoid(x)`, which is two
    # roundings and which is what MAX itself spells
    # (`max/kernels/src/nn/activations.mojo:249`). Sabotage
    # `S20_SILU_MUL_SIGMOID` must move `silu.out`.
    if opts.act_is_silu():
        # CPU SPEED (lane neural-cpu): the statement below as lanes.
        st.silu_out = host_f32_uninit(m * inter)
        if gated:
            _span_silu_par(st.gate_proj_out, m * inter, st.silu_out)
        else:
            _span_silu_par(st.up_proj_out, m * inter, st.silu_out)
    var scalar_act = 0 if opts.act_is_silu() else m * inter
    host_tick(hton, htk, "fwd.silu")
    for i in range(scalar_act):
        var z: Float32
        if gated:
            z = ftz(st.gate_proj_out[i])
        else:
            z = ftz(st.up_proj_out[i])
        if opts.act_is_silu():
            st.silu_out.append(ftz(identical_silu(z)))
        elif opts.act_is_gelu_tanh():
            st.silu_out.append(ftz(identical_gelu_tanh(z)))
        else:
            st.silu_out.append(ftz(identical_gelu_erf(z)))

    # S21: one product, so `pinned_mul`. Absent under an ungated MLP.
    if gated:
        st.mlp_gated = host_f32_uninit(m * inter)
        _span_mul_par(st.silu_out, st.up_proj_out, m * inter, st.mlp_gated)
        st.down_proj_out = _oracle_product(st.mlp_gated, w.w_down, m, dm, inter, int15)
    else:
        st.down_proj_out = _oracle_product(st.silu_out, w.w_down, m, dm, inter, int15)
    if opts.mlp_bias:
        var db = st.down_proj_out.copy()
        add_bias_into(db, w.b_down, m, dm)
        st.down_proj_out = db^

    # ---- S23, the second residual (LDL:323) ------------------------------
    st.residual2_out = host_f32_uninit(m * dm)
    _span_add_par(st.residual1_out, st.down_proj_out, m * dm, st.residual2_out)
    host_tick(hton, htk, "fwd.down_proj_residual2")

    return st^


def transformer_prefill_then_decode(
    w: TransformerWeights,
    x: List[Float32],
    b: Int,
    l: Int,
    split: Int,
    mut cache: TransformerKVCache,
    rope: RopeTable,
) raises -> TransformerStages:
    """Two calls carrying the cache, returning the SECOND call's stages.

    The shape contract clause (d) is about, offered here so that the check
    file does not have to re-derive the token slicing and so that a reader
    can see in one place that the second call is the same function with the
    same arguments and a shorter `l`.

    **NO PLANT.** A planted score in a two-call fixture would make it
    impossible to say whether a divergence came from the plant or from the
    cache, and clause (d)'s whole value is that it separates those.

    **THIS FUNCTION IS NOT A GATE AND MUST NOT BE MISTAKEN FOR ONE.** It
    produces the decode side of a comparison; the comparison, and the
    negative control that two DIFFERENT positions must differ, belong in
    `transformer_check.mojo`. Contract section 7.2 is explicit that a gate
    comparing a decode step against a prefill token, without that negative
    control, passes for ever when the index map is broken."""
    if split <= 0 or split >= l:
        raise Error(
            String("transformer: split ")
            + String(split)
            + " must be in (0, l) for a prefill-then-decode run"
        )
    var dm = w.dims.d_model
    var head = List[Float32]()
    var tail = List[Float32]()
    for bb in range(b):
        for t in range(l):
            for j in range(dm):
                var v = x[(bb * l + t) * dm + j]
                if t < split:
                    head.append(v)
                else:
                    tail.append(v)
    var first = transformer_block_oracle(
        w, head, b, split, cache, rope, ScorePlant.none()
    )
    _ = first^
    var second = transformer_block_oracle(
        w, tail, b, l - split, cache, rope, ScorePlant.none()
    )
    return second^
