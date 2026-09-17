# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The decoder block's OPTIONS record (lane/block-options, 2026-09-17).
one plain-data struct that the host oracle, the device kernels, the host
forward and the three bindings all read, so that the constructor

    TransformerBlock(weights, *, n_heads, n_kv_heads=None, head_dim=None,
        window=0, rope_theta=10000.0, rope_scaling=None, rope_dim=None,
        max_positions=8192, qkv_bias=False, o_bias=False, norm="rmsnorm",
        norm_eps=1e-6, norm_bias=False, mlp="swiglu", mlp_bias=False,
        qk_norm=False, attn_softcap=None)

has ONE encoding on every side of every boundary.

WHERE THIS FILE SITS, AND WHY. `transformer/impl/llama/modeling_llama.mojo`
imports nothing from `transformer/checks/` by design (its header: "no
fixture type crosses this boundary"), and the oracle must not import the
device file. A record both must read therefore lives at the package level,
beside neither. It carries NO arithmetic: every seam an option adds is
spelled twice, once in the oracle and once in the kernels, through
`checks/numerics.mojo`'s helpers, exactly as the contract's existing seams
are.

THE DEFAULT RECORD IS TODAY'S BLOCK, BIT FOR BIT. `BlockOptions()` is
theta 10000.0 (0x461C4000), no scaling, full rotary, the 8192 ceiling,
no bias anywhere, RMSNorm at eps 1e-6 (0x358637BD; the profile's frozen
constant, `transformer_fixture.mojo::RMS_EPS`, NOT the 1e-5 the lane brief
guessed), SwiGLU, no qk norm, no softcap. `is_default()` is what the
Python side uses to decide whether to send the tails at all: a default
block sends the OLD list lengths and reaches the OLD code path.

THE TWO TAILS, WORD FOR WORD (mirrored in `python/mojolearn/_transformer_impl.py`
and in every binding docstring that accepts them). Both are appended AFTER
the entry point's existing list; an entry point accepts the old length (every
option at its default) or the old length plus the tail, and refuses any other
length by name. Float32 options cross as their IEEE-754 BIT PATTERN in an
int, because a params list is ints and a decimal round trip is
`[[mojo-string-float-roundtrip]]`'s hazard.

    params tail (BLOCK_OPTION_PARAMS = 17 ints)
      +0   rope_theta_bits           Float32 bits of the RoPE base
      +1   rope_scaling              0 none, 1 linear, 2 llama3
      +2   rope_factor_bits          Float32 bits (`factor`, linear and llama3)
      +3   rope_low_freq_factor_bits Float32 bits (llama3)
      +4   rope_high_freq_factor_bits Float32 bits (llama3)
      +5   rope_original_max_positions  int (llama3's original_max_position_embeddings)
      +6   rope_dim                  int, 0 = head_dim (partial rotary otherwise)
      +7   max_positions             int, the absolute-position ceiling this model declares
      +8   qkv_bias                  0 / 1
      +9   o_bias                    0 / 1
      +10  norm_kind                 0 rmsnorm, 1 layernorm, 2 rmsnorm_offset
      +11  norm_eps_bits             Float32 bits
      +12  norm_bias                 0 / 1
      +13  mlp_kind                  0 swiglu, 1 gelu, 2 gelu_tanh, 3 geglu, 4 geglu_tanh
      +14  mlp_bias                  0 / 1
      +15  qk_norm                   0 / 1
      +16  attn_softcap_bits         Float32 bits, 0x00000000 = none

    addrs tail (BLOCK_OPTION_ADDRS = 11 addresses, 0 = absent)
      +0   q_proj.bias               [n_heads*head_dim]        qkv_bias
      +1   k_proj.bias               [n_kv_heads*head_dim]     qkv_bias
      +2   v_proj.bias               [n_kv_heads*head_dim]     qkv_bias
      +3   o_proj.bias               [d_model]                 o_bias
      +4   input_layernorm.bias      [d_model]                 norm_bias
      +5   post_attention_layernorm.bias [d_model]             norm_bias
      +6   up_proj.bias              [intermediate]            mlp_bias
      +7   down_proj.bias            [d_model]                 mlp_bias
      +8   gate_proj.bias            [intermediate]            mlp_bias and gated
      +9   q_norm.weight             [head_dim]                qk_norm
      +10  k_norm.weight             [head_dim]                qk_norm

A tensor whose flag is off must be ABSENT and one whose flag is on must be
PRESENT; each side refuses the mismatch by the tensor's name. With an
ungated MLP (`gelu`, `gelu_tanh`) the existing `gate_proj.weight` slot of
the base list carries 0 and the projection is never run.

DEVIATION NUMBERS. This lane owns 2930-2939 and 2943-2959 (2940-2942 were
already taken in the tree by the resident decode session and its
neighbours when this lane started). A change that moves no bits takes no
number; each NEW numerical option takes one, cited at every site that
spells it:

    2930 rope_theta as a parameter
    2931 rope_scaling "linear"
    2932 rope_scaling "llama3"
    2933 max_positions, and the ceiling restated as the ANGLE domain
    2934 qkv_bias
    2935 o_bias
    2936 norm "layernorm" with norm_bias
    2937 norm_eps as a parameter
    2938 norm "rmsnorm_offset" (Gemma's `(1 + w)`)
    2939 mlp "gelu" (ungated, erf)
    2943 mlp "gelu_tanh" (ungated, tanh form)
    2944 mlp "geglu" and "geglu_tanh" (gated, GELU in place of SiLU)
    2945 mlp_bias
    2946 qk_norm
    2947 attn_softcap
    2948 rope_dim (partial rotary)
"""

from std.memory import bitcast


comptime ROPE_SCALING_NONE = 0
comptime ROPE_SCALING_LINEAR = 1
comptime ROPE_SCALING_LLAMA3 = 2

comptime NORM_RMSNORM = 0
comptime NORM_LAYERNORM = 1
comptime NORM_RMSNORM_OFFSET = 2

comptime MLP_SWIGLU = 0
comptime MLP_GELU = 1
comptime MLP_GELU_TANH = 2
comptime MLP_GEGLU = 3
comptime MLP_GEGLU_TANH = 4

comptime BLOCK_OPTION_PARAMS = 17
comptime BLOCK_OPTION_ADDRS = 11

comptime DEFAULT_ROPE_THETA_BITS: UInt32 = 0x461C4000
"""10000.0, contract section 3's frozen base."""
comptime DEFAULT_NORM_EPS_BITS: UInt32 = 0x358637BD
"""1e-6, `LlamaConfig.rms_norm_eps`, contract section 3. Today's eps."""
comptime DEFAULT_MAX_POSITIONS = 8192
"""Contract DEVIATION 812's ceiling, now the DEFAULT declared ceiling."""
comptime ROPE_ANGLE_DOMAIN_BITS: UInt32 = 0x46000000
"""8192.0: the Cody-Waite domain of `_cephes_sincosf_core`, the bound the
ROTATION ANGLE (`position * inv_freq`) must stay strictly below. DEVIATION
2933 restates DEVIATION 812 as a bound on the angle rather than on the
position, which is what the reduction actually bounds; at the default
options the two coincide because `inv_freq[0] == 1.0`."""


def f32_bits_int(v: Float32) -> Int:
    """A Float32's bit pattern as a non-negative Int, the params spelling."""
    return Int(bitcast[DType.uint32](v))


def f32_from_bits_int(v: Int, what: String) raises -> Float32:
    """The inverse, refusing anything outside the 32-bit range by name."""
    if v < 0 or v > 0xFFFFFFFF:
        raise Error(
            String("transformer options: ")
            + what
            + " must be a Float32 bit pattern in [0, 2^32), got "
            + String(v)
        )
    return bitcast[DType.float32](UInt32(v))


def _is_finite_positive(v: Float32) -> Bool:
    var b = bitcast[DType.uint32](v)
    if (b & UInt32(0x7F800000)) == UInt32(0x7F800000):
        return False
    return v > Float32(0.0)


def rope_angle_domain() -> Float32:
    return bitcast[DType.float32](ROPE_ANGLE_DOMAIN_BITS)


struct BlockOptions(Copyable, Movable):
    """The record. Every field is a plain scalar; see the module docstring
    for the meaning and the params order."""

    var rope_theta: Float32
    var rope_scaling: Int
    var rope_factor: Float32
    var rope_low_freq_factor: Float32
    var rope_high_freq_factor: Float32
    var rope_original_max_positions: Int
    var rope_dim: Int
    var max_positions: Int
    var qkv_bias: Bool
    var o_bias: Bool
    var norm_kind: Int
    var norm_eps: Float32
    var norm_bias: Bool
    var mlp_kind: Int
    var mlp_bias: Bool
    var qk_norm: Bool
    var attn_softcap: Float32

    def __init__(out self):
        """TODAY'S BLOCK: every field at the value the frozen profile pins."""
        self.rope_theta = bitcast[DType.float32](DEFAULT_ROPE_THETA_BITS)
        self.rope_scaling = ROPE_SCALING_NONE
        self.rope_factor = Float32(1.0)
        self.rope_low_freq_factor = Float32(1.0)
        self.rope_high_freq_factor = Float32(4.0)
        self.rope_original_max_positions = DEFAULT_MAX_POSITIONS
        self.rope_dim = 0
        self.max_positions = DEFAULT_MAX_POSITIONS
        self.qkv_bias = False
        self.o_bias = False
        self.norm_kind = NORM_RMSNORM
        self.norm_eps = bitcast[DType.float32](DEFAULT_NORM_EPS_BITS)
        self.norm_bias = False
        self.mlp_kind = MLP_SWIGLU
        self.mlp_bias = False
        self.qk_norm = False
        self.attn_softcap = Float32(0.0)

    @staticmethod
    def from_params(p: List[Int], base: Int) raises -> Self:
        """The tail read out of a params list whose first `base` entries are
        the entry point's own. `len(p) == base` is the default record;
        `len(p) == base + BLOCK_OPTION_PARAMS` is the tail; anything else is
        refused by name. `validate` is NOT called here: the caller knows
        `head_dim`."""
        var o = Self()
        if len(p) == base:
            return o^
        if len(p) != base + BLOCK_OPTION_PARAMS:
            raise Error(
                String("transformer options: params must hold ")
                + String(base)
                + " values or "
                + String(base)
                + " + "
                + String(BLOCK_OPTION_PARAMS)
                + " (the block options tail), got "
                + String(len(p))
            )
        o.rope_theta = f32_from_bits_int(p[base + 0], "rope_theta_bits")
        o.rope_scaling = p[base + 1]
        o.rope_factor = f32_from_bits_int(p[base + 2], "rope_factor_bits")
        o.rope_low_freq_factor = f32_from_bits_int(
            p[base + 3], "rope_low_freq_factor_bits"
        )
        o.rope_high_freq_factor = f32_from_bits_int(
            p[base + 4], "rope_high_freq_factor_bits"
        )
        o.rope_original_max_positions = p[base + 5]
        o.rope_dim = p[base + 6]
        o.max_positions = p[base + 7]
        o.qkv_bias = p[base + 8] != 0
        o.o_bias = p[base + 9] != 0
        o.norm_kind = p[base + 10]
        o.norm_eps = f32_from_bits_int(p[base + 11], "norm_eps_bits")
        o.norm_bias = p[base + 12] != 0
        o.mlp_kind = p[base + 13]
        o.mlp_bias = p[base + 14] != 0
        o.qk_norm = p[base + 15] != 0
        o.attn_softcap = f32_from_bits_int(p[base + 16], "attn_softcap_bits")
        return o^

    def to_params(self) -> List[Int]:
        """The tail, in the module docstring's order. `from_params(to_params)`
        round-trips every field bit for bit."""
        var p = List[Int]()
        p.append(f32_bits_int(self.rope_theta))
        p.append(self.rope_scaling)
        p.append(f32_bits_int(self.rope_factor))
        p.append(f32_bits_int(self.rope_low_freq_factor))
        p.append(f32_bits_int(self.rope_high_freq_factor))
        p.append(self.rope_original_max_positions)
        p.append(self.rope_dim)
        p.append(self.max_positions)
        p.append(1 if self.qkv_bias else 0)
        p.append(1 if self.o_bias else 0)
        p.append(self.norm_kind)
        p.append(f32_bits_int(self.norm_eps))
        p.append(1 if self.norm_bias else 0)
        p.append(self.mlp_kind)
        p.append(1 if self.mlp_bias else 0)
        p.append(1 if self.qk_norm else 0)
        p.append(f32_bits_int(self.attn_softcap))
        return p^

    def is_default(self) -> Bool:
        """Whether every field holds the frozen profile's value, BY BITS for
        the floats. A record that is default reaches the untouched code
        paths on every side."""
        var d = Self()
        if bitcast[DType.uint32](self.rope_theta) != bitcast[DType.uint32](d.rope_theta):
            return False
        if self.rope_scaling != ROPE_SCALING_NONE:
            return False
        if self.rope_dim != 0:
            return False
        if self.max_positions != DEFAULT_MAX_POSITIONS:
            return False
        if self.qkv_bias or self.o_bias or self.norm_bias or self.mlp_bias:
            return False
        if self.qk_norm:
            return False
        if self.norm_kind != NORM_RMSNORM or self.mlp_kind != MLP_SWIGLU:
            return False
        if bitcast[DType.uint32](self.norm_eps) != bitcast[DType.uint32](d.norm_eps):
            return False
        if bitcast[DType.uint32](self.attn_softcap) != UInt32(0):
            return False
        return True

    def gated(self) -> Bool:
        """Whether the MLP carries a gate projection (SwiGLU and the GEGLU
        forms). An ungated MLP is `down(act(up(x)))`."""
        return (
            self.mlp_kind == MLP_SWIGLU
            or self.mlp_kind == MLP_GEGLU
            or self.mlp_kind == MLP_GEGLU_TANH
        )

    def act_is_silu(self) -> Bool:
        return self.mlp_kind == MLP_SWIGLU

    def act_is_gelu_tanh(self) -> Bool:
        return self.mlp_kind == MLP_GELU_TANH or self.mlp_kind == MLP_GEGLU_TANH

    def has_gate_bias(self) -> Bool:
        return self.mlp_bias and self.gated()

    def has_softcap(self) -> Bool:
        return bitcast[DType.uint32](self.attn_softcap) != UInt32(0)

    def rope_dim_of(self, head_dim: Int) -> Int:
        """The number of rotated columns: `rope_dim`, or `head_dim` at 0."""
        if self.rope_dim <= 0:
            return head_dim
        return self.rope_dim

    def qk_norm_kind(self) -> Int:
        """The RMSNorm form applied to q and k under `qk_norm`: Gemma's
        offset form when the block's norm is the offset form, the plain
        form otherwise (Qwen3). Never LayerNorm; `validate` refuses that."""
        if self.norm_kind == NORM_RMSNORM_OFFSET:
            return NORM_RMSNORM_OFFSET
        return NORM_RMSNORM

    def validate(self, head_dim: Int) raises:
        """Every unsupported value or combination, REFUSED BY NAME with the
        exact value in the message. Called on every side that builds a
        block from a record."""
        if not _is_finite_positive(self.rope_theta):
            raise Error(
                String("transformer options: rope_theta must be a finite")
                + " positive float, got "
                + String(self.rope_theta)
            )
        if (
            self.rope_scaling != ROPE_SCALING_NONE
            and self.rope_scaling != ROPE_SCALING_LINEAR
            and self.rope_scaling != ROPE_SCALING_LLAMA3
        ):
            raise Error(
                String("transformer options: rope_scaling type ")
                + String(self.rope_scaling)
                + " is not supported (0 none, 1 linear, 2 llama3)"
            )
        if self.rope_scaling != ROPE_SCALING_NONE:
            if not _is_finite_positive(self.rope_factor):
                raise Error(
                    String("transformer options: rope_scaling factor must be")
                    + " a finite positive float, got "
                    + String(self.rope_factor)
                )
        if self.rope_scaling == ROPE_SCALING_LLAMA3:
            if not _is_finite_positive(self.rope_low_freq_factor):
                raise Error(
                    String("transformer options: rope_scaling llama3")
                    + " low_freq_factor must be a finite positive float, got "
                    + String(self.rope_low_freq_factor)
                )
            if not _is_finite_positive(self.rope_high_freq_factor):
                raise Error(
                    String("transformer options: rope_scaling llama3")
                    + " high_freq_factor must be a finite positive float, got "
                    + String(self.rope_high_freq_factor)
                )
            if self.rope_high_freq_factor <= self.rope_low_freq_factor:
                raise Error(
                    String("transformer options: rope_scaling llama3 needs")
                    + " high_freq_factor > low_freq_factor, got "
                    + String(self.rope_high_freq_factor)
                    + " <= "
                    + String(self.rope_low_freq_factor)
                    + " (the smoothing divides by their difference)"
                )
            if self.rope_original_max_positions <= 0:
                raise Error(
                    String("transformer options: rope_scaling llama3")
                    + " original_max_position_embeddings must be positive, got "
                    + String(self.rope_original_max_positions)
                )
        if self.rope_dim < 0 or self.rope_dim > head_dim:
            raise Error(
                String("transformer options: rope_dim ")
                + String(self.rope_dim)
                + " must be 0 (= head_dim) or in (0, head_dim="
                + String(head_dim)
                + "]"
            )
        if self.rope_dim % 2 != 0:
            raise Error(
                String("transformer options: rope_dim ")
                + String(self.rope_dim)
                + " must be even (RoPE pairs halves)"
            )
        if self.max_positions <= 0:
            raise Error(
                String("transformer options: max_positions must be positive,")
                + " got "
                + String(self.max_positions)
            )
        if (
            self.norm_kind != NORM_RMSNORM
            and self.norm_kind != NORM_LAYERNORM
            and self.norm_kind != NORM_RMSNORM_OFFSET
        ):
            raise Error(
                String("transformer options: norm kind ")
                + String(self.norm_kind)
                + " is not supported (0 rmsnorm, 1 layernorm, 2 rmsnorm_offset)"
            )
        if not _is_finite_positive(self.norm_eps):
            raise Error(
                String("transformer options: norm_eps must be a finite")
                + " positive float, got "
                + String(self.norm_eps)
            )
        if self.norm_bias and self.norm_kind != NORM_LAYERNORM:
            raise Error(
                String("transformer options: norm_bias=True needs")
                + " norm='layernorm'; norm kind "
                + String(self.norm_kind)
                + " carries no bias"
            )
        if self.mlp_kind < MLP_SWIGLU or self.mlp_kind > MLP_GEGLU_TANH:
            raise Error(
                String("transformer options: mlp kind ")
                + String(self.mlp_kind)
                + " is not supported (0 swiglu, 1 gelu, 2 gelu_tanh,"
                + " 3 geglu, 4 geglu_tanh)"
            )
        if self.qk_norm and self.norm_kind == NORM_LAYERNORM:
            raise Error(
                String("transformer options: qk_norm=True with")
                + " norm='layernorm' is not supported (no reference family"
                + " normalizes q and k with LayerNorm; use rmsnorm or"
                + " rmsnorm_offset)"
            )
        var cap_bits = bitcast[DType.uint32](self.attn_softcap)
        if cap_bits != UInt32(0) and not _is_finite_positive(self.attn_softcap):
            raise Error(
                String("transformer options: attn_softcap must be None or a")
                + " finite positive float, got "
                + String(self.attn_softcap)
            )

    def describe(self) -> String:
        """One line for refusal messages and logs."""
        var s = String("theta=") + String(self.rope_theta)
        s += " scaling=" + String(self.rope_scaling)
        s += " rope_dim=" + String(self.rope_dim)
        s += " max_positions=" + String(self.max_positions)
        s += " qkv_bias=" + String(self.qkv_bias)
        s += " o_bias=" + String(self.o_bias)
        s += " norm=" + String(self.norm_kind)
        s += " eps=" + String(self.norm_eps)
        s += " norm_bias=" + String(self.norm_bias)
        s += " mlp=" + String(self.mlp_kind)
        s += " mlp_bias=" + String(self.mlp_bias)
        s += " qk_norm=" + String(self.qk_norm)
        s += " softcap=" + String(self.attn_softcap)
        return s
