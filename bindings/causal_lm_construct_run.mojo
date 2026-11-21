# SPDX-License-Identifier: Apache-2.0
"""Typed decisions of CausalLM construction (RUN, no Python).

`causal_transformer_block` turns one transformer layer's admitted options into
the block's words: head arithmetic, the norm and MLP codes, the 17-word
options tail (float32 bit patterns of the scalar options), the extended flag,
the optional-tensor flags and widths and the nine base weight shapes.
`causal_recurrent_block` gives a Mamba layer's configuration words and its
weight shapes. `causal_f32_bits` is the finite / positive / float32-range
check of a scalar option and its bit pattern. `causal_row_slice_copy` copies
a projection row slice out of a fused checkpoint tensor (phi3).

The boundary door (`bindings/causal_lm_construct_door.mojo`) is UNPACK and
PACK: it reads the option values and weight presence into the request
structs here and publishes the words on the block's fields. A refusal whose
message quotes a Python repr is returned as a code (`refusal`) for PACK.

Parallel plan: none. One layer's words are a few dozen dependent integer
steps once per layer at construction; the door's layer loop is the unit and
each layer's block object is built under the GIL. The row slice copy is one
contiguous memcpy.
"""
from std.math import isfinite
from std.memory import bitcast, unsafe_memcpy
from core.hostptr import f32_ptr

comptime CAUSAL_BLOCK_OK = 0
comptime CAUSAL_BLOCK_BAD_NORM = 1
comptime CAUSAL_BLOCK_BAD_MLP = 2
comptime CAUSAL_BLOCK_GATE_MISSING = 3
comptime CAUSAL_BLOCK_GATE_PRESENT = 4
comptime CAUSAL_BLOCK_BAD_ROPE_SCALING = 5
comptime CAUSAL_BLOCK_OPTIONAL_PRESENT = 6
comptime CAUSAL_BLOCK_OPTIONAL_MISSING = 7

comptime CAUSAL_OPTIONAL_COUNT = 11


def causal_f32_bits(number: Float64, name: String) raises -> Int:
    """The float32 bit pattern of a finite positive scalar option."""
    if not isfinite(number) or number <= 0:
        raise Error("CausalLM: " + name + " must be finite and positive")
    var rounded = Float32(number)
    if not isfinite(rounded):
        raise Error("CausalLM: " + name + " exceeds float32 range")
    return Int(bitcast[DType.uint32](rounded))


struct CausalTransformerRequest(Defaultable, Movable):
    """One transformer layer's options and weight presence, read by UNPACK.

    `has_*` is False for an option given as None. Scalars are the admitted
    `float(value)`; the rope scaling words are read only when `has_scaling`."""
    var d_model: Int
    var n_heads: Int
    var has_n_kv_heads: Bool
    var n_kv_heads: Int
    var has_head_dim: Bool
    var head_dim: Int
    var has_rope_dim: Bool
    var rope_dim: Int
    var mlp: String
    var norm: String
    var has_gate: Bool
    # Rows of the gate and up projections; -1 when absent or not 2-D.
    var gate_rows: Int
    var up_rows: Int
    var rope_theta: Float64
    var has_scaling: Bool
    var scaling_kind: String
    var scaling_factor: Float64
    var low_freq_factor: Float64
    var high_freq_factor: Float64
    var original_max_positions: Int
    var max_positions: Int
    var qkv_bias: Bool
    var o_bias: Bool
    var norm_eps: Float64
    var norm_bias: Bool
    var mlp_bias: Bool
    var qk_norm: Bool
    var has_softcap: Bool
    var softcap: Float64
    var window: Int
    # Presence of the eleven optional tensors (`optional_names` order).
    var optional_present: List[Bool]

    def __init__(out self):
        self.d_model = 0
        self.n_heads = 0
        self.has_n_kv_heads = False
        self.n_kv_heads = 0
        self.has_head_dim = False
        self.head_dim = 0
        self.has_rope_dim = False
        self.rope_dim = 0
        self.mlp = String()
        self.norm = String()
        self.has_gate = False
        self.gate_rows = -1
        self.up_rows = -1
        self.rope_theta = 0.0
        self.has_scaling = False
        self.scaling_kind = String()
        self.scaling_factor = 0.0
        self.low_freq_factor = 0.0
        self.high_freq_factor = 0.0
        self.original_max_positions = 0
        self.max_positions = 0
        self.qkv_bias = False
        self.o_bias = False
        self.norm_eps = 0.0
        self.norm_bias = False
        self.mlp_bias = False
        self.qk_norm = False
        self.has_softcap = False
        self.softcap = 0.0
        self.window = 0
        self.optional_present = List[Bool](length=CAUSAL_OPTIONAL_COUNT, fill=False)


struct CausalTransformerWords(Movable):
    """A transformer layer's block words (PACK publishes them)."""
    var refusal: Int
    # The optional tensor index of an OPTIONAL_* refusal.
    var refusal_index: Int
    var n_kv_heads: Int
    var head_dim: Int
    var intermediate: Int
    var rope_dim: Int
    var gated: Bool
    var extended: Bool
    var tail: List[Int]
    var optional_flags: List[Bool]
    var optional_widths: List[Int]
    var shapes: List[List[Int]]

    def __init__(out self):
        self.refusal = CAUSAL_BLOCK_OK
        self.refusal_index = -1
        self.n_kv_heads = 0
        self.head_dim = 0
        self.intermediate = 0
        self.rope_dim = 0
        self.gated = True
        self.extended = False
        self.tail = List[Int](length=17, fill=0)
        self.optional_flags = List[Bool](length=CAUSAL_OPTIONAL_COUNT, fill=False)
        self.optional_widths = List[Int](length=CAUSAL_OPTIONAL_COUNT, fill=0)
        self.shapes = List[List[Int]]()


def causal_transformer_block(r: CausalTransformerRequest) raises -> CausalTransformerWords:
    """The block words of one transformer layer, refusals in the door's order."""
    var words = CausalTransformerWords()
    var dm = r.d_model
    var nh = r.n_heads
    if nh < 1:
        raise Error("CausalLM: n_heads must be positive")
    var nkv = r.n_kv_heads if r.has_n_kv_heads else nh
    var hd = r.head_dim if r.has_head_dim else dm // nh
    if nh < 1 or nkv < 1 or hd < 1 or dm != nh * hd:
        raise Error("CausalLM: d_model must equal n_heads * head_dim, all positive")
    var norm_code = 0 if r.norm == "rmsnorm" else (1 if r.norm == "layernorm" else (2 if r.norm == "rmsnorm_offset" else -1))
    var mk = 0 if r.mlp == "swiglu" else (1 if r.mlp == "gelu" else (2 if r.mlp == "gelu_tanh" else (3 if r.mlp == "geglu" else (4 if r.mlp == "geglu_tanh" else -1))))
    if norm_code < 0:
        words.refusal = CAUSAL_BLOCK_BAD_NORM
        return words^
    if mk < 0:
        words.refusal = CAUSAL_BLOCK_BAD_MLP
        return words^
    var gated = mk == 0 or mk == 3 or mk == 4
    if gated and not r.has_gate:
        words.refusal = CAUSAL_BLOCK_GATE_MISSING
        return words^
    if not gated and r.has_gate:
        words.refusal = CAUSAL_BLOCK_GATE_PRESENT
        return words^
    var it = r.gate_rows if gated else r.up_rows
    if it < 1:
        raise Error("CausalLM: projection must have positive intermediate size")
    var tail = List[Int](length=17, fill=0)
    tail[0] = causal_f32_bits(r.rope_theta, "rope_theta")
    tail[2] = 0x3f800000
    tail[3] = 0x3f800000
    tail[4] = 0x40800000
    tail[5] = 8192
    if r.has_scaling:
        var sk = r.scaling_kind
        if sk != "linear" and sk != "llama3":
            words.refusal = CAUSAL_BLOCK_BAD_ROPE_SCALING
            return words^
        tail[1] = 1 if sk == "linear" else 2
        tail[2] = causal_f32_bits(r.scaling_factor, "rope_scaling factor")
        if sk == "llama3":
            tail[3] = causal_f32_bits(r.low_freq_factor, "low_freq_factor")
            tail[4] = causal_f32_bits(r.high_freq_factor, "high_freq_factor")
            tail[5] = r.original_max_positions
            if r.high_freq_factor <= r.low_freq_factor or tail[5] < 1:
                raise Error("CausalLM: llama3 rope_scaling needs high_freq_factor > low_freq_factor "
                            + "and a positive original_max_position_embeddings")
    var rd = r.rope_dim if r.has_rope_dim else hd
    if rd < 1 or rd > hd or rd % 2 != 0:
        raise Error("CausalLM: rope_dim must be positive even and no larger than head_dim")
    tail[6] = 0 if rd == hd else rd
    tail[7] = r.max_positions
    tail[8] = Int(r.qkv_bias)
    tail[9] = Int(r.o_bias)
    tail[10] = norm_code
    tail[11] = causal_f32_bits(r.norm_eps, "norm_eps")
    tail[12] = Int(r.norm_bias)
    tail[13] = mk
    tail[14] = Int(r.mlp_bias)
    tail[15] = Int(r.qk_norm)
    tail[16] = causal_f32_bits(r.softcap, "attn_softcap") if r.has_softcap else 0
    if tail[7] < 1:
        raise Error("CausalLM: max_positions must be positive")
    if tail[12] != 0 and norm_code != 1:
        raise Error("CausalLM: norm_bias=True needs norm='layernorm'")
    if tail[15] != 0 and norm_code == 1:
        raise Error("CausalLM: qk_norm=True with norm='layernorm' is not supported (qk_norm is an RMSNorm)")
    if r.window < 0:
        raise Error("CausalLM: sliding window must be nonnegative")
    var default_tail = [0x461c4000, 0, 0x3f800000, 0x3f800000, 0x40800000, 8192, 0, 8192, 0, 0, 0, 0x358637bd, 0, 0, 0, 0, 0]
    var extended = False
    for i in range(17):  # driver-loop: over the fixed block options record
        extended = extended or tail[i] != default_tail[i]
    var flags = [tail[8], tail[8], tail[8], tail[9], tail[12], tail[12], tail[14], tail[14], tail[14] if gated else 0, tail[15], tail[15]]
    var widths = [nh * hd, nkv * hd, nkv * hd, dm, dm, dm, it, dm, it, hd, hd]
    for i in range(CAUSAL_OPTIONAL_COUNT):  # driver-loop: over optional block tensors
        words.optional_flags[i] = flags[i] != 0
        words.optional_widths[i] = widths[i]
        if r.optional_present[i] and flags[i] == 0:
            words.refusal = CAUSAL_BLOCK_OPTIONAL_PRESENT
            words.refusal_index = i
            return words^
        if not r.optional_present[i] and flags[i] != 0:
            words.refusal = CAUSAL_BLOCK_OPTIONAL_MISSING
            words.refusal_index = i
            return words^
    words.n_kv_heads = nkv
    words.head_dim = hd
    words.intermediate = it
    words.rope_dim = rd
    words.gated = gated
    words.extended = extended
    words.tail = tail^
    words.shapes = [[dm], [dm], [nh * hd, dm], [nkv * hd, dm], [nkv * hd, dm], [dm, nh * hd], [it, dm], [it, dm], [dm, it]]
    return words^


struct CausalRecurrentWords(Movable):
    """A Mamba layer's configuration words and base weight shapes."""
    var d_inner: Int
    var dt_rank: Int
    var nheads: Int
    var conv_dim: Int
    var d_in_proj: Int
    var shapes: List[List[Int]]

    def __init__(out self):
        self.d_inner = 0
        self.dt_rank = 0
        self.nheads = 0
        self.conv_dim = 0
        self.d_in_proj = 0
        self.shapes = List[List[Int]]()


def causal_recurrent_block(kind: String, dm: Int) raises -> CausalRecurrentWords:
    """Configuration words and weight shapes of a mamba1 / mamba2 / mamba3 layer."""
    var words = CausalRecurrentWords()
    var di = 2 * dm
    words.d_inner = di
    if kind == "mamba1":
        var rank = (dm + 15) // 16
        words.dt_rank = rank
        words.shapes = [[dm], [2 * di, dm], [di, 1, 4], [di], [rank + 32, di], [di, rank], [di], [di, 16], [di], [dm, di]]
    elif kind == "mamba2":
        if dm % 32 != 0:
            raise Error("CausalLM: Mamba2 d_model must be a multiple of 32")
        var nh = di // 64
        var cd = di + 256
        words.nheads = nh
        words.conv_dim = cd
        words.shapes = [[dm], [2 * di + 256 + nh, dm], [cd, 1, 4], [cd], [nh], [nh], [nh], [di], [dm, di]]
    elif kind == "mamba3":
        if dm % 32 != 0:
            raise Error("Mamba3Block: d_model must be a multiple of 32")
        var nh = di // 64
        var dip = 2 * di + 256 + 3 * nh + 32
        words.nheads = nh
        words.d_in_proj = dip
        words.shapes = [[dm], [dip, dm], [nh], [128], [128], [nh, 128], [nh, 128], [nh], [dm, di]]
    else:
        raise Error("neural block: unsupported block kind " + kind)
    return words^


def causal_row_slice_copy(dst: Int, src: Int, rows: Int, cols: Int, begin: Int, end: Int) raises:
    """Copy rows [begin, end) of a row-major float32 [rows, cols] tensor to `dst`."""
    if begin < 0 or end < begin or end > rows:
        raise Error("row slice out of range")
    if end > begin:
        unsafe_memcpy(dest=f32_ptr(dst), src=f32_ptr(src) + begin * cols, count=(end - begin) * cols)
