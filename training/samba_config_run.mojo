# SPDX-License-Identifier: Apache-2.0
"""RUN half of the Samba configuration door (bindings/samba_config_native_door.mojo):
the configuration check with its defaults, and the names and shapes of one
block's tensors. Typed Mojo only, no Python import
(tools/python_free_dirs.sh).

Shape metadata of one configuration: a handful of integers per call, with no
independent units to split.
"""

comptime SAMBA_LAYER_MAMBA3 = 0
comptime SAMBA_LAYER_ATTENTION = 1


@fieldwise_init
struct SambaShape(Copyable, Movable):
    """A Samba configuration as typed words. The four attention fields are
    optional in the public constructor: `has_*` says whether one was given
    (or defaulted by `samba_shape_check`)."""
    var vocab: Int
    var d_model: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var intermediate: Int
    var has_n_heads: Bool
    var has_n_kv_heads: Bool
    var has_head_dim: Bool
    var has_intermediate: Bool
    var tie_embeddings: Bool
    var norm_eps: Float64
    var dropout: Float64


def samba_layer_code(kind: String) raises -> Int:
    """The layer word of a configured kind name, refused in the
    constructor's words for any other spelling."""
    if kind == "mamba3":
        return SAMBA_LAYER_MAMBA3
    if kind == "attention":
        return SAMBA_LAYER_ATTENTION
    raise Error("SambaConfig: layer kind must be mamba3 or attention")


def samba_layer_name(code: Int) -> String:
    return String("mamba3") if code == SAMBA_LAYER_MAMBA3 else String("attention")


def samba_shape_check(mut shape: SambaShape, layers: List[Int]) raises:
    """Check one configuration and fill its defaults (`head_dim`,
    `n_kv_heads`), in the constructor's order and words. The two float
    settings are stored at float32 precision."""
    shape.norm_eps = Float64(Float32(shape.norm_eps))
    shape.dropout = Float64(Float32(shape.dropout))
    var attention = False
    var mamba = False
    for i in range(len(layers)):  # driver-loop: over configured model layer kinds
        attention = attention or layers[i] == SAMBA_LAYER_ATTENTION
        mamba = mamba or layers[i] == SAMBA_LAYER_MAMBA3
    var dm = shape.d_model
    if shape.vocab < 2 or dm < 1 or len(layers) < 1:
        raise Error("SambaConfig: vocab >= 2, d_model >= 1 and at least one layer are required")
    # Mamba3's published profile fixes head width 64 and expansion 2;
    # this shape constraint is unrelated to benchmark dimensions.
    if mamba and dm % 32 != 0:
        raise Error("SambaConfig: a mamba3 layer needs d_model to be a multiple of 32")
    if attention:
        if not shape.has_n_heads or not shape.has_intermediate:
            raise Error("SambaConfig: an attention layer needs n_heads and intermediate")
        var heads = shape.n_heads
        if heads < 1:
            raise Error("SambaConfig: n_heads must be positive")
        if not shape.has_head_dim:
            shape.head_dim = dm // heads
            shape.has_head_dim = True
        if not shape.has_n_kv_heads:
            shape.n_kv_heads = heads
            shape.has_n_kv_heads = True
        if heads * shape.head_dim != dm:
            raise Error("SambaConfig: d_model must equal n_heads * head_dim")
    if not (shape.dropout >= 0.0 and shape.dropout < 1.0):
        raise Error("SambaConfig: dropout must be in [0, 1)")
    if not (shape.norm_eps >= 0.0):
        raise Error("SambaConfig: norm_eps must be >= 0")


def samba_block_names(kind: Int) -> List[String]:
    """The tensor names of one block, in the block's weight order."""
    if kind == SAMBA_LAYER_MAMBA3:
        return [String("block_norm.weight"), String("in_proj.weight"), String("dt_bias"),
                String("B_norm.weight"), String("C_norm.weight"), String("B_bias"), String("C_bias"),
                String("D"), String("out_proj.weight")]
    return [String("input_layernorm.weight"), String("post_attention_layernorm.weight"),
            String("q_proj.weight"), String("k_proj.weight"), String("v_proj.weight"), String("o_proj.weight"),
            String("gate_proj.weight"), String("up_proj.weight"), String("down_proj.weight")]


def samba_block_dims(shape: SambaShape, kind: Int) -> List[List[Int]]:
    """The tensor shapes of one block, in `samba_block_names` order. A
    mamba3 block reads `d_model` only; an attention block reads the four
    attention fields too."""
    var result = List[List[Int]]()
    var dm = shape.d_model
    if kind == SAMBA_LAYER_MAMBA3:
        var di = 2 * dm
        var heads = di // 64
        result.append([dm])
        result.append([2 * di + 256 + 3 * heads + 32, dm])
        result.append([heads])
        result.append([128])
        result.append([128])
        result.append([heads, 128])
        result.append([heads, 128])
        result.append([heads])
        result.append([dm, di])
    else:
        var qw = shape.n_heads * shape.head_dim
        var kw = shape.n_kv_heads * shape.head_dim
        var it = shape.intermediate
        result.append([dm])
        result.append([dm])
        result.append([qw, dm])
        result.append([kw, dm])
        result.append([kw, dm])
        result.append([dm, qw])
        result.append([it, dm])
        result.append([it, dm])
        result.append([dm, it])
    return result^
