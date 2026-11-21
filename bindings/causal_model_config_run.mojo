# SPDX-License-Identifier: Apache-2.0
"""Typed planning of a causal model configuration (RUN, no Python).

`causal_config_plan` decides a `ModelPlan` from the consumed configuration
fields: the family refusals by name, the integer and number checks, the
attention head arithmetic, the block options, the Mamba profile constants
and the phi3 fused-projection row split. `causal_layer_weights` expands one
layer's checkpoint names; `causal_checkpoint_names` enumerates every
checkpoint name of a plan in insertion order without duplicates.

The boundary door (`bindings/causal_model_config_door.mojo`) is UNPACK and
PACK: it reads each consumed field of the captured configuration once into a
`CausalConfigValue` (its exact JSON type, value and truth), and builds the
Python plan from the `CausalConfigPlan` this file returns. A refusal by name
is returned as words (`refused`, field, reason) because its message quotes
the Python repr of the field's value, which PACK renders.

Parallel plan: none. One configuration is planned per model construction and
each step reads the previous step's words (serial by nature, a few dozen
comparisons); the checkpoint name enumeration is a dedupe in insertion order,
one pass over layers x entries.
"""
from std.collections import Set
from std.math import inf

comptime CONFIG_ABSENT = 0
comptime CONFIG_NULL = 1
comptime CONFIG_BOOL = 2
comptime CONFIG_INT = 3
comptime CONFIG_FLOAT = 4
comptime CONFIG_STR = 5
comptime CONFIG_OTHER = 6

# Layer tables connected at import (`config.py` `_NATIVE_CONTEXT`).
comptime CONFIG_TABLE_LLAMA = 0
comptime CONFIG_TABLE_QKV_BIAS = 1
comptime CONFIG_TABLE_QK_NORM = 2
comptime CONFIG_TABLE_MAMBA1 = 3
comptime CONFIG_TABLE_MAMBA2 = 4
comptime CONFIG_TABLE_BYTE = 5


struct CausalConfigValue(Copyable, Movable):
    """One configuration field as UNPACK read it: its exact JSON type (`tag`),
    its integer / number / text value and its Python truth."""
    var tag: Int
    var i: Int
    var f: Float64
    var s: String
    var truthy: Bool

    def __init__(out self):
        self.tag = CONFIG_ABSENT
        self.i = 0
        self.f = 0.0
        self.s = String()
        self.truthy = False

    def __init__(out self, tag: Int, i: Int, f: Float64, var s: String, truthy: Bool):
        self.tag = tag
        self.i = i
        self.f = f
        self.s = s^
        self.truthy = truthy

    def present(self) -> Bool:
        """Python `data.get(field) is not None`."""
        return self.tag != CONFIG_ABSENT and self.tag != CONFIG_NULL

    def numeric(self) -> Bool:
        return self.tag == CONFIG_BOOL or self.tag == CONFIG_INT or self.tag == CONFIG_FLOAT


def config_equal_int(v: CausalConfigValue, n: Int) -> Bool:
    """Python `v == n` for a JSON value and an int (bool and float compare by value)."""
    if v.tag == CONFIG_BOOL or v.tag == CONFIG_INT:
        return v.i == n
    if v.tag == CONFIG_FLOAT:
        return v.f == Float64(n)
    return False


def config_equal(a: CausalConfigValue, b: CausalConfigValue) -> Bool:
    """Python `a == b` for two JSON scalars (lists and dicts never equal a scalar)."""
    if a.numeric() and b.numeric():
        if a.tag != CONFIG_FLOAT and b.tag != CONFIG_FLOAT:
            return a.i == b.i
        return a.f == b.f
    if a.tag == CONFIG_STR and b.tag == CONFIG_STR:
        return a.s == b.s
    if a.tag == CONFIG_NULL and b.tag == CONFIG_NULL:
        return True
    return False


struct CausalLayerEntry(Copyable, Movable):
    """One `(block_key, checkpoint_name, row_slice_or_None)` layer map entry."""
    var key: String
    var name: String
    var has_rows: Bool
    var begin: Int
    var end: Int

    def __init__(out self, var key: String, var name: String, has_rows: Bool, begin: Int, end: Int):
        self.key = key^
        self.name = name^
        self.has_rows = has_rows
        self.begin = begin
        self.end = end


@fieldwise_init
struct CausalFixedField(Copyable, Movable):
    """A byte-LM config field and the one value the stack spells."""
    var name: String
    var value: CausalConfigValue


struct CausalConfigTables(Defaultable, Movable):
    """The static layer tables and byte-LM fixed fields, typed once at connect."""
    var tables: List[List[CausalLayerEntry]]
    var byte_fixed: List[CausalFixedField]

    def __init__(out self):
        self.tables = List[List[CausalLayerEntry]]()
        for _ in range(6):  # small-loop(the six connected layer tables)
            self.tables.append(List[CausalLayerEntry]())
        self.byte_fixed = List[CausalFixedField]()


struct CausalConfigRequest(Movable):
    """The consumed fields of one configuration, read once by UNPACK.

    `fields` holds every consumed name of `source` (the configuration, or
    for the byte-LM format its renamed HF view); `fixed_got` the byte-LM
    fixed names as the configuration spells them; `present` the names whose
    key exists (for gemma's `hidden_activation` choice)."""
    var model_type: String
    var byte_lm: Bool
    var fields: Dict[String, CausalConfigValue]
    var fixed_got: Dict[String, CausalConfigValue]
    var hidden_activation_key: Bool
    var layer_kinds: Int
    var dt_limit_given: Bool
    var dt_lo: Float64
    var dt_hi: Float64

    def __init__(out self):
        self.model_type = String()
        self.byte_lm = False
        self.fields = Dict[String, CausalConfigValue]()
        self.fixed_got = Dict[String, CausalConfigValue]()
        self.hidden_activation_key = False
        self.layer_kinds = 0
        self.dt_limit_given = False
        self.dt_lo = 0.0
        self.dt_hi = 0.0

    def get(self, name: String) -> CausalConfigValue:
        var found = self.fields.get(name)
        if found:
            return found.value().copy()
        return CausalConfigValue()


struct CausalConfigPlan(Movable):
    """The plan words PACK publishes as the `ModelPlan` fields."""
    var refused: Bool
    var refuse_field: String
    var refuse_why: String
    # When set, PACK appends repr(source.get(refuse_suffix_field)) + ")" to the reason.
    var refuse_suffix_field: String
    var kind: String
    var transformer: Bool
    var n_layers: Int
    var d_model: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var intermediate: Int
    var vocab_size: Int
    var tied: Bool
    var norm_eps: Float64
    var max_positions: Int
    # Block options (transformer): -1 publishes None.
    var opt_n_kv_heads: Int
    var opt_head_dim: Int
    var window: Int
    var rope_theta: Float64
    var qkv_bias: Bool
    var mlp_bias: Bool
    var qk_norm: Bool
    # Block kwargs (Mamba-2).
    var has_dt_limit: Bool
    var dt_lo: Float64
    var dt_hi: Float64
    var layer_map: List[CausalLayerEntry]
    var embed_name: String
    var norm_name: String
    var has_norm_name: Bool
    var head_name: String
    var has_head_name: Bool

    def __init__(out self):
        self.refused = False
        self.refuse_field = String()
        self.refuse_why = String()
        self.refuse_suffix_field = String()
        self.kind = String()
        self.transformer = False
        self.n_layers = 0
        self.d_model = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.head_dim = 0
        self.intermediate = 0
        self.vocab_size = 0
        self.tied = False
        self.norm_eps = 0.0
        self.max_positions = 0
        self.opt_n_kv_heads = -1
        self.opt_head_dim = -1
        self.window = 0
        self.rope_theta = 0.0
        self.qkv_bias = False
        self.mlp_bias = False
        self.qk_norm = False
        self.has_dt_limit = False
        self.dt_lo = 0.0
        self.dt_hi = 0.0
        self.layer_map = List[CausalLayerEntry]()
        self.embed_name = String()
        self.norm_name = String()
        self.has_norm_name = False
        self.head_name = String()
        self.has_head_name = False

    def refuse(mut self, field: String, why: String, suffix_field: String):
        self.refused = True
        self.refuse_field = field
        self.refuse_why = why
        self.refuse_suffix_field = suffix_field

    def refuse(mut self, field: String, why: String):
        self.refuse(field, why, String())


def _config_int(r: CausalConfigRequest, field: String, default: Int, has_default: Bool) raises -> Int:
    # `_config_int` of the door before this split: an absent field takes the
    # default; a null or absent-without-default field is required; the value
    # must be an exact int (a bool is refused).
    var v = r.get(field)
    if v.tag == CONFIG_ABSENT and has_default:
        return default
    if v.tag == CONFIG_ABSENT or v.tag == CONFIG_NULL:
        raise Error("model config: " + field + ": the field is required and the config lacks it")
    if v.tag != CONFIG_INT:
        raise Error("model config: " + field + ": an integer is required")
    return v.i


def _config_int(r: CausalConfigRequest, field: String) raises -> Int:
    return _config_int(r, field, 0, False)


def _config_number(r: CausalConfigRequest, field: String, default: Float64) raises -> Float64:
    var v = r.get(field)
    if v.tag == CONFIG_ABSENT:
        return default
    if v.tag == CONFIG_INT:
        return Float64(v.i)
    if v.tag == CONFIG_FLOAT:
        return v.f
    raise Error("model config: " + field + " requires a number")


def _config_truth(r: CausalConfigRequest, field: String, default: Bool) -> Bool:
    var v = r.get(field)
    if v.tag == CONFIG_ABSENT:
        return default
    return v.truthy


def _config_text_is(r: CausalConfigRequest, field: String, default: String, a: String, b: String) -> Bool:
    # Python `String(data.get(field, default)) in (a, b)`: only a JSON string can spell a or b.
    var v = r.get(field)
    if v.tag == CONFIG_ABSENT:
        return default == a or default == b
    return v.tag == CONFIG_STR and (v.s == a or v.s == b)


def _config_fixed(r: CausalConfigRequest, mut plan: CausalConfigPlan, names: List[String], values: List[Int]) -> Bool:
    for i in range(len(names)):  # small-loop(fixed config field words)
        var v = r.get(names[i])
        if v.tag != CONFIG_ABSENT and not config_equal_int(v, values[i]):
            plan.refuse(names[i], "fixed at " + String(values[i]))
            return False
    return True


def _config_transformer(r: CausalConfigRequest, tables: CausalConfigTables, mut plan: CausalConfigPlan) raises:
    var mt = r.model_type
    for field in [String("num_local_experts"), String("num_experts"), String("num_experts_per_tok"), String("n_routed_experts")]:  # small-loop(the four mixture-of-experts field names)
        var v = r.get(field)
        if v.present() and not config_equal_int(v, 0) and not config_equal_int(v, 1):
            plan.refuse(field, "mixture of experts")
            return
    if r.get("rope_scaling").present():
        plan.refuse("rope_scaling", "rope type default only")
        return
    var rt = r.get("rope_type")
    if rt.tag != CONFIG_ABSENT and rt.tag != CONFIG_NULL and not (rt.tag == CONFIG_STR and rt.s == "default"):
        plan.refuse("rope_type", "default only")
        return
    if not _config_text_is(r, "hidden_act", "silu", "silu", "swish"):
        plan.refuse("hidden_act", "requires silu/swish SwiGLU")
        return
    if _config_truth(r, "use_sliding_window", False):
        plan.refuse("use_sliding_window", "sliding-window-alternating layers (max_window_layers=", "max_window_layers")
        return
    if r.layer_kinds > 1:
        raise Error("model config: layer_types requests alternating attention kinds")
    var dm = _config_int(r, "hidden_size")
    var nh = _config_int(r, "num_attention_heads")
    var n_kv = _config_int(r, "num_key_value_heads", nh, True)
    if dm < 1 or nh < 1 or n_kv < 1:
        raise Error("model config: hidden_size and attention head counts must be positive")
    var hd_field = r.get("head_dim")
    var hd: Int
    if hd_field.tag == CONFIG_ABSENT or hd_field.tag == CONFIG_NULL:
        hd = dm // nh
    elif hd_field.tag != CONFIG_INT:
        raise Error("model config: head_dim: an integer is required")
    else:
        hd = hd_field.i
    if nh * hd != dm:
        raise Error("model config: head_dim * num_attention_heads must equal hidden_size")
    if nh % n_kv != 0:
        plan.refuse("num_key_value_heads", "num_attention_heads is not a multiple of it (must divide num_attention_heads)")
        return
    if hd % 2 != 0:
        raise Error("model config: head_dim must be even")
    var inter = _config_int(r, "intermediate_size")
    var layers = _config_int(r, "num_hidden_layers")
    var vocab = _config_int(r, "vocab_size")
    var eps = _config_number(r, "rms_norm_eps", 1e-6)
    var theta = _config_number(r, "rope_theta", 10000.0)
    var window_value = r.get("sliding_window")
    var window = 0
    if window_value.tag == CONFIG_INT:
        window = window_value.i
    elif window_value.tag != CONFIG_ABSENT and window_value.tag != CONFIG_NULL:
        raise Error("model config: sliding_window requires an integer width or null")
    if window < 0:
        raise Error("model config: sliding_window requires an integer width or null")
    var bias = _config_truth(r, "attention_bias", mt == "qwen2")
    var mlp_bias = _config_truth(r, "mlp_bias", False)
    var dropout = r.get("attention_dropout")
    if dropout.tag != CONFIG_ABSENT and dropout.tag != CONFIG_NULL and (not dropout.numeric() or dropout.f != 0):
        raise Error("model config: attention_dropout must be zero: inference has no dropout")
    var tied = _config_truth(r, "tie_word_embeddings", False)
    var maxpos = _config_int(r, "max_position_embeddings", 8192, True)
    plan.transformer = True
    plan.kind = "transformer"
    plan.n_layers = layers
    plan.d_model = dm
    plan.n_heads = nh
    plan.n_kv_heads = n_kv
    plan.head_dim = hd
    plan.intermediate = inter
    plan.vocab_size = vocab
    plan.tied = tied
    plan.norm_eps = eps
    plan.max_positions = maxpos
    plan.opt_n_kv_heads = n_kv if n_kv != nh else -1
    plan.opt_head_dim = hd if hd != dm // nh else -1
    plan.window = window
    plan.rope_theta = theta
    plan.qkv_bias = bias
    plan.mlp_bias = mlp_bias
    plan.qk_norm = mt == "qwen3"
    plan.layer_map = tables.tables[CONFIG_TABLE_LLAMA].copy()
    if bias:
        plan.layer_map.extend(tables.tables[CONFIG_TABLE_QKV_BIAS].copy())
    if mt == "qwen3":
        plan.layer_map.extend(tables.tables[CONFIG_TABLE_QK_NORM].copy())
    plan.embed_name = "model.embed_tokens.weight"
    plan.norm_name = "model.norm.weight"
    plan.has_norm_name = True
    plan.has_head_name = not tied
    plan.head_name = "lm_head.weight"


def _config_mamba(r: CausalConfigRequest, tables: CausalConfigTables, mut plan: CausalConfigPlan) raises:
    var two = r.model_type == "mamba2"
    var dm = _config_int(r, "hidden_size")
    var layers = _config_int(r, "num_hidden_layers")
    var vocab = _config_int(r, "vocab_size")
    if not _config_fixed(r, plan, ["state_size", "conv_kernel", "expand", "use_bias", "use_conv_bias"],
                         [128 if two else 16, 4, 2, 0, 1]):
        return
    if two:
        if not _config_fixed(r, plan, ["head_dim", "n_groups", "chunk_size", "rms_norm"], [64, 1, 256, 1]):
            return
        if _config_truth(r, "norm_before_gate", False):
            raise Error("model config: norm_before_gate must be false")
        if dm % 32 != 0:
            plan.refuse("hidden_size", "must be a multiple of 32")
            return
        var nh = r.get("num_heads")
        if nh.present() and not config_equal_int(nh, 2 * dm // 64):
            raise Error("model config: num_heads must equal expand*hidden_size/head_dim")
        plan.has_dt_limit = True
        if r.dt_limit_given:
            plan.dt_lo = r.dt_lo
            plan.dt_hi = r.dt_hi
        else:
            plan.dt_lo = 0.0
            plan.dt_hi = inf[DType.float64]()  # Python float("inf")
    else:
        var rank = r.get("time_step_rank")
        var auto = rank.tag == CONFIG_ABSENT or (rank.tag == CONFIG_STR and rank.s == "auto")
        if not auto and not config_equal_int(rank, (dm + 15) // 16):
            raise Error("model config: time_step_rank must be auto or ceil(hidden_size/16)")
        if not _config_text_is(r, "hidden_act", "silu", "silu", "silu"):
            raise Error("model config: hidden_act requires silu")
    var eps = _config_number(r, "layer_norm_epsilon", 1e-5)
    if eps != 1e-5:
        plan.refuse("layer_norm_epsilon", "fixed at 1e-5")
        return
    var tied = _config_truth(r, "tie_word_embeddings", True)
    plan.transformer = False
    plan.kind = "mamba2" if two else "mamba1"
    plan.n_layers = layers
    plan.d_model = dm
    plan.vocab_size = vocab
    plan.tied = tied
    plan.norm_eps = eps
    plan.layer_map = tables.tables[CONFIG_TABLE_MAMBA2 if two else CONFIG_TABLE_MAMBA1].copy()
    plan.embed_name = "backbone.embeddings.weight"
    plan.norm_name = "backbone.norm_f.weight"
    plan.has_norm_name = True
    plan.has_head_name = not tied
    plan.head_name = "lm_head.weight"


def _config_phi3(r: CausalConfigRequest, mut plan: CausalConfigPlan) raises:
    var original = r.get("original_max_position_embeddings")
    if original.present() and not config_equal_int(original, plan.max_positions):
        raise Error("model config: original_max_position_embeddings requests longrope extended context")
    var qr = plan.n_heads * plan.head_dim
    var kr = plan.n_kv_heads * plan.head_dim
    var it = plan.intermediate
    for ref entry in plan.layer_map:  # driver-loop: over model layer descriptors
        var key = entry.key
        var begin = -1
        var end = -1
        if key == "q_proj.weight" or key == "k_proj.weight" or key == "v_proj.weight":
            entry.name = "model.layers.{i}.self_attn.qkv_proj.weight"
            begin = 0 if key == "q_proj.weight" else (qr if key == "k_proj.weight" else qr + kr)
            end = qr if key == "q_proj.weight" else (qr + kr if key == "k_proj.weight" else qr + 2 * kr)
        elif key == "gate_proj.weight" or key == "up_proj.weight":
            entry.name = "model.layers.{i}.mlp.gate_up_proj.weight"
            begin = 0 if key == "gate_proj.weight" else it
            end = it if key == "gate_proj.weight" else 2 * it
        if begin >= 0:
            entry.has_rows = True
            entry.begin = begin
            entry.end = end


def causal_config_plan(r: CausalConfigRequest, tables: CausalConfigTables) raises -> CausalConfigPlan:
    """Plan one admitted configuration (model_type and family row already admitted)."""
    var plan = CausalConfigPlan()
    var mt = r.model_type
    if mt == "gemma2":
        for field in [String("attn_logit_softcapping"), String("final_logit_softcapping"), String("query_pre_attn_scalar")]:  # small-loop(the three fixed gemma2 field names)
            if r.get(field).present():
                plan.refuse(field, "unsupported for gemma2")
                return plan^
    if mt == "gemma" or mt == "gemma2":
        plan.refuse("hidden_activation" if r.hidden_activation_key else "hidden_act", "unsupported GeGLU, RMSNorm offset and embedding scale")
        return plan^
    if r.byte_lm:
        for ref fixed in tables.byte_fixed:  # small-loop(the fixed byte-LM config fields)
            var got = r.fixed_got.get(fixed.name)
            if got and got.value().tag != CONFIG_ABSENT and not config_equal(got.value(), fixed.value):
                raise Error("model config: byte-LM " + fixed.name + " differs from required value")
        _config_transformer(r, tables, plan)
        if plan.refused:
            return plan^
        plan.layer_map = tables.tables[CONFIG_TABLE_BYTE].copy()
        plan.embed_name = "embed"
        plan.has_norm_name = False
        plan.head_name = "lm_head"
        plan.has_head_name = True
    elif mt == "mamba" or mt == "mamba2":
        _config_mamba(r, tables, plan)
    else:
        _config_transformer(r, tables, plan)
        if not plan.refused and mt == "phi3":
            _config_phi3(r, plan)
    return plan^


def causal_layer_weights(entries: List[CausalLayerEntry], layer: Int) -> List[CausalLayerEntry]:
    """One layer's `(block_key, checkpoint_name, rows)` entries, `{i}` replaced by `layer`."""
    var result = List[CausalLayerEntry](capacity=len(entries))
    var index = String(layer)
    for ref entry in entries:  # driver-loop: expand admitted checkpoint names for one model layer
        result.append(CausalLayerEntry(entry.key, entry.name.replace("{i}", index), entry.has_rows, entry.begin, entry.end))
    return result^


def causal_checkpoint_names(outer: List[String], entries: List[CausalLayerEntry], n_layers: Int) -> List[String]:
    """Every checkpoint name of a plan: the outer names, then each layer's, first occurrence kept."""
    var names = List[String]()
    var seen = Set[String]()
    for ref name in outer:  # small-loop(three outer checkpoint weight names)
        names.append(name)
        seen.add(name)
    for i in range(n_layers):  # driver-loop: enumerate complete model checkpoint names
        var index = String(i)
        for ref entry in entries:  # driver-loop: deduplicate native checkpoint names in insertion order
            var name = entry.name.replace("{i}", index)
            if name not in seen:
                seen.add(name)
                names.append(name^)
    return names^
