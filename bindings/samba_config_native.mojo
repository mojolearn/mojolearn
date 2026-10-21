# SPDX-License-Identifier: Apache-2.0
"""Native Samba configuration and ordered parameter registry metadata."""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.samba_admission import samba_admit


def samba_config_init(config: PythonObject, args: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    config.vocab = bi.int(args[0])
    config.d_model = bi.int(args[1])
    var kinds = bi.list(args[2])
    var layers = bi.list()
    var attention = False
    var mamba = False
    for i in range(len(kinds)):  # driver-loop: over configured model layer kinds
        var kind = bi.str(kinds[i])
        if String(kind) != "mamba3" and String(kind) != "attention":
            raise Error("SambaConfig: layer kind must be mamba3 or attention")
        attention = attention or String(kind) == "attention"
        mamba = mamba or String(kind) == "mamba3"
        layers.append(kind)
    config.layers = bi.tuple(layers)
    config.n_heads = PythonObject(None) if args[3] is None else bi.int(args[3])
    config.n_kv_heads = PythonObject(None) if args[4] is None else bi.int(args[4])
    config.head_dim = PythonObject(None) if args[5] is None else bi.int(args[5])
    config.intermediate = PythonObject(None) if args[6] is None else bi.int(args[6])
    config.tie_embeddings = bi.bool(args[7])
    config.norm_eps = PythonObject(Float64(Float32(Float64(py=bi.float(args[8])))))
    config.dropout = PythonObject(Float64(Float32(Float64(py=bi.float(args[9])))))
    var dm = Int(py=config.d_model)
    if Int(py=config.vocab) < 2 or dm < 1 or len(layers) < 1:
        raise Error("SambaConfig: vocab >= 2, d_model >= 1 and at least one layer are required")
    # Mamba3's published profile fixes head width 64 and expansion 2;
    # this shape constraint is unrelated to benchmark dimensions.
    if mamba and dm % 32 != 0:
        raise Error("SambaConfig: a mamba3 layer needs d_model to be a multiple of 32")
    if attention:
        if config.n_heads is None or config.intermediate is None:
            raise Error("SambaConfig: an attention layer needs n_heads and intermediate")
        var heads = Int(py=config.n_heads)
        if heads < 1:
            raise Error("SambaConfig: n_heads must be positive")
        if config.head_dim is None:
            config.head_dim = PythonObject(dm // heads)
        if config.n_kv_heads is None:
            config.n_kv_heads = config.n_heads
        if heads * Int(py=config.head_dim) != dm:
            raise Error("SambaConfig: d_model must equal n_heads * head_dim")
    var drop = Float64(py=config.dropout)
    if not (drop >= 0.0 and drop < 1.0):
        raise Error("SambaConfig: dropout must be in [0, 1)")
    if not (Float64(py=config.norm_eps) >= 0.0):
        raise Error("SambaConfig: norm_eps must be >= 0")
    return PythonObject(None)


def samba_config_dict(config: PythonObject) raises -> PythonObject:
    var result = Python.dict()
    result["vocab"] = config.vocab
    result["d_model"] = config.d_model
    result["layers"] = Python.import_module("builtins").list(config.layers)
    result["n_heads"] = config.n_heads
    result["n_kv_heads"] = config.n_kv_heads
    result["head_dim"] = config.head_dim
    result["intermediate"] = config.intermediate
    result["tie_embeddings"] = config.tie_embeddings
    result["norm_eps"] = config.norm_eps
    result["dropout"] = config.dropout
    return result


def samba_config_from_dict(cls: PythonObject, value: PythonObject) raises -> PythonObject:
    var bi = Python.import_module("builtins")
    var args = bi.list()
    for name in [String("vocab"), String("d_model"), String("layers"), String("n_heads"), String("n_kv_heads"), String("head_dim"), String("intermediate"), String("tie_embeddings"), String("norm_eps"), String("dropout")]:  # driver-loop: over fixed configuration keys
        args.append(value[name])
    var result = bi.object.__new__(cls)
    _ = samba_config_init(result, args)
    return result


def _samba_registry_item(name: String, dims: List[Int]) raises -> PythonObject:
    var values = Python.list()
    for dimension in dims:  # driver-loop: over tensor shape metadata
        values.append(dimension)
    return Python.tuple(PythonObject(name), Python.import_module("builtins").tuple(values))


def samba_block_shapes(config: PythonObject, kind: String) raises -> PythonObject:
    var result = Python.list()
    var dm = Int(py=config.d_model)
    if kind == "mamba3":
        var di = 2 * dm
        var heads = di // 64
        result.append(_samba_registry_item("block_norm.weight", [dm]))
        result.append(_samba_registry_item("in_proj.weight", [2 * di + 256 + 3 * heads + 32, dm]))
        result.append(_samba_registry_item("dt_bias", [heads]))
        result.append(_samba_registry_item("B_norm.weight", [128]))
        result.append(_samba_registry_item("C_norm.weight", [128]))
        result.append(_samba_registry_item("B_bias", [heads, 128]))
        result.append(_samba_registry_item("C_bias", [heads, 128]))
        result.append(_samba_registry_item("D", [heads]))
        result.append(_samba_registry_item("out_proj.weight", [dm, di]))
    else:
        var qw = Int(py=config.n_heads) * Int(py=config.head_dim)
        var kw = Int(py=config.n_kv_heads) * Int(py=config.head_dim)
        var it = Int(py=config.intermediate)
        result.append(_samba_registry_item("input_layernorm.weight", [dm]))
        result.append(_samba_registry_item("post_attention_layernorm.weight", [dm]))
        result.append(_samba_registry_item("q_proj.weight", [qw, dm]))
        result.append(_samba_registry_item("k_proj.weight", [kw, dm]))
        result.append(_samba_registry_item("v_proj.weight", [kw, dm]))
        result.append(_samba_registry_item("o_proj.weight", [dm, qw]))
        result.append(_samba_registry_item("gate_proj.weight", [it, dm]))
        result.append(_samba_registry_item("up_proj.weight", [it, dm]))
        result.append(_samba_registry_item("down_proj.weight", [dm, it]))
    return result


def samba_registry(config: PythonObject) raises -> PythonObject:
    var dm = Int(py=config.d_model)
    var vocab = Int(py=config.vocab)
    var result = Python.list(_samba_registry_item("embed.weight", [vocab, dm]))
    for i in range(len(config.layers)):  # driver-loop: over configured model layers
        var entries = samba_block_shapes(config, String(config.layers[i]))
        for j in range(len(entries)):  # driver-loop: over per-layer tensor registry metadata
            result.append(Python.tuple(PythonObject("layers." + String(i) + "." + String(entries[j][0])), entries[j][1]))
    result.append(_samba_registry_item("norm_f.weight", [dm]))
    if not Bool(py=config.tie_embeddings):
        result.append(_samba_registry_item("lm_head.weight", [vocab, dm]))
    return result


def samba_config_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var operation = String(args[0])
        var values = args[1]
        samba_admit(operation, values, args[2])
        var result = PythonObject(None)
        if operation == "initialize":
            result = samba_config_init(values[0], values[1])
        elif operation == "dict":
            result = samba_config_dict(values[0])
        elif operation == "from_dict":
            result = samba_config_from_dict(values[0], values[1])
        elif operation == "block_shapes":
            result = samba_block_shapes(values[0], String(values[1]))
        elif operation == "registry":
            result = samba_registry(values[0])
        else:
            raise Error("SambaConfig: unknown native operation " + operation)
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error, ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error, ExceptionType("PyExc_NotImplementedError"))
        return raise_python_exception(error, ExceptionType.ValueError)
