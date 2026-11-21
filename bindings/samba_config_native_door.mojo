# SPDX-License-Identifier: Apache-2.0
"""Native Samba configuration and ordered parameter registry metadata.

UNPACK (`_samba_config_unpack`, `_samba_shape_of`) turns the constructor
arguments or a constructed configuration into the typed `SambaShape`; RUN
(training/samba_config_run.mojo, no Python) checks it and derives the block
schemas; PACK (`_samba_config_pack`, `_samba_block_pack`, `samba_registry`)
writes the public objects. `samba_config_dict` / `samba_config_from_dict` are
field copies between two Python forms with no RUN step."""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.samba_admission_door import samba_admit
from bindings.py_balanced import py_call, py_item, py_setattr, py_setitem
from bindings.samba_keys_unpack import samba_keys, SAMBA_OPN_BLOCK_SHAPES, SAMBA_OPN_DICT, SAMBA_OPN_FROM_DICT, SAMBA_OPN_INITIALIZE, SAMBA_OPN_REGISTRY
from bindings.py_keys import py_op_code
# RUN: typed check and block schemas, no Python import.
from training.samba_config_run import (
    SambaShape, SAMBA_LAYER_MAMBA3, SAMBA_LAYER_ATTENTION, samba_layer_code, samba_layer_name,
    samba_shape_check, samba_block_names, samba_block_dims,
)


def _optional_int(value: PythonObject, bi: PythonObject) raises -> Tuple[Int, Bool]:
    """UNPACK: one optional integer constructor argument."""
    if value is None:
        return (0, False)
    return (Int(py=py_call(bi.int, value)), True)


def _samba_config_unpack(args: PythonObject, mut layers: List[Int], bi: PythonObject) raises -> SambaShape:
    """UNPACK: the ten constructor arguments as typed words, converted in
    the constructor's order (each conversion's own Python error, then the
    layer-kind refusal, exactly where they were raised before)."""
    var vocab = Int(py=py_call(bi.int, args[0]))
    var d_model = Int(py=py_call(bi.int, args[1]))
    var kinds = py_call(bi.list, args[2])
    for i in range(len(kinds)):  # driver-loop: over configured model layer kinds
        layers.append(samba_layer_code(String(py_call(bi.str, kinds[i]))))
    var n_heads = _optional_int(args[3], bi)
    var n_kv_heads = _optional_int(args[4], bi)
    var head_dim = _optional_int(args[5], bi)
    var intermediate = _optional_int(args[6], bi)
    var tie = Bool(py=py_call(bi.bool, args[7]))
    var norm_eps = Float64(py=py_call(bi.float, args[8]))
    var dropout = Float64(py=py_call(bi.float, args[9]))
    return SambaShape(vocab, d_model, n_heads[0], n_kv_heads[0], head_dim[0], intermediate[0],
        n_heads[1], n_kv_heads[1], head_dim[1], intermediate[1], tie, norm_eps, dropout)


def _samba_config_pack(config: PythonObject, shape: SambaShape, layers: List[Int], bi: PythonObject) raises:
    """PACK: publish a checked configuration on its object. Nothing is
    written before the check passed."""
    var kinds = bi.list()
    for i in range(len(layers)):  # driver-loop: over configured model layer kinds
        _ = py_call(kinds.append, PythonObject(samba_layer_name(layers[i])))
    py_setattr(config, "vocab", PythonObject(shape.vocab))
    py_setattr(config, "d_model", PythonObject(shape.d_model))
    py_setattr(config, "layers", py_call(bi.tuple, kinds))
    py_setattr(config, "n_heads", PythonObject(shape.n_heads) if shape.has_n_heads else PythonObject(None))
    py_setattr(config, "n_kv_heads", PythonObject(shape.n_kv_heads) if shape.has_n_kv_heads else PythonObject(None))
    py_setattr(config, "head_dim", PythonObject(shape.head_dim) if shape.has_head_dim else PythonObject(None))
    py_setattr(config, "intermediate", PythonObject(shape.intermediate) if shape.has_intermediate else PythonObject(None))
    py_setattr(config, "tie_embeddings", PythonObject(shape.tie_embeddings))
    py_setattr(config, "norm_eps", PythonObject(shape.norm_eps))
    py_setattr(config, "dropout", PythonObject(shape.dropout))


def samba_config_init(config: PythonObject, args: PythonObject) raises -> PythonObject:
    # UNPACK
    var bi = native_loaded_module("builtins")
    var layers = List[Int]()
    var shape = _samba_config_unpack(args, layers, bi)
    # RUN (training/samba_config_run.mojo): the check and the defaults.
    samba_shape_check(shape, layers)
    # PACK
    _samba_config_pack(config, shape, layers, bi)
    return PythonObject(None)


def samba_config_dict(config: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var result = Python.dict()
    py_setitem(result, nk[].k_vocab, config.vocab)
    py_setitem(result, nk[].k_d_model, config.d_model)
    py_setitem(result, nk[].k_layers, py_call(native_loaded_module("builtins").list, config.layers))
    py_setitem(result, nk[].k_n_heads, config.n_heads)
    py_setitem(result, nk[].k_n_kv_heads, config.n_kv_heads)
    py_setitem(result, nk[].k_head_dim, config.head_dim)
    py_setitem(result, nk[].k_intermediate, config.intermediate)
    py_setitem(result, nk[].k_tie_embeddings, config.tie_embeddings)
    py_setitem(result, nk[].k_norm_eps, config.norm_eps)
    py_setitem(result, nk[].k_dropout, config.dropout)
    return result


def samba_config_from_dict(cls: PythonObject, value: PythonObject) raises -> PythonObject:
    var nk = samba_keys()
    var bi = native_loaded_module("builtins")
    var args = bi.list()
    # The fixed configuration keys, in constructor order (held keys: no str
    # build or key list per call).
    _ = py_call(args.append, py_item(value, nk[].k_vocab))
    _ = py_call(args.append, py_item(value, nk[].k_d_model))
    _ = py_call(args.append, py_item(value, nk[].k_layers))
    _ = py_call(args.append, py_item(value, nk[].k_n_heads))
    _ = py_call(args.append, py_item(value, nk[].k_n_kv_heads))
    _ = py_call(args.append, py_item(value, nk[].k_head_dim))
    _ = py_call(args.append, py_item(value, nk[].k_intermediate))
    _ = py_call(args.append, py_item(value, nk[].k_tie_embeddings))
    _ = py_call(args.append, py_item(value, nk[].k_norm_eps))
    _ = py_call(args.append, py_item(value, nk[].k_dropout))
    var result = py_call(bi.object.__new__, cls)
    _ = samba_config_init(result, args)
    return result


def _samba_registry_item(name: String, dims: List[Int], tuple_type: PythonObject) raises -> PythonObject:
    var values = Python.list()
    for dimension in dims:  # driver-loop: over tensor shape metadata
        _ = py_call(values.append, dimension)
    return Python.tuple(PythonObject(name), py_call(tuple_type, values))


def _samba_shape_of(config: PythonObject, attention: Bool, whole: Bool, mut layers: List[Int]) raises -> SambaShape:
    """UNPACK: the typed words of a constructed configuration that one
    operation reads. A mamba3 block reads `d_model` only; the attention
    fields are read (and their conversion refused) only when an attention
    block is asked for, as before; `whole` adds the registry's fields."""
    var shape = SambaShape(0, Int(py=config.d_model), 0, 0, 0, 0, False, False, False, False, True, 0.0, 0.0)
    var wants_attention = attention
    if whole:
        shape.vocab = Int(py=config.vocab)
        var kinds = config.layers
        for i in range(len(kinds)):  # driver-loop: over configured model layers
            # Any kind that is not mamba3 has the attention schema, as before.
            var code = SAMBA_LAYER_MAMBA3 if String(kinds[i]) == "mamba3" else SAMBA_LAYER_ATTENTION
            wants_attention = wants_attention or code == SAMBA_LAYER_ATTENTION
            layers.append(code)
    if wants_attention:
        shape.n_heads = Int(py=config.n_heads)
        shape.head_dim = Int(py=config.head_dim)
        shape.n_kv_heads = Int(py=config.n_kv_heads)
        shape.intermediate = Int(py=config.intermediate)
    if whole:
        shape.tie_embeddings = Bool(py=config.tie_embeddings)
    return shape.copy()


def _samba_block_pack(shape: SambaShape, kind: Int, tuple_type: PythonObject) raises -> PythonObject:
    """RUN + PACK of one block schema: the typed names and shapes
    (training/samba_config_run.mojo) as the public list of (name, shape)."""
    var names = samba_block_names(kind)
    var dims = samba_block_dims(shape, kind)
    var result = Python.list()
    for j in range(len(names)):  # driver-loop: over one block's tensor registry metadata
        _ = py_call(result.append, _samba_registry_item(names[j], dims[j], tuple_type))
    return result


def samba_block_shapes(config: PythonObject, kind: String) raises -> PythonObject:
    var unused = List[Int]()
    var code = SAMBA_LAYER_MAMBA3 if kind == "mamba3" else SAMBA_LAYER_ATTENTION
    var shape = _samba_shape_of(config, code == SAMBA_LAYER_ATTENTION, False, unused)
    return _samba_block_pack(shape, code, native_loaded_module("builtins").tuple)


def samba_registry(config: PythonObject) raises -> PythonObject:
    # Shared by every registry item and both lazily prepared block schemas.
    var tuple_type = native_loaded_module("builtins").tuple
    # UNPACK: the configuration's words, once for the whole registry.
    var layers = List[Int]()
    var shape = _samba_shape_of(config, False, True, layers)
    var dm = shape.d_model
    var vocab = shape.vocab
    var result = Python.list(_samba_registry_item("embed.weight", [vocab, dm], tuple_type))
    # Every layer of one kind has the same parameter schema; build each kind
    # on first use. Only immutable shape tuples escape into distinct
    # per-layer records. A later public call builds new schemas from its
    # newly admitted config.
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var mamba_shapes = Optional[PythonObject]()
    var attention_shapes = Optional[PythonObject]()
    for i in range(len(layers)):  # driver-loop: over configured model layers
        var entries = PythonObject(None)
        if layers[i] == SAMBA_LAYER_MAMBA3:
            if not mamba_shapes:
                mamba_shapes = _samba_block_pack(shape, SAMBA_LAYER_MAMBA3, tuple_type)
            entries = mamba_shapes.value()
        else:
            if not attention_shapes:
                attention_shapes = _samba_block_pack(shape, SAMBA_LAYER_ATTENTION, tuple_type)
            entries = attention_shapes.value()
        var prefix = "layers." + String(i) + "."
        for j in range(len(entries)):  # driver-loop: over per-layer tensor registry metadata
            var entry = entries[j]
            _ = py_call(result.append, Python.tuple(PythonObject(prefix + String(entry[0])), entry[1]))
    _ = py_call(result.append, _samba_registry_item("norm_f.weight", [dm], tuple_type))
    if not shape.tie_embeddings:
        _ = py_call(result.append, _samba_registry_item("lm_head.weight", [vocab, dm], tuple_type))
    return result


def samba_config_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        var nk = samba_keys()
        var operation = py_op_code(nk[].ops, args[0])
        var values = args[1]
        samba_admit(operation, values, args[2])
        var result = PythonObject(None)
        if operation == SAMBA_OPN_INITIALIZE:
            result = samba_config_init(values[0], values[1])
        elif operation == SAMBA_OPN_DICT:
            result = samba_config_dict(values[0])
        elif operation == SAMBA_OPN_FROM_DICT:
            result = samba_config_from_dict(values[0], values[1])
        elif operation == SAMBA_OPN_BLOCK_SHAPES:
            result = samba_block_shapes(values[0], String(values[1]))
        elif operation == SAMBA_OPN_REGISTRY:
            result = samba_registry(values[0])
        else:
            raise Error("SambaConfig: unknown native operation " + String(args[0]))
        return result.steal_data()
    except error:
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
