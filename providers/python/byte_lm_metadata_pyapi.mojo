# SPDX-License-Identifier: Apache-2.0
# Shared ByteLM metadata snapshot for host/GPU frontends. Numeric dimensions,
# registry shapes/names and profile spelling are owned here after admission.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_contains, py_fields, py_setitem
from bindings.classical_pandas import classical_admit_value
from bindings.native_buffer_owner import native_buffer_view, native_storage_allocate
from bindings.classical_attributes import classical_connect_numpy
from core.training_state_validation import training_byte_shape_validate
from bindings.optimizer_public_words import stash_python_error


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")


# ---------------------------------------------------------------- the shape

struct _Shape(Copyable, Movable):
    """`ByteLanguageModelConfig`'s numbers: (batch, length, d_model, n_heads,
    n_kv, head_dim, intermediate, n_layers, vocab_size) and the chunked head
    flag."""
    var d: List[Int]
    var chunked: Bool

    def __init__(out self, var d: List[Int], chunked: Bool):
        self.d = d^
        self.chunked = chunked

    def batch(self) -> Int:
        return self.d[0]

    def length(self) -> Int:
        return self.d[1]

    def vocab(self) -> Int:
        return self.d[8]

    def layers(self) -> Int:
        return self.d[7]

    def n_tensors(self) -> Int:
        return 2 + 9 * self.d[7]

    def shapes(self) -> List[List[Int]]:
        """`parameter_shapes`, in packing order."""
        var dm = self.d[2]
        var kd = self.d[4] * self.d[5]
        var ff = self.d[6]
        var out = List[List[Int]]()
        out.append([self.d[8], dm])
        for _ in range(self.d[7]):  # driver-loop: over the decoder layers (model structure)
            out.append([dm])
            out.append([dm, dm])
            out.append([kd, dm])
            out.append([kd, dm])
            out.append([dm, dm])
            out.append([dm])
            out.append([ff, dm])
            out.append([ff, dm])
            out.append([dm, ff])
        out.append([self.d[8], dm])
        return out^

    def names(self) -> List[String]:
        """`parameter_names`, in packing order."""
        var block = [String("norm1_w"), String("w_q"), String("w_k"), String("w_v"), String("w_o"),
                     String("norm2_w"), String("w_gate"), String("w_up"), String("w_down")]
        var out = List[String]()
        out.append(String("embed"))
        for layer in range(self.d[7]):  # driver-loop: over the decoder layers (model structure)
            for k in range(9):  # small-loop(the nine tensors of one block)
                out.append(String("block") + String(layer) + "." + block[k])
        out.append(String("lm_head"))
        return out^

    def n_total(self) -> Int:
        var total = 0
        var shapes = self.shapes()
        for i in range(len(shapes)):  # driver-loop: over the parameter tensors (model structure)
            var count = 1
            for j in range(len(shapes[i])):  # small-loop(one tensor's shape words)
                count *= shapes[i][j]
            total += count
        return total

    def native_shape(self) raises -> PythonObject:
        """`list(shape.native_shape)`: the nine dimensions, and 1 with the
        chunked head."""
        var out = Python.list()
        for i in range(9):  # small-loop(the nine dimensions)
            _ = py_call(out.append, self.d[i])
        if self.chunked:
            _ = py_call(out.append, 1)
        return out

    def profile(self, default_profile: String) -> String:
        """`shape.profile`."""
        var b = self.d[0]
        var l = self.d[1]
        var dm = self.d[2]
        var h = self.d[3]
        var kv = self.d[4]
        var hd = self.d[5]
        var ff = self.d[6]
        var layers = self.d[7]
        var vocab = self.d[8]
        if (not self.chunked and b == 2 and l == 32 and dm == 32 and h == 4 and kv == 2 and hd == 8 and ff == 64
                and layers == 2 and vocab == 256):
            return default_profile
        var suffix = String("-v256-blocks2.fp32.v2") if (layers == 2 and vocab == 256) else (
            String("-v") + String(vocab) + "-blocks" + String(layers) + ".fp32.v3")
        var head = String("-lmhead-chunk256-v2") if self.chunked else String("")
        return (String("mojolearn.byte-lm.b") + String(b) + "-l" + String(l) + "-d" + String(dm) + "-h" + String(h)
                + "-kv" + String(kv) + "-hd" + String(hd) + "-ff" + String(ff) + suffix + head)


def _shape_of(state: PythonObject) raises -> _Shape:
    """`state_shape(state)` (the config's dimension refusals included)."""
    var b = _b()
    var dims: List[Int] = [2, 32, 32, 4, 2, 8, 64, 2, 256]
    if not py_contains(state, "model_shape"):
        return _Shape(dims^, False)
    var value = state["model_shape"]
    var names = [String("batch"), String("length"), String("d_model"), String("n_heads"), String("n_kv"),
                 String("head_dim"), String("intermediate"), String("n_layers"), String("vocab_size")]
    if py_type(value) is not b.dict:
        raise Error("ValueError: Byte-LM model_shape has missing or unknown dimensions")
    var keys = py_call(b.set, value)
    var all_keys = b.set()
    for i in range(9):  # small-loop(the nine dimension names)
        _ = py_call(all_keys.add, names[i])
    _ = py_call(all_keys.add, "chunked_lm_head_v2")
    var no_chunk = py_call(all_keys.difference, Python.list("chunked_lm_head_v2"))
    var no_layers = py_call(all_keys.difference, Python.list("n_layers", "vocab_size"))
    var no_both = py_call(all_keys.difference, Python.list("n_layers", "vocab_size", "chunked_lm_head_v2"))
    if not (Bool(py=keys == all_keys) or Bool(py=keys == no_chunk) or Bool(py=keys == no_layers) or Bool(py=keys == no_both)):
        raise Error("ValueError: Byte-LM model_shape has missing or unknown dimensions")
    for i in range(9):  # small-loop(the nine dimension names)
        if py_contains(value, names[i]):
            var v = value[names[i]]
            if py_type(v) is b.bool or String(py=py_type(v).__name__) == "bool_":
                raise Error("ValueError: Byte-LM shape requires integer dimensions")
            var n = 0
            try:
                n = Int(py=py_call(native_loaded_module("operator").index, v))
            except:
                raise Error("ValueError: Byte-LM shape requires integer dimensions")
            if not (0 < n and n <= (1 << 20)):
                raise Error("ValueError: Byte-LM dimensions must be in [1, 2**20]")
            dims[i] = n
    var chunked = False
    if py_contains(value, "chunked_lm_head_v2"):
        if py_type(value["chunked_lm_head_v2"]) is not b.bool:
            raise Error("ValueError: chunked_lm_head_v2 must be a boolean")
        chunked = Bool(py=value["chunked_lm_head_v2"])
    return _Shape(dims^, chunked)


# ---------------------------------------------------------------- inputs and outputs


def byte_lm_lengths(source: PythonObject, batch: Int, length: Int, context: PythonObject) raises -> List[Int]:
    # Native scalar admission preserves integer index types (including NumPy
    # integers in lists) without permitting user-defined index callbacks.
    classical_admit_value(source, context)
    var bi = _b()
    var cls = py_type(source)
    var values = source
    if cls is context["array_class"]:
        if len(source.shape) != 1:
            raise Error("lengths must be one-dimensional")
        values = source._mv
    elif cls is not bi.list and cls is not bi.tuple and cls is not bi.range:
        values = native_buffer_view(source)
        if Int(py=values.ndim) != 1:
            raise Error("lengths must be one-dimensional")
    if len(values) != batch:
        raise Error("one length is required per batch row")
    var result = List[Int]()
    for i in range(batch):  # driver-loop: over one admitted length metadata word per sequence
        var value = values[i]
        if py_type(value) is bi.bool or String(py=py_type(value).__name__) == "bool_":
            raise Error("TypeError: lengths entries must be integers, not booleans")
        var scalar = PythonObject(None)
        try:
            scalar = py_call(native_loaded_module("operator").index, value)
        except:
            raise Error("TypeError: lengths entries must be integers")
        if Bool(py=scalar < PythonObject(1)) or Bool(py=scalar > PythonObject(length)):
            raise Error("lengths must be in [1, L]")
        result.append(Int(py=scalar))
    return result^


def _config_names() -> List[String]:
    return [String("batch"), String("length"), String("d_model"), String("n_heads"), String("n_kv"),
            String("head_dim"), String("intermediate"), String("n_layers"), String("vocab_size")]


def _config_store(value: PythonObject, shape: _Shape) raises:
    var fields = py_fields(value)
    var names = _config_names()
    for i in range(9):  # driver-loop: over normalized immutable configuration fields
        py_setitem(fields, names[i], shape.d[i])
    py_setitem(fields, "chunked_lm_head_v2", shape.chunked)


def _config_new(shape: _Shape, context: PythonObject) raises -> PythonObject:
    var value = native_storage_allocate(context["byte_lm_config_class"])
    _config_store(value, shape)
    return value


def _config_capture(value: PythonObject, context: PythonObject) raises -> _Shape:
    if py_type(value) is not context["byte_lm_config_class"]:
        raise Error("TypeError: shape requires the connected native ByteLanguageModelConfig class")
    var fields = py_fields(value)
    classical_admit_value(fields, context)
    var state = Python.dict()
    py_setitem(state, "model_shape", fields)
    return _shape_of(state)


def byte_lm_config_operation(value: PythonObject, operation: String, args: PythonObject,
                              context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var bi = _b()
    if operation == "config_require":
        if value is None:
            return _config_new(_shape_of(Python.dict()), context)
        if py_type(value) is not context["byte_lm_config_class"]:
            raise Error("TypeError: shape must be a ByteLanguageModelConfig")
        return value
    if operation == "config_state":
        classical_admit_value(value, context)
        if py_type(value) is not bi.dict:
            raise Error("TypeError: state requires native dictionary storage")
        var shape = _shape_of(value)
        training_byte_shape_validate(shape.d, shape.chunked)
        return _config_new(shape, context)
    if operation == "config_construct":
        if py_type(value) is not context["byte_lm_config_class"]:
            raise Error("TypeError: shape requires the connected native ByteLanguageModelConfig class")
        classical_admit_value(args, context)
        var fields = Python.dict()
        var names = _config_names()
        for i in range(9):  # driver-loop: over public configuration arguments
            py_setitem(fields, names[i], args[i])
        py_setitem(fields, "chunked_lm_head_v2", args[9])
        var state = Python.dict()
        py_setitem(state, "model_shape", fields)
        var shape = _shape_of(state)
        training_byte_shape_validate(shape.d, shape.chunked)
        _config_store(value, shape)
        return PythonObject(None)
    if operation == "config_setattr" or operation == "config_delattr":
        # Keep FrozenInstanceError rather than weakening the frozen dataclass
        # contract to a generic AttributeError. No Python exception initializer.
        if py_type(args[0]) is not bi.str:
            raise Error("TypeError: attribute name must be an exact native string")
        var category = context["byte_lm_config_frozen_error"]
        var verb = String("assign to" if operation == "config_setattr" else "delete")
        var message = verb + " field " + String(py_call(bi.repr, args[0]))
        var exception = py_call(bi.BaseException.__new__, category, PythonObject("cannot " + message))
        ref cpy = Python().cpython()
        cpy.lib.get_function[NoneType]("PyErr_SetObject")(category._obj_ptr, exception._obj_ptr)
        var failure = stash_python_error()
        raise failure^
    if operation == "config_equal" and py_type(args[0]) is not context["byte_lm_config_class"]:
        return bi.NotImplemented
    var shape = _config_capture(value, context)
    if operation == "config_validate":
        training_byte_shape_validate(shape.d, shape.chunked)
        _config_store(value, shape)
        return PythonObject(None)
    if operation == "config_native_shape":
        return py_call(bi.tuple, shape.native_shape())
    if operation == "config_profile":
        return PythonObject(shape.profile(String(context["byte_lm_default_profile"])))
    if operation == "config_n_tensors":
        return PythonObject(shape.n_tensors())
    if operation == "config_n_total":
        return PythonObject(shape.n_total())
    if operation == "config_equal":
        var other = _config_capture(args[0], context)
        var equal = shape.chunked == other.chunked
        for i in range(9):  # driver-loop: over the nine immutable dimension fields
            equal = equal and shape.d[i] == other.d[i]
        return PythonObject(equal)
    if operation == "config_hash":
        var words = shape.native_shape()
        if not shape.chunked:
            _ = py_call(words.append, False)
        else:
            py_setitem(words, 9, True)
        return py_call(bi.hash, py_call(bi.tuple, words))
    var names = _config_names()
    if operation == "config_dict":
        var result = Python.dict()
        for i in range(9):  # driver-loop: over completed configuration fields
            py_setitem(result, names[i], shape.d[i])
        py_setitem(result, "chunked_lm_head_v2", shape.chunked)
        return result
    if operation == "config_repr":
        var text = String(context["byte_lm_config_class"].__qualname__) + "("
        for i in range(9):  # driver-loop: over formatted configuration metadata
            if i > 0:
                text += ", "
            text += names[i] + "=" + String(shape.d[i])
        text += ", chunked_lm_head_v2=" + String("True" if shape.chunked else "False") + ")"
        return PythonObject(text)
    if operation == "config_names":
        var out = Python.list()
        for name in shape.names():  # driver-loop: over configured parameter names
            _ = py_call(out.append, name)
        return py_call(bi.tuple, out)
    var shapes = shape.shapes()
    if operation == "config_shapes":
        var out = Python.list()
        for dims in shapes:  # driver-loop: over configured parameter shapes
            var words = Python.list()
            for width in dims:  # driver-loop: over one parameter tensor's rank metadata
                _ = py_call(words.append, width)
            _ = py_call(out.append, py_call(bi.tuple, words))
        return py_call(bi.tuple, out)
    if operation == "config_offsets":
        var out = Python.list(0)
        var offset = 0
        for dims in shapes:  # driver-loop: over configured parameter tensor offsets
            var count = 1
            for width in dims:  # driver-loop: over one parameter tensor's rank metadata
                count *= width
            offset += count
            _ = py_call(out.append, offset)
        return py_call(bi.tuple, out)
    raise Error("RuntimeError: unknown ByteLM configuration operation")
