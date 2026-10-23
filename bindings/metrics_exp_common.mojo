# SPDX-License-Identifier: Apache-2.0
# The shared pieces of the classification, ranking and clustering metrics'
# native owner (bindings/metrics_exp_native.mojo): the x_metrics PROGRAM
# (`_expansion_metrics._Prog`, with the counted output ranges), its run
# through the shared program runner (bindings/arena_prog_native.mojo
# `_metrics_run`, the one native call the Python `_run` made), the arena
# reads, and the label input step the Python made in `_metrics_impl`
# (`_classification_encoded`, `_classification_labels`, `_selected_labels`,
# `_label_map`) and `_buffer.shape_of`.
#
# Labels: an integer or bool buffer (a mojolearn Array or an ndarray of one
# dimension) is encoded by the host label encoder (core/label_cast.mojo,
# the sorted classes and one int32 code per row, the encoder the Python
# reached through `prepare_targets`); every other input (a list, str
# labels, pandas, a buffer of another kind) is the label input step on
# Python objects, done with builtins over the whole container (set, sorted,
# map, dict): # cpu-route: Python object labels, the explicit label input step.
# Both give the sorted distinct classes and the codes into them, which is
# all the metrics read, so the programs below see the same words.
from std.memory import memcpy, bitcast
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setattr, py_setitem, py_fields
from bindings.native_buffer_owner import native_storage_allocate, native_buffer_view
from bindings.classical_arrays import (
    classical_address, classical_is_array, classical_dtype_str, classical_wrap_array, classical_cast_kind,
    classical_c_order,
)
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_validate import classical_view
from bindings.arena_prog_native import _metrics_run
from bindings.optimizer_public_words import stash_python_error
from bindings.family_common import (
    FAMILY_F32P, FAMILY_F64P, FAMILY_I32P, FAMILY_I64P, family_repr, family_str, family_isinstance,
    family_call, family_call_kw, family_array, family_shape_repr, family_cast_view, family_view,
)
from bindings.metrics_reg_native import _reg_weights
from core.family_cast_impl import family_all_finite_f32
from core.cast_in import CastIn, CAST_KIND_FLOAT, label_kind_of
from core.label_cast import host_unique_inverse_cast
from core.metrics_exp_impl import mexp_gather_i32

comptime MX_NONE = -1
comptime MX_PARAMS = 14


# ---------------------------------------------------------------- small Python helpers


def mx_b() raises -> PythonObject:
    return Python.import_module("builtins")


def mx_len(value: PythonObject) raises -> Int:
    return Int(py=py_call(mx_b().len, value))


def mx_attr(value: PythonObject, name: String) raises -> PythonObject:
    """`getattr(value, name)` (names PythonObject spells as its own methods)."""
    return py_call(mx_b().getattr, value, PythonObject(name))


def mx_truthy(value: PythonObject) raises -> Bool:
    return Bool(py=py_call(mx_b().bool, value))


def mx_is_none(value: PythonObject) -> Bool:
    return value is None


def mx_eq(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a == b)`."""
    return Bool(py=py_call(mx_b().bool, py_call(Python.import_module("operator").eq, a, b)))


def mx_is_bool(value: PythonObject) raises -> Bool:
    """`_labels.is_bool`: Python's bool or NumPy's bool_."""
    return family_isinstance(value, mx_b().bool) or String(py_type(value).__name__) == "bool_"


def mx_is_str(value: PythonObject) raises -> Bool:
    return family_isinstance(value, mx_b().str)


def mx_is_real(value: PythonObject, context: PythonObject) raises -> Bool:
    """`isinstance(value, numbers.Real)`."""
    return family_isinstance(value, context["family_metrics_real"])


def mx_is_integral(value: PythonObject, context: PythonObject) raises -> Bool:
    """`isinstance(value, numbers.Integral)`."""
    return family_isinstance(value, context["family_mexp_integral"])


def mx_float(value: PythonObject) raises -> Float64:
    """`float(value)`."""
    return Float64(py=family_call(mx_b().float, value))


def mx_nan() -> Float64:
    return bitcast[DType.float64](UInt64(0x7FF8000000000000))


def mx_inf() -> Float64:
    return bitcast[DType.float64](UInt64(0x7FF0000000000000))


def mx_isnan(x: Float64) -> Bool:
    return x != x


def mx_isinf(x: Float64) -> Bool:
    return x == x and x - x != 0.0


def mx_raise_obj(cls: PythonObject, argument: PythonObject) raises:
    """Raise `cls(argument)` as the Python exception itself (a class the
    door has no prefix for: ZeroDivisionError, KeyError, UserWarning-free
    refusals), restored unchanged at the door."""
    var exception = py_call(cls, argument)
    ref cpy = Python().cpython()
    cpy.lib.get_function[NoneType]("PyErr_SetObject")(cls._obj_ptr, exception._obj_ptr)
    var failure = stash_python_error()
    raise failure^


def mx_zero_division(message: String) raises:
    mx_raise_obj(mx_b().ZeroDivisionError, PythonObject(message))


def mx_warn(message: String, category: PythonObject, stacklevel: Int) raises:
    var keywords = Python.dict()
    py_setitem(keywords, "stacklevel", PythonObject(stacklevel))
    _ = family_call_kw(Python.import_module("warnings").warn, keywords, PythonObject(message), category)


def mx_undefined(message: String, context: PythonObject) raises:
    """`_undefined_warning`: scikit-learn's UndefinedMetricWarning when the
    caller imported sklearn.exceptions, else mojolearn's."""
    var category = context["family_metrics_undefined"]
    var sk = py_get(Python.import_module("sys").modules, PythonObject("sklearn.exceptions"), PythonObject(None))
    if sk is not None:
        category = py_call(mx_b().getattr, sk, PythonObject("UndefinedMetricWarning"), category)
    mx_warn(message, category, 2)


def mx_sorted(values: PythonObject) raises -> PythonObject:
    return py_call(mx_b().sorted, values)


def mx_set(values: PythonObject) raises -> PythonObject:
    return py_call(mx_b().set, values)


def mx_list(values: PythonObject) raises -> PythonObject:
    return py_call(mx_b().list, values)


def mx_index(order: PythonObject) raises -> PythonObject:
    """`{c: i for i, c in enumerate(order)}`."""
    var b = mx_b()
    return py_call(b.dict, py_call(b.zip, order, py_call(b.range, py_call(b.len, order))))


def mx_union_sorted(a: PythonObject, c: PythonObject) raises -> PythonObject:
    """`sorted(set(a) | set(c))`."""
    return mx_sorted(py_call(mx_set(a).union, c))


def mx_missing(present: PythonObject, classes: PythonObject) raises -> Bool:
    """`bool(set(present) - set(classes))`."""
    return mx_len(py_call(mx_set(present).difference, classes)) > 0


def mx_label_order(chosen: PythonObject, present: PythonObject) raises -> PythonObject:
    """`_label_order`: `chosen` then every present label not in it."""
    var it = Python.import_module("itertools")
    var rest = mx_list(py_call(it.filterfalse, mx_attr(mx_set(chosen), "__contains__"), present))
    return mx_list(chosen) + rest


def mx_str_list(values: PythonObject) raises -> PythonObject:
    """`[str(v) for v in values]`."""
    return mx_list(py_call(mx_b().map, mx_b().str, values))


# ---------------------------------------------------------------- the program


struct MxProg(Movable):
    """`_expansion_metrics._Prog`: an arena layout, the inputs copied into
    it, the stage records and the declared outputs, as the Python lists the
    shared runner reads."""
    var size: Int
    var inputs: PythonObject
    var stages: PythonObject
    var skip: PythonObject
    var outs: PythonObject
    var ops: PythonObject

    def __init__(out self, context: PythonObject) raises:
        self.size = 0
        self.inputs = Python.list()
        self.stages = Python.list()
        self.skip = Python.list()
        self.outs = PythonObject(None)
        self.ops = context["family_metrics_ops"]

    def alloc(mut self, n: Int) raises -> Int:
        var off = self.size
        self.size += max(n, 0)
        if self.size > 2147483647:
            raise Error("mojolearn metrics: the problem exceeds the Int32 arena bound")
        return off

    def put(mut self, array: PythonObject) raises -> Int:
        """A float32 (or int32, stored as its bits) C-ordered array -> its offset."""
        var count = Int(py=array.size)
        var off = self.alloc(count)
        _ = py_call(self.inputs.append, Python.tuple(PythonObject(off), array))
        _ = py_call(self.skip.append, Python.tuple(PythonObject(off), PythonObject(off + count)))
        return off

    def put_i32(mut self, array: PythonObject) raises -> Int:
        return self.put(array)

    def scratch(mut self, n: Int) raises -> Int:
        var off = self.alloc(n)
        if n > 0:
            _ = py_call(self.skip.append, Python.tuple(PythonObject(off), PythonObject(off + n)))
        return off

    def stage(mut self, op: String, total: Int, params: List[Int]) raises:
        if len(params) > MX_PARAMS:
            raise Error("ValueError: x_metrics: too many stage parameters")
        var record = Python.list()
        _ = py_call(record.append, self.ops[PythonObject(op)])
        _ = py_call(record.append, PythonObject(total))
        for i in range(len(params)):  # small-loop(stage words: at most 14 parameters)
            _ = py_call(record.append, PythonObject(params[i]))
        for _ in range(MX_PARAMS - len(params)):  # small-loop(stage words: the zero padding of one record)
            _ = py_call(record.append, PythonObject(0))
        _ = py_call(self.stages.append, record)

    def want_count(mut self, off: Int, n: Int, count: Int, mult: Int) raises -> Int:
        """`_Prog.want(off, n, count, mult)`: declare an output range."""
        if self.outs is None:
            self.outs = Python.list()
        if n > 0:
            _ = py_call(self.outs.append, Python.tuple(PythonObject(off), PythonObject(off + n),
                                                       PythonObject(count), PythonObject(mult)))
        return off

    def want(mut self, off: Int, n: Int) raises -> Int:
        return self.want_count(off, n, -1, 1)

    def group(mut self, key: Int, n: Int, m: Int, values: Int, vstride: Int, weights: Int,
              width: Int) raises -> Tuple[Int, Int]:
        """`_group`: the stable counting sort and the PairSums; (OFF, OUT)."""
        var off = self.alloc(m + 1)
        var order = self.scratch(n)
        var out = self.alloc(m * width)
        self.stage("group_sort", 1, [key, n, m, off, order])
        if m * width:
            self.stage("group_sum", m * width, [off, order, values, vstride, weights, out, width])
        return (off, out)

    def counts(mut self, off: Int, m: Int) raises -> Int:
        """`_counts`: the m group sizes as an Int32 slot (off_diff)."""
        var out = self.alloc(m)
        if m:
            self.stage("off_diff", m, [off, m, out])
        return out

    def scan_flag(mut self, src: Int, n: Int, mode: Int) raises -> Int:
        var flag = self.alloc(1)
        if n:
            self.stage("flag_scan", n, [src, n, mode, flag])
        return flag

    def f64_words(mut self, value: Float64, context: PythonObject) raises -> Int:
        """`_f64_words`: a binary64 scalar as two Int32 words (low first), an input."""
        var words = family_array(Python.tuple(2), "<i4", context)
        FAMILY_F64P(unsafe_from_address=Int(py=classical_address(words)))[0] = value
        return self.put_i32(words)


struct MxArena(Copyable, Movable):
    """A run program's arena (the float32 `array.array` the runner made)."""
    var store: PythonObject
    var base: Int

    def __init__(out self, store: PythonObject) raises:
        self.store = store
        self.base = Int(py=py_call(store.buffer_info)[0])

    def i32(self, off: Int) -> Int:
        return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=self.base + 4 * off)[0])

    def i64(self, off: Int) -> Int:
        return Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=self.base + 4 * off)[0])

    def f64(self, off: Int) -> Float64:
        """The binary64 value of arena[off], arena[off + 1], low first (`words(off, 2, "d")`)."""
        var words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=self.base + 4 * off)
        return bitcast[DType.float64](UInt64(words[0]) | (UInt64(words[1]) << 32))

    def f32(self, off: Int) -> Float64:
        """`floats(off, 1)[0]`: the Float32 word, widened."""
        return Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=self.base + 4 * off)[0])

    def words(self, off: Int, words: Int, shape: PythonObject, typestr: String,
              context: PythonObject) raises -> PythonObject:
        """`Array._owned(words(off, words, code), shape, typestr)`: a copy of
        the arena words as a new Array."""
        var result = family_array(shape, typestr, context)
        if words > 0:
            memcpy(dest=MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result))),
                   src=MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=self.base + 4 * off), count=words)
        return result


def mx_run(prog: MxProg, mode: PythonObject, context: PythonObject) raises -> MxArena:
    """`_run(prog, numeric_mode)`: one program handed to the shared runner."""
    var prog_context = context["family_metrics_prog_context"]
    classical_connect_numpy(prog_context)
    var program = native_storage_allocate(prog_context["prog_class"])
    py_setattr(program, "size", PythonObject(prog.size))
    py_setattr(program, "_inputs", prog.inputs)
    py_setattr(program, "_stages", prog.stages)
    py_setattr(program, "arena", PythonObject(None))
    py_setattr(program, "_outs", prog.outs)
    py_setattr(program, "_skip", prog.skip)
    py_setattr(program, "rows", PythonObject(None))
    _ = _metrics_run(program, mode, prog_context)
    return MxArena(py_fields(program)["arena"])


def mx_f64_scores(values: List[Float64], context: PythonObject) raises -> PythonObject:
    """binary64 values as an Int32 word array (low words first): `_f64_bits`."""
    var words = family_array(Python.tuple(2 * len(values)), "<i4", context)
    var dst = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(words)))
    for i in range(len(values)):  # driver-loop: over the metric's per-class scores
        dst[i] = values[i]
    return words


# ---------------------------------------------------------------- labels


struct MxLab(Copyable, Movable):
    """One label vector: its kind ("integer" or "string"), the sorted
    distinct classes (a Python list) and one int32 code per row into them."""
    var kind: String
    var classes: PythonObject
    var codes: PythonObject
    var n: Int

    def __init__(out self, kind: String, classes: PythonObject, codes: PythonObject, n: Int):
        self.kind = kind
        self.classes = classes
        self.codes = codes
        self.n = n


def mx_shape_of(obj: PythonObject) raises -> PythonObject:
    """`_buffer.shape_of`: `.shape`, or a list's nesting, without converting."""
    var b = mx_b()
    var shape = py_call(b.getattr, obj, PythonObject("shape"), PythonObject(None))
    if shape is not None:
        return py_call(b.tuple, shape)
    if family_isinstance(obj, Python.tuple(b.list, b.tuple)):
        var n = mx_len(obj)
        if n > 0 and family_isinstance(obj[0], Python.tuple(b.list, b.tuple)):
            return Python.tuple(PythonObject(n), PythonObject(mx_len(obj[0])))
        return Python.tuple(PythonObject(n))
    if Bool(py=py_call(b.hasattr, obj, PythonObject("__len__"))):
        return Python.tuple(PythonObject(mx_len(obj)))
    return Python.tuple()


def mx_refuse_multilabel(y: PythonObject, name: String, caller: String) raises:
    var shape = mx_shape_of(y)
    if mx_len(shape) == 2 and Int(py=shape[1]) != 1:
        raise Error("NotImplementedError: mojolearn " + caller + ": " + name
                    + " is 2-D; multilabel-indicator targets are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv), "
                    + "only binary and multiclass 1-D labels")


def _mx_flatten(y: PythonObject) raises -> PythonObject:
    """`_labels.flatten_labels` of a one-dimensional label input."""
    var b = mx_b()
    var values = y
    if Bool(py=py_call(b.hasattr, values, PythonObject("tolist"))) and not family_isinstance(values, Python.tuple(b.list, b.tuple)):
        values = py_call(values.tolist)
    if family_isinstance(values, Python.tuple(b.str, b.bytes)):
        raise Error("mojolearn: y must be a sequence of labels")
    return mx_list(values)


def mx_list_labels(values: PythonObject, name: String, allow_empty: Bool,
                   context: PythonObject) raises -> Tuple[PythonObject, String]:
    """`_classification_labels`: (the labels as a list, their kind)."""
    var b = mx_b()
    var it = Python.import_module("itertools")
    var op = Python.import_module("operator")
    if mx_len(mx_shape_of(values)) != 1:
        raise Error(name + " must be one-dimensional single-label data")
    var labels = _mx_flatten(values)
    if not allow_empty and mx_len(labels) == 0:
        raise Error(name + " must contain at least one label")
    if Bool(py=py_call(b.all, py_call(b.map, b.isinstance, labels, py_call(it.repeat, b.str)))):
        return (mx_list(py_call(b.map, b.str, labels)), String("string"))
    # every label an Integral or a bool (NumPy's bool_ by its type name):
    # the distinct types, then the ones that are neither, by name
    var types = mx_list(mx_set(py_call(b.map, py_type, labels)))
    var integral = Python.tuple(context["family_mexp_integral"], b.bool)
    var fits = mx_list(py_call(b.map, b.issubclass, types, py_call(it.repeat, integral)))
    var rest = mx_list(py_call(it.compress, types, py_call(b.map, op.not_, fits)))
    var names = py_call(b.map, py_call(op.attrgetter, PythonObject("__name__")), rest)
    if Bool(py=py_call(b.all, py_call(b.map, op.eq, names, py_call(it.repeat, PythonObject("bool_"))))):
        return (mx_list(py_call(b.map, b.int, labels)), String("integer"))
    raise Error("TypeError: " + name + " must contain only strings or only integers; "
                + "floating labels, missing labels and mixed types are unsupported")


def _mx_class_code(kind: Int, item: Int) -> String:
    """The `array.array` code holding one class item of a label buffer."""
    if kind == 1:  # signed
        if item == 1:
            return "b"
        if item == 2:
            return "h"
        if item == 4:
            return "i"
        return "q"
    if item == 1:
        return "B"
    if item == 2:
        return "H"
    if item == 4:
        return "I"
    return "Q"


def _mx_buffer_labels(values: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_native_classification_labels`: (classes, codes) of a 1-D integer or
    bool buffer through the host label encoder, or None for anything the
    Python label step reads (every other input)."""
    var b = mx_b()
    var source = values
    if py_type(values) is b.list or py_type(values) is b.tuple:
        # a flat list of exact Python ints as an int64 buffer (OverflowError
        # for an int outside int64 keeps the list route)
        var n = mx_len(values)
        if n == 0 or not mx_eq(mx_set(py_call(b.map, py_type, values)), mx_set(Python.list(b.int))):
            return PythonObject(None)
        var lo = py_call(b.min, values)
        var hi = py_call(b.max, values)
        if Bool(py=lo < PythonObject(Int.MIN)) or Bool(py=hi > PythonObject(Int.MAX)):
            return PythonObject(None)
        var store = py_call(context["array_type"], PythonObject("q"), values)
        source = classical_wrap_array(store, Python.tuple(PythonObject(n)), "<i8", "C", context)
    elif not classical_is_array(values, context):
        var nd_class = py_get(context, "ndarray")
        if nd_class is None or py_type(values) is not nd_class:
            return PythonObject(None)
    if Int(py=source.ndim) != 1:
        return PythonObject(None)
    var n = Int(py=source.size)
    if n < 1:
        return PythonObject(None)
    var letter = String(source.dtype[1]) if classical_is_array(source, context) else String(source.dtype.kind)
    if letter != "i" and letter != "u" and letter != "b":
        return PythonObject(None)
    if not classical_is_array(source, context) and not Bool(py=source.dtype.isnative):
        return PythonObject(None)
    var item = Int(py=source.itemsize)
    var kind = classical_cast_kind(source, item, name, context)
    var s0 = Int(py=source.strides[0])
    var src = CastIn(Int(py=classical_address(source)), kind, item, n, 1, s0, item)
    var lk = label_kind_of(kind, item, name)
    var code_kind = 1 if kind == 1 else 0
    var raw = py_call(context["array_type"], PythonObject(_mx_class_code(code_kind, item)),
                      py_call(context["native_bytes_type"], n * item))
    var codes = family_array(Python.tuple(PythonObject(n)), "<i4", context)
    var k = host_unique_inverse_cast(src, lk, Int(py=py_call(raw.buffer_info)[0]),
                                     Int(py=classical_address(codes)))
    _ = source^
    if k < 0:
        return PythonObject(None)
    var classes = mx_list(py_call(raw.tolist)[py_call(b.slice, PythonObject(0), PythonObject(k))])
    return Python.tuple(classes, codes)


def mx_encode(values: PythonObject, name: String, context: PythonObject) raises -> MxLab:
    """`_classification_encoded(values, name)`: the classes and the codes."""
    var b = mx_b()
    if not family_isinstance(values, Python.tuple(b.str, b.bytes)) and mx_len(mx_shape_of(values)) == 1:
        var fast = _mx_buffer_labels(values, name, context)
        if fast is not None:
            return MxLab("integer", fast[0], fast[1], Int(py=fast[1].size))
    var listed = mx_list_labels(values, name, False, context)
    var labels = listed[0]
    var classes = mx_sorted(mx_set(labels))
    var index = mx_index(classes)
    var n = mx_len(labels)
    var store = py_call(context["array_type"], PythonObject("i"), py_call(b.map, mx_attr(index, "__getitem__"), labels))
    var codes = classical_wrap_array(store, Python.tuple(PythonObject(n)), "<i4", "C", context)
    return MxLab(listed[1], classes, codes, n)


def mx_selected(labels: PythonObject, kind: String, observed: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`_selected_labels(labels, kind, observed)`."""
    if labels is None:
        return mx_list(observed)
    var listed = mx_list_labels(labels, "labels", False, context)
    if listed[1] != kind:
        raise Error("TypeError: labels must use the same type as the targets")
    if mx_len(mx_set(listed[0])) != mx_len(listed[0]):
        raise Error("labels must be unique")
    return listed[0]


def _mx_gather(lab: MxLab, table: PythonObject, strict: Bool, context: PythonObject) raises -> PythonObject:
    """`table[codes[i]]` per row (`table` an `array.array("i")` over the
    classes); `strict` raises the KeyError `index[v]` raised for a -1."""
    var k = mx_len(lab.classes)
    var out = family_array(Python.tuple(PythonObject(lab.n)), "<i4", context)
    var first = mexp_gather_i32(FAMILY_I32P(unsafe_from_address=Int(py=py_call(table.buffer_info)[0])), k,
                                FAMILY_I32P(unsafe_from_address=Int(py=classical_address(lab.codes))), lab.n,
                                FAMILY_I32P(unsafe_from_address=Int(py=classical_address(out))))
    if strict and first >= 0:
        var code = Int(FAMILY_I32P(unsafe_from_address=Int(py=classical_address(lab.codes)))[first])
        mx_raise_obj(mx_b().KeyError, lab.classes[code])
    return out


def mx_codes(lab: MxLab, order: PythonObject, strict: Bool, context: PythonObject) raises -> PythonObject:
    """`_codes(labels, order)`: each row's index in `order`, -1 for a label
    outside it; `strict` raises the KeyError `index[v]` raised for one."""
    var b = mx_b()
    var index = mx_index(order)
    var table = py_call(context["array_type"], PythonObject("i"),
                        py_call(b.map, index.get, lab.classes, py_call(Python.import_module("itertools").repeat, PythonObject(-1))))
    return _mx_gather(lab, table, strict, context)


def mx_map_eq(lab: MxLab, pos: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_label_map(labels, lambda v: int(v == pos))`: the flag of each class, gathered per row."""
    var b = mx_b()
    var hits = py_call(b.map, Python.import_module("operator").eq, lab.classes,
                       py_call(Python.import_module("itertools").repeat, pos))
    var table = py_call(context["array_type"], PythonObject("i"), py_call(b.map, b.int, py_call(b.map, b.bool, hits)))
    return _mx_gather(lab, table, False, context)


def mx_zeros_i32(n: Int, context: PythonObject) raises -> PythonObject:
    return family_array(Python.tuple(PythonObject(n)), "<i4", context)


# ---------------------------------------------------------------- shared validation


struct MxPair(Movable):
    """`_pair`'s (true, pred, kind, present, w)."""
    var true: MxLab
    var pred: MxLab
    var kind: String
    var present: PythonObject
    var w: PythonObject

    def __init__(out self, var true: MxLab, var pred: MxLab, kind: String, present: PythonObject, w: PythonObject):
        self.true = true^
        self.pred = pred^
        self.kind = kind
        self.present = present
        self.w = w


def mx_weights(sample_weight: PythonObject, n: Int, caller: String, context: PythonObject) raises -> PythonObject:
    """`_weights`: None, or `_sample_weight_f32` (float32, checked)."""
    return _reg_weights(sample_weight, n, caller, context)


def mx_pair(y_true: PythonObject, y_pred: PythonObject, sample_weight: PythonObject, caller: String,
            context: PythonObject) raises -> MxPair:
    mx_refuse_multilabel(y_true, "y_true", caller)
    mx_refuse_multilabel(y_pred, "y_pred", caller)
    var true = mx_encode(y_true, "y_true", context)
    var pred = mx_encode(y_pred, "y_pred", context)
    if true.n != pred.n:
        raise Error("mojolearn " + caller + ": y_true and y_pred lengths differ")
    if true.kind != pred.kind:
        raise Error("TypeError: mojolearn " + caller + ": y_true and y_pred must use the same label type")
    if true.n > 2147483647:
        raise Error("mojolearn " + caller + ": at most INT32_MAX rows")
    var w = mx_weights(sample_weight, true.n, caller, context)
    var present = mx_union_sorted(true.classes, pred.classes)
    var kind = true.kind
    return MxPair(true^, pred^, kind, present, w)


def mx_targets(y_true: PythonObject, caller: String, context: PythonObject) raises -> MxLab:
    """`_targets`: the encoded y_true (its classes are `present`)."""
    mx_refuse_multilabel(y_true, "y_true", caller)
    return mx_encode(y_true, "y_true", context)


def mx_pos_label(pos_label: PythonObject, kind: String, classes: PythonObject, caller: String,
                 context: PythonObject) raises -> PythonObject:
    """`_pos_label`."""
    if pos_label is None:
        var allowed = Python.tuple(Python.list(PythonObject(0)), Python.list(PythonObject(1)), Python.list(PythonObject(-1)),
                                   Python.list(PythonObject(0), PythonObject(1)), Python.list(PythonObject(-1), PythonObject(1)))
        if kind != "integer" or not py_contains(allowed, classes):
            raise Error("y_true takes value in " + family_str(classes) + " and pos_label is not specified: either make "
                        + "y_true take value in {0, 1} or {-1, 1} or pass pos_label explicitly.")
        return PythonObject(1)
    var listed = mx_list_labels(Python.list(pos_label), "pos_label", False, context)
    if listed[1] != kind:
        raise Error("pos_label must use the same type as y_true")
    return listed[0][0]


def mx_pos_label_ok(kind: String, classes: PythonObject) raises -> Bool:
    """True when `_pos_label(None, ...)` answers (1) instead of refusing."""
    var allowed = Python.tuple(Python.list(PythonObject(0)), Python.list(PythonObject(1)), Python.list(PythonObject(-1)),
                               Python.list(PythonObject(0), PythonObject(1)), Python.list(PythonObject(-1), PythonObject(1)))
    return kind == "integer" and py_contains(allowed, classes)


struct MxMat(Copyable, Movable):
    """A validated float32 C-ordered matrix: (rows, cols) and its rank."""
    var a: PythonObject
    var rows: Int
    var cols: Int
    var ndim: Int

    def __init__(out self, a: PythonObject, rows: Int, cols: Int, ndim: Int):
        self.a = a
        self.rows = rows
        self.cols = cols
        self.ndim = ndim


def mx_f32_view(source: PythonObject, name: String, context: PythonObject) raises -> Tuple[PythonObject, String]:
    """`materialize_f32_lists(source, name)`: the admitted view (a list of
    numbers cast to float32) and its typestr."""
    var b = mx_b()
    var view = classical_view(source, context, name)
    var typestr = classical_dtype_str(view, context)
    if (typestr == "<f8" or typestr == "<i8") and not classical_is_array(source, context):
        var listed = py_type(source) is b.list or py_type(source) is b.tuple
        var buffered = True
        if not listed:
            try:
                _ = native_buffer_view(source)
            except:
                buffered = False
        if listed or not buffered:
            view = family_cast_view(view, name, context)
            typestr = "<f4"
    return (view, typestr)


def mx_shape_text(view: PythonObject, rows: Int, cols: Int, nd: Int) raises -> String:
    if nd == 1:
        return "(" + String(rows) + ",)"
    if nd == 2:
        return "(" + String(rows) + ", " + String(cols) + ")"
    return family_shape_repr(view)


def mx_scores(source: PythonObject, n: Int, caller: String, ndim: Int, name: String,
              context: PythonObject) raises -> MxMat:
    """`_scores(y_score, n, caller, ndim=ndim, name=name)`; ndim < 0 = any."""
    var got = mx_f32_view(source, name, context)
    var view = got[0]
    if got[1] != "<f4":
        raise Error("TypeError: mojolearn " + caller + ": " + name + " must have dtype float32; cast explicitly")
    var nd = Int(py=view.ndim)
    var rows = Int(py=view.shape[0]) if nd >= 1 else 0
    var cols = Int(py=view.shape[1]) if nd >= 2 else 1
    if nd == 2 and cols == 1 and ndim == 1:
        nd = 1
    if ndim >= 0 and nd != ndim:
        raise Error("mojolearn " + caller + ": " + name + " must be " + String(ndim) + "-D, got shape "
                    + mx_shape_text(view, rows, cols, nd))
    if rows != n:
        raise Error("mojolearn " + caller + ": " + name + " has " + String(rows) + " rows for " + String(n) + " samples")
    var checked = family_view(view, -1, name, context)
    var a = family_cast_view(checked, name, context)
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(a))), rows * cols):
        raise Error("mojolearn " + caller + ": " + name + " must be finite")
    return MxMat(a, rows, cols, nd)


def mx_zero_division_value(zero_division: PythonObject, context: PythonObject) raises -> Float64:
    """`_zero_division_value`."""
    if mx_is_str(zero_division):
        if String(py=zero_division) == "warn":
            return 0.0
    elif not mx_is_bool(zero_division) and mx_is_real(zero_division, context):
        var zero_one = Python.tuple(PythonObject(0), PythonObject(1))
        if py_contains(zero_one, zero_division):
            return mx_float(zero_division)
        if mx_isnan(mx_float(zero_division)):
            return mx_nan()
    raise Error("zero_division must be 'warn', 0.0, 1.0 or np.nan")
