# SPDX-License-Identifier: Apache-2.0
# The shared pieces of the classification metrics owner
# (bindings/metrics_cls_native.mojo, lane mcb2-mcls): what
# python/mojolearn/_metrics_impl.py and the _expansion_metrics.py helpers it
# called did before the public functions became one native call each.
#
# - the label input step (`_classification_labels`, `_classification_encoded`,
#   `_label_set`, `_label_map`, `_selected_labels`): integer label buffers
#   are encoded by the base binding's `prepare_targets` (the entry
#   `_labels._encode_labels_native` called) and mapped by its `gather_i32`
#   (the entry `_label_map` called); str labels and Python ints beyond int64
#   are the explicit label input step, read through the builtins (C-level
#   set, sorted, map and dict lookups: no per-label Mojo loop);
# - the 1-D loaders of the cuML kernels (`_as_i32_1d`, `_as_f32_1d`,
#   `_as_regression_f32_1d`, `_pair_1d`) with their refusals, word for word;
# - the x_metrics PROGRAM (`_expansion_metrics._Prog`): the same arena
#   layout, the same stage records in the same order, the same declared
#   outputs, run by the shared runner (bindings/arena_prog_native.mojo
#   `_metrics_run`, the one native call `_run` made); the result words read
#   back from the arena as `prog.ints` / `prog.words` read them.
from std.memory import memcpy, bitcast
from core.metrics_program_plan import MetricProgramPlan
from bindings.admitted_family import admit_family
from bindings.metrics_plan_admission import MetricPlanOpcodes, MetricArenaOwner
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from bindings.metrics_native_support import metric_is_real, metric_float, metric_integer, metric_is_integral_type, metric_warn, metric_undefined
from bindings.metrics_native_inputs import metric_shape, metric_shape_admitted, metric_flatten, metric_flatten_admitted
from bindings.classical_pandas import classical_admit_value
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.py_attr import py_attr
from bindings.native_buffer_owner import native_buffer_view
from bindings.classical_arrays import classical_address, classical_is_array, classical_dtype_str, classical_wrap_array
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_validate import classical_view
from bindings.arena_prog_native import _metrics_run_plan
from bindings.family_common import (
    FAMILY_F32P, FAMILY_F64P, FAMILY_I32P, FAMILY_I64P, family_repr, family_str, family_isinstance, family_call,
    family_call_kw, family_array,
)
from core.host_finite_scan import host_all_finite_f32
from core.admitted_ids import admitted_ids_i64, admitted_narrow_i32
from core.cast_in import CastIn, CAST_KIND_SIGNED
from core.label_cast import host_unique_inverse_cast
from core.metrics_exp_impl import mexp_gather_i32
from providers.python.metrics_vector_input_pyapi import metric_vector_exact, metric_i32_vector, metric_f32_vector
from bindings.metrics_reg_native import _reg_weights
from bindings.native_array_storage import native_array_zeros
from bindings.p3_labels_metrics_native import (
    p3_integral_list, p3_text_codes, p3_text_classes, P3_TEXT_OK)
from bindings.cpython_table import cpython_table

comptime MC_NONE = -1
comptime MC_PARAMS = 14
comptime MC_INT32_MAX = 2147483647


# ---------------------------------------------------------------- Python glue


def mc_b() raises -> PythonObject:
    return native_loaded_module("builtins")


def mc_len(value: PythonObject) raises -> Int:
    """`len(value)`; a Python exception it raises is restored unchanged."""
    # p3-labels-metrics: an exact list, tuple, dict, set or str answers
    # through the C length slot (it cannot raise); anything else keeps the
    # builtins call, whose exception class the door restores.
    var b = mc_b()
    var cls = py_type(value)
    if cls is b.list or cls is b.tuple or cls is b.dict or cls is b.set or cls is b.str:
        return len(value)
    return Int(py=family_call(b.len, value))


def mc_bool(value: PythonObject) raises -> Bool:
    """`bool(value)` (Python truthiness; an ambiguous array raises as it did)."""
    var b = mc_b()
    if py_type(value) is b.bool:
        return value is PythonObject(True)
    return Bool(py=family_call(b.bool, value))


def mc_eq(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a == b)`."""
    return mc_bool(family_call(native_loaded_module("operator").eq, a, b))


def mc_ne(a: PythonObject, b: PythonObject) raises -> Bool:
    """`bool(a != b)`."""
    return mc_bool(family_call(native_loaded_module("operator").ne, a, b))


def mc_in(value: PythonObject, container: PythonObject) raises -> Bool:
    """`value in container` (equality, as Python's `in`)."""
    return mc_bool(family_call(native_loaded_module("operator").contains, container, value))


def mc_is_str(value: PythonObject) raises -> Bool:
    return family_isinstance(value, mc_b().str)


def mc_is_list_like(value: PythonObject) raises -> Bool:
    var b = mc_b()
    return family_isinstance(value, Python.tuple(b.list, b.tuple))


def mc_is_bool(value: PythonObject) raises -> Bool:
    """`_labels.is_bool`: Python's bool or NumPy's bool_."""
    return family_isinstance(value, mc_b().bool) or String(py_type(value).__name__) == "bool_"


def mc_is_real(value: PythonObject, context: PythonObject) raises -> Bool:
    return metric_is_real(value, context)


def mc_float(value: PythonObject, context: PythonObject) raises -> Float64:
    """`float(value)`."""
    return metric_float(value, context)


def mc_int(value: PythonObject) raises -> Int:
    """`int(value)`."""
    return Int(py=family_call(mc_b().int, value))


def mc_sorted(values: PythonObject) raises -> PythonObject:
    return family_call(mc_b().sorted, values)


def mc_set(values: PythonObject) raises -> PythonObject:
    return family_call(mc_b().set, values)


def mc_list(values: PythonObject) raises -> PythonObject:
    return family_call(mc_b().list, values)


def mc_types(values: PythonObject) raises -> PythonObject:
    """`set(map(type, values))`: the distinct exact types of a flat list."""
    var b = mc_b()
    return family_call(b.set, family_call(b.map, b.type, values))


def mc_nan() raises -> PythonObject:
    return family_call(mc_b().float, PythonObject("nan"))


def mc_index(order: PythonObject) raises -> PythonObject:
    """`{label: i for i, label in enumerate(order)}`."""
    var b = mc_b()
    return family_call(b.dict, family_call(b.zip, order, family_call(b.range, PythonObject(mc_len(order)))))


def mc_warn(message: String, category: PythonObject, context: PythonObject) raises:
    metric_warn(message, category, context)


def mc_undefined(message: String, context: PythonObject) raises:
    metric_undefined(message, context)


def mc_entry(binding: PythonObject, name: String) raises -> PythonObject:
    # Exact module-dictionary lookup refuses Python replacements and module
    # __getattr__ hooks before a numerical entry can be called.
    return neural_descriptor_entry(binding, name)



def mc_i32_array(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array.from_list(values, "<i4")` of a short Python int list (a class
    table): one `array.array` conversion."""
    var store = family_call(native_loaded_module("array").array, PythonObject("i"), values)
    return classical_wrap_array(store, Python.tuple(mc_len(store)), "<i4", "C", context)


def mc_f64_array(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Array.from_list(values, "<f8")` of a short Python float list."""
    var store = family_call(native_loaded_module("array").array, PythonObject("d"), values)
    return classical_wrap_array(store, Python.tuple(mc_len(store)), "<f8", "C", context)


# ------------------------------------------------------------ shapes, formats


def mc_shape_of(obj: PythonObject, context: PythonObject) raises -> PythonObject:
    return metric_shape(obj, context)


def mc_rank(shape: PythonObject) raises -> Int:
    return mc_len(shape)


def mc_dim(shape: PythonObject, axis: Int) raises -> Int:
    return mc_int(shape[axis])


comptime MC_NO_FORMAT = "-"


def mc_buffer_format(obj: PythonObject, context: PythonObject) raises -> String:
    """`linear_model._buffer_format`: the struct format code of a buffer,
    its byte-order prefix stripped; MC_NO_FORMAT for an object without a
    buffer (a list, a scalar)."""
    classical_admit_value(obj, context)
    return mc_buffer_format_admitted(obj, context)


def mc_buffer_format_admitted(obj: PythonObject, context: PythonObject) raises -> String:
    """`mc_buffer_format` of a value already admitted in this native call."""
    if classical_is_array(obj, context):
        var dtype = String(obj.dtype)
        if dtype == "<f4":
            return "f"
        if dtype == "<f8":
            return "d"
        if dtype == "<i4":
            return "i"
        if dtype == "<i8":
            return "q"
        if dtype == "<u4":
            return "I"
        if dtype == "<u1":
            return "B"
        if dtype == "<f2":
            return "e"
        return MC_NO_FORMAT
    var view: PythonObject
    try:
        view = native_buffer_view(obj)
    except:
        return MC_NO_FORMAT
    var fmt = String(view.format)
    var start = 0
    while start < fmt.byte_length():  # small-loop(prefix: the byte-order characters of one format): at most two
        var c = String(fmt[byte=start:start + 1])
        if c == "<" or c == ">" or c == "=" or c == "@" or c == "!" or c == "|":
            start += 1
        else:
            break
    return String(fmt[byte=start:fmt.byte_length()])


def mc_dtype_name(obj: PythonObject, context: PythonObject) raises -> String:
    """`linear_model._dtype_name`."""
    var fmt = mc_buffer_format(obj, context)
    if fmt != MC_NO_FORMAT:
        return "buffer format " + family_repr(PythonObject(fmt))
    return "type " + String(py_type(obj).__name__)


def mc_flatten(obj: PythonObject, context: PythonObject) raises -> PythonObject:
    return metric_flatten(obj, context)


def mc_types_all(types: PythonObject, classes: PythonObject) raises -> Bool:
    var listed = mc_list(types)
    for i in range(mc_len(listed)):  # driver-loop: no Python metaclass/subclass hook executes
        if listed[i] is not classes:
            return False
    return True


def mc_types_integral(types: PythonObject, context: PythonObject) raises -> Bool:
    var listed = mc_list(types)
    for i in range(mc_len(listed)):  # driver-loop: exact native label type identities
        if not metric_is_integral_type(listed[i], context):
            return False
    return True


def mc_is_integer_labels(obj: PythonObject, context: PythonObject) raises -> Bool:
    """`linear_model._is_integer_labels`: an integer buffer format (bools
    are not integers), or a list whose every leaf is an exact int."""
    # One admission walk for the format probe and the flatten below.
    classical_admit_value(obj, context)
    var fmt = mc_buffer_format_admitted(obj, context)
    if fmt != MC_NO_FORMAT:
        return fmt.byte_length() == 1 and String("bBhHiIlLqQnN").find(fmt) >= 0
    var leaves = metric_flatten_admitted(obj, context)
    if mc_len(leaves) == 0:
        return False
    var types = mc_types(leaves)
    return mc_len(types) == 1 and py_contains(types, mc_b().int)


def mc_is_float64(obj: PythonObject, context: PythonObject) raises -> Bool:
    """`_metrics_impl._is_float64`: a buffer of format 'd', or a list with a
    Python float among its leaves."""
    classical_admit_value(obj, context)
    var fmt = mc_buffer_format_admitted(obj, context)
    if fmt != MC_NO_FORMAT:
        return fmt == "d"
    return py_contains(mc_types(metric_flatten_admitted(obj, context)), mc_b().float)


# ------------------------------------------------------------------- labels


struct McLabels(Movable):
    """One label vector after the label input step: its kind ("integer" or
    "string"), its length, its sorted distinct labels (`_label_set`) as a
    Python list and one int32 code per row into them."""
    var kind: String
    var n: Int
    var classes: PythonObject
    var codes: PythonObject

    def __init__(out self, kind: String, n: Int, classes: PythonObject, codes: PythonObject):
        self.kind = String(kind)
        self.n = n
        self.classes = classes
        self.codes = codes


def mc_label_list(values: PythonObject, name: String, allow_empty: Bool, context: PythonObject,
                  admitted: Bool = False) raises -> PythonObject:
    """`_metrics_impl._classification_labels`: (labels, kind) of a 1-D
    label sequence, the explicit label input step."""
    var b = mc_b()
    # `admitted`: the caller ran classical_admit_value(values) in this call.
    var shape = metric_shape_admitted(values, context) if admitted else mc_shape_of(values, context)
    if mc_rank(shape) != 1:
        raise Error(name + " must be one-dimensional single-label data")
    var source = values
    if family_isinstance(source, Python.tuple(b.str, b.bytes)):
        raise Error("mojolearn: y must be a sequence of labels")
    var labels = metric_flatten_admitted(source, context)  # admitted by mc_shape_of above
    var size = mc_len(labels)
    if not allow_empty and size == 0:
        raise Error(name + " must contain at least one label")
    var types = mc_types(labels)
    if mc_types_all(types, b.str):
        return Python.tuple(labels, PythonObject("string"))
    if mc_types_all(types, b.int):
        # every leaf is an exact Python int: `int(v)` is v itself, so the
        # flattened list (a fresh list) is the converted one; no per-label call
        return Python.tuple(labels, PythonObject("integer"))
    if mc_types_integral(types, context):
        # p3-labels-metrics: `metric_integer` per label (type tests, a scalar
        # buffer decode, a PyLong parsed from text, a Python append) becomes
        # one presized list filled by C calls: exact ints kept, bools 0 / 1,
        # NumPy integer scalars their exact int (bindings/p3_labels_metrics_native.mojo).
        return Python.tuple(p3_integral_list(labels, py_get(context, "bool_types")), PythonObject("integer"))
    raise Error("TypeError: " + name + " must contain only strings or only integers; "
                + "floating labels, missing labels and mixed types are unsupported")


def _view_kind(dtype: String) raises -> Int:
    """The integer item kind of an array typestr: 1 signed, 2 unsigned
    (bool bytes read as unsigned), 0 none (floats, str, object, uint64)."""
    if dtype.byte_length() < 3:
        return 0
    var tail = String(dtype[byte=1:dtype.byte_length()])
    if tail == "i1" or tail == "i2" or tail == "i4" or tail == "i8":
        return 1
    if tail == "u1" or tail == "u2" or tail == "u4" or tail == "b1":
        return 2
    return 0


@fieldwise_init
struct MetricIntegerView(Movable):
    var view: PythonObject
    var count: Int
    var item: Int
    var stride: Int
    var signed: Bool


def mc_integer_view(values: PythonObject, context: PythonObject) raises -> Optional[MetricIntegerView]:
    """The existing integer-store admission; storage choice follows it."""
    var view: PythonObject
    try:
        view = classical_view(values, context, "y")
    except:
        return None
    if Int(py=view.ndim) != 1:
        return None
    var kind = _view_kind(classical_dtype_str(view, context))
    if kind == 0:
        return None
    var count = Int(py=view.shape[0])
    if count < 1:
        return None
    var item = Int(py=view.itemsize)
    var stride = Int(py=view.strides[0])
    return MetricIntegerView(view, count, item, stride, kind == 1)


def mc_int_store(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """Legacy Array store wrapper over the shared admitted view."""
    var input = mc_integer_view(values, context)
    if not input:
        return PythonObject(None)
    var count = input.value().count
    # Keep the existing zeroed Int64 store and native conversion route.
    var store = native_array_zeros(String("q"), count)
    var dst = FAMILY_I64P(unsafe_from_address=Int(py=py_call(store.buffer_info)[0]))
    var admitted = admitted_ids_i64(Int(py=classical_address(input.value().view)), input.value().stride,
        0, count, 1, input.value().item, input.value().signed, False, dst)
    _ = input  # retain the source owner through success and refusal
    _ = store
    if not admitted:
        return PythonObject(None)
    return Python.tuple(store, PythonObject(count))


def mc_encode_signed(addr: Int, item: Int, count: Int, classes_addr: Int, codes_addr: Int) raises -> Int:
    """The ORDER RULE over `count` signed little-endian integer labels of
    `item` bytes (contiguous): the sorted distinct values written to
    `classes_addr` (items of the same width) and one dense int32 code per
    row; returns k. The host label encoder called natively: no Python call
    into the base binding's `prepare_targets` entry, the same classes and
    codes (exact integers)."""
    var src = CastIn(addr, CAST_KIND_SIGNED, item, count, 1, item, item)
    var k = host_unique_inverse_cast(src, 1, classes_addr, codes_addr)
    if k < 0:
        raise Error("RuntimeError: prepare_targets refused integer labels")
    return k


def mc_encode_store(store: PythonObject, count: Int, context: PythonObject) raises -> PythonObject:
    """`(classes, codes)` of `count` int64 labels (the ORDER RULE: sorted
    distinct values, one dense int32 code per row), encoded natively: the
    classes as a Python int list."""
    # p3-labels-metrics: the class words land in a native scratch list and
    # only the k classes become Python ints (one presized list), instead of
    # an n-slot bytes + array store, a slice and a tolist.
    var classes = List[Int64](length=max(count, 1), fill=0)
    var codes = family_array(Python.tuple(count), "<i4", context)
    var k = mc_encode_signed(Int(py=py_call(store.buffer_info)[0]), 8, count,
                             Int(classes.unsafe_ptr()), Int(py=classical_address(codes)))
    _ = store  # keep the label store alive through the native call above
    _ = codes  # keep the Python object alive through the native call above
    ref cpy = Python().cpython()
    var t = cpython_table()
    var raw = t[].PyList_New(k)
    if not raw:
        raise cpy.unsafe_get_error()
    var listed = PythonObject(from_owned=raw)
    for c in range(k):  # driver-loop: over the k distinct classes
        var item = t[].PyLong_FromLongLong(classes[c])
        if not item:
            raise cpy.unsafe_get_error()
        _ = t[].PyList_SetItem(listed._obj_ptr, c, item)
    _ = classes^
    return Python.tuple(listed, codes)


def mc_fast_labels(values: PythonObject, context: PythonObject, admitted: Bool = False) raises -> PythonObject:
    """`_native_classification_labels`: (classes, codes, n) of an integer
    or bool label buffer, or a flat list of exact Python ints within int64;
    None for everything the label input step reads."""
    var b = mc_b()
    if family_isinstance(values, Python.tuple(b.str, b.bytes)):
        return PythonObject(None)
    var store = PythonObject(None)
    var count = 0
    if py_type(values) is b.list or py_type(values) is b.tuple:
        count = mc_len(values)
        if count == 0:
            return PythonObject(None)
        var types = mc_types(values)
        if mc_len(types) != 1 or not py_contains(types, b.int):
            return PythonObject(None)
        try:
            store = py_call(native_loaded_module("array").array, PythonObject("q"), values)
        except:
            return PythonObject(None)   # an int outside int64: a class only Python holds
    else:
        if not classical_is_array(values, context):
            try:
                _ = native_buffer_view(values)
            except:
                return PythonObject(None)
        var shape = metric_shape_admitted(values, context) if admitted else mc_shape_of(values, context)
        if mc_rank(shape) != 1:
            return PythonObject(None)
        var packed = mc_int_store(values, context)
        if packed is None:
            return PythonObject(None)
        store = packed[0]
        count = Int(py=packed[1])
    var encoded = mc_encode_store(store, count, context)
    return Python.tuple(encoded[0], encoded[1], PythonObject(count))


def mc_labels(values: PythonObject, name: String, context: PythonObject, admitted: Bool = False) raises -> McLabels:
    """`_classification_encoded(values, name)`: the native encoder for an
    integer buffer, else the label input step, then one code per row."""
    var b = mc_b()
    var fast = mc_fast_labels(values, context, admitted)
    if fast is not None:
        return McLabels("integer", Int(py=fast[2]), fast[0], fast[1])
    var listed = mc_label_list(values, name, False, context, admitted)
    var labels = listed[0]
    if String(listed[1]) == "integer":
        # The converted labels are exact Python ints (pandas, uint64 or
        # nested inputs flattened above): within int64 they take the native
        # encoder as one int64 store; an int outside int64 keeps the Python
        # label step below.
        var store = PythonObject(None)
        try:
            store = py_call(native_loaded_module("array").array, PythonObject("q"), labels)
        except:
            store = PythonObject(None)
        if store is not None:
            var count = mc_len(store)
            var encoded = mc_encode_store(store, count, context)
            return McLabels("integer", count, encoded[0], encoded[1])
    if String(listed[1]) == "string":
        # p3-labels-metrics: exact str labels hashed to their k distinct
        # values by UTF-8 bytes, only the k classes sorted (code-point order,
        # as `sorted(set(...))`), int32 codes written natively.
        var fast = p3_text_codes(labels, False)
        if fast.status == P3_TEXT_OK:
            var n = len(fast.codes)
            var fast_codes = family_array(Python.tuple(n), "<i4", context)
            var target = FAMILY_I32P(unsafe_from_address=Int(py=classical_address(fast_codes)))
            for i in range(n):  # driver-loop: native copy of the n codes
                target[i] = fast.codes[i]
            _ = fast_codes  # keep the Python object alive through the native writes above
            return McLabels("string", n, p3_text_classes(labels, fast.first), fast_codes)
    var classes = mc_sorted(mc_set(labels))
    var index = mc_index(classes)
    # cpu-route: Python ints outside int64 or str with no UTF-8 form, the explicit label input step (dict lookups by builtins map)
    var codes = mc_i32_array(family_call(b.list, family_call(b.map, py_attr(index, "__getitem__"), labels)), context)
    return McLabels(String(listed[1]), mc_len(labels), classes, codes)


def mc_union(a: McLabels, b: McLabels) raises -> PythonObject:
    """`sorted(_label_set(a) | _label_set(b))`."""
    return mc_sorted(family_call(mc_set(a.classes).union, b.classes))


def mc_gather(labels: McLabels, table: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_label_map`: the per-class int32 `table` gathered per row, natively
    (the codes are in range by construction; no Python call into the base
    binding's `gather_i32` entry)."""
    var t = mc_i32_array(table, context)
    var out = family_array(Python.tuple(labels.n), "<i4", context)
    _ = mexp_gather_i32(FAMILY_I32P(unsafe_from_address=Int(py=classical_address(t))), mc_len(labels.classes),
                        FAMILY_I32P(unsafe_from_address=Int(py=classical_address(labels.codes))), labels.n,
                        FAMILY_I32P(unsafe_from_address=Int(py=classical_address(out))))
    _ = t  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = t^
    return out


def mc_map_order(labels: McLabels, order: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_label_map(labels, lambda v: index.get(v, -1))` over `order`."""
    var b = mc_b()
    var index = mc_index(order)
    var table = family_call(b.list, family_call(b.map, index.get, labels.classes,
                                                family_call(native_loaded_module("itertools").repeat, PythonObject(-1))))
    return mc_gather(labels, table, context)


def mc_map_equal(labels: McLabels, value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_label_map(labels, lambda v: int(v == value))`."""
    var b = mc_b()
    var eq = family_call(b.map, native_loaded_module("operator").eq, labels.classes,
                         family_call(native_loaded_module("itertools").repeat, value))
    return mc_gather(labels, family_call(b.list, family_call(b.map, b.int, eq)), context)


def mc_selected(labels: PythonObject, kind: String, observed: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_metrics_impl._selected_labels`."""
    if labels is None:
        return family_call(py_attr(observed, "copy"))
    var listed = mc_label_list(labels, "labels", False, context)
    if String(listed[1]) != kind:
        raise Error("TypeError: labels must use the same type as the targets")
    var selected = listed[0]
    if mc_len(mc_set(selected)) != mc_len(selected):
        raise Error("labels must be unique")
    return selected


def mc_extend_order(chosen: PythonObject, present: PythonObject) raises -> PythonObject:
    """`chosen + [v for v in present if v not in set(chosen)]`
    (`_label_order`)."""
    var b = mc_b()
    var keep = family_call(b.set, chosen)
    var out = family_call(b.list, chosen)
    for i in range(mc_len(present)):  # driver-loop: over the present classes
        var v = present[i]
        if not py_contains(keep, v):
            _ = family_call(out.append, v)
    return out


def mc_positive(pos_label: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_classification_labels([pos_label], "pos_label")`: (value, kind)."""
    var listed = mc_label_list(Python.list(pos_label), "pos_label", False, context)
    return Python.tuple(listed[0][0], listed[1])


def mc_refuse_multilabel(y: PythonObject, name: String, caller: String, context: PythonObject,
                         admitted: Bool = False) raises:
    """`_expansion_metrics._refuse_multilabel`."""
    var shape = metric_shape_admitted(y, context) if admitted else mc_shape_of(y, context)
    if mc_rank(shape) == 2 and mc_dim(shape, 1) != 1:
        raise Error("NotImplementedError: mojolearn " + caller + ": " + name + " is 2-D; multilabel-indicator "
                    + "targets are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv), only binary and multiclass 1-D labels")


# ------------------------------------------------------------- the 1-D loaders


def mc_vector_shape(x: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_vector_shape`: (shape, column) of a 1-D input or an (n, 1) column."""
    var shape = mc_shape_of(x, context)
    var column = mc_rank(shape) == 2 and mc_dim(shape, 1) == 1
    if mc_rank(shape) != 1 and not column:
        raise Error("mojolearn metrics: " + name + " must be 1-D, got shape " + family_str(shape))
    if mc_dim(shape, 0) == 0:
        raise Error("mojolearn metrics: " + name + " is empty")
    return Python.tuple(shape, PythonObject(column))


def mc_i32_1d(x: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_metrics_impl._as_i32_1d`: integer labels that fit in int32, as an
    int32 C-order Array of n entries."""
    if metric_vector_exact(x, context):
        return metric_i32_vector(x, name, context)
    var b = mc_b()
    _ = mc_vector_shape(x, name, context)
    if not mc_is_integer_labels(x, context):
        raise Error("mojolearn metrics: " + name + " must be an integer label array, got "
                    + mc_dtype_name(x, context) + "; the ported kernels take int32 labels")
    var store: PythonObject
    var count: Int
    if mc_is_list_like(x):
        # `as_i64_c` of a list: its exact Python ints (a column list flattened)
        var flat = metric_flatten_admitted(x, context)  # admitted by mc_is_integer_labels above
        store = family_call(native_loaded_module("array").array, PythonObject("q"), flat)
        count = mc_len(store)
    else:
        var packed = mc_int_store(x, context)
        if packed is None:
            raise Error("mojolearn metrics: " + name + " must be an integer label array, got "
                        + mc_dtype_name(x, context) + "; the ported kernels take int32 labels")
        store = packed[0]
        count = Int(py=packed[1])
    var out = family_array(Python.tuple(count), "<i4", context)
    if not admitted_narrow_i32(FAMILY_I64P(unsafe_from_address=Int(py=py_call(store.buffer_info)[0])), count,
                           FAMILY_I32P(unsafe_from_address=Int(py=classical_address(out)))):
        raise Error("mojolearn metrics: " + name + " does not fit in int32; the implemented "
                    + "kernels are the int32 instantiation (the int64 overload of "
                    + "adjusted_rand_index is the same code at a wider type and is "
                    + "not instantiated, metrics/NOT_IMPLEMENTED.tsv)")
    _ = out  # keep the Python object alive through the native calls above
    _ = store^
    return out


def mc_f32_1d(x: PythonObject, name: String, require_finite: Bool, context: PythonObject) raises -> PythonObject:
    """`_metrics_impl._as_f32_1d`: a float32 C-order array of n entries
    (an (n, 1) column is the same words)."""
    if metric_vector_exact(x, context):
        return metric_f32_vector(x, name, require_finite, False, context)
    var vs = mc_vector_shape(x, name, context)
    if mc_is_float64(x, context):
        raise Error("TypeError: mojolearn metrics: " + name + " is float64; the float64 overloads of "
                    + "r2_score, kl_divergence and silhouette_score are NOT implemented "
                    + "(no float64 on this GPU). Cast to float32 yourself so the "
                    + "precision you run at is the one you chose.")
    var out = admit_family(x, mc_rank(vs[0]), name, context).array
    if require_finite and not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out))),
                                                    Int(py=out.size)):
        raise Error("mojolearn metrics: " + name + " contains NaN or infinity; refused "
                    + "here (metrics/README.md HAND-OFF ask 3) rather than letting it "
                    + "reach a kernel whose answer for it is a washed sentinel")
    _ = out  # keep the Python object alive through the native calls above
    return out


def mc_f32_lists(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_buffer.materialize_f32_lists(source, name)[0]`: the input as an
    array; nested Python numbers (float64 or int64 when packed) are cast to
    float32 once, a buffer keeps its own dtype."""
    var b = mc_b()
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
            view = admit_family(view, -1, name, context, admitted=True, empty=True).array
    return view


def mc_regression_f32_1d(x: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_metrics_impl._as_regression_f32_1d`."""
    if metric_vector_exact(x, context):
        return metric_f32_vector(x, name, True, True, context)
    if mc_buffer_format(x, context) != "f" and not mc_is_list_like(x):
        raise Error("TypeError: mojolearn regression metrics: " + name
                    + " must have dtype float32; cast explicitly before scoring")
    var a = mc_f32_lists(x, "input", context)
    var dtype = classical_dtype_str(a, context)
    if dtype != "<f4":
        raise Error("TypeError: mojolearn regression metrics: " + name + " must have dtype float32, got "
                    + dtype + "; cast explicitly before scoring")
    return mc_f32_1d(a, name, True, context)


def mc_pair_check(x: PythonObject, y: PythonObject, name_a: String, name_b: String) raises:
    """`_pair_1d`'s length rule."""
    var nx = Int(py=x.size)
    var ny = Int(py=y.size)
    if nx != ny:
        raise Error("mojolearn metrics: " + name_a + " has " + String(nx) + " entries and "
                    + name_b + " has " + String(ny) + "; they must be the same length")


def mc_weights(source: PythonObject, n: Int, caller: String, context: PythonObject) raises -> PythonObject:
    """`_metrics_impl._sample_weight_f32` (None stays None): 1-D, n entries,
    finite and nonnegative, a positive total, cast to float32."""
    return _reg_weights(source, n, caller, context)


# ------------------------------------------------------------- the program


struct McProg(Movable):
    # Pure native stage/layout words; only boundary input owners live here.
    var plan: MetricProgramPlan
    var inputs: List[PythonObject]
    var ops: MetricPlanOpcodes

    def __init__(out self, context: PythonObject) raises:
        self.plan = MetricProgramPlan()
        self.inputs = List[PythonObject]()
        self.ops = MetricPlanOpcodes(context["family_metrics_ops"])

    def alloc(mut self, n: Int) raises -> Int:
        return self.plan.alloc(n)

    def put(mut self, array: PythonObject) raises -> Int:
        """A float32 (or int32, stored as its bits) C-ordered array -> its offset."""
        var off = self.plan.input(Int(py=array.size))
        self.inputs.append(array)
        return off

    def scratch(mut self, n: Int) raises -> Int:
        return self.plan.scratch(n)

    def stage(mut self, op: String, total: Int, params: List[Int]) raises:
        if len(params) > MC_PARAMS:
            raise Error("x_metrics: too many stage parameters")
        var opcode = self.ops.get(op)
        self.plan.stage(opcode, total, params)

    def want(mut self, off: Int, n: Int, count: Int = MC_NONE, mult: Int = 1) raises -> Int:
        return self.plan.want(off, n, count, mult)


def mc_run(prog: McProg, mode: PythonObject, context: PythonObject) raises -> MetricArenaOwner:
    """`_run(prog, numeric_mode)`: the program handed to the shared runner
    (the one call the Python `_run` made); returns the arena store."""
    var prog_context = context["family_metrics_prog_context"]
    return _metrics_run_plan(prog.plan, prog.inputs, mode, prog_context)


def mc_arena_base(arena: MetricArenaOwner) raises -> Int:
    return arena.base


def mc_arena_int(arena: MetricArenaOwner, off: Int) raises -> Int:
    """`prog.ints(off, 1)[0]`."""
    var result = Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=arena.base + 4 * off)[0])
    _ = arena.store
    return result


def mc_arena_float(arena: MetricArenaOwner, off: Int) raises -> Float64:
    """`prog.floats(off, 1)[0]`: the exact image of a Float32 word."""
    var result = Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=arena.base + 4 * off)[0])
    _ = arena.store
    return result


def mc_arena_double(arena: MetricArenaOwner, off: Int) raises -> Float64:
    """`prog.words(off, 2, "d")[0]`: the binary64 at arena[off], arena[off + 1], low first."""
    var words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=arena.base + 4 * off)
    var result = bitcast[DType.float64](UInt64(words[0]) | (UInt64(words[1]) << 32))
    _ = arena.store
    return result


def mc_arena_i64(arena: MetricArenaOwner, off: Int) raises -> Int:
    """`prog.words(off, 2, "q")[0]`."""
    var words = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=arena.base + 4 * off)
    var result = Int(bitcast[DType.int64](UInt64(words[0]) | (UInt64(words[1]) << 32)))
    _ = arena.store
    return result


def mc_arena_f64_array(arena: MetricArenaOwner, off: Int, count: Int, shape: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`Array._owned(prog.words(off, 2 * count, "d"), shape, "<f8")`."""
    var out = family_array(shape, "<f8", context)
    if count > 0:
        memcpy(dest=FAMILY_F64P(unsafe_from_address=Int(py=classical_address(out))),
               src=FAMILY_F64P(unsafe_from_address=arena.base + 4 * off), count=count)
        _ = out  # keep both owners alive through the native copy above
        _ = arena.store
    return out


def mc_arena_i64_array(arena: MetricArenaOwner, off: Int, count: Int, shape: PythonObject,
                       context: PythonObject) raises -> PythonObject:
    """`Array._owned(prog.words(off, 2 * count, "q"), shape, "<i8")`."""
    var out = family_array(shape, "<i8", context)
    if count > 0:
        memcpy(dest=FAMILY_I64P(unsafe_from_address=Int(py=classical_address(out))),
               src=FAMILY_I64P(unsafe_from_address=arena.base + 4 * off), count=count)
        _ = out  # keep both owners alive through the native copy above
        _ = arena.store
    return out


def mc_f64_words(values: List[Float64], context: PythonObject) raises -> PythonObject:
    """Binary64 values as their Int32 word pairs (low first), an int32
    Array (`array.array("d", values).tobytes()` read as "i")."""
    var out = family_array(Python.tuple(2 * len(values)), "<i4", context)
    var dst = FAMILY_F64P(unsafe_from_address=Int(py=classical_address(out)))
    _ = out  # keep the Python object alive through the native call above
    for i in range(len(values)):  # small-loop(values: the binary64 parameters of one stage): bounded by the call site
        dst[i] = values[i]
        _ = out  # keep the Python object alive through the native call above
    return out


# ------------------------------------------------------------ program pieces


def mc_group(mut prog: McProg, key: Int, n: Int, m: Int, values: Int, vstride: Int, weights: Int,
             width: Int) raises -> Tuple[Int, Int]:
    """`_group`: the stable counting sort and the PairSums; (OFF, OUT)."""
    var off = prog.alloc(m + 1)
    var order = prog.scratch(n)
    var out = prog.alloc(m * width)
    prog.stage("group_sort", 1, [key, n, m, off, order])
    if m * width:
        prog.stage("group_sum", m * width, [off, order, values, vstride, weights, out, width])
    return (off, out)


def mc_counts(mut prog: McProg, off: Int, m: Int) raises -> Int:
    """`_counts`: the m group sizes (off_diff)."""
    var out = prog.alloc(m)
    if m:
        prog.stage("off_diff", m, [off, m, out])
    return out
