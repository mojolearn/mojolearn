# SPDX-License-Identifier: Apache-2.0
# Shared pieces of the SVM family owner (bindings/svc_native.mojo): the
# argument conversions, the label encoding and decoding, the per-row label
# maps and the two `score` metrics the Python methods of
# python/mojolearn/_svm_impl.py, svm.py and _expansion_neighbors.py
# (OneClassSVM) used, with their refusals in the same wording and order.
#
# Every compiled step is the one the Python called: the base binding's host
# helpers (`gather_i64`, `threshold_labels_i64`, `argmax_rows_f32`, ...),
# resolved through `_buffer._native` (the module registers it as
# `family_svc_native`), and the x_metrics programs of `_r2_sums` and
# `accuracy_fraction`, rebuilt here stage for stage (the `_Prog` of
# bindings/metrics_reg_native.mojo, run by the shared runner). The label
# ORDER RULE (`_labels.sorted_classes`) is the builtins' `sorted` over the
# first-seen labels, as the Python label input step ran it: the classes and
# codes are the ones `encode_labels` returned for a flat label list.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.py_attr import py_attr
from bindings.classical_arrays import classical_address, classical_is_array, classical_wrap_array
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense
from providers.python.array_metadata_pyapi import ArrayMetadata
from bindings.family_common import (
    FAMILY_F32P, FAMILY_F64P, FAMILY_I32P, FAMILY_I64P, family_repr, family_isinstance, family_call,
    family_entry, family_array, family_view, family_cast_view, family_as_f32, family_shape_repr, family_f32_snapshot,
)
from bindings.metrics_reg_native import _Prog, _Reg, _reg_run, _arena_double, _arena_int, _RE_DIV
from core.family_cast_impl import family_all_finite_f32

comptime SVC_NONE = -1
# cm_epi COUNT (x_metrics/cm_epi.mojo), `_expansion_metrics._CM_COUNT`.
comptime _CM_COUNT = 2


# ---------------------------------------------------------------- small reads


def svc_builtins() raises -> PythonObject:
    return native_loaded_module("builtins")


def svc_field(fields: PythonObject, name: String) raises -> PythonObject:
    """`self.<name>` of an instance attribute (the instance dictionary, so no
    attribute hook runs); refused as the lookup refused it."""
    var value = py_get(fields, PythonObject(name), PythonObject(None))
    if value is None and not py_contains(fields, PythonObject(name)):
        raise Error("AttributeError: object has no attribute '" + name + "'")
    return value


def svc_attr(model: PythonObject, name: String) raises -> PythonObject:
    """`getattr(model, name)` for a class-level constant or a parameter a
    subclass may define on the class."""
    return family_call(svc_builtins().getattr, model, PythonObject(name))


def svc_has(fields: PythonObject, name: String) raises -> Bool:
    """`hasattr(self, name)` for an attribute fit sets on the instance."""
    return py_contains(fields, PythonObject(name))


def svc_name(model: PythonObject) raises -> String:
    """`type(self).__name__`."""
    return String(py_type(model).__name__)


def svc_float(value: PythonObject) raises -> Float64:
    """`float(value)` with Python's conversion (and its refusals)."""
    return Float64(py=family_call(svc_builtins().float, value))


def svc_int(value: PythonObject) raises -> Int:
    """`int(value)` with Python's conversion (and its refusals)."""
    return Int(py=family_call(svc_builtins().int, value))


def svc_truthy(value: PythonObject) raises -> Bool:
    return Bool(py=family_call(svc_builtins().bool, value))


def svc_round_f32(value: Float64) -> Float64:
    """`linear_model._round_f32`: one round-to-nearest-even to binary32."""
    return Float64(Float32(value))


def svc_isfinite(value: Float64) -> Bool:
    return value == value and value - value == 0.0


def svc_base(name: String, context: PythonObject) raises -> PythonObject:
    """The base binding's host helper `name` (`_buffer._native(name)`)."""
    return family_call(context["family_svc_native"], PythonObject(name))


def svc_mode(fields: PythonObject) raises -> PythonObject:
    """`getattr(self, "numeric_mode", None)`."""
    return py_get(fields, PythonObject("numeric_mode"), PythonObject(None))


def svc_binding(name: String, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_backend.binding(name, mode)`: binding acquisition."""
    return family_call(context["backend_fields"]["binding"], PythonObject(name), mode)


def svc_zero_division(float_division: Bool) raises:
    """Raise Python's ZeroDivisionError with the message the Python division
    raised (`division by zero` for ints, `float division by zero`)."""
    var operator = Python.import_module("operator")
    if float_division:
        _ = family_call(operator.truediv, PythonObject(1.0), PythonObject(0.0))
    _ = family_call(operator.truediv, PythonObject(1), PythonObject(0))
    raise Error("RuntimeError: division by zero")


# ---------------------------------------------------------------- arrays


def svc_prefix(source: PythonObject, count: Int, shape: PythonObject, typestr: String,
               context: PythonObject) raises -> PythonObject:
    """`source[:count].reshape(shape)`: a new compact C-order Array holding
    the first `count` items of the C-order `source` (an exact copy)."""
    var out = family_array(shape, typestr, context)
    var width = 8 if typestr == "<f8" or typestr == "<i8" else 4
    if count > 0:
        memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(out))),
               src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(source))),
               count=count * width)
        _ = out  # keep the Python object alive through the native call above
    return out


def svc_scalar_array(value: Float64, typestr: String, context: PythonObject) raises -> PythonObject:
    """`Array.from_list([value], typestr)` for "<f4" or "<f8"."""
    var out = family_array(Python.tuple(1), typestr, context)
    if typestr == "<f4":
        FAMILY_F32P(unsafe_from_address=Int(py=classical_address(out)))[0] = Float32(value)
        _ = out  # keep the Python object alive through the native call above
    else:
        FAMILY_F64P(unsafe_from_address=Int(py=classical_address(out)))[0] = value
        _ = out  # keep the Python object alive through the native call above
    return out


def svc_read_f64(array: PythonObject, index: Int) raises -> Float64:
    return FAMILY_F64P(unsafe_from_address=Int(py=classical_address(array)))[index]


def svc_read_f32(array: PythonObject, index: Int) raises -> Float64:
    return Float64(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(array)))[index])


def svc_read_i32(array: PythonObject, index: Int) raises -> Int:
    return Int(FAMILY_I32P(unsafe_from_address=Int(py=classical_address(array)))[index])


def svc_f32(source: PythonObject, ndim: Int, name: String, dense: Bool,
            context: PythonObject) raises -> PythonObject:
    """`_buffer.as_f32_c` (or `as_f32_dense_c` with `dense`): the rank and
    emptiness refusals, the cast; returns (array, copied). A scipy.sparse
    matrix is densified exactly first (its implicit entries are zeros)."""
    var snapshot = svc_snapshot(source, ndim, name, dense, context)
    return Python.tuple(snapshot.keep, PythonObject(snapshot.copied))


def svc_snapshot(source: PythonObject, ndim: Int, name: String, dense: Bool,
                 context: PythonObject) raises -> ArrayMetadata:
    var b = svc_builtins()
    var sequence = py_type(source) is b.list or py_type(source) is b.tuple or py_type(source) is b.range
    var snapshot = family_f32_snapshot(source, ndim, name, context, dense)
    snapshot.copied = snapshot.copied or sequence
    return snapshot^


# ---------------------------------------------------------------- shapes and labels


def svc_shape_of(obj: PythonObject) raises -> PythonObject:
    """`_buffer.shape_of`: `.shape` when the object has one, the nesting of a
    list or tuple otherwise, `(len,)` for a sized object, else ()."""
    var b = svc_builtins()
    var shape = family_call(b.getattr, obj, PythonObject("shape"), PythonObject(None))
    if shape is not None:
        return family_call(b.tuple, shape)
    var seq = Python.tuple(b.list, b.tuple)
    if family_isinstance(obj, seq):
        if len(obj) > 0 and family_isinstance(obj[0], seq):
            return Python.tuple(PythonObject(len(obj)), PythonObject(len(obj[0])))
        return Python.tuple(PythonObject(len(obj)))
    if Bool(py=family_call(b.hasattr, obj, PythonObject("__len__"))):
        return Python.tuple(family_call(b.len, obj))
    return Python.tuple()


def _svc_flatten_into(item: PythonObject, out_buf: PythonObject) raises:
    """Depth-first leaves of a nested list/tuple, the order the level-by-level
    chain of `flatten_labels` keeps."""
    var b = svc_builtins()
    var seq = Python.tuple(b.list, b.tuple)
    if not family_isinstance(item, seq):
        _ = py_call(out_buf.append, item)
        return
    for i in range(len(item)):  # driver-loop: over the nesting of a user label list (the label input step)
        var leaf = item[i]
        if family_isinstance(leaf, seq):
            _svc_flatten_into(leaf, out_buf)
        else:
            _ = py_call(out_buf.append, leaf)


def svc_plain_labels(labels: PythonObject, context: PythonObject) raises -> String:
    """`_labels._plain_labels`: "number" when every label is exactly an int,
    float or bool, "str" when every one is exactly a str, else ""."""
    var b = svc_builtins()
    if len(labels) == 0:
        return ""
    var kinds = family_call(b.set, family_call(b.map, b.type, labels))
    var numbers = family_call(b.frozenset, Python.list(b.int, b.float, b.bool))
    if Bool(py=family_call(kinds.issubset, numbers)):
        return "number"
    if len(kinds) == 1 and py_contains(kinds, b.str):
        return "str"
    return ""


def svc_flatten_labels(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.flatten_labels(y)`: a flat Python list of the label objects
    (`tolist()` for an array, nested lists and tuples flattened)."""
    var b = svc_builtins()
    var seq = Python.tuple(b.list, b.tuple)
    var src = y
    if not family_isinstance(y, seq) and Bool(py=family_call(b.hasattr, y, PythonObject("tolist"))):
        if classical_is_array(y, context) and Int(py=y.ndim) == 1 and String(y.dtype) != "<f2":
            # the Array's own memoryview lists its values in C
            src = family_call(y._mv.tolist)
        else:
            src = family_call(family_call(b.getattr, y, PythonObject("tolist")))
    if family_isinstance(src, Python.tuple(b.str, b.bytes)):
        raise Error("mojolearn: y must be a sequence of labels")
    if (py_type(src) is b.list or py_type(src) is b.tuple) and svc_plain_labels(src, context).byte_length() > 0:
        return family_call(b.list, src)
    var out = Python.list()
    _svc_flatten_into(src, out)
    return out


def svc_labels_1d(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`linear_model._labels_1d`: (labels list or None, shape)."""
    var shape = svc_shape_of(y)
    if len(shape) != 1:
        return Python.tuple(PythonObject(None), shape)
    return Python.tuple(svc_flatten_labels(y, context), shape)


comptime _NAN_LABEL = "mojolearn: y contains a NaN label; NaN is not a class"


def _svc_check_label_kinds(labels: PythonObject, context: PythonObject) raises:
    """`_labels._check_label_kinds`: the first label that is neither a number
    nor a str (or a NaN before it), then a mix of the two kinds."""
    var b = svc_builtins()
    var operator = Python.import_module("operator")
    var types = family_call(b.list, family_call(b.set, family_call(b.map, b.type, labels)))
    var refused = Python.list()
    var has_str = False
    var has_number = False
    for i in range(len(types)):  # driver-loop: over the distinct label types
        var t = types[i]
        if Bool(py=family_call(b.issubclass, t, b.str)):
            has_str = True
        elif Bool(py=family_call(b.issubclass, t, context["family_svc_real"])) or String(t.__name__) == "bool_":
            has_number = True
        else:
            _ = py_call(refused.append, t)
    var stop = -1
    var head = labels
    if len(refused) > 0:
        var refused_set = family_call(b.frozenset, refused)
        var flags = family_call(b.list, family_call(b.map, py_attr(refused_set, "__contains__"), family_call(b.map, b.type, labels)))
        stop = Int(py=family_call(flags.index, PythonObject(True)))
        head = family_call(operator.getitem, labels, family_call(b.slice, PythonObject(stop)))
    if Bool(py=family_call(b.any, family_call(b.map, operator.ne, head, head))):
        raise Error(_NAN_LABEL)
    if stop >= 0:
        raise Error("mojolearn: label " + family_repr(labels[stop]) + " is neither a number nor a str")
    if has_str and has_number:
        raise Error("mojolearn: y mixes numeric and str labels; encode one kind before fitting")


def svc_sorted_classes(labels: PythonObject, context: PythonObject) raises -> PythonObject:
    """`encode_labels` of a flat label list: (classes list, int32 codes) under
    the ORDER RULE (first-seen representative of equal labels, sorted by
    Python's comparison; a NaN label refused)."""
    var b = svc_builtins()
    var operator = Python.import_module("operator")
    var n = len(labels)
    if n == 0:
        raise Error("mojolearn: y is empty")
    var kind = svc_plain_labels(labels, context)
    if kind == "number":
        if Bool(py=family_call(b.any, family_call(b.map, operator.ne, labels, labels))):
            raise Error(_NAN_LABEL)
    elif kind.byte_length() == 0:
        _svc_check_label_kinds(labels, context)
    var classes = family_call(b.sorted, family_call(b.dict.fromkeys, labels))
    var index = family_call(b.dict, family_call(b.zip, classes, family_call(b.range, PythonObject(len(classes)))))
    var store = family_call(context["array_type"], PythonObject("i"), family_call(b.map, py_attr(index, "__getitem__"), labels))
    var codes = classical_wrap_array(store, Python.tuple(PythonObject(n)), "<i4", "C", context)
    return Python.tuple(classes, codes)


def svc_label_kind(classes: PythonObject, context: PythonObject) raises -> Int:
    """`_labels.label_kind`: 0 'int' (every class Integral, no bool), 1
    'float' (every class Real, no bool), 2 'object'."""
    var integral = True
    var real = True
    for i in range(len(classes)):  # driver-loop: over the fitted classes
        var c = classes[i]
        var boolean = family_isinstance(c, svc_builtins().bool) or String(py_type(c).__name__) == "bool_"
        if boolean or not family_isinstance(c, context["family_svc_integral"]):
            integral = False
        if boolean or not family_isinstance(c, context["family_svc_real"]):
            real = False
    if integral:
        return 0
    if real:
        return 1
    return 2


def svc_decode(classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.decode_labels(classes, codes)` for an int64 code Array this
    owner made: an int64 Array for int classes, a float64 Array for real
    classes (the base binding's `gather_i64` / `gather_f64`), the list of
    label objects otherwise."""
    var b = svc_builtins()
    var kind = svc_label_kind(classes, context)
    var n = Int(py=codes.size)
    var k = len(classes)
    if kind < 2 and k > 0:
        var table = family_array(Python.tuple(PythonObject(k)), "<i8" if kind == 0 else "<f8", context)
        var packed = True
        for i in range(k):  # driver-loop: over the fitted classes
            if not packed:
                continue
            try:
                if kind == 0:
                    var value = py_call(b.int, classes[i])
                    if Bool(py=value < PythonObject(-9223372036854775807 - 1)) or Bool(py=value > PythonObject(9223372036854775807)):
                        packed = False
                    else:
                        FAMILY_I64P(unsafe_from_address=Int(py=classical_address(table)))[i] = Int64(Int(py=value))
                        _ = table  # keep the Python object alive through the native call above
                else:
                    FAMILY_F64P(unsafe_from_address=Int(py=classical_address(table)))[i] = Float64(py=py_call(b.float, classes[i]))
                    _ = table  # keep the Python object alive through the native call above
            except:
                packed = False
        if packed:
            var out = family_array(Python.tuple(PythonObject(n)), "<i8" if kind == 0 else "<f8", context)
            if n > 0:
                _ = family_call(svc_base("gather_i64" if kind == 0 else "gather_f64", context), classical_address(table),
                                PythonObject(k), classical_address(codes), PythonObject(n), classical_address(out))
                _ = table  # keep the Python object alive through the native call above
                _ = out  # keep the Python object alive through the native call above
            _ = table^
            return out
    # the label output step: the class objects, one per code (C-level map
    # over the int64 store)
    var values = family_call(b.list, family_call(b.map, py_attr(classes, "__getitem__"), codes._store))
    if kind == 2:
        return values
    try:
        var store = py_call(context["array_type"], PythonObject("q" if kind == 0 else "d"),
                            py_call(b.map, b.int if kind == 0 else b.float, values))
        return classical_wrap_array(store, Python.tuple(PythonObject(n)), "<i8" if kind == 0 else "<f8", "C", context)
    except:
        return values


def svc_threshold_codes(scores: PythonObject, threshold: Float64, strict: Bool, below: Int, above: Int,
                        context: PythonObject) raises -> PythonObject:
    """`_labels.threshold_codes` of a C-order float32 Array: int64 codes."""
    var n = Int(py=scores.size)
    var out = family_array(Python.tuple(PythonObject(n)), "<i8", context)
    if n > 0:
        _ = family_call(svc_base("threshold_labels_i64", context), classical_address(scores), PythonObject(0),
                        PythonObject(n), PythonObject(threshold), PythonObject(1 if strict else 0),
                        PythonObject(below), PythonObject(above), classical_address(out))
        _ = out  # keep the Python object alive through the native call above
    return out


def svc_argmax_rows(scores: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.argmax_rows` of a C-order (n, c) float32 or float64 Array."""
    var n = Int(py=scores.shape[0])
    var c = Int(py=scores.shape[1])
    var out = family_array(Python.tuple(PythonObject(n)), "<i8", context)
    if n == 0:
        return out
    if c == 0:
        raise Error("argmax_rows: scores have no columns")
    var name = "argmax_rows_f32" if String(scores.dtype) == "<f4" else "argmax_rows_f64"
    _ = family_call(svc_base(name, context), classical_address(scores), PythonObject(n), PythonObject(c),
                    classical_address(out))
    _ = out  # keep the Python object alive through the native call above
    return out


# ---------------------------------------------------------------- score metrics


def svc_r2_sums(pred: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`linear_model._r2_sums(pred, y)`: (SS_res, SS_tot) from the one
    x_metrics program `_r2_sums_of` ran: the y mean (Float32 column), then
    both sums of squares, binary64 out."""
    var t = family_as_f32(y, 1, "y", context)
    var p = family_as_f32(pred, 1, "predictions", context)
    var n = Int(py=t.size)
    var m = Int(py=p.size)
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(t))), n):
        raise Error("mojolearn score: y_true contains NaN or infinity")
    _ = t  # keep the Python object alive through the native calls above
    if not family_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(p))), m):
        raise Error("mojolearn score: y_pred contains NaN or infinity")
    _ = p  # keep the Python object alive through the native calls above
    if n != m:
        raise Error("mojolearn score: y_true and y_pred have different shapes (" + String(n) + ", 1) and ("
                    + String(m) + ", 1)")
    var r = _Reg(t, p, n, 1, PythonObject(None), "uniform_average", PythonObject(None), context["family_metrics_ops"])
    var sq = Int(py=context["family_metrics_terms"]["sq"])
    r.program(0.0, context)
    var y_off = r.y_off
    var p_off = r.p_off
    var y_sums = r.sums(y_off)
    var y_mean = r.fin32(y_sums, _RE_DIV)
    var res_terms = r.term(sq, y_off, p_off, False)
    var res_sums = r.sums(res_terms)
    var num = r.fin(res_sums, 0)
    var tot_terms = r.term(sq, y_off, y_mean, True)
    var tot_sums = r.sums(tot_terms)
    var den = r.fin(tot_sums, 0)
    _ = r.prog.want(num, 2)
    _ = r.prog.want(den, 2)
    var arena = _reg_run(r.prog, PythonObject(None), context)
    var ss_res = _arena_double(arena, num)
    var ss_tot = _arena_double(arena, den)
    _ = t^
    _ = p^
    return Python.tuple(PythonObject(ss_res), PythonObject(ss_tot))


def _svc_classification_labels(values: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_metrics_impl._classification_labels`: (labels list, kind)."""
    var b = svc_builtins()
    if len(svc_shape_of(values)) != 1:
        raise Error(name + " must be one-dimensional single-label data")
    var labels = svc_flatten_labels(values, context)
    if len(labels) == 0:
        raise Error(name + " must contain at least one label")
    # `all(isinstance(v, str))` and `all(isinstance(v, Integral) or
    # is_bool(v))`, asked once per distinct label type.
    var types = family_call(b.list, family_call(b.set, family_call(b.map, b.type, labels)))
    var strings = True
    var integers = True
    for i in range(len(types)):  # driver-loop: over the distinct label types
        var t = types[i]
        if not Bool(py=family_call(b.issubclass, t, b.str)):
            strings = False
        if not (Bool(py=family_call(b.issubclass, t, context["family_svc_integral"])) or String(t.__name__) == "bool_"):
            integers = False
    if strings:
        return Python.tuple(family_call(b.list, family_call(b.map, b.str, labels)), PythonObject("string"))
    if integers:
        return Python.tuple(family_call(b.list, family_call(b.map, b.int, labels)), PythonObject("integer"))
    raise Error("TypeError: " + name + " must contain only strings or only integers; "
                + "floating labels, missing labels and mixed types are unsupported")


def svc_accuracy(y_true: PythonObject, y_pred: PythonObject, n_pred: Int, context: PythonObject) raises -> Float64:
    """`linear_model._accuracy_host(pred, y)`: the shape refusal, then
    `_expansion_metrics.accuracy_fraction(y, pred)`, the exact matched count
    over the row count from the x_metrics COUNT program (`_Sums` with the
    `_cm_stage` COUNT tail, unweighted)."""
    var b = svc_builtins()
    var shape = svc_shape_of(y_true)
    if len(shape) != 1 or Int(py=shape[0]) != n_pred:
        raise Error("mojolearn: y must be 1-D with one entry per row of X")
    var caller = String("accuracy_score")
    var tl = _svc_classification_labels(y_true, "y_true", context)
    var pl = _svc_classification_labels(y_pred, "y_pred", context)
    var true = tl[0]
    var pred = pl[0]
    var n = len(true)
    if n != len(pred):
        raise Error("mojolearn " + caller + ": y_true and y_pred lengths differ")
    if String(tl[1]) != String(pl[1]):
        raise Error("TypeError: mojolearn " + caller + ": y_true and y_pred must use the same label type")
    if n > 2147483647:
        raise Error("mojolearn " + caller + ": at most INT32_MAX rows")
    var present = family_call(b.sorted, family_call(family_call(b.set, true).union, pred))
    var big_l = len(present)
    var index = family_call(b.dict, family_call(b.zip, present, family_call(b.range, PythonObject(big_l))))
    var yt = classical_wrap_array(family_call(context["array_type"], PythonObject("i"), family_call(b.map, py_attr(index, "__getitem__"), true)),
                                  Python.tuple(PythonObject(n)), "<i4", "C", context)
    var yp = classical_wrap_array(family_call(context["array_type"], PythonObject("i"), family_call(b.map, py_attr(index, "__getitem__"), pred)),
                                  Python.tuple(PythonObject(n)), "<i4", "C", context)
    var prog = _Prog(context["family_metrics_ops"])
    var a = prog.put_i32(yt)
    var bo = prog.put_i32(yp)
    var matched = prog.scratch(n)
    prog.stage("pair_key", n, [a, bo, matched, big_l, 1])
    # `_group` for the match, true and pred keys (no values, no weights).
    var offs = List[Int]()
    var keys: List[Int] = [matched, a, bo]
    for g in range(3):  # small-loop(groupings: match, true, pred)
        var off = prog.alloc(big_l + 1)
        var order = prog.scratch(n)
        var out = prog.alloc(big_l)
        prog.stage("group_sort", 1, [keys[g], n, big_l, off, order])
        if big_l > 0:
            prog.stage("group_sum", big_l, [off, order, SVC_NONE, 1, SVC_NONE, out, 1])
        offs.append(off)
    # `_counts` of the three groupings.
    for g in range(3):  # small-loop(groupings: match, true, pred)
        var cnt = prog.alloc(big_l)
        if big_l > 0:
            prog.stage("off_diff", big_l, [offs[g], big_l, cnt])
    # `_cm_stage(..., _CM_COUNT, 11)`: the tail over the match OFF table.
    var tail = prog.alloc(11)
    prog.stage("cm_epi", 1, [_CM_COUNT, offs[0], big_l, 0, n, tail])
    _ = prog.want(tail, 11)
    var arena = _reg_run(prog, PythonObject(None), context)
    _ = yt^
    _ = yp^
    return _arena_double(arena, tail + 8)
