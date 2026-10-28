# SPDX-License-Identifier: Apache-2.0
# The classical estimators' shared native label, weight and score steps:
# `_labels.encode_labels` / `decode_labels` / `argmax_rows`,
# `_sample_weight.check_sample_weight`, `_class_weights._class_weight_rows`,
# `_buffer.shape_of` / `_as_typed`, and `_expansion_metrics.accuracy_fraction`
# (scikit-learn's ClassifierMixin.score), with their refusals and wording and
# the same native helpers in the same order (the base binding's casts,
# `prepare_targets`, gathers and argmax; the x_trees weight kernels; the
# x_metrics COUNT program run by the shared native program runner), so every
# byte a kernel reads is unchanged.
#
# Derived from the tree owner's input and score steps (lane migrate-trees,
# bindings/trees_public_inputs.mojo and trees_public_score.mojo) for the
# linear, discriminant and naive Bayes classifiers; fold the two copies into
# one module once both lanes are on main.
#
# No Python function runs here: inputs are read through the buffer protocol
# and the Array slots, outputs are mojolearn Arrays assembled from their
# slots (classical_wrap_array), and the only callables invoked are compiled
# entry points and C builtins connected at import
# (python/mojolearn/_native_labels_bridge.py `label_context`).
from bindings.native_array_storage import native_array_from_bytes, native_array_from_values, native_array_zeros
from bindings.native_loaded_module import native_loaded_module
from std.math import isfinite
from std.memory import bitcast, memcpy
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_setattr, py_fields
from bindings.py_attr import py_attr
from std.python._cpython import PyObjectPtr
from bindings.py_typed_call import py_call_typed, typed_python_error
from bindings.native_buffer_owner import native_buffer_view
from bindings.classical_arrays import classical_wrap_array, _store_code, _buffer_typestr
from bindings.classical_pandas import native_pandas_exact, native_pandas_f32, _pd_fields, _pd_get, classical_admit_value
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.sequential_inputs import sequential_sequence
from bindings.arena_prog_native import _metrics_run
from bindings.classical_attributes import classical_connect_numpy

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]



# ------------------------------------------------------------ connection


def labels_builtins() raises -> PythonObject:
    return native_loaded_module("builtins")


def labels_type_name(t: PythonObject) raises -> String:
    """A type's name through type's own C descriptor: no metaclass
    attribute hook runs (error messages name refused label types)."""
    var b = labels_builtins()
    return String(py_call(b.type.__dict__["__name__"].__get__, t))


def _labels_one_of(t: PythonObject, types: PythonObject) raises -> Bool:
    for i in range(len(types)):  # driver-loop: over a fixed tuple of connected exact scalar types
        if t is types[i]:
            return True
    return False


def _labels_connect(context: PythonObject) raises:
    """Connect NumPy's exact scalar types once NumPy is loaded, so label
    admission recognizes them by identity (never numbers ABC callbacks)."""
    if (py_get(context, "ndarray") is None and py_contains(context, "modules")
            and py_contains(context, "native_module_type")):
        classical_connect_numpy(context)


def labels_bool_type(t: PythonObject, context: PythonObject) raises -> Bool:
    """Exactly `bool` or NumPy's `bool_`."""
    var b = labels_builtins()
    if t is b.bool:
        return True
    var bools = py_get(context, "bool_types")
    return bools is not None and _labels_one_of(t, bools)


def labels_real_type(t: PythonObject, context: PythonObject) raises -> Bool:
    """Exactly int, float, bool or a connected NumPy numeric/bool scalar
    type: the former numbers.Real test (plus NumPy bool_) by identity."""
    var b = labels_builtins()
    if t is b.int or t is b.float or t is b.bool:
        return True
    _labels_connect(context)
    var scalars = py_get(context, "scalar_types")
    return scalars is not None and _labels_one_of(t, scalars)


def labels_integral_type(t: PythonObject, context: PythonObject) raises -> Bool:
    """Exactly int or a connected NumPy integer scalar type (bool excluded):
    the former numbers.Integral test by identity."""
    var b = labels_builtins()
    if t is b.int:
        return True
    _labels_connect(context)
    var integers = py_get(context, "integer_types")
    return integers is not None and _labels_one_of(t, integers)


def labels_text_type(t: PythonObject, context: PythonObject) raises -> Bool:
    """Exactly str or NumPy's str_ (a connected C value type)."""
    var b = labels_builtins()
    if t is b.str:
        return True
    _labels_connect(context)
    var values = py_get(context, "native_value_types")
    # issubclass runs only on a connected C type, never on a user class.
    return values is not None and _labels_one_of(t, values) and Bool(py=py_call(b.issubclass, t, b.str))


def labels_table(context: PythonObject, mode: String, name: String) raises -> PythonObject:
    """The connected entry table of binding `name` in `mode`, or the
    refusal recorded at import ("<Class>: <message>")."""
    var tables = py_get(context["labels_bindings"], PythonObject(mode))
    if tables is None:
        raise Error("numeric_mode must be fast, identical or None")
    var table = py_get(tables, PythonObject(name))
    if table is None:
        raise Error("ImportError: mojolearn: " + name + " is not connected to the tree owner")
    if py_type(table) is py_type(PythonObject("")):
        raise Error(String(table))
    return table


def labels_entry(table: PythonObject, name: String) raises -> PythonObject:
    """A connected compiled entry point, or None when the binary has none."""
    return py_get(table, PythonObject(name))


def labels_need(table: PythonObject, name: String, message: String, kind: String = "RuntimeError") raises -> PythonObject:
    """A connected entry point the operation requires, refused with the
    Python wording when the binary predates it."""
    var entry = py_get(table, PythonObject(name))
    if entry is None:
        raise Error(kind + ": " + message)
    return entry


def labels_helper(context: PythonObject, name: String) raises -> PythonObject:
    """A base binding helper (`_buffer._native`), or the recorded refusal."""
    var entry = py_get(context["labels_helpers"], PythonObject(name))
    if entry is None or py_type(entry) is py_type(PythonObject("")):
        if entry is None:
            raise Error("ImportError: mojolearn: the base binding has no `" + name + "`")
        raise Error(String(entry))
    return entry


def labels_call(entry: PythonObject, args: PythonObject) raises -> PythonObject:
    """`entry(*args)` for a compiled entry point and a Python list of its
    arguments, keeping the class of a refusal the entry raises."""
    ref cpy = Python().cpython()
    var envelope = py_call(labels_builtins().tuple, args)
    var result = cpy.lib.get_function[PyObjectPtr]("PyObject_CallObject")(entry._obj_ptr, envelope._obj_ptr)
    if not result:
        raise typed_python_error()
    return PythonObject(from_owned=result)


def labels_env(name: String, default: String) -> String:
    var value = getenv(name)
    return value if value else default


# --------------------------------------------------------------- outputs


def labels_itemsize(typestr: String) raises -> Int:
    if typestr == "<f8" or typestr == "<i8":
        return 8
    if typestr == "<f4" or typestr == "<i4" or typestr == "<u4":
        return 4
    if typestr == "<f2" or typestr == "<u2":
        return 2
    if typestr == "<u1" or typestr == "<i1":
        return 1
    raise Error("TypeError: " + typestr + " is not an Array dtype")


def labels_empty(shape: PythonObject, typestr: String, context: PythonObject, order: String = "C") raises -> PythonObject:
    """A new zero-filled mojolearn Array (`_buffer.empty`), assembled from
    its slots; never an ndarray, whatever is installed."""
    var count = 1
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        count *= Int(py=shape[k])
    var width = labels_itemsize(typestr)
    var store = native_array_zeros(String(_store_code(typestr)), count)
    return classical_wrap_array(store, shape, typestr, order, context)


def labels_vector(count: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    return labels_empty(Python.tuple(count), typestr, context)


def labels_full_f32(count: Int, value: Float64, context: PythonObject) raises -> PythonObject:
    """`_buffer.full((count,), value, '<f4')`: one float32 conversion of
    `value`, the store repeated in C (`array.array.__mul__`)."""
    var one = native_array_from_values(String("f"), Python.list(PythonObject(value)))
    var store = py_call(py_attr(one, "__mul__"), PythonObject(count))
    return classical_wrap_array(store, Python.tuple(count), "<f4", "C", context)


def labels_reshape(array: PythonObject, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`array.reshape(shape)` of a fresh C-order Array: the same store under
    a new shape (the Array the Python reshape returned)."""
    return classical_wrap_array(array._store, shape, String(array.dtype), "C", context)


def labels_freeze(array: PythonObject) raises:
    """A fitted model array the caller cannot write through (the copy
    `_mojolearn_after_fit` took): the fresh export's own storage, marked
    read-only."""
    py_setattr(array, "_readonly", PythonObject(True))


def labels_address(array: PythonObject) raises -> Int:
    return Int(py=array._addr)


# ----------------------------------------------------------------- inputs


struct LabelsView(Copyable, Movable):
    """One admitted input read in place: its owner (kept alive across the
    kernel), data address, shape, byte strides and Array dtype."""
    var keep: PythonObject
    var addr: Int
    var shape: List[Int]
    var strides: List[Int]
    var typestr: String
    var item: Int

    def __init__(out self, keep: PythonObject, addr: Int, var shape: List[Int], var strides: List[Int],
                 typestr: String, item: Int):
        self.keep = keep
        self.addr = addr
        self.shape = shape^
        self.strides = strides^
        self.typestr = typestr
        self.item = item

    def ndim(self) -> Int:
        return len(self.shape)

    def size(self) -> Int:
        var n = 1
        for k in range(len(self.shape)):  # driver-loop: over array rank metadata
            n *= self.shape[k]
        return n

    def c_order(self) -> Bool:
        var step = self.item
        for k in range(len(self.shape)):  # driver-loop: over array rank metadata
            var axis = len(self.shape) - 1 - k
            if self.shape[axis] != 1 and self.strides[axis] != step:
                return False
            step *= self.shape[axis]
        return True

    def f_order(self) -> Bool:
        var step = self.item
        for axis in range(len(self.shape)):  # driver-loop: over array rank metadata
            if self.shape[axis] != 1 and self.strides[axis] != step:
                return False
            step *= self.shape[axis]
        return True

    def both_orders(self) -> Bool:
        """`Array._both_orders`: at most one dimension larger than 1."""
        var big = 0
        for k in range(len(self.shape)):  # driver-loop: over array rank metadata
            if self.shape[k] != 1:
                big += 1
        return big <= 1


def labels_shape_repr(view: LabelsView) raises -> String:
    """The Python repr of the shape tuple, as `_as_typed` printed it."""
    var b = labels_builtins()
    var items = Python.list()
    for k in range(len(view.shape)):  # driver-loop: over array rank metadata
        _ = py_call(items.append, view.shape[k])
    return String(py_call(b.repr, py_call(b.tuple, items)))


def _view_of_array(value: PythonObject) raises -> LabelsView:
    var shape = List[Int]()
    var strides = List[Int]()
    var rank = Int(py=value.ndim)
    for k in range(rank):  # driver-loop: over array rank metadata
        shape.append(Int(py=value.shape[k]))
        strides.append(Int(py=value.strides[k]))
    var typestr = String(value.dtype)
    return LabelsView(value, Int(py=value._addr), shape^, strides^, typestr, Int(py=value.itemsize))


def _any_contiguous_address(value: PythonObject) raises -> Int:
    """The first element's address of a C- or F-contiguous export
    (PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT); the export is released at once,
    the caller keeps the owner alive."""
    ref cpy = Python().cpython()
    var storage = List[Int](length=10, fill=0)
    var pointer = storage.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    if cpy.lib.get_function[Int32]("PyObject_GetBuffer")(value._obj_ptr, pointer, Int32(0x9C)) != 0:
        raise cpy.unsafe_get_error()
    var address = storage[0]
    cpy.lib.get_function[NoneType]("PyBuffer_Release")(pointer)
    _ = storage^
    return address


def _view_of_buffer(value: PythonObject, name: String, context: PythonObject) raises -> LabelsView:
    var view = native_buffer_view(value)
    var typestr = _buffer_typestr(view, name)
    var item = Int(py=view.itemsize)
    var rank = Int(py=view.ndim)
    var shape = List[Int]()
    var strides = List[Int]()
    for k in range(rank):  # driver-loop: over array rank metadata
        shape.append(Int(py=view.shape[k]))
        strides.append(Int(py=view.strides[k]))
    if rank == 0:
        shape.append(1)
        strides.append(item)
    if not Bool(py=view.c_contiguous) and not Bool(py=view.f_contiguous):
        # `_materialize`: a strided export is linearized in C order once
        # (`memoryview.tobytes()`, in C) into an Array of its dtype.
        var store = native_array_from_bytes(String(_store_code(typestr)), py_call(view.tobytes, "C"))
        var pyshape = Python.list()
        for k in range(len(shape)):  # driver-loop: over array rank metadata
            _ = py_call(pyshape.append, shape[k])
        return _view_of_array(classical_wrap_array(store, py_call(labels_builtins().tuple, pyshape), typestr, "C", context))
    var _keep_ret_view = LabelsView(view, _any_contiguous_address(view), shape^, strides^, typestr, item)
    _ = view  # keep the Python object alive through the native call above
    return _keep_ret_view^


def _pandas_values(source: PythonObject) raises -> PythonObject:
    """The storage array of an exact single-block pandas Series (what its
    `tolist()` lists), read through pandas' native slots."""
    var manager = py_get(_pd_fields(source), "_mgr")
    var blocks = py_get(_pd_fields(manager), "blocks")
    if py_type(blocks) is not labels_builtins().tuple or len(blocks) != 1:
        raise Error("TypeError: pandas Series requires native single-block storage")
    return _pd_get("pandas._libs.internals", "Block", "values", blocks[0])


def labels_view(source: PythonObject, name: String, context: PythonObject) raises -> LabelsView:
    """`_buffer._materialize(source, name)` read in place: an Array, any
    buffer exporter (an ndarray, an `array.array`, a memoryview), a pandas
    frame or series, a torch tensor, or a nested list or tuple of numbers.
    The refusals keep `_materialize`'s classes and wording."""
    var b = labels_builtins()
    var cls = py_type(source)
    if cls is context["array_class"]:
        return _view_of_array(source)
    if cls is b.str or cls is b.bytes:
        raise Error("TypeError: mojolearn: " + name + " is a " + labels_type_name(cls) + ", not an array")
    if native_pandas_exact(source):
        return _view_of_buffer(native_pandas_f32(source, context, False, True), name, context)
    if _ordinary_torch(cls, context):
        return _view_of_buffer(native_torch_numpy(source), name, context)
    if cls is b.list or cls is b.tuple:
        var packed = sequential_sequence(source, context)
        var kind = String(packed[1])
        var typestr = String("<f8")
        if kind == "i" or kind == "b":
            typestr = "<i8"
        elif kind != "f":
            raise Error("TypeError: mojolearn: " + name + " holds a str, not a number")
        var store = packed[3]
        if kind == "b":
            # A bool leaf is the int 0 or 1 (`_materialize` lists it as int64).
            store = native_array_from_values(String("q"), store)
        var array = classical_wrap_array(store, packed[0], typestr, "C", context)
        return _view_of_array(array)
    try:
        return _view_of_buffer(source, name, context)
    except error:
        if String(error).startswith("TypeError: " + name):
            raise error^
        raise Error("TypeError: mojolearn: " + name + " holds a " + labels_type_name(cls) + ", not a number")


def labels_check_rank(view: LabelsView, ndim: Int, name: String) raises:
    """`_as_typed`'s rank and emptiness refusals."""
    if ndim >= 0 and view.ndim() != ndim:
        var hint = String("")
        if ndim == 2 and view.ndim() < 2:
            hint = (". Reshape your data either using array.reshape(-1, 1) if your data "
                    + "has a single feature or array.reshape(1, -1) if it contains a single sample.")
        raise Error("mojolearn: " + name + " must be " + String(ndim) + "-D, got " + String(view.ndim())
                    + "-D shape " + labels_shape_repr(view) + hint)
    if view.size() == 0:
        var what = String("feature(s)") if view.ndim() == 2 and view.shape[0] != 0 else String("sample(s)")
        var shown = labels_shape_repr(view)
        raise Error("mojolearn: " + name + " is empty, shape " + shown + ": Found array with 0 " + what
                    + " (shape=" + shown + ") while a minimum of 1 is required.")


def _native_code(typestr: String) raises -> Int:
    """`_array._NATIVE_CODE`: the core helpers' dtype numbering."""
    if typestr == "<f4":
        return 0
    if typestr == "<f8":
        return 1
    if typestr == "<i4":
        return 2
    if typestr == "<i8":
        return 3
    if typestr == "<u4":
        return 4
    if typestr == "<u1":
        return 5
    if typestr == "<u2":
        return 6
    if typestr == "<i1":
        return 7
    if typestr == "<f2":
        return 8
    raise Error("TypeError: " + typestr + " has no native element code")


def labels_cast(src: Int, source: String, dst: Int, target: String, count: Int, context: PythonObject) raises:
    """`Array.astype(target)` of a contiguous block (`_native_astype`): the
    core helper `cast_elements`, one conversion per element in storage
    order."""
    if count == 0:
        return
    var refused = Int(py=py_call_typed(labels_helper(context, "cast_elements"), PythonObject(src),
                                      PythonObject(_native_code(source)), PythonObject(dst),
                                      PythonObject(_native_code(target)), PythonObject(count)))
    if refused != 0:
        raise Error("TypeError: mojolearn: an element does not fit " + target)


struct LabelsMatrix(Copyable, Movable):
    """A float32 X as a kernel reads it: the owner of its storage, its
    address, its shape and whether it is row-major (C order)."""
    var keep: PythonObject
    var addr: Int
    var rows: Int
    var cols: Int
    var row_major: Bool

    def __init__(out self, keep: PythonObject, addr: Int, rows: Int, cols: Int, row_major: Bool):
        self.keep = keep
        self.addr = addr
        self.rows = rows
        self.cols = cols
        self.row_major = row_major


def labels_f32(source: PythonObject, ndim: Int, order: String, name: String, context: PythonObject) raises -> LabelsMatrix:
    """`_buffer._as_typed(source, '<f4', order, ndim, name)`: the input
    itself when it is already float32 in `order` (no copy), otherwise one
    converted copy through the base binding's casts (a float64 or float32
    source) or the classical element cast (integers and float16)."""
    var view = labels_view(source, name, context)
    labels_check_rank(view, ndim, name)
    return labels_f32_view(view^, order, context)


def labels_f32_view(var view: LabelsView, order: String, context: PythonObject) raises -> LabelsMatrix:
    var rows = view.shape[0]
    var cols = view.shape[1] if view.ndim() == 2 else 1
    var two = view.ndim() == 2
    var want_f = order == "F"
    var has_order = view.both_orders() or (view.f_order() if want_f else view.c_order())
    if view.typestr == "<f4" and has_order:
        var _keep_ret_view = LabelsMatrix(view.keep, view.addr, rows, cols, not want_f)
        _ = view  # keep the Python object alive through the native call above
        return _keep_ret_view^
    var shape = Python.tuple(rows, cols) if two else Python.tuple(rows)
    var out = labels_empty(shape, "<f4", context, "F" if want_f and two else "C")
    var dst = labels_address(out)
    _ = out  # keep the Python object alive through the native call above
    var contiguous_c = view.c_order()
    var contiguous_f = view.f_order()
    if (view.typestr == "<f8" or view.typestr == "<f4") and (contiguous_c or contiguous_f):
        if two and not view.both_orders() and contiguous_f != want_f:
            # `_native_to_f32`: the one fused cast-and-transpose (or the
            # float32 transpose) over the source's [rows, cols] layout.
            var r = rows
            var c = cols
            if contiguous_f:
                r = cols
                c = rows
            var fn_obj = labels_helper(context, "cast_colmajor_f64_to_f32" if view.typestr == "<f8" else "transpose_f32")
            _ = py_call_typed(fn_obj, PythonObject(view.addr), PythonObject(dst), PythonObject(r), PythonObject(c))
            _ = out  # keep the Python object alive through the native call above
            _ = view  # keep the Python object alive through the native call above
            return LabelsMatrix(out, dst, rows, cols, not want_f)
        if view.typestr == "<f8":
            # The flat cast keeps the source's storage order, which is the
            # requested one here (or the two coincide).
            _ = py_call_typed(labels_helper(context, "cast_f64_to_f32"), PythonObject(view.addr), PythonObject(dst),
                              PythonObject(view.size()))
            _ = out  # keep the Python object alive through the native call above
            _ = view  # keep the Python object alive through the native call above
            return LabelsMatrix(out, dst, rows, cols, not want_f)
    # `astype('<f4')` keeps the source's storage order; `_as_order` then
    # transposes when the requested order differs (the float32 transpose).
    if not two or view.both_orders() or contiguous_f == want_f:
        labels_cast(view.addr, view.typestr, dst, "<f4", view.size(), context)
        _ = out  # keep the Python object alive through the native call above
        _ = view  # keep the Python object alive through the native call above
        return LabelsMatrix(out, dst, rows, cols, not want_f)
    var staged = labels_vector(view.size(), "<f4", context)
    labels_cast(view.addr, view.typestr, labels_address(staged), "<f4", view.size(), context)
    _ = staged  # keep the Python object alive through the native call above
    _ = view  # keep the Python object alive through the native call above
    var r = cols if contiguous_f else rows
    var c = rows if contiguous_f else cols
    _ = py_call_typed(labels_helper(context, "transpose_f32"), PythonObject(labels_address(staged)), PythonObject(dst),
                      PythonObject(r), PythonObject(c))
    _ = staged  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = staged^
    return LabelsMatrix(out, dst, rows, cols, not want_f)


def labels_all_finite(addr: Int, count: Int, wide: Bool, context: PythonObject) raises -> Bool:
    """`_buffer.all_finite`: the base binding's scan."""
    if count == 0:
        return True
    var fn_obj = labels_helper(context, "all_finite_f64" if wide else "all_finite_f32")
    return Int(py=py_call_typed(fn_obj, PythonObject(addr), PythonObject(count))) == 1


# ---------------------------------------------------------------- weights


def labels_check_sample_weight(source: PythonObject, n: Int, context: PythonObject) raises -> LabelsMatrix:
    """`_sample_weight.check_sample_weight(source, n)` (float32): a Python
    or 0-d scalar fills a vector; a vector must hold n entries; the
    finiteness, sign and all-zero refusals are `prepare_targets`' with
    scikit-learn's wording. The vector may borrow the caller's float32
    buffer (the fit only reads it)."""
    var b = labels_builtins()
    if source is None:
        var ones = labels_full_f32(n, 1.0, context)
        var _keep_ret_ones = LabelsMatrix(ones, labels_address(ones), n, 1, True)
        _ = ones  # keep the Python object alive through the native call above
        return _keep_ret_ones^
    var cls = py_type(source)
    var scalar = False
    var value = Float64(0)
    if cls is b.int or cls is b.float or (labels_real_type(cls, context) and not labels_bool_type(cls, context)
                                          and not labels_integral_type(cls, context)):
        scalar = True
        value = Float64(py=py_call(b.float, source))
    elif cls is not context["array_class"] and cls is not b.list and cls is not b.tuple:
        var zero_d = False
        try:
            zero_d = Int(py=native_buffer_view(source).ndim) == 0
        except:
            zero_d = False
        if zero_d:
            scalar = True
            value = Float64(py=py_call(b.float, source))
    if not scalar:
        var view = labels_view(source, "sample_weight", context)
        if view.size() == 0:
            if n != 0:
                raise Error("sample_weight.shape == (0,), expected (" + String(n) + ",)!")
            labels_check_rank(view, -1, "sample_weight")
        if view.ndim() != 1:
            raise Error("Sample weights must be 1D array or scalar")
        if view.shape[0] != n:
            raise Error("sample_weight.shape == " + labels_shape_repr(view) + ", expected (" + String(n) + ",)!")
        var w = labels_f32_view(view^, "C", context)
        if n:
            labels_prepare_weights(w.addr, n, context)
            _ = w  # keep the Python object alive through the native call above
        return w^
    var out = labels_full_f32(n, value, context)
    if n:
        labels_prepare_weights(labels_address(out), n, context)
        _ = out  # keep the Python object alive through the native call above
    var _keep_ret_out = LabelsMatrix(out, labels_address(out), n, 1, True)
    _ = out  # keep the Python object alive through the native call above
    return _keep_ret_out^


def labels_prepare_weights(addr: Int, n: Int, context: PythonObject) raises:
    """`prepare_targets(None, w, n)` on float32 weights: the weight scan,
    refused with scikit-learn's wording."""
    var status = Int(py=py_call_typed(labels_helper(context, "prepare_targets"), PythonObject(0), PythonObject(0),
                                     PythonObject(n), PythonObject(0), PythonObject(0), PythonObject(addr),
                                     PythonObject(0), PythonObject(0)))
    if status == -3:
        raise Error("Input sample_weight contains NaN or infinity.")
    if status == -4:
        raise Error("Negative values in data passed to `sample_weight`")
    if status == -5:
        raise Error("Sample weights must contain at least one non-zero number.")


# ----------------------------------------------------------------- labels


def _encode_code(typestr: String) -> Int:
    """`_labels._NATIVE_ENCODE`, or -1."""
    if typestr == "<f4":
        return 0
    if typestr == "<f8":
        return 1
    if typestr == "<i4":
        return 2
    if typestr == "<i8":
        return 3
    if typestr == "<u4":
        return 4
    if typestr == "<u1":
        return 5
    return -1


struct ClassicalLabels(Copyable, Movable):
    """`(classes, codes)`: the class list (Python label objects, the
    `classes_` attribute) and one int32 code per row (an Array)."""
    var classes: PythonObject
    var codes: PythonObject
    var n: Int

    def __init__(out self, classes: PythonObject, codes: PythonObject, n: Int):
        self.classes = classes
        self.codes = codes
        self.n = n


def _native_labels(view: LabelsView, bool_source: Bool, context: PythonObject) raises -> ClassicalLabels:
    """`_labels._encode_labels_native`: one `prepare_targets` call over a
    numeric label vector, the k class values as Python numbers."""
    var n = view.size()
    var wide_float = view.typestr == "<f4" or view.typestr == "<f8"
    var classes_store = labels_vector(n, "<f8" if wide_float else "<i8", context)
    var codes = labels_vector(n, "<i4", context)
    # Every view here is contiguous (`_view_of_buffer` linearizes a strided
    # export; an Array is C or F), so a vector's storage is its logical order.
    var y_addr = view.addr
    var keep = view.keep
    var k = 0
    try:
        k = Int(py=py_call_typed(labels_helper(context, "prepare_targets"), PythonObject(y_addr),
                                PythonObject(_encode_code(view.typestr)), PythonObject(n),
                                PythonObject(labels_address(classes_store)), PythonObject(labels_address(codes)),
                                PythonObject(0), PythonObject(0), PythonObject(0)))
        _ = codes  # keep the Python object alive through the native call above
        _ = classes_store  # keep the Python object alive through the native call above
    except error:
        var message = String(error)
        if "NaN label" in message:
            if message.startswith("ValueError: "):
                raise error^
            raise Error("ValueError: " + message)
        raise error^
    _ = keep^
    var b = labels_builtins()
    var classes = Python.list()
    for i in range(k):  # driver-loop: over the k distinct classes
        if wide_float:
            _ = py_call(classes.append, PythonObject(_F64P(unsafe_from_address=labels_address(classes_store))[i]))
            _ = classes_store  # keep the Python object alive through the native call above
        else:
            var v = Int(_I64P(unsafe_from_address=labels_address(classes_store))[i])
            _ = classes_store  # keep the Python object alive through the native call above
            if bool_source:
                _ = py_call(classes.append, PythonObject(v != 0))
                _ = classes_store  # keep the Python object alive through the native call above
            else:
                _ = py_call(classes.append, PythonObject(v))
                _ = classes_store  # keep the Python object alive through the native call above
    var _keep_ret_classes_store = ClassicalLabels(classes, codes, n)
    _ = classes_store  # keep the Python object alive through the native call above
    return _keep_ret_classes_store^


def _label_list(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`y.tolist()` of a label container that has one (an ndarray, an
    `array.array`, a pandas Series, an Array): the C methods of the exact
    native classes; any other buffer through its memoryview."""
    var b = labels_builtins()
    var cls = py_type(y)
    var ndarray = py_get(context, "ndarray")
    if ndarray is not None and cls is ndarray:
        return py_call(ndarray.tolist, y)
    if cls is context["array_type"]:
        return py_call(context["array_type"].tolist, y)
    if native_pandas_exact(y):
        return _label_list(_pandas_values(y), context)
    if cls is context["array_class"]:
        return py_call(native_buffer_view(y._mv).tolist)
    # Only connected C buffer exporters: a Python class's __buffer__ hook
    # would run inside PyMemoryView_FromObject.
    var owners = py_get(context, "buffer_owner_types")
    if cls is not b.memoryview and (owners is None or not _labels_one_of(cls, owners)):
        raise Error("TypeError: mojolearn: y must be a list, tuple, array or native buffer of labels, not "
                    + labels_type_name(cls))
    return py_call(native_buffer_view(y).tolist)


def _flatten_labels(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.flatten_labels`: a flat Python list of the label objects,
    nested lists and tuples flattened depth first."""
    var b = labels_builtins()
    var item = y
    var cls = py_type(y)
    if cls is not b.list and cls is not b.tuple and cls is not b.str and cls is not b.bytes:
        item = _label_list(y, context)
        cls = py_type(item)
    if cls is b.str or cls is b.bytes:
        raise Error("mojolearn: y must be a sequence of labels")
    var out = Python.list()
    _flatten_into(item, out, b)
    return out


def _flatten_into(item: PythonObject, out_buf: PythonObject, b: PythonObject) raises:
    var cls = py_type(item)
    if cls is b.list or cls is b.tuple:
        for k in range(len(item)):  # cpu-route: Python label objects, the explicit label input step
            _flatten_into(item[k], out_buf, b)
        return
    _ = py_call(out_buf.append, item)


def _label_kind(t: PythonObject, b: PythonObject, context: PythonObject) raises -> Int:
    """`_labels._kind_of_type`: 1 str, 2 number, 0 refused. Exact native
    types only: a subclass's Python comparison/hash hooks never run."""
    if labels_text_type(t, context):
        return 1
    if labels_real_type(t, context):
        return 2
    return 0


def _sorted_classes(labels: PythonObject, context: PythonObject) raises -> ClassicalLabels:
    """`_labels.sorted_classes`: the ORDER RULE over Python label objects
    (numbers by value, strings by code point, the first of equal labels
    kept), its refusals in walk order, one dense code per label."""
    var b = labels_builtins()
    var n = len(labels)
    if n == 0:
        raise Error("mojolearn: y is empty")
    var seen_str = False
    var seen_number = False
    for i in range(n):  # cpu-route: Python label objects, the explicit label input step
        var item = labels[i]
        var t = py_type(item)
        var kind = _label_kind(t, b, context)
        if kind == 0:
            raise Error("mojolearn: label of type " + labels_type_name(t) + " is neither a number nor a str")
        if kind == 2 and Bool(py=item != item):
            raise Error("mojolearn: y contains a NaN label; NaN is not a class")
        seen_str = seen_str or kind == 1
        seen_number = seen_number or kind == 2
    if seen_str and seen_number:
        raise Error("mojolearn: y mixes numeric and str labels; encode one kind before fitting")
    var classes = py_call(b.sorted, py_call(b.dict.fromkeys, labels))
    var index = Python.dict()
    for i in range(len(classes)):  # driver-loop: over class index table (classes-sized: distinct classes)
        py_setitem(index, classes[i], PythonObject(i))
    var codes = labels_vector(n, "<i4", context)
    var p = _I32P(unsafe_from_address=labels_address(codes))
    _ = codes  # keep the Python object alive through the native call above
    for i in range(n):  # cpu-route: Python label objects, the explicit label input step
        p[i] = Int32(Int(py=index[labels[i]]))
        _ = codes  # keep the Python object alive through the native call above
    return ClassicalLabels(classes, codes, n)


def _numeric_label_view(y: PythonObject, context: PythonObject, mut bool_source: Bool) raises -> Optional[LabelsView]:
    """The numeric label vector `_labels._encode_labels_native` encodes, or
    None for what only the label input step reads (lists, str or object
    buffers, a pandas Series, a matrix, an empty vector)."""
    var b = labels_builtins()
    var cls = py_type(y)
    if cls is b.list or cls is b.tuple or cls is b.str or cls is b.bytes or native_pandas_exact(y):
        return None
    var found: Optional[LabelsView] = None
    try:
        if cls is context["array_class"]:
            found = _view_of_array(y)
        else:
            var probe = native_buffer_view(y)
            bool_source = String(probe.format) == "?"
            found = _view_of_buffer(y, "y", context)
    except:
        return None
    var view = found.value().copy()
    var n = view.size()
    var big = 0
    for k in range(view.ndim()):  # driver-loop: over array rank metadata
        if view.shape[k] > big:
            big = view.shape[k]
    if _encode_code(view.typestr) < 0 or n == 0 or n != big:
        return None
    return view^


def _list_labels(y: PythonObject, context: PythonObject) raises -> Optional[ClassicalLabels]:
    """`_labels._encode_label_list_native`: a flat list or tuple whose every
    label is exactly an int, or exactly a float, through the native encoder;
    None for anything else (and for an int outside int64)."""
    var b = labels_builtins()
    var n = len(y)
    if n == 0:
        return None
    var all_int = True
    var all_float = True
    for i in range(n):  # cpu-route: Python label objects, the explicit label input step
        var t = py_type(y[i])
        all_int = all_int and t is b.int
        all_float = all_float and t is b.float
    if not all_int and not all_float:
        return None
    var store = PythonObject(None)
    try:
        store = native_array_from_values(String("q" if all_int else "d"), y)
    except:
        return None  # an int outside int64: a class only Python can hold
    var array = classical_wrap_array(store, Python.tuple(n), "<i8" if all_int else "<f8", "C", context)
    return _native_labels(_view_of_array(array), False, context)


def labels_encode_labels(y: PythonObject, context: PythonObject) raises -> ClassicalLabels:
    """`_labels.encode_labels(y)`: one native `prepare_targets` call for a
    numeric label vector (a buffer or a flat list of exact ints or exact
    floats); the ORDER RULE over Python objects for str and object labels."""
    var b = labels_builtins()
    var bool_source = False
    var numeric = _numeric_label_view(y, context, bool_source)
    if numeric:
        return _native_labels(numeric.value(), bool_source, context)
    var flat = _flatten_labels(y, context)
    var listed = _list_labels(flat, context)
    if listed:
        return listed.value().copy()
    return _sorted_classes(flat, context)


def labels_label_kind(classes: PythonObject, context: PythonObject) raises -> String:
    """`_labels.label_kind`: 'int', 'float' or 'object'."""
    var b = labels_builtins()
    var all_int = True
    var all_real = True
    for i in range(len(classes)):  # driver-loop: over kind test of the class labels (classes-sized: distinct classes)
        var c = classes[i]
        var t = py_type(c)
        var is_bool = labels_bool_type(t, context)
        var integral = labels_integral_type(t, context) and not is_bool
        var real = labels_real_type(t, context) and not is_bool
        all_int = all_int and integral
        all_real = all_real and real
    if all_int:
        return "int"
    if all_real:
        return "float"
    return "object"


def labels_decode_labels(classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.decode_labels(classes, codes)` for int64 or int32 codes:
    integer classes an int64 Array, real classes a float64 Array (the base
    binding's gathers), str or bool classes a Python list."""
    var b = labels_builtins()
    var kind = labels_label_kind(classes, context)
    var n = Int(py=codes.size)
    var k = len(classes)
    var wide_codes = codes
    if String(codes.dtype) == "<i4":
        wide_codes = labels_vector(n, "<i8", context)
        labels_cast(labels_address(codes), "<i4", labels_address(wide_codes), "<i8", n, context)
        _ = wide_codes  # keep the Python object alive through the native call above
    if (kind == "int" or kind == "float") and k > 0:
        var table = labels_vector(k, "<i8" if kind == "int" else "<f8", context)
        var fits = True
        for i in range(k):  # driver-loop: over the k class values into the gather table
            if kind == "int":
                var v = classes[i]
                if Bool(py=v < PythonObject(-9223372036854775808)) or Bool(py=v > PythonObject(9223372036854775807)):
                    fits = False
                    break
                _I64P(unsafe_from_address=labels_address(table))[i] = Int64(Int(py=py_call(b.int, v)))
                _ = table  # keep the Python object alive through the native call above
            else:
                _F64P(unsafe_from_address=labels_address(table))[i] = Float64(py=py_call(b.float, classes[i]))
                _ = table  # keep the Python object alive through the native call above
        if fits:
            var out = labels_vector(n, "<i8" if kind == "int" else "<f8", context)
            var fn_obj = labels_helper(context, "gather_i64" if kind == "int" else "gather_f64")
            _ = py_call_typed(fn_obj, PythonObject(labels_address(table)), PythonObject(k),
                              PythonObject(labels_address(wide_codes)), PythonObject(n), PythonObject(labels_address(out)))
            _ = table  # keep the Python object alive through the native call above
            _ = out  # keep the Python object alive through the native call above
            _ = wide_codes  # keep the Python object alive through the native call above
            return out
    var values = Python.list()
    var p = _I64P(unsafe_from_address=labels_address(wide_codes))
    _ = wide_codes  # keep the Python object alive through the native call above
    for i in range(n):  # cpu-route: str or bool classes decode to Python objects, the explicit label output step
        _ = py_call(values.append, classes[Int(p[i])])
        _ = wide_codes  # keep the Python object alive through the native call above
    return values


def labels_argmax_rows(scores: PythonObject, rows: Int, cols: Int, wide: Bool, context: PythonObject) raises -> PythonObject:
    """`_labels.argmax_rows` of a C-order float32 or float64 block: the
    base binding's first-max scan, int64 codes."""
    var out = labels_vector(rows, "<i8", context)
    if rows == 0:
        return out
    if cols == 0:
        raise Error("argmax_rows: scores have no columns")
    var fn_obj = labels_helper(context, "argmax_rows_f64" if wide else "argmax_rows_f32")
    _ = py_call_typed(fn_obj, PythonObject(labels_address(scores)), PythonObject(rows), PythonObject(cols),
                      PythonObject(labels_address(out)))
    _ = out  # keep the Python object alive through the native call above
    return out


# ---------------------------------------------------------- class weights


def labels_class_weight_rows(class_weight: PythonObject, labels: ClassicalLabels, xtrees: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    """`_class_weights._class_weight_rows(class_weight, classes, codes,
    xbind)` with int32 codes on the x_trees binding: per-row float32
    weights from a dict or 'balanced', or None when every class weight is
    1. The counts and the per-row gather are the binding's kernels; only
    the k class weights are read here."""
    var b = labels_builtins()
    if class_weight is None:
        return PythonObject(None)
    var classes = labels.classes
    var k = len(classes)
    var n = labels.n
    var objects = Python.list()
    if py_type(class_weight) is b.str:
        var counts = labels_vector(k, "<i4", context)
        var count_entry = labels_need(xtrees, "x_trees_code_counts", "the x_trees binding has no x_trees_code_counts",
                                     "ImportError")
        _ = py_call_typed(count_entry, PythonObject(labels_address(labels.codes) if n else 0),
                          PythonObject(labels_address(counts)), Python.list(n, k))
        _ = counts  # keep the Python object alive through the native call above
        var wide = labels_vector(k, "<i8", context)
        labels_cast(labels_address(counts), "<i4", labels_address(wide), "<i8", k, context)
        _ = wide  # keep the Python object alive through the native call above
        _ = counts  # keep the Python object alive through the native call above
        var w64 = labels_vector(k, "<f8", context)
        if k and Int(py=py_call_typed(labels_helper(context, "class_ratio_f64"), PythonObject(labels_address(wide)),
                                      PythonObject(k), PythonObject(n), PythonObject(0), PythonObject(labels_address(w64)))):
            raise Error("ZeroDivisionError: division by zero")
        _ = wide  # keep the Python object alive through the native calls above
        _ = w64  # keep the Python object alive through the native calls above
        var p = _F64P(unsafe_from_address=labels_address(w64))
        _ = w64  # keep the Python object alive through the native call above
        for i in range(k):  # driver-loop: over the k class weights for the argument checks below
            _ = py_call(objects.append, PythonObject(p[i]))
            _ = w64  # keep the Python object alive through the native call above
    else:
        if py_type(class_weight) is not b.dict:
            raise Error("TypeError: class_weight must be 'balanced', None or a dict")
        var keys = py_call(b.list, class_weight)
        for i in range(len(keys)):  # driver-loop: over user class_weight keys, admitted before hashing/comparison
            var kt = py_type(keys[i])
            if not labels_text_type(kt, context) and not labels_real_type(kt, context):
                raise Error("TypeError: class_weight keys must be native numbers or strings")
        var unknown = py_call(py_call(b.set, keys).difference, classes)
        if len(unknown) > 0:
            raise Error("class_weight contains labels absent from y")
        for i in range(k):  # driver-loop: over user class_weight dict per class
            _ = py_call(objects.append, py_call(class_weight.get, classes[i], PythonObject(1.0)))
    var values = List[Float64]()
    var all_one = True
    for i in range(k):  # driver-loop: over class weight argument checks
        var v = objects[i]
        var t = py_type(v)
        if labels_bool_type(t, context) or not labels_real_type(t, context):
            raise Error("class weights must be finite nonnegative real numbers")
        var f = Float64(py=py_call(b.float, v))
        if not isfinite(f) or f < 0:
            raise Error("class weights must be finite nonnegative real numbers")
        values.append(f)
        all_one = all_one and Bool(py=v == PythonObject(1))
    if all_one:
        return PythonObject(None)
    var narrowed = labels_vector(k, "<f4", context)
    var q = _F32P(unsafe_from_address=labels_address(narrowed))
    _ = narrowed  # keep the Python object alive through the native call above
    var top = Float32(0)
    for i in range(k):  # driver-loop: over the k class weights narrowed to float32
        var f32 = Float32(values[i])
        if not isfinite(f32) or (values[i] > 0 and f32 == 0):
            raise Error("class weights must remain finite and nonzero when positive in Float32")
        q[i] = f32
        _ = narrowed  # keep the Python object alive through the native call above
        if i == 0 or f32 > top:
            top = f32
    if top <= 0:
        raise Error("class weights must have positive total")
    var out = labels_vector(n, "<f4", context)
    if n:
        var rows_entry = labels_need(xtrees, "x_trees_class_rows", "the x_trees binding has no x_trees_class_rows",
                                    "ImportError")
        _ = py_call_typed(rows_entry, PythonObject(labels_address(labels.codes)), PythonObject(labels_address(narrowed)),
                          PythonObject(labels_address(out)), Python.list(n, k))
        _ = narrowed  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
    return out


def labels_weighted_rows(sample_weight: PythonObject, class_weight: PythonObject, labels: ClassicalLabels,
                        xtrees: PythonObject, context: PythonObject) raises -> LabelsMatrix:
    """`_expansion_trees._trees_weighted_rows`: the checked sample weights
    times the class_weight row weights, one correctly rounded float32
    product per row (`x_trees_mul_f32`); the sample weights alone when no
    class weight differs from 1."""
    var sw = labels_check_sample_weight(sample_weight, labels.n, context)
    if class_weight is None:
        return sw^
    var rows = labels_class_weight_rows(class_weight, labels, xtrees, context)
    if rows is None:
        return sw^
    var out = labels_vector(labels.n, "<f4", context)
    var mul = labels_need(xtrees, "x_trees_mul_f32", "the x_trees binding has no x_trees_mul_f32", "ImportError")
    _ = py_call_typed(mul, PythonObject(sw.addr), PythonObject(labels_address(rows)), PythonObject(labels_address(out)),
                      Python.list(labels.n))
    _ = out  # keep the Python object alive through the native call above
    _ = rows  # keep the Python object alive through the native call above
    _ = sw^
    var _keep_ret_out = LabelsMatrix(out, labels_address(out), labels.n, 1, True)
    _ = out  # keep the Python object alive through the native call above
    return _keep_ret_out^


def labels_class_rows_only(class_weight: PythonObject, labels: ClassicalLabels, xtrees: PythonObject,
                          context: PythonObject) raises -> Optional[LabelsMatrix]:
    """The class_weight row weights without sample weights, or None."""
    var rows = labels_class_weight_rows(class_weight, labels, xtrees, context)
    if rows is None:
        return None
    var _keep_ret_rows = LabelsMatrix(rows, labels_address(rows), labels.n, 1, True)
    _ = rows  # keep the Python object alive through the native call above
    return _keep_ret_rows^


# x_metrics stage ops (`_expansion_metrics._OPS`) and cm_epi kinds.
comptime _OP_GROUP_SORT = 0
comptime _OP_GROUP_SUM = 1
comptime _OP_PAIR_KEY = 2
comptime _OP_OFF_DIFF = 56
comptime _OP_CM_EPI = 59
comptime _CM_COUNT = 2
comptime _CM_WSUM_LEAF = 7
comptime _CM_WSUM_FOLD = 8
comptime _WSUM_BLOCK = 256
comptime _NONE = -1
comptime _PARAMS = 14
comptime _INT32_MAX = 2147483647



# ------------------------------------------------------------ shapes


def labels_shape_of(value: PythonObject, context: PythonObject) raises -> List[Int]:
    """Shape from admitted native storage, without Array properties or protocols."""
    classical_admit_value(value, context)
    var b = labels_builtins()
    var cls = py_type(value)
    var shape = List[Int]()
    if cls is context["array_class"]:
        var dimensions = py_call(context["array_fields"]["shape"], value)
        for k in range(len(dimensions)):  # driver-loop: native Array shape slots
            shape.append(Int(py=dimensions[k]))
        return shape^
    if native_pandas_exact(value):
        var storage = native_pandas_f32(value, context, False, True)
        var dimensions = storage.shape
        for k in range(len(dimensions)):  # driver-loop: admitted native pandas storage dimensions
            shape.append(Int(py=dimensions[k]))
        return shape^
    if cls is b.list or cls is b.tuple:
        shape.append(len(value))
        if len(value) > 0:
            var first = value[0]
            if py_type(first) is b.list or py_type(first) is b.tuple:
                shape.append(len(first))
        return shape^
    var storage = native_torch_numpy(value) if _ordinary_torch(cls, context) else value
    var view = native_buffer_view(storage)
    for k in range(Int(py=view.ndim)):  # driver-loop: native buffer dimensions
        shape.append(Int(py=view.shape[k]))
    _ = storage^
    return shape^


def _shape_text(shape: List[Int]) raises -> String:
    """The Python repr of a shape tuple."""
    var b = labels_builtins()
    var items = Python.list()
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        _ = py_call(items.append, shape[k])
    return String(py_call(b.repr, py_call(b.tuple, items)))


def _buffer_format(value: PythonObject, context: PythonObject) raises -> String:
    """`linear_model._buffer_format`: the struct format code, byte order
    stripped; "" when the value exports no buffer."""
    var b = labels_builtins()
    if py_type(value) is context["array_class"]:
        var dtype = String(value.dtype)
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
        return ""
    try:
        return String(py_call(py_call(b.memoryview, value).format.lstrip, "<>=@!|"))
    except:
        return ""


def _has_float_leaf(value: PythonObject) raises -> Bool:
    """A Python float among the leaves of a nested list or tuple."""
    var b = labels_builtins()
    if py_type(value) is b.float:
        return True
    if py_type(value) is b.list or py_type(value) is b.tuple:
        for k in range(len(value)):  # cpu-route: the leaf types of a user Python list, the explicit input step
            if _has_float_leaf(value[k]):
                return True
    return False


# ------------------------------------------------------------ r2


def _as_f32_1d(source: PythonObject, name: String, context: PythonObject) raises -> LabelsMatrix:
    """`_metrics_impl._as_f32_1d`: a 1-D (or (n, 1) column) float32 vector,
    float64 refused by name, NaN or infinity refused."""
    var shape = labels_shape_of(source, context)
    var column = len(shape) == 2 and shape[1] == 1
    if len(shape) != 1 and not column:
        raise Error("mojolearn metrics: " + name + " must be 1-D, got shape " + _shape_text(shape))
    if shape[0] == 0:
        raise Error("mojolearn metrics: " + name + " is empty")
    var fmt = _buffer_format(source, context)
    var float64 = fmt == "d" if fmt.byte_length() > 0 else _has_float_leaf(source)
    if float64:
        raise Error("TypeError: mojolearn metrics: " + name + " is float64; the float64 overloads of "
                    + "r2_score, kl_divergence and silhouette_score are NOT implemented "
                    + "(no float64 on this GPU). Cast to float32 yourself so the "
                    + "precision you run at is the one you chose.")
    var out = labels_f32(source, len(shape), "C", name, context)
    if not labels_all_finite(out.addr, out.rows * out.cols, False, context):
        raise Error("mojolearn metrics: " + name + " contains NaN or infinity; refused "
                    + "here (metrics/README.md HAND-OFF ask 3) rather than letting it "
                    + "reach a kernel whose answer for it is a washed sentinel")
    _ = out  # keep the Python object alive through the native calls above
    return out^


def _reduce(addr: Int, count: Int, what: Int, context: PythonObject) raises -> PythonObject:
    """`Array.min()` (0) / `max()` (1) of a float32 vector: `reduce_stat`."""
    return py_call_typed(labels_helper(context, "reduce_stat"), PythonObject(addr), PythonObject(0),
                         PythonObject(count), PythonObject(what))


def labels_sample_weight_f32(source: PythonObject, n: Int, caller: String, context: PythonObject) raises -> LabelsMatrix:
    """`_metrics_impl._sample_weight_f32`: one finite, non-negative float32
    weight per sample with a positive total."""
    var shape = labels_shape_of(source, context)
    if len(shape) != 1:
        raise Error("mojolearn " + caller + ": sample_weight must be 1-D, got shape " + _shape_text(shape))
    if shape[0] != n:
        raise Error("mojolearn " + caller + ": sample_weight has " + String(shape[0]) + " entries for "
                    + String(n) + " samples")
    var w = labels_f32(source, 1, "C", "sample_weight", context)
    if not labels_all_finite(w.addr, n, False, context) or Bool(py=_reduce(w.addr, n, 0, context) < PythonObject(0)):
        raise Error("mojolearn " + caller + ": sample_weight must have finite nonnegative entries")
    _ = w  # keep the Python object alive through the native calls above
    if not Bool(py=_reduce(w.addr, n, 1, context) > PythonObject(0)):
        raise Error("mojolearn " + caller + ": sample_weight must have positive total weight")
    _ = w  # keep the Python object alive through the native calls above
    return w^


def _metrics_mode(mode: PythonObject, context: PythonObject) raises -> String:
    """`_metrics_impl._get_binding`'s mode: the requested one or the
    package default."""
    var b = labels_builtins()
    var selected = mode
    if selected is not None:
        if py_type(selected) is not b.str:
            raise Error("numeric_mode must be 'fast' or 'identical'")
        var clean = String(py_call(b.str.lower, py_call(b.str.strip, selected)))
        if clean != "fast" and clean != "identical":
            raise Error("numeric_mode must be 'fast' or 'identical'")
        return clean
    return String(py_call(py_call(context["backend_fields"]["_DEFAULT_MODE"].strip).lower))


def labels_r2(y_true: PythonObject, prediction: PythonObject, sample_weight: PythonObject, mode: PythonObject,
             context: PythonObject) raises -> Float64:
    """`metrics.r2_score(y_true, prediction, sample_weight=..., numeric_mode=mode)`
    for 1-D targets (the default multioutput): the float32 R^2 kernel."""
    var y = _as_f32_1d(y_true, "y_true", context)
    var yh = _as_f32_1d(prediction, "y_pred", context)
    if y.rows != yh.rows:
        raise Error("mojolearn metrics: y_true has " + String(y.rows) + " entries and "
                    + "y_pred has " + String(yh.rows) + "; they must be the same length")
    var selected = _metrics_mode(mode, context)
    var metrics = labels_table(context, selected, "_mojolearn_metrics")
    var read_mode = labels_entry(metrics, "metrics_numeric_mode")
    if read_mode is None:
        read_mode = labels_entry(metrics, "umap_numeric_mode")
    if read_mode is not None:
        var code = Int(py=py_call_typed(read_mode))
        var actual = String("fast") if code == 0 else (String("identical") if code == 1 else String("None"))
        if actual != selected:
            raise Error("RuntimeError: mojolearn metrics: requested " + selected + ", binary reports "
                        + actual + "; rebuild it")
    var result = PythonObject(None)
    if sample_weight is not None:
        var w = labels_sample_weight_f32(sample_weight, y.rows, "r2_score", context)
        result = py_call_typed(metrics["r2_score_weighted"], PythonObject(y.addr), PythonObject(yh.addr),
                               PythonObject(w.addr), Python.list(PythonObject(y.rows)))
        _ = yh  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        _ = w^
    else:
        result = py_call_typed(metrics["r2_score"], PythonObject(y.addr), PythonObject(yh.addr),
                               Python.list(PythonObject(y.rows)))
        _ = yh  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
    _ = y^
    _ = yh^
    return Float64(py=py_call(labels_builtins().float, result))


# ------------------------------------------------------------ accuracy


struct ScoreLabels(Copyable, Movable):
    """`_classification_encoded`'s answer: encoded integer labels (sorted
    distinct Python ints and one int32 code per row) or a list of Python
    str or int labels; `kind` 'integer' or 'string'."""
    var encoded: Bool
    var classes: PythonObject
    var codes: PythonObject
    var values: PythonObject
    var n: Int
    var kind: String

    def __init__(out self, encoded: Bool, classes: PythonObject, codes: PythonObject, values: PythonObject,
                 n: Int, kind: String):
        self.encoded = encoded
        self.classes = classes
        self.codes = codes
        self.values = values
        self.n = n
        self.kind = kind


def _native_classification(values: PythonObject, context: PythonObject) raises -> Optional[ScoreLabels]:
    """`_metrics_impl._native_classification_labels`: an integer or bool
    buffer (or a flat list of exact ints) through the native encoder."""
    var b = labels_builtins()
    var cls = py_type(values)
    if cls is b.str or cls is b.bytes:
        return None
    var view: LabelsView
    var bool_source = False
    if cls is b.list or cls is b.tuple:
        var n = len(values)
        if n == 0:
            return None
        for i in range(n):  # cpu-route: the label types of a user Python list, the explicit label input step
            if py_type(values[i]) is not b.int:
                return None
        var store = PythonObject(None)
        try:
            store = native_array_from_values(String("q"), values)
        except:
            return None
        view = _view_of_array(classical_wrap_array(store, Python.tuple(n), "<i8", "C", context))
    else:
        if native_pandas_exact(values):
            return None
        if len(labels_shape_of(values, context)) != 1:
            return None
        try:
            if cls is context["array_class"]:
                view = _view_of_array(values)
            else:
                bool_source = String(py_call(b.memoryview, values).format) == "?"
                view = _view_of_buffer(values, "input", context)
        except:
            return None
    if view.ndim() != 1 or view.size() < 1:
        return None
    var typestr = view.typestr
    if typestr == "<u2" or typestr == "<i1":
        # exact; the encoder has no 8- or 16-bit arm (`astype('<i4')`)
        var wide = labels_vector(view.size(), "<i4", context)
        labels_cast(view.addr, typestr, labels_address(wide), "<i4", view.size(), context)
        _ = wide  # keep the Python object alive through the native call above
        view = _view_of_array(wide)
    if view.typestr != "<i4" and view.typestr != "<i8" and view.typestr != "<u4" and view.typestr != "<u1":
        return None
    # A bool buffer's classes are read as the ints 0 and 1 (`int(c)`).
    var labels = _native_labels(view, False, context)
    return ScoreLabels(True, labels.classes, labels.codes, PythonObject(None), labels.n, "integer")


def _classification_labels(values: PythonObject, name: String, context: PythonObject) raises -> ScoreLabels:
    """`_metrics_impl._classification_labels`: one-dimensional single-label
    data of str or integer labels (bools read as ints)."""
    var b = labels_builtins()
    var shape = labels_shape_of(values, context)
    if len(shape) != 1:
        raise Error(name + " must be one-dimensional single-label data")
    var labels = _flatten_labels(values, context)
    var n = len(labels)
    if n == 0:
        raise Error(name + " must contain at least one label")
    var all_str = True
    var all_int = True
    for i in range(n):  # cpu-route: the label types of user labels, the explicit label input step
        var v = labels[i]
        var t = py_type(v)
        all_str = all_str and labels_text_type(t, context)
        all_int = all_int and (labels_integral_type(t, context) or labels_bool_type(t, context))
    var out = Python.list()
    if all_str:
        for i in range(n):  # cpu-route: str labels, the explicit label input step
            _ = py_call(out.append, py_call(b.str, labels[i]))
        return ScoreLabels(False, PythonObject(None), PythonObject(None), out, n, "string")
    if all_int:
        for i in range(n):  # cpu-route: Python int labels, the explicit label input step
            _ = py_call(out.append, py_call(b.int, labels[i]))
        return ScoreLabels(False, PythonObject(None), PythonObject(None), out, n, "integer")
    raise Error("TypeError: " + name + " must contain only strings or only integers; "
                + "floating labels, missing labels and mixed types are unsupported")


def _classification_encoded(values: PythonObject, name: String, context: PythonObject) raises -> ScoreLabels:
    var fast = _native_classification(values, context)
    if fast:
        return fast.value().copy()
    return _classification_labels(values, name, context)


def _label_set(labels: ScoreLabels) raises -> PythonObject:
    var b = labels_builtins()
    return py_call(b.set, labels.classes if labels.encoded else labels.values)


def _codes(labels: ScoreLabels, order: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_metrics._codes`: each label's index in `order` (-1 when
    absent) as an int32 Array; encoded labels map their class table once and
    gather per row (`gather_i32`)."""
    var index = Python.dict()
    for i in range(len(order)):  # driver-loop: over the distinct class labels
        py_setitem(index, order[i], PythonObject(i))
    var out = labels_vector(labels.n, "<i4", context)
    if labels.encoded:
        var k = len(labels.classes)
        var table = labels_vector(k, "<i4", context)
        var t = _I32P(unsafe_from_address=labels_address(table))
        _ = table  # keep the Python object alive through the native call above
        for i in range(k):  # driver-loop: over the distinct class labels
            t[i] = Int32(Int(py=py_get(index, labels.classes[i], PythonObject(-1))))
            _ = table  # keep the Python object alive through the native call above
        _ = py_call_typed(labels_helper(context, "gather_i32"), PythonObject(labels_address(table)), PythonObject(k),
                          PythonObject(labels_address(labels.codes)), PythonObject(labels.n),
                          PythonObject(labels_address(out)))
        _ = table  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        return out
    var p = _I32P(unsafe_from_address=labels_address(out))
    _ = out  # keep the Python object alive through the native call above
    for i in range(labels.n):  # cpu-route: Python object labels mapped per row, the explicit label input step
        p[i] = Int32(Int(py=py_get(index, labels.values[i], PythonObject(-1))))
        _ = out  # keep the Python object alive through the native call above
    return out


def _refuse_multilabel(y: PythonObject, name: String, caller: String, context: PythonObject) raises:
    var shape = labels_shape_of(y, context)
    if len(shape) == 2 and shape[1] != 1:
        raise Error("NotImplementedError: mojolearn " + caller + ": " + name + " is 2-D; multilabel-indicator targets are NOT "
                    + "IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv), only binary and multiclass 1-D labels")


struct _Program(Movable):
    """The fields of an `_expansion_metrics._Prog` the native runner reads:
    its size, input records, stage records and skipped ranges."""
    var size: Int
    var inputs: PythonObject
    var stages: PythonObject
    var skip: PythonObject

    def __init__(out self) raises:
        self.size = 0
        self.inputs = Python.list()
        self.stages = Python.list()
        self.skip = Python.list()

    def alloc(mut self, n: Int) raises -> Int:
        var off = self.size
        self.size += max(n, 0)
        if self.size > _INT32_MAX:
            raise Error("mojolearn metrics: the problem exceeds the Int32 arena bound")
        return off

    def put(mut self, array: PythonObject, count: Int) raises -> Int:
        """`put` / `put_i32` of a C-order float32 or int32 Array."""
        var off = self.alloc(count)
        _ = py_call(self.inputs.append, Python.tuple(PythonObject(off), array))
        _ = py_call(self.skip.append, Python.tuple(PythonObject(off), PythonObject(off + count)))
        return off

    def scratch(mut self, n: Int) raises -> Int:
        var off = self.alloc(n)
        if n > 0:
            _ = py_call(self.skip.append, Python.tuple(PythonObject(off), PythonObject(off + n)))
        return off

    def stage(mut self, op: Int, total: Int, params: List[Int]) raises:
        if len(params) > _PARAMS:
            raise Error("x_metrics: too many stage parameters")
        var record = Python.list(PythonObject(op), PythonObject(total))
        for i in range(_PARAMS):  # small-loop(params): the fourteen fixed parameter words of one stage
            _ = py_call(record.append, PythonObject(params[i] if i < len(params) else 0))
        _ = py_call(self.stages.append, record)

    def group(mut self, key: Int, n: Int, m: Int, weights: Int) raises -> Tuple[Int, Int]:
        """`_expansion_metrics._group` (values=_NONE, vstride=1, width=1)."""
        var off = self.alloc(m + 1)
        var order = self.scratch(n)
        var out = self.alloc(m)
        self.stage(_OP_GROUP_SORT, 1, [key, n, m, off, order])
        if m:
            self.stage(_OP_GROUP_SUM, m, [off, order, _NONE, 1, weights, out, 1])
        return (off, out)

    def counts(mut self, off: Int, m: Int) raises -> Int:
        """`_expansion_metrics._counts`."""
        var out = self.alloc(m)
        if m:
            self.stage(_OP_OFF_DIFF, m, [off, m, out])
        return out


def _run(var program: _Program, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_run(prog, numeric_mode)`: the shared native program runner over a
    program object holding these fields. Returns the arena store."""
    var b = labels_builtins()
    var sys = native_loaded_module("sys")
    var module = py_get(sys.modules, "mojolearn._expansion_metrics")
    if module is None:
        raise Error("ImportError: mojolearn: the metrics programs are not connected")
    var prog_context = py_get(py_fields(module), "_PROG_CONTEXT")
    if prog_context is None:
        raise Error("ImportError: mojolearn: the metrics programs are not connected")
    var prog = py_call(context["labels_namespace"])
    var fields = py_fields(prog)
    py_setitem(fields, "size", PythonObject(program.size))
    py_setitem(fields, "_inputs", program.inputs)
    py_setitem(fields, "_stages", program.stages)
    py_setitem(fields, "_skip", program.skip)
    py_setitem(fields, "_outs", PythonObject(None))
    py_setitem(fields, "arena", PythonObject(None))
    _ = _metrics_run(prog, mode, prog_context)
    return fields["arena"]


def _arena_address(arena: PythonObject) raises -> Int:
    return Int(py=py_call(arena.buffer_info)[0])


def labels_accuracy(y_true: PythonObject, prediction: PythonObject, sample_weight: PythonObject, mode: PythonObject,
                   context: PythonObject) raises -> Float64:
    """`_expansion_metrics.accuracy_fraction(y_true, prediction,
    sample_weight, mode)`: the (weighted) matched fraction from the device
    COUNT tail."""
    var b = labels_builtins()
    var caller = String("accuracy_score")
    _refuse_multilabel(y_true, "y_true", caller, context)
    _refuse_multilabel(prediction, "y_pred", caller, context)
    var true = _classification_encoded(y_true, "y_true", context)
    var pred = _classification_encoded(prediction, "y_pred", context)
    if true.n != pred.n:
        raise Error("mojolearn " + caller + ": y_true and y_pred lengths differ")
    if true.kind != pred.kind:
        raise Error("TypeError: mojolearn " + caller + ": y_true and y_pred must use the same label type")
    if true.n > _INT32_MAX:
        raise Error("mojolearn " + caller + ": at most INT32_MAX rows")
    var n = true.n
    var weighted = sample_weight is not None
    var w: Optional[LabelsMatrix] = None
    if weighted:
        w = labels_sample_weight_f32(sample_weight, n, caller, context)
    var present = py_call(b.sorted, py_call(_label_set(true).union, _label_set(pred)))
    var L = len(present)
    var yt = _codes(true, present, context)
    var yp = _codes(pred, present, context)
    # `_Sums(true, pred, w, present, mode, tail=_cm_stage(..., COUNT, 11))`.
    var prog = _Program()
    var a = prog.put(yt, n)
    var bb = prog.put(yp, n)
    var matched = prog.scratch(n)
    prog.stage(_OP_PAIR_KEY, n, [a, bb, matched, L, 1])
    var W = _NONE
    var w_array = PythonObject(None)
    if weighted:
        w_array = w.value().keep if py_type(w.value().keep) is context["array_class"] else _owned_f32(w.value(), context)
        W = prog.put(w_array, n)
    var g0 = prog.group(matched, n, L, W)
    var g1 = prog.group(a, n, L, W)
    var g2 = prog.group(bb, n, L, W)
    if weighted:
        # every row in group 0: the total weight grouping
        var zero = prog.scratch(n)
        prog.stage(_OP_PAIR_KEY, n, [a, a, zero, 1, 2])
        _ = prog.group(zero, n, 1, W)
    if not weighted:
        _ = prog.counts(g0[0], L)
        _ = prog.counts(g1[0], L)
        _ = prog.counts(g2[0], L)
    # `_cm_stage(prog, groups, w, n, L, _CM_COUNT, 11)`.
    var out = prog.alloc(11)
    var tp = g0[1] if weighted else g0[0]
    var tot = n
    if weighted:
        # `_wsum_sf64(prog, L)`: the soft binary64 hit and weight total.
        var nb = max(1, (n + _WSUM_BLOCK - 1) // _WSUM_BLOCK)
        var s = prog.scratch(4 * nb)
        prog.stage(_OP_CM_EPI, nb, [_CM_WSUM_LEAF, matched, W, n, s, L])
        var stride = 1
        while stride < nb:  # driver-loop: over the fold levels (log2 of the block count)
            prog.stage(_OP_CM_EPI, (nb - stride + 2 * stride - 1) // (2 * stride), [_CM_WSUM_FOLD, s, stride, nb])
            stride *= 2
        tot = s
    prog.stage(_OP_CM_EPI, 1, [_CM_COUNT, tp, L, 1 if weighted else 0, tot, out])
    var arena = _run(prog^, mode, context)
    var base = _arena_address(arena)
    _ = arena  # keep the Python object alive through the native call above
    _ = yt^
    _ = yp^
    _ = w^
    _ = w_array^
    if weighted and Int(_I32P(unsafe_from_address=base + 4 * out)[0]) != 0:
        raise Error("ZeroDivisionError: float division by zero")
    _ = arena  # keep the Python object alive through the native calls above
    var fraction = _F64P(unsafe_from_address=base + 4 * (out + 8))[0]
    _ = arena  # keep the Python object alive through the native call above
    _ = arena^
    return fraction


def _owned_f32(w: LabelsMatrix, context: PythonObject) raises -> PythonObject:
    """A float32 Array copy of weights that borrow a foreign buffer (the
    program runner copies Arrays into its arena; `put` took `as_f32_c`'s
    Array). Copy the already-admitted contiguous float32 words natively."""
    var count = w.rows * w.cols
    var out = labels_vector(count, "<f4", context)
    # `keep` owns the admitted storage; it need not itself export bytes.
    # Re-entering its buffer protocol is unnecessary after labels_f32.
    memcpy(dest=_F32P(unsafe_from_address=labels_address(out)),
           src=_F32P(unsafe_from_address=w.addr), count=count)
    _ = w  # retain the source export until the copy finishes
    return out




# ------------------------------------------------------------ public API
# The names the classical owners call.


def classical_encode_labels(y: PythonObject, context: PythonObject) raises -> ClassicalLabels:
    """`_labels.encode_labels(y)`: `classes` the Python class list and one
    int32 code per row (an Array), the same classes, codes and refusals."""
    return labels_encode_labels(y, context)


def classical_decode_labels(classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.decode_labels(classes, codes)` for an int32 or int64 codes
    Array."""
    return labels_decode_labels(classes, codes, context)


def classical_accuracy(y_true: PythonObject, prediction: PythonObject, sample_weight: PythonObject,
                       mode: PythonObject, context: PythonObject) raises -> Float64:
    """`_expansion_metrics.accuracy_fraction(y_true, prediction,
    sample_weight, mode)`."""
    return labels_accuracy(y_true, prediction, sample_weight, mode, context)
