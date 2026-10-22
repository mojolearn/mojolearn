# SPDX-License-Identifier: Apache-2.0
# Input, label, weight and output handling of the tree-ensemble owner
# (bindings/trees_public_native.mojo). These are the steps the Python
# helpers `_buffer.as_f32_c` / `as_f32_colmajor` / `as_f32_forest_layout`,
# `_labels.encode_labels` / `decode_labels` / `argmax_rows`,
# `_sample_weight.check_sample_weight` and `_class_weights._class_weight_rows`
# ran before each forest launch, with their refusals and their wording, and
# with the same native helpers in the same order (the base binding's casts,
# transposes, finiteness scans, `prepare_targets` and gathers; the x_trees
# binding's weight kernels), so every byte a kernel reads is unchanged.
#
# No Python function runs here: inputs are read through the buffer protocol
# and the Array slots, outputs are mojolearn Arrays assembled from their
# slots (classical_wrap_array), and the only callables invoked are compiled
# entry points connected at import (python/mojolearn/_native_trees_bridge.py)
# and C builtins.
from std.math import isfinite
from std.memory import bitcast
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_setattr
from bindings.py_typed_call import py_call_typed
from bindings.native_buffer_owner import native_buffer_view
from bindings.classical_arrays import classical_wrap_array, _store_code, _buffer_typestr
from bindings.classical_pandas import native_pandas_exact, native_pandas_f32, _pd_fields, _pd_get
from bindings.sequential_admission import _ordinary_torch
from bindings.native_torch_admission import native_torch_numpy
from bindings.sequential_inputs import sequential_sequence

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]

#: The forests' refusal of a non-finite X (randomforest.py / extratrees.py
#: `_NONFINITE_REFUSAL`).
comptime TREES_NONFINITE = "X contains NaN or infinity; the forest has no missing-value arm"


# ------------------------------------------------------------ connection


def trees_builtins() raises -> PythonObject:
    return Python.import_module("builtins")


def trees_table(context: PythonObject, mode: String, name: String) raises -> PythonObject:
    """The connected entry table of binding `name` in `mode`, or the
    refusal recorded at import ("<Class>: <message>")."""
    var tables = py_get(context["trees_bindings"], PythonObject(mode))
    if tables is None:
        raise Error("numeric_mode must be fast, identical or None")
    var table = py_get(tables, PythonObject(name))
    if table is None:
        raise Error("ImportError: mojolearn: " + name + " is not connected to the tree owner")
    if py_type(table) is py_type(PythonObject("")):
        raise Error(String(table))
    return table


def trees_entry(table: PythonObject, name: String) raises -> PythonObject:
    """A connected compiled entry point, or None when the binary has none."""
    return py_get(table, PythonObject(name))


def trees_need(table: PythonObject, name: String, message: String, kind: String = "RuntimeError") raises -> PythonObject:
    """A connected entry point the operation requires, refused with the
    Python wording when the binary predates it."""
    var entry = py_get(table, PythonObject(name))
    if entry is None:
        raise Error(kind + ": " + message)
    return entry


def trees_helper(context: PythonObject, name: String) raises -> PythonObject:
    """A base binding helper (`_buffer._native`), or the recorded refusal."""
    var entry = py_get(context["trees_helpers"], PythonObject(name))
    if entry is None or py_type(entry) is py_type(PythonObject("")):
        if entry is None:
            raise Error("ImportError: mojolearn: the base binding has no `" + name + "`")
        raise Error(String(entry))
    return entry


def trees_env(name: String, default: String) -> String:
    var value = getenv(name)
    return value if value else default


# --------------------------------------------------------------- outputs


def trees_itemsize(typestr: String) raises -> Int:
    if typestr == "<f8" or typestr == "<i8":
        return 8
    if typestr == "<f4" or typestr == "<i4" or typestr == "<u4":
        return 4
    if typestr == "<f2" or typestr == "<u2":
        return 2
    if typestr == "<u1" or typestr == "<i1":
        return 1
    raise Error("TypeError: " + typestr + " is not an Array dtype")


def trees_empty(shape: PythonObject, typestr: String, context: PythonObject, order: String = "C") raises -> PythonObject:
    """A new zero-filled mojolearn Array (`_buffer.empty`), assembled from
    its slots; never an ndarray, whatever is installed."""
    var count = 1
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        count *= Int(py=shape[k])
    var width = trees_itemsize(typestr)
    var store = py_call(context["array_type"], PythonObject(_store_code(typestr)),
                        py_call(context["native_bytes_type"], count * width))
    return classical_wrap_array(store, shape, typestr, order, context)


def trees_vector(count: Int, typestr: String, context: PythonObject) raises -> PythonObject:
    return trees_empty(Python.tuple(count), typestr, context)


def trees_full_f32(count: Int, value: Float64, context: PythonObject) raises -> PythonObject:
    """`_buffer.full((count,), value, '<f4')`: one float32 conversion of
    `value`, the store repeated in C (`array.array.__mul__`)."""
    var one = py_call(context["array_type"], PythonObject("f"), Python.list(PythonObject(value)))
    var store = py_call(one.__mul__, PythonObject(count))
    return classical_wrap_array(store, Python.tuple(count), "<f4", "C", context)


def trees_reshape(array: PythonObject, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`array.reshape(shape)` of a fresh C-order Array: the same store under
    a new shape (the Array the Python reshape returned)."""
    return classical_wrap_array(array._store, shape, String(array.dtype), "C", context)


def trees_freeze(array: PythonObject) raises:
    """A fitted model array the caller cannot write through (the copy
    `_mojolearn_after_fit` took): the fresh export's own storage, marked
    read-only."""
    py_setattr(array, "_readonly", PythonObject(True))


def trees_address(array: PythonObject) raises -> Int:
    return Int(py=array._addr)


# ----------------------------------------------------------------- inputs


struct TreesView(Copyable, Movable):
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


def trees_shape_repr(view: TreesView) raises -> String:
    """The Python repr of the shape tuple, as `_as_typed` printed it."""
    var b = trees_builtins()
    var items = Python.list()
    for k in range(len(view.shape)):  # driver-loop: over array rank metadata
        _ = py_call(items.append, view.shape[k])
    return String(py_call(b.repr, py_call(b.tuple, items)))


def _view_of_array(value: PythonObject) raises -> TreesView:
    var shape = List[Int]()
    var strides = List[Int]()
    var rank = Int(py=value.ndim)
    for k in range(rank):  # driver-loop: over array rank metadata
        shape.append(Int(py=value.shape[k]))
        strides.append(Int(py=value.strides[k]))
    var typestr = String(value.dtype)
    return TreesView(value, Int(py=value._addr), shape^, strides^, typestr, Int(py=value.itemsize))


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


def _view_of_buffer(value: PythonObject, name: String, context: PythonObject) raises -> TreesView:
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
        var store = py_call(context["array_type"], PythonObject(_store_code(typestr)), py_call(view.tobytes, "C"))
        var pyshape = Python.list()
        for k in range(len(shape)):  # driver-loop: over array rank metadata
            _ = py_call(pyshape.append, shape[k])
        return _view_of_array(classical_wrap_array(store, py_call(trees_builtins().tuple, pyshape), typestr, "C", context))
    return TreesView(view, _any_contiguous_address(view), shape^, strides^, typestr, item)


def _pandas_values(source: PythonObject) raises -> PythonObject:
    """The storage array of an exact single-block pandas Series (what its
    `tolist()` lists), read through pandas' native slots."""
    var manager = py_get(_pd_fields(source), "_mgr")
    var blocks = py_get(_pd_fields(manager), "blocks")
    if py_type(blocks) is not trees_builtins().tuple or len(blocks) != 1:
        raise Error("TypeError: pandas Series requires native single-block storage")
    return _pd_get("pandas._libs.internals", "Block", "values", blocks[0])


def trees_view(source: PythonObject, name: String, context: PythonObject) raises -> TreesView:
    """`_buffer._materialize(source, name)` read in place: an Array, any
    buffer exporter (an ndarray, an `array.array`, a memoryview), a pandas
    frame or series, a torch tensor, or a nested list or tuple of numbers.
    The refusals keep `_materialize`'s classes and wording."""
    var b = trees_builtins()
    var cls = py_type(source)
    if cls is context["array_class"]:
        return _view_of_array(source)
    if cls is b.str or cls is b.bytes:
        raise Error("TypeError: mojolearn: " + name + " is a " + String(cls.__name__) + ", not an array")
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
            store = py_call(context["array_type"], "q", store)
        var array = classical_wrap_array(store, packed[0], typestr, "C", context)
        return _view_of_array(array)
    try:
        return _view_of_buffer(source, name, context)
    except error:
        if String(error).startswith("TypeError: " + name):
            raise error^
        raise Error("TypeError: mojolearn: " + name + " holds a " + String(cls.__name__) + ", not a number")


def trees_check_rank(view: TreesView, ndim: Int, name: String) raises:
    """`_as_typed`'s rank and emptiness refusals."""
    if ndim >= 0 and view.ndim() != ndim:
        var hint = String("")
        if ndim == 2 and view.ndim() < 2:
            hint = (". Reshape your data either using array.reshape(-1, 1) if your data "
                    + "has a single feature or array.reshape(1, -1) if it contains a single sample.")
        raise Error("mojolearn: " + name + " must be " + String(ndim) + "-D, got " + String(view.ndim())
                    + "-D shape " + trees_shape_repr(view) + hint)
    if view.size() == 0:
        var what = String("feature(s)") if view.ndim() == 2 and view.shape[0] != 0 else String("sample(s)")
        var shown = trees_shape_repr(view)
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


def trees_cast(src: Int, source: String, dst: Int, target: String, count: Int, context: PythonObject) raises:
    """`Array.astype(target)` of a contiguous block (`_native_astype`): the
    core helper `cast_elements`, one conversion per element in storage
    order."""
    if count == 0:
        return
    var refused = Int(py=py_call_typed(trees_helper(context, "cast_elements"), PythonObject(src),
                                      PythonObject(_native_code(source)), PythonObject(dst),
                                      PythonObject(_native_code(target)), PythonObject(count)))
    if refused != 0:
        raise Error("TypeError: mojolearn: an element does not fit " + target)


struct TreesMatrix(Copyable, Movable):
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


def trees_f32(source: PythonObject, ndim: Int, order: String, name: String, context: PythonObject) raises -> TreesMatrix:
    """`_buffer._as_typed(source, '<f4', order, ndim, name)`: the input
    itself when it is already float32 in `order` (no copy), otherwise one
    converted copy through the base binding's casts (a float64 or float32
    source) or the classical element cast (integers and float16)."""
    var view = trees_view(source, name, context)
    trees_check_rank(view, ndim, name)
    return trees_f32_view(view^, order, context)


def trees_f32_view(var view: TreesView, order: String, context: PythonObject) raises -> TreesMatrix:
    var rows = view.shape[0]
    var cols = view.shape[1] if view.ndim() == 2 else 1
    var two = view.ndim() == 2
    var want_f = order == "F"
    var has_order = view.both_orders() or (view.f_order() if want_f else view.c_order())
    if view.typestr == "<f4" and has_order:
        return TreesMatrix(view.keep, view.addr, rows, cols, not want_f)
    var shape = Python.tuple(rows, cols) if two else Python.tuple(rows)
    var out = trees_empty(shape, "<f4", context, "F" if want_f and two else "C")
    var dst = trees_address(out)
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
            var fn = trees_helper(context, "cast_colmajor_f64_to_f32" if view.typestr == "<f8" else "transpose_f32")
            _ = py_call_typed(fn, PythonObject(view.addr), PythonObject(dst), PythonObject(r), PythonObject(c))
            return TreesMatrix(out, dst, rows, cols, not want_f)
        if view.typestr == "<f8":
            # The flat cast keeps the source's storage order, which is the
            # requested one here (or the two coincide).
            _ = py_call_typed(trees_helper(context, "cast_f64_to_f32"), PythonObject(view.addr), PythonObject(dst),
                              PythonObject(view.size()))
            return TreesMatrix(out, dst, rows, cols, not want_f)
    # `astype('<f4')` keeps the source's storage order; `_as_order` then
    # transposes when the requested order differs (the float32 transpose).
    if not two or view.both_orders() or contiguous_f == want_f:
        trees_cast(view.addr, view.typestr, dst, "<f4", view.size(), context)
        return TreesMatrix(out, dst, rows, cols, not want_f)
    var staged = trees_vector(view.size(), "<f4", context)
    trees_cast(view.addr, view.typestr, trees_address(staged), "<f4", view.size(), context)
    var r = cols if contiguous_f else rows
    var c = rows if contiguous_f else cols
    _ = py_call_typed(trees_helper(context, "transpose_f32"), PythonObject(trees_address(staged)), PythonObject(dst),
                      PythonObject(r), PythonObject(c))
    _ = staged^
    return TreesMatrix(out, dst, rows, cols, not want_f)


def trees_forest_layout(source: PythonObject, name: String, context: PythonObject) raises -> TreesMatrix:
    """`_buffer.as_f32_forest_layout`: a 2-D float32 C-order input as is
    (row-major, a borrow); every other input column-major
    (`as_f32_colmajor`)."""
    var view = trees_view(source, name, context)
    if view.ndim() == 2 and view.typestr == "<f4" and view.size() > 0 and view.c_order() and not view.both_orders():
        return TreesMatrix(view.keep, view.addr, view.shape[0], view.shape[1], True)
    trees_check_rank(view, 2, name)
    return trees_f32_view(view^, "F", context)


def trees_all_finite(addr: Int, count: Int, wide: Bool, context: PythonObject) raises -> Bool:
    """`_buffer.all_finite`: the base binding's scan."""
    if count == 0:
        return True
    var fn = trees_helper(context, "all_finite_f64" if wide else "all_finite_f32")
    return Int(py=py_call_typed(fn, PythonObject(addr), PythonObject(count))) == 1


# ---------------------------------------------------------------- weights


def trees_check_sample_weight(source: PythonObject, n: Int, context: PythonObject) raises -> TreesMatrix:
    """`_sample_weight.check_sample_weight(source, n)` (float32): a Python
    or 0-d scalar fills a vector; a vector must hold n entries; the
    finiteness, sign and all-zero refusals are `prepare_targets`' with
    scikit-learn's wording. The vector may borrow the caller's float32
    buffer (the fit only reads it)."""
    var b = trees_builtins()
    var cls = py_type(source)
    var scalar = False
    var value = Float64(0)
    if Bool(py=py_call(b.isinstance, source, Python.tuple(b.int, b.float))) and not Bool(py=py_call(b.isinstance, source, b.bool)):
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
        var view = trees_view(source, "sample_weight", context)
        if view.size() == 0:
            if n != 0:
                raise Error("sample_weight.shape == (0,), expected (" + String(n) + ",)!")
            trees_check_rank(view, -1, "sample_weight")
        if view.ndim() != 1:
            raise Error("Sample weights must be 1D array or scalar")
        if view.shape[0] != n:
            raise Error("sample_weight.shape == " + trees_shape_repr(view) + ", expected (" + String(n) + ",)!")
        var w = trees_f32_view(view^, "C", context)
        if n:
            trees_prepare_weights(w.addr, n, context)
        return w^
    var out = trees_full_f32(n, value, context)
    if n:
        trees_prepare_weights(trees_address(out), n, context)
    return TreesMatrix(out, trees_address(out), n, 1, True)


def trees_prepare_weights(addr: Int, n: Int, context: PythonObject) raises:
    """`prepare_targets(None, w, n)` on float32 weights: the weight scan,
    refused with scikit-learn's wording."""
    var status = Int(py=py_call_typed(trees_helper(context, "prepare_targets"), PythonObject(0), PythonObject(0),
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


struct TreesLabels(Copyable, Movable):
    """`(classes, codes)`: the class list (Python label objects, the
    `classes_` attribute) and one int32 code per row (an Array)."""
    var classes: PythonObject
    var codes: PythonObject
    var n: Int

    def __init__(out self, classes: PythonObject, codes: PythonObject, n: Int):
        self.classes = classes
        self.codes = codes
        self.n = n


def _native_labels(view: TreesView, bool_source: Bool, context: PythonObject) raises -> TreesLabels:
    """`_labels._encode_labels_native`: one `prepare_targets` call over a
    numeric label vector, the k class values as Python numbers."""
    var n = view.size()
    var wide_float = view.typestr == "<f4" or view.typestr == "<f8"
    var classes_store = trees_vector(n, "<f8" if wide_float else "<i8", context)
    var codes = trees_vector(n, "<i4", context)
    # Every view here is contiguous (`_view_of_buffer` linearizes a strided
    # export; an Array is C or F), so a vector's storage is its logical order.
    var y_addr = view.addr
    var keep = view.keep
    var k = 0
    try:
        k = Int(py=py_call_typed(trees_helper(context, "prepare_targets"), PythonObject(y_addr),
                                PythonObject(_encode_code(view.typestr)), PythonObject(n),
                                PythonObject(trees_address(classes_store)), PythonObject(trees_address(codes)),
                                PythonObject(0), PythonObject(0), PythonObject(0)))
    except error:
        var message = String(error)
        if "NaN label" in message:
            if message.startswith("ValueError: "):
                raise error^
            raise Error("ValueError: " + message)
        raise error^
    _ = keep^
    var b = trees_builtins()
    var classes = Python.list()
    for i in range(k):  # driver-loop: over the k distinct classes
        if wide_float:
            _ = py_call(classes.append, PythonObject(_F64P(unsafe_from_address=trees_address(classes_store))[i]))
        else:
            var v = Int(_I64P(unsafe_from_address=trees_address(classes_store))[i])
            if bool_source:
                _ = py_call(classes.append, PythonObject(v != 0))
            else:
                _ = py_call(classes.append, PythonObject(v))
    return TreesLabels(classes, codes, n)


def _label_list(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`y.tolist()` of a label container that has one (an ndarray, an
    `array.array`, a pandas Series, an Array): the C methods of the exact
    native classes; any other buffer through its memoryview."""
    var b = trees_builtins()
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
    return py_call(native_buffer_view(y).tolist)


def _flatten_labels(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.flatten_labels`: a flat Python list of the label objects,
    nested lists and tuples flattened depth first."""
    var b = trees_builtins()
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


def _flatten_into(item: PythonObject, out: PythonObject, b: PythonObject) raises:
    var cls = py_type(item)
    if Bool(py=py_call(b.isinstance, item, Python.tuple(b.list, b.tuple))):
        for k in range(len(item)):  # cpu-route: Python label objects, the explicit label input step
            _flatten_into(item[k], out, b)
        return
    _ = py_call(out.append, item)


def _label_kind(t: PythonObject, b: PythonObject, context: PythonObject) raises -> Int:
    """`_labels._kind_of_type`: 1 str, 2 number, 0 refused."""
    if Bool(py=py_call(b.issubclass, t, b.str)):
        return 1
    if Bool(py=py_call(b.issubclass, t, context["trees_real_type"])) or String(t.__name__) == "bool_":
        return 2
    return 0


def _sorted_classes(labels: PythonObject, context: PythonObject) raises -> TreesLabels:
    """`_labels.sorted_classes`: the ORDER RULE over Python label objects
    (numbers by value, strings by code point, the first of equal labels
    kept), its refusals in walk order, one dense code per label."""
    var b = trees_builtins()
    var n = len(labels)
    if n == 0:
        raise Error("mojolearn: y is empty")
    var kinds = Python.dict()
    var seen_str = False
    var seen_number = False
    for i in range(n):  # cpu-route: Python label objects, the explicit label input step
        var item = labels[i]
        var t = py_type(item)
        var kind_obj = py_get(kinds, t)
        var kind = 0
        if kind_obj is None:
            kind = _label_kind(t, b, context)
            py_setitem(kinds, t, PythonObject(kind))
        else:
            kind = Int(py=kind_obj)
        if kind == 0:
            raise Error("mojolearn: label " + String(py_call(b.repr, item)) + " is neither a number nor a str")
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
    var codes = trees_vector(n, "<i4", context)
    var p = _I32P(unsafe_from_address=trees_address(codes))
    for i in range(n):  # cpu-route: Python label objects, the explicit label input step
        p[i] = Int32(Int(py=index[labels[i]]))
    return TreesLabels(classes, codes, n)


def _numeric_label_view(y: PythonObject, context: PythonObject, mut bool_source: Bool) raises -> Optional[TreesView]:
    """The numeric label vector `_labels._encode_labels_native` encodes, or
    None for what only the label input step reads (lists, str or object
    buffers, a pandas Series, a matrix, an empty vector)."""
    var b = trees_builtins()
    var cls = py_type(y)
    if cls is b.list or cls is b.tuple or cls is b.str or cls is b.bytes or native_pandas_exact(y):
        return None
    var view: TreesView
    try:
        if cls is context["array_class"]:
            view = _view_of_array(y)
        else:
            var probe = native_buffer_view(y)
            bool_source = String(probe.format) == "?"
            view = _view_of_buffer(y, "y", context)
    except:
        return None
    var n = view.size()
    var big = 0
    for k in range(view.ndim()):  # driver-loop: over array rank metadata
        if view.shape[k] > big:
            big = view.shape[k]
    if _encode_code(view.typestr) < 0 or n == 0 or n != big:
        return None
    return view^


def _list_labels(y: PythonObject, context: PythonObject) raises -> Optional[TreesLabels]:
    """`_labels._encode_label_list_native`: a flat list or tuple whose every
    label is exactly an int, or exactly a float, through the native encoder;
    None for anything else (and for an int outside int64)."""
    var b = trees_builtins()
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
        store = py_call(context["array_type"], "q" if all_int else "d", y)
    except:
        return None  # an int outside int64: a class only Python can hold
    var array = classical_wrap_array(store, Python.tuple(n), "<i8" if all_int else "<f8", "C", context)
    return _native_labels(_view_of_array(array), False, context)


def trees_encode_labels(y: PythonObject, context: PythonObject) raises -> TreesLabels:
    """`_labels.encode_labels(y)`: one native `prepare_targets` call for a
    numeric label vector (a buffer or a flat list of exact ints or exact
    floats); the ORDER RULE over Python objects for str and object labels."""
    var b = trees_builtins()
    var bool_source = False
    var numeric = _numeric_label_view(y, context, bool_source)
    if numeric:
        return _native_labels(numeric.value(), bool_source, context)
    var flat = _flatten_labels(y, context)
    var listed = _list_labels(flat, context)
    if listed:
        return listed.value().copy()
    return _sorted_classes(flat, context)


def trees_label_kind(classes: PythonObject, context: PythonObject) raises -> String:
    """`_labels.label_kind`: 'int', 'float' or 'object'."""
    var b = trees_builtins()
    var all_int = True
    var all_real = True
    for i in range(len(classes)):  # driver-loop: over kind test of the class labels (classes-sized: distinct classes)
        var c = classes[i]
        var t = py_type(c)
        var is_bool = t is b.bool or String(t.__name__) == "bool_"
        var integral = Bool(py=py_call(b.isinstance, c, context["trees_integral_type"])) and not is_bool
        var real = Bool(py=py_call(b.isinstance, c, context["trees_real_type"])) and not is_bool
        all_int = all_int and integral
        all_real = all_real and real
    if all_int:
        return "int"
    if all_real:
        return "float"
    return "object"


def trees_decode_labels(classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.decode_labels(classes, codes)` for int64 or int32 codes:
    integer classes an int64 Array, real classes a float64 Array (the base
    binding's gathers), str or bool classes a Python list."""
    var b = trees_builtins()
    var kind = trees_label_kind(classes, context)
    var n = Int(py=codes.size)
    var k = len(classes)
    var wide_codes = codes
    if String(codes.dtype) == "<i4":
        wide_codes = trees_vector(n, "<i8", context)
        trees_cast(trees_address(codes), "<i4", trees_address(wide_codes), "<i8", n, context)
    if (kind == "int" or kind == "float") and k > 0:
        var table = trees_vector(k, "<i8" if kind == "int" else "<f8", context)
        var fits = True
        for i in range(k):  # driver-loop: over the k class values into the gather table
            if kind == "int":
                var v = classes[i]
                if Bool(py=v < PythonObject(-9223372036854775808)) or Bool(py=v > PythonObject(9223372036854775807)):
                    fits = False
                    break
                _I64P(unsafe_from_address=trees_address(table))[i] = Int64(Int(py=py_call(b.int, v)))
            else:
                _F64P(unsafe_from_address=trees_address(table))[i] = Float64(py=py_call(b.float, classes[i]))
        if fits:
            var out = trees_vector(n, "<i8" if kind == "int" else "<f8", context)
            var fn = trees_helper(context, "gather_i64" if kind == "int" else "gather_f64")
            _ = py_call_typed(fn, PythonObject(trees_address(table)), PythonObject(k),
                              PythonObject(trees_address(wide_codes)), PythonObject(n), PythonObject(trees_address(out)))
            return out
    var values = Python.list()
    var p = _I64P(unsafe_from_address=trees_address(wide_codes))
    for i in range(n):  # cpu-route: str or bool classes decode to Python objects, the explicit label output step
        _ = py_call(values.append, classes[Int(p[i])])
    return values


def trees_argmax_rows(scores: PythonObject, rows: Int, cols: Int, wide: Bool, context: PythonObject) raises -> PythonObject:
    """`_labels.argmax_rows` of a C-order float32 or float64 block: the
    base binding's first-max scan, int64 codes."""
    var out = trees_vector(rows, "<i8", context)
    if rows == 0:
        return out
    if cols == 0:
        raise Error("argmax_rows: scores have no columns")
    var fn = trees_helper(context, "argmax_rows_f64" if wide else "argmax_rows_f32")
    _ = py_call_typed(fn, PythonObject(trees_address(scores)), PythonObject(rows), PythonObject(cols),
                      PythonObject(trees_address(out)))
    return out


# ---------------------------------------------------------- class weights


def trees_class_weight_rows(class_weight: PythonObject, labels: TreesLabels, xtrees: PythonObject,
                            context: PythonObject) raises -> PythonObject:
    """`_class_weights._class_weight_rows(class_weight, classes, codes,
    xbind)` with int32 codes on the x_trees binding: per-row float32
    weights from a dict or 'balanced', or None when every class weight is
    1. The counts and the per-row gather are the binding's kernels; only
    the k class weights are read here."""
    var b = trees_builtins()
    if class_weight is None:
        return PythonObject(None)
    var classes = labels.classes
    var k = len(classes)
    var n = labels.n
    var objects = Python.list()
    if Bool(py=py_call(b.isinstance, class_weight, b.str)):
        var counts = trees_vector(k, "<i4", context)
        var count_entry = trees_need(xtrees, "x_trees_code_counts", "the x_trees binding has no x_trees_code_counts",
                                     "ImportError")
        _ = py_call_typed(count_entry, PythonObject(trees_address(labels.codes) if n else 0),
                          PythonObject(trees_address(counts)), Python.list(n, k))
        var wide = trees_vector(k, "<i8", context)
        trees_cast(trees_address(counts), "<i4", trees_address(wide), "<i8", k, context)
        var w64 = trees_vector(k, "<f8", context)
        if k and Int(py=py_call_typed(trees_helper(context, "class_ratio_f64"), PythonObject(trees_address(wide)),
                                      PythonObject(k), PythonObject(n), PythonObject(0), PythonObject(trees_address(w64)))):
            raise Error("ZeroDivisionError: division by zero")
        var p = _F64P(unsafe_from_address=trees_address(w64))
        for i in range(k):  # driver-loop: over the k class weights for the argument checks below
            _ = py_call(objects.append, PythonObject(p[i]))
    else:
        var unknown = py_call(py_call(b.set, class_weight).difference, classes)
        if len(unknown) > 0:
            raise Error("class_weight contains labels absent from y")
        for i in range(k):  # driver-loop: over user class_weight dict per class
            _ = py_call(objects.append, py_call(class_weight.get, classes[i], PythonObject(1.0)))
    var values = List[Float64]()
    var all_one = True
    for i in range(k):  # driver-loop: over class weight argument checks
        var v = objects[i]
        var t = py_type(v)
        if t is b.bool or String(t.__name__) == "bool_" or not Bool(py=py_call(b.isinstance, v, context["trees_real_type"])):
            raise Error("class weights must be finite nonnegative real numbers")
        var f = Float64(py=py_call(b.float, v))
        if not isfinite(f) or f < 0:
            raise Error("class weights must be finite nonnegative real numbers")
        values.append(f)
        all_one = all_one and Bool(py=v == PythonObject(1))
    if all_one:
        return PythonObject(None)
    var narrowed = trees_vector(k, "<f4", context)
    var q = _F32P(unsafe_from_address=trees_address(narrowed))
    var top = Float32(0)
    for i in range(k):  # driver-loop: over the k class weights narrowed to float32
        var f32 = Float32(values[i])
        if not isfinite(f32) or (values[i] > 0 and f32 == 0):
            raise Error("class weights must remain finite and nonzero when positive in Float32")
        q[i] = f32
        if i == 0 or f32 > top:
            top = f32
    if top <= 0:
        raise Error("class weights must have positive total")
    var out = trees_vector(n, "<f4", context)
    if n:
        var rows_entry = trees_need(xtrees, "x_trees_class_rows", "the x_trees binding has no x_trees_class_rows",
                                    "ImportError")
        _ = py_call_typed(rows_entry, PythonObject(trees_address(labels.codes)), PythonObject(trees_address(narrowed)),
                          PythonObject(trees_address(out)), Python.list(n, k))
    return out


def trees_weighted_rows(sample_weight: PythonObject, class_weight: PythonObject, labels: TreesLabels,
                        xtrees: PythonObject, context: PythonObject) raises -> TreesMatrix:
    """`_expansion_trees._trees_weighted_rows`: the checked sample weights
    times the class_weight row weights, one correctly rounded float32
    product per row (`x_trees_mul_f32`); the sample weights alone when no
    class weight differs from 1."""
    var sw = trees_check_sample_weight(sample_weight, labels.n, context)
    if class_weight is None:
        return sw^
    var rows = trees_class_weight_rows(class_weight, labels, xtrees, context)
    if rows is None:
        return sw^
    var out = trees_vector(labels.n, "<f4", context)
    var mul = trees_need(xtrees, "x_trees_mul_f32", "the x_trees binding has no x_trees_mul_f32", "ImportError")
    _ = py_call_typed(mul, PythonObject(sw.addr), PythonObject(trees_address(rows)), PythonObject(trees_address(out)),
                      Python.list(labels.n))
    _ = sw^
    return TreesMatrix(out, trees_address(out), labels.n, 1, True)


def trees_class_rows_only(class_weight: PythonObject, labels: TreesLabels, xtrees: PythonObject,
                          context: PythonObject) raises -> Optional[TreesMatrix]:
    """The class_weight row weights without sample weights, or None."""
    var rows = trees_class_weight_rows(class_weight, labels, xtrees, context)
    if rows is None:
        return None
    return TreesMatrix(rows, trees_address(rows), labels.n, 1, True)
