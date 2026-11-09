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
from std.ffi import _Global, c_int
from std.math import isfinite
from std.memory import bitcast
from std.os import getenv
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_setattr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_attr import py_attr
from std.python._cpython import PyObjectPtr
from bindings.py_typed_call import py_call_typed, typed_python_error
from bindings.cpython_table import cpython_table
from bindings.native_buffer_owner import native_buffer_view
from bindings.classical_arrays import classical_wrap_array, _store_code, _buffer_typestr
from bindings.native_array_storage import native_array_zeros, native_scratch_bytes, native_scratch_address
from bindings.classical_pandas import native_pandas_exact, native_pandas_f32, _pd_fields, _pd_get
from bindings.sequential_admission import _ordinary_torch, _one_of
from bindings.native_torch_admission import native_torch_numpy
from bindings.sequential_inputs import sequential_sequence
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.metrics_native_support import metric_is_bool, metric_is_integral, metric_is_real, metric_float
from bindings.classical_labels import (
    ClassicalLabels,
    _encode_label_view as classical_encode_label_view,
    _native_labels as classical_native_labels,
    _flat_list_labels as classical_flat_list_labels,
    _text_buffer_labels as classical_text_buffer_labels,
    _text_list_labels as classical_text_list_labels,
)
# The array admission helpers (lane consol-a0-admission, 2026-11-08): this
# file kept a near-verbatim copy of bindings/classical_labels.mojo's block;
# the one merged copy lives there and is imported here under the trees_*
# names its callers use.
# TOMBSTONE: trees_public_inputs array helpers copy (duplicate of classical_labels) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch.
from bindings.classical_labels import (
    labels_itemsize as trees_itemsize,
    labels_empty as trees_empty,
    labels_vector as trees_vector,
    labels_full_f32 as trees_full_f32,
    labels_reshape as trees_reshape,
    labels_freeze as trees_freeze,
    labels_address as trees_address,
    LabelsView as TreesView,
    labels_shape_repr as trees_shape_repr,
    _view_of_array,
    _any_contiguous_address,
    _view_of_buffer,
    _pandas_values,
    labels_view as trees_view,
    labels_check_rank as trees_check_rank,
    _native_code,
    labels_cast as trees_cast,
    LabelsMatrix as TreesMatrix,
    labels_f32 as trees_f32,
    labels_f32_view as trees_f32_view,
    labels_forest_layout as trees_forest_layout,
    labels_all_finite as trees_all_finite,
    labels_check_sample_weight as trees_check_sample_weight,
    labels_prepare_weights as trees_prepare_weights,
)
from bindings.label_input_sources import labels_storage_source
from bindings.perf3_label_doors_text import labels_all_exact_str

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]

#: The forests' refusal of a non-finite X (randomforest.py / extratrees.py
#: `_NONFINITE_REFUSAL`).
comptime TREES_NONFINITE = "X contains NaN or infinity; the forest has no missing-value arm"


# ------------------------------------------------------------ connection


def trees_builtins() raises -> PythonObject:
    # The admitted builtins module (one stamp compare per call) instead of
    # `Python.import_module`, which ran the import machinery's `__import__`
    # on every call (lane perf-sweep-trees, 2026-11-05: dozens of calls per
    # public predict).
    return native_loaded_module("builtins")


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


def trees_call(entry: PythonObject, args: PythonObject) raises -> PythonObject:
    """`entry(*args)` for a compiled entry point and a Python list of its
    arguments, keeping the class of a refusal the entry raises."""
    ref cpy = Python().cpython()
    var envelope = py_call(trees_builtins().tuple, args)
    # The interpreter's own PyObject_Call entry (no by-name symbol lookup per
    # call, p3-trees); the same call as PyObject_CallObject with a tuple.
    var result = cpy.PyObject_Call(entry._obj_ptr, envelope._obj_ptr, PyObjectPtr())
    if not result:
        raise typed_python_error()
    return PythonObject(from_owned=result)


def trees_env(name: String, default: String) -> String:
    var value = getenv(name)
    return value if value else default


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
    var b = trees_builtins()
    var classes = Python.list()
    for i in range(k):  # driver-loop: over the k distinct classes
        if wide_float:
            _ = py_call(classes.append, PythonObject(_F64P(unsafe_from_address=trees_address(classes_store))[i]))
            _ = classes_store  # keep the Python object alive through the native call above
        else:
            var v = Int(_I64P(unsafe_from_address=trees_address(classes_store))[i])
            _ = classes_store  # keep the Python object alive through the native call above
            if bool_source:
                _ = py_call(classes.append, PythonObject(v != 0))
                _ = classes_store  # keep the Python object alive through the native call above
            else:
                _ = py_call(classes.append, PythonObject(v))
                _ = classes_store  # keep the Python object alive through the native call above
    var _keep_ret_classes_store = TreesLabels(classes, codes, n)
    _ = classes_store  # keep the Python object alive through the native call above
    return _keep_ret_classes_store^


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


def _flatten_into(item: PythonObject, out_buf: PythonObject, b: PythonObject) raises:
    var cls = py_type(item)
    if Bool(py=py_call(b.isinstance, item, Python.tuple(b.list, b.tuple))):
        for k in range(len(item)):  # cpu-route: Python label objects, the explicit label input step
            _flatten_into(item[k], out_buf, b)
        return
    _ = py_call(out_buf.append, item)


def _label_kind(t: PythonObject, b: PythonObject, context: PythonObject) raises -> Int:
    """`_labels._kind_of_type`: 1 str, 2 number, 0 refused."""
    if t is b.str:
        return 1
    if t is b.int or t is b.float or t is b.bool or _one_of(t, context["scalar_types"]):
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
        if py_type(t) is not b.type:
            raise Error("TypeError: native tree labels require ordinary numeric or string storage without Python hooks")
        var kind_obj = py_get(kinds, t)
        var kind = 0
        if kind_obj is None:
            kind = _label_kind(t, b, context)
            py_setitem(kinds, t, PythonObject(kind))
        else:
            kind = Int(py=kind_obj)
        if kind == 0:
            raise Error("mojolearn: labels require native numbers or strings without Python hooks")
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
    _ = codes  # keep the Python object alive through the native call above
    for i in range(n):  # cpu-route: Python label objects, the explicit label input step
        p[i] = Int32(Int(py=index[labels[i]]))
        _ = codes  # keep the Python object alive through the native call above
    return TreesLabels(classes, codes, n)


def _numeric_label_view(y: PythonObject, context: PythonObject, mut bool_source: Bool) raises -> Optional[TreesView]:
    """The numeric label vector `_labels._encode_labels_native` encodes, or
    None for what only the label input step reads (lists, str or object
    buffers, a pandas Series, a matrix, an empty vector)."""
    var b = trees_builtins()
    var cls = py_type(y)
    if cls is b.list or cls is b.tuple or cls is b.str or cls is b.bytes:
        return None
    if native_pandas_exact(y):
        # An exact pandas Series over one plain NumPy integer, unsigned,
        # bool or float block (lane perf2-trees, 2026-11-06): its `tolist()`
        # lists exactly that block's values as Python ints, bools or floats,
        # which the label input step then encoded through the same native
        # `prepare_targets` (ints, floats) or sorted to the same bool classes
        # (`bool_source`). The block is read in place instead of listed and
        # walked per label. Nullable, categorical, string, object and any
        # other storage keep the walk. Same classes and codes: no bits change.
        var values = PythonObject(None)
        try:
            values = _pandas_values(y)
        except:
            return None
        var ndarray = py_get(context, "ndarray")
        if ndarray is None or py_type(values) is not ndarray or Int(py=values.ndim) != 1:
            return None
        var kind = String(values.dtype.kind)
        if not (kind == "i" or kind == "u" or kind == "b" or kind == "f"):
            return None
        return _numeric_label_view(values, context, bool_source)
    var found: Optional[TreesView] = None
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
    # perf3-label-doors (2026-11-06): the classical encoder's one-read paths
    # (bindings/classical_labels.mojo, perf2-labels), each giving exactly the
    # classes and codes the walk below gives the same input, so no bits
    # change. A numeric ndarray / Array / numpy-backed Series this copy's view
    # does not take (a C-ordered matrix, a bool or one-byte '|' Array,
    # int8 / uint16 / float16 storage) is encoded from its buffer; only
    # numeric storage is handed over, so the classical admission it runs
    # first meets nothing this encoder would refuse differently.
    var stored = labels_storage_source(y, context)
    if stored is not None:
        var kind: String
        if py_type(stored) is py_get(context, "ndarray"):
            kind = String(stored.dtype.kind)
        else:
            var typestr = String(stored.dtype)
            kind = String(typestr[byte=1:2])
        if kind == "i" or kind == "u" or kind == "b" or kind == "f":
            var wide_bool = False
            var wide = classical_encode_label_view(y, context, wide_bool)
            if wide:
                return _trees_of(classical_native_labels(wide.value(), wide_bool, context))
    # An exact flat list / tuple of exact ints, floats or bools: one native
    # encode from the container's C storage (no flatten, no type walk).
    var direct = classical_flat_list_labels(y, context)
    if direct:
        return _trees_of(direct.value())
    # An exact '<U' ndarray: UCS4 read from its buffer, native stable sort.
    var text = classical_text_buffer_labels(y, context)
    if text:
        return _trees_of(text.value())
    var flat = _flatten_labels(y, context)
    var listed = _list_labels(flat, context)
    if listed:
        return listed.value().copy()
    # A flat list of exact str labels only (this copy's ORDER RULE refuses
    # str subclasses, which the native str reader would take): code points
    # read natively and sorted by the stable merge sort, the same classes
    # (first of each run) and codes as the ORDER RULE.
    if labels_all_exact_str(flat):
        var named = classical_text_list_labels(flat, context)
        if named:
            return _trees_of(named.value())
    return _sorted_classes(flat, context)


def _trees_of(labels: ClassicalLabels) -> TreesLabels:
    return TreesLabels(labels.classes, labels.codes, labels.n)


def trees_label_kind(var classes: PythonObject, context: PythonObject) raises -> String:
    """`_labels.label_kind`: 'int', 'float' or 'object'."""
    classes = classifier_label_list(classes, context)
    var all_int = True
    var all_real = True
    for i in range(len(classes)):  # driver-loop: over kind test of the class labels (classes-sized: distinct classes)
        var c = classes[i]
        var is_bool = metric_is_bool(c, context)
        var integral = metric_is_integral(c, context) and not is_bool
        var real = metric_is_real(c, context) and not is_bool
        all_int = all_int and integral
        all_real = all_real and real
    if all_int:
        return "int"
    if all_real:
        return "float"
    return "object"


# ------------------------------------------------- decode tables, per classes_
#
# Every classifier predict decoded through `classes_` re-derived the label
# kind (three type tests per class) and rebuilt the int64/float64 gather table
# (one `int()`/`float()` and two range compares per class). Both depend only
# on the class objects, so they are kept for the last few `classes_` seen
# (lane perf2-trees, 2026-11-06). An entry is reused only while the caller's
# classes object is the recorded one AND its listed items are the recorded
# objects, one by one (identity only: no user `__eq__` runs), so an in-place
# edit of `classes_`, a refit or a replaced attribute rebuilds it. The table
# is read-only (the gather's source); entries hold strong references to at
# most _DECODE_SLOTS small class lists.


comptime _DECODE_SLOTS = 8


struct _DecodeTables(Defaultable, Movable):
    var owners: List[PythonObject]
    var items: List[PythonObject]
    var kinds: List[Int]
    var tables: List[PythonObject]
    var next: Int

    def __init__(out self):
        self.owners = List[PythonObject]()
        self.items = List[PythonObject]()
        self.kinds = List[Int]()
        self.tables = List[PythonObject]()
        self.next = 0


comptime _DECODE = _Global[StorageType=_DecodeTables, name="MojolearnTreesDecodeTablesV1",
                           init_fn=_DecodeTables.__init__]


def _decode_slot(owner: PythonObject, classes: PythonObject) raises -> Int:
    """The cache slot recorded for `owner` whose listed classes are still
    the same objects, or -1."""
    var g = _DECODE.get_or_create_ptr()
    for s in range(len(g[].owners)):  # small-loop(_DECODE_SLOTS): at most eight recorded classes_
        if not (g[].owners[s] is owner):
            continue
        var held = g[].items[s]
        if len(held) != len(classes):
            return -1
        for i in range(len(classes)):  # driver-loop: over the k class objects (identity only)
            if not (held[i] is classes[i]):
                return -1
        return s
    return -1


def _decode_record(owner: PythonObject, classes: PythonObject, kind: Int, table: PythonObject) raises:
    """Record `kind` (0 int, 1 float, 2 object) and the gather table."""
    var g = _DECODE.get_or_create_ptr()
    var copy = py_call(trees_builtins().list, classes)
    for s in range(len(g[].owners)):  # small-loop(_DECODE_SLOTS): replace this owner's stale entry
        if g[].owners[s] is owner:
            g[].items[s] = copy
            g[].kinds[s] = kind
            g[].tables[s] = table
            return
    if len(g[].owners) < _DECODE_SLOTS:
        g[].owners.append(owner)
        g[].items.append(copy)
        g[].kinds.append(kind)
        g[].tables.append(table)
        return
    var s = g[].next
    g[].next = (s + 1) % _DECODE_SLOTS
    # Hold the displaced objects until the slot is rewritten, so no
    # destructor runs while the cache is being changed.
    var old_owner = g[].owners[s]
    var old_items = g[].items[s]
    var old_table = g[].tables[s]
    g[].owners[s] = owner
    g[].items[s] = copy
    g[].kinds[s] = kind
    g[].tables[s] = table
    _ = old_owner^
    _ = old_items^
    _ = old_table^


def trees_decode_labels(var classes: PythonObject, codes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_labels.decode_labels(classes, codes)` for int64 or int32 codes:
    integer classes an int64 Array, real classes a float64 Array (the base
    binding's gathers), str or bool classes a Python list."""
    var b = trees_builtins()
    var owner = classes
    # ClassLabels inherits list but its __getitem__ is Python. Copy its
    # verified native list storage once, preserving the caller's label objects.
    classes = classifier_label_list(classes, context)
    var slot = _decode_slot(owner, classes)
    var kind: String
    var cached_table = PythonObject(None)
    if slot >= 0:
        var g = _DECODE.get_or_create_ptr()
        var code = g[].kinds[slot]
        kind = String("int") if code == 0 else (String("float") if code == 1 else String("object"))
        cached_table = g[].tables[slot]
    else:
        kind = trees_label_kind(classes, context)
    var n = Int(py=codes.size)
    var k = len(classes)
    var wide_codes = codes
    if String(codes.dtype) == "<i4":
        wide_codes = trees_vector(n, "<i8", context)
        trees_cast(trees_address(codes), "<i4", trees_address(wide_codes), "<i8", n, context)
        _ = wide_codes  # keep the Python object alive through the native call above
    if slot < 0 and not ((kind == "int" or kind == "float") and k > 0):
        _decode_record(owner, classes, 2 if kind != "int" and kind != "float" else (0 if kind == "int" else 1),
                       PythonObject(None))
    if (kind == "int" or kind == "float") and k > 0 and (slot < 0 or cached_table is not None):
        var table = cached_table
        var fits = True
        if slot < 0:
            table = trees_vector(k, "<i8" if kind == "int" else "<f8", context)
            for i in range(k):  # driver-loop: over the k class values into the gather table
                if kind == "int":
                    var v = classes[i]
                    if Bool(py=v < PythonObject(-9223372036854775808)) or Bool(py=v > PythonObject(9223372036854775807)):
                        fits = False
                        break
                    _I64P(unsafe_from_address=trees_address(table))[i] = Int64(Int(py=py_call(b.int, v)))
                    _ = table  # keep the Python object alive through the native call above
                else:
                    _F64P(unsafe_from_address=trees_address(table))[i] = Float64(py=py_call(b.float, classes[i]))
                    _ = table  # keep the Python object alive through the native call above
            _decode_record(owner, classes, 0 if kind == "int" else 1, table if fits else PythonObject(None))
        if fits:
            var out = trees_vector(n, "<i8" if kind == "int" else "<f8", context)
            var fn_obj = trees_helper(context, "gather_i64" if kind == "int" else "gather_f64")
            _ = py_call_typed(fn_obj, PythonObject(trees_address(table)), PythonObject(k),
                              PythonObject(trees_address(wide_codes)), PythonObject(n), PythonObject(trees_address(out)))
            _ = table  # keep the Python object alive through the native call above
            _ = out  # keep the Python object alive through the native call above
            _ = wide_codes  # keep the Python object alive through the native call above
            return out
    var p = _I64P(unsafe_from_address=trees_address(wide_codes))
    _ = wide_codes  # keep the Python object alive through the native call above
    # The k class objects read once (lane perf-sweep-trees, 2026-11-05). A
    # code indexes as the Python list did: a negative code counts from the
    # end, any other out-of-range code is refused. p3-trees: the output is
    # one PyList_New(n) filled slot by slot (PyList_SetItem with a new
    # reference), not one bound `append` call per row.
    var labels = List[PythonObject]()
    for c in range(k):  # driver-loop: over the k class objects
        labels.append(classes[c])
    ref cpy = Python().cpython()
    var raw = cpy.lib.get_function[PyObjectPtr]("PyList_New")(n)
    if not raw:
        raise cpy.unsafe_get_error()
    # Owns the NULL-filled list: a refusal below releases the filled slots.
    var values = PythonObject(from_owned=raw)
    var table = cpython_table()
    for i in range(n):  # cpu-route: str or bool classes decode to Python objects, the explicit label output step
        var code = Int(p[i])
        if code < 0:
            code += k
        if code < 0 or code >= k:
            raise Error("IndexError: list index out of range")
        # SetItem steals the new reference on success and failure alike.
        if table[].PyList_SetItem(values._obj_ptr, i, cpy.Py_NewRef(labels[code]._obj_ptr)) != 0:
            raise cpy.unsafe_get_error()
    _ = wide_codes  # keep the Python object alive through the reads above
    _ = labels^
    return values


def trees_argmax_rows(scores: PythonObject, rows: Int, cols: Int, wide: Bool, context: PythonObject) raises -> PythonObject:
    """`_labels.argmax_rows` of a C-order float32 or float64 block: the
    base binding's first-max scan, int64 codes."""
    var out = trees_vector(rows, "<i8", context)
    if rows == 0:
        return out
    if cols == 0:
        raise Error("argmax_rows: scores have no columns")
    var fn_obj = trees_helper(context, "argmax_rows_f64" if wide else "argmax_rows_f32")
    _ = py_call_typed(fn_obj, PythonObject(trees_address(scores)), PythonObject(rows), PythonObject(cols),
                      PythonObject(trees_address(out)))
    _ = out  # keep the Python object alive through the native call above
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
        _ = counts  # keep the Python object alive through the native call above
        var wide = trees_vector(k, "<i8", context)
        trees_cast(trees_address(counts), "<i4", trees_address(wide), "<i8", k, context)
        _ = wide  # keep the Python object alive through the native call above
        _ = counts  # keep the Python object alive through the native call above
        var w64 = trees_vector(k, "<f8", context)
        if k and Int(py=py_call_typed(trees_helper(context, "class_ratio_f64"), PythonObject(trees_address(wide)),
                                      PythonObject(k), PythonObject(n), PythonObject(0), PythonObject(trees_address(w64)))):
            raise Error("ZeroDivisionError: division by zero")
        _ = wide  # keep the Python object alive through the native calls above
        _ = w64  # keep the Python object alive through the native calls above
        var p = _F64P(unsafe_from_address=trees_address(w64))
        _ = w64  # keep the Python object alive through the native call above
        for i in range(k):  # driver-loop: over the k class weights for the argument checks below
            _ = py_call(objects.append, PythonObject(p[i]))
            _ = w64  # keep the Python object alive through the native call above
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
        if metric_is_bool(v, context) or not metric_is_real(v, context):
            raise Error("class weights must be finite nonnegative real numbers")
        var f = metric_float(v, context)
        if not isfinite(f) or f < 0:
            raise Error("class weights must be finite nonnegative real numbers")
        values.append(f)
        all_one = all_one and Bool(py=v == PythonObject(1))
    if all_one:
        return PythonObject(None)
    var narrowed = trees_vector(k, "<f4", context)
    var q = _F32P(unsafe_from_address=trees_address(narrowed))
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
    var out = trees_vector(n, "<f4", context)
    if n:
        var rows_entry = trees_need(xtrees, "x_trees_class_rows", "the x_trees binding has no x_trees_class_rows",
                                    "ImportError")
        _ = py_call_typed(rows_entry, PythonObject(trees_address(labels.codes)), PythonObject(trees_address(narrowed)),
                          PythonObject(trees_address(out)), Python.list(n, k))
        _ = narrowed  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
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
    _ = out  # keep the Python object alive through the native call above
    _ = rows  # keep the Python object alive through the native call above
    _ = sw^
    var _keep_ret_out = TreesMatrix(out, trees_address(out), labels.n, 1, True)
    _ = out  # keep the Python object alive through the native call above
    return _keep_ret_out^


def trees_class_rows_only(class_weight: PythonObject, labels: TreesLabels, xtrees: PythonObject,
                          context: PythonObject) raises -> Optional[TreesMatrix]:
    """The class_weight row weights without sample weights, or None."""
    var rows = trees_class_weight_rows(class_weight, labels, xtrees, context)
    if rows is None:
        return None
    var _keep_ret_rows = TreesMatrix(rows, trees_address(rows), labels.n, 1, True)
    _ = rows  # keep the Python object alive through the native call above
    return _keep_ret_rows^
