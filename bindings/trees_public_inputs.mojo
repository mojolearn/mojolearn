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
from bindings.classical_labels import ClassicalLabels, classical_encode_labels
# The array admission helpers (lane consol-a0-admission, 2026-11-08): this
# file kept a near-verbatim copy of bindings/classical_labels.mojo's block;
# the one merged copy lives there and is imported here under the trees_*
# names its callers use.
# TOMBSTONE: trees_public_inputs array helpers copy (duplicate of classical_labels) deleted 2026-11-08 by lane consol-a0-admission; code recoverable at ef4bce3a7e.
# Restore: experiments/removed/A0_ADMISSION_HELPERS.patch.
from bindings.classical_labels import (
    labels_itemsize as trees_itemsize,
    labels_full_f32 as trees_full_f32,
    labels_reshape as trees_reshape,
    labels_freeze as trees_freeze,
    labels_address as trees_address,
    _pandas_values,
    _native_code,
    labels_cast as trees_cast,
)
# The X / weight / view admission is the one path (bindings/admitted_input.mojo).
# TOMBSTONE: trees_empty / trees_vector / TreesView / trees_view / trees_check_rank / trees_shape_repr / TreesMatrix / trees_f32 / trees_f32_view / trees_forest_layout / trees_all_finite / trees_check_sample_weight / trees_prepare_weights aliases, and the trees label encoder copy (_native_labels, _label_list, _flatten_labels, _flatten_into, _label_kind, _sorted_classes, _numeric_label_view, _list_labels) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from bindings.admitted_input import (
    AdmittedX, admit_f32, admit_x_unrecorded, admit_weight_words, admitted_words, admitted_array, admitted_vector,
    admit_c_view, admitted_dtype,
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


def trees_encode_labels(y: PythonObject, context: PythonObject) raises -> TreesLabels:
    """`_labels.encode_labels(y)`: THE one label encoder
    (bindings/classical_labels.mojo classical_encode_labels; the trees
    context carries the label helpers through label_context())."""
    # Lane a1b-doors-opt-in-trees-linear: this was a second encoder (its own
    # numeric view, flatten, ORDER RULE and list paths, then the classical
    # one-read paths). The classical encoder gives the same classes and
    # codes for every input it takes; a str subclass label, which this copy
    # refused, is now encoded as the classical ORDER RULE encodes it.
    return _trees_of(classical_encode_labels(y, context))


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
        wide_codes = admitted_vector(n, "<i8", context)
        trees_cast(trees_address(codes), "<i4", trees_address(wide_codes), "<i8", n, context)
        _ = wide_codes  # keep the Python object alive through the native call above
    if slot < 0 and not ((kind == "int" or kind == "float") and k > 0):
        _decode_record(owner, classes, 2 if kind != "int" and kind != "float" else (0 if kind == "int" else 1),
                       PythonObject(None))
    if (kind == "int" or kind == "float") and k > 0 and (slot < 0 or cached_table is not None):
        var table = cached_table
        var fits = True
        if slot < 0:
            table = admitted_vector(k, "<i8" if kind == "int" else "<f8", context)
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
            var out = admitted_vector(n, "<i8" if kind == "int" else "<f8", context)
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
    var out = admitted_vector(rows, "<i8", context)
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
        var counts = admitted_vector(k, "<i4", context)
        var count_entry = trees_need(xtrees, "x_trees_code_counts", "the x_trees binding has no x_trees_code_counts",
                                     "ImportError")
        _ = py_call_typed(count_entry, PythonObject(trees_address(labels.codes) if n else 0),
                          PythonObject(trees_address(counts)), Python.list(n, k))
        _ = counts  # keep the Python object alive through the native call above
        var wide = admitted_vector(k, "<i8", context)
        trees_cast(trees_address(counts), "<i4", trees_address(wide), "<i8", k, context)
        _ = wide  # keep the Python object alive through the native call above
        _ = counts  # keep the Python object alive through the native call above
        var w64 = admitted_vector(k, "<f8", context)
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
    var narrowed = admitted_vector(k, "<f4", context)
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
    var out = admitted_vector(n, "<f4", context)
    if n:
        var rows_entry = trees_need(xtrees, "x_trees_class_rows", "the x_trees binding has no x_trees_class_rows",
                                    "ImportError")
        _ = py_call_typed(rows_entry, PythonObject(trees_address(labels.codes)), PythonObject(trees_address(narrowed)),
                          PythonObject(trees_address(out)), Python.list(n, k))
        _ = narrowed  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
    return out


def trees_weighted_rows(sample_weight: PythonObject, class_weight: PythonObject, labels: TreesLabels,
                        xtrees: PythonObject, context: PythonObject) raises -> AdmittedX:
    """`_expansion_trees._trees_weighted_rows`: the checked sample weights
    times the class_weight row weights, one correctly rounded float32
    product per row (`x_trees_mul_f32`); the sample weights alone when no
    class weight differs from 1."""
    var sw = admit_weight_words(sample_weight, labels.n, context)
    if class_weight is None:
        return sw^
    var rows = trees_class_weight_rows(class_weight, labels, xtrees, context)
    if rows is None:
        return sw^
    var out = admitted_vector(labels.n, "<f4", context)
    var mul = trees_need(xtrees, "x_trees_mul_f32", "the x_trees binding has no x_trees_mul_f32", "ImportError")
    _ = py_call_typed(mul, PythonObject(sw.address), PythonObject(trees_address(rows)), PythonObject(trees_address(out)),
                      Python.list(labels.n))
    _ = out  # keep the Python object alive through the native call above
    _ = rows  # keep the Python object alive through the native call above
    _ = sw^
    return admitted_words(out, trees_address(out), labels.n, 1)


def trees_class_rows_only(class_weight: PythonObject, labels: TreesLabels, xtrees: PythonObject,
                          context: PythonObject) raises -> Optional[AdmittedX]:
    """The class_weight row weights without sample weights, or None."""
    var rows = trees_class_weight_rows(class_weight, labels, xtrees, context)
    if rows is None:
        return None
    return admitted_words(rows, trees_address(rows), labels.n, 1)
