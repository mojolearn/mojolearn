# SPDX-License-Identifier: Apache-2.0
# The tree ensembles' `score` in one native call: predict and score without
# returning to Python between the two.
#
# `trees_accuracy` is `_expansion_metrics.accuracy_fraction` (scikit-learn's
# ClassifierMixin.score): the `_pair` label checks and wording
# (`_refuse_multilabel`, `_metrics_impl._classification_encoded`, the length
# and kind refusals, `_sample_weight_f32`), the sorted label union, the
# per-row codes (`_codes`: the base binding's `gather_i32` for encoded
# integer labels), and the x_metrics COUNT program of `_cm_count` (`_Sums`:
# pair_key, the three group_sort/group_sum groupings, the weighted total
# grouping, off_diff counts; `_cm_stage` with `_wsum_sf64` when weighted)
# with the same stage words in the same order, run by the shared native
# program runner (bindings/arena_prog_native.mojo `_metrics_run`). The
# fraction is the binary64 the device epilogue wrote; a zero weight total
# raises ZeroDivisionError as the division did.
#
# `trees_r2` is `_metrics_impl.r2_score`'s default 1-D path: `_pair_1d` over
# `_as_f32_1d` (shape, float64 and finiteness refusals), `_sample_weight_f32`,
# then `r2_score` / `r2_score_weighted` on the metrics binding of the mode.
#
# The prediction is a host Array (the forests' predict writes it there), and
# the metrics programs stage their inputs from host memory; no existing entry
# reads a forest's device vote, so the prediction crosses the host once, as
# before. What is gone is the Python between predict and score and the
# re-materialization of y and the prediction.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem, py_fields
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_wrap_array
from bindings.arena_prog_native import _metrics_run
from bindings.trees_public_inputs import (
    TreesView, TreesMatrix, trees_builtins, trees_helper, trees_table, trees_entry, trees_vector,
    trees_address, trees_view, trees_f32, trees_f32_view, trees_check_rank, trees_all_finite, trees_cast,
    trees_shape_repr, _view_of_array, _view_of_buffer, _pandas_values, _flatten_labels, _native_labels,
    _encode_code,
)
from bindings.classical_pandas import native_pandas_exact, native_pandas_f32
from bindings.metrics_native_support import metric_is_integral
from bindings.trees_public_forest import (
    forest_predict, forest_predict_codes, forest_public_predict, forest_effective_mode, forest_kind,
)
from bindings.trees_public_inputs import trees_decode_labels
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.classical_labels import _text_buffer_labels as score_text_buffer_labels, _text_list_labels as score_text_list_labels
from bindings.label_input_sources import labels_list_scalar_kind, LABELS_LIST_INT
from bindings.classical_labels import classical_score_from_codes

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]

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


def trees_shape_of(value: PythonObject, context: PythonObject) raises -> List[Int]:
    """`_buffer.shape_of`: the shape of an array-like without converting
    it (an Array's slot, a buffer's shape, a list's nesting, a length)."""
    var b = trees_builtins()
    var cls = py_type(value)
    var shape = List[Int]()
    if cls is context["array_class"]:
        for k in range(Int(py=value.ndim)):  # driver-loop: over array rank metadata
            shape.append(Int(py=value.shape[k]))
        return shape^
    if native_pandas_exact(value):
        var storage = native_pandas_f32(value, context, False, True)
        for k in range(Int(py=storage.ndim)):  # driver-loop: over array rank metadata
            shape.append(Int(py=storage.shape[k]))
        return shape^
    if cls is b.list or cls is b.tuple:
        shape.append(len(value))
        if len(value) > 0:
            var first = value[0]
            if py_type(first) is b.list or py_type(first) is b.tuple or Bool(py=py_call(b.isinstance, first, Python.tuple(b.list, b.tuple))):
                shape.append(len(first))
        return shape^
    try:
        var view = py_call(b.memoryview, value)
        for k in range(Int(py=view.ndim)):  # driver-loop: over array rank metadata
            shape.append(Int(py=view.shape[k]))
        return shape^
    except:
        pass
    try:
        shape.append(Int(py=py_call(b.len, value)))
    except:
        pass
    return shape^


def _shape_text(shape: List[Int]) raises -> String:
    """The Python repr of a shape tuple."""
    var b = trees_builtins()
    var items = Python.list()
    for k in range(len(shape)):  # driver-loop: over array rank metadata
        _ = py_call(items.append, shape[k])
    return String(py_call(b.repr, py_call(b.tuple, items)))


def _buffer_format(value: PythonObject, context: PythonObject) raises -> String:
    """`linear_model._buffer_format`: the struct format code, byte order
    stripped; "" when the value exports no buffer."""
    var b = trees_builtins()
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
    var b = trees_builtins()
    if py_type(value) is b.float:
        return True
    if py_type(value) is b.list or py_type(value) is b.tuple:
        for k in range(len(value)):  # cpu-route: the leaf types of a user Python list, the explicit input step
            if _has_float_leaf(value[k]):
                return True
    return False


# ------------------------------------------------------------ r2


def _as_f32_1d(source: PythonObject, name: String, context: PythonObject) raises -> TreesMatrix:
    """`_metrics_impl._as_f32_1d`: a 1-D (or (n, 1) column) float32 vector,
    float64 refused by name, NaN or infinity refused."""
    var shape = trees_shape_of(source, context)
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
    var out = trees_f32(source, len(shape), "C", name, context)
    if not trees_all_finite(out.addr, out.rows * out.cols, False, context):
        raise Error("mojolearn metrics: " + name + " contains NaN or infinity; refused "
                    + "here (metrics/README.md HAND-OFF ask 3) rather than letting it "
                    + "reach a kernel whose answer for it is a washed sentinel")
    _ = out  # keep the Python object alive through the native calls above
    return out^


def _reduce(addr: Int, count: Int, what: Int, context: PythonObject) raises -> PythonObject:
    """`Array.min()` (0) / `max()` (1) of a float32 vector: `reduce_stat`."""
    return py_call_typed(trees_helper(context, "reduce_stat"), PythonObject(addr), PythonObject(0),
                         PythonObject(count), PythonObject(what))


def trees_sample_weight_f32(source: PythonObject, n: Int, caller: String, context: PythonObject) raises -> TreesMatrix:
    """`_metrics_impl._sample_weight_f32`: one finite, non-negative float32
    weight per sample with a positive total."""
    var shape = trees_shape_of(source, context)
    if len(shape) != 1:
        raise Error("mojolearn " + caller + ": sample_weight must be 1-D, got shape " + _shape_text(shape))
    if shape[0] != n:
        raise Error("mojolearn " + caller + ": sample_weight has " + String(shape[0]) + " entries for "
                    + String(n) + " samples")
    var w = trees_f32(source, 1, "C", "sample_weight", context)
    if not trees_all_finite(w.addr, n, False, context) or Bool(py=_reduce(w.addr, n, 0, context) < PythonObject(0)):
        raise Error("mojolearn " + caller + ": sample_weight must have finite nonnegative entries")
    _ = w  # keep the Python object alive through the native calls above
    if not Bool(py=_reduce(w.addr, n, 1, context) > PythonObject(0)):
        raise Error("mojolearn " + caller + ": sample_weight must have positive total weight")
    _ = w  # keep the Python object alive through the native calls above
    return w^


def _metrics_mode(mode: PythonObject, context: PythonObject) raises -> String:
    """`_metrics_impl._get_binding`'s mode: the requested one or the
    package default."""
    var b = trees_builtins()
    var selected = mode
    if selected is not None:
        if not Bool(py=py_call(b.isinstance, selected, b.str)):
            raise Error("numeric_mode must be 'fast' or 'identical'")
        var clean = String(py_call(py_call(selected.strip).lower))
        if clean != "fast" and clean != "identical":
            raise Error("numeric_mode must be 'fast' or 'identical'")
        return clean
    return String(py_call(py_call(context["backend_fields"]["_DEFAULT_MODE"].strip).lower))


def trees_r2(y_true: PythonObject, prediction: PythonObject, sample_weight: PythonObject, mode: PythonObject,
             context: PythonObject) raises -> Float64:
    """`metrics.r2_score(y_true, prediction, sample_weight=..., numeric_mode=mode)`
    for 1-D targets (the default multioutput): the float32 R^2 kernel."""
    var y = _as_f32_1d(y_true, "y_true", context)
    var yh = _as_f32_1d(prediction, "y_pred", context)
    if y.rows != yh.rows:
        raise Error("mojolearn metrics: y_true has " + String(y.rows) + " entries and "
                    + "y_pred has " + String(yh.rows) + "; they must be the same length")
    var selected = _metrics_mode(mode, context)
    var metrics = trees_table(context, selected, "_mojolearn_metrics")
    var read_mode = trees_entry(metrics, "metrics_numeric_mode")
    if read_mode is None:
        read_mode = trees_entry(metrics, "umap_numeric_mode")
    if read_mode is not None:
        var code = Int(py=py_call_typed(read_mode))
        var actual = String("fast") if code == 0 else (String("identical") if code == 1 else String("None"))
        if actual != selected:
            raise Error("RuntimeError: mojolearn metrics: requested " + selected + ", binary reports "
                        + actual + "; rebuild it")
    var result = PythonObject(None)
    if sample_weight is not None:
        var w = trees_sample_weight_f32(sample_weight, y.rows, "r2_score", context)
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
    return Float64(py=py_call(trees_builtins().float, result))


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
    var b = trees_builtins()
    var cls = py_type(values)
    if cls is b.str or cls is b.bytes:
        return None
    var view: TreesView
    var bool_source = False
    if cls is b.list or cls is b.tuple:
        var n = len(values)
        if n == 0:
            return None
        # One C-level pass over the item classes (perf3-nbr-msel-trees):
        # exact ints only, as the per-item walk it replaces.
        if labels_list_scalar_kind(values) != LABELS_LIST_INT:
            return None
        var store = PythonObject(None)
        try:
            store = py_call(context["array_type"], "q", values)
        except:
            return None
        view = _view_of_array(classical_wrap_array(store, Python.tuple(n), "<i8", "C", context))
    else:
        if native_pandas_exact(values):
            return None
        if len(trees_shape_of(values, context)) != 1:
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
        var wide = trees_vector(view.size(), "<i4", context)
        trees_cast(view.addr, typestr, trees_address(wide), "<i4", view.size(), context)
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
    var b = trees_builtins()
    var shape = trees_shape_of(values, context)
    if len(shape) != 1:
        raise Error(name + " must be one-dimensional single-label data")
    # perf3-nbr-msel-trees: str labels are encoded natively (classes and one
    # int32 code per row) instead of a per-label type walk, a str() per label
    # and a dict lookup per row in `_codes`. A '<U' ndarray is read from its
    # buffer; a flat list of str labels after the flatten by the native str
    # reader. Both give the sorted distinct strs and their dense codes, so
    # `_label_set` and `_codes` see the same classes as the walk's values.
    var text = score_text_buffer_labels(values, context)
    if text:
        var found = text.value().copy()
        return ScoreLabels(True, found.classes, found.codes, PythonObject(None), found.n, "string")
    var labels = _flatten_labels(values, context)
    var n = len(labels)
    if n == 0:
        raise Error(name + " must contain at least one label")
    var listed = score_text_list_labels(labels, context)
    if listed:
        var found = listed.value().copy()
        return ScoreLabels(True, found.classes, found.codes, PythonObject(None), found.n, "string")
    var all_str = True
    var all_int = True
    # Builtins, bound methods and exact types read once, not per label; an
    # exact str (int) is its own str() (int()) in CPython, so those calls are
    # skipped for exact types only (lane perf-sweep-trees, 2026-11-05).
    var isinstance_ = b.isinstance
    var str_t = b.str
    var int_t = b.int
    for i in range(n):  # cpu-route: the label types of user labels, the explicit label input step
        var v = labels[i]
        if all_str:
            all_str = py_type(v) is str_t or Bool(py=py_call(isinstance_, v, str_t))
        all_int = all_int and metric_is_integral(v, context)
    var out = Python.list()
    var append = out.append
    if all_str:
        for i in range(n):  # cpu-route: str labels, the explicit label input step
            var v = labels[i]
            _ = py_call(append, v if py_type(v) is str_t else py_call(str_t, v))
        return ScoreLabels(False, PythonObject(None), PythonObject(None), out, n, "string")
    if all_int:
        for i in range(n):  # cpu-route: Python int labels, the explicit label input step
            var v = labels[i]
            _ = py_call(append, v if py_type(v) is int_t else py_call(int_t, v))
        return ScoreLabels(False, PythonObject(None), PythonObject(None), out, n, "integer")
    raise Error("TypeError: " + name + " must contain only strings or only integers; "
                + "floating labels, missing labels and mixed types are unsupported")


def _classification_encoded(values: PythonObject, name: String, context: PythonObject) raises -> ScoreLabels:
    var fast = _native_classification(values, context)
    if fast:
        return fast.value().copy()
    return _classification_labels(values, name, context)


def _label_set(labels: ScoreLabels) raises -> PythonObject:
    var b = trees_builtins()
    return py_call(b.set, labels.classes if labels.encoded else labels.values)


def _codes(labels: ScoreLabels, order: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_metrics._codes`: each label's index in `order` (-1 when
    absent) as an int32 Array; encoded labels map their class table once and
    gather per row (`gather_i32`)."""
    var index = Python.dict()
    for i in range(len(order)):  # driver-loop: over the distinct class labels
        py_setitem(index, order[i], PythonObject(i))
    var out = trees_vector(labels.n, "<i4", context)
    if labels.encoded:
        var k = len(labels.classes)
        var table = trees_vector(k, "<i4", context)
        var t = _I32P(unsafe_from_address=trees_address(table))
        _ = table  # keep the Python object alive through the native call above
        for i in range(k):  # driver-loop: over the distinct class labels
            t[i] = Int32(Int(py=py_get(index, labels.classes[i], PythonObject(-1))))
            _ = table  # keep the Python object alive through the native call above
        _ = py_call_typed(trees_helper(context, "gather_i32"), PythonObject(trees_address(table)), PythonObject(k),
                          PythonObject(trees_address(labels.codes)), PythonObject(labels.n),
                          PythonObject(trees_address(out)))
        _ = table  # keep the Python object alive through the native call above
        _ = out  # keep the Python object alive through the native call above
        return out
    var p = _I32P(unsafe_from_address=trees_address(out))
    _ = out  # keep the Python object alive through the native call above
    var lookup = index.get  # read once, not per row
    var missing = PythonObject(-1)
    for i in range(labels.n):  # cpu-route: Python object labels mapped per row, the explicit label input step
        p[i] = Int32(Int(py=py_call(lookup, labels.values[i], missing)))
        _ = out  # keep the Python object alive through the native call above
    return out


def _refuse_multilabel(y: PythonObject, name: String, caller: String, context: PythonObject) raises:
    var shape = trees_shape_of(y, context)
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
        var append = record.append  # read once, not once per word
        for i in range(_PARAMS):  # small-loop(params): the fourteen fixed parameter words of one stage
            _ = py_call(append, PythonObject(params[i] if i < len(params) else 0))
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
    var b = trees_builtins()
    var sys = native_loaded_module("sys")
    var module = py_get(sys.modules, "mojolearn._expansion_metrics")
    if module is None:
        raise Error("ImportError: mojolearn: the metrics programs are not connected")
    var prog_context = py_get(py_fields(module), "_PROG_CONTEXT")
    if prog_context is None:
        raise Error("ImportError: mojolearn: the metrics programs are not connected")
    var prog = py_call(context["trees_namespace"])
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


def trees_accuracy(y_true: PythonObject, prediction: PythonObject, sample_weight: PythonObject, mode: PythonObject,
                   context: PythonObject) raises -> Float64:
    """`_expansion_metrics.accuracy_fraction(y_true, prediction,
    sample_weight, mode)`: the (weighted) matched fraction from the device
    COUNT tail."""
    var caller = String("accuracy_score")
    _refuse_multilabel(y_true, "y_true", caller, context)
    _refuse_multilabel(prediction, "y_pred", caller, context)
    var true = _classification_encoded(y_true, "y_true", context)
    var pred = _classification_encoded(prediction, "y_pred", context)
    return _accuracy_encoded(true, pred, sample_weight, mode, context)


def trees_accuracy_labels(y_true: PythonObject, pred: ScoreLabels, sample_weight: PythonObject, mode: PythonObject,
                          context: PythonObject) raises -> Float64:
    """`trees_accuracy` with the prediction already encoded (a classifier's
    own class codes): y_true is checked and encoded as before."""
    var caller = String("accuracy_score")
    _refuse_multilabel(y_true, "y_true", caller, context)
    var true = _classification_encoded(y_true, "y_true", context)
    return _accuracy_encoded(true, pred, sample_weight, mode, context)


def _accuracy_encoded(true: ScoreLabels, pred: ScoreLabels, sample_weight: PythonObject, mode: PythonObject,
                      context: PythonObject) raises -> Float64:
    var b = trees_builtins()
    var caller = String("accuracy_score")
    if true.n != pred.n:
        raise Error("mojolearn " + caller + ": y_true and y_pred lengths differ")
    if true.kind != pred.kind:
        raise Error("TypeError: mojolearn " + caller + ": y_true and y_pred must use the same label type")
    if true.n > _INT32_MAX:
        raise Error("mojolearn " + caller + ": at most INT32_MAX rows")
    var n = true.n
    var weighted = sample_weight is not None
    var w: Optional[TreesMatrix] = None
    if weighted:
        w = trees_sample_weight_f32(sample_weight, n, caller, context)
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
    # The COUNT epilogue (x_metrics/cm_epi.mojo `_count`) reads only the
    # matched grouping's counts when unweighted, and only the folded wsum
    # slot when weighted (TP is not read). The y_true and y_pred groupings,
    # their counts and the weighted total grouping were queued and never
    # read: two to four O(n) device sorts per score (p3-trees). Only the
    # grouping the epilogue reads is queued; the result is the same.
    var tp = matched
    if not weighted:
        var g0 = prog.group(matched, n, L, W)
        _ = prog.counts(g0[0], L)
        tp = g0[0]
    # `_cm_stage(prog, groups, w, n, L, _CM_COUNT, 11)`.
    var out = prog.alloc(11)
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


def _owned_f32(w: TreesMatrix, context: PythonObject) raises -> PythonObject:
    """A float32 Array copy of weights that borrow a foreign buffer (the
    program runner copies Arrays into its arena; `put` took `as_f32_c`'s
    Array). The bytes are copied in C (`array.array.frombytes`)."""
    # copied from the matrix's own address (its owner need not be a
    # bytes-like object: frombytes refused a viewed NumPy weight vector)
    var count = w.rows * w.cols
    var out = trees_vector(count, "<f4", context)
    trees_cast(w.addr, "<f4", trees_address(out), "<f4", count, context)
    _ = out  # keep the Python object alive through the native call above
    return out


# ------------------------------------------------------------ the forests


def forest_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ForestProtocol.score(X, y, sample_weight)`: the forest's predict and
    the accuracy (classifiers) or float32 R^2 (regressors) in one call."""
    var shape = trees_shape_of(args[1], context)
    if len(shape) != 1:
        raise Error("score requires one-dimensional targets")
    if forest_kind(model, context).endswith("_classifier"):
        return _score_codes(model, forest_predict_codes(model, Python.tuple(args[0]), context), args[1], args[2],
                            shape[0], context)
    return _score_values(model, forest_predict(model, Python.tuple(args[0]), context), args[1], args[2], shape[0],
                         context)


def forest_score_public(model: PythonObject, kind: String, x: TreesMatrix, y: PythonObject,
                        sample_weight: PythonObject, context: PythonObject) raises -> PythonObject:
    """`forest_score` over the X the public door already admitted and
    checked (no second view or width check)."""
    var shape = trees_shape_of(y, context)
    if len(shape) != 1:
        raise Error("score requires one-dimensional targets")
    if kind.endswith("_classifier"):
        return _score_codes(model, forest_public_predict(model, kind, x, False, True, context), y, sample_weight,
                            shape[0], context)
    return _score_values(model, forest_public_predict(model, kind, x, False, False, context), y, sample_weight,
                         shape[0], context)


def _score_values(model: PythonObject, prediction: PythonObject, y: PythonObject, sample_weight: PythonObject,
                  n: Int, context: PythonObject) raises -> PythonObject:
    var rows = Int(py=prediction.shape[0]) if py_type(prediction) is context["array_class"] else len(prediction)
    if rows != n:
        raise Error("score target and prediction lengths differ")
    var mode = PythonObject(forest_effective_mode(py_fields(model), context))
    var y32 = trees_f32(y, 1, "C", "y", context)
    var p32 = trees_f32(prediction, 1, "C", "prediction", context)
    var result = trees_r2(_as_array(y32, context), _as_array(p32, context), sample_weight, mode, context)
    _ = y32^
    _ = p32^
    return PythonObject(result)


def _score_codes(model: PythonObject, pair: PythonObject, y: PythonObject, sample_weight: PythonObject,
                 n: Int, context: PythonObject) raises -> PythonObject:
    """A classifier's accuracy from its `(classes_, codes)` prediction.
    Lane ne1-h-labels (plan section H, N435): the one shared score from
    codes (bindings/classical_labels.mojo `classical_score_from_codes`): y
    is encoded once against `classes_` and the predicted codes go to the
    COUNT program as they are, for str and integer classes alike (lane
    perf-trees-owed scored str classes in codes through a private copy,
    folded here; lesson 8). Other classes decode as before inside it. The
    match count and the soft binary64 weight sums do not depend on label
    numbering: the same bits."""
    var classes = pair[0]
    var codes = pair[1]
    var rows = Int(py=codes.size)
    if rows != n:
        raise Error("score target and prediction lengths differ")
    var mode = PythonObject(forest_effective_mode(py_fields(model), context))
    return PythonObject(classical_score_from_codes(y, classes, codes, sample_weight, mode, context))


def _as_array(m: TreesMatrix, context: PythonObject) raises -> PythonObject:
    """The float32 vector as the Array `as_f32_c` returned (a borrowed
    foreign buffer is passed as its own owner, which reads the same)."""
    return m.keep
