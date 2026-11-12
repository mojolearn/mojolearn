# SPDX-License-Identifier: Apache-2.0
# The tree ensembles' `score` in one native call: predict and score without
# returning to Python between the two.
#
# Accuracy is the one shared `labels_accuracy` / `classical_score_from_codes`
# (bindings/classical_labels.mojo); `trees_class_codes` maps labels to a
# fitted class table for the adapter and meta scores.
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
from bindings.base_helpers_key_unpack import base_helpers_key
from bindings.labels_stat_boundary_pack import labels_stat_pack
from bindings.base_helpers_released_unpack import base_reduce_stat
from bindings.py_balanced import py_call, py_type, py_setitem, py_fields
from bindings.py_typed_call import py_call_typed
from bindings.trees_public_inputs import trees_builtins, trees_table, trees_entry, trees_address
from bindings.classical_labels import ScoreLabels, LABELS_GATHER_I32
from bindings.labels_descriptor_boundary_door import labels_gather
from bindings.labels_score_boundary_door import score_keys_index
from core.labels_score_keys import score_key_codes
from std.python._cpython import GILReleased
from bindings.admitted_input import AdmittedX, admit_f32, admitted_vector
# TOMBSTONE: trees_f32 / trees_all_finite / TreesView / TreesMatrix / trees_vector / _view_of_array / _view_of_buffer uses (the trees admission aliases) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from bindings.classical_pandas_door import native_pandas_exact, native_pandas_f32
from bindings.trees_public_forest import (
    forest_predict, forest_predict_codes, forest_public_predict, forest_effective_mode, forest_kind,
)
from bindings.classical_labels import classical_score_from_codes
from xtrees.trees_score_run import score_shape_text

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


def _as_f32_1d(source: PythonObject, name: String, context: PythonObject) raises -> AdmittedX:
    """`_metrics_impl._as_f32_1d`: a 1-D (or (n, 1) column) float32 vector,
    float64 refused by name, NaN or infinity refused."""
    var shape = trees_shape_of(source, context)
    var column = len(shape) == 2 and shape[1] == 1
    if len(shape) != 1 and not column:
        raise Error("mojolearn metrics: " + name + " must be 1-D, got shape " + score_shape_text(shape))
    if shape[0] == 0:
        raise Error("mojolearn metrics: " + name + " is empty")
    var fmt = _buffer_format(source, context)
    var float64 = fmt == "d" if fmt.byte_length() > 0 else _has_float_leaf(source)
    if float64:
        raise Error("TypeError: mojolearn metrics: " + name + " is float64; the float64 overloads of "
                    + "r2_score, kl_divergence and silhouette_score are NOT implemented "
                    + "(no float64 on this GPU). Cast to float32 yourself so the "
                    + "precision you run at is the one you chose.")
    # The one admission (admit_f32): the finite test rides in its cast or
    # its one parallel scan; the metrics wording is kept.
    try:
        return admit_f32(source, len(shape), name, context)
    except error:
        if "contains NaN or infinity" in String(error):
            raise Error("mojolearn metrics: " + name + " contains NaN or infinity; refused "
                        + "here (metrics/README.md HAND-OFF ask 3) rather than letting it "
                        + "reach a kernel whose answer for it is a washed sentinel")
        raise error^


def _reduce(addr: Int, count: Int, what: Int, context: PythonObject) raises -> PythonObject:
    """`Array.min()` (0) / `max()` (1) of a float32 vector: `reduce_stat`
    through the base helper table; PACK of its answer as the Python float."""
    return labels_stat_pack(base_reduce_stat(base_helpers_key(context["trees_helpers"]), addr, 0, count, what))


def trees_sample_weight_f32(source: PythonObject, n: Int, caller: String, context: PythonObject) raises -> AdmittedX:
    """`_metrics_impl._sample_weight_f32`: one finite, non-negative float32
    weight per sample with a positive total."""
    var shape = trees_shape_of(source, context)
    if len(shape) != 1:
        raise Error("mojolearn " + caller + ": sample_weight must be 1-D, got shape " + score_shape_text(shape))
    if shape[0] != n:
        raise Error("mojolearn " + caller + ": sample_weight has " + String(shape[0]) + " entries for "
                    + String(n) + " samples")
    var refusal = "mojolearn " + caller + ": sample_weight must have finite nonnegative entries"
    var w = AdmittedX(PythonObject(None), PythonObject(None), 0, 0, 0, 1, False)
    try:
        w = admit_f32(source, 1, "sample_weight", context)
    except error:
        if "contains NaN or infinity" in String(error):
            raise Error(refusal)
        raise error^
    if Bool(py=_reduce(w.address, n, 0, context) < PythonObject(0)):
        raise Error(refusal)
    _ = w  # keep the Python object alive through the native calls above
    if not Bool(py=_reduce(w.address, n, 1, context) > PythonObject(0)):
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
        result = py_call_typed(metrics["r2_score_weighted"], PythonObject(y.address), PythonObject(yh.address),
                               PythonObject(w.address), Python.list(PythonObject(y.rows)))
        _ = yh  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
        _ = w  # keep the Python object alive through the native call above
        _ = w^
    else:
        result = py_call_typed(metrics["r2_score"], PythonObject(y.address), PythonObject(yh.address),
                               Python.list(PythonObject(y.rows)))
        _ = yh  # keep the Python object alive through the native call above
        _ = y  # keep the Python object alive through the native call above
    _ = y^
    _ = yh^
    return Float64(py=py_call(trees_builtins().float, result))


# ------------------------------------------------------------ class codes
# Lane loops-trees-ab (2026-11-11): the private accuracy copy that lived here
# (ScoreLabels, _native_classification, _classification_labels,
# _classification_encoded, _label_set, _codes, _refuse_multilabel, _run,
# _arena_address, trees_accuracy, trees_accuracy_labels, _accuracy_encoded,
# _owned_f32 and the xtrees/trees_score_run.mojo COUNT program words) is
# deleted: its callers use the one shared accuracy and label encoder in
# bindings/classical_labels.mojo (`labels_accuracy`, `_classification_encoded`),
# which already reads labels by class pointer and runs the same COUNT program
# with the same stage words (no bit change: the hit count and the weight sums
# do not depend on label numbering). The per-row CPython loops of the copy
# (a type walk, a `str()`/`int()` and a list append per label, a dict lookup
# per row) are gone with it. Code recoverable at bf33328698 (the lane base).


def trees_class_codes(target: ScoreLabels, var classes: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_label_map(target, vocabulary.get(label, -1))`: each label's index in
    `classes` (the last of duplicate classes winning, -1 for a label outside
    them) as an int32 Array. Python equality runs once per DISTINCT label (one
    dict lookup each, the semantics the per-row lookup had: `True == 1`,
    `"a"` never equal to `1`); every row's code comes from the native key
    index and the base gather, with no CPython call per row."""
    var vocabulary = Python.dict()
    for i in range(len(classes)):  # driver-loop: over the fitted classes (the vocabulary)
        py_setitem(vocabulary, classes[i], PythonObject(i))
    var lookup = vocabulary.get  # read once, not per label
    var missing = PythonObject(-1)
    var n = target.n
    var out = admitted_vector(n, "<i4", context)
    var k: Int
    var table: List[Int32]
    var row_codes = List[Int32]()
    var rows_address: Int
    if target.encoded:
        # Integer labels the native encoder wrote: k distinct class words and
        # one int32 code per row (`target.codes`).
        k = len(target.owner.keys)
        table = List[Int32](length=max(k, 1), fill=-1)
        for d in range(k):  # driver-loop: over the distinct target classes
            var word = PythonObject(Int(target.owner.keys[d].number))
            table[d] = Int32(Int(py=py_call(lookup, word, missing)))
        rows_address = trees_address(target.codes)
    else:
        # Admitted str or int labels (one native key per row): the distinct
        # keys in sorted order, then each row's position among them.
        var index = score_keys_index(target.owner.keys)
        k = len(index.sorted_rows)
        table = List[Int32](length=max(k, 1), fill=-1)
        for d in range(k):  # driver-loop: over the distinct target labels
            table[d] = Int32(Int(py=py_call(lookup, target.owner.values[index.sorted_rows[d]], missing)))
        row_codes = List[Int32](length=max(n, 1), fill=0)
        var row_address = Int(row_codes.unsafe_ptr())
        with GILReleased(Python()):
            score_key_codes(index, target.owner.keys, row_address, True)
        _ = index^
        rows_address = row_address
    labels_gather(context, LABELS_GATHER_I32, Int(table.unsafe_ptr()), k, rows_address, n, trees_address(out))
    _ = table^
    _ = row_codes^
    _ = target  # keep the label owners (and their pins) alive through the reads above
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


def forest_score_public(model: PythonObject, kind: String, x: AdmittedX, y: PythonObject,
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
    # NaN is refused by trees_r2's own check, with the metrics wording.
    var y32 = admit_f32(y, 1, "y", context, finite=False)
    var p32 = admit_f32(prediction, 1, "prediction", context, finite=False)
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


def _as_array(m: AdmittedX, context: PythonObject) raises -> PythonObject:
    """The float32 vector as the Array `as_f32_c` returned (a borrowed
    foreign buffer is passed as its own owner, which reads the same)."""
    return m.array
