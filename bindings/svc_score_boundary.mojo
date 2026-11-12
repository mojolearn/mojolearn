# SPDX-License-Identifier: Apache-2.0
"""SVM accuracy UNPACK and pins over the original label and COUNT providers.

Native code/arena storage replaces private prediction and encoded Arrays.
Exact scalar reads and reference ownership remain under the GIL, once per
USED class: a decoded prediction keeps its rows as native int32 slots into
the used class objects (SvcDecodedScore), so no CPython call runs per row. Label
encoding and remapping retain labels_prepare_targets/labels_gather, which
are distinct from the GP scorer's existing host-only label path.
"""
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.cpython_table import cpython_table
from bindings.p3_labels_metrics_native import p3_item
from bindings.py_balanced import py_type, py_get
from bindings.classical_arrays import classical_address
from bindings.classical_pandas import classical_admit_value
from bindings.classical_labels import (
    _native_classification_input, _classification_labels, _encode_code,
    _refuse_multilabel, _accuracy_context,
)
from bindings.label_admission_native import classifier_label_list, LabelTypeKinds
from bindings.label_input_sources import LABELS_LIST_INT
from bindings.gp_label_input_boundary import gp_label_input
from core.gp_native_label_table import GP_LABEL_OBJECT, GP_LABEL_I64, GP_LABEL_F64
from bindings.svc_common import svc_builtins, svc_cached_class_table, svc_shape_of
from bindings.labels_descriptor_boundary import labels_prepare_targets, labels_gather
from bindings.labels_score_boundary import ScoreKeyOwner, score_keys_unpack, score_keys_union, score_keys_index
from core.labels_score_keys import ScoreKeyIndex, score_integer_keys, score_key_codes
from core.svc_score_codes import svc_score_codes_valid, svc_used_classes, svc_gather_words, SvcUsedClasses
from core.gp_native_accuracy import GPCAccuracyProgram
from bindings.arena_prog_native import _family_entries
from bindings.arena_prog_run import _switch
from bindings.x_prep_native_owner import _resident_scope
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once
from x_metrics.descriptor_schema import METRICS_PROGRAM_ABI, MP_OPERATION_COUNT


struct SvcScoreLabels(Movable):
    var owner: ScoreKeyOwner
    var codes: List[Int32]
    var rows: Int
    var encoded: Bool
    var kind: String

    def __init__(out self, rows: Int, encoded: Bool, kind: String):
        self.owner = ScoreKeyOwner()
        self.codes = List[Int32](length=rows if encoded else 0, fill=0)
        self.rows = rows
        self.encoded = encoded
        self.kind = kind


struct SvcDecodedScore(Movable):
    """A decoded prediction for SVC score. Numeric kinds hold one int64 or
    float64 word per row. GP_LABEL_OBJECT holds the USED class objects once
    each (`pins`, in first-appearance order) and one int32 slot per row into
    them: the CPython work on object labels is per used class, never per row."""
    var rows: Int
    var kind: Int
    var words: List[UInt64]
    var pins: List[PythonObject]
    var slots: List[Int32]

    def __init__(out self, rows: Int, kind: Int):
        self.rows = rows
        self.kind = kind
        self.words = List[UInt64]()
        self.pins = List[PythonObject]()
        self.slots = List[Int32]()


def svc_decode_score(owner: PythonObject, var classes: PythonObject, codes: List[Int64],
                     context: PythonObject) raises -> SvcDecodedScore:
    # Preserve the public decoder's cache lookup/build and numeric gather
    # before any truth-shape admission. Mutable classes use the same key.
    classes = classifier_label_list(classes, context)
    var built = svc_cached_class_table(owner, classes, context)
    var kind = Int(py=built[0])
    var table = built[1]
    var count = len(codes)
    if table is not None:
        var result = SvcDecodedScore(count, GP_LABEL_I64 if kind == 0 else GP_LABEL_F64)
        result.words = List[UInt64](length=count, fill=0)
        if count > 0:
            labels_gather(context, 1 if kind == 0 else 2, Int(py=classical_address(table)), len(classes),
                Int(codes.unsafe_ptr()), count, Int(result.words.unsafe_ptr()))
        _ = codes
        _ = table
        return result^
    if not svc_score_codes_valid(codes, len(classes)):
        raise Error("IndexError: list index out of range")
    # UNPACK per USED class (the rows only index them, in native code): the
    # class objects a row-by-row decode would have referenced, once each.
    # The row scan is data-sized native work over plain words: the GIL is
    # released around it like the gathers below (lane perf-audit-2-a).
    var n_classes = len(classes)
    var used = SvcUsedClasses(0)
    with GILReleased(Python()):
        used = svc_used_classes(codes, n_classes)
    var result = SvcDecodedScore(count, GP_LABEL_OBJECT)
    var tuple_input = py_type(classes) is svc_builtins().tuple
    for i in range(len(used.order)):  # UNPACK one borrowed reference per used class (C slot, no protocol)
        result.pins.append(PythonObject(from_borrowed=p3_item(classes, used.order[i], tuple_input)))
    swap(result.slots, used.slots)
    _ = classes
    _ = codes
    if kind == 2:
        return result^
    # The original failed-table decoder attempted array('q'/'d', map(...))
    # over selected labels. Retain selected-only overflow and normalization:
    # one exact scalar conversion per used class, then a native gather.
    var class_words = List[UInt64]()
    var overflow = List[Int32](length=1, fill=0)
    var api = cpython_table()
    ref cpy = Python().cpython()
    try:
        class_words = List[UInt64](length=len(result.pins), fill=0)
        for i in range(len(result.pins)):  # UNPACK exact admitted numeric scalar C conversions, one per used class
            var raw = api[].PyNumber_Long(result.pins[i]._obj_ptr) if kind == 0 else api[].PyNumber_Float(result.pins[i]._obj_ptr)
            if not raw:
                raise cpy.unsafe_get_error()
            var number = PythonObject(from_owned=raw)
            if kind == 0:
                overflow[0] = 0
                var value = api[].PyLong_AsLongLongAndOverflow(raw, Int(overflow.unsafe_ptr()))
                if overflow[0] != 0:
                    return result^
                if value == -1 and api[].PyErr_Occurred():
                    raise cpy.unsafe_get_error()
                class_words[i] = bitcast[DType.uint64](value)
            else:
                var value = api[].PyFloat_AsDouble(raw)
                if value == -1 and api[].PyErr_Occurred():
                    raise cpy.unsafe_get_error()
                class_words[i] = bitcast[DType.uint64](value)
            _ = number
    except:
        return result^
    _ = overflow
    var words = List[UInt64]()
    with GILReleased(Python()):
        words = svc_gather_words(class_words, result.slots)
    _ = class_words
    swap(result.words, words)
    result.kind = GP_LABEL_I64 if kind == 0 else GP_LABEL_F64
    result.pins = List[PythonObject]()
    result.slots = List[Int32]()
    return result^


def _svc_integer_score(source: Int, code: Int, rows: Int, context: PythonObject) raises -> SvcScoreLabels:
    var result = SvcScoreLabels(rows, True, String("integer"))
    var classes = List[Int64](unsafe_uninit_length=rows)
    var count = labels_prepare_targets(context, source, code, rows, Int(classes.unsafe_ptr()),
        Int(result.codes.unsafe_ptr()), 0, 0, 0)
    with GILReleased(Python()):
        result.owner.keys = score_integer_keys(Int(classes.unsafe_ptr()), count)
    _ = classes
    return result^


def _svc_score_truth(values: PythonObject, context: PythonObject) raises -> SvcScoreLabels:
    var input = _native_classification_input(values, context)
    if input:
        var view = input.value().view
        var result = _svc_integer_score(Int(py=classical_address(view)), _encode_code(input.value().typestr),
            Int(py=view.size), context)
        _ = view
        _ = input
        _ = values
        return result^
    var labels = _classification_labels(values, "y_true", context)
    var result = SvcScoreLabels(labels.n, False, labels.kind)
    swap(result.owner, labels.owner)
    return result^


def _svc_score_prediction(prediction: SvcDecodedScore, context: PythonObject) raises -> SvcScoreLabels:
    if prediction.rows == 0:
        raise Error("y_pred must contain at least one label")
    if prediction.kind == GP_LABEL_I64:
        var result = _svc_integer_score(Int(prediction.words.unsafe_ptr()), 3, prediction.rows, context)
        _ = prediction
        return result^
    if prediction.kind == GP_LABEL_OBJECT:
        # Match _native_classification's exact-int list fast path, including
        # its all-or-nothing int64 conversion before general scalar keys. The
        # rows hold only the used classes, so admitting each once decides the
        # same all-or-nothing outcome; the row words are a native gather.
        var words = gp_label_input(prediction.pins, LABELS_LIST_INT)
        if words:
            var row_words = List[UInt64]()
            with GILReleased(Python()):
                row_words = svc_gather_words(words.value().values, prediction.slots)
            var result = _svc_integer_score(Int(row_words.unsafe_ptr()), 3, prediction.rows, context)
            _ = row_words
            _ = words
            _ = prediction
            return result^
        var all_str = True
        var all_int = True
        var kinds = LabelTypeKinds()
        for i in range(len(prediction.pins)):  # UNPACK exact scalar identity under GIL, one per used class
            var kind = kinds.admit_raw(prediction.pins[i]._obj_ptr, context)
            all_str = all_str and kind == 1
            all_int = all_int and (kind == 2 or kind == 3)
        _ = kinds^
        if all_str or all_int:
            # Encoded form: one key per used class, one int32 slot per row
            # (_svc_score_remap gathers the joined codes natively).
            var result = SvcScoreLabels(0, False, String("string") if all_str else String("integer"))
            result.owner = score_keys_unpack(prediction.pins, all_str, context)
            result.codes = prediction.slots.copy()
            result.rows = prediction.rows
            result.encoded = True
            return result^
    raise Error("TypeError: y_pred must contain only strings or only integers; "
        + "floating labels, missing labels and mixed types are unsupported")


def _svc_score_remap(labels: SvcScoreLabels, index: ScoreKeyIndex, context: PythonObject) raises -> List[Int32]:
    var result = List[Int32](length=labels.rows, fill=0)
    if labels.encoded:
        var table = List[Int32](unsafe_uninit_length=len(labels.owner.keys))
        with GILReleased(Python()):
            score_key_codes(index, labels.owner.keys, Int(table.unsafe_ptr()))
        labels_gather(context, 0, Int(table.unsafe_ptr()), len(table), Int(labels.codes.unsafe_ptr()),
            labels.rows, Int(result.unsafe_ptr()))
        _ = table
    else:
        with GILReleased(Python()):
            score_key_codes(index, labels.owner.keys, Int(result.unsafe_ptr()))
    _ = labels
    _ = index
    return result^


def svc_accuracy_native(truth: PythonObject, prediction: SvcDecodedScore,
                        context: PythonObject) raises -> Float64:
    var shape = svc_shape_of(truth, context)
    if len(shape) != 1 or Int(py=shape[0]) != prediction.rows:
        raise Error("mojolearn: y must be 1-D with one entry per row of X")
    _refuse_multilabel(truth, "y_true", "accuracy_score", context)
    if prediction.kind == GP_LABEL_OBJECT:
        # The same element admission before truth encoding, once per used
        # class in first-appearance order (the first failing row's class
        # fails first, with the same refusal).
        for i in range(len(prediction.pins)):  # UNPACK element admission, one per used class
            classical_admit_value(prediction.pins[i], context, 1)
    var true_labels = _svc_score_truth(truth, context)
    var pred_labels = _svc_score_prediction(prediction, context)
    if true_labels.rows != pred_labels.rows:
        raise Error("mojolearn accuracy_score: y_true and y_pred lengths differ")
    if true_labels.kind != pred_labels.kind:
        raise Error("TypeError: mojolearn accuracy_score: y_true and y_pred must use the same label type")
    if true_labels.rows > 2147483647:
        raise Error("mojolearn accuracy_score: at most INT32_MAX rows")
    var joined = score_keys_union(true_labels.owner, pred_labels.owner)
    var index = score_keys_index(joined)
    var true_codes = _svc_score_remap(true_labels, index, context)
    var pred_codes = _svc_score_remap(pred_labels, index, context)
    var class_count = len(index.sorted_rows)
    _ = joined
    _ = index
    _ = true_labels
    _ = pred_labels
    # Read the same live module context as labels_accuracy; do not substitute
    # the GP caller's connected context or cache a rebinding-sensitive value.
    var prog_context = _accuracy_context(context)
    var entries = _family_entries(PythonObject(None), "x_metrics", prog_context)
    var provider = neural_descriptor_connect_factory_once(entries["metrics_program_native_api_v1"],
        String(METRICS_PROGRAM_ABI), MP_OPERATION_COUNT)
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var program = GPCAccuracyProgram(prediction.rows, class_count,
        Int(true_codes.unsafe_ptr()), Int(pred_codes.unsafe_ptr()))
    _ = true_codes
    _ = pred_codes
    var ranges = _switch("MOJOLEARN_ARENA_RANGES") and py_get(entries, "x_metrics_run_ranges") is not None
    if ranges and prediction.rows > 0:
        # Same thread-local admission. Metrics store exports were tombstoned;
        # connected x_metrics images have no resident cache or input uploads.
        var scope = _resident_scope(entries, prog_context, "x_metrics")
        if scope is not None:
            raise Error("TypeError: x_metrics native accuracy requires its connected nonresident input ABI")
    var output_ranges = py_get(entries, "x_metrics_run_out") is not None
    var result = Float64(0)
    try:
        with GILReleased(Python()):
            result = program.run(entry, ranges, output_ranges)
    finally:
        _ = provider
        _ = entries
        _ = prog_context
        _ = program
        _ = truth
        _ = prediction
    return result
