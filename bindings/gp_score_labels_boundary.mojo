# SPDX-License-Identifier: Apache-2.0
"""GP score label UNPACK: exact public pins, native code and metric plans."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.os import getenv
from core.admitted_ids import admitted_ids_i64
from core.gp_native_label_table import GP_LABEL_OBJECT, GP_LABEL_I64
from core.gp_native_score_labels import GPScoreLabelCodes, gp_score_integer_labels, gp_score_key_labels, gp_score_pair
from core.gp_native_accuracy import GPCAccuracyProgram
from bindings.gp_score_label_storage import GPDecodedScoreLabels
from bindings.gp_label_input_boundary import gp_label_input
from bindings.label_input_sources import LABELS_LIST_INT
from bindings.labels_score_boundary import ScoreKeyOwner, score_keys_unpack
from bindings.mcls_common import mc_integer_view, mc_label_list, mc_refuse_multilabel, MC_INT32_MAX
from bindings.metrics_native_inputs import metric_shape_admitted
from bindings.metrics_native_support import metric_is_integral_type
from bindings.classical_pandas import classical_admit_value
from bindings.classical_arrays import classical_address, classical_is_array
from bindings.native_buffer_owner import native_buffer_view
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type
from bindings.arena_prog_native import _family_entries
from providers.python.neural_descriptor_pyapi import NeuralDescriptorProvider, neural_descriptor_connect_factory_once
from x_metrics.descriptor_schema import METRICS_PROGRAM_ABI, MP_OPERATION_COUNT


@fieldwise_init
struct GPMetricLabels(Movable):
    var kind: String
    # Public scalar identity and immutable text/magnitude bytes stay at
    # UNPACK. The native plans read keys and spans only.
    var owner: ScoreKeyOwner
    var plan: GPScoreLabelCodes


def _gp_score_integer(source: Int, rows: Int) raises -> GPMetricLabels:
    var plan = gp_score_integer_labels(source, rows)
    return GPMetricLabels(String("integer"), ScoreKeyOwner(), plan^)


def _gp_score_keys(var owner: ScoreKeyOwner, kind: String) raises -> GPMetricLabels:
    if kind == "integer":
        # Normalize native scalar pins once, then retain the existing signed
        # host encoder whenever the same array('q') conversion would fit.
        var words = gp_label_input(owner.values, LABELS_LIST_INT)
        if words:
            var result = _gp_score_integer(words.value().address(), len(words.value().values))
            _ = words
            _ = owner
            return result^
    var plan = gp_score_key_labels(owner.keys)
    return GPMetricLabels(String(kind), owner^, plan^)


def _gp_score_truth(values: PythonObject, context: PythonObject) raises -> GPMetricLabels:
    var b = native_loaded_module("builtins")
    var cls = py_type(values)
    if cls is b.list or cls is b.tuple:
        var words = gp_label_input(values, LABELS_LIST_INT)
        if words:
            var result = _gp_score_integer(words.value().address(), len(words.value().values))
            _ = words
            _ = values
            return result^
    else:
        var can_view = classical_is_array(values, context)
        if not can_view:
            try:
                _ = native_buffer_view(values)
                can_view = True
            except:
                pass
        if can_view:
            var shape = metric_shape_admitted(values, context)
            can_view = len(shape) == 1
        if can_view:
            var input = mc_integer_view(values, context)
            if input:
                var count = input.value().count
                var words = List[Int64](length=count, fill=0)
                var admitted = admitted_ids_i64(Int(py=classical_address(input.value().view)), input.value().stride,
                    0, count, 1, input.value().item, input.value().signed, False,
                    MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(words.unsafe_ptr())))
                _ = input  # raw source is retained on both success and refusal
                if admitted:
                    var result = _gp_score_integer(Int(words.unsafe_ptr()), count)
                    _ = words
                    _ = values
                    return result^
    # This shared boundary still owns general flatten/admission metadata.
    # Preserve its error order, supported containers and exact scalar rules.
    var listed = mc_label_list(values, "y_true", False, context, True)
    var kind = String(listed[1])
    if kind == "integer":
        var words = gp_label_input(listed[0], LABELS_LIST_INT)
        if words:
            var result = _gp_score_integer(words.value().address(), len(words.value().values))
            _ = words
            _ = listed
            return result^
    var owner = score_keys_unpack(listed[0], kind == "string", context)
    _ = listed
    return _gp_score_keys(owner^, kind)


def _gp_score_prediction_admit(prediction: GPDecodedScoreLabels, context: PythonObject) raises -> Bool:
    # Original mc_pair admits every decoded object before encoding truth.
    # A generated numeric vector is already an exact native one-D span.
    if prediction.kind != GP_LABEL_OBJECT:
        return False
    for i in range(len(prediction.pins)):  # UNPACK retained public class objects under GIL
        classical_admit_value(prediction.pins[i], context, 1)
    var b = native_loaded_module("builtins")
    var nested = False
    if len(prediction.pins) > 0:
        var cls = py_type(prediction.pins[0])
        nested = cls is b.list or cls is b.tuple
        if nested and len(prediction.pins[0]) != 1:
            raise Error("NotImplementedError: mojolearn accuracy_score: y_pred is 2-D; multilabel-indicator "
                + "targets are NOT IMPLEMENTED (metrics/NOT_IMPLEMENTED.tsv), only binary and multiclass 1-D labels")
    return nested


def _gp_score_prediction(prediction: GPDecodedScoreLabels, nested: Bool,
                         context: PythonObject) raises -> GPMetricLabels:
    if prediction.kind == GP_LABEL_I64:
        # Keep the old generated-Array -> integer-store snapshot before the
        # host encoder. The storage is native but the conversion is unchanged.
        var words = List[Int64](length=prediction.rows, fill=0)
        _ = admitted_ids_i64(Int(prediction.words.unsafe_ptr()), 8, 0, prediction.rows, 1, 8, True, False,
            MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(words.unsafe_ptr())))
        _ = prediction
        var result = _gp_score_integer(Int(words.unsafe_ptr()), prediction.rows)
        _ = words
        return result^
    if nested:
        raise Error("y_pred must be one-dimensional single-label data")
    if prediction.rows == 0:
        raise Error("y_pred must contain at least one label")
    var strings = prediction.kind == GP_LABEL_OBJECT
    var integers = strings
    var b = native_loaded_module("builtins")
    for i in range(len(prediction.pins)):  # UNPACK exact scalar type identities, no Python set/map
        var cls = py_type(prediction.pins[i])
        strings = strings and cls is b.str
        integers = integers and metric_is_integral_type(cls, context)
    if not strings and not integers:
        raise Error("TypeError: y_pred must contain only strings or only integers; "
            + "floating labels, missing labels and mixed types are unsupported")
    var owner = score_keys_unpack(prediction.pins, strings, context)
    return _gp_score_keys(owner^, String("string") if strings else String("integer"))


def gpc_score_result(truth: PythonObject, prediction: GPDecodedScoreLabels,
                     context: PythonObject) raises -> Float64:
    mc_refuse_multilabel(truth, "y_true", "accuracy_score", context, True)
    var nested = _gp_score_prediction_admit(prediction, context)
    var true_labels = _gp_score_truth(truth, context)
    var pred_labels = _gp_score_prediction(prediction, nested, context)
    if true_labels.plan.rows != pred_labels.plan.rows:
        raise Error("mojolearn accuracy_score: y_true and y_pred lengths differ")
    if true_labels.kind != pred_labels.kind:
        raise Error("TypeError: mojolearn accuracy_score: y_true and y_pred must use the same label type")
    if true_labels.plan.rows > MC_INT32_MAX:
        raise Error("mojolearn accuracy_score: at most INT32_MAX rows")
    var pair = gp_score_pair(true_labels.plan, pred_labels.plan)
    var program = GPCAccuracyProgram(pair.rows, pair.classes, Int(pair.truth.unsafe_ptr()), Int(pair.prediction.unsafe_ptr()))
    _ = pair
    _ = true_labels
    _ = pred_labels
    var entries = _family_entries(PythonObject(None), "x_metrics", context["family_metrics_prog_context"])
    var provider = neural_descriptor_connect_factory_once(entries["metrics_program_native_api_v1"],
        String(METRICS_PROGRAM_ABI), MP_OPERATION_COUNT)
    var entry = provider.downcast_value_ptr[NeuralDescriptorProvider]()[].entry
    var ranges = String(getenv("MOJOLEARN_ARENA_RANGES", "1").strip()) != "0"
    var result = Float64(0.0)
    var py = Python()
    try:
        with GILReleased(py):
            result = program.run(entry, ranges)
    finally:
        _ = provider
        _ = entries
        _ = program
        _ = truth
        _ = prediction
    return result
