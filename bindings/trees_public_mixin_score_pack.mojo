# SPDX-License-Identifier: Apache-2.0
# `score` of the tree ensembles that inherit scikit-learn's mixin answer
# (`_mode.ClassifierScoreMixin.score`: the accuracy of `predict(X)`;
# `_mode.RegressorScoreMixin.score`: the float32 R^2 of `predict(X)`), in one
# native call: the estimator's native predict (bindings/trees_public_boost.mojo)
# and the native accuracy / R^2 (bindings/trees_public_score.mojo), with no
# Python between the prediction and the metric. The metric kernels and their
# order are the ones the mixins called, so the score keeps its bits.
# The PACK half of the former `bindings/trees_public_mixin_score.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/trees_public_mixin_score_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.base_helpers_key_unpack import base_helpers_key
from bindings.base_helpers_released_unpack import base_reduce_stat_int
from bindings.py_balanced import py_call, py_get, py_fields, py_contains
from bindings.classifier_public_lifecycle_door import classifier_label_list
from bindings.py_typed_call import py_call_typed
from bindings.trees_public_inputs import (
    trees_builtins, trees_table, trees_entry,
    trees_address, trees_cast, trees_decode_labels,
)
from bindings.admitted_input import admit_f32, admitted_vector
from core.host_finite_scan import host_all_finite_f32
# TOMBSTONE: trees_f32 / trees_all_finite / trees_vector uses (the trees admission aliases) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from bindings.trees_public_score_door import (
    trees_r2, trees_shape_of, trees_sample_weight_f32, _metrics_mode, trees_class_codes,
)
from bindings.classical_labels import classical_score_from_codes, _classification_encoded as labels_classification_encoded
from bindings.trees_public_gbdt_door import _adapter_fitted, adapter_codes, adapter_decision_function
from bindings.trees_public_boost import boost_operation, boost_kind, boost_predict_codes


def boost_score(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`score(X, y, sample_weight)` of Bagging*, AdaBoost* and DART*."""
    var X = args[0]
    var y = args[1]
    var sample_weight = args[2]
    # The mixins pass the estimator's own numeric_mode (None: the default).
    var mode = py_get(py_fields(model), "numeric_mode")
    var kind = boost_kind(model, context)
    if kind.endswith("_classifier"):
        # Scored from the predicted class codes (N043, ne1-a-trees): classes
        # are not decoded to n Python labels and re-encoded. Integration
        # (integ/ne-20261107): ne1-h-labels folded the forest door's private
        # `_string_class_codes` into the one shared score from codes
        # (bindings/classical_labels.mojo `classical_score_from_codes`, the
        # forest door's route now), so this door calls it too (lesson 8).
        var pair = boost_predict_codes(model, kind, X, context)
        return PythonObject(classical_score_from_codes(y, pair[0], pair[1], sample_weight, mode, context))
    var prediction = boost_operation(model, "trees_boost_predict", Python.tuple(X), context)
    # `as_f32_c(y, ndim=None)` and `as_f32_c(prediction, ndim=None)`, then
    # `r2_score`'s 1-D path.
    # NaN is refused by trees_r2's own check, with the metrics wording.
    var truth = admit_f32(y, -1, "y", context, finite=False)
    var pred = admit_f32(prediction, -1, "prediction", context, finite=False)
    var result = trees_r2(truth.array, pred.array, sample_weight, mode, context)
    _ = truth^
    _ = pred^
    return PythonObject(result)


# ---------------------------------------------------- the GBDT adapters
