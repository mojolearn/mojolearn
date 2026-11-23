# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a classifier member's `class_weight` (lane mf-adm3,
2026-11-23); the RUN is core/member_class_weight.mojo (its module note has
the tail layout and the family steps).

  * `member_cw_kind` (each classifier admit): which tail the estimator's
    class_weight takes: none, 'balanced' (no label needed: the member FIT
    computes it from its codes), an exact dict (a PENDING tail: the dict is
    keyed by labels, which only the caller holds), or not admitted (any
    other value: the estimator's own fit refuses it in its words, so the
    caller keeps that path for it).
  * `member_cw_fill` (the caller's UNPACK, after the admit, with the
    caller's labels in code order): the pending tail becomes the table,
    read with the estimator's OWN dict admission and in its words:
    LogisticRegression bindings/classical_labels_door.mojo
    `labels_class_weight_rows` (`class_weight_admit_dictionary`, then
    `class_weight_dictionary_values` over the label list); LinearSVC
    bindings/svc_linear_boundary_unpack.mojo `_qn_row_weights`
    (`class_weight_dictionary_values[CLASS_WEIGHT_SVC]`); SVC
    bindings/svc_kernel_params_unpack.mojo `svm_weights_unpack` (its dict
    loop and refusal, `svc_float`); the x_linear kinds (SGD*,
    RidgeClassifier, LogisticRegressionCV)
    bindings/x_linear_common_unpack.mojo `xl_class_weights_unpack` (its
    float32 words). A key the caller's classes do not hold raises those
    words here, before any fit, as each fold's own fit would.
    False: a pending tail and no labels (a caller that does not hand them
    over keeps its own path for such an estimator, as before).

SERIAL BY NATURE: one dictionary read per member admission (a few entries).
"""
from std.python import Python, PythonObject
from bindings.class_weight_boundary_unpack import (
    CLASS_WEIGHT_SVC, class_weight_admit_dictionary, class_weight_dictionary_values,
)
from bindings.classical_labels_door import labels_address
from bindings.family_common_unpack import family_repr
from bindings.label_admission_native_pack import classifier_label_list
from bindings.labels_descriptor_boundary_door import labels_scope
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_call, py_contains, py_get, py_type
from bindings.svc_common_unpack import svc_float
from bindings.x_linear_common_unpack import xl_class_weights_unpack
from core.member_class_weight import (
    MEMBER_CW_NONE, MEMBER_CW_BALANCED, MEMBER_CW_DICT, member_cw_start, member_cw_pending, member_cw_table_words,
)
from core.member_families import (
    MEMBER_FAMILY_LOGISTIC, MEMBER_FAMILY_XLINEAR, MEMBER_FAMILY_SGD, MEMBER_FAMILY_LINEAR_SVM,
    MEMBER_FAMILY_KERNEL_SVM,
)


def member_cw_kind(class_weight: PythonObject, b: PythonObject) raises -> Int:
    """MEMBER_CW_NONE (None), MEMBER_CW_BALANCED (an exact str 'balanced'),
    MEMBER_CW_DICT (an exact dict), -1 (anything else: not admitted)."""
    if class_weight is None:
        return MEMBER_CW_NONE
    var kind = py_type(class_weight)
    if kind is b.str:
        return MEMBER_CW_BALANCED if String(class_weight) == "balanced" else -1
    if kind is b.dict:
        return MEMBER_CW_DICT
    return -1


def _svc_values(class_weight: PythonObject, classes: PythonObject, b: PythonObject) raises -> List[Float64]:
    """`svm_weights_unpack`'s dict loop: 1.0 per class, then each entry's
    `float(value)` at its class; a key outside the classes is refused in its
    words."""
    var k = len(classes)
    var values = List[Float64](length=k, fill=1.0)
    var items = py_call(b.list, py_call(py_call(b.dict, class_weight).items))
    for i in range(len(items)):  # driver-loop: over the user's class_weight entries
        var key = items[i][0]
        if not py_contains(classes, key):
            raise Error("mojolearn SVC: class_weight names the label " + family_repr(key)
                        + ", which is not in y's classes " + family_repr(classes))
        values[Int(py=py_call(classes.index, key))] = svc_float(items[i][1])
    return values^


def member_cw_fill(mut owner: MemberOwner, model: PythonObject, fields: PythonObject, labels: PythonObject,
    context: PythonObject, b: PythonObject) raises -> Bool:
    """The caller's UNPACK (module note): True when the member's words are
    complete (no tail, 'balanced', or the dict table now read over
    `labels`, the caller's classes in code order); False for a pending tail
    without labels. Raises the estimator's own words for a dict it
    refuses. `context` is the member's family context."""
    var start = member_cw_start(owner.plan.family, owner.plan.kind, owner.plan.params)
    if not member_cw_pending(owner.plan.params, start):
        return True
    if labels is None:
        return False
    var class_weight = py_get(fields, PythonObject("class_weight"), PythonObject(None))
    var family = owner.plan.family
    var listed = py_call(b.list, labels)
    var values = List[Float64]()
    if family == MEMBER_FAMILY_LOGISTIC:
        var scope = labels_scope(model, context)
        class_weight_admit_dictionary(class_weight, scope)
        listed = classifier_label_list(labels, scope)
        var inputs = class_weight_dictionary_values(class_weight, listed, scope)
        values = inputs.values.copy()
        _ = inputs^
    elif family == MEMBER_FAMILY_LINEAR_SVM:
        var inputs = class_weight_dictionary_values[CLASS_WEIGHT_SVC](class_weight, listed, context)
        values = inputs.values.copy()
        _ = inputs^
    elif family == MEMBER_FAMILY_KERNEL_SVM:
        values = _svc_values(class_weight, listed, b)
    elif family == MEMBER_FAMILY_SGD or family == MEMBER_FAMILY_XLINEAR:
        var got = xl_class_weights_unpack(class_weight, listed, context)
        var words = got[0]
        var k = len(listed)
        var at = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=labels_address(words))
        values = List[Float64](length=k, fill=1.0)
        for c in range(k):  # driver-loop: over the classes (the door's float32 words)
            values[c] = Float64(at[c])
        _ = words  # keep the words' owner alive through the reads above
    else:
        return False
    var named = List[Bool](length=len(values), fill=False)
    for c in range(len(values)):  # driver-loop: over the caller's classes (one dict lookup each)
        named[c] = py_contains(class_weight, listed[c])
    member_cw_table_words(owner.plan.params, start, values, named)
    _ = listed
    return True
