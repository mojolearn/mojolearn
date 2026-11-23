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
  * `member_cw_fill_keyed` (the UNPACK of a caller whose class labels are
    its target's numeric values, known only in RUN: IterativeImputer's
    classifier steps): the pending tail becomes the KEYED tail, each entry's
    weight read by the same family admission over the dict's own keys (so
    no key is absent there). A key that is not an exact int, float or bool
    equals none of a step's numeric classes: after the family's own key
    admission (its TypeErrors) the tail has no entry and is marked absent,
    so every classifier step's fit refuses it in the family's absent words,
    where the step's own fit refused it (lane mf-left4).
  * INVALID VALUES (lane mf-left4): a class_weight that is neither None,
    'balanced' nor an exact dict, and that the estimator's own fit door
    refuses (`member_cw_invalid_message`: the door's condition and words),
    is admitted with the MEMBER_CW_INVALID tail; `member_cw_fill_plan`
    (every FIT caller's UNPACK) raises those words before any fit. A value
    the door accepts that the member does not read as words (a str subclass
    equal to 'balanced', a dict subclass) is not admitted (the caller's own
    path, as before). Before this lane every invalid value took the
    caller's door path, whose first member fit raised the same words.

SERIAL BY NATURE: one dictionary read per member admission (a few entries).
"""
from std.python import Python, PythonObject
from bindings.class_weight_boundary_unpack import (
    CLASS_WEIGHT_SVC, class_weight_admit_dictionary, class_weight_dictionary_values,
)
from bindings.label_admission_native_unpack import labels_primitive_kind
from bindings.logistic_config_boundary_unpack import logistic_scalar_text
from bindings.svc_linear_boundary_unpack import _class_weight_message
from bindings.classical_labels_door import labels_address
from bindings.family_common_unpack import family_repr
from bindings.label_admission_native_pack import classifier_label_list
from bindings.labels_descriptor_boundary_door import labels_scope
from bindings.member_owner_unpack import MemberOwner
from bindings.py_balanced import py_call, py_contains, py_get, py_type
from bindings.svc_common_unpack import svc_float
from bindings.x_linear_common_unpack import xl_class_weights_unpack
from core.class_weight_plan import class_weight_dict_refused
from core.member_abi import MemberPlan
from core.member_class_weight import (
    MEMBER_CW_NONE, MEMBER_CW_BALANCED, MEMBER_CW_DICT, MEMBER_CW_INVALID, member_cw_start, member_cw_pending,
    member_cw_mode, member_cw_table_words, member_cw_keyed_words,
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


def member_cw_kind_accepted(family: Int, model: PythonObject, class_weight: PythonObject,
    b: PythonObject) raises -> Int:
    """`member_cw_kind`, and (lane mf-split4) a str or dict SUBCLASS that the
    family's fit door accepts (its condition: `member_cw_invalid_message`
    answers ""; LinearSVC's door takes exact types only): MEMBER_CW_BALANCED
    for a str subclass whose value is 'balanced', MEMBER_CW_DICT for a dict
    subclass, whose table `member_cw_fill_plan` / `member_cw_fill_keyed`
    read from ONE exact dict copy of it. -1: not admitted."""
    var kind = member_cw_kind(class_weight, b)
    if kind >= 0 or family == MEMBER_FAMILY_LINEAR_SVM:
        return kind
    if member_cw_invalid_message(family, model, class_weight, b).byte_length() != 0:
        return -1
    if Bool(py=py_call(b.isinstance, class_weight, b.str)):
        return MEMBER_CW_BALANCED if String(class_weight) == "balanced" else -1
    if Bool(py=py_call(b.isinstance, class_weight, b.dict)):
        return MEMBER_CW_DICT
    return -1


def _exact_dict(class_weight: PythonObject, b: PythonObject) raises -> PythonObject:
    """A dict subclass's ONE exact copy (lane mf-split4); an exact dict as it is."""
    if class_weight is None or py_type(class_weight) is b.dict:
        return class_weight
    return py_call(b.dict, class_weight)


def _svc_values(class_weight: PythonObject, classes: PythonObject, b: PythonObject) raises -> List[Float64]:
    """`svm_weights_unpack`'s dict loop: 1.0 per class, then each entry's
    `float(value)` at its class; a key outside the classes is refused in its
    words when a class is also left unweighted (`class_weight_dict_refused`),
    else ignored."""
    var k = len(classes)
    var values = List[Float64](length=k, fill=1.0)
    var named = List[Bool](length=k, fill=False)
    var unknown = 0
    var first_unknown = PythonObject(None)
    var items = py_call(b.list, py_call(py_call(b.dict, class_weight).items))
    for i in range(len(items)):  # driver-loop: over the user's class_weight entries
        var key = items[i][0]
        if not py_contains(classes, key):
            if unknown == 0:
                first_unknown = key
            unknown += 1
            continue
        var c = Int(py=py_call(classes.index, key))
        values[c] = svc_float(items[i][1])
        named[c] = True
    if unknown > 0:
        var unweighted = 0
        for c in range(k):  # driver-loop: over the classes (the dict's unnamed ones)
            if not named[c]:
                unweighted += 1
        if class_weight_dict_refused(unknown, unweighted):
            raise Error("mojolearn SVC: class_weight names the label " + family_repr(first_unknown)
                        + ", which is not in y's classes " + family_repr(classes))
    return values^


def member_cw_invalid_message(family: Int, model: PythonObject, class_weight: PythonObject,
    b: PythonObject) raises -> String:
    """The words the estimator's own fit door raises for `class_weight`
    (a value `member_cw_kind` does not admit), with that door's condition;
    "" when the door accepts it (the member then keeps the caller's own path
    for it). Read from the doors: LogisticRegression
    bindings/logistic_native_door.mojo `_row_weights` (isinstance str /
    dict); LinearSVC bindings/svc_linear_boundary_unpack.mojo
    `_qn_row_weights` (exact types, `_class_weight_message`); SVC
    bindings/svc_kernel_params_unpack.mojo `svm_weights_unpack` (a str other
    than 'balanced'; a value neither str nor dict: the settings admission's
    words, `svm_settings_unpack`, which the member's admit now leaves to this
    tail, lane mf-split4b); the x_linear kinds
    bindings/x_linear_common_unpack.mojo `xl_class_weights_unpack` (equal to
    'balanced', or isinstance dict; `model` is not read there)."""
    var is_str = Bool(py=py_call(b.isinstance, class_weight, b.str))
    var is_dict = Bool(py=py_call(b.isinstance, class_weight, b.dict))
    if family == MEMBER_FAMILY_LOGISTIC:
        # The door's fit admits every consumed value first
        # (bindings/logistic_admission_unpack.mojo `_parameter` through
        # `logistic_fields`): a value that is neither a mapping nor a native
        # scalar (a list, a tuple, an array: a class with `__len__`, read on
        # the class, never the instance) raises its TypeError there and never
        # reaches `_row_weights` (gate 5645984a8f mf_left4 inv_lr_2_* rows).
        if (not is_str and not is_dict and py_type(class_weight) is not b.bool and py_type(class_weight) is not b.int
                and py_type(class_weight) is not b.float
                and Bool(py=py_call(b.hasattr, py_type(class_weight), PythonObject("__len__")))):
            return String("TypeError: LogisticRegression parameter class_weight requires native scalar configuration "
                          + "without conversion hooks")
        if (is_str and Bool(py=class_weight != PythonObject("balanced"))) or (not is_str and not is_dict):
            return ("The 'class_weight' parameter of " + String(py=py_type(model).__name__) + " must be an "
                    + "instance of 'dict', a str among {'balanced'} or None. "
                    + "Got " + logistic_scalar_text(class_weight) + " instead.")
        return String("")
    if family == MEMBER_FAMILY_LINEAR_SVM:
        var kind = py_type(class_weight)
        if (kind is b.str and String(class_weight) != "balanced") or (kind is not b.str and kind is not b.dict):
            return _class_weight_message(model, class_weight)
        return String("")
    if family == MEMBER_FAMILY_KERNEL_SVM:
        if is_str and String(class_weight) != "balanced":
            return "mojolearn SVC: class_weight is a dict, 'balanced' or None, got " + family_repr(class_weight)
        if not is_str and not is_dict:
            # The settings admission's words (bindings/svc_kernel_params_unpack.mojo
            # `svm_settings_unpack`: the value's type name), lane mf-split4b.
            return ("mojolearn SVC: class_weight is a dict, 'balanced' or None, got "
                    + String(py_type(class_weight).__name__))
        return String("")
    if family == MEMBER_FAMILY_SGD or family == MEMBER_FAMILY_XLINEAR:
        if not Bool(py=class_weight == PythonObject("balanced")) and not is_dict:
            return String("mojolearn: class_weight must be None, 'balanced' or a dict")
        return String("")
    return String("")


def member_cw_fill(mut owner: MemberOwner, model: PythonObject, fields: PythonObject, labels: PythonObject,
    context: PythonObject, b: PythonObject) raises -> Bool:
    """`member_cw_fill_plan` over the owner's plan."""
    return member_cw_fill_plan(owner.plan, model, fields, labels, context, b)


def member_cw_fill_plan(mut plan: MemberPlan, model: PythonObject, fields: PythonObject, labels: PythonObject,
    context: PythonObject, b: PythonObject) raises -> Bool:
    """The caller's UNPACK (module note): True when the member's words are
    complete (no tail, 'balanced', or the dict table now read over
    `labels`, the caller's classes in code order); False for a pending tail
    without labels. Raises the estimator's own words for a dict it
    refuses, and for an invalid value (MEMBER_CW_INVALID: with or without
    labels, before any fit). `context` is the member's family context."""
    var start = member_cw_start(plan.family, plan.kind, plan.params)
    if not member_cw_pending(plan.params, start):
        return True
    var class_weight = py_get(fields, PythonObject("class_weight"), PythonObject(None))
    if member_cw_mode(plan.params, start) == MEMBER_CW_INVALID:
        raise Error(member_cw_invalid_message(plan.family, model, class_weight, b))
    if labels is None:
        return False
    class_weight = _exact_dict(class_weight, b)
    var listed = py_call(b.list, labels)
    var values = List[Float64]()
    if not _dict_values(plan.family, model, class_weight, listed, context, b, values):
        return False
    var named = List[Bool](length=len(values), fill=False)
    for c in range(len(values)):  # driver-loop: over the caller's classes (one dict lookup each)
        named[c] = py_contains(class_weight, listed[c])
    member_cw_table_words(plan.params, start, values, named)
    _ = listed
    return True


def member_cw_fill_keyed(mut owner: MemberOwner, model: PythonObject, fields: PythonObject, context: PythonObject,
    b: PythonObject) raises -> Bool:
    """The UNPACK of a caller whose labels are its target's values (module
    note): True when the words are complete (no tail, 'balanced', or the
    KEYED tail now read); False for a family this does not read. Raises the
    estimator's own words for an invalid value, for a dict value it refuses
    and for a key its key admission refuses: the caller keeps them for its
    RUN's first classifier fit, where the estimator's own fit raised them."""
    var start = member_cw_start(owner.plan.family, owner.plan.kind, owner.plan.params)
    if not member_cw_pending(owner.plan.params, start):
        return True
    var class_weight = py_get(fields, PythonObject("class_weight"), PythonObject(None))
    if member_cw_mode(owner.plan.params, start) == MEMBER_CW_INVALID:
        raise Error(member_cw_invalid_message(owner.plan.family, model, class_weight, b))
    class_weight = _exact_dict(class_weight, b)
    var keys = py_call(b.list, class_weight)
    var numbers = List[Float64](capacity=len(keys))
    var foreign = False
    for e in range(len(keys)):  # driver-loop: over the dict's keys (exact native numbers only)
        var kind = py_type(keys[e])
        if kind is not b.int and kind is not b.float and kind is not b.bool:
            foreign = True
            break
        numbers.append(Float64(py=py_call(b.float, keys[e])))
    if foreign:
        # A key no numeric class equals: every classifier step's fit refuses
        # the dict as absent from its y, after the family's own key
        # admission, which runs first in its fit (its TypeErrors raise here).
        _keys_admit(owner.plan.family, model, class_weight, keys, context, b)
        member_cw_keyed_words(owner.plan.params, start, List[Float64](), List[Float64](), True)
        _ = keys
        return True
    var values = List[Float64]()
    if not _dict_values(owner.plan.family, model, class_weight, keys, context, b, values):
        return False
    member_cw_keyed_words(owner.plan.params, start, numbers, values, False)
    _ = keys
    return True


def _keys_admit(family: Int, model: PythonObject, class_weight: PythonObject, keys: PythonObject,
    context: PythonObject, b: PythonObject) raises:
    """The key checks each family's fit makes before its absent test
    (module note): LogisticRegression `class_weight_admit_dictionary` (keys
    and values native primitives), LinearSVC the SVC profile's key test of
    `class_weight_dictionary_values`; SVC and the x_linear kinds test no key
    type before it."""
    if family == MEMBER_FAMILY_LOGISTIC:
        class_weight_admit_dictionary(class_weight, labels_scope(model, context))
    elif family == MEMBER_FAMILY_LINEAR_SVM:
        for i in range(len(keys)):  # driver-loop: over the dict's keys (exact key UNPACK, before hashing)
            if labels_primitive_kind(py_type(keys[i]), context) == 0:
                raise Error("TypeError: class_weight keys must be native numbers or strings")


def _dict_values(family: Int, model: PythonObject, class_weight: PythonObject, var listed: PythonObject,
    context: PythonObject, b: PythonObject, mut values: List[Float64]) raises -> Bool:
    """The dict's weights over `listed` (labels in code order), in the
    family's own admission and words (module note). False: a family that
    takes no class_weight dict."""
    if family == MEMBER_FAMILY_LOGISTIC:
        var scope = labels_scope(model, context)
        class_weight_admit_dictionary(class_weight, scope)
        listed = classifier_label_list(listed, scope)
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
    _ = listed
    return True
