# SPDX-License-Identifier: Apache-2.0
# The LDA owner reads only connected native class storage and admitted values.
from std.math import isfinite, floor
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.classical_attributes import classical_check_layout
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.classical_pandas import classical_admit_value
from bindings.classical_labels import _flatten_labels, labels_shape_of
from bindings.classical_exceptions import classical_not_fitted_class
from bindings.classical_metadata import classical_is_fitted
from bindings.optimizer_public_words import stash_python_error
from bindings.metrics_native_support import metric_float, metric_is_real, metric_is_integral, metric_warn
from bindings.classical_warnings import COLUMN_WARNING, _warning_category
from bindings.label_integer_storage import labels_integer_storage, labels_float_storage
from bindings.label_input_sources import labels_storage_source, labels_list_integer_storage, labels_text_storage
from bindings.native_typed_context import native_typed_context
from bindings.native_loaded_module import NativeModules
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_key_add, prepared_key_same
from bindings.native_model_records import (
    records_stamped_get, records_stamped_put, records_stampable, records_key_same_exact, ContentRecord,
    content_value, RECORDS_PREPARED,
)
from bindings.py_balanced import py_fields
from bindings.native_class_admission import class_record, class_admit, class_watch_ready
from bindings.label_admission_native import _labels_class as native_labels_class
from bindings.py_attr import py_attr
from bindings.native_model_records import bump_model_generation


def _lda_storage(model: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(model) is not py_get(context, "lda_native_class"):
        raise Error("TypeError: native LDA requires its connected class without Python subclass hooks")
    classical_check_layout(model, context["lda_admission_context"])
    return py_call(context["lda_fields_get"], model, py_type(model))


def _lda_parameter(name: PythonObject, value: PythonObject, context: PythonObject, configure: Bool = False) raises:
    # The constructor, get_params and set_params store values verbatim and
    # validate nothing beyond native admission (scikit-learn's convention,
    # check_do_not_raise_errors_in_init_or_set_params); the refusal below
    # runs when a consuming operation (fit, predict, ...) reads the value.
    if String(name) == "covariance_estimator" and value is not None:
        if configure:
            # Stored and returned as the same reference; nothing reads it
            # (no attribute, protocol or call), so no Python runs. A consuming
            # operation refuses it by name below.
            return
        # There is no compiled covariance-estimator object API in this
        # package. Calling sklearn/custom fit from Mojo would execute Python.
        raise Error("TypeError: covariance_estimator cannot invoke Python callbacks; use covariance_estimator=None and the native solver/shrinkage configuration")
    classical_admit_value(value, context)


def lda_native_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _lda_storage(model, context)
    var names = context["lda_parameters"]
    if len(args) != len(names):
        raise Error("TypeError: native LDA initializer requires its complete constructor configuration")
    for i in range(len(names)):  # driver-loop: seven unchanged constructor values, before publication
        _lda_parameter(names[i], args[i], context, True)
    for i in range(len(names)):  # driver-loop: native instance dictionary writes, no __setattr__ hooks
        py_setitem(fields, names[i], args[i])
    return PythonObject(None)


comptime _LABELS_CLASS_SLOT = "prep_labels_class_checked_v1"
comptime _LDA_FIELDS_SLOT = "prep_lda_fields_admitted_v1"
comptime _LDA_STAMP = "prep_lda_fields_stamp_v1"


def _lda_stamp(model: PythonObject, fields: PythonObject, names: PythonObject, fitted: PythonObject,
               plain: PythonObject, configure: Bool) raises:
    """Stamp the admitted LDA fields with the model's generation (lane
    ne1-e-records, row N258). The generation proves every stored name's
    identity and presence; the content words cover exact containers item by
    item (and other values' classes). Only when `fields` is the model's own
    instance dictionary (the record holds it)."""
    if not (py_fields(model) is fields) or not records_stampable(model):
        return
    var content = ContentRecord()
    for i in range(len(names)):  # driver-loop: fixed constructor configuration (content words, once per record)
        content_value(content, fields[names[i]], 0, 5, False)
    for i in range(len(fitted)):  # driver-loop: consumed fitted state (content words, once per record)
        if py_contains(fields, fitted[i]):
            content_value(content, fields[fitted[i]], 0, 5, False)
    _ = records_stamped_put(RECORDS_PREPARED, model, _LDA_STAMP, content^,
                            Python.tuple(fields, names, fitted, plain, PythonObject(configure)))


def _lda_stamped(model: PythonObject, fields: PythonObject, names: PythonObject, fitted: PythonObject,
                 configure: Bool, context: PythonObject) raises -> PythonObject:
    """`(fields, plain)` from the stamped record, or None (run the key path).
    `plain` is derived from `classes_` again unless the record's `classes_`
    is itself the plain list: an ndarray's elements can change in place, so
    the derived labels are compared as the key path compares them."""
    var stamped = records_stamped_get(RECORDS_PREPARED, model, _LDA_STAMP)
    if stamped is None or not (stamped[0] is fields) or not (stamped[1] is names) or not (stamped[2] is fitted):
        return PythonObject(None)
    if Bool(py=stamped[4]) != configure:
        return PythonObject(None)
    var kept = stamped[3]
    if kept is None or not py_contains(fields, "classes_"):
        return Python.tuple(fields, kept)
    var value = fields["classes_"]
    if kept is value:
        return Python.tuple(fields, kept)
    var again = PythonObject(None)
    try:
        again = prep_plain_classes(value, context)
    except:
        # The key path raises it at its original position.
        return PythonObject(None)
    var list_t = py_type(Python.list())
    if py_type(kept) is not list_t or py_type(again) is not list_t or not records_key_same_exact(kept, again):
        return PythonObject(None)
    return Python.tuple(fields, again)


def prep_plain_classes(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`classifier_label_list`: the plain list copy of a ClassLabels
    `classes_` (any other value as is). The ClassLabels class check
    (ancestry and namespace against the import-time connection) runs once
    per class state: the result is kept in the class-admission registry
    (type-watcher epoch; any class or ancestor assignment drops it; Python
    3.10/3.11 keep no record and check every call, as before) (perf2-prep)."""
    var cls = py_get(context, "classifier_labels_class")
    if cls is None or py_type(value) is not cls:
        return value
    var connected = context["classifier_labels_namespace"]
    var mods = NativeModules()
    var record = class_record(mods, _LABELS_CLASS_SLOT, cls)
    if record is None or record[0] is not connected:
        _ = native_labels_class(context)
        if class_watch_ready():  # no payload where no record can be kept (3.10/3.11)
            class_admit(mods, _LABELS_CLASS_SLOT, cls, Python.tuple(connected))
    # list.copy is the native list implementation, not an iterable protocol.
    return py_call(py_attr(py_type(Python.list()), "copy"), value)


def lda_native_fields_plain(model: PythonObject, context: PythonObject, configure: Bool = False) raises -> PythonObject:
    """(fields, plain classes_ or None): `lda_native_fields` plus the plain
    `classes_` list it admitted, so the caller does not copy it again.

    The 7 constructor values and up to 18 fitted fields were each re-admitted
    (`classical_admit_value`) on every predict, transform and score. Now a
    per-model record in the prepared-owner registry (keyed weakly by the
    model, outside its `__dict__`) holds the identity key of what the last
    full admission passed: each field name, the value object and, for exact
    list/tuple/dict values, their items (bindings/native_prepared_owner.mojo
    `prepared_key_add`), and the items of the plain `classes_`. A call whose
    key is the same skips the admissions; any set_params, refit, attribute
    reassignment, container edit or `classes_` edit misses and admits in full
    (perf2-prep). Not noticed: an element written in place into an ndarray or
    Array value (its admission is of the object, its dtype and shape, which
    such a write does not change), or an in-place dtype/shape reassignment of
    an ndarray. The covariance_estimator refusal runs on every call."""
    var fields = _lda_storage(model, context)
    var names = context["lda_parameters"]
    var hit = _lda_stamped(model, fields, names, context["lda_fitted_fields"], configure, context)
    if hit is not None:
        return hit
    var mods = NativeModules()
    var key = Python.list()
    # A missing parameter or a covariance_estimator refusal takes the full
    # path below, which raises at the original position and order.
    var full = False
    for i in range(len(names)):  # driver-loop: fixed constructor configuration identities
        if not py_contains(fields, names[i]):
            full = True
            break
        var value = fields[names[i]]
        if not configure and String(names[i]) == "covariance_estimator" and value is not None:
            full = True
            break
        _ = py_call(key.append, names[i])
        prepared_key_add(mods, key, value)
    var fitted = context["lda_fitted_fields"]
    var plain = PythonObject(None)
    for i in range(len(fitted)):  # driver-loop: consumed fitted state identities
        if full:
            break
        if py_contains(fields, fitted[i]):
            var value = fields[fitted[i]]
            _ = py_call(key.append, fitted[i])
            if String(fitted[i]) == "classes_":
                try:
                    plain = prep_plain_classes(value, context)
                except:
                    # Refused by the class check: the full path raises it at
                    # its original position, after the parameters.
                    full = True
                if full:
                    break
                _ = py_call(key.append, value)
                if plain is not value:
                    for k in range(len(plain)):  # driver-loop: class labels of the fitted classes_ (identity key)
                        _ = py_call(key.append, plain[k])
                else:
                    prepared_key_add(mods, key, value)
            else:
                prepared_key_add(mods, key, value)
    var held = PythonObject(None) if full else prepared_get(mods, _LDA_FIELDS_SLOT, model)
    if held is not None and prepared_key_same(held, key):
        _lda_stamp(model, fields, names, fitted, plain, configure)
        return Python.tuple(fields, plain)
    for i in range(len(names)):  # driver-loop: fixed constructor configuration, before comparisons or conversion
        if not py_contains(fields, names[i]):
            raise Error("AttributeError: LDA constructor parameter is missing: " + String(names[i]))
        _lda_parameter(names[i], fields[names[i]], context, configure)
    for i in range(len(fitted)):  # driver-loop: consumed fitted state cannot introduce attribute or scalar hooks
        if py_contains(fields, fitted[i]):
            if String(fitted[i]) == "classes_":
                if plain is None:
                    plain = prep_plain_classes(fields[fitted[i]], context)
                classical_admit_value(plain, context)
            else:
                classical_admit_value(fields[fitted[i]], context)
    if not full:
        prepared_put(mods, _LDA_FIELDS_SLOT, model, key)
        _lda_stamp(model, fields, names, fitted, plain, configure)
    return Python.tuple(fields, plain)


def lda_native_fields(model: PythonObject, context: PythonObject, configure: Bool = False) raises -> PythonObject:
    return lda_native_fields_plain(model, context, configure)[0]


def lda_native_get_params(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = Python.dict()
    var names = context["lda_parameters"]
    for i in range(len(names)):  # driver-loop: the fixed constructor parameter schema
        py_setitem(result, names[i], fields[names[i]])
    return result


def lda_native_set_params(model: PythonObject, fields: PythonObject, values: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    # Generation word (bindings/native_model_records.mojo, lane ne1-e-records):
    # every stamped record of this object misses from here on.
    _ = bump_model_generation(model)
    if py_type(values) is not py_type(Python.dict()):
        raise Error("TypeError: native LDA parameter updates require an ordinary dictionary")
    var names = context["lda_parameters"]
    var keys = py_call(py_type(Python.list()), values)
    for i in range(len(keys)):  # driver-loop: argument names before equality or conversion
        if py_type(keys[i]) is not py_type(PythonObject("")):
            raise Error("TypeError: LDA parameter names must be native strings")
    for i in range(len(keys)):  # driver-loop: preserve the existing ordered parameter update
        var name = keys[i]
        if not py_contains(names, name):
            raise Error("mojolearn: invalid parameter " + String(py_call(native_loaded_module("builtins").repr, name))
                        + " for LinearDiscriminantAnalysis")
        _lda_parameter(name, values[name], context, True)
        py_setitem(fields, name, values[name])
    return model


def lda_native_labels(value: PythonObject, context: PythonObject, fitting: Bool = False, name: String = "LinearDiscriminantAnalysis") raises:
    if fitting and value is None:
        raise Error(name + " requires y to be passed, but the target y is None.")
    classical_admit_value(value, context)
    # The exact ndarray / Array itself, or the stored ndarray of a numpy-backed
    # pandas Series (1-D, its labels in the same order, so the same refusals at
    # the same first label and never the column warning, as the walk), else
    # None (perf3-label-doors; bindings/label_input_sources.mojo).
    var source = labels_storage_source(value, context)
    # An exact integer/bool ndarray or Array holds no element any check below
    # can refuse (tolist yields exact ints/bools; the float loop never runs for
    # it), so it is admitted from its dtype metadata, not label by label: the
    # flatten + per-label admission cost scales with the rows of every
    # LDA/QDA/NB/logistic/forest/GBDT-adapter classifier fit. Same refusals,
    # same warning, nothing returned, so no bits change.
    if source is not None and labels_integer_storage(source, context):
        if fitting:
            var shape = labels_shape_of(source, context)
            if len(shape) == 2 and shape[1] == 1:
                metric_warn(COLUMN_WARNING, _warning_category(context), context)
        return
    # An exact float32/float64 ndarray or Array holds only floats, so the one
    # check below that can refuse an element (NaN/infinity, then non-integral)
    # reads the buffer in one native pass (bindings/label_integer_storage.mojo,
    # core/label_storage_impl.mojo) instead of tolist + admission + three
    # scalar calls per label (~0.15 s per 20000-label fit). Same warning, same
    # refusals at the same first label, nothing returned: no bits change.
    var float_refusal = -1
    if source is not None:
        float_refusal = labels_float_storage(source, context)
    if float_refusal >= 0:
        if fitting:
            var shape = labels_shape_of(source, context)
            if len(shape) == 2 and shape[1] == 1:
                metric_warn(COLUMN_WARNING, _warning_category(context), context)
            if float_refusal == 1:
                raise Error("Input y contains NaN or infinity (" + name + ").")
            if float_refusal == 2:
                raise Error("Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.")
        return
    # An exact flat list/tuple of exact ints (or bools): every label is admitted
    # and the float loop below never fires for it (its first label is
    # integral), and a list/tuple never takes the column warning. Read through
    # the container's C storage, one type compare per label.
    if labels_list_integer_storage(value):
        return
    # An exact '<U' ndarray lists exact str only: admitted, and not numeric
    # float, so the walk refuses nothing; only its column warning remains.
    if labels_text_storage(value, context):
        if fitting:
            var shape = labels_shape_of(value, context)
            if len(shape) == 2 and shape[1] == 1:
                metric_warn(COLUMN_WARNING, _warning_category(context), context)
        return
    # An exact object ndarray may still hold arbitrary Python scalar objects.
    # Reading/flattening the references is native; admission precedes hashing,
    # sorting, equality and numeric conversion in the label encoder.
    var flat = _flatten_labels(value, context)
    classical_admit_value(flat, context)
    if fitting:
        var b = native_loaded_module("builtins")
        var shape = labels_shape_of(value, context)
        if len(shape) == 2 and shape[1] == 1 and py_type(value) is not b.list and py_type(value) is not b.tuple and py_type(value) is not b.memoryview:
            metric_warn(COLUMN_WARNING, _warning_category(context), context)
        var numeric_float = True
        if py_type(value) is context["array_class"]:
            numeric_float = String(value.dtype).startswith("<f")
        elif py_type(value) is py_get(context, "ndarray"):
            numeric_float = String(value.dtype.kind) == "f"
        elif py_type(value) is b.list or py_type(value) is b.tuple:
            numeric_float = len(flat) > 0 and metric_is_real(flat[0], context) and not metric_is_integral(flat[0], context)
        # Exact int / bool / float labels are classified by type pointer and
        # a float read natively (p3-prep); other classes keep the metric
        # real / integral checks. Same refusals at the same first label.
        var int_type = b.int
        var bool_type = b.bool
        var float_type = b.float
        for i in range(len(flat) if numeric_float else 0):  # driver-loop: preserve the public classifier target admission before encoding
            var item = flat[i]
            var cls = py_type(item)
            if cls is int_type or cls is bool_type:
                continue
            var number: Float64
            if cls is float_type:
                number = Float64(py=item)
            elif metric_is_real(item, context) and not metric_is_integral(item, context):
                number = metric_float(item, context)
            else:
                continue
            if not isfinite(number):
                raise Error("Input y contains NaN or infinity (" + name + ").")
            if floor(number) != number:
                raise Error("Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.")


def lda_require_fitted(model: PythonObject, operation: String, context: PythonObject) raises:
    if not Bool(py=classical_is_fitted(model)):
        var cls = classical_not_fitted_class(context)
        var message = "This LinearDiscriminantAnalysis instance is not fitted yet. Call 'fit' with appropriate arguments before using " + operation + "."
        ref cpy = Python().cpython()
        cpy.PyErr_SetString(cls._obj_ptr, message.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin())
        var failure = stash_python_error()
        raise failure^


def lda_native_context(context: PythonObject) raises -> PythonObject:
    # The established label algorithms use isinstance against their connected
    # real/integral classes. Concrete native type tuples preserve those tests
    # without executing numbers.Real/Integral's Python ABC machinery. Built
    # once per connected class context (bindings/native_typed_context.mojo),
    # not copied and rebuilt on every public call.
    return native_typed_context(context, True, False)
