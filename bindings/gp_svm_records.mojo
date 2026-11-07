# SPDX-License-Identifier: Apache-2.0
# Per-model admitted-state records of the Gaussian process, kernel SVM,
# OneClassSVM and kernel-method public doors (lane perf2-gp-svm, 2026-11-06).
#
# Each predict-side public call of these estimators re-admitted the whole
# fitted state: a class-ancestry scan for Python fit callbacks, about sixteen
# scalar fields, seven array fields, two kernel-tree snapshots, the class list
# and, for the one-vs-rest classifier, every member's arrays; the decoders
# rebuilt their class tables; the GP owners repacked the kernel. Those results
# depend only on objects the model already holds, so they are kept here:
#
# - Class-state checks (the fit-callback scan, the binary storage class) use
#   the per-class records of bindings/native_class_admission.mojo (CPython
#   type watchers; Python 3.10/3.11 run the full scan every call, as before).
# - Model records live in the GP_SVM slot of the one record table
#   (bindings/native_model_records.mojo, lane ne1-e-records), keyed by a
#   weak reference to the model, outside its `__dict__`: predict leaves
#   `__dict__` unchanged, pickles and copies never carry a record, and a copy
#   or an unpickled model rebuilds its record on first use. A refit drops it.
# - A record is valid while its key matches: the identities of every object
#   the skipped admission read (exact containers item by item, each non-scalar
#   with its class). Values are compared by identity (exact ints by value), so
#   no user `__eq__`/`__hash__` runs. A value whose admission can depend on
#   more than its identity and class (a 0-d ndarray scalar field, a torch or
#   container-subclass array field) is admitted again on every call.
#
# Not noticed (as before: the full admissions never read them either): in-place
# writes into a fitted ndarray's elements. Nothing here computes on data or
# changes an arithmetic or fold order.
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.native_loaded_module import NativeModules
from bindings.native_class_admission import class_record, class_admit
from bindings.trees_side_records import trees_key_shallow
from bindings.native_model_records import records_get, records_put, records_drop, records_key_append, RECORDS_GP_SVM
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.mode_metadata_native import _class_fields
from bindings.classical_attributes import _type_mro


comptime _CALLBACK_SLOT = "gpsvm_no_fit_callback_class_v1"


# ------------------------------------------------------------ class state


def gpsvm_no_fit_callbacks(model: PythonObject, fields: PythonObject, message: String) raises:
    """Refuse a model whose class ancestry or instance defines the Python fit
    callback `_mojolearn_after_fit` (`message` is the refusal). The ancestry
    scan runs once per class state; the instance field is read every call."""
    var mods = NativeModules()
    var cls = py_type(model)
    if class_record(mods, _CALLBACK_SLOT, cls) is None:
        var ancestry = _type_mro(cls)
        for i in range(len(ancestry)):  # driver-loop: no completion callback may execute from a native operation
            if py_contains(_class_fields(ancestry[i]), "_mojolearn_after_fit"):
                raise Error(message)
        # The payload never retains the class (a weak-keyed record).
        class_admit(mods, _CALLBACK_SLOT, cls, PythonObject(True))
    if py_contains(fields, "_mojolearn_after_fit"):
        raise Error(message)


# ------------------------------------------------------------ storage


def gpsvm_side_get(slot: String, owner: PythonObject) raises -> PythonObject:
    """The GP / SVM / kernel-method record `slot` holds for `owner`, or None."""
    return records_get(RECORDS_GP_SVM, owner, slot)


def gpsvm_side_put(slot: String, owner: PythonObject, record: PythonObject) raises -> Bool:
    """Store `record` for `owner`; False when `owner` takes no weak reference."""
    return records_put(RECORDS_GP_SVM, owner, slot, record)


def gpsvm_side_drop(slot: String, owner: PythonObject) raises:
    """Forget `owner`'s record in `slot` now (a refit)."""
    records_drop(RECORDS_GP_SVM, owner, slot)


# ------------------------------------------------------------ helpers


def gpsvm_builtin_scalar(mut mods: NativeModules, value: PythonObject) raises -> Bool:
    var b = mods.module("builtins")
    var cls = py_type(value)
    return (cls is b.int or cls is b.float or cls is b.bool or cls is b.str or cls is b.bytes
            or value is None)


def gpsvm_key_fields(mut mods: NativeModules, key: PythonObject, fields: PythonObject, skip: List[String]) raises:
    """Append an exact field dictionary's entries (names, values, exact
    containers item by item, non-scalars with their class), except the
    call-written `skip` names, then the count of entries appended."""
    var b = mods.module("builtins")
    var items = py_call(b.list, py_call(fields.items))
    var count = 0
    for i in range(len(items)):  # driver-loop: over a fitted model's stored fields (identity only)
        var name = items[i][0]
        if py_type(name) is b.str:
            var text = String(name)
            var skipped = False
            for j in range(len(skip)):  # small-loop(skip: call-written field names)
                if skip[j] == text:
                    skipped = True
                    break
            if skipped:
                continue
        records_key_append(key, name)
        trees_key_shallow(mods, key, items[i][1], 0)
        count += 1
    records_key_append(key, PythonObject(count))


def gpsvm_context_key(key: PythonObject, context: PythonObject, names: List[String]) raises:
    """The admission context and the connected entries the skipped admission
    read, by identity."""
    records_key_append(key, context)
    for i in range(len(names)):  # small-loop(names: fixed connected context entries)
        records_key_append(key, py_get(context, PythonObject(names[i])))


def gpsvm_key_value(mut mods: NativeModules, key: PythonObject, residual: PythonObject, value: PythonObject,
                    context: PythonObject, depth: Int) raises -> Bool:
    """Append the identities `classical_admit_value(value)` reads: the value,
    exact list/tuple/set/frozenset/dict containers item by item (with their
    lengths), and every other value's class. An exact ndarray, Array or
    connected NumPy scalar is admitted by its class alone; any other value
    (RandomState, torch, container subclasses, ...) is also appended to
    `residual`, to be admitted again on every call. False when the nesting
    is too deep to record."""
    if depth > 64:
        return False
    records_key_append(key, value)
    if value is None:
        return True
    var b = mods.module("builtins")
    var cls = py_type(value)
    if cls is b.bool or cls is b.int or cls is b.float or cls is b.str or cls is b.bytes:
        return True
    if cls is b.dict:
        # items() of an exact dict never hashes or compares keys.
        var pairs = py_call(b.list, py_call(value.items))
        records_key_append(key, PythonObject(len(pairs)))
        for k in range(len(pairs)):  # driver-loop: stored mapping entries (identity only)
            if not gpsvm_key_value(mods, key, residual, pairs[k][0], context, depth + 1):
                return False
            if not gpsvm_key_value(mods, key, residual, pairs[k][1], context, depth + 1):
                return False
        return True
    if cls is b.list or cls is b.tuple or cls is b.set or cls is b.frozenset:
        # Iterating an exact container runs no item protocol.
        var items = py_call(b.list, value)
        records_key_append(key, PythonObject(len(items)))
        for k in range(len(items)):  # driver-loop: stored container items (identity only)
            if not gpsvm_key_value(mods, key, residual, items[k], context, depth + 1):
                return False
        return True
    records_key_append(key, cls)
    if cls is context["array_class"] or cls is py_get(context, "ndarray"):
        return True
    var scalars = py_get(context, "scalar_types")
    if scalars is not None and py_type(scalars) is b.tuple:
        for i in range(len(scalars)):  # driver-loop: over the connected NumPy scalar types, by identity
            if cls is scalars[i]:
                return True
    records_key_append(residual, value)
    return True


# ------------------------------------------------------------ class tables


def _label_same(mut mods: NativeModules, a: PythonObject, c: PythonObject) raises -> Bool:
    """The same label for a class table: the same object, or two exact
    builtin scalars of one type with equal value (floats bit for bit, so
    -0.0 and 0.0 differ). No user `__eq__` runs."""
    if a is c:
        return True
    var b = mods.module("builtins")
    var cls = py_type(a)
    if cls is not py_type(c):
        return False
    if cls is b.float:
        return bitcast[DType.uint64](Float64(py=a)) == bitcast[DType.uint64](Float64(py=c))
    if cls is b.int or cls is b.str or cls is b.bytes:
        return Bool(py=(a == c))
    return False


def _labels_recordable(mut mods: NativeModules, labels: PythonObject) raises -> Bool:
    """Only label lists of exact builtin scalars are recorded: other label
    objects (NumPy scalars made fresh per call) would miss every time."""
    for i in range(len(labels)):  # driver-loop: over the fitted classes
        if not gpsvm_builtin_scalar(mods, labels[i]):
            return False
    return True


def gpsvm_class_table(owner: PythonObject, slot: String, labels: PythonObject) raises -> PythonObject:
    """The class table recorded for `owner` under `slot` when `labels` (the
    native label list of `classes_`) holds the same labels; else None."""
    var held = gpsvm_side_get(slot, owner)
    if held is None:
        return PythonObject(None)
    var kept = held[0]
    if len(kept) != len(labels):
        return PythonObject(None)
    var mods = NativeModules()
    for i in range(len(labels)):  # driver-loop: over the fitted classes (identity or exact scalar value)
        if not _label_same(mods, kept[i], labels[i]):
            return PythonObject(None)
    return held[1]


def gpsvm_class_table_put(owner: PythonObject, slot: String, labels: PythonObject, table: PythonObject) raises:
    var mods = NativeModules()
    if not _labels_recordable(mods, labels):
        return
    var b = mods.module("builtins")
    _ = gpsvm_side_put(slot, owner, Python.tuple(py_call(b.list, labels), table))


# ------------------------------------------------------------ OneClassSVM


comptime _OCSVM_COEF_SLOT = "ocsvm_coef_snapshot_v2"


def _ocsvm_coef_layout(dual: PythonObject) raises -> PythonObject:
    """The object's own layout words: shape, dtype and strides. A NumPy array
    can be reinterpreted in place (`a.dtype = ...`, `a.shape = ...`,
    `a.resize(...)`) without becoming another object, so identity alone does
    not prove the snapshot is still the object itself (lane p3-svm-gp-kernel,
    row 953)."""
    return Python.tuple(dual.shape, dual.dtype, dual.strides)


def ocsvm_coef_reused(owner: PythonObject, dual: PythonObject) raises -> Bool:
    """`dual` is the coefficient object whose float32 snapshot was the object
    itself last call (no conversion), with the same shape, dtype and strides:
    the snapshot can be skipped."""
    var held = gpsvm_side_get(_OCSVM_COEF_SLOT, owner)
    if held is None or not (held[0] is dual) or not (held[1] is py_type(dual)):
        return False
    return Bool(py=held[2] == _ocsvm_coef_layout(dual))


def ocsvm_coef_remember(owner: PythonObject, dual: PythonObject, coef: PythonObject) raises:
    if coef is dual:
        _ = gpsvm_side_put(_OCSVM_COEF_SLOT, owner, Python.tuple(dual, py_type(dual), _ocsvm_coef_layout(dual)))
    else:
        gpsvm_side_drop(_OCSVM_COEF_SLOT, owner)
