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
# - Model records live in the prepared-owner registry
#   (bindings/native_prepared_owner.mojo, through trees_side_records), keyed
#   by a weak reference to the model, outside its `__dict__`: predict leaves
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
from bindings.trees_side_records import trees_side_get, trees_side_put, trees_side_drop, trees_key_shallow
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
        _ = py_call(key.append, name)
        trees_key_shallow(mods, key, items[i][1], 0)
        count += 1
    _ = py_call(key.append, PythonObject(count))


def gpsvm_context_key(key: PythonObject, context: PythonObject, names: List[String]) raises:
    """The admission context and the connected entries the skipped admission
    read, by identity."""
    _ = py_call(key.append, context)
    for i in range(len(names)):  # small-loop(names: fixed connected context entries)
        _ = py_call(key.append, py_get(context, PythonObject(names[i])))


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
    var held = trees_side_get(slot, owner)
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
    _ = trees_side_put(slot, owner, Python.tuple(py_call(b.list, labels), table))


# ------------------------------------------------------------ OneClassSVM


comptime _OCSVM_COEF_SLOT = "ocsvm_coef_snapshot_v1"


def ocsvm_coef_reused(owner: PythonObject, dual: PythonObject) raises -> Bool:
    """`dual` is the coefficient object whose float32 snapshot was the object
    itself last call (no conversion): the snapshot can be skipped."""
    var held = trees_side_get(_OCSVM_COEF_SLOT, owner)
    return held is not None and held[0] is dual and held[1] is py_type(dual)


def ocsvm_coef_remember(owner: PythonObject, dual: PythonObject, coef: PythonObject) raises:
    if coef is dual:
        _ = trees_side_put(_OCSVM_COEF_SLOT, owner, Python.tuple(dual, py_type(dual)))
    else:
        trees_side_drop(_OCSVM_COEF_SLOT, owner)
