# SPDX-License-Identifier: Apache-2.0
# Constructor-configuration admission of the meta-estimators (Voting,
# Stacking, MultiOutput, OneVsRest, CalibratedClassifierCV), once per
# configuration instead of once per public call.
#
# The admission itself is the model-selection owner's `__admit__`: a full
# native clone of the meta-estimator and every member, which validates every
# constructor value's schema. Lane perf-sweep-trees (2026-11-05): every public
# meta call (each predict, predict_proba, transform and score) paid that
# clone. The record below holds the identities of everything the clone reads:
# the constructor parameter values of the meta-estimator, recursively through
# containers and nested native estimators (their own constructor parameters).
# While every identity is unchanged, nothing the clone validates can have
# changed (constructor values are immutable scalars, exact containers walked
# item by item, or estimators walked parameter by parameter), so the clone is
# skipped. A replaced member, a `set_params`, an in-place change to a list,
# tuple, set or dict parameter, or a changed member parameter dict changes the
# record and the full admission runs again. The walk only reads exact-type
# identities and the connected plans' field dictionaries: no Python protocol of
# a value runs before the full admission has accepted it.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.prep_native_estimator import prep_estimator_call, prep_estimator_plan
from bindings.trees_side_records import trees_side_get, trees_side_put


comptime _RECORD_SLOT = "trees_meta_admitted_v1"
comptime _MAX_DEPTH = 64


def _record_into(record: PythonObject, value: PythonObject, context: PythonObject, depth: Int) raises -> Bool:
    """Append the identities the native clone reads under `value`; False when
    the configuration is too deep to record (the full admission then runs and
    reports it)."""
    if depth > _MAX_DEPTH:
        return False
    var b = native_loaded_module("builtins")
    _ = py_call(record.append, value)
    if value is None:
        return True
    var kind = py_type(value)
    if kind is b.bool or kind is b.int or kind is b.float or kind is b.str:
        return True
    if kind is b.dict:
        var pairs = py_call(b.list, value.items())
        # The pair count travels with the record: an emptied and refilled
        # mapping with the same members still differs in length.
        _ = py_call(record.append, PythonObject(len(pairs)))
        for k in range(len(pairs)):  # driver-loop: constructor parameter mapping entries
            if not _record_into(record, pairs[k][0], context, depth + 1):
                return False
            if not _record_into(record, pairs[k][1], context, depth + 1):
                return False
        return True
    if kind is b.list or kind is b.tuple or kind is b.set or kind is b.frozenset:
        var items = py_call(b.list, value)
        _ = py_call(record.append, PythonObject(len(items)))
        for k in range(len(items)):  # driver-loop: constructor parameter container items
            if not _record_into(record, items[k], context, depth + 1):
                return False
        return True
    var plan = prep_estimator_plan(value, context)
    if plan is None:
        # Any other value (a NumPy scalar or array, a splitter class, ...) is
        # recorded by identity alone; the clone admits it by type.
        return True
    var names = plan["parameters"]
    var fields = py_call(plan["fields_get"], value, kind)
    for k in range(len(names)):  # driver-loop: a member's fixed constructor parameters
        if not _record_into(record, py_get(fields, names[k]), context, depth + 1):
            return False
    return True


def _record_of(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The identity record of `model`'s constructor configuration, or None
    when it cannot be recorded."""
    var plan = prep_estimator_plan(model, context)
    if plan is None:
        return PythonObject(None)
    var record = Python.list()
    var names = plan["parameters"]
    var fields = py_call(plan["fields_get"], model, py_type(model))
    for k in range(len(names)):  # driver-loop: the meta-estimator's fixed constructor parameters
        if not _record_into(record, py_get(fields, names[k]), context, 1):
            return PythonObject(None)
    return record


def _same_record(held: PythonObject, record: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    if py_type(held) is not b.list or len(held) != len(record):
        return False
    for k in range(len(record)):  # driver-loop: recorded configuration identities
        var a = held[k]
        var c = record[k]
        if a is c:
            continue
        # Counts are fresh int objects on each walk: compare them by value.
        if py_type(a) is b.int and py_type(c) is b.int and Int(py=a) == Int(py=c):
            continue
        return False
    return True


def meta_admit_configuration(model: PythonObject, context: PythonObject) raises:
    """The meta-estimator's constructor admission (`__admit__`: one native
    clone of the model and every member), skipped while the identity record
    stored by the last successful admission still matches."""
    var record = _record_of(model, context)
    if record is not None:
        var held = trees_side_get(_RECORD_SLOT, model)
        if held is not None and _same_record(held, record):
            return
    _ = prep_estimator_call(model, "__admit__", Python.tuple(), context)
    if record is not None:
        _ = trees_side_put(_RECORD_SLOT, model, record)
