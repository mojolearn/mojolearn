# SPDX-License-Identifier: Apache-2.0
# Per-model native side records of the tree family (lane perf-trees-owed,
# 2026-11-06).
#
# A tree estimator's public call used to redo work whose result depends only
# on objects the model already holds: the admission of every constructor
# parameter (forests), of every fitted member (boosting), the meta-estimators'
# clone admission, DART's per-tree address lists, the isolation forest's
# resident token. Those results are now kept in the prepared-owner registry
# (bindings/native_prepared_owner.mojo): keyed by the model object through a
# weak reference, outside the model's `__dict__`. So predict, score and
# transform leave `__dict__` unchanged (scikit-learn's estimator checks),
# pickles and copies never carry a process-local record, and a record dies
# with its model. A copy or an unpickled model has no record and simply
# rebuilds it on first use.
#
# A record is valid while its key matches: the identities of every object
# the cached work read, walked as the work walked them (exact containers item
# by item, with their lengths; anything else by identity). Values are
# compared by identity only (counts by value), so no user `__eq__`/`__hash__`
# or other Python protocol runs. A record never holds its own model.
from std.python import Python, PythonObject
from bindings.native_loaded_module import NativeModules
from bindings.native_prepared_owner import prepared_get, prepared_put, prepared_drop
from bindings.py_balanced import py_call, py_type, py_contains
from bindings.classical_pandas import classical_admit_value

comptime _MAX_DEPTH = 64


def trees_side_get(slot: String, owner: PythonObject) raises -> PythonObject:
    """The record `slot` holds for `owner`, or None."""
    var mods = NativeModules()
    return prepared_get(mods, slot, owner)


def trees_side_put(slot: String, owner: PythonObject, record: PythonObject) raises -> Bool:
    """Store `record` for `owner` (replacing any older one). False when the
    owner takes no weak reference: then nothing is stored and the caller
    redoes its work on every call, as before."""
    var mods = NativeModules()
    prepared_put(mods, slot, owner, record)
    return prepared_get(mods, slot, owner) is not None


def trees_side_drop(slot: String, owner: PythonObject) raises:
    """Forget `owner`'s record in `slot` (a refit: its arrays are released
    now, not at the next call)."""
    var mods = NativeModules()
    prepared_drop(mods, slot, owner)


def trees_key_same(held: PythonObject, key: PythonObject) raises -> Bool:
    """Two keys are the same: equal length and every entry the same object
    (exact ints, the recorded counts, compared by value)."""
    if held is None or len(held) != len(key):
        return False
    var int_t = py_type(PythonObject(0))
    for k in range(len(key)):  # small-loop(key: recorded identities): pointer identity compare only
        var a = held[k]
        var c = key[k]
        if a is c:
            continue
        # Exact ints compare with int's own equality (any size, no hook).
        if py_type(a) is int_t and py_type(c) is int_t and Bool(py=(a == c)):
            continue
        return False
    return True


def trees_key_value(mut mods: NativeModules, key: PythonObject, residual: PythonObject, value: PythonObject,
                    depth: Int) raises -> Bool:
    """Append the identities an admission of `value` reads: the value, and for
    exact list/tuple/set/frozenset/dict containers their length and items
    (dict keys and values), recursively. A value of any other type (an
    ndarray, a RandomState, a container subclass, an estimator) is appended by
    identity and also to `residual` as `(value, depth)`: its admission may
    depend on more than its identity, so the caller admits it on every call.
    False when the nesting is too deep to record (the full admission then runs
    and reports it)."""
    if depth > _MAX_DEPTH:
        return False
    _ = py_call(key.append, value)
    if value is None:
        return True
    var b = mods.module("builtins")
    var cls = py_type(value)
    if cls is b.bool or cls is b.int or cls is b.float or cls is b.str or cls is b.bytes:
        return True
    if cls is b.dict:
        # items() of an exact dict never hashes or compares keys.
        var pairs = py_call(b.list, py_call(value.items))
        _ = py_call(key.append, PythonObject(len(pairs)))
        for k in range(len(pairs)):  # driver-loop: constructor parameter mapping entries
            if not trees_key_value(mods, key, residual, pairs[k][0], depth + 1):
                return False
            if not trees_key_value(mods, key, residual, pairs[k][1], depth + 1):
                return False
        return True
    if cls is b.list or cls is b.tuple or cls is b.set or cls is b.frozenset:
        # Iterating an exact container runs no item protocol.
        var items = py_call(b.list, value)
        _ = py_call(key.append, PythonObject(len(items)))
        for k in range(len(items)):  # driver-loop: constructor parameter container items
            if not trees_key_value(mods, key, residual, items[k], depth + 1):
                return False
        return True
    _ = py_call(key.append, cls)
    _ = py_call(residual.append, Python.tuple(value, PythonObject(depth)))
    return True


def trees_key_fields(mut mods: NativeModules, key: PythonObject, fields: PythonObject) raises:
    """Append an exact field dictionary's length, then every key and value
    (exact list/tuple/dict values with their items, to a bounded depth;
    deeper values by identity)."""
    var b = mods.module("builtins")
    var items = py_call(b.list, py_call(fields.items))
    _ = py_call(key.append, PythonObject(len(items)))
    for i in range(len(items)):  # driver-loop: over a fitted model's stored fields (identity only)
        _ = py_call(key.append, items[i][0])
        _key_shallow(mods, key, items[i][1], 0)


def _key_shallow(mut mods: NativeModules, key: PythonObject, value: PythonObject, depth: Int) raises:
    _ = py_call(key.append, value)
    if depth >= 4 or value is None:
        return
    var b = mods.module("builtins")
    var cls = py_type(value)
    if cls is b.list or cls is b.tuple:
        _ = py_call(key.append, PythonObject(len(value)))
        for i in range(len(value)):  # driver-loop: over a stored container's items (identity only)
            _key_shallow(mods, key, value[i], depth + 1)
    elif cls is b.dict:
        var items = py_call(b.list, py_call(value.items))
        _ = py_call(key.append, PythonObject(len(items)))
        for i in range(len(items)):  # driver-loop: over a stored mapping's items (identity only)
            _ = py_call(key.append, items[i][0])
            _key_shallow(mods, key, items[i][1], depth + 1)
    elif not (cls is b.bool or cls is b.int or cls is b.float or cls is b.str or cls is b.bytes):
        # A class can be reassigned on an instance: its admission is by type.
        _ = py_call(key.append, cls)


comptime _PARAMS_SLOT = "trees_forest_params_v1"


def trees_admit_parameters(model: PythonObject, fields: PythonObject, names: PythonObject,
                           context: PythonObject) raises:
    """`classical_admit_value` of every stored constructor parameter in
    `names`, once per configuration.

    Each forest public call (fit, predict, predict_proba, score) admitted all
    ~26 parameters, each a full value admission (~40-60 C-API calls,
    bindings/classical_pandas.mojo). The parameters are recorded by identity
    (exact scalars, exact containers item by item); while the record matches,
    those admissions would accept again, so only values whose admission can
    depend on more than their identity (ndarrays, RandomState, container
    subclasses, ...) are admitted again. A `set_params`, an assignment or an
    in-place container edit changes the key and the full admission runs, in
    the original order, with its original refusals."""
    var mods = NativeModules()
    var key = Python.list(names)
    var residual = Python.list()
    var recordable = True
    for i in range(len(names)):  # driver-loop: stored constructor arguments (identity record)
        var name = names[i]
        if py_contains(fields, name):
            _ = py_call(key.append, name)
            if not trees_key_value(mods, key, residual, fields[name], 0):
                recordable = False
                break
    if recordable:
        var held = trees_side_get(_PARAMS_SLOT, model)
        if held is not None and trees_key_same(held, key):
            for i in range(len(residual)):  # driver-loop: parameters admitted by more than identity
                classical_admit_value(residual[i][0], context, Int(py=residual[i][1]))
            return
    for i in range(len(names)):  # driver-loop: stored constructor arguments before native conversions
        if py_contains(fields, names[i]):
            classical_admit_value(fields[names[i]], context)
    if recordable:
        _ = trees_side_put(_PARAMS_SLOT, model, key)
