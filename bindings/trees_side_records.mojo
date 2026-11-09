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
# Storage (lane ne1-e-records, 2026-11-06): the records live in the TREES
# family slot of bindings/native_model_records.mojo (one weak entry per
# object for every family); this module keeps the tree family's key walks.
#
# A record is valid while its key matches: the identities of every object
# the cached work read, walked as the work walked them (exact containers item
# by item, with their lengths; anything else by identity). Values are
# compared by identity only (counts by value), so no user `__eq__`/`__hash__`
# or other Python protocol runs. A record never holds its own model.
from std.python import Python, PythonObject
from bindings.native_loaded_module import NativeModules
from bindings.native_model_records import (
    records_get, records_put, records_drop, records_key_append, records_key_same_exact,
    records_stamped_get, records_stamped_put, records_stampable, ContentRecord, content_value, RECORDS_TREES,
)
from bindings.py_balanced import py_call, py_type, py_contains, py_fields
from bindings.classical_pandas import classical_admit_value
from core.shared_slot import SharedSlot

comptime _MAX_DEPTH = 64


def trees_side_get(slot: String, owner: PythonObject) raises -> PythonObject:
    """The record `slot` holds for `owner`, or None."""
    return records_get(RECORDS_TREES, owner, slot)


def trees_side_put(slot: String, owner: PythonObject, record: PythonObject) raises -> Bool:
    """Store `record` for `owner` (replacing any older one). False when the
    owner takes no weak reference: then nothing is stored and the caller
    redoes its work on every call, as before."""
    return records_put(RECORDS_TREES, owner, slot, record)


def trees_side_drop(slot: String, owner: PythonObject) raises:
    """Forget `owner`'s record in `slot` (a refit: its arrays are released
    now, not at the next call)."""
    records_drop(RECORDS_TREES, owner, slot)


def trees_key_same(held: PythonObject, key: PythonObject) raises -> Bool:
    """Two keys are the same: equal length and every entry the same object
    (exact ints, the recorded counts, compared by value)."""
    if held is None:
        return False
    var list_t = py_type(Python.list())
    if py_type(held) is list_t and py_type(key) is list_t:
        # Borrowed native reads (lane ne1-e-records, row N255): no boxed
        # indexing per entry.
        return records_key_same_exact(held, key)
    if len(held) != len(key):
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
        for k in range(len(pairs)):  # driver-loop: constructor parameter mapping entries
            if not trees_key_value(mods, key, residual, pairs[k][0], depth + 1):
                return False
            if not trees_key_value(mods, key, residual, pairs[k][1], depth + 1):
                return False
        return True
    if cls is b.list or cls is b.tuple or cls is b.set or cls is b.frozenset:
        # Iterating an exact container runs no item protocol.
        var items = py_call(b.list, value)
        records_key_append(key, PythonObject(len(items)))
        for k in range(len(items)):  # driver-loop: constructor parameter container items
            if not trees_key_value(mods, key, residual, items[k], depth + 1):
                return False
        return True
    records_key_append(key, cls)
    records_key_append(residual, Python.tuple(value, PythonObject(depth)))
    return True


def trees_key_fields(mut mods: NativeModules, key: PythonObject, fields: PythonObject) raises:
    """Append an exact field dictionary's length, then every key and value
    (exact list/tuple/dict values with their items, to a bounded depth;
    deeper values by identity)."""
    var b = mods.module("builtins")
    var items = py_call(b.list, py_call(fields.items))
    records_key_append(key, PythonObject(len(items)))
    for i in range(len(items)):  # driver-loop: over a fitted model's stored fields (identity only)
        records_key_append(key, items[i][0])
        trees_key_shallow(mods, key, items[i][1], 0)


def trees_key_shallow(mut mods: NativeModules, key: PythonObject, value: PythonObject, depth: Int) raises:
    """Append `value` and, for exact list/tuple/dict values, their length and
    items (to depth 4; deeper by identity); a non-scalar's class too."""
    records_key_append(key, value)
    if depth >= 4 or value is None:
        return
    var b = mods.module("builtins")
    var cls = py_type(value)
    if cls is b.list or cls is b.tuple:
        records_key_append(key, PythonObject(len(value)))
        for i in range(len(value)):  # driver-loop: over a stored container's items (identity only)
            trees_key_shallow(mods, key, value[i], depth + 1)
    elif cls is b.dict:
        var items = py_call(b.list, py_call(value.items))
        records_key_append(key, PythonObject(len(items)))
        for i in range(len(items)):  # driver-loop: over a stored mapping's items (identity only)
            records_key_append(key, items[i][0])
            trees_key_shallow(mods, key, items[i][1], depth + 1)
    elif not (cls is b.bool or cls is b.int or cls is b.float or cls is b.str or cls is b.bytes):
        # A class can be reassigned on an instance: its admission is by type.
        records_key_append(key, cls)


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
    # Generation fast path (lane ne1-e-records, row N255): while the model's
    # generation word (bindings/native_model_records.mojo: its watched
    # instance dictionary, class and explicit bumps) and the native content
    # words of its container parameters are unchanged, the key below would
    # match again: no key list is built and only the residual values are
    # admitted again, as on a key hit.
    var stamped = records_stamped_get(RECORDS_TREES, model, _PARAMS_STAMP)
    if stamped is not None and stamped[0] is names and stamped[2] is fields:
        var kept = stamped[1]
        for i in range(len(kept)):  # driver-loop: parameters admitted by more than identity
            classical_admit_value(kept[i][0], context, Int(py=kept[i][1]))
        return
    var mods = NativeModules()
    var key = Python.list(names)
    var residual = Python.list()
    var recordable = True
    for i in range(len(names)):  # driver-loop: stored constructor arguments (identity record)
        var name = names[i]
        if py_contains(fields, name):
            records_key_append(key, name)
            if not trees_key_value(mods, key, residual, fields[name], 0):
                recordable = False
                break
    if recordable:
        var held = trees_side_get(_PARAMS_SLOT, model)
        if held is not None and trees_key_same(held, key):
            for i in range(len(residual)):  # driver-loop: parameters admitted by more than identity
                classical_admit_value(residual[i][0], context, Int(py=residual[i][1]))
            _trees_params_stamp(model, fields, names, residual)
            return
    for i in range(len(names)):  # driver-loop: stored constructor arguments before native conversions
        if py_contains(fields, names[i]):
            classical_admit_value(fields[names[i]], context)
    if recordable:
        _ = trees_side_put(_PARAMS_SLOT, model, key)
        _trees_params_stamp(model, fields, names, residual)


comptime _PARAMS_STAMP = "trees_forest_params_stamp_v1"


def _trees_params_stamp(model: PythonObject, fields: PythonObject, names: PythonObject,
                        residual: PythonObject) raises:
    """Stamp the admitted configuration with the model's generation. Only
    when `fields` is the model's own instance dictionary (the one its
    generation watches); the record holds `fields`, so its identity cannot be
    recycled. Content words cover exact lists and dicts item by item and
    every other non-scalar's class, as trees_key_value does; a set, a
    non-str/int dict key or a nested object makes the record unusable."""
    if not (py_fields(model) is fields) or not records_stampable(model):
        return
    var content = ContentRecord()
    for i in range(len(names)):  # driver-loop: stored constructor arguments (content words, once per record)
        if py_contains(fields, names[i]):
            content_value(content, fields[names[i]], 0, _MAX_DEPTH + 1, True)
    _ = records_stamped_put(RECORDS_TREES, model, _PARAMS_STAMP, content^, Python.tuple(names, residual, fields))


# ------------------------------------------------- finite-scanned fit inputs
#
# The forest public door admits X with classical_check_x, which refuses NaN
# and infinity while casting; the forest fit then scanned the same X again
# (`_finite_or_refuse`, O(n d) per fit). The door registers its admitted X
# here for the duration of its fit, and a float32 copy made from a
# registered source is registered too. A scan is skipped only for a matrix
# whose owner is registered, so a matrix from any other caller (a member
# fit, a meta-estimator) is scanned as before. Entries are strong
# references (no address reuse while registered) and the door clears them
# when its fit ends; a concurrent clear only causes an extra scan.


struct TreesFiniteSources(Defaultable, Movable):
    var items: List[PythonObject]

    def __init__(out self):
        self.items = List[PythonObject]()


comptime TREES_FINITE = SharedSlot[TreesFiniteSources, "MojolearnTreesFiniteSourcesV2"]


def trees_finite_known(value: PythonObject) raises -> Bool:
    """`value` was admitted finite by the public door of the running fit."""
    var g = TREES_FINITE.get_or_create_ptr()
    for i in range(len(g[].items)):  # small-loop(items: the running fit's registered inputs, one or two)
        if g[].items[i] is value:
            return True
    return False


def trees_finite_mark(value: PythonObject) raises:
    var g = TREES_FINITE.get_or_create_ptr()
    g[].items.append(value)


def trees_finite_clear() raises:
    var g = TREES_FINITE.get_or_create_ptr()
    # Hold a second reference while the registry empties, so no object is
    # released (no destructor runs) while the registry is being changed.
    var items = g[].items.copy()
    g[].items.clear()
    _ = items^
