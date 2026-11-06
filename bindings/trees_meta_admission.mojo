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
from std.python._cpython import PyObjectPtr
from bindings.perf2_shared_cpython_table import perf2_cpython_table
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


# Perf (perf3-nbr-msel-trees, trees R5): a call compares the held record
# while it walks the configuration, in the same order `_record_into` wrote it,
# instead of building a fresh record list (one append per identity, a list
# copy per container) and comparing the two afterwards. It stops at the first
# difference. Lists, tuples and dicts are read in place through the CPython
# accessors (borrowed references, no Python temporaries); sets keep the
# `list(value)` copy. A miss builds the record exactly as before.
def _held_at(held: PythonObject, pos: Int) raises -> PythonObject:
    var table = perf2_cpython_table()
    return PythonObject(from_borrowed=table[].PyList_GetItem(held._obj_ptr, pos))


def _held_count(held: PythonObject, mut pos: Int, count: Int) raises -> Bool:
    if pos >= len(held):
        return False
    var item = _held_at(held, pos)
    if py_type(item) is not native_loaded_module("builtins").int or Int(py=item) != count:
        return False
    pos += 1
    return True


def _match_into(held: PythonObject, mut pos: Int, value: PythonObject, context: PythonObject,
                depth: Int) raises -> Bool:
    """True when the identities under `value` equal the held record from
    `pos` on (advancing `pos`), in `_record_into`'s order."""
    if depth > _MAX_DEPTH or pos >= len(held):
        return False
    if _held_at(held, pos) is not value:
        return False
    pos += 1
    if value is None:
        return True
    var b = native_loaded_module("builtins")
    var kind = py_type(value)
    if kind is b.bool or kind is b.int or kind is b.float or kind is b.str:
        return True
    var table = perf2_cpython_table()
    if kind is b.dict:
        var size = len(value)
        if not _held_count(held, pos, size):
            return False
        ref cpy = Python().cpython()
        var at = Int(0)
        var key = PyObjectPtr()
        var item = PyObjectPtr()
        # main's compiled form (bindings/causal_model_config.mojo:75); the
        # shared tables carry no PyDict_Next entry (pass 2, C3). The symbol
        # is resolved once per dictionary, not once per entry (p3-trees).
        var dict_next = cpy.lib.get_function[Int32]("PyDict_Next")
        for _ in range(size):  # driver-loop: constructor parameter mapping entries
            if dict_next(value._obj_ptr, MutPointer(to=at), MutPointer(to=key), MutPointer(to=item)) == 0:
                return False
            if not _match_into(held, pos, PythonObject(from_borrowed=key), context, depth + 1):
                return False
            if not _match_into(held, pos, PythonObject(from_borrowed=item), context, depth + 1):
                return False
        return True
    if kind is b.list or kind is b.tuple:
        var size = len(value)
        if not _held_count(held, pos, size):
            return False
        var is_list = kind is b.list
        for k in range(size):  # driver-loop: constructor parameter container items
            # Pass 2 H12: each borrowed item becomes an owned reference
            # (`from_borrowed` takes one) before anything else runs, and the
            # walk below runs no Python code (exact-type dict lookups and the
            # `__dict__` descriptor). A list is still re-measured before each
            # read, so a list that shrank can never hand back a NULL item.
            if is_list and k >= len(value):
                return False
            var entry = PythonObject(from_borrowed=table[].PyList_GetItem(value._obj_ptr, k)) if is_list else PythonObject(from_borrowed=table[].PyTuple_GetItem(value._obj_ptr, k))
            if not _match_into(held, pos, entry, context, depth + 1):
                return False
        return True
    if kind is b.set or kind is b.frozenset:
        var items = py_call(b.list, value)
        if not _held_count(held, pos, len(items)):
            return False
        for k in range(len(items)):  # driver-loop: constructor parameter container items
            if not _match_into(held, pos, items[k], context, depth + 1):
                return False
        return True
    var plan = prep_estimator_plan(value, context)
    if plan is None:
        return True
    var names = plan["parameters"]
    var fields = py_call(plan["fields_get"], value, kind)
    for k in range(len(names)):  # driver-loop: a member's fixed constructor parameters
        if not _match_into(held, pos, py_get(fields, names[k]), context, depth + 1):
            return False
    return True


def _held_matches(held: PythonObject, model: PythonObject, context: PythonObject) raises -> Bool:
    """True when `held` (a record `_record_of` built) still describes `model`."""
    if py_type(held) is not native_loaded_module("builtins").list:
        return False
    var plan = prep_estimator_plan(model, context)
    if plan is None:
        return False
    var names = plan["parameters"]
    var fields = py_call(plan["fields_get"], model, py_type(model))
    var pos = Int(0)
    for k in range(len(names)):  # driver-loop: the meta-estimator's fixed constructor parameters
        if not _match_into(held, pos, py_get(fields, names[k]), context, 1):
            return False
    return pos == len(held)


def meta_admit_configuration(model: PythonObject, context: PythonObject) raises:
    """The meta-estimator's constructor admission (`__admit__`: one native
    clone of the model and every member), skipped while the identity record
    stored by the last successful admission still matches."""
    var held = trees_side_get(_RECORD_SLOT, model)
    if held is not None and _held_matches(held, model, context):
        return
    var record = _record_of(model, context)
    _ = prep_estimator_call(model, "__admit__", Python.tuple(), context)
    if record is not None:
        _ = trees_side_put(_RECORD_SLOT, model, record)
