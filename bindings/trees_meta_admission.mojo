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
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.prep_native_estimator import prep_estimator_call, prep_estimator_plan
from bindings.trees_side_records import trees_side_get, trees_side_put
from bindings.native_model_records import (
    records_stamped_get, records_stamped_put, records_stampable, ContentRecord, content_value, content_object,
    RECORDS_TREES,
)


comptime _RECORD_SLOT = "trees_meta_admitted_v1"
comptime _STAMP_SLOT = "trees_meta_admitted_stamp_v1"
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
    var table = cpython_table()
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
    var table = cpython_table()
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


# Generation stamp (lane ne2-b-members, row N071; the generation word of
# bindings/native_model_records.mojo, lane ne1-e-records): after an admission
# (or a record match) the configuration is stamped with the meta-estimator's
# generation word plus native content words for what that word does not
# prove: the items of every exact list and dict the walk reads, the class of
# every other non-scalar value, and every nested estimator's class and own
# generation word, recursively through its constructor parameters, in the
# walk's order. A call whose words all hold skips the walk (one word per
# object instead of one identity per constructor value). Any set or
# frozenset, a dict keyed by anything but exact str / int, a member whose
# fields are not its instance dictionary, too deep a configuration, or
# Python 3.10 / 3.11 (no dictionary watcher) stores no stamp: the identity
# record above decides, exactly as before. `set_params`, a member assignment
# and any attribute write change the owner's or the member's dictionary and
# so its generation; an in-place edit of a list or dict parameter changes its
# content words.
def _stamp_into(mut rec: ContentRecord, value: PythonObject, context: PythonObject, depth: Int) raises:
    # Iterative (ne-fix-hang2): an explicit stack in place of the former
    # self-recursion (the compile hang was the dead-key ring, not this). Children
    # are pushed in reverse so they pop in order: the same depth-first
    # pre-order of content words, the same `_MAX_DEPTH` refusal and the same
    # stop once `rec.ok` is False.
    if not rec.ok:
        return
    var b = native_loaded_module("builtins")
    var pending = List[PythonObject]()
    var depths = List[Int]()
    pending.append(value)
    depths.append(depth)
    while len(pending) > 0 and rec.ok:  # driver-loop: over a configuration's nested values (once per stamp)
        var item = pending.pop()
        var level = depths.pop()
        if item is None:
            continue
        if level > _MAX_DEPTH:
            rec.ok = False
            return
        var kind = py_type(item)
        if kind is b.bool or kind is b.int or kind is b.float or kind is b.str:
            continue
        if kind is b.set or kind is b.frozenset:
            rec.ok = False
            return
        if kind is b.dict:
            # The entries' identities (exact str / int keys only), one level.
            content_value(rec, item, 0, 1, True)
            var pairs = py_call(b.list, item.items())
            for k in range(len(pairs) - 1, -1, -1):  # driver-loop: constructor parameter mapping entries (once per stamp)
                pending.append(pairs[k][1])
                depths.append(level + 1)
            continue
        if kind is b.list or kind is b.tuple:
            # A list's length and item identities (a tuple cannot change), then
            # each item.
            content_value(rec, item, 0, 1, True)
            for k in range(len(item) - 1, -1, -1):  # driver-loop: constructor parameter container items (once per stamp)
                pending.append(item[k])
                depths.append(level + 1)
            continue
        var plan = prep_estimator_plan(item, context)
        if plan is None:
            # Identity alone, as the record; its class is kept too.
            content_value(rec, item, 0, 1, True)
            continue
        var fields = py_call(plan["fields_get"], item, kind)
        if not (fields is py_fields(item)):
            rec.ok = False
            return
        # The member's class and generation word (limit 0: its values are walked
        # below, parameter by parameter, as the record walks them).
        content_object(rec, item, 0)
        var names = plan["parameters"]
        for k in range(len(names) - 1, -1, -1):  # driver-loop: a member's fixed constructor parameters (once per stamp)
            pending.append(py_get(fields, names[k]))
            depths.append(level + 1)


def _stamp(model: PythonObject, context: PythonObject) raises:
    """Stamp `model`'s admitted configuration (see `_stamp_into`)."""
    if not records_stampable(model):
        return
    var plan = prep_estimator_plan(model, context)
    if plan is None:
        return
    var fields = py_call(plan["fields_get"], model, py_type(model))
    if not (fields is py_fields(model)):
        return
    var rec = ContentRecord()
    var names = plan["parameters"]
    for k in range(len(names)):  # driver-loop: the meta-estimator's fixed constructor parameters (once per stamp)
        _stamp_into(rec, py_get(fields, names[k]), context, 1)
    _ = records_stamped_put(RECORDS_TREES, model, _STAMP_SLOT, rec^, PythonObject(True))


def meta_admit_configuration(model: PythonObject, context: PythonObject) raises:
    """The meta-estimator's constructor admission (`__admit__`: one native
    clone of the model and every member), skipped while the generation stamp,
    or else the identity record, stored by the last successful admission
    still matches."""
    if records_stamped_get(RECORDS_TREES, model, _STAMP_SLOT) is not None:
        return
    var held = trees_side_get(_RECORD_SLOT, model)
    if held is not None and _held_matches(held, model, context):
        _stamp(model, context)
        return
    var record = _record_of(model, context)
    _ = prep_estimator_call(model, "__admit__", Python.tuple(), context)
    if record is not None:
        _ = trees_side_put(_RECORD_SLOT, model, record)
        _stamp(model, context)
