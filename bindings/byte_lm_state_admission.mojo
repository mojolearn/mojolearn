# SPDX-License-Identifier: Apache-2.0
"""Admit-once record of a Byte-LM trainer's fitted `_state`.

Every train step, logits call and checkpoint export used to run
`classical_admit_value` over the whole `_state` dictionary (parameter names,
shapes, offsets, optimizer configuration, model shape and data schedule) even
though only the six step words change between calls. The record below is kept
in the binding connection context (never in the trainer `__dict__`), keyed by
the trainer's object address. It holds strong references to the admitted
stable values, so a recorded identity can never be recycled for a different
object while the record lives.

A call takes the fast path only when `_state` is still an exact dict with the
same number of fields, every stable field holds the very object admitted
before (and an exact list/tuple/dict of the same size), and the same step
fields are present; the step fields alone are then admitted. Any other change
(a field added, removed or replaced, a stable container resized, a new state
from export/load) falls back to the full admission and re-records. In-place
replacement of an element inside a stable container at an unchanged size is
outside the private-state contract and not re-checked, as in the causal
owner's identity record.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_loaded_module import NativeModules
from bindings.classical_pandas import classical_admit_value

# Bound on the per-connection table; dead trainers' entries are dropped when
# it fills (one clear, then the live trainers re-record on their next call).
comptime _BYTE_STATE_RECORDS_MAX = 64


def _byte_state_step_fields() -> List[String]:
    # The fields every step, session adoption or load rewrites.
    return [String("parameters"), String("m"), String("v"), String("flags"), String("completed_steps"),
            String("next_batch_index")]


def _byte_state_is_step_field(name: String) -> Bool:
    var steps = _byte_state_step_fields()
    for i in range(len(steps)):  # small-loop(the six step fields)
        if steps[i] == name:
            return True
    return False


def _byte_state_size(value: PythonObject, bi: PythonObject) raises -> Int:
    # len() only on exact builtin containers (C length, no __len__ hook).
    var cls = py_type(value)
    if cls is bi.list or cls is bi.tuple or cls is bi.dict:
        return len(value)
    return -1


def _byte_state_record(state: PythonObject, bi: PythonObject) raises -> PythonObject:
    # (field count, step fields present, [(name, value, size)] of stable fields)
    var stable = Python.list()
    var present = 0
    var keys = py_call(bi.list, state.keys())
    for i in range(len(keys)):  # driver-loop: over the state record's top-level fields
        var name = keys[i]
        if py_type(name) is bi.str and _byte_state_is_step_field(String(py=name)):
            present += 1
            continue
        var value = state[name]
        _ = py_call(stable.append, Python.tuple(name, value, PythonObject(_byte_state_size(value, bi))))
    return Python.tuple(PythonObject(len(keys)), PythonObject(present), stable)


def _byte_state_matches(record: PythonObject, state: PythonObject, bi: PythonObject) raises -> Bool:
    if len(state) != Int(py=record[0]):
        return False
    var stable = record[2]
    for i in range(len(stable)):  # driver-loop: over the recorded stable fields
        var entry = stable[i]
        var name = entry[0]
        if not py_contains(state, name):
            return False
        var value = state[name]
        if value is not entry[1] or _byte_state_size(value, bi) != Int(py=entry[2]):
            return False
    var steps = _byte_state_step_fields()
    var present = 0
    for i in range(len(steps)):  # small-loop(the six step fields)
        if py_contains(state, PythonObject(steps[i])):
            present += 1
    return present == Int(py=record[1])


def byte_state_admit(mut mods: NativeModules, trainer: PythonObject, state: PythonObject,
                     context: PythonObject) raises:
    """`classical_admit_value(state, context)`, admitted once per state
    layout: later calls admit only the step fields while the stable fields
    are the same objects at the same sizes."""
    var bi = mods.module("builtins")
    if py_type(state) is not bi.dict:
        classical_admit_value(state, context)
        return
    var key = PythonObject(Int(trainer._obj_ptr))
    var table = py_get(context, "_byte_lm_state_records")
    if table is not None and py_type(table) is bi.dict:
        var record = py_get(table, key)
        if record is not None and _byte_state_matches(record, state, bi):
            var steps = _byte_state_step_fields()
            var values = Python.list()
            for i in range(len(steps)):  # small-loop(the six step fields)
                var name = PythonObject(steps[i])
                if py_contains(state, name):
                    _ = py_call(values.append, state[name])
            classical_admit_value(values, context)
            return
    classical_admit_value(state, context)
    if table is None or py_type(table) is not bi.dict:
        table = py_call(bi.dict)
        py_setitem(context, "_byte_lm_state_records", table)
    if len(table) >= _BYTE_STATE_RECORDS_MAX:
        _ = py_call(table.clear)
    py_setitem(table, key, _byte_state_record(state, bi))
