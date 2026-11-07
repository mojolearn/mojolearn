# SPDX-License-Identifier: Apache-2.0
"""Admit-once records for `msel_estimator_plan` (lane perf2-msel).

`msel_estimator_plan` runs on every native cross-validation member call,
every delegated search call (`GridSearchCV.predict` and friends) and every
native clone. It checked two things each time:

1. the class rows: the live MRO equals the plan's and every connected method
   or storage hook (one row per ancestor and protected name) is still the
   identity connected at import;
2. the instance dictionary: every key an exact str, no protected name
   overriding a connected method, `_has_constructor_parameters` an exact bool.

(1) depends only on the class and is kept per class in the
native_class_admission type-watcher records (any class or ancestor
modification, `__bases__` assignment included, invalidates it). (2) depends
only on the instance dictionary and is kept per model (its record slot) with
the dictionary's address and the dictionary-watcher epoch of
native_class_admission: any insertion,
replacement, deletion, clear or deallocation of the watched dictionary bumps
the epoch and forces the full check again. The record also holds the plan's
address, so the same dictionary under another class plan is checked again.

Python 3.10 and 3.11 have neither watcher: every call there runs both checks
exactly as before (no record is ever admitted).

This map is separate from `dict_keys_admit`'s shared admitted map on purpose:
that one only says "every key is an exact str", while this record also
covers the protected-name refusal.
"""
from std.python import PythonObject
from bindings.native_class_admission import dict_keys_admit, dict_keys_admitted
from bindings.native_loaded_module import shared_dict_epoch, shared_dict_stamped
from bindings.native_model_records import records_live, records_open, records_slot_msel_plan

# Storage (lane ne1-e-records, 2026-11-06): the record lives in the model's
# MSEL_PLAN slot of bindings/native_model_records.mojo, keyed by the MODEL
# through a weak reference, so it is released when the model dies. The old
# map was keyed by dictionary address and kept one entry per model ever
# admitted (bounded only by a 4096-entry clear).

#: The class-record slot of native_class_admission for the class rows check.
comptime MSEL_PLAN_CLASS_SLOT = "msel_estimator_plan_class_v1"


def msel_fields_admitted(model: PythonObject, fields: PythonObject, plan: PythonObject) raises -> Bool:
    """True when `fields` (the instance dictionary of `model`) passed the full
    instance check for `plan` and the watched dictionary has not changed
    since."""
    if not shared_dict_stamped():
        return False
    if not records_live(model):
        return False
    var slot = records_slot_msel_plan(Int(model._obj_ptr))
    if not slot[].admitted:
        return False
    var address = Int(fields._obj_ptr)
    if slot[].fields_address != address or not (slot[].plan is plan):
        return False
    return slot[].fields_epoch == shared_dict_epoch(address)


def msel_fields_admit(model: PythonObject, fields: PythonObject, plan: PythonObject) raises:
    """Record that `fields` (an exact dict whose keys are all exact str, the
    instance dictionary of `model`) just passed the full instance check for
    `plan`. No-op without watchers."""
    # Arms the dictionary watcher on `fields` (and records the exact-str key
    # admission, which the caller has just made).
    dict_keys_admit(fields)
    if not shared_dict_stamped():
        return
    # `dict_keys_admit` returns silently when PyDict_Watch refuses this
    # dictionary (for example a watcher id registered by another
    # interpreter). An unwatched dictionary never bumps its epoch, so a
    # record written for it would stay valid after `est.predict = hook` and
    # skip the protected-name refusal. Write the record only when the key
    # admission (made just now) is held, which happens only when the watch
    # was accepted.
    if not dict_keys_admitted(fields):
        return
    var key = records_open(model)
    if key == 0:
        return
    var address = Int(fields._obj_ptr)
    var epoch = shared_dict_epoch(address)
    var slot = records_slot_msel_plan(key)
    var previous = slot[].plan
    slot[].plan = plan
    slot[].fields_address = address
    slot[].fields_epoch = epoch
    slot[].admitted = True
    _ = previous^
