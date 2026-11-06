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
only on the instance dictionary and is kept here per dictionary address with
the dictionary-watcher epoch of native_class_admission: any insertion,
replacement, deletion, clear or deallocation of the watched dictionary bumps
the epoch and forces the full check again. The record also holds the plan's
address, so the same dictionary under another class plan is checked again.

Python 3.10 and 3.11 have neither watcher: every call there runs both checks
exactly as before (no record is ever admitted).

This map is separate from `dict_keys_admit`'s shared admitted map on purpose:
that one only says "every key is an exact str", while this record also
covers the protected-name refusal.
"""
from std.ffi import _Global
from std.python import PythonObject
from bindings.native_class_admission import dict_keys_admit, dict_keys_admitted
from bindings.native_loaded_module import shared_dict_epoch, shared_dict_watcher


struct MselFieldsRecords(Defaultable, Movable):
    var epochs: Dict[Int, Int]
    var plans: Dict[Int, Int]

    def __init__(out self):
        self.epochs = Dict[Int, Int]()
        self.plans = Dict[Int, Int]()


comptime MSEL_FIELDS_RECORDS = _Global[
    StorageType=MselFieldsRecords, name="MojolearnMselFieldsRecordsV1",
    init_fn=MselFieldsRecords.__init__,
]

#: The class-record slot of native_class_admission for the class rows check.
comptime MSEL_PLAN_CLASS_SLOT = "msel_estimator_plan_class_v1"

#: Bound on the records kept. A record is never removed when its dictionary
#: dies (the dead address only fails the epoch test), so without a bound the
#: map grows by one entry per model ever admitted. The records are an
#: optimization only: dropping them all just re-runs the full check once.
comptime _MSEL_FIELDS_RECORDS_MAX = 4096


def msel_fields_admitted(fields: PythonObject, plan: PythonObject) raises -> Bool:
    """True when `fields` passed the full instance check for `plan` and the
    watched dictionary has not changed since."""
    if shared_dict_watcher() < 0:
        return False
    var address = Int(fields._obj_ptr)
    var g = MSEL_FIELDS_RECORDS.get_or_create_ptr()
    if address not in g[].epochs:
        return False
    if g[].plans[address] != Int(plan._obj_ptr):
        return False
    return g[].epochs[address] == shared_dict_epoch(address)


def msel_fields_admit(fields: PythonObject, plan: PythonObject) raises:
    """Record that `fields` (an exact dict whose keys are all exact str) just
    passed the full instance check for `plan`. No-op without watchers."""
    # Arms the dictionary watcher on `fields` (and records the exact-str key
    # admission, which the caller has just made).
    dict_keys_admit(fields)
    if shared_dict_watcher() < 0:
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
    var address = Int(fields._obj_ptr)
    var g = MSEL_FIELDS_RECORDS.get_or_create_ptr()
    if address not in g[].epochs and len(g[].epochs) >= _MSEL_FIELDS_RECORDS_MAX:
        g[].epochs.clear()
        g[].plans.clear()
    g[].epochs[address] = shared_dict_epoch(address)
    g[].plans[address] = Int(plan._obj_ptr)
