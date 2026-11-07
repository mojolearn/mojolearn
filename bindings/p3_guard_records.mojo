# SPDX-License-Identifier: Apache-2.0
"""Layout-guard rows checked once per class state, not once per call.

Samba and optimizer admission keep import-time guard rows: a class, its MRO
snapshot and the descriptor identities native code relies on. Each row
depends only on its class rows, so after it passed, the class-watcher record
(bindings/native_class_admission.mojo) stays valid until the class or an
ancestor is modified (`PyType_Modified` bumps the class epoch). Python 3.10 and
3.11 have no type watcher: every call there keeps the full check.

A record is per (family, row index) and stores the address of the row it
admitted, so a class guarded by several rows with different name sets, or a
context whose guard tuple was replaced, never reuses another row's verdict.
The rows live in the import-time connection context for the life of the
process, so a row address is never recycled while its record is live. The
payload is a plain int and never retains the class.
"""
from std.python import PythonObject
from bindings.native_loaded_module import NativeModules
from bindings.native_class_admission import _class_arm, _class_epoch, class_watch_ready
from bindings.native_model_records import records_live, records_open, records_slot_guard

# Storage (lane ne1-e-records, 2026-11-06): the rows live in the GUARD slot
# of the class's entry in bindings/native_model_records.mojo (released if the
# class dies), with the class epoch at admission; validity is unchanged.


def guard_row_admitted(mut mods: NativeModules, family: String, index: Int, cls: PythonObject, row: PythonObject) raises -> Bool:
    if not class_watch_ready():
        return False
    if not records_live(cls):
        return False
    var name = family + String(index)
    var epoch = _class_epoch(cls)
    var slot = records_slot_guard(Int(cls._obj_ptr))
    if name not in slot[].rows:
        return False
    return slot[].rows[name] == Int(row._obj_ptr) and slot[].epochs[name] == epoch


def guard_row_admit(mut mods: NativeModules, family: String, index: Int, cls: PythonObject, row: PythonObject) raises:
    if not _class_arm(cls):
        return
    var epoch = _class_epoch(cls)
    var key = records_open(cls)
    if key == 0:
        return
    var name = family + String(index)
    var slot = records_slot_guard(key)
    slot[].rows[name] = Int(row._obj_ptr)
    slot[].epochs[name] = epoch
