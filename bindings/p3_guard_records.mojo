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
from bindings.native_class_admission import class_record, class_admit


def guard_row_admitted(mut mods: NativeModules, family: String, index: Int, cls: PythonObject, row: PythonObject) raises -> Bool:
    var record = class_record(mods, family + String(index), cls)
    if record is None:
        return False
    return Int(py=record) == Int(row._obj_ptr)


def guard_row_admit(mut mods: NativeModules, family: String, index: Int, cls: PythonObject, row: PythonObject) raises:
    class_admit(mods, family + String(index), cls, PythonObject(Int(row._obj_ptr)))
