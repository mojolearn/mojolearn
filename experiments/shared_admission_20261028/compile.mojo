# Compile-only coverage. These exported entries are never executed.
from std.python import PythonObject
from bindings.native_loaded_module import NativeModules, native_loaded_module
from bindings.native_array_snapshot import NativeArrayAdmission, native_array_snapshot

@export
def compile_array_admission(context: PythonObject, value: PythonObject) raises -> PythonObject:
    var admission = NativeArrayAdmission(context)
    admission.recheck(context)
    var mods = NativeModules()
    _ = mods.module("builtins")
    _ = mods.module("builtins")
    mods.released()
    _ = mods.module("builtins")
    return native_array_snapshot(mods, value, context)

from bindings.native_identity_key import NativeIdentityKey
@export
def compile_identity_walk(context: PythonObject, value: PythonObject, fields: PythonObject) raises -> Bool:
    var mods = NativeModules()
    var key = NativeIdentityKey(mods)
    key.add(value)
    key.fields(fields, List[String]())
    key.watch_arrays(mods, context)
    var position = 0
    return key.matches(value, position) and key.matches_fields(fields, List[String](), position) and key.complete(position) and key.arrays_match(mods, context)
