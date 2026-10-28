from bindings.native_identity_key import NativeIdentityKey
from bindings.native_loaded_module import NativeModules
from std.python import PythonObject

def main() raises:
    var mods = NativeModules()
    var key = NativeIdentityKey(mods)
    var value = PythonObject(None)
    key.add(value)
    var pos = 0
    _ = key.matches(value, pos)
    key.fields(value, List[String]())
    _ = key.matches_fields(value, List[String](), pos)
    key.watch_arrays(mods, value)
    _ = key.arrays_match(mods, value)
    _ = key.retains(value)
