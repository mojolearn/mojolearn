# Compile-only changed operation; no execution.
from std.python import PythonObject
from bindings.user_objects_native import user_objects_call


def main() raises:
    var value = PythonObject(None)
    _ = user_objects_call(value)
