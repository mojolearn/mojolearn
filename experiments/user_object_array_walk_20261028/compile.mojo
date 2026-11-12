# Compile-only changed Array flatten closure; never executed.
from std.python import PythonObject
from bindings.user_object_flatten_door import UserObjectFlatten


def main() raises:
    var reader = UserObjectFlatten()
    var value = PythonObject(None)
    reader.array_into(value, value)
