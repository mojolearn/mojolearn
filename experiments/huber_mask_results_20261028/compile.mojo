# Compile-only affected fit closure; this program is never executed.
from std.python import PythonObject
from bindings.x_linear_native import xl_fit_huber


def main() raises:
    var value = PythonObject(None)
    _ = xl_fit_huber(value, value, value)
