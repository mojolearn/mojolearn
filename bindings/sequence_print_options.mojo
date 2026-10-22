# SPDX-License-Identifier: Apache-2.0
# Read NumPy's public print configuration through admitted native metadata.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields


def sequence_print_mode(context: PythonObject, release: Int) raises -> Int:
    # set_printoptions is Python application configuration, outside product
    # execution. Reading its native dictionary/ContextVar needs no Python
    # get_printoptions/_get_legacy_print_mode callback. NumPy 2.3 source and
    # release notes specify the new float32 threshold and legacy<=2.2 rule:
    # https://numpy.org/doc/2.3/release/2.3.0-notes.html#changes
    var name = "numpy._core.arrayprint" if release >= 200 else "numpy.core.arrayprint"
    var module = py_get(context["modules"], PythonObject(name))
    if module is None or py_type(module) is not context["module_type"]:
        raise Error("TypeError: native text output requires NumPy's print configuration module")
    var b = Python.import_module("builtins")
    var fields = py_fields(module)
    var options = py_get(fields, "_format_options")
    if options is None:
        var variable = py_get(fields, "format_options")
        if py_type(variable) is not context["sequence_print_context_type"]:
            raise Error("TypeError: native text output requires a native print configuration ContextVar")
        options = py_call(context["sequence_print_context_get"], variable)
    if py_type(options) is not b.dict:
        raise Error("TypeError: native text output requires a native print configuration dictionary")
    var keys = py_call(b.list, py_call(options.keys))
    for i in range(len(keys)):  # driver-loop: over NumPy's fixed print option names before keyed reads
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native print configuration requires native string keys")
    var legacy = py_get(options, "legacy")
    if py_type(legacy) is not b.int:
        raise Error("TypeError: native text output requires NumPy's native integer legacy mode")
    var mode = Int(py=legacy)
    return min(mode, 202) if release < 203 else mode
